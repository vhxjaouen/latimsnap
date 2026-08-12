"""Geometry-correct, 2D-per-slice inference of a 2D generator over a 3D volume.

Reimplements the notebook's MONAI ``SliceInferer`` semantics (2D gaussian
sliding window, replicate padding) plus optional multi-planar ensembling, in
pure numpy + the model runner. The output volume is in the exact source grid.

The public API is the callable ``model.apply(volume, progress)`` consumed by
``latimsnap-i2i``'s generic loader (see :mod:`cbctdenoise.models.running`).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from cbctdenoise import ensembling


def _plane_slice_count(vol: np.ndarray, plane: str) -> int:
    # vol: (C, d0, d1, d2)
    if plane == "axial":
        return vol.shape[3]
    if plane == "coronal":
        return vol.shape[2]
    if plane == "sagittal":
        return vol.shape[1]
    raise ValueError(f"unknown plane {plane!r}")


def _run_on_plane(runner, vol, output, oc, config, progress, start, end):
    """vol (C,d0,d1,d2) -> output (OC,d0,d1,d2) filled slice-by-slice."""
    c = vol.shape[0]
    plane = config.get("plane", "axial")
    ph, pw = config.get("patch_size", [256, 256])
    overlap = float(config.get("overlap", 0.0))
    n = _plane_slice_count(vol, plane)

    for i in range(n):
        if plane == "axial":
            sl = vol[:, :, :, i]            # (C,d0,d1)
            out = _sw2d(runner, sl, oc, ph, pw, overlap)
            output[:, :, :, i] = out
        elif plane == "coronal":
            sl = vol[:, :, i, :]            # (C,d0,d2)
            out = _sw2d(runner, sl, oc, ph, pw, overlap)
            output[:, :, i, :] = out
        else:  # sagittal
            sl = vol[:, i, :, :]            # (C,d1,d2)
            out = _sw2d(runner, sl, oc, ph, pw, overlap)
            output[:, i, :, :] = out
        progress(start + (end - start) * (i + 1) / n)


def _sw2d(runner, im, oc, ph, pw, overlap):
    """2D sliding-window inference with gaussian blending + replicate padding."""
    c = im.shape[0]
    h, w = im.shape[1], im.shape[2]
    acc = np.zeros((oc, h, w), dtype=np.float32)
    wgt = np.zeros((h, w), dtype=np.float32)

    w2 = _gauss2d(ph, pw)
    step_h = max(1, int(ph * (1.0 - overlap)))
    step_w = max(1, int(pw * (1.0 - overlap)))

    for yy in range(0, h, step_h):
        for xx in range(0, w, step_w):
            sy = np.clip(np.arange(yy, yy + ph), 0, h - 1)
            sx = np.clip(np.arange(xx, xx + pw), 0, w - 1)
            patch = im[:, sy][:, :, sx]      # (C, ph, pw) replicate padded
            out = runner(patch)              # (OC, ph, pw)
            hh = min(ph, h - yy)
            ww = min(pw, w - xx)
            wpart = w2[:hh, :ww][None]          # (1, H, W) broadcast over channels
            acc[:, yy:yy + hh, xx:xx + ww] += out[:, :hh, :ww] * wpart
            wgt[yy:yy + hh, xx:xx + ww] += w2[:hh, :ww]
    return acc / np.maximum(1e-8, wgt)


def _gauss2d(ph: int, pw: int) -> np.ndarray:
    if ph <= 1 or pw <= 1:
        return np.ones((ph, pw), dtype=np.float32)
    gy = np.exp(-((np.arange(ph) - (ph - 1) / 2.0) ** 2) / (2 * ((ph / 4.0) ** 2)))
    gx = np.exp(-((np.arange(pw) - (pw - 1) / 2.0) ** 2) / (2 * ((pw / 4.0) ** 2)))
    return np.outer(gy, gx).astype(np.float32)


def apply_2d_model_to_volume(runner, volume: np.ndarray, config: dict,
                             progress: Optional[Callable[[float], None]] = None):
    """Run a 2D model over a 3D volume, optionally ensembling multiple planes.

    * ``volume``: (C, X, Y, Z) float32 (channel-first, ITK order).
    * ``runner``: callable taking a (C, H, W) patch and returning (OC, H, W).
    * ``config``: dict with plane / patch_size / overlap / ensemble_* keys.
    * Returns (OC, X, Y, Z) float32 in the source grid.
    """
    progress = progress or (lambda f: None)
    planes = config.get("ensemble_planes") or [config.get("plane", "axial")]
    oc = config.get("output_channels", 1)
    out_all = []

    for k, plane in enumerate(planes):
        cfg = dict(config)
        cfg["plane"] = plane
        out = np.zeros((oc,) + volume.shape[1:], dtype=np.float32)
        base = k / max(1, len(planes))
        span = 1.0 / max(1, len(planes))
        _run_on_plane(runner, volume, out, oc, cfg, progress,
                      base * 0.9, (base + span) * 0.9)
        out_all.append(out)

    if len(out_all) == 1:
        return out_all[0]

    mode = config.get("ensemble", "fourier_burst")
    return ensembling.ensemble_views(out_all, mode, config.get("ensemble_p", 5.0))
