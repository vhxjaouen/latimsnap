"""Intensity scaling between HU and [-1, 1].

Replicates the notebook's MONAI ``ScaleIntensityRanged(min=-1000, max=3000,
b_min=-1, b_max=1, clip=True/thin)`` transforms, as pure numpy (no MONAI
needed at serving time).
"""

from __future__ import annotations

import numpy as np


def _hu_to_norm(voxels, lo=-1000.0, hi=3000.0):
    """HU -> [-1, 1], clipped exactly like ``ScaleIntensityRanged(clip=True)``."""
    v = np.clip(np.asarray(voxels, dtype=np.float32), lo, hi)
    span = hi - lo
    return 2.0 * (v - lo) / span - 1.0


def _norm_to_hu(voxels, lo=-1000.0, hi=3000.0, clip=True):
    """[-1, 1] -> HU (denormalizes; optionally clips the normalized input first)."""
    v = np.asarray(voxels, dtype=np.float32)
    if clip:
        v = np.clip(v, -1.0, 1.0)
    span = hi - lo
    return (v + 1.0) / 2.0 * span + lo


def make_scaler(lo: float, hi: float):
    """Return (forward, inverse) closures bound to a (lo, hi) range."""
    return (lambda a: _hu_to_norm(a, lo, hi),
            lambda b: _norm_to_hu(b, lo, hi))
