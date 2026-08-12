"""Multi-view ensembling (mean / Fourier Burst Accumulation).

``FourierBurstAccumulation`` is ported verbatim from the notebook's
production inference cell (3-plane ensembling).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def mean_ensemble(images: Sequence[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(images, axis=0), axis=0)


def fourier_burst_accumulation(images: Sequence[np.ndarray], p: float = 5.0) -> np.ndarray:
    """Frequency-domain, magnitude-weighted fusion of aligned images."""
    imgs = [np.asarray(im, dtype=np.float32) for im in images]
    vs_hat = [np.fft.rfftn(im) for im in imgs]

    if p == float("inf"):
        out_hat = np.max(vs_hat, axis=0)
    else:
        power = [np.abs(v) ** p for v in vs_hat]
        denominator = np.sum(power, axis=0)
        ws = [pw / denominator for pw in power]
        out_hat = sum(w * v for w, v in zip(ws, vs_hat))

    return np.fft.irfftn(out_hat, axes=list(range(out_hat.ndim)), s=imgs[0].shape).astype(np.float32)


def ensemble_views(images: Sequence[np.ndarray], mode: str = "fourier_burst", p: float = 5.0) -> np.ndarray:
    if mode == "fourier_burst":
        return fourier_burst_accumulation(images, p=p)
    return mean_ensemble(images)
