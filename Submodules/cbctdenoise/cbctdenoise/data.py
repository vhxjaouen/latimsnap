"""Data loading / transforms for the research (training) path.

Replicates the notebook's 3D->2D slicing and HU scaling. Requires the ``[train]``
extra (MONAI) — imported lazily so the serving path never pulls it in.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from cbctdenoise import preprocess


def split_3d_to_2d(volume: np.ndarray, axis: int = -1):
    """Split a (C, H, W, D) volume into a list of (C, H, W) slices."""
    return [volume[..., i] for i in range(volume.shape[-1])]


class SliceDataset:
    """Minimal (C, H, W) slice dataset from a list of 3D volumes.

    Keep it dependency-light; for MONAI/PersistentDataset see the notebook.
    """

    def __init__(self, volumes, hu_clip: Tuple[float, float] = (-1000.0, 3000.0),
                 patch_size: int = 256, random_crop: bool = True, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.hu_clip = hu_clip
        self.patch_size = patch_size
        self.random_crop = random_crop
        self._slices = []
        for vol in volumes:
            for sl in split_3d_to_2d(vol):
                self._slices.append(sl.astype(np.float32))

    def __len__(self):
        return len(self._slices)

    def __getitem__(self, idx):
        sl = self._slices[idx]
        sl = preprocess._hu_to_norm(sl, *self.hu_clip)
        if self.random_crop:
            h, w = sl.shape[1], sl.shape[2]
            ph = pw = self.patch_size
            sy = self.rng.integers(0, max(1, h - ph + 1))
            sx = self.rng.integers(0, max(1, w - pw + 1))
            sl = sl[:, sy:sy + ph, sx:sx + pw]
        return sl
