"""Regression test: pre/postprocessing must match MONAI ScaleIntensityRanged."""
import numpy as np
from latimsnap_i2i.model_loader import _apply_preprocess, _apply_postprocess


def test_minmax_roundtrip():
    vol = np.linspace(-1500, 3500, 100, dtype=np.float32).reshape(1, 10, 10, 1)
    cfg = {"min": -1000.0, "max": 3000.0}
    norm = _apply_preprocess("minmax", cfg, vol)
    assert norm.min() == -1.0 and norm.max() == 1.0        # clipped + scaled to [-1,1]
    inv = _apply_postprocess("denorm_minmax", cfg, norm)
    assert np.allclose(inv, np.clip(vol, -1000, 3000), atol=0.5)  # -1000..3000, no -5000


def test_minmax_matches_monai_formula():
    a = np.array([-1000.0, 0.0, 1000.0, 3000.0], dtype=np.float32)
    norm = _apply_preprocess("minmax", {"min": -1000.0, "max": 3000.0}, a)
    expected = (a + 1000) / 4000 * 2 - 1
    assert np.allclose(norm, expected)
