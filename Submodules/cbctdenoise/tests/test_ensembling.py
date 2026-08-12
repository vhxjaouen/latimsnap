import numpy as np

from cbctdenoise.ensembling import fourier_burst_accumulation, mean_ensemble


def test_mean():
    a = np.full((4, 4), 1.0, dtype=np.float32)
    b = np.full((4, 4), 3.0, dtype=np.float32)
    assert np.allclose(mean_ensemble([a, b]), 2.0)


def test_fba_shape_and_dtype():
    a = np.random.rand(4, 5, 6).astype(np.float32)
    b = np.random.rand(4, 5, 6).astype(np.float32)
    out = fourier_burst_accumulation([a, b], p=5.0)
    assert out.shape == a.shape
    assert out.dtype == np.float32
    assert np.all(np.isfinite(out))