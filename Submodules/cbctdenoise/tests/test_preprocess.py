import numpy as np
import pytest

from cbctdenoise import preprocess


def test_roundtrip():
    fwd, inv = preprocess.make_scaler(-1000.0, 3000.0)
    v = np.array([-2000.0, -1000.0, 1000.0, 3000.0, 4000.0], dtype=np.float32)
    n = fwd(v)
    assert n[0] == -1.0 and n[-1] == 1.0  # clipped
    assert n[1] == -1.0 and n[3] == 1.0
    back = inv(n)
    assert np.allclose(back, np.clip(v, -1000, 3000), atol=1e-3)


def test_clip_faithful_to_monai():
    # ScaleIntensityRanged: b = (a-a_min)/(a_max-a_min)*(b_max-b_min)+b_min
    fwd, _ = preprocess.make_scaler(-1000.0, 3000.0)
    v = np.array([1500.0], dtype=np.float32)
    expected = (1500 - (-1000)) / (3000 - (-1000)) * 2 - 1
    assert np.isclose(fwd(v)[0], expected)
