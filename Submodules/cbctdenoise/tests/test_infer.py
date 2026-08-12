import numpy as np

from cbctdenoise.infer import apply_2d_model_to_volume


def _identity_runner(patch):
    # identity 2D model: (C,H,W) -> (C,H,W)
    return patch.copy()


def test_axial_reassembly_shape():
    vol = np.random.rand(1, 20, 24, 8).astype(np.float32)
    cfg = {"plane": "axial", "patch_size": (16, 16), "overlap": 0.5,
           "ensemble_planes": ["axial"], "ensemble": "none", "output_channels": 1}
    out = apply_2d_model_to_volume(_identity_runner, vol, cfg)
    assert out.shape == (1, 20, 24, 8)


def test_identity_preserves_value():
    vol = np.full((1, 20, 24, 8), 3.0, dtype=np.float32)
    cfg = {"plane": "axial", "patch_size": (16, 16), "overlap": 0.5,
           "ensemble_planes": ["axial"], "ensemble": "none", "output_channels": 1}
    out = apply_2d_model_to_volume(_identity_runner, vol, cfg)
    assert np.allclose(out, 3.0, atol=1e-4)


def test_threepane_mean_ensemble():
    vol = np.full((1, 20, 24, 8), 2.0, dtype=np.float32)
    cfg = {"plane": "axial", "patch_size": (16, 16), "overlap": 0.5,
           "ensemble_planes": ["axial", "coronal", "sagittal"],
           "ensemble": "mean", "output_channels": 1}
    out = apply_2d_model_to_volume(_identity_runner, vol, cfg)
    assert out.shape == (1, 20, 24, 8)
    assert np.allclose(out, 2.0, atol=1e-3)