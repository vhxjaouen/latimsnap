"""Tests for runtime axis selection, 2D dim inference and slice padding."""
import numpy as np
import pytest

from latimsnap_i2i.model_loader import (
    _in_plane_axes,
    _pad_slices_to_patch,
)
from latimsnap_i2i.server import _resolve_axis_planes, _resolve_fusion
from latimsnap_i2i.model_loader import _fourier_burst


def _spec(preprocess=None):
    preprocess = preprocess or {"type": "minmax", "min": -1000.0}
    return type("S", (), {"preprocess": preprocess})


def test_resolve_axis_planes():
    assert _resolve_axis_planes("axial") == ["axial"]
    assert _resolve_axis_planes("sagittal") == ["sagittal"]
    assert _resolve_axis_planes("frontal") == ["coronal"]
    assert _resolve_axis_planes("coronal") == ["coronal"]
    assert _resolve_axis_planes("all") == ["axial", "sagittal", "coronal"]
    with pytest.raises(Exception):
        _resolve_axis_planes("bogus")


def test_in_plane_axes():
    assert _in_plane_axes(["axial"]) == {0, 1}
    assert _in_plane_axes(["sagittal"]) == {1, 2}
    assert _in_plane_axes(["coronal"]) == {0, 2}
    assert _in_plane_axes(["axial", "sagittal", "coronal"]) == {0, 1, 2}


def test_pad_small_planes_are_background_filled():
    spatial = {"mode": "slice", "patch_size": [256, 256]}
    vol = np.full((1, 48, 48, 24), -1000.0, np.float32)
    vol[..., 0] = 500.0  # a spot of real data

    # axial pads only its two in-plane axes (Z, Y); X (the slice axis) stays.
    padded, crop = _pad_slices_to_patch(vol, spatial, _spec(), ["axial"])
    assert padded.shape == (1, 256, 256, 24)
    assert crop == (208, 208, 0)
    # original data preserved at the lower-left corner, background fills the rest
    assert np.array_equal(padded[:, :48, :48, :24], vol)
    assert padded[0, 0, 0, -1] == -1000.0

    # single-plane sagittal pads only (Y, X)
    padded, crop = _pad_slices_to_patch(vol, spatial, _spec(), ["sagittal"])
    assert padded.shape == (1, 48, 256, 256)
    assert crop == (0, 208, 232)

    # all three planes pad every axis
    padded, crop = _pad_slices_to_patch(
        vol, spatial, _spec(), ["axial", "sagittal", "coronal"])
    assert padded.shape == (1, 256, 256, 256)
    assert crop == (208, 208, 232)


def test_pad_value_from_preprocess_floor():
    spatial = {"mode": "slice", "patch_size": [256, 256]}
    vol = np.zeros((1, 48, 48, 24), np.float32)
    padded, _ = _pad_slices_to_patch(vol, spatial, _spec({"min": -1000.0}), ["axial"])
    assert padded[0, 200, 5, 0] == -1000.0          # CT background (Z padded)

    padded_mr, _ = _pad_slices_to_patch(vol, spatial, _spec({"min": 0.0}), ["axial"])
    assert padded_mr[0, 200, 5, 0] == 0.0           # MR background


def test_pad_skipped_when_axes_large_enough():
    spatial = {"mode": "slice", "patch_size": [256, 256]}
    vol = np.zeros((1, 512, 512, 128), np.float32)
    padded, crop = _pad_slices_to_patch(vol, spatial, _spec(), ["axial"])
    # axial in-plane (Z=512, Y=512) already >= 256 -> no padding
    assert padded.shape == vol.shape
    assert crop == (0, 0, 0)

def test_resolve_fusion():
    assert _resolve_fusion("average") == "mean"
    assert _resolve_fusion("avg") == "mean"
    assert _resolve_fusion("median") == "median"
    assert _resolve_fusion("fba") == "fourier_burst"
    with pytest.raises(Exception):
        _resolve_fusion("bogus")


def test_median_fusion_shape_matches_fba():
    rng = np.random.default_rng(3)
    res = [rng.normal(0, 100, (1, 1, 20, 18, 16)).astype(np.float32) for _ in range(3)]
    mean = np.mean(res, axis=0)
    median = np.median(res, axis=0)
    fba = _fourier_burst(res, 5.0)[None]
    assert mean.shape == median.shape == fba.shape == (1, 1, 20, 18, 16)
    # median and average should generally differ from each other
    assert np.abs(median - mean).max() > 0
