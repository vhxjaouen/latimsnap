"""Tests for the wire codec shared with LaTIM-SNAP."""

import base64
import gzip

import numpy as np
import pytest

from latimsnap_i2i import imgcodec


def test_roundtrip_scalar():
    vol = (np.random.rand(1, 4, 5, 6) * 1000).astype(np.float32)
    raw, meta = imgcodec.encode_raw(vol)
    decoded = imgcodec.decode_raw(raw, meta)
    assert decoded.shape == vol.shape
    assert np.allclose(decoded, vol)


def test_roundtrip_multicomponent():
    vol = np.random.rand(3, 4, 5, 6).astype(np.float32)
    raw, meta = imgcodec.encode_raw(vol)
    assert meta["components_per_pixel"] == 3
    decoded = imgcodec.decode_raw(raw, meta)
    assert np.allclose(decoded, vol)


def test_encode_result_shape():
    vol = np.random.rand(1, 4, 5, 6).astype(np.float32)
    res = imgcodec.encode_result(vol)
    payload = base64.b64decode(res["result"])
    raw = gzip.decompress(payload)
    decoded = imgcodec.decode_raw(raw, res["metadata"])
    assert np.allclose(decoded, vol)
