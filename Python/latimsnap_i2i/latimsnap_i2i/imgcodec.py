"""Encoding/decoding of image buffers exchanged with the LaTIM-SNAP client.

The client sends images as multipart parts:
  file     : gzip stream of raw little-endian voxel data
  metadata : JSON with dimensions/spacing/origin/direction/components/type

The server returns a transferred image as:
  result   : base64(gzip(raw little-endian voxel data))
  metadata : JSON with the same schema

Raw voxel buffer layout is component-major per pixel: for an image of shape
D = (Z, Y, X) with C components per pixel, the buffer has D*C elements laid
out as interleaved [pix0_c0, pix0_c1, ..., pix1_c0, ...].
"""

import base64
import gzip
import io
import json

import numpy as np

NUMPY_DTYPES = {
    "int8": np.dtype(np.int8),
    "uint8": np.dtype(np.uint8),
    "int16": np.dtype(np.int16),
    "uint16": np.dtype(np.uint16),
    "int32": np.dtype(np.int32),
    "uint32": np.dtype(np.uint32),
    "int64": np.dtype(np.int64),
    "uint64": np.dtype(np.uint64),
    "float32": np.dtype(np.float32),
    "float64": np.dtype(np.float64),
}

DTYPE_NAMES = {v: k for k, v in NUMPY_DTYPES.items()}


def dtype_name(dtype):
    return DTYPE_NAMES.get(np.dtype(dtype), "float32")


def _metadata_to_dict(image, components):
    return {
        "dimensions": [int(s) for s in image.shape],   # [Z, Y, X]
        "spacing": [float(s) for s in image.spacing_] if hasattr(image, "spacing_") else [1.0] * 3,
        "origin": [float(o) for o in image.origin_] if hasattr(image, "origin_") else [0.0] * 3,
        "direction": sum((list(image.direction_[i]) for i in range(3)), []),
        "components_per_pixel": int(components),
        "component_type": dtype_name(image.dtype),
    }


def gzip_bytes(data):
    return gzip.compress(data)


def gunzip_bytes(data):
    return gzip.decompress(data)


def decode_raw(raw_bytes, metadata):
    """Decode raw little-endian voxel bytes + metadata dict into a numpy array.

    ``metadata["dimensions"]`` is in ITK order [X, Y, Z] (X fastest in the raw
    buffer). Returns a (C, X, Y, Z) numpy array where C=components_per_pixel.
    """
    if "component_type" not in metadata:
        raise ValueError("metadata missing 'component_type'")
    dtype = NUMPY_DTYPES.get(metadata.get("component_type", "float32"))
    if dtype is None:
        raise ValueError("unsupported component_type")

    dims = [int(d) for d in metadata.get("dimensions", [])]
    comps = int(metadata.get("components_per_pixel", 1))
    if len(dims) != 3:
        raise ValueError("expected 3D dimensions")

    total = int(np.prod(dims)) * comps
    expected_bytes = total * dtype.itemsize
    if len(raw_bytes) != expected_bytes:
        raise ValueError(
            "raw buffer size mismatch: got %d bytes, expected %d" % (len(raw_bytes), expected_bytes))

    arr = np.frombuffer(raw_bytes, dtype=dtype)
    arr = arr.reshape([comps] + dims)
    return arr


def encode_raw(volume):
    """Encode a (C, X, Y, Z) numpy array into (raw_bytes, metadata)."""
    volume = np.ascontiguousarray(volume, dtype=volume.dtype)
    c = int(volume.shape[0])
    dims = [int(s) for s in volume.shape[1:]]  # [X, Y, Z] (ITK order)
    interleaved = volume.transpose(1, 2, 3, 0)  # (X, Y, Z, C)
    raw = interleaved.tobytes()
    metadata = {
        "dimensions": dims,
        "spacing": [1.0, 1.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        "components_per_pixel": int(c),
        "component_type": dtype_name(volume.dtype),
    }
    return raw, metadata


def encode_result(volume, base_metadata=None):
    """Encode a (C, X, Y, Z) numpy array into the client result JSON dict.

    Physical geometry (spacing/origin/direction) is inherited from the source
    image description (``base_metadata``, i.e. the uploaded source) so the
    result registers in the same space as the source.
    """
    volume = np.ascontiguousarray(volume, dtype=volume.dtype)
    c = int(volume.shape[0])
    dims = [int(s) for s in volume.shape[1:]]
    raw, _ = encode_raw(volume)
    payload = gzip_bytes(raw)
    metadata = {
        "dimensions": dims,
        "components_per_pixel": c,
        "component_type": dtype_name(volume.dtype),
    }
    if base_metadata:
        ans = dict(base_metadata)
        ans.update(metadata)
        metadata = ans
    else:
        raw2, metadata = encode_raw(volume)
        _ = raw2
    return {
        "result": base64.b64encode(payload).decode("ascii"),
        "metadata": metadata,
    }
