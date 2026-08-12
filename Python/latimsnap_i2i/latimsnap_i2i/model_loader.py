"""Generic model loader + runner for the image-to-image server.

The server is deliberately framework-agnostic. A "model spec" (JSON) describes
where the encoder lives and how input/output volumes should be pre/post
processed. Supported frameworks:

  * ONNX Runtime (``framework: "onnx"``) - most portable; no torch needed.
  * PyTorch     (``framework: "torch"``) - requires a ``module``/``class`` so
    the architecture can be instantiated, plus an optional weights file.

Volumes are handled as channel-first numpy arrays of shape (C, Z, Y, X).
"""

import importlib
import json
import logging
import os

import numpy as np

log = logging.getLogger("latimsnap_i2i.model")


# ---------------------------------------------------------------------------
# Preprocessing / postprocessing
# ---------------------------------------------------------------------------

def _apply_preprocess(name, cfg, vol):
    """vol is (C,Z,Y,X) float32. Returns (C,Z,Y,X) float32."""
    if name in (None, "identity", ""):
        return vol
    if name == "minmax":
        lo = float(cfg.get("min", 0.0))
        hi = float(cfg.get("max", 1.0))
        return np.clip(vol, lo, hi)
    if name == "percentile":
        plo = float(cfg.get("pmin", 0.5))
        phi = float(cfg.get("pmax", 99.5))
        flat = vol.reshape(vol.shape[0], -1)
        vmin = np.percentile(flat, plo, axis=1, keepdims=True).astype(np.float32)
        vmax = np.percentile(flat, phi, axis=1, keepdims=True).astype(np.float32)
        denom = np.maximum(vmax - vmin, 1e-8)
        out = (vol - vmin) / denom
        out = np.clip(out, 0.0, 1.0)
        return out.astype(np.float32)
    if name == "normalize":
        mean = float(cfg.get("mean", 0.0))
        std = float(cfg.get("std", 1.0))
        return ((vol - mean) / max(float(std), 1e-8)).astype(np.float32)
    raise ValueError("unknown preprocessing type %r" % name)


def _apply_postprocess(name, cfg, vol):
    if name in (None, "identity", ""):
        return vol
    if name == "clip":
        return np.clip(vol, float(cfg.get("min", 0.0)), float(cfg.get("max", 1.0)))
    if name == "denorm_minmax":
        lo = float(cfg.get("min", 0.0))
        hi = float(cfg.get("max", 1.0))
        return (vol * (hi - lo) + lo).astype(np.float32)
    raise ValueError("unknown postprocessing type %r" % name)


# ---------------------------------------------------------------------------
# Model spec
# ---------------------------------------------------------------------------

class ModelSpec:
    def __init__(self, spec):
        self.spec = spec
        self.id = spec["id"]
        self.name = spec.get("name", self.id)
        self.framework = spec.get("framework", "onnx")
        self.input_channels = int(spec.get("input_channels", 1))
        self.output_channels = int(spec.get("output_channels", 1))
        self.preprocess = spec.get("preprocessing") or {"type": "identity"}
        self.postprocess = spec.get("postprocessing") or {"type": "identity"}
        self.spatial = spec.get("spatial") or {"mode": "whole"}
        self.output_dtype = spec.get("output_dtype", "float32")
        self.model = spec.get("model") or {}
        self.base_dir = os.path.dirname(os.path.abspath(spec.get("_path", "")))


def _resolve_path(spec, key):
    p = spec.model.get(key)
    if not p:
        return None
    if os.path.isabs(p):
        return p
    return os.path.join(spec.base_dir, p)


# ---------------------------------------------------------------------------
# Framework runners
# ---------------------------------------------------------------------------

def _load_onnx(spec):
    import onnxruntime as ort
    model_path = _resolve_path(spec, "onnx_path")
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError("onnx_path not found: %r" % model_path)
    # Use CPUExecutionProvider by default (AI engines may use others).
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    for p in ort.get_available_providers():
        if p in ("CUDAExecutionProvider", "TensorrtExecutionProvider"):
            providers.insert(0, p)
    sess = ort.InferenceSession(
        model_path, sess_options=sess_options, providers=providers)
    return sess, sess.get_inputs()[0].name


def _load_torch(spec, progress):
    import torch
    module_name = spec.model.get("module")
    class_name = spec.model.get("class")
    if not module_name or not class_name:
        raise ValueError("torch framework requires model.module and model.class")
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    kwargs = spec.model.get("kwargs") or {}
    net = cls(**kwargs)
    net.eval()
    weights = _resolve_path(spec, "weights")
    if weights:
        if os.path.exists(weights):
            state = torch.load(weights, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            net.load_state_dict(state)

    def _run(x):
        # x: (1, C, Z, Y, X) torch
        with torch.no_grad():
            return net(x)

    return _run, None


# ---------------------------------------------------------------------------
# Spatial execution (whole-volume vs patches)
# ---------------------------------------------------------------------------

def _run_inference(runner, input_name, vol, output_channels, progress,
                   spatial):
    """vol: (1, C, Z, Y, X) float32. Returns (1, OC, Z, Y, X)."""
    mode = spatial.get("mode", "whole")
    if mode == "whole":
        progress(0.35)
        out = _run_once(runner, input_name, vol, output_channels)
        progress(0.85)
        return out

    if mode == "patches":
        return _run_patches(runner, input_name, vol, output_channels,
                            progress, spatial)

    raise ValueError("unknown spatial mode %r" % mode)


def _run_once(runner, input_name, vol, output_channels):
    if input_name is None:  # torch runner
        out = runner(vol)
        return out
    # onnx runner
    out = runner.run(None, {input_name: vol})
    return np.asarray(out[0])


def _run_patches(runner, input_name, vol, output_channels, progress, spatial):
    import numpy as np
    c, z, y, x = vol.shape[1:]
    pz, py, px = spatial.get("patch_size", [min(z, 64), min(y, 64), min(x, 64)])
    overlap = float(spatial.get("overlap", 0.25))
    acc = np.zeros((1, output_channels, z, y, x), dtype=np.float32)
    wsum = np.zeros((1, 1, z, y, x), dtype=np.float32)

    def window(pl):
        # linear ramp weighting for seamless blending
        if pl <= 1:
            return np.ones((pl,), dtype=np.float32)
        ramp = np.linspace(0.0, 1.0, int(pl // 2) + 1)
        half = np.concatenate([ramp, np.ones(pl - 2 * len(ramp) + 1), ramp[::-1]])
        half = np.clip(half, 0, 1)
        return half

    wz, wy, wx = window(pz), window(py), window(px)
    w3 = np.einsum("i,j,k->ijk", wz, wy, wx).astype(np.float32)

    starts = []
    step_z = max(1, int(pz * (1 - overlap)))
    step_y = max(1, int(py * (1 - overlap)))
    step_x = max(1, int(px * (1 - overlap)))
    for zz in range(0, z, step_z):
        for yy in range(0, y, step_y):
            for xx in range(0, x, step_x):
                starts.append((zz, yy, xx))

    total = len(starts)
    for i, (zz, yy, xx) in enumerate(starts):
        ze = min(zz + pz, z)
        ye = min(yy + py, y)
        xe = min(xx + px, x)
        patch = np.zeros((1, c, pz, py, px), dtype=np.float32)
        patch[:, :, :ze - zz, :ye - yy, :xe - xx] = vol[:, :, zz:ze, yy:ye, xx:xe]
        out = _run_once(runner, input_name, patch, output_channels)
        ps = out.shape
        # write valid region with patch weights
        wz_, wy_, wx_ = w3[:ze - zz, :ye - yy, :xe - xx]
        out_w = out[:, :, :ze - zz, :ye - yy, :xe - xx] * wz_[None, None, :, :, :]
        acc[:, :, zz:ze, yy:ye, xx:xe] += out_w * wy_[None, None, :, :][..., None] * wx_[None, None, :]
        wsum[:, :, zz:ze, yy:ye, xx:xe] += (wz_ * wy_ * wx_)[None, None]
        progress(0.35 + 0.5 * (i + 1) / total)
    wsum = np.maximum(wsum, 1e-8)
    return acc / wsum


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------

class ModelRunner:
    """Holds one loaded model + spec; run() performs a full transfer."""

    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self._runner = None
        self._input_name = "input"

    def load(self):
        if self.spec.framework == "identity":
            # Built-in no-op for contract smoke tests and as a reference.
            self._runner = lambda x: x
            self._input_name = None
        elif self.spec.framework == "onnx":
            self._runner, self._input_name = _load_onnx(self.spec)
        elif self.spec.framework == "torch":
            self._runner, self._input_name = _load_torch(self.spec, lambda f: None)
        else:
            raise ValueError("unsupported framework %r" % self.spec.framework)

    def run(self, vol, progress=lambda f: None):
        """vol: (C, Z, Y, X) float32. Returns (OC, Z, Y, X) float32."""
        spec = self.spec
        progress(0.05)
        vol = vol.astype(np.float32)
        vol = _apply_preprocess(spec.preprocess.get("type"), spec.preprocess, vol)
        progress(0.12)

        inp = vol[None, ...]  # (1, C, Z, Y, X)
        out = _run_inference(self._runner, self._input_name, inp,
                             spec.output_channels, progress, spec.spatial)
        out = np.asarray(out[0])  # (OC, Z, Y, X)
        out = _apply_postprocess(spec.postprocess.get("type"), spec.postprocess, out)
        out = np.asarray(out, dtype=np.float32)
        progress(0.95)
        return out


def load_model_specs(models_dir, model_registry):
    """Load all model specs from a directory; update the registry dict."""
    if not models_dir:
        return model_registry
    if not os.path.isdir(models_dir):
        log.error("models dir not found: %s", models_dir)
        return model_registry
    for fname in sorted(os.listdir(models_dir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(models_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            data["_path"] = path
            spec = ModelSpec(data)
            model_registry[spec.id] = spec
        except Exception as exc:  # noqa: BLE001
            log.error("failed to load model spec %s: %s", path, exc)
    return model_registry
