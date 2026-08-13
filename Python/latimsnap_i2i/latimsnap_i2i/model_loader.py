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
        b_lo = float(cfg.get("b_min", -1.0))
        b_hi = float(cfg.get("b_max", 1.0))
        # Faithful to MONAI ScaleIntensityRanged(min,max -> b_min,b_max, clip=True).
        a = np.clip(vol, lo, hi)
        return ((a - lo) / max(hi - lo, 1e-8) * (b_hi - b_lo) + b_lo).astype(np.float32)
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
        # Faithful to MONAI's inverse ScaleIntensityRanged(..., clip=True):
        # map the model output from the normalized [-1,1] range back to
        # physical units [lo, hi].
        return ((np.clip(vol, -1.0, 1.0) + 1.0) / 2.0 * (hi - lo) + lo).astype(np.float32)
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
    # Prefer GPU when available. TensorRT is intentionally not listed (it needs
    # a separate install and would otherwise print init errors and fall back).
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    log.info("onnx inference providers requested: %s", providers)
    sess = ort.InferenceSession(
        model_path, sess_options=sess_options, providers=providers)
    return sess, sess.get_inputs()[0].name


def _load_torch(spec, progress):
    import sys  # noqa: PLC0415
    import torch  # noqa: PLC0415
    module_name = spec.model.get("module")
    class_name = spec.model.get("class")
    if not module_name or not class_name:
        raise ValueError("torch framework requires model.module and model.class")

    # Make the model's directory importable so vendored sibling modules
    # (e.g. ``vjnetworks``) resolve next to the declared module.
    if spec.base_dir and spec.base_dir not in sys.path:
        sys.path.insert(0, spec.base_dir)

    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    kwargs = spec.model.get("kwargs") or {}
    net = cls(**kwargs)
    net.eval()

    weights = _resolve_path(spec, "weights")
    if weights:
        if not os.path.exists(weights):
            raise FileNotFoundError("weights not found: %r" % weights)
        state = torch.load(weights, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            log.info("checkpoint top-level keys: %s", list(state.keys()))
            state = state["model"]
        elif isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(k.startswith("module.") for k in state):
            log.info("stripping 'module.' prefix (DDP checkpoint)")
            state = {k[len("module."):]: v for k, v in state.items()}

        # Verbose diagnostics: surface a checkpoint/architecture mismatch
        # (surfaces in the LaTIM-SNAP server log / dialog).
        if isinstance(state, dict):
            model_keys = set(net.state_dict().keys())
            missing = sorted(model_keys - set(state.keys()))
            unexpected = sorted(set(state.keys()) - model_keys)
            if missing:
                log.warning("missing keys (%d), e.g. %s", len(missing), missing[:6])
            if unexpected:
                log.warning("unexpected keys (%d), e.g. %s", len(unexpected), unexpected[:6])
        net.load_state_dict(state, strict=True)

    def _run(x):
        # x: (1, C, [H, W] | Z, Y, X) torch
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
    if mode == "slice":
        return _run_slice(runner, input_name, vol, output_channels,
                          progress, spatial)
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
# 2D models applied per-slice to a 3D volume (spatial.mode == "slice")
# ---------------------------------------------------------------------------

def _invoke(runner, input_name, batched_np):
    """Run a batched ``(N, C, ph, pw)`` tensor and return ``(N, OC, ph, pw)``."""
    if input_name is None:  # torch runner
        import torch  # noqa: PLC0415
        out = runner(torch.from_numpy(np.ascontiguousarray(batched_np, np.float32)))
        return np.asarray(out.detach().cpu().numpy(), dtype=np.float32)
    # onnx runner
    return np.asarray(runner.run(None, {input_name: batched_np})[0], dtype=np.float32)


def _sw2d(runner, input_name, im, output_channels, ph, pw, overlap):
    """2D sliding-window inference with gaussian blending + replicate padding.

    All windows of the slice are collected and run in a **single batched**
    inference call, which is dramatically faster on GPU than one small forward
    per window. ``im``: (1, C, H, W). Returns (1, OC, H, W).
    """
    c = im.shape[1]
    h, w = im.shape[2], im.shape[3]
    acc = np.zeros((1, output_channels, h, w), dtype=np.float32)
    wgt = np.zeros((1, 1, h, w), dtype=np.float32)

    w2 = _gauss2d(ph, pw)
    step_h = max(1, int(ph * (1.0 - overlap)))
    step_w = max(1, int(pw * (1.0 - overlap)))

    windows = []
    coords = []
    for yy in range(0, h, step_h):
        for xx in range(0, w, step_w):
            sy = np.clip(np.arange(yy, yy + ph), 0, h - 1)
            sx = np.clip(np.arange(xx, xx + pw), 0, w - 1)
            windows.append(im[:, :, sy][:, :, :, sx])  # (1, C, ph, pw)
            coords.append((yy, xx))

    if windows:
        batched = np.concatenate(windows, axis=0)     # (N, C, ph, pw)
        outs = _invoke(runner, input_name, batched)    # (N, OC, ph, pw)
        for n, (yy, xx) in enumerate(coords):
            out = outs[n]                              # (OC, ph, pw)
            hh = min(ph, h - yy)
            ww = min(pw, w - xx)
            wpart = w2[:hh, :ww][None, None]
            acc[:, :, yy:yy + hh, xx:xx + ww] += out[None, :, :hh, :ww] * wpart
            wgt[:, :, yy:yy + hh, xx:xx + ww] += wpart
    return acc / np.maximum(1e-8, wgt)


def _gauss2d(ph, pw, sigma_scale=0.5):
    """MONAI-style gaussian importance map.

    ``sigma = size * sigma_scale`` per axis and the map is clamped to a small
    positive floor (>=1e-3) so overlapping windows always get non-zero weight
    (faithful to monai.compute_importance_map).
    """
    if ph <= 1 or pw <= 1:
        return np.ones((ph, pw), dtype=np.float32)

    def g(n):
        x = np.arange(-(n - 1) / 2.0, (n - 1) / 2.0 + 1, dtype=np.float32)
        sig = n * sigma_scale
        return np.exp(x * x / (-2.0 * sig * sig))

    w = np.outer(g(ph), g(pw)).astype(np.float32)
    mn = max(float(w.min()), 1e-3)
    return np.clip(w, mn, None)


def _monai_starts(image_size, patch_size, scan_interval):
    """Window start indices exactly as MONAI's dense_patch_slices.

    The last window is pulled back so it ends at the image edge (no overshoot,
    hence no padding/replication needed).
    """
    starts = []
    for dim in range(len(image_size)):
        n = 1 if scan_interval[dim] == 0 else int(np.ceil(image_size[dim] / scan_interval[dim]))
        if scan_interval[dim] != 0:
            sd = next((k for k in range(n)
                       if k * scan_interval[dim] + patch_size[dim] >= image_size[dim]), None)
            n = (sd + 1) if sd is not None else 1
        ds = []
        for idx in range(n):
            s = idx * scan_interval[dim]
            s -= max(s + patch_size[dim] - image_size[dim], 0)
            ds.append(s)
        starts.append(ds)
    return starts


def _sw2d(runner, input_name, im, output_channels, ph, pw, overlap, sigma_scale=0.5):
    """2D sliding-window inference, MONAI-compatible.

    Window starts match MONAI's dense_patch_slices (last window snug to the
    edge, windows always fit inside the image - no replicate padding). All
    windows of the slice run in one batched call for GPU speed.

    ``im``: (1, C, H, W). Returns (1, OC, H, W).
    """
    h, w = im.shape[2], im.shape[3]
    acc = np.zeros((1, output_channels, h, w), dtype=np.float32)
    wgt = np.zeros((1, 1, h, w), dtype=np.float32)

    w2 = _gauss2d(ph, pw, sigma_scale)[None, None]
    step_h = max(1, int(ph * (1.0 - overlap)))
    step_w = max(1, int(pw * (1.0 - overlap)))
    starts_h, starts_w = _monai_starts([h, w], [ph, pw], [step_h, step_w])

    windows = []
    coords = []
    for yy in starts_h:
        for xx in starts_w:
            windows.append(im[:, :, yy:yy + ph, xx:xx + pw])  # always inside
            coords.append((yy, xx))

    if windows:
        batched = np.concatenate(windows, axis=0)   # (N, C, ph, pw)
        outs = _invoke(runner, input_name, batched)  # (N, OC, ph, pw)
        for n, (yy, xx) in enumerate(coords):
            acc[:, :, yy:yy + ph, xx:xx + pw] += outs[n][None] * w2
            wgt[:, :, yy:yy + ph, xx:xx + pw] += w2
    return acc / np.maximum(1e-8, wgt)


def _run_plane(runner, input_name, vol, output, output_channels, spatial, plane,
               progress, pstart, pend):
    """Fill ``output`` (1, OC, X, Y, Z) by running the 2D model per slice."""
    ph, pw = spatial.get("patch_size", [256, 256])[:2]
    overlap = float(spatial.get("overlap", 0.0))
    sigma_scale = float(spatial.get("sigma_scale", 0.5))
    c = vol.shape[1]
    if plane == "axial":
        n = vol.shape[4]
        for i in range(n):
            out = _sw2d(runner, input_name, vol[:, :, :, :, i], output_channels, ph, pw, overlap, sigma_scale)
            output[:, :, :, :, i] = out
            progress(pstart + (pend - pstart) * (i + 1) / n)
    elif plane == "coronal":
        n = vol.shape[3]
        for i in range(n):
            out = _sw2d(runner, input_name, vol[:, :, :, i, :], output_channels, ph, pw, overlap, sigma_scale)
            output[:, :, :, i, :] = out
            progress(pstart + (pend - pstart) * (i + 1) / n)
    elif plane == "sagittal":
        n = vol.shape[2]
        for i in range(n):
            out = _sw2d(runner, input_name, vol[:, :, i, :, :], output_channels, ph, pw, overlap, sigma_scale)
            output[:, :, i, :, :] = out
            progress(pstart + (pend - pstart) * (i + 1) / n)
    else:
        raise ValueError("unknown plane %r" % plane)


def _run_slice(runner, input_name, vol, output_channels, progress, spatial):
    """Apply a 2D model to a 3D volume, optionally ensembling multiple planes."""
    planes = spatial.get("ensemble_planes") or [spatial.get("plane", "axial")]
    n_planes = len(planes)
    results = []
    for k, plane in enumerate(planes):
        out = np.zeros((1, output_channels) + vol.shape[2:], dtype=np.float32)
        p0 = 0.15 + k * (0.7 / n_planes)
        p1 = p0 + 0.7 / n_planes
        _run_plane(runner, input_name, vol, out, output_channels, spatial, plane,
                   progress, p0, p1)
        results.append(out)

    if n_planes == 1:
        return results[0]

    mode = spatial.get("ensemble", "fourier_burst")
    stacked = np.concatenate(results, axis=0)  # (n, OC, X, Y, Z)
    if mode == "mean":
        return stacked.mean(axis=0, keepdims=True)
    if mode == "fourier_burst":
        fba = _fourier_burst(stacked, float(spatial.get("ensemble_p", 5.0)))
        return fba[None]
    raise ValueError("unknown ensemble mode %r" % mode)


def _fourier_burst(stacked, p):
    vs = [np.fft.rfftn(v) for v in np.moveaxis(stacked, 0, 0)]
    power = [np.abs(v) ** p for v in vs]
    denom = np.sum(power, axis=0)
    ws = [pw / denom for pw in power]
    out_hat = sum(w * v for w, v in zip(ws, vs))
    return np.fft.irfftn(out_hat, s=stacked.shape[1:]).astype(np.float32)


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
