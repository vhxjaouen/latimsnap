"""Verify a served CBCT->CT model on a dummy volume.

Runs the exported artifact (ONNX, or a torch wrapper) through
:func:`cbctdenoise.infer.apply_2d_model_to_volume` and asserts the output
shape/components/finiteness. This catches dirty-checkpoint/architecture
mismatches before the model ever reaches the LaTIM-SNAP app.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from cbctdenoise import preprocess
from cbctdenoise.infer import apply_2d_model_to_volume


def make_onnx_runner(onnx_path: str):
    import onnxruntime as ort  # noqa: PLC0415
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]

    def runner(patch: np.ndarray) -> np.ndarray:
        out = sess.run(None, {inp.name: patch[None].astype(np.float32)})[0]
        return out[0]

    return runner


def make_torch_runner(generator, device="cpu"):
    """Runner over a torch generator expecting (B,C,H,W)."""
    import torch  # noqa: PLC0415
    gen = generator.to(device).eval()

    def runner(patch: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.from_numpy(patch[None].astype(np.float32)).to(device)
            return gen(t).detach().cpu().numpy()[0]

    return runner


def verify_artifact(runner: Callable, output_channels: int = 1,
                    in_channels: int = 1, volume_shape=(64, 96, 128),
                    config: Optional[dict] = None,
                    hu_clip=(-1000.0, 3000.0)) -> dict:
    config = config or {"plane": "axial", "patch_size": (256, 256),
                        "overlap": 0.66, "ensemble_planes": ["axial"]}
    rng = np.random.default_rng(0)
    vol = rng.standard_normal((in_channels,) + volume_shape).astype(np.float32)
    fwd, inv = preprocess.make_scaler(*hu_clip)
    vol_norm = fwd(vol)

    out = apply_2d_model_to_volume(runner, vol_norm, config)
    out = inv(out)

    expected = (output_channels,) + volume_shape
    ok = out.shape == expected and np.all(np.isfinite(out))
    report = {"ok": bool(ok), "shape": tuple(out.shape), "expected": expected,
              "range": (float(out.min()), float(out.max()))}
    return report


if __name__ == "__main__":  # smoke
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path")
    a = parser.parse_args()
    r = verify_artifact(make_onnx_runner(a.onnx_path))
    print(r)
    raise SystemExit(0 if r["ok"] else 1)
