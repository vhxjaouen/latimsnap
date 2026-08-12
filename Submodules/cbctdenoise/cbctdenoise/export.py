"""Export a research checkpoint to a standard serving artifact.

The research notebook stores weights as ``torch.load(path)['model']`` for the
full :class:`Pix2PixRRDB`. This module treats that ``.h5`` purely as an input
and converts it into a clean, standard artifact:

  * **ONNX** (primary) — ``cbct2ct.onnx``, portable, no torch required at
    serving time;
  * **normalized .pt** (fallback) — bare ``state_dict`` of the generator.

Export performs *loud* validation: it reports the checkpoint's top-level keys,
missing/unexpected keys, and shape mismatches, so a ``.h5`` that does not
match the declared architecture fails here rather than silently at serving.
"""

from __future__ import annotations

import sys
from typing import Optional

import torch

from cbctdenoise.config import CBCTDenoiseConfig


def _report(checkpoint_keys, model_keys, strict) -> None:
    model_keys = set(model_keys)
    missing = [k for k in model_keys if k not in checkpoint_keys]
    unexpected = [k for k in checkpoint_keys if k not in model_keys]
    if missing:
        print(f"[export] WARNING: {len(missing)} missing model key(s), e.g. {missing[:5]}")
    if unexpected:
        print(f"[export] NOTE: {len(unexpected)} unexpected checkpoint key(s), e.g. {unexpected[:5]}")
    if strict and missing:
        raise RuntimeError(
            "checkpoint does not match the declared architecture: "
            f"{len(missing)} missing key(s)")


def _infer_arch(state):
    """Infer generator architecture from the checkpoint's state_dict keys.

    Returns kwargs for :class:`~cbctdenoise.models.rrdb.RRDBGenerator`, so a
    `.h5` is exported without relying on the (heavy) research stack and without
    assuming the architecture dimensions.
    """
    arch = {"in_channels": 1, "out_channels": 1, "num_rrdb": 9,
            "num_dense_layers": 2, "growth_rate": 32, "feature_channels": 64,
            "use_se": False}
    for k, v in state.items():
        p = k.split(".")
        if v.dim() < 2:
            continue  # skip bias/running stats (1-D)
        if p[0] == "conv_in":
            arch["in_channels"] = v.shape[1]
            arch["feature_channels"] = v.shape[0]
        elif p[0] == "conv_out":
            arch["out_channels"] = v.shape[0]
        elif p[0] == "rrdb_trunk":
            arch["num_rrdb"] = max(arch["num_rrdb"], int(p[1]) + 1)
            if "dense_layers" in p and "conv2" in p:
                arch["growth_rate"] = v.shape[0]
                arch["num_dense_layers"] = max(arch["num_dense_layers"],
                                               int(p[p.index("dense_layers") + 1]) + 1)
        if p[0].startswith("se_"):
            arch["use_se"] = True
    print(f"[export] inferred architecture: {arch}")
    return arch


def _build_lean_generator(arch):
    from cbctdenoise.models.rrdb import RRDBGenerator  # noqa: PLC0415
    return RRDBGenerator(
        in_channels=arch["in_channels"],
        out_channels=arch["out_channels"],
        num_rrdb=arch["num_rrdb"],
        num_dense_layers=arch["num_dense_layers"],
        growth_rate=arch["growth_rate"],
        feature_channels=arch["feature_channels"],
        use_se=arch["use_se"],
    )


def load_generator(checkpoint_path: str, cfg: CBCTDenoiseConfig):
    """Load the generator from a research checkpoint; returns net + diagnostics."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
        print(f"[export] checkpoint top-level keys: {list(checkpoint.keys())}")
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        # bare state_dict
        state = checkpoint

    # Normalize a possible DDP 'module.' prefix
    if any(k.startswith("module.") for k in state):
        print("[export] stripping 'module.' prefix (DDP checkpoint)")
        state = {k[len("module."):]: v for k, v in state.items()}

    # Generator sub-state: keep only keys under 'generator_A_to_B.' ; a bare
    # generator state_dict (no prefix) is also accepted.
    gen_keys = {k[len("generator_A_to_B."):]: v
                for k, v in state.items() if k.startswith("generator_A_to_B.")}
    if gen_keys:
        state = gen_keys
        print(f"[export] extracting generator keys (n={len(state)})")

    return state


def export_onnx(checkpoint_path: str, onnx_path: str,
                cfg: Optional[CBCTDenoiseConfig] = None, opset: int = 17) -> None:
    cfg = cfg or CBCTDenoiseConfig()
    state = load_generator(checkpoint_path, cfg)
    arch = _infer_arch(state)

    gen = _build_lean_generator(arch)
    model_keys = set(gen.state_dict().keys())
    _report(set(state.keys()), model_keys, strict=False)
    missing = [k for k in model_keys if k not in state]
    if missing:
        raise RuntimeError(
            "checkpoint is missing generator weights required by the "
            f"architecture: {missing[:8]}")

    gen.load_state_dict(state, strict=True)
    gen.eval()

    dummy = torch.zeros(1, arch["in_channels"], cfg.patch_size[0], cfg.patch_size[1])
    torch.onnx.export(
        gen, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "H", 3: "W"}, "output": {2: "H", 3: "W"}},
        opset_version=opset,
        # Legacy exporter keeps weights inline (single self-contained .onnx).
        dynamo=False,
    )
    print(f"[export] wrote ONNX: {onnx_path}")


def export_pt(checkpoint_path: str, pt_path: str,
              cfg: Optional[CBCTDenoiseConfig] = None) -> None:
    cfg = cfg or CBCTDenoiseConfig()
    state = load_generator(checkpoint_path, cfg)
    arch = _infer_arch(state)

    # Save the state dict of the serving wrapper (keys prefixed 'generator.')
    # so the latimsnap-i2i torch loader can load it directly.
    from cbctdenoise.models.serving import MRToCTGenerator  # noqa: PLC0415
    serving = MRToCTGenerator(
        in_channels=arch["in_channels"], out_channels=arch["out_channels"],
        num_rrdb_G=arch["num_rrdb"], num_dense_layers_G=arch["num_dense_layers"],
        growth_rate_G=arch["growth_rate"], feature_channels_G=arch["feature_channels"])
    bare = {k[len("generator."):]: v for k, v in serving.state_dict().items()
            if k.startswith("generator.")}
    _report(set(state.keys()), set(bare.keys()), strict=False)
    serving.generator.load_state_dict(
        {k: v for k, v in state.items()}, strict=True)
    torch.save(serving.state_dict(), pt_path)
    print(f"[export] wrote serving state_dict: {pt_path}")


if __name__ == "__main__":
    # Simple CLI fallback (full CLI lives in cli.py)
    if len(sys.argv) < 3:
        print("usage: python -m cbctdenoise.export <checkpoint.h5> <out.onnx>")
        sys.exit(2)
    export_onnx(sys.argv[1], sys.argv[2])
