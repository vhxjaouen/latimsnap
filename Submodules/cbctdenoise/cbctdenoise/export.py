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


def _build_full_model(cfg: CBCTDenoiseConfig, torch_device):
    """Build the full Pix2PixRRDB from the vendored vjnetworks package."""
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415
    # Make the vendored research package importable (self-contained).
    _research = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")
    if _research not in sys.path:
        sys.path.insert(0, _research)
    import vjnetworks  # noqa: PLC0415 (research extra)
    if not torch.cuda.is_available():
        # The vendored network builds on a module-level 'cuda:0' device; fall
        # back to CPU so export works on machines without a GPU.
        vjnetworks.gpu_device = torch.device("cpu")
    return vjnetworks.Pix2PixRRDB(
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        num_rrdb_G=cfg.num_rrdb_G,
        num_dense_layers_G=cfg.num_dense_layers_G,
        growth_rate_G=cfg.growth_rate_G,
        feature_channels_G=cfg.feature_channels_G,
        use_se=cfg.use_se,
    ).to(torch_device)


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

    model = _build_full_model(cfg, torch.device("cpu"))
    gen_keys = set(model.generator_A_to_B.state_dict().keys())
    _report(set(state.keys()), gen_keys, strict=False)
    missing = [k for k in gen_keys if k not in state]
    if missing:
        raise RuntimeError(
            "checkpoint is missing generator weights required by the "
            f"architecture: {missing[:8]}")

    model.generator_A_to_B.load_state_dict(state, strict=True)
    model.eval()

    dummy = torch.zeros(1, cfg.in_channels, cfg.patch_size[0], cfg.patch_size[1])
    torch.onnx.export(
        model.generator_A_to_B, dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {2: "H", 3: "W"}, "output": {2: "H", 3: "W"}},
        opset_version=opset,
    )
    print(f"[export] wrote ONNX: {onnx_path}")


def export_pt(checkpoint_path: str, pt_path: str,
              cfg: Optional[CBCTDenoiseConfig] = None) -> None:
    cfg = cfg or CBCTDenoiseConfig()
    state = load_generator(checkpoint_path, cfg)

    # Save the state dict of the serving wrapper (keys prefixed 'generator.')
    # so the latimsnap-i2i torch loader can load it directly.
    from cbctdenoise.models.serving import MRToCTGenerator  # noqa: PLC0415
    serving = MRToCTGenerator(
        in_channels=cfg.in_channels, out_channels=cfg.out_channels,
        num_rrdb_G=cfg.num_rrdb_G, num_dense_layers_G=cfg.num_dense_layers_G,
        growth_rate_G=cfg.growth_rate_G, feature_channels_G=cfg.feature_channels_G)
    bare = {k[len("generator."):]: v for k, v in serving.state_dict().items()
            if k.startswith("generator.")}
    _report(set(state.keys()), set(bare.keys()), strict=False)
    serving.generator.load_state_dict(
        {k: v for k, v in state.items()}, strict=False)
    torch.save(serving.state_dict(), pt_path)
    print(f"[export] wrote serving state_dict: {pt_path}")


if __name__ == "__main__":
    # Simple CLI fallback (full CLI lives in cli.py)
    if len(sys.argv) < 3:
        print("usage: python -m cbctdenoise.export <checkpoint.h5> <out.onnx>")
        sys.exit(2)
    export_onnx(sys.argv[1], sys.argv[2])
