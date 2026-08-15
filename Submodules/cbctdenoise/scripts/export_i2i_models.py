#!/usr/bin/env python
"""Export the 10 chosen SynthRAD2025 CBCT->CT models to ONNX for the I2I server.

Each model checkpoint is a PyTorch ``{model: OrderedDict}`` (keys prefixed
``generator_A_to_B.``) saved with a ``.h5`` extension. Two generator
topologies are supported:

  * ``rrdb``      - the standard RRDB VJ016 (``num_rrdb=9, ndl=2, gr=32, fc=64``)
  * ``resnet1``   - the 1-channel Triplet-Attention U-Net (``MyUNet1(32)``)

For every entry a self-contained ONNX graph and a matching model spec JSON are
written into ``SPECS_DIR`` so the ``latimsnap_i2i`` server can advertise it.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import torch

# Make cbctdenoise importable when run from its scripts/ directory
HERE = os.path.dirname(os.path.abspath(__file__))
CBCT = os.path.abspath(os.path.join(HERE, ".."))
if CBCT not in sys.path:
    sys.path.insert(0, CBCT)

from cbctdenoise.research import vjnetworks as VJ          # noqa: E402
from cbctdenoise.research.vjnetworks_attn import MyUNet1   # noqa: E402

DEFAULT_SPECS_DIR = os.path.join(CBCT, "specs")

# name guard -> (kind, display-short-label)
def arch(kind):
    return kind


def strip_state(sd):
    """Pull the generator sub-dict out of a checkpoint and drop its prefix."""
    out = {}
    for k, v in sd.items():
        if k.startswith("generator_A_to_B."):
            out[k[len("generator_A_to_B."):]] = v
    return out


def build_generator(kind):
    if kind == "rrdb":
        return VJ.RRDBGenerator(
            in_channels=1, out_channels=1, num_rrdb=9,
            num_dense_layers=2, growth_rate=32, feature_channels=64)
    if kind == "resnet1":
        return MyUNet1(in_ch=32, act=True)
    raise ValueError(f"unknown model kind {kind!r}")


def load_checkpoint(path, kind):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    sd = strip_state(sd)
    gen = build_generator(kind)
    gen.load_state_dict(sd, strict=True)
    gen.eval()
    return gen


def parse_metrics(path):
    b = os.path.basename(path)
    psnr = re.search(r"_(\d+\.\d+)dB", b)
    mae = re.search(r"MAE[_\s]*([0-9.]+)", b)
    return (float(psnr.group(1)) if psnr else None,
            float(mae.group(1).rstrip(".")) if mae else None)


def make_spec(entry, onnx_path):
    psnr, mae = entry["psnr"], entry["mae"]
    metrics = []
    if psnr is not None:
        metrics.append(f"{psnr:.2f}dB")
    if mae is not None:
        metrics.append(f"MAE {mae:.1f}")
    display = f"{entry['name']}"
    if metrics:
        display += f" [{','.join(metrics)}]"
    return {
        "id": entry["id"],
        "name": display,
        "framework": "onnx",
        "model": {"onnx_path": os.path.basename(onnx_path)},
        "input_channels": 1,
        "output_channels": 1,
        "preprocessing": {"type": "minmax", "min": -1000, "max": 3000},
        "postprocessing": {"type": "denorm_minmax", "min": -1000, "max": 3000},
        "spatial": {
            "mode": "slice",
            "dim": 2,
            "patch_size": [256, 256],
            "overlap": 0.25,
            "plane": "axial",
            "padding": "replicate",
            "ensemble_planes": ["axial"],
            "ensemble": "none",
            "ensemble_p": 5.0,
            "sigma_scale": 0.5,
        },
        "output_dtype": "float32",
    }


def export_one(entry, specs_dir, device="cpu"):
    src = entry["ckpt"]
    gen = load_checkpoint(src, entry["kind"])
    psnr, mae = parse_metrics(src)
    entry["psnr"], entry["mae"] = psnr, mae

    onnx_path = os.path.join(specs_dir, entry["id"] + ".onnx")
    dummy = torch.randn(1, 1, 256, 256)
    torch.onnx.export(
        gen.to(device), dummy, onnx_path,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "N", 2: "H", 3: "W"},
                      "output": {0: "N", 2: "H", 3: "W"}},
        opset_version=17)

    # Re-embed any external weight data so each model ships as ONE .onnx file
    import onnx
    model = onnx.load(onnx_path)
    for init in model.graph.initializer:
        init.ClearField("data_location")
    onnx.save_model(model, onnx_path)
    data = onnx_path + ".data"
    if os.path.exists(data):
        os.remove(data)

    spec = make_spec(entry, onnx_path)
    with open(os.path.join(specs_dir, entry["id"] + ".json"), "w") as fh:
        json.dump(spec, fh, indent=2)
    return onnx_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/amadeo/DATA/PUBLIC/SynthRAD2025/RESULTS")
    ap.add_argument("--specs", default=DEFAULT_SPECS_DIR)
    args = ap.parse_args()
    R = args.results
    os.makedirs(args.specs, exist_ok=True)

    def rglob(pat):
        import glob
        hits = glob.glob(os.path.join(R, pat))
        return hits[0] if hits else None

    resnet_dir = rglob(
        "2025-NECSR_ResNetSynthRAD_VJ016_HU1000to3000_ndl2_gr32_fc64_ep100_*aNGF_0.25_*")
    assert resnet_dir, "ResNet experiment dir not found"

    def best_resnet(ep):
        import glob
        cand = [f for f in glob.glob(os.path.join(resnet_dir, "*.h5"))
                if re.search(r"_e%04d_" % ep, os.path.basename(f))]
        assert cand, f"no ResNet ckpt for epoch {ep}"
        return max(cand, key=lambda f: float(
            re.search(r"_(\d+\.\d+)dB", os.path.basename(f)).group(1)))

    def rrdb_ckpt(*subs):
        """Best-scored RRDB checkpoint in the dir whose name contains all subs."""
        import glob
        hits = [d for d in glob.glob(os.path.join(R, "2025-NECSR_RRDB_VJ016_*"))
                if os.path.isdir(d) and all(s in d for s in subs)]
        assert hits, f"RRDB config {subs!r} not found"
        d = hits[0]
        fs = [f for f in glob.glob(os.path.join(d, "*.h5"))
              if re.search(r"_(\d+\.\d+)dB", os.path.basename(f))]
        assert fs, f"no scored RRDB ckpt in {d}"
        return max(fs, key=lambda f: float(
            re.search(r"_(\d+\.\d+)dB", os.path.basename(f)).group(1)))

    manifest = [
        # ResNet Triplet-Attention, best checkpoint per epoch
        dict(kind="resnet1", id="cbct2ct_resnet_a025_e18",
             name="CBCT->CT ResNet aNGF0.25 (e18)",
             ckpt=best_resnet(18)),
        dict(kind="resnet1", id="cbct2ct_resnet_a025_e07",
             name="CBCT->CT ResNet aNGF0.25 (e07)",
             ckpt=best_resnet(7)),
        dict(kind="resnet1", id="cbct2ct_resnet_a025_e12",
             name="CBCT->CT ResNet aNGF0.25 (e12)",
             ckpt=best_resnet(12)),
        dict(kind="resnet1", id="cbct2ct_resnet_a025_e17",
             name="CBCT->CT ResNet aNGF0.25 (e17)",
             ckpt=best_resnet(17)),
        dict(kind="resnet1", id="cbct2ct_resnet_a025_e09",
             name="CBCT->CT ResNet aNGF0.25 (e09)",
             ckpt=best_resnet(9)),
        # RRDB VJ016 variants
        dict(kind="rrdb", id="cbct2ct_rrdb_im600_a05",
             name="CBCT->CT RRDB im600x50 aNGF0.05",
             ckpt=rrdb_ckpt("im600x50", "aNGF_0.05")),
        dict(kind="rrdb", id="cbct2ct_rrdb_im600_a20",
             name="CBCT->CT RRDB im600x50 aNGF0.20",
             ckpt=rrdb_ckpt("im600x50", "aNGF_0.20")),
        dict(kind="rrdb", id="cbct2ct_rrdb_a25",
             name="CBCT->CT RRDB ndl2 aNGF0.25",
             ckpt=rrdb_ckpt("aNGF_0.25")),
        dict(kind="rrdb", id="cbct2ct_rrdb_L1100_a15",
             name="CBCT->CT RRDB ndl2 L1 100 aNGF0.15",
             ckpt=rrdb_ckpt("aNGF_0.15")),
        # Legacy RRDB run
        dict(kind="rrdb", id="cbct2ct_rrdb_vj004",
             name="CBCT->CT RRDB VJ004 /RUN007",
             ckpt=rglob("VJ004_ALL_*.h5")),
    ]
    assert all(m["ckpt"] for m in manifest), "a manifest entry resolved to None"

    for m in manifest:
        print(f"[export] {m['id']} <- {os.path.basename(m['ckpt'])}")
        p = export_one(m, args.specs)
        print(f"   -> {os.path.basename(p)}  PSNR={m['psnr']} MAE={m['mae']}")
    print("\nDone. Specs + onnx written into", args.specs)


if __name__ == "__main__":
    main()