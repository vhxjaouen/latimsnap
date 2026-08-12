"""Command-line entry points (console scripts)."""

from __future__ import annotations

import argparse

import numpy as np

from cbctdenoise.config import CBCTDenoiseConfig


def _add_common(parser):
    parser.add_argument("--checkpoint", required=True, help="research .h5 checkpoint")
    parser.add_argument("--out", required=True, help="output artifact path")
    parser.add_argument("--fc", type=int, default=64)
    parser.add_argument("--num-rrdb", type=int, default=9)
    parser.add_argument("--ndl", type=int, default=2)
    parser.add_argument("--gr", type=int, default=32)


def export_main(argv=None):
    p = argparse.ArgumentParser("cbctdenoise-export",
                                description="research .h5 -> ONNX/pt serving artifact")
    _add_common(p)
    p.add_argument("--format", choices=["onnx", "pt"], default="onnx")
    a = p.parse_args(argv)
    cfg = CBCTDenoiseConfig(num_rrdb_G=a.num_rrdb, num_dense_layers_G=a.ndl,
                            growth_rate_G=a.gr, feature_channels_G=a.fc)
    if a.format == "onnx":
        from cbctdenoise.export import export_onnx
        export_onnx(a.checkpoint, a.out, cfg)
    else:
        from cbctdenoise.export import export_pt
        export_pt(a.checkpoint, a.out, cfg)


def verify_main(argv=None):
    p = argparse.ArgumentParser("cbctdenoise-verify",
                                description="run a dummy volume through an artifact")
    p.add_argument("artifact", help=".onnx or .pt")
    a = p.parse_args(argv)
    from cbctdenoise.verify import make_onnx_runner, make_torch_runner, verify_artifact
    if a.artifact.endswith(".onnx"):
        runner = make_onnx_runner(a.artifact)
    else:
        import torch
        from cbctdenoise.models.serving import MRToCTGenerator  # noqa: PLC0415
        gen = MRToCTGenerator(); gen.load_state_dict(torch.load(a.artifact, map_location="cpu"))
        runner = make_torch_runner(gen, "cpu")
    r = verify_artifact(runner)
    print(r)
    return 0 if r["ok"] else 1


def infer_main(argv=None):
    p = argparse.ArgumentParser("cbctdenoise-infer",
                                description="CBCT volume -> CT volume (NIfTI)")
    p.add_argument("artifact", help=".onnx or .pt")
    p.add_argument("input_nii")
    p.add_argument("output_nii")
    p.add_argument("--plane", default="axial")
    p.add_argument("--patch", nargs=2, type=int, default=[256, 256])
    p.add_argument("--overlap", type=float, default=0.66)
    p.add_argument("--ensemble-planes", nargs="+", default=["axial"])
    p.add_argument("--ensemble", default="fourier_burst")
    a = p.parse_args(argv)

    import nibabel as nib
    from cbctdenoise import preprocess
    from cbctdenoise.infer import apply_2d_model_to_volume

    if a.artifact.endswith(".onnx"):
        from cbctdenoise.verify import make_onnx_runner
        runner = make_onnx_runner(a.artifact)
    else:
        import torch
        from cbctdenoise.models.serving import MRToCTGenerator  # noqa: PLC0415
        gen = MRToCTGenerator(); gen.load_state_dict(torch.load(a.artifact, map_location="cpu"))
        from cbctdenoise.verify import make_torch_runner
        runner = make_torch_runner(gen, "cpu")

    nii = nib.load(a.input_nii)
    vol = nii.get_fdata().astype(np.float32)[None, ...]   # (C=1, X, Y, Z)
    fwd, inv = preprocess.make_scaler(*CBCTDenoiseConfig().hu_clip)
    config = {"plane": a.plane, "patch_size": tuple(a.patch), "overlap": a.overlap,
              "ensemble_planes": a.ensemble_planes, "ensemble": a.ensemble,
              "output_channels": 1}
    out = inv(apply_2d_model_to_volume(runner, fwd(vol), config))
    nib.save(nib.Nifti1Image(out[0], nii.affine, nii.header), a.output_nii)
    print(f"wrote {a.output_nii}")


def evaluate_main(argv=None):
    p = argparse.ArgumentParser("cbctdenoise-evaluate", description="PSNR / MAE / SSIM")
    p.add_argument("pred")
    p.add_argument("ref")
    a = p.parse_args(argv)
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity  # noqa: PLC0415
    import nibabel as nib
    import numpy as np
    pred = nib.load(a.pred).get_fdata().astype(np.float32)
    ref = nib.load(a.ref).get_fdata().astype(np.float32)
    mae = float(np.mean(np.abs(pred - ref)))
    psnr = float(peak_signal_noise_ratio(ref, pred, data_range=ref.max() - ref.min()))
    ssim = float(structural_similarity(ref, pred, data_range=ref.max() - ref.min()))
    print(f"PSNR={psnr:.2f} dB  MAE={mae:.2f}  SSIM={ssim:.4f}")
    return 0


def train_main(argv=None):
    p = argparse.ArgumentParser("cbctdenoise-train", description="research training loop")
    p.add_argument("--results-dir", default=".")
    p.add_argument("--epochs", type=int, default=100)
    a = p.parse_args(argv)
    from cbctdenoise.config import CBCTDenoiseConfig
    from cbctdenoise.train import train
    cfg = CBCTDenoiseConfig(num_epochs=a.epochs, results_dir=a.results_dir)
    train(cfg, a.results_dir)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(verify_main())