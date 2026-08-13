#!/usr/bin/env python
"""Offline end-to-end check for the LaTIM-SNAP CBCT->CT I2I server path.

Runs the exact server inference (ModelRunner.run on RAW HU) on a CBCT volume,
writes the result as a NIfTI you can open in a viewer, and prints rich stats.

Optionally verifies parity with the notebook's MONAI SliceInferer on the same
volume, so you can confirm the server reproduces the reference.

Usage:
  python scripts/e2e_check_cbct.py INPUT_CBCT.nii.gz [OUTPUT.nii.gz]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import nibabel as nib

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "Python/latimsnap_i2i"))

SPEC = os.path.join(REPO, "Submodules", "cbctdenoise", "specs", "cbct2ct.json")


def run_server_path(raw_hu):
    from latimsnap_i2i.model_loader import ModelRunner, ModelSpec

    spec_dict = json.load(open(SPEC))
    spec_dict["_path"] = SPEC
    spec = ModelSpec(spec_dict)
    runner = ModelRunner(spec)
    runner.load()
    out = runner.run(raw_hu[None])  # (1, OC, X, Y, Z) in HU
    return out[0, 0, :, :, :] if out.ndim == 5 else out[0]


def monai_parity(raw_hu, stats_rows):
    """Compare our result to MONAI SliceInferer on the same volume (parity)."""
    try:
        import torch
        from monai.inferers import SliceInferer
        from cbctdenoise.export import load_generator, _infer_arch, _build_lean_generator
        from latimsnap_i2i.model_loader import _apply_preprocess
    except Exception as exc:  # noqa: BLE001
        print("  (parity check skipped:", exc, ")")
        return
    spec = json.load(open(SPEC))
    cpt = os.path.join(REPO, "LaTIM-weights")
    hs = [f for f in os.listdir(cpt) if f.endswith(".h5")]
    if not hs:
        print("  (parity check skipped: no .h5 in LaTIM-weights/)")
        return
    state = load_generator(os.path.join(cpt, hs[0]), None)
    gen = _build_lean_generator(_infer_arch(state)).eval().cuda()
    gen.load_state_dict(state, strict=True)
    norm = _apply_preprocess("minmax", spec["preprocessing"], raw_hu[None])
    inf = SliceInferer(roi_size=(256, 256), sw_batch_size=1, overlap=0.66,
                       mode="gaussian", sigma_scale=0.5, spatial_dim=2,
                       device="cuda", padding_mode="replicate")
    with torch.no_grad():
        mono = inf(torch.from_numpy(np.ascontiguousarray(norm[None])).to("cuda"), gen)
        mono = mono.detach().cpu().numpy()[0, 0]
    mono_hu = ((np.clip(mono, -1, 1) + 1) / 2 * 4000 - 1000)
    d = np.abs(output_hu - mono_hu)
    print(f"  parity vs MONAI SliceInferer: RMS|diff| = {d.mean():.2f} HU")


def report(o, name):
    print(f"== {name} ==")
    print(f"  HU range: {o.min():.0f} .. {o.max():.0f}, mean {o.mean():.0f}")
    print(f"  air (<=-900): {(o<=-900).mean()*100:.1f}% | tissue (0..200): {((o>0)&(o<200)).mean()*100:.1f}% | bone (>=500): {(o>=500).mean()*100:.1f}%")
    for z in np.linspace(5, o.shape[2]-5, 3, dtype=int):
        s = o[:, :, int(z)]
        print(f"  z={int(z):3d}: p1 {np.percentile(s,1):6.0f} p25 {np.percentile(s,25):6.0f} p50 {np.percentile(s,50):6.0f} p95 {np.percentile(s,95):6.0f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    inp = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/opencode/cbct2ct_result.nii.gz"
    n = nib.load(inp)
    raw = n.get_fdata().astype(np.float32)
    print(f"input {inp}: shape {raw.shape}, HU {raw.min():.0f}..{raw.max():.0f}")
    global output_hu
    output_hu = run_server_path(raw)
    report(output_hu, "SERVER PATH (raw -> run)")
    monai_parity(raw, None)
    out_ni = nib.Nifti1Image(output_hu, n.affine, n.header)
    nib.save(out_ni, out)
    print(f"wrote {out}")