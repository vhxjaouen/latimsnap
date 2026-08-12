# cbctdenoise — CBCT → CT (RRDB, SynthRAD2025 Task 2)

A clean, self-contained factorization of the research notebook
`NECSRv3_RRDB_SynthRAD2025_Task2_VJ016_AB_TH.ipynb`, at the code standard of
**LaTIM-SNAP** (typed, documented, config-driven, tested, no hardcoded paths).

It is a **2D** RRDB generator (single-channel) applied **per-slice** to a 3D
volume, optionally ensembled across the three anatomical planes (mean or
Fourier-burst accumulation), producing a denoised CT in the **source grid**.

## Layout

```
cbctdenoise/
  config.py       # CBCTDenoiseConfig dataclass (was the notebook's Options)
  preprocess.py   # HU <-> [-1,1] scaling (ScaleIntensityRanged faithful)
  models/rrdb.py  # lean, torch-only RRDB generator (serving path)
  models/serving.py  # MRToCTGenerator wrapper for the generic i2i loader
  infer.py        # 2D-per-slice inference over a 3D volume + ensembling
  ensembling.py   # mean / Fourier Burst Accumulation
  export.py       # research .h5 -> standard ONNX (or normalized .pt)
  verify.py       # dummy-volume validation of an artifact
  data.py, losses.py, train.py   # research/training path (needs [train] extra)
  cli.py          # console scripts
specs/cbct2ct.json   # latimsnap-i2i model spec
tests/              # unit tests
```

## Serving in LaTIM-SNAP (`latimsnap-i2i`)

1. **Export** the research checkpoint to ONNX:
   ```bash
   cbctdenoise-export /path/to/...e0094_38.56dB.h5 specs/cbct2ct.onnx
   # checkpoint top-level keys are reported; missing/unexpected keys warn loudly
   ```
2. **Verify** it:
   ```bash
   cbctdenoise-verify specs/cbct2ct.onnx   # runs a dummy volume
   ```
3. Point the LaTIM-SNAP image-to-image server at these specs
   (`--models-dir <this repo>/Submodules/cbctdenoise/specs`) or copy
   `cbct2ct.json` + `cbct2ct.onnx` into the server's models dir. The model
   appears as **CBCT → CT denoising (RRDB VJ016)** in `Tools → Image-to-Image`.

The spec defaults to **axial-only** inference; edit `spec.json` to enable
3-plane + FBA:
```jsonc
"spatial": {
  "mode": "slice", "dim": 2, "patch_size": [256,256], "overlap": 0.66,
  "plane": "axial", "padding": "replicate",
  "ensemble_planes": ["axial","coronal","sagittal"],
  "ensemble": "fourier_burst", "ensemble_p": 5.0
}
```

## Standalone use
```bash
cbctdenoise-infer  cbct2ct.onnx  input_cbct.nii.gz  output_ct.nii.gz
cbctdenoise-infer  cbct2ct.onnx  input.nii.gz  out.nii.gz --ensemble-planes axial coronal sagittal --ensemble fourier_burst
cbctdenoise-evaluate pred.nii.gz ref.nii.gz
```
Note: the exported ONNX has dynamic height/width axes, so any patch size works.

## Research / training
Requires the heavy extras (MONAI generative, kornia):
```bash
uv pip install -e Submodules/cbctdenoise[train]
```
`cbctdenoise-train --results-dir ./runs --epochs 100` is a cleaned skeleton of
the GAN/NGF/MSSSIM loop; the vendored `vjnetworks` package provides the full
`Pix2PixRRDB` used by `export.py` for exact weight loading.

## Testing
```bash
pytest Submodules/cbctdenoise/tests
```
