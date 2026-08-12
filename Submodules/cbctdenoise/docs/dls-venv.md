# DLS / I2I Python environment (uv recipe)

LaTIM-SNAP's deep-learning servers are launched by spawning a python
executable from a virtual environment that you point the app at
(Preferences → Deep Learning → server → venv/python path). One venv can host
everything (segmentation `itksnap_dls`, image-to-image `latimsnap_i2i`, and
the CBCT→CT model).

This page is the **reproducible recipe**: given the repository and the model
weights (a `.h5`), anyone can rebuild the environment, export the model to
ONNX, and serve it — without needing the original GPU/training machine.

## Prerequisites
- `uv` (https://docs.astral.sh/uv/) — `uv --version`
- the weights file, e.g. `LaTIM-weights/2025-NECSR_RRDB_VJ016_...e0092_38.28dB_MAE_27.9.h5`
  (not committed; drop it into `LaTIM-weights/`)

## 1. Create the venv (Python 3.12)

The notebook ran on Python 3.12; torch/onnx have well-tested 3.12 wheels.

```bash
cd <repo>
uv venv .venv --python 3.12
```

## 2. Install requirements

Two requirement files are provided in `Submodules/cbctdenoise/`:

| file | purpose |
|---|---|
| `requirements-serving.txt` | minimal: export to ONNX + serve (`torch`, `onnx`, `onnxscript`, `onnxruntime`, `numpy`, `nibabel`) |
| `requirements-notebook.txt` | full research env faithful to the notebook (`-r serving` + `monai`, `monai-generative`, `kornia`, `scipy`, `scikit-image`, `scikit-learn`, `pandas`, `matplotlib`, `imageio`, `natsort`, `tqdm`, `pillow`, `opencv-python-headless`) |

### CPU-only box (recommended to start)
```bash
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv -r Submodules/cbctdenoise/requirements-serving.txt
```
(Installing `torch` first from the CPU index avoids pulling the huge CUDA
wheel; the subsequent `-r` resolves the rest from PyPI.)

### GPU box / full research environment
```bash
uv pip install --python .venv -r Submodules/cbctdenoise/requirements-notebook.txt
```

### Install the model package itself
```bash
uv pip install --python .venv -e Submodules/cbctdenoise[serving]
```

## 3. Export the weights to ONNX

```bash
.venv/bin/cbctdenoise-export \
  --checkpoint LaTIM-weights/2025-NECSR_RRDB_VJ016_...e0092_38.28dB_MAE_27.9.h5 \
  --out Submodules/cbctdenoise/specs/cbct2ct.onnx
```

The exporter prints the checkpoint's structure and **auto-infers the
architecture** (`num_rrdb`/`num_dense_layers`/`growth_rate`/`feature_channels`)
from the state dict, then validates every weight (`strict=True`). A `.h5` that
does not match an RRDB generator fails loudly here.

## 4. Verify

```bash
.venv/bin/cbctdenoise-verify Submodules/cbctdenoise/specs/cbct2ct.onnx
# => {'ok': True, 'shape': (1, 16, 24, 32), 'expected': (1, 16, 24, 32), ...}
```
Small patch/volume for speed; the ONNX graph has dynamic H/W so the real
256×256 inference path is identical.

## 5. Point LaTIM-SNAP at it

* Set the server's **venv path** to `<repo>/.venv`.
* The I2I server reads model specs from its `--models-dir` (the *Installed
  package's* `models/`). Point it at the cbctdenoise specs:
  `python -m latimsnap_i2i --models-dir <repo>/Submodules/cbctdenoise/specs`
  (or copy `cbct2ct.json` + `cbct2ct.onnx` into that `models/` folder).
* `Tools → Image-to-Image` now lists **CBCT → CT denoising (RRDB VJ016)**.

Edge-to-edge this needs `latimsnap_i2i` importable in the same venv:
```bash
uv pip install --python .venv -e Python/latimsnap_i2i
```

## pip fallback
Same commands with `python -m venv .venv` and `pip install`.