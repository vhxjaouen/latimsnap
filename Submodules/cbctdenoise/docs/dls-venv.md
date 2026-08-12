# DLS / I2I Python environment (uv recipe)

LaTIM-SNAP's deep-learning servers are launched by spawning a python
executable from a virtual environment that you point the app at
(Preferences → Deep Learning → server → venv/python path). One venv can host
everything (segmentation `itksnap_dls`, image-to-image `latimsnap_i2i`, and
the CBCT→CT model).

## Create the venv with `uv`

```bash
# from the repository root
uv venv .venv --python 3.11

# I2I server + CBCT denoising model (serving path: torch, onnxruntime)
uv pip install -e Python/latimsnap_i2i
uv pip install -e Submodules/cbctdenoise[serving]

# OPTIONAL: interactive segmentation (nnInteractive) in the same venv
uv pip install "itksnap-dls>=0.0.6"

# OPTIONAL: research extras for cbctdenoise (retraining / .h5 -> ONNX export)
uv pip install -e Submodules/cbctdenoise[train]
```

## Point LaTIM-SNAP at it

Set the server's **venv path** to `<repo>/.venv` (and let the app use
`.venv/bin/python`), then connect. The I2I server will spawn
`python -m latimsnap_i2i` from that venv.

## Make the CBCT→CT model appear in the app

The I2I server reads model specs from its `--models-dir` (defaults to the
`models/` folder inside the installed `latimsnap_i2i` package). Point it at
the cbctdenoise specs:

* either copy `Submodules/cbctdenoise/specs/cbct2ct.{json,onnx}` into the
  package's `models/` folder, or
* launch with `python -m latimsnap_i2i --models-dir
  <repo>/Submodules/cbctdenoise/specs`.

Then `Tools → Image-to-Image` lists **CBCT → CT denoising (RRDB VJ016)**.

## pip fallback

Replace `uv pip install -e` with `pip install -e` (create the venv with
`python -m venv .venv`). Same packages.
