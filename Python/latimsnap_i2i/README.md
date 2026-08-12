# latimsnap-i2i

GPL-licensed, in-tree image-to-image (I2I) deep learning server for
**LaTIM-SNAP**. It runs an encoder model (ONNX or PyTorch) in the background
and returns a new image (e.g. MR to CT synthesis) over HTTP. The server is
launched by LaTIM-SNAP itself as a local subprocess:

```
python -m latimsnap_i2i --port <port> [--models-dir <dir>] [--setup-only]
```

## Wire protocol

Mirrors the existing `itksnap_dls` segmentation server so LaTIM-SNAP reuses its
gzip/JSON encode helpers.

| Endpoint | Purpose |
|---|---|
| `GET /status` | Health/version + list of available models |
| `GET /start_session` | Open a session; returns `{"session_id": ...}` |
| `POST /upload_raw/{sid}` | Upload source image (multipart `file`=gzip raw voxels, `metadata`=JSON) |
| `GET /transfer_models` | List model specs |
| `POST /run_transfer/{sid}?model=<id>` | Start a background transfer; returns `{"job_id": ...}` |
| `GET /transfer_progress/{sid}?job=<id>` | `{"status","progress"(0..1),"message"}` |
| `GET /transfer_result/{sid}?job=<id>` | `{"result": base64(gzip(raw)), "metadata": {...}}` |

## Model specs

A model spec is a JSON file in the `--models-dir`. See
`models/identity.json` and `models/example_mr_to_ct.json`.

* `framework`: `onnx` | `torch` | `identity` (built-in smoke test)
* `model.onnx_path`: path to an `.onnx` file (framework `onnx`)
* `model.module` / `model.class` / `model.kwargs` / `model.weights`:
  architecture to instantiate (framework `torch`)
* `input_channels`, `output_channels`
* `preprocessing.type`: `identity` | `minmax` | `percentile` | `normalize`
* `postprocessing.type`: `identity` | `clip` | `denorm_minmax`
* `spatial.mode`: `whole` (model accepts arbitrary shape) or `patches`
  (with `patch_size` + `overlap` for seamless stitching / limited memory)

## Development

```bash
python -m pip install -e ./Python/latimsnap_i2i
python -m latimsnap_i2i --port 8912 --models-dir Python/latimsnap_i2i/models
python -m pytest Python/latimsnap_i2i/tests
```
