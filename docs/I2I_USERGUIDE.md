# LaTIM-SNAP Deep-Learning Servers — User Guide

LaTIM-SNAP ships two "deep-learning server" features that call Python machine
learning models over HTTP:

1. **Interactive Segmentation server** (`itksnap_dls`, wraps nnInteractive) —
   used by the paintbrush in its deep-learning "smart mode".
2. **Image-to-Image (I2I) server** (`latimsnap_i2i`, plugs in any ONNX/torch
   model) — e.g. *CBCT → CT denoising*. Produces a **new image** added as an
   overlay. Run from **Tools → Image-to-Image…**.

Both share the same server setup/connection machinery and the same mental
model: **the C++ client is "dumb" — it sends raw image data to a Python
server and interprets what the server returns. All the cleverness
(preprocessing, inference, postprocessing) lives server-side, configured by a
small JSON "model card".**

```
 [LaTIM-SNAP client: your desktop]          [Python server: where the model runs]
 ─────────────────────────────────          ─────────────────────────────────────
  choose image + model
  start_session →                           (server keeps a session)
  upload_raw: gzip(raw float voxels)  ───►  gunzip → float volume
    + metadata (spacing/origin/direction)   apply spec.preprocessing
        (no scaling here)                   run model (GPU)
  transfer_result: gzip(raw float)  ◄───    apply spec.postprocessing
    + metadata (same geometry)              gzip raw result
  added as new overlay                       |
  (segmentation: merged into the label layer instead)
```

---

## 1. Concepts

### 1.1 It is a client–server architecture

- **Client** = LaTIM-SNAP itself. It never runs PyTorch/ONNX.
- **Server** = a Python process that either
  - is **spawned locally** by LaTIM-SNAP as a subprocess (venv Python), or
  - runs on a **remote machine** (e.g. a GPU box) that LaTIM-SNAP connects to
    over the network (directly or through an SSH tunnel).
- **Where preprocessing & inference happen = wherever the server runs.**
  The client only rounds-trips raw voxel data and geometry metadata.

### 1.2 Task types

A configured server has a **Task Type** (set in the server editor):

| Task Type | Server module | What it returns | Where you see it |
|---|---|---|---|
| Segmentation (nnInteractive) | `itksnap_dls` | a label map | merged into the segmentation layer via the paintbrush |
| Image-to-Image | `latimsnap_i2i` | a new image | new overlay, from **Tools → Image-to-Image…** |

### 1.3 The model card (only for I2I)

I2I servers read JSON **model specs** from a `--models-dir` (default: the
`models/` folder inside the installed `latimsnap_i2i` package). A spec
declares *everything* the server needs — no Python to write for a new model:

```jsonc
{
  "id": "cbct2ct_rrdb",
  "name": "CBCT -> CT denoising (RRDB VJ016)",
  "framework": "onnx",                 // onnx | torch | identity
  "model": { "onnx_path": "cbct2ct.onnx" },
  "input_channels": 1, "output_channels": 1,
  "preprocessing":  { "type": "minmax", "min": -1000, "max": 3000 },
  "postprocessing": { "type": "denorm_minmax", "min": -1000, "max": 3000 },
  "spatial": {
    "mode": "slice", "dim": 2, "patch_size": [256,256], "overlap": 0.66,
    "plane": "axial", "padding": "replicate",
    "ensemble_planes": ["axial"], "ensemble": "none", "ensemble_p": 5.0
  },
  "output_dtype": "float32"
}
```

Because of this, **preprocessing is a spec setting, not app code**: change
`preprocessing.type` (identity | minmax | percentile | normalize) or
`postprocessing.type` (identity | clip | denorm_minmax) and the served model
behaves differently — no rebuild of LaTIM-SNAP.

---

## 2. Environment: one Python venv for everything

Everything below assumes a Python 3.12 virtual environment built with `uv`
(see `Submodules/cbctdenoise/docs/dls-venv.md` for the full, reproducible
recipe):

```bash
uv venv .venv --python 3.12
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cpu   # CPU box
uv pip install --python .venv -r Submodules/cbctdenoise/requirements-serving.txt
uv pip install --python .venv -e Python/latimsnap_i2i
uv pip install --python .venv -e Submodules/cbctdenoise[serving]
# OPTIONAL segmentation server:
uv pip install --python .venv "itksnap-dls>=0.0.6"
# OPTIONAL GPU inference (I2I). CPU onnxruntime = slow for full volumes:
uv pip install --python .venv onnxruntime-gpu        # match CUDA to your box
```

Point LaTIM-SNAP at it: **Preferences → Deep Learning → New/Edit server →
venv path** → `.venv`. Connect.

---

## 3. Interactive Segmentation server (paintbrush)

1. Set up the venv (above) + `itksnap-dls`.
2. Preferences → Deep Learning → add a server, **Task Type = Segmentation
   (nnInteractive)**, set the venv path, run **Configure Packages** (the
   editor installs `itksnap-dls` and runs its `--setup-only`).
3. Connect (status should flip to Connected).
4. Select the **Paintbrush**, and in the paintbrush panel switch its smart
   mode to deep learning.
5. Click/brush on the image → each interaction is sent to the server, which
   replies with an updated segmentation that is **merged into the active
   label layer** (paint as foreground/background using the current drawing
   label).

The server subprocess output appears in the **Deep Learning panel's log**
widget — watch it when things misbehave.

---

## 4. Image-to-Image server (Tools → Image-to-Image…)

### 4.1 Add the server
1. Preferences → Deep Learning → **New…**, **Task Type = Image-to-Image**,
   set the venv path, connect.
2. Make the CBCT→CT model visible. The I2I server lists models from its
   `--models-dir`. Provide yours:
   - copy `cbct2ct.json` + `cbct2ct.onnx` into
     `<venv>/lib/python3.12/site-packages/latimsnap_i2i/models/`, **or**
   - launch with `python -m latimsnap_i2i --models-dir <path>`
     (see §5/§6 for how the server is started).
3. Reconnect → `Tools → Image-to-Image…` should list **CBCT → CT denoising**.

### 4.2 Export your weights once (research `.h5` → ONNX)
```bash
.venv/bin/cbctdenoise-export \
  --checkpoint LaTIM-weights/2025-NECSR_RRDB_VJ016_...MAE_27.9.h5 \
  --out Submodules/cbctdenoise/specs/cbct2ct.onnx
.venv/bin/cbctdenoise-verify Submodules/cbctdenoise/specs/cbct2ct.onnx
```
The exporter prints the checkpoint structure, **auto-infers the architecture**
(num_rrdb, dense layers, growth, features), and fails loudly if a `.h5` does
not match an RRDB generator.

### 4.3 Run a transfer
1. `Tools → Image-to-Image…`.
2. Choose the **source image** (main image or an overlay) and the **model**.
3. **Run** → a progress bar tracks the server-side job.
4. On completion, the result is added as a **new overlay** (“transferred”) in
   the same space/geometry as the source — denormalized back to physical units
   (e.g. HU). Save it from the layer context menu.

---

## 5. Local vs remote: where things run

| | Local server | Remote server |
|---|---|---|
| Server process | spawned by LaTIM-SNAP (venv python) | you run it on the GPU machine |
| Preprocessing | on your machine (in the server) | **on the GPU machine** |
| Inference (GPU) | your GPU (if onnxruntime-gpu) | the remote GPU |
| Model files (`.onnx`/`.json`) | local `--models-dir` / package models | must be **on the remote** |
| Server log | shown in the Deep Learning panel | check the remote console |

**Which machine runs inference is whichever machine runs the server process.**
The client never loads the model.

### Remote, secured (recommended): SSH tunnel
- On the GPU machine: install venv + `latimsnap_i2i` + `onnxruntime-gpu`, put
  `cbct2ct.json`/`cbct2ct.onnx` in its `models/` dir, then:
  ```bash
  python -m latimsnap_i2i --port 8912          # stays on 127.0.0.1
  ```
- In the app: server editor → **Network connection to GPU server** →
  Hostname `localhost`, Port `8912`, tick **Use SSH tunnel**, set SSH
  username + key. LaTIM-SNAP opens a tunnel and forwards requests — the server
  is never exposed publicly (there is **no authentication** on the server).

### Remote, direct (trusted network / VPN only)
- On the GPU machine, bind a reachable interface (it has no auth!):
  ```bash
  python -m latimsnap_i2i --host 0.0.0.0 --port 8912 --models-dir <specs>
  ```
- In the app: **Network connection**, Hostname = the server IP, Port `8912`,
  tunnel **off**. Ensure the firewall allows the port.
- Test reachability: `curl http://<host>:8912/status` should return
  `{"status":"ok","type":"image-to-image","models":[...]}`.

---

## 6. Appendix: endpoints & troubleshooting

### 6.1 HTTP contract (both servers)
- `GET /status` — health + version (+ `models` list for I2I).
- `GET /start_session`, `POST /upload_raw/{sid}` (multipart gzip-raw + JSON
  metadata), and per-task endpoints
  (`process_*_interaction/{sid}` for segmentation; `run_transfer/{sid}`,
  `transfer_progress/{sid}`, `transfer_result/{sid}` for I2I).

### 6.2 Troubleshooting
- **“No models” / empty model list in Image-to-Image** → the I2I server's
  `--models-dir` doesn't contain valid specs (or `cbct2ct.onnx` is missing).
  Confirm `curl /transfer_models`.
- **Server won't connect** → wrong venv/python path, server not running,
  wrong port, or tunnel not established. Read `Error contacting server: …`
  shown in the Deep Learning panel.
- **Weight/architecture mismatch during export/serve** → the verbose server
  log prints missing/unexpected keys. A `.h5` built for a different generator
  (e.g. ResNet vs RRDB) fails loudly — pick the checkpoint that matches the
  architecture.
- **Transfers slow** → CPU onnxruntime. Install `onnxruntime-gpu` where the
  server runs.
- **Result looks wrong / not in MU** → check `spec.json` `preprocessing` /
  `postprocessing` ranges match the model's training normalization.

### 6.3 FAQ
- **Is GPU inference done?** Yes — by the server process, when it has
  `onnxruntime-gpu` (CUDA provider is preferred automatically). Export and
  verify are separate CPU utilities.
- **Why is the result an HU overlay?** The spec denormalizes the model output
  back to physical units, and the client wraps it in the source geometry.
- **Can I serve remotely?** Yes — see §5. Use an SSH tunnel; there is no
  server-side authentication.
- **Do I need to rebuild LaTIM-SNAP to try another model/normalization?** No —
  it's all in the model card (`spec.json`).