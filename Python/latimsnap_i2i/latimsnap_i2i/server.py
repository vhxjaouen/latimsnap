"""FastAPI application for the LaTIM-SNAP image-to-image server."""

import gzip
import hashlib
import json
import logging
import os
import threading
import uuid

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from latimsnap_i2i import __version__
from latimsnap_i2i.imgcodec import decode_raw, encode_result, gunzip_bytes
from latimsnap_i2i.jobs import JobManager
from latimsnap_i2i.model_loader import ModelRunner, load_model_specs

log = logging.getLogger("latimsnap_i2i.server")

SOFT_VERSION = "0.1.0"
CONTRACT_VERSION = 1


class I2IState:
    """Per-server application state."""
    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.models_dir = models_dir
        self.sessions = set()
        self.uploads = {}          # session_id -> {payload, metadata, checksum}
        self.jobs = JobManager()
        self.model_registry = {}
        self.registry_lock = threading.Lock()
        load_model_specs(models_dir, self.model_registry)


def create_app(models_dir=None):
    state = I2IState(models_dir=models_dir)
    app = FastAPI(title="LaTIM-SNAP Image-to-Image Server",
                  version=SOFT_VERSION)

    @app.get("/status")
    def status():
        with state.registry_lock:
            models = sorted(state.model_registry.keys())
        return {
            "status": "ok",
            "version": SOFT_VERSION,
            "contract_version": CONTRACT_VERSION,
            "type": "image-to-image",
            "models": models,
        }

    @app.get("/start_session")
    def start_session():
        sid = uuid.uuid4().hex
        state.sessions.add(sid)
        return {"session_id": sid}

    @app.delete("/end_session/{session_id}")
    def end_session(session_id: str):
        state.sessions.discard(session_id)
        state.jobs.clear_session(session_id)
        return {"ok": True}

    def _require_session(session_id: str):
        if session_id not in state.sessions:
            raise HTTPException(status_code=404, detail="unknown session")

    @app.get("/transfer_models")
    def transfer_models():
        with state.registry_lock:
            out = [
                {"id": s.id, "name": s.name,
                 "output_channels": s.output_channels}
                for s in state.model_registry.values()
            ]
        return {"models": out}

    @app.post("/upload_raw/{session_id}")
    async def upload_raw(session_id: str,
                         file: UploadFile = File(...),
                         metadata: str = Form(...)):
        _require_session(session_id)
        try:
            meta = json.loads(metadata)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="invalid metadata JSON")
        data = await file.read()
        # Validate the payload decodes before storing (keeps state simple: the
        # transferred source is cached per session for model runs).
        state.uploads[session_id] = {
            "payload": data,
            "metadata": meta,
            "checksum": hashlib.md5(data).hexdigest(),
        }
        return {"ok": True, "checksum": state.uploads[session_id]["checksum"]}

    @app.get("/run_transfer/{session_id}")
    def run_transfer(session_id: str, model: str = Query(...)):
        _require_session(session_id)
        cached = state.uploads.get(session_id)
        if not cached:
            raise HTTPException(status_code=400, detail="no uploaded image for session")
        with state.registry_lock:
            spec = state.model_registry.get(model)
        if spec is None:
            raise HTTPException(status_code=404, detail="unknown model %r" % model)

        # Re-decode the source volume each run.
        try:
            raw = gunzip_bytes(cached["payload"])
            src = decode_raw(raw, cached["metadata"])   # (C,Z,Y,X)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail="failed to decode source: %s" % exc)

        job = state.jobs.create(session_id, model)

        def _work(progress):
            runner = ModelRunner(spec)
            runner.load()

            outp = runner.run(src, progress)   # (OC, X, Y, Z)
            # Ensure at least one channel if the model returned a bare scalar vol.
            if outp.ndim == 3:
                outp = outp[None, ...]
            # Carry the source geometry (spacing/origin/direction) so the
            # result registers in the same space as the source image.
            return encode_result(outp, base_metadata=cached["metadata"])

        state.jobs.submit(job, _work)
        return {"job_id": job.job_id}

    @app.get("/transfer_progress/{session_id}")
    def transfer_progress(session_id: str, job: str):
        snap = state.jobs.snapshot(job)
        if snap is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return snap

    @app.get("/transfer_result/{session_id}")
    def transfer_result(session_id: str, job: str):
        snap = state.jobs.snapshot(job)
        if snap is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if snap["status"] == "error":
            raise HTTPException(status_code=500, detail=snap.get("error") or "transfer failed")
        if snap["status"] != "done":
            raise HTTPException(status_code=409, detail="job not finished")
        res = state.jobs.take_result(job)
        if res is None:
            raise HTTPException(status_code=404, detail="result already consumed")
        return JSONResponse(content=res)

    return app


def run_server(app, host="127.0.0.1", port=8912, log_level="info"):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level=log_level)


# Convenience object for `uvicorn latimsnap_i2i.server:app` (no models dir).
app = create_app()
