"""latimsnap_i2i - image-to-image deep learning server for LaTIM-SNAP.

This is the GPL-licensed, in-tree server used by LaTIM-SNAP's Image-to-Image
pipeline. It runs a user-supplied encoder model (ONNX or PyTorch) in the
background and returns a new image (e.g. MR->CT synthesis) over HTTP.

The wire protocol mirrors the existing interactive segmentation server
(itksnap_dls) so the LaTIM-SNAP C++ client can reuse its gzip/JSON encode
helpers.

Endpoints:
  GET  /status
  GET  /start_session
  POST /upload_raw/{session_id}          (multipart: file gz + metadata json)
  GET  /transfer_models
  POST /run_transfer/{session_id}        ({model_id}) -> {job_id}
  GET  /transfer_progress/{session_id}   ({job_id})  -> {status, progress, message}
  GET  /transfer_result/{session_id}     ({job_id})  -> {result, metadata}
"""

__version__ = "0.1.0"
