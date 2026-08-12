"""CBCT denoising (CBCT -> CT) — factorized, latimsnap-standard code.

This package is a clean, self-contained factorization of the research
``NECSRv3_RRDB_SynthRAD2025_Task2_VJ016_AB_TH.ipynb`` notebook. It provides:

  * the 2D RRDB generator (lean, torch-only) and a serving wrapper;
  * HU <-> normalized scaling and geometry-correct 2D-per-slice inference on
    3D volumes (with optional multi-planar + Fourier-burst ensembling);
  * checkpoint export to a standard ONNX artifact (the research ``.h5`` is
    treated purely as an *input* to ``cbctdenoise.export``);
  * a ``latimsnap-i2i`` model spec so the LaTIM-SNAP image-to-image server can
    serve the model directly.

The training loop (GAN + NGF + MSSSIM losses) is kept separate from the
serving path and only needs the ``[train]`` extra.
"""

__version__ = "0.1.0"
