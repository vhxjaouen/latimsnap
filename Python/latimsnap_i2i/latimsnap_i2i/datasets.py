"""Environment validation used by ``--setup-only``.

The LaTIM-SNAP package installer runs ``python -m latimsnap_i2i --setup-only``
after creating the venv / installing dependencies, to confirm the runtime can
import everything it needs and report which model backends are available.
"""

import importlib
import sys


def _mod(name):
    try:
        return importlib.import_module(name)
    except Exception:  # noqa: BLE001
        return None


def check_environment():
    print("latimsnap-i2i %s setup check:" % "0.1.0")
    ok = True

    for name in ("numpy", "fastapi", "uvicorn", "pydantic"):
        m = _mod(name)
        if m is None:
            print("  [MISSING] %s" % name)
            ok = False
        else:
            print("  [ok] %s %s" % (name, getattr(m, "__version__", "?")))

    # Optional backends
    for name, label in (("onnxruntime", "onnx"), ("torch", "torch")):
        m = _mod(name)
        print("  %s backend: %s" % (label, "available" if m else "not installed"))

    if not ok:
        print("Some required packages are missing; run pip install with the "
              "latimsnap-i2i dependency extras and try again.")
        sys.exit(1)
