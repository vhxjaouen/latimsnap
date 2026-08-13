"""Command line entry point: python -m latimsnap_i2i --port <port>"""

import argparse
import logging
import os
import sys


def _add_nvidia_libs_to_path():
    """Expose bundled NVIDIA CUDA libraries (nvidia-* wheels) to the loader.

    onnxruntime's CUDAExecutionProvider looks for libcublasLt/cudnn etc. via
    LD_LIBRARY_PATH. Those libraries ship inside this venv's nvidia/*/lib
    folders, so prepend them so GPU inference works without a system CUDA
    toolkit or any manual LD_LIBRARY_PATH setup.
    """
    import sysconfig
    site = sysconfig.get_paths().get("purelib", "")
    nvidia_root = os.path.join(site, "nvidia")
    if not os.path.isdir(nvidia_root):
        return
    lib_dirs = []
    for name in sorted(os.listdir(nvidia_root)):
        lib = os.path.join(nvidia_root, name, "lib")
        if os.path.isdir(lib):
            lib_dirs.append(lib)
    if not lib_dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    merged = ":".join(lib_dirs)
    if existing:
        merged = merged + ":" + existing
    os.environ["LD_LIBRARY_PATH"] = merged


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="latimsnap-i2i",
        description="LaTIM-SNAP image-to-image deep learning server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8912)
    parser.add_argument("--models-dir", default=None,
                        help="Directory containing model spec JSON files")
    parser.add_argument("--use-colors", action="store_true")
    parser.add_argument("--setup-only", action="store_true",
                        help="Validate the environment and exit (used by the "
                             "LaTIM-SNAP package installer)")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    # Expose bundled CUDA libs so GPU (onnxruntime CUDA provider) is usable.
    _add_nvidia_libs_to_path()

    if args.use_colors:
        os.environ.setdefault("FORCE_COLOR", "1")

    if args.setup_only:
        from latimsnap_i2i.datasets import check_environment
        check_environment()
        print("latimsnap-i2i environment OK")
        return 0

    from latimsnap_i2i.server import create_app, run_server
    app = create_app(models_dir=args.models_dir)
    run_server(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
