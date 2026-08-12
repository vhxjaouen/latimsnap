"""Command line entry point: python -m latimsnap_i2i --port <port>"""

import argparse
import logging
import os
import sys


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
