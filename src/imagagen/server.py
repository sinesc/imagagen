"""Entry point: `python -m imagagen` or the `imagagen` console script."""

from __future__ import annotations

import argparse

import uvicorn

from .app import create_app
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(prog="imagagen", description="AI image generator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--root", default=None, help="project root (default: $IMAGEN_ROOT or cwd)")
    parser.add_argument(
        "--mock", action="store_true", help="force the CPU mock backend (default when no CUDA)"
    )
    args = parser.parse_args()

    config = load_config(args.root)
    if args.mock:
        from .backends import MockBackend

        backend = MockBackend()
    else:
        # Imported lazily: pulls in torch.
        from .sdxl_backend import SdxlBackend

        backend = SdxlBackend.from_config(config) or MockBackend()
    app = create_app(config, backend=backend)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
