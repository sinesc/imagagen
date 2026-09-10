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
        "--mock", action="store_true", help="force the CPU mock backend (no GPU/models needed)"
    )
    args = parser.parse_args()

    config = load_config(args.root)
    if args.mock:
        from .backends import MockBackend

        app = create_app(config, backend=MockBackend())
    else:
        # Real mode: the app owns a ModelRegistry and loads models lazily
        # on first generation / model select (nothing is in VRAM until asked).
        app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
