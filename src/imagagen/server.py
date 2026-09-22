"""Entry point: `python -m imagagen` or the `imagagen` console script."""

from __future__ import annotations

import argparse
import logging

import uvicorn

from .app import create_app
from .config import load_config


def _configure_logging() -> None:
    """INFO logs for imagagen.* to stderr.

    Attaches to the `imagagen` package logger only: uvicorn's own loggers
    (and their format) stay untouched, and nothing is re-configured on
    import, so test/dev-script usage of create_app() logs nothing unless
    it configures logging itself.
    """
    logger = logging.getLogger("imagagen")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(prog="imagagen", description="AI image generator")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--root", default=None, help="project root (default: $IMAGEN_ROOT or cwd)")
    parser.add_argument(
        "--mock", action="store_true", help="force the CPU mock backend (no GPU/models needed)"
    )
    args = parser.parse_args()

    _configure_logging()

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
