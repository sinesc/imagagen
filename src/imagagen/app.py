"""FastAPI application factory."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .backends import GenerateRequest, MockBackend
from .config import Config, ensure_dirs, load_config
from .paths import next_output_path

STATIC_DIR = Path(__file__).parent / "static"


def create_app(config: Config | None = None, backend=None) -> FastAPI:
    config = config or load_config()
    ensure_dirs(config)
    if backend is None:
        backend = MockBackend()

    app = FastAPI(title="imagagen")
    app.state.config = config
    app.state.backend = backend

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "root": str(config.root),
            "backend": backend.name,
        }

    @app.post("/api/generate")
    def generate(req: GenerateRequest) -> dict:
        """Phase 0 smoke endpoint (synchronous, single image, no queue yet)."""
        result = backend.generate(req)
        out_path = next_output_path(config.output_dir, req.prompt)
        from PIL import Image

        Image.fromarray(result.image).save(out_path)
        return {
            "seed": result.seed,
            "width": req.width,
            "height": req.height,
            "output": out_path.name,
        }

    @app.get("/api/output")
    def list_output() -> list[dict]:
        files = sorted(config.output_dir.iterdir()) if config.output_dir.is_dir() else []
        return [{"name": f.name, "size": f.stat().st_size} for f in files if f.is_file()]

    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    return app
