"""Model registry: discovery, loading, single-model VRAM residency.

Phase 1: one SDXL checkpoint family (models/sdxl/*.safetensors). Only one
model is in VRAM at a time; switching unloads the previous one (Phase 6
adds RAM residency for the last-unloaded model).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelInfo:
    id: str  # relative path, e.g. "sdxl/sd_xl_base_1.0.safetensors"
    name: str
    path: Path

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name}


class ModelRegistry:
    def __init__(self, config, emit=None) -> None:
        self.config = config
        self.emit = emit or (lambda e: None)
        self.models: list[ModelInfo] = self.discover()
        self._lock = threading.Lock()  # serializes load/unload/generate
        self._current_id: str | None = None
        self._backend = None

    def discover(self) -> list[ModelInfo]:
        models = []
        sdxl_dir = self.config.models_dir / "sdxl"
        if sdxl_dir.is_dir():
            for f in sorted(sdxl_dir.glob("*.safetensors")):
                rel = f.relative_to(self.config.root).as_posix()
                models.append(ModelInfo(id=rel, name=f.stem, path=f))
        return models

    def list(self) -> list[dict]:
        return [m.to_dict() for m in self.models]

    def get(self, model_id: str) -> ModelInfo | None:
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def current_backend(self):
        with self._lock:
            return self._backend

    def current_id(self) -> str | None:
        with self._lock:
            return self._current_id

    def backend_for(self, model_id: str):
        """Return the backend for model_id, loading it (and unloading the
        previous one) if necessary. Blocks while a generation is running."""
        info = self.get(model_id)
        if info is None:
            raise ValueError(f"unknown model: {model_id}")
        with self._lock:
            if self._current_id == model_id and self._backend is not None:
                return self._backend
            self._unload_locked()
            self.emit({"type": "status", "message": f"loading {info.name} …"})
            from .sdxl_backend import SdxlBackend

            self._backend = SdxlBackend(info)
            self._current_id = model_id
            self.emit({"type": "status", "message": f"loaded {info.name}"})
            return self._backend

    def _unload_locked(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
            self._current_id = None
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
