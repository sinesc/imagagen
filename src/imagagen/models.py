"""Model registry: discovery, loading, single-model VRAM residency.

Phase 1: one SDXL checkpoint family (models/sdxl/, subdirectories are
scanned recursively). Only one
model is in VRAM at a time; switching unloads the previous one (Phase 6
adds RAM residency for the last-unloaded model).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path


_MODEL_SUFFIXES = (".safetensors", ".ckpt")


def _walk_files(base: Path) -> list[Path]:
    """All ``*.safetensors``/``*.ckpt`` under ``base``, recursive.

    An explicit scandir-style walk instead of ``rglob``: since Python 3.13,
    ``**`` does NOT descend into symlinked directories (it used to), so a
    symlinked folder full of checkpoints would silently go undetected.
    Cycles via symlink chains are broken by tracking resolved dirs.
    """
    out: list[Path] = []
    seen: set[Path] = {base.resolve()}

    def walk(d: Path) -> None:
        try:
            entries = sorted(d.iterdir())
        except OSError:
            return  # vanished/unreadable dir: skip
        for e in entries:
            try:
                if e.is_dir():
                    real = e.resolve()
                    if real not in seen:
                        seen.add(real)
                        walk(e)
                elif e.suffix in _MODEL_SUFFIXES:
                    out.append(e)
            except OSError:
                continue  # broken symlink etc.

    walk(base)
    return out


def list_loras(config) -> list[str]:
    """All LoRAs under ``loras/<model-type>/`` (subdirectories included,
    symlinked dirs followed), as names relative to the model-type directory,
    without extension (e.g. ``"dmd2"``, ``"test-dir/FuturisticXL"``)."""
    out: list[str] = []
    loras_dir = config.loras_dir
    if not loras_dir.is_dir():
        return out
    for sub in sorted(loras_dir.iterdir()):
        if not sub.is_dir():
            continue
        files = _walk_files(sub)
        files = sorted(f for f in files if f.suffix == ".safetensors") + \
            sorted(f for f in files if f.suffix == ".ckpt")
        for f in files:
            out.append(f.relative_to(sub).with_suffix("").as_posix())
    return out


@dataclass
class ModelInfo:
    id: str  # relative path within models/sdxl, no extension, e.g. "test-dir/JuggernautXL9.0"
    name: str  # same as id (the path makes nested models distinguishable in the UI)
    path: Path

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name}


class ModelRegistry:
    def __init__(self, config, emit=None, gpu: int | None = None) -> None:
        self.config = config
        self.emit = emit or (lambda e: None)
        self.models: list[ModelInfo] = self.discover()
        self._lock = threading.Lock()  # serializes load/unload/generate
        self._current_id: str | None = None
        self._backend = None
        self.gpu = gpu  # None -> resolved on first use (ini/auto-pick)

    def discover(self) -> list[ModelInfo]:
        models = []
        sdxl_dir = self.config.models_dir / "sdxl"
        if sdxl_dir.is_dir():
            files = sorted(f for f in _walk_files(sdxl_dir)
                           if f.suffix == ".safetensors")
            for f in files:
                rel = f.relative_to(sdxl_dir).with_suffix("").as_posix()
                models.append(ModelInfo(id=rel, name=rel, path=f))
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

            self._backend = SdxlBackend(info, self._resolve_device(),
                                        config=self.config)
            self._current_id = model_id
            self.emit({"type": "status", "message": f"loaded {info.name}"})
            return self._backend

    def _resolve_device(self):
        """Pick the torch device for generation (cached after first call).

        Explicit ``gpu`` (from /api/health-time selection) wins; otherwise
        the ini settings decide: primary-gpu, or auto-pick the most-free
        non-ignored GPU.
        """
        import torch

        if self.gpu is None:
            from .gpus import select_gpu

            self.gpu = select_gpu(self.config.primary_gpu,
                                  self.config.ignore_gpus)
        if self.gpu is None:
            raise RuntimeError(
                f"no usable GPU for generation (primary-gpu="
                f"{self.config.primary_gpu if self.config.primary_gpu is not None else 'auto'}, "
                f"ignore-gpus={sorted(self.config.ignore_gpus) or '-'}); "
                "check nvidia-smi and imagagen.ini"
            )
        return torch.device(f"cuda:{self.gpu}")

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
