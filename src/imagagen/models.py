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

from .metadata import forge_lora_hash


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


def _discover_loras(config) -> list[tuple[str, str]]:
    """(name, filename) for all LoRAs under ``loras/<model-type>/``
    (subdirectories included, symlinked dirs followed), name relative to
    the model-type directory, without extension (e.g.
    ``("test-dir/FuturisticXL", "FuturisticXL.safetensors")``)."""
    out: list[tuple[str, str]] = []
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
            out.append((f.relative_to(sub).with_suffix("").as_posix(), f.name))
    return out


def list_loras(config) -> list[str]:
    """All LoRAs under ``loras/<model-type>/`` (subdirectories included,
    symlinked dirs followed), as names relative to the model-type directory,
    without extension (e.g. ``"dmd2"``, ``"test-dir/FuturisticXL"``)."""
    return [name for name, _ in _discover_loras(config)]


def lora_files(config) -> dict[str, str]:
    """Discovered LoRA name -> file name (extension included)."""
    return dict(_discover_loras(config))


def resolve_lora_names(
    names: list[str], config, hashes: dict[str, str] | None = None
) -> list[str]:
    """Map imported LoRA names to the names of the LoRAs on disk.

    A1111/Forge record LoRAs by (base) name only, so when a LoRA lives in a
    subfolder the imported name is not a discovered name and generation
    would fail. Resolution order per name:

    1. exact match with a discovered (relative) name,
    2. unique basename match among the discovered LoRAs,
    3. Forge's ``Lora hashes`` value (see
       :func:`imagagen.metadata.forge_lora_hash`): matched against ALL
       discovered LoRAs when the name matches no basename. The hash is
       derived from the file's basename, so it is NOT a content fingerprint
       (a locally renamed file won't match) -- it only helps when the
       prompt token's name differs from the basename (e.g. a LoRA with an
       internal display name).

    Several LoRAs sharing a basename: the hash can tell them apart only
    when the extensions differ (``Twin.safetensors`` vs ``Twin.ckpt`` --
    different basenames, different hashes); identical basenames hash
    identically and are equally (un)distinguishable, in which case the
    FIRST candidate in registry order (sorted discovery) is picked so that
    generation still works. Truly unknown names are returned unchanged.
    """
    _STRIP = str.maketrans("", "", ":,")  # what Forge strips from hash keys
    def _hash_for(name: str) -> str | None:
        if not hashes:
            return None
        h = hashes.get(name)
        return h or hashes.get(name.translate(_STRIP))
    files = lora_files(config)
    by_base: dict[str, list[str]] = {}
    for name in files:
        by_base.setdefault(name.rsplit("/", 1)[-1], []).append(name)
    out = []
    for name in names:
        if name in files:
            out.append(name)
            continue
        cand = by_base.get(name, [])
        if len(cand) == 1:
            out.append(cand[0])
            continue
        h = _hash_for(name)
        if h:
            pool = cand if cand else list(files)
            matches = [
                c for c in pool if forge_lora_hash(files[c]) == h.lower()[:12]
            ]
            if len(matches) == 1:
                out.append(matches[0])
                continue
        if cand:
            # ambiguous (and the hash, derived from the basename, cannot
            # tell them apart): pick the first in registry order rather
            # than leaving the bare basename unresolvable
            out.append(cand[0])
            continue
        out.append(name)
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
