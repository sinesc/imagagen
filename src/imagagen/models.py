"""Model registry: discovery, loading, single-model VRAM residency.

Two checkpoint families live under ``models/<family>/`` (subdirectories
are scanned recursively): ``sdxl`` (SDXL, ``.safetensors``) and ``flux``
(FLUX.1, ``.gguf``). The directory name IS the family and selects the
backend class (:meth:`ModelRegistry._new_backend`). Only one model is in
VRAM at a time; switching unloads the previous one (Phase 6 adds RAM
residency for the last-unloaded model).
"""

from __future__ import annotations

import logging
import time
import threading
from dataclasses import dataclass
from pathlib import Path

from .backends import format_loras
from .metadata import forge_lora_hash

log = logging.getLogger(__name__)


_MODEL_SUFFIXES = (".safetensors", ".ckpt", ".gguf")


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


def _discover_loras(config, family: str | None = None) -> list[tuple[str, str, Path]]:
    """(name, filename, path) for all LoRAs under
    ``loras/<model-type>/`` (subdirectories included, symlinked dirs
    followed), name relative to the model-type directory, without
    extension (e.g.
    ``("test-dir/FuturisticXL", "FuturisticXL.safetensors", <path>)``).

    ``family`` restricts discovery to ``loras/<family>/`` (the UI filters
    by the selected model's family, so SDXL LoRAs are not selectable for
    Flux models and vice versa); ``None`` scans every family dir.
    """
    out: list[tuple[str, str, Path]] = []
    loras_dir = config.loras_dir
    if not loras_dir.is_dir():
        return out
    subs = ([loras_dir / family] if family is not None
            else [sub for sub in sorted(loras_dir.iterdir()) if sub.is_dir()])
    for sub in subs:
        if not sub.is_dir():
            continue
        files = _walk_files(sub)
        files = sorted(f for f in files if f.suffix == ".safetensors") + \
            sorted(f for f in files if f.suffix == ".ckpt")
        for f in files:
            out.append((f.relative_to(sub).with_suffix("").as_posix(),
                        f.name, f))
    return out


def list_loras(config, family: str | None = None) -> list[str]:
    """All LoRAs under ``loras/<model-type>/`` (subdirectories included,
    symlinked dirs followed), as names relative to the model-type directory,
    without extension (e.g. ``"dmd2"``, ``"test-dir/FuturisticXL"``).

    ``family`` (``"sdxl"``/``"flux"``) restricts to that model-type dir;
    ``None`` (default) lists every family.
    """
    return [name for name, _, _ in _discover_loras(config, family)]


def lora_files(config) -> dict[str, str]:
    """Discovered LoRA name -> file name (extension included)."""
    return {name: fname for name, fname, _ in _discover_loras(config)}


def lora_paths(config) -> dict[str, Path]:
    """Discovered LoRA name -> its file path (e.g. for the settings .json
    stored next to the .safetensors)."""
    return {name: path for name, _, path in _discover_loras(config)}


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
    id: str  # relative path within models/<family>, no extension
    name: str  # same as id (the path makes nested models distinguishable in the UI)
    path: Path
    family: str = "sdxl"  # "sdxl" | "flux" -- selects the backend class

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "family": self.family}


class _GpuState:
    """Per-GPU residency state: one backend copy, one RLock.

    The lock serializes load/unload/generate on that GPU. A backend must
    never be moved or swapped while another thread is inside
    ``generate()`` (concurrent ``.to()``/forward on the same module
    corrupts CUDA state and can segfault the process).
    """

    __slots__ = ("gpu", "lock", "backend", "model_id")

    def __init__(self, gpu: int) -> None:
        self.gpu = gpu
        self.lock = threading.RLock()
        self.backend = None
        self.model_id: str | None = None


class ModelRegistry:
    def __init__(self, config, emit=None,
                 gpus: list[int] | None = None) -> None:
        self.config = config
        self.emit = emit or (lambda e: None)
        self.models: list[ModelInfo] = self.discover()
        if gpus is None:
            from .gpus import gpu_order

            gpus = gpu_order(config.primary_gpu, config.ignore_gpus)
        self.gpus: list[int] = list(gpus)
        self.primary: int | None = self.gpus[0] if self.gpus else None
        self._gpu_states: dict[int, _GpuState] = {}
        self._meta_lock = threading.Lock()
        self._current_id: str | None = None

    # (family, models/<family>/, allowed file suffixes). The directory name
    # IS the family. Only the GGUF transformer is loadable for flux right
    # now; the ~12 GB fp8/nf4 .safetensors variants are deliberately left
    # out of discovery so they never surface as un-loadable entries.
    _FAMILIES = (
        ("sdxl", ".safetensors"),
        ("flux", ".gguf"),
    )

    def discover(self) -> list[ModelInfo]:
        models = []
        for family, *suffixes in self._FAMILIES:
            fam_dir = self.config.models_dir / family
            if not fam_dir.is_dir():
                continue
            files = sorted(f for f in _walk_files(fam_dir)
                           if f.suffix in suffixes)
            for f in files:
                rel = f.relative_to(fam_dir).with_suffix("").as_posix()
                models.append(ModelInfo(id=rel, name=rel, path=f,
                                        family=family))
        return models

    def list(self) -> list[dict]:
        return [m.to_dict() for m in self.models]

    def get(self, model_id: str) -> ModelInfo | None:
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def current_backend(self):
        if self.primary is None or self.primary not in self._gpu_states:
            return None
        return self._gpu_states[self.primary].backend

    def current_id(self) -> str | None:
        with self._meta_lock:
            return self._current_id

    def _get_gpu_state(self, gpu: int) -> _GpuState:
        with self._meta_lock:
            return self._gpu_states.setdefault(gpu, _GpuState(gpu))

    def _unload_state_locked(self, state: _GpuState) -> None:
        if state.backend is not None:
            log.info("model unload: %s on GPU %d", state.backend.name, state.gpu)
            state.backend.unload()
            state.backend = None
            state.model_id = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                with torch.cuda.device(state.gpu):
                    torch.cuda.empty_cache()
        except Exception:
            pass

    def _new_backend(self, info: ModelInfo, device):
        """Instantiate the backend for ``info.family``.

        Lazy import so a family with no checkpoint on disk never forces its
        (heavy) dependencies to load.
        """
        import torch

        if info.family == "flux":
            from .flux_backend import FluxBackend

            return FluxBackend(info, torch.device(device), config=self.config)
        from .sdxl_backend import SdxlBackend

        return SdxlBackend(info, torch.device(device), config=self.config)

    def _load_backend_locked(
        self, state: _GpuState, model_id: str,
        loras: list[tuple[str, float]] | None = None,
    ):
        """Ensure ``model_id`` is loaded on ``state.gpu`` and return it.

        Caller must hold ``state.lock``. ``loras`` is the LoRA set this load
        is being prepared for (the queue applies it per generation); it is
        only logged.
        """
        if state.backend is not None and state.model_id == model_id:
            return state.backend
        self._unload_state_locked(state)
        info = self.get(model_id)
        if info is None:
            raise ValueError(f"unknown model: {model_id}")
        import torch

        log.info("model load start: %s on GPU %d (LoRAs: %s)",
                 info.name, state.gpu, format_loras(loras or []))
        started = time.monotonic()
        self.emit({"type": "status",
                   "message": f"loading {info.name} on GPU {state.gpu} …"})
        state.backend = self._new_backend(info, torch.device(f"cuda:{state.gpu}"))
        state.model_id = model_id
        with self._meta_lock:
            self._current_id = model_id
        log.info("model load done: %s on GPU %d (%.1fs)",
                 info.name, state.gpu, time.monotonic() - started)
        self.emit({"type": "status",
                   "message": f"loaded {info.name} on GPU {state.gpu}"})
        return state.backend

    def backend_for(self, model_id: str, gpu: int | None = None):
        """Return the backend for model_id on ``gpu`` (or the primary),
        loading it (and unloading any other model on that GPU) if needed.

        Blocks while a generation is running on that GPU. Used by
        /api/models/select pre-warm, ``from_config`` and dev scripts; the
        generation queue uses :meth:`run` instead.
        """
        if gpu is None:
            gpu = self.primary
        if gpu is None:
            from .gpus import select_gpu

            gpu = select_gpu(self.config.primary_gpu, self.config.ignore_gpus)
        if gpu is None:
            raise RuntimeError(
                f"no usable GPU for generation (primary-gpu="
                f"{self.config.primary_gpu if self.config.primary_gpu is not None else 'auto'}, "
                f"ignore-gpus={sorted(self.config.ignore_gpus) or '-'}); "
                "check nvidia-smi and imagagen.ini"
            )
        state = self._get_gpu_state(gpu)
        with state.lock:
            return self._load_backend_locked(state, model_id)

    def run(self, model_id: str, gpu: int, request, progress=None):
        """Atomically ensure ``model_id`` is loaded on ``gpu`` and run one
        generation.

        Returns ``(GenerateResult, ModelBackend)``. The per-GPU lock is held
        for the whole generation, so a model swap on the same GPU blocks
        until it finishes; other GPUs are unaffected.
        """
        state = self._get_gpu_state(gpu)
        with state.lock:
            backend = self._load_backend_locked(
                state, model_id, loras=getattr(request, "loras", None))
            return backend.generate(request, progress), backend

    def cancel(self, gpu: int | None = None) -> None:
        """Cancel in-flight generation on one GPU (or all loaded GPUs).

        This deliberately does NOT take the per-GPU generation lock: an
        interrupt must not block behind the very generation it is trying to
        cancel. ``ModelBackend.cancel`` only flips a flag checked per step.
        """
        with self._meta_lock:
            if gpu is None:
                states = list(self._gpu_states.values())
            else:
                state = self._gpu_states.get(gpu)
                states = [state] if state is not None else []
        for state in states:
            backend = state.backend
            if backend is not None:
                backend.cancel()
