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
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from . import memdbg
from .backends import format_loras
from .metadata import forge_lora_hash

log = logging.getLogger(__name__)


_MODEL_SUFFIXES = (".safetensors", ".ckpt", ".gguf")


def _walk_files(base: Path,
                suffixes: tuple[str, ...] = _MODEL_SUFFIXES) -> list[Path]:
    """All files with one of ``suffixes`` under ``base``, recursive.

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
                elif e.suffix in suffixes:
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


def filter_loras_for_family(
    loras: list[tuple[str, float]], family: str, config
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Split a ``(name, weight)`` LoRA list into LoRAs that live under
    ``loras/<family>/`` and LoRAs discovered under a *different* family
    dir (the first path component under ``loras/`` IS the family).

    Foreign-family entries are dropped rather than passed to a backend
    that cannot load them (generation continues with the matching ones);
    names that are not discovered at all stay in the kept list so the
    backend still reports them as "not found".
    """
    paths = lora_paths(config)
    kept: list[tuple[str, float]] = []
    foreign: list[tuple[str, float]] = []
    for name, weight in loras:
        p = paths.get(name)
        if p is not None:
            rel = p.relative_to(config.loras_dir)
            if len(rel.parts) > 1 and rel.parts[0] != family:
                foreign.append((name, weight))
                continue
        kept.append((name, weight))
    return kept, foreign


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
    """Per-GPU residency state: one GPU-resident copy, one RAM-parked copy,
    one RLock.

    The lock serializes load/unload/generate on that GPU. A backend must
    never be moved or swapped while another thread is inside
    ``generate()`` (concurrent ``.to()``/forward on the same module
    corrupts CUDA state and can segfault the process).

    After a batch finishes the queue parks the resident model (weights in
    RAM, VRAM freed); ``parked`` keeps that copy so the next use of the
    same model is a GPU move instead of a checkpoint re-read. At most one
    parked model is kept per GPU.
    """

    __slots__ = ("gpu", "lock", "backend", "model_id", "parked", "parked_id")

    def __init__(self, gpu: int) -> None:
        self.gpu = gpu
        self.lock = threading.RLock()
        self.backend = None
        self.model_id: str | None = None
        self.parked = None
        self.parked_id: str | None = None


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
    # IS the family. For flux the loadable formats are the GGUF Q4_K file
    # and the ~12 GB Kijai FP8 / bnb NF4 .safetensors variants (all
    # dequantize per-op on the GPU, see imagagen.gguf_ops).
    _FAMILIES = (
        ("sdxl", ".safetensors"),
        ("flux", ".gguf", ".safetensors"),
        ("zimage", ".safetensors"),
        ("krea2", ".safetensors"),
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

    def gpu_lock(self, gpu: int) -> threading.RLock:
        """The per-GPU lock, so external GPU work (upscaling) serializes
        against load/swap/generate on that GPU. Only called with a valid
        CUDA index (callers skip it when the registry has no GPU)."""
        return self._get_gpu_state(gpu).lock

    def _park_state_locked(self, state: _GpuState) -> None:
        """Move the GPU-resident model to RAM and keep it parked.

        Caller must hold ``state.lock``. A previously parked model is
        dropped first so at most one parked copy is kept per GPU. Backends
        whose type cannot be moved (``can_park`` is False) are left
        resident: they simply stay in VRAM.
        """
        if state.parked is not None:
            log.info("model park: dropping %s parked on GPU %d",
                     state.parked_id, state.gpu)
            # MEMDBG: a whole checkpoint copy (~7 GiB for SDXL fp16) is about
            # to become unreachable. If no "evt=freed" line follows, the
            # dropped copy is still referenced by something -> RAM leak.
            memdbg.stamp("park-drop", gpu=state.gpu, model=state.parked_id,
                         obj=id(state.parked))
            state.parked = None
            state.parked_id = None
        if state.backend is not None and not state.backend.can_park:
            return  # this model type cannot leave VRAM; leave it resident
        if state.backend is not None:
            log.info("model park: %s on GPU %d -> RAM", state.backend.name, state.gpu)
            started = time.monotonic()
            memdbg.stamp("park-begin", gpu=state.gpu, model=state.model_id,
                         obj=id(state.backend))
            state.backend.unload()
            state.parked = state.backend
            state.parked_id = state.model_id
            state.backend = None
            state.model_id = None
            memdbg.stamp("park-end", gpu=state.gpu, model=state.parked_id,
                         secs=round(time.monotonic() - started, 1))
        import gc

        gc.collect()
        # park churns a multi-GiB pageable host copy; glibc otherwise
        # caches the freed segments in the calling thread's arena (see
        # load_config's mallopt_arenas note for the measured effect)
        memdbg.malloc_trim("park")
        try:
            import torch

            if torch.cuda.is_available():
                with torch.cuda.device(state.gpu):
                    torch.cuda.empty_cache()
        except Exception:
            pass

    def unload_all(self) -> None:
        """Move every GPU-resident model into RAM to release VRAM.

        Called after a batch finishes so the app does not hold VRAM while
        idle. The parked copies stay reachable and a later request for the
        same model restores it with a GPU move instead of a re-read.
        """
        with self._meta_lock:
            states = list(self._gpu_states.values())
        memdbg.stamp("unload_all", gpus=",".join(str(s.gpu) for s in states))
        for state in states:
            with state.lock:
                self._park_state_locked(state)
        memdbg.stamp("unload_all-end")

    def _new_backend(self, info: ModelInfo, device):
        """Instantiate the backend for ``info.family``.

        Lazy import so a family with no checkpoint on disk never forces its
        (heavy) dependencies to load.
        """
        import torch

        if info.family == "flux":
            from .backends.flux import FluxBackend

            return FluxBackend(info, torch.device(device), config=self.config)
        if info.family == "zimage":
            from .backends.zimage import ZImageBackend

            return ZImageBackend(info, torch.device(device), config=self.config)
        if info.family == "krea2":
            from .backends.krea2 import Krea2Backend

            return Krea2Backend(info, torch.device(device), config=self.config)
        from .backends.sdxl import SdxlBackend

        return SdxlBackend(info, torch.device(device), config=self.config)

    def _load_backend_locked(
        self, state: _GpuState, model_id: str,
        loras: list[tuple[str, float]] | None = None,
    ):
        """Ensure ``model_id`` is loaded on ``state.gpu`` and return it.

        Caller must hold ``state.lock``. ``loras`` is the LoRA set this load
        is being prepared for (the queue applies it per generation); it is
        only logged. Loading is deferred: nothing loads a model until a
        generation (or an explicit pre-warm) asks for it.
        """
        if state.backend is not None and state.model_id == model_id:
            return state.backend
        info = self.get(model_id)
        if info is None:
            raise ValueError(f"unknown model: {model_id}")
        # Fast path: this exact model is still parked in RAM from a
        # previous batch/switch. The currently resident model (if any)
        # becomes the new parked copy, then the parked backend moves back
        # to the GPU instead of re-reading the checkpoint.
        if state.parked is not None and state.parked_id == model_id:
            restoring = state.parked
            state.parked = None
            state.parked_id = None
            self._park_state_locked(state)
            state.backend = restoring
            state.model_id = model_id
            log.info("model load start: %s on GPU %d (parked restore, LoRAs: %s)",
                     info.name, state.gpu, format_loras(loras or []))
            memdbg.stamp("restore-begin", gpu=state.gpu, model=model_id,
                         obj=id(restoring),
                         also_parked=state.parked_id)
            started = time.monotonic()
            self.emit({"type": "status",
                       "message": f"loading {info.name} on GPU {state.gpu} …"})
            state.backend.load()
            with self._meta_lock:
                self._current_id = model_id
            log.info("model load done: %s on GPU %d (%.1fs)",
                     info.name, state.gpu, time.monotonic() - started)
            memdbg.stamp("restore-end", gpu=state.gpu, model=model_id,
                         secs=round(time.monotonic() - started, 1))
            self.emit({"type": "status",
                       "message": f"loaded {info.name} on GPU {state.gpu}"})
            return state.backend
        # Normal path: park whatever is resident, then read the checkpoint.
        self._park_state_locked(state)
        import torch

        log.info("model load start: %s on GPU %d (LoRAs: %s)",
                 info.name, state.gpu, format_loras(loras or []))
        started = time.monotonic()
        self.emit({"type": "status",
                   "message": f"loading {info.name} on GPU {state.gpu} …"})
        memdbg.stamp("read-checkpoint-begin", gpu=state.gpu, model=model_id)
        state.backend = self._new_backend(info, torch.device(f"cuda:{state.gpu}"))
        memdbg.track(state.backend, f"backend[{model_id}@gpu{state.gpu}]")
        state.model_id = model_id
        memdbg.stamp("read-checkpoint-end", gpu=state.gpu, model=model_id)
        # the checkpoint read churns several GiB of host buffers (see
        # _park_state_locked / load_config on why the heap is trimmed)
        import gc

        gc.collect()
        memdbg.malloc_trim("load")
        with self._meta_lock:
            self._current_id = model_id
        log.info("model load done: %s on GPU %d (%.1fs)",
                 info.name, state.gpu, time.monotonic() - started)
        self.emit({"type": "status",
                   "message": f"loaded {info.name} on GPU {state.gpu}"})
        return state.backend

    def backend_for(self, model_id: str, gpu: int | None = None):
        """Return the backend for model_id on ``gpu`` (or the primary),
        loading it (and parking any other model on that GPU) if needed.

        Blocks while a generation is running on that GPU. Used by
        ``from_config`` and dev scripts; the generation queue uses
        :meth:`run` instead. Model selection (``/api/models/select``)
        deliberately does NOT call this: loading is deferred until the
        first generation for the model.
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

    def pre_encode(self, model_id: str, gpu: int | None, pairs,
                   reserve: int | None = None):
        """Ensure ``model_id`` is loaded on ``gpu`` (or the primary) and
        encode every ``(prompt, negative_prompt)`` pair of a batch in one
        text-encoder session (see :meth:`ModelBackend.pre_encode`).

        The generation queue calls this for wildcard batches BEFORE any of
        their tasks runs, so the batch's per-image prompt picks each get
        one encoder load for the whole batch instead of one per image. The
        per-GPU lock is held across the encode, so a model swap can never
        park the backend mid-encode.
        """
        if gpu is None:
            gpu = self.primary
        state = self._get_gpu_state(gpu)
        with state.lock:
            backend = self._load_backend_locked(state, model_id)
            backend.pre_encode(pairs, reserve=reserve)

    def prefetch_text_encoder(self, model_id: str, prompt: str,
                              negative: str, force: bool = False) -> str:
        """Prompt-change prefetch (imagagen.ini prefetch-text-encoder).

        Move every GPU that currently has ``model_id`` VRAM-resident into
        its "about to encode" layout (see
        :meth:`ModelBackend.prepare_text_encode`): text encoders to the
        GPU (parking the transformer when the two cannot co-reside) so the
        expensive bus swap happens while the user still edits the prompt
        instead of at generation start. Deliberately lazy like model
        selection: a model that is parked or was never loaded is NOT
        woken up (``not-resident``), and a cached prompt needs no encoder
        at all (``cached``) -- unless ``force``, set for wildcard prompts
        whose per-seed resolutions always miss the cache.

        Never queues behind an in-flight generation: a GPU whose lock is
        busy is skipped (its generation encodes on its own anyway). The
        per-GPU lock IS taken while preparing, so the moves never race a
        load/unload/generate. Returns a status string:
        ``not-resident`` | ``cached`` | ``prepared`` | ``busy`` | ``failed``.
        ``failed`` means every attempt raised (a broken prepare_text_encode,
        not a busy GPU) -- the tracebacks are in the log.
        """
        with self._meta_lock:
            states = [s for s in self._gpu_states.values()
                      if s.backend is not None and s.model_id == model_id]
        if not states:
            memdbg.stamp("prefetch", model=model_id, action="not-resident",
                         prompt=len(prompt), force=force)
            return "not-resident"
        # the prompt-encoding cache is shared across the model's per-GPU
        # instances, so one check decides for all of them
        if not force and states[0].backend.encoding_cached(prompt, negative):
            memdbg.stamp("prefetch", model=model_id, action="cached",
                         prompt=len(prompt), force=force)
            return "cached"
        prepared = failed = 0
        for state in states:
            if not state.lock.acquire(blocking=False):
                continue  # generating on this GPU: its encode runs anyway
            try:
                backend = state.backend
                if backend is None or state.model_id != model_id:
                    continue  # parked/switched between snapshot and lock
                # whether the transformer has to make room is backend-
                # specific (SDXL's UNet always co-resides with the CLIP
                # towers and deliberately stays in VRAM), so the message
                # must not promise a park
                log.info(
                    "prompt prefetch: %s on GPU %d -- moving the text "
                    "encoders to their encode device (prompt cache miss)",
                    backend.name, state.gpu)
                started = time.monotonic()
                backend.prepare_text_encode()
                prepared += 1
                memdbg.stamp("prefetch-end", gpu=state.gpu, model=model_id,
                             secs=round(time.monotonic() - started, 2))
            except Exception:
                # a broken prepare must not be reported as "busy": the
                # feature is failing, and silently so without this counter
                log.exception("prompt prefetch failed on GPU %d", state.gpu)
                memdbg.stamp("prefetch-failed", gpu=state.gpu, model=model_id)
                failed += 1
            finally:
                state.lock.release()
        action = ("prepared" if prepared else
                  "failed" if failed else "busy")
        memdbg.stamp("prefetch", model=model_id, action=action,
                     prompt=len(prompt), force=force,
                     gpus=",".join(str(s.gpu) for s in states))
        return action

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

    def caption(self, image, gpu: int | None = None, progress=None) -> str:
        """Describe ``image`` (HxWx3 uint8) with the reference VLM.

        Runs under the per-GPU lock (never races a generation on that
        GPU); a resident model is parked first so the captioning towers
        (~5.3 GB) fit on the card. The result is served by
        ``POST /api/reference/caption`` and shared with generation-time
        captioning through the sha1 cache (see imagagen.reference).
        ``progress`` receives the captioning phase messages (tower
        load / describing / done).
        """
        from .reference import caption_reference

        if gpu is None:
            gpu = self.primary
        if gpu is None:
            raise RuntimeError("no usable GPU for reference captioning")
        state = self._get_gpu_state(gpu)
        with state.lock:
            if state.backend is not None:
                self._park_state_locked(state)
            return caption_reference(image, f"cuda:{gpu}", self.config,
                                     progress=progress)

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
