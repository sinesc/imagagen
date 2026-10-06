"""Model backend interface.

The rest of imagagen (queue, generation, API) talks to this interface only,
so the CPU MockBackend can stand in for a real SDXL backend in tests and in
environments without a GPU.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..sag import SelfAttentionGuidance
from ..soft_inpaint import SoftInpainting

# Serializes every operation that CREATES torch parameters across ALL
# backends (one per GPU, each on its own worker thread): model init
# (SdxlBackend.__init__ / from_single_file) and LoRA apply/unload
# (_apply_loras: peft adapters in SDXL, registered delta parameters in
# Flux). They go through peft/transformers, whose ``_init_on_device``
# context manager MONKEY-PATCHES GLOBAL torch state
# (``nn.Module.register_parameter``, ``torch.empty``/``zeros``/...);
# that patch is not thread-safe. Two threads creating parameters
# concurrently interleave the global patches, and a parameter
# registration runs through the wrong thread's ``register_empty_parameter``,
# which copies ``param.__dict__`` (including transformers'
# ``_is_hf_initialized`` flag) into ``Parameter.__new__`` kwargs ->
# ``TypeError: Parameter.__new__() got an unexpected keyword argument
# "_is_hf_initialized"``. Holding this lock makes parameter creation
# single-threaded process-wide. (It is always acquired innermost -- after
# state.lock / self._lock -- so no deadlock.)
_PARAM_INIT_LOCK = threading.RLock()

#: built-in CFG when neither the user (sentinel -1) nor the model
#: configuration provide an override (A1111's universal default)
DEFAULT_CFG = 7.0

#: default LRU cap for a model's prompt-encoding cache
_ENC_CACHE_MAX = 8

#: how many models keep prompt-encoding caches at once (per-model LRU)
_MAX_CACHED_MODELS = 8


class EncCache(OrderedDict):
    """LRU store of CPU-side encoded prompt tensors for one model.

    Shared across that model's per-GPU backend instances (each holds a
    reference to the same object): an encoding is a deterministic function
    of the encoder weights and the prompt, so every instance of the model
    on every GPU can reuse it. The generation queue pre-encodes a wildcard
    batch in one encoder session on the primary GPU, and every task of the
    batch is then a cache hit wherever it runs.

    ``max`` is the LRU cap; ``bump`` grows it (a pre-encoded batch must not
    be evicted before all of its tasks have run).
    """

    def __init__(self, max_entries: int = _ENC_CACHE_MAX) -> None:
        super().__init__()
        self.max = max_entries

    def bump(self, entries: int) -> None:
        if entries > self.max:
            self.max = entries


_shared_enc_caches: OrderedDict[tuple[str, str], EncCache] = OrderedDict()


def shared_enc_cache(family: str, model_id: str) -> EncCache:
    """The shared prompt-encoding cache for one (family, model_id)."""
    key = (family, model_id)
    cache = _shared_enc_caches.get(key)
    if cache is None:
        _shared_enc_caches[key] = cache = EncCache()
    _shared_enc_caches.move_to_end(key)
    while len(_shared_enc_caches) > _MAX_CACHED_MODELS:
        _shared_enc_caches.popitem(last=False)
    return cache


class GenerateRequest(BaseModel):
    # init_image/init_mask are numpy arrays passed in-process (never from JSON);
    # JSON callers leave them as None.
    model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20  # <= 0 = "model default": the backend resolves its built-in
    cfg: float = DEFAULT_CFG  # <= 0 = "model default": the backend resolves its built-in
    seed: int = -1  # -1 => random; batch callers pass explicit seeds
    sampler: str | None = None  # None => backend default
    scheduler: str | None = None
    model_id: str = ""
    model_name: str = ""
    loras: list[tuple[str, float]] = Field(default_factory=list)
    # Flux: the CFG value is the embedded guidance (distilled, one pass per
    # step, negative prompt ignored) when True; classic CFG with the
    # negative prompt (two passes per step) when False. Inert for SDXL.
    distilled: bool = True
    # Krea 2: True selects the turbo checkpoint class (8 steps, no CFG,
    # mu pinned to 1.15) over the raw class (12 steps, standard CFG 1.5,
    # resolution-derived mu). Set from the model config entry, never from
    # JSON. Inert for every other family.
    turbo: bool = False
    # inpainting
    init_image: np.ndarray | None = None  # HxWx3 uint8
    init_mask: np.ndarray | None = None  # HxW uint8, 0..255 (255 = inpaint here)
    denoise: float = 0.75
    # soft inpainting (Forge builtin extension): per-step latent blend
    # replacing the hard content lock + difference-based pixel composite
    # (inpaint modes only; ignored by backends that do not support it)
    soft_inpainting: SoftInpainting | None = None
    # self attention guidance (Forge builtin extension): an extra
    # unconditional pass sharpens the regions the model's self-attention
    # focuses on (all modes; requires cfg > 1, ignored by backends that
    # do not support it)
    sag: SelfAttentionGuidance | None = None

    def resolved_seed(self) -> int:
        import random

        return random.randrange(2**31) if self.seed == -1 else self.seed


@dataclass
class GenerateResult:
    image: np.ndarray  # HxWx3 uint8
    seed: int
    # The effective parameters the backend actually used (they differ from
    # the request only when it carried the "model default" sentinel, steps/
    # cfg <= 0); the PNG metadata records these, never the raw request
    # values.
    steps: int | None = None
    cfg: float | None = None
    # Flux only: the effective distilled/classic-CFG mode (None = the
    # backend is not Flux). A schnell-class checkpoint silently demotes a
    # distilled request to classic CFG, so this is not necessarily the
    # request's flag.
    distilled: bool | None = None
    # Krea 2 only: the effective raw/turbo checkpoint class (None = the
    # backend is not Krea 2).
    turbo: bool | None = None


ProgressCallback = Callable[[float, str], None]  # fraction 0..1, message


def format_loras(loras: list[tuple[str, float]] | tuple[tuple[str, float], ...]) -> str:
    """`name (weight), ...` for log messages; `-` when empty."""
    if not loras:
        return "-"
    return ", ".join(f"{name} ({weight:g})" for name, weight in loras)


class ModelBackend(ABC):
    """A loadable, generatable model (e.g. one SDXL checkpoint)."""

    #: unique id, e.g. the relative path "test-dir/JuggernautXL9.0"
    model_id: str
    #: display name in the UI
    name: str
    #: backend built-in sampler/scheduler names (used when request leaves them None)
    default_sampler: str = "Euler a"
    default_scheduler: str = "Karras"
    #: md5[:16] of the first 100MB of the checkpoint ("" for mocks)
    model_hash: str = ""
    #: whether attached LoRAs are actually applied to the weights (False:
    #: they are ignored and kept out of the PNG metadata)
    supports_loras: bool = True
    #: the model is a Flux (the "Distilled:" PNG metadata line applies)
    is_flux: bool = False
    #: the model is a Z-Image (single-pass: no CFG, effective cfg 1.0)
    is_zimage: bool = False
    #: the model is a Krea 2 (the raw/turbo checkpoint-class flag applies)
    is_krea2: bool = False
    #: whether the model can be moved out of VRAM (to RAM) after use
    #: (False: unload is a no-op and the model stays resident)
    can_park: bool = False

    @abstractmethod
    def generate(
        self, request: GenerateRequest, progress: ProgressCallback | None = None
    ) -> GenerateResult:
        """Run a single generation. `progress` may be called at will."""

    @abstractmethod
    def cancel(self) -> None:
        """Request cancellation of the in-flight generation (checked per step)."""

    def unload(self) -> None:
        """Move the model out of VRAM into RAM (inverse of :meth:`load`).

        The default does nothing: model types that cannot be moved simply
        stay in VRAM.
        """

    def pre_encode(self, pairs: list[tuple[str, str]],
                   reserve: int | None = None) -> None:
        """Encode every ``(prompt, negative_prompt)`` pair into the shared
        prompt-encoding cache (see :func:`shared_enc_cache`), ideally in a
        single text-encoder session.

        The generation queue calls this for wildcard batches BEFORE any of
        their tasks runs: without it every task of the batch carries a
        different resolved prompt, misses the cache, and re-loads / swaps
        the (multi-GiB) text encoder for itself. After this call every task
        of the batch is a cache hit and the encoder crosses the bus once
        per batch, not once per image. The default is a no-op (mock).
        """

    def load(self) -> None:
        """Move a previously parked model back to its GPU (inverse of
        :meth:`unload`). The default does nothing for non-parkable types."""


class MockBackend(ModelBackend):
    """Deterministic, CPU-only backend for tests and GPU-less development.

    Produces a seeded gradient + noise image so inpaint/resize/mask logic can
    be verified without any model weights.
    """

    def __init__(self, model_id: str = "mock/mock.safetensors", steps_delay: float = 0.0):
        self.model_id = model_id
        self.name = "Mock Model"
        self._cancelled = None
        self._steps_delay = steps_delay

    def cancel(self) -> None:
        self._cancelled = object()

    def generate(
        self, request: GenerateRequest, progress: ProgressCallback | None = None
    ) -> GenerateResult:
        import time

        seed = request.resolved_seed()
        self._cancelled = None
        # the "model default" sentinel (steps/cfg <= 0) resolves to the
        # built-in values, as the real backends do (the PNG metadata
        # records the effective values the backend used)
        steps = request.steps if request.steps > 0 else 20
        cfg = request.cfg if request.cfg > 0 else DEFAULT_CFG
        rng = np.random.default_rng(seed)
        h, w = request.height, request.width
        yy, xx = np.mgrid[0:h, 0:w]
        img = np.dstack(
            [
                (xx / max(w - 1, 1)) * 255,  # horizontal ramp, encodes width
                (yy / max(h - 1, 1)) * 255,  # vertical ramp, encodes height
                np.full((h, w), (seed % 251) * 255 // 250),  # encodes seed
            ]
        ).astype(np.float32)
        img += rng.standard_normal((h, w, 3)) * 12  # small noise, encodes seed
        if request.init_image is not None:
            # blend toward the init image where there is no mask
            if request.init_mask is not None:
                mask = request.init_mask.astype(np.float32) / 255.0
            else:
                mask = 0.0
            mask = mask[..., None] * request.denoise
            img = img * mask + request.init_image.astype(np.float32) * (1.0 - mask)
        for i in range(max(1, steps)):
            if self._cancelled is not None:
                raise InterruptedError("generation cancelled")
            if self._steps_delay:
                time.sleep(self._steps_delay)
            if progress:
                progress((i + 1) / steps, f"step {i + 1}/{steps}")
        return GenerateResult(
            image=np.clip(img, 0, 255).astype(np.uint8), seed=seed,
            steps=steps, cfg=cfg)
