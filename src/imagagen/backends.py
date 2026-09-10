"""Model backend interface.

The rest of imagagen (queue, generation, API) talks to this interface only,
so the CPU MockBackend can stand in for a real SDXL backend in tests and in
environments without a GPU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class GenerateRequest(BaseModel):
    # init_image/init_mask are numpy arrays passed in-process (never from JSON);
    # JSON callers leave them as None.
    model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg: float = 7.0
    seed: int = -1  # -1 => random; batch callers pass explicit seeds
    sampler: str | None = None  # None => backend default
    scheduler: str | None = None
    loras: list[tuple[str, float]] = Field(default_factory=list)
    # inpainting
    init_image: np.ndarray | None = None  # HxWx3 uint8
    init_mask: np.ndarray | None = None  # HxW uint8, 0..255 (255 = inpaint here)
    denoise: float = 0.75

    def resolved_seed(self) -> int:
        import random

        return random.randrange(2**31) if self.seed == -1 else self.seed


@dataclass
class GenerateResult:
    image: np.ndarray  # HxWx3 uint8
    seed: int


ProgressCallback = Callable[[float, str], None]  # fraction 0..1, message


class ModelBackend(ABC):
    """A loadable, generatable model (e.g. one SDXL checkpoint)."""

    #: unique id, e.g. the relative path "sdxl/foo.safetensors"
    model_id: str
    #: display name in the UI
    name: str

    @abstractmethod
    def generate(
        self, request: GenerateRequest, progress: ProgressCallback | None = None
    ) -> GenerateResult:
        """Run a single generation. `progress` may be called at will."""

    @abstractmethod
    def cancel(self) -> None:
        """Request cancellation of the in-flight generation (checked per step)."""


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
        for i in range(max(1, request.steps)):
            if self._cancelled is not None:
                raise InterruptedError("generation cancelled")
            if self._steps_delay:
                time.sleep(self._steps_delay)
            if progress:
                progress((i + 1) / request.steps, f"step {i + 1}/{request.steps}")
        return GenerateResult(image=np.clip(img, 0, 255).astype(np.uint8), seed=seed)
