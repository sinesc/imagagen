"""SDXL model backend. Phase 1 implementation; stub for now.

Contract: ``SdxlBackend.from_config(config)`` returns a backend instance, or
``None`` when no usable CUDA device or no model is found (caller falls back
to the MockBackend).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .backends import GenerateRequest, GenerateResult, ModelBackend

if TYPE_CHECKING:
    from .config import Config


class SdxlBackend(ModelBackend):
    def __init__(self, model_id: str = "", name: str = ""):
        self.model_id = model_id
        self.name = name

    @classmethod
    def from_config(cls, config: Config) -> ModelBackend | None:
        # Phase 1: detect CUDA + models/sdxl/*.safetensors, build diffusers pipeline.
        return None

    def generate(self, request: GenerateRequest, progress=None) -> GenerateResult:
        raise NotImplementedError("SDXL backend lands in Phase 1")

    def cancel(self) -> None:
        raise NotImplementedError("SDXL backend lands in Phase 1")
