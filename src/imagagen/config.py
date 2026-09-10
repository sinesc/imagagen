"""Project configuration and directory layout."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Root of the imagagen project plus all derived data directories."""

    root: Path

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def loras_dir(self) -> Path:
        return self.root / "loras"

    @property
    def brushes_dir(self) -> Path:
        return self.root / "brushes"

    @property
    def presets_dir(self) -> Path:
        return self.root / "presets"

    @property
    def model_configs_dir(self) -> Path:
        return self.root / "model-configs"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def segmentation_models_dir(self) -> Path:
        return self.models_dir / "segmentation"

    @property
    def hf_cache_dir(self) -> Path:
        """Persistent HuggingFace hub cache (tokenizers, configs, ...).

        Kept inside the project so the ~2 MB of hub files a pipeline's
        `from_single_file` still fetches are downloaded once and survive
        container rebuilds (the container's ~/.cache is ephemeral).
        """
        return self.models_dir / "hf_cache"


def load_config(root: Path | str | None = None) -> Config:
    """Build a Config from an explicit root, $IMAGEN_ROOT, or the cwd."""
    if root is None:
        root = os.environ.get("IMAGEN_ROOT", Path.cwd())
    config = Config(root=Path(root).expanduser().resolve())
    # Must happen before huggingface_hub is imported (it reads these at
    # import time): cache hub files in the project, keep the console quiet.
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(config.hf_cache_dir))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    # torchvision-less image-processor fallbacks are exactly what we want
    # (PIL backend); stop transformers from announcing them on first import
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    import logging

    # Keep the console clean: these loggers are noisy at import time but
    # harmless for our fp16 GPU-only usage.
    for name in ("huggingface_hub", "transformers"):
        logging.getLogger(name).setLevel(logging.ERROR)
    return config


def ensure_dirs(config: Config) -> None:
    """Create the data-directory layout if it does not exist yet."""
    for d in (
        config.models_dir,
        config.segmentation_models_dir,
        config.loras_dir,
        config.brushes_dir,
        config.presets_dir,
        config.model_configs_dir,
        config.output_dir,
        config.hf_cache_dir,
    ):
        d.mkdir(parents=True, exist_ok=True)
