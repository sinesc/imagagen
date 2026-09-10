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


def load_config(root: Path | str | None = None) -> Config:
    """Build a Config from an explicit root, $IMAGEN_ROOT, or the cwd."""
    if root is None:
        root = os.environ.get("IMAGEN_ROOT", Path.cwd())
    return Config(root=Path(root).expanduser().resolve())


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
    ):
        d.mkdir(parents=True, exist_ok=True)
