"""Setting presets (user/presets/*.json).

A preset captures some of the image-generation settings. Each setting has a
checkbox ("controls") deciding whether the preset controls that setting: on
select, only the checked settings are written into the form, everything else
is left untouched (concept.md).

File layout (one JSON per preset, filename = sanitized name):

    {"name": "Big square",
     "group": null,  # optional group label for the dropdown's optgroups
     "width": 1024, "height": 1024,
     "steps": null, "cfg": null, "seed": null, "batch": null,
     "sampler": null, "scheduler": null, "mode": null, "denoise": null,
     "mask_padding": null, "loras": null,
     "controls": {"width": true, "height": true}}
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .paths import sanitize_prompt

#: the settings a preset may control, in UI order
PRESET_FIELDS = (
    "width",
    "height",
    "rotation",
    "steps",
    "cfg",
    "seed",
    "batch",
    "sampler",
    "scheduler",
    "mode",
    "denoise",
    "mask_padding",
    "oversample",
    "loras",
)


class PresetError(Exception):
    """User-facing preset error."""


class Preset(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    # optional group label (the main UI's preset dropdown renders one
    # optgroup per group; empty-group presets are plain options)
    group: str | None = Field(default=None, max_length=120)
    # -1 = the auto-size sentinel (mask bounding box, inpaint modes only)
    width: int | None = Field(default=None, ge=-1, le=4096)
    height: int | None = Field(default=None, ge=-1, le=4096)
    # source rotation (degrees, positive = clockwise; None = leave as-is)
    rotation: int | None = Field(default=None, ge=-180, le=180)
    # steps/cfg: -1 = the "use model default" sentinel (same as the form)
    steps: int | None = Field(default=None, ge=-1, le=150)
    cfg: float | None = Field(default=None, ge=-1.0)
    seed: int | None = None
    batch: int | None = Field(default=None, ge=1, le=32)
    sampler: str | None = None
    scheduler: str | None = None
    mode: str | None = Field(default=None, pattern="^(generate|inpaint_masked|inpaint_not_masked)$")
    denoise: float | None = Field(default=None, ge=0.0, le=1.0)
    mask_padding: int | None = Field(default=None, ge=0, le=2048)
    oversample: float | None = Field(default=None, ge=0.1, le=4.0)
    loras: list[tuple[str, float]] | None = None
    controls: dict[str, bool] = Field(default_factory=dict)

    @field_validator("group", mode="before")
    @classmethod
    def _group_empty_to_none(cls, v):
        """The UI sends null for an empty group; be lenient about ""."""
        if v is not None and str(v).strip() == "":
            return None
        return v

    @field_validator("rotation")
    @classmethod
    def _rotation_multiple_of_45(cls, v: int | None) -> int | None:
        if v is not None and v % 45:
            raise ValueError("rotation must be a multiple of 45")
        return v

    def to_dict(self) -> dict:
        d = self.model_dump()
        # full field list, unchecked = false (unknown keys dropped)
        d["controls"] = {k: bool(d["controls"].get(k, False)) for k in PRESET_FIELDS}
        return d

    def controlled(self) -> dict:
        """Only the settings this preset controls (for UI application)."""
        out = {}
        d = self.model_dump()
        for k in PRESET_FIELDS:
            if d["controls"].get(k):
                out[k] = d[k]
        return out


class PresetSaveRequest(Preset):
    """POST /api/presets body. ``old_name`` names the preset being edited;
    when the name is changed, the old file is deleted (a rename instead of
    creating a new preset)."""

    old_name: str | None = Field(default=None, max_length=120)


class PresetStore:
    """Loads/saves user/presets/*.json. Thread-safe (saves are rare)."""

    def __init__(self, presets_dir: Path) -> None:
        self.dir = presets_dir
        self._lock = threading.RLock()

    def _path(self, name: str) -> Path:
        return self.dir / f"{sanitize_prompt(name, max_len=120)}.json"

    def all(self) -> list[Preset]:
        """Every loadable preset, in filename (name) order."""
        with self._lock:
            out = []
            if self.dir.is_dir():
                for f in sorted(self.dir.glob("*.json")):
                    p = self.get(f.stem)
                    if p is not None:
                        out.append(p)
            return out

    def names(self) -> list[str]:
        return [p.name for p in self.all()]

    def get(self, id_: str) -> Preset | None:
        """Look up by sanitized id or by display name (both map to the
        same file, since files are named after the sanitized name)."""
        with self._lock:
            f = self.dir / f"{sanitize_prompt(id_, max_len=120)}.json"
            if not f.is_file():
                return None
            try:
                return Preset.model_validate(json.loads(f.read_text("utf-8")))
            except (OSError, json.JSONDecodeError, Exception):
                return None

    def save(self, preset: Preset, old_name: str | None = None) -> str:
        """Save the preset (in place if the name is unchanged). If ``old_name``
        is given and maps to a different file, the old file is deleted:
        renaming a preset renames it instead of creating a new one."""
        with self._lock:
            id_ = sanitize_prompt(preset.name, max_len=120)
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path(preset.name).with_suffix(".json.tmp")
            tmp.write_text(json.dumps(preset.to_dict(), indent=2, ensure_ascii=False) + "\n",
                           "utf-8")
            tmp.replace(self._path(preset.name))
            if old_name:
                old_id = sanitize_prompt(old_name, max_len=120)
                if old_id != id_:
                    self._path(old_name).unlink(missing_ok=True)
            return id_

    def delete(self, name: str) -> bool:
        with self._lock:
            f = self._path(name)
            if not f.is_file():
                return False
            f.unlink()
            return True
