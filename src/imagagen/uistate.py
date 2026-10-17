"""Persistence of the web UI's input state across server sessions.

The last-used form values (model, prompt, negative prompt, width, height,
steps, CFG, seed, batch, sampler, scheduler, denoise strength) are
serialized as JSON to
``<root>/user/ui-state.json``. A restarted server serves the saved state via
``GET /api/ui-state`` and the UI restores it into the form, so the page
picks up exactly where the user left off.

JSON was chosen over INI/TOML because the state is a flat bag of typed
values (strings, ints, floats, nulls) that the UI produces as JSON anyway;
the INI file is reserved for operator configuration.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


# Canvas tools, same list as the TOOLS array in tools.js
TOOL_NAMES = ("brush", "mask", "clone", "smudge", "crop", "copy", "paste")


class ToolSettings(BaseModel):
    """What one tool remembers for the next time it is used.

    Same limits as the toolbar UI: brush size clamped to 10-500 px,
    opacity slider 0.05-1. The colour applies to the brush tool only
    (mask strokes ignore it; clone/smudge/crop/copy/paste have none).
    """

    size: int = Field(10, ge=10, le=500)
    opacity: float = Field(1.0, ge=0.05, le=1.0)
    brush: str = "circle"  # shape name as listed by /api/brushes
    screen: bool = False  # size in screen px instead of canvas px
    color: str | None = None  # "#rrggbb"

    @field_validator("brush")
    @classmethod
    def _brush_name(cls, v: str) -> str:
        # a brush file name, never a path
        if not v or "/" in v or "\\" in v:
            raise ValueError("invalid brush name")
        return v

    @field_validator("color")
    @classmethod
    def _color_hex(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"#[0-9a-fA-F]{6}", v):
            raise ValueError("colour must be #rrggbb")
        return v


class ToolState(BaseModel):
    """Current canvas tool + the per-tool remembered settings."""

    # `copy_area` is validated from its alias too, so the JSON the browser
    # posts ("copy") and a plain model_dump ("copy_area") both load
    model_config = ConfigDict(populate_by_name=True)

    current: str = "brush"
    # spec defaults match tools.js (the mask tool starts at 30 px)
    brush: ToolSettings = Field(default_factory=ToolSettings)
    mask: ToolSettings = Field(default_factory=lambda: ToolSettings(size=30))
    clone: ToolSettings = Field(default_factory=ToolSettings)
    smudge: ToolSettings = Field(default_factory=ToolSettings)
    crop: ToolSettings = Field(default_factory=ToolSettings)
    # copy/paste drag no brush at all (copy drags a rectangle, paste drops
    # the copied area); entries kept so every tool has a settings row. The
    # copy tool's field is `copy_area` -- a field named `copy` would shadow
    # pydantic's BaseModel.copy(); the JSON key stays "copy" (alias, and
    # saved/served with model_dump(by_alias=True))
    copy_area: ToolSettings = Field(
        default_factory=ToolSettings, alias="copy", serialization_alias="copy"
    )
    paste: ToolSettings = Field(default_factory=ToolSettings)

    @field_validator("current")
    @classmethod
    def _current_tool(cls, v: str) -> str:
        if v not in TOOL_NAMES:
            raise ValueError(f"unknown tool {v!r}")
        return v


class UiState(BaseModel):
    """The UI form fields, with the same limits as the HTML inputs.

    Validation here (not just in the browser) keeps garbage from being
    persisted: the browser's min/max attributes do not constrain the JSON
    the UI POSTs.
    """

    model: str | None = None
    prompt: str = ""
    negative_prompt: str = ""
    # -1 = auto size: mask bounding box (inpaint modes only; see generation.py)
    width: int = Field(default=1024, ge=-1, le=4096)
    height: int = Field(default=1024, ge=-1, le=4096)
    # -1 = "use model default" (the model configuration's value, then built-in);
    # the default IS the sentinel, so an unconfigured field rides on the
    # resolution chain instead of hard-coding a value
    steps: int = Field(default=-1, ge=-1, le=150)
    cfg: float = Field(default=-1.0, ge=-1)
    seed: int = -1  # -1 = random (A1111 convention)
    batch: int = Field(default=1, ge=1, le=32)
    sampler: str | None = None  # null = "use model default"
    scheduler: str | None = None  # null = "use model default"
    denoise: float = Field(1.0, ge=0.0, le=1.0)
    # context padding around the masked area (canvas px per side),
    # "inpaint masked" only; same limits as the HTML input
    mask_padding: int = Field(default=8, ge=0, le=2048)
    # scales the generation size (user or mask-derived) before rounding up
    # to a multiple of 16 (inpaint modes only); same limits as the HTML input
    oversample: float = Field(default=1.0, ge=0.1, le=4.0)
    # rotate the source before generating (UI dropdown; 0 = default)
    rotation: int = Field(default=0, ge=-180, le=180)

    @field_validator("rotation")
    @classmethod
    def _rotation_multiple_of_45(cls, v: int) -> int:
        if v % 45:
            raise ValueError("rotation must be a multiple of 45")
        return v
    # soft inpainting: the enabled flag AND the option values are
    # model-configuration settings (per t2i/i2i, enabled by default), not
    # persisted here
    # self attention guidance: the enabled flag AND the option values are
    # model-configuration settings (per t2i/i2i), not persisted here
    # user-selected LoRAs, [[name, weight], ...] (model-config LoRAs are
    # prepended server-side at generation time)
    loras: list[tuple[str, float]] = Field(default_factory=list)
    # LoRAs shown toggled off in the UI (not applied, kept in the list)
    disabled_loras: list[tuple[str, float]] = Field(default_factory=list)
    # user-selected styles (names from styles.csv), in list order
    styles: list[str] = Field(default_factory=list)
    # style names shown toggled off in the UI (not applied, kept in the list)
    disabled_styles: list[str] = Field(default_factory=list)
    # person mask generation (Phase 5): the selected segmentation model
    # (file name in models/segmentation/), the last-used class selection
    # (empty = "all classes except background"), the confidence threshold
    # (0..1) and the mask edge feather radius in canvas px (0 = hard edge)
    seg_model: str | None = None
    seg_classes: list[str] = Field(default_factory=list)
    seg_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    seg_feather: float = Field(default=0.0, ge=0.0, le=64.0)
    # the thumbnail checked as "reference image" (canvas entity id; the
    # generation conditions on it). Not a generation parameter -- it is a
    # UI selection, allowed by the ui-state exclusions rule. The server
    # drops it on serve when the entity no longer exists (ids do not
    # survive a server restart).
    reference_image_id: str | None = None
    # the caption sections enabled in the UI tag row (None = all;
    # dropped together with reference_image_id when the entity is gone)
    reference_sections: list[str] | None = None
    # sidebar (control panel) width in px, set by dragging its right border;
    # same clamped range as the UI's drag handler
    sidebar_width: int = Field(default=300, ge=280, le=800)
    # canvas tools: current tool + per-tool remembered settings (the
    # toolbar used to keep these in localStorage; they now ride in the
    # ui-state so a fresh browser on the same server restores them)
    tools: ToolState = Field(default_factory=ToolState)

    # The generation mode and the selected preset are deliberately NOT
    # persisted: the UI always starts in "generate" mode (there is nothing
    # to inpaint yet) with the preset dropdown at "—". Old state files
    # that still contain "mode"/"preset" keys are loaded fine (unknown
    # keys are ignored).


class UiStateStore:
    """Loads/saves :class:`UiState` as JSON at a fixed path.

    The latest state is always kept in memory: a running server serves it
    via :meth:`load` even when disk persistence is disabled
    (``persist=False``), so a browser reload still restores the settings.
    ``persist`` only controls whether saves are also written to disk.

    Corrupt or partial files never crash a server start: they fall back
    to defaults. Saves are atomic (write temp file + rename) so a crash
    mid-write cannot leave a truncated state file behind.
    """

    def __init__(self, path: Path, persist: bool = True) -> None:
        self.path = path
        self.persist = persist
        self._state: UiState | None = None  # in-memory copy of the latest state

    def load(self) -> UiState:
        if self._state is not None:
            return self._state
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = None
        if not isinstance(raw, dict):
            state = UiState()
        else:
            try:
                state = UiState.model_validate(raw)
            except ValidationError:
                # e.g. an old file whose limits changed: start fresh rather
                # than fail to boot
                state = UiState()
        self._state = state
        return state

    def save(self, state: UiState) -> None:
        self._state = state
        if not self.persist:
            return
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(state.model_dump(by_alias=True), indent=2, ensure_ascii=False) + "\n",
            "utf-8",
        )
        tmp.replace(self.path)
