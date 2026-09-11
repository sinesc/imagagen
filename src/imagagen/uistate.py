"""Persistence of the web UI's input state across server sessions.

The last-used form values (model, prompt, negative prompt, width, height,
steps, CFG, seed, batch, sampler, scheduler) are serialized as JSON to
``<root>/ui-state.json``. A restarted server serves the saved state via
``GET /api/ui-state`` and the UI restores it into the form, so the page
picks up exactly where the user left off.

JSON was chosen over INI/TOML because the state is a flat bag of typed
values (strings, ints, floats, nulls) that the UI produces as JSON anyway;
the INI file is reserved for operator configuration.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class UiState(BaseModel):
    """The UI form fields, with the same limits as the HTML inputs.

    Validation here (not just in the browser) keeps garbage from being
    persisted: the browser's min/max attributes do not constrain the JSON
    the UI POSTs.
    """

    model: str | None = None
    prompt: str = ""
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=64, le=2048)
    height: int = Field(default=1024, ge=64, le=2048)
    steps: int = Field(default=25, ge=1, le=150)
    cfg: float = Field(default=7.0, ge=0)
    seed: int = -1  # -1 = random (A1111 convention)
    batch: int = Field(default=1, ge=1, le=32)
    sampler: str | None = None
    scheduler: str | None = None


class UiStateStore:
    """Loads/saves :class:`UiState` as JSON at a fixed path.

    Corrupt or partial files never crash a server start: they fall back
    to defaults. Saves are atomic (write temp file + rename) so a crash
    mid-write cannot leave a truncated state file behind.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> UiState:
        try:
            raw = json.loads(self.path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return UiState()
        if not isinstance(raw, dict):
            return UiState()
        try:
            return UiState.model_validate(raw)
        except ValidationError:
            # e.g. an old file whose limits changed: start fresh rather
            # than fail to boot
            return UiState()

    def save(self, state: UiState) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(state.model_dump(), indent=2, ensure_ascii=False) + "\n",
            "utf-8",
        )
        tmp.replace(self.path)
