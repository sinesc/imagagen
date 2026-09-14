"""Prompt styles (styles.csv, A1111-style).

A style is a named fragment of prompt text (plus optional negative-prompt
text and an optional optgroup for grouping it in the UI dropdown). The UI
lets the user pick an ordered list of styles; before generation the styles
are applied to the prompt in list order (see :func:`apply_styles`).

File layout (single CSV at the project root, utf-8-sig to survive an
editor-inserted BOM):

    name,prompt,negative_prompt,optgroup
    Analog Film,analog film photo {prompt} . faded film, 35mm photo,,"Photo Styles"

Temporary styles (created when importing image metadata whose styles are
missing from the CSV, e.g. deleted or from another install) live only in
memory, use the "Session" optgroup, and are never written to the CSV.
"""

from __future__ import annotations

import csv
import threading
from pathlib import Path

from pydantic import BaseModel, Field

#: optgroup used for in-memory temporary styles (concept: "Session")
TEMP_GROUP = "Session"

_FIELDS = ("name", "prompt", "negative_prompt", "optgroup")


class StyleError(Exception):
    """User-facing style error."""


class Style(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    prompt: str = ""
    negative_prompt: str = ""
    optgroup: str = ""

    def to_dict(self, temporary: bool = False) -> dict:
        d = self.model_dump()
        d["temporary"] = temporary
        return d


class StyleStore:
    """Loads/saves styles.csv plus the in-memory temporary styles.

    Thread-safe (saves are rare). The CSV is re-read on every access --
    it is small and single-user, and this keeps the store in sync with
    manual edits of the file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self._temp: dict[str, Style] = {}

    # --- file styles ---

    def _read(self) -> list[Style]:
        if not self.path.is_file():
            return []
        out: list[Style] = []
        with self.path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                try:
                    s = Style(
                        name=(row.get("name") or "").strip(),
                        prompt=row.get("prompt") or "",
                        negative_prompt=row.get("negative_prompt") or "",
                        optgroup=(row.get("optgroup") or "").strip(),
                    )
                except Exception:
                    continue  # skip malformed rows (e.g. empty name)
                out.append(s)
        return out

    def _write(self, styles: list[Style]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".csv.tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(_FIELDS))
            w.writeheader()
            for s in styles:
                w.writerow(s.model_dump())
        tmp.replace(self.path)

    # --- unified view (file styles + temporaries) ---

    def all(self) -> list[dict]:
        """All styles as dicts (temporaries flagged), file order + temp order."""
        with self._lock:
            out = [s.to_dict() for s in self._read()]
            out += [s.to_dict(temporary=True) for s in self._temp.values()]
            return out

    def get(self, name: str) -> Style | None:
        """Look up by exact name (file styles first, then temporaries)."""
        with self._lock:
            for s in self._read():
                if s.name == name:
                    return s
            return self._temp.get(name)

    def save(self, style: Style) -> None:
        """Create or update a file style (upsert by name)."""
        with self._lock:
            styles = self._read()
            for i, s in enumerate(styles):
                if s.name == style.name:
                    styles[i] = style
                    break
            else:
                styles.append(style)
            self._write(styles)
            # a real style now exists: a same-named temporary is redundant
            self._temp.pop(style.name, None)

    def delete(self, name: str) -> bool:
        with self._lock:
            styles = self._read()
            kept = [s for s in styles if s.name != name]
            if len(kept) == len(styles):
                return False
            self._write(kept)
            return True

    # --- temporary styles (in-memory only) ---

    def add_temporary(self, style: Style) -> None:
        """Register a style seen in imported metadata that is missing from
        the CSV. Never persisted; grouped under TEMP_GROUP."""
        with self._lock:
            if any(s.name == style.name for s in self._read()):
                return  # a real style now exists: nothing to temp
            self._temp[style.name] = style.model_copy(
                update={"optgroup": TEMP_GROUP}
            )


def _apply_one(text: str, current: str) -> str:
    """Apply one style text to the accumulated prompt: the {prompt}
    placeholder is replaced with everything so far; without it the style
    text is appended (comma-separated, the A1111 convention)."""
    if not text:
        return current
    if "{prompt}" in text:
        return text.replace("{prompt}", current)
    if current:
        return f"{current}, {text}"
    return text


def apply_styles(
    prompt: str,
    negative_prompt: str,
    names: list[str],
    get,
) -> tuple[str, str]:
    """Apply styles (by name, in list order) to prompt / negative prompt.

    ``get(name)`` returns a :class:`Style` or None; unknown names (e.g.
    deleted between UI render and generation) are skipped.
    """
    for name in names:
        s = get(name)
        if s is None:
            continue
        prompt = _apply_one(s.prompt, prompt)
        negative_prompt = _apply_one(s.negative_prompt, negative_prompt)
    return prompt, negative_prompt
