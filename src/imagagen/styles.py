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
import re
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

    def extract_styles(
        self, prompt: str, negative_prompt: str
    ) -> tuple[list[Style], str, str]:
        """Forge-style restoration of the styles applied to a prompt.

        Given a fully expanded (styles already applied) positive and negative
        prompt, determine which of this store's styles were applied and the
        prompts with them removed. Peels one recognised style off at a time
        until none matches, then returns the peeled styles in the order that
        re-applies them to reproduce the input (i.e. the reverse of peel
        order).

        Used to import A1111/Forge images, whose metadata stores the expanded
        prompt rather than the style list. Returns
        ``(styles, prompt, negative_prompt)`` where ``styles`` is a list of
        :class:`Style` (empty when nothing matched).
        """
        with self._lock:
            raw_styles = self._read() + list(self._temp.values())
        # canonical style texts so matching is tolerant of whitespace / comma
        # formatting differences in the (expanded) import prompt
        applicable = [
            (normalize_prompt(s.prompt), normalize_prompt(s.negative_prompt), s)
            for s in raw_styles
        ]
        prompt = normalize_prompt(prompt)
        negative_prompt = normalize_prompt(negative_prompt)
        extracted: list[Style] = []
        while True:
            found = None
            for style_prompt, style_negative, style in applicable:
                if not style_prompt and not style_negative:
                    continue
                ok_pos, new_prompt = _extract_style_text(style_prompt, prompt)
                if not ok_pos:
                    continue
                ok_neg, new_negative = _extract_style_text(
                    style_negative, negative_prompt
                )
                if not ok_neg:
                    continue
                found = style
                prompt, negative_prompt = new_prompt, new_negative
                applicable.remove((style_prompt, style_negative, style))
                break
            if found is None:
                break
            extracted.append(found)
        return list(reversed(extracted)), prompt, negative_prompt

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


def normalize_prompt(text: str) -> str:
    """Canonical form of a prompt for tolerant style matching.

    Comma- and line-separated terms are split, whitespace inside each term is
    collapsed, empty terms dropped, and everything re-joined with a single
    ", " separator. This makes style identification (and the LoRA removal
    that feeds it) indifferent to extra spaces, newlines, doubled or missing
    commas, etc.
    """
    terms = [re.sub(r"\s+", " ", term).strip() for term in re.split(r"[,\n]+", text)]
    return ", ".join(term for term in terms if term)


# A prompt line whose first non-space/non-tab character is # is a
# comment: it is dropped before prompt processing (styles, wildcards,
# token counting, model encoding), but the RAW prompt -- comments
# included -- is what gets recorded in the image metadata.
_PROMPT_COMMENT_RE = re.compile(r"^[ \t]*#")


def strip_prompt_comments(text: str) -> str:
    """Remove comment lines from a prompt.

    Every other line is kept verbatim (blank lines and line endings
    included), so this only changes a prompt by deleting whole lines."""
    return "\n".join(
        line for line in text.split("\n")
        if not _PROMPT_COMMENT_RE.match(line)
    )


def _extract_style_text(style_text: str, prompt: str) -> tuple[bool, str]:
    """Inverse of :func:`_apply_one`, in canonical form (see
    :func:`normalize_prompt`). If ``style_text`` was applied last to yield
    ``prompt`` (via :func:`apply_styles`), return ``(True, prompt before it
    was applied)``; otherwise ``(False, prompt)`` unchanged.

    Mirrors Forge's ``extract_style_text_from_prompt``: a style carrying a
    ``{prompt}`` placeholder must wrap the whole prompt (prefix + suffix
    around it); a plain style must trail it (with the joining ", " removed).
    """
    if "{prompt}" in style_text:
        left, _, right = style_text.partition("{prompt}")
        if prompt.startswith(left) and prompt.endswith(right):
            return True, prompt[len(left): len(prompt) - len(right)]
    else:
        if prompt.endswith(style_text):
            remainder = prompt[: len(prompt) - len(style_text)]
            if remainder.endswith(", "):
                remainder = remainder[:-2]
            return True, remainder
    return False, prompt


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
