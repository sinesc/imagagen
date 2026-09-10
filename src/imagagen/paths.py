"""Output-file naming: leading integer derived from the highest existing one."""

from __future__ import annotations

import re
from pathlib import Path

_LEADING_INT_RE = re.compile(r"^(\d+)-")


def sanitize_prompt(prompt: str, max_len: int = 80) -> str:
    """Lowercase, [a-z0-9-] only, runs collapsed to single dashes."""
    s = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    s = s[:max_len].rstrip("-")
    return s or "image"


def next_output_number(output_dir: Path) -> int:
    """Highest leading integer among existing files in output_dir, plus one."""
    highest = 0
    if output_dir.is_dir():
        for p in output_dir.iterdir():
            if not p.is_file():
                continue
            m = _LEADING_INT_RE.match(p.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return highest + 1


def next_output_path(output_dir: Path, prompt: str) -> Path:
    """e.g. output/ contains '123-foo.png' -> output/124-<sanitized prompt>.png"""
    return output_dir / f"{next_output_number(output_dir)}-{sanitize_prompt(prompt)}.png"
