"""Prompt wildcard processing.

Wildcard text files (``*.txt``) live in the project's ``user/wildcards/``
directory. A prompt (or negative prompt) may include one via ``__name__``,
where ``name`` is the file's path relative to the wildcards folder without
its extension, wrapped in double underscores -- e.g. ``__weather__`` or
``__test-dir/clouds__`` for a file in a subfolder.

A wildcard file may itself reference other wildcards (same syntax), in the
same folder or subfolders (parent folders are not addressable), so wildcards
can be composed and re-used.

On prompt processing each ``__name__`` is replaced by a pseudo-random
non-empty line of its file. The pick is deterministic for a given seed: the
same seed + prompt always resolves to the same text (each reference rolls
independently, in the order the references are encountered, which is what
makes the result depend on the seed *and* the processing order).

Two user-facing error conditions are raised as :class:`WildcardError`:

- **infinite recursion** -- while resolving a wildcard, one of its
  ancestors is referenced again (``a`` -> ``b`` -> ``a``);
- **missing wildcard** -- a referenced file does not exist (or points
  outside the wildcards folder).
"""

from __future__ import annotations

import random
import re
from pathlib import Path

# A wildcard token: ``__name__``. ``name`` is a relative path (no extension)
# of one or more well-formed segments separated by single slashes. A segment
# is word chars / dashes, optionally with single underscores, and can never
# start or end with a slash or underscore. This keeps the doubled-underscore
# delimiter unambiguous and rules out leading slashes, trailing slashes and
# parent (``..``) traversal.
_SEG = r"[A-Za-z0-9-]+(?:_[A-Za-z0-9-]+)*"
_WILDCARD_RE = re.compile(rf"__({_SEG}(?:/{_SEG})*)__")


class WildcardError(Exception):
    """User-facing wildcard error (missing file or infinite recursion)."""


def has_wildcards(text: str) -> bool:
    """Whether ``text`` contains at least one ``__name__`` reference."""
    return _WILDCARD_RE.search(text) is not None


def _find(base: Path, name: str) -> Path | None:
    """Resolve a wildcard name to its file, or None if it is not usable.

    ``name`` is a path RELATIVE TO ``base`` (no extension): a wildcard file
    can reference files in its own folder or subfolders, so ``base`` is the
    folder of the file doing the referencing (the wildcards root at the top
    level). Absolute paths and parent-folder traversal (``..``) are rejected,
    so a reference can never escape ``base`` (parent folders are out).
    """
    if not name or name.startswith("/") or name.startswith("\\"):
        return None
    if ".." in Path(name).parts:
        return None
    candidate = base / (name + ".txt")
    if not candidate.is_file():
        return None
    return candidate


def _lines(path: Path) -> list[str]:
    """Non-empty lines of a wildcard file (each stripped of surrounding
    whitespace); a trailing newline must not produce a phantom blank line."""
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def resolve(text: str, wildcards_dir: Path, seed: int) -> str:
    """Replace every ``__name__`` in ``text`` with a pseudo-random line of the
    referenced file.

    ``text`` is resolved against the wildcards root. A referenced file's own
    references are resolved relative to that file's folder, so a file can
    address files in its own folder or subfolders (but not parent folders).

    Deterministic for a given ``(text, wildcards_dir, seed)``. Nested
    wildcards are resolved depth-first (left to right); a single shared RNG
    seeded with ``seed`` is consumed in processing order, so the result
    depends on both the seed and the order the references are met.

    Raises :class:`WildcardError` on a missing file or infinite recursion.
    """
    rng = random.Random(seed)
    return _replace(text, wildcards_dir, rng, frozenset())


def _replace(text: str, base: Path, rng: random.Random, stack: frozenset[str]) -> str:
    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in stack:
            raise WildcardError(
                "infinite wildcard recursion: " + " -> ".join([*sorted(stack), name])
            )
        path = _find(base, name)
        if path is None:
            raise WildcardError(f"wildcard not found: __{name}__")
        lines = _lines(path)
        if not lines:
            raise WildcardError(f"wildcard has no lines: __{name}__")
        line = lines[rng.randrange(len(lines))]
        # the file's references are relative to its own folder
        sub_base = base / Path(name).parent if "/" in name else base
        return _replace(line, sub_base, rng, stack | {name})

    return _WILDCARD_RE.sub(repl, text)


def validate(text: str, wildcards_dir: Path) -> None:
    """Structural dry-run of ``text``: raises :class:`WildcardError` if any
    referenced wildcard is missing or would recurse, without caring which
    line gets picked (the seed is irrelevant to these errors). A text with no
    wildcard references validates trivially."""
    if has_wildcards(text):
        resolve(text, wildcards_dir, seed=0)
