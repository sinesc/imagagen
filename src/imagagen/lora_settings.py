"""LoRA settings: a .json file stored next to each LoRA (.safetensors).

Field keys match Forge's files ("preferred weight", ...) plus the
imagagen-only "optional activation text" (toggleable prompt tags, see
:func:`apply_lora_texts`). Unknown extra fields (e.g. Forge's "sd version")
are ignored on read and preserved when saving, so files remain
Forge-compatible.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from pydantic import BaseModel, Field

from .models import list_loras, lora_paths
from .styles import apply_one


class LoraSettings(BaseModel):
    description: str = ""
    preferred_weight: float = Field(default=0.0, ge=-10, le=10)
    activation_text: str = ""
    # imagagen-only addition (Forge files never carry it): the same idea as
    # "activation text", but comma separated into TAGS the UI can switch on
    # and off individually
    optional_activation_text: str = ""
    negative_text: str = ""
    notes: str = ""


#: settings field -> key used in the .json files (Forge-compatible)
_FILE_KEYS = {
    "description": "description",
    "preferred_weight": "preferred weight",
    "activation_text": "activation text",
    "optional_activation_text": "optional activation text",
    "negative_text": "negative text",
    "notes": "notes",
}

_lock = threading.Lock()  # settings saves are rare; serialize them


def settings_path(name: str, config) -> Path | None:
    """The .json path for a discovered LoRA (next to its .safetensors),
    or None when the name is not a discovered LoRA."""
    f = lora_paths(config).get(name)
    return f.with_suffix(".json") if f else None


def _read_file(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _from_file(data: dict) -> LoraSettings:
    kwargs = {}
    for field, key in _FILE_KEYS.items():
        if key in data:
            kwargs[field] = data[key]
    try:
        return LoraSettings.model_validate(kwargs)
    except Exception:
        return LoraSettings()


def _to_file(s: LoraSettings) -> dict:
    return {key: getattr(s, field) for field, key in _FILE_KEYS.items()}


def get_settings(name: str, config) -> dict | None:
    """Settings for one LoRA as a dict, or None when the LoRA is unknown or
    has no settings file (a file may miss individual fields; those
    default)."""
    p = settings_path(name, config)
    if p is None or not p.is_file():
        return None
    with _lock:
        return _to_file(_from_file(_read_file(p)))


def split_tags(text: str) -> list[str]:
    """Comma-separated optional activation text -> its individual tags
    (surrounding whitespace stripped, empty parts dropped, order kept,
    duplicates kept).

    The UI shows one toggle button per tag of the selected LoRAs, so a
    single text like ``blue eyes, 8k`` can be switched on and off tag by
    tag. The UI mirrors this split (app.js ``splitTags``).
    """
    return [t.strip() for t in (text or "").split(",") if t.strip()]


def activation_tag_universe(
    loras: list[tuple[str, float]], config
) -> list[str]:
    """Every optional-activation-text tag the given (enabled) LoRAs
    contribute, in LoRA list order, deduped: the tag universe of a
    request."""
    out: list[str] = []
    for name, _weight in loras:
        s = get_settings(name, config)
        if s is None:
            continue
        for tag in split_tags(s.get("optional activation text", "")):
            if tag not in out:
                out.append(tag)
    return out


def apply_lora_texts(
    prompt: str,
    negative_prompt: str,
    loras: list[tuple[str, float]],
    config,
    disabled_tags: list[str] | None = None,
) -> tuple[str, str]:
    """Apply each enabled LoRA's activation/negative text to the prompt /
    negative prompt (in LoRA list order).

    Each text is applied exactly like a style's text (see :func:`apply_one`):
    a ``{prompt}`` placeholder is replaced with the prompt so far, otherwise
    the text is appended comma-separated; empty texts are a no-op. This is
    processing-only: the saved image metadata records the LoRAs by
    name/weight as before, not the expanded texts.

    A LoRA's plain ``activation text`` always applies in full (Forge
    behaviour). Its ``optional activation text`` is comma separated into
    tags: ``disabled_tags`` lists the ones the UI switched OFF, which are
    dropped from that text before it is applied (the remaining ones keep
    their order). Negative text is never split into tags.
    """
    off = set(disabled_tags or ())
    for name, _weight in loras:
        s = get_settings(name, config)
        if s is None:
            continue
        # get_settings dicts carry the .json file keys ("activation text",
        # "optional activation text", ...)
        prompt = apply_one(s.get("activation text", ""), prompt)
        optional = s.get("optional activation text", "")
        if optional and off:
            optional = ", ".join(
                t for t in split_tags(optional) if t not in off)
        prompt = apply_one(optional, prompt)
        negative_prompt = apply_one(s.get("negative text", ""), negative_prompt)
    return prompt, negative_prompt


def all_settings(config) -> dict:
    """name -> settings dict for every discovered LoRA that has a
    settings file."""
    return {
        n: s
        for n in list_loras(config)
        if (s := get_settings(n, config)) is not None
    }


def save_settings(name: str, s: LoraSettings, config) -> dict:
    """Create or update a LoRA's settings file, preserving any unknown
    extra fields already in the file. Returns the saved settings dict."""
    p = settings_path(name, config)
    if p is None:
        raise KeyError(name)
    with _lock:
        data = _read_file(p) if p.is_file() else {}
        data.update(_to_file(s))
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(data, indent=4, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp.replace(p)
    return _to_file(s)
