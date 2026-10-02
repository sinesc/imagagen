"""PNG generation metadata.

Generated images carry their settings in a `parameters` text chunk as a
JSON document (imagagen format, keyed exactly like the legacy A1111/Forge
text parse result, so both formats normalize to the same dict on read).

The raw (unapplied) prompt and negative prompt are recorded, not the
style-applied ones -- the style LIST travels in the document as well, so
re-import re-applies the styles exactly like a new generation would.

Reading is lenient: A1111/Forge images (and pre-JSON imagagen PNGs) store
the classic "parameters" TEXT format and are still importable -- the legacy
text parser in :func:`parse_parameters` covers those, including the
Forge-specific bits (LoRAs baked into the prompt text, `Lora hashes`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from PIL import Image, PngImagePlugin

from .sag import SelfAttentionGuidance
from .soft_inpaint import SoftInpainting

_FORMAT = "imagagen"
_VERSION = "v0.2.0"

# A1111/Forge embed LoRAs in the prompt text as ``<lora:NAME:WEIGHT>`` (unlike
# imagagen, which records them on dedicated ``Lora:`` meta lines). The name is
# everything up to the last colon; the weight is the (signed) number after it.
_LORA_RE = re.compile(r"<lora:(.+?):(-?\d+(?:\.\d+)?)>")

# one entry of a Forge "Lora hashes" value: `name: <12-char hex>`
_LORA_HASH_ENTRY_RE = re.compile(r"^(.*?):\s*([0-9a-fA-F]{12})$")


def forge_lora_hash(filename: str) -> str:
    """The hash Forge stores per LoRA in the `Lora hashes:` meta line.

    Despite the name, Forge does NOT hash the file contents: it is the first
    12 hex chars of the sha256 of the file's *basename* (extension included
    -- LoRA shorthash, vs 10 chars for checkpoints) --
    `forge_fake_calculate_sha256`, see
    `extensions-builtin/sd_forge_lora/network.py::read_hash`.
    """
    return hashlib.sha256(os.path.basename(filename).encode("utf-8")).hexdigest()[:12]


def _split_lora_hash_entries(text: str) -> list[tuple[str, str]]:
    """Split a (quote-stripped) `Lora hashes` value into (name, hash) pairs."""
    entries = []
    for part in text.split(", "):
        name, sep, h = part.rpartition(": ")
        if sep and re.fullmatch(r"[0-9a-fA-F]{12}", h):
            entries.append((name, h.lower()))
    return entries


def extract_prompt_loras(prompt: str) -> tuple[list[tuple[str, float]], str]:
    """Pull ``<lora:NAME:WEIGHT>`` tokens out of a (style-expanded) prompt.

    A1111/Forge bake LoRAs into the prompt text instead of recording them, so
    they must be recovered on import and handed to the normal LoRA selection
    logic. Returns ``(loras, remainder)``: ``loras`` is a list of
    ``(name, weight)`` in order of appearance, and ``remainder`` is the prompt
    with the tokens removed (their surrounding separator commas / whitespace
    are cleaned up by :func:`imagagen.styles.normalize_prompt` downstream).
    """
    loras = [(m.group(1).strip(), float(m.group(2))) for m in _LORA_RE.finditer(prompt)]
    return loras, _LORA_RE.sub("", prompt)


def build_parameters(
    prompt: str,
    negative_prompt: str,
    steps: int,
    sampler: str,
    scheduler: str,
    cfg: float,
    seed: int,
    width: int,
    height: int,
    model_name: str,
    model_hash: str,
    denoise: float | None = None,
    loras: list[tuple[str, float]] | None = None,
    disabled_loras: list[tuple[str, float]] | None = None,
    styles: list[dict] | None = None,
    soft_inpainting: SoftInpainting | None = None,
    sag: SelfAttentionGuidance | None = None,
    distilled: bool | None = None,
) -> str:
    """Serialize the `parameters` text chunk as a JSON document.

    The document carries exactly the keys (and compatible value shapes) the
    legacy A1111/Forge text parser in :func:`parse_parameters` used to
    return, so old and new PNGs normalize to identical dicts on read:

    * ``prompt`` / ``negative_prompt`` are the RAW texts -- newlines and
      comments included (the line-based legacy format could not store a
      multi-line negative prompt at all).
    * ``loras`` / ``disabled_loras``: ``name@weight`` strings, like the
      legacy ``Lora:`` / ``Disabled LoRAs:`` entries.
    * ``styles``: entries with "name", "prompt", "negative_prompt" and
      "enabled" -- the raw (unapplied) prompt is recorded, not the
      style-applied one, so re-import re-applies the styles exactly like a
      new generation would.
    * ``soft_inpainting`` / ``sag``: the enabled flag plus the option values
      under the legacy meta keys (``Soft inpainting enabled`` & co.,
      ``sag_enabled`` & co.). The enabled flags are written unconditionally
      (like the Forge extension) so a dropped image can restore the
      checkbox in BOTH states; absent = the option group was not used.
    * ``distilled``: Flux only (None = not a Flux model, key omitted);
      written in BOTH states like the SAG flags.
    """
    doc: dict[str, Any] = {
        "format": _FORMAT,
        "version": _VERSION,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "sampler": sampler,
        "scheduler": scheduler,
        "cfg": cfg,
        "seed": seed,
        "width": width,
        "height": height,
        "model": model_name,
        "model_hash": model_hash,
    }
    if denoise is not None:
        # A1111/Forge's inpaint strength: importing restores the slider
        doc["denoise"] = denoise
    if loras:
        doc["loras"] = [f"{n}@{w:g}" for n, w in loras]
    if disabled_loras:
        doc["disabled_loras"] = [f"{n}@{w:g}" for n, w in disabled_loras]
    if styles:
        doc["styles"] = [
            {
                "name": s.get("name", ""),
                "prompt": s.get("prompt", ""),
                "negative_prompt": s.get("negative_prompt", ""),
                "enabled": bool(s.get("enabled", True)),
            }
            for s in styles
        ]
    if soft_inpainting is not None:
        # written unconditionally, so a dropped image can restore the
        # checkbox in BOTH states
        s = soft_inpainting
        doc["Soft inpainting enabled"] = _bool_str(s.enabled)
        doc["Soft inpainting schedule bias"] = s.mask_blend_power
        doc["Soft inpainting preservation strength"] = s.mask_blend_scale
        doc["Soft inpainting transition contrast boost"] = s.inpaint_detail_preservation
        doc["Soft inpainting mask influence"] = s.composite_mask_influence
        doc["Soft inpainting difference threshold"] = s.composite_difference_threshold
        doc["Soft inpainting difference contrast"] = s.composite_difference_contrast
    if distilled is not None:
        # written unconditionally, so a dropped image records the mode in
        # BOTH states
        doc["Distilled"] = _bool_str(distilled)
    if sag is not None:
        # written unconditionally, so a dropped image can restore the
        # checkbox in BOTH states
        doc["sag_enabled"] = _bool_str(sag.enabled)
        doc["sag_scale"] = sag.scale
        doc["sag_blur_sigma"] = sag.blur_sigma
        doc["sag_threshold"] = sag.threshold
    return json.dumps(doc, ensure_ascii=False, separators=(",", ": "))


def _bool_str(b: bool) -> str:
    # "True"/"False" (str(bool)): matches what the legacy text parser used to
    # return for the enabled flags and what the UI compares against
    return str(bool(b))


def parse_parameters(text: str) -> dict[str, Any]:
    """Parse a `parameters` chunk back into settings.

    Accepts both the imagagen JSON format and the legacy A1111/Forge text
    format (needed to import foreign images and pre-JSON imagagen PNGs).
    Both normalize to the same dict shape. Lenient: unknown keys are kept
    verbatim, missing ones are omitted.
    """
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            doc = json.loads(stripped)
        except json.JSONDecodeError:
            doc = None
        # a genuine imagagen document, not just text that happens to be
        # valid JSON (e.g. a single-number prompt)
        if isinstance(doc, dict) and doc.get("format") == _FORMAT and "prompt" in doc:
            return doc
    out: dict[str, Any] = {}
    lines = text.splitlines()
    prompt_lines: list[str] = []
    meta_line = None
    for line in lines:
        is_meta = line.startswith("Negative prompt:") or bool(
            re.match(r"^\s*Steps:\s*", line)
        )
        if meta_line is None and is_meta:
            meta_line = line
        if meta_line is None:
            prompt_lines.append(line)
    if meta_line is None:
        meta_line = ""
    if meta_line.startswith("Negative prompt:"):
        # negative prompt is line 2, the meta key:value list is line 3+
        idx = lines.index(meta_line)
        neg = meta_line.split(":", 1)[1].strip()
        prompt = "\n".join(prompt_lines).strip()
        meta_line = lines[idx + 1] if idx + 1 < len(lines) else ""
        out["negative_prompt"] = neg
    else:
        prompt = "\n".join(prompt_lines).strip()
    out["prompt"] = prompt

    # "Lora hashes" (Forge): its value is one or more `name: <12-hex>`
    # entries joined with the same ", " the meta line itself uses, so a
    # multi-LoRA value is split across parts. Re-join the entry parts
    # before the generic key:value split below (and mark them so the split
    # doesn't misread the later parts as their own keys).
    parts = meta_line.split(", ")
    lora_hash_entries: list[tuple[str, str]] = []
    skip: set[int] = set()
    i = 0
    while i < len(parts):
        if parts[i].split(":", 1)[0].strip() != "Lora hashes":
            i += 1
            continue
        value = parts[i].split(":", 1)[1].strip()
        if value.startswith('"'):
            # whole value quoted (some Forge versions)
            buf, j = value[1:], i
            while '"' not in buf and j + 1 < len(parts):
                j += 1
                buf = buf + ", " + parts[j]
                skip.add(j)
            close = buf.find('"')
            found = _split_lora_hash_entries(
                buf[:close] if close != -1 else buf
            )
            if found:
                lora_hash_entries.extend(found)
                skip.add(i)
            i = j + 1
            continue
        m = _LORA_HASH_ENTRY_RE.match(value)
        if m:
            lora_hash_entries.append((m.group(1), m.group(2).lower()))
            skip.add(i)
            j = i + 1
            while j < len(parts) and _LORA_HASH_ENTRY_RE.match(parts[j]):
                hm = _LORA_HASH_ENTRY_RE.match(parts[j])
                lora_hash_entries.append((hm.group(1), hm.group(2).lower()))
                skip.add(j)
                j += 1
            i = j
            continue
        i += 1
    if lora_hash_entries:
        out["lora_hashes"] = {k: v for k, v in lora_hash_entries}

    for idx, part in enumerate(parts):
        if idx in skip:
            continue
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        key, value = key.strip(), value.strip()
        if key == "Negative prompt":
            out["negative_prompt"] = value
        elif key == "Steps":
            out["steps"] = int(value)
        elif key == "Sampler":
            out["sampler"] = value
        elif key == "Scheduler":
            out["scheduler"] = value
        elif key == "CFG scale":
            out["cfg"] = float(value)
        elif key == "Seed":
            out["seed"] = int(value)
        elif key == "Denoising strength":
            out["denoise"] = float(value)
        elif key == "Size":
            m = re.match(r"^(\d+)\s*[x×]\s*(\d+)$", value)
            if m:
                out["width"], out["height"] = int(m.group(1)), int(m.group(2))
        elif key == "Model":
            out["model"] = value
        elif key == "Model hash":
            out["model_hash"] = value
        elif key == "Lora":
            out.setdefault("loras", []).append(value)
        elif key == "Version":
            out["version"] = value
        else:
            out[key] = value
    # imagagen extension: a trailing `Styles: [json]` line (kept off the
    # meta line on purpose: style prompts contain ", " and would split it)
    for line in lines:
        if line.startswith("Styles:"):
            try:
                styles = json.loads(line.split(":", 1)[1].strip())
                if isinstance(styles, list):
                    out["styles"] = styles
            except json.JSONDecodeError:
                pass
            break
    # imagagen extension: a trailing `Disabled LoRAs: name@w, ...` line,
    # same `name@weight` shape as the `Lora:` meta entries (kept as strings,
    # like out["loras"])
    for line in lines:
        if line.startswith("Disabled LoRAs:"):
            out["disabled_loras"] = [
                p.strip() for p in line.split(":", 1)[1].split(",") if p.strip()
            ]
            break
    return out


def attach_metadata(
    image: Image.Image, parameters_text: str
) -> tuple[Image.Image, PngImagePlugin.PngInfo]:
    """Return (image, pnginfo). The caller must pass `pnginfo=pnginfo` to
    `Image.save(...)` for the text chunk to be written.

    The JSON document is the single source of truth (Pillow writes it as an
    iTXt chunk when it contains non-ASCII text; :func:`read_metadata`
    handles both).
    """
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", parameters_text)
    return image, info


def read_metadata(png_bytes: bytes) -> dict[str, Any] | None:
    """Extract settings from a PNG's text chunks (Forge/A1111 or ours)."""
    from io import BytesIO

    img = Image.open(BytesIO(png_bytes))
    # Text chunks written AFTER the image data (as A1111/Forge do) are only
    # parsed into info/text when the image is decoded, so force a load first
    # or the metadata is silently missing (e.g. on cropped re-saves).
    try:
        img.load()
    except Exception:
        pass  # truncated/corrupt: use whatever was already parsed
    info = img.info or {}
    params = info.get("parameters")
    if isinstance(params, bytes):
        params = params.decode("utf-8", "replace")
    elif params is not None and not isinstance(params, str):
        params = str(params)  # e.g. an iTXt chunk
    if params:
        return parse_parameters(params)
    # fall back to standalone chunks
    if "prompt" in info:
        out: dict[str, Any] = {}
        for field, key in (
            ("prompt", "prompt"),
            ("Negative prompt", "negative_prompt"),
            ("Steps", "steps"),
            ("Sampler", "sampler"),
            ("Scheduler", "scheduler"),
            ("CFG scale", "cfg"),
            ("Seed", "seed"),
            ("Model", "model"),
            ("Model hash", "model_hash"),
        ):
            if field in info:
                v = info[field]
                if key in ("steps",):
                    v = int(v)
                elif key in ("cfg",):
                    v = float(v)
                elif key in ("seed",):
                    v = int(v)
                out[key] = v
        if "Size" in info:
            m = re.match(r"^(\d+)\s*[x×]\s*(\d+)$", str(info["Size"]))
            if m:
                out["width"], out["height"] = int(m.group(1)), int(m.group(2))
        return out
    return None
