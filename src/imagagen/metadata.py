"""PNG metadata in the A1111/Forge "parameters" text-chunk format.

Generated images carry a `parameters` chunk (plus the individual standard
fields Forge also reads) so Forge can import them and so "Import" in imagagen
can round-trip settings.

imagagen extension: styles travel on a dedicated `Styles: [json]` line
AFTER the comma-separated meta line (each entry carries the style's name
plus its actual prompt / negative-prompt contents). The `parameters` text
then records the user's raw prompt, not the style-applied one -- re-import
re-applies the styles exactly like a new generation would. A1111/Forge
ignore the extra trailing line.
"""

from __future__ import annotations

import json
import re
from typing import Any

from PIL import Image, PngImagePlugin

from .soft_inpaint import SoftInpainting

_VERSION = "v0.1.0"

# fields that travel both inside "parameters" and as standalone chunks
_FIELDS = (
    "Negative prompt",
    "Steps",
    "Sampler",
    "Scheduler",
    "CFG scale",
    "Seed",
    "Size",
    "Model",
    "Model hash",
)


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
    loras: list[tuple[str, float]] | None = None,
    styles: list[dict] | None = None,
    soft_inpainting: SoftInpainting | None = None,
) -> str:
    """Format the `parameters` text chunk exactly like A1111/Forge does.

    ``styles`` (imagagen extension): entries with "name", "prompt" and
    "negative_prompt"; serialized as JSON on a trailing `Styles:` line.

    ``soft_inpainting``: when set, the enabled flag and the six option
    values travel on the meta line under Forge's exact extra-parameter
    labels, so Forge (or this file's parser) can read them back.
    """
    lines = [prompt]
    if negative_prompt:
        lines.append(f"Negative prompt: {negative_prompt}")
    tail = [
        f"Steps: {steps}",
        f"Sampler: {sampler}",
        f"Scheduler: {scheduler}",
        f"CFG scale: {cfg:g}",
        f"Seed: {seed}",
        f"Size: {width}x{height}",
        f"Model: {model_name}",
        f"Model hash: {model_hash}",
    ]
    for name, weight in loras or []:
        tail.append(f"Lora: {name}@{weight:g}")
    if soft_inpainting is not None:
        tail.append("Soft inpainting enabled: True")
        tail.append(
            f"Soft inpainting schedule bias: {soft_inpainting.mask_blend_power:g}"
        )
        tail.append(
            "Soft inpainting preservation strength: "
            f"{soft_inpainting.mask_blend_scale:g}"
        )
        tail.append(
            "Soft inpainting transition contrast boost: "
            f"{soft_inpainting.inpaint_detail_preservation:g}"
        )
        tail.append(
            "Soft inpainting mask influence: "
            f"{soft_inpainting.composite_mask_influence:g}"
        )
        tail.append(
            "Soft inpainting difference threshold: "
            f"{soft_inpainting.composite_difference_threshold:g}"
        )
        tail.append(
            "Soft inpainting difference contrast: "
            f"{soft_inpainting.composite_difference_contrast:g}"
        )
    tail.append(f"Version: {_VERSION}")
    lines.append(", ".join(tail))
    if styles:
        entries = [
            {
                "name": s.get("name", ""),
                "prompt": s.get("prompt", ""),
                "negative_prompt": s.get("negative_prompt", ""),
            }
            for s in styles
        ]
        # no newlines in the JSON (prompts keep theirs as \n escapes)
        lines.append("Styles: " + json.dumps(entries, ensure_ascii=False))
    return "\n".join(lines)


def parse_parameters(text: str) -> dict[str, Any]:
    """Parse an A1111/Forge `parameters` chunk back into settings.

    Lenient: unknown keys are kept verbatim, missing ones are omitted.
    """
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

    for part in meta_line.split(", "):
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
    return out


def attach_metadata(
    image: Image.Image, parameters_text: str
) -> tuple[Image.Image, PngImagePlugin.PngInfo]:
    """Return (image, pnginfo). The caller must pass `pnginfo=pnginfo` to
    `Image.save(...)` for the text chunks to be written."""
    info = PngImagePlugin.PngInfo()
    info.add_text("parameters", parameters_text)
    parsed = parse_parameters(parameters_text)
    for field in _FIELDS:
        key = {
            "Negative prompt": "negative_prompt",
            "Steps": "steps",
            "Sampler": "sampler",
            "Scheduler": "scheduler",
            "CFG scale": "cfg",
            "Seed": "seed",
            "Size": "width",
        }.get(field)
        if key is None or key not in parsed:
            continue
        if field == "Size":
            info.add_text("Size", f"{parsed['width']}x{parsed['height']}")
        else:
            info.add_text(field, str(parsed[key]))
    info.add_text("Model", parsed.get("model", ""))
    info.add_text("Model hash", parsed.get("model_hash", ""))
    return image, info


def read_metadata(png_bytes: bytes) -> dict[str, Any] | None:
    """Extract settings from a PNG's text chunks (Forge/A1111 or ours)."""
    from io import BytesIO

    img = Image.open(BytesIO(png_bytes))
    info = img.info or {}
    params = info.get("parameters")
    if isinstance(params, bytes):
        params = params.decode("utf-8", "replace")
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
