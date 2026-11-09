"""CLI tool: print an image file's generation metadata as imagagen-format JSON.

Reads the generation settings embedded in an image (our own JSON
`parameters` chunk, the legacy A1111/Forge text `parameters` chunk, JPEG
comments carrying the same, or standalone text chunks) and prints them as
the imagagen JSON document -- foreign (A1111/Forge) metadata is converted
to our format: LoRA tokens baked into the prompt text are recovered into
`loras` entries and, when a styles store is available, the applied styles
are peeled out of the expanded prompt into a `styles` list (exactly like
the UI's PNG import does).

Usage: python -m imagagen.pngmeta IMAGE [--styles PATH] [--no-styles]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from . import metadata
from .styles import StyleStore

# project root: src/imagagen/pngmeta.py -> parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# canonical key order of our format; unknown keys follow, keeping insertion
# order (lenient: Forge extras like "VAE" travel along verbatim)
_KEY_ORDER = [
    "format",
    "version",
    "prompt",
    "negative_prompt",
    "steps",
    "sampler",
    "scheduler",
    "cfg",
    "seed",
    "width",
    "height",
    "model",
    "model_hash",
    "model_config",
    "model_family",
    "denoise",
    "rotation",
    "loras",
    "disabled_loras",
    "styles",
]


def convert_to_imagagen(
    meta: dict[str, Any],
    width: int,
    height: int,
    style_store: StyleStore | None = None,
) -> dict[str, Any]:
    """Normalize a parsed metadata dict to the imagagen JSON format.

    Our own images are already in the format (the parsed JSON document is
    returned verbatim). A1111/Forge dicts are converted:

    * ``<lora:NAME:WEIGHT>`` tokens are extracted from the prompt into
      ``loras`` ``name@weight`` entries (Forge bakes them into the prompt),
    * applied styles are peeled out of the (expanded) prompt into the
      ``styles`` list via the style store, when one is given,
    * ``width``/``height`` are filled from the image when absent,
    * the ``format`` key marks the result as our format; the recorded
      ``version`` (the generating program's) is kept as documentation.
    """
    doc = dict(meta)
    if doc.get("format") == metadata._FORMAT:
        return doc
    prompt = doc.get("prompt", "")
    negative = doc.get("negative_prompt", "")
    loras, base_prompt = metadata.extract_prompt_loras(prompt)
    if loras:
        doc.setdefault("loras", []).extend(f"{n}@{w:g}" for n, w in loras)
    doc["prompt"] = base_prompt
    if style_store is not None:
        found, base_prompt, base_negative = style_store.extract_styles(
            base_prompt, negative
        )
        doc["prompt"] = base_prompt
        if found:
            doc["styles"] = [s.to_dict() for s in found]
            if "negative_prompt" in doc:
                doc["negative_prompt"] = base_negative
    doc.setdefault("width", width)
    doc.setdefault("height", height)
    doc["format"] = metadata._FORMAT
    # canonical key order first, remaining (Forge) keys verbatim after
    ordered = {k: doc[k] for k in _KEY_ORDER if k in doc}
    ordered.update(doc)
    return ordered


def read_image_metadata(path: Path) -> tuple[dict[str, Any] | None, int, int]:
    """(meta, width, height) from an image file; meta is None if it has none.

    PNG/WebP text chunks go through :func:`imagagen.metadata.read_metadata`;
    JPEG (where A1111/Forge store the parameters text in the JPEG comment)
    gets an extra fallback.
    """
    data = path.read_bytes()
    meta = metadata.read_metadata(data)
    img = Image.open(path)
    width, height = img.size
    if meta is not None:
        return meta, width, height
    comment = (img.info or {}).get("comment")
    if comment:
        text = comment.decode("utf-8", "replace") if isinstance(comment, bytes) else str(comment)
        # A1111/Forge write JPEG comments as "parameters: <text>"
        if text.startswith("parameters:"):
            text = text.split(":", 1)[1].strip()
        return metadata.parse_parameters(text), width, height
    return None, width, height


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pngmeta",
        description="Print an image's generation metadata as imagagen-format JSON.",
    )
    parser.add_argument("image", help="image file (PNG/JPEG/WebP/...)")
    parser.add_argument(
        "--styles",
        type=Path,
        default=_PROJECT_ROOT / "user" / "styles.csv",
        help="styles CSV used to peel applied styles out of expanded "
        "A1111/Forge prompts (default: this install's user/styles.csv)",
    )
    parser.add_argument(
        "--no-styles",
        action="store_true",
        help="do not attempt style extraction (keep the expanded prompt verbatim)",
    )
    args = parser.parse_args(argv)

    path = Path(args.image)
    if not path.is_file():
        print(f"pngmeta: no such file: {path}", file=sys.stderr)
        return 2
    try:
        meta, width, height = read_image_metadata(path)
    except Exception as exc:
        print(f"pngmeta: could not read image: {exc}", file=sys.stderr)
        return 2
    if meta is None:
        print(f"pngmeta: no generation metadata found in {path}", file=sys.stderr)
        return 1

    style_store: StyleStore | None = None
    if not args.no_styles and args.styles.is_file():
        style_store = StyleStore(args.styles)

    doc = convert_to_imagagen(meta, width, height, style_store)
    print(json.dumps(doc, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
