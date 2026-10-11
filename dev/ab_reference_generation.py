#!/usr/bin/env python3
"""Phase 1 A/B check: generation with vs. without a reference image.

For each tested family (SDXL, Krea 2):
  * baseline generation (no reference),
  * the same seed + prompt WITH a reference image (the VLM caption is
    appended to the positive prompt),
  * the caption preview path (ModelRegistry.caption, i.e. the
    /api/reference/caption endpoint route) is exercised first.

Outputs land in dev/scratch/ab_reference/ for visual (agent-vision)
comparison; the baseline must show NO reference metadata line, the
reference run must carry the caption.

Usage: .venv/bin/python dev/ab_reference_generation.py [--gpu 1]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.models import ModelRegistry  # noqa: E402

OUT = ROOT / "dev" / "scratch" / "ab_reference"
REF_IMAGE = ROOT / "dev" / "references" / "purple-flower.png"

MODELS = [
    ("epicrealismXL11.0", "a lighthouse by the sea"),
    ("krea2_Int8Extended", "a lighthouse by the sea"),
]


def save(arr: np.ndarray, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(OUT / f"{name}.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    config = load_config()
    gpu = args.gpu
    registry = ModelRegistry(config, gpus=[gpu])

    ref = np.asarray(Image.open(REF_IMAGE).convert("RGB"))

    # 1) caption preview path (the /api/reference/caption route): runs on
    # the registry with no model loaded
    t0 = time.time()
    preview = registry.caption(ref, gpu)
    print(f"preview caption ({time.time() - t0:.1f}s): {preview[:120]}...")

    for model_id, prompt in MODELS:
        if registry.get(model_id) is None:
            print(f"skip {model_id} (not present)")
            continue
        base = GenerateRequest(prompt=prompt, width=768, height=768,
                               seed=42, model_id=model_id)
        t0 = time.time()
        res0, _ = registry.run(model_id, gpu, base)
        save(res0.image, f"{model_id.split('/')[0]}_baseline")
        print(f"{model_id}: baseline {time.time() - t0:.1f}s, "
              f"caption={res0.reference_caption}")
        t0 = time.time()
        res1, _ = registry.run(
            model_id, gpu, base.model_copy(update={"reference_image": ref}))
        save(res1.image, f"{model_id.split('/')[0]}_reference")
        print(f"{model_id}: with reference {time.time() - t0:.1f}s, "
              f"caption={res1.reference_caption!r}")
        assert res1.reference_caption, "reference run must report a caption"
        assert res0.reference_caption is None
    print(f"outputs in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
