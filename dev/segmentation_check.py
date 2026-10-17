"""Person mask generation smoke test (real MediaPipe model, CPU).

Runs the segmentation service against models/segmentation/*.tflite on a
reference image (or a CLI image) and writes:
  - dev/references/segmentation-preview-<model-stem>.png  (mask over image)
  - dev/references/segmentation-mask-<model-stem>.png     (raw alpha)

Usage:
  .venv/bin/python dev/segmentation_check.py [image.png] [--threshold 0.25] [--feather 6]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from imagagen.segmentation import SegmentationService  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?",
                    default=str(REPO / "dev/references/woman-with-dog-mask-generator-testing.png"))
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--feather", type=float, default=0.0)
    args = ap.parse_args()

    image_path = Path(args.image)
    img = Image.open(image_path).convert("RGB")
    rgb = np.asarray(img)
    print(f"image: {image_path.name} {img.size[0]}x{img.size[1]} "
          f"threshold={args.threshold} feather={args.feather}")

    svc = SegmentationService(REPO / "models" / "segmentation")
    models = svc.list_models()
    if not models:
        print("no models in models/segmentation/ (put a .tflite there)")
        return 1

    out_dir = REPO / "dev" / "references"
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for m in models:
        # all person classes except background
        classes = [label for label in m.labels if label != "background"]
        t0 = time.time()
        mask = svc.generate(m.id, classes, rgb,
                            threshold=args.threshold, feather=args.feather)
        dt = time.time() - t0
        px = int((mask > 0).sum())
        soft = int(((mask > 0) & (mask < 255)).sum())
        print(f"{m.id}: classes={classes}  masked={px} px "
              f"({100 * px / mask.size:.1f}%, {soft} feathered)  in {dt:.2f}s")
        for label in m.labels:
            single = svc.generate(m.id, [label], rgb, threshold=args.threshold,
                                  feather=args.feather)
            print(f"    {label:>12}: {int((single > 0).sum())} px")

        overlay = img.copy()
        red = Image.new("RGB", img.size, (255, 0, 0))
        overlay.paste(red, mask=Image.fromarray(mask, mode="L"))
        overlay.save(out_dir / f"segmentation-preview-{m.name}.png")
        Image.fromarray(mask, mode="L").save(out_dir / f"segmentation-mask-{m.name}.png")
        print(f"    wrote segmentation-preview-{m.name}.png "
              f"(+ segmentation-mask-{m.name}.png)")
        ok &= px > 0
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
