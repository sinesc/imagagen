"""Person mask generation smoke test (real MediaPipe model, CPU).

Runs the segmentation service against models/segmentation/*.tflite on a
reference image (or a CLI image) and writes:
  - dev/references/segmentation-preview-<model-stem>.png  (mask over image)
  - dev/references/segmentation-mask-<model-stem>.png     (raw alpha)

Usage:
  .venv/bin/python dev/segmentation_check.py [image.png]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from imagagen.segmentation import SegmentationService  # noqa: E402


def main() -> int:
    image_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else REPO / "dev/references/woman-with-dog-mask-generator-testing.png"
    )
    img = Image.open(image_path).convert("RGB")
    rgb = np.asarray(img)
    print(f"image: {image_path.name} {img.size[0]}x{img.size[1]}")

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
        mask = svc.generate(m.id, classes, rgb)
        dt = time.time() - t0
        px = int((mask > 0).sum())
        print(f"{m.id}: classes={classes}  masked={px} px "
              f"({100 * px / mask.size:.1f}%)  in {dt:.2f}s")
        for label in m.labels:
            single = svc.generate(m.id, [label], rgb)
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
