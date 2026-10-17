#!/usr/bin/env python3
"""Reproduce the caption-preview OOM: SDXL resident on GPU1, then
registry.caption() with a 1024x1280 image. Logs VRAM at each stage."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from imagagen.config import load_config
from imagagen.models import ModelRegistry


def vram(tag):
    a = torch.cuda.memory_allocated(1) / 2**30
    r = torch.cuda.memory_reserved(1) / 2**30
    print(f"[vram] {tag}: allocated={a:.2f}GiB reserved={r:.2f}GiB", flush=True)

config = load_config()
reg = ModelRegistry(config, gpus=[1])
img = np_img = None
import numpy as np
from PIL import Image as PImage

im = PImage.open("dev/references/cute-kitten-dmd2.png").convert("RGB")
img = np.asarray(im)
from imagagen.backends import GenerateRequest  # noqa: E402

print("image", img.shape)
vram("baseline")
res, _ = reg.run(
    "epicrealismXL11.0", 1,
    GenerateRequest(prompt="a cat", width=640, height=640, seed=1,
                    model_id="epicrealismXL11.0", steps=4))
vram("after generation (SDXL resident)")
cap = reg.caption(img, 1)
vram("after caption")
print("caption:", cap[:80])
