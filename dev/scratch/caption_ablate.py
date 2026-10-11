#!/usr/bin/env python3
"""Caption template ablation: does the missing empty think block explain
the 'distorted image' hallucination? Captions one clean image with the
current CAPTION_PARTS and with the reference's trailing empty think
block appended (built via chr() so no marker text appears anywhere)."""
import sys
sys.path.insert(0, "src")
import numpy as np
import torch
from PIL import Image

from imagagen.config import load_config
import imagagen.reference as R
from imagagen.encoders.caption_prompt import CAPTION_PARTS

THINK = chr(60) + "think" + chr(62) + "\n\n" + chr(60) + "/think" + chr(62) + "\n\n"

img = np.asarray(Image.open("dev/references/purple-flower.png").convert("RGB"))
cfg = load_config()

def run(parts, tag):
    orig = list(R.CAPTION_PARTS)
    R.CAPTION_PARTS[:] = parts
    # bypass the sha1 cache
    cache_file = R.reference_cache_path(cfg)
    import os
    if os.path.exists(cache_file):
        os.rename(cache_file, str(cache_file) + ".bak")
    try:
        cap = R.caption_reference(img, "cuda:1", cfg)
    finally:
        R.CAPTION_PARTS[:] = orig
        if os.path.exists(str(cache_file) + ".bak"):
            os.rename(str(cache_file) + ".bak", cache_file)
    print(f"=== {tag}\n{cap}\n", flush=True)

run(list(CAPTION_PARTS), "current template")
run(list(CAPTION_PARTS) + [("text", THINK)], "with empty think block")
