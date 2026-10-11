#!/usr/bin/env python3
"""Phase 1 reference-captioning smoke test.

Runs the universal VLM caption tier (imagagen.reference) on sample images
and checks:
  * the captions are sensible (visual check by the agent),
  * captioning is bit-stable: a fresh (cache-cleared) run reproduces the
    exact same strings,
  * the sha1 cache round-trips (second call is instant and identical).

Usage:
    .venv/bin/python dev/smoke_reference_caption.py [--gpu 1] [img ...]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imagagen.config import load_config  # noqa: E402
from imagagen.reference import (  # noqa: E402
    caption_reference,
    reference_cache_path,
)

DEFAULT_IMAGES = [
    Path("dev/references/purple-flower.png"),
    Path("dev/references/cute-kitten-dmd2.png"),
    Path("dev/references/lighthouse-keeper.png"),
]

#: regression fixtures: the exact captions the pinned sample images
#: produce (sha1 of the caption text). Greedy decoding is deterministic;
#: a mismatch means the caption path changed (vision tower, template,
#: stream assembly, or the text tower) -- update deliberately.
PINNED_CAPTION_SHA1 = {
    "dev/references/purple-flower.png":
        "95885ce26eafe4831171374a024dd90d0f805902",
    "dev/references/cute-kitten-dmd2.png":
        "424ce1e56c7a66a7ba4794ec0ab1c3c0d049a208",
    "dev/references/lighthouse-keeper.png":
        "f252b7ecf05f5e0b974b76c2340b8f96ae0336b1",
}


def load_image(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--gpu", type=int, default=1)
    args = ap.parse_args()
    config = load_config()
    device = f"cuda:{args.gpu}"
    images = args.images or DEFAULT_IMAGES

    cache = reference_cache_path(config)
    results: dict[str, str] = {}
    for p in images:
        arr = load_image(p)
        # fresh run: drop the cache entry so the VLM really runs
        cache.unlink(missing_ok=True)
        t0 = time.time()
        cap1 = caption_reference(arr, device, config)
        t1 = time.time()
        # cache hit: instant and identical
        t2 = time.time()
        cap2 = caption_reference(arr, device, config)
        t3 = time.time()
        # determinism: a second fresh run must reproduce the exact string
        cache.unlink(missing_ok=True)
        cap3 = caption_reference(arr, device, config)
        stable = cap1 == cap2 == cap3
        results[str(p)] = cap1
        print(f"=== {p} ({arr.shape[1]}x{arr.shape[0]})")
        print(f"    caption ({t1 - t0:.1f}s fresh, {t3 - t2:.4f}s cached):")
        print(f"    {cap1!r}")
        print(f"    bit-stable across runs: {stable}")
        if not stable:
            print(f"    cap2={cap2!r} cap3={cap3!r}")
            return 1
        pinned = PINNED_CAPTION_SHA1.get(str(p))
        if pinned is not None:
            got = hashlib.sha1(cap1.encode()).hexdigest()
            if got != pinned:
                print(f"    FIXTURE MISMATCH: sha1={got} != {pinned}")
                return 1
            print(f"    pinned fixture matches (sha1 {got[:12]})")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
