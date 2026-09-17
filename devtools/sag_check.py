"""Self attention guidance (SAG) end-to-end check against the real SDXL
backend (no server, no UI).

Verifies the invariants the unit tests can't (they can't load a 6.9 GB
UNet):
 1. SAG on: a face portrait generates without error, in more time than
    the base (the extra unconditional UNet pass) and at a sane peak VRAM
    (the mask accumulation is chunked -- no full (heads, n, n) matrix).
 2. SAG with scale 0.0 is bit-identical to SAG off (the correction term
    is multiplied by zero; the extra pass and the attention capture must
    not perturb the base generation).
 3. SAG requested with cfg <= 1 is skipped (no unconditional pass exists,
    Forge prints and continues) -- the result equals the cfg-1 base.

Saves all images next to this script's run dir for a visual compare.

Usage (from the project root):
    .venv/bin/python devtools/sag_check.py --out /tmp/sag
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from imagagen.backends import GenerateRequest
from imagagen.config import load_config
from imagagen.models import ModelRegistry
from imagagen.sag import SelfAttentionGuidance


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=5.5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--model", default=None)
    p.add_argument("--out", default="/tmp/sag")
    args = p.parse_args()

    reg = ModelRegistry(load_config())
    model = reg.models[0] if args.model is None else next(
        m for m in reg.models if m.id == args.model or m.path.name == args.model)
    backend = reg.backend_for(model.id)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    common = dict(
        prompt="portrait of a woman, detailed eyes, soft light",
        negative_prompt="ugly, blurry",
        width=args.width, height=args.height, steps=args.steps,
        cfg=args.cfg, seed=args.seed,
    )

    device = torch.device(backend.device)

    def peak() -> float:
        return torch.cuda.max_memory_allocated(device) / 2**30

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    base = backend.generate(GenerateRequest(**common),
                            progress=lambda f, s: None)
    t_base = time.time() - t0
    peak_base = peak()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    sag = backend.generate(GenerateRequest(
        **common, sag=SelfAttentionGuidance(scale=0.5)),
        progress=lambda f, s: None)
    t_sag = time.time() - t0
    peak_sag = peak()

    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    sag_zero = backend.generate(GenerateRequest(
        **common, sag=SelfAttentionGuidance(scale=0.0)),
        progress=lambda f, s: None)
    t_zero = time.time() - t0
    peak_zero = peak()

    # cfg <= 1: SAG is skipped (no uncond pass); the result must equal a
    # plain cfg-1 run
    cfg1 = backend.generate(GenerateRequest(**{**common, "cfg": 1.0}),
                            progress=lambda f, s: None)
    cfg1_sag = backend.generate(GenerateRequest(
        **{**common, "cfg": 1.0}, sag=SelfAttentionGuidance(scale=0.5)),
        progress=lambda f, s: None)

    Image.fromarray(base.image).save(out / "base.png")
    Image.fromarray(sag.image).save(out / "sag.png")
    Image.fromarray(sag_zero.image).save(out / "sag_scale0.png")

    diff = np.abs(sag.image.astype(np.int16) - base.image.astype(np.int16))
    print(f"backend device: {backend.device}")
    print(f"base : {t_base:6.1f}s  peak {peak_base:5.2f} GiB")
    print(f"sag  : {t_sag:6.1f}s  peak {peak_sag:5.2f} GiB  (ratio {t_sag / t_base:.2f}x)")
    print(f"sag0 : {t_zero:6.1f}s  peak {peak_zero:5.2f} GiB")
    print(f"pixel diff base vs sag: mean {diff.mean():.2f} max {diff.max()} "
          f"({(diff.max(2) > 8).mean() * 100:.1f}% of pixels differ >8)")
    ok = True
    if not np.array_equal(sag_zero.image, base.image):
        d = np.abs(sag_zero.image.astype(np.int16) - base.image.astype(np.int16))
        print(f"FAIL: sag scale 0 is NOT bit-identical to base (max {d.max()})")
        ok = False
    else:
        print("OK: sag scale 0 bit-identical to base")
    if not np.array_equal(cfg1_sag.image, cfg1.image):
        d = np.abs(cfg1_sag.image.astype(np.int16) - cfg1.image.astype(np.int16))
        print(f"FAIL: cfg1+sag is NOT bit-identical to cfg1 (max {d.max()})")
        ok = False
    else:
        print("OK: cfg<=1 skips SAG (bit-identical to plain cfg1)")
    if not (t_sag > t_base * 1.1):
        print(f"WARN: sag not slower than base ({t_sag:.1f}s vs {t_base:.1f}s)?")
    if peak_sag > peak_base + 1.5:
        print(f"WARN: sag peak VRAM much higher than base "
              f"({peak_sag:.2f} vs {peak_base:.2f} GiB) -- chunking broken?")
    print("images in", out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
