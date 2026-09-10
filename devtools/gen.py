"""Generate an image directly through the backend (no server, no UI).

Prints timing, image statistics and a structural check, and saves the result.
The structure check (std of a 16x16 downscale) distinguishes real content
from all-noise output: pure noise has structure ~ std, a soft real image has
structure << std.

Usage (from the project root):
    .venv/bin/python devtools/gen.py "a red cube on a wooden table" \
        --width 1024 --height 1024 --steps 20 --cfg 5.5 --seed 7 \
        --sampler "DPM++ 2M" --scheduler Karras --model models/sdxl/sd_xl_base_1.0.safetensors \
        --out /tmp/cube.png
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from PIL import Image

from imagagen.backends import GenerateRequest
from imagagen.config import load_config
from imagagen.models import ModelRegistry


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("prompt")
    p.add_argument("--negative", default="")
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--cfg", type=float, default=5.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sampler", default=None)
    p.add_argument("--scheduler", default=None)
    p.add_argument("--model", default=None,
                   help="model id as listed by /api/models (default: current)")
    p.add_argument("--out", default="/tmp/gen.png")
    args = p.parse_args()

    reg = ModelRegistry(load_config())
    model = reg.models[0] if args.model is None else next(
        m for m in reg.models if m.id == args.model or m.path.name == args.model)
    backend = reg.backend_for(model.id)

    free, _ = torch.cuda.mem_get_info(0)
    t0 = time.time()
    res = backend.generate(GenerateRequest(
        prompt=args.prompt, negative_prompt=args.negative,
        width=args.width, height=args.height, steps=args.steps, cfg=args.cfg,
        seed=args.seed, sampler=args.sampler, scheduler=args.scheduler),
        progress=lambda f, s: None)
    dt = time.time() - t0
    free2, _ = torch.cuda.mem_get_info(0)

    a = res.image.astype(float)
    small = np.array(Image.fromarray(res.image).resize((16, 16))).astype(float)
    Image.fromarray(res.image).save(args.out)
    print(f"model:   {model.id}")
    print(f"time:    {dt:.1f}s   VRAM free: {free/2**30:.2f} -> {free2/2**30:.2f} GiB")
    print(f"std:     {a.std():.1f}   structure(16x16): {small.std():.1f}   "
          f"(noise-only output would have structure ~ std)")
    print(f"saved:   {args.out}   (seed {res.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
