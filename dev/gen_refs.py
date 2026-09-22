"""Regenerate the Forge reference cases with the local backend for comparison.

Usage: uv run python dev/gen_refs.py [dmd2|zoom3|all]
Writes PNGs to dev/references/imagagen/ (same prompt/params as Forge refs).
"""
import sys
from pathlib import Path

sys.path.insert(0, "src")
import torch
from imagagen.config import load_config
from imagagen.models import ModelRegistry
from imagagen.backends import GenerateRequest

CASES = {
    "dmd2": dict(
        prompt="professional photo of a cute kitten playing with a woolen ball",
        width=1024, height=1024, steps=12, cfg=1.5, seed=1666281685,
        sampler="Euler a", scheduler="Polyexponential",
        loras=[("DMD2fp32XL1.0", 0.7)],
    ),
    "zoom3": dict(
        prompt="a cute white bunny in a field of green grass, sniffing a yellow flower",
        width=1024, height=1024, steps=20, cfg=6, seed=4,
        sampler="DPM++ 2M", scheduler="Karras",
        loras=[("EnvyZoomSliderXL1.0", -3.0)],
    ),
}

def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = load_config()
    reg = ModelRegistry(cfg, gpus=[1])  # leave GPU 0 to the user
    backend = reg.backend_for([m for m in reg.models if m.name == "epicrealismXL11.0"][0].id)
    out_dir = Path("dev/references/imagagen")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, c in CASES.items():
        if which not in ("all", name):
            continue
        req = GenerateRequest(
            prompt=c["prompt"], negative_prompt="",
            width=c["width"], height=c["height"], steps=c["steps"],
            cfg=c["cfg"], seed=c["seed"], sampler=c["sampler"],
            scheduler=c["scheduler"], loras=c["loras"],
            model_id=backend.model_id, model_name=backend.name,
        )
        res = backend.generate(req)
        import numpy as np
        from PIL import Image
        img = np.ascontiguousarray(res.image)
        Image.fromarray(img).save(out_dir / (name + ".png"))
        print("saved", out_dir / (name + ".png"))

main()
