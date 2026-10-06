"""Verify the decode-park decision across model classes:
- NF4 (6.1 GiB transformer, co-resident encoders): decode should fit, no park
- FP8 1408x1408 (untiled, ~5.4 GiB needed): park expected
- FP8 img2img 1024x1280: decode should park (init encode unaffected)
"""

import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.backends.flux import FluxBackend  # noqa: E402
from imagagen.models import ModelInfo  # noqa: E402

config = load_config(ROOT)


def make(model_file: str) -> FluxBackend:
    path = config.root / "models" / "flux" / model_file
    return FluxBackend(
        ModelInfo(id=model_file.rsplit(".", 1)[0], name=model_file.rsplit(".", 1)[0],
                  path=path, family="flux"),
        torch.device("cuda:0"), config)


def run(tag: str, backend: FluxBackend, req: GenerateRequest) -> None:
    t0 = time.time()
    backend.generate(req)
    print(f"[{tag}] done in {time.time() - t0:.1f}s", flush=True)


nf4 = make("flux1SchnellNF4.safetensors")
run("NF4 1024x1280", nf4, GenerateRequest(
    prompt="a lighthouse in a storm", width=1024, height=1280,
    steps=-1, cfg=-1, seed=7))
nf4.unload()
torch.cuda.empty_cache()

fp8 = make("flux1SchnellFP811GBKijai.safetensors")
run("FP8 1408x1408", fp8, GenerateRequest(
    prompt="a red fox sitting on a bench in the rain, cinematic",
    width=1408, height=1408, steps=-1, cfg=-1, seed=8))
img = np.full((1024, 1280, 3), 128, np.uint8)
run("FP8 img2img 1024x1280", fp8, GenerateRequest(
    prompt="a red fox sitting on a bench in the rain, cinematic",
    width=1024, height=1280, steps=-1, cfg=-1, seed=9,
    init_image=img, denoise=0.5))
print("ALL DONE", flush=True)
