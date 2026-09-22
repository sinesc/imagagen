"""E2E VRAM test: 3440x1440 txt2img (+LoRA) and inpaint through the real backend."""
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

W, H = 3440, 1440

reg = ModelRegistry(load_config())
model = reg.models[0]
backend = reg.backend_for(model.id)
dev = backend.device
free, total = torch.cuda.mem_get_info(dev)
print(f"device={dev} free={free/2**30:.2f} GiB total={total/2**30:.2f} GiB", flush=True)
torch.cuda.reset_peak_memory_stats(dev)

t0 = time.time()
res = backend.generate(GenerateRequest(
    prompt="a lighthouse on a cliff, dramatic sky, cinematic lighting",
    negative_prompt="", width=W, height=H, steps=20, cfg=5.5, seed=1234,
    sampler="DPM++ 2M", scheduler="Karras",
    loras=[("EnvyZoomSliderXL1.0", 0.8)]),
    progress=lambda f, s: None)
print(f"txt2img: {time.time()-t0:.1f} s", flush=True)
print(f"txt2img peak: {torch.cuda.max_memory_allocated(dev)/2**30:.2f} GiB", flush=True)
Image.fromarray(res.image).save("/tmp/e2e_3440.txt2img.png")

h, w = res.image.shape[:2]
# outpaint scenario: protect a central box, regenerate the rest
mask = np.full((h, w), 255, np.uint8)
mask[240:1200, 680:2760] = 0
torch.cuda.reset_peak_memory_stats(dev)
t0 = time.time()
res2 = backend.generate(GenerateRequest(
    prompt="a lighthouse on a cliff, dramatic sky, cinematic lighting",
    negative_prompt="", width=W, height=H, steps=20, cfg=5.5, seed=1234,
    sampler="DPM++ 2M", scheduler="Karras", denoise=1.0,
    loras=[("EnvyZoomSliderXL1.0", 0.8)],
    init_image=res.image, init_mask=mask),
    progress=lambda f, s: None)
print(f"inpaint: {time.time()-t0:.1f} s", flush=True)
print(f"inpaint peak: {torch.cuda.max_memory_allocated(dev)/2**30:.2f} GiB", flush=True)
Image.fromarray(res2.image).save("/tmp/e2e_3440.inpaint.png")
small = np.array(Image.fromarray(res2.image).resize((16, 16))).astype(float)
print(f"structure(16x16): {small.std():.1f}  std: {res2.image.astype(float).std():.1f}", flush=True)
print("done", flush=True)
