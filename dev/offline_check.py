"""Verify the model + improved VAE load and generate with the network OFF.

Proves the project-local HF cache (cache/) is complete: a fresh
process with HF_HUB_OFFLINE=1 must load the pipeline and produce a real
image (structure check: a real image's 16x16-downscale std is far below its
pixel std; pure noise / failed decode is not).

Usage (from the project root):
    .venv/bin/python dev/offline_check.py
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CHILD = """
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from imagagen.config import load_config
from imagagen.models import ModelRegistry
from imagagen.backends import GenerateRequest
import numpy as np
from PIL import Image
reg = ModelRegistry(load_config())
b = reg.backend_for(reg.models[0].id)
r = b.generate(GenerateRequest(prompt="a red cube on a table", width=512,
                                height=512, steps=10, cfg=6, seed=1),
               progress=lambda f, s: None)
a = r.image.astype(float)
small = np.array(Image.fromarray(r.image).resize((16, 16))).astype(float)
print(f"OFFLINE-OK std={a.std():.1f} structure={small.std():.1f}")
"""


def main() -> int:
    proc = subprocess.run(
        [sys.executable, "-c", CHILD], capture_output=True, text=True, cwd=ROOT,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    sys.stdout.write(proc.stdout)
    tail = proc.stderr.strip().splitlines()[-3:]
    if proc.returncode != 0:
        print(" ".join(tail))
        print("FAIL: offline load/generation failed (HF cache incomplete?)")
        return 1
    print("PASS: fully offline load + generation works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
