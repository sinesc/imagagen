"""Regression check: fixed seeds must produce byte-identical images.

Catches regressions in the seeded-noise path (initial noise and the
per-step ancestral/SDE streams, which must both be derived from the seed).

Usage (from the project root):
    .venv/bin/python dev/seed_check.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from imagagen.backends import GenerateRequest
from imagagen.config import load_config
from imagagen.models import ModelRegistry

PROMPT = "a blue sphere on a green floor"


def main() -> int:
    reg = ModelRegistry(load_config())
    backend = reg.backend_for(reg.models[0].id)

    def gen(seed: int, sampler: str | None = None) -> np.ndarray:
        return backend.generate(GenerateRequest(
            prompt=PROMPT, width=512, height=512, steps=15, cfg=6.0,
            seed=seed, sampler=sampler), progress=lambda f, s: None).image

    ok = True
    for sampler in (None, "DPM++ 2M SDE"):  # ODE + ancestral/SDE paths
        label = sampler or "default (Euler a)"
        out = [gen(1, sampler) for _ in range(3)]
        same = np.array_equal(out[0], out[1]) and np.array_equal(out[0], out[2])
        ok &= same
        print(f"{label:20s} seed=1 x3 identical: {same}")

    diff = gen(1) != gen(2)
    ok &= bool(diff.any())
    print(f"{'different seeds':20s} differ:         {bool(diff.any())}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
