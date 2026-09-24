"""Live Flux LoRA smoke test (GPU).

Mirrors dev/smoke_sdxl_lora.py for the flux model family, using the real
LoRA in ``loras/flux/`` (standard ComfyUI/SimpleTrainer Flux LoRA, rank
16, no alpha tensors):

  1. baseline (no LoRA)
  2. MetalSuit @ 1.0   -> must differ from baseline (deltas are active),
                          output must be valid (no NaN, not uniform)
  3. MetalSuit @ 1.0 again -> identical to (2) (deterministic seed; the
                             unchanged set is served from the parse cache
                             and the deltas are already installed)
  4. MetalSuit @ 0.0   -> identical to baseline (zero-weight LoRA installs
                          no deltas)
  5. no LoRA again     -> deltas cleared, identical to baseline

Also reports the per-module delta state (how many GgufLinears carry
deltas, how many extra GEMM pairs that is, and the delta parameter
footprint) so the runtime-delta cost is visible.

Usage (from the project root; needs GPU + models/flux/*.gguf):
    .venv/bin/python dev/smoke_flux_lora.py [gpu-index]
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.gguf_ops import GgufLinear  # noqa: E402
from imagagen.models import ModelRegistry, list_loras  # noqa: E402

LORA = "MetalSuitFlux1.0"


def delta_report(backend):
    """How much LoRA state the transformer currently carries."""
    mods = [m for m in backend.transformer.modules()
            if isinstance(m, GgufLinear) and m.lora_deltas]
    n_pairs = sum(len(m.lora_deltas) for m in mods)
    n_params = sum(p.numel() for m in mods
                  for name, p in m.named_parameters(
                      recurse=False, remove_duplicate=True)
                  if name.startswith("lora_"))
    print(f"  delta state: {len(mods)} modules, {n_pairs} GEMM pairs, "
          f"{n_params * 2 / 2**20:.1f} MB (bf16)", flush=True)
    return n_pairs


def main(gpu: int | None) -> None:
    cfg = load_config(ROOT)
    found = list_loras(cfg, family="flux")
    print("flux loras:", found, flush=True)
    assert LORA in found, f"expected {LORA!r} in loras/flux/, got {found!r}"

    reg = ModelRegistry(cfg, gpus=[gpu])
    info = next(m for m in reg.models if m.family == "flux")
    print("loading", info.name, "...", flush=True)
    t0 = time.time()
    backend = reg.backend_for(info.id)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)
    torch.cuda.reset_peak_memory_stats()

    def gen(loras):
        req = GenerateRequest(prompt="a metallic armor suit, studio light",
                              width=512, height=512, steps=4, seed=42,
                              loras=loras)
        t0 = time.time()
        img = backend.generate(req).image
        dt = time.time() - t0
        print(f"  {loras or '(no lora)'}  {dt:.1f}s", flush=True)
        return img, dt

    def diff(a, b) -> float:
        return float(np.abs(a.astype(int) - b.astype(int)).mean())

    print("\n[1] baseline 512x512 x4 (no runtime deltas)", flush=True)
    base, base_dt = gen([])
    assert delta_report(backend) == 0
    assert not np.isnan(base.astype(float)).any(), "NaN in baseline"

    print(f"\n[2] {LORA} @ 1.0 (expect clearly different)", flush=True)
    a1, lora_dt = gen([(LORA, 1.0)])
    pairs = delta_report(backend)
    assert pairs > 0, "no runtime deltas installed"
    d = diff(a1, base)
    print(f"  mean abs diff vs baseline: {d:.2f} "
          f"(runtime delta {lora_dt / base_dt:.2f}x baseline time)",
          flush=True)
    assert d > 1, f"LoRA had no visible effect (diff={d:.2f})"
    assert not np.isnan(a1.astype(float)).any(), "NaN with LoRA"
    assert a1.std() > 10, f"output looks uniform (std={a1.std():.1f})"
    peak = torch.cuda.max_memory_allocated(backend.device) / 2**30
    print(f"  peak VRAM: {peak:.2f} GiB", flush=True)

    print(f"\n[3] {LORA} @ 1.0 again (cache hit, expect identical to [2])",
          flush=True)
    a2, _ = gen([(LORA, 1.0)])
    d = diff(a2, a1)
    print(f"  mean abs diff vs [2]: {d:.2f}", flush=True)
    assert d == 0, f"same seed + same LoRAs should be deterministic (diff={d:.2f})"

    print(f"\n[4] {LORA} @ 0.0 (expect identical to baseline)", flush=True)
    a0, _ = gen([(LORA, 0.0)])
    d = diff(a0, base)
    print(f"  mean abs diff vs baseline: {d:.2f}", flush=True)
    assert d == 0, f"zero-weight LoRA should not change the image (diff={d:.2f})"

    print("\n[5] no LoRA again (deltas cleared, expect baseline)", flush=True)
    base2, _ = gen([])
    d = diff(base2, base)
    print(f"  mean abs diff vs baseline: {d:.2f}", flush=True)
    assert delta_report(backend) == 0, "deltas were not cleared"
    assert d == 0, f"deltas were not cleared (diff={d:.2f})"

    print("\nOK: Flux LoRA parse, key mapping, runtime deltas, caching "
          "and unload all fine")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
