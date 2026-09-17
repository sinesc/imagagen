"""Live SDXL LoRA smoke test (GPU).

Verifies the backend's adapter plumbing end to end, using the real LoRA in
``loras/sdxl/`` (A1111/Kohya on-disk format) plus one fabricated rank-4
LoRA ("beta") written to a temp dir:

  0. discovery: the project's loras/ must list the real LoRA
  1. baseline (no LoRA)
  2. real LoRA @ 1.0      -> must differ from baseline (adapter is active)
  3. real LoRA @ 1.0 again-> identical to (2) (deterministic seed; the
                             unchanged set is served from the injection cache)
  4. real LoRA @ 0.0      -> identical to baseline (zero-weight adapter)
  5. real @ 1.0 + beta @ 1.0 -> differs from both single-LoRA runs
  6. no LoRA again        -> adapters unloaded, identical to baseline

The model registry is pointed at a temp root (models/ is a symlink to the
project's, loras/ is a temp copy), so the script never writes into the
project's data dirs.

Usage (from the project root; needs GPU + models/sdxl/*.safetensors):
    .venv/bin/python devtools/smoke_sdxl_lora.py [gpu-index]
"""
import dataclasses
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.model_configs import ConfigStore  # noqa: E402
from imagagen.models import ModelRegistry, list_loras  # noqa: E402

# attention projections the fabricated LoRA patches (SDXL diffusers layout:
# down_blocks[0] has no attention; .attentions is a ModuleList)
TARGETS = [
    "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_q",
]


def fabricate_lora(unet, seed: int, path: Path, rank: int = 4) -> None:
    """Write a small LoRA checkpoint over a few attn projections.

    Emitted in the *diffusers-native* format (``unet.<module>`` prefix,
    PEFT ``lora_A``/``lora_B``) rather than the A1111/Kohya on-disk format:
    the module paths are the real UNet's, so the targets are guaranteed to
    be found, and the A1111-conversion path is already exercised by the real
    FuturisticXL LoRA. This one is purely a second adapter for the merge test.
    """
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for p in TARGETS:
        m = unet
        for part in p.split("."):
            m = m[int(part)] if part.isdigit() else getattr(m, part)
        scale = 0.3
        sd[f"unet.{p}.lora_A.weight"] = torch.randn(
            rank, m.in_features, generator=g) * scale
        sd[f"unet.{p}.lora_B.weight"] = torch.randn(
            m.out_features, rank, generator=g) * scale
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path))


def main(gpu: int | None) -> None:
    # temp root: models/ is a symlink to the project's (incl. the HF cache);
    # loras/ is a copy (so the fabricated LoRA stays out of the project)
    tmp = Path(tempfile.mkdtemp(prefix="imagagen_lora_"))
    (tmp / "models").symlink_to(ROOT / "models")
    src_loras = ROOT / "loras"
    if src_loras.is_dir():
        shutil.copytree(src_loras, tmp / "loras", dirs_exist_ok=True)
    cfg = dataclasses.replace(load_config(ROOT), root=tmp)

    # 0. discovery (against the *project* loras/, not the temp copy)
    found = list_loras(dataclasses.replace(cfg, root=ROOT))
    print("project loras:", found, flush=True)
    assert any("FuturisticXL" in n for n in found), (
        f"expected the copied-in LoRA in loras/sdxl/, got {found!r}"
    )
    store = ConfigStore(cfg, lambda: [])
    assert store.entries() == []  # empty model-configs dir in the temp root

    reg = ModelRegistry(cfg, gpu=gpu)
    info = reg.models[0]
    print("loading", info.name, "...", flush=True)
    t0 = time.time()
    backend = reg.backend_for(info.id)
    print(f"loaded in {time.time() - t0:.1f}s", flush=True)

    fabricate_lora(backend.pipe.unet, 11, cfg.loras_dir / "sdxl" / "beta.safetensors")
    lora_a = "FuturisticXL"

    def gen(loras):
        req = GenerateRequest(prompt="a blue cube on a table", width=512,
                              height=512, steps=8, seed=42, loras=loras)
        t0 = time.time()
        img = backend.generate(req).image
        print(f"  {loras or '(no lora)'}  {time.time() - t0:.1f}s", flush=True)
        return img

    def diff(a, b) -> float:
        return float(np.abs(a.astype(int) - b.astype(int)).mean())

    print("\n[1] baseline", flush=True)
    base = gen([])
    print(f"\n[2] {lora_a} @ 1.0 (expect clearly different)", flush=True)
    a1 = gen([(lora_a, 1.0)])
    d = diff(a1, base)
    print(f"  mean abs diff vs baseline: {d:.2f}")
    assert d > 5, f"LoRA had no effect (diff={d:.2f})"
    print(f"\n[3] {lora_a} @ 1.0 again (cache hit, expect identical to [2])", flush=True)
    a2 = gen([(lora_a, 1.0)])
    d = diff(a2, a1)
    print(f"  mean abs diff vs [2]: {d:.2f}")
    assert d == 0, f"same seed + same LoRAs should be deterministic (diff={d:.2f})"
    print(f"\n[4] {lora_a} @ 0.0 (expect identical to baseline)", flush=True)
    a0 = gen([(lora_a, 0.0)])
    d = diff(a0, base)
    print(f"  mean abs diff vs baseline: {d:.2f}")
    assert d == 0, f"zero-weight LoRA should not change the image (diff={d:.2f})"
    print("\n[5] FuturisticXL @ 1.0 + beta @ 1.0 (expect different from both)",
          flush=True)
    ab = gen([(lora_a, 1.0), ("beta", 1.0)])
    d1, d2 = diff(ab, a1), diff(ab, base)
    print(f"  mean abs diff vs [2]: {d1:.2f}, vs baseline: {d2:.2f}")
    assert d1 > 5 and d2 > 5, "merged LoRA set indistinguishable from singles"
    print("\n[6] no LoRA again (adapters unloaded, expect baseline)", flush=True)
    base2 = gen([])
    d = diff(base2, base)
    print(f"  mean abs diff vs baseline: {d:.2f}")
    assert d == 0, f"adapters were not unloaded (diff={d:.2f})"

    print("\nOK: LoRA discovery, injection, per-LoRA weights, caching and "
          "unload all fine")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
