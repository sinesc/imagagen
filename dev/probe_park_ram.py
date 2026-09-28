"""Probe: system RAM behaviour while cycling through different models.

Loads and parks (moves to RAM) three backends in sequence on one GPU,
mimicking "testing different models" in the app, and reports process RSS
and the registry's resident/parked state after each step. This answers
whether parked models accumulate in RAM or are bounded/dropped.

Usage: .venv/bin/python dev/probe_park_ram.py [gpu]
"""
from __future__ import annotations

import gc
import sys
import time

import torch

from imagagen.config import load_config
from imagagen.models import ModelRegistry


def rss_gb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1e6
    raise RuntimeError("no VmRSS")


def cpu_torch_gb() -> float:
    """Total bytes held by CPU tensors still alive in the process."""
    total = 0
    seen = set()
    for obj in gc.get_objects():
        if torch.is_tensor(obj) and obj.device.type == "cpu" and id(obj) not in seen:
            seen.add(id(obj))
            total += obj.numel() * obj.element_size()
    return total / 1e9


def main() -> None:
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    torch.cuda.init()
    torch.cuda.set_device(gpu)
    cfg = load_config()
    reg = ModelRegistry(cfg, gpus=[gpu])

    print("discovered:", [m["id"] for m in reg.list()])
    candidates = ["epicrealismXL11.0", "agFluxQ4KS_v21",
                  "intorealism_zitV40"]
    available = [c for c in candidates if reg.get(c) is not None]
    print(f"GPU: {gpu}  models: {available}")
    st = reg._get_gpu_state(gpu)

    def report(label: str) -> None:
        print(f"{label}: RSS={rss_gb():.1f} GB  "
              f"CPU-tensors={cpu_torch_gb():.1f} GB  "
              f"resident={st.model_id!r} parked={st.parked_id!r}", flush=True)

    report("baseline")
    for mid in available:
        t0 = time.monotonic()
        reg.backend_for(mid, gpu=gpu)
        print(f"  loaded {mid} in {time.monotonic() - t0:.0f}s", flush=True)
        report(f"after load {mid} (resident)")
        reg.unload_all()
        gc.collect()
        report(f"after park {mid} (unload_all)")

    # go back to the first model: the last parked one must be dropped
    first = available[0]
    t0 = time.monotonic()
    reg.backend_for(first, gpu=gpu)
    print(f"  reloaded {first} in {time.monotonic() - t0:.0f}s", flush=True)
    report("after full cycle (back to first model)")
    reg.unload_all()
    gc.collect()
    report("final (idle, parked)")


if __name__ == "__main__":
    main()
