"""Smoke test: memdbg census against a REAL resident model (needs a GPU).

Loads one SDXL checkpoint on the given GPU with memory-debug enabled,
runs one small generation, parks it, restores+parks it again, and prints
the census at each step. Verifies that:

* the resident/parked backend's weights are attributed per copy
  (``census backend ...`` lines) and NOT double-counted as UNATTRIBUTED,
* the park/restore/prefetch events stamp with RSS,
* the dropped parked copy reports ``evt=freed`` (weakref callback).

Usage: .venv/bin/python dev/smoke_memdbg.py [gpu]   (default 1)
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s: %(message)s")

from imagagen import memdbg  # noqa: E402
from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.models import ModelRegistry  # noqa: E402

gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 1
config = load_config()
memdbg.configure(True, interval=3600, census_every=999,
                 log_path=config.user_dir / "memdbg-smoke.log")
registry = ModelRegistry(config, gpus=[gpu])
info = next(m for m in registry.models if m.family == "sdxl")
print(f"== model: {info.id} on GPU {gpu}")

registry.run(info.id, gpu, GenerateRequest(prompt="a cat", width=512,
                                           height=512, steps=4, seed=1))
memdbg.stamp("smoke-generated")
print(memdbg.census("after-generation (resident)"))

registry.unload_all()          # park to RAM
print(memdbg.census("after-park (CPU weights)"))

registry.run(info.id, gpu, GenerateRequest(prompt="a dog", width=512,
                                           height=512, steps=4, seed=2))
registry.unload_all()          # park again; same copy re-parked (no drop)
time.sleep(1.0)
print(memdbg.census("after second park").splitlines()[-1])

# force the parked copy out of the registry: only now may it die; a
# missing "evt=freed" line here means something else holds it
registry._gpu_states[gpu].parked = None
registry._gpu_states[gpu].parked_id = None
import gc

gc.collect()
time.sleep(1.0)
ok = memdbg._counts.get("freed", 0) >= 1
print("events:", dict(memdbg._counts))
print("OK" if ok else "FAIL: no freed line for the dropped parked copy")
