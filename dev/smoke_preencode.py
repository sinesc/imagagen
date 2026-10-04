"""Live check: a wildcard batch is pre-encoded in ONE text-encoder session.

Submits a 4-image batch whose prompt contains a wildcard (so every task
resolves a different prompt) against the FP8 Flux model on GPU 1 -- the
model where the ~4.8 GiB T5/CLIP encoders cannot co-reside with the
~11.1 GiB transformer and are swapped in/out per encode.

Old behaviour: every task missed the prompt-encoding cache, so the
transformer was swapped to the CPU (and back) ONCE PER IMAGE -- four
~11 GiB device crossings each way, plus four encoder loads.

New behaviour: the queue resolves every task's wildcard picks at submit
time and pre-encodes the batch in one encoder session BEFORE any task runs;
each task then hits the shared cache. Expect in the log:
  - exactly ONE "transformer swapped to the CPU for text encoding of 4
    prompt(s)" line, emitted BEFORE the first task's progress,
  - "job 1: pre-encoded 4 prompt(s) in one encoder session",
  - four distinct resolved prompts in the results' metadata.
"""
import logging
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s %(name)s: %(message)s")

from imagagen.backends import GenerateRequest  # noqa: E402
from imagagen.canvas import Canvas  # noqa: E402
from imagagen.config import load_config  # noqa: E402
from imagagen.models import ModelRegistry  # noqa: E402
from imagagen.queue import GenerationQueue  # noqa: E402

cfg = load_config(ROOT)
reg = ModelRegistry(cfg, gpus=[1])
info = next(m for m in reg.models if "FP8" in m.name)
print("model:", info.name, flush=True)

wc = Path(tempfile.mkdtemp(prefix="wc-"))
(wc / "mood.txt").write_text("sunny\nstormy\nmisty\nfoggy\n",
                             encoding="utf-8")


def emit(e):
    print("EVENT", e.get("type"), "job", e.get("job"),
          "idx", e.get("index", "-"), e.get("message", ""), flush=True)


q = GenerationQueue(registry=reg, canvas=Canvas(), emit=emit,
                    wildcards_dir=wc)
q.start(gpus=[1])

job = q.submit(info.id, GenerateRequest(
    prompt="a landscape with __mood__ skies", width=512, height=512,
    steps=4, seed=123), 4)

deadline = time.time() + 1200
while time.time() < deadline:
    j = q.job(job.id)
    if j is not None and j.done_emitted:
        break
    time.sleep(1)

j = q.job(job.id)
print("done:", j.done_emitted,
      "results:", len(j.results), "of", j.batch_size, flush=True)
assert j.done_emitted and len(j.results) == 4, "batch incomplete"

# the per-task metadata must carry one valid resolved prompt per image
# (the picks are independent random draws from the 4-line file, so
# duplicates between tasks are expected sometimes)
from imagagen import metadata  # noqa: E402

MOODS = ("sunny", "stormy", "misty", "foggy")
prompts = []
for i in range(4):
    m = metadata.read_metadata(j.results[i])
    print("image", i, "prompt:", m["prompt"], "seed:", m["seed"], flush=True)
    assert "__mood__" not in m["prompt"]
    assert any(w in m["prompt"] for w in MOODS)
    prompts.append(m["prompt"])
print("distinct resolved prompts:", len(set(prompts)), flush=True)
print("OK", flush=True)
