"""Live check of the inpaint context-padding fix on the real SDXL backend.

Replicates the user's scenario: the 4-...bird.png macro photo, a soft blob
mask over the sky behind the bird, mode=inpaint_masked denoise=0.6.
Result saved to /tmp/inpaint_padded.png for visual inspection.
"""
import io
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
port = 8794
tmp = Path(tempfile.mkdtemp(prefix="imagagen-pad-"))
(tmp / "models").symlink_to(ROOT / "models")
(tmp / "imagagen.ini").write_text("[imagagen]\nprimary-gpu = 1\n")

proc = subprocess.Popen(
    [sys.executable, "-m", "imagagen.server", "--port", str(port),
     "--root", str(tmp)],
    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
base = f"http://127.0.0.1:{port}"


def fail(msg):
    print(f"\nFAILED: {msg}")
    proc.terminate()
    print((proc.stdout.read() or b"").decode()[-3000:])
    sys.exit(1)


try:
    for _ in range(120):
        try:
            if httpx.get(base + "/api/health", timeout=0.5).status_code == 200:
                break
        except Exception:
            pass
        time.sleep(0.5)
    else:
        fail("server did not come up")

    c = httpx.Client(base_url=base, timeout=120)

    # canvas = the original bird photo
    buf = io.BytesIO()
    Image.open(ROOT / "output/4-professional-macro-photo-"
                     "of-a-beautiful-bird-sitting-in-a-tree.png").save(buf, "PNG")
    r = c.post("/api/canvas/upload", content=buf.getvalue())
    assert r.status_code == 200, r.text

    # soft blob mask over the sky behind the bird (user's soft24 area)
    pts = [[x, y] for y in range(340, 760, 14) for x in range(400, 700, 14)]
    r = c.post("/api/canvas/stroke", json={
        "points": pts, "size": 48, "opacity": 1.0, "brush": "circle",
        "color": "#ff0000", "target": "mask"})
    assert r.status_code == 200, r.text
    m = np.asarray(Image.open(io.BytesIO(c.get("/api/mask.png").content)).convert("L"))
    print(f"mask painted: {int((m > 200).sum())} px > 200", flush=True)

    r = c.post("/api/generate", json={
        "prompt": "professional macro photo of a beautiful bird sitting in a "
                  "tree, a sliver, a bright sky",
        "width": 1024, "height": 1024, "steps": 25, "seed": 11,
        "mode": "inpaint_masked", "denoise": 0.6})
    if r.status_code != 200:
        fail(f"generate -> {r.status_code} {r.text}")
    print("job", r.json()["job_id"], flush=True)

    before = np.asarray(Image.open(io.BytesIO(c.get("/api/canvas.png").content)))
    t0 = time.time()
    after = None
    while time.time() - t0 < 600:
        time.sleep(2)
        cur = np.asarray(Image.open(io.BytesIO(c.get("/api/canvas.png").content)))
        if not (cur == before).all():
            after = cur
            break
    if after is None:
        fail("canvas never changed")
    print(f"done in {time.time() - t0:.0f}s", flush=True)

    core = m[420:680, 480:620] > 128
    d_core = np.abs(after.astype(int)[420:680, 480:620]
                    - before.astype(int)[420:680, 480:620])[core].mean()
    d_corner = np.abs(after.astype(int)[:150, :150]
                      - before.astype(int)[:150, :150]).mean()
    print(f"masked core mean diff: {d_core:.2f}")
    print(f"unmasked corner mean diff: {d_corner:.2f}")
    Image.fromarray(after).save("/tmp/inpaint_padded.png")
    print("saved /tmp/inpaint_padded.png", flush=True)
finally:
    proc.terminate()
    proc.wait(timeout=10)
