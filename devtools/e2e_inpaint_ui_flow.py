"""E2E: replicate the user's web-UI flow against the real SDXL backend.

1. generate a bird image (canvas auto-set)
2. paint a mask blob in the centre
3. generate again with mode=inpaint_masked, denoise=0.6
4. assert the masked region changed, the unmasked region was preserved
"""
import hashlib
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
port = 8793
tmp = Path(tempfile.mkdtemp(prefix="imagagen-e2e-"))
(tmp / "models").symlink_to(ROOT / "models")
(tmp / "imagagen.ini").write_text("[imagagen]\nprimary-gpu = 1\n")

proc = subprocess.Popen(
    [sys.executable, "-m", "imagagen.server", "--port", str(port),
     "--root", str(tmp)],
    cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
base = f"http://127.0.0.1:{port}"


def fail(msg):
    print(f"\nE2E FAILED: {msg}")
    print("---- server log ----")
    proc.terminate()
    print((proc.stdout.read() or b"").decode()[-3000:])
    sys.exit(1)


try:
    # wait for health
    ok = False
    for _ in range(120):
        try:
            r = httpx.get(base + "/api/health", timeout=0.5)
            if r.status_code == 200:
                ok = True
                print("health:", r.json().get("gpu"), flush=True)
                break
        except Exception:
            time.sleep(0.5)
    if not ok:
        fail("server did not come up")

    c = httpx.Client(base_url=base, timeout=60)

    # 1. bird
    print("\n[1] generating bird 512x512 x20 ...", flush=True)
    r = c.post("/api/generate", json={
        "prompt": "a small red bird on a branch, forest",
        "width": 512, "height": 512, "steps": 20, "seed": 7})
    if r.status_code != 200:
        fail(f"generate bird -> {r.status_code} {r.text}")
    job = r.json()["job_id"]
    print("  job", job, flush=True)
    before = None
    for _ in range(300):  # wait for the first image auto-set
        time.sleep(2)
        r2 = c.get("/api/canvas.png")
        if r2.status_code == 200 and len(r2.content) > 100:
            before = r2.content
            break
    if before is None:
        fail("canvas empty after bird generation (job may have errored)")
    print(f"  bird on canvas ({len(before)} bytes)", flush=True)

    # 2. paint a mask blob in the centre
    pts = [[x, y] for y in range(220, 301, 4) for x in range(220, 301, 4)]
    r = c.post("/api/canvas/stroke", json={
        "points": pts, "size": 18, "opacity": 1.0, "brush": "circle",
        "color": "#ff0000", "target": "mask"})
    if r.status_code != 200:
        fail(f"mask stroke -> {r.status_code} {r.text}")
    m = np.asarray(
        Image.open(io.BytesIO(c.get("/api/mask.png").content)).convert("L"))
    if m.max() < 200:
        fail("mask painting produced no mask")
    print(f"  mask painted: {int((m > 200).sum())} px > 200", flush=True)

    # 3. inpaint masked, denoise 0.6 (the user's exact request)
    print("\n[3] inpaint_masked denoise=0.6 (different prompt) ...", flush=True)
    r = c.post("/api/generate", json={
        "prompt": "a small green frog on a branch, forest",
        "width": 512, "height": 512, "steps": 20, "seed": 8,
        "mode": "inpaint_masked", "denoise": 0.6})
    if r.status_code != 200:
        fail(f"generate inpaint -> {r.status_code} {r.text}")
    job2 = r.json()["job_id"]
    print("  job", job2, flush=True)

    # 4. wait for the canvas to change (first image auto-set)
    t0 = time.time()
    after = None
    while time.time() - t0 < 420:
        time.sleep(2)
        cur = c.get("/api/canvas.png").content
        if hashlib.md5(cur).digest() != hashlib.md5(before).digest():
            after = cur
            break
    if after is None:
        fail("canvas never changed (job may have errored; see log)")
    print(f"  canvas updated after {time.time() - t0:.0f}s", flush=True)

    b = np.asarray(Image.open(io.BytesIO(before)).convert("RGB")).astype(int)
    a = np.asarray(Image.open(io.BytesIO(after)).convert("RGB")).astype(int)

    # core of the painted mask (well inside the blob)
    core = (m[235:285, 235:285] > 128)
    d_core = np.abs(a[235:285, 235:285] - b[235:285, 235:285]).mean()
    # far corner, definitely unmasked
    d_corner = np.abs(a[:80, :80] - b[:80, :80]).mean()
    # mask size must be retained
    m2 = np.asarray(
        Image.open(io.BytesIO(c.get("/api/mask.png").content)).convert("L"))
    print(f"\n  masked-region   mean abs diff: {d_core:.2f}  (expect clearly > 0)")
    print(f"  unmasked corner mean abs diff: {d_corner:.2f}  (expect ~0)")
    print(f"  mask retained:   {bool((m2 > 200).sum() > 0)}")
    if d_core < 3:
        fail(f"masked region barely changed ({d_core:.2f})")
    if d_corner > 3:
        fail(f"unmasked region changed ({d_corner:.2f})")
    if (m2 > 200).sum() == 0:
        fail("mask was lost across generation")

    print("\nE2E PASSED", flush=True)
    Image.fromarray(a.astype(np.uint8)).save("/tmp/e2e_after.png")
    Image.fromarray(b.astype(np.uint8)).save("/tmp/e2e_before.png")
finally:
    proc.terminate()
    proc.wait(timeout=10)
