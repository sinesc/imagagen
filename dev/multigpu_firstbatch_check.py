"""Fresh-server first-batch black-image check (requires >= 2 free GPUs).

Starts a real server in a FRESH process and sends a multi-image batch as
the VERY FIRST request -- the scenario that reliably produced black images
before the queue's warm-up gate. Verifies:

  1. image 0 (the job's first image) completes FIRST: the gate holds the
     rest of the batch until it has fully generated + decoded on the
     primary GPU
  2. all images are clean (no NaN, not black/flat)
  3. a second batch afterwards is also clean (post-warm parallel runs)

Usage (from the project root):
    .venv/bin/python dev/multigpu_firstbatch_check.py [port]
"""

import asyncio
import json
import subprocess
import sys
import time
import urllib.request

import numpy as np
import websockets
from PIL import Image

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8901
BASE = f"http://127.0.0.1:{PORT}"
WS_URL = f"ws://127.0.0.1:{PORT}/api/ws"
SERVER_LOG = "/tmp/imagagen_firstbatch_check_server.log"


def http_json(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def fetch_image(path: str) -> np.ndarray:
    with urllib.request.urlopen(BASE + path, timeout=60) as resp:
        return np.asarray(Image.open(resp).convert("RGB"))


def check(tag: str, arr: np.ndarray) -> bool:
    f64 = arr.astype(np.float64)
    nan = bool(np.isnan(f64).any())
    mean = float(f64.mean())
    small = np.asarray(Image.fromarray(arr).resize((16, 16)))
    flat = small.std() < 3.0
    ok = (not nan) and (not flat) and 5.0 < mean < 250.0
    print(f"  {tag}: ok={ok} nan={nan} mean={mean:.1f} "
          f"small16_std={small.std():.1f}")
    Image.fromarray(arr).save(f"/tmp/firstbatch_{tag}.png")
    return ok


async def generate_batch(ws, seed: int, batch_size: int) -> tuple[int, dict[int, int], float]:
    """POST /api/generate, consume WS until 'done'.

    Returns (job_id, index -> completion order, elapsed seconds).
    """
    req = urllib.request.Request(
        f"{BASE}/api/generate",
        data=json.dumps({
            "prompt": "a red cube on a wooden table",
            "width": 512, "height": 512, "steps": 20, "cfg": 6,
            "seed": seed, "batch_size": batch_size,
        }).encode(),
        headers={"content-type": "application/json"})
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(None, urllib.request.urlopen, req)
    job_id = json.loads(resp.read())["job_id"]
    t0 = time.time()
    order: dict[int, int] = {}
    n = 0
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=900))
        if msg.get("job") != job_id:
            continue
        if msg["type"] == "image":
            order[msg["index"]] = n
            n += 1
        elif msg["type"] == "error":
            raise RuntimeError(f"job {job_id} error: {msg.get('message')}")
        elif msg["type"] == "done":
            return job_id, order, time.time() - t0


async def main() -> int:
    server_log = open(SERVER_LOG, "wb")
    server = subprocess.Popen(
        [sys.executable, "-m", "imagagen", "--port", str(PORT)],
        stdout=server_log, stderr=subprocess.STDOUT)
    ok = True
    try:
        health = None
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                health = http_json("GET", "/api/health")
                break
            except Exception:
                time.sleep(1)
        if health is None:
            print("FAIL: server did not become healthy in time")
            return 1
        pool = health.get("pool", [])
        print(f"primary={health.get('primary')} pool={pool}")
        if len(pool) < 2:
            print("FAIL: need >= 2 usable non-ignored GPUs (see /api/health pool)")
            return 1

        async with websockets.connect(WS_URL) as ws:
            # --- the cold first batch: the user's failure scenario --------
            print("first batch of 3 (cold server) ...", flush=True)
            job_id, order, dt = await generate_batch(ws, seed=1000, batch_size=3)
            done_order = sorted(order, key=order.get)
            print(f"  done in {dt:.1f}s, completion order: {done_order}")
            if done_order[0] != 0:
                print("FAIL: first image did not complete first (warm-up gate?)")
                return 1
            for i in done_order:
                arr = await asyncio.get_running_loop().run_in_executor(
                    None, fetch_image, f"/api/thumb/{job_id}/{i}.png")
                if not check(f"cold-{i}", arr):
                    ok = False

            # --- a warm second batch: normal parallel runs ----------------
            print("second batch of 3 (warm) ...", flush=True)
            job_id2, order2, dt2 = await generate_batch(ws, seed=2000, batch_size=3)
            done_order2 = sorted(order2, key=order2.get)
            print(f"  done in {dt2:.1f}s, completion order: {done_order2}")
            for i in done_order2:
                arr = await asyncio.get_running_loop().run_in_executor(
                    None, fetch_image, f"/api/thumb/{job_id2}/{i}.png")
                if not check(f"warm-{i}", arr):
                    ok = False
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()
        server_log.close()

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
