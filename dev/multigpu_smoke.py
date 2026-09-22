"""End-to-end multi-GPU smoke test (requires >= 2 free real GPUs).

Starts the real server (no --mock), then verifies:
  1. a lone image loads/runs on the primary GPU only
  2. a batch of 3 on 2 GPUs: total time ~ 2x single-image time (not 3x)
  3. both GPUs show activity (utilisation) during the batch
  4. all 3 images complete

Usage (from the project root):
    .venv/bin/python dev/multigpu_smoke.py [port]
"""

import asyncio
import json
import subprocess
import sys
import threading
import time
import urllib.request

import websockets

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
BASE = f"http://127.0.0.1:{PORT}"
WS_URL = f"ws://127.0.0.1:{PORT}/api/ws"
SERVER_LOG = "/tmp/imagagen_multigpu_smoke_server.log"


def http(method: str, path: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def gpu_state() -> dict[int, tuple[int, int]]:
    """{gpu index: (memory.used MiB, utilization.gpu %)}"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu", "index,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    state: dict[int, tuple[int, int]] = {}
    for line in out.strip().splitlines():
        idx, used, util = line.split(", ")
        state[int(idx)] = (int(used), int(util))
    return state


class GpuSampler:
    """Background poller recording peak per-GPU VRAM + utilisation."""

    def __init__(self, gpus: list[int]):
        self.gpus = gpus
        self.peak_mem = {g: 0 for g in gpus}
        self.peak_util = {g: 0 for g in gpus}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                state = gpu_state()
            except Exception:
                self._stop.wait(0.2)
                continue
            for g in self.gpus:
                if g in state:
                    used, util = state[g]
                    if used > self.peak_mem[g]:
                        self.peak_mem[g] = used
                    if util > self.peak_util[g]:
                        self.peak_util[g] = util
            self._stop.wait(0.2)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()


async def generate_and_wait(ws, body: dict, timeout: float = 600) -> tuple[int, float]:
    """POST /api/generate, consume WS until 'done'; return (images, elapsed s)."""
    req = urllib.request.Request(
        f"{BASE}/api/generate", data=json.dumps(body).encode(),
        headers={"content-type": "application/json"})
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(None, urllib.request.urlopen, req)
    job_id = json.loads(resp.read())["job_id"]
    t0 = time.time()
    imgs = 0
    while True:
        msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
        if msg.get("job") != job_id:
            continue
        if msg["type"] == "image":
            imgs += 1
        elif msg["type"] == "error":
            raise RuntimeError(f"job {job_id} error: {msg.get('message')}")
        elif msg["type"] == "done":
            return imgs, time.time() - t0


async def main() -> int:
    # start the real server (no --mock) from the project root
    server_log = open(SERVER_LOG, "wb")
    server = subprocess.Popen(
        [sys.executable, "-m", "imagagen", "--port", str(PORT)],
        stdout=server_log, stderr=subprocess.STDOUT)
    try:
        # wait for health (fresh process: torch import + app setup can be slow)
        health = None
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                health = http("GET", "/api/health")
                break
            except Exception:
                time.sleep(1)
        if health is None:
            print("FAIL: server did not become healthy in time")
            return 1
        pool = health.get("pool", [])
        primary = health.get("primary")
        print(f"primary={primary} pool={pool}")
        if len(pool) < 2:
            print("FAIL: need >= 2 usable non-ignored GPUs (see /api/health pool)")
            return 1
        state0 = gpu_state()
        pool = [g for g in pool if g in state0]
        if len(pool) < 2:
            print("FAIL: pool GPUs not visible to nvidia-smi")
            return 1
        baseline = {g: s[0] for g, s in state0.items()}

        async with websockets.connect(WS_URL) as ws:
            # --- warmup: 1 image (loads the model on the primary) ----------
            print("warmup: 1 image (loads model on primary) ...", flush=True)
            imgs, dt = await generate_and_wait(ws, {
                "prompt": "a red cube on a wooden table",
                "width": 512, "height": 512, "steps": 10, "cfg": 6, "seed": 1})
            assert imgs == 1
            print(f"  ok in {dt:.1f}s")

            # a lone image must leave only the primary GPU loaded
            state = gpu_state()
            single_ok = (
                state[primary][0] - baseline[primary] > 2000
                and all(g == primary or state[g][0] - baseline[g] < 2000
                        for g in pool)
            )
            print(f"  single-image residency: "
                  f"{ {g: state[g][0] - baseline[g] for g in pool} } (MiB) -> "
                  f"{'OK (primary only)' if single_ok else 'UNEXPECTED'}")

            # --- pre-warm the second GPU (tiny batch: img2 loads on it) ----
            print("pre-warm: 2x small images (loads model on 2nd GPU) ...", flush=True)
            imgs, dt = await generate_and_wait(ws, {
                "prompt": "a red cube on a wooden table",
                "width": 256, "height": 256, "steps": 2, "cfg": 6, "seed": 2,
                "batch_size": 2})
            assert imgs == 2
            print(f"  ok in {dt:.1f}s")

            # --- baseline: single-image time (both models now resident) ----
            imgs, t1 = await generate_and_wait(ws, {
                "prompt": "a red cube on a wooden table",
                "width": 512, "height": 512, "steps": 10, "cfg": 6, "seed": 3})
            assert imgs == 1
            print(f"single image time: {t1:.1f}s")

            pre_batch = {g: s[0] for g, s in gpu_state().items()}
            sampler = GpuSampler(pool)
            sampler.start()

            # --- the batch: 3 images should run on 2 GPUs in parallel ------
            imgs, t3 = await generate_and_wait(ws, {
                "prompt": "a red cube on a wooden table",
                "width": 512, "height": 512, "steps": 10, "cfg": 6, "seed": 4,
                "batch_size": 3})
            sampler.stop()
            assert imgs == 3, f"expected 3 images, got {imgs}"

            rise = {g: sampler.peak_mem[g] - pre_batch[g] for g in pool}
            # utilisation is the activity metric: a pre-warmed GPU reuses
            # its cached CUDA-allocator pool, so memory.used does not rise
            # on every generation
            util = {g: sampler.peak_util[g] for g in pool}
            both_active = all(u >= 1 for u in util.values())
            print(f"batch of 3 time: {t3:.1f}s "
                  f"(~{t3 / t1:.2f}x single; sequential would be ~3.00x)")
            print(f"batch VRAM rise per GPU (MiB): {rise}")
            print(f"batch peak utilisation per GPU (%): {util}")
            timing_ok = t3 < 2.6 * t1
            print(f"timing: {'OK (parallel)' if timing_ok else 'SUSPECT (serial?)'}")
            print(f"both GPUs active during batch: {both_active}")

        ok = single_ok and timing_ok and both_active
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except Exception:
            server.kill()
        server_log.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
