"""End-to-end server smoke test: one generation per sampler/scheduler combo.

The server must already be running (default port 8899). Exercises the full
stack: HTTP generate request -> queue -> backend -> WebSocket events.

Usage (from the project root, server running):
    .venv3.13/bin/python devtools/server_smoke.py [port]
"""

import asyncio
import json
import sys
import urllib.request

import websockets

PROMPT = "a red cube on a wooden table"
COMBOS = [
    ("Euler a", "Karras"),
    ("DPM++ 2M", "Karras"),
    ("DPM++ 2M SDE", "Exponential"),
    ("Euler a", "Polyexponential"),
]


async def post(port: int, body: dict) -> dict:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/generate",
        data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(None, urllib.request.urlopen, req)
    return json.loads(resp.read())


async def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
    ok = True
    async with websockets.connect(f"ws://127.0.0.1:{port}/api/ws") as ws:
        for sampler, sched in COMBOS:
            job = (await post(port, {
                "prompt": PROMPT, "width": 512, "height": 512,
                "steps": 10, "cfg": 6, "seed": 1,
                "sampler": sampler, "scheduler": sched}))["job_id"]
            imgs, result = 0, "timeout"
            while True:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=120))
                if msg["type"] == "image":
                    imgs += 1
                if msg["type"] == "done":
                    result = f"ok, {imgs} img"
                    break
                if msg["type"] == "error":
                    result = f"ERROR: {msg.get('message')}"
                    break
            ok &= result.startswith("ok")
            print(f"{sampler:14s} {sched:16s} job={job} -> {result}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
