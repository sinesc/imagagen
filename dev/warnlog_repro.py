"""Reproduce a cold server start + one generation and capture the console log.

Usage (project root):
    .venv/bin/python dev/warnlog_repro.py /tmp/warnlog.txt [port]

Starts the real server on the given port, hits the endpoints a UI session
hits on startup (segmentation models), submits one txt2img generation with
one LoRA, waits for it to finish, then shuts the server down. The server's
stdout+stderr are captured into the output file.
"""

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = "http://127.0.0.1:{port}"


def http(url, method="GET", body=None):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req) as r:
        return r.status, r.read()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/warnlog.txt"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8091
    p = subprocess.Popen(
        ["sh", "launch.sh", "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True)
    try:
        base = ROOT.format(port=port)
        for _ in range(120):
            if p.poll() is not None:
                sys.exit("server died on startup")
            try:
                http(base + "/")
                break
            except Exception:
                time.sleep(0.5)
        else:
            sys.exit("server did not start in time")
        # same calls the UI makes on load
        http(base + "/api/segmentation/models")
        http(base + "/api/models")
        http(base + "/api/loras")
        status, body = http(base + "/api/generate", "POST", {
            "prompt": "a red cube on a wooden table",
            "width": 768, "height": 768, "steps": 20, "cfg": 7.0,
            "seed": 7, "loras": [["EnvyZoomSliderXL1.0", 1.0]],
        })
        job_id = json.loads(body)["job_id"]
        # wait for the single image to be ready
        for _ in range(1200):
            time.sleep(1)
            try:
                st, _ = http(base + f"/api/thumb/{job_id}/0.png")
            except Exception:
                continue
            if st == 200:
                break
        else:
            sys.exit("generation did not finish in time")
        http(base + "/api/canvas.png")
        time.sleep(2)  # let in-flight WS/log writes flush
        os.killpg(os.getpgid(p.pid), signal.SIGINT)
        try:
            p.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    finally:
        if p.poll() is None:
            p.kill()
    with open(out, "w") as f:
        f.write(p.stdout.read())
    print(f"log written to {out}")


if __name__ == "__main__":
    main()
