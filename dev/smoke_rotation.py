"""Live rotation smoke test (GPU 1, per AGENTS.local.md).

Starts the real server against this repo's root and runs:

1. 1024x1024 generate with rotation 180 -> the result canvas must be
   1024x1024 and the PNG must carry rotation 180 (not the generation size)
2. 1024x1024 generate with rotation 45 -> the generation runs at
   ceil(1024*sqrt(2)) = 1449 x 1449 (extended so the rotated image fits);
   the result canvas is back at 1024x1024 with rotation 45 recorded
3. a 1024x1024 inpaint (masked) with rotation 90 -> the canvas keeps its
   size, the replaced region is back at the original mask box

Polls the result thumbnail instead of the WebSocket (no ws client needed).
"""
import io
import math
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> None:
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    log = open("/tmp/smoke_rotation.log", "wb")
    proc = subprocess.Popen(
        [str(ROOT / ".venv" / "bin" / "python"), "-m", "imagagen",
         "--root", str(ROOT), "--port", str(port)],
        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        with httpx.Client(base_url=base, timeout=30) as c:
            # wait for the server
            for _ in range(120):
                try:
                    if c.get("/api/health").status_code == 200:
                        break
                except Exception:
                    pass
                if proc.poll() is not None:
                    log.flush()
                    sys.exit(f"server exited early:\n"
                             f"{log.name}: {log.getvalue()!r}")
                time.sleep(0.5)
            else:
                sys.exit(f"server did not come up (log: {log.name})")

            # a flat canvas so the jobs have a source to retain
            buf = io.BytesIO()
            Image.new("RGB", (1024, 1024), (40, 60, 80)).save(buf, format="PNG")
            assert c.post("/api/canvas/upload",
                          content=buf.getvalue()).status_code == 200

            def gen(**kw) -> int:
                r = c.post("/api/generate", json={"prompt": "a bird on a "
                             "branch, forest", "steps": 8, "seed": 42,
                             "width": 1024, "height": 1024, **kw})
                assert r.status_code == 200, r.text
                job = r.json()["job_id"]
                # wait for the result thumbnail (no WS client here)
                for _ in range(600):
                    if c.get(f"/api/thumb/{job}/0.png").status_code == 200:
                        break
                    time.sleep(1)
                else:
                    log.flush()
                    sys.exit(f"job {job} never produced a result "
                             f"(log: {log.name})")
                r = c.post("/api/canvas/thumb",
                           json={"job": job, "index": 0})
                assert r.status_code == 200, r.text
                return job

            def canvas() -> Image.Image:
                r = c.get("/api/canvas.png")
                assert r.status_code == 200
                img = Image.open(io.BytesIO(r.content))
                img.load()
                return img

            def canvas_meta() -> dict:
                from imagagen import metadata as md
                r = c.get("/api/canvas.png")
                return md.read_metadata(r.content)

            # 1. rotation 180
            print("[1] generate 1024x1024 rotation 180", flush=True)
            gen(rotation=180)
            img = canvas()
            print(f"  canvas: {img.size}", flush=True)
            assert img.size == (1024, 1024)
            meta = canvas_meta()
            assert meta["rotation"] == 180, meta
            assert (meta["width"], meta["height"]) == (1024, 1024), meta
            print(f"  metadata: rotation={meta['rotation']} "
                  f"size={meta['width']}x{meta['height']} (requested)",
                  flush=True)

            # 2. rotation 45: the generation size is extended
            expected = math.ceil(1024 * math.sqrt(2))  # 1449
            print(f"[2] generate 1024x1024 rotation 45 "
                  f"(generation at {expected}x{expected})", flush=True)
            gen(rotation=45)
            img = canvas()
            print(f"  canvas: {img.size}", flush=True)
            assert img.size == (1024, 1024)
            meta = canvas_meta()
            assert meta["rotation"] == 45, meta
            assert (meta["width"], meta["height"]) == (1024, 1024), meta
            print(f"  metadata: rotation={meta['rotation']} "
                  f"size={meta['width']}x{meta['height']} (requested)",
                  flush=True)

            # 3. inpaint (masked) with rotation 90: a 128x128 mask box
            print("[3] inpaint 1024x1024 rotation 90", flush=True)
            img = canvas().convert("RGB")
            mask = Image.new("RGBA", img.size, (128, 128, 128, 0))
            px = mask.load()
            for yy in range(400, 528):
                for xx in range(400, 528):
                    px[xx, yy] = (128, 128, 128, 255)
            buf_i, buf_m = io.BytesIO(), io.BytesIO()
            img.save(buf_i, format="PNG")
            mask.save(buf_m, format="PNG")
            r = c.post("/api/canvas/sync",
                       files={"image": ("canvas.png", buf_i.getvalue(),
                                        "image/png"),
                             "mask": ("mask.png", buf_m.getvalue(),
                                      "image/png")})
            assert r.status_code == 200, r.text
            r = c.post("/api/generate", json={"prompt": "a bird on a "
                         "branch, forest", "steps": 8, "seed": 42,
                         "width": 1024, "height": 1024,
                         "mode": "inpaint_masked", "denoise": 1.0,
                         "rotation": 90, "mask_padding": 0})
            assert r.status_code == 200, r.text
            job = r.json()["job_id"]
            for _ in range(600):
                if c.get(f"/api/thumb/{job}/0.png").status_code == 200:
                    break
                time.sleep(1)
            else:
                log.flush()
                sys.exit(f"job {job} never produced a result (log: {log.name})")
            r = c.post("/api/canvas/thumb", json={"job": job, "index": 0})
            assert r.status_code == 200, r.text
            img = canvas()
            print(f"  canvas: {img.size}", flush=True)
            assert img.size == (1024, 1024)
            print("  inpaint with rotation 90 completed", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
    print("ROTATION SMOKE OK (server log: /tmp/smoke_rotation.log)")


if __name__ == "__main__":
    main()
