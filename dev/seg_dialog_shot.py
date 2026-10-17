"""One-off eyeball shot of the person-mask (segment) dialog.

Starts a mock server in a temp root, opens the segment dialog, and
screenshots it to dev/references/seg-dialog.png.

Usage: .venv/bin/python dev/seg_dialog_shot.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main() -> int:
    from playwright.sync_api import sync_playwright

    port = free_port()
    root = Path("/tmp/seg-dialog-shot")
    subprocess.run(["rm", "-rf", str(root)], check=False)
    root.mkdir(parents=True)
    proc = subprocess.Popen(
        [sys.executable, "-m", "imagagen.server", "--port", str(port),
         "--root", str(root), "--mock"],
        cwd=REPO, env={**os.environ},
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(150):
            try:
                httpx.get(base + "/api/health", timeout=0.5)
                break
            except Exception:
                time.sleep(0.1)
        else:
            print("server did not come up")
            return 1

        # the real selfie model so the dialog has a model to list
        src = REPO / "models/segmentation/selfie_multiclass_256x256.tflite"
        (root / "models/segmentation").mkdir(parents=True, exist_ok=True)
        if src.is_file():
            (root / "models/segmentation" / src.name).write_bytes(src.read_bytes())

        with sync_playwright() as pw:
            b = pw.chromium.launch()
            page = b.new_page(viewport={"width": 1600, "height": 900})
            page.goto(base)
            page.wait_for_selector("#toolbar")
            page.click("#seg-run")
            page.wait_for_selector("#dialog:not([hidden])")
            page.locator("#dialog-box").screenshot(
                path=REPO / "dev/references/seg-dialog.png")
            # toolbar around the Segment button (no model dropdown, no sep)
            page.locator("#toolbar").screenshot(
                path=REPO / "dev/references/seg-toolbar.png")
            b.close()
        print("wrote dev/references/seg-dialog.png + seg-toolbar.png")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
