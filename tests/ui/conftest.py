"""Fixtures and small helpers for the browser (Playwright) UI tests.

These tests are the "eyeball testing" safety net for the dependency-free
frontend: they drive a real Chromium against a live mock-backend server in
a temp root. Skipped automatically when playwright's chromium is not
installed (see the `page` fixture).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture()
def server(tmp_path: Path):
    """Live mock-backend server on a free port, rooted at tmp_path."""
    port = _free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "imagagen.server",
            "--port", str(port),
            "--root", str(tmp_path),
            "--mock",
        ],
        cwd=REPO,
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    up = False
    try:
        for _ in range(150):
            try:
                httpx.get(base + "/api/health", timeout=0.5)
                up = True
                break
            except Exception:
                if proc.poll() is not None:
                    log = proc.stdout.read().decode()
                    raise RuntimeError(f"server exited early:\n{log}") from None
                time.sleep(0.1)
        assert up, "server did not come up"
        yield base, tmp_path
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def browser():
    """One Chromium per test MODULE (the `page` fixture hands out a fresh
    tab per test).

    Launching a browser per test cost ~0.5 s of every UI test's setup. A
    fresh page still means a fresh document, and the app keeps no
    browser-side state (no localStorage/sessionStorage; ui-state lives on
    the server), so sharing the process is safe.

    Module- and NOT session-scoped: a live sync-Playwright browser breaks
    pytest-asyncio's runner for the async tests that run later in the same
    session (see tests/ui/test_loading_overlay_slow_api.py), so the browser
    has to go away at the end of the module that created it.
    """
    try:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        b = pw.chromium.launch()
    except Exception as e:  # ImportError, missing browser build, no libs
        pytest.skip(f"playwright/chromium not available: {e}")
    yield b
    b.close()
    pw.stop()


@pytest.fixture()
def page(browser):
    pg = browser.new_page(viewport={"width": 1280, "height": 800})
    yield pg
    pg.close()


# ---------------- searchable combobox (makeCombobox) helpers ----------------
# #model / #preset / .lora-name / .style-name are not native <select>s: a
# .combo-input inside the root opens a .combo-list of .combo-item entries
# (grouped ones are separated by .combo-group header entries).

def combo_open(page, root):
    """Open the list of the combobox at `root` (a Playwright locator)."""
    root.locator(".combo-input").click()
    try:
        root.locator(".combo-list.open").wait_for(timeout=500)
    except Exception:
        # a pick on another row may have just armed the 250ms ghost-click
        # guard: one more click opens the list for sure
        root.locator(".combo-input").click()
        root.locator(".combo-list.open").wait_for(timeout=5000)


def combo_pick(page, root, value: str):
    """Pick the option whose value is `value` from the combobox."""
    combo_open(page, root)
    root.locator(f".combo-item[data-value='{value}']").click()


def combo_seq(page, root):
    """Open the list and return its children as [kind, text] pairs, where
    kind is 'group' (section header) or 'item'. Closes the list again."""
    combo_open(page, root)
    seq = root.locator(".combo-list > li").evaluate_all(
        "els => els.map(e => [e.classList.contains('combo-group') ? "
        "'group' : 'item', e.textContent])")
    page.keyboard.press("Escape")
    return seq


def combo_items(page, root):
    """Open the list and return the item texts (headers excluded)."""
    return [t for k, t in combo_seq(page, root) if k == "item"]


# ---------------- canvas state helpers (server-side, authoritative) ----


def set_canvas(base: str, w: int = 200, h: int = 150, rgb=(20, 30, 40)):
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), rgb).save(buf, format="PNG")
    with httpx.Client(base_url=base, timeout=10) as c:
        r = c.post("/api/canvas/upload", content=buf.getvalue())
        assert r.status_code == 200, r.text


def canvas_px(base: str, x: int, y: int) -> tuple[int, int, int]:
    import io

    from PIL import Image

    with httpx.Client(base_url=base, timeout=10) as c:
        img = Image.open(io.BytesIO(c.get("/api/canvas.png").content))
        return img.getpixel((x, y))


def mask_arr(base: str):
    """The mask as an ndarray, or None while the canvas is empty."""
    import io

    import numpy as np
    from PIL import Image

    with httpx.Client(base_url=base, timeout=10) as c:
        r = c.get("/api/mask.png")
        if r.status_code == 404:
            return None
        return np.asarray(Image.open(io.BytesIO(r.content)))


def wait_until(fn, timeout: float = 10.0, interval: float = 0.1) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if fn():
            return True
        time.sleep(interval)
    return False


# ---------------- pointer / coordinate helpers ----------------


def screen_pos(page, x: float, y: float) -> tuple[float, float]:
    """Canvas-pixel (x, y) -> CSS viewport coordinates (uses view.*/rect)."""
    return page.evaluate(
        """([x, y]) => {
            const r = document.getElementById('canvas').getBoundingClientRect();
            return [r.left + view.ox + x * view.scale, r.top + view.oy + y * view.scale];
        }""",
        [x, y],
    )


def stroke(page, pts, shift: bool = False, ctrl: bool = False, button: str = "left"):
    """Pointer stroke through canvas-pixel points.

    The client commits strokes locally (no round trip), so the helper ends
    with a forceSync: the server store catches up immediately and the
    server-pixel assertions (canvas_px / mask_arr) stay valid.
    """
    sx, sy = screen_pos(page, *pts[0])
    page.mouse.move(sx, sy)
    if shift:
        page.keyboard.down("Shift")
    if ctrl:
        page.keyboard.down("Control")
    page.mouse.down(button=button)
    for p in pts[1:]:
        sx, sy = screen_pos(page, *p)
        page.mouse.move(sx, sy, steps=5)
    page.mouse.up(button=button)
    if ctrl:
        page.keyboard.up("Control")
    if shift:
        page.keyboard.up("Shift")
    page.evaluate("forceSync()")


def dot(page, x: float, y: float, shift: bool = False, ctrl: bool = False):
    """A single-point stroke (one stamp)."""
    stroke(page, [[x, y]], shift=shift, ctrl=ctrl)


def sync(page):
    """Push the client's pending canvas/mask edits to the server right now.

    The client does this on its own after SYNC_IDLE_MS (2 s) of idleness, so
    tests that assert on SERVER state after a non-stroke action (mask ops,
    undo/redo, paste, rotation) call this instead of paying that 2 s idle
    timer in a wait_until loop. It never weakens an assertion: the sync is
    either already scheduled or forced here, and the server-side state the
    assertions read is the same either way.
    """
    page.evaluate("forceSync()")


def flush(page, timeout: int = 1500):
    """sync() for edits that land ASYNCHRONOUSLY (undo/redo, mask ops, the
    selection composite).

    Their handlers await (morph worker, history restore), so the canvas is
    re-dirtied a tick AFTER the click/keypress: a bare sync() would push the
    PRE-edit state, and the assertion would then wait on the 2 s idle timer.
    Wait for the re-dirty first, then force the sync.
    """
    try:
        page.wait_for_function(
            "() => canvasDirty || syncTimer !== null || syncPromise", timeout=timeout)
    except Exception:
        pass  # nothing re-dirtied: forceSync below + the idle timer still run
    page.evaluate("forceSync()")


def set_input(page, selector: str, value):
    """Set an input's value and fire input+change (fill() is unreliable on
    range/color inputs across Playwright versions)."""
    page.evaluate(
        """([sel, v]) => {
            const el = document.querySelector(sel);
            el.value = v;
            el.dispatchEvent(new Event('input', {bubbles: true}));
            el.dispatchEvent(new Event('change', {bubbles: true}));
        }""",
        [selector, str(value)],
    )
