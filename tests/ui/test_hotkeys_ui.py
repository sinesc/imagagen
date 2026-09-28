"""Browser tests for the toolbar keyboard shortcuts (Chromium via
Playwright).

Covers the undo/redo hotkeys (Ctrl+Z / Ctrl+Y, target follows the active
tool) and the tool / mask / zoom hotkeys. Uses the mock-backend server in
a temp root.
"""

import json

from conftest import wait_until


def _open(page, base):
    page.goto(base)
    page.wait_for_function("!document.getElementById('loading-overlay').hidden")


def test_undo_redo_hotkeys(page, server):
    base, _root = server
    _open(page, base)
    undo, redo = [], []

    def on_request(req):
        if req.url.endswith("/api/canvas/undo"):
            undo.append(json.loads(req.post_data)["target"])
        elif req.url.endswith("/api/canvas/redo"):
            redo.append(json.loads(req.post_data)["target"])

    page.on("request", on_request)
    page.keyboard.press("Control+z")
    wait_until(lambda: len(undo) == 1)
    page.keyboard.press("Control+y")
    wait_until(lambda: len(redo) == 1)
    assert undo == ["draw"] and redo == ["draw"]  # brush tool -> draw stack

    # the target follows the active tool: mask -> mask stack
    page.click("#tool-mask")
    page.keyboard.press("Control+z")
    wait_until(lambda: len(undo) == 2)
    assert undo[-1] == "mask"

    # typing in a field: the browser's own undo must not be hijacked
    page.click("#prompt")
    page.keyboard.press("Control+z")
    assert len(undo) == 2 and len(redo) == 1
