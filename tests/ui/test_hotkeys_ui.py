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
    # .hidden === true means the overlay is GONE (boot settled)
    page.wait_for_function("document.getElementById('loading-overlay').hidden === true")


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


def test_tool_and_view_hotkeys(page, server):
    base, _root = server
    _open(page, base)

    for key, tool in {"d": "brush", "s": "smudge", "c": "clone",
                      "m": "mask", "p": "crop"}.items():
        page.keyboard.press(key)
        assert page.evaluate("Tools.current") == tool

    # mask ops: the hotkeys fire the same endpoints as the buttons
    ops = []

    def on_request(req):
        if req.url.endswith("/api/canvas/mask"):
            ops.append(json.loads(req.post_data).get("op"))

    page.on("request", on_request)
    for key in ("n", "k", "r"):
        page.keyboard.press(key)
    wait_until(lambda: len(ops) == 3)
    assert ops == ["dilate", "erode", "clear"]

    # f fits: a no-op without an image, but must not throw
    page.keyboard.press("f")
    assert page.evaluate("view.has") is False

    # typing in a field: none of the hotkeys fire
    page.click("#prompt")
    page.keyboard.press("m")
    assert page.evaluate("Tools.current") == "crop"


def test_hotkey_buttons_show_their_character_underlined(page, server):
    base, _root = server
    _open(page, base)
    # the hotkey character is part of the button text and underlined
    for sel, text in [
        ("#tool-brush", "draw"),
        ("#tool-smudge", "smudge"),
        ("#tool-clone", "clone"),
        ("#tool-mask", "mask"),
        ("#tool-crop", "crop p"),
        ("#mask-dilate", "expand"),
        ("#mask-erode", "shrink k"),
        ("#mask-clear", "clear mask"),
        ("#zoom-fit", "fit"),
    ]:
        btn = page.locator(sel)
        assert btn.evaluate("e => e.querySelector('u').textContent").isalpha()
        assert btn.evaluate("e => e.textContent") == text
