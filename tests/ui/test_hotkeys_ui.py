"""Browser tests for the toolbar keyboard shortcuts (Chromium via
Playwright).

Covers the undo/redo hotkeys (Ctrl+Z / Ctrl+Y, target follows the active
tool), the tool / mask / zoom hotkeys, and the save (Ctrl+S) / generate
(Ctrl+G) hotkeys. Uses the mock-backend server in a temp root.
"""

from conftest import dot, set_canvas, set_input, wait_until


def _open(page, base):
    page.goto(base)
    # .hidden === true means the overlay is GONE (boot settled)
    page.wait_for_function("document.getElementById('loading-overlay').hidden === true")


def test_undo_redo_hotkeys(page, server):
    # The undo/redo hotkeys act on the client's local stack (no server
    # round trip): Ctrl+Z reverts the painted pixels, Ctrl+Y brings them
    # back, and the target follows the active tool (mask -> mask stack)
    BG = (20, 30, 40)  # set_canvas default
    base, _ = server
    _open(page, base)
    set_canvas(base)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")
    set_input(page, "#brush-color", "#ff0000")

    px = """([x, y]) => {
        const c = document.getElementById('canvas');
        const d = c.getContext('2d').getImageData(
            Math.round((view.ox + x * view.scale) * view.dpr),
            Math.round((view.oy + y * view.scale) * view.dpr), 1, 1).data;
        return [d[0], d[1], d[2]];
    }"""

    dot(page, 100, 75)
    assert wait_until(lambda: page.evaluate(px, [100, 75]) == [255, 0, 0])

    page.keyboard.press("Control+z")  # brush tool -> draw stack
    assert wait_until(lambda: page.evaluate(px, [100, 75]) == list(BG))
    page.keyboard.press("Control+y")
    assert wait_until(lambda: page.evaluate(px, [100, 75]) == [255, 0, 0])

    # the target follows the active tool: mask -> mask stack
    page.click("#tool-mask")
    dot(page, 60, 40)
    m = ("([x, y]) => view.maskValues.getContext('2d', "
         "{willReadFrequently: true}).getImageData(x, y, 1, 1).data[3]")
    assert wait_until(lambda: page.evaluate(m, [60, 40]) == 255)
    page.keyboard.press("Control+z")
    assert wait_until(lambda: page.evaluate(m, [60, 40]) == 0)
    page.keyboard.press("Control+y")
    assert wait_until(lambda: page.evaluate(m, [60, 40]) == 255)
    # ...and the draw pixels are untouched by the mask stack
    assert page.evaluate(px, [100, 75]) == [255, 0, 0]

    # typing in a field: the browser's own undo must not be hijacked
    page.click("#prompt")
    page.keyboard.press("Control+z")
    assert page.evaluate(px, [100, 75]) == [255, 0, 0]  # unchanged


def test_tool_and_view_hotkeys(page, server):
    base, _root = server
    _open(page, base)

    for key, tool in {"d": "brush", "s": "smudge", "c": "clone",
                      "m": "mask", "p": "crop"}.items():
        page.keyboard.press(key)
        assert page.evaluate("Tools.current") == tool

    # mask ops: the hotkeys fire the same client-side ops as the buttons
    # (no server round trip any more); without a canvas they are no-ops
    for key in ("n", "k", "r"):
        page.keyboard.press(key)
    assert page.evaluate("view.has") is False

    # f fits: a no-op without an image, but must not throw
    page.keyboard.press("f")
    assert page.evaluate("view.has") is False

    # typing in a field: none of the hotkeys fire
    page.click("#prompt")
    page.keyboard.press("m")
    assert page.evaluate("Tools.current") == "crop"


def test_save_and_generate_hotkeys(page, server):
    base, root = server
    _open(page, base)
    set_canvas(base)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")

    # Ctrl+G starts generation: a job gets queued on the mock backend and
    # the status line reflects it
    page.keyboard.press("Control+g")
    assert wait_until(lambda: page.evaluate("state.job") is not None)
    assert wait_until(lambda: page.evaluate("state.generating") is False)

    # Ctrl+S saves the server's canvas copy into the output directory
    page.keyboard.press("Control+s")
    status = page.locator("#status-line")
    assert wait_until(lambda: status.evaluate("e => e.textContent").startswith("saved "))
    assert list((root / "output").glob("*.png"))


def test_hotkey_buttons_show_their_character_underlined(page, server):
    base, _root = server
    _open(page, base)
    # the hotkey character is part of the button text and underlined
    for sel, text in [
        ("#tool-brush", "draw"),
        ("#tool-smudge", "smudge"),
        ("#tool-clone", "clone"),
        ("#tool-mask", "mask"),
        ("#tool-crop", "crop"),
        ("#mask-dilate", "expand"),
        ("#mask-erode", "shrink"),
        ("#mask-clear", "clear"),
        ("#zoom-fit", "fit"),
    ]:
        btn = page.locator(sel)
        assert btn.evaluate("e => e.querySelector('u').textContent").isalpha()
        assert btn.evaluate("e => e.textContent") == text
