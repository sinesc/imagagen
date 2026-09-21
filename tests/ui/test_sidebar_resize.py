"""Browser tests for dragging the sidebar's right border and persisting
the resulting width in ui-state.json."""

from conftest import wait_until


def _open(page, base):
    page.goto(base)
    page.wait_for_function("brushNames.length >= 1")


def _width(page) -> float:
    return page.eval_on_selector("#controls", "e => e.getBoundingClientRect().width")


def _drag(page, dx: int):
    """Drag the resizer dx pixels (integer coords -> exact delta)."""
    box = page.locator("#sidebar-resizer").bounding_box()
    sx = round(box["x"]) + 2  # inside the 5px handle
    sy = round(box["y"]) + 50
    page.mouse.move(sx, sy)
    page.mouse.down()
    page.mouse.move(sx + dx, sy, steps=5)
    page.mouse.up()


def test_drag_border_resizes_sidebar(page, server):
    base, _ = server
    _open(page, base)
    assert _width(page) == 300

    _drag(page, 150)
    assert _width(page) == 450
    _drag(page, -50)
    assert _width(page) == 400


def test_width_is_clamped(page, server):
    base, _ = server
    _open(page, base)
    # drag far left -> clamped to the 280px minimum
    _drag(page, -280)
    assert _width(page) == 280

    # drag far right -> clamped to the 800px maximum
    _drag(page, 700)
    assert _width(page) == 800


def test_width_is_restored_from_ui_state(page, server):
    base, root = server
    _open(page, base)

    _drag(page, 70)
    assert _width(page) == 370

    # the debounced save lands in ui-state.json
    wait_until(lambda: (root / "user" / "ui-state.json").exists())
    wait_until(lambda: '"sidebar_width": 370' in
               (root / "user" / "ui-state.json").read_text())

    # a fresh page load restores the saved width
    page.goto(base)
    page.wait_for_function("brushNames.length >= 1")
    wait_until(lambda: _width(page) == 370)
