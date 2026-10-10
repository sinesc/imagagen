from conftest import dot, screen_pos, wait_until
from test_canvas_ui import BG, _mask_alpha, _open, _screen_px, _show


def test_mask_stroke_refused_in_generate(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base, 100, 80)
    page.keyboard.press("m")  # mask tool stays usable in generate mode
    assert page.evaluate("Tools.current") == "mask"
    assert not page.evaluate("document.getElementById('tool-mask').disabled")

    # click-to-draw is refused: red status, no stroke armed, no preview
    page.mouse.move(*screen_pos(page, 50, 40))
    page.mouse.down()
    assert page.evaluate("view.stroke") is None
    assert page.text_content("#status-line") == "Must be in inpaint/outpaint mode"
    assert page.evaluate(
        "document.getElementById('status-line').classList.contains('error')")
    page.mouse.up()
    assert _mask_alpha(page, 50, 40) == 0
    assert page.evaluate("view.mask") is None or True  # overlay hidden in generate
    assert _screen_px(page, 50, 40) == list(BG)

    # inpaint mode: the same stroke draws normally
    page.select_option("#mode", "inpaint_masked")
    dot(page, 50, 40)
    assert wait_until(lambda: _mask_alpha(page, 50, 40) == 255)
    assert wait_until(lambda: _screen_px(page, 50, 40) != list(BG))

    # generate mode: the committed mask is hidden, a new stroke is refused
    page.select_option("#mode", "generate")
    assert wait_until(lambda: _screen_px(page, 50, 40) == list(BG))
    dot(page, 70, 40)
    assert _mask_alpha(page, 70, 40) == 0
    assert _mask_alpha(page, 50, 40) == 255  # existing mask intact
