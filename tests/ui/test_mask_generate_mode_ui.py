from conftest import dot, mask_arr, wait_until
from test_canvas_ui import BG, _mask_alpha, _open, _screen_px, _show


def test_mask_stroke_allowed_in_generate(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base, 100, 80)  # default mode: generate

    # the mask tool is usable in "generate" mode: the stroke arms, draws,
    # and commits like in the inpaint modes
    page.keyboard.press("m")
    assert page.evaluate("Tools.current") == "mask"
    dot(page, 50, 40)
    assert wait_until(lambda: _mask_alpha(page, 50, 40) == 255)
    assert wait_until(
        lambda: mask_arr(base) is not None
        and mask_arr(base)[40, 50] == 255)

    # the committed overlay renders in generate mode too (grey 128 at 0.9
    # alpha over BG -> ~(117,118,119) at the mask centre)
    def overlay_shown():
        r, g, b = _screen_px(page, 50, 40)
        return abs(r - 117) < 12 and abs(g - 118) < 12 and abs(b - 119) < 12
    assert wait_until(overlay_shown), \
        f"overlay not rendered in generate mode: {_screen_px(page, 50, 40)}"

    # the tool-gated hiding is unchanged: switching to the brush tool hides
    # the overlay (visual only), while the bbox outline stays drawn
    page.keyboard.press("d")
    assert wait_until(lambda: _screen_px(page, 50, 40) == list(BG)), \
        f"overlay not hidden outside mask mode: {_screen_px(page, 50, 40)}"
    assert page.evaluate("maskOutlineBox()") is not None
    assert _mask_alpha(page, 50, 40) == 255  # mask data untouched
