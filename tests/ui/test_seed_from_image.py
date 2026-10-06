"""Browser tests for the Seed field: sidebar order (Steps before Seed), the
seed reset button and the copy-seed-from-selected-image button."""
from conftest import set_input
from test_canvas_ui import _open, _show


def test_seed_controls_and_copy_button(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base, 200, 150)

    # in the sidebar row, Steps sits left of Seed and the copy button is a
    # single-glyph mini button next to the seed field
    assert page.evaluate(
        """() => {
            const row = document.getElementById('seed').closest('.row');
            const a = row.querySelector('label:has(#steps)').offsetLeft;
            const b = row.querySelector('label:has(#seed)').offsetLeft;
            const btn = document.getElementById('seed-from-image');
            const reset = document.getElementById('seed-reset');
            return a < b && btn.classList.contains('mini')
                && btn.classList.contains('icon') && btn.closest('.row') === row
                && reset.classList.contains('mini') && reset.classList.contains('icon')
                && reset.closest('.row') === row;
        }"""
    )
    # and it sits in line with the inputs, not the Seed caption
    btn_box = page.locator("#seed-from-image").bounding_box()
    in_box = page.locator("#seed").bounding_box()
    assert abs(btn_box["y"] - in_box["y"]) < 2

    # the reset button sits between the seed field and the copy button, also
    # in line with the inputs, and sets the field back to -1
    reset_box = page.locator("#seed-reset").bounding_box()
    assert abs(reset_box["y"] - in_box["y"]) < 2
    assert in_box["x"] < reset_box["x"] < btn_box["x"]
    set_input(page, "#seed", "1234567890")
    page.click("#seed-reset")
    assert page.locator("#seed").input_value() == "-1"

    # the seed slot is a third wider than the old equal thirds (long seeds),
    # paid for by the CFG/steps slots (short values, still equal to each other)
    cfg_w = page.locator("#cfg").bounding_box()["width"]
    steps_w = page.locator("#steps").bounding_box()["width"]
    seed_w = in_box["width"]
    assert abs(cfg_w - steps_w) < 2
    assert 1.5 < seed_w / cfg_w < 1.7
    # the extra button must not overflow the sidebar row
    assert page.evaluate(
        """() => {
            const row = document.getElementById('seed').closest('.row');
            const last = document.getElementById('seed-from-image').getBoundingClientRect();
            return row.scrollWidth <= row.clientWidth + 1
                && last.right <= row.getBoundingClientRect().right + 1;
        }""")

    # generate with a fixed seed
    set_input(page, "#seed", "12345")
    page.click("#btn-generate")
    page.wait_for_function(
        "state.thumbs.some(t => t.source === undefined && t.seed === 12345)",
        timeout=15000,
    )

    # no image selected: the field is untouched
    page.evaluate("state.currentThumb = -1")
    page.click("#seed-from-image")
    assert page.locator("#seed").input_value() == "12345"

    # source thumbnail selected (no seed): the field is untouched
    page.locator("#thumbs img").first.click()
    page.wait_for_function("state.currentThumb === 0")
    page.click("#seed-from-image")
    assert page.locator("#seed").input_value() == "12345"

    # a result selected: its seed is copied into the field
    page.locator("#thumbs img").nth(1).click()
    page.wait_for_function("state.thumbs[state.currentThumb].seed === 12345")
    page.click("#seed-from-image")
    assert page.locator("#seed").input_value() == "12345"
