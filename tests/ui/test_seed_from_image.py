"""Browser tests for the Seed field: sidebar order (Steps before Seed) and
the copy-seed-from-selected-image button."""
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
            return a < b && btn.classList.contains('mini')
                && btn.classList.contains('icon') && btn.closest('.row') === row;
        }"""
    )
    # and it sits in line with the inputs, not the Seed caption
    btn_box = page.locator("#seed-from-image").bounding_box()
    in_box = page.locator("#seed").bounding_box()
    assert abs(btn_box["y"] - in_box["y"]) < 2

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
