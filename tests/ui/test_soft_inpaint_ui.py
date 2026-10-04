"""Browser tests for the soft-inpainting row (Forge builtin extension):
checkbox left + options gear right in the row above the Generate/Save
buttons, mode gating, and the six-slider options dialog with persistence.
"""

from conftest import wait_until


def _open(page, base):
    page.goto(base)
    assert page.is_visible("#tool-brush")
    page.wait_for_function("brushNames.length >= 1")


def _dialog_visible(page) -> bool:
    return page.evaluate("document.getElementById('dialog') !== null && "
                         "!document.getElementById('dialog').hidden")


def _ui_state_key(page, key: str):
    return page.evaluate(
        "fetch('/api/ui-state').then(r => r.json())"
        ".then(j => j.state['" + key + "'])"
    )


def test_soft_row_layout_and_mode_gating(page, server):
    base, _ = server
    _open(page, base)

    # the row sits directly above the Generate/Save buttons, with the
    # checkbox first and the gear last (right-aligned). (Self attention
    # guidance has no main-form row: its enable flag and options live in
    # the model configuration.)
    rows = page.locator("#controls > .row")
    classes = [rows.nth(i).get_attribute("class") or "" for i in range(rows.count())]
    assert "actions" in classes[-1]
    assert "soft-row" in classes[-2]
    row = page.locator(".soft-row")
    # checkbox (left) first, gear (right) last
    first = row.locator("> :first-child")
    assert first.evaluate("e => e.querySelector('input')?.id") == "soft-inpaint"
    assert row.locator("> :last-child").get_attribute("id") == "soft-inpaint-cfg"
    # the label sits to the RIGHT of the checkbox (one line, not stacked):
    # the sidebar's #controls label rule is flex-column, so this is a real
    # layout regression if it ever breaks
    box = page.locator("#soft-inpaint").bounding_box()
    lab = row.locator(".soft-check span").bounding_box()
    assert abs(lab["y"] - box["y"]) < 5
    assert lab["x"] > box["x"] + box["width"]

    # off by default; disabled in "generate" mode, enabled in inpaint modes;
    # the label text dims (#aaa) while the checkbox is disabled
    lab = page.locator(".soft-row .soft-check")
    assert not page.locator("#soft-inpaint").is_checked()
    assert page.locator("#soft-inpaint").is_disabled()
    assert lab.evaluate("e => getComputedStyle(e).color") == "rgb(170, 170, 170)"
    page.select_option("#mode", "inpaint_masked")
    assert not page.locator("#soft-inpaint").is_disabled()
    assert lab.evaluate("e => getComputedStyle(e).color") == "rgb(204, 204, 204)"
    page.select_option("#mode", "inpaint_not_masked")
    assert not page.locator("#soft-inpaint").is_disabled()
    page.select_option("#mode", "generate")
    assert page.locator("#soft-inpaint").is_disabled()
    assert lab.evaluate("e => getComputedStyle(e).color") == "rgb(170, 170, 170)"


def test_soft_options_dialog(page, server):
    base, _ = server
    _open(page, base)
    # the gear opens the options dialog with the six Forge sliders
    page.click("#soft-inpaint-cfg")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert "Soft inpainting" in page.locator("#dialog-title").inner_text()
    sliders = page.locator("#dialog-body input[type=range]")
    assert sliders.count() == 6
    # Forge slider ranges/steps
    expect = [
        ("0", "8", "0.1"),       # schedule bias
        ("0", "8", "0.05"),      # preservation strength
        ("1", "32", "0.5"),      # transition contrast boost
        ("0", "1", "0.05"),      # mask influence
        ("0", "8", "0.25"),      # difference threshold
        ("0", "8", "0.25"),      # difference contrast
    ]
    for i, (mn, mx, step) in enumerate(expect):
        s = sliders.nth(i)
        assert s.get_attribute("min") == mn
        assert s.get_attribute("max") == mx
        assert s.get_attribute("step") == step
    # the Forge defaults
    for i, d in enumerate(["1", "0.5", "4", "0", "0.5", "2"]):
        assert sliders.nth(i).input_value() == d
    # the pixel composite section heading divides the two groups
    assert "Pixel composite settings" in page.locator("#dialog-body").inner_text()

    # changing a slider updates its live value; saving persists to ui-state
    # (range inputs only accept integer fill values, so set a decimal via the
    # real input event)
    page.evaluate("(() => { const i = document.querySelectorAll('#dialog-body input[type=range]')[0]; "
                  "i.value = '2.2'; "
                  "i.dispatchEvent(new Event('input', { bubbles: true })); })()")
    assert sliders.nth(0).input_value() == "2.2"
    page.click("#dialog-ok")
    assert wait_until(lambda: not _dialog_visible(page), timeout=3)
    assert wait_until(lambda: _ui_state_key(page, "soft_mask_blend_power") == 2.2,
                      timeout=5)

    # the dialog reopens at the persisted value
    page.click("#soft-inpaint-cfg")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert page.locator("#dialog-body input[type=range]").first.input_value() == "2.2"


def test_soft_checkbox_persists_but_mode_comes_back_generate(page, server):
    base, _ = server
    _open(page, base)
    page.select_option("#mode", "inpaint_masked")
    page.check("#soft-inpaint")
    page.evaluate("saveUiState()")
    assert wait_until(lambda: _ui_state_key(page, "soft_inpainting") is True)
    page.reload()
    page.wait_for_function("brushNames.length >= 1")
    # the mode always comes back "generate" (not persisted, there is
    # nothing to inpaint until the user has an image); the checkbox value
    # is persisted but stays disabled in "generate" mode
    assert page.locator("#mode").input_value() == "generate"
    # (applyUiState runs when the ui-state fetch resolves; poll for it)
    assert wait_until(lambda: page.locator("#soft-inpaint").is_checked())
    assert page.locator("#soft-inpaint").is_disabled()
    page.select_option("#mode", "inpaint_masked")
    assert page.locator("#soft-inpaint").is_checked()
    assert not page.locator("#soft-inpaint").is_disabled()
