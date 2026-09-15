"""Browser tests for the self-attention-guidance row (Forge builtin
extension): checkbox left + options gear right in the row above the
Generate/Save buttons (below the soft-inpainting row, always enabled --
SAG works in all modes), and the three-slider options dialog with
persistence.
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


def test_sag_row_layout_and_always_enabled(page, server):
    base, _ = server
    _open(page, base)

    # the row sits between the soft-inpainting row and the Generate/Save
    # buttons, with the checkbox first and the gear last (right-aligned)
    rows = page.locator("#controls > .row")
    classes = [rows.nth(i).get_attribute("class") or "" for i in range(rows.count())]
    assert "actions" in classes[-1]
    assert "sag-row" in classes[-2]
    assert "soft-row" in classes[-3]
    row = page.locator(".sag-row")
    first = row.locator("> :first-child")
    assert first.evaluate("e => e.querySelector('input')?.id") == "sag"
    assert row.locator("> :last-child").get_attribute("id") == "sag-cfg"
    # the label sits to the RIGHT of the checkbox (one line, not stacked)
    box = page.locator("#sag").bounding_box()
    lab = row.locator(".soft-check span").bounding_box()
    assert abs(lab["y"] - box["y"]) < 5
    assert lab["x"] > box["x"] + box["width"]

    # unlike soft inpainting, SAG works in all modes: never disabled
    assert not page.locator("#sag").is_checked()
    assert not page.locator("#sag").is_disabled()
    page.select_option("#mode", "inpaint_masked")
    assert not page.locator("#sag").is_disabled()
    page.select_option("#mode", "inpaint_not_masked")
    assert not page.locator("#sag").is_disabled()
    page.select_option("#mode", "generate")
    assert not page.locator("#sag").is_disabled()


def test_sag_options_dialog(page, server):
    base, _ = server
    _open(page, base)
    # the gear opens the options dialog with the three Forge sliders
    page.click("#sag-cfg")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert "Self attention guidance" in page.locator("#dialog-title").inner_text()
    sliders = page.locator("#dialog-body input[type=range]")
    assert sliders.count() == 3
    # Forge slider ranges/steps
    expect = [
        ("-2", "5", "0.01"),    # scale
        ("0", "10", "0.01"),    # blur sigma
        ("0", "4", "0.01"),     # blur mask threshold
    ]
    for i, (mn, mx, step) in enumerate(expect):
        s = sliders.nth(i)
        assert s.get_attribute("min") == mn
        assert s.get_attribute("max") == mx
        assert s.get_attribute("step") == step
    # the Forge defaults
    for i, d in enumerate(["0.5", "2", "1"]):
        assert sliders.nth(i).input_value() == d

    # changing a slider updates its live value; saving persists to ui-state
    page.evaluate("(() => { const i = document.querySelectorAll('#dialog-body input[type=range]')[0]; "
                  "i.value = '1.25'; "
                  "i.dispatchEvent(new Event('input', { bubbles: true })); })()")
    assert sliders.nth(0).input_value() == "1.25"
    page.click("#dialog-ok")
    assert wait_until(lambda: not _dialog_visible(page), timeout=3)
    assert wait_until(lambda: _ui_state_key(page, "sag_scale") == 1.25, timeout=5)

    # the dialog reopens at the persisted value
    page.click("#sag-cfg")
    assert wait_until(lambda: _dialog_visible(page), timeout=3)
    assert page.locator("#dialog-body input[type=range]").first.input_value() == "1.25"


def test_sag_checkbox_and_options_persist_across_reload(page, server):
    base, _ = server
    _open(page, base)
    page.check("#sag")
    page.evaluate("saveUiState()")
    assert wait_until(lambda: _ui_state_key(page, "sag") is True)
    page.reload()
    page.wait_for_function("brushNames.length >= 1")
    # (applyUiState runs when the ui-state fetch resolves; poll for it)
    assert wait_until(lambda: page.locator("#sag").is_checked())
    # SAG stays enabled in "generate" mode (no mode gating)
    assert not page.locator("#sag").is_disabled()
