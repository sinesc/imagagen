"""Regression: picking a combobox item on a slow press (button held down
longer than the 120 ms close-on-blur timer). The mousedown on the item
blurs the input, which closes (hides) the list after 120 ms; a click-based
pick then lost its target on the mouseup and the old value remained.
Items are picked on mousedown, so the press alone applies the value."""

import json

from conftest import combo_open, combo_pick


def _seed_presets(root):
    # preset files are named by the *sanitized* (lowercased) name
    for name in ("Alpha", "Beta"):
        p = root / "user" / "presets" / f"{name.lower()}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"name": name, "controls": {}}), "utf-8")


def _slow_press(page, item):
    """Mousedown on `item`, hold the button 300 ms, then release."""
    box = item.bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.wait_for_timeout(300)
    page.mouse.up()


def test_slow_press_picks_value(page, server):
    base, root = server
    _seed_presets(root)
    page.goto(base)
    page.wait_for_function("brushNames.length >= 1 && state.presets.length === 2")

    # a press held down well past the 120 ms blur-close timer must still
    # apply the value (this is the failure mode that used to keep the old
    # value selected)
    combo_pick(page, page.locator("#preset"), "Alpha")
    assert page.evaluate("document.getElementById('preset').value") == "Alpha"
    combo_open(page, page.locator("#preset"))
    _slow_press(page, page.locator("#preset .combo-item[data-value='Beta']"))
    assert page.evaluate("document.getElementById('preset').value") == "Beta"
    assert page.locator("#preset .combo-list.open").count() == 0


def test_quick_press_still_picks(page, server):
    base, root = server
    _seed_presets(root)
    page.goto(base)
    page.wait_for_function("brushNames.length >= 1 && state.presets.length === 2")
    combo_pick(page, page.locator("#preset"), "Beta")
    assert page.evaluate("document.getElementById('preset').value") == "Beta"


def test_right_click_does_not_pick(page, server):
    base, root = server
    _seed_presets(root)
    page.goto(base)
    page.wait_for_function("brushNames.length >= 1 && state.presets.length === 2")
    combo_pick(page, page.locator("#preset"), "Alpha")
    combo_open(page, page.locator("#preset"))
    box = page.locator("#preset .combo-item[data-value='Beta']").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.mouse.down(button="right")
    page.mouse.up(button="right")
    assert page.evaluate("document.getElementById('preset').value") == "Alpha"
