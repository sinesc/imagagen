"""Canvas border highlights with the focus color while tool hotkeys are live."""


def _border(page):
    return page.evaluate(
        "getComputedStyle(document.getElementById('canvas')).borderTopColor"
    )


def test_hotkey_border(page, server):
    base, _root = server
    page.goto(base)
    page.wait_for_function("document.getElementById('loading-overlay').hidden")
    # initial: body focused -> hotkeys live -> focus color #777
    page.wait_for_function(
        "document.getElementById('canvas').classList.contains('hotkeys')"
    )
    assert _border(page) == "rgb(119, 119, 119)"
    # focusing a text field disables the hotkeys -> plain #333
    page.locator("#seed").click()
    page.wait_for_function(
        "!document.getElementById('canvas').classList.contains('hotkeys')"
    )
    assert _border(page) == "rgb(51, 51, 51)"
    # blur back to the document -> hotkeys live again
    page.evaluate("document.getElementById('seed').blur()")
    page.wait_for_function(
        "document.getElementById('canvas').classList.contains('hotkeys')"
    )
    assert _border(page) == "rgb(119, 119, 119)"
