"""Placeholder text in the empty thumbnail strip.

While the strip is empty it shows a centred muted-grey hint (styled like
the #canvas-empty message). The hint must vanish once a thumbnail exists
and must not swallow drops onto the strip."""

import pytest  # noqa: F401  (fixtures come via conftest)
from conftest import wait_until
from test_canvas_ui import BG, _drop_png, _meta_png, _open


def _before(page, prop):
    return page.evaluate(
        "(p) => getComputedStyle(document.getElementById('thumbs'),"
        "'::before')[p]",
        prop,
    )


def test_empty_thumbs_strip_shows_placeholder(page, server):
    base, _ = server
    _open(page, base)
    assert page.locator("#thumbs").evaluate("el => el.classList.contains('empty')")
    text = _before(page, "content")
    assert "No thumbnails yet" in text
    assert "Drop PNGs here to use as source or reference images" in text
    # same muted styling as #canvas-empty (.muted: #888 / 0.8rem, centred)
    assert _before(page, "color") == "rgb(136, 136, 136)"
    assert _before(page, "fontSize") == "12.8px"
    assert _before(page, "textAlign") in ("center", "center, center")
    # the hint is purely decorative: the strip below stays the dropzone
    assert _before(page, "pointerEvents") == "none"


def test_placeholder_gone_after_drop(page, server):
    base, _ = server
    _open(page, base)
    _drop_png(page, "#thumbs", _meta_png(120, 90, rgb=BG), "pin.png")
    assert wait_until(lambda: page.locator("#thumbs img").count() == 1)
    assert not page.locator("#thumbs").evaluate("el => el.classList.contains('empty')")
    assert _before(page, "content") == "none"
