"""The global "loading…" overlay covers the UI until the websocket connects.

The overlay ships *visible* in the markup (so it is up from the first paint,
before any JS runs) and app.js hides it on ws.onopen.
"""

import httpx

from playwright.sync_api import expect


def test_loading_overlay_visible_until_ws(server, page):
    base, _ = server

    # the overlay is in the markup without a hidden attribute
    html = httpx.get(base + "/", timeout=10).text
    assert 'id="loading-overlay"' in html
    assert 'id="loading-overlay" hidden' not in html

    # track the visible -> hidden transition inside the page itself (the
    # connection can finish before the test code gets to look at the overlay)
    page.add_init_script(
        """
        window.__overlayEverHidden = false;
        new MutationObserver(() => {
            const o = document.getElementById('loading-overlay');
            if (o && o.hidden) window.__overlayEverHidden = true;
        }).observe(document, {subtree: true, attributes: true, attributeFilter: ["hidden"]});
        """
    )
    page.goto(base)

    # hidden once the websocket is connected, and it got hidden (transition
    # observed, not absent from the start)
    expect(page.locator("#loading-overlay")).not_to_be_visible()
    assert page.evaluate("window.__overlayEverHidden") is True

    # stays hidden (no reconnect flapping)
    page.wait_for_timeout(300)
    expect(page.locator("#loading-overlay")).not_to_be_visible()
