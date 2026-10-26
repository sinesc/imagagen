"""The global "loading…" overlay covers the UI until it is settled.

"Settled" means: all initial requests whose responses rewrite the form
have arrived (init: models/presets/ui-state/..., brushes, canvas). The
websocket is deliberately NOT part of the gate (its upgrade can lag the
HTTP requests well behind, e.g. when the browser throttles queued
reconnect attempts after the server was offline). The overlay ships
*visible* in the markup (so it is up from the first paint, before any JS
runs) and app.js hides it once the boot Promise.all settles. While it is
up it must also block input, so changes typed before the state load
arrives cannot be clobbered by it.

The tests that need a DELIBERATELY SLOW initial API response live in
test_loading_overlay_slow_api.py: their route handler must not block, so
they use Playwright's ASYNC API -- and a live sync-Playwright browser
(conftest's shared one) inside the same module breaks pytest-asyncio's
runner ("Runner.run() cannot be called from a running event loop"). Keep
that file async-only.
"""

import httpx
from playwright.sync_api import expect


def test_loading_overlay_visible_until_settled(server, page):
    base, _ = server

    # the overlay is in the markup without a hidden attribute
    html = httpx.get(base + "/", timeout=10).text
    assert 'id="loading-overlay"' in html
    assert 'id="loading-overlay" hidden' not in html

    # track the visible -> hidden transition inside the page itself (the
    # load can finish before the test code gets to look at the overlay)
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

    # hidden once the UI is settled, and it got hidden (transition
    # observed, not absent from the start)
    expect(page.locator("#loading-overlay")).not_to_be_visible()
    assert page.evaluate("window.__overlayEverHidden") is True

    # stays hidden (no reconnect flapping)
    page.wait_for_timeout(300)
    expect(page.locator("#loading-overlay")).not_to_be_visible()
