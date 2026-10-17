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
"""

import asyncio

import httpx
import pytest
from playwright.async_api import expect as aexpect
from playwright.sync_api import TimeoutError as PWTimeoutError
from playwright.sync_api import expect


@pytest.fixture()
async def apage():
    from playwright.async_api import async_playwright

    pw = await async_playwright().start()
    try:
        browser = await pw.chromium.launch()
        yield await browser.new_page(viewport={"width": 1280, "height": 800})
        await browser.close()
    finally:
        await pw.stop()


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



async def _hold_route(page, pattern, delay: float):
    """Route `pattern` through a delayed continue so its response settles
    after the websocket (the async API's handlers are coroutines, so the
    sleep does not block the connection loop)."""
    async def hold(route):
        await asyncio.sleep(delay)
        await route.continue_()

    await page.route(pattern, hold)


async def test_loading_overlay_waits_for_slow_api(server, apage):
    """The overlay stays up while an initial API response is slow, even if
    the websocket is already connected, and drops once it settles."""
    base, _ = server
    await _hold_route(apage, "**/api/presets", 1.5)
    await apage.goto(base)
    # the websocket is up ...
    await apage.wait_for_function("ws.readyState === 1")
    # ... but the overlay is still up (presets have not settled)
    await aexpect(apage.locator("#loading-overlay")).to_be_visible()
    # ... and hides once everything settles
    await aexpect(apage.locator("#loading-overlay")).not_to_be_visible(timeout=15000)


async def test_loading_overlay_blocks_input_until_settled(server, apage):
    """Typing is blocked while the overlay is up (the load would clobber it)
    and works once it has settled."""
    base, _ = server
    await _hold_route(apage, "**/api/presets", 1.5)
    await apage.goto(base)
    await apage.wait_for_function("ws.readyState === 1")

    # the overlay eats pointer events: clicking the prompt times out
    prompt = apage.locator("#prompt")
    with pytest.raises(PWTimeoutError):  # the overlay eats pointer events
        await prompt.click(timeout=500)

    # Tab/programmatic focus can still reach the field behind the overlay,
    # but the capture-phase keydown guard blocks the typing
    await prompt.focus()
    await apage.keyboard.type("a photo of a frog")
    assert await prompt.input_value() == ""

    # settled: typing works now
    await aexpect(apage.locator("#loading-overlay")).not_to_be_visible(timeout=15000)
    await apage.keyboard.type("x")
    assert await prompt.input_value() == "x"
