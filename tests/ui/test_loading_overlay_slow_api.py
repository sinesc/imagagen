"""The loading overlay under a DELIBERATELY SLOW initial API response.

Split out of test_loading_overlay.py: the delayed route handler must not
block the (sync) connection loop, so these tests use Playwright's ASYNC
API. A live sync-Playwright browser -- conftest's shared one -- inside a
module that also runs async tests breaks pytest-asyncio's runner
("Runner.run() cannot be called from a running event loop"), so this file
must stay ASYNC-ONLY: do not add tests that use conftest's `page`.
"""

import asyncio

import pytest
from playwright.async_api import expect as aexpect
from playwright.sync_api import TimeoutError as PWTimeoutError


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
