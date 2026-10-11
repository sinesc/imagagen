"""Browser tests for the per-thumbnail "reference image" checkbox (phase 0).

Layout (below the pin checkbox), the single-reference invariant, auto-pin
on check, the /api/generate payload, the caption preview tooltip, and
ui-state persistence across a reload. The mock backend echoes the caption
("mock"), so the preview round trip is exercised too.
"""

import json

from conftest import set_canvas, wait_until

BG = (20, 30, 40)


def _open(page, base):
    page.goto(base)
    assert page.is_visible("#tool-brush")
    page.wait_for_function("brushNames.length >= 1")


def _show(page, base, w=200, h=150):
    set_canvas(base, w, h, BG)
    page.evaluate("viewRefresh()")
    page.wait_for_function("document.getElementById('canvas').hidden === false")


def _generate(page, base):
    page.click("#btn-generate")
    page.wait_for_function("state.thumbs.length >= 2")  # source + result
    page.wait_for_function("!state.generating", timeout=15000)


def _rect(page, sel):
    return page.locator(sel).evaluate(
        "el => { const r = el.getBoundingClientRect(); "
        "return [r.left, r.top, r.width, r.height]; }"
    )


def test_reference_checkbox_sits_below_the_pin(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _generate(page, base)

    pins = page.locator("#thumbs .thumb-pin")
    refs = page.locator("#thumbs .thumb-ref")
    assert pins.count() == refs.count() == len(page.evaluate("state.thumbs"))

    for i in range(refs.count()):
        p = _rect(page, f"#thumbs .thumb-pin >> nth={i}")
        r = _rect(page, f"#thumbs .thumb-ref >> nth={i}")
        # stacked in the top-right corner: the reference below the pin,
        # right edges aligned, no overlap
        assert r[1] >= p[1] + p[3]
        assert abs((r[0] + r[2]) - (p[0] + p[2])) < 1.0


def test_single_reference_and_auto_pin(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _generate(page, base)

    # checking a thumbnail marks it as the reference and pins it
    # (unpinned thumbnails are dropped at submit; a reference must survive)
    page.click("#thumbs .thumb-ref >> nth=1")
    assert wait_until(
        lambda: page.evaluate("state.thumbs[1].pinned === true"))
    assert page.evaluate(
        "state.referenceImageId === state.thumbs[1].image_id")
    assert page.locator("#thumbs .thumb-pin >> nth=1").is_checked()

    # the caption preview (mock echo) lands in the tooltip
    assert wait_until(lambda: "mock" in page.evaluate(
        "document.querySelectorAll('#thumbs .thumb-ref')[1].title"))

    # checking a second thumbnail clears the first (single reference).
    # The handler re-renders asynchronously (behind the auto-pin await),
    # so poll the checkbox states rather than reading them right after
    # the click.
    page.click("#thumbs .thumb-ref >> nth=0")
    assert wait_until(lambda: page.evaluate(
        "state.referenceImageId === state.thumbs[0].image_id"))
    assert wait_until(lambda: not page.locator(
        "#thumbs .thumb-ref >> nth=1").is_checked())
    assert page.locator("#thumbs .thumb-ref >> nth=0").is_checked()

    # unchecking clears the selection entirely
    page.click("#thumbs .thumb-ref >> nth=0")
    assert wait_until(lambda: page.evaluate("state.referenceImageId === null"))


def test_generate_payload_carries_the_reference_id(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _generate(page, base)
    ref_id = page.evaluate("state.thumbs[1].image_id")
    page.click("#thumbs .thumb-ref >> nth=1")
    assert wait_until(
        lambda: page.evaluate("state.referenceImageId") == ref_id)

    with page.expect_request("**/api/generate") as req_info:
        page.click("#btn-generate")
    # the payload check uses the id captured BEFORE the second generate:
    # generate() prunes unpinned thumbnails synchronously, so reading
    # state.thumbs after the request is a race
    body = json.loads(req_info.value.post_data)
    assert body["reference_image_id"] == ref_id
    page.wait_for_function("!state.generating", timeout=15000)


def test_reference_selection_survives_a_reload(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _generate(page, base)
    page.click("#thumbs .thumb-ref >> nth=1")
    assert wait_until(
        lambda: page.evaluate("state.referenceImageId !== null"))
    ref_id = page.evaluate("state.referenceImageId")

    # persist, reload, and let the strip rebuild from /api/queue + ui-state
    page.evaluate("saveUiState()")  # fire-and-forget: poll the server
    assert wait_until(lambda: page.evaluate(
        "fetch('/api/ui-state').then(r=>r.json()).then("
        "j=>j.state.reference_image_id === '%s')" % ref_id))

    page.reload()
    assert page.is_visible("#tool-brush")
    page.wait_for_function("state.thumbs.length >= 1")
    assert page.evaluate("state.referenceImageId") == ref_id
    # the matching restored thumbnail shows the box checked
    assert page.evaluate(
        "Array.from(document.querySelectorAll('#thumbs .thumb-ref'))"
        ".some((c, i) => c.checked && state.thumbs[i].image_id === '%s')"
        % ref_id)
