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


def test_section_reference_fallback_button(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _generate(page, base)

    # a caption with no sections -> a single "Reference" tag toggling
    # the whole description (driven directly, like the spinner test)
    page.evaluate("state.referenceImageId = 'x:1';"
                  "state.referenceCaption = 'one flat paragraph';"
                  "renderRefSections();")
    assert page.evaluate(
        "document.querySelectorAll('#ref-sections .ref-tag').length") == 1
    assert page.evaluate(
        "document.querySelector('#ref-sections .ref-tag').textContent") == "Reference"
    assert page.evaluate(
        "document.querySelector('#ref-sections .ref-tag')"
        ".classList.contains('active')")
    # toggling off selects nothing ([] = the server applies no caption)
    page.click("#ref-sections .ref-tag")
    assert page.evaluate("state.referenceSections") == []
    assert not page.evaluate(
        "document.querySelector('#ref-sections .ref-tag')"
        ".classList.contains('active')")
    # toggling back restores the default (null = all)
    page.click("#ref-sections .ref-tag")
    assert page.evaluate("state.referenceSections") is None

    # an empty caption renders no buttons at all
    page.evaluate("state.referenceCaption = ''; renderRefSections();")
    assert page.evaluate("getComputedStyle("
                         "document.getElementById('ref-sections')).display") == "none"


def test_section_processing_spinner(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _generate(page, base)

    # checking a reference streams the captioner's phase messages to the
    # status line (the mock emits loading/describing/done back to back,
    # so the last one is what remains on the line)
    page.click("#thumbs .thumb-ref >> nth=1")
    assert wait_until(lambda: page.evaluate(
        "document.getElementById('status-line').textContent")
        == "Reference caption ready")

    # drive the placeholder directly (the mock preview resolves too fast
    # to catch through the real check flow): a checked reference with no
    # caption yet renders the animated placeholder
    page.evaluate("state.referenceImageId = 'x:1';"
                  "state.referenceCaption = null; renderRefSections();")
    txt = page.evaluate("document.querySelector('#ref-sections .ref-processing')"
                        ".textContent")
    assert txt.startswith("Processing")
    # the dots cycle "." -> ".." -> "..." every 500 ms
    seen = set()
    for _ in range(8):
        seen.add(page.evaluate(
            "document.querySelector('#ref-sections .ref-processing').textContent"))
        page.wait_for_timeout(200)
    assert seen == {"Processing.", "Processing..", "Processing..."}

    # resolving the caption replaces the placeholder with the tags and
    # stops the timer
    page.evaluate("state.referenceCaption = 'intro\\n\\n**Subject:** b';"
                  "renderRefSections();")
    assert page.evaluate(
        "document.querySelectorAll('#ref-sections .ref-tag').length") == 2
    assert page.evaluate(
        "document.querySelectorAll('#ref-sections .ref-processing').length") == 0

    # clearing the reference hides the row
    page.evaluate("state.referenceImageId = null; renderRefSections();")
    assert page.evaluate("getComputedStyle("
                         "document.getElementById('ref-sections')).display") == "none"


def test_section_tags_row(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _generate(page, base)

    # no reference: the tag row is hidden (and takes no layout space)
    assert page.evaluate("getComputedStyle("
                         "document.getElementById('ref-sections')).display") == "none"

    page.click("#thumbs .thumb-ref >> nth=1")
    # the mock caption carries Overview + Subject + Summary -> three tags,
    # all active by default
    assert wait_until(lambda: page.evaluate(
        "Array.from(document.querySelectorAll('#ref-sections .ref-tag'))"
        ".map(b => b.textContent).join(',')") == "Overview,Subject,Summary")
    assert page.evaluate(
        "Array.from(document.querySelectorAll('#ref-sections .ref-tag'))"
        ".every(b => b.classList.contains('active'))")
    assert not page.evaluate("document.getElementById('ref-sections')"
                             ".classList.contains('hidden')")

    # unchecking the reference hides the row again
    page.click("#thumbs .thumb-ref >> nth=1")
    assert wait_until(lambda: page.evaluate(
        "getComputedStyle(document.getElementById('ref-sections')).display") == "none")

    # re-check, then remove the reference's thumbnail (unpinning drops
    # the thumbnail): the row goes with it
    page.click("#thumbs .thumb-ref >> nth=1")
    assert wait_until(lambda: page.evaluate(
        "getComputedStyle(document.getElementById('ref-sections')).display") != "none")
    page.click("#thumbs .thumb-pin >> nth=1")
    assert wait_until(lambda: page.evaluate("state.referenceImageId") is None)
    assert wait_until(lambda: page.evaluate(
        "getComputedStyle(document.getElementById('ref-sections')).display") == "none")


def test_section_toggle_payload_and_reload(page, server):
    base, _ = server
    _open(page, base)
    _show(page, base)
    _generate(page, base)
    ref_id = page.evaluate("state.thumbs[1].image_id")
    page.click("#thumbs .thumb-ref >> nth=1")
    assert wait_until(lambda: page.evaluate(
        "document.querySelectorAll('#ref-sections .ref-tag').length") == 3)

    # toggle Subject off -> only Overview + Summary remain selected
    page.click("#ref-sections .ref-tag:nth-child(2)")
    assert page.evaluate("state.referenceSections") == ["Overview",
                                                        "Summary"]
    assert not page.evaluate(
        "document.querySelectorAll('#ref-sections .ref-tag')[1]"
        ".classList.contains('active')")

    # the generate payload carries the selection
    with page.expect_request("**/api/generate") as req_info:
        page.click("#btn-generate")
    body = json.loads(req_info.value.post_data)
    assert body["reference_image_id"] == ref_id
    assert body["reference_sections"] == ["Overview", "Summary"]
    page.wait_for_function("!state.generating", timeout=15000)

    # persist, reload: the selection is restored (the preview refetch
    # re-renders the row with Subject inactive)
    page.evaluate("saveUiState()")
    assert wait_until(lambda: page.evaluate(
        "fetch('/api/ui-state').then(r=>r.json()).then("
        "j=>JSON.stringify(j.state.reference_sections)"
        "==='[\"Overview\",\"Summary\"]')"))
    page.reload()
    assert page.is_visible("#tool-brush")
    assert page.evaluate("state.referenceSections") == ["Overview",
                                                        "Summary"]
    assert wait_until(lambda: page.evaluate(
        "document.querySelectorAll('#ref-sections .ref-tag').length") == 3)
    assert page.evaluate(
        "Array.from(document.querySelectorAll('#ref-sections .ref-tag'))"
        ".map(b => b.classList.contains('active'))") == [True, False, True]


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
