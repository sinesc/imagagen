"""Browser tests for the LoRA activation-tag toggles (the green buttons in
the tag row under the negative prompt).

A LoRA's comma-separated "optional activation text" (its settings .json,
edited in the "Edit LoRA" dialog) is split into tags: one toggle per tag of
the selected LoRAs, placed after the reference image's caption tags and split
off by a separator. All tags are on by default; only the enabled ones reach
the prompt, while a LoRA's plain "activation text" always does. Uses the
mock-backend server in a temp root.
"""

import json

from conftest import combo_pick, set_canvas, wait_until


def _seed_lora(root, name, tags, activation="", negative=""):
    d = root / "loras" / "sdxl"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.safetensors").write_bytes(b"")
    (d / f"{name}.json").write_text(json.dumps({
        "activation text": activation,
        "optional activation text": tags,
        "negative text": negative,
    }))


def _open(page, base, root):
    page.goto(base)
    page.wait_for_function(
        "state.loras.length >= 2 && Object.keys(state.loraSettings).length >= 2"
    )


def _add_lora(page, name):
    page.click("#lora-add")
    combo_pick(page, page.locator("#loras .lora-row").last, name)


def _tags(page):
    """[text, active] per LoRA activation tag button."""
    return page.evaluate(
        """() => Array.from(
              document.querySelectorAll('#ref-sections .lora-tag'))
            .map(b => [b.textContent, b.classList.contains('active')])"""
    )


def test_activation_tag_buttons_toggle_on_off(page, server):
    base, root = server
    # "sig style" is plain activation text: it always applies and never
    # becomes a toggle button
    _seed_lora(root, "alpha", "blue eyes, freckles", activation="sig style")
    _seed_lora(root, "beta", "freckles, 8k")
    _open(page, base, root)

    # no LoRA selected: the tag row carries nothing
    assert page.evaluate("getActivationTags()") == []
    assert page.evaluate("getComputedStyle("
                         "document.getElementById('ref-sections')).display") == "none"

    _add_lora(page, "alpha")
    # every optional tag of the selected LoRA shows up, enabled by default
    assert wait_until(lambda: _tags(page) == [["blue eyes", True],
                                              ["freckles", True]])
    assert page.evaluate("state.disabledActivationTags") == []
    assert "sig style" not in page.evaluate("getActivationTags()")

    # toggle "freckles" off: the button loses .active, the state records it
    page.click("#ref-sections .lora-tag:has-text('freckles')")
    assert _tags(page) == [["blue eyes", True], ["freckles", False]]
    assert page.evaluate("state.disabledActivationTags") == ["freckles"]
    # toggling it again clears the selection completely
    page.click("#ref-sections .lora-tag:has-text('freckles')")
    assert page.evaluate("state.disabledActivationTags") == []

    # a second LoRA adds its tags; a tag both LoRAs share stays ONE button
    _add_lora(page, "beta")
    assert _tags(page) == [["blue eyes", True], ["freckles", True],
                           ["8k", True]]

    # switching the LoRA row off takes its tags with it (its activation text
    # is not applied at all then)
    page.locator("#loras .lora-row").first.locator("button.lora-toggle").click()
    assert wait_until(lambda: _tags(page) == [["freckles", True],
                                              ["8k", True]])
    assert page.evaluate("state.disabledActivationTags") == []


def test_activation_tag_buttons_are_green_and_width_limited(page, server):
    base, root = server
    _seed_lora(root, "alpha", "blue eyes")
    _seed_lora(root, "beta", "a very long activation tag that would "
                             "need much more than ten em of width")
    _open(page, base, root)
    _add_lora(page, "alpha")
    _add_lora(page, "beta")

    style = page.evaluate(
        """() => {
            const b = document.querySelector('#ref-sections .lora-tag');
            const s = getComputedStyle(b);
            const [r, g, bl] = s.backgroundColor.match(/\\d+/g).map(Number);
            return {green: g > r && g > bl, inactive: s.backgroundColor,
                    maxWidth: parseFloat(s.maxWidth),
                    fontSize: parseFloat(s.fontSize)};
        }"""
    )
    assert style["green"], style  # enabled tags are green
    # max-width 10em of the button's own font size
    assert abs(style["maxWidth"] - 10 * style["fontSize"]) < 1.0, style

    # the long tag is cut to that width (ellipsised) but its title carries it
    long_tag = "a very long activation tag that would need much more than " \
               "ten em of width"
    box = page.evaluate(
        """(tag) => {
            const b = Array.from(document.querySelectorAll(
                '#ref-sections .lora-tag')).find(x => x.textContent === tag);
            return [b.scrollWidth > b.clientWidth, b.title.includes(tag)];
        }""", long_tag)
    assert box == [True, True]


def test_separator_splits_lora_tags_from_the_reference_tags(page, server):
    base, root = server
    _seed_lora(root, "alpha", "blue eyes")
    _seed_lora(root, "beta", "8k")
    _open(page, base, root)
    # a source image on the canvas: the generation then produces the source
    # thumbnail as well (the reference checkbox has to live on a result thumb)
    set_canvas(base)
    page.evaluate("viewRefresh()")
    page.wait_for_function(
        "document.getElementById('canvas').hidden === false")
    page.click("#btn-generate")
    page.wait_for_function("state.thumbs.length >= 2")
    page.wait_for_function("!state.generating", timeout=15000)

    # reference only: caption tags, no separator
    page.click("#thumbs .thumb-ref >> nth=1")
    assert wait_until(lambda: page.evaluate(
        "document.querySelectorAll('#ref-sections .ref-tag').length") == 3)
    assert page.evaluate(
        "document.querySelectorAll('#ref-sections .tag-sep').length") == 0
    assert page.evaluate(
        "document.querySelectorAll('#ref-sections .lora-tag').length") == 0

    # LoRA added: its tags follow the caption tags, split off by a separator
    _add_lora(page, "alpha")
    assert wait_until(lambda: page.evaluate(
        "document.querySelectorAll('#ref-sections .lora-tag').length") == 1)
    order = page.evaluate(
        """() => Array.from(document.getElementById('ref-sections').children)
            .map(e => e.classList.contains('ref-tag') ? 'ref'
                 : e.classList.contains('lora-tag') ? 'lora'
                 : e.classList.contains('tag-sep') ? 'sep' : 'other')""")
    assert order == ["ref", "ref", "ref", "sep", "lora"]


def test_generate_payload_and_reload_keep_the_selection(page, server):
    base, root = server
    _seed_lora(root, "alpha", "blue eyes, freckles")
    _seed_lora(root, "beta", "8k")
    _open(page, base, root)
    _add_lora(page, "alpha")
    _add_lora(page, "beta")
    page.click("#ref-sections .lora-tag:has-text('freckles')")
    assert page.evaluate("state.disabledActivationTags") == ["freckles"]

    with page.expect_request("**/api/generate") as req_info:
        page.click("#btn-generate")
    body = json.loads(req_info.value.post_data)
    assert body["disabled_activation_tags"] == ["freckles"]
    page.wait_for_function("!state.generating", timeout=15000)

    # persist, reload: the selection (and its button state) is restored
    page.evaluate("saveUiState()")
    assert wait_until(lambda: page.evaluate(
        "fetch('/api/ui-state').then(r=>r.json()).then("
        "j=>JSON.stringify(j.state.disabled_activation_tags)"
        "==='[\"freckles\"]')"))
    page.reload()
    assert page.is_visible("#tool-brush")
    assert page.evaluate("state.disabledActivationTags") == ["freckles"]
    assert wait_until(lambda: _tags(page) == [["blue eyes", True],
                                              ["freckles", False],
                                              ["8k", True]])
