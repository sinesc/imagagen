"""Browser tests for the styles UI (Chromium via Playwright).

Covers the style dropdown rows (+ / − buttons, optgroup sorting), the
style configuration dialog (create / edit / delete), ui-state restore,
the styles sent on generate, and the "Session" temporary styles created
by PNG metadata import. Uses the mock-backend server in a temp root.
"""

import httpx
from conftest import combo_pick, combo_seq, wait_until

CSV = "name,prompt,negative_prompt,optgroup"
SEED_ROWS = "\n".join(
    [
        CSV,
        'Cinematic,"cinematic film still {prompt} . moody",,"Photo Styles"',
        'Dunes,"in the dunes, white sand",,Background',
        'Negs,,"digital art, cartoon",Negatives',
        "Plain,plain text,,",
    ]
)


def _open(page, base, root, seed=True):
    if seed:
        (root / "user" / "styles.csv").write_text(SEED_ROWS + "\n", "utf-8")
    page.goto(base)
    page.wait_for_function("brushNames.length >= 1 && state.styles.length > 0")


def _options(page):
    """The (group, item text) pairs of the first style row's dropdown,
    mirroring the old (optgroup label, option text) shape: a group header
    is ['\x00' + group, None], an item is [its group, text]."""
    g = ""
    out = []
    for kind, text in combo_seq(page, page.locator("#styles .style-row").first):
        if kind == "group":
            g = text
            out.append(["\x00" + g, None])
        else:
            out.append([g, text])
    return out


# ---------------- rows ----------------


def test_style_add_remove_and_groups(page, server):
    base, root = server
    _open(page, base, root)

    # starts empty
    assert page.locator("#styles .style-row").count() == 0

    # "+ Style" adds a dropdown row
    page.click("#style-add")
    row = page.locator("#styles .style-row")
    assert row.count() == 1
    # dropdown is grouped by optgroup (sorted), names sorted inside,
    # ungrouped styles come first as plain options
    assert _options(page) == [
        ["", "Plain"],
        ["\x00Background", None],
        ["Background", "Dunes"],
        ["\x00Negatives", None],
        ["Negatives", "Negs"],
        ["\x00Photo Styles", None],
        ["Photo Styles", "Cinematic"],
    ]

    # each row keeps its own selection; a second row starts unselected
    combo_pick(page, row.first, "Cinematic")
    page.click("#style-add")
    assert row.count() == 2
    combo_pick(page, row.nth(1), "Dunes")
    assert row.nth(0).locator(".style-name").evaluate("e => e.value") == "Cinematic"

    # − removes its own row only (the remaining row is the second one)
    row.first.locator("button.style-rm").click()
    assert row.count() == 1
    assert row.first.locator(".style-name").evaluate("e => e.value") == "Dunes"
    # the surviving selection is read back as names for generate
    assert page.evaluate("getStyles()") == ["Dunes"]

    # the "+ Style" button stretches so the gear's right edge aligns with
    # the rows' remove (−) buttons
    geom = page.evaluate(
        """() => {
            const right = (e) => e.getBoundingClientRect().right;
            return {
                minus: right(document.querySelector('#styles .style-row button.style-rm')),
                gear: right(document.getElementById('style-manage')),
            };
        }"""
    )
    assert abs(geom["gear"] - geom["minus"]) < 1


def test_style_selection_restored_from_ui_state(page, server):
    base, root = server
    _open(page, base, root)
    page.click("#style-add")
    combo_pick(page, page.locator("#styles .style-row"), "Dunes")
    page.click("#style-add")
    combo_pick(page, page.locator("#styles .style-row").nth(1), "Negs")
    import time

    time.sleep(1.0)  # let the debounced ui-state save land
    page.reload()
    wait_until(lambda: page.locator("#styles .style-row").count() == 2)
    assert page.eval_on_selector_all(
        "#styles .style-row .style-name", "els => els.map(e => e.value)"
    ) == ["Dunes", "Negs"]


# ---------------- style configuration dialog ----------------


def test_style_manage_create_edit_delete(page, server):
    base, root = server
    _open(page, base, root)

    # gear opens the configuration list
    page.click("#style-manage")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    # the list keeps the CSV file order (only the dropdown is sorted)
    names = page.eval_on_selector_all(
        ".preset-manage-name", "els => els.map(e => e.textContent)"
    )
    assert names == [
        "Cinematic — Photo Styles", "Dunes — Background",
        "Negs — Negatives", "Plain",
    ]
    # the close-only list dialog has no redundant Cancel button
    assert not page.locator("#dialog-cancel").is_visible()

    # create: name + group + prompt + negative
    page.click("text=+ New style")
    wait_until(
        lambda: page.evaluate("document.getElementById('dialog-title').textContent")
        == "New style"
    )
    # the create dialog saves, so it keeps its Cancel button
    assert page.locator("#dialog-cancel").is_visible()
    page.locator(".style-dialog input[type=text]").first.fill("Analog Film")
    page.locator(".style-dialog input[type=text]").nth(1).fill("Photo Styles")
    page.locator(".style-prompt").first.fill("analog film photo {prompt}")
    page.locator(".style-prompt").nth(1).fill("clean")
    page.click("#dialog-ok")
    # saving returns to the management list (re-rendered with the new style)
    wait_until(lambda: page.evaluate(
        "document.getElementById('dialog-title').textContent") == "Style configuration")
    text = (root / "user" / "styles.csv").read_text("utf-8")
    assert "Analog Film" in text and "Photo Styles" in text

    # edit the new style
    rows = page.locator(".preset-manage-row")
    idx = page.eval_on_selector_all(
        ".preset-manage-name", "els => els.map(e => e.textContent)"
    ).index("Analog Film — Photo Styles")
    rows.nth(idx).locator("button").first.click()  # ✎
    wait_until(
        lambda: page.evaluate("document.getElementById('dialog-title').textContent")
        == "Edit style"
    )
    assert page.locator(".style-prompt").first.input_value() == "analog film photo {prompt}"
    page.locator(".style-prompt").first.fill("vintage {prompt}")
    page.click("#dialog-ok")
    # saving returns to the management list again
    wait_until(lambda: page.evaluate(
        "document.getElementById('dialog-title').textContent") == "Style configuration")
    assert "vintage {prompt}" in (root / "user" / "styles.csv").read_text("utf-8")

    # required name validation keeps the dialog open
    page.click("text=+ New style")
    wait_until(
        lambda: page.evaluate("document.getElementById('dialog-title').textContent")
        == "New style"
    )
    page.locator(".style-dialog input[type=text]").first.fill("   ")
    page.click("#dialog-ok")
    wait_until(lambda: page.locator(".dialog-error").count() >= 1)
    page.keyboard.press("Escape")

    # delete (confirm() accepts); the list is open again from the Escape above
    page.on("dialog", lambda d: d.accept())
    wait_until(lambda: page.evaluate(
        "document.getElementById('dialog-title').textContent") == "Style configuration")
    idx = page.eval_on_selector_all(
        ".preset-manage-name", "els => els.map(e => e.textContent)"
    ).index("Plain")
    page.locator(".preset-manage-row").nth(idx).locator("button").nth(1).click()  # ✗
    # the list stays open after a delete; "Plain" is gone from it (4 remain)
    wait_until(lambda: page.locator(".preset-manage-name").count() == 4)
    assert "Plain" not in (root / "user" / "styles.csv").read_text("utf-8")


# ---------------- generate + temporary styles ----------------


def test_generate_sends_selected_styles(page, server):
    base, root = server
    _open(page, base, root)
    page.click("#style-add")
    combo_pick(page, page.locator("#styles .style-row"), "Cinematic")

    seen = {}

    def on_request(req):
        if req.url.endswith("/api/generate"):
            import json

            seen["body"] = json.loads(req.post_data)

    page.on("request", on_request)
    page.click("#btn-generate")
    wait_until(lambda: "body" in seen, timeout=30)
    assert seen["body"]["styles"] == ["Cinematic"]
    assert seen["body"]["prompt"] == ""  # styles travel separately


def test_temporary_style_from_imported_metadata(page, server):
    base, root = server
    _open(page, base, root)
    # simulate the drag&drop import path (PNG -> /api/png-meta -> applyPngMeta)
    ok = page.evaluate(
        """async () => {
            await applyPngMeta({styles: [
              {name: "Cinematic", prompt: "changed elsewhere", negative_prompt: ""},
              {name: "Lost Style", prompt: "lost {prompt}", negative_prompt: "x"},
            ]}, 64, 64);
            return true;
        }"""
    )
    assert ok
    # two rows appear, "Lost Style" is selectable as a temporary
    wait_until(lambda: page.locator("#styles .style-row").count() == 2)
    assert page.eval_on_selector_all(
        "#styles .style-row .style-name", "els => els.map(e => e.value)"
    ) == ["Cinematic", "Lost Style"]
    # until /api/styles/temp answers the unknown name shows as
    # "(temporary)"; afterwards it appears as a regular (Session-grouped)
    # item
    def lost_normal():
        seq = combo_seq(page, page.locator("#styles .style-row").nth(1))
        return ("item", "Lost Style") in [tuple(x) for x in seq] and \
            ("item", "Lost Style (temporary)") not in [tuple(x) for x in seq]

    wait_until(lost_normal)
    # server: temporary under "Session", not written to the CSV
    r = httpx.get(base + "/api/styles").json()["styles"]
    lost = [s for s in r if s["name"] == "Lost Style"]
    assert lost and lost[0]["temporary"] is True
    assert lost[0]["optgroup"] == "Session"
    assert "Lost Style" not in (root / "user" / "styles.csv").read_text("utf-8")
    # the existing file style was NOT overwritten by the import
    assert not [s for s in r if s["name"] == "Cinematic"][0]["temporary"]


# ---------------- style on/off toggle ----------------


def _style_row_state(page, index=-1):
    """[enabled, comboDisabled] of a sidebar style row."""
    return page.evaluate(
        """(i) => {
            const rows = document.querySelectorAll('#styles .style-row');
            const r = i < 0 ? rows[rows.length + i] : rows[i];
            return [
                !r.classList.contains('off'),
                r.querySelector('.combo-input').disabled,
            ];
        }""", index)


def test_style_toggle_state_and_position(page, server):
    base, root = server
    _open(page, base, root)
    page.click("#style-add")
    combo_pick(page, page.locator("#styles .style-row").last, "Cinematic")
    page.click("#style-add")
    combo_pick(page, page.locator("#styles .style-row").last, "Negs")
    wait_until(lambda: page.locator("#styles .style-row").count() == 2)

    # the toggle sits between the dropdown and the − button
    assert page.eval_on_selector(
        "#styles .style-row:last-child",
        "r => [...r.children].map(c => c.classList.contains('style-name') ? 'combo'\n"
        "  : c.classList.contains('style-toggle') ? 'toggle' : 'rm')") == \
        ["combo", "toggle", "rm"]

    # off: the row is dimmed and the dropdown uses the disabled UI state
    page.locator("#styles .style-row").last.locator("button.style-toggle").click()
    assert _style_row_state(page) == [False, True]
    assert _style_row_state(page, 0) == [True, False]
    # only the enabled style is read back for generation
    assert page.evaluate("getStyles()") == ["Cinematic"]
    assert page.evaluate("getAllStyles()") == \
        [["Cinematic", True], ["Negs", False]]

    # on again
    page.locator("#styles .style-row").last.locator("button.style-toggle").click()
    assert _style_row_state(page) == [True, False]
    assert page.evaluate("getStyles()") == ["Cinematic", "Negs"]


def test_generate_sends_disabled_styles(page, server):
    base, root = server
    _open(page, base, root)
    page.click("#style-add")
    combo_pick(page, page.locator("#styles .style-row").last, "Cinematic")
    page.click("#style-add")
    combo_pick(page, page.locator("#styles .style-row").last, "Negs")
    page.locator("#styles .style-row").last.locator("button.style-toggle").click()

    seen = {}

    def on_request(req):
        if req.url.endswith("/api/generate"):
            import json

            seen["body"] = json.loads(req.post_data)

    page.on("request", on_request)
    page.click("#btn-generate")
    wait_until(lambda: "body" in seen, timeout=30)
    # only the enabled style is applied; the disabled one is recorded
    assert seen["body"]["styles"] == ["Cinematic"]
    assert seen["body"]["disabled_styles"] == ["Negs"]


def test_ui_state_keeps_styles_disabled(page, server):
    base, root = server
    _open(page, base, root)
    page.click("#style-add")
    combo_pick(page, page.locator("#styles .style-row").last, "Cinematic")
    page.click("#style-add")
    combo_pick(page, page.locator("#styles .style-row").last, "Negs")
    page.locator("#styles .style-row").last.locator("button.style-toggle").click()
    # ui-state is saved debounced; give it a beat before reloading
    page.wait_for_timeout(500)
    page.reload()
    wait_until(lambda: page.locator("#styles .style-row").count() == 2)
    assert _style_row_state(page, 0) == [True, False]
    assert _style_row_state(page, 1) == [False, True]


def test_import_restores_disabled_styles(page, server):
    base, root = server
    _open(page, base, root)
    # simulate the drag&drop import path (PNG -> /api/png-meta -> applyPngMeta)
    ok = page.evaluate(
        """async () => {
            await applyPngMeta({styles: [
              {name: "Cinematic", prompt: "cinematic film still {prompt} . moody",
               negative_prompt: "", enabled: true},
              {name: "Negs", prompt: "", negative_prompt: "digital art, cartoon",
               enabled: false},
              {name: "NoFlag", prompt: "x", negative_prompt: ""},
            ]}, 64, 64);
            return true;
        }"""
    )
    assert ok
    wait_until(lambda: page.locator("#styles .style-row").count() == 3)
    # enabled flag restored per row; absent flag defaults to enabled
    assert _style_row_state(page, 0) == [True, False]
    assert _style_row_state(page, 1) == [False, True]
    assert _style_row_state(page, 2) == [True, False]
    assert page.evaluate("getStyles()") == ["Cinematic", "NoFlag"]
    # only the enabled styles reach the prompt-token counter's request
    assert page.evaluate("getAllStyles()") == \
        [["Cinematic", True], ["Negs", False], ["NoFlag", True]]
