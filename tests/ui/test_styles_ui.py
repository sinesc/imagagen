"""Browser tests for the styles UI (Chromium via Playwright).

Covers the style dropdown rows (+ / − buttons, optgroup sorting), the
style configuration dialog (create / edit / delete), ui-state restore,
the styles sent on generate, and the "Session" temporary styles created
by PNG metadata import. Uses the mock-backend server in a temp root.
"""

import httpx
from conftest import wait_until

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
        (root / "styles.csv").write_text(SEED_ROWS + "\n", "utf-8")
    page.goto(base)
    page.wait_for_function("brushNames.length >= 1 && state.styles.length > 0")


def _options(page):
    """The (optgroup label, option text) pairs of the first style row."""
    sel = "#styles .style-row select"
    return page.eval_on_selector_all(
        f"{sel} optgroup, {sel} option",
        "els => els.map(e => e.tagName === 'OPTGROUP'"
        " ? ['\x00' + e.label, null]"
        " : [e.closest('optgroup')?.label ?? '', e.textContent])",
    )


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
    row.first.locator("select").select_option("Cinematic")
    page.click("#style-add")
    assert row.count() == 2
    row.nth(1).locator("select").select_option("Dunes")
    assert row.nth(0).locator("select").input_value() == "Cinematic"

    # − removes its own row only (the remaining row is the second one)
    row.first.locator("button").click()
    assert row.count() == 1
    assert row.first.locator("select").input_value() == "Dunes"
    # the surviving selection is read back as names for generate
    assert page.evaluate("getStyles()") == ["Dunes"]

    # the "+ Style" button stretches so the gear's right edge aligns with
    # the rows' remove (−) buttons
    geom = page.evaluate(
        """() => {
            const right = (e) => e.getBoundingClientRect().right;
            return {
                minus: right(document.querySelector('#styles .style-row button')),
                gear: right(document.getElementById('style-manage')),
            };
        }"""
    )
    assert abs(geom["gear"] - geom["minus"]) < 1


def test_style_selection_restored_from_ui_state(page, server):
    base, root = server
    _open(page, base, root)
    page.click("#style-add")
    page.locator("#styles .style-row select").select_option("Dunes")
    page.click("#style-add")
    page.locator("#styles .style-row").nth(1).locator("select").select_option("Negs")
    import time

    time.sleep(1.0)  # let the debounced ui-state save land
    page.reload()
    wait_until(lambda: page.locator("#styles .style-row").count() == 2)
    assert page.eval_on_selector_all(
        "#styles .style-row select", "els => els.map(e => e.value)"
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

    # create: name + group + prompt + negative
    page.click("text=+ New style")
    wait_until(
        lambda: page.evaluate("document.getElementById('dialog-title').textContent")
        == "New style"
    )
    page.locator(".style-dialog input[type=text]").first.fill("Analog Film")
    page.locator(".style-dialog input[type=text]").nth(1).fill("Photo Styles")
    page.locator(".style-prompt").first.fill("analog film photo {prompt}")
    page.locator(".style-prompt").nth(1).fill("clean")
    page.click("#dialog-ok")
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))
    text = (root / "styles.csv").read_text("utf-8")
    assert "Analog Film" in text and "Photo Styles" in text

    # the management list is open again? No -- it was closed on Create; reopen
    page.click("#style-manage")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
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
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))
    assert "vintage {prompt}" in (root / "styles.csv").read_text("utf-8")

    # required name validation keeps the dialog open
    page.click("#style-manage")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    page.click("text=+ New style")
    wait_until(
        lambda: page.evaluate("document.getElementById('dialog-title').textContent")
        == "New style"
    )
    page.locator(".style-dialog input[type=text]").first.fill("   ")
    page.click("#dialog-ok")
    wait_until(lambda: page.locator(".dialog-error").count() >= 1)
    page.keyboard.press("Escape")

    # delete (confirm() accepts)
    page.on("dialog", lambda d: d.accept())
    page.click("#style-manage")
    wait_until(lambda: not page.evaluate("document.getElementById('dialog').hidden"))
    idx = page.eval_on_selector_all(
        ".preset-manage-name", "els => els.map(e => e.textContent)"
    ).index("Plain")
    page.locator(".preset-manage-row").nth(idx).locator("button").nth(1).click()  # ✗
    wait_until(lambda: page.evaluate("document.getElementById('dialog').hidden"))
    assert "Plain" not in (root / "styles.csv").read_text("utf-8")


# ---------------- generate + temporary styles ----------------


def test_generate_sends_selected_styles(page, server):
    base, root = server
    _open(page, base, root)
    page.click("#style-add")
    page.locator("#styles .style-row select").select_option("Cinematic")

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
    vals = page.eval_on_selector_all("#styles .style-row select", "els => els.map(e => e.value)")
    assert vals == ["Cinematic", "Lost Style"]
    # until /api/styles/temp answers the unknown name shows as "(temporary)";
    # afterwards it appears as a regular (Session-grouped) option
    labels = page.eval_on_selector_all(
        "#styles .style-row option", "els => els.map(e => e.textContent)"
    )
    assert "Lost Style" in labels
    # server: temporary under "Session", not written to the CSV
    r = httpx.get(base + "/api/styles").json()["styles"]
    lost = [s for s in r if s["name"] == "Lost Style"]
    assert lost and lost[0]["temporary"] is True
    assert lost[0]["optgroup"] == "Session"
    assert "Lost Style" not in (root / "styles.csv").read_text("utf-8")
    # the existing file style was NOT overwritten by the import
    assert not [s for s in r if s["name"] == "Cinematic"][0]["temporary"]
