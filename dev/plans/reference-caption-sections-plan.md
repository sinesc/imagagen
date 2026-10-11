# Reference caption sections — plan

## Goal

VLM reference captions are structured: an unlabeled preamble paragraph,
several `**Title:**` sections (Subject, Clothing, Art Style, Colors,
Composition, Lighting, …), and a trailing unlabeled "In summary, …"
paragraph. Let the user pick which sections get folded into the
positive prompt: when a reference image is checked, a row of toggle
tags (one per section, labeled by title) appears below the negative
prompt; only enabled sections are applied. Default: all enabled.

## Section model

`parse_caption_sections(caption) -> list[(title, text)]` in caption
order. Split on blank lines (`\n\n`); a paragraph that is exactly
`**Title:**` (optionally with inline body text after the colon) starts
a section; unlabeled paragraphs attach to the current section.
Unlabeled paragraphs **before** the first heading become section
`Overview`; unlabeled paragraphs **after** the last heading become
section `Summary`. Duplicate titles get ` (2)`, ` (3)` suffixes.
Sections with empty text are dropped.

Selection semantics: `reference_sections: list[str] | None` — `None`
means ALL sections (default); a list means exactly those titles;
`[]` means apply nothing (prompt unchanged).

## Server changes

### `src/imagagen/reference.py`

```python
import re
_SECTION_HEADING = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$", re.S)

def parse_caption_sections(caption: str) -> list[tuple[str, str]]:
    """[(title, text)] in caption order (see plan: Overview/Summary
    for unlabeled leading/trailing paragraphs, deduped titles)."""
    paras = [p.strip() for p in caption.split("\n\n") if p.strip()]
    sections: list[tuple[str | None, list[str]]] = []
    for p in paras:
        m = _SECTION_HEADING.match(p)
        if m:
            title = m.group(1).strip()
            body = [m.group(2).strip()] if m.group(2).strip() else []
            sections.append((title, body))
        elif sections:
            sections[-1][1].append(p)
        else:
            sections.append((None, [p]))
    out: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    n = len(sections)
    for i, (title, body) in enumerate(sections):
        if title is None:
            title = "Overview" if i == 0 else "Summary"
        c = seen.get(title, 0) + 1
        seen[title] = c
        if c > 1:
            title = f"{title} ({c})"
        text = "\n\n".join(body).strip()
        if text:
            out.append((title, text))
    return out
```

`apply_reference` gains the selection (keep the old signature working):

```python
def apply_reference(prompt: str, caption: str,
                    sections: list[str] | None = None) -> str:
    if not caption:
        return prompt
    parsed = parse_caption_sections(caption)
    text = "\n\n".join(x for t, x in parsed
                       if sections is None or t in sections)
    if not text:
        return prompt
    return f"{prompt} {_CAPTION_PREFIX}{text}."
```

Also export a tiny helper `selected_caption_titles(caption, sections)`
(the applied titles, in order) so backends can record what was applied.

### Request/result plumbing

- `GenerateRequest` (backends/__init__.py): `reference_sections: list[str] | None = None` (rides `model_dump` through the queue automatically).
- `GenerateResult`: `reference_sections: list[str] | None = None` — the titles actually applied (None when no reference).
- All four backends: pass `request.reference_sections` into `apply_reference` at the existing call sites (sdxl, flux, zimage, krea2) and set `reference_sections=` on the result.
- `app.py` `GenerateBody`: `reference_sections: list[str] | None = None`, forwarded to the request.
- `metadata.py` `build_parameters`: add `reference_sections` (the applied titles) next to the existing `reference_caption` key. `reference_caption` keeps storing the FULL caption (reproducibility: the full caption is cache-keyed; the applied subset is reconstructible from sections + caption).
- `uistate.py`: `reference_sections: list[str] | None`; dropped together with `reference_image_id` when the canvas entity is gone.

## UI changes (`static/index.html`, `app.js`, `style.css`)

- `index.html`: `<div id="ref-sections" class="hidden"></div>` directly after the `#negative` textarea.
- `app.js`:
  - Mirror the parser: `parseCaptionSections(caption)` (same rules: blank-line paragraphs, `**Title:**` headings, Overview/Summary, dedup suffixes).
  - `state.referenceSections = null` (null = all enabled). Reset to null whenever `state.referenceImageId` changes.
  - `renderRefSections()`: called after the caption preview resolves (where `state.referenceCaption` is set) and on reference clear. Builds one `<button class="ref-tag mini" data-title="Colors">Colors</button>` per section; `active` class when enabled (enabled = `state.referenceSections === null` or title in list). Hide the row when the caption has <= 1 section (nothing to toggle).
  - Click handler: materialize the full enabled set from the buttons, toggle the clicked one, store the array in `state.referenceSections` (all-enabled stores null), re-render button classes.
  - Generate payload: `reference_sections: state.referenceSections`.
  - ui-state save/restore: `reference_sections` (on restore, re-fetch the caption preview for the restored `reference_image_id` exactly like the check handler does, then render the row with the restored selection).
- `style.css`: `.ref-tag` pill buttons in a wrapping flex row (`#ref-sections { display:flex; flex-wrap:wrap; gap:4px; }`), active = filled accent, inactive = greyed (`#777` border, muted text) per the existing UI conventions (`.mini`, 1.75rem controls).

## Mock + tests

- `MockBackend` caption preview: replace the `"mock"` echo with a fixed multi-section caption (Overview + `**Subject:**` + Summary paragraphs) so UI/integration tests can exercise tags. Update the integration assertions that expect `"mock"`.
- Unit (`tests/unit/test_reference_caption.py`): parser (full structure, Overview/Summary, inline heading body, no headings -> single Overview, duplicate titles deduped, empty sections dropped); `apply_reference` with `None`/subset/empty selection.
- Integration (`tests/integration/test_reference_api.py`): generate with `reference_sections` subset -> result echoes applied titles; ui-state round-trip of `reference_sections`; dropped when the reference entity is gone.
- UI (`tests/ui/test_reference_thumb_ui.py`): checking a reference renders the tag row with the mock caption's titles, all active; toggling one off persists to ui-state; row hidden when no reference.

## Verification

1. `.venv/bin/python -m pytest tests/unit tests/integration -q` (full), UI tests for the reference files only.
2. `uv run ruff check` on touched files (24 pre-existing findings acceptable).
3. Real-GPU smoke (`--gpu 1`): check a reference in the browser, confirm tags appear from the real caption preview, toggle off Colors + Summary, generate, and confirm the PNG metadata `reference_sections` lists the applied titles and the prompt contains only the selected paragraphs.

## Acceptance criteria

- Caption preview with sections renders one tag per section (incl. Overview/Summary), all active by default.
- Disabled sections are absent from the applied prompt text; enabled ones appear in caption order.
- Selection survives reload (ui-state) and resets when the reference changes.
- No-reference generation is byte-identical to today's behavior.

## Open questions (defaults assumed)

- Labels `Overview` (preamble) and `Summary` (trailing paragraph) — assumed OK.
- Tag row hidden when the caption has <= 1 section — assumed.
- Selection persisted in ui-state — assumed yes.
