# Outpaint source placement — implementation plan

## Goal

Outpaint ("inpaint_not_masked") currently AR-scales the source to touch the target
(`fit_ar`) and centers it in the leftover slack (`paste_centered`, `(W-sw)//2`).
Make the placement user-controllable along the axis/axes that have slack, via a
target bounding-box overlay on the canvas with draggable edge handles placed
slightly outside the source image. The overlay is only shown when source and
target aspect ratios differ.

## Design decisions (resolved)

- **Placement parameter**: normalized `(px, py)` in `[0, 1]`; maps to offsets
  `ox = floor(px * (W - sw))`, `oy = floor(py * (H - sh))` in target px.
  Default `(0.5, 0.5)` reproduces today's centering **exactly**
  (`floor(0.5*gap) == gap//2`), so all existing geometry tests stay green.
  `px=0/1` = flush at the respective target edge.
- Placement applies in the **display frame** (the user's W×H); the server applies
  it inside `fit_ar`, before the target rotation — so rotated jobs and the
  rotation-ignoring UI overlay agree without extra math.
- The mask travels with the source: `resize_shift_mask` already takes `(ox, oy)`;
  feeding it the placement-derived offsets is all that is needed.
- The overlay shows the target box in canvas space at the scale where the
  *scaled* source fills the source's display rect. Box dims = `W/s × H/s` canvas
  px with `s = sw/cw`; the box touches the source on the axes with zero slack,
  exactly as specified.
- The displayed box must use the **resolved** target size (outpaint sizes are
  scaled by `oversample` and rounded UP to a multiple of 16, clamped 64..4096 —
  mirror `_round16_up` in JS). The auto-size sentinel (-1) cannot be previewed
  client-side (resolved from the mask server-side): overlay is hidden for -1;
  the server still applies placement to the resolved size.
- Persist placement in ui-state (`placement: [x, y]`), write it to PNG metadata
  (only when non-default), restore on PNG import, and add it to presets.

## 1. Backend geometry — `src/imagagen/generation.py`

```python
def fit_ar(w, h, W, H, px: float = 0.5, py: float = 0.5):
    """AR-preserving scale of (w,h) so it "touches" the inside of (W,H).
    Returns (sw, sh, ox, oy); ox/oy position the scaled source inside the
    target: 0 = flush left/top, 1 = flush right/bottom, 0.5 = centered."""
    scale = min(W / w, H / h)
    sw = max(1, int(round(w * scale)))
    sh = max(1, int(round(h * scale)))
    return sw, sh, int(px * (W - sw)), int(py * (H - sh))
```

- Generalize `paste_centered(scaled, W, H)` → `paste_at(scaled, W, H, ox, oy)`
  (same edge-replicating `np.pad`, offsets from the parameters). Update the two
  unit tests that import `paste_centered` (rename + pass offsets); keep the
  centering behaviour by computing the offsets with `fit_ar`'s formula.
- `with_inpaint(..., placement: tuple[float, float] = (0.5, 0.5))` (new last
  kwarg). Clamp `px, py` to `[0, 1]`. In the `inpaint_not_masked` branch only:

```python
px, py = (min(1.0, max(0.0, placement[0])), min(1.0, max(0.0, placement[1])))
sw, sh, ox, oy = fit_ar(cw, ch, request.width, request.height, px, py)
target = paste_at(resize_image(image, sw, sh), request.width, request.height, ox, oy)
new_mask = (resize_shift_mask(mask, sw, sh, ox, oy, request.width, request.height)
            if mask.any() else zeros)
```

  (`_resolve_gen_size` has already scaled `request.width/height`, so placement
  applies to the resolved target — matching the JS overlay.)
- `InpaintPlan` gains `placement: tuple[float, float] = (0.5, 0.5)` (metadata
  round-trip carrier; geometry is already baked into `init_image`).
- Update the module docstring wording ("centered" → "placed per `placement`").
- `src/imagagen/canvas.py` `_inherit_mask_locked` keeps calling `fit_ar` with
  defaults (centered) — untouched.

## 2. API — `src/imagagen/app.py`

`GenerateBody` gains:

```python
placement: tuple[float, float] = (0.5, 0.5)  # outpaint source position
# field_validator: clamp each component to [0, 1]
```

Pass it through at the `with_inpaint` call (~line 915): `placement=body.placement`.

## 3. PNG metadata

- `metadata.build_parameters(..., placement: tuple[float, float] | None = None)`:
  emits JSON key `"outpaint_placement": [px, py]` when not None (JSON pass-through
  in `parse_parameters`, no legacy-format entry needed).
- `queue._result_to_png` (~line 915, next to the existing `denoise=...` plan
  gate): pass `placement` only for
  `p is not None and p.mode == "inpaint_not_masked" and tuple(p.placement) != (0.5, 0.5)`
  → default placement leaves metadata clean.

## 4. Persistence — presets + ui-state

- `uistate.py`: `placement: tuple[float, float] = Field(default=(0.5, 0.5))`
  + clamp validator (mirror the GenerateBody validator).
- `presets.py`: add `"placement_x"`, `"placement_y"` to `PRESET_FIELDS`; `Preset`
  fields `placement_x/ placement_y: float | None = Field(default=None, ge=0, le=1)`.
  Splitting into two scalar fields reuses the existing preset machinery
  (checkbox per field, number row in the dialog) — no tuple handling.
- `app.js`:
  - `state.outpaintPlacement = { x: 0.5, y: 0.5 }` on the `state` object.
  - `readUiState()`: `placement: [state.outpaintPlacement.x, state.outpaintPlacement.y]`;
    in the ui-state apply path set `state.outpaintPlacement` from `st.placement`
    (skip null/undefined — the null-clobber gotcha), then `viewRedraw()`.
  - Preset dialog: add `["placement_x", "Placement X", "number", ...]`,
    `["placement_y", "Placement Y", "number", ...]` rows (step 0.05) and extend
    the preset key list (~line 1773); preset *apply* is a special case (no form
    element): write into `state.outpaintPlacement` next to the existing
    `st.seg_classes`-style special cases + `viewRedraw()`; preset *save* reads
    from `state.outpaintPlacement`.
  - `generate()` body: `placement: [state.outpaintPlacement.x, state.outpaintPlacement.y]`.
  - `applyPngMeta`: `if (Array.isArray(meta.outpaint_placement))` → set
    `state.outpaintPlacement`, `viewRedraw()`, `scheduleSaveUiState()`.

## 5. Canvas overlay — `src/imagagen/static/canvas.js`

New section (kept pure/inspectable for tests), mirroring the mask-outline /
selection-handle conventions. All geometry is canvas px (world space of
`viewRedraw`'s base transform); `u = 1 / view.scale` keeps strokes/handles
zoom-independent, same as `drawMaskOutline`.

```js
function resolveGenSize(w, h, ov) { // mirror of server _resolve_gen_size
  const r16 = (v) => Math.min(4096, Math.max(64, Math.ceil(v * ov / 16) * 16));
  return [r16(w), r16(h)];
}
function outpaintGeom() {
  if ($("mode").value !== "inpaint_not_masked" || !view.base) return null;
  const W0 = parseInt($("width").value, 10), H0 = parseInt($("height").value, 10);
  if (!(W0 > 0 && H0 > 0)) return null;               // -1 sentinel: hidden
  const [W, H] = resolveGenSize(W0, H0, parseFloat($("oversample").value) || 1);
  const s = Math.min(W / view.w, H / view.h);
  const sw = Math.max(1, Math.round(view.w * s));
  const sh = Math.max(1, Math.round(view.h * s));
  const gx = W - sw, gy = H - sh;
  if (gx <= 0 && gy <= 0) return null;                // same AR: nothing to place
  const px = state.outpaintPlacement.x, py = state.outpaintPlacement.y;
  const ox = Math.floor(px * gx), oy = Math.floor(py * gy);
  return { W, H, s, gx, gy, ox, oy,
           bx: -ox / s, by: -oy / s, bw: W / s, bh: H / s };
}
function outpaintHandles(g) {
  const u = 1 / view.scale, off = 9 * u;              // handle centre offset
  const out = [];
  if (g.gx > 0) {
    out.push({ axis: "x", x: -off, y: view.h / 2, cursor: "ew-resize" });
    out.push({ axis: "x", x: view.w + off, y: view.h / 2, cursor: "ew-resize" });
  }
  if (g.gy > 0) {
    out.push({ axis: "y", x: view.w / 2, y: -off, cursor: "ns-resize" });
    out.push({ axis: "y", x: view.w / 2, y: view.h + off, cursor: "ns-resize" });
  }
  return out;
}
function hitOutpaintHandle(p) { // canvas px -> handle or null
  const g = outpaintGeom(); if (!g) return null;
  const t = 10 / view.scale;
  return outpaintHandles(g).find((h) => Math.abs(p.x - h.x) <= t && Math.abs(p.y - h.y) <= t) || null;
}
```

- `drawOutpaintOverlay(ctx)`: stroke the box rect (`rgba(128,128,128,0.6)`,
  `lineWidth = u`), the `${W}×${H}` label at the box top-left (same screen-space
  text style as the mask-bbox label), and filled handle squares (16·u,
  `rgba(128,128,128,0.85)`) at the handle centres. Call it from `viewRedraw`
  right after `drawMaskOutline(ctx)` (before the selection overlay).
- Drag: pointerdown — before the tool arming, `const h = hitOutpaintHandle(p)`;
  if hit: `view.outDrag = { axis: h.axis, x0: p.x, y0: p.y,
  base: { ...state.outpaintPlacement } }`, capture, `return` (no stroke arms).
  pointermove — first branch: while `view.outDrag`,
  `dx = p.x - x0` (canvas px) and for axis "x":
  `state.outpaintPlacement.x = clamp01(base.x + dx * g.s / g.gx)` (axis "y"
  analogous with `dy * g.s / g.gy`), then `viewRedraw()`.
  pointerup — clear `view.outDrag` and `scheduleSaveUiState()`.
  (1 canvas display px = `s` target px: `ox = px*gx` shifts the box by `ox/s`
  display px, hence the `s/gx` factor; dragging a handle moves the SOURCE that
  way inside the box.)
- Cursor: in `syncCanvasCursor` (selection handles keep priority) set the handle
  `cursor` ("ew-resize"/"ns-resize") when `hitOutpaintHandle(cursorCanvas(e))`.
- Redraw hooks (`app.js`): `$("width")`, `$("height")`, `$("oversample")`
  `input` listeners additionally call `viewRedraw()` (same pattern as the
  existing `$("mask-padding").addEventListener("input", viewRedraw)`), and
  `syncMode()` ends with `viewRedraw()` so mode switches repaint the overlay.

## 6. Tests

Unit `tests/unit/test_generation.py`:
- `fit_ar` placement: `fit_ar(100,100,250,200, 0, .5) == (200,200,0,0)`,
  `px=1 → ox=50`, `py` irrelevant when `gy=0`; the four existing `fit_ar` tests
  must pass unchanged (default 0.5 ≡ old centering).
- `paste_at` (renamed test): content at explicit `(ox, oy)`, border edge-replicated.
- `with_inpaint` placement: 64×64 source with a unique marker column, target
  128×64, `placement=(1, .5)` → `request.init_image`'s rightmost 64 columns equal
  the resized source, left 64 are edge-replicated; a mask stroke's
  `request.init_mask` bbox starts at `ox`; plan carries `placement`.

Integration `tests/integration/test_generation_modes.py` (mock backend, low
denoise so output tracks `init_image`):
- outpaint `placement=[0, 0.5]` → source content flush-left in the result;
  entity `parameters` JSON contains `"outpaint_placement": [0.0, 0.5]`;
  default placement → key absent.

UI `tests/ui/test_outpaint_placement_ui.py` (mock server; canvas-px↔screen
mapping via canvas rect + `view.ox/oy/scale`, as in existing overlay tests):
- box visible in outpaint mode with differing source/target AR; hidden in
  generate mode, with equal ARs, and with `#width = -1`;
- drag the right handle → `state.outpaintPlacement.x` moves toward 1 (clamped),
  `ui-state.json` gains `placement` (reload-restore assert per the 500 ms
  debounce pattern);
- cursor over handles is `ew-resize`/`ns-resize`.

Run: `uv run pytest tests/unit/test_generation.py tests/integration/test_generation_modes.py`
+ the new UI test file (targeted only). Geometry-only change — no backend
sampling code touched, so no GPU smoke required. Commit when green
(past-tense message, e.g. "Added user-controllable outpaint source placement").

## Suggested implementation order

1. generation.py geometry + plan field, unit tests green.
2. app.py body field + queue/metadata wiring, integration test.
3. canvas.js overlay + drag + hooks, app.js state/generate/applyUiState/
   applyPngMeta/preset rows.
4. UI test, full targeted test run, commit.
