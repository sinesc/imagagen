# Undo/redo rework — implementation plan

Reworks the single combined canvas undo/redo stack into two independent
systems: a global, image-independent **mask** history and per-image
**draw** histories. Image selection is no longer an undoable change; every
canvas image is a persistent entity with its own draw stack, and images are
evicted (with their history) when they are no longer on the canvas and no
longer referenced by a live job.

## Resulting behaviour

- **Mask history** (global): one undo/redo stack for every mask change —
  mask brush strokes (draw/erase), expand, shrink, clear, segmentation
  union, and the new mask set by "inpaint not masked". Independent of the
  image: switching images does not touch it.
- **Draw history** (per image): one undo/redo stack per canvas image
  holding its pre-stroke states. Selecting a different thumbnail switches
  the active image *and* its stack; selecting it again restores the
  history.
- Image changes (thumb selection, upload/import, generation, clear) push
  nothing onto any stack. Each generated/inpainted result is a new image
  with an empty draw history; "inpaint masked" composites onto the
  submit-time base image exactly like today, but as a new entity (in-flight
  edits survive on the old image, reachable via the source thumbnail).
- The mask keeps its own (native) size. When it is displayed or used for
  inpainting while the canvas image has a different size, it is
  aspect-preserving **fit-within, centered** (scale = min, same geometry as
  the outpaint `fit_ar`), touching the image's inner edges; the remainder
  of the image area is unmasked.
- Buttons: one undo/redo pair. Label and enabled state follow the active
  tool: "↩ undo mask"/"↪ redo mask" in mask mode, "↩ undo draw"/
  "↪ redo draw" in brush mode.
- The unused `POST /api/canvas/clear` endpoint is deleted (no UI button or
  tooling calls it).

## Image entity lifecycle

- Entities live in `Canvas._images: dict[id, CanvasImage]`.
  Ids: `job:{job_id}:{index}` (results), an existing canvas-image id
  (a job's *source* references the canvas image, no new entity),
  `upload:{n}` (upload/drag&drop).
- An entity holds: `png` bytes (cold store, always), `info` (PNG text
  chunks), `plan_mask` bytes (inpaint-not-masked jobs only), `image` numpy
  (hot cache, only while the entity is the canvas image), `undo`/`redo`
  lists of PNG bytes.
- Hot cache: the numpy array is freed when the entity leaves the canvas
  and re-decoded from the entity's PNG when it is selected again.
- Eviction: the queue maintains the set of job-anchored ids
  (`Canvas.anchor`, §2). `Canvas.evict_stale()` drops every entity that is
  neither the current canvas image nor job-anchored. A new `submit()` drops
  every **finished** job (`done_emitted`) from `queue._jobs`, re-anchors,
  and evicts (in-flight jobs stay). `set_current`/`import_image` also
  evict, so an uploaded image that has no thumbnail is dropped as soon as
  it leaves the canvas. Client behaviour matches: the thumb strip is
  cleared on every new generation, so only the latest job's PNG endpoints
  are still referenced.
- Uploads have no job reference: they are pruned as soon as they leave the
  canvas (pixels and history lost by design).

## History storage

- All history entries (draw and mask) are PNG-compressed bytes, not raw
  numpy — ~100–300 KB per 1024² entry instead of ~3.2 MB.
- Caps: `MASK_HISTORY = 200` (global mask stack), `DRAW_HISTORY = 50`
  (per-image draw stack). Oldest entries drop off; the redo list is
  unbounded (bounded in practice by the undo cap).
- No-ops record no step (existing change-detection behaviour kept: a mask
  op that changes nothing, or setting an identical mask, pushes nothing).

---

## 1. `src/imagagen/canvas.py` — core rework

### New `CanvasImage` dataclass

```python
@dataclass
class CanvasImage:
    id: str
    png: bytes                     # cold store (always present)
    info: dict = field(default_factory=dict)   # PNG text chunks
    plan_mask: bytes | None = None             # inpaint-not-masked result
    image: np.ndarray | None = None            # hot cache: on canvas only
    undo: list[bytes] = field(default_factory=list)
    redo: list[bytes] = field(default_factory=list)
```

### `Canvas` state

- `self._images: dict[str, CanvasImage]`, `self._current: str | None`
- `self._mask: np.ndarray | None` (native size)
- `self._mask_undo: list[bytes | None]`, `self._mask_redo: list[bytes | None]`
  (`None` = "no mask" state)
- Constants: `MASK_HISTORY = 200`, `DRAW_HISTORY = 50` (replace
  `MAX_HISTORY`)

### Entity management (all under `self._lock`)

- `add_result(job_id, index, png, info, plan_mask=None) -> str` — creates
  `job:{job_id}:{index}`, returns id.
- `import_image(data: bytes) -> str` — parses PNG (RGB + text chunks as
  today in `set_from_png_bytes`), creates `upload:{n}`, sets it current
  (decodes to hot cache).
- `current_id() -> str | None`.
- `set_current(id)` — switch canvas image:
  1. old entity: `_images[old].image = None` (drop hot cache).
  2. new entity: if `entity.image is None`, decode `entity.png`;
     `self._current = id`.
  3. if `entity.plan_mask is not None` and differs from the current mask
     (after scaling comparison): `set_mask(np.frombuffer decoded)` —
     one mask-history step. This covers both auto-set and thumb
     selection of "inpaint not masked" results.
- `anchor(ids: set[str])` — the queue records which entity ids are
  referenced by a live job (results + source); stored as
  `self._anchored`. Called whenever the job set changes (§2).
- `evict_stale()` — delete every entity with id `!= self._current`
  **and** `not in self._anchored` (its undo/redo/png go with it). Called
  from `set_current` and `import_image` (so a stale upload is dropped the
  moment it leaves the canvas) and after the queue drops jobs (§2).

### Mask helpers

- `mask_scaled() -> np.ndarray | None` — the mask displayed/used against
  the current image: `None` when no mask or no image. Otherwise
  `sw, sh, ox, oy = fit_ar(mw, mh, W, H)` (import from `.generation`) and
  `resize_shift_mask(self._mask, sw, sh, ox, oy, W, H)` (already in
  `.generation`: scales and places on a zero (H,W) canvas).
- `mask_fit_params() -> tuple[float, int, int] | None` — returns
  `(scale, ox, oy)` with `scale = sw / mw` for canvas→mask coordinate
  remapping; `None` when there is no image (or mask size unknown).
  Same-size case: identity (scale 1, offset 0).
- `set_mask(mask: np.ndarray) -> bool` — replacement (segmentation,
  plan masks). Change-detected against the *native* mask: identical
  arrays push no step. Sets native size to the new mask's shape.
- Mask history push/pop: `_mask_push()` snapshots the current mask as PNG
  bytes (or `None`), caps at `MASK_HISTORY`, clears `_mask_redo`;
  `undo_mask()`/`redo_mask()` swap like today but only the mask is
  restored (size comes with the array).

### Strokes

- `stroke(..., target)` keeps its signature.
  - `target="image"`: requires a current entity with hot cache (raises
    `ValueError` for an empty canvas, as today). Snapshot:
    `entity.undo.append(<current png bytes>)` (cap `DRAW_HISTORY`),
    `entity.redo.clear()`, stamp onto the numpy, re-encode
    `entity.png` (hot cache and cold store stay in sync), commit only when
    pixels actually changed (compare before/after; no change → no entry —
    so the pushed snapshot must be made *after* the change check, i.e.
    stamp on a copy first, then commit).
  - `target="mask"`: remap each point
    `mx = (x - ox) / scale`, `my = (y - oy) / scale` using
    `mask_fit_params()`, stamp onto the **native** mask array (existing
    stamping code unchanged, clipping in native bounds), push the mask
    history entry on change.
- `undo(target="image"|"mask")`, `redo(target)`,
  `can_undo_draw/can_redo_draw` (current entity),
  `can_undo_mask/can_redo_mask` (mask: global). `undo`/`redo` for draw
  swap the hot-cache top entry with the popped history PNG and update
  `entity.png`.

### Mask operations

- `mask_dilate`/`mask_erode`: run on the **native** mask (scipy filters
  unchanged, footprint in native px). History entry on change.
- `mask_clear`: zeros the native mask; no-op (no step) when already empty.
- Remove: `set_image`, `set_image_keep_mask`, `set_from_png_bytes`,
  `clear`, the old combined `_undo`/`_redo`/`_state`/`_commit_locked`,
  `undo`, `redo`, `can_undo`, `can_redo` (all superseded).
- Keep: `image` (hot cache of current entity, `None` when empty),
  `has_image`, `size`, `to_png_bytes` (from current entity; carries
  `entity.info`), `to_mask_png_bytes` → now serves `mask_scaled()` bytes
  (404 upstream when it is `None`), `png_info` (current entity's `info`),
  brushes API, `stamp_positions`.

## 2. `src/imagagen/queue.py` — entities, source ids, job eviction

- `Job`: add `source_image_id: str | None = None`.
- `submit()`:
  - `job.source_image_id = self.canvas.current_id()` (replaces the
    `canvas.has_image()` check; `source_png` snapshot for strip display is
    captured exactly as today, only when `source_image_id is not None`).
  - After registering the new job: drop finished jobs —
    `for jid, j in list(self._jobs.items()): if j.done_emitted and jid != job.id: del self._jobs[jid]` —
    then `self.canvas.anchor(self.live_image_ids())` and
    `self.canvas.evict_stale()`.
  - `live_image_ids()`: `{f"job:{j.id}:{i}" for j in self._jobs.values()
    for i in j.results} | {j.source_image_id for j in self._jobs.values()
    if j.source_image_id}` — also refreshed (anchor + evict_stale) after
    each result is added in `_run_task`.
- `_run_task` result handling:
  - `eid = self.canvas.add_result(job.id, task.index, png, info,
    plan_mask=(job.plan.new_mask bytes if inpaint_not_masked else None))`
    — encode `plan_mask` once per job (cache on the Job) and share.
  - WS `image` event gains `"image_id": eid`.
  - First-result auto-set: `self.canvas.set_current(eid)` (instead of
    `set_image`/`set_image_keep_mask`); WS `canvas_set` unchanged.
- `source_image` event gains `"image_id": job.source_image_id`.
- Nothing else changes; worker/condvar logic untouched.

## 3. `src/imagagen/app.py` — API

- Response contract: every canvas-mutating endpoint (stroke, mask op,
  segmentation, thumb, upload, select, undo, redo) returns
  `{"ok": ..., "undo_mask": b, "redo_mask": b, "undo_draw": b,
  "redo_draw": b}` (replace today's `undo`/`redo` keys) via one helper
  `canvas_flags()`.
- `POST /api/canvas/undo` and `/api/canvas/redo`: body
  `{"target": "mask" | "draw"}` (422/400 on other values); performs
  `canvas.undo(target)` / `canvas.redo(target)`.
- `POST /api/canvas/select`: body `{"id": str}` → `canvas.set_current(id)`
  (unknown id → 400 `{"error": ...}`). Thumb-selection and source-restore
  both go through it.
- `POST /api/canvas/thumb`: body unchanged; resolves
  `id = f"job:{body.job}:{body.index}"` and calls the same
  `set_current` path (drop the `plan_mask` argument logic — the entity
  carries it).
- `POST /api/canvas/upload`: `canvas.import_image(data)`; response gains
  the flags.
- Delete `POST /api/canvas/clear` (no UI or tooling uses it; with the mask
  being its own system, its old "wipe everything" meaning no longer fits).
- `GET /api/mask.png`: `canvas.to_mask_png_bytes()` now returns the
  *scaled* mask (see §1) → client overlay needs no changes.
- `with_inpaint` in `generation.py`: `mask = canvas.mask_scaled()` instead
  of `canvas.mask`; `base_image = canvas.image` (hot cache, always
  available while the canvas shows an image). Everything downstream
  (bbox, padding, init image/mask, plan) is unchanged — `base_mask` in the
  plan is the scaled mask, and `postprocess` blends against
  `base_image` with it, both at image size. For "inpaint not masked",
  `resize_shift_mask` now runs on the scaled mask (one extra LANCZOS pass,
  acceptable).
- Segmentation endpoint: `existing = canvas.mask_scaled()`;
  `merged = np.maximum(existing, seg)` (existing is None → seg), then
  `canvas.set_mask(merged)` (change-detected → repeated identical runs
  push no mask step). Native mask size becomes the image size.
- `/api/save`, `/api/output`, `/api/canvas.png`, brushes, ui-state:
  unchanged.

## 4. Frontend

### `static/index.html`
- `btn-undo`/`btn-redo` keep their ids; initial text "↩ undo draw" /
  "↪ redo draw"; titles updated dynamically by JS.

### `static/app.js`
- Thumb objects gain `image_id`:
  - WS `source_image`: `{source: true, url: msg.url, image_id: msg.image_id}`
  - WS `image`: `{job, index, url, seed, image_id: msg.image_id}`
- `restoreThumb(t)`: single path for both kinds —
  `await api("/api/canvas/select", {id: t.image_id})` (replaces the
  source fetch+upload).
- Undo/redo:
  ```js
  let undoFlags = { undo_mask: false, redo_mask: false,
                    undo_draw: false, redo_draw: false };
  function syncUndoButtons() {
    const t = Tools.current === "mask" ? "mask" : "draw";
    $("btn-undo").textContent = `↩ undo ${t}`;
    $("btn-redo").textContent = `↪ redo ${t}`;
    $("btn-undo").disabled = !undoFlags[`undo_${t}`];
    $("btn-redo").disabled = !undoFlags[`redo_${t}`];
  }
  ```
  - `btn-undo`/`btn-redo` onclick → `api("/api/canvas/undo", {target: t})`
    / redo, store the four flags, `syncUndoButtons()`, `viewRefresh()`.
  - Every other canvas-mutating handler (stroke commit, `maskOp`,
    segmentation, `restoreThumb`, upload/drop, clear) stores the returned
    flags and calls `syncUndoButtons()`.
  - `switchTool()` calls `syncUndoButtons()`.
  - Initial state: flags all false → buttons disabled until the first
    canvas response (consistent, no GET needed).
- No changes in `canvas.js`: the mask overlay already stretches the served
  mask PNG to the image rect, and the server now serves it pre-scaled to
  the image size; the optimistic stroke preview is unaffected.
- Keep the no-flicker invariant: stroke flow (preview → commit →
  `viewRefresh`) unchanged.

## 5. Tests

### `tests/unit/test_canvas.py`
- Rewrite the undo/redo section:
  - image strokes undo/redo via the draw stack; mask ops via the mask
    stack; interleaving one does not disturb the other's redo;
  - per-image histories: image A (upload bytes), stroke; switch to B;
    stroke; switch back to A → A's undo depth and pixels restored, B's
    stack intact;
  - draw stack cap `DRAW_HISTORY = 50`, mask cap `MASK_HISTORY = 200`;
  - eviction: an upload that leaves the canvas is dropped by
    `evict_stale()` (history gone); a job-anchored entity (`anchor({id})`)
    survives leaving the canvas until it is un-anchored;
  - mask scaling: mask 100×100 over a 200×100 image → `to_mask_png_bytes`
    is 200×100 with the mask at x=50..150 (fit_ar); a mask stroke at
    canvas (100, 50) lands on native (50, 50); dilate runs on native
    resolution;
  - `set_mask` with identical bytes pushes no step.
- Adapt existing stroke/morphology tests that touched `_undo` internals
  (lines ~207–281) to the new stacks.

### `tests/integration/test_canvas_api.py`
- Undo/redo endpoint with `target`; 400 on bad target.
- All mutating endpoints return the four flags (replace assertions on
  `undo`/`redo` keys).
- `/api/canvas/select` by id (result, upload, unknown → error).
- `/api/mask.png` dimensions follow the *image* size, not the mask's
  native size (generate/upload a second size, assert).
- `/api/canvas/clear` keeps the mask.
- Thumb selection preserves each image's draw history (stroke on A,
  select B, select A, undo → A's pre-stroke state).
- Rewrite `test_undo_redo_over_api_with_mask`: it used `/api/canvas/clear`
  as an undoable mutation (endpoint deleted) and expected image strokes to
  be globally undoable; replace the clear/undo tail with the per-target
  semantics (image stroke undoable via `target=draw`, mask via
  `target=mask`, each independent).

### `tests/unit/test_generation.py` + `tests/integration/test_generation_modes.py`
- `with_inpaint("inpaint_masked")` with a mask whose native size differs
  from the image (uses the scaled mask; bbox/blend at image size).
- "inpaint not masked": resulting mask = `plan.new_mask` (image/target
  size), one mask-history step on selection; repeated selection of the
  same result pushes no further step (change detection).
- Existing mask-retention assertions (e.g. `test_inpaint_masked_*`) stay
  true (mask is now trivially retained across image switches) — keep them
  as regression guards.

### `tests/ui/test_canvas_ui.py`
- Button labels follow the tool: after an image stroke in brush mode the
  buttons read "undo draw"/"redo draw"; pressing M flips them to
  "undo mask"/"redo mask"; enabled state per system (the existing
  dilate/clear/undo/redo block at ~lines 128–135 works unchanged in mask
  mode; add the label assertions and a brush-mode case where an image
  stroke does not enable "undo mask").

### `tests/unit/test_queue.py`
- New: finished jobs are dropped on the next `submit()`;
  `live_image_ids()` contains results + source ids; in-flight jobs are
  not dropped.

### Devtools
- `devtools/e2e_inpaint_ui_flow.py`: still valid as-is (mask retention now
  trivial; thumb selection via `/api/canvas/thumb` unchanged). Re-run it
  after the change to confirm end-to-end.

## 6. Implementation order

1. `canvas.py` + `app.py` + `generation.py` together (the endpoint
   signatures depend on the new Canvas API): `CanvasImage`, stacks, mask
   scaling/fit params, stroke remapping, entity mgmt, `prune`,
   `with_inpaint` on the scaled mask — with `test_canvas.py` and the
   integration tests.
2. `queue.py`: entities on result, `source_image_id`, job dropping +
   `prune`, WS `image_id` — with `test_queue.py`.
3. Frontend (`app.js`, `index.html`) — with the UI tests.
4. Full suite: `tests/unit`, `tests/integration`, `tests/ui` (Playwright),
   then `devtools/e2e_inpaint_ui_flow.py` for the live GPU pass.

## Notes

- All `Canvas` mutations stay under the single `RLock`; queue workers and
  request handlers already share it.
- `plan.base_image` still snapshots at submit time (generation
  overwrites in-flight edits *of that result's base* — the edits remain
  safe on the old entity).
- Brush size constants, toolbar CSS classes, and the stroke no-flicker
  invariant are untouched.
