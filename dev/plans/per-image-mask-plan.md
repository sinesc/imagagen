# Per-image mask history — implementation plan

Replaces the global, image-independent mask (one shared mask buffer + one
global undo/redo stack on `Canvas`) with a **per-image** mask: each
`CanvasImage` entity carries its own mask state (current mask + undo/redo
stacks), exactly like its draw stacks.

## Why (bug fixed)

With the global mask, selecting an outpaint result applied its plan mask to
the *global* mask, destroying the source image's mask: re-selecting the
source then displayed the plan mask scaled again ("broken" mask), and the
mask's native size (the outpaint size) no longer matched other images, so
brush strokes landed outside the fit-within remap and vanished (top/bottom
borders un-maskable, even on fresh images). Per-image masks fix both: every
image keeps its own mask, and each mask's native size equals its image's
size, so strokes map 1:1 and the whole canvas is always maskable.

## Invariants

- A mask's native size is always the size of its own image. Display,
  inpaint usage, and brush strokes all work in that size (no remap). The
  old fit-within display scaling and canvas->native stroke remap are gone.
- Selecting an image (thumb click / `/api/canvas/select`) is a pure entity
  switch: no mask is created, replaced, or scaled on selection. The old
  "apply `plan_mask` when the entity is entered" logic is deleted — the
  mask is baked in when the entity is created.
- A new entity inherits the mask state of the image it derives from:
  - generation result -> the job's source image (`job.source_image_id`,
    the green-bordered thumbnail),
  - upload -> the image currently displayed (the one being replaced).
- Inherited state = current mask + both history stacks.

## Inheritance mechanics

- **Current mask**: materialized at entity creation, transformed to the
  new image's size with the same mapping the image itself receives
  (fit-within, centered = `fit_ar` + `resize_shift_mask`; identity when
  the size is unchanged). For an `inpaint_not_masked` result this is
  exactly the plan mask, so the `plan_mask` field and its application are
  obsolete and deleted.
- **History frames**: inherited by reference — the new entity's
  `mask_undo`/`mask_redo` start as shallow copies of the source's frame
  lists. Frames are cold PNG bytes *at the source image's native size*;
  no resize work happens at creation. When a frame is popped (undo/redo),
  it is materialized: decoded, and if its size differs from the entity's
  image size it is transformed (fit-within, centered). Frames pushed later
  (snapshots of the current mask) are at the entity's own size, so the
  decoded frame's own size always decides which way it is materialized.
- No source (empty-canvas generate, source entity gone) -> fresh empty
  (zero) mask, empty stacks.
- Every entity's mask is always present (zero when nothing is masked) —
  the old "no mask" state is represented by an empty mask, so
  `GET /api/mask.png` is 200 as soon as any image is displayed (404 only
  for an empty canvas), and undoing the first mask write reverts to the
  all-zero mask rather than to "no mask".

## Resulting behaviour

- Original image's mask survives selection of any outpaint/inpaint result
  and is shown unmodified when the original is re-selected.
- Every image's mask covers its full size: the whole canvas is maskable on
  every image (no dead borders), and undo/redo in the mask tool steps
  through that image's own mask history (inherited from its source, so a
  fresh result can undo the strokes drawn on the source).
- Mask brush/expand/shrink/clear/segmentation operate on the displayed
  entity's mask (all at image size; the segmentation union is a direct
  element-wise max, no scaling).
- `GET /api/mask.png` serves the displayed entity's mask at the image size
  (404 only when the canvas is empty).
- Undo/redo API and the four response flags are unchanged; the flags now
  report the displayed entity's stacks.
- `POST /api/canvas/stroke` (mask): points are canvas-pixel coordinates,
  which now *are* mask-native coordinates (no remap, brush size unchanged).

## Code changes

- `canvas.py`:
  - `CanvasImage` gains `mask` (ndarray at the image's size, always
    present — zero when nothing is masked), `mask_undo`, `mask_redo`
    (PNG bytes); loses `plan_mask`.
  - `Canvas` loses `_mask`/`_mask_undo`/`_mask_redo`, `mask_scaled`,
    `_mask_fit_params_locked`, plan-mask-on-entry; `add_result(...,
    source_id)` and `import_image` build the inherited mask state;
    all mask ops run on the current entity; `_materialize` transforms
    popped frames.
- `generation.py`: `with_inpaint` reads `canvas.mask` (image size) instead
  of `canvas.mask_scaled()`; the plan still computes `new_mask` for the
  backend init mask. `postprocess` returns only the image (the returned
  mask was only used for the deleted `plan_mask`).
- `queue.py`: `add_result(..., source_id=job.source_image_id)`; the
  `canvas_set` WS event carries the four undo/redo flags.
- `app.py`: segmentation union uses `canvas.mask`; `canvas_set` response
  unchanged otherwise.
- `static/app.js`: `canvas_set` handler stores the event's flags
  (`storeUndoFlags`) instead of manually resetting the draw flags.

## Tests

- `tests/unit/test_canvas.py`: mask section rewritten for per-image
  semantics (entity fields, no global buffer), plus new tests: source
  mask survives result selection, result inherits source mask + history
  (undo on the result reverts a source stroke, source untouched),
  cross-size inheritance transforms mask and history frames, upload
  inherits the previous image's mask, strokes reach the canvas borders.
- `tests/integration/test_canvas_api.py`: existing upload-based tests now
  pass via inheritance (dot scales 2x across an upload); new API test:
  select back to a previous upload restores its own mask.
- Other suites (generation modes, segmentation, queue, UI) exercise the
  same endpoints and are expected to pass unchanged; the full unit +
  integration + UI suite is run.
