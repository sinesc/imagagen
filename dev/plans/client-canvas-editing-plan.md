# Client-authoritative canvas editing — implementation plan

## Goal

Move brush editing (paint, mask add/erase, clone, smudge, undo/redo, mask
dilate/erode/clear) from the server to the client. The client already computes
the **pixel-exact commit result** (the preview layers *are* the commit), so
applying them locally costs nothing new. The server keeps its role as **store +
generation engine**; the client pushes the full canvas state to it at defined
sync points.

This eliminates, at the root:

- the per-stroke server round trip and the heavy refresh landing (full-PNG
  encode server-side + decode + mask pixel loop client-side per burst),
- the preview→commit pixel pop on refresh,
- the server-side brush cost that scales with image size (smudge's full
  float32 work copy; the full-canvas PIL PNG re-encode after **every** stroke
  op in `Canvas._stroke_*_locked` / `_stamp_mask`).

## New division of responsibilities

- **Client (editor):** owns the displayed entity's live state:
  `view.base` (offscreen canvas, W×H canvas px = image px, committed RGB) and
  `view.maskValues` (offscreen canvas W×H, grey 128, **alpha = mask value
  0..255**). All edit ops mutate these locally and synchronously.
- **Server (store):** entity lifecycle (upload, crop, results, select,
  thumbs, retention/eviction), generation (inpaint reads canvas at submit),
  segmentation (MediaPipe on the server image), save.
- **Sync (client → server, one-way, full state):**
  - `POST /api/canvas/sync` (new; multipart `image` + `mask` PNGs) replaces
    the displayed entity's state.
  - **Mandatory before** `/api/generate`, `/api/segmentation/mask`,
    `/api/save`, `/api/canvas/crop` (server-side reads of canvas state).
  - **Idle background sync:** 2 s after the last local edit (keeps the server
    fresh for reload/multi-tab).
  - **beforeunload:** best-effort `fetch(..., {keepalive: true})` if dirty.

## Server changes

### 1. New endpoint + `Canvas.replace_state` (add first, nothing else changes)

`src/imagagen/canvas.py`:

```python
def replace_state(self, img: np.ndarray, mask: np.ndarray) -> None:
    """Client-authoritative state push: replace the displayed entity's
    image and mask. PNG text chunks (metadata) are preserved."""
    with self._lock:
        ent = self._images[self._current]
        ent.image = img
        ent.mask = mask
        ent.png = _encode_png(img, ent.info)
```

`src/imagagen/canvas.py` — the client uploads `maskValues` as an RGBA PNG
with **alpha = value** (RGB is the constant grey 128), so `_decode_mask`
must take the alpha channel for such images (its own L-mode mask PNGs are
unchanged):

```python
def _decode_mask(data: bytes) -> np.ndarray:
    im = Image.open(BytesIO(data))
    if "A" in im.getbands():  # client value canvas: alpha = value
        im = im.getchannel("A")
    return np.asarray(im.convert("L"))
```

`src/imagagen/app.py` (next to the other canvas routes; import
`_decode_png`/`_decode_mask` from `.canvas`):

```python
@app.post("/api/canvas/sync")
async def canvas_sync(request: Request) -> dict:
    if not canvas.has_image():
        return JSONResponse({"error": "canvas is empty"}, status_code=400)
    form = await request.form()
    img_file, mask_file = form.get("image"), form.get("mask")
    if img_file is None or mask_file is None:
        return JSONResponse({"error": "image and mask required"}, status_code=400)
    try:
        img = _decode_png(await img_file.read())
        mask = _decode_mask(await mask_file.read())
    except Exception:
        return JSONResponse({"error": "could not read image"}, status_code=400)
    w, h = canvas.size
    if img.shape != (h, w, 3) or mask.shape != (h, w):
        return JSONResponse({"error": "size mismatch"}, status_code=400)
    canvas.replace_state(img, mask)
    return {"ok": True}
```

### 2. Removals (after the client has stopped calling them)

- **app.py endpoints:** `POST /api/canvas/stroke`, `/clone`, `/smudge`,
  `/mask`, `/undo`, `/redo`. **Body models:** `StrokeBody`, `CloneBody`,
  `SmudgeBody`, `MaskOpBody`, `UndoRedoBody`.
- **`canvas_flags()` helper** and the four flag fields in every response that
  carries them (`upload`, `thumb`, `select`, `crop`, `segmentation/mask`) and
  in the `canvas_set` WS event (`queue.py` ~line 519-523).
- **canvas.py methods:** `stroke`, `clone`, `smudge`, `_stroke_image_locked`,
  `_stroke_clone_locked`, `_stroke_smudge_locked`, `_stroke_mask_locked`,
  `_stamp_mask`, `_disk`, `mask_dilate`, `mask_erode`, `_mask_morphology`,
  `mask_clear`, `stamp_positions`, `undo`, `redo`, `can_undo_draw`,
  `can_redo_draw`, `can_undo_mask`, `can_redo_mask`, `_undo_draw_locked`,
  `_redo_draw_locked`, `_undo_mask_locked`, `_redo_mask_locked`,
  `_mask_snapshot_locked`, `_materialize_mask_locked`.
- **`CanvasImage` fields:** `undo`, `redo`, `mask_undo`, `mask_redo` (and all
  construction sites: `add_result` — `_inherit_mask_state_locked` then returns
  only the transformed current mask —, `import_image`, `crop`, and any test
  fixtures). Keep current-mask inheritance (transformed to result size); only
  the history-stack-by-reference plumbing goes.
- `set_mask` (used by segmentation) stays, minus its mask-history push.
- `list_brushes`, `brush_shape`, `brush_png`, `brush_cache` **stay** (the
  client still fetches brush shapes via `/api/brushes` and
  `/api/brush/{name}.png`).

## Client changes (static/canvas.js + static/app.js)

### 3. View state

Replace `view.img` (HTMLImageElement) and `view.mask` (0.9 overlay) with:

```js
view.base = null;      // offscreen canvas W×H (canvas px): committed image
view.maskValues = null; // offscreen canvas W×H: grey 128, alpha = mask value
view.mask = null;       // display overlay derived from maskValues
```

Remove `view.holdStrokes` and the entire deferred-refresh mechanism from the
previous commit: `STROKE_REFRESH_IDLE_MS`, `strokeRefreshTimer/Gen/InFlight/
Dirty`, `strokeSeq`, `committedSeq`, `commitChain`, `dropHeldStrokes`,
`cancelStrokeRefresh`, `enqueueCommit`, `scheduleStrokeRefresh`,
`fireStrokeRefresh`, and the `viewRefresh(options)` machinery
(`fromStroke`/`appliedSeq`/`gen`). Remove per-stroke `s.settings`
(the commit now happens synchronously at pointerup with the same live
settings the preview used; `s.work` and `s.img` stay).

`viewRefresh()` (unchanged call sites: boot, select, thumb, upload, crop,
`canvas_set` WS) now:

- fetches `/api/canvas.png` + `/api/mask.png` (as today),
- builds `view.base` (new canvas, `drawImage` of the decoded image) and
  `view.maskValues` via `maskValuesFromPng` (below), then `buildMaskOverlay()`,
- **resets the four local undo/redo stacks and the sync state**
  (`canvasDirty = false`, cancel `syncTimer`),
- keeps the existing fit/resize/cloneSource/crop logic; empty canvas clears
  all three canvases.

```js
// decoded mask Image (luminance = value) -> value canvas
function maskValuesFromPng(img) {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth; c.height = img.naturalHeight;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, c.width, c.height);
  const d = data.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i] = d[i + 1] = d[i + 2] = 128;
    d[i + 3] = d[i]; // alpha = luminance
  }
  ctx.putImageData(data, 0, 0);
  return c;
}
// 0.9 see-through overlay, GPU-only (replaces buildMaskLayer's pixel loop)
function buildMaskOverlay() {
  if (!view.maskValues) { view.mask = null; return; }
  const o = document.createElement("canvas");
  o.width = view.maskValues.width; o.height = view.maskValues.height;
  const ctx = o.getContext("2d");
  ctx.drawImage(view.maskValues, 0, 0);
  ctx.globalCompositeOperation = "destination-in"; // alpha *= 0.9
  ctx.globalAlpha = 0.9;
  ctx.fillRect(0, 0, o.width, o.height);
  view.mask = o;
}
```

`viewRedraw` draws `view.base` instead of `view.img`; the `maskLayered`
skip-rule becomes: skip the base overlay when the **active** stroke's tool is
`mask` or `smudge` (no held strokes anymore). `newSmudgeWork` copies
`view.base`. `drawPreview` becomes single-stroke again (`view.stroke` only).
`clearBrushStamps` moves back to `endStroke` (after the local commit).

### 4. Local stroke commit (endStroke v2)

No POST, no refresh. At pointerup, synchronously:

- **smudge:** `view.base.getContext("2d").drawImage(s.work, 0, 0)`.
- **clone / brush (paint):** run the existing preview branch of
  `drawStrokePreview` with the **base canvas's context** (identity
  transform, layers already in canvas px) instead of the screen ctx — the
  paint/clone branches blit their layer with `globalAlpha = st.opacity`,
  exactly the commit blend. (Refactor `drawStrokePreview(ctx, s)` so the
  paint/clone branches can be invoked with either ctx; the smudge and mask
  branches keep their own commit paths.)
- **mask add:** `m = max(m, cov × 255 × opacity)` per pixel over the stroke's
  stamp bbox, where `cov` is the existing `strokeMaxCoverage` canvas (alpha
  channel) — same math as the preview's overlay build, writing **values**
  into `view.maskValues` (getImageData/putImageData over the union bbox of
  footprints, the bbox walk already exists in the preview).
- **mask erase:** `m = m − covU × (m − min(m, cap))`, `cap = 255 ×
  (1 − opacity)`, `covU` = union stamp coverage (same as the preview's
  capped-copy math, values instead of the 0.9 overlay).

Then: push the snapshot (below), `markDirty()`, `syncUndoButtons()`,
`viewRedraw()`. The preview was showing exactly this result, so the swap is
invisible (the no-flicker hand-off problem no longer exists).

**Snapshots (undo input), taken at pointerdown** (overlaps the drag):
brush/clone/smudge → `s.snapP = snapCanvas(view.base)`; mask →
`s.snapMaskP = snapCanvas(view.maskValues)`. In `endStroke`: `await s.snapP`
and push onto `undoDraw`/`undoMask`.

### 5. Client undo/redo

```js
const UNDO_DRAW = 50, UNDO_MASK = 200; // same depths as the server had
const undoDraw = [], redoDraw = [], undoMask = [], redoMask = [];

async function snapCanvas(c) {
  const b = await c.toBlob("image/png");
  return new Uint8Array(await b.arrayBuffer());
}
async function restoreCanvas(c, bytes) {
  const img = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height); // drawImage composites
  ctx.drawImage(img, 0, 0);
  img.close();
}
// (for maskValues the alpha channel round-trips exactly; the grey RGB may
// drift ±1 through premultiplied storage -- irrelevant, only alpha is used)
```

- Push: `stack.push(snap); if (stack.length > DEPTH) stack.shift();` and clear
  the matching redo stack (one pair follows `Tools.current`, as today:
  mask tool → mask pair; brush/clone/smudge → draw pair).
- Undo: pop `stack`, push `await snapCanvas(current)` onto the redo stack,
  `restoreCanvas` into `view.base`/`view.maskValues`, `buildMaskOverlay()`
  for mask, `markDirty()`, `syncUndoButtons()`, `viewRedraw()`. Redo mirrors
  it. (Undo/redo make the state diverge from the server → idle sync
  re-pushes it.)
- Reset all four stacks in `viewRefresh` (entity switch/result/upload/crop).
- `syncUndoButtons()` reads the local stack lengths; **remove**
  `undoFlags`/`storeUndoFlags` from app.js and every `storeUndoFlags(...)`
  call site (the WS `canvas_set` handler keeps only `viewRefresh()`).

### 6. Mask morphology + clear (Web Worker)

New file `static/mask-morph.js` (served at `/mask-morph.js`; static dir is
mounted at `/`). Protocol: `{w, h, r, dilate, plane: Uint8Array}` (alpha
plane, transferred) → reply `{plane}`.

Disk is the server's old `_disk`: `dx² + dy² ≤ r²`, `r = round(px)`.
Row-band sliding-window extremum (deque), matching scipy's
`maximum_filter`/`minimum_filter` (reflect borders == clamped indexes for
max/min):

```js
self.onmessage = ({ data: m }) => {
  const { w, h, r, dilate } = m;
  const src = m.plane;
  const acc = new Uint8Array(w * h);
  if (dilate) acc.fill(0); else acc.fill(255);
  const win = new Uint8Array(w);
  const dq = new Int32Array(w); // deque of row indexes
  for (let dy = -r; dy <= r; dy++) {
    const halfw = Math.floor(Math.sqrt(r * r - dy * dy));
    for (let y = 0; y < h; y++) {
      const sy = Math.min(h - 1, Math.max(0, y + dy)); // clamped == reflect
      const row = sy * w;
      let head = 0, tail = 0;
      for (let x = 0; x < w; x++) {
        // window for output x is [x-halfw, x+halfw]: the newly entering
        // element is x+halfw (clamped at the right edge)
        const xi = Math.min(w - 1, x + halfw);
        while (
          tail > head &&
          (dilate
            ? src[row + dq[tail - 1]] <= src[row + xi]
            : src[row + dq[tail - 1]] >= src[row + xi])
        ) tail--;
        dq[tail++] = xi;
        while (head < tail && dq[head] < x - halfw) head++; // stale
        win[x] = src[row + dq[head]];
      }
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        acc[i] = dilate ? Math.max(acc[i], win[x]) : Math.min(acc[i], win[x]);
      }
    }
  }
  self.postMessage({ plane: acc }, [acc.buffer]);
};
```

Client op (replaces the `/api/canvas/mask` calls for dilate/erode/clear):

```js
async function maskMorph(dilate) {
  const px = Math.round(Number($("#mask-px").value));
  if (!px || !view.maskValues) return;
  const c = view.maskValues;
  const d = c.getContext("2d", { willReadFrequently: true })
             .getImageData(0, 0, c.width, c.height);
  const plane = new Uint8Array(c.width * c.height);
  for (let i = 0; i < plane.length; i++) plane[i] = d.data[i * 4 + 3];
  const snap = await snapCanvas(c); // BEFORE the mutation
  const res = await workerPost({ w: c.width, h: c.height, r: px,
                                 dilate, plane }, plane.buffer);
  for (let i = 0; i < plane.length; i++) d.data[i * 4 + 3] = res[i];
  c.getContext("2d").putImageData(d, 0, 0);
  pushMaskUndo(snap);
  buildMaskOverlay(); markDirty(); syncUndoButtons(); viewRedraw();
}
```

Same for clear (snapshot, then fill the plane with 0). Skip the op entirely when the mask is all-zero
(server parity: no change, no history step).

### 7. Sync wiring (canvas.js)

```js
let canvasDirty = false, syncTimer = null, syncPromise = null;
const canvasToBlob = (c) => c.toBlob("image/png");

async function forceSync() {
  if (!view.has || !canvasDirty || syncPromise) return syncPromise;
  if (syncTimer) { clearTimeout(syncTimer); syncTimer = null; }
  syncPromise = (async () => {
    try {
      const fd = new FormData();
      fd.append("image", await canvasToBlob(view.base), "canvas.png");
      fd.append("mask", await canvasToBlob(view.maskValues), "mask.png");
      const r = await fetch("/api/canvas/sync", { method: "POST", body: fd });
      if (r.ok) canvasDirty = false;
      // failure (size mismatch after an entity switch, server down):
      // stay dirty and retry on the next idle tick; viewRefresh() resets
      // the flag when a fresh server state lands
      if (canvasDirty) syncTimer = setTimeout(forceSync, 2000);
    } finally { syncPromise = null; }
  })();
  return syncPromise;
}
function markDirty() {
  canvasDirty = true;
  if (syncTimer) clearTimeout(syncTimer);
  syncTimer = setTimeout(forceSync, 2000);
}
```

`beforeunload` (app.js): if dirty, fire-and-forget the same POST with
`keepalive: true` (best effort — browsers often kill large requests on exit;
the 2 s idle sync is the real safety net).

**`await forceSync()` call sites in app.js, before the existing POST:**
- generate (app.js:857),
- segmentation (app.js:1553),
- save,
- crop commit (app.js:799).

Segmentation additionally: push a mask snapshot (local mask == synced state
after `forceSync`) **before** the POST, then `viewRefresh()` on success
(the server's unioned mask comes back via the fetch).

## Behavioural changes (accepted)

- **Reload/crash within ~2 s of the last edit** can lose unsynced strokes
  (beforeunload is best effort).
- **Multi-tab:** last sync wins (was shared server state). Single-user tool;
  documented limitation.
- **Server-side undo/redo history is gone**; the UI's undo/redo is fully
  client-side (session-scoped, reset on entity switch). Crop was never
  undo-able (new entity); no regression there.
- The four flag fields disappear from mutating responses and the
  `canvas_set` WS event.
- `POST /api/canvas/stroke|clone|smudge|mask|undo|redo` no longer exist.
- Dirty generate/segment/save/crop pays **one** full-canvas upload (~0.1–0.3 s
  at 5 MP) instead of per-stroke round trips.
- Risk to measure during implementation: `toBlob` snapshot cost per stroke on
  large canvases (browser PNG encodes are typically threaded; if jank shows
  up, cap `UNDO_DRAW` for large canvases).

## Tests

- **tests/unit/test_canvas.py:** delete stroke/clone/smudge/mask-stroke/
  morphology/undo-redo/`stamp_positions` tests; keep brush-shape, entity
  create/select/evict/anchor/pin, `add_result` (current-mask inheritance
  only), `crop`, `import_image`, `to_png_bytes`/`to_mask_png_bytes`,
  `_transform_mask`; update `CanvasImage` fixtures (no history fields).
  Add `replace_state` tests.
- **tests/integration/test_canvas_api.py:** delete the stroke/clone/smudge/
  mask/undo/redo endpoint tests (and their 422 value-validation tests);
  add `/api/canvas/sync` tests: ok (state replaced, info chunks preserved),
  empty canvas 400, missing part 400, size mismatch 400.
- **tests/ui/conftest.py:** `stroke()` and `dot()` end with
  `page.evaluate("forceSync()")` (Playwright awaits the returned promise) so
  the existing server-pixel assertions (`canvas_px`/`mask_arr`) stay valid
  unchanged.
- **tests/ui/test_canvas_ui.py:** delete
  `test_stroke_preview_held_until_server_response`,
  `test_cursor_tracks_without_flicker_during_commit`,
  `test_rapid_strokes_keep_previous_preview_until_refresh` (mechanism gone).
  Add `test_stroke_commits_locally_without_round_trip`: block
  `/api/canvas/sync` via the fetch wrapper, stroke, assert the pixel is on
  screen **immediately** and the server canvas is unchanged; unblock,
  `forceSync()`, assert the server pixel matches.
  `test_mask_tool_shift_erase_and_morph`: keep the assertions, ops are now
  client-side + `forceSync` via the conftest helpers (validates the worker
  end-to-end). Add client undo/redo tests (draw + mask: edit → pixel changes
  → undo → original pixel → redo) and a client mask-clear test.

## Commit order

1. **Server:** add `POST /api/canvas/sync` + `Canvas.replace_state` +
   unit/integration tests. (No behaviour change.)
2. **Client:** local state (`view.base`/`view.maskValues`/overlay), local
   stroke commits, local undo/redo, remove the deferred-refresh/holdStrokes
   mechanism, rewire app.js undo buttons + drop `storeUndoFlags`,
   conftest `forceSync`, rewrite the affected UI tests.
3. **Client:** mask morphology worker + local dilate/erode/clear,
   `forceSync` call sites (generate/segmentation/save/crop), idle sync,
   beforeunload.
4. **Server cleanup:** remove the six endpoints, body models, canvas.py
   brush/morphology/history code, `CanvasImage` history fields,
   `canvas_flags` and the flag fields in responses/WS event; delete the
   corresponding tests.
5. Update project memory (canvas architecture, test notes).

## Verification

- Unit + integration suites green after each commit.
- Targeted Playwright runs after step 2/3: the rewritten stroke tests, the
  mask tests, undo/redo, crop + thumb suites.
- Manual smoke on a large image (3440×1440): rapid brush/mask/clone/smudge
  strokes feel immediate with no pop; pausing settles locally (no server
  round trip visible); generate → source thumb shows the latest state;
  reload 2 s after the last edit keeps the state; reload 0.5 s after may
  lose the last stroke (accepted).
