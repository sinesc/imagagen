# Thumbnail freshness & upload-entity eviction — plan

## Symptoms

1. Hovering a thumbnail (or its pin/ref checkbox) for an image that was edited on the
   canvas shows the pre-edit image in both the strip thumbnail and the hover preview.
2. A PNG dropped onto the canvas gets an `upload:{n}` thumbnail that becomes
   unclickable once another thumbnail is selected and it is clicked again:
   status shows `unknown canvas image: 'upload:18'`.

## Root causes

**A. Upload entities are evicted on the first thumbnail switch** (`src/imagagen/canvas.py`).
`import_image()` registers the entity and makes it current, but it belongs to none of
`_anchored` / `_pinned` / `_user_pinned`. Any later `set_current()` runs
`_enter_entity_locked()` → `_evict_stale_locked()`, which deletes it. Crop entities survive
because `crop()` pins both sides via `_pin_locked()`; uploads never got that.
(The existing drop test only survives because after a generation the upload is the source
of a retained job, i.e. anchored.)

**B. Thumbnails/previews of the edited image read the server's cold PNG.**
Strokes commit locally (`commitStroke` → `markDirty()`); the server copy updates only on
the 2 s idle sync / `forceSync()`. Thumbnail `<img>`s and `thumbPreviewShow(t.url)` point
at `/api/canvas/image.png?id=…` (or `/api/thumb/{job}/{i}.png`) — a fixed URL that is
never reloaded, and Chromium memory-caches the first bitmap anyway. So the strip and the
preview show the image as it was when the page first decoded that URL.

**C. Entity switches never sync.** `restoreThumb()` (app.js) calls `/api/canvas/select`
without a preceding `forceSync()`: edits made within the idle window are neither pushed
to the old entity's PNG nor visible in its thumbnail.

## Fix 1 — server: keep upload entities alive (canvas.py + app.py + app.js)

Crop-style lifecycle: pinned in `_pinned` (cleared by `release_pins()` at the start of
every generation, exactly when the client drops unpinned thumbnails).

1. `Canvas.import_image(data, replace_id: str | None = None)`:
   - before `_enter_entity_locked(eid)`: `self._pin_locked(eid)`.
   - replacement (the client re-points an upload thumbnail instead of appending, because
     the canvas only tracks one canvas-upload entity): if `replace_id` names a known
     entity, hand over the user pin and un-pin the old one so `_evict_stale_locked()`
     (already invoked by `_enter_entity_locked`) drops it — skip the removal entirely if
     `replace_id` is anchored (a live job's source keeps its PNG until the job drops):
     ```python
     if (replace_id and replace_id != eid and replace_id in self._images
             and replace_id not in self._anchored):
         was_user = replace_id in self._user_pinned
         self._user_pinned.discard(replace_id)
         self._pinned.discard(replace_id)
         if was_user:
             self._user_pinned.add(eid)  # the re-pointed thumb stays pinned
     ```
2. `POST /api/canvas/upload`: pass through an optional query param —
   `canvas.import_image(data, replace_id=request.query_params.get("replace"))`.
3. `wireCanvasDrop()` (app.js): the "second drop replaces the upload thumbnail" rule
   applies only when that thumbnail IS the entity currently on the canvas (see
   `currentEntityId` in Fix 2) — otherwise a strip-dropped pinned upload sitting last in
   the strip would be wrongly replaced. Send the replaced id to the server:
   ```js
   const last = state.thumbs[state.thumbs.length - 1];
   const replace = last?.upload && last.image_id === currentEntityId
     ? last.image_id : null;
   const up = await fetch("/api/canvas/upload" +
     (replace ? `?replace=${encodeURIComponent(replace)}` : ""),
     { method: "POST", body: await file.arrayBuffer() });
   // ...on replace: last.image_id = image_id; last.url = url;  (keep last.pinned)
   // ...else: state.thumbs.push({ image_id, upload: true, url });
   ```
   (`import_pinned()` already pins to `_pinned` + `_user_pinned` — unchanged.)

## Fix 2 — client: thumbnails track local edits (app.js + canvas.js)

Since the client is authoritative for the displayed image, the current entity's
thumbnail is refreshed from local state on every pixel change; the hover preview reads
`t.url`, so it inherits the fix. No URL re-fetching, no cache-busting, works during the
unsynced window.

1. **Track the displayed entity** — module-level `let currentEntityId = null;` in app.js,
   updated at every point the canvas entity changes:
   - boot: `restoreQueueThumbs(q)` → `currentEntityId = q.current ?? null`
     (`/api/queue` already returns `current`),
   - `restoreThumb()` after a successful select → `t.image_id`,
   - `wireCanvasDrop()` → the new `image_id`,
   - `addCropThumbs()` → `r.crop_image_id`,
   - WS `case "canvas_set"` → `job:${msg.job}:${msg.index}` (before `viewRefresh()`).
   (`import-pin` drops and generations that keep the canvas image do NOT change it.)
2. **Refresh the thumbnail from `view.base`** — new app.js function:
   ```js
   function refreshCurrentThumbPixels() {
     if (!view.has || !currentEntityId) return;
     const t = state.thumbs.find((x) => x.image_id === currentEntityId);
     if (!t) return;
     view.base.toBlob((b) => {
       if (!b) return;
       const prev = t.url, url = URL.createObjectURL(b);
       t.url = url;
       const i = state.thumbs.indexOf(t);
       const img = $("thumbs").querySelectorAll(".thumb-item img")[i];
       if (img) img.src = url;               // targeted: no full renderThumbs()
       if (thumbPreviewUrl === prev) {       // preview is open on this thumb
         if (prev.startsWith("blob:")) URL.revokeObjectURL(prev);
         thumbPreviewShow(url);
       } else if (prev.startsWith("blob:")) {
         URL.revokeObjectURL(prev);
       }
     }, "image/png");
   }
   ```
   Blob sizes are trivial (a canvas PNG); revoke every superseded blob URL (only the
   current entity ever gets blob thumbs).
3. **Hook it to base-image edits only** — `markDirty(pixels = false)` in canvas.js ends
   with `if (pixels) refreshCurrentThumbPixels();` (canvas.js already calls app.js
   globals like `syncUndoButtons`; scripts load in that order). Call `markDirty(true)` at
   the three sites that mutate `view.base`:
   - `commitStroke` non-`s.sel` branch: `markDirty(s.tool !== "mask")` (the draw-family
     tools are `brush`/`clone`/`smudge`; plain `markDirty()` for mask strokes),
   - `undoStep`/`redoStep` when `target === "draw"`,
   - `commitSelection()`.
4. **Sync before every entity switch** — in `renderThumbs()`'s `image.onclick`:
   `await forceSync();` before `restoreThumb(t)` (no-op when clean). This keeps the old
   entity's server PNG current, so that entity's (still local-blob or later server-URL)
   thumbnail matches it.
5. **Server thumbnails after reload** — in `GET /api/thumb/{job}/{index}.png`, prefer the
   live canvas entity when it exists (the entity is authoritative once synced, and
   replace_state updates its cold PNG):
   ```python
   data = canvas.entity_png(f"job:{job_id}:{index}")
   if data is None and job is not None:
       data = job.results.get(index)
   ```
   Same swap for `/api/thumb/{job}/source.png` using `job.source_image_id` first,
   falling back to `job.source_png`.

With 1+2 the current entity's thumb/preview are always live; with 4, after a switch the
other entity's thumbnail keeps its fresh blob (identical to the server copy) and reloads
rebuild it from the now-current server PNG.

## Tests

Integration (`tests/integration/test_canvas_api.py`):
- `test_upload_entity_survives_switching`: upload, `import-pin` another PNG,
  `select` the pinned one, `select` `upload:0` again → 200 and `/api/canvas.png` has the
  upload's shape (today: 400 `unknown canvas image`).
- `test_upload_replace_evicts_superseded`: upload A, upload B with
  `?replace=upload:0` → `/api/canvas/image.png?id=upload:0` → 404, `upload:1` → 200;
  user-pin transfer: pin `upload:0`, replace it, assert the new id is in
  `/api/queue`'s `pinned` and select still works.

Unit (`tests/unit/test_canvas.py`): direct `import_image()` / `set_current()` round-trip
and the `replace_id` pin-transfer logic (anchored replace leaves the entity alone).

UI (`tests/ui/`, mock server like existing `test_canvas_ui.py` drop tests):
- `test_upload_thumbnail_restorable_after_switching`: drop a PNG on `#thumbs`
  (pinned upload), drop another on `#canvas-holder`, click the strip drop's thumbnail,
  click the canvas drop's thumbnail again → no status error, canvas pixels are the
  dropped PNG (this is the reported `upload:18` repro).
- `test_second_drop_after_strip_drop_appends_not_replaces`: strip drop last, canvas
  drop → two upload thumbnails, old one still selectable.
- `test_thumb_and_preview_track_local_edits` (in `test_thumb_preview_ui.py`): generate,
  draw a stroke on the canvas with the current thumb selected, wait ~300 ms, hover the
  thumb: its `src` is a `blob:` URL, `#thumb-preview-img` has the same src, and drawing
  it into an offscreen canvas in `page.evaluate` shows the stroke's colour at the stroke
  pixel.
- `test_edits_survive_thumbnail_switch`: generate (source thumb), draw a distinctive
  stroke, click the result thumb, click the source thumb back → `canvas_px(base, x, y)`
  at the stroke still shows the stroke colour on the SERVER (proves the pre-select
  `forceSync`).

## Verification

```
uv run pytest tests/unit/test_canvas.py tests/integration/test_canvas_api.py
uv run pytest tests/ui/test_canvas_ui.py -k "drop or upload or thumb or edit"
uv run pytest tests/ui/test_thumb_preview_ui.py
```
Also run ruff per the project lint conventions, then commit (past-tense message, e.g.
"Fixed upload thumbnail eviction and stale thumbnails of locally edited images").
