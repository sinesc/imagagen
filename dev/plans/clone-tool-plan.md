# Clone tool — implementation plan

Adds the **clone tool** from the initial concept: *shift-click to select a
source, drawing copies from the source offset by `current position` −
`drawing start position`*. The hard requirement is a **live-updating canvas
preview** — the copied pixels must appear under the cursor while the stroke
is in progress, not only after the server round-trip.

## What the tool does

- **Pick source:** with the clone tool active, **shift + left-click** a point
  `S` (canvas px). A crosshair marks `S`. Shift-click again to move it.
- **Clone:** a plain **left drag** starts a clone stroke. The drag start is
  `D0`; the offset `O = S − D0` is frozen for the whole stroke. For every point
  `C` along the stroke, the brush-shaped patch is **sampled from `S + (C − D0)`**
  (= `C + O`) and painted at `C`, blended by brush alpha × opacity.
- **Semantics:** a clone stroke samples from the **pre-stroke image** (the
  committed canvas at stroke start), not from the progressively edited image.
  That makes the operation deterministic and identical on client and server,
  which is what makes the live preview trustworthy.
- Undoable as a **draw** step (it modifies the image), on the current image's
  draw stack.

## Why the preview can be live (and flicker-free)

The canvas is server-authoritative, but the client already holds the
pre-stroke image on screen (`view.img`, loaded from the last committed
`/api/canvas.png`). A clone patch is a pure *copy from the same image*, so the
client can compute it locally by sampling `view.img` — no round-trip needed
while dragging. This is exactly the existing optimistic-preview model used by
the brush/mask tools (client previews, server commits, identical result), so
the **no-flicker invariant is preserved unchanged**: `endStroke` drops
`view.stroke` without redrawing and keeps the last rendered frame (the clone
preview over the old image) on screen until `viewRefresh()` lands the
committed pixels. The `view.refreshing` guard (skip `viewRedraw` on
pointermove/pointerleave) already covers it.

**Client/server parity** is what keeps the hand-off invisible:
- Both sample the *same* pre-stroke image (client `view.img` == server
  `ent.image` at stroke start — all prior strokes are committed).
- Both use the *same* stamp layout (`stamp_positions`, spacing `max(1, size/12)`).
- The offset `O` is rounded to **integer canvas px** on the client and sent
  whole; the server uses it as-is (sub-pixel sampling is the only
  client/server difference, and it is imperceptible — the same accepted
  approximation as the circle-vs-exact-brush preview today).
- Both blend with brush alpha × opacity. The client previews with a circular
  clip; the server stamps the exact brush shape. A custom-brush clone therefore
  previews as a disc and commits at full shape — identical to how the brush
  tool already behaves.

## Server (single source of truth)

`src/imagagen/canvas.py`
- New public op `Canvas.clone(points, offset, size, opacity, brush) -> bool`
  (acquires `self._lock`, validates, delegates to `_stroke_clone_locked`),
  mirroring `stroke`.
- `_stroke_clone_locked(points, size, opacity, brush, offset)`:
  - Require a current hot image (else `ValueError`, like `_stroke_image_locked`).
  - `src = ent.image` (pre-stroke, read-only during the loop),
    `img2 = ent.image.astype(np.float32)` (working copy), `a = brush_shape(...) * opacity`.
  - `Ox, Oy = round(offset[0]), round(offset[1])`.
  - For each stamp centre `C=(cx,cy)` from `stamp_positions(points, max(1, size/12))`:
    - destination footprint origin `(dx0,dy0) = (round(cx−sw/2), round(cy−sh/2))`.
    - Compute the shape-local sub-rect valid for **both** the destination and the
      source (= destination shifted by `(Ox,Oy)`):
      `j in [max(0,−dx0,−dx0−Ox), min(sw, w−dx0, w−dx0−Ox))`,
      `i in [max(0,−dy0,−dy0−Oy), min(sh, h−dy0, h−dy0−Oy))`.
    - If the sub-rect is non-empty, blend the whole block at once:
      `dst = img2[dy0+i0:dy0+i1, dx0+j0:dx0+j1]`,
      `s = src[dy0+i0+Oy:dy0+i1+Oy, dx0+j0+Ox:dx0+j1+Ox]`,
      `img2[...] = clip(dst*(1−a_sub) + s*a_sub, 0, 255)` with `a_sub = a[i0:i1,j0:j1][...,None]`.
    - Out-of-bounds source is skipped/clipped per stamp (no crash at edges).
  - If `img2` is unchanged → return `False`.
  - Otherwise record **one draw-history step** exactly like
    `_stroke_image_locked` (push `ent.png` onto `ent.undo`, cap at
    `DRAW_HISTORY`, clear `ent.redo`, set `ent.image`/`ent.png`).

`src/imagagen/app.py`
- New `CloneBody` (Pydantic): `points: list[list[float]]` (min 1),
  `offset: list[float]` (exactly 2), `size` / `opacity` / `brush` with the
  same bounds as `StrokeBody` (`MIN_BRUSH_SIZE`..`MAX_BRUSH_SIZE_ZOOMED`).
- `POST /api/canvas/clone` → calls `canvas.clone(...)`, returns
  `{"ok": True, **canvas_flags()}` (the four undo/redo flags, same as
  `/api/canvas/stroke`). `ValueError`/`KeyError` → 400, bad body → 422.

## Client

`src/imagagen/static/tools.js`
- `TOOLS = ["brush", "mask", "clone"]`.
- `state.clone = { size: 10, opacity: 1, brush: "circle", screen: false }`
  (no colour — clone copies pixels). Persisted with the rest in
  `localStorage "imagagen.tools"`, so it remembers its last brush/opacity.

`src/imagagen/static/canvas.js`
- `view.cloneSource: [x,y] | null` — the picked source `S` (canvas px). Reset
  to `null` when the displayed image changes (size/identity) in `viewRefresh`.
- `pointerdown` (button 0, clone tool):
  - `e.shiftKey` → set `view.cloneSource = p`, `viewRedraw()`, **no stroke**.
  - else → if a source is set, start `view.stroke = { tool:"clone", points:[p],
    size: brushSizeCanvas(), cloneOffset:[S−p] }` (offset frozen, canvas px,
    rounded to int); if no source, show a status hint and do not start a stroke.
- `drawPreview` — new **clone branch**: for each `stampPositions` centre `C`,
  circular-clip at `C` (radius `size/2`) and
  `ctx.drawImage(view.img, Cx+Ox−r, Cy+Oy−r, 2r,2r, Cx−r, Cy−r, 2r,2r)` at
  `globalAlpha = opacity`. (Off-image source is clipped by `drawImage`.)
- `viewRedraw` / cursor: when the clone tool is active draw a crosshair at
  `view.cloneSource`; while a clone stroke is in progress also draw a crosshair
  at the **live sample point** (cursor + offset) and a faint line between the
  two, so the user sees exactly where each pixel is being copied from.
- `endStroke` — branch on `view.stroke.tool`:
  - clone → `POST /api/canvas/clone { points, offset, size, opacity, brush }`;
  - otherwise the existing `/api/canvas/stroke` path.
  Both then follow the same no-flicker tail: `storeUndoFlags(resp)`,
  `view.stroke = null` (no redraw), `view.refreshing = true` around
  `await viewRefresh()`.

`src/imagagen/static/index.html`
- `<button id="tool-clone" class="tool mini" title="clone tool (C); shift-click to pick the source, then drag to copy">clone</button>`
  next to the brush/mask buttons.

`src/imagagen/static/app.js`
- `syncToolbar`: toggle `.active` on `tool-clone`; **disable the colour swatch**
  when the clone tool is active (it has no colour).
- `switchTool("clone")` from `wireToolbar` (`tool-clone.onclick`).
- Keyboard: `c`/`C` → `switchTool("clone")` (alongside `b`/`m`, only when not
  typing in a field).

## Notes on "implemented as a plugin"

The initial concept lists clone (and smudge) as plugin tools, but there is no
tool-plugin system in the codebase yet (only a comment in `sdxl_backend.py`
that samplers/schedulers will grow into plugins in Phase 5). This plan ships
clone as a **first-class tool**, built the same way as the existing
brush/mask tools (client tool + one standalone server op). Keeping the server
side a single self-contained `Canvas.clone` method means it can be lifted into
a tool-plugin system later without rework. **Building the generic tool-plugin
system is a separate follow-up, not part of this feature.** (Confirm if you'd
rather scaffold the plugin system first.)

## Tests

- `tests/unit/test_canvas.py` — new clone section:
  - a solid two-colour test image; clone a region from a source offset; assert
    the destination pixels now equal the source pixels (offset respected).
  - offset that pushes the sample off the top/left edge → clipped, no crash,
    in-bounds part correct.
  - opacity < 1 blends (intermediate value), opacity 1 is a hard copy.
  - samples the **pre-stroke** image: two overlapping stamps in one stroke both
    read the original (the second does not read the first's output).
  - records exactly one draw step; `undo("image")` reverts the clone; `redo`
    reapplies; `can_undo_draw`/`can_redo_draw` track it.
  - no-op clone (identical pixels) records no history step (returns `False`).
- `tests/integration/test_canvas_api.py` — `POST /api/canvas/clone` happy path
  returns the four flags and changes `/api/canvas.png`; bad `offset` length and
  out-of-bounds `size`/`opacity` → 422; unknown brush → 400.
- `tests/ui/test_canvas_ui.py` (Playwright) —
  - switch to clone tool, **shift-click** sets a source (crosshair appears,
    no stroke), then a drag copies: assert the committed `/api/canvas.png`
    changed where expected.
  - **live preview**: during the drag (before `pointerup`), the canvas
    element already shows the copied pixels under the cursor (sample a pixel
    via the page / a small screenshot region), proving the preview is live and
    not only post-commit.
  - the preview survives the commit with no flash (reuse the pattern of
    `test_stroke_preview_held_until_server_response` /
    `test_cursor_moves_do_not_redraw_during_commit`).
  - tool state (clone size/opacity/brush) persists across a reload.
- Run the targeted unit + integration + UI tests for canvas first, then the
  full unit + integration + UI suite (redirect output to a temp file to
  inspect).

## Smudge tool (future, same architecture)

The initial concept also calls for a **smudge** tool (smear/blur regions to
roughly fix problems before inpainting). It fits this same
server-authoritative + optimistic-preview architecture with **no change** to
the clone design. Two points for the future implementer:

- Smudge only samples from the current canvas image (no external source), so
  it is local like brush/clone — the client can preview it from `view.img`.
- Unlike brush/clone, smudge is **sequential within a stroke**: each stamp
  drags the pixels it just moved, so stamp N reads stamp N−1's output. The
  client preview must therefore be **incremental** — a persistent offscreen
  working canvas that accumulates the drag as the pointer moves — rather than
  recomputed from scratch each frame like `drawPreview` does today (this is
  also how a real smudge works). The server commit is the same sequential
  accumulation over a working copy.
- The algorithm (as implemented in `smudgeAccumulate`, preview and commit
  share it): for each consecutive pair of stamps, BLEND the brush-shaped
  footprint at the current centre toward the pixels it came from (the
  previous centre's positions), weighted by the brush shape and a per-stamp
  strength capped at half the opacity, accumulating on a copy of the
  pre-stroke image. The blend must stay PARTIAL: a rigid copy overwrites
  the pixel ahead of the brush (destination's leading edge is 1px past the
  copied region) and leaves a one-way valve, so a back-and-forth scrub
  could only push a boundary in the direction of the first leg (one side
  eats the other -- reported as "everything ends up dark"). A mirror blend
  on the footprint's leading edge closes the valve; scrubbing then
  converges to a blurred band centred on the original boundary.

Deliberately **not** moving to client-authoritative editing (client computes
pixels, uploads the result) to make previews exact: it would break the
server-is-the-source-of-truth invariant that inpaint/outpaint generation
depends on, move undo/redo off the server (lost on reload, harder to
reconcile with generation results), and add per-stroke image-upload
bandwidth — none of which the required preview accuracy needs.
