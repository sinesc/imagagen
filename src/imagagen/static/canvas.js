/* Canvas view: renders the canvas image with a transparent grey mask
   overlay, handles zoom (wheel, cursor-anchored) and pan (middle-button
   drag) as pure client-side transforms, and captures left pointer strokes
   as point lists in canvas-pixel coordinates.

   The client is the editor: stroke/mask commits are applied locally and
   synchronously to view.base / view.maskValues (the preview IS the commit
   result, so the hand-off is invisible), and undo/redo + the full-canvas
   server sync are client-side. The server is the store -- it keeps a copy
   of the displayed entity for generation, segmentation, save and crop --
   and the client pushes the state to /api/canvas/sync before those reads
   and on an idle timer (see the sync section below). */
"use strict";

const view = {
  base: null, // offscreen canvas (W×H canvas px): the committed image
  maskValues: null, // offscreen canvas (W×H): grey 128, alpha = mask value
  mask: null, // display overlay derived from maskValues (0.9 see-through)
  has: false,
  w: 0,
  h: 0,
  scale: 1, // screen = canvas * scale + o (CSS px)
  ox: 0,
  oy: 0,
  dpr: 1,
  stroke: null, // {tool, points: [[x,y]...], erase, cloneOffset, smudgeStamps}
  cloneSource: null, // [x,y] canvas px: clone tool's picked source S
  cloneGesture: null, // [ox,oy] canvas px: CTRL held -> the source follows
  // the cursor at this relative offset (a temporary move of the picked S)
  crop: null, // {sx, sy, ex, ey} raw canvas px of the crop tool's drag; the
  // region used is its clamped intersection with the image (cropRect)
  pan: null,
  cursor: null, // [mx,my] in CSS px relative to the canvas element
};

const cv = () => $("canvas");

// content box of the canvas holder (clientWidth includes its padding)
function holderBox() {
  const holder = $("canvas-holder");
  const cs = getComputedStyle(holder);
  return {
    w: holder.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight),
    h: holder.clientHeight - parseFloat(cs.paddingTop) - parseFloat(cs.paddingBottom),
  };
}

// ---------------- rendering ----------------

function viewResize() {
  const { w, h } = holderBox();
  view.dpr = window.devicePixelRatio || 1;
  const bw = Math.max(1, Math.floor(w * view.dpr));
  const bh = Math.max(1, Math.floor(h * view.dpr));
  if (bw !== cv().width || bh !== cv().height) {
    cv().width = bw;
    cv().height = bh;
  }
  viewRedraw();
}

// The holder box can change WITHOUT a window resize: dragging the sidebar
// resizer, the persisted sidebar width arriving late from /api/ui-state
// (the restore lands after boot, racing the canvas image load), a manually
// resized prompt textarea, toolbar reflow. #canvas is a replaced element
// clamped to the holder (max-width/max-height: 100%), so a backing store
// sized for the old box is silently rescaled to fit the new one -- the
// whole render (image, brush outline, strokes) drifts from the mouse by a
// factor of distance-from-top-left. Keep the backing store in step.
// (document.getElementById: canvas.js is parsed before app.js defines $)
new ResizeObserver(viewResize).observe(document.getElementById("canvas-holder"));

function viewFit() {
  if (!view.has) return;
  const { w, h } = holderBox();
  const s = Math.min(w / view.w, h / view.h) * 0.98;
  view.scale = s;
  view.ox = (w - view.w * s) / 2;
  view.oy = (h - view.h * s) / 2;
  viewRedraw();
}

function viewZoomAt(factor, mx, my) {
  const ns = Math.min(64, Math.max(1 / 512, view.scale * factor));
  const k = ns / view.scale;
  view.ox = mx - (mx - view.ox) * k;
  view.oy = my - (my - view.oy) * k;
  view.scale = ns;
  viewRedraw();
}

function viewRedraw() {
  const c = cv();
  const ctx = c.getContext("2d");
  $("canvas-empty").hidden = view.has;
  c.hidden = !view.has;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, c.width, c.height);
  if (!view.has) {
    $("zoom-label").textContent = "";
    return;
  }
  ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
  ctx.imageSmoothingEnabled = view.scale < 4; // crisp pixels up close
  ctx.translate(view.ox, view.oy);
  ctx.scale(view.scale, view.scale);
  if (view.base) ctx.drawImage(view.base, 0, 0);
  // a mask stroke (add or erase) and smudge each draw their own full copy
  // of the mask inside drawStrokePreview (add: base + new, erase: capped,
  // smudge: base on top of the working image); drawing the base mask
  // underneath would show through and double the opacity of the existing
  // mask -- so skip it while such a stroke is active
  const s0 = view.stroke;
  const maskLayered = !!s0 && (s0.tool === "mask" || s0.tool === "smudge");
  if (view.mask && !maskLayered) ctx.drawImage(view.mask, 0, 0, view.w, view.h);
  if (view.stroke) drawStrokePreview(ctx, view.stroke);
  if (view.crop) drawCropOverlay(ctx);
  drawCursor(ctx);
  drawCloneCrosshairs(ctx);
  $("zoom-label").textContent = `${Math.round(view.scale * 100)}%`;
}

// decoded mask PNG (greyscale, luminance = value) -> value canvas
function maskValuesFromPng(img) {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const data = ctx.getImageData(0, 0, c.width, c.height);
  const d = data.data;
  for (let i = 0; i < d.length; i += 4) {
    const a = d[i]; // greyscale: R = luminance
    d[i] = d[i + 1] = d[i + 2] = 128;
    d[i + 3] = a; // alpha = luminance
  }
  ctx.putImageData(data, 0, 0);
  return c;
}

// display overlay: constant grey with alpha = mask alpha x 0.9, derived
// GPU-only (destination-in alpha scale -- no per-pixel loop). The grey fill
// is the COLOUR source: maskValues only contributes its alpha. (Tinting
// from maskValues' own pixels would render black where alpha was 0 -- those
// pixels are stored premultiplied with RGB 0 even after a morph raises the
// alpha, since the write-back only touches the alpha channel.)
function buildMaskOverlay() {
  if (!view.maskValues) {
    view.mask = null;
    return;
  }
  const o = document.createElement("canvas");
  o.width = view.maskValues.width;
  o.height = view.maskValues.height;
  const ctx = o.getContext("2d");
  ctx.fillStyle = "rgb(128,128,128)";
  ctx.fillRect(0, 0, o.width, o.height);
  ctx.globalCompositeOperation = "destination-in"; // alpha = mask x 0.9
  ctx.globalAlpha = 0.9;
  ctx.drawImage(view.maskValues, 0, 0);
  view.mask = o;
}

// fetch image + mask from the server and re-render. The server's state
// REPLACES the local state: a fresh entity (entity switch, generation
// result, upload, crop, boot) or a server-side op (segmentation) landed, so
// the local undo/redo stacks reset (client history is per-entity) and the
// sync state resets (we ARE the server state now). A segmentation op stashes
// its pre-op mask snapshot first (pendingMaskUndo) so the step stays
// undoable across the refresh.
async function viewRefresh() {
  const [ri, rm] = await Promise.all([
    fetch("/api/canvas.png"),
    fetch("/api/mask.png"),
  ]);
  const had = view.has;
  const oldW = view.w;
  const oldH = view.h;
  undoDraw.length = redoDraw.length = undoMask.length = redoMask.length = 0;
  if (pendingMaskUndo) {
    undoMask.push(pendingMaskUndo);
    pendingMaskUndo = null;
  }
  cancelSync();
  view.has = ri.ok;
  if (!view.has) {
    view.base = null;
    view.maskValues = null;
    view.mask = null;
    view.w = view.h = 0;
    clearBrushStamps();
    viewRedraw();
    return;
  }
  const img = new Image();
  img.src = URL.createObjectURL(await ri.blob());
  await new Promise((r) => {
    img.onload = r;
  });
  const base = document.createElement("canvas");
  base.width = img.naturalWidth;
  base.height = img.naturalHeight;
  base.getContext("2d").drawImage(img, 0, 0);
  view.base = base;
  view.w = img.naturalWidth;
  view.h = img.naturalHeight;
  view.maskValues = null;
  if (rm.ok) {
    const m = new Image();
    m.src = URL.createObjectURL(await rm.blob());
    await new Promise((r) => {
      m.onload = r;
    });
    view.maskValues = maskValuesFromPng(m);
  }
  buildMaskOverlay();
  // keep the current zoom/pan unless the image appeared or changed size
  if (!had || view.w !== oldW || view.h !== oldH) {
    view.cloneSource = null; // old source point no longer makes sense
    view.cloneGesture = null;
    viewResize(); // the canvas element may have been hidden (sized to 0)
    viewFit();
  }
  clearBrushStamps();
  viewRedraw();
}

// ---------------- server sync ----------------
//
// The client is the editor and the server the store: generation,
// segmentation, save and crop read the server's copy, so the client pushes
// the full state (image + mask) to /api/canvas/sync -- awaited by app.js
// right before those reads, on an idle timer SYNC_IDLE_MS after the last
// edit (keeps the server fresh for reload / multi-tab), and best-effort on
// beforeunload (app.js).
const SYNC_IDLE_MS = 2000;
let canvasDirty = false; // local state has diverged from the server's
let syncTimer = null; // the pending idle sync
let syncPromise = null; // at most one sync outstanding

async function forceSync() {
  if (!view.has || !canvasDirty || syncPromise) return syncPromise;
  if (syncTimer) {
    clearTimeout(syncTimer);
    syncTimer = null;
  }
  syncPromise = (async () => {
    try {
      const fd = new FormData();
      fd.append("image", await blobP(view.base), "canvas.png");
      fd.append("mask", await blobP(view.maskValues), "mask.png");
      const r = await fetch("/api/canvas/sync", { method: "POST", body: fd });
      if (r.ok) canvasDirty = false;
      // failure (an entity switch changed the size underneath us, the
      // server is down): stay dirty and retry on the next idle tick;
      // viewRefresh() clears the flag when fresh server state lands
      if (canvasDirty) syncTimer = setTimeout(forceSync, SYNC_IDLE_MS);
    } finally {
      syncPromise = null;
    }
  })();
  return syncPromise;
}

function markDirty() {
  canvasDirty = true;
  if (syncTimer) clearTimeout(syncTimer);
  syncTimer = setTimeout(forceSync, SYNC_IDLE_MS);
}

// fresh server state is about to replace the local state (viewRefresh): we
// are the server state now, so the dirty flag and any pending sync clear
function cancelSync() {
  canvasDirty = false;
  if (syncTimer) {
    clearTimeout(syncTimer);
    syncTimer = null;
  }
}

// ---------------- local undo/redo ----------------
//
// The server used to own the per-image history; the client owns it now: PNG
// snapshots of view.base (draw) / view.maskValues (mask) taken BEFORE each
// edit (at pointerdown, so the encode overlaps the drag), same depths the
// server had. viewRefresh resets the stacks (new entity / server state).
const UNDO_DRAW = 50;
const UNDO_MASK = 200;
const undoDraw = [], redoDraw = [], undoMask = [], redoMask = [];
let pendingMaskUndo = null; // a pre-op mask snapshot that viewRefresh pushes
// onto undoMask instead of dropping it (segmentation / server-side mask ops)

// toBlob as a promise (this Chromium requires the callback form -- the
// promise form throws "parameter 1 is not of type 'Function'")
function blobP(c) {
  return new Promise((resolve, reject) =>
    c.toBlob((b) => (b ? resolve(b) : reject(new Error("toBlob failed"))),
             "image/png"));
}

function snapCanvas(c) {
  return blobP(c).then((b) => b.arrayBuffer()).then((buf) => new Uint8Array(buf));
}

async function restoreCanvas(c, bytes) {
  const img = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height); // drawImage composites
  ctx.drawImage(img, 0, 0);
  img.close();
}

function pushUndo(tool, snap) {
  const mask = tool === "mask";
  const undo = mask ? undoMask : undoDraw;
  const redo = mask ? redoMask : redoDraw;
  undo.push(snap);
  if (undo.length > (mask ? UNDO_MASK : UNDO_DRAW)) undo.shift();
  redo.length = 0;
}

function stashMaskUndo(snap) {
  pendingMaskUndo = snap;
}

async function undoStep(target) { // "draw" | "mask"
  const undo = target === "mask" ? undoMask : undoDraw;
  const redo = target === "mask" ? redoMask : redoDraw;
  const prev = await undo.pop(); // stack entries may be snapshot promises
  if (prev === undefined) return;
  const c = target === "mask" ? view.maskValues : view.base;
  redo.push(await snapCanvas(c));
  await restoreCanvas(c, prev);
  if (target === "mask") buildMaskOverlay();
  markDirty();
  syncUndoButtons(); // app.js
  viewRedraw();
}

async function redoStep(target) { // "draw" | "mask"
  const undo = target === "mask" ? undoMask : undoDraw;
  const redo = target === "mask" ? redoMask : redoDraw;
  const nxt = await redo.pop(); // stack entries may be snapshot promises
  if (nxt === undefined) return;
  const c = target === "mask" ? view.maskValues : view.base;
  undo.push(await snapCanvas(c));
  await restoreCanvas(c, nxt);
  if (target === "mask") buildMaskOverlay();
  markDirty();
  syncUndoButtons(); // app.js
  viewRedraw();
}

// ---------------- mask morphology (worker) ----------------
// The old server's /api/canvas/mask (scipy max/min filter over the disk)
// runs client-side in a worker with the same disk, so the result is the
// same; the op lands locally + one history step, like a stroke.
let morphWorker = null;
function workerPost(msg, transfer) {
  if (!morphWorker) morphWorker = new Worker("/mask-morph.js");
  return new Promise((resolve, reject) => {
    const onMsg = (e) => {
      morphWorker.removeEventListener("message", onMsg);
      resolve(e.data.plane);
    };
    const onErr = (e) => {
      morphWorker.removeEventListener("message", onMsg);
      reject(e.error || e);
    };
    morphWorker.addEventListener("message", onMsg);
    morphWorker.addEventListener("error", onErr);
    morphWorker.postMessage(msg, transfer);
  });
}

async function maskMorph(dilate) {
  const px = Math.round(Number($("mask-px").value));
  if (!px || !view.maskValues) return;
  const c = view.maskValues;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  const d = ctx.getImageData(0, 0, c.width, c.height);
  const plane = new Uint8Array(c.width * c.height);
  for (let i = 0; i < plane.length; i++) plane[i] = d.data[i * 4 + 3];
  const snap = await snapCanvas(c); // BEFORE the mutation
  const res = await workerPost({ w: c.width, h: c.height, r: px,
                                 dilate, plane }, [plane.buffer]);
  for (let i = 0; i < res.length; i++) d.data[i * 4 + 3] = res[i];
  ctx.putImageData(d, 0, 0);
  pushUndo("mask", snap);
  buildMaskOverlay();
  markDirty();
  syncUndoButtons(); // app.js
  viewRedraw();
}

async function maskClear() {
  const c = view.maskValues;
  if (!c) return;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  const d = ctx.getImageData(0, 0, c.width, c.height);
  let any = false;
  for (let i = 3; i < d.data.length; i += 4) {
    if (d.data[i]) { any = true; break; }
  }
  if (!any) return; // nothing to clear (no history step, like the old server)
  const snap = await snapCanvas(c);
  for (let i = 3; i < d.data.length; i += 4) d.data[i] = 0;
  ctx.putImageData(d, 0, 0);
  pushUndo("mask", snap);
  buildMaskOverlay();
  markDirty();
  syncUndoButtons(); // app.js
  viewRedraw();
}

// ---------------- optimistic stroke preview ----------------

// even stamp centres along the stroke: consecutive stamps are at least
// `spacing` apart (1-D Poisson disk), the last point is always stamped
function stampPositions(points, spacing) {
  if (!points.length) return [];
  if (points.length === 1) return [points[0]];
  const out = [points[0]];
  let acc = 0;
  let last = points[0];
  for (const p of points.slice(1)) {
    let seg = Math.hypot(p[0] - last[0], p[1] - last[1]);
    while (seg >= spacing - acc) {
      const t = seg > 0 ? (spacing - acc) / seg : 1;
      last = [last[0] + t * (p[0] - last[0]), last[1] + t * (p[1] - last[1])];
      out.push(last);
      seg = Math.hypot(p[0] - last[0], p[1] - last[1]);
      acc = 0;
    }
    acc += seg;
    last = p;
  }
  return out;
}

// footprint of the current brush in canvas px: in screen mode the slider
// value is screen px (constant while zooming), so divide by the zoom
function brushSizeCanvas() {
  const st = Tools.get();
  return st.screen ? st.size / view.scale : st.size;
}

// ---------------- brush shapes ----------------
// The commit stamps with the EXACT brush shape: "circle" is built in, other
// brushes are greyscale PNGs (luminance = alpha) resized to the stamp size.
// The preview uses the same pixel data so a soft brush previews soft: the
// brush's PNG bytes come from /api/brush/{name}.png once per name, and the
// luminance is converted to alpha the same way as the mask values
// (maskValuesFromPng).
const brushImages = {}; // name -> {canvas, w, h, ok}
async function loadBrushImage(name) {
  const entry = brushImages[name];
  if (entry) return entry;
  const e = { canvas: null, w: 0, h: 0, ok: false };
  brushImages[name] = e;
  try {
    const r = await fetch(`/api/brush/${encodeURIComponent(name)}.png`);
    if (!r.ok) return e;
    const img = new Image();
    img.src = URL.createObjectURL(await r.blob());
    await new Promise((res) => {
      img.onload = res;
    });
    e.w = img.naturalWidth;
    e.h = img.naturalHeight;
    const base = document.createElement("canvas");
    base.width = e.w;
    base.height = e.h;
    const bctx = base.getContext("2d", { willReadFrequently: true });
    bctx.drawImage(img, 0, 0);
    const imgData = bctx.getImageData(0, 0, e.w, e.h);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      const a = d[i]; // greyscale: R = luminance
      d[i] = d[i + 1] = d[i + 2] = 128;
      d[i + 3] = a;
    }
    bctx.putImageData(imgData, 0, 0);
    e.canvas = base;
    e.ok = true;
  } catch {
    // keep ok=false: previews fall back to the disc
  }
  return e;
}

// the brush stamp at a concrete footprint: the base shape scaled to
// (w, h) -- largest dim = round(size), aspect preserved (exactly
// Canvas.brush_shape) -- optionally source-in filled so the alpha stays
// the shape and the colour becomes `fill` (an rgba() with alpha < 1 scales
// the alpha, e.g. the mask overlay's 0.9)
const brushStamps = {}; // `${name}@${w}x${h}:${fill}` -> canvas
function brushStamp(name, size, fill) {
  const e = brushImages[name];
  if (!e || !e.ok) return null;
  const scale = Math.max(1, Math.round(size)) / Math.max(e.w, e.h);
  const w = Math.max(1, Math.round(e.w * scale));
  const h = Math.max(1, Math.round(e.h * scale));
  const key = `${name}@${w}x${h}:${fill || ""}`;
  const hit = brushStamps[key];
  if (hit) return hit;
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(e.canvas, 0, 0, w, h);
  if (fill) {
    ctx.globalCompositeOperation = "source-in";
    ctx.fillStyle = fill;
    ctx.fillRect(0, 0, w, h);
  } else {
    // the raw shape's pixel data, for incremental max-coverage builds
    c.alphaData = ctx.getImageData(0, 0, w, h);
  }
  // keyed by (name, size, fill): ctrl-wheel size storms can fill it up,
  // so bound it with a full clear
  if (Object.keys(brushStamps).length > 128) clearBrushStamps();
  brushStamps[key] = c;
  return c;
}

function clearBrushStamps() {
  for (const k in brushStamps) delete brushStamps[k];
}

// brush footprint outline: dark halo + thin white ring, visible on any
// background, 1-screen-px wide
function drawBrushRing(ctx, x, y) {
  const size = brushSizeCanvas();
  // a custom brush: the disc is the wrong footprint, so outline the actual
  // shape as a translucent white silhouette (the disc fallback applies
  // until the brush image has loaded)
  if (Tools.get().brush !== "circle") {
    const shape = brushStamp(Tools.get().brush, size, "#ffffff");
    if (shape) {
      ctx.save();
      ctx.globalAlpha = 0.35;
      ctx.drawImage(
        shape,
        Math.round(x - shape.width / 2),
        Math.round(y - shape.height / 2)
      );
      ctx.restore();
      return;
    }
  }
  const r = size / 2;
  if (r < 0.5) return;
  ctx.save();
  ctx.lineWidth = 3 / view.scale;
  ctx.strokeStyle = "rgba(0,0,0,0.7)";
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = 1 / view.scale;
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

// outline of the brush footprint at the cursor (pre-draw preview)
function drawCursor(ctx) {
  if (!view.has || !view.cursor) return;
  if (Tools.current === "crop") return; // the crop tool shows a rectangle, not a brush
  if (Tools.current === "brush" && shiftDown)
    return; // colour-picker mode: no brush preview under the mouse
  const x = (view.cursor[0] - view.ox) / view.scale;
  const y = (view.cursor[1] - view.oy) / view.scale;
  drawBrushRing(ctx, x, y);
}

// shift held: with the draw tool this is colour-picker mode (the brush
// preview is hidden and a click picks the canvas colour at the cursor)
let shiftDown = false;
document.addEventListener("keydown", (e) => {
  if (e.key === "Shift" && !shiftDown) { shiftDown = true; viewRedraw(); }
});
document.addEventListener("keyup", (e) => {
  if (e.key === "Shift" && shiftDown) { shiftDown = false; viewRedraw(); }
});

// draw tool + shift held: pick the committed image's colour at the click as
// the brush tool's colour (not a stroke; no pointer capture either, so the
// following pointerup has nothing to commit)
function pickBrushColor(p) {
  const x = Math.min(view.w - 1, Math.max(0, Math.round(p[0])));
  const y = Math.min(view.h - 1, Math.max(0, Math.round(p[1])));
  const d = view.base.getContext("2d").getImageData(x, y, 1, 1).data;
  const hex =
    "#" + [d[0], d[1], d[2]].map((v) => v.toString(16).padStart(2, "0")).join("");
  Tools.state.brush.color = hex;
  Tools.save();
  syncToolbar(); // app.js: update the swatch
  setStatus(`picked colour ${hex}`);
}

// CTRL gesture (clone tool, a source S is set): the source moves 1:1 with
// the cursor, at the relative offset frozen when CTRL went down. Strokes
// pressed during the gesture use the moved source; releasing CTRL reverts
// the source to S (only after the current stroke, if one is in flight).
let cloneCtrl = false;

function cloneGestureStart() {
  if (cloneCtrl) return; // first press freezes the offset
  cloneCtrl = true;
  if (view.cloneGesture || Tools.current !== "clone" || !view.cloneSource) return;
  const p = cursorCanvas();
  if (!p) return;
  // source = cursor + (S - cursor): it starts at S and tracks the cursor
  view.cloneGesture = [view.cloneSource[0] - p[0], view.cloneSource[1] - p[1]];
  viewRedraw();
}

function cloneGestureEnd() {
  if (!cloneCtrl) return;
  cloneCtrl = false;
  if (!view.cloneGesture) return;
  if (view.stroke) return; // only takes effect after the stroke ends
  view.cloneGesture = null;
  viewRedraw();
}

// the current cursor in canvas px (null when off the canvas)
function cursorCanvas() {
  if (!view.cursor) return null;
  return [(view.cursor[0] - view.ox) / view.scale,
          (view.cursor[1] - view.oy) / view.scale];
}

// the clone source as it currently is: the picked S, or (while the CTRL
// gesture is active) the moved source following the cursor
function cloneSourcePoint() {
  if (view.cloneGesture) {
    const p = cursorCanvas();
    if (p) return [p[0] + view.cloneGesture[0], p[1] + view.cloneGesture[1]];
  }
  return view.cloneSource;
}

// crosshairs for the clone tool: at the source, and (during a clone
// stroke) at the live sample point (cursor + frozen offset)
function drawCloneCrosshairs(ctx) {
  if (!view.has || Tools.current !== "clone") return;
  const line = (x1, y1, x2, y2, w, style) => {
    ctx.lineWidth = w;
    ctx.strokeStyle = style;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  };
  const cross = (x, y) => {
    const L = 10 / view.scale; // arm length: constant on screen
    line(x - L, y, x + L, y, 3 / view.scale, "rgba(0,0,0,0.7)");
    line(x, y - L, x, y + L, 3 / view.scale, "rgba(0,0,0,0.7)");
    line(x - L, y, x + L, y, 1 / view.scale, "rgba(255,255,255,0.95)");
    line(x, y - L, x, y + L, 1 / view.scale, "rgba(255,255,255,0.95)");
  };
  ctx.save();
  const src = cloneSourcePoint();
  // during a clone stroke the brush follows the live sample point (the
  // current source); at rest it sits at the picked source
  const stroking = view.stroke && view.stroke.tool === "clone" && view.cursor;
  if (src) {
    cross(...src);
    if (!stroking) drawBrushRing(ctx, ...src);
  }
  if (stroking) {
    const [x, y] = cloneCursorSample();
    cross(x, y);
    drawBrushRing(ctx, x, y);
  }
  ctx.restore();
}

// live sample point under the cursor: cursor + the stroke's frozen offset
function cloneCursorSample() {
  const s = view.stroke;
  const x = (view.cursor[0] - view.ox) / view.scale + s.cloneOffset[0];
  const y = (view.cursor[1] - view.oy) / view.scale + s.cloneOffset[1];
  return [x, y];
}

// the smudge preview's working image: a copy of the pre-stroke image at
// canvas-px resolution, accumulated stamp by stamp (drawStrokePreview)
function newSmudgeWork() {
  const canvas = document.createElement("canvas");
  canvas.width = view.w;
  canvas.height = view.h;
  canvas.getContext("2d").drawImage(view.base, 0, 0);
  return { canvas };
}

// scratch canvases for the stroke previews (layer-sized): the capped mask
// overlay (erase), the stroke's stamp coverage (erase + clone), and the
// clone's translated-image layer
const stampLayer = {
  canvas: document.createElement("canvas"),
  w: 0,
  h: 0,
};
let stampCoverage = null; // the stamp-coverage canvas (layer-sized)
let smudgeTemp = null; // smudge: stamp-sized buffer for the shape-masked copy

// a scratch layer at (w, h) + its context
function layerCtx(L, w, h) {
  if (L.w !== w || L.h !== h) {
    L.canvas.width = w;
    L.canvas.height = h;
    L.w = w;
    L.h = h;
  }
  return L.canvas.getContext("2d");
}

function coverageCtx(w, h) {
  if (!stampCoverage) stampCoverage = document.createElement("canvas");
  if (stampCoverage.width !== w || stampCoverage.height !== h) {
    stampCoverage.width = w;
    stampCoverage.height = h;
  }
  return stampCoverage.getContext("2d");
}

// the non-stacking tools' (brush, mask-add) coverage canvas: per pixel,
// the MAX brush alpha over the stroke's stamp footprints (== the commit's
// coverage). Built incrementally on the stroke object (s.cov / s.covN):
// only the stamps newer than the last frame are folded in, so the cost is
// amortized O(stamp area) per stamp. Pixels take grey 128 so the canvas
// can also be blitted directly as the mask overlay.
function strokeMaxCoverage(s, pts, raw) {
  if (!raw) return null;
  const cov = s.cov || (s.cov = document.createElement("canvas"));
  if (cov.width !== view.w || cov.height !== view.h) {
    cov.width = view.w;
    cov.height = view.h;
  }
  const covCtx = cov.getContext("2d", { willReadFrequently: true });
  const da = raw.alphaData.data;
  for (let i = s.covN || 0; i < pts.length; i++) {
    const x0 = Math.round(pts[i][0] - raw.width / 2);
    const y0 = Math.round(pts[i][1] - raw.height / 2);
    const rx = Math.max(0, x0);
    const ry = Math.max(0, y0);
    const rw = Math.min(view.w, x0 + raw.width) - rx;
    const rh = Math.min(view.h, y0 + raw.height) - ry;
    if (rw <= 0 || rh <= 0) continue;
    const c = covCtx.getImageData(rx, ry, rw, rh);
    const cd = c.data;
    // per-row indexing: the destination row pitch (rw) differs from the
    // source row pitch (raw.width) whenever the stamp is clipped at the
    // left/right edge -- a flat walk would smear source rows across
    const sy = ry - y0;
    const sx = rx - x0;
    for (let jy = 0; jy < rh; jy++) {
      let k = ((sy + jy) * raw.width + sx) * 4;
      let j = jy * rw * 4;
      for (let jx = 0; jx < rw; jx++, k += 4, j += 4) {
        const a = da[k + 3];
        if (a > cd[j + 3]) {
          cd[j] = cd[j + 1] = cd[j + 2] = 128;
          cd[j + 3] = a;
        }
      }
    }
    covCtx.putImageData(c, rx, ry);
  }
  s.covN = pts.length;
  return cov;
}

// the smudge's sequential stamp accumulation, shared by the live preview
// and the local commit (the commit finishes any stamps the preview has not
// drawn yet) -- each stamp drags the brush-shaped region at the previous
// centre to the current one (integer displacement, same size, no
// resampling) and alpha-blends it by the brush shape x the opacity
function smudgeAccumulate(s, st, pts, raw, r) {
  const wctx = s.work.canvas.getContext("2d");
  while (s.smudgeStamps < pts.length) {
    const i = s.smudgeStamps++; // new centre; i-1 is its predecessor
    const [ox, oy] = pts[i - 1];
    const [cx, cy] = pts[i];
    const dx = Math.round(cx - ox);
    const dy = Math.round(cy - oy);
    if (!dx && !dy) continue;
    if (!raw) {
      // fallback: clip to the stamp's disc (brush image not loaded yet)
      wctx.save();
      wctx.beginPath();
      wctx.arc(cx, cy, r, 0, Math.PI * 2);
      wctx.clip();
      wctx.globalAlpha = st.opacity;
      wctx.drawImage(s.work.canvas, dx, dy);
      wctx.restore();
      continue;
    }
    const x0 = Math.round(cx - raw.width / 2);
    const y0 = Math.round(cy - raw.height / 2);
    if (
      !smudgeTemp ||
      smudgeTemp.width !== raw.width ||
      smudgeTemp.height !== raw.height
    ) {
      smudgeTemp = document.createElement("canvas");
      smudgeTemp.width = raw.width;
      smudgeTemp.height = raw.height;
    }
    const tctx = smudgeTemp.getContext("2d");
    // temp[q] = work[q + (x0, y0) - (dx, dy)]: the pixels the PREVIOUS
    // stamp just moved (off-image source leaves dest as-is) -- the same
    // read the sequential smear does over its working copy
    tctx.clearRect(0, 0, raw.width, raw.height);
    tctx.drawImage(s.work.canvas, dx - x0, dy - y0);
    // mask the copy by the brush shape, then blend it by the opacity
    tctx.globalCompositeOperation = "destination-in";
    tctx.globalAlpha = 1;
    tctx.drawImage(raw, 0, 0);
    tctx.globalCompositeOperation = "source-over";
    wctx.globalAlpha = st.opacity;
    wctx.drawImage(smudgeTemp, x0, y0);
    wctx.globalAlpha = 1;
  }
}

function drawStrokePreview(ctx, s) {
  const st = Tools.get();
  const pts = stampPositions(s.points, Math.max(1, s.size / 12));
  const r = Math.max(s.size / 2, 0.5);
  // the selected brush at the stroke's frozen size: the exact shape the
  // commit stamps with (null until the brush image has loaded, then the
  // branches fall back to the disc)
  const raw = brushStamp(st.brush, s.size, null);
  ctx.globalAlpha = st.opacity;
  if (s.tool === "smudge") {
    // live smudge preview: SEQUENTIAL smear, accumulated on an offscreen
    // working image (like a real smudge, unlike brush/clone which recompute
    // from scratch) -- see smudgeAccumulate
    smudgeAccumulate(s, st, pts, raw, r);
    // blit the working image; the mask overlay goes on top of it so the
    // stack matches the base rendering (image, then mask)
    ctx.globalAlpha = 1;
    ctx.drawImage(s.work.canvas, 0, 0);
    if (view.mask) ctx.drawImage(view.mask, 0, 0, view.w, view.h);
    return;
  }
  if (s.tool === "clone") {
    // live clone preview: paint the pre-stroke image (view.base) translated
    // by the frozen offset, masked by the UNION of the stamp footprints,
    // in ONE composite so the opacity applies to the stroke as a whole
    // (overlapping stamps must not stack). The coverage canvas accumulates
    // the shape alphas source-over: 1 - prod(1 - shape_i); off-image source
    // reads as transparent.
    const [ox, oy] = s.cloneOffset;
    if (!raw) {
      // fallback: clip to the union of the stamp discs
      ctx.save();
      ctx.beginPath();
      for (const [x, y] of pts) {
        ctx.moveTo(x + r, y);
        ctx.arc(x, y, r, 0, Math.PI * 2);
      }
      ctx.clip();
      ctx.globalAlpha = st.opacity;
      // dest pixel p must show img[p + (ox, oy)]: drawImage(img, dx, dy)
      // shows img[p - (dx, dy)], so the image goes at -offset
      ctx.drawImage(s.img || view.base, -ox, -oy);
      ctx.restore();
      ctx.globalAlpha = 1;
      return;
    }
    const lctx = layerCtx(stampLayer, view.w, view.h);
    const cctx = coverageCtx(view.w, view.h);
    lctx.globalCompositeOperation = "source-over";
    lctx.globalAlpha = 1;
    lctx.clearRect(0, 0, view.w, view.h);
    // dest pixel p must show img[p + (ox, oy)]: drawImage(img, dx, dy) shows
    // img[p - (dx, dy)], so the image goes at -offset. s.img is the
    // pre-stroke image the commit sampled
    lctx.drawImage(s.img || view.base, -ox, -oy);
    cctx.globalCompositeOperation = "source-over";
    cctx.globalAlpha = 1;
    cctx.clearRect(0, 0, view.w, view.h);
    for (const [x, y] of pts) {
      cctx.drawImage(
        raw,
        Math.round(x - raw.width / 2),
        Math.round(y - raw.height / 2)
      );
    }
    // keep only the stroke's coverage (alpha = 1 - prod(1 - shape_i))
    lctx.globalCompositeOperation = "destination-in";
    lctx.drawImage(cctx.canvas, 0, 0);
    lctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = st.opacity;
    ctx.drawImage(stampLayer.canvas, 0, 0);
    ctx.globalAlpha = 1;
    return;
  }
  if (s.tool === "mask" && s.erase) {
    // erasing caps the mask alpha at brush x 255*(1-opacity) (non-
    // stacking: sequential stamps give min(region, cap x max shape)). The
    // overlay alpha is the mask alpha x 0.9 (buildMaskOverlay), so cap it
    // at 0.9*255*(1-opacity), approached via the union stamp coverage
    // (the preview's AA edge / overlap model)
    if (view.mask) {
      const lctx = layerCtx(stampLayer, view.w, view.h);
      const cctx = coverageCtx(view.w, view.h);
      lctx.globalCompositeOperation = "source-over";
      lctx.globalAlpha = 1;
      lctx.clearRect(0, 0, view.w, view.h);
      lctx.drawImage(view.mask, 0, 0);
      cctx.globalCompositeOperation = "source-over";
      cctx.globalAlpha = 1;
      cctx.clearRect(0, 0, view.w, view.h);
      cctx.fillStyle = "#000";
      const sw = raw ? raw.width : 2 * r;
      const sh = raw ? raw.height : 2 * r;
      // union bbox of the stamped footprints, clamped to the layer
      let bx0 = 1 / 0, by0 = 1 / 0, bx1 = -1 / 0, by1 = -1 / 0;
      for (const [x, y] of pts) {
        const x0 = Math.round(x - sw / 2);
        const y0 = Math.round(y - sh / 2);
        if (raw) cctx.drawImage(raw, x0, y0);
        else {
          cctx.beginPath();
          cctx.arc(x, y, r, 0, Math.PI * 2);
          cctx.fill();
        }
        bx0 = Math.min(bx0, x0);
        by0 = Math.min(by0, y0);
        bx1 = Math.max(bx1, x0 + sw);
        by1 = Math.max(by1, y0 + sh);
      }
      bx0 = Math.max(0, Math.floor(bx0));
      by0 = Math.max(0, Math.floor(by0));
      bx1 = Math.min(view.w, Math.ceil(bx1));
      by1 = Math.min(view.h, Math.ceil(by1));
      if (bx1 > bx0 && by1 > by0) {
        const bw = bx1 - bx0, bh = by1 - by0;
        const d = lctx.getImageData(bx0, by0, bw, bh);
        const c = cctx.getImageData(bx0, by0, bw, bh);
        const cap = 229.5 * (1 - st.opacity);
        for (let i = 0; i < d.data.length; i += 4) {
          const covA = c.data[i + 3] / 255;
          if (!covA) continue;
          const a = d.data[i + 3];
          d.data[i + 3] = a - covA * (a - Math.min(a, cap));
        }
        lctx.putImageData(d, bx0, by0);
      }
      // the layer carries the WHOLE capped mask: draw it at full alpha
      ctx.globalAlpha = 1;
      ctx.drawImage(stampLayer.canvas, 0, 0);
    }
    ctx.globalAlpha = 1;
    return;
  }
  if (s.tool === "mask") {
    // adding is NON-stacking, exactly like the local commit: the mask
    // value is the fixed brush x 255 x opacity where the brush touched
    // (max over the stamps), and the overlay alpha is 0.9 x that
    // (buildMaskOverlay) -- so build the overlay as max(base overlay,
    // 0.9 x 255 x s_max x opacity), pixel-exact to the committed mask
    const cov = strokeMaxCoverage(s, pts, raw);
    if (!cov) {
      // fallback: the old stacked discs (brush image not loaded yet)
      ctx.fillStyle = "rgba(128,128,128,0.9)";
      for (const [x, y] of pts) {
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
      return;
    }
    if (view.mask) {
      const lctx = layerCtx(stampLayer, view.w, view.h);
      lctx.globalCompositeOperation = "source-over";
      lctx.globalAlpha = 1;
      lctx.clearRect(0, 0, view.w, view.h);
      lctx.drawImage(view.mask, 0, 0);
      const cctx = cov.getContext("2d", { willReadFrequently: true });
      // union bbox of the stamped footprints, clamped to the layer
      let bx0 = 1 / 0, by0 = 1 / 0, bx1 = -1 / 0, by1 = -1 / 0;
      for (const [x, y] of pts) {
        bx0 = Math.min(bx0, x - raw.width / 2);
        by0 = Math.min(by0, y - raw.height / 2);
        bx1 = Math.max(bx1, x + raw.width / 2);
        by1 = Math.max(by1, y + raw.height / 2);
      }
      bx0 = Math.max(0, Math.floor(bx0));
      by0 = Math.max(0, Math.floor(by0));
      bx1 = Math.min(view.w, Math.ceil(bx1));
      by1 = Math.min(view.h, Math.ceil(by1));
      if (bx1 > bx0 && by1 > by0) {
        const bw = bx1 - bx0, bh = by1 - by0;
        const d = lctx.getImageData(bx0, by0, bw, bh);
        const c = cctx.getImageData(bx0, by0, bw, bh);
        const target = 229.5 * st.opacity;
        // the overlay is always grey 128; alpha-0 pixels read back with
        // undefined RGB (0), so the RGB must be set alongside the alpha
        for (let i = 0; i < d.data.length; i += 4) {
          d.data[i] = d.data[i + 1] = d.data[i + 2] = 128;
          const t = target * (c.data[i + 3] / 255);
          if (t > d.data[i + 3]) d.data[i + 3] = t;
        }
        lctx.putImageData(d, bx0, by0);
      }
      ctx.globalAlpha = 1;
      ctx.drawImage(stampLayer.canvas, 0, 0);
    } else {
      // no base mask: the overlay is just the grey coverage itself
      ctx.globalAlpha = 0.9 * st.opacity;
      ctx.drawImage(cov, 0, 0);
    }
    ctx.globalAlpha = 1;
    return;
  }
  // brush tool: colour x brush x opacity, NON-stacking like the commit:
  // the colour is blended in ONCE by the stroke's max coverage x
  // opacity (overlapping stamps do not darken the core)
  const cov = strokeMaxCoverage(s, pts, raw);
  if (cov) {
    const lctx = layerCtx(stampLayer, view.w, view.h);
    lctx.globalCompositeOperation = "source-over";
    lctx.globalAlpha = 1;
    lctx.fillStyle = st.color;
    lctx.fillRect(0, 0, view.w, view.h);
    // keep only the stroke's coverage (alpha = max shape over stamps)
    lctx.globalCompositeOperation = "destination-in";
    lctx.drawImage(cov, 0, 0);
    lctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = st.opacity;
    ctx.drawImage(stampLayer.canvas, 0, 0);
    ctx.globalAlpha = 1;
    return;
  }
  // fallback: the old stacked discs (brush image not loaded yet)
  ctx.fillStyle = st.color;
  for (const [x, y] of pts) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

// ---------------- local stroke commit ----------------

// mask ADD, values: m = max(m, cov x 255 x opacity) per pixel -- the
// stroke's max stamp coverage (the preview's strokeMaxCoverage canvas)
// scaled to the value range the commit writes
function commitMaskAdd(s, st) {
  const pts = stampPositions(s.points, Math.max(1, s.size / 12));
  const raw = brushStamp(st.brush, s.size, null);
  const r = Math.max(s.size / 2, 0.5);
  const cov = raw ? strokeMaxCoverage(s, pts, raw) : null;
  const mctx = view.maskValues.getContext("2d", { willReadFrequently: true });
  let bx0, by0, bx1, by1;
  if (raw) {
    // union bbox of the stamped footprints, clamped to the canvas
    bx0 = by0 = 1 / 0; bx1 = by1 = -1 / 0;
    for (const [x, y] of pts) {
      bx0 = Math.min(bx0, x - raw.width / 2);
      by0 = Math.min(by0, y - raw.height / 2);
      bx1 = Math.max(bx1, x + raw.width / 2);
      by1 = Math.max(by1, y + raw.height / 2);
    }
  } else {
    // fallback (brush image not loaded): per-pixel disc coverage, max over
    // the stamps (non-stacking, same model as the brush path)
    bx0 = by0 = 1 / 0; bx1 = by1 = -1 / 0;
    for (const [x, y] of pts) {
      bx0 = Math.min(bx0, x - r);
      by0 = Math.min(by0, y - r);
      bx1 = Math.max(bx1, x + r);
      by1 = Math.max(by1, y + r);
    }
  }
  bx0 = Math.max(0, Math.floor(bx0));
  by0 = Math.max(0, Math.floor(by0));
  bx1 = Math.min(view.w, Math.ceil(bx1));
  by1 = Math.min(view.h, Math.ceil(by1));
  if (bx1 <= bx0 || by1 <= by0) return; // entirely off-canvas
  const bw = bx1 - bx0, bh = by1 - by0;
  const d = mctx.getImageData(bx0, by0, bw, bh);
  const target = 255 * st.opacity;
  if (raw) {
    const c = cov.getContext("2d", { willReadFrequently: true })
            .getImageData(bx0, by0, bw, bh).data;
    for (let i = 0; i < d.data.length; i += 4) {
      const t = target * (c[i + 3] / 255);
      if (t > d.data[i + 3]) {
        // the overlay is always grey 128; alpha-0 pixels read back with
        // undefined RGB (0), so the RGB must be set alongside the alpha
        d.data[i] = d.data[i + 1] = d.data[i + 2] = 128;
        d.data[i + 3] = t;
      }
    }
  } else {
    for (let i = 0; i < d.data.length; i += 4) {
      const x = bx0 + (i / 4) % bw;
      const y = by0 + Math.floor(i / 4 / bw);
      for (const [px, py] of pts) {
        const dx = x - px, dy = y - py;
        if (dx * dx + dy * dy <= r * r) {
          d.data[i] = d.data[i + 1] = d.data[i + 2] = 128;
          d.data[i + 3] = Math.max(d.data[i + 3], target);
          break;
        }
      }
    }
  }
  mctx.putImageData(d, bx0, by0);
}

// mask ERASE, values: m = m - covU x (m - min(m, cap)) with cap = 255 x
// (1 - opacity) and covU the UNION stamp coverage (source-over alpha
// accumulation -- sequential stamps compose to the union, matching the
// per-stamp affine blends the old server commit used)
function commitMaskErase(s, st) {
  const pts = stampPositions(s.points, Math.max(1, s.size / 12));
  const raw = brushStamp(st.brush, s.size, null);
  const r = Math.max(s.size / 2, 0.5);
  const cctx = coverageCtx(view.w, view.h);
  cctx.globalCompositeOperation = "source-over";
  cctx.globalAlpha = 1;
  cctx.clearRect(0, 0, view.w, view.h);
  const sw = raw ? raw.width : 2 * r;
  const sh = raw ? raw.height : 2 * r;
  let bx0 = 1 / 0, by0 = 1 / 0, bx1 = -1 / 0, by1 = -1 / 0;
  for (const [x, y] of pts) {
    const x0 = Math.round(x - sw / 2);
    const y0 = Math.round(y - sh / 2);
    if (raw) cctx.drawImage(raw, x0, y0);
    else {
      cctx.beginPath();
      cctx.arc(x, y, r, 0, Math.PI * 2);
      cctx.fill();
    }
    bx0 = Math.min(bx0, x0);
    by0 = Math.min(by0, y0);
    bx1 = Math.max(bx1, x0 + sw);
    by1 = Math.max(by1, y0 + sh);
  }
  bx0 = Math.max(0, Math.floor(bx0));
  by0 = Math.max(0, Math.floor(by0));
  bx1 = Math.min(view.w, Math.ceil(bx1));
  by1 = Math.min(view.h, Math.ceil(by1));
  if (bx1 <= bx0 || by1 <= by0) return; // entirely off-canvas
  const bw = bx1 - bx0, bh = by1 - by0;
  const mctx = view.maskValues.getContext("2d", { willReadFrequently: true });
  const d = mctx.getImageData(bx0, by0, bw, bh);
  const c = cctx.getImageData(bx0, by0, bw, bh).data;
  const cap = 255 * (1 - st.opacity);
  for (let i = 0; i < d.data.length; i += 4) {
    const covA = c[i + 3] / 255;
    if (!covA) continue;
    const a = d.data[i + 3];
    d.data[i + 3] = a - covA * (a - Math.min(a, cap));
  }
  mctx.putImageData(d, bx0, by0);
}

// apply a finished stroke to the committed local state. The commit IS the
// last preview frame (the previews are built to be pixel-exact), so the
// hand-off is invisible.
function commitStroke(s, st) {
  const baseCtx = view.base.getContext("2d");
  if (s.tool === "brush" || s.tool === "clone") {
    // the preview's paint/clone branch blits the stroke's layer (colour x
    // coverage, or the translated image x coverage) with the stroke's
    // opacity onto the given ctx -- invoke it with the base canvas's
    // context (identity transform; the layers are in canvas px)
    drawStrokePreview(baseCtx, s);
    return;
  }
  if (s.tool === "smudge") {
    // finish any stamps the preview has not drawn yet, then blit the
    // working image (the mask overlay stays in view.mask)
    const pts = stampPositions(s.points, Math.max(1, s.size / 12));
    smudgeAccumulate(s, st, pts, brushStamp(st.brush, s.size, null),
                     Math.max(s.size / 2, 0.5));
    baseCtx.drawImage(s.work.canvas, 0, 0);
    return;
  }
  if (s.tool === "mask") {
    if (s.erase) commitMaskErase(s, st);
    else commitMaskAdd(s, st);
    buildMaskOverlay();
  }
}

// wheel delta with the shift-fallback described above
function wheelDelta(e) {
  return e.deltaY !== 0 ? e.deltaY : e.deltaX;
}

// ---------------- crop tool ----------------

// The crop selection's actual region: the drag's bounding rect clamped to
// the image bounds, so a selection that starts or ends outside the image
// still crops the intersection. Null when the intersection is empty.
function cropRect() {
  if (!view.crop) return null;
  const { sx, sy, ex, ey } = view.crop;
  const cl = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const x0 = cl(Math.min(sx, ex), 0, view.w);
  const y0 = cl(Math.min(sy, ey), 0, view.h);
  const x1 = cl(Math.max(sx, ex), 0, view.w);
  const y1 = cl(Math.max(sy, ey), 0, view.h);
  if (x1 - x0 < 1 || y1 - y0 < 1) return null;
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

// preview while dragging: dim everything outside the crop rect and outline it
function drawCropOverlay(ctx) {
  const r = cropRect();
  if (!r) return;
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, view.w, view.h);
  ctx.rect(r.x, r.y, r.w, r.h);
  ctx.fillStyle = "rgba(0,0,0,0.5)";
  ctx.fill("evenodd");
  ctx.lineWidth = 2 / view.scale; // dark halo, constant on screen
  ctx.strokeStyle = "rgba(0,0,0,0.7)";
  ctx.strokeRect(r.x, r.y, r.w, r.h);
  ctx.lineWidth = 1 / view.scale;
  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.strokeRect(r.x, r.y, r.w, r.h);
  ctx.restore();
}

// pointerup on a finished drag: commit the (already clamped) rect to the
// server; a zero-area selection (plain click) is a no-op
function commitCrop() {
  const r = cropRect();
  view.crop = null;
  viewRedraw();
  if (!r) return;
  doCrop(r.x, r.y, r.w, r.h); // app.js: POST + thumbnail update + refresh
}

// Esc / pointercancel / tool switch: abandon the selection, no request
function abortCrop() {
  if (!view.crop) return;
  view.crop = null;
  viewRedraw();
}

// ---------------- input ----------------

function canvasPoint(e) {
  const r = cv().getBoundingClientRect();
  const mx = e.clientX - r.left;
  const my = e.clientY - r.top;
  view.cursor = [mx, my]; // screen px: the brush outline stays under the mouse
  return [(mx - view.ox) / view.scale, (my - view.oy) / view.scale];
}

function setupCanvasInput() {
  const c = cv();

  c.addEventListener("pointerdown", (e) => {
    if (!view.has) return;
    canvasPoint(e); // record the cursor for the brush outline
    if (e.button === 1) {
      e.preventDefault(); // middle: pan (also kills autoscroll)
      view.pan = { x: e.clientX, y: e.clientY, ox: view.ox, oy: view.oy };
      c.setPointerCapture(e.pointerId);
      return;
    }
    if (e.button !== 0) return;
    const tool = Tools.current;
    const p = canvasPoint(e);
    if (tool === "crop") {
      // start a rectangle selection; the pointer may be anywhere on the
      // element (including outside the image) -- the used region is the
      // clamped intersection, computed in cropRect
      view.crop = { sx: p[0], sy: p[1], ex: p[0], ey: p[1] };
      c.setPointerCapture(e.pointerId);
      viewRedraw();
      return;
    }
    if (tool === "brush" && e.shiftKey) {
      // colour-picker mode: set the brush colour from the clicked pixel
      pickBrushColor(p);
      return;
    }
    if (tool === "clone") {
      if (e.shiftKey) {
        // pick/move the clone source S; not a stroke (no pointer capture,
        // so the following pointerup has nothing to commit)
        view.cloneSource = [p[0], p[1]];
        // re-anchor an active CTRL gesture: the moved source lands on S
        // (the cursor is on S, so the relative offset is zero)
        if (view.cloneGesture) view.cloneGesture = [0, 0];
        viewRedraw();
        return;
      }
      const src = cloneSourcePoint();
      if (!src) {
        setStatus("clone: shift-click to pick a source point first");
        return;
      }
      view.stroke = {
        tool,
        points: [p],
        size: brushSizeCanvas(),
        // frozen for the whole stroke, integer canvas px, so the preview
        // and the commit sample identical pixels
        cloneOffset: [Math.round(src[0] - p[0]), Math.round(src[1] - p[1])],
        // the pre-stroke image the commit samples
        img: view.base,
        // pre-stroke snapshot for the draw undo stack; toBlob captures at
        // call time, so the encode overlaps the drag
        snapP: snapCanvas(view.base),
      };
      c.setPointerCapture(e.pointerId);
      viewRedraw();
      return;
    }
    if (tool === "smudge") {
      // smudge is a drag: the first stamp samples itself (a no-op), so the
      // stamp counter starts at 1; the working image starts as the current
      // (pre-stroke) canvas image and rides on the stroke object, so the
      // held preview keeps working after the stroke ends
      view.stroke = {
        tool,
        points: [p],
        size: brushSizeCanvas(),
        smudgeStamps: 1,
        work: newSmudgeWork(),
        snapP: snapCanvas(view.base),
      };
      c.setPointerCapture(e.pointerId);
      viewRedraw();
      return;
    }
    view.stroke = {
      tool,
      points: [p],
      // freeze the canvas-px footprint at stroke start (the commit stamps
      // with a single size, so a mid-stroke zoom can't change it)
      size: brushSizeCanvas(),
      erase: tool === "mask" && e.shiftKey, // shift = mask eraser
      // pre-stroke snapshot for the right undo stack (draw vs mask)
      snapP: tool === "mask" ? null : snapCanvas(view.base),
      snapMaskP: tool === "mask" ? snapCanvas(view.maskValues) : null,
    };
    c.setPointerCapture(e.pointerId);
    viewRedraw();
  });

  c.addEventListener("pointermove", (e) => {
    if (view.pan) {
      view.ox = view.pan.ox + (e.clientX - view.pan.x);
      view.oy = view.pan.oy + (e.clientY - view.pan.y);
      viewRedraw();
      return;
    }
    if (!view.has) return;
    const oldCursor = view.cursor;
    const p = canvasPoint(e); // also updates view.cursor
    if (view.crop) {
      view.crop.ex = p[0];
      view.crop.ey = p[1];
      viewRedraw();
      return;
    }
    let redraw = false;
    if (view.stroke) {
      const n = view.stroke.points.length;
      const [lx, ly] = view.stroke.points[n - 1];
      // ignore sub-pixel jitter: at least 1 screen px between recorded points
      if (Math.hypot(p[0] - lx, p[1] - ly) * view.scale >= 1) {
        view.stroke.points.push(p);
        redraw = true;
      }
    } else if (
      (!oldCursor ||
        Math.hypot(
          p[0] - (oldCursor[0] - view.ox) / view.scale,
          p[1] - (oldCursor[1] - view.oy) / view.scale
        ) * view.scale >= 1)
    ) {
      // cursor outline moved at least 1 screen px
      redraw = true;
    }
    if (redraw) viewRedraw();
  });
  c.addEventListener("pointerleave", () => {
    view.cursor = null;
    viewRedraw();
  });

  const endStroke = async () => {
    if (view.pan) {
      view.pan = null;
      return;
    }
    if (!view.stroke) return;
    const s = view.stroke;
    view.stroke = null;
    const st = Tools.get();
    // CTRL released during the stroke: the source reverts only now
    if (view.cloneGesture && !cloneCtrl) view.cloneGesture = null;
    if (s.tool === "smudge" && s.points.length < 2) {
      // a single smudge click smears nothing: no commit, no history step
      clearBrushStamps();
      viewRedraw();
      return;
    }
    commitStroke(s, st); // local, synchronous: the preview's own result
    clearBrushStamps(); // the preview is committed; its stamp cache is done
    // the snapshot promise lands on the stack immediately (its encoding
    // overlaps the drag); undoStep/redoStep await it on pop
    pushUndo(s.tool, s.snapP || s.snapMaskP);
    markDirty(); // the server's copy is behind; idle sync (or the next
    // forceSync before a server-side read) pushes the state
    syncUndoButtons(); // app.js
    viewRedraw();
  };
  c.addEventListener("pointerup", (e) => {
    if (view.crop) { commitCrop(); return; }
    endStroke();
  });
  c.addEventListener("pointercancel", (e) => {
    if (view.crop) { abortCrop(); return; }
    endStroke();
  });

  // wheel zoom, anchored at the cursor; CTRL+wheel = brush size,
  // SHIFT+wheel = opacity (the browser turns a shifted vertical wheel into
  // horizontal delta, so fall back to deltaX when deltaY is 0)
  c.addEventListener(
    "wheel",
    (e) => {
      if (!view.has) return;
      e.preventDefault();
      if (e.ctrlKey) { wheelBrushSize(Math.pow(1.0015, -wheelDelta(e))); return; }
      if (e.shiftKey) { wheelBrushOpacity(Math.pow(1.0015, -wheelDelta(e))); return; }
      const r = c.getBoundingClientRect();
      viewZoomAt(
        Math.pow(1.0015, -e.deltaY),
        e.clientX - r.left,
        e.clientY - r.top
      );
    },
    { passive: false }
  );
  c.addEventListener("dblclick", () => viewFit());
  c.addEventListener("auxclick", (e) => {
    if (e.button === 1) e.preventDefault();
  });
}
