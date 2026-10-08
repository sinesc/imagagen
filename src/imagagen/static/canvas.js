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
  mask: null, // display overlay derived from maskValues (maskOverlayOpacity
  // see-through)
  maskBbox: null, // {x,y,w,h} canvas px of the current mask (alpha > 0);
  // null when the mask is empty; refreshed with the overlay
  has: false,
  w: 0,
  h: 0,
  scale: 1, // screen = canvas * scale + o (CSS px)
  ox: 0,
  oy: 0,
  dpr: 1,
  stroke: null, // {tool, points: [[x,y]...], erase, cloneOffset, smudgeStamps}
  cloneSource: null, // [x,y] canvas px: clone tool's picked source S
  cloneGesture: null, // [ox,oy] canvas px: SHIFT held -> the source follows
  // the cursor at this relative offset (a temporary move of the picked S)
  crop: null, // {sx, sy, ex, ey} raw canvas px of the crop tool's drag; the
  // region used is its clamped intersection with the image (cropRect). The
  // copy tool drags the same rectangle (commitCrop dispatches on the tool)
  clip: null, // the copied area: {img, w, h} -- img is an offscreen canvas
  // (w x h, opaque) taken from view.base; paste drops it as a floating
  // selection. In-memory only: it survives entity switches, not a reload
  sel: null, // floating selection {img, sw, sh, cx, cy, angle}: img is the
  // copied image (its own resolution), sw/sh the CURRENT display size (the
  // image is stretched to fill it, resampled once at Apply), cx/cy its
  // centre and angle its rotation, all in canvas px / radians
  selDrag: null, // in-flight handle drag {kind, F: frame frozen at drag start}
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
    syncSelActions(); // no canvas -> no selection overlay buttons
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
  if (view.stroke && !view.stroke.sel) drawStrokePreview(ctx, view.stroke);
  drawMaskOutline(ctx);
  if (view.crop) drawCropOverlay(ctx);
  if (view.sel) drawSelectionOverlay(ctx);
  drawCursor(ctx);
  drawCloneCrosshairs(ctx);
  syncSelActions(); // the apply / discard pair follows the selection
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

// ---------------- mask overlay opacity ----------------

// Alpha multiplier applied to the mask when displaying it (0.9 = the
// historical default). Ini-configurable (mask-overlay-opacity) and
// delivered by /api/ui-config at boot; the local default matches the ini
// default, so a failed/absent fetch is invisible.
let maskOverlayOpacity = 0.9;
let overlayConfigPromise = null;

function ensureOverlayConfig() {
  if (!overlayConfigPromise) {
    overlayConfigPromise = fetch("/api/ui-config")
      .then((r) => (r.ok ? r.json() : null))
      .then((cfg) => {
        if (cfg && Number.isFinite(cfg.mask_overlay_opacity)) {
          maskOverlayOpacity = cfg.mask_overlay_opacity;
          // the boot viewRefresh may already have built the overlay with
          // the default factor: rebuild so the configured value applies
          if (view.maskValues) buildMaskOverlay();
        }
      })
      .catch(() => {});
  }
  return overlayConfigPromise;
}

// display overlay: constant grey with alpha = mask alpha x
// maskOverlayOpacity, derived GPU-only (destination-in alpha scale -- no
// per-pixel loop). The grey fill
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
  ctx.globalCompositeOperation = "destination-in";
  // alpha = mask x maskOverlayOpacity
  ctx.globalAlpha = maskOverlayOpacity;
  ctx.drawImage(view.maskValues, 0, 0);
  view.mask = o;
  computeMaskBbox();
  syncGenerateEnabled(); // app.js: the Generate button tracks mask emptiness
}

// bounding box of the current mask (alpha > 0) in canvas px, or null when
// the mask is empty. Recomputed exactly where the overlay is -- i.e. every
// time the mask changes (commit, morph, clear, refresh, undo/redo).
function computeMaskBbox() {
  view.maskBbox = null;
  if (!view.maskValues) return;
  const c = view.maskValues;
  const data = c.getContext("2d", { willReadFrequently: true })
    .getImageData(0, 0, c.width, c.height).data;
  let x0 = Infinity, y0 = Infinity, x1 = -1, y1 = -1;
  for (let y = 0, i = 3; y < c.height; y++) {
    for (let x = 0; x < c.width; x++, i += 4) {
      if (data[i]) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 >= x0) view.maskBbox = { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
}

// live mask extent in image-space px: the committed mask bbox united with
// the in-flight mask stroke's footprint (r = s.size/2), so it updates
// continuously while drawing; null when there is neither.
function maskBboxLive() {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  const b = view.maskBbox;
  if (b) {
    x0 = b.x; y0 = b.y; x1 = b.x + b.w; y1 = b.y + b.h;
  }
  const s = view.stroke;
  if (s && s.tool === "mask" && s.points.length) {
    const r = Math.max(s.size / 2, 0.5);
    for (const [px, py] of s.points) {
      if (px - r < x0) x0 = px - r;
      if (py - r < y0) y0 = py - r;
      if (px + r > x1) x1 = px + r;
      if (py + r > y1) y1 = py + r;
    }
  }
  if (x1 <= x0 || y1 <= y0) return null; // empty mask, no live stroke
  const ix0 = Math.floor(x0), iy0 = Math.floor(y0);
  const ix1 = Math.ceil(x1), iy1 = Math.ceil(y1);
  return { x: ix0, y: iy0, w: ix1 - ix0, h: iy1 - iy0 };
}

// faint outline rect for the current mask: the live mask extent grown by
// the "Mask pad" value on each side (capped at the mask's largest extent,
// like the server's inpaint box) and clamped to the canvas; null when the
// mask is empty.
function maskOutlineBox() {
  const b = maskBboxLive();
  if (!b) return null;
  let pad = parseInt(
    document.getElementById("mask-padding").value, 10) || 0;
  pad = Math.max(0, Math.min(pad, Math.max(b.w, b.h)));
  const ox0 = Math.max(0, b.x - pad);
  const oy0 = Math.max(0, b.y - pad);
  const ox1 = Math.min(view.w, b.x + b.w + pad);
  const oy1 = Math.min(view.h, b.y + b.h + pad);
  if (ox1 <= ox0 || oy1 <= oy0) return null;
  return { x: ox0, y: oy0, w: ox1 - ox0, h: oy1 - oy0 };
}

function drawMaskOutline(ctx) {
  const r = maskOutlineBox();
  if (!r) return;
  ctx.save();
  ctx.lineWidth = 1 / view.scale; // 1 screen px, faint grey
  ctx.strokeStyle = "rgba(128,128,128,0.6)";
  ctx.strokeRect(r.x, r.y, r.w, r.h);
  // live mask extent in image-space px (includes the in-flight stroke,
  // so it updates while drawing), tucked inside the box's bottom-right
  // corner (the drawn box is padded, the number is the raw mask extent)
  const b = maskBboxLive();
  if (b) {
    const fs = 15 / view.scale; // ~15 screen px
    ctx.font = `${fs}px system-ui, sans-serif`;
    ctx.fillStyle = "rgba(128,128,128,0.9)";
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    // thin dark drop shadow so the grey text stays readable on bright canvas
    // (shadowBlur/offset are in device px, unaffected by the view transform)
    ctx.shadowColor = "rgba(0,0,0,0.85)";
    ctx.shadowBlur = 1;
    ctx.shadowOffsetY = 1;
    const inset = 2 / view.scale;
    ctx.fillText(`${b.w}x${b.h}`, r.x + r.w - inset, r.y + r.h - inset);
  }
  ctx.restore();
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
    ensureOverlayConfig(),
  ]);
  const had = view.has;
  const oldW = view.w;
  const oldH = view.h;
  undoDraw.length = redoDraw.length = undoMask.length = redoMask.length = 0;
  if (pendingMaskUndo) {
    undoMask.push(pendingMaskUndo);
    pendingMaskUndo = null;
  }
  // fresh server state replaces the canvas: a floating selection rides on the
  // OLD image, so it is dropped with it (the copied area survives -- it is
  // the client's own buffer, not canvas state)
  view.sel = null;
  view.selDrag = null;
  view.crop = null;
  cancelSync();
  view.has = ri.ok;
  if (!view.has) {
    view.base = null;
    view.maskValues = null;
    view.mask = null;
    view.maskBbox = null;
    view.w = view.h = 0;
    clearBrushStamps();
    viewRedraw();
    syncGenerateEnabled(); // app.js: no image -> empty mask
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
  const tool = Tools.current;
  if (tool === "crop" || tool === "copy") return; // these show a rectangle
  if (tool === "paste") return drawPasteGhost(ctx);
  if (tool === "brush" && ctrlDown) return; // colour-picker: no brush preview
  const p = [(view.cursor[0] - view.ox) / view.scale,
             (view.cursor[1] - view.oy) / view.scale];
  if (selStrokeTool(tool)) {
    // the stroke lands in the selection's own frame, so the footprint
    // outline is drawn there too (it rotates with the selection)
    const s = view.sel;
    ctx.save();
    ctx.translate(s.cx, s.cy);
    ctx.rotate(s.angle);
    const q = selCentred(p);
    drawBrushRing(ctx, q[0], q[1]);
    ctx.restore();
    return;
  }
  drawBrushRing(ctx, p[0], p[1]);
}

// shift held: with a floating selection active the draw tool erases (see
// drawStrokePreview); with the mask tool it is the mask eraser
let shiftDown = false;
// ctrl held: with the draw tool this is colour-picker mode (the brush preview
// is hidden and a click picks the canvas colour at the cursor)
let ctrlDown = false;
document.addEventListener("keydown", (e) => {
  if (e.key === "Shift" && !shiftDown) { shiftDown = true; viewRedraw(); }
  if (e.key === "Control" && !ctrlDown) { ctrlDown = true; viewRedraw(); }
});
document.addEventListener("keyup", (e) => {
  if (e.key === "Shift" && shiftDown) { shiftDown = false; viewRedraw(); }
  if (e.key === "Control" && ctrlDown) { ctrlDown = false; viewRedraw(); }
});

// draw tool + ctrl held: pick the committed image's colour at the click as
// the brush tool's colour (not a stroke; no pointer capture either, so the
// following pointerup has nothing to commit)
function pickBrushColor(p) {
  const x = Math.min(view.w - 1, Math.max(0, Math.round(p[0])));
  const y = Math.min(view.h - 1, Math.max(0, Math.round(p[1])));
  const d = view.base.getContext("2d").getImageData(x, y, 1, 1).data;
  const hex =
    "#" + [d[0], d[1], d[2]].map((v) => v.toString(16).padStart(2, "0")).join("");
  Tools.state.brush.color = hex;
  scheduleSaveUiState(); // app.js
  syncToolbar(); // app.js: update the swatch
  setStatus(`picked colour ${hex}`);
}

// SHIFT gesture (clone tool, a source S is set): the source moves 1:1 with
// the cursor, at the relative offset frozen when SHIFT went down. Strokes
// pressed during the gesture use the moved source; releasing SHIFT reverts
// the source to S (only after the current stroke, if one is in flight).
let cloneShift = false;

function cloneGestureStart() {
  if (cloneShift) return; // first press freezes the offset
  cloneShift = true;
  if (view.cloneGesture || Tools.current !== "clone" || !view.cloneSource) return;
  const p = cursorCanvas();
  if (!p) return;
  // source = cursor + (S - cursor): it starts at S and tracks the cursor
  view.cloneGesture = [view.cloneSource[0] - p[0], view.cloneSource[1] - p[1]];
  viewRedraw();
}

function cloneGestureEnd() {
  if (!cloneShift) return;
  cloneShift = false;
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

// the clone source as it currently is: the picked S, or (while the SHIFT
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

// the smudge preview's working image: a copy of the pre-stroke image at the
// stroke's target resolution, accumulated stamp by stamp (drawStrokePreview)
function newSmudgeWork(img, w, h) {
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  canvas.getContext("2d").drawImage(img, 0, 0);
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
  if (cov.width !== s.tw || cov.height !== s.th) {
    cov.width = s.tw;
    cov.height = s.th;
  }
  const covCtx = cov.getContext("2d", { willReadFrequently: true });
  const da = raw.alphaData.data;
  for (let i = s.covN || 0; i < pts.length; i++) {
    const x0 = Math.round(pts[i][0] - raw.width / 2);
    const y0 = Math.round(pts[i][1] - raw.height / 2);
    const rx = Math.max(0, x0);
    const ry = Math.max(0, y0);
    const rw = Math.min(s.tw, x0 + raw.width) - rx;
    const rh = Math.min(s.th, y0 + raw.height) - ry;
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
// drawn yet). Each stamp blends the brush-shaped footprint at the current
// centre TOWARD the pixels it came from (the previous centre's positions),
// weighted by the brush shape x the per-stamp strength. The blend is
// PARTIAL (capped at half), never a full copy: a rigid copy overwrites the
// pixel ahead of the brush (its destination's leading edge lies 1px past
// the copied region, so content is destroyed there) and leaves a one-way
// valve -- a back-and-forth scrub could then only push a boundary in the
// direction of the first leg, one side eating the other. The partial blend
// advects the content with the drag AND diffuses it, so scrubbing back and
// forth converges to a blurred band around the original boundary. The
// footprint's leading edge (the 1px the return pass cannot reach) gets a
// mirror blend toward the pixel ahead of the brush, closing the valve.
function smudgeAccumulate(s, st, pts, raw, r) {
  const canvas = s.work.canvas;
  const wctx =
    s.workCtx ||
    (s.workCtx = canvas.getContext("2d", { willReadFrequently: true }));
  const W = canvas.width, H = canvas.height;
  // per-stamp strength: capped at half of the opacity so even a 100%
  // smudge stays a blend (the erosion-free regime), not a rigid copy
  const t = 0.5 * st.opacity;
  const tm = t; // the leading-edge mirror: equal strength closes the valve
  const fw = raw ? raw.width : Math.ceil(2 * r + 1);
  const fh = raw ? raw.height : Math.ceil(2 * r + 1);
  const ad = raw ? raw.alphaData.data : null;
  const r2 = r * r;
  while (s.smudgeStamps < pts.length) {
    const i = s.smudgeStamps++; // new centre; i-1 is its predecessor
    const [ox, oy] = pts[i - 1];
    const [cx, cy] = pts[i];
    const dx = Math.round(cx - ox);
    const dy = Math.round(cy - oy);
    if (!dx && !dy) continue;
    const x0 = Math.round(cx - fw / 2);
    const y0 = Math.round(cy - fh / 2);
    // the stamp's footprint, clipped to the image
    const x1 = Math.max(0, x0), y1 = Math.max(0, y0);
    const x2 = Math.min(W, x0 + fw), y2 = Math.min(H, y0 + fh);
    if (x2 <= x1 || y2 <= y1) continue;
    // read the footprint, its source (shifted back by the displacement),
    // and the leading-edge mirror neighbours (1px ahead) in one go; the
    // walk then blends in place from PRE-STAMP values -- it runs against
    // the motion, so a pixel's source is never read after this stamp has
    // modified it
    const sx0 = x0 - dx, sy0 = y0 - dy;
    const ux0 = Math.max(0, Math.min(x0, sx0) - (dx < 0 ? 1 : 0));
    const uy0 = Math.max(0, Math.min(y0, sy0) - (dy < 0 ? 1 : 0));
    const ux1 = Math.min(W, Math.max(x2, sx0 + fw) + (dx > 0 ? 1 : 0));
    const uy1 = Math.min(H, Math.max(y2, sy0 + fh) + (dy > 0 ? 1 : 0));
    const img = wctx.getImageData(ux0, uy0, ux1 - ux0, uy1 - uy0);
    const d = img.data;
    const uw = ux1 - ux0;
    const id4 = (x, y) => ((y - uy0) * uw + (x - ux0)) * 4;
    const shape = (x, y) => {
      if (raw) return ad[((y - y0) * fw + (x - x0)) * 4 + 3] / 255;
      const qx = x - cx, qy = y - cy;
      return qx * qx + qy * qy <= r2 ? 1 : 0; // disc fallback
    };
    // against the motion: d>0 -> the source lies at smaller coords, so
    // visit the larger ones first; d<0 -> the other way round
    const xStart = dx >= 0 ? x2 - 1 : x1, xStep = dx >= 0 ? -1 : 1;
    const yStart = dy >= 0 ? y2 - 1 : y1, yStep = dy >= 0 ? -1 : 1;
    for (let y = yStart; dy >= 0 ? y >= y1 : y < y2; y += yStep) {
      for (let x = xStart; dx >= 0 ? x >= x1 : x < x2; x += xStep) {
        const a = shape(x, y);
        if (!a) continue;
        const px = x - dx, py = y - dy; // where this pixel came from
        if (px < 0 || py < 0 || px >= W || py >= H) continue;
        const j = id4(x, y), k = id4(px, py);
        const f = t * a;
        d[j] += f * (d[k] - d[j]);
        d[j + 1] += f * (d[k + 1] - d[j + 1]);
        d[j + 2] += f * (d[k + 2] - d[j + 2]);
      }
    }
    // the leading edge: the footprint's outer row/column in the motion
    // direction. Its mirror blend reads the pixel just AHEAD of the brush
    // (out of reach of the return pass without it), so scrubbing back
    // reaches the boundary the forward pass pushed out there
    const mirror = (x, y, nx, ny) => {
      if (nx < 0 || ny < 0 || nx >= W || ny >= H) return;
      const j = id4(x, y), k = id4(nx, ny);
      const f = tm * shape(x, y);
      if (!f) return;
      d[j] += f * (d[k] - d[j]);
      d[j + 1] += f * (d[k + 1] - d[j + 1]);
      d[j + 2] += f * (d[k + 2] - d[j + 2]);
    };
    if (dx) {
      const x = dx > 0 ? x2 - 1 : x1;
      for (let y = y1; y < y2; y++) mirror(x, y, x + Math.sign(dx), y);
    }
    if (dy) {
      const y = dy > 0 ? y2 - 1 : y1;
      for (let x = x1; x < x2; x++) mirror(x, y, x, y + Math.sign(dy));
    }
    wctx.putImageData(img, ux0, uy0);
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
    // stack matches the base rendering (image, then mask). A stroke on a
    // floating selection has no mask layer (the mask belongs to the canvas)
    ctx.globalAlpha = 1;
    ctx.drawImage(s.work.canvas, 0, 0);
    if (view.mask && !s.sel) ctx.drawImage(view.mask, 0, 0, view.w, view.h);
    return;
  }
  if (s.tool === "clone") {
    // live clone preview: paint the pre-stroke image (s.img) translated
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
      ctx.drawImage(s.img, -ox, -oy);
      ctx.restore();
      ctx.globalAlpha = 1;
      return;
    }
    const lctx = layerCtx(stampLayer, s.tw, s.th);
    const cctx = coverageCtx(s.tw, s.th);
    lctx.globalCompositeOperation = "source-over";
    lctx.globalAlpha = 1;
    lctx.clearRect(0, 0, s.tw, s.th);
    // dest pixel p must show img[p + (ox, oy)]: drawImage(img, dx, dy) shows
    // img[p - (dx, dy)], so the image goes at -offset. s.img is the
    // pre-stroke image the commit sampled
    lctx.drawImage(s.img, -ox, -oy);
    cctx.globalCompositeOperation = "source-over";
    cctx.globalAlpha = 1;
    cctx.clearRect(0, 0, s.tw, s.th);
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
    // overlay alpha is the mask alpha x maskOverlayOpacity
    // (buildMaskOverlay), so cap it at 255*maskOverlayOpacity*(1-opacity),
    // approached via the union stamp coverage (the preview's AA edge /
    // overlap model)
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
        const cap = 255 * maskOverlayOpacity * (1 - st.opacity);
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
    // (max over the stamps), and the overlay alpha is
    // maskOverlayOpacity x that (buildMaskOverlay) -- so build the overlay
    // as max(base overlay, maskOverlayOpacity x 255 x s_max x opacity),
    // pixel-exact to the committed mask
    const cov = strokeMaxCoverage(s, pts, raw);
    if (!cov) {
      // fallback: the old stacked discs (brush image not loaded yet)
      ctx.fillStyle = `rgba(128,128,128,${maskOverlayOpacity})`;
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
        const target = 255 * maskOverlayOpacity * st.opacity;
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
      ctx.globalAlpha = maskOverlayOpacity * st.opacity;
      ctx.drawImage(cov, 0, 0);
    }
    ctx.globalAlpha = 1;
    return;
  }
  // brush tool: colour x brush x opacity, NON-stacking like the commit:
  // the colour is blended in ONCE by the stroke's max coverage x
  // opacity (overlapping stamps do not darken the core). On a floating
  // selection, shift is the eraser: the same layer composited
  // destination-out, which lowers the affected pixels' alpha
  const cov = strokeMaxCoverage(s, pts, raw);
  if (cov) {
    const lctx = layerCtx(stampLayer, s.tw, s.th);
    lctx.globalCompositeOperation = "source-over";
    lctx.globalAlpha = 1;
    lctx.fillStyle = st.color;
    lctx.fillRect(0, 0, s.tw, s.th);
    // keep only the stroke's coverage (alpha = max shape over stamps)
    lctx.globalCompositeOperation = "destination-in";
    lctx.drawImage(cov, 0, 0);
    lctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = st.opacity;
    if (s.erase) ctx.globalCompositeOperation = "destination-out";
    ctx.drawImage(stampLayer.canvas, 0, 0);
    ctx.globalCompositeOperation = "source-over";
    ctx.globalAlpha = 1;
    return;
  }
  // fallback: the old stacked discs (brush image not loaded yet)
  ctx.fillStyle = st.color;
  if (s.erase) ctx.globalCompositeOperation = "destination-out";
  for (const [x, y] of pts) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  if (s.erase) ctx.globalCompositeOperation = "source-over";
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
  const baseCtx = s.img.getContext("2d"); // the canvas, or the selection image
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

// ---------------- copy / paste / floating selection ----------------
//
// The copy tool drags the crop tool's rectangle and stores that region of
// view.base (view.clip, fully opaque, in-memory). The paste tool drops it as
// a FLOATING SELECTION: an image riding on top of the canvas with its own
// position, size and rotation, with handles to reshape it and an apply /
// discard pair in its middle. It is drawn as an overlay only -- view.base is
// untouched until Apply composites it, so sync, snapshots and the draw undo
// stack never see it. While it is active the draw / smudge / clone tools
// paint on the selection (in its local frame) and the crop tool crops it;
// the mask tool keeps working on the canvas's own mask.
//
// Geometry: the selection is (cx, cy) centre + angle + display size (sw, sh)
// in canvas px; its image keeps its own resolution and is STRETCHED to fill
// sw x sh, so dragging a handle is a pure transform (no resampling) and the
// resample happens once, at Apply / stroke start (selMaterialize).

// the selection's display size never goes below this (px) or above that
const SEL_MIN = 8;
const SEL_MAX = 8192;

// tools whose strokes land on the selection while one is active
function selStrokeTool(tool) {
  return !!view.sel && (tool === "brush" || tool === "clone" || tool === "smudge");
}

// canvas px <-> selection-local px (local: x in [0, sw], y in [0, sh])
function frameLocal(F, p) {
  const ca = Math.cos(-F.angle), sa = Math.sin(-F.angle);
  const dx = p[0] - F.cx, dy = p[1] - F.cy;
  return [dx * ca - dy * sa + F.sw / 2, dx * sa + dy * ca + F.sh / 2];
}

function frameWorld(F, q) {
  const ca = Math.cos(F.angle), sa = Math.sin(F.angle);
  const dx = q[0] - F.sw / 2, dy = q[1] - F.sh / 2;
  return [F.cx + dx * ca - dy * sa, F.cy + dx * sa + dy * ca];
}

const selLocal = (p) => frameLocal(view.sel, p);
const selWorld = (q) => frameWorld(view.sel, q);

// the same point in the CENTRED local frame (x in [-sw/2, sw/2]): what the
// transform-drawn overlay (border, handles, cursor ring) lives in, while
// selLocal's [0, sw] frame is the selection image's own pixel grid
function selCentred(p) {
  const q = selLocal(p);
  return [q[0] - view.sel.sw / 2, q[1] - view.sel.sh / 2];
}

// the selection's image resampled to its current display size (the stretched
// view of it, in canvas px): the layer strokes paint on and the layer Apply
// composites
function selMaterialize(s) {
  const w = Math.max(1, Math.round(s.sw));
  const h = Math.max(1, Math.round(s.sh));
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(s.img, 0, 0, s.img.width, s.img.height, 0, 0, w, h);
  return c;
}

// the handles, in selection-local px: edge midpoints resize that edge (the
// opposite edge stays put), TR/BL scale with the aspect locked (opposite
// corner pinned), TL/BR rotate around the centre
function selHandles(s) {
  const w = s.sw / 2, h = s.sh / 2;
  return [
    { kind: "edge", edge: "l", x: -w, y: 0 },
    { kind: "edge", edge: "r", x: w, y: 0 },
    { kind: "edge", edge: "t", x: 0, y: -h },
    { kind: "edge", edge: "b", x: 0, y: h },
    { kind: "rot", x: -w, y: -h },
    { kind: "scale", corner: "tr", x: w, y: -h },
    { kind: "scale", corner: "bl", x: -w, y: h },
    { kind: "rot", x: w, y: h },
  ];
}

// grab a handle: the pointer must land within a screen-constant radius of it
function startSelDrag(p) {
  const s = view.sel;
  const q = selCentred(p);
  const tol = Math.max(8 / view.scale, 7 / view.scale);
  let best = null, bestD = 1 / 0;
  for (const h of selHandles(s)) {
    const d = Math.hypot(q[0] - h.x, q[1] - h.y);
    if (d <= tol && d < bestD) { best = h; bestD = d; }
  }
  if (!best) return false;
  const d = { kind: best.kind, F: { cx: s.cx, cy: s.cy, angle: s.angle, sw: s.sw, sh: s.sh } };
  if (d.kind === "rot") d.base = Math.atan2(best.y, best.x); // corner's own direction
  if (d.kind === "edge") d.edge = best.edge;
  if (d.kind === "scale") d.corner = best.corner;
  view.selDrag = d;
  return true;
}

const clampN = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

// apply a handle drag to the selection; the frame F frozen at drag start is
// the reference, so the fixed edge / corner never drifts
function applySelDrag(p) {
  const d = view.selDrag, F = d.F, s = view.sel;
  const q = frameLocal(F, p);
  const ca = Math.cos(F.angle), sa = Math.sin(F.angle);
  if (d.kind === "rot") {
    s.angle = Math.atan2(p[1] - F.cy, p[0] - F.cx) - d.base;
    return;
  }
  if (d.kind === "edge") {
    const horiz = d.edge === "l" || d.edge === "r";
    if (horiz) {
      const w = Math.round(clampN(d.edge === "r" ? q[0] : F.sw - q[0], SEL_MIN, SEL_MAX));
      const a = frameWorld(F, d.edge === "r" ? [0, F.sh / 2] : [F.sw, F.sh / 2]);
      const k = (d.edge === "r" ? 1 : -1) * (w / 2);
      s.sw = w;
      s.cx = a[0] + k * ca;
      s.cy = a[1] + k * sa;
    } else {
      const h = Math.round(clampN(d.edge === "b" ? q[1] : F.sh - q[1], SEL_MIN, SEL_MAX));
      const a = frameWorld(F, d.edge === "b" ? [F.sw / 2, 0] : [F.sw / 2, F.sh]);
      const k = (d.edge === "b" ? 1 : -1) * (h / 2);
      s.sh = h;
      s.cx = a[0] - k * sa;
      s.cy = a[1] + k * ca;
    }
    return;
  }
  // aspect-locked scale: the dragged corner follows the cursor along the
  // diagonal through the pinned corner
  const a = F.sw / F.sh, dd = a * a + 1;
  const t = clampN(
    d.corner === "tr" ? (q[0] * a + F.sh - q[1]) / dd : ((F.sw - q[0]) * a + q[1]) / dd,
    SEL_MIN / a, SEL_MAX / a
  );
  const w = Math.round(t * a), h = Math.round(t);
  const c = frameWorld(F, d.corner === "tr" ? [w / 2, F.sh - h / 2] : [F.sw - w / 2, h / 2]);
  s.sw = w;
  s.sh = h;
  s.cx = c[0];
  s.cy = c[1];
}

// the live stroke preview on a selection is built on a scratch layer (the
// stroke's own frame), then blitted inside the selection's transform
const selPreview = { canvas: document.createElement("canvas"), w: 0, h: 0 };

function drawSelectionOverlay(ctx) {
  const s = view.sel;
  ctx.save();
  ctx.translate(s.cx, s.cy);
  ctx.rotate(s.angle);
  ctx.imageSmoothingEnabled = true; // the selection is resampled content
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(s.img, 0, 0, s.img.width, s.img.height, -s.sw / 2, -s.sh / 2, s.sw, s.sh);
  const st = view.stroke;
  if (st && st.sel) {
    const pc = selPreview.canvas;
    if (selPreview.w !== st.tw || selPreview.h !== st.th) {
      pc.width = st.tw;
      pc.height = st.th;
      selPreview.w = st.tw;
      selPreview.h = st.th;
    }
    const pctx = pc.getContext("2d");
    pctx.setTransform(1, 0, 0, 1, 0, 0);
    pctx.globalCompositeOperation = "source-over";
    pctx.globalAlpha = 1;
    pctx.clearRect(0, 0, st.tw, st.th);
    pctx.imageSmoothingEnabled = true;
    pctx.drawImage(st.img, 0, 0);
    drawStrokePreview(pctx, st);
    // the preview layer IS the selection image + stroke, so it goes exactly
    // where the image itself was drawn (centred on the transform origin) --
    // blitting it at 0,0 would show a second copy offset by half its size
    ctx.drawImage(pc, -st.tw / 2, -st.th / 2, st.tw, st.th);
  }
  // faint border + handles (constant on screen)
  ctx.lineWidth = 3 / view.scale;
  ctx.strokeStyle = "rgba(0,0,0,0.55)";
  ctx.strokeRect(-s.sw / 2, -s.sh / 2, s.sw, s.sh);
  ctx.lineWidth = 1 / view.scale;
  ctx.strokeStyle = "rgba(255,255,255,0.75)";
  ctx.strokeRect(-s.sw / 2, -s.sh / 2, s.sw, s.sh);
  const hs = 7 / view.scale;
  for (const h of selHandles(s)) {
    ctx.fillStyle = "rgba(255,255,255,0.95)";
    ctx.fillRect(h.x - hs / 2, h.y - hs / 2, hs, hs);
    ctx.lineWidth = 1 / view.scale;
    ctx.strokeStyle = "rgba(0,0,0,0.6)";
    ctx.strokeRect(h.x - hs / 2, h.y - hs / 2, hs, hs);
  }
  ctx.restore();
}

// the paste tool carries the copied area at the cursor until it is dropped
// (clipped to the image, so it is only visible on the canvas)
function drawPasteGhost(ctx) {
  const cl = view.clip;
  if (!cl || view.sel) return;
  const x = Math.round((view.cursor[0] - view.ox) / view.scale - cl.w / 2);
  const y = Math.round((view.cursor[1] - view.oy) / view.scale - cl.h / 2);
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, view.w, view.h);
  ctx.clip();
  ctx.globalAlpha = 0.6;
  ctx.drawImage(cl.img, x, y);
  ctx.globalAlpha = 1;
  ctx.lineWidth = 1 / view.scale;
  ctx.strokeStyle = "rgba(255,255,255,0.9)";
  ctx.strokeRect(x, y, cl.w, cl.h);
  ctx.restore();
}

// paste tool click: drop the copied area, centred on the cursor. It may hang
// off the canvas (Apply clips it); a new drop replaces an active selection.
function pasteAt(p) {
  if (!view.clip) {
    setStatus("nothing copied yet \u2014 drag a rectangle with the copy tool first");
    return;
  }
  view.stroke = null;
  view.selDrag = null;
  view.crop = null;
  view.sel = { img: view.clip.img, sw: view.clip.w, sh: view.clip.h, cx: p[0], cy: p[1], angle: 0 };
  viewRedraw();
  setStatus("floating selection \u2014 drag its handles, Enter applies, Esc discards");
}

// copy tool pointerup: store the region of the committed image (the copied
// area is opaque; erasing to transparency happens on the floating selection)
function commitCopy(r) {
  // integer canvas px: the copied area is a pixel grid, not a sub-pixel rect
  const x = Math.round(r.x), y = Math.round(r.y);
  const w = Math.max(1, Math.round(r.w)), h = Math.max(1, Math.round(r.h));
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  c.getContext("2d").drawImage(view.base, x, y, w, h, 0, 0, w, h);
  view.clip = { img: c, w, h };
  syncToolbar(); // app.js: the paste button becomes enabled
  setStatus(`copied ${w}\u00d7${h} \u2014 paste tool (V), click to place it`);
}

// crop tool with an active selection: crop the SELECTION to the dragged
// rectangle (the canvas entity is untouched, no new entity is created)
function cropSelection(r) {
  const s = view.sel;
  const cs = [[r.x, r.y], [r.x + r.w, r.y], [r.x, r.y + r.h], [r.x + r.w, r.y + r.h]]
    .map(selLocal);
  // rounded to the selection's pixel grid: the pointer's sub-pixel position
  // must not shift the crop by a whole row / column
  const x0 = clampN(Math.round(Math.min(...cs.map((q) => q[0]))), 0, s.sw - 1);
  const y0 = clampN(Math.round(Math.min(...cs.map((q) => q[1]))), 0, s.sh - 1);
  const x1 = clampN(Math.round(Math.max(...cs.map((q) => q[0]))), x0 + 1, s.sw);
  const y1 = clampN(Math.round(Math.max(...cs.map((q) => q[1]))), y0 + 1, s.sh);
  const w = x1 - x0, h = y1 - y0;
  if (w < 2 || h < 2) {
    setStatus("the crop must keep at least 2\u00d72 px of the selection");
    return;
  }
  const m = selMaterialize(s);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  c.getContext("2d").drawImage(m, x0, y0, w, h, 0, 0, w, h);
  const centre = selWorld([x0 + w / 2, y0 + h / 2]);
  s.img = c;
  s.sw = w;
  s.sh = h;
  s.cx = centre[0];
  s.cy = centre[1];
  viewRedraw();
  setStatus(`cropped the selection to ${w}\u00d7${h}`);
}

// Apply: composite the selection (stretched + rotated, resampled once) onto
// the committed image with its per-pixel alpha, as ONE draw-undo step
async function commitSelection() {
  const s = view.sel;
  if (!s) return;
  const snap = await snapCanvas(view.base);
  const m = selMaterialize(s);
  const ctx = view.base.getContext("2d");
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.translate(Math.round(s.cx), Math.round(s.cy));
  ctx.rotate(s.angle);
  ctx.drawImage(m, -m.width / 2, -m.height / 2);
  ctx.restore();
  view.sel = null;
  view.selDrag = null;
  pushUndo("draw", snap);
  markDirty();
  syncUndoButtons(); // app.js
  viewRedraw();
  setStatus("applied the floating selection");
}

// Cancel: throw the selection away (no history step, the base never changed)
function cancelSelection() {
  if (!view.sel) return;
  view.sel = null;
  view.selDrag = null;
  view.stroke = null;
  viewRedraw();
  setStatus("discarded the floating selection");
}

// the apply / discard pair, centred on the selection (holder-relative
// anchoring, same pattern as the thumbnail hover preview)
function syncSelActions() {
  const el = document.getElementById("sel-actions");
  if (!el) return;
  if (!view.sel || !view.has) {
    el.hidden = true;
    return;
  }
  const holder = document.getElementById("canvas-holder").getBoundingClientRect();
  const c = cv().getBoundingClientRect();
  el.hidden = false;
  el.style.left = `${c.left - holder.left + view.ox + view.sel.cx * view.scale}px`;
  el.style.top = `${c.top - holder.top + view.oy + view.sel.cy * view.scale}px`;
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

// pointerup on a finished drag: commit the (already clamped) rect -- the copy
// tool stores it, the crop tool crops the floating selection (while one is
// active) or the canvas image; a zero-area selection (plain click) is a no-op
function commitCrop() {
  const r = cropRect();
  const tool = Tools.current;
  view.crop = null;
  viewRedraw();
  if (!r) return;
  if (tool === "copy") { commitCopy(r); return; }
  if (view.sel) { cropSelection(r); return; }
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
    // a floating selection is active: its handles grab the pointer first
    if (view.sel && startSelDrag(p)) {
      c.setPointerCapture(e.pointerId);
      viewRedraw();
      return;
    }
    if (tool === "paste") {
      pasteAt(p); // drop the copied area as a floating selection
      return;
    }
    if (tool === "crop" || tool === "copy") {
      // start a rectangle selection; the pointer may be anywhere on the
      // element (including outside the image) -- the used region is the
      // clamped intersection, computed in cropRect
      view.crop = { sx: p[0], sy: p[1], ex: p[0], ey: p[1] };
      c.setPointerCapture(e.pointerId);
      viewRedraw();
      return;
    }
    if (tool === "brush" && e.ctrlKey) {
      // colour-picker mode: set the brush colour from the clicked pixel
      pickBrushColor(p);
      return;
    }
    // with a floating selection active, the draw / clone / smudge strokes
    // land on IT, in its own (rotated) frame: the stroke runs on the
    // selection's image resampled to its display size, in local px
    const onSel = selStrokeTool(tool);
    const pt = onSel ? selLocal(p) : p;
    const tw = onSel ? Math.round(view.sel.sw) : view.w;
    const th = onSel ? Math.round(view.sel.sh) : view.h;
    const img = onSel ? selMaterialize(view.sel) : view.base;
    if (tool === "clone") {
      if (e.ctrlKey) {
        // pick/move the clone source S; not a stroke (no pointer capture,
        // so the following pointerup has nothing to commit). Stored in
        // canvas px: on a selection it maps into the selection's frame per
        // stroke, so moving the selection carries the sampled content along
        view.cloneSource = [p[0], p[1]];
        // re-anchor an active SHIFT gesture: the moved source lands on S
        // (the cursor is on S, so the relative offset is zero)
        if (view.cloneGesture) view.cloneGesture = [0, 0];
        viewRedraw();
        return;
      }
      const src = cloneSourcePoint();
      if (!src) {
        setStatus("clone: ctrl-click to pick a source point first");
        return;
      }
      const srcL = onSel ? selLocal(src) : src;
      view.stroke = {
        tool,
        points: [pt],
        size: brushSizeCanvas(),
        // frozen for the whole stroke, integer canvas px, so the preview
        // and the commit sample identical pixels
        cloneOffset: [Math.round(srcL[0] - pt[0]), Math.round(srcL[1] - pt[1])],
        // the pre-stroke image the commit samples
        img,
        tw,
        th,
        sel: onSel,
        // pre-stroke snapshot for the draw undo stack; toBlob captures at
        // call time, so the encode overlaps the drag. A stroke on a floating
        // selection is no canvas edit, so it gets no history step
        snapP: onSel ? null : snapCanvas(view.base),
      };
      c.setPointerCapture(e.pointerId);
      viewRedraw();
      return;
    }
    if (tool === "smudge") {
      // smudge is a drag: the first stamp samples itself (a no-op), so the
      // stamp counter starts at 1; the working image starts as the current
      // (pre-stroke) target image and rides on the stroke object, so the
      // held preview keeps working after the stroke ends
      view.stroke = {
        tool,
        points: [pt],
        size: brushSizeCanvas(),
        smudgeStamps: 1,
        work: newSmudgeWork(img, tw, th),
        img,
        tw,
        th,
        sel: onSel,
        snapP: onSel ? null : snapCanvas(view.base),
      };
      c.setPointerCapture(e.pointerId);
      viewRedraw();
      return;
    }
    view.stroke = {
      tool,
      points: [pt],
      // freeze the canvas-px footprint at stroke start (the commit stamps
      // with a single size, so a mid-stroke zoom can't change it)
      size: brushSizeCanvas(),
      // shift = mask eraser, and (on a floating selection) draw eraser
      erase: e.shiftKey && (tool === "mask" || (tool === "brush" && onSel)),
      img,
      tw,
      th,
      sel: onSel,
      // pre-stroke snapshot for the right undo stack (draw vs mask)
      snapP: tool === "mask" || onSel ? null : snapCanvas(view.base),
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
    if (view.selDrag) {
      applySelDrag(p);
      viewRedraw();
      return;
    }
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
      const q = view.stroke.sel ? selLocal(p) : p;
      if (Math.hypot(q[0] - lx, q[1] - ly) * view.scale >= 1) {
        view.stroke.points.push(q);
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
    // SHIFT released during the stroke: the source reverts only now
    if (view.cloneGesture && !cloneShift) view.cloneGesture = null;
    if (s.tool === "smudge" && s.points.length < 2) {
      // a single smudge click smears nothing: no commit, no history step
      clearBrushStamps();
      viewRedraw();
      return;
    }
    commitStroke(s, st); // local, synchronous: the preview's own result
    clearBrushStamps(); // the preview is committed; its stamp cache is done
    if (s.sel) {
      // the edited selection image IS the committed result: no canvas edit,
      // so no history step and no sync (view.base never changed)
      view.sel.img = s.img;
      viewRedraw();
      return;
    }
    // the snapshot promise lands on the stack immediately (its encoding
    // overlaps the drag); undoStep/redoStep await it on pop
    pushUndo(s.tool, s.snapP || s.snapMaskP);
    markDirty(); // the server's copy is behind; idle sync (or the next
    // forceSync before a server-side read) pushes the state
    syncUndoButtons(); // app.js
    viewRedraw();
  };
  c.addEventListener("pointerup", (e) => {
    if (view.selDrag) {
      view.selDrag = null; // the transform is live: nothing to resample
      viewRedraw();
      return;
    }
    if (view.crop) { commitCrop(); return; }
    endStroke();
  });
  c.addEventListener("pointercancel", (e) => {
    if (view.selDrag) {
      view.selDrag = null;
      viewRedraw();
      return;
    }
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
  c.addEventListener("auxclick", (e) => {
    if (e.button === 1) e.preventDefault();
  });
}
