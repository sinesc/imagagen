/* Canvas view: renders the server-authoritative image with a transparent
   grey mask overlay, handles zoom (wheel, cursor-anchored) and pan
   (middle-button drag) as pure client-side transforms, and captures left
   pointer strokes as point lists in canvas-pixel coordinates.

   Strokes are stamped server-side (POST /api/canvas/stroke) -- the server
   is the single source of truth. While a stroke is in flight the client
   draws an optimistic preview (same stamp layout) so the brush feels
   responsive; the authoritative pixels arrive with the canvas re-fetch. */
"use strict";

const view = {
  img: null, // HTMLImageElement: server image
  mask: null, // offscreen canvas: grey mask overlay
  has: false,
  w: 0,
  h: 0,
  scale: 1, // screen = canvas * scale + o (CSS px)
  ox: 0,
  oy: 0,
  dpr: 1,
  stroke: null, // {tool, points: [[x,y]...], erase, cloneOffset, smudgeStamps}
  smudgeWork: null, // {canvas}: smudge tool's offscreen working image (canvas
  // px), the pre-stroke image plus the drag accumulated stamp by stamp
  cloneSource: null, // [x,y] canvas px: clone tool's picked source S
  cloneGesture: null, // [ox,oy] canvas px: CTRL held -> the source follows
  // the cursor at this relative offset (a temporary move of the picked S)
  crop: null, // {sx, sy, ex, ey} raw canvas px of the crop tool's drag; the
  // region used is its clamped intersection with the image (cropRect)
  pan: null,
  cursor: null, // [mx,my] in CSS px relative to the canvas element
  refreshing: false, // stroke commit in flight: don't clobber the preview
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
  if (view.img) ctx.drawImage(view.img, 0, 0);
  // an active erase stroke draws its own punched copy of the mask; the base
  // mask would show through underneath it (double opacity)
  const erasing = view.stroke && view.stroke.erase;
  if (view.mask && !erasing) ctx.drawImage(view.mask, 0, 0, view.w, view.h);
  if (view.stroke) drawPreview(ctx);
  if (view.crop) drawCropOverlay(ctx);
  drawCursor(ctx);
  drawCloneCrosshairs(ctx);
  $("zoom-label").textContent = `${Math.round(view.scale * 100)}%`;
}

// mask PNG (greyscale, luminance = alpha) -> transparent grey overlay
function buildMaskLayer(maskImg) {
  const layer = document.createElement("canvas");
  layer.width = maskImg.naturalWidth;
  layer.height = maskImg.naturalHeight;
  const lctx = layer.getContext("2d", { willReadFrequently: true });
  lctx.drawImage(maskImg, 0, 0);
  const data = lctx.getImageData(0, 0, layer.width, layer.height);
  const d = data.data;
  for (let i = 0; i < d.length; i += 4) {
    const a = d[i]; // greyscale: R=G=B=luminance
    d[i] = d[i + 1] = d[i + 2] = 128;
    d[i + 3] = Math.min(255, a * 0.9);
  }
  lctx.putImageData(data, 0, 0);
  return layer;
}

// fetch image + mask from the server and re-render
async function viewRefresh() {
  const [ri, rm] = await Promise.all([
    fetch("/api/canvas.png"),
    fetch("/api/mask.png"),
  ]);
  const had = view.has;
  const oldW = view.w;
  const oldH = view.h;
  view.has = ri.ok;
  if (!view.has) {
    view.img = null;
    view.mask = null;
    view.w = view.h = 0;
    viewRedraw();
    return;
  }
  const img = new Image();
  img.src = URL.createObjectURL(await ri.blob());
  await new Promise((r) => {
    img.onload = r;
  });
  view.img = img;
  view.w = img.naturalWidth;
  view.h = img.naturalHeight;
  if (rm.ok) {
    const m = new Image();
    m.src = URL.createObjectURL(await rm.blob());
    await new Promise((r) => {
      m.onload = r;
    });
    view.mask = buildMaskLayer(m);
  } else {
    view.mask = null;
  }
  // keep the current zoom/pan unless the image appeared or changed size
  if (!had || view.w !== oldW || view.h !== oldH) {
    view.cloneSource = null; // old source point no longer makes sense
    view.cloneGesture = null;
    view.smudgeWork = null; // a working canvas is size-bound
    viewResize(); // the canvas element may have been hidden (sized to 0)
    viewFit();
  }
  viewRedraw();
}

// ---------------- optimistic stroke preview ----------------

// mirror of Canvas.stamp_positions (server is authoritative on commit)
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
// The server stamps with the EXACT brush shape: "circle" is built in, other
// brushes are greyscale PNGs (luminance = alpha) resized to the stamp size.
// The preview uses the same pixel data so a soft brush previews soft: the
// brush's PNG bytes come from /api/brush/{name}.png once per name, and the
// luminance is converted to alpha the same way as the mask overlay
// (buildMaskLayer).
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
  const x = (view.cursor[0] - view.ox) / view.scale;
  const y = (view.cursor[1] - view.oy) / view.scale;
  drawBrushRing(ctx, x, y);
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

// crosshairs for the clone tool: at the source, and (during a clone stroke)
// at the live sample point (cursor + frozen offset), with a faint line
// between the two so the user sees where pixels are copied from
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
  if (src) {
    cross(...src);
    drawBrushRing(ctx, ...src); // the brush at the source too
  }
  if (view.stroke && view.stroke.tool === "clone" && src && view.cursor) {
    const [x, y] = cloneCursorSample();
    line(src[0], src[1], x, y, 1 / view.scale, "rgba(255,255,255,0.35)");
    cross(x, y);
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
// canvas-px resolution, accumulated stamp by stamp (see drawPreview)
function newSmudgeWork() {
  const canvas = document.createElement("canvas");
  canvas.width = view.w;
  canvas.height = view.h;
  canvas.getContext("2d").drawImage(view.img, 0, 0);
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

function drawPreview(ctx) {
  const st = Tools.get();
  const s = view.stroke;
  const pts = stampPositions(s.points, Math.max(1, s.size / 12));
  const r = Math.max(s.size / 2, 0.5);
  // the selected brush at the stroke's frozen size: the exact shape the
  // server stamps with (null until the brush image has loaded, then the
  // branches fall back to the disc)
  const raw = brushStamp(st.brush, s.size, null);
  ctx.globalAlpha = st.opacity;
  if (s.tool === "smudge") {
    // live smudge preview: SEQUENTIAL smear, accumulated on an offscreen
    // working image (like a real smudge, unlike brush/clone which recompute
    // from scratch). For each new stamp centre, copy the brush-shaped
    // region at the previous centre onto the current one (integer
    // displacement, same size, so no resampling) and alpha-blend it by the
    // brush shape x the opacity. This is the same sequential accumulation
    // the server commit does over its working copy (only the brush-edge
    // antialiasing differs), so the no-flicker hand-off stays invisible.
    const wctx = view.smudgeWork.canvas.getContext("2d");
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
        wctx.drawImage(view.smudgeWork.canvas, dx, dy);
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
      // read as the server's src sub-rect
      tctx.clearRect(0, 0, raw.width, raw.height);
      tctx.drawImage(view.smudgeWork.canvas, dx - x0, dy - y0);
      // mask the copy by the brush shape, then blend it by the opacity
      tctx.globalCompositeOperation = "destination-in";
      tctx.globalAlpha = 1;
      tctx.drawImage(raw, 0, 0);
      tctx.globalCompositeOperation = "source-over";
      wctx.globalAlpha = st.opacity;
      wctx.drawImage(smudgeTemp, x0, y0);
      wctx.globalAlpha = 1;
    }
    // blit the working image; the mask overlay goes on top of it so the
    // stack matches the base rendering (image, then mask)
    ctx.globalAlpha = 1;
    ctx.drawImage(view.smudgeWork.canvas, 0, 0);
    if (view.mask) ctx.drawImage(view.mask, 0, 0, view.w, view.h);
    return;
  }
  if (s.tool === "clone") {
    // live clone preview: paint the pre-stroke image (view.img) translated
    // by the frozen offset, masked by the UNION of the stamp footprints,
    // in ONE composite so the opacity applies to the stroke as a whole
    // (overlapping stamps must not stack, same as the server commit).
    // The coverage canvas accumulates the shape alphas source-over: 1 -
    // prod(1 - shape_i), exactly what the server commits; off-image source
    // reads as transparent, same as the server skipping it.
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
      ctx.drawImage(view.img, -ox, -oy);
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
    // img[p - (dx, dy)], so the image goes at -offset
    lctx.drawImage(view.img, -ox, -oy);
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
    // stacking, same as the server: sequential stamps give min(region,
    // cap x max shape)). The overlay alpha is the mask alpha x 0.9
    // (buildMaskLayer), so cap it at 0.9*255*(1-opacity), approached via
    // the union stamp coverage (the preview's AA edge / overlap model)
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
    // adding paints the fixed brush x 255 x opacity (no stacking); the
    // overlay alpha is the mask alpha x 0.9 (buildMaskLayer), so the
    // stamp's own alpha is 0.9 x shape
    const tint = raw
      ? brushStamp(st.brush, s.size, "rgba(128,128,128,0.9)")
      : null;
    if (tint) {
      for (const [x, y] of pts) {
        ctx.drawImage(
          tint,
          Math.round(x - tint.width / 2),
          Math.round(y - tint.height / 2)
        );
      }
    } else {
      ctx.fillStyle = "rgba(128,128,128,0.9)";
      for (const [x, y] of pts) {
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    return;
  }
  // brush tool: colour x brush x opacity, stamped sequentially so
  // overlapping stamps stack like the server commit
  const tint = raw ? brushStamp(st.brush, s.size, st.color) : null;
  if (tint) {
    for (const [x, y] of pts) {
      ctx.drawImage(
        tint,
        Math.round(x - tint.width / 2),
        Math.round(y - tint.height / 2)
      );
    }
  } else {
    ctx.fillStyle = st.color;
    for (const [x, y] of pts) {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.globalAlpha = 1;
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
        // frozen for the whole stroke, integer canvas px, so the client
        // preview and the server commit sample identical pixels
        cloneOffset: [Math.round(src[0] - p[0]), Math.round(src[1] - p[1])],
      };
      c.setPointerCapture(e.pointerId);
      viewRedraw();
      return;
    }
    if (tool === "smudge") {
      // smudge is a drag: the first stamp samples itself (a no-op), so the
      // stamp counter starts at 1; the working image starts as the current
      // (pre-stroke) canvas image
      view.smudgeWork = newSmudgeWork();
      view.stroke = {
        tool,
        points: [p],
        size: brushSizeCanvas(),
        smudgeStamps: 1,
      };
      c.setPointerCapture(e.pointerId);
      viewRedraw();
      return;
    }
    view.stroke = {
      tool,
      points: [p],
      // freeze the canvas-px footprint at stroke start (the server stamps
      // with a single size, so a mid-stroke zoom can't change it)
      size: brushSizeCanvas(),
      erase: tool === "mask" && e.shiftKey, // shift = mask eraser
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
      // while the commit round-trip is in flight the last frame (with the
      // optimistic preview) must be kept on screen -- redrawing for the
      // cursor outline would flash the un-stroked image (see endStroke)
      !view.refreshing &&
      (!oldCursor ||
        Math.hypot(
          p[0] - (oldCursor[0] - view.ox) / view.scale,
          p[1] - (oldCursor[1] - view.oy) / view.scale
        ) * view.scale >= 1)
    ) {
      redraw = true; // cursor outline moved at least 1 screen px
    }
    if (redraw) viewRedraw();
  });
  c.addEventListener("pointerleave", () => {
    view.cursor = null;
    if (!view.refreshing) viewRedraw();
  });

  const endStroke = async () => {
    if (view.pan) {
      view.pan = null;
      return;
    }
    if (!view.stroke) return;
    const s = view.stroke;
    const st = Tools.get();
    // CTRL released during the stroke: the source reverts only now
    // (no redraw -- the held frame stays until viewRefresh redraws)
    if (view.cloneGesture && !cloneCtrl) view.cloneGesture = null;
    // Freeze the stroke BEFORE the round trip: stop recording points and
    // hold the last frame (with the preview) until the committed pixels
    // arrive -- otherwise the stroke would keep following the cursor while
    // the request is in flight, and the preview would vanish (flicker).
    view.stroke = null;
    view.refreshing = true;
    let resp;
    try {
      if (s.tool === "clone") {
        resp = await api("/api/canvas/clone", {
          points: s.points,
          offset: s.cloneOffset,
          size: s.size, // canvas px (screen mode already divided by zoom)
          opacity: st.opacity,
          brush: st.brush,
        });
      } else if (s.tool === "smudge") {
        resp = await api("/api/canvas/smudge", {
          points: s.points,
          size: s.size, // canvas px (screen mode already divided by zoom)
          opacity: st.opacity,
          brush: st.brush,
        });
        view.smudgeWork = null;
      } else {
        resp = await api("/api/canvas/stroke", {
          points: s.points,
          size: s.size, // canvas px (screen mode already divided by zoom)
          opacity: st.opacity,
          brush: st.brush,
          color: st.color || "#ffffff",
          // Tools.current is "brush"|"mask"; the API wants "image"|"mask"
          target: s.tool === "mask" ? "mask" : "image",
          erase: s.erase,
        });
      }
    } catch (err) {
      view.smudgeWork = null;
      clearBrushStamps();
      view.refreshing = false;
      setStatus(`stroke failed: ${err.message}`, true);
      viewRedraw();
      return;
    }
    storeUndoFlags(resp); // the response carries the four undo/redo flags
    try {
      await viewRefresh();
    } finally {
      view.refreshing = false;
      clearBrushStamps(); // the stamp cache was sized for this stroke
    }
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
