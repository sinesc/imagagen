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
  cv().width = Math.max(1, Math.floor(w * view.dpr));
  cv().height = Math.max(1, Math.floor(h * view.dpr));
  viewRedraw();
}

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

// brush footprint outline: dark halo + thin white ring, visible on any
// background, 1-screen-px wide
function drawBrushRing(ctx, x, y) {
  const r = brushSizeCanvas() / 2;
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

// scratch canvas for the erase preview (punched-out mask overlay)
const eraseLayer = {
  canvas: document.createElement("canvas"),
  w: 0,
  h: 0,
};

function drawPreview(ctx) {
  const st = Tools.get();
  const s = view.stroke;
  const pts = stampPositions(s.points, Math.max(1, s.size / 12));
  const r = Math.max(s.size / 2, 0.5);
  ctx.globalAlpha = st.opacity;
  if (s.tool === "smudge") {
    // live smudge preview: SEQUENTIAL smear, accumulated on an offscreen
    // working image (like a real smudge, unlike brush/clone which recompute
    // from scratch). For each new stamp centre, copy the working image at
    // the previous centre onto the current one (integer displacement, same
    // size, so no resampling) and alpha-blend it by the opacity, clipped to
    // the stamp's disc. This is the same sequential accumulation the server
    // commit does over its working copy (only the brush-edge antialiasing
    // differs), so the no-flicker hand-off stays invisible.
    const wctx = view.smudgeWork.canvas.getContext("2d");
    while (s.smudgeStamps < pts.length) {
      const i = s.smudgeStamps++; // new centre; i-1 is its predecessor
      const [ox, oy] = pts[i - 1];
      const [cx, cy] = pts[i];
      const dx = Math.round(cx - ox);
      const dy = Math.round(cy - oy);
      if (!dx && !dy) continue;
      wctx.save();
      wctx.beginPath();
      wctx.arc(cx, cy, r, 0, Math.PI * 2);
      wctx.clip();
      wctx.globalAlpha = st.opacity;
      // drawImage(img, dx, dy) shows img[p - (dx, dy)] at dest p: the whole
      // working image at -displacement, so dest shows the pixels the
      // PREVIOUS stamp just moved (off-image source leaves dest as-is)
      wctx.drawImage(view.smudgeWork.canvas, -dx, -dy);
      wctx.restore();
    }
    // blit the working image; the mask overlay goes on top of it so the
    // stack matches the base rendering (image, then mask)
    ctx.globalAlpha = 1;
    ctx.drawImage(view.smudgeWork.canvas, 0, 0);
    if (view.mask) ctx.drawImage(view.mask, 0, 0, view.w, view.h);
    ctx.globalAlpha = 1;
    return;
  }
  if (s.tool === "clone") {
    // live clone preview: paint the pre-stroke image (view.img) translated
    // by the frozen offset, clipped to the UNION of the stamp footprints,
    // in ONE composite so the opacity applies to the stroke as a whole
    // (overlapping stamps must not stack, same as the server commit).
    // Off-image source reads as transparent, same as the server skips it.
    const [ox, oy] = s.cloneOffset;
    ctx.save();
    ctx.beginPath();
    for (const [x, y] of pts) {
      ctx.moveTo(x + r, y);
      ctx.arc(x, y, r, 0, Math.PI * 2);
    }
    ctx.clip();
    ctx.globalAlpha = st.opacity;
    // dest pixel p must show img[p + (ox, oy)]: drawImage(img, dx, dy) shows
    // img[p - (dx, dy)], so the image goes at -offset
    ctx.drawImage(view.img, -ox, -oy);
    ctx.restore();
    ctx.globalAlpha = 1;
    return;
  }
  if (s.tool === "mask" && s.erase) {
    // erasing removes the grey overlay: copy it into a scratch canvas, punch
    // out the stamped spots (destination-out), draw it over the image
    if (view.mask) {
      let L = eraseLayer;
      if (L.w !== view.w || L.h !== view.h) {
        L.canvas.width = view.w;
        L.canvas.height = view.h;
        L.w = view.w;
        L.h = view.h;
      }
      const lctx = L.canvas.getContext("2d");
      lctx.globalCompositeOperation = "source-over";
      lctx.globalAlpha = 1;
      lctx.clearRect(0, 0, L.canvas.width, L.canvas.height);
      lctx.drawImage(view.mask, 0, 0);
      lctx.globalCompositeOperation = "destination-out";
      lctx.globalAlpha = st.opacity; // the server scales removal by opacity
      lctx.fillStyle = "#000";
      for (const [x, y] of pts) {
        lctx.beginPath();
        lctx.arc(x, y, r, 0, Math.PI * 2);
        lctx.fill();
      }
      lctx.globalCompositeOperation = "source-over";
      lctx.globalAlpha = 1;
      ctx.drawImage(L.canvas, 0, 0);
    }
    ctx.globalAlpha = 1;
    return;
  }
  ctx.fillStyle = s.tool === "mask" ? "rgba(128,128,128,0.9)" : st.color;
  for (const [x, y] of pts) {
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
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
    }
  };
  c.addEventListener("pointerup", endStroke);
  c.addEventListener("pointercancel", endStroke);

  // wheel zoom, anchored at the cursor
  c.addEventListener(
    "wheel",
    (e) => {
      if (!view.has) return;
      e.preventDefault();
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
