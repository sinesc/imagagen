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
  stroke: null, // {tool, points: [[x,y]...], erase}
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
  if (view.mask) ctx.drawImage(view.mask, 0, 0, view.w, view.h);
  if (view.stroke) drawPreview(ctx);
  drawCursor(ctx);
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

// outline of the brush footprint at the cursor (pre-draw preview)
function drawCursor(ctx) {
  if (!view.has || !view.cursor) return;
  const r = brushSizeCanvas() / 2;
  if (r < 0.5) return;
  const x = (view.cursor[0] - view.ox) / view.scale;
  const y = (view.cursor[1] - view.oy) / view.scale;
  // dark halo + thin white ring: visible on any background, 1-screen-px wide
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

function drawPreview(ctx) {
  const st = Tools.get();
  const s = view.stroke;
  if (s.tool === "mask" && s.erase) return; // erasing: committed state is truth
  const pts = stampPositions(s.points, Math.max(1, s.size / 12));
  const r = Math.max(s.size / 2, 0.5);
  ctx.globalAlpha = st.opacity;
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
    let resp;
    try {
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
    } catch (err) {
      view.stroke = null;
      setStatus(`stroke failed: ${err.message}`);
      viewRedraw();
      return;
    }
    storeUndoFlags(resp); // the response carries the four undo/redo flags
    // Drop the stroke WITHOUT redrawing in between: the screen keeps the
    // last rendered frame (preview over the old image) until viewRefresh()
    // has fetched and drawn the committed pixels, so the stroke never
    // vanishes during the request round trip (no flicker on release).
    view.stroke = null;
    view.refreshing = true;
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
