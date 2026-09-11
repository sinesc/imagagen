/* imagagen UI: controls, thumbnails, WS events, toolbar wiring.
   Canvas rendering/input live in canvas.js, tool state in tools.js. */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  generating: false,
  job: null,
  thumbs: [], // {job, index, url, seed}
  currentThumb: -1,
};

function setStatus(text) {
  $("status-line").textContent = text || "";
}

// POST-only API helper. Every call is a POST (some endpoints, like
// /api/canvas/undo, take no body -- an empty {} is sent, which FastAPI
// ignores for body-less endpoints).
async function api(path, body = {}) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    // surface the server's error message (e.g. "mask is empty") instead of
    // a bare status code when one was sent
    let msg = `${path}: ${res.status}`;
    try {
      const j = await res.json();
      if (j && j.error) msg = j.error;
    } catch (e) { /* non-JSON error body */ }
    throw new Error(msg);
  }
  return res.json();
}

function renderThumbs() {
  const strip = $("thumbs");
  strip.innerHTML = "";
  state.thumbs.forEach((t, i) => {
    const el = document.createElement("img");
    el.src = t.url;
    el.title = `#${i} seed ${t.seed}`;
    if (i === state.currentThumb) el.classList.add("selected");
    el.onclick = async () => {
      await api("/api/canvas/thumb", { job: t.job, index: t.index });
      state.currentThumb = i;
      renderThumbs();
      viewRefresh();
    };
    strip.appendChild(el);
  });
}

function clearThumbs() {
  state.thumbs = [];
  state.currentThumb = -1;
  renderThumbs();
}

function setGenerating(on) {
  state.generating = on;
  const btn = $("btn-generate");
  btn.textContent = on ? "Interrupt" : "Generate";
  btn.classList.toggle("danger", on);
  $("progress-wrap").hidden = !on;
  if (!on) {
    $("progress-fill").style.width = "0%";
    $("progress-label").textContent = "";
  }
}

async function generate() {
  if (state.generating) {
    await api("/api/interrupt", { job: state.job });
    setGenerating(false);
    return;
  }
  const body = {
    prompt: $("prompt").value,
    negative_prompt: $("negative").value,
    width: parseInt($("width").value, 10),
    height: parseInt($("height").value, 10),
    steps: parseInt($("steps").value, 10),
    cfg: parseFloat($("cfg").value),
    seed: parseInt($("seed").value, 10),
    batch_size: parseInt($("batch").value, 10),
    sampler: $("sampler").value || null,
    scheduler: $("scheduler").value || null,
    model_id: $("model").value || null,
    mode: $("mode").value,
    denoise: parseFloat($("denoise").value),
  };
  clearThumbs();
  setStatus("queued…");
  saveUiState(); // persist the settings actually about to be used
  try {
    const r = await api("/api/generate", body);
    state.job = r.job_id;
    setGenerating(true);
  } catch (e) {
    setStatus(`generate failed: ${e.message}`);
  }
}

async function selectModel(id) {
  setStatus(`loading ${id}…`);
  await api("/api/models/select", { id });
  setStatus("");
}

// ---------------- toolbar: tools, brushes, mask ops, zoom ----------------
let brushNames = ["circle"];

async function loadBrushes() {
  try {
    brushNames = (await (await fetch("/api/brushes")).json()).brushes;
  } catch (e) {
    return;
  }
  const sel = $("brush");
  const cur = sel.value;
  sel.innerHTML = "";
  brushNames.forEach((n) => {
    const o = document.createElement("option");
    o.value = n;
    o.textContent = n === "circle" ? "circle (default)" : n;
    sel.appendChild(o);
  });
  if (brushNames.includes(cur)) sel.value = cur;
  syncToolbar();
}

// log-scale size: slider 0..1 maps to 0.1..100 px (x1000 across the range)
const sizeFromSlider = (v) => 0.1 * Math.pow(1000, v);
const sliderFromSize = (s) => Math.log(Math.max(s, 0.1) / 0.1) / Math.log(1000);

function syncToolbar() {
  const st = Tools.get();
  $("tool-brush").classList.toggle("active", Tools.current === "brush");
  $("tool-mask").classList.toggle("active", Tools.current === "mask");
  // The colour swatch always edits the brush tool's remembered colour
  // (mask strokes ignore colour), so it stays live in mask mode too.
  $("brush-color").value = Tools.state.brush.color || "#ffffff";
  if (brushNames.includes(st.brush)) $("brush").value = st.brush;
  $("brush-size").value = sliderFromSize(st.size);
  $("brush-size-val").textContent =
    st.size < 1 ? st.size.toFixed(2) : st.size.toFixed(1);
  $("brush-opacity").value = st.opacity;
  $("brush-opacity-val").textContent = st.opacity.toFixed(2);
}

function switchTool(t) {
  Tools.switch(t);
  syncToolbar();
}

async function maskOp(op) {
  try {
    await api("/api/canvas/mask", {
      op,
      px: parseInt($("mask-px").value, 10) || 0,
    });
    await viewRefresh();
  } catch (e) {
    setStatus(`mask ${op} failed: ${e.message}`);
  }
}

function zoomCenter(f) {
  const r = cv().getBoundingClientRect();
  viewZoomAt(f, r.width / 2, r.height / 2);
}

// denoising strength only applies to the inpaint modes (concept.md: "disabled
// when not inpainting")
function syncMode() {
  $("denoise").disabled = $("mode").value === "generate";
}

function wireToolbar() {
  $("tool-brush").onclick = () => switchTool("brush");
  $("tool-mask").onclick = () => switchTool("mask");
  $("brush").onchange = (e) => {
    Tools.get().brush = e.target.value;
    Tools.save();
  };
  $("brush-color").oninput = (e) => {
    Tools.state.brush.color = e.target.value;
    Tools.save();
  };
  $("brush-size").oninput = (e) => {
    // round to 0.1 px so the readout and the server request agree
    Tools.get().size = Math.round(sizeFromSlider(parseFloat(e.target.value)) * 10) / 10;
    Tools.save();
    syncToolbar();
  };
  $("brush-opacity").oninput = (e) => {
    Tools.get().opacity = parseFloat(e.target.value);
    Tools.save();
    syncToolbar();
  };
  $("mode").onchange = () => {
    syncMode();
    scheduleSaveUiState(0);
  };
  $("mask-dilate").onclick = () => maskOp("dilate");
  $("mask-erode").onclick = () => maskOp("erode");
  $("mask-clear").onclick = () => maskOp("clear");
  $("zoom-in").onclick = () => zoomCenter(1.25);
  $("zoom-out").onclick = () => zoomCenter(0.8);
  $("zoom-fit").onclick = () => viewFit();
}

// ---------------- UI state persistence (ui-state.json) ----------------
function readUiState() {
  return {
    model: $("model").value || null,
    prompt: $("prompt").value,
    negative_prompt: $("negative").value,
    width: parseInt($("width").value, 10),
    height: parseInt($("height").value, 10),
    steps: parseInt($("steps").value, 10),
    cfg: parseFloat($("cfg").value),
    seed: parseInt($("seed").value, 10),
    batch: parseInt($("batch").value, 10),
    sampler: $("sampler").value || null,
    scheduler: $("scheduler").value || null,
    mode: $("mode").value,
    denoise: parseFloat($("denoise").value),
  };
}

function applyUiState(st) {
  if (!st) return;
  const fields = [
    ["prompt", "prompt"],
    ["negative", "negative_prompt"],
    ["width", "width"],
    ["height", "height"],
    ["steps", "steps"],
    ["cfg", "cfg"],
    ["seed", "seed"],
    ["batch", "batch"],
    ["sampler", "sampler"],
    ["scheduler", "scheduler"],
    ["mode", "mode"],
    ["denoise", "denoise"],
  ];
  for (const [el, key] of fields) {
    if (st[key] !== undefined && st[key] !== null) $(el).value = st[key];
  }
  // display-only restore: the model itself is NOT preloaded (it loads
  // lazily on the next generation, matching the registry design)
  if (st.model) $("model").value = st.model;
}

let saveTimer = null;
function saveUiState() {
  return api("/api/ui-state", readUiState()).catch(() => {});
}
function scheduleSaveUiState(delay = 500) {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveUiState, delay);
}

// populate the form from the server, then apply the persisted state
async function init() {
  let models = { models: [], current: null };
  let samplers = { samplers: [], schedulers: [] };
  let saved = null;
  try {
    const [m, s, u] = await Promise.all([
      fetch("/api/models").then((r) => r.json()),
      fetch("/api/samplers").then((r) => r.json()),
      fetch("/api/ui-state").then((r) => (r.ok ? r.json() : null)),
    ]);
    models = m;
    samplers = s;
    saved = u && u.state;
  } catch (e) {
    console.error("init:", e);
    return;
  }
  const sel = $("model");
  sel.innerHTML = "";
  models.models.forEach((x) => {
    const o = document.createElement("option");
    o.value = x.id;
    o.textContent = x.name;
    if (x.id === models.current) o.selected = true;
    sel.appendChild(o);
  });
  for (const [id, names] of [
    ["sampler", samplers.samplers],
    ["scheduler", samplers.schedulers],
  ]) {
    const s2 = $(id);
    s2.innerHTML = "";
    names.forEach((n) => {
      const o = document.createElement("option");
      o.value = n;
      o.textContent = n;
      s2.appendChild(o);
    });
  }
  applyUiState(saved);
  syncMode();
}

// ---------------- WebSocket ----------------
let ws = null;
function connectWS() {
  ws = new WebSocket(`ws://${location.host}/api/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    switch (msg.type) {
      case "status":
        setStatus(msg.message);
        break;
      case "progress":
        $("progress-fill").style.width = `${Math.round(msg.fraction * 100)}%`;
        $("progress-label").textContent = msg.message;
        break;
      case "image":
        state.thumbs.push({ job: msg.job, index: msg.index, url: msg.url, seed: msg.seed });
        if (state.currentThumb === -1) state.currentThumb = state.thumbs.length - 1;
        renderThumbs();
        break;
      case "canvas_set":
        viewRefresh();
        break;
      case "done":
        setGenerating(false);
        setStatus("");
        break;
      case "job_interrupted":
        setGenerating(false);
        setStatus("interrupted (partial images discarded)");
        break;
      case "error":
        setStatus(`error: ${msg.message}`);
        break;
    }
  };
  ws.onclose = () => setTimeout(connectWS, 1000);
}

// ---------------- events ----------------
$("btn-generate").onclick = generate;
$("btn-save").onclick = async () => {
  try {
    const r = await api("/api/save", { prompt: $("prompt").value });
    setStatus(`saved ${r.path}`);
  } catch (e) {
    setStatus(`save failed: ${e.message}`);
  }
};
$("btn-undo").onclick = async () => {
  await api("/api/canvas/undo");
  viewRefresh();
};
$("btn-redo").onclick = async () => {
  await api("/api/canvas/redo");
  viewRefresh();
};
$("model").onchange = (e) => {
  selectModel(e.target.value);
  scheduleSaveUiState(0);
};

// left/right keys switch between thumbnail variants (when not typing)
document.addEventListener("keydown", (e) => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "");
  if (typing || state.thumbs.length === 0) return;
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  e.preventDefault();
  const d = e.key === "ArrowRight" ? 1 : -1;
  const next = (state.currentThumb + d + state.thumbs.length) % state.thumbs.length;
  const t = state.thumbs[next];
  api("/api/canvas/thumb", { job: t.job, index: t.index }).then(() => {
    state.currentThumb = next;
    renderThumbs();
    viewRefresh();
  });
});

// B / M switch tools (when not typing in a field)
document.addEventListener("keydown", (e) => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "");
  if (typing) return;
  if (e.key === "b" || e.key === "B") switchTool("brush");
  else if (e.key === "m" || e.key === "M") switchTool("mask");
});

// ---------------- boot ----------------
Tools.load();
syncToolbar();
wireToolbar();
loadBrushes();
init();
viewResize();
viewRefresh();
setupCanvasInput();
window.addEventListener("resize", viewResize);
connectWS();

["prompt", "negative", "width", "height", "steps", "cfg", "seed", "batch", "denoise"].forEach((id) => {
  $(id).addEventListener("input", () => scheduleSaveUiState());
});
["sampler", "scheduler"].forEach((id) => {
  $(id).onchange = () => scheduleSaveUiState(0);
});
// last-ditch save on tab close (sendBeacon is fire-and-forget, always
// delivered; the endpoint ignores content-type to accept its text/plain)
window.addEventListener("beforeunload", () => {
  try {
    navigator.sendBeacon(
      "/api/ui-state",
      new Blob([JSON.stringify(readUiState())], { type: "text/plain" })
    );
  } catch (e) {}
});
