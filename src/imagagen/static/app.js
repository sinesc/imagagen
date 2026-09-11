/* imagagen MVP UI: controls + canvas + thumbnails + WS events */
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

async function api(path, body) {
  const res = await fetch(path, body === undefined ? undefined : {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json();
}

function refreshCanvas() {
  const img = $("canvas");
  img.onload = () => {
    img.hidden = false;
    $("canvas-empty").hidden = true;
  };
  fetch("/api/canvas.png")
    .then((r) => {
      if (!r.ok) throw new Error("empty");
      return r.blob();
    })
    .then((b) => {
      img.src = URL.createObjectURL(b);
    })
    .catch(() => {
      img.hidden = true;
      img.removeAttribute("src");
      $("canvas-empty").hidden = false;
    });
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
      refreshCanvas();
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
  };
  clearThumbs();
  setStatus("queued…");
  saveUiState(); // persist the settings actually about to be used
  const r = await api("/api/generate", body);
  state.job = r.job_id;
  setGenerating(true);
}

async function selectModel(id) {
  setStatus(`loading ${id}…`);
  await api("/api/models/select", { id });
  setStatus("");
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
        refreshCanvas();
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
  refreshCanvas();
};
$("btn-redo").onclick = async () => {
  await api("/api/canvas/redo");
  refreshCanvas();
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
    refreshCanvas();
  });
});

// ---------------- boot ----------------
init();
refreshCanvas();
connectWS();

["prompt", "negative", "width", "height", "steps", "cfg", "seed", "batch"].forEach((id) => {
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
