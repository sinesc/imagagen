/* imagagen UI: controls, thumbnails, WS events, toolbar wiring, and the
   Phase 4 configuration layer (model configs, LoRAs, presets).
   Canvas rendering/input live in canvas.js, tool state in tools.js. */
"use strict";

const $ = (id) => document.getElementById(id);

const state = {
  generating: false,
  job: null,
  thumbs: [], // {job, index, url, seed}
  currentThumb: -1,
  entries: [], // model configurations from /api/models
  samplers: [],
  schedulers: [],
  loras: [], // lora names from /api/loras
  presets: [], // preset names
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
  if (!res.ok) throw await apiError(res, path);
  return res.json();
}

// surface the server's error message (e.g. "mask is empty") instead of
// a bare status code when one was sent
async function apiError(res, path) {
  let msg = `${path}: ${res.status}`;
  try {
    const j = await res.json();
    if (j && j.error) msg = j.error;
    else if (j && Array.isArray(j.detail)) {
      // FastAPI validation errors: ["loc", "msg"] pairs
      msg = j.detail.map((d) => `${(d.loc || []).join(".")}: ${d.msg}`).join("; ");
    } else if (j && j.detail) msg = String(j.detail);
  } catch (e) { /* non-JSON error body */ }
  return new Error(msg);
}

// ---------------- small DOM helpers ----------------
function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (k === "class") e.className = v;
    else if (k.startsWith("on") && typeof v === "function") e[k] = v;
    else e.setAttribute(k, v);
  }
  for (const c of children) if (c !== undefined && c !== null) e.append(c);
  return e;
}
const opt = (value, text) => {
  const o = el("option", { value }, text);
  return o;
};

// ---------------- generic dialog ----------------
let dialogOk = null;
function openDialog(title, body, onOk, okText = "Save") {
  $("dialog-title").textContent = title;
  const b = $("dialog-body");
  b.innerHTML = "";
  b.append(body);
  dialogOk = onOk;
  $("dialog-ok").textContent = okText;
  $("dialog-ok").onclick = async () => {
    if (!dialogOk) { closeDialog(); return; }
    const r = await dialogOk(); // returning false keeps the dialog open
    if (r !== false) closeDialog();
  };
  $("dialog").hidden = false;
}
function closeDialog() {
  $("dialog").hidden = true;
  dialogOk = null;
}
$("dialog-cancel").onclick = closeDialog;
$("dialog").onclick = (e) => { if (e.target === $("dialog")) closeDialog(); };
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("dialog").hidden) closeDialog();
});
function dialogError(text) {
  $("dialog-body").append(el("div", { class: "dialog-error" }, text));
}

// ---------------- LoRA list rows (form + dialogs) ----------------
function loraRow(name, weight, onChange) {
  const sel = el("select", { class: "lora-name" });
  state.loras.forEach((n) => sel.append(opt(n, n)));
  if (name && !state.loras.includes(name)) sel.append(opt(name, name));
  sel.value = name || "";
  const wt = el("input", {
    class: "lora-weight", type: "number", min: "-10", max: "10", step: "0.05",
  });
  wt.value = weight;
  const rm = el("button", { type: "button", class: "mini lora-rm", title: "remove" }, "−");
  rm.onclick = () => { rm.closest(".lora-row").remove(); onChange && onChange(); };
  sel.onchange = () => onChange && onChange();
  wt.oninput = () => onChange && onChange();
  return el("div", { class: "lora-row" }, sel, wt, rm);
}
function loraAddButton(listEl, onChange) {
  const add = el("button", { type: "button", class: "mini", title: "add a LoRA" }, "+ LoRA");
  add.onclick = () => {
    const row = loraRow("", 1, onChange);
    listEl.append(row);
    row.querySelector("select").focus();
    onChange && onChange();
  };
  return add;
}
function readLoraList(listEl) {
  return [...listEl.querySelectorAll(".lora-row")].map((r) => [
    r.querySelector(".lora-name").value,
    parseFloat(r.querySelector(".lora-weight").value) || 0,
  ]);
}
// the main form's LoRA list
function renderMainLoras(items) {
  const c = $("loras");
  c.innerHTML = "";
  (items || []).forEach(([n, w]) => c.append(loraRow(n, w, scheduleSaveUiState)));
}
const getLoras = () => readLoraList($("loras")).filter((x) => x[0]);

// ---------------- thumbnails ----------------
// put a thumbnail's image onto the canvas. Results come from the job;
// the inpaint source (green border) is not a job result, so its PNG is
// fetched and re-uploaded.
async function restoreThumb(t) {
  if (t.source) {
    const res = await fetch(t.url);
    if (!res.ok) throw new Error(`source: ${res.status}`);
    const data = await res.arrayBuffer();
    const up = await fetch("/api/canvas/upload", { method: "POST", body: data });
    if (!up.ok) throw new Error(`upload: ${up.status}`);
    return;
  }
  await api("/api/canvas/thumb", { job: t.job, index: t.index });
}

function renderThumbs() {
  const strip = $("thumbs");
  strip.hidden = state.thumbs.length === 0;
  strip.innerHTML = "";
  state.thumbs.forEach((t, i) => {
    const image = document.createElement("img");
    image.src = t.url;
    image.title = t.source
      ? "inpaint source (click to restore)"
      : `#${i} seed ${t.seed}`;
    if (t.source) image.classList.add("source");
    if (i === state.currentThumb) image.classList.add("selected");
    image.onclick = async () => {
      try {
        await restoreThumb(t);
      } catch (e) {
        setStatus(e.message);
        return;
      }
      state.currentThumb = i;
      renderThumbs();
      viewRefresh();
    };
    strip.appendChild(image);
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
  statusIsError = false;
  const body = {
    prompt: $("prompt").value,
    negative_prompt: $("negative").value,
    width: parseInt($("width").value, 10),
    height: parseInt($("height").value, 10),
    steps: parseInt($("steps").value, 10),
    cfg: parseFloat($("cfg").value), // -1 = "use model default"
    seed: parseInt($("seed").value, 10),
    batch_size: parseInt($("batch").value, 10),
    sampler: $("sampler").value || null, // null = "use model default"
    scheduler: $("scheduler").value || null,
    loras: getLoras(),
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
    if (finishedJobs.has(r.job_id)) {
      // The job already settled (fast error, interrupt, ...) before this
      // POST response arrived: the WS handler processed done/interrupt
      // first, so re-arming the button would stick it on "Interrupt".
      setGenerating(false);
    } else {
      setGenerating(true);
    }
  } catch (e) {
    setStatus(`generate failed: ${e.message}`);
  }
}

// ---------------- model configurations (Phase 4) ----------------
function currentEntry() {
  return state.entries.find((e) => e.id === $("model").value) || null;
}

async function selectModel(id) {
  setStatus(`loading ${id}…`);
  try {
    await api("/api/models/select", { id });
    setStatus("");
  } catch (e) {
    setStatus(`model select failed: ${e.message}`);
  }
}

async function refreshModels(keepSelection = true) {
  const m = await (await fetch("/api/models")).json();
  if (m.error) throw new Error(m.error);
  const sel = $("model");
  const cur = keepSelection ? sel.value : null;
  state.entries = m.models;
  sel.innerHTML = "";
  m.models.forEach((x) => {
    sel.append(opt(x.id, x.is_default ? `${x.name} (default)` : x.name));
  });
  if (cur && m.models.some((x) => x.id === cur)) sel.value = cur;
  else if (m.current) sel.value = m.current;
}

function modeFieldset(title, section) {
  // section: {sampler, scheduler, cfg, loras: [{name, weight}]}
  const fs = el("fieldset", { class: "cfg-mode" }, el("legend", {}, title));
  const s = el("select", { class: "cfg-sampler" });
  s.append(opt("", "Use model default"));
  state.samplers.forEach((n) => s.append(opt(n, n)));
  s.value = section.sampler || "";
  const sc = el("select", { class: "cfg-scheduler" });
  sc.append(opt("", "Use model default"));
  state.schedulers.forEach((n) => sc.append(opt(n, n)));
  sc.value = section.scheduler || "";
  const steps = el("input", {
    class: "cfg-steps", type: "number", min: "1", step: "1",
    title: "empty = use the built-in default",
  });
  steps.value = section.steps === null ? "" : section.steps;
  const cfg = el("input", {
    class: "cfg-cfg", type: "number", min: "0", step: "0.1",
    title: "empty = use the model's built-in default",
  });
  cfg.value = section.cfg === null ? "" : section.cfg;
  const loraList = el("div", { class: "lora-list" });
  (section.loras || []).forEach((l) => loraList.append(loraRow(l.name, l.weight)));
  fs.append(
    el("div", { class: "row" },
      el("label", {}, "sampler", s),
      el("label", {}, "scheduler", sc)),
    el("div", { class: "row" },
      el("label", { title: "steps override; empty = built-in default" }, "steps", steps),
      el("label", { title: "CFG override; empty = model default" }, "CFG", cfg)),
    el("div", { class: "loras" }, loraList, loraAddButton(loraList)),
  );
  return fs;
}

function readCfgSection(fs) {
  return {
    sampler: fs.querySelector(".cfg-sampler").value || null,
    scheduler: fs.querySelector(".cfg-scheduler").value || null,
    steps: fs.querySelector(".cfg-steps").value === "" ? null : parseInt(fs.querySelector(".cfg-steps").value, 10),
    cfg: fs.querySelector(".cfg-cfg").value === "" ? null : parseFloat(fs.querySelector(".cfg-cfg").value),
    loras: readLoraList(fs.querySelector(".lora-list")).map(([name, weight]) => ({ name, weight })),
  };
}

function openConfigDialog(entry) {
  const nameIn = el("input", { class: "cfg-name", type: "text", maxlength: "120" });
  nameIn.value = entry.name;
  const modelIn = el("input", { class: "cfg-model", type: "text", readonly: "" });
  modelIn.value = entry.model_id;
  const f1 = modeFieldset("text to image", entry.t2i);
  const f2 = modeFieldset("image to image / inpaint", entry.i2i);
  const box = el("div", { class: "cfg-dialog" },
    el("label", {}, "name", nameIn),
    el("label", { class: "muted" }, "model", modelIn),
    f1, f2);
  openDialog("Model configuration", box, async () => {
    const name = nameIn.value.trim();
    if (!name) { dialogError("name is required"); return false; }
    try {
      // pass the entry's id: saving the in-memory *default* of a model
      // materializes it into a file configuration (it gets a new id based
      // on the name, so follow it below)
      const r = await api("/api/model-configs", {
        id: entry.id,
        name,
        model: entry.model_id,
        t2i: readCfgSection(f1),
        i2i: readCfgSection(f2),
      });
      $("model").value = r.id;
    } catch (e) {
      dialogError(e.message);
      return false;
    }
    try {
      await refreshModels(true);
    } catch (e) {
      setStatus(e.message);
    }
    scheduleSaveUiState(0);
  });
}

function createConfig() {
  const cur = currentEntry() || state.entries[0];
  if (!cur) return;
  const nameIn = el("input", { type: "text", maxlength: "120", class: "cfg-name" });
  nameIn.value = `${cur.name} (variant)`;
  openDialog("New model configuration", el("div", {},
    el("div", { class: "muted" }, "starts as a copy of the selected configuration"),
    el("label", {}, "name", nameIn)),
    async () => {
      const name = nameIn.value.trim();
      if (!name) { dialogError("name is required"); return false; }
      try {
        const entry = await api("/api/model-configs/create", { name, based_on: cur.id });
        $("model").value = entry.id;
        await refreshModels(true);
        scheduleSaveUiState(0);
        return true;
      } catch (e) {
        dialogError(e.message);
        return false;
      }
    },
    "Create");
}

async function deleteConfig() {
  const cur = currentEntry();
  if (!cur) return;
  if (cur.is_default) {
    setStatus("in-memory default configurations cannot be deleted");
    return;
  }
  if (!confirm(`Delete model configuration "${cur.name}"?`)) return;
  try {
    await api("/api/model-configs/delete", { id: cur.id });
    await refreshModels(false);
    scheduleSaveUiState(0);
  } catch (e) {
    setStatus(`delete config failed: ${e.message}`);
  }
}

// ---------------- presets (Phase 4) ----------------
const PRESET_KEYS = ["width", "height", "steps", "cfg", "seed", "batch",
  "sampler", "scheduler", "mode", "denoise", "loras"];

async function applyPreset(name) {
  if (!name) return false;
  let p;
  try {
    const res = await fetch(`/api/presets/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`preset: ${res.status}`);
    p = (await res.json()).preset;
  } catch (e) {
    setStatus(e.message);
    return false;
  }
  const controls = p.controls || {};
  for (const key of PRESET_KEYS) {
    if (!controls[key] || p[key] === null || p[key] === undefined) continue;
    // "" resets sampler/scheduler to "use model default" (the main UI's
    // empty option); mode has no empty state
    if (p[key] === "" && key !== "sampler" && key !== "scheduler") continue;
    if (key === "loras") renderMainLoras(p[key]);
    else if (key === "mode") {
      $("mode").value = p[key];
      syncMode();
    } else {
      $(key).value = p[key];
    }
  }
  scheduleSaveUiState(0);
  setStatus(`preset "${name}" applied`);
  return true;
}

async function loadPresets() {
  const r = await (await fetch("/api/presets")).json();
  state.presets = r.presets || [];
  const sel = $("preset");
  const cur = sel.value;
  sel.innerHTML = "";
  sel.append(opt("", "—"));
  state.presets.forEach((n) => sel.append(opt(n, n)));
  if (state.presets.includes(cur)) sel.value = cur;
}

function presetValueInput(kind, value, emptyLabel = "—") {
  if (kind === "loras") {
    const input = el("div", { class: "lora-list preset-loras" });
    if (value) value.forEach((x) => input.append(loraRow(x[0], x[1])));
    input.append(loraAddButton(input));
    return input;
  }
  if (kind === "select") {
    const input = el("select", { class: "preset-val" });
    // sampler/scheduler: "" = reset the form to "use model default" (same
    // option as the main UI); mode has no such state, so plain "—"
    input.append(opt("", emptyLabel));
    return input;
  }
  const input = el("input", { class: "preset-val", type: "number" });
  if (value !== null && value !== undefined) input.value = value;
  return input;
}

function openPresetDialog(existing) {
  const p = existing || {};
  const controls = p.controls || {};
  const nameIn = el("input", { type: "text", maxlength: "120" });
  nameIn.value = p.name || "";
  const defs = [
    ["width", "width", "number", p.width],
    ["height", "height", "number", p.height],
    ["steps", "steps", "number", p.steps],
    ["cfg", "CFG", "number", p.cfg],
    ["seed", "seed", "number", p.seed],
    ["batch", "batch size", "number", p.batch],
    ["sampler", "sampler", "select", p.sampler],
    ["scheduler", "scheduler", "select", p.scheduler],
    ["mode", "mode", "select", p.mode],
    ["denoise", "denoise strength", "number", p.denoise],
    ["loras", "LORAs", "loras", p.loras],
  ];
  const rows = el("div", { class: "preset-rows" });
  for (const [key, label, kind, value] of defs) {
    const cb = el("input", { type: "checkbox", class: "preset-cb" });
    cb.checked = !!controls[key];
    let input;
    if (kind === "number" || kind === "loras") {
      input = presetValueInput(kind, value);
    } else {
      // sampler/scheduler: "" means "use model default" (applied on select);
      // mode has no such state
      input = presetValueInput(kind, value, key === "mode" ? "—" : "Use model default");
      if (key === "sampler") state.samplers.forEach((n) => input.append(opt(n, n)));
      if (key === "scheduler") state.schedulers.forEach((n) => input.append(opt(n, n)));
      if (key === "mode") [
        ["generate", "generate"],
        ["inpaint_masked", "inpaint masked"],
        ["inpaint_not_masked", "inpaint not masked"],
      ].forEach(([v, t]) => input.append(opt(v, t)));
      input.value = value || "";
    }
    input.dataset.presetKey = key;
    input.dataset.presetKind = kind;
    rows.append(el("div", { class: "preset-row" }, cb, el("span", { class: "preset-label" }, label), input));
  }
  const box = el("div", { class: "preset-dialog" },
    el("label", {}, "name", nameIn),
    el("div", { class: "muted" }, "checked settings are applied when the preset is selected"),
    rows);
  openDialog(existing ? "Edit preset" : "New preset", box, async () => {
    const name = nameIn.value.trim();
    if (!name) { dialogError("name is required"); return false; }
    const body = { name, controls: {} };
    for (const row of rows.children) {
      const input = row.querySelector("[data-preset-key]");
      const key = input.dataset.presetKey;
      body.controls[key] = row.querySelector(".preset-cb").checked;
      if (input.dataset.presetKind === "loras") body[key] = readLoraList(input);
      else if (input.dataset.presetKind === "select") {
        // sampler/scheduler: "" is a real value ("use model default", the
        // main UI's empty option); mode has no empty state, so "" -> null
        body[key] = input.value === ""
          ? (key === "sampler" || key === "scheduler" ? "" : null)
          : input.value;
      }
      else body[key] = input.value === "" ? null : parseFloat(input.value);
    }
    try {
      await api("/api/presets", body);
    } catch (e) {
      dialogError(e.message);
      return false;
    }
    await loadPresets();
    scheduleSaveUiState(0);
  });
}

function openPresetManage() {
  const list = el("div", { class: "preset-manage" });
  const newBtn = el("button", { class: "mini", type: "button" }, "+ New preset");
  const render = () => {
    list.querySelectorAll(".preset-manage-row").forEach((r) => r.remove());
    if (!state.presets.length) list.append(el("div", { class: "muted" }, "no presets yet"));
    state.presets.forEach((name) => {
      const nameEl = el("span", { class: "preset-manage-name" }, name);
      const edit = el("button", { class: "mini", type: "button", title: "edit" }, "✎");
      const del = el("button", { class: "mini", type: "button", title: "delete" }, "✗");
      edit.onclick = async () => {
        closeDialog();
        let p;
        try {
          p = (await (await fetch(`/api/presets/${encodeURIComponent(name)}`)).json()).preset;
        } catch (e) { setStatus(e.message); return; }
        openPresetDialog(p);
      };
      del.onclick = async () => {
        if (!confirm(`Delete preset "${name}"?`)) return;
        try {
          await api("/api/presets/delete", { name });
        } catch (e) { setStatus(e.message); return; }
        await loadPresets();
        render();
      };
      list.append(el("div", { class: "preset-manage-row" }, nameEl, edit, del));
    });
    list.append(newBtn);
  };
  render();
  newBtn.onclick = () => { closeDialog(); openPresetDialog(null); };
  openDialog("Setting presets", el("div", { class: "preset-manage-wrap" }, list), null, "Close");
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

// Brush size bounds; keep BRUSH_SIZE_MAX in sync with MAX_BRUSH_SIZE in app.py
const BRUSH_SIZE_MIN = 1;
const BRUSH_SIZE_MAX = 300;
// log-scale size: slider 0..1 maps to BRUSH_SIZE_MIN..BRUSH_SIZE_MAX px
const sizeFromSlider = (v) => BRUSH_SIZE_MIN * Math.pow(BRUSH_SIZE_MAX / BRUSH_SIZE_MIN, v);
const sliderFromSize = (s) => Math.log(Math.max(s, BRUSH_SIZE_MIN) / BRUSH_SIZE_MIN) / Math.log(BRUSH_SIZE_MAX / BRUSH_SIZE_MIN);

function syncToolbar() {
  const st = Tools.get();
  $("tool-brush").classList.toggle("active", Tools.current === "brush");
  $("tool-mask").classList.toggle("active", Tools.current === "mask");
  // The colour swatch always edits the brush tool's remembered colour
  // (mask strokes ignore colour), so it stays live in mask mode too.
  $("brush-color").value = Tools.state.brush.color || "#ffffff";
  if (brushNames.includes(st.brush)) $("brush").value = st.brush;
  $("brush-size").value = sliderFromSize(st.size);
  $("brush-size-val").textContent = Math.round(st.size);
  $("brush-screen").checked = !!st.screen;
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
    // round to whole px: size is an absolute pixel size
    Tools.get().size = Math.round(sizeFromSlider(parseFloat(e.target.value)));
    Tools.save();
    syncToolbar();
  };
  $("brush-screen").onchange = (e) => {
    Tools.get().screen = e.target.checked;
    Tools.save();
    if (view.has) viewRedraw(); // cursor outline changes footprint
  };
  $("brush-opacity").oninput = (e) => {
    Tools.get().opacity = parseFloat(e.target.value);
    Tools.save();
    syncToolbar();
  };
  $("mode").onchange = () => syncMode(); // mode is not persisted
  $("mask-dilate").onclick = () => maskOp("dilate");
  $("mask-erode").onclick = () => maskOp("erode");
  $("mask-clear").onclick = () => maskOp("clear");
  $("zoom-in").onclick = () => zoomCenter(1.25);
  $("zoom-out").onclick = () => zoomCenter(0.8);
  $("zoom-fit").onclick = () => viewFit();
}

// Phase 4 wiring: model configs, LoRAs, presets
function wireConfig() {
  $("cfg-create").onclick = createConfig;
  $("cfg-edit").onclick = () => {
    const cur = currentEntry();
    if (cur) openConfigDialog(cur);
  };
  $("cfg-delete").onclick = deleteConfig;
  $("lora-add").onclick = () => {
    const row = loraRow("", 1, scheduleSaveUiState);
    $("loras").append(row);
    row.querySelector("select").focus();
  };
  $("preset-manage").onclick = openPresetManage;
  // apply on change, then reset to "—" so the same preset can be
  // re-selected (change only fires on a different selection)
  $("preset").onchange = (e) => {
    const sel = e.target;
    applyPreset(sel.value).then((ok) => { if (ok) sel.value = ""; });
  };
}

// ---------------- PNG drag&drop import (prompt / negative prompt) ----------------
// Dropping a PNG onto the prompt or negative-prompt field sets the image on
// the canvas and applies its embedded metadata to the controls. Without
// metadata, only the width/height come from the image dimensions.
async function importPng(file) {
  try {
    const metaRes = await fetch("/api/png-meta", { method: "POST", body: await file.arrayBuffer() });
    if (!metaRes.ok) {
      const j = await metaRes.json().catch(() => null);
      throw new Error(j?.error || `png-meta: ${metaRes.status}`);
    }
    const { meta, width, height } = await metaRes.json();
    const up = await fetch("/api/canvas/upload", { method: "POST", body: await file.arrayBuffer() });
    if (!up.ok) throw new Error(`canvas upload: ${up.status}`);
    applyPngMeta(meta, width, height);
    viewRefresh();
    setStatus(meta ? `imported ${file.name} (metadata applied)` : `loaded ${file.name} onto the canvas (no metadata)`);
  } catch (e) {
    setStatus(`png import failed: ${e.message}`);
  }
}

function applyPngMeta(meta, width, height) {
  if (!meta) {
    $("width").value = width;
    $("height").value = height;
    scheduleSaveUiState();
    return;
  }
  if (meta.prompt !== undefined && meta.prompt !== null) $("prompt").value = meta.prompt;
  if (meta.negative_prompt !== undefined && meta.negative_prompt !== null) $("negative").value = meta.negative_prompt;
  for (const id of ["steps", "cfg", "seed", "width", "height"]) {
    if (meta[id] !== undefined && meta[id] !== null) $(id).value = meta[id];
  }
  for (const id of ["sampler", "scheduler"]) {
    const sel = $(id);
    if (meta[id] && [...sel.options].some((o) => o.value === meta[id])) sel.value = meta[id];
  }
  if (meta.loras?.length) renderMainLoras(parseMetaLoras(meta.loras));
  // meta "Model" carries the model's display name; match it against the
  // loaded configurations (fall back to id for robustness)
  if (meta.model) {
    const entry = state.entries.find((e) => e.name === meta.model || e.id === meta.model);
    if (entry) {
      $("model").value = entry.id;
      selectModel(entry.id);
    }
  }
  scheduleSaveUiState();
}

// meta LoRAs travel as "name@weight" strings (A1111/Forge format)
function parseMetaLoras(items) {
  return (items || []).map((s) => {
    const m = String(s).match(/^(.*)@(-?\d+(?:\.\d+)?)$/);
    return m ? [m[1], parseFloat(m[2])] : [String(s), 1];
  });
}

// Dropping a PNG directly on the canvas just sets the image there --
// applying metadata is what dropping on the prompt fields is for.
function wireCanvasDrop() {
  const holder = $("canvas-holder");
  holder.addEventListener("dragover", (e) => {
    if (!(e.dataTransfer?.types || []).includes("Files")) return;
    e.preventDefault();
    holder.classList.add("drop");
  });
  holder.addEventListener("dragleave", () => holder.classList.remove("drop"));
  holder.addEventListener("drop", async (e) => {
    e.preventDefault();
    holder.classList.remove("drop");
    const files = [...(e.dataTransfer?.files || [])];
    const file = files.find((f) => f.type === "image/png" || /\.png$/i.test(f.name));
    if (!file) {
      setStatus("drop a PNG file");
      return;
    }
    try {
      const up = await fetch("/api/canvas/upload", { method: "POST", body: await file.arrayBuffer() });
      if (!up.ok) throw new Error(`canvas upload: ${up.status}`);
      viewRefresh();
      setStatus(`loaded ${file.name} onto the canvas`);
    } catch (err) {
      setStatus(`canvas drop failed: ${err.message}`);
    }
  });
}

function wirePromptDrop() {
  for (const id of ["prompt", "negative"]) {
    const t = $(id);
    t.addEventListener("dragover", (e) => {
      if (!(e.dataTransfer?.types || []).includes("Files")) return;
      e.preventDefault(); // needed to allow the drop
      t.classList.add("drop");
    });
    t.addEventListener("dragleave", () => t.classList.remove("drop"));
    t.addEventListener("drop", (e) => {
      e.preventDefault();
      t.classList.remove("drop");
      const files = [...(e.dataTransfer?.files || [])];
      const file = files.find((f) => f.type === "image/png" || /\.png$/i.test(f.name));
      if (!file) {
        setStatus("drop a PNG file");
        return;
      }
      importPng(file);
    });
  }
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
    // mode is intentionally not persisted: the UI always opens in
    // "generate" mode since there is nothing to inpaint yet
    denoise: parseFloat($("denoise").value),
    loras: getLoras(),
    // preset is not persisted either: the dropdown always starts at "—"
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
    ["denoise", "denoise"],
  ];
  for (const [elm, key] of fields) {
    if (st[key] !== undefined && st[key] !== null) $(elm).value = st[key];
  }
  renderMainLoras(st.loras || []);
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
  let m = { models: [], current: null };
  let s = { samplers: [], schedulers: [] };
  let u = null;
  let pr = { presets: [] };
  let lr = { loras: [] };
  try {
    [m, s, u, pr, lr] = await Promise.all([
      fetch("/api/models").then((r) => r.json()),
      fetch("/api/samplers").then((r) => r.json()),
      fetch("/api/ui-state").then((r) => (r.ok ? r.json() : null)),
      fetch("/api/presets").then((r) => r.json()),
      fetch("/api/loras").then((r) => r.json()),
    ]);
  } catch (e) {
    console.error("init:", e);
    return;
  }
  state.entries = m.models;
  state.samplers = s.samplers;
  state.schedulers = s.schedulers;
  state.presets = pr.presets;
  state.loras = lr.loras;
  const sel = $("model");
  sel.innerHTML = "";
  m.models.forEach((x) => {
    sel.append(opt(x.id, x.is_default ? `${x.name} (default)` : x.name));
  });
  if (m.current && m.models.some((x) => x.id === m.current)) sel.value = m.current;
  for (const id of ["sampler", "scheduler"]) {
    const s2 = $(id);
    s2.innerHTML = "";
    s2.append(opt("", "Use model default"));
    (id === "sampler" ? s.samplers : s.schedulers).forEach((n) => s2.append(opt(n, n)));
  }
  const pres = $("preset");
  pres.innerHTML = "";
  pres.append(opt("", "—"));
  state.presets.forEach((n) => pres.append(opt(n, n)));
  applyUiState(u && u.state);
  syncMode();
}

// ---------------- WebSocket ----------------
// Job ids whose terminal event (done / job_interrupted) already arrived.
// A job can settle before the /api/generate POST response reaches us
// (fast failures), so generate() checks this before flipping the button.
const finishedJobs = new Set();
// true while the status line shows an error: "done" must not wipe it
let statusIsError = false;
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
      case "source_image":
        // inpaint source snapshot, shown before any result arrives
        state.thumbs.unshift({ source: true, url: msg.url });
        renderThumbs();
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
        finishedJobs.add(msg.job);
        setGenerating(false);
        if (!statusIsError) setStatus("");
        break;
      case "job_interrupted":
        finishedJobs.add(msg.job);
        setGenerating(false);
        setStatus("interrupted (partial images discarded)");
        break;
      case "error":
        statusIsError = true;
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
  restoreThumb(t).then(() => {
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
wireConfig();
wirePromptDrop();
wireCanvasDrop();
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
