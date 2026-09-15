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
  styles: [], // styles from /api/styles ({name, prompt, negative_prompt, optgroup, temporary})
  segModels: [], // segmentation models from /api/segmentation/models
  segClasses: [], // last-used person-mask class selection ([] = default)
  // soft inpainting (Forge builtin extension) options; the enabled flag
  // lives on the checkbox. Defaults = Forge's built-in defaults.
  soft: { power: 1, scale: 0.5, detail: 4, maskInf: 0, threshold: 0.5, contrast: 2 },
  // self attention guidance (Forge builtin extension) options; the
  // enabled flag lives on the checkbox. Defaults = Forge's built-in
  // defaults.
  sag: { scale: 0.5, blur: 2, threshold: 1 },
};

function setStatus(text) {
  $("status-line").textContent = text || "";
}

// POST-only API helper. Every call is a POST; body-less endpoints simply
// ignore the empty {} JSON body that is always sent.
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

// ---------------- prompt token counter ----------------
// Live CLIP token count for the prompt field, shown in the caption row. The
// count comes from the server (same BPE vocabulary the SDXL backend uses), so
// it matches what actually gets encoded. Debounced; responses are sequenced so
// a slow early request can never overwrite a newer one.
let tokenTimer = null;
let tokenSeq = 0;
async function updatePromptTokens() {
  const seq = ++tokenSeq;
  const out = $("prompt-tokens");
  let r;
  try {
    // styles are applied server-side (exactly as generation does), so the
    // count reflects the full prompt the model will actually see
    r = await api("/api/prompt-tokens", {
      prompt: $("prompt").value,
      styles: getStyles(),
    });
  } catch (e) {
    return;
  }
  if (seq !== tokenSeq) return; // a newer request superseded this one
  if (r.tokens == null) { out.textContent = ""; out.classList.remove("over"); return; }
  out.textContent = String(r.tokens);
  out.classList.toggle("over", r.windows > 1); // >75 tokens -> split across windows
}
function schedulePromptTokens() {
  clearTimeout(tokenTimer);
  tokenTimer = setTimeout(updatePromptTokens, 200);
}
// selected styles feed the (server-side applied) prompt, so any change to the
// style list must refresh the counter as well as the persisted UI state
function onStylesChanged() {
  scheduleSaveUiState();
  schedulePromptTokens();
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
  const rm = el("button", { type: "button", class: "mini icon lora-rm", title: "remove" }, "−");
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

// ---------------- style list rows (styles.csv) ----------------
// One dropdown per selected style (grouped by optgroup, everything sorted
// alphabetically) + a "-" button, mirroring the LoRA rows. "Session" styles
// are server-side temporaries (imported from image metadata, not in the CSV).
function fillStyleOptions(sel, selected) {
  sel.innerHTML = "";
  const groups = new Map(); // optgroup ("" = top level) -> [names]
  for (const s of state.styles) {
    const g = s.optgroup || "";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(s.name);
  }
  for (const [g, names] of [...groups.entries()].sort((a, b) =>
    a[0].localeCompare(b[0])
  )) {
    names.sort((a, b) => a.localeCompare(b));
    if (!g) {
      names.forEach((n) => sel.append(opt(n, n)));
    } else {
      const og = el("optgroup", { label: g });
      names.forEach((n) => og.append(opt(n, n)));
      sel.append(og);
    }
  }
  sel.value = selected || "";
  if (selected && sel.value !== selected) {
    // selected but (temporarily) absent from the list: keep it selectable
    sel.append(opt(selected, `${selected} (temporary)`));
    sel.value = selected;
  }
}

function styleRow(name, onChange) {
  const sel = el("select", { class: "style-name" });
  fillStyleOptions(sel, name);
  const rm = el("button", { type: "button", class: "mini icon style-rm", title: "remove" }, "−");
  rm.onclick = () => { rm.closest(".style-row").remove(); onChange && onChange(); };
  sel.onchange = () => onChange && onChange();
  return el("div", { class: "style-row" }, sel, rm);
}

function renderMainStyles(names) {
  const c = $("styles");
  c.innerHTML = "";
  (names || []).forEach((n) => c.append(styleRow(n, onStylesChanged)));
}
const getStyles = () => [...$("styles").querySelectorAll(".style-row")]
  .map((r) => r.querySelector(".style-name").value)
  .filter(Boolean);

async function loadStyles() {
  const r = await (await fetch("/api/styles")).json();
  state.styles = r.styles || [];
  // re-render existing rows against the fresh list, keeping the selection
  renderMainStyles(getStyles());
  schedulePromptTokens(); // a selected style's contents may have changed
}

function openStyleDialog(existing) {
  const s = existing || {};
  const nameIn = el("input", { type: "text", maxlength: "120" });
  nameIn.value = s.name || "";
  const groupIn = el("input", { type: "text", maxlength: "80" });
  groupIn.value = s.optgroup || "";
  const promptIn = el("textarea", { class: "style-prompt", rows: "4" });
  promptIn.value = s.prompt || "";
  const negIn = el("textarea", { class: "style-prompt", rows: "3" });
  negIn.value = s.negative_prompt || "";
  const box = el("div", { class: "style-dialog" },
    el("label", {}, "name", nameIn),
    el("label", { title: "optgroup: groups styles in the UI dropdown (\"Session\" is reserved for imported temporary styles)" }, "group", groupIn),
    el("label", { title: "{prompt} is replaced with the prompt so far at generation time; without it the text is appended to the prompt" }, "prompt", promptIn),
    el("label", {}, "negative prompt", negIn));
  openDialog(existing ? "Edit style" : "New style", box, async () => {
    const name = nameIn.value.trim();
    if (!name) { dialogError("name is required"); return false; }
    try {
      await api("/api/styles", {
        name,
        optgroup: groupIn.value.trim(),
        prompt: promptIn.value,
        negative_prompt: negIn.value,
      });
      await loadStyles();
      scheduleSaveUiState(0);
    } catch (e) {
      dialogError(e.message);
      return false;
    }
  });
}

function openStyleManage() {
  const list = el("div", { class: "preset-manage" });
  const newBtn = el("button", { class: "mini", type: "button" }, "+ New style");
  const render = () => {
    list.querySelectorAll(".preset-manage-row").forEach((r) => r.remove());
    if (!state.styles.length) list.append(el("div", { class: "muted" }, "no styles yet"));
    state.styles.forEach((s) => {
      const nameEl = el("span", { class: "preset-manage-name" },
        s.name + (s.optgroup ? ` — ${s.optgroup}` : "") + (s.temporary ? " (temporary)" : ""));
      const edit = el("button", { class: "mini icon", type: "button", title: s.temporary ? "copy to styles.csv" : "edit" }, "✎");
      const del = el("button", { class: "mini icon", type: "button", title: s.temporary ? "drop (temporary styles are not in styles.csv)" : "delete" }, "✗");
      edit.onclick = () => { closeDialog(); openStyleDialog(s); };
      del.onclick = async () => {
        if (!confirm(`Delete style "${s.name}"?`)) return;
        try {
          if (s.temporary) {
            // temporaries are in-memory only: drop the row, keep everything else
            renderMainStyles(getStyles().filter((n) => n !== s.name));
          } else {
            await api("/api/styles/delete", { name: s.name });
            await loadStyles();
          }
          scheduleSaveUiState(0);
          schedulePromptTokens();
        } catch (e) { setStatus(e.message); return; }
        render();
      };
      list.append(el("div", { class: "preset-manage-row" }, nameEl, edit, del));
    });
    list.append(newBtn);
  };
  render();
  newBtn.onclick = () => { closeDialog(); openStyleDialog(null); };
  openDialog("Style configuration", el("div", { class: "preset-manage-wrap" }, list), null, "Close");
}

// ---------------- thumbnails ----------------
// put a thumbnail's image onto the canvas. Results come from the job;
// the source image (green border, canvas snapshot at submit time --
// inpaint always, generate when the canvas was non-empty) is not a job
// result, so its PNG is fetched and re-uploaded.
// Both kinds of thumbnail (results and the green-bordered source snapshot)
// are live canvas image entities now, so selecting one is always a select
// by id; the response carries the four undo/redo flags.
async function restoreThumb(t) {
  const r = await api("/api/canvas/select", { id: t.image_id });
  storeUndoFlags(r);
}

// hover preview: the thumbnail's full image, zoom-to-fit, laid over the
// canvas box. A purely visual overlay -- it never touches the view
// (zoom/pan), the mask, or the undo stacks. The container is positioned
// by JS onto the <canvas> element's on-screen rect so it fits it exactly.
let thumbPreviewHideTimer = null;
function thumbPreviewShow(url) {
  if (!view.has) return; // no canvas to overlay on
  clearTimeout(thumbPreviewHideTimer);
  const pv = $("thumb-preview");
  $("thumb-preview-img").src = url;
  const holder = $("canvas-holder").getBoundingClientRect();
  const c = cv().getBoundingClientRect();
  pv.style.left = `${c.left - holder.left}px`;
  pv.style.top = `${c.top - holder.top}px`;
  pv.style.width = `${c.width}px`;
  pv.style.height = `${c.height}px`;
  pv.hidden = false;
  void pv.offsetWidth; // flush styles so the fade-in runs from opacity 0
  pv.classList.add("on");
}
function thumbPreviewHide() {
  const pv = $("thumb-preview");
  pv.classList.remove("on");
  clearTimeout(thumbPreviewHideTimer);
  // let the 0.25s fade-out finish, then drop the element
  thumbPreviewHideTimer = setTimeout(() => (pv.hidden = true), 250);
}

function renderThumbs() {
  const strip = $("thumbs");
  strip.hidden = state.thumbs.length === 0;
  strip.innerHTML = "";
  state.thumbs.forEach((t, i) => {
    const image = document.createElement("img");
    image.src = t.url;
    image.title = t.source
      ? "source image (click to restore)"
      : `#${i} seed ${t.seed}`;
    if (t.source) image.classList.add("source");
    if (i === state.currentThumb) image.classList.add("selected");
    image.onmouseenter = () => thumbPreviewShow(t.url);
    image.onmouseleave = () => thumbPreviewHide();
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
    styles: getStyles(),
    model_id: $("model").value || null,
    mode: $("mode").value,
    denoise: parseFloat($("denoise").value),
    mask_padding: parseInt($("mask-padding").value, 10) || 0,
    soft_inpainting: softInpaintBody(), // null = off
    sag: sagBody(), // null = off
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
  s.append(opt("", "Model default"));
  state.samplers.forEach((n) => s.append(opt(n, n)));
  s.value = section.sampler || "";
  const sc = el("select", { class: "cfg-scheduler" });
  sc.append(opt("", "Model default"));
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
  "sampler", "scheduler", "mode", "denoise", "mask_padding", "loras"];

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
    // "" resets sampler/scheduler to "Model default" (the main UI's
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

// state.presets: [{name, group|null}] (name-sorted, from /api/presets)
function fillPresetSelect(sel) {
  sel.innerHTML = "";
  sel.append(opt("", "—"));
  // ungrouped presets first as plain options, then one optgroup per group
  // (first-appearance order; within a group the name order is kept)
  state.presets.filter((p) => !p.group)
    .forEach((p) => sel.append(opt(p.name, p.name)));
  const groups = [...new Set(state.presets.map((p) => p.group).filter(Boolean))];
  for (const g of groups) {
    const og = el("optgroup", { label: g });
    state.presets.filter((p) => p.group === g)
      .forEach((p) => og.append(opt(p.name, p.name)));
    sel.append(og);
  }
}

async function loadPresets() {
  const r = await (await fetch("/api/presets")).json();
  state.presets = r.presets || [];
  const sel = $("preset");
  const cur = sel.value;
  fillPresetSelect(sel);
  if (state.presets.some((p) => p.name === cur)) sel.value = cur;
}

function presetValueInput(kind, value, emptyLabel = "—") {
  if (kind === "loras") {
    const list = el("div", { class: "lora-list preset-loras" });
    if (value) value.forEach((x) => list.append(loraRow(x[0], x[1])));
    // same structure as the sidebar/config dialog: rows first, the "+ LoRA"
    // button after the list (added rows append to the list, above the button)
    return el("div", { class: "loras" }, list, loraAddButton(list));
  }
  if (kind === "select") {
    const input = el("select", { class: "preset-val" });
    // sampler/scheduler: "" = reset the form to "Model default" (same
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
  const groupIn = el("input", { type: "text", maxlength: "120",
    placeholder: "optional — presets in the same group are listed together" });
  groupIn.value = p.group || "";
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
    ["mask_padding", "mask padding", "number", p.mask_padding],
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
      // sampler/scheduler: "" means "Model default" (applied on select);
      // mode has no such state
      input = presetValueInput(kind, value, key === "mode" ? "—" : "Model default");
      if (key === "sampler") state.samplers.forEach((n) => input.append(opt(n, n)));
      if (key === "scheduler") state.schedulers.forEach((n) => input.append(opt(n, n)));
      if (key === "mode") [
        ["generate", "Generate"],
        ["inpaint_masked", "Inpaint"],
        ["inpaint_not_masked", "Outpaint"],
      ].forEach(([v, t]) => input.append(opt(v, t)));
      input.value = value || "";
    }
    input.dataset.presetKey = key;
    input.dataset.presetKind = kind;
    rows.append(el("div", { class: "preset-row" }, cb, el("span", { class: "preset-label" }, label), input));
  }
  const box = el("div", { class: "preset-dialog" },
    el("label", {}, "name", nameIn),
    el("label", {}, "group", groupIn),
    el("div", { class: "muted" }, "checked settings are applied when the preset is selected"),
    rows);
  openDialog(existing ? "Edit preset" : "New preset", box, async () => {
    const name = nameIn.value.trim();
    if (!name) { dialogError("name is required"); return false; }
    const body = { name, group: groupIn.value.trim() || null, controls: {} };
    for (const row of rows.children) {
      const input = row.querySelector("[data-preset-key]");
      const key = input.dataset.presetKey;
      body.controls[key] = row.querySelector(".preset-cb").checked;
      if (input.dataset.presetKind === "loras") body[key] = readLoraList(input);
      else if (input.dataset.presetKind === "select") {
        // sampler/scheduler: "" is a real value ("Model default", the
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
    state.presets.forEach((p) => {
      const name = p.name;
      const nameEl = el("span", { class: "preset-manage-name" }, name);
      const edit = el("button", { class: "mini icon", type: "button", title: "edit" }, "✎");
      const del = el("button", { class: "mini icon", type: "button", title: "delete" }, "✗");
      edit.onclick = async () => {
        closeDialog();
        let preset;
        try {
          preset = (await (await fetch(`/api/presets/${encodeURIComponent(name)}`)).json()).preset;
        } catch (e) { setStatus(e.message); return; }
        openPresetDialog(preset);
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
  syncUndoButtons();
}

// ---------------- undo / redo (per-image draw + global mask) ----------------
// The server reports, on every canvas-mutating response, how far each of
// the two independent stacks can go: the per-image draw stack of the
// CURRENT image, and the image-independent global mask. One toolbar
// undo/redo pair follows the active tool: "mask" drives the mask stack, any
// other tool drives the draw stack.
let undoFlags = { undo_mask: false, redo_mask: false,
                  undo_draw: false, redo_draw: false };
function syncUndoButtons() {
  const t = Tools.current === "mask" ? "mask" : "draw";
  $("btn-undo").textContent = `↩ undo ${t}`;
  $("btn-redo").textContent = `↪ redo ${t}`;
  $("btn-undo").disabled = !undoFlags[`undo_${t}`];
  $("btn-redo").disabled = !undoFlags[`redo_${t}`];
}
// Store the four flags from a canvas-mutating response and re-sync the
// tool-dependent button state (called by every such handler, incl. the
// stroke commit in canvas.js).
function storeUndoFlags(r) {
  if (r && typeof r === "object") {
    for (const k of ["undo_mask", "redo_mask", "undo_draw", "redo_draw"]) {
      if (typeof r[k] === "boolean") undoFlags[k] = r[k];
    }
  }
  syncUndoButtons();
}

async function maskOp(op) {
  try {
    const r = await api("/api/canvas/mask", {
      op,
      px: parseInt($("mask-px").value, 10) || 0,
    });
    storeUndoFlags(r);
    await viewRefresh();
  } catch (e) {
    setStatus(`mask ${op} failed: ${e.message}`);
  }
}

// ---------------- person mask generation (Phase 5 segmentation) ----------------
// models/segmentation/*.tflite -> /api/segmentation/models; the class
// checkboxes come from the model's own labels (background excluded). The
// result replaces the mask buffer (one undo step).
function openSegDialog() {
  const model =
    state.segModels.find((m) => m.id === $("seg-model").value) || null;
  if (!model) {
    openDialog(
      "Generate person mask",
      el(
        "div",
        { class: "muted" },
        "no segmentation model selected — put a .tflite (e.g. " +
          "selfie_multiclass_256x256.tflite) into models/segmentation/ and reload"
      ),
      null,
      "Close"
    );
    return;
  }
  const defaults = model.labels.filter((l) => l !== "background");
  const prev = state.segClasses.length ? state.segClasses : defaults;
  const rows = defaults.map((label) => {
    const cb = el("input", { type: "checkbox" });
    cb.checked = prev.includes(label);
    return el("label", { class: "seg-class" }, cb, label);
  });
  const body = el("div", {},
    el("div", { class: "muted" },
      `classes to add to the mask (union with the current mask, undoable — clear it first to replace):`),
    el("div", { class: "seg-classes" }, ...rows),
  );
  openDialog("Generate person mask", body, async () => {
    const classes = rows.filter((r) => r.querySelector("input").checked)
      .map((r) => r.textContent.trim());
    if (!classes.length) return false; // keep the dialog open
    try {
      const r = await api("/api/segmentation/mask", { model: model.id, classes });
      storeUndoFlags(r);
      state.segClasses = classes;
      scheduleSaveUiState(0);
      setStatus(`mask generated: ${r.pixels.toLocaleString()} px (undo to revert)`);
      viewRefresh();
    } catch (e) {
      statusIsError = true;
      setStatus(`segmentation failed: ${e.message}`);
      return false; // keep the dialog open so the class set can be fixed
    }
  }, "Generate");
}

function zoomCenter(f) {
  const r = cv().getBoundingClientRect();
  viewZoomAt(f, r.width / 2, r.height / 2);
}

// denoising strength only applies to the inpaint modes (concept.md: "disabled
// when not inpainting"); mask padding only to "inpaint masked"; soft
// inpainting only makes sense with an inpaint region
function syncMode() {
  $("denoise").disabled = $("mode").value === "generate";
  $("mask-padding").disabled = $("mode").value !== "inpaint_masked";
  $("soft-inpaint").disabled = $("mode").value === "generate";
}

// ---------------- soft inpainting (Forge builtin extension) ----------------

// The six Forge sliders: [state key, label, min, max, step, help]. The
// first group is the per-step latent blend, the second the pixel composite.
const SOFT_LATENT_OPTS = [
  ["power", "Schedule bias", 0, 8, 0.1,
   "Shifts when the original content is preserved during denoising. 1 is balanced; below 1 the original content is preserved more strongly at the end of denoising (low noise), above 1 more strongly at the beginning (high noise)."],
  ["scale", "Preservation strength", 0, 8, 0.05,
   "Skews whether partially masked regions preserve the original content or favour the inpainted content. Low values favour the inpainted content, high values the original. 0 ignores the mask in the per-step blend entirely."],
  ["detail", "Transition contrast boost", 1, 32, 0.5,
   "How the original and denoised latent vectors are interpolated: their magnitudes are interpolated under this exponent, biased towards the larger of the two. Low values blend more softly (can lose contrast); high values keep contrast but can over-sharpen or over-saturate the transition."],
];
const SOFT_PIXEL_OPTS = [
  ["maskInf", "Mask influence", 0, 1, 0.05,
   "How strongly the original mask biases the pixel-composite difference threshold. 0 ignores the mask (only the image content decides where the original is kept), 1 follows it closely."],
  ["threshold", "Difference threshold", 0, 8, 0.25,
   "The latent-space distance at which the original pixels are kept at 50% opacity. Low: the generated content must be nearly identical for any original pixels to be kept. High: even clearly different content keeps the original pixels."],
  ["contrast", "Difference contrast", 0, 8, 0.25,
   "Contrast of the opacity transition between original and inpainted content. Low: gradual transitions (can ghost); high: abrupt transitions (no ghosting, but the seam can be visible)."],
];

function openSoftInpaintDialog() {
  const fields = [];
  const make = ([key, label, min, max, step, help]) => {
    const input = el("input", { type: "range", min, max, step });
    input.value = state.soft[key];
    const val = el("b", {}, String(state.soft[key]));
    input.oninput = () => { val.textContent = input.value; };
    fields.push({ key, input });
    return el("label", { class: "soft-opt" },
      el("div", { class: "soft-opt-head" }, el("span", {}, label), val),
      input,
      el("div", { class: "soft-opt-help" }, help),
    );
  };
  openDialog("Soft inpainting options", el("div", {},
    el("div", { class: "soft-lead" },
      "Soft inpainting (Forge builtin extension) blends the original content with the inpainted content according to the mask opacity instead of locking the protected region exactly. A high mask blur is recommended."),
    ...SOFT_LATENT_OPTS.map(make),
    el("div", { class: "soft-section" }, "Pixel composite settings"),
    ...SOFT_PIXEL_OPTS.map(make),
  ), () => {
    fields.forEach(({ key, input }) => { state.soft[key] = parseFloat(input.value); });
    scheduleSaveUiState(0);
  });
}

// The generate body's soft_inpainting field: null when off or in
// "generate" mode (the backend only applies it with an init mask anyway)
function softInpaintBody() {
  if (!$("soft-inpaint").checked || $("mode").value === "generate") return null;
  const s = state.soft;
  return {
    mask_blend_power: s.power,
    mask_blend_scale: s.scale,
    inpaint_detail_preservation: s.detail,
    composite_mask_influence: s.maskInf,
    composite_difference_threshold: s.threshold,
    composite_difference_contrast: s.contrast,
  };
}

// ---------------- self attention guidance (Forge builtin extension) ----------------

// The three Forge sliders: [state key, label, min, max, step, help].
const SAG_OPTS = [
  ["scale", "Scale", -2, 5, 0.01,
   "How strongly the guidance is applied: the difference between the degraded image and its re-denoised version is multiplied by this and added to the CFG result. 0 disables the effect; negative values work in reverse."],
  ["blur", "Blur Sigma", 0, 10, 0.01,
   "The Gaussian blur strength used to degrade the regions the model's self-attention focuses on. 0 leaves the image unchanged."],
  ["threshold", "Blur mask threshold", 0, 4, 0.01,
   "How concentrated the attention has to be for a region to be degraded. 0 blurs everything, higher values target only the most focused regions."],
];

function openSagDialog() {
  const fields = [];
  const make = ([key, label, min, max, step, help]) => {
    const input = el("input", { type: "range", min, max, step });
    input.value = state.sag[key];
    const val = el("b", {}, String(state.sag[key]));
    input.oninput = () => { val.textContent = input.value; };
    fields.push({ key, input });
    return el("label", { class: "soft-opt" },
      el("div", { class: "soft-opt-head" }, el("span", {}, label), val),
      input,
      el("div", { class: "soft-opt-help" }, help),
    );
  };
  openDialog("Self attention guidance options", el("div", {},
    el("div", { class: "soft-lead" },
      "Self attention guidance (Forge builtin extension) sharpens high-detail regions (faces, eyes, fine structures): the unconditional x0 is degraded where the model's own self-attention is concentrated, the degraded copy is re-denoised with an extra unconditional UNet pass, and the difference is fed back into the result. Needs CFG > 1 and adds one extra UNet pass per step (~50% more model time)."),
    ...SAG_OPTS.map(make),
  ), () => {
    fields.forEach(({ key, input }) => { state.sag[key] = parseFloat(input.value); });
    scheduleSaveUiState(0);
  });
}

// The generate body's sag field: null when off. All modes; the backend
// needs cfg > 1 to apply it (skips it like Forge when there is no
// unconditional pass)
function sagBody() {
  if (!$("sag").checked) return null;
  const s = state.sag;
  return { scale: s.scale, blur_sigma: s.blur, threshold: s.threshold };
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
  $("seg-model").onchange = () => scheduleSaveUiState(0);
  $("seg-run").onclick = () => openSegDialog();
  $("soft-inpaint").onchange = () => scheduleSaveUiState(0);
  $("soft-inpaint-cfg").onclick = openSoftInpaintDialog;
  $("sag").onchange = () => scheduleSaveUiState(0);
  $("sag-cfg").onclick = openSagDialog;
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
  $("style-add").onclick = () => {
    const row = styleRow("", onStylesChanged);
    $("styles").append(row);
    row.querySelector("select").focus();
    onStylesChanged();
  };
  $("style-manage").onclick = openStyleManage;
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
    storeUndoFlags(await up.json());
    await applyPngMeta(meta, width, height);
    viewRefresh();
    setStatus(meta ? `imported ${file.name} (metadata applied)` : `loaded ${file.name} onto the canvas (no metadata)`);
  } catch (e) {
    setStatus(`png import failed: ${e.message}`);
  }
}

async function applyPngMeta(meta, width, height) {
  if (!meta) {
    $("width").value = width;
    $("height").value = height;
    scheduleSaveUiState();
    return;
  }
  if (meta.prompt !== undefined && meta.prompt !== null) $("prompt").value = meta.prompt;
  if (meta.negative_prompt !== undefined && meta.negative_prompt !== null) $("negative").value = meta.negative_prompt;
  updatePromptTokens(); // imported metadata may have set the prompt
  for (const id of ["steps", "cfg", "seed", "width", "height"]) {
    if (meta[id] !== undefined && meta[id] !== null) $(id).value = meta[id];
  }
  for (const id of ["sampler", "scheduler"]) {
    const sel = $(id);
    if (meta[id] && [...sel.options].some((o) => o.value === meta[id])) sel.value = meta[id];
  }
  if (meta.loras?.length) renderMainLoras(parseMetaLoras(meta.loras));
  // styles travel as name + actual contents; the server registers missing
  // ones as in-memory "Session" temporaries (not written to styles.csv)
  if (meta.styles?.length) {
    renderMainStyles(meta.styles.map((s) => s.name));
    try {
      const r = await api("/api/styles/temp", {
        styles: meta.styles.map((s) => ({
          name: s.name,
          prompt: s.prompt || "",
          negative_prompt: s.negative_prompt || "",
        })),
      });
      state.styles = r.styles;
      renderMainStyles(getStyles()); // options now include the temporaries
    } catch (e) {
      console.error("styles/temp:", e);
    }
  }
  // soft inpainting (Forge extra parameters) round-trips through the PNG
  if (meta["Soft inpainting enabled"]) {
    $("soft-inpaint").checked = true;
    const softMeta = {
      power: "Soft inpainting schedule bias",
      scale: "Soft inpainting preservation strength",
      detail: "Soft inpainting transition contrast boost",
      maskInf: "Soft inpainting mask influence",
      threshold: "Soft inpainting difference threshold",
      contrast: "Soft inpainting difference contrast",
    };
    for (const [key, label] of Object.entries(softMeta)) {
      const v = parseFloat(meta[label]);
      if (Number.isFinite(v)) state.soft[key] = v;
    }
  }
  // self attention guidance (Forge SAG extra parameters) round-trips
  // through the PNG under the extension's raw snake_case keys
  if (meta.sag_enabled) {
    $("sag").checked = true;
    for (const [key, label] of Object.entries({
      scale: "sag_scale",
      blur: "sag_blur_sigma",
      threshold: "sag_threshold",
    })) {
      const v = parseFloat(meta[label]);
      if (Number.isFinite(v)) state.sag[key] = v;
    }
  }
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
      storeUndoFlags(await up.json());
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
    mask_padding: parseInt($("mask-padding").value, 10) || 0,
    soft_inpainting: $("soft-inpaint").checked,
    soft_mask_blend_power: state.soft.power,
    soft_mask_blend_scale: state.soft.scale,
    soft_inpaint_detail_preservation: state.soft.detail,
    soft_composite_mask_influence: state.soft.maskInf,
    soft_composite_difference_threshold: state.soft.threshold,
    soft_composite_difference_contrast: state.soft.contrast,
    sag: $("sag").checked,
    sag_scale: state.sag.scale,
    sag_blur_sigma: state.sag.blur,
    sag_threshold: state.sag.threshold,
    loras: getLoras(),
    styles: getStyles(),
    // preset is not persisted either: the dropdown always starts at "—"
    seg_model: $("seg-model").value || null,
    seg_classes: state.segClasses,
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
    ["mask-padding", "mask_padding"],
  ];
  for (const [elm, key] of fields) {
    if (st[key] !== undefined && st[key] !== null) $(elm).value = st[key];
  }
  renderMainLoras(st.loras || []);
  renderMainStyles(st.styles || []);
  // display-only restore: the model itself is NOT preloaded (it loads
  // lazily on the next generation, matching the registry design)
  if (st.model) $("model").value = st.model;
  // person mask generation: class selection lives in JS state, the model
  // select is restored here (after loadSegModels populated the dropdown)
  if (Array.isArray(st.seg_classes)) state.segClasses = st.seg_classes;
  if (st.seg_model && $("seg-model").options.length) $("seg-model").value = st.seg_model;
  // soft inpainting: enabled flag + the six options (JS state object)
  if (typeof st.soft_inpainting === "boolean") $("soft-inpaint").checked = st.soft_inpainting;
  for (const [key, k] of [
    ["soft_mask_blend_power", "power"],
    ["soft_mask_blend_scale", "scale"],
    ["soft_inpaint_detail_preservation", "detail"],
    ["soft_composite_mask_influence", "maskInf"],
    ["soft_composite_difference_threshold", "threshold"],
    ["soft_composite_difference_contrast", "contrast"],
  ]) {
    if (typeof st[key] === "number") state.soft[k] = st[key];
  }
  // self attention guidance: enabled flag + the three options (JS state
  // object)
  if (typeof st.sag === "boolean") $("sag").checked = st.sag;
  for (const [key, k] of [
    ["sag_scale", "scale"],
    ["sag_blur_sigma", "blur"],
    ["sag_threshold", "threshold"],
  ]) {
    if (typeof st[key] === "number") state.sag[k] = st[key];
  }
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
  let sy = { styles: [] };
  let sm = { models: [] };
  try {
    [m, s, u, pr, lr, sy, sm] = await Promise.all([
      fetch("/api/models").then((r) => r.json()),
      fetch("/api/samplers").then((r) => r.json()),
      fetch("/api/ui-state").then((r) => (r.ok ? r.json() : null)),
      fetch("/api/presets").then((r) => r.json()),
      fetch("/api/loras").then((r) => r.json()),
      fetch("/api/styles").then((r) => (r.ok ? r.json() : { styles: [] })),
      fetch("/api/segmentation/models").then((r) => (r.ok ? r.json() : { models: [] })),
    ]);
  } catch (e) {
    console.error("init:", e);
    return;
  }
  state.segModels = sm.models;
  const segSel = $("seg-model");
  segSel.innerHTML = "";
  if (!state.segModels.length) segSel.append(opt("", "no models"));
  state.segModels.forEach((x) => segSel.append(opt(x.id, x.name)));
  state.entries = m.models;
  state.samplers = s.samplers;
  state.schedulers = s.schedulers;
  state.presets = pr.presets;
  state.loras = lr.loras;  // [{name, group|null}]
  state.styles = sy.styles;
  const sel = $("model");
  sel.innerHTML = "";
  m.models.forEach((x) => {
    sel.append(opt(x.id, x.is_default ? `${x.name} (default)` : x.name));
  });
  if (m.current && m.models.some((x) => x.id === m.current)) sel.value = m.current;
  for (const id of ["sampler", "scheduler"]) {
    const s2 = $(id);
    s2.innerHTML = "";
    s2.append(opt("", "Model default"));
    (id === "sampler" ? s.samplers : s.schedulers).forEach((n) => s2.append(opt(n, n)));
  }
  fillPresetSelect($("preset"));
  applyUiState(u && u.state);
  syncMode();
  updatePromptTokens(); // the restored prompt may be non-empty
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
        // canvas snapshot at submit time (inpaint, or generate on a
        // non-empty canvas), shown before any result arrives; the id lets
        // it be restored as a live canvas entity
        state.thumbs.unshift({ source: true, url: msg.url, image_id: msg.image_id });
        renderThumbs();
        break;
      case "image":
        state.thumbs.push({ job: msg.job, index: msg.index, url: msg.url, seed: msg.seed,
                            image_id: msg.image_id });
        if (state.currentThumb === -1) {
          // highlight whatever is currently on the canvas: a source
          // thumbnail exists exactly when the canvas keeps its image (no
          // auto-set), otherwise the empty canvas got the first result
          // auto-set
          state.currentThumb = state.thumbs[0].source
            ? 0
            : state.thumbs.length - 1;
        }
        renderThumbs();
        break;
      case "canvas_set":
        // the canvas auto-set to a fresh result entity: its draw history
        // is empty (only the global mask is retained), so reset the draw
        // flags before re-syncing the buttons
        undoFlags.undo_draw = false;
        undoFlags.redo_draw = false;
        syncUndoButtons();
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
function currentUndoTarget() {
  return Tools.current === "mask" ? "mask" : "draw";
}
$("btn-undo").onclick = async () => {
  const t = currentUndoTarget();
  try {
    storeUndoFlags(await api("/api/canvas/undo", { target: t }));
    await viewRefresh();
  } catch (e) {
    setStatus(`undo failed: ${e.message}`);
  }
};
$("btn-redo").onclick = async () => {
  const t = currentUndoTarget();
  try {
    storeUndoFlags(await api("/api/canvas/redo", { target: t }));
    await viewRefresh();
  } catch (e) {
    setStatus(`redo failed: ${e.message}`);
  }
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
syncUndoButtons();
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

["prompt", "negative", "width", "height", "steps", "cfg", "seed", "batch", "denoise", "mask-padding"].forEach((id) => {
  $(id).addEventListener("input", () => scheduleSaveUiState());
});
$("prompt").addEventListener("input", schedulePromptTokens);
$("swap-dims").onclick = () => {
  const w = $("width").value;
  $("width").value = $("height").value;
  $("height").value = w;
  scheduleSaveUiState();
};
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
