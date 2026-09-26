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
  entries: [], // model configurations from /api/models (incl. family)
  samplers: [],
  schedulers: [],
  loras: [], // lora names from /api/loras
  loraSettings: {}, // name -> settings .json next to the .safetensors ({"preferred weight", ...})
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
  // the preset currently matching the form: {name, keys} where keys are its
  // enabled control keys. While set, the preset stays selected; a user edit of
  // any covered field clears it (clearActivePreset).
  activePreset: null,
};

function setStatus(text, isError = false) {
  const el = $("status-line");
  el.textContent = text || "";
  el.classList.toggle("error", isError);
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

// ---------------- searchable combobox ----------------
// Turns a plain element into a type-to-filter dropdown. The element keeps a
// .value accessor (id/value = the option's value, display = the option's text)
// and fires a `change` event on pick, so it drops in for the plain <select>s it
// replaces. The option list is rendered in an absolutely positioned <ul> that
// overlays the page, so it never changes the layout/footprint of the control.
//   makeCombobox(root, options, cfg?)
//   options: array of {value, text, group?} (group => rendered as a sticky
//            section header, e.g. a style optgroup)
//   cfg: { placeholder, initial }
// Shared across every combobox: when any list closes we record the time so a
// stray click (e.g. the browser re-dispatching the trailing mouseup onto a
// neighbouring input after a picked item's list collapses) can't immediately
// reopen a sibling. A genuine click on another box happens well after this.
let comboRecentClose = 0;
const COMBO_REOPEN_GUARD_MS = 250;

function makeCombobox(root, options, cfg = {}) {
  root.classList.add("combobox");
  root.innerHTML = "";
  const input = el("input", {
    class: "combo-input", type: "text", spellcheck: "false",
    autocomplete: "off", placeholder: cfg.placeholder || "",
  });
  const list = el("ul", { class: "combo-list", role: "listbox" });
  root.append(input, list);

  let opts = options;
  let value = cfg.initial || "";
  let active = -1;

  const curOpts = () => (typeof opts === "function" ? opts() : opts);
  const itemEls = () => [...list.querySelectorAll(".combo-item")];
  const textFor = (v) => {
    const o = curOpts().find((x) => x.value === v);
    return o ? o.text : v;
  };

  function render(filter = "") {
    const f = filter.trim().toLowerCase();
    list.replaceChildren();
    active = -1;
    let lastGroup = null;
    let any = false;
    for (const o of curOpts()) {
      if (f && !String(o.text).toLowerCase().includes(f)) continue;
      if (o.group && o.group !== lastGroup) {
        list.append(el("li", { class: "combo-group" }, o.group));
        lastGroup = o.group;
      }
      const li = el("li", { class: "combo-item", role: "option" }, o.text);
      li.dataset.value = o.value;
      li.onclick = () => pick(o.value, true);
      list.append(li);
      any = true;
    }
    if (!any) list.append(el("li", { class: "combo-empty" }, "no match"));
    const idx = itemEls().findIndex((li) => li.dataset.value === value);
    if (idx >= 0) setActive(idx);
  }

  function setActive(i) {
    const items = itemEls();
    items.forEach((li) => li.classList.remove("active"));
    active = i;
    if (i >= 0 && items[i]) {
      items[i].classList.add("active");
      items[i].scrollIntoView({ block: "nearest" });
    }
  }

  function open(filter = "") {
    render(filter);
    list.classList.add("open");
  }
  function close() {
    list.classList.remove("open");
    active = -1;
  }

  function pick(v, fromMouse = false) {
    value = v;
    input.value = v ? textFor(v) : "";
    close();
    root.dispatchEvent(new Event("change"));
    // Standard form behaviour: the combobox keeps focus after a selection.
    input.focus();
    if (!fromMouse) return;
    // A mouse pick collapses the list; the browser then re-dispatches the
    // trailing click onto a neighbouring input (always the first row, as the
    // scrollable pane re-anchors). Mark the time so that stray click is
    // swallowed (see the click handler) and, if focus lands on another combobox
    // shortly after, pull it back here.
    comboRecentClose = Date.now();
    const steal = (ev) => {
      const t = ev ? ev.target : null;
      const other = t && t.closest ? t.closest(".combobox") : null;
      if (!other || other === root) return; // our own focus / not a combobox
      // a stray always lands on a *closed* box; if the sibling's list is open it
      // is a genuine focus (e.g. a row that was just added), so leave it be
      const otherList = other.querySelector ? other.querySelector(".combo-list") : null;
      if (otherList && otherList.classList.contains("open")) { cleanup(); return; }
      if (Date.now() - comboRecentClose < COMBO_REOPEN_GUARD_MS) input.focus();
      cleanup();
    };
    const cleanup = () => {
      removeEventListener("focusin", steal, true);
      clearTimeout(timer);
    };
    const timer = setTimeout(cleanup, COMBO_REOPEN_GUARD_MS);
    addEventListener("focusin", steal, true);
  }

  // NOTE: focus alone deliberately does NOT open the list. Opening is driven
  // by an explicit user action (click / typing / any key) or by the caller
  // (root.open() when a row is added). Opening on mere focus made a
  // previously-closed sibling pop back open, because after a pick the browser
  // transiently moves focus onto a neighbouring combobox's input.
  input.addEventListener("click", () => {
    // swallow the trailing/ghost click that follows a pick on another row
    if (Date.now() - comboRecentClose < COMBO_REOPEN_GUARD_MS) return;
    open("");
  });
  input.addEventListener("input", () => open(input.value));
  input.addEventListener("blur", () => setTimeout(close, 120));
  input.addEventListener("keydown", (e) => {
    // keyboard: any key while focused shows the list (typing filters it,
    // arrows/enter navigate) before handling the key
    if (!list.classList.contains("open")) open(input.value);
    const items = itemEls();
    if (e.key === "ArrowDown") { e.preventDefault(); if (active < items.length - 1) setActive(active + 1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); if (active > 0) setActive(active - 1); }
    else if (e.key === "Enter") {
      e.preventDefault();
      if (active >= 0 && items[active]) pick(items[active].dataset.value);
      else if (items.length === 1) pick(items[0].dataset.value);
      else close();
    } else if (e.key === "Escape") {
      input.value = value ? textFor(value) : "";
      close();
    }
  });

  Object.defineProperty(root, "value", {
    get: () => value,
    set: (v) => {
      value = v || "";
      input.value = value ? textFor(value) : "";
    },
  });
  root.setOptions = (o) => {
    opts = o;
    if (list.classList.contains("open")) open(input.value);
  };
  root.focus = () => input.focus();
  // show the list AND focus the input; used when a row is added so the new
  // dropdown opens immediately. Kept separate from root.focus (which never
  // opens) so that a plain focus can't pop the list back up. The list is opened
  // *before* focus so that a focusin handler (see the post-pick focus guard in
  // pick) can see this box is open and leave the new row's focus alone.
  root.open = (filter = "") => { open(filter); input.focus(); };

  input.value = value ? textFor(value) : "";
  render();
  return root;
}

// ---------------- generic dialog ----------------
let dialogOk = null;
let dialogClose = null;
// quick-filter widget for the dialog title row. `onFilter` gets the trimmed,
// lowercased query (debounced); the x clears it, Escape in the input does too
// (without closing the dialog).
function makeFilterInput(onFilter) {
  const input = el("input", { type: "text", placeholder: "filter", title: "filter the list" });
  const clear = el("button", { type: "button", class: "dialog-filter-clear", title: "clear filter" }, "×");
  clear.hidden = true;
  let t = null;
  const apply = () => {
    clear.hidden = !input.value;
    onFilter(input.value.trim().toLowerCase());
  };
  input.addEventListener("input", () => {
    clearTimeout(t);
    t = setTimeout(apply, 150); // short debounce: live-filter while typing
  });
  clear.onclick = () => {
    input.value = "";
    clearTimeout(t);
    apply();
    input.focus();
  };
  input.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      e.stopPropagation(); // clear the filter, don't close the dialog
      input.value = "";
      clearTimeout(t);
      apply();
    }
  });
  return el("div", { class: "dialog-filter" }, input, clear);
}
function openDialog(title, body, onOk, okText = "Save", onClose = null, onFilter = null) {
  $("dialog-title").textContent = title;
  const f = $("dialog-filter");
  f.innerHTML = "";
  f.hidden = !onFilter;
  if (onFilter) f.append(makeFilterInput(onFilter));
  const b = $("dialog-body");
  b.innerHTML = "";
  b.append(body);
  dialogOk = onOk;
  dialogClose = onClose;
  // a close-only dialog (no onOk) has the ok button as its single action,
  // so the functionally identical Cancel button is hidden
  $("dialog-cancel").hidden = !onOk;
  $("dialog-ok").textContent = okText;
  $("dialog-ok").onclick = async () => {
    if (!dialogOk) { closeDialog(); return; }
    const r = await dialogOk(); // returning false keeps the dialog open
    if (r !== false) closeDialog();
  };
  $("dialog").hidden = false;
  if (onFilter) f.querySelector("input").focus(); // list dialogs: ready to type
}
function closeDialog() {
  $("dialog").hidden = true;
  const onClose = dialogClose;
  dialogOk = null;
  dialogClose = null;
  if (onClose) onClose(); // e.g. return to the management list dialog
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
  const root = el("span", { class: "lora-name" });
  const names = state.loras.slice();
  if (name && !names.includes(name)) names.push(name);
  makeCombobox(root, names.map((n) => ({ value: n, text: n })), {
    initial: name || "",
  });
  const wt = el("input", {
    class: "lora-weight", type: "number", min: "-10", max: "10", step: "0.05",
  });
  wt.value = weight;
  const rm = el("button", { type: "button", class: "mini icon lora-rm", title: "remove" }, "−");
  rm.onclick = () => { rm.closest(".lora-row").remove(); onChange && onChange(); };
  root.onchange = () => {
    // selecting a LoRA defaults the weight to its preferred weight
    // (the settings .json stored next to the .safetensors)
    const s = state.loraSettings[root.value];
    if (s && typeof s["preferred weight"] === "number")
      wt.value = s["preferred weight"];
    onChange && onChange();
  };
  wt.oninput = () => onChange && onChange();
  return el("div", { class: "lora-row" }, root, wt, rm);
}
function loraAddButton(listEl, onChange) {
  const add = el("button", { type: "button", class: "mini", title: "add a LoRA" }, "+ LoRA");
  add.onclick = () => {
    const row = loraRow("", 1, onChange);
    listEl.append(row);
    row.querySelector(".lora-name").open();
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
// the main form's LoRA list. Edits clear an active preset that covers LoRAs
// (applyPreset sets these rows programmatically, so it doesn't fire this).
function mainLoraChange() {
  onUserFieldChange("loras");
  scheduleSaveUiState();
}
function renderMainLoras(items) {
  const c = $("loras");
  c.innerHTML = "";
  (items || []).forEach(([n, w]) => c.append(loraRow(n, w, mainLoraChange)));
}
const getLoras = () => readLoraList($("loras")).filter((x) => x[0]);

// ---------------- style list rows (styles.csv) ----------------
// One dropdown per selected style (grouped by optgroup, everything sorted
// alphabetically) + a "-" button, mirroring the LoRA rows. "Session" styles
// are server-side temporaries (imported from image metadata, not in the CSV).
// one {value, text, group?} option per style, grouped/sorted as the old
// <select> was ("" group = top level, first; named optgroups alphabetically).
// `selected` (a style imported from image metadata but not in styles.csv) is
// appended as a temporary entry so it stays selectable.
function styleOptions(selected) {
  const groups = new Map(); // optgroup ("" = top level) -> [names]
  for (const s of state.styles) {
    const g = s.optgroup || "";
    if (!groups.has(g)) groups.set(g, []);
    groups.get(g).push(s.name);
  }
  const out = [];
  for (const [g, names] of [...groups.entries()].sort((a, b) =>
    a[0].localeCompare(b[0])
  )) {
    names.sort((a, b) => a.localeCompare(b));
    names.forEach((n) => out.push({ value: n, text: n, group: g || undefined }));
  }
  if (selected && !state.styles.some((s) => s.name === selected)) {
    out.push({ value: selected, text: `${selected} (temporary)` });
  }
  return out;
}

function styleRow(name, onChange) {
  const root = el("span", { class: "style-name" });
  makeCombobox(root, styleOptions(name), { initial: name || "" });
  const rm = el("button", { type: "button", class: "mini icon style-rm", title: "remove" }, "−");
  rm.onclick = () => { rm.closest(".style-row").remove(); onChange && onChange(); };
  root.onchange = () => onChange && onChange();
  return el("div", { class: "style-row" }, root, rm);
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

function openStyleDialog(existing, onClose = null) {
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
  }, "Save", onClose);
}

function openStyleManage() {
  const list = el("div", { class: "preset-manage" });
  const newBtn = el("button", { class: "mini", type: "button" }, "+ New style");
  let filter = "";
  const render = () => {
    list.querySelectorAll(".preset-manage-row, .muted").forEach((r) => r.remove());
    const shown = state.styles.filter((s) =>
      !filter || (s.name + " " + (s.optgroup || "")).toLowerCase().includes(filter));
    if (!state.styles.length) list.append(el("div", { class: "muted" }, "no styles yet"));
    else if (!shown.length) list.append(el("div", { class: "muted" }, "no matches"));
    shown.forEach((s) => {
      const nameEl = el("span", { class: "preset-manage-name" },
        s.name + (s.optgroup ? ` — ${s.optgroup}` : "") + (s.temporary ? " (temporary)" : ""));
      const edit = el("button", { class: "mini icon", type: "button", title: s.temporary ? "copy to styles.csv" : "edit" }, "✎");
      const del = el("button", { class: "mini icon", type: "button", title: s.temporary ? "drop (temporary styles are not in styles.csv)" : "delete" }, "✗");
      // saving/cancelling returns to this management list
      edit.onclick = () => { closeDialog(); openStyleDialog(s, openStyleManage); };
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
        } catch (e) { setStatus(e.message, true); return; }
        render();
      };
      list.append(el("div", { class: "preset-manage-row" }, nameEl, edit, del));
    });
    list.append(newBtn);
  };
  render();
  newBtn.onclick = () => { closeDialog(); openStyleDialog(null, openStyleManage); };
  openDialog("Style configuration", el("div", { class: "preset-manage-wrap" }, list), null, "Close",
    null, (q) => { filter = q; render(); });
}

// ---------------- LoRA settings (the .json next to each .safetensors) --------
// Informational per-LoRA settings (Forge-compatible files); the UI uses only
// "preferred weight" so far (the weight default on LoRA selection).
function openLoraDialog(name, onClose = null) {
  const s = state.loraSettings[name] || {};
  const nameIn = el("input", { type: "text" });
  nameIn.value = name;
  nameIn.readOnly = true;
  const descIn = el("textarea", { class: "style-prompt", rows: "4" });
  descIn.value = s.description || "";
  const wtIn = el("input", { type: "number", min: "-10", max: "10", step: "0.05" });
  wtIn.value = s["preferred weight"] ?? 0;
  const actIn = el("textarea", { class: "style-prompt", rows: "3" });
  actIn.value = s["activation text"] || "";
  const negIn = el("textarea", { class: "style-prompt", rows: "3" });
  negIn.value = s["negative text"] || "";
  const notesIn = el("textarea", { class: "style-prompt", rows: "3" });
  notesIn.value = s.notes || "";
  const box = el("div", { class: "style-dialog" },
    el("label", {}, "name", nameIn),
    el("label", {}, "description", descIn),
    el("label", {}, "preferred weight", wtIn),
    el("label", {}, "activation text", actIn),
    el("label", {}, "negative text", negIn),
    el("label", {}, "notes", notesIn));
  openDialog("Edit LoRA", box, async () => {
    const body = {
      name,
      description: descIn.value,
      preferred_weight: parseFloat(wtIn.value) || 0,
      activation_text: actIn.value,
      negative_text: negIn.value,
      notes: notesIn.value,
    };
    try {
      const r = await api("/api/loras/settings", body);
      state.loraSettings[name] = r.settings;
    } catch (e) {
      dialogError(e.message);
      return false;
    }
  }, "Save", onClose);
}

function openLoraManage() {
  const list = el("div", { class: "preset-manage" });
  let filter = "";
  const render = () => {
    list.querySelectorAll(".preset-manage-row, .muted").forEach((r) => r.remove());
    const shown = state.loras.filter((n) => !filter || n.toLowerCase().includes(filter));
    if (!state.loras.length) list.append(el("div", { class: "muted" }, "no LoRAs yet"));
    else if (!shown.length) list.append(el("div", { class: "muted" }, "no matches"));
    shown.forEach((n) => {
      const nameEl = el("span", { class: "preset-manage-name" }, n);
      const edit = el("button", { class: "mini icon", type: "button", title: "edit" }, "✎");
      edit.onclick = () => { closeDialog(); openLoraDialog(n, openLoraManage); };
      list.append(el("div", { class: "preset-manage-row" }, nameEl, edit));
    });
  };
  render();
  openDialog("LoRA configuration", el("div", { class: "preset-manage-wrap" }, list), null, "Close",
    null, (q) => { filter = q; render(); });
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
      : t.crop
        ? "crop (click to restore)"
        : `#${i} seed ${t.seed}`;
    if (t.source) image.classList.add("source");
    if (i === state.currentThumb) image.classList.add("selected");
    image.onmouseenter = () => thumbPreviewShow(t.url);
    image.onmouseleave = () => thumbPreviewHide();
    image.onclick = async () => {
      try {
        await restoreThumb(t);
      } catch (e) {
        setStatus(e.message, true);
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

// ---------------- crop tool (drag a rectangle) ----------------
// A crop turns the displayed image into a NEW canvas entity: the pre-crop
// image stays (pinned server-side) as a restorable thumbnail and the crop
// becomes a new, immediately selected thumbnail — unlike image generation,
// which only auto-selects on an empty canvas. The rect sent is already the
// clamped image/selection intersection (canvas.js clamps it to the bounds);
// the server re-clamps defensively.
function addCropThumbs(r) {
  // the not-yet-cropped image becomes a thumbnail, unless it already is one
  // (e.g. a generation result currently on the canvas)
  if (!state.thumbs.some((t) => t.image_id === r.prev_image_id)) {
    state.thumbs.push({ image_id: r.prev_image_id, url: r.prev_url, crop: true });
  }
  // the crop is a new thumbnail and, unlike generation, immediately selected
  state.thumbs.push({ image_id: r.crop_image_id, url: r.crop_url, crop: true });
  state.currentThumb = state.thumbs.length - 1;
  renderThumbs();
}

async function doCrop(x, y, w, h) {
  try {
    const r = await api("/api/canvas/crop", { x, y, width: w, height: h });
    storeUndoFlags(r);
    addCropThumbs(r);
    await viewRefresh();
  } catch (e) {
    setStatus(`crop failed: ${e.message}`, true);
    viewRedraw();
  }
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
    setStatus(`generate failed: ${e.message}`, true);
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
    setStatus(`model select failed: ${e.message}`, true);
  }
}

// Refetch the sampler/scheduler lists for the selected model's family and
// rebuild the main dropdowns. A Flux model offers only its Euler sampler
// (no schedulers); an SDXL model offers the full SDXL sets. The selection is
// kept only when it is still valid for the new family, else it resets to
// "Model default".
async function refreshSamplers() {
  const sel = $("model");
  const entry = state.entries.find((x) => x.id === sel.value) || state.entries[0];
  if (!entry) return;
  let s;
  try {
    s = await (await fetch(
      `/api/samplers?model_id=${encodeURIComponent(entry.model_id)}`
    )).json();
  } catch (e) {
    console.error("samplers:", e);
    return;
  }
  state.samplers = s.samplers;
  state.schedulers = s.schedulers;
  for (const id of ["sampler", "scheduler"]) {
    const s2 = $(id);
    const cur = s2.value;
    const list = id === "sampler" ? state.samplers : state.schedulers;
    s2.innerHTML = "";
    s2.append(opt("", "Model default"));
    list.forEach((n) => s2.append(opt(n, n)));
    s2.value = list.includes(cur) ? cur : "";
  }
}

// Family-specific main-form controls: Z-Image is a single-pass model (no
// CFG/guidance, no negative prompt), so disable those inputs instead of
// letting the user set values the backend silently ignores. The family
// comes from /api/models (state.entries), not a samplers fetch (which
// would race the model load).
function applyFamilyControls() {
  const sel = $("model");
  const entry = state.entries.find((x) => x.id === sel.value) || state.entries[0];
  if (!entry) return;
  const zimage = entry.family === "zimage";
  const cfgIn = $("cfg");
  const negIn = $("negative");
  cfgIn.disabled = zimage;
  negIn.disabled = zimage;
  cfgIn.title = zimage
    ? "Z-Image is single-pass: CFG has no effect (always 1.0)"
    : "-1 = use the model configuration's CFG";
}

// Refetch the LoRA list for the selected model's family and drop any
// main-form rows whose LoRA is no longer in that family's list (e.g. an
// SDXL LoRA selected before switching to a Flux model). Mirrors
// refreshSamplers().
async function refreshLoras() {
  const sel = $("model");
  const entry = state.entries.find((x) => x.id === sel.value) || state.entries[0];
  if (!entry) return;
  let r;
  try {
    r = await (await fetch(
      `/api/loras?family=${encodeURIComponent(entry.family)}`
    )).json();
  } catch (e) {
    console.error("loras:", e);
    return;
  }
  state.loras = r.loras;
  if (r.settings) state.loraSettings = r.settings;
  const list = $("loras");
  [...list.querySelectorAll(".lora-row")].forEach((row) => {
    const name = row.querySelector(".lora-name").value;
    if (name && !r.loras.includes(name)) row.remove();
  });
}

async function refreshModels(keepSelection = true) {
  const m = await (await fetch("/api/models")).json();
  if (m.error) throw new Error(m.error);
  const sel = $("model");
  const cur = keepSelection ? sel.value : null;
  state.entries = m.models;
  sel.setOptions(m.models.map((x) => ({
    value: x.id,
    text: x.is_default ? `${x.name} (default)` : x.name,
  })));
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
  // distilled model: Flux-only flag (the CFG value becomes the embedded
  // guidance); hidden for SDXL configurations, where it is inert.
  // entry.family comes from /api/models, so it is correct even if the
  // user opens the dialog before the model finished loading.
  const isFlux = entry.family === "flux";
  const distilledIn = el("input", { class: "cfg-distilled", type: "checkbox" });
  distilledIn.checked = entry.distilled !== false;
  const f1 = modeFieldset("text to image", entry.t2i);
  const f2 = modeFieldset("image to image / inpaint", entry.i2i);
  const box = el("div", { class: "cfg-dialog" },
    el("label", {}, "name", nameIn),
    el("label", { class: "muted" }, "model", modelIn),
    ...(isFlux ? [el("label", { class: "cfg-distilled-row", title: "checked " +
        "(default): the CFG values below are the embedded guidance " +
        "(distilled CFG; one pass per step, the negative prompt is ignored). " +
        "unchecked: classic CFG with the negative prompt (two passes per " +
        "step, about 2x the time)." }, distilledIn, "distilled model")] : []),
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
        distilled: isFlux ? distilledIn.checked : entry.distilled,
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
      setStatus(e.message, true);
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
    setStatus(`delete config failed: ${e.message}`, true);
  }
}

// ---------------- presets (Phase 4) ----------------
const PRESET_KEYS = ["width", "height", "steps", "cfg", "seed", "batch",
  "sampler", "scheduler", "mode", "denoise", "mask_padding", "loras",
  "soft_inpaint", "sag"]; // enable/disable of the sidebar checkboxes

// The preset box keeps its selection while the form still matches the applied
// preset. Programmatic changes (applyPreset itself, PNG import) don't fire
// input/change events, so they never clear it by accident; the user-change
// handlers below call onUserFieldChange for each field they touch.
function clearActivePreset() {
  state.activePreset = null;
  $("preset").value = "";
}
function onUserFieldChange(id) {
  if (!state.activePreset) return;
  const key = id.replace(/-/g, "_"); // mask-padding -> mask_padding, etc.
  if (state.activePreset.keys.includes(key)) clearActivePreset();
}

async function applyPreset(name) {
  if (!name) return false;
  let p;
  try {
    const res = await fetch(`/api/presets/${encodeURIComponent(name)}`);
    if (!res.ok) throw new Error(`preset: ${res.status}`);
    p = (await res.json()).preset;
  } catch (e) {
    setStatus(e.message, true);
    $("preset").value = ""; // apply failed: don't leave a stale selection
    return false;
  }
  const controls = p.controls || {};
  try {
    for (const key of PRESET_KEYS) {
      if (!controls[key] || p[key] === null || p[key] === undefined) continue;
      // "" resets sampler/scheduler to "Model default" (the main UI's
      // empty option); mode has no empty state
      if (p[key] === "" && key !== "sampler" && key !== "scheduler") continue;
      if (key === "loras") renderMainLoras(p[key]);
      else if (key === "mode") {
        $("mode").value = p[key];
        syncMode();
      }
      else if (key === "soft_inpaint") $("soft-inpaint").checked = p[key];
      else if (key === "sag") $("sag").checked = p[key];
      // sidebar ids are hyphenated but preset keys use underscores
      // (e.g. mask_padding -> #mask-padding); the rest map 1:1
      else $(key.replace(/_/g, "-")).value = p[key];
    }
  } catch (e) {
    setStatus(`preset "${name}": ${e.message}`, true);
    $("preset").value = ""; // apply failed: don't leave a stale selection
    return false;
  }
  // remember what we applied so later user edits of these fields clear the box
  state.activePreset = { name, keys: Object.keys(controls).filter((k) => controls[k]) };
  $("preset").value = name; // keep the selection: the form now matches it
  scheduleSaveUiState(0);
  setStatus(`preset "${name}" applied`);
  return true;
}

// state.presets: [{name, group|null}] (name-sorted, from /api/presets)
function fillPresetSelect(sel) {
  // ungrouped presets first as plain options, then one group section per
  // group: groups alphabetical, names alphabetical within a group (same
  // convention as the style dropdowns). The empty ("—") state is the
  // combobox's placeholder, not a real option.
  const opts = [];
  state.presets.filter((p) => !p.group)
    .sort((a, b) => a.name.localeCompare(b.name))
    .forEach((p) => opts.push({ value: p.name, text: p.name }));
  const groups = [...new Set(state.presets.map((p) => p.group).filter(Boolean))]
    .sort((a, b) => a.localeCompare(b));
  for (const g of groups) {
    state.presets.filter((p) => p.group === g)
      .sort((a, b) => a.name.localeCompare(b.name))
      .forEach((p) => opts.push({ value: p.name, text: p.name, group: g }));
  }
  sel.setOptions(opts);
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
  if (kind === "bool") {
    // second checkbox in the row: the enable/disable value itself
    const input = el("input", { class: "preset-val", type: "checkbox" });
    input.checked = value === true;
    return input;
  }
  const input = el("input", { class: "preset-val", type: "number" });
  if (value !== null && value !== undefined) input.value = value;
  return input;
}

function openPresetDialog(existing, onClose = null) {
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
    ["soft_inpaint", "soft inpainting", "bool", p.soft_inpaint],
    ["sag", "self attention guidance", "bool", p.sag],
  ];
  const rows = el("div", { class: "preset-rows" });
  for (const [key, label, kind, value] of defs) {
    const cb = el("input", { type: "checkbox", class: "preset-cb", id: `preset-cb-${key}` });
    cb.checked = !!controls[key];
    let input;
    if (kind === "number" || kind === "loras" || kind === "bool") {
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
    rows.append(el("div", { class: "preset-row" }, cb, el("label", { class: "preset-label", for: `preset-cb-${key}` }, label), input));
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
      else if (input.dataset.presetKind === "bool") body[key] = input.checked;
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
  }, "Save", onClose);
}

function openPresetManage() {
  const list = el("div", { class: "preset-manage" });
  const newBtn = el("button", { class: "mini", type: "button" }, "+ New preset");
  let filter = "";
  const render = () => {
    list.querySelectorAll(".preset-manage-row, .muted").forEach((r) => r.remove());
    const shown = state.presets.filter((p) => !filter || p.name.toLowerCase().includes(filter));
    if (!state.presets.length) list.append(el("div", { class: "muted" }, "no presets yet"));
    else if (!shown.length) list.append(el("div", { class: "muted" }, "no matches"));
    shown.forEach((p) => {
      const name = p.name;
      const nameEl = el("span", { class: "preset-manage-name" }, name);
      const edit = el("button", { class: "mini icon", type: "button", title: "edit" }, "✎");
      const del = el("button", { class: "mini icon", type: "button", title: "delete" }, "✗");
      edit.onclick = async () => {
        closeDialog();
        let preset;
        try {
          preset = (await (await fetch(`/api/presets/${encodeURIComponent(name)}`)).json()).preset;
        } catch (e) { setStatus(e.message, true); return; }
        // saving/cancelling returns to this management list
        openPresetDialog(preset, openPresetManage);
      };
      del.onclick = async () => {
        if (!confirm(`Delete preset "${name}"?`)) return;
        try {
          await api("/api/presets/delete", { name });
        } catch (e) { setStatus(e.message, true); return; }
        await loadPresets();
        render();
      };
      list.append(el("div", { class: "preset-manage-row" }, nameEl, edit, del));
    });
    list.append(newBtn);
  };
  render();
  newBtn.onclick = () => { closeDialog(); openPresetDialog(null, openPresetManage); };
  openDialog("Setting presets", el("div", { class: "preset-manage-wrap" }, list), null, "Close",
    null, (q) => { filter = q; render(); });
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
const BRUSH_SIZE_MIN = 10;
const BRUSH_SIZE_MAX = 300;
// log-scale size: slider 0..1 maps to BRUSH_SIZE_MIN..BRUSH_SIZE_MAX px
const sizeFromSlider = (v) => BRUSH_SIZE_MIN * Math.pow(BRUSH_SIZE_MAX / BRUSH_SIZE_MIN, v);
const sliderFromSize = (s) => Math.log(Math.max(s, BRUSH_SIZE_MIN) / BRUSH_SIZE_MIN) / Math.log(BRUSH_SIZE_MAX / BRUSH_SIZE_MIN);

function syncToolbar() {
  const st = Tools.get();
  $("tool-brush").classList.toggle("active", Tools.current === "brush");
  $("tool-mask").classList.toggle("active", Tools.current === "mask");
  $("tool-clone").classList.toggle("active", Tools.current === "clone");
  $("tool-smudge").classList.toggle("active", Tools.current === "smudge");
  $("tool-crop").classList.toggle("active", Tools.current === "crop");
  const cropMode = Tools.current === "crop";
  // The colour swatch always edits the brush tool's remembered colour
  // (mask strokes ignore colour), so it stays live in mask mode too;
  // the clone and smudge tools have no colour at all, so disable it there.
  // Crop disables all four brush controls entirely.
  $("brush-color").disabled = ["clone", "smudge", "mask", "crop"].includes(Tools.current);
  $("brush-size").disabled = cropMode;
  $("brush-screen").disabled = cropMode;
  $("brush-opacity").disabled = cropMode;
  $("brush-color").value = Tools.state.brush.color || "#ffffff";
  if (brushNames.includes(st.brush)) $("brush").value = st.brush;
  $("brush-size").value = sliderFromSize(st.size);
  $("brush-size-val").textContent = Math.round(st.size);
  $("brush-screen").checked = !!st.screen;
  $("brush-opacity").value = st.opacity;
  $("brush-opacity-val").textContent = st.opacity.toFixed(2);
}

function switchTool(t) {
  // switching to a DIFFERENT tool abandons an in-progress crop selection
  // (re-arming the crop tool itself, e.g. tapping X mid-drag, does not)
  if (view.crop && Tools.current !== t) abortCrop();
  Tools.switch(t);
  syncToolbar();
  syncUndoButtons();
}

// ---------------- undo / redo (per-image draw + per-image mask) ---------
// The server reports, on every canvas-mutating response, how far each of
// the two independent stacks of the CURRENT image can go: its draw stack
// and its mask stack (each image owns its mask, at the image's size;
// new images inherit the source image's mask state). One toolbar
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
    setStatus(`mask ${op} failed: ${e.message}`, true);
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
      setStatus(`segmentation failed: ${e.message}`, true);
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

// The generate body's soft_inpainting field: null in "generate" mode (the
// backend only applies it with an init mask anyway); in the inpaint modes
// it is always sent, enabled or not, so the PNG metadata carries
// "Soft inpainting enabled" unconditionally (like the Forge extension) and
// a dropped image can restore the checkbox in both states.
function softInpaintBody() {
  if ($("mode").value === "generate") return null;
  const s = state.soft;
  return {
    enabled: $("soft-inpaint").checked,
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

// The generate body's sag field: always sent (all modes), enabled or not,
// so the PNG metadata carries sag_enabled unconditionally (like the Forge
// extension) and a dropped image can restore the checkbox in both states.
// The backend needs cfg > 1 to apply it (skips it like Forge when there
// is no unconditional pass)
function sagBody() {
  const s = state.sag;
  return {
    enabled: $("sag").checked,
    scale: s.scale,
    blur_sigma: s.blur,
    threshold: s.threshold,
  };
}

function wireToolbar() {
  $("tool-brush").onclick = () => switchTool("brush");
  $("tool-mask").onclick = () => switchTool("mask");
  $("tool-clone").onclick = () => switchTool("clone");
  $("tool-smudge").onclick = () => switchTool("smudge");
  $("tool-crop").onclick = () => switchTool("crop");
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
  $("mode").onchange = () => { onUserFieldChange("mode"); syncMode(); }; // mode is not persisted
  $("mask-dilate").onclick = () => maskOp("dilate");
  $("mask-erode").onclick = () => maskOp("erode");
  $("mask-clear").onclick = () => maskOp("clear");
  $("seg-model").onchange = () => scheduleSaveUiState(0);
  $("seg-run").onclick = () => openSegDialog();
  $("soft-inpaint").onchange = () => { onUserFieldChange("soft-inpaint"); scheduleSaveUiState(0); };
  $("soft-inpaint-cfg").onclick = openSoftInpaintDialog;
  $("sag").onchange = () => { onUserFieldChange("sag"); scheduleSaveUiState(0); };
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
    const row = loraRow("", 1, mainLoraChange);
    $("loras").append(row);
    row.querySelector(".lora-name").open();
  };
  $("lora-manage").onclick = openLoraManage;
  $("preset-manage").onclick = openPresetManage;
  $("style-add").onclick = () => {
    const row = styleRow("", onStylesChanged);
    $("styles").append(row);
    row.querySelector(".style-name").open();
    onStylesChanged();
  };
  $("style-manage").onclick = openStyleManage;
  // applying a preset keeps it selected (applyPreset); it clears on its own
  // once the user edits one of the fields the preset covers
  $("preset").onchange = (e) => applyPreset(e.target.value);
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
    setStatus(`png import failed: ${e.message}`, true);
  }
}

async function applyPngMeta(meta, width, height) {
  // importing replaces the form with the image's settings, so any active
  // preset no longer describes it
  clearActivePreset();
  if (!meta) {
    $("width").value = width;
    $("height").value = height;
    scheduleSaveUiState();
    return;
  }
  // Importing a fresh image replaces the current LoRA/style selection: clear
  // whatever was previously chosen first, so a prompt that resolves to no
  // LoRAs/styles doesn't silently keep the user's old picks in the UI.
  renderMainLoras([]);
  renderMainStyles([]);
  if (meta.prompt !== undefined && meta.prompt !== null) $("prompt").value = meta.prompt;
  if (meta.negative_prompt !== undefined && meta.negative_prompt !== null) $("negative").value = meta.negative_prompt;
  updatePromptTokens(); // imported metadata may have set the prompt
  for (const id of ["steps", "cfg", "seed", "width", "height", "denoise"]) {
    if (meta[id] !== undefined && meta[id] !== null) $(id).value = meta[id];
  }
  for (const id of ["sampler", "scheduler"]) {
    const sel = $(id);
    if (meta[id] && [...sel.options].some((o) => o.value === meta[id])) sel.value = meta[id];
  }
  // The configuration the import ends up selecting: the image's model if
  // a configuration matches it, else the currently selected one. That
  // configuration's LoRAs were baked into the image at generation time
  // (always prepended to the UI selection), so they are dropped from the
  // metadata list before the remaining LoRAs are applied -- otherwise
  // they would load twice (once via the config, once via the UI).
  const matchedEntry = meta.model
    ? state.entries.find((e) => e.name === meta.model || e.id === meta.model) || null
    : null;
  const targetEntry = matchedEntry || currentEntry();
  if (meta.loras?.length) {
    let loras = parseMetaLoras(meta.loras);
    if (targetEntry) {
      // match on name AND weight (a user may have added a config LoRA at
      // another weight in the UI; only the config-provided copy is removed)
      const configLoras = new Set(
        [...(targetEntry.t2i?.loras || []), ...(targetEntry.i2i?.loras || [])]
          .map((l) => `${l.name}@${l.weight}`)
      );
      loras = loras.filter(([name, weight]) => !configLoras.has(`${name}@${weight}`));
    }
    renderMainLoras(loras);
  }
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
  // soft inpainting (Forge extra parameters) round-trips through the PNG;
  // the enabled flag is written unconditionally, so a dropped image
  // restores the checkbox in BOTH states (absent key = leave as-is)
  if (meta["Soft inpainting enabled"] !== undefined
      && meta["Soft inpainting enabled"] !== null) {
    $("soft-inpaint").checked = String(meta["Soft inpainting enabled"]) === "True";
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
  // through the PNG under the extension's raw snake_case keys; the
  // enabled flag is written unconditionally, so a dropped image restores
  // the checkbox in BOTH states (absent key = leave as-is)
  if (meta.sag_enabled !== undefined && meta.sag_enabled !== null) {
    $("sag").checked = String(meta.sag_enabled) === "True";
    for (const [key, label] of Object.entries({
      scale: "sag_scale",
      blur: "sag_blur_sigma",
      threshold: "sag_threshold",
    })) {
      const v = parseFloat(meta[label]);
      if (Number.isFinite(v)) state.sag[key] = v;
    }
  }
  // switch to the image's model (see matchedEntry above)
  if (matchedEntry) {
    $("model").value = matchedEntry.id;
    selectModel(matchedEntry.id);
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
      setStatus(`canvas drop failed: ${err.message}`, true);
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
    // sidebar (control panel) width set by dragging its right border
    sidebar_width: Math.round($("controls").getBoundingClientRect().width),
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
  // sidebar width set by dragging its right border
  if (typeof st.sidebar_width === "number") $("controls").style.width = `${st.sidebar_width}px`;
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
  state.loras = lr.loras;  // lora names (str) from /api/loras
  state.loraSettings = lr.settings || {};  // name -> settings (file .jsons)
  state.styles = sy.styles;
  // the preset and model fields are searchable comboboxes (makeCombobox):
  // build them first so the .value / setOptions below work, then feed options
  const sel = makeCombobox($("model"), [], {});
  makeCombobox($("preset"), [], { placeholder: "—" });
  sel.setOptions(m.models.map((x) => ({
    value: x.id,
    text: x.is_default ? `${x.name} (default)` : x.name,
  })));
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
  await refreshSamplers(); // family-correct sampler lists for the selected model
  applyFamilyControls(); // family-specific main-form controls
  await refreshLoras(); // family-correct LoRA list for the selected model
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
// resolves once the first websocket connection is up; the boot section
// keeps the loading overlay visible until this AND all initial fetches settle
let resolveWsConnected = () => {};
const wsConnected = new Promise((r) => (resolveWsConnected = r));
function connectWS() {
  ws = new WebSocket(`ws://${location.host}/api/ws`);
  ws.onopen = resolveWsConnected;
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
        // the canvas auto-set to a fresh result entity: its draw history is
        // empty and its mask history was inherited from the source image;
        // re-sync both flag pairs from the event
        storeUndoFlags(msg);
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
        setStatus(`error: ${msg.message}`, true);
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
    setStatus(`save failed: ${e.message}`, true);
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
    setStatus(`undo failed: ${e.message}`, true);
  }
};
$("btn-redo").onclick = async () => {
  const t = currentUndoTarget();
  try {
    storeUndoFlags(await api("/api/canvas/redo", { target: t }));
    await viewRefresh();
  } catch (e) {
    setStatus(`redo failed: ${e.message}`, true);
  }
};
$("model").onchange = (e) => {
  selectModel(e.target.value);
  refreshSamplers(); // sampler lists follow the model's family
  applyFamilyControls(); // family-specific main-form controls
  refreshLoras(); // LoRA list follows the model's family
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

// B / M / C / S switch tools (when not typing in a field); CTRL (clone
// tool, a source set) makes the source follow the cursor until it is released
document.addEventListener("keydown", (e) => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "");
  if (typing) return;
  if (e.key === "b" || e.key === "B") switchTool("brush");
  else if (e.key === "m" || e.key === "M") switchTool("mask");
  else if (e.key === "c" || e.key === "C") switchTool("clone");
  else if (e.key === "s" || e.key === "S") switchTool("smudge");
  else if (e.key === "x" || e.key === "X") switchTool("crop");
  // Esc while dragging the crop rectangle aborts the selection
  else if (e.key === "Escape" && view.crop) abortCrop();
  else if (e.key === "Control") cloneGestureStart();
});
document.addEventListener("keyup", (e) => {
  if (e.key === "Control") cloneGestureEnd();
});

// -------- sidebar resize: drag the border between the sidebar and stage --------
const SIDEBAR_MIN_W = 280;
const SIDEBAR_MAX_W = 800;
$("sidebar-resizer").addEventListener("mousedown", (e) => {
  e.preventDefault();
  const start = { x: e.clientX, w: $("controls").getBoundingClientRect().width };
  const move = (ev) => {
    const w = Math.round(
      Math.min(SIDEBAR_MAX_W, Math.max(SIDEBAR_MIN_W, start.w + ev.clientX - start.x))
    );
    $("controls").style.width = `${w}px`;
  };
  const up = () => {
    window.removeEventListener("mousemove", move);
    window.removeEventListener("mouseup", up);
    document.body.classList.remove("resizing-sidebar");
    scheduleSaveUiState();
  };
  document.body.classList.add("resizing-sidebar");
  window.addEventListener("mousemove", move);
  window.addEventListener("mouseup", up);
});

// ---------------- boot ----------------
Tools.load();
syncToolbar();
syncUndoButtons();
wireToolbar();
wireConfig();
wirePromptDrop();
wireCanvasDrop();
// the loading overlay covers the UI until it is settled: the websocket AND
// every request whose response rewrites the form (init), plus the brush
// list and the canvas. While it is up, pointer input hits the overlay and
// the keydown guard below blocks the keyboard, so nothing the user does
// can be clobbered by the state load. Fail open: hide it if any of these
// rejects instead of leaving a permanent overlay.
const pInit = init();
const pBrushes = loadBrushes();
viewResize();
const pCanvas = viewRefresh();
setupCanvasInput();
window.addEventListener("resize", viewResize);
connectWS();
Promise.all([pInit, pBrushes, pCanvas, wsConnected])
  .then(() => { $("loading-overlay").hidden = true; })
  .catch((e) => { console.error("boot:", e); $("loading-overlay").hidden = true; });

// while the overlay is up, block keyboard input into the fields behind it
// (the overlay already eats pointer events; Tab can still focus them)
document.addEventListener(
  "keydown",
  (e) => {
    if ($("loading-overlay").hidden) return;
    const t = document.activeElement;
    if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) e.preventDefault();
  },
  true
);

["prompt", "negative", "width", "height", "steps", "cfg", "seed", "batch", "denoise", "mask-padding"].forEach((id) => {
  $(id).addEventListener("input", () => { onUserFieldChange(id); scheduleSaveUiState(); });
});
$("prompt").addEventListener("input", schedulePromptTokens);
$("seed-from-image").onclick = () => {
  const t = state.thumbs[state.currentThumb];
  // no image selected, or a source image (restored from the canvas, no seed)
  if (!t || typeof t.seed !== "number") return;
  $("seed").value = String(t.seed);
  onUserFieldChange("seed");
  scheduleSaveUiState();
};
$("swap-dims").onclick = () => {
  const w = $("width").value;
  $("width").value = $("height").value;
  $("height").value = w;
  onUserFieldChange("width");
  onUserFieldChange("height");
  scheduleSaveUiState();
};
["sampler", "scheduler"].forEach((id) => {
  $(id).onchange = () => { onUserFieldChange(id); scheduleSaveUiState(0); };
});
// last-ditch save on tab close (sendBeacon is fire-and-forget, always
// delivered; the endpoint ignores content-type to accept its text/plain)
window.addEventListener("beforeunload", () => {
  // the load is still in flight: the server's persisted state is about to
  // fill the form, so don't clobber it with the not-yet-applied defaults
  if (!$("loading-overlay").hidden) return;
  try {
    navigator.sendBeacon(
      "/api/ui-state",
      new Blob([JSON.stringify(readUiState())], { type: "text/plain" })
    );
  } catch (e) {}
});
