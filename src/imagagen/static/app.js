/* imagagen UI: controls, thumbnails, WS events, toolbar wiring, and the
   Phase 4 configuration layer (model configs, LoRAs, presets).
   Canvas rendering/input live in canvas.js, tool state in tools.js.
   Generic UI primitives ($, el, makeCombobox, openDialog, ...) live in
   ui-elements.js, which is loaded before this file. */
"use strict";

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
  segModel: null, // last-used model file name (the dialog's dropdown)
  segClasses: [], // last-used person-mask class selection ([] = default)
  segThreshold: 0.25, // confidence cut (0..1) of the segmenter dialog
  segFeather: 0, // mask edge feather radius (canvas px) of that dialog
  // the preset currently matching the form: {name, keys} where keys are its
  // enabled control keys. While set, the preset stays selected; a user edit of
  // any covered field clears it (clearActivePreset).
  activePreset: null,
  // the thumbnail checked as "reference image" (canvas entity id; exactly
  // one at a time — checking a new one clears the previous). The
  // generation describes that image with the VLM and injects the caption
  // into the prompt; the caption preview rides in the checkbox tooltip.
  referenceImageId: null,
  referenceCaption: null, // preview text for the checked reference (null = fetching/none)
  // enabled caption sections for the checked reference (null = all;
  // see renderRefSections for the tag row under the negative prompt)
  referenceSections: null,
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
    // styles and LoRA activation texts are applied server-side (exactly
    // as generation does), so the count reflects the full prompt the
    // model will actually see
    r = await api("/api/prompt-tokens", {
      prompt: $("prompt").value,
      styles: getStyles(),
      loras: getLoras().filter((x) => x[2]).map(([n, w]) => [n, w]),
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

const opt = (value, text) => {
  const o = el("option", { value }, text);
  return o;
};

// ---------------- LoRA list rows (form + dialogs) ----------------
// withToggle = the main form's on/off button (disabled LoRAs are kept in
// the list but not applied, and round-trip through the PNG metadata and
// ui-state); config/preset dialogs pass false (their LoRAs are always on)
function loraRow(name, weight, onChange, enabled = true, withToggle = true) {
  const root = el("span", { class: "lora-name" });
  // options as a function so the dropdown always shows the *current*
  // family's LoRAs (state.loras changes on model switch); a row's selected
  // name is kept in the list even if it's no longer in state.loras (its text
  // then falls back to the raw value)
  makeCombobox(root, () => pathGroupOptions(
    state.loras.concat(name && !state.loras.includes(name) ? [name] : [])
      .map((n) => ({
        value: n, text: n,
        title: state.loraSettings[n]?.description || undefined,
      }))
  ), { initial: name || "" });
  const wt = el("input", {
    class: "lora-weight", type: "number", min: "-10", max: "10", step: "0.05",
  });
  wt.value = weight;
  const tog = withToggle ? el("button", {
    type: "button", class: "mini icon lora-toggle", title: "toggle on/off (off = not applied)",
  }, "◐") : null;
  const rm = el("button", { type: "button", class: "mini icon lora-rm", title: "remove" }, "−");
  const row = el("div", { class: "lora-row" }, root, wt, tog, rm);
  const setInput = root.querySelector(".combo-input");
  const setEnabled = (on) => {
    row.classList.toggle("off", !on);
    // disabled UI state for the dropdown: dimmed (CSS) and not openable
    setInput.disabled = !on;
    if (tog)
      tog.title = on
        ? "disable this LoRA (kept in the list, not applied)"
        : "enable this LoRA";
  };
  if (tog)
    tog.onclick = () => {
      setEnabled(row.classList.contains("off"));
      onChange && onChange();
    };
  rm.onclick = () => { row.remove(); onChange && onChange(); };
  root.onchange = () => {
    // selecting a LoRA defaults the weight to its preferred weight
    // (the settings .json stored next to the .safetensors)
    const s = state.loraSettings[root.value];
    if (s && typeof s["preferred weight"] === "number")
      wt.value = s["preferred weight"];
    onChange && onChange();
  };
  wt.oninput = () => onChange && onChange();
  setEnabled(enabled);
  return row;
}
function loraAddButton(listEl, onChange, withToggle = true) {
  const add = el("button", { type: "button", class: "mini", title: "add a LoRA" }, "+ LoRA");
  add.onclick = () => {
    const row = loraRow("", 0.8, onChange, true, withToggle);
    listEl.append(row);
    row.querySelector(".lora-name").open();
    onChange && onChange();
  };
  return add;
}
// [name, weight, enabled] per row (rows without a toggle are always on)
function readLoraList(listEl) {
  return [...listEl.querySelectorAll(".lora-row")].map((r) => [
    r.querySelector(".lora-name").value,
    parseFloat(r.querySelector(".lora-weight").value) || 0,
    !r.classList.contains("off"),
  ]);
}
// the main form's LoRA list. Edits clear an active preset that covers LoRAs
// (applyPreset sets these rows programmatically, so it doesn't fire this).
function mainLoraChange() {
  onUserFieldChange("loras");
  scheduleSaveUiState();
  // a LoRA's activation text is part of the model prompt
  schedulePromptTokens();
}
// items: [name, weight] or [name, weight, enabled]
function renderMainLoras(items) {
  const c = $("loras");
  c.innerHTML = "";
  (items || []).forEach(([n, w, en]) =>
    c.append(loraRow(n, w, mainLoraChange, en !== false)));
}
// [name, weight, enabled] per row
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

// like loraRow: the main form gets an on/off toggle button (withToggle);
// disabled styles are kept in the list but not applied, and round-trip
// through the PNG metadata (Styles JSON enabled flag) and ui-state
function styleRow(name, onChange, enabled = true, withToggle = true) {
  const root = el("span", { class: "style-name" });
  makeCombobox(root, styleOptions(name), { initial: name || "" });
  const tog = withToggle ? el("button", {
    type: "button", class: "mini icon style-toggle", title: "toggle on/off (off = not applied)",
  }, "◐") : null;
  const rm = el("button", { type: "button", class: "mini icon style-rm", title: "remove" }, "\u2212");
  const row = el("div", { class: "style-row" }, root, tog, rm);
  const setInput = root.querySelector(".combo-input");
  const setEnabled = (on) => {
    row.classList.toggle("off", !on);
    // disabled UI state for the dropdown: dimmed (CSS) and not openable
    setInput.disabled = !on;
    if (tog)
      tog.title = on
        ? "disable this style (kept in the list, not applied)"
        : "enable this style";
  };
  if (tog)
    tog.onclick = () => {
      setEnabled(row.classList.contains("off"));
      onChange && onChange();
    };
  rm.onclick = () => { row.remove(); onChange && onChange(); };
  root.onchange = () => onChange && onChange();
  setEnabled(enabled);
  return row;
}

// items: a style name, or [name, enabled]
function renderMainStyles(items) {
  const c = $("styles");
  c.innerHTML = "";
  (items || []).forEach((x) => c.append(styleRow(
    Array.isArray(x) ? x[0] : x, onStylesChanged,
    !Array.isArray(x) || x[1] !== false)));
}
// enabled style names only (what the server applies)
const getStyles = () => getAllStyles()
  .filter(([n, e]) => e).map(([n]) => n);
// [name, enabled] per row
const getAllStyles = () => [...$("styles").querySelectorAll(".style-row")]
  .map((r) => [r.querySelector(".style-name").value,
               !r.classList.contains("off")])
  .filter(([n]) => n);

async function loadStyles() {
  const r = await (await fetch("/api/styles")).json();
  state.styles = r.styles || [];
  // re-render existing rows against the fresh list, keeping the selection
  // (on/off state included)
  renderMainStyles(getAllStyles());
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
    el("label", {}, "Name", nameIn),
    el("label", { title: "optgroup: groups styles in the UI dropdown (\"Session\" is reserved for imported temporary styles)" }, "Group", groupIn),
    el("label", { title: "{prompt} is replaced with the prompt so far at generation time; without it the text is appended to the prompt" }, "Prompt", promptIn),
    el("label", {}, "Negative prompt", negIn));
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

let styleManageFilter = ""; // survives dialog round-trips (edit/new returns here)
function openStyleManage() {
  const list = el("div", { class: "preset-manage" });
  const newBtn = el("button", { class: "mini", type: "button" }, "+ New style");
  let filter = styleManageFilter;
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
  };
  render();
  newBtn.onclick = () => { closeDialog(); openStyleDialog(null, openStyleManage); };
  // the new-style button stays below the list (outside the scrollable area)
  openDialog("Style configuration", el("div", { class: "preset-manage-wrap" }, list, newBtn), null, "Close",
    null, (q) => { filter = styleManageFilter = q; render(); }, styleManageFilter);
}

// ---------------- LoRA settings (the .json next to each .safetensors) --------
// Informational per-LoRA settings (Forge-compatible files); the UI uses only
// "preferred weight" so far (the weight default on LoRA selection).
function openLoraDialog(name, onClose = null) {
  const s = state.loraSettings[name] || {};
  const nameIn = el("input", { type: "text" });
  nameIn.value = name;
  nameIn.readOnly = true;
  const descIn = el("textarea", {
    class: "style-prompt", rows: "4",
    placeholder: "Shown in LoRA tooltip",
  });
  descIn.value = s.description || "";
  const wtIn = el("input", { type: "number", min: "-10", max: "10", step: "0.05" });
  wtIn.value = s["preferred weight"] ?? 0.8;
  const actIn = el("textarea", {
    class: "style-prompt", rows: "3",
    placeholder: "Will be appended to prompt, or use {prompt} to place prompt in text.",
  });
  actIn.value = s["activation text"] || "";
  const negIn = el("textarea", {
    class: "style-prompt", rows: "3",
    placeholder: "Will be appended to negative prompt, or use {prompt} to place negative prompt in text.",
  });
  negIn.value = s["negative text"] || "";
  const notesIn = el("textarea", { class: "style-prompt", rows: "3" });
  notesIn.value = s.notes || "";
  const box = el("div", { class: "style-dialog" },
    el("label", {}, "Name", nameIn),
    el("label", {}, "Description", descIn),
    el("label", {}, "Preferred weight", wtIn),
    el("label", {}, "Activation text", actIn),
    el("label", {}, "Negative text", negIn),
    el("label", {}, "Notes", notesIn));
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

let loraManageFilter = ""; // survives dialog round-trips (edit returns here)
function openLoraManage() {
  const list = el("div", { class: "preset-manage" });
  let filter = loraManageFilter;
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
    null, (q) => { filter = loraManageFilter = q; render(); }, loraManageFilter);
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
  await api("/api/canvas/select", { id: t.image_id });
}

// hover preview: the thumbnail's full image laid over the canvas box. A
// purely visual overlay -- it never touches the view (zoom/pan), the mask,
// or the undo stacks. The container is positioned by JS onto the <canvas>
// element's on-screen rect so it fits it exactly (same border/background),
// and it lays the image out in one of two ways:
//  - scale-to-fit: the whole image scaled into the box with the SAME padding
//    the zoom-to-fit button gives the canvas image (viewFit's 0.98 factor),
//    centred, never cropped. Used while SHIFT is held, and whenever the
//    thumbnail's dimensions differ from the image on the canvas.
//  - match-the-canvas: when the thumbnail has the SAME dimensions as the
//    canvas image and SHIFT is NOT held, the image is placed at the canvas
//    view's exact zoom and pan (the box crops it), so hovering compares the
//    two images in the identical viewport -- same area, same zoom.
let thumbPreviewHideTimer = null;
let thumbPreviewUrl = null; // url currently shown (guards the async probe)
let thumbPreviewSize = null; // [w,h] intrinsic size of that image, null = unknown

// (re-)apply the layout of the visible preview; also called from viewRedraw
// (zoom/pan changes) and on SHIFT key changes, so it tracks the canvas view
function thumbPreviewLayout() {
  const pv = $("thumb-preview");
  if (pv.hidden) return;
  const img = $("thumb-preview-img");
  const match =
    !shiftDown &&
    !!thumbPreviewSize &&
    thumbPreviewSize[0] === view.w &&
    thumbPreviewSize[1] === view.h;
  if (match) {
    // the exact canvas viewport: same zoom, same pan, cropped by the box
    img.classList.add("match");
    img.style.left = `${view.ox}px`;
    img.style.top = `${view.oy}px`;
    img.style.width = `${view.w * view.scale}px`;
    img.style.height = `${view.h * view.scale}px`;
    img.style.imageRendering = view.scale >= 4 ? "pixelated" : "auto";
  } else {
    img.classList.remove("match");
    img.style.left = img.style.top = img.style.width = img.style.height = "";
    img.style.imageRendering = "";
  }
}

function thumbPreviewShow(url) {
  if (!view.has) return; // no canvas to overlay on
  clearTimeout(thumbPreviewHideTimer);
  const pv = $("thumb-preview");
  const img = $("thumb-preview-img");
  img.src = url;
  thumbPreviewUrl = url;
  // the layout depends on the image's intrinsic size (usually cached: the
  // strip thumbnail is the same URL), probe it when it is not
  thumbPreviewSize =
    img.complete && img.naturalWidth ? [img.naturalWidth, img.naturalHeight] : null;
  const holder = $("canvas-holder").getBoundingClientRect();
  const c = cv().getBoundingClientRect();
  pv.style.left = `${c.left - holder.left}px`;
  pv.style.top = `${c.top - holder.top}px`;
  pv.style.width = `${c.width}px`;
  pv.style.height = `${c.height}px`;
  pv.hidden = false;
  thumbPreviewLayout();
  if (!thumbPreviewSize) {
    const probe = new Image();
    probe.onload = () => {
      if (thumbPreviewUrl !== url) return; // preview moved on to another thumb
      thumbPreviewSize = [probe.naturalWidth, probe.naturalHeight];
      thumbPreviewLayout();
    };
    probe.src = url;
  }
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

// SHIFT held/unheld switches the preview between match-the-canvas and
// scale-to-fit live, while the mouse stays on the thumbnail
for (const ev of ["keydown", "keyup"])
  document.addEventListener(ev, (e) => {
    if (e.key === "Shift") thumbPreviewLayout();
  });

// the pin checkbox of a thumbnail: checked thumbnails (and, server-side,
// their canvas entities) survive new generations until unchecked.
// Unchecking NEVER removes the thumbnail — unpinned entries are dropped at
// the START of the next generation (generate()), which is also when the
// server evicts their entities
function pinToggle(t) {
  return el("input", {
    type: "checkbox",
    class: "thumb-pin",
    title: "Pin thumbnail",
    "aria-label": "Pin thumbnail",
    onchange: async () => {
      const on = !t.pinned;
      try {
        if (on) {
          await api("/api/canvas/pin", { id: t.image_id, pin: true });
          t.pinned = true;
        } else {
          // the entity pin only goes away when no other thumbnail shows the
          // same image (e.g. a crop that reused an existing thumbnail)
          const stillShown = state.thumbs.some(
            (o) => o !== t && o.image_id === t.image_id);
          if (!stillShown)
            await api("/api/canvas/pin", { id: t.image_id, pin: false });
          t.pinned = false;
          // A reference has to survive until submit time, so an unpinned
          // thumbnail cannot stay the reference image
          if (state.referenceImageId === t.image_id) {
            state.referenceImageId = null;
            state.referenceCaption = null;
            state.referenceSections = null;
            renderRefSections(); // the tag row goes with the reference
          }
        }
      } catch (e) {
        setStatus(e.message, true);
        renderThumbs(); // re-sync the checkbox with t.pinned
        return;
      }
      renderThumbs();
    },
  });
}

// Mirror of imagagen.reference.parse_caption_sections (keep in sync):
// blank-line paragraphs, "**Title:**" headings, unlabeled leading
// paragraphs -> Overview, trailing ones -> Summary, deduped titles.
function parseCaptionSections(caption) {
  const paras = (caption || "").split(/\n\n+/)
    .map((s) => s.trim()).filter(Boolean);
  const raw = [];
  const BOLD = /^\*\*(.+?):\*\*\s*([\s\S]*)$/;
  // plain-text heading variant (mirror of the Python parser): title-case
  // words (at most three) + colon, so ordinary sentences never match
  const PLAIN = /^(?!\*\*)([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*){0,2}):\s*([\s\S]*)$/;
  for (const p of paras) {
    const m = p.match(BOLD) || p.match(PLAIN);
    if (m) raw.push({ title: m[1].trim(), paras: m[2].trim() ? [m[2].trim()] : [] });
    else if (raw.length) raw[raw.length - 1].paras.push(p);
    else raw.push({ title: null, paras: [p] });
  }
  // trailing unlabeled paragraphs after the LAST heading are the
  // summary (mirror of the Python parser: the last section keeps its
  // first paragraph, the rest becomes Summary)
  if (raw.length && raw[raw.length - 1].title !== null
      && raw[raw.length - 1].paras.length > 1) {
    const last = raw[raw.length - 1];
    raw.push({ title: null, paras: last.paras.slice(1) });
    last.paras = last.paras.slice(0, 1);
  }
  const out = [];
  const seen = {};
  raw.forEach((sec, i) => {
    let title = sec.title ?? (i === 0 ? "Overview" : "Summary");
    const n = (seen[title] || 0) + 1;
    seen[title] = n;
    if (n > 1) title = `${title} (${n})`;
    const text = sec.paras.join("\n\n").trim();
    if (text) out.push({ title, text });
  });
  return out;
}

// The caption-section tag row under the negative prompt: one toggle per
// section of the checked reference's caption preview, all enabled by
// default (state.referenceSections null = all). Hidden when no
// reference is checked or the caption has <= 1 section. While the
// preview is still pending, an animated "Processing..." placeholder
// spins (the dots cycle every 500 ms).
let refDotsTimer = null;
function renderRefSections() {
  const root = $("ref-sections");
  root.textContent = "";
  if (refDotsTimer) { clearInterval(refDotsTimer); refDotsTimer = null; }
  if (state.referenceImageId && state.referenceCaption === null) {
    const dots = el("span", {}, ".");
    root.appendChild(el("span", { class: "ref-processing" },
                        "Processing", dots));
    root.classList.remove("hidden");
    refDotsTimer = setInterval(() => {
      dots.textContent = ".".repeat(dots.textContent.length % 3 + 1);
    }, 500);
    return;
  }
  const secs = state.referenceImageId && state.referenceCaption
    ? parseCaptionSections(state.referenceCaption) : [];
  // fallback: a caption the parser found no sections in gets a single
  // "Reference" tag toggling the whole description (enabled = null =
  // all sections; off = [] = apply nothing)
  const tags = secs.length === 1
    ? [{ title: "Reference", text: secs[0].text }] : secs;
  if (tags.length <= 0) {
    root.classList.add("hidden");
    return;
  }
  root.classList.remove("hidden");
  const enabled = (t) => !Array.isArray(state.referenceSections)
    || state.referenceSections.includes(t);
  for (const s of tags) {
    root.appendChild(el("button", {
      type: "button",
      class: `ref-tag mini${enabled(s.title) ? " active" : ""}`,
      title: `include the "${s.title}" caption section in the prompt`,
      onclick: () => {
        const titles = tags.map((x) => x.title);
        const next = new Set(titles.filter(enabled));
        if (next.has(s.title)) next.delete(s.title);
        else next.add(s.title);
        const all = titles.every((t) => next.has(t));
        state.referenceSections = all ? null
          : titles.filter((t) => next.has(t));
        renderRefSections();
      },
    }, s.title));
  }
}

// Fetch the caption preview for the checked reference (shared by the
// checkbox handler and the ui-state restore path).
async function fetchReferencePreview() {
  const id = state.referenceImageId;
  if (!id) return;
  try {
    const r = await api("/api/reference/caption", { image_id: id });
    if (state.referenceImageId === id) {
      state.referenceCaption = r.caption;
      renderThumbs();
      renderRefSections();
    }
  } catch (e) {
    if (state.referenceImageId === id) {
      // empty (not null) marks the preview as finished-failed so the
      // tag row stops spinning instead of showing the placeholder
      state.referenceCaption = "";
      renderRefSections();
      setStatus(`reference caption preview failed: ${e.message}`, true);
    }
  }
}

function renderThumbs() {
  const strip = $("thumbs");
  // an empty strip stays visible as a dropzone (see wireThumbsDrop), styled
  // by the .empty class; `hidden` would display:none it away from drops
  strip.hidden = false;
  strip.classList.toggle("empty", state.thumbs.length === 0);
  strip.innerHTML = "";
  state.thumbs.forEach((t, i) => {
    const item = el("div", { class: "thumb-item" });
    const image = document.createElement("img");
    // The title names the image's dimensions too, but those are only known
    // once the file has been decoded, so it is (re)built on load.
    const setTitle = () => {
      const dims = image.naturalWidth
        ? ` ${image.naturalWidth}\u00d7${image.naturalHeight}` : "";
      image.title = t.source
        ? `source image${dims}${typeof t.seed === "number" ? ` (seed ${t.seed})` : ""} (click to restore)`
        : t.crop
          ? `crop${dims} (click to restore)`
          : t.upload
            ? `uploaded image${dims} (click to restore)`
            : `#${i} seed ${t.seed}${dims}`;
    };
    image.onload = setTitle;
    setTitle(); // dimensions unknown yet: seed-only title until the decode lands
    image.src = t.url;
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
    const pin = pinToggle(t);
    pin.checked = !!t.pinned;
    // the checkbox overlaps the image's corner: keep the hover preview
    // stable across it
    pin.onmouseenter = () => thumbPreviewShow(t.url);
    pin.onmouseleave = () => thumbPreviewHide();
    // "reference image": describe this thumbnail into the prompt of the
    // next generation. Single reference: checking one clears the others
    // (the re-render below resets every sibling checkbox).
    const ref = el("input", {
      type: "checkbox",
      class: "thumb-ref",
      "aria-label": "Reference image",
      onchange: async () => {
        if (!ref.checked) {
          state.referenceImageId = null;
          state.referenceCaption = null;
          state.referenceSections = null;
          renderRefSections();
          renderThumbs();
          return;
        }
        state.referenceImageId = t.image_id;
        state.referenceCaption = null;
        state.referenceSections = null; // new reference: all sections on
        renderRefSections();
        if (!t.pinned) {
          // unpinned thumbnails are dropped at submit: a reference must
          // survive, so auto-pin it (same API as the pin checkbox)
          try {
            await api("/api/canvas/pin", { id: t.image_id, pin: true });
            t.pinned = true;
          } catch (e) {
            setStatus(e.message, true);
            state.referenceImageId = null;
            renderThumbs(); // re-sync both checkboxes with the server state
            return;
          }
        }
        renderThumbs();
        // caption preview: show what the generation will inject
        await fetchReferencePreview();
      },
    });
    ref.checked = state.referenceImageId === t.image_id;
    ref.title = ref.checked && state.referenceCaption
      ? `Reference image — will inject: “${state.referenceCaption}”`
      : "Reference image: describe this image into the prompt of the next generation";
    ref.onmouseenter = () => thumbPreviewShow(t.url);
    ref.onmouseleave = () => thumbPreviewHide();
    item.append(image, pin, ref);
    strip.appendChild(item);
  });
}

// page load: rebuild the strip from the server's jobs. The strip normally
// accumulates from WS 'source_image'/'image' events, which are lost across
// a reload. The server only retains in-flight jobs plus the last finished
// one (older jobs are dropped on submit), so only those are restorable.
function restoreQueueThumbs(q) {
  const pinned = new Set(q && q.pinned);  // user pins outlive their jobs
  for (const j of [...((q && q.jobs) || [])].reverse()) {  // newest job first
    if (j.source) state.thumbs.push({ source: true, url: j.source.url,
                                      image_id: j.source.image_id,
                                      seed: j.source.seed, // null: no seed in its metadata
                                      pinned: pinned.has(j.source.image_id) });
    for (const r of j.results) {
      state.thumbs.push({ job: j.id, index: r.index, url: r.url, seed: r.seed,
                          image_id: r.image_id,
                          pinned: pinned.has(r.image_id) });
    }
  }
  // pinned images whose job was dropped before the reload: still live canvas
  // entities, shown at the end of the strip (oldest material last)
  const shown = new Set(state.thumbs.map((t) => t.image_id));
  for (const id of pinned) {
    if (shown.has(id)) continue;
    state.thumbs.push({ image_id: id,
                        url: `/api/canvas/image.png?id=${encodeURIComponent(id)}`,
                        pinned: true });
  }
  // highlight the entity currently on the canvas, if it is a thumbnail
  if (q && q.current) {
    const i = state.thumbs.findIndex((t) => t.image_id === q.current);
    if (i >= 0) state.currentThumb = i;
  }
  if (state.thumbs.length) renderThumbs();
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
    await forceSync(); // the server reads the canvas (pre-crop entity)
    const r = await api("/api/canvas/crop", { x, y, width: w, height: h });
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
  syncGenerateEnabled();
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
    rotation: parseInt($("rotation").value, 10),
    steps: parseInt($("steps").value, 10),
    cfg: parseFloat($("cfg").value), // -1 = "use model default"
    seed: parseInt($("seed").value, 10),
    batch_size: parseInt($("batch").value, 10),
    sampler: $("sampler").value || null, // null = "use model default"
    scheduler: $("scheduler").value || null,
    // disabled LoRAs are not applied but travel so the image metadata can
    // record them (restored disabled on import)
    loras: getLoras().filter((x) => x[2]).map(([n, w]) => [n, w]),
    disabled_loras: getLoras().filter((x) => !x[2]).map(([n, w]) => [n, w]),
    // disabled styles are not applied but travel so the image metadata can
    // record them (restored disabled on import)
    styles: getStyles(),
    disabled_styles: getAllStyles().filter((x) => !x[1]).map((x) => x[0]),
    model_id: $("model").value || null,
    mode: $("mode").value,
    reference_image_id: state.referenceImageId || null,
    reference_sections: Array.isArray(state.referenceSections)
      ? state.referenceSections : null,
    denoise: parseFloat($("denoise").value),
    mask_padding: parseInt($("mask-padding").value, 10) || 0,
    oversample: parseFloat($("oversample").value) || 1,
  };
  // THIS is where thumbnails are removed: unpinning alone never deletes
  // one, the generation drops every unpinned entry now (the server evicts
  // their entities at the same moment). Pinned thumbnails are retained (the
  // images stay pinned server-side): their jobs get dropped, so source
  // snapshots switch to the entity's PNG (result thumbnails keep their URLs
  // — the server falls back to the pinned entity for dropped jobs)
  state.thumbs = state.thumbs.filter((t) => {
    if (!t.pinned) return false;
    if (t.source && t.image_id)
      t.url = `/api/canvas/image.png?id=${encodeURIComponent(t.image_id)}`;
    return true;
  });
  state.currentThumb = -1;
  renderThumbs();
  setStatus("queued…");
  saveUiState(); // persist the settings actually about to be used
  try {
    await forceSync(); // the server snapshots the canvas (source, inpaint)
    const r = await api("/api/generate", body);
    state.job = r.job_id;
    // A different server_id means the server restarted since our last
    // generate: job ids restart at 1, so the terminal events recorded
    // above belong to a dead server and must not veto this job.
    if (lastServerId !== r.server_id) {
      finishedJobs.clear();
      lastServerId = r.server_id;
    }
    if (r.done || finishedJobs.has(r.job_id)) {
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
// CFG/guidance, no negative prompt). Self attention guidance (SAG) has no
// main-form controls: its enable flag and options live in the model
// configuration (per-direction checkboxes + options dialog, SDXL entries
// only). The family comes from /api/models (state.entries), not a samplers
// fetch (which would race the model load).
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
function modelOptions(models) {
  // one option per configuration; default entries' names are model paths and
  // get path-grouped (file configurations have free-form names and stay
  // ungrouped unless the user put a slash in the name)
  return pathGroupOptions(models.map((x) => ({
    value: x.id,
    text: x.is_default ? `${x.name} (default)` : x.name,
  })));
}

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
  sel.setOptions(modelOptions(m.models));
  if (cur && m.models.some((x) => x.id === cur)) sel.value = cur;
  else if (m.current) sel.value = m.current;
}

function modeFieldset(title, section, showSag, openSagOptions, openSoftOptions, showSoft) {
  // section: {sampler, scheduler, cfg, sag, sag_scale, sag_blur_sigma,
  //           sag_threshold, soft, soft_mask_blend_power, ... (six soft
  //           options), loras: [{name, weight}]}
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
  (section.loras || []).forEach((l) =>
    loraList.append(loraRow(l.name, l.weight, undefined, true, false)));
  const rows = [
    el("div", { class: "row" },
      el("label", {}, "Sampler", s),
      el("label", {}, "Scheduler", sc)),
    el("div", { class: "row" },
      el("label", { title: "steps override; empty = built-in default" }, "Steps", steps),
      el("label", { title: "CFG override; empty = model default" }, "CFG", cfg)),
  ];
  // soft inpainting (Forge builtin extension): enable checkbox + options
  // gear, offered for EVERY family (all four implement it, unlike SAG) but
  // ONLY in the i2i section (showSoft) — soft inpainting needs an init
  // mask, so the t2i section has no soft controls (the section's soft keys
  // are inert there). The row sits directly under steps/CFG, aligned with
  // the SAG row below it (same label+gear structure). Enabled by default: a
  // section without the key (old config files, in-memory defaults) shows it
  // checked. The six option values live in hidden inputs: the options
  // dialog edits them and readCfgSection picks them up (cf. the SAG inputs)
  const softInputs = [];
  if (showSoft) {
    const soft = el("input", { class: "cfg-soft", type: "checkbox" });
    soft.checked = section.soft !== false;
    const softGear = el("button", { type: "button", class: "mini icon cfg-soft-cfg",
      title: "soft inpainting options" }, "⚙");
    softGear.onclick = () => openSoftOptions(fs);
    rows.push(el("div", { class: "row" },
      el("label", { class: "cfg-soft-row", title: "soft inpainting (Forge builtin extension): the protected content is blended with the inpainted content according to the mask opacity instead of being locked exactly; a high mask blur is recommended" },
        soft, "Soft inpainting"),
      softGear));
    for (const [cls, key, dflt] of [
      ["cfg-soft-mask-blend-power", "soft_mask_blend_power", 1],
      ["cfg-soft-mask-blend-scale", "soft_mask_blend_scale", 0.5],
      ["cfg-soft-detail-preservation", "soft_inpaint_detail_preservation", 4],
      ["cfg-soft-mask-influence", "soft_composite_mask_influence", 0],
      ["cfg-soft-diff-threshold", "soft_composite_difference_threshold", 0.5],
      ["cfg-soft-diff-contrast", "soft_composite_difference_contrast", 2],
    ]) {
      const i = el("input", { class: cls, type: "number", hidden: "" });
      i.value = section[key] ?? dflt;
      softInputs.push(i);
    }
  }
  // SAG is implemented for the SDXL UNet only; the checkbox row is omitted
  // (not shown disabled) for the other families. The option values live in
  // hidden inputs: the options dialog edits them and readCfgSection picks
  // them up (the dialog replaces this one, so they cannot stay in sliders)
  const sagInputs = [];
  if (showSag) {
    const sag = el("input", { class: "cfg-sag", type: "checkbox" });
    sag.checked = !!section.sag;
    const gear = el("button", { type: "button", class: "mini icon cfg-sag-cfg",
      title: "self attention guidance options" }, "⚙");
    gear.onclick = () => openSagOptions(fs);
    rows.push(el("div", { class: "row" },
      el("label", { class: "cfg-sag-row", title: "self attention guidance (Forge builtin extension): sharpens high-detail regions (faces, eyes, fine structures) by feeding the model's own self-attention back through an extra unconditional denoise (this direction, needs CFG > 1)" },
        sag, "Self attention guidance"),
      gear));
    for (const [cls, key, dflt] of [
      ["cfg-sag-scale", "sag_scale", 0.5],
      ["cfg-sag-blur-sigma", "sag_blur_sigma", 2],
      ["cfg-sag-threshold", "sag_threshold", 1],
    ]) {
      const i = el("input", { class: cls, type: "number", hidden: "" });
      i.value = section[key] ?? dflt;
      sagInputs.push(i);
    }
  }
  fs.append(...rows, ...sagInputs, ...softInputs,
    el("div", { class: "loras" }, loraList, loraAddButton(loraList, null, false)),
  );
  return fs;
}

function readCfgSection(fs) {
  // the .cfg-sag* inputs only exist for SDXL entries; undefined (dropped
  // by JSON.stringify) leaves the server-side defaults (false / Forge
  // option values). The .cfg-soft* inputs exist for every family.
  const num = (cls) => {
    const i = fs.querySelector(cls);
    return i ? parseFloat(i.value) : undefined;
  };
  const sag = fs.querySelector(".cfg-sag");
  const soft = fs.querySelector(".cfg-soft");
  return {
    sampler: fs.querySelector(".cfg-sampler").value || null,
    scheduler: fs.querySelector(".cfg-scheduler").value || null,
    steps: fs.querySelector(".cfg-steps").value === "" ? null : parseInt(fs.querySelector(".cfg-steps").value, 10),
    cfg: fs.querySelector(".cfg-cfg").value === "" ? null : parseFloat(fs.querySelector(".cfg-cfg").value),
    sag: sag ? sag.checked : undefined,
    sag_scale: num(".cfg-sag-scale"),
    sag_blur_sigma: num(".cfg-sag-blur-sigma"),
    sag_threshold: num(".cfg-sag-threshold"),
    // the soft controls exist in the i2i fieldset only; undefined (dropped
    // by JSON.stringify) in t2i leaves the server-side defaults (inert)
    soft: soft ? soft.checked : undefined,
    soft_mask_blend_power: num(".cfg-soft-mask-blend-power"),
    soft_mask_blend_scale: num(".cfg-soft-mask-blend-scale"),
    soft_inpaint_detail_preservation: num(".cfg-soft-detail-preservation"),
    soft_composite_mask_influence: num(".cfg-soft-mask-influence"),
    soft_composite_difference_threshold: num(".cfg-soft-diff-threshold"),
    soft_composite_difference_contrast: num(".cfg-soft-diff-contrast"),
    loras: readLoraList(fs.querySelector(".lora-list")).map(([name, weight]) => ({ name, weight })),
  };
}

function openConfigDialog(entry, prefill = null) {
  // prefill: unsaved dialog state ({name, distilled, t2i, i2i}) captured
  // before a nested dialog (the SAG options) replaced this one, so the
  // detour back to this dialog keeps every edit
  const nameIn = el("input", { class: "cfg-name", type: "text", maxlength: "120" });
  nameIn.value = prefill ? prefill.name : entry.name;
  const modelIn = el("input", { class: "cfg-model", type: "text", readonly: "" });
  modelIn.value = entry.model_id;
  // distilled model: Flux-only flag (the CFG value becomes the embedded
  // guidance); hidden for SDXL configurations, where it is inert.
  // entry.family comes from /api/models, so it is correct even if the
  // user opens the dialog before the model finished loading.
  const isFlux = entry.family === "flux";
  const isKrea2 = entry.family === "krea2";
  const distilledIn = el("input", { class: "cfg-distilled", type: "checkbox" });
  distilledIn.checked = prefill ? !!prefill.distilled : entry.distilled !== false;
  // turbo model: Krea 2-only flag (the turbo checkpoint class: 8 steps,
  // no CFG, pinned mu); hidden for every other family, where it is inert
  const turboIn = el("input", { class: "cfg-turbo", type: "checkbox" });
  turboIn.checked = prefill ? !!prefill.turbo : !!entry.turbo;
  // SAG is implemented for the SDXL UNet only: the per-direction checkbox
  // + options gear are offered for SDXL entries only. Soft inpainting has
  // no family gate (all four families implement it) but lives in the i2i
  // fieldset only (it needs an init mask; the inpaint modes read the i2i
  // section) — see modeFieldset's showSoft.
  const showSag = entry.family === "sdxl";
  // the SAG options dialog replaces this one (single #dialog element), so
  // the click handler snapshots the whole dialog and reopens it
  const snapCfg = () => ({
    name: nameIn.value,
    distilled: distilledIn.checked,
    turbo: turboIn.checked,
    t2i: readCfgSection(f1),
    i2i: readCfgSection(f2),
  });
  const openSagOptions = (fs) => {
    const key = fs === f1 ? "t2i" : "i2i";
    const cur = readCfgSection(fs);
    const snap = snapCfg();
    // on save AND on cancel the config dialog reopens with the captured
    // state, so unsaved edits survive the detour
    const reopen = () => openConfigDialog(entry, snap);
    openSagDialog({
      scale: cur.sag_scale, blur: cur.sag_blur_sigma, threshold: cur.sag_threshold,
    }, (v) => {
      snap[key] = {
        ...snap[key],
        sag_scale: v.scale, sag_blur_sigma: v.blur, sag_threshold: v.threshold,
      };
      reopen();
    }, reopen);
  };
  // the soft inpainting options dialog: same detour pattern as the SAG one
  const openSoftOptions = (fs) => {
    const key = fs === f1 ? "t2i" : "i2i";
    const cur = readCfgSection(fs);
    const snap = snapCfg();
    const reopen = () => openConfigDialog(entry, snap);
    openSoftDialog({
      power: cur.soft_mask_blend_power, scale: cur.soft_mask_blend_scale,
      detail: cur.soft_inpaint_detail_preservation,
      maskInf: cur.soft_composite_mask_influence,
      threshold: cur.soft_composite_difference_threshold,
      contrast: cur.soft_composite_difference_contrast,
    }, (v) => {
      snap[key] = {
        ...snap[key],
        soft_mask_blend_power: v.power, soft_mask_blend_scale: v.scale,
        soft_inpaint_detail_preservation: v.detail,
        soft_composite_mask_influence: v.maskInf,
        soft_composite_difference_threshold: v.threshold,
        soft_composite_difference_contrast: v.contrast,
      };
      reopen();
    }, reopen);
  };
  const f1 = modeFieldset("text to image", prefill?.t2i ?? entry.t2i, showSag,
                          openSagOptions, openSoftOptions, false);
  const f2 = modeFieldset("image to image / inpaint", prefill?.i2i ?? entry.i2i,
                          showSag, openSagOptions, openSoftOptions, true);
  const box = el("div", { class: "cfg-dialog" },
    el("label", {}, "Name", nameIn),
    el("label", { class: "muted" }, "Model", modelIn),
    ...(isFlux ? [el("label", { class: "cfg-distilled-row", title: "checked " +
        "(default): the CFG values below are the embedded guidance " +
        "(distilled CFG; one pass per step, the negative prompt is ignored). " +
        "unchecked: classic CFG with the negative prompt (two passes per " +
        "step, about 2x the time)." }, distilledIn, "Distilled model")] : []),
    ...(isKrea2 ? [el("label", { class: "cfg-distilled-row", title: "checked " +
        "(default): the raw checkpoint class (12 steps, standard CFG 1.5, " +
        "resolution-derived noise schedule; the negative prompt is used). " +
        "unchecked: the turbo class (8 steps, no CFG, pinned noise schedule; " +
        "faster, tuned for 8-step sampling)." }, turboIn, "Turbo model")] : []),
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
        turbo: isKrea2 ? turboIn.checked : entry.turbo,
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
    el("label", {}, "Name", nameIn)),
    async () => {
      const name = nameIn.value.trim();
      if (!name) { dialogError("name is required"); return false; }
      try {
        const entry = await api("/api/model-configs/create", { name, based_on: cur.id });
        $("model").value = entry.id;
        await refreshModels(true);
        scheduleSaveUiState(0);
        // go straight to the new configuration's edit dialog so the copy can
        // be tuned without opening it manually. Deferred: this dialog is
        // closed right after onOk returns, which would close the new one too.
        const created = state.entries.find((e) => e.id === entry.id);
        if (created) setTimeout(() => openConfigDialog(created), 0);
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
const PRESET_KEYS = ["width", "height", "rotation", "steps", "cfg", "seed", "batch",
  "sampler", "scheduler", "mode", "denoise", "mask_padding", "oversample", "loras"];
// enable/disable of the sidebar checkboxes (soft inpainting and SAG are not
// preset fields: their flags live in the model configuration)

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
    if (value) value.forEach((x) =>
      list.append(loraRow(x[0], x[1], undefined, true, false)));
    // same structure as the sidebar/config dialog: rows first, the "+ LoRA"
    // button after the list (added rows append to the list, above the button)
    return el("div", { class: "loras" }, list, loraAddButton(list, null, false));
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

// clone prefill: "<name> copy" (then "copy 2", "copy 3", ...) so saving the
// clone cannot silently overwrite the source preset
function uniquePresetName(base) {
  const names = new Set(state.presets.map((p) => p.name.toLowerCase()));
  if (!names.has(base.toLowerCase())) return base;
  for (let i = 2; ; i++) {
    const cand = `${base} ${i}`;
    if (!names.has(cand.toLowerCase())) return cand;
  }
}

// isClone: the fields come from `existing` but the dialog creates a NEW
// preset (no old_name, so the source is kept and the name is made unique)
function openPresetDialog(existing, onClose = null, isClone = false) {
  const p = existing || {};
  const controls = p.controls || {};
  const nameIn = el("input", { type: "text", maxlength: "120" });
  nameIn.value = isClone ? uniquePresetName(`${p.name} copy`) : (p.name || "");
  const groupIn = el("input", { type: "text", maxlength: "120",
    placeholder: "optional — presets in the same group are listed together" });
  groupIn.value = p.group || "";
  const defs = [
    ["width", "Width", "number", p.width],
    ["height", "Height", "number", p.height],
    ["rotation", "Rotation", "select", p.rotation],
    ["steps", "Steps", "number", p.steps],
    ["cfg", "CFG", "number", p.cfg],
    ["seed", "Seed", "number", p.seed],
    ["batch", "Batch size", "number", p.batch],
    ["sampler", "Sampler", "select", p.sampler],
    ["scheduler", "Scheduler", "select", p.scheduler],
    ["mode", "Mode", "select", p.mode],
    ["denoise", "Denoise strength", "number", p.denoise],
    ["mask_padding", "Mask padding", "number", p.mask_padding],
    ["oversample", "Scale", "number", p.oversample],
    ["loras", "LORAs", "loras", p.loras],
  ];
  const rows = el("div", { class: "preset-rows" });
  for (const [key, label, kind, value] of defs) {
    const cb = el("input", { type: "checkbox", class: "preset-cb", id: `preset-cb-${key}` });
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
      if (key === "rotation") [-180, -135, -90, -45, 0, 45, 90, 135, 180].forEach((d) => input.append(opt(d, d + "°")));
      input.value = value || "";
    }
    input.dataset.presetKey = key;
    input.dataset.presetKind = kind;
    rows.append(el("div", { class: "preset-row" }, cb, el("label", { class: "preset-label", for: `preset-cb-${key}` }, label), input));
  }
  const box = el("div", { class: "preset-dialog" },
    el("label", {}, "Name", nameIn),
    el("label", {}, "Group", groupIn),
    el("div", { class: "muted" }, "checked settings are applied when the preset is selected"),
    rows);
  openDialog(existing && !isClone ? "Edit preset" : "New preset", box, async () => {
    const name = nameIn.value.trim();
    if (!name) { dialogError("name is required"); return false; }
    const body = { name, group: groupIn.value.trim() || null, controls: {} };
    // editing an existing preset: the server renames it (deletes the old
    // file) when the name changed; creating a new preset omits old_name
    if (existing && !isClone) body.old_name = existing.name;
    for (const row of rows.children) {
      const input = row.querySelector("[data-preset-key]");
      const key = input.dataset.presetKey;
      body.controls[key] = row.querySelector(".preset-cb").checked;
      if (input.dataset.presetKind === "loras")
        body[key] = readLoraList(input).map(([n, w]) => [n, w]);
      else if (input.dataset.presetKind === "bool") body[key] = input.checked;
      else if (key === "rotation")
        body[key] = input.value === "" ? null : parseInt(input.value, 10);
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

let presetManageFilter = ""; // survives dialog round-trips (edit/new returns here)
function openPresetManage() {
  const list = el("div", { class: "preset-manage" });
  const newBtn = el("button", { class: "mini", type: "button" }, "+ New preset");
  let filter = presetManageFilter;
  const render = () => {
    list.querySelectorAll(".preset-manage-row, .muted").forEach((r) => r.remove());
    const shown = state.presets.filter((p) => !filter || p.name.toLowerCase().includes(filter));
    if (!state.presets.length) list.append(el("div", { class: "muted" }, "no presets yet"));
    else if (!shown.length) list.append(el("div", { class: "muted" }, "no matches"));
    shown.forEach((p) => {
      const name = p.name;
      const nameEl = el("span", { class: "preset-manage-name" }, name);
      const edit = el("button", { class: "mini icon", type: "button", title: "edit" }, "✎");
      const clone = el("button", { class: "mini icon", type: "button", title: "clone" }, "⧉");
      const del = el("button", { class: "mini icon", type: "button", title: "delete" }, "✗");
      // both edit and clone fetch the full preset, then reopen the dialog
      // (saving/cancelling returns to this management list)
      const openEditor = async (isClone) => {
        closeDialog();
        let preset;
        try {
          preset = (await (await fetch(`/api/presets/${encodeURIComponent(name)}`)).json()).preset;
        } catch (e) { setStatus(e.message, true); return; }
        openPresetDialog(preset, openPresetManage, isClone);
      };
      edit.onclick = () => openEditor(false);
      clone.onclick = () => openEditor(true);
      del.onclick = async () => {
        if (!confirm(`Delete preset "${name}"?`)) return;
        try {
          await api("/api/presets/delete", { name });
        } catch (e) { setStatus(e.message, true); return; }
        await loadPresets();
        render();
      };
      list.append(el("div", { class: "preset-manage-row" }, nameEl, edit, clone, del));
    });
  };
  render();
  newBtn.onclick = () => { closeDialog(); openPresetDialog(null, openPresetManage); };
  // the new-preset button stays below the list (outside the scrollable area)
  openDialog("Setting presets", el("div", { class: "preset-manage-wrap" }, list, newBtn), null, "Close",
    null, (q) => { filter = presetManageFilter = q; render(); }, presetManageFilter);
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
  // fetch each brush's pixel data once so the stroke preview can stamp
  // the exact shape (falls back to a disc if this is still in flight)
  brushNames.forEach((n) => loadBrushImage(n));
  syncToolbar();
}

// Brush size bounds; keep in sync with ToolSettings.size in uistate.py
const BRUSH_SIZE_MIN = 10;
const BRUSH_SIZE_MAX = 500;
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
  $("tool-copy").classList.toggle("active", Tools.current === "copy");
  $("tool-paste").classList.toggle("active", Tools.current === "paste");
  const cropMode = ["crop", "copy", "paste"].includes(Tools.current);
  // The colour swatch always edits the brush tool's remembered colour
  // (mask strokes ignore colour), so it stays live in mask mode too;
  // the clone and smudge tools have no colour at all, so disable it there.
  // Crop, copy and paste use no brush at all and disable all four controls.
  $("brush-color").disabled = ["clone", "smudge", "mask", "crop", "copy", "paste"].includes(Tools.current);
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

// CTRL+wheel (size) / SHIFT+wheel (opacity) adjustments, called from the
// canvas wheel handler (canvas.js): multiplicative, same 1.0015^delta
// curve as wheel zoom, clamped to the slider ranges
function wheelBrushSize(factor) {
  const st = Tools.get();
  const s = Math.min(BRUSH_SIZE_MAX, Math.max(BRUSH_SIZE_MIN,
                 Math.round(st.size * factor)));
  if (s === st.size) return;
  st.size = s;
  scheduleSaveUiState();
  syncToolbar();
  viewRedraw(); // cursor outline changes footprint
}
function wheelBrushOpacity(factor) {
  const st = Tools.get();
  // round to the slider's 0.01 step
  const o = Math.round(Math.min(1, Math.max(0.05, st.opacity * factor)) * 100) / 100;
  if (o === st.opacity) return;
  st.opacity = o;
  scheduleSaveUiState();
  syncToolbar();
}

function switchTool(t) {
  // switching to a DIFFERENT tool abandons an in-progress crop/copy rectangle
  // (re-arming the same tool itself, e.g. tapping P mid-drag, does not). A
  // floating selection SURVIVES a tool switch: the tools act on it
  if (view.crop && Tools.current !== t) abortCrop();
  Tools.switch(t);
  scheduleSaveUiState();
  syncToolbar();
  syncUndoButtons();
  viewRedraw(); // the cursor preview depends on the active tool
}

// ---------------- undo / redo (per-image draw + per-image mask) ---------
// One toolbar undo/redo pair follows the active tool: "mask" drives the
// client's mask stack, any other tool drives the draw stack (the client
// owns the stacks now; see canvas.js).
function syncUndoButtons() {
  const mask = Tools.current === "mask";
  const t = mask ? "mask" : "draw";
  $("btn-undo").textContent = `↩ Undo ${t}`;
  $("btn-redo").textContent = `↪ Redo ${t}`;
  $("btn-undo").disabled = (mask ? undoMask : undoDraw).length === 0;
  $("btn-redo").disabled = (mask ? redoMask : redoDraw).length === 0;
}

async function maskOp(op) {
  try {
    // client-side now (canvas.js): worker morphology / local clear
    if (op === "clear") await maskClear();
    else await maskMorph(op === "dilate");
  } catch (e) {
    setStatus(`mask ${op} failed: ${e.message}`, true);
  }
}

// ---------------- person mask generation (Phase 5 segmentation) ----------------
// models/segmentation/*.tflite -> /api/segmentation/models. The dialog picks
// the model, the classes (from the model's own labels, background excluded),
// the confidence threshold and the edge feather; the result is unioned into
// the mask buffer (one undo step).
function openSegDialog() {
  if (!state.segModels.length) {
    openDialog(
      "Generate person mask",
      el(
        "div",
        { class: "muted" },
        "no segmentation models found — put a .tflite (e.g. " +
          "selfie_multiclass_256x256.tflite) into models/segmentation/ and reload"
      ),
      null,
      "Close"
    );
    return;
  }

  const modelSel = el("select", { class: "seg-model" });
  state.segModels.forEach((m) => modelSel.append(opt(m.id, m.name)));
  modelSel.value = state.segModels.some((m) => m.id === state.segModel)
    ? state.segModel
    : state.segModels[0].id;

  // The class checkboxes follow the selected model's labels, so they are
  // rebuilt whenever the model changes.
  const classWrap = el("div", { class: "seg-classes" });
  let rows = [];
  const renderClasses = () => {
    const model = state.segModels.find((m) => m.id === modelSel.value);
    const labels = model.labels.filter((l) => l !== "background");
    const prev = state.segClasses.length ? state.segClasses : labels;
    classWrap.innerHTML = "";
    rows = labels.map((label) => {
      const cb = el("input", { type: "checkbox" });
      cb.checked = prev.includes(label);
      const row = el("label", { class: "seg-class" }, cb, label);
      classWrap.append(row);
      return row;
    });
  };
  modelSel.onchange = renderClasses;
  renderClasses();

  const threshold = el("input", {
    class: "seg-threshold", type: "number", min: "0", max: "1", step: "0.05",
    title: "segmenter confidence cut (0–1): at or below it a pixel stays unmasked, above it the mask is fully opaque",
  });
  threshold.value = state.segThreshold;
  const feather = el("input", {
    class: "seg-feather", type: "number", min: "0", max: "64", step: "1",
    title: "feather the mask edge by this many pixels (0 = hard edge; the opaque core stays opaque)",
  });
  feather.value = state.segFeather;

  const body = el("div", { class: "seg-dialog" },
    el("label", {}, "Model", modelSel),
    el("div", { class: "muted" },
      `classes to add to the mask (union with the current mask, undoable — clear it first to replace):`),
    classWrap,
    el("div", { class: "row" },
      el("label", {}, "Threshold", threshold),
      el("label", {}, "Feather (px)", feather)),
  );
  openDialog("Generate person mask", body, async () => {
    const classes = rows.filter((r) => r.querySelector("input").checked)
      .map((r) => r.textContent.trim());
    if (!classes.length) return false; // keep the dialog open
    const thr = parseFloat(threshold.value);
    const fea = parseFloat(feather.value);
    if (!Number.isFinite(thr) || thr < 0 || thr > 1) {
      dialogError("threshold must be between 0 and 1");
      return false;
    }
    if (!Number.isFinite(fea) || fea < 0 || fea > 64) {
      dialogError("feather must be between 0 and 64 pixels");
      return false;
    }
    try {
      await forceSync(); // the server unions with ITS copy of the mask
      const snap = view.maskValues ? await snapCanvas(view.maskValues) : null;
      const r = await api("/api/segmentation/mask", {
        model: modelSel.value, classes, threshold: thr, feather: fea,
      });
      if (r.changed && snap) stashMaskUndo(snap); // undoable across the refresh
      state.segModel = modelSel.value;
      state.segClasses = classes;
      state.segThreshold = thr;
      state.segFeather = fea;
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
// when not inpainting"); mask padding only to "inpaint masked". Soft
// inpainting has no main-form control: its enable flag and options live in
// the model configuration (i2i section, all families).
function syncMode() {
  const gen = $("mode").value === "generate";
  $("denoise").disabled = gen;
  $("mask-padding").disabled = $("mode").value !== "inpaint_masked";
  $("oversample").disabled = gen;
  syncGenerateEnabled();
  viewRedraw(); // mask overlay is mask-mode-only (visual only), bbox stays
}

// Inpaint with an empty mask is a server-side no-op (rejected with
// "mask is empty"), so keep the button off until a mask exists. view.maskBbox
// is null exactly when the mask is empty; it is recomputed by canvas.js in
// buildMaskOverlay() on every mask change, which calls back here.
function syncGenerateEnabled() {
  $("btn-generate").disabled =
    !state.generating &&
    $("mode").value === "inpaint_masked" &&
    !view.maskBbox;
}

// ---------------- soft inpainting (Forge builtin extension) ----------------

// The six Forge sliders: [state key, label, min, max, step, help]. The
// first group is the per-step latent blend, the second the pixel composite.
const SOFT_LATENT_OPTS = [
  ["power", "Schedule bias", 0, 8, 0.1,
   "Controls when the original image is preserved during generation. 1 is balanced; below 1 it is preserved more at the end, above 1 more at the start."],
  ["scale", "Preservation strength", 0, 8, 0.05,
   "Biases partially masked areas toward the original image or the new content. Low favours the new content, high the original. 0 ignores the mask entirely."],
  ["detail", "Transition contrast boost", 1, 32, 0.5,
   "Controls the contrast where original and new content meet. Low blends softly and can look flat; high keeps contrast but can over-sharpen the transition."],
];
const SOFT_PIXEL_OPTS = [
  ["maskInf", "Mask influence", 0, 1, 0.05,
   "How much the mask guides where original pixels are kept. 0 lets the image content decide alone, 1 follows the mask closely."],
  ["threshold", "Difference threshold", 0, 8, 0.25,
   "How far the new content may deviate from the original. Low: original pixels survive only when the content is nearly identical. High: they survive even clear differences."],
  ["contrast", "Difference contrast", 0, 8, 0.25,
   "Sharpness of the blend between original and new content. Low fades gradually (can ghost); high switches abruptly (no ghosting, but seams can show)."],
];

// The soft inpainting options dialog (from the model configuration's
// per-direction gear). ``vals`` = {power, scale, detail, maskInf, threshold,
// contrast} to initialize the sliders; onApply receives the edited values;
// onClose (cancel/escape/backdrop) fires so the caller can restore the
// dialog it replaced (cf. openSagDialog).
function openSoftDialog(vals, onApply, onClose = null) {
  const fields = [];
  const make = ([key, label, min, max, step, help]) => {
    const input = el("input", { type: "range", min, max, step });
    input.value = vals[key];
    const val = el("b", {}, String(vals[key]));
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
      "Soft inpainting blends the original image with the inpainted area according to the mask opacity instead of locking the masked region exactly. A high mask blur is recommended."),
    ...SOFT_LATENT_OPTS.map(make),
    el("div", { class: "soft-section" }, "Pixel composite settings"),
    ...SOFT_PIXEL_OPTS.map(make),
  ), () => {
    const v = {};
    fields.forEach(({ key, input }) => { v[key] = parseFloat(input.value); });
    onApply(v);
    // the caller replaces this dialog (reopening the config dialog): skip
    // the closeDialog() this handler would run after onApply returned
    return false;
  }, "Save", onClose);
}

// The generate body carries no soft inpainting fields: the enabled flag AND
// the option values are model-configuration settings (i2i section only —
// soft inpainting needs an init mask — every family, enabled by default),
// resolved server-side from the selected
// configuration's section. The server builds the request object from the
// configuration, so the inpaint PNG metadata carries Forge's "Soft
// inpainting ..." lines unconditionally (like the extension).

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

// The SAG options dialog (from the model configuration's per-direction
// gear). ``vals`` = {scale, blur, threshold} to initialize the sliders;
// onApply receives the edited values; onClose (cancel/escape/backdrop)
// fires so the caller can restore the dialog it replaced.
function openSagDialog(vals, onApply, onClose = null) {
  const fields = [];
  const make = ([key, label, min, max, step, help]) => {
    const input = el("input", { type: "range", min, max, step });
    input.value = vals[key];
    const val = el("b", {}, String(vals[key]));
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
    const v = {};
    fields.forEach(({ key, input }) => { v[key] = parseFloat(input.value); });
    onApply(v);
    // the caller replaces this dialog (reopening the config dialog): skip
    // the closeDialog() this handler would run after onApply returned
    return false;
  }, "Save", onClose);
}

// The generate body carries no SAG fields: the enabled flag AND the option
// values are model-configuration settings (per direction, SDXL only),
// resolved server-side. The server builds the request object from the
// selected configuration, so the PNG metadata carries the Forge sag_* keys
// unconditionally (like the extension).

function wireToolbar() {
  $("tool-brush").onclick = () => switchTool("brush");
  $("tool-mask").onclick = () => switchTool("mask");
  $("tool-clone").onclick = () => switchTool("clone");
  $("tool-smudge").onclick = () => switchTool("smudge");
  $("tool-crop").onclick = () => switchTool("crop");
  $("tool-copy").onclick = () => switchTool("copy");
  // The button stays clickable without a copy so it can say why nothing
  // happens (the E hotkey still arms the tool unconditionally).
  $("tool-paste").onclick = () => {
    if (!view.clip) {
      setStatus("Copy a region first");
      return;
    }
    switchTool("paste");
  };
  $("sel-ok").onclick = () => commitSelection();
  $("sel-cancel").onclick = () => cancelSelection();
  $("sel-flip-h").onclick = () => flipSel(true);
  $("sel-flip-v").onclick = () => flipSel(false);
  $("brush").onchange = (e) => {
    Tools.get().brush = e.target.value;
    scheduleSaveUiState();
    viewRedraw(); // cursor outline changes shape
  };
  $("brush-color").oninput = (e) => {
    Tools.state.brush.color = e.target.value;
    scheduleSaveUiState();
  };
  $("brush-size").oninput = (e) => {
    // round to whole px: size is an absolute pixel size
    Tools.get().size = Math.round(sizeFromSlider(parseFloat(e.target.value)));
    scheduleSaveUiState();
    syncToolbar();
    viewRedraw(); // cursor outline changes footprint
  };
  $("brush-screen").onchange = (e) => {
    Tools.get().screen = e.target.checked;
    scheduleSaveUiState();
    if (view.has) viewRedraw(); // cursor outline changes footprint
  };
  $("brush-opacity").oninput = (e) => {
    Tools.get().opacity = parseFloat(e.target.value);
    scheduleSaveUiState();
    syncToolbar();
  };
  $("mode").onchange = () => { onUserFieldChange("mode"); syncMode(); }; // mode is not persisted
  $("mask-dilate").onclick = () => maskOp("dilate");
  $("mask-erode").onclick = () => maskOp("erode");
  $("mask-clear").onclick = () => maskOp("clear");
  $("seg-run").onclick = () => openSegDialog();
  $("rotation").onchange = () => { onUserFieldChange("rotation"); scheduleSaveUiState(0); };
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
    const row = loraRow("", 0.8, mainLoraChange);
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
    await up.json();
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
  // the source rotation round-trips through the PNG (absent = leave as-is)
  if (meta.rotation !== undefined && meta.rotation !== null)
    $("rotation").value = String(meta.rotation);
  for (const id of ["sampler", "scheduler"]) {
    const sel = $(id);
    if (meta[id] && [...sel.options].some((o) => o.value === meta[id])) sel.value = meta[id];
  }
  // The configuration the import ends up selecting: the server resolves it
  // into meta.config_id (imagagen PNGs record the configuration NAME; for
  // A1111/Forge PNGs it scores the configurations of the image's checkpoint
  // by LoRA match, so "checkpoint + DMD2@0.7" wins over a bare-checkpoint
  // one). Without a config_id (no model in the metadata, server-side match
  // failed), fall back to a plain model name/id match, else keep the current
  // configuration. That configuration's LoRAs were baked into the image at
  // generation time (always prepended to the UI selection), so they are
  // dropped from the metadata list before the remaining LoRAs are applied --
  // otherwise they would load twice (once via the config, once via the UI).
  const matchedEntry = meta.config_id
    ? state.entries.find((e) => e.id === meta.config_id) || null
    : meta.model
      ? state.entries.find((e) => e.name === meta.model || e.id === meta.model) ||
        null
      : null;
  const targetEntry = matchedEntry || currentEntry();
  if (meta.loras?.length || meta.disabled_loras?.length) {
    let loras = parseMetaLoras(meta.loras || []);
    // the Disabled LoRAs metadata line: restored as disabled rows
    let off = parseMetaLoras(meta.disabled_loras || [])
      .map(([n, w]) => [n, w, false]);
    if (targetEntry) {
      // match on name AND weight (a user may have added a config LoRA at
      // another weight in the UI; only the config-provided copy is removed)
      const configLoras = new Set(
        [...(targetEntry.t2i?.loras || []), ...(targetEntry.i2i?.loras || [])]
          .map((l) => `${l.name}@${l.weight}`)
      );
      loras = loras.filter(([name, weight]) => !configLoras.has(`${name}@${weight}`));
      off = off.filter(([name, weight]) => !configLoras.has(`${name}@${weight}`));
    }
    renderMainLoras([...loras, ...off]);
  }
  // styles travel as name + actual contents; the server registers missing
  // ones as in-memory "Session" temporaries (not written to styles.csv)
  if (meta.styles?.length) {
    // the enabled flag (default: enabled) restores each row's on/off state
    renderMainStyles(meta.styles.map((s) => [s.name, s.enabled !== false]));
    try {
      const r = await api("/api/styles/temp", {
        styles: meta.styles.map((s) => ({
          name: s.name,
          prompt: s.prompt || "",
          negative_prompt: s.negative_prompt || "",
        })),
      });
      state.styles = r.styles;
      renderMainStyles(getAllStyles()); // options now include the temporaries
    } catch (e) {
      console.error("styles/temp:", e);
    }
  }
  // soft inpainting (Forge extra parameters) round-trips through the PNG,
  // but is NOT imported: the enabled flag and the option values are
  // model-configuration settings (i2i section, every family), not
  // importable UI fields (cf. the SAG block below)
  // self attention guidance (Forge SAG extra parameters) round-trips
  // through the PNG under the extension's raw snake_case keys, but is NOT
  // imported: the enabled flag and the option values are model-configuration
  // settings (per direction, SDXL only), not importable UI fields
  // switch to the image's model (see matchedEntry above)
  if (matchedEntry) {
    $("model").value = matchedEntry.id;
    selectModel(matchedEntry.id);
    // same follow-up work as the model dropdown's onchange: the sampler
    // lists, family-specific controls and the LoRA list all follow the
    // model's family (an import can land on another family's model)
    refreshSamplers();
    applyFamilyControls();
    refreshLoras();
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
// applying metadata is what dropping on the prompt fields is for. The
// upload becomes a canvas entity, so it also gets a thumbnail (selected,
// like the crop is): restorable via the strip after a generation puts a
// result on the canvas, and pinnable like any other thumbnail. Dropping
// onto an upload thumbnail REPLACES it (the canvas only ever holds one
// upload entity: the server evicts the superseded one, so its thumbnail
// would go dead), while any other canvas content (a result, a crop, a
// source snapshot) keeps its thumbnail and gets a new one appended.
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
      const { image_id } = await up.json();
      const url = `/api/canvas/image.png?id=${encodeURIComponent(image_id)}`;
      const last = state.thumbs[state.thumbs.length - 1];
      if (last?.upload) {
        last.image_id = image_id; // re-point it at the new entity
        last.url = url;
      } else {
        state.thumbs.push({ image_id, upload: true, url });
      }
      state.currentThumb = state.thumbs.length - 1; // the canvas shows it
      renderThumbs();
      viewRefresh();
      setStatus(`loaded ${file.name} onto the canvas`);
    } catch (err) {
      setStatus(`canvas drop failed: ${err.message}`, true);
    }
  });
}

// The thumbnail strip is a dropzone too: dropping a PNG there registers it as
// a PINNED canvas entity (a permanent thumbnail) without touching the canvas
// — the strip's own "Pin thumbnail" checkbox semantics, applied at import
// time. The strip stays a drop target while empty (no `hidden`, just a thin
// dashed placeholder), so the first thumbnail can be dropped in too.
function wireThumbsDrop() {
  const strip = $("thumbs");
  strip.addEventListener("dragover", (e) => {
    if (!(e.dataTransfer?.types || []).includes("Files")) return;
    e.preventDefault();
    strip.classList.add("drop");
  });
  strip.addEventListener("dragleave", () => strip.classList.remove("drop"));
  strip.addEventListener("drop", async (e) => {
    e.preventDefault();
    strip.classList.remove("drop");
    const files = [...(e.dataTransfer?.files || [])];
    const file = files.find((f) => f.type === "image/png" || /\.png$/i.test(f.name));
    if (!file) {
      setStatus("drop a PNG file");
      return;
    }
    try {
      const r = await fetch("/api/canvas/import-pin", { method: "POST", body: await file.arrayBuffer() });
      if (!r.ok) throw new Error(`import-pin: ${r.status}`);
      const { image_id } = await r.json();
      // pinned from the start; NOT selected — the canvas still shows its own
      // image, so the current thumbnail highlight is cleared
      state.thumbs.push({ image_id, upload: true, pinned: true,
                          url: `/api/canvas/image.png?id=${encodeURIComponent(image_id)}` });
      state.currentThumb = -1;
      renderThumbs();
      setStatus(`added ${file.name} as a pinned thumbnail`);
    } catch (err) {
      setStatus(`thumbnail drop failed: ${err.message}`, true);
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
    rotation: parseInt($("rotation").value, 10),
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
    oversample: parseFloat($("oversample").value) || 1,
    // soft inpainting (enable flag + options) and self attention guidance
    // are model configuration settings, not persisted here
    loras: getLoras().filter((x) => x[2]).map(([n, w]) => [n, w]),
    disabled_loras: getLoras().filter((x) => !x[2]).map(([n, w]) => [n, w]),
    styles: getStyles(),
    disabled_styles: getAllStyles().filter((x) => !x[1]).map((x) => x[0]),
    // preset is not persisted either: the dropdown always starts at "—"
    seg_model: state.segModel,
    seg_classes: state.segClasses,
    seg_threshold: state.segThreshold,
    seg_feather: state.segFeather,
    // the checked reference thumbnail (the server drops it when the
    // entity no longer exists)
    reference_image_id: state.referenceImageId || null,
    reference_sections: Array.isArray(state.referenceSections)
      ? state.referenceSections : null,
    // canvas tools: current tool + per-tool remembered settings
    tools: { current: Tools.current, ...Tools.state },
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
    ["rotation", "rotation"],
    ["sampler", "sampler"],
    ["scheduler", "scheduler"],
    ["denoise", "denoise"],
    ["mask-padding", "mask_padding"],
    ["oversample", "oversample"],
  ];
  for (const [elm, key] of fields) {
    if (st[key] !== undefined && st[key] !== null) $(elm).value = st[key];
  }
  renderMainLoras([
    ...(st.loras || []).map((x) => [x[0], x[1], true]),
    ...(st.disabled_loras || []).map((x) => [x[0], x[1], false]),
  ]);
  renderMainStyles([
    ...(st.styles || []).map((n) => [n, true]),
    ...(st.disabled_styles || []).map((n) => [n, false]),
  ]);
  // display-only restore: the model itself is NOT preloaded (it loads
  // lazily on the next generation, matching the registry design)
  if (st.model) $("model").value = st.model;
  // person mask generation: model, class selection, threshold and feather
  // all live in JS state (the dialog reads them, the save writes them back)
  if (Array.isArray(st.seg_classes)) state.segClasses = st.seg_classes;
  if (typeof st.seg_model === "string") state.segModel = st.seg_model;
  if (Number.isFinite(st.seg_threshold)) state.segThreshold = st.seg_threshold;
  if (Number.isFinite(st.seg_feather)) state.segFeather = st.seg_feather;
  // reference image: the checked thumbnail id (restoreQueueThumbs renders
  // the strip afterwards; renderThumbs re-checks the matching thumb)
  if (st.reference_image_id) state.referenceImageId = st.reference_image_id;
  if (Array.isArray(st.reference_sections))
    state.referenceSections = st.reference_sections;
  if (state.referenceImageId) fetchReferencePreview(); // renders the tag row
  // soft inpainting and self attention guidance (enable flags + options)
  // are model configuration settings, not persisted in the ui-state
  // sidebar width set by dragging its right border
  if (typeof st.sidebar_width === "number") $("controls").style.width = `${st.sidebar_width}px`;
  // canvas tools: current tool + per-tool remembered settings
  if (st.tools) {
    for (const [t, saved] of Object.entries(st.tools)) {
      if (Tools.state[t] && saved && typeof saved === "object") {
        for (const [k, v] of Object.entries(saved))
          if (k in Tools.state[t]) Tools.state[t][k] = v;
      }
    }
    if (TOOLS.includes(st.tools.current)) Tools.current = st.tools.current;
    syncToolbar();
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
  let q = { jobs: [] };
  try {
    [m, s, u, pr, lr, sy, sm, q] = await Promise.all([
      fetch("/api/models").then((r) => r.json()),
      fetch("/api/samplers").then((r) => r.json()),
      fetch("/api/ui-state").then((r) => (r.ok ? r.json() : null)),
      fetch("/api/presets").then((r) => r.json()),
      fetch("/api/loras").then((r) => r.json()),
      fetch("/api/styles").then((r) => (r.ok ? r.json() : { styles: [] })),
      fetch("/api/segmentation/models").then((r) => (r.ok ? r.json() : { models: [] })),
      fetch("/api/queue").then((r) => (r.ok ? r.json() : null)),
    ]);
  } catch (e) {
    console.error("init:", e);
    return;
  }
  state.segModels = sm.models;
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
  sel.setOptions(modelOptions(m.models));
  if (m.current && m.models.some((x) => x.id === m.current)) sel.value = m.current;
  for (const id of ["sampler", "scheduler"]) {
    const s2 = $(id);
    s2.innerHTML = "";
    s2.append(opt("", "Model default"));
    (id === "sampler" ? s.samplers : s.schedulers).forEach((n) => s2.append(opt(n, n)));
  }
  fillPresetSelect($("preset"));
  applyUiState(u && u.state);
  restoreQueueThumbs(q); // thumbnails are not persisted; pull the server's jobs
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
// the server_id reported by the last /api/generate response (null until
// the first generate); every server restart changes it
let lastServerId = null;
// true while the status line shows an error: "done" must not wipe it
let statusIsError = false;
let ws = null;
// escalating delay between reconnect attempts; reset on a successful open
let wsRetryDelay = 1000;
function connectWS() {
  ws = new WebSocket(`ws://${location.host}/api/ws`);
  ws.onopen = () => { wsRetryDelay = 1000; };
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
      case "phase":
        // "Encoding prompt" / "Inpainting @ 1024x1280" goes to the STATUS
        // line: the progress label belongs to the step counter. An error
        // that surfaced mid-batch keeps the line (a later phase event from
        // another task of the same batch must not hide it).
        if (!statusIsError) setStatus(msg.message);
        break;
      case "source_image":
        // canvas snapshot at submit time (inpaint, or generate on a
        // non-empty canvas), shown before any result arrives; the id lets
        // it be restored as a live canvas entity
        state.thumbs.unshift({ source: true, url: msg.url, image_id: msg.image_id,
                            seed: msg.seed }); // null: no seed in its metadata
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
        // the canvas auto-set to a fresh result entity: viewRefresh resets
        // the local history stacks with the new state
        viewRefresh();
        break;
      case "done":
        finishedJobs.add(msg.job);
        setGenerating(false);
        // the batch summary ("Completed 4 images in 32.4s") replaces the
        // cleared status line; an error that ended the batch stays shown
        if (!statusIsError) setStatus(msg.message || "");
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
  ws.onclose = () => {
    // Back off (1s -> 2s -> ... -> 10s cap) while the server is down: a
    // steady 1/s retry loop piles up half-dead connection attempts that
    // the browser queues behind its per-origin connection limit, which
    // can delay the next successful connect by tens of seconds.
    setTimeout(connectWS, wsRetryDelay);
    wsRetryDelay = Math.min(wsRetryDelay * 2, 10000);
  };
}

// ---------------- events ----------------
$("btn-generate").onclick = generate;
$("btn-save").onclick = async () => {
  try {
    await forceSync(); // the server saves ITS copy
    const r = await api("/api/save", { prompt: $("prompt").value });
    setStatus(`saved ${r.path}`);
  } catch (e) {
    setStatus(`save failed: ${e.message}`, true);
  }
};
// the server is best-effort caught up before the tab goes: the 2 s idle
// sync is the real safety net (the browser often kills large requests on
// exit, and keepalive POSTs are size-limited)
window.addEventListener("beforeunload", () => {
  if (!canvasDirty || !view.has) return;
  Promise.all([blobP(view.base), blobP(view.maskValues)])
    .then(([img, mask]) => {
      const fd = new FormData();
      fd.append("image", img, "canvas.png");
      fd.append("mask", mask, "mask.png");
      fetch("/api/canvas/sync", { method: "POST", body: fd, keepalive: true });
    });
});
function currentUndoTarget() {
  return Tools.current === "mask" ? "mask" : "draw";
}
$("btn-undo").onclick = async () => {
  const t = currentUndoTarget();
  try {
    await undoStep(t); // canvas.js: local stack, no round trip
  } catch (e) {
    setStatus(`undo failed: ${e.message}`, true);
  }
};
$("btn-redo").onclick = async () => {
  const t = currentUndoTarget();
  try {
    await redoStep(t); // canvas.js: local stack, no round trip
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

// ---------------- ESC clears the field focus ----------------
// The canvas hotkeys are live only while no form field has focus (the
// handlers below check activeElement), so ESC on a main-form field (prompt,
// seed, combobox, slider, ...) clears that focus and the canvas is live
// again without having to click it. Runs in the CAPTURE phase, i.e. before
// the field's own keydown handlers: a combobox whose list is OPEN swallows
// the FIRST ESC (it only closes the list), the next one blurs. Fields inside
// a dialog are left to the dialog's own ESC handling (ESC closes it).
// escBlurFlag marks the event whose focus the blur cleared, so the hotkey
// handlers below -- which after the blur see "not typing" -- do not also
// abort an in-flight crop rectangle or discard the floating selection.
let escBlurFlag = false;
document.addEventListener("keydown", (e) => {
  escBlurFlag = false;
  if (e.key !== "Escape") return;
  const el = document.activeElement;
  if (!el || !/INPUT|TEXTAREA|SELECT/.test(el.tagName)) return;
  if (el.closest("#dialog")) return;
  const box = el.closest(".combobox");
  if (box && box.querySelector(".combo-list.open")) return;
  el.blur();
  escBlurFlag = true;
}, true);

// D / S / C / M / P / Y / E switch tools (draw, smudge, clone, mask, crop,
// copy, paste), N / K / R act on the mask, F fits the image, CTRL+S saves,
// CTRL+G generates and CTRL+Z / CTRL+Y undo / redo; ENTER applies a floating
// selection and ESC aborts an in-flight rectangle or discards the selection
// (all when not typing in a field; ESC on a focused field only clears that
// focus -- see the ESC handler above);
// SHIFT (clone tool, a source set) makes the source follow the cursor
// until it is released
document.addEventListener("keydown", (e) => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "");
  if (typing) return;
  if (e.key === "Escape" && escBlurFlag) return; // ESC only cleared a focus
  if (e.ctrlKey && (e.key === "z" || e.key === "Z")) {
    e.preventDefault();
    $("btn-undo").onclick();
    return;
  }
  if (e.ctrlKey && (e.key === "y" || e.key === "Y")) {
    e.preventDefault();
    $("btn-redo").onclick();
    return;
  }
  if (e.ctrlKey && !e.shiftKey && (e.key === "s" || e.key === "S")) {
    e.preventDefault(); // keep the browser's save dialog from popping up
    $("btn-save").onclick();
    return;
  }
  if (e.ctrlKey && !e.shiftKey && (e.key === "g" || e.key === "G")) {
    e.preventDefault();
    $("btn-generate").onclick();
    return;
  }
  // the plain-letter hotkeys are unmodified: anything else with CTRL / META /
  // ALT belongs to the browser (CTRL+Z/Y/S/G were handled above)
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "d" || e.key === "D") switchTool("brush");
  else if (e.key === "m" || e.key === "M") switchTool("mask");
  else if (e.key === "c" || e.key === "C") switchTool("clone");
  else if (e.key === "s" || e.key === "S") switchTool("smudge");
  else if (e.key === "p" || e.key === "P") switchTool("crop");
  else if (e.key === "y" || e.key === "Y") switchTool("copy");
  else if (e.key === "e" || e.key === "E") switchTool("paste");
  else if (e.key === "n" || e.key === "N") $("mask-dilate").onclick();
  else if (e.key === "k" || e.key === "K") $("mask-erode").onclick();
  else if (e.key === "r" || e.key === "R") $("mask-clear").onclick();
  else if (e.key === "f" || e.key === "F") viewFit();
  // Esc aborts an in-flight rectangle first, else discards the floating
  // selection; Enter applies it (both only while no dialog is open)
  else if (e.key === "Escape" && view.crop) abortCrop();
  else if (e.key === "Escape" && view.sel && $("dialog").hidden) cancelSelection();
  else if (e.key === "Enter" && view.sel && $("dialog").hidden) commitSelection();
  else if (e.key === "Shift") cloneGestureStart();
});
document.addEventListener("keyup", (e) => {
  if (e.key === "Shift") cloneGestureEnd();
});

// Highlight the canvas border with the focus color while the tool hotkeys
// are live — i.e. no input/select/textarea has focus (the exact condition
// the hotkey handlers above check). The hover preview overlay mimics the
// canvas border, so it gets the same class.
const syncHotkeyBorder = () => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "");
  for (const id of ["canvas", "thumb-preview"])
    $(id).classList.toggle("hotkeys", !typing);
};
// focusout fires before activeElement moves to the next field, so re-check
// after the focus machinery settled; debounce 100ms so combo-box selections
// (which blur/refocus the input mid-pick) don't flash the border
let hotkeyBorderTimer = 0;
const applyHotkeyBorder = () => {
  const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "");
  for (const id of ["canvas", "thumb-preview"])
    $(id).classList.toggle("hotkeys", !typing);
};
for (const ev of ["focusin", "focusout"])
  document.addEventListener(ev, () => {
    clearTimeout(hotkeyBorderTimer);
    hotkeyBorderTimer = setTimeout(applyHotkeyBorder, 100);
  });
applyHotkeyBorder();

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
syncToolbar();
syncUndoButtons();
wireToolbar();
wireConfig();
wirePromptDrop();
wireCanvasDrop();
wireThumbsDrop();
renderThumbs(); // shows the empty strip as a dropzone (boot: index.html has it hidden)
// the loading overlay covers the UI until it is settled: every request
// whose response rewrites the form (init), plus the brush list and the
// canvas. The websocket is deliberately NOT part of this gate: its
// upgrade can lag the HTTP requests well behind (the browser throttles
// queued reconnect attempts after the server was offline), and it is not
// needed for the state load — job events simply arrive whenever the
// connection comes up. While it is up, pointer input hits the overlay and
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
Promise.all([pInit, pBrushes, pCanvas])
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

["prompt", "negative", "width", "height", "steps", "cfg", "seed", "batch", "denoise", "mask-padding", "oversample"].forEach((id) => {
  $(id).addEventListener("input", () => { onUserFieldChange(id); scheduleSaveUiState(); });
});
$("prompt").addEventListener("input", schedulePromptTokens);
// the mask bbox outline is grown by this field: redraw as it changes
$("mask-padding").addEventListener("input", viewRedraw);
$("seed-reset").onclick = () => {
  // -1 = the form's "new random seed for every image" state
  $("seed").value = "-1";
  onUserFieldChange("seed");
  scheduleSaveUiState();
};
$("seed-from-image").onclick = () => {
  const t = state.thumbs[state.currentThumb];
  // no image selected, or the selected image has no known seed
  // (e.g. a plain upload without parameters metadata)
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
