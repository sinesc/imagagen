/* imagagen UI primitives: the searchable combobox and the generic modal
   dialog, shared by app.js. Loaded before app.js in index.html; the DOM
   helpers defined here ($, el) are also used by canvas.js / tools.js. */
"use strict";

const $ = (id) => document.getElementById(id);

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
  const titleFor = (v) => {
    const o = curOpts().find((x) => x.value === v);
    return (o && o.title) || "";
  };

  function render(filter = "") {
    const f = filter.trim().toLowerCase();
    list.replaceChildren();
    active = -1;
    let lastGroup = null;
    let any = false;
    for (const o of curOpts()) {
      // match the item text OR its group header (for path-grouped options the
      // directory became the header, so typing part of the path still finds it)
      if (f && !String(o.text).toLowerCase().includes(f) &&
          !(o.group && String(o.group).toLowerCase().includes(f))) continue;
      if (o.group && o.group !== lastGroup) {
        list.append(el("li", { class: "combo-group" }, o.group));
        lastGroup = o.group;
      }
      const li = el("li", { class: "combo-item", role: "option", title: o.title }, o.text);
      li.dataset.value = o.value;
      // Pick on mousedown (like a native select), not click: the mousedown
      // blurs the input, and the blur handler closes the list after 120 ms.
      // On a slow press (>120 ms) the list hides before the trailing click
      // lands, so a click-based pick never fires and the old value remains.
      li.onmousedown = (e) => { if (e.button === 0) pick(o.value, true); };
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
    input.title = v ? titleFor(v) : "";
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
      input.title = value ? titleFor(value) : "";
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
  input.title = value ? titleFor(value) : "";
  render();
  return root;
}

// The combobox equivalent of an <optgroup> for path-like names: options whose
// text is a relative path get grouped by its directory ("g/s/name" -> group
// "g/s", item "name"), so the closed field shows the bare name and the list
// shows the directory as a section header. Texts without a directory stay in
// the (first) ungrouped section. Order: ungrouped first, groups by
// localeCompare, items by localeCompare within a group (same convention as
// the style/preset dropdowns).
function pathGroupOptions(options) {
  const ungrouped = [];
  const groups = new Map(); // directory -> options
  for (const o of options) {
    const i = o.text.lastIndexOf("/");
    if (i <= 0) { ungrouped.push(o); continue; }
    const dir = o.text.slice(0, i);
    if (!groups.has(dir)) groups.set(dir, []);
    groups.get(dir).push({ ...o, text: o.text.slice(i + 1) });
  }
  const byText = (a, b) => a.text.localeCompare(b.text);
  const out = [...ungrouped].sort(byText);
  for (const [dir, opts] of [...groups.entries()].sort((a, b) =>
    a[0].localeCompare(b[0])
  ))
    [...opts].sort(byText).forEach((o) => out.push({ ...o, group: dir }));
  return out;
}

// ---------------- generic dialog ----------------
let dialogOk = null;
let dialogClose = null;
let dialogCancel = null;  // cancel-override (openDialog's onCancel param)
// quick-filter widget for the dialog title row. `onFilter` gets the trimmed,
// lowercased query (debounced); the x clears it, Escape in the input does too
// (without closing the dialog).
function makeFilterInput(onFilter, initial = "") {
  const input = el("input", { type: "text", placeholder: "filter", title: "filter the list" });
  input.value = initial;
  const clear = el("button", { type: "button", class: "dialog-filter-clear", title: "clear filter" }, "×");
  clear.hidden = !input.value;
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
function openDialog(title, body, onOk, okText = "Save", onClose = null, onFilter = null, initialFilter = "", onCancel = null) {
  // list dialogs (they have the quick-filter) pin the title row and the
  // actions and scroll only their item list (.list in style.css)
  $("dialog-box").classList.toggle("list", !!onFilter);
  $("dialog-title").textContent = title;
  const f = $("dialog-filter");
  f.innerHTML = "";
  f.hidden = !onFilter;
  if (onFilter) f.append(makeFilterInput(onFilter, initialFilter));
  const b = $("dialog-body");
  b.innerHTML = "";
  b.append(body);
  dialogOk = onOk;
  dialogClose = onClose;
  dialogCancel = onCancel;
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
  dialogCancel = null;
  if (onClose) onClose(); // e.g. return to the management list dialog
}
// every cancel path (Cancel button, Escape, backdrop click) routes through
// the dialog's onCancel override when it has one (e.g. the Upscale dialog
// aborts its running request instead of closing)
function requestDialogCancel() {
  if (dialogCancel) dialogCancel();
  else closeDialog();
}
$("dialog-cancel").onclick = () => requestDialogCancel();
$("dialog").onclick = (e) => {
  if (e.target === $("dialog")) requestDialogCancel();
};
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !$("dialog").hidden) requestDialogCancel();
});
function dialogError(text) {
  $("dialog-body").append(el("div", { class: "dialog-error" }, text));
}
