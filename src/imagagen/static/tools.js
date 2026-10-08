/* Tool state: current tool + per-tool remembered brush/opacity/etc.
   Spec: each tool remembers the last settings it was used with.
   The state is in-memory here; persistence is handled by app.js,
   which includes it in the ui-state saved to the server. */
"use strict";

const TOOLS = ["brush", "mask", "clone", "smudge", "crop", "copy", "paste"];

const Tools = {
  current: "brush",
  // spec default brush: circle, 10 px diameter, full opacity
  // screen: size in screen px (constant while zooming) instead of canvas px
  state: {
    brush: { size: 10, opacity: 1, brush: "circle", color: "#ffffff", screen: false },
    mask: { size: 30, opacity: 1, brush: "circle", screen: false },
    // no colour: clone copies pixels from the picked source point
    clone: { size: 10, opacity: 1, brush: "circle", screen: false },
    // no colour either: smudge smears the image's own pixels
    smudge: { size: 10, opacity: 1, brush: "circle", screen: false },
    // crop uses no brush at all; entry kept so Tools.get() never returns undefined
    crop: { size: 10, opacity: 1, brush: "circle", screen: false },
    // copy drags a rectangle and paste drops the copied area: no brush here
    // either; entries kept so Tools.get() never returns undefined
    copy: { size: 10, opacity: 1, brush: "circle", screen: false },
    paste: { size: 10, opacity: 1, brush: "circle", screen: false },
  },
  // restored from the server's ui-state by applyUiState() in app.js
  get() {
    return this.state[this.current];
  },
  switch(name) {
    if (!TOOLS.includes(name)) return;
    this.current = name;
  },
};
