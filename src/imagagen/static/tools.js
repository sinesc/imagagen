/* Tool state: current tool + per-tool remembered brush/opacity.
   Spec: each tool remembers the last brush and opacity it was used with.
   Persisted in localStorage so the choice survives a page reload. */
"use strict";

const TOOLS = ["brush", "mask", "clone"];

const Tools = {
  current: "brush",
  // spec default brush: circle, 10 px diameter, full opacity
  // screen: size in screen px (constant while zooming) instead of canvas px
  state: {
    brush: { size: 10, opacity: 1, brush: "circle", color: "#ffffff", screen: false },
    mask: { size: 30, opacity: 1, brush: "circle", screen: false },
    // no colour: clone copies pixels from the picked source point
    clone: { size: 10, opacity: 1, brush: "circle", screen: false },
  },
  load() {
    try {
      const raw = localStorage.getItem("imagagen.tools");
      if (!raw) return;
      const s = JSON.parse(raw);
      for (const t of TOOLS) if (s[t]) Object.assign(this.state[t], s[t]);
      if (TOOLS.includes(s.current)) this.current = s.current;
    } catch (e) {
      console.warn("tools load failed:", e);
    }
  },
  save() {
    try {
      localStorage.setItem("imagagen.tools", JSON.stringify({ current: this.current, ...this.state }));
    } catch (e) { /* private mode etc. */ }
  },
  get() {
    return this.state[this.current];
  },
  switch(name) {
    if (!TOOLS.includes(name)) return;
    this.current = name;
    this.save();
  },
};
