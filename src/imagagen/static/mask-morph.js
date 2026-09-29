/* Mask morphology worker: dilate/erode a mask's VALUE plane (alpha 0..255)
   with the disk neighbourhood dx²+dy² ≤ r² (the disk the old server used
   with scipy's maximum_filter/minimum_filter). Row-band sliding-window
   extremum (deque per row, O(H·W·(2r+1)) total); clamped row indexes equal
   scipy's reflect borders for max/min filters.

   in : {w, h, r, dilate, plane: Uint8Array} (plane transferred)
   out: {plane: Uint8Array} (transferred)
*/
"use strict";

self.onmessage = ({ data: m }) => {
  const { w, h, r, dilate } = m;
  const src = m.plane;
  const acc = new Uint8Array(w * h);
  if (dilate) acc.fill(0); else acc.fill(255);
  const win = new Uint8Array(w);
  // deque of row indexes; at most one push per output x plus the initial
  // window preload, so 2w+2 entries cover a row
  const dq = new Int32Array(2 * w + 2);
  for (let dy = -r; dy <= r; dy++) {
    const halfw = Math.floor(Math.sqrt(r * r - dy * dy));
    for (let y = 0; y < h; y++) {
      const sy = Math.min(h - 1, Math.max(0, y + dy)); // clamp == reflect
      const row = sy * w;
      let head = 0, tail = 0;
      const push = (xi) => {
        // monotone deque: drop tail entries that can never win again
        // (equals: the NEWER index wins, it leaves the window later)
        while (
          tail > head &&
          (dilate
            ? src[row + dq[tail - 1]] <= src[row + xi]
            : src[row + dq[tail - 1]] >= src[row + xi])
        ) tail--;
        dq[tail++] = xi;
      };
      // preload the initial window [0, min(halfw, w-1)], then for each
      // output x push the newly entering index x+halfw (clamped at the
      // right edge) and drop stale indexes (< x-halfw)
      for (let xi = 0; xi <= Math.min(w - 1, halfw); xi++) push(xi);
      for (let x = 0; x < w; x++) {
        if (x > 0) push(Math.min(w - 1, x + halfw));
        while (head < tail && dq[head] < x - halfw) head++; // stale
        win[x] = src[row + dq[head]];
      }
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        acc[i] = dilate ? Math.max(acc[i], win[x]) : Math.min(acc[i], win[x]);
      }
    }
  }
  self.postMessage({ plane: acc }, [acc.buffer]);
};
