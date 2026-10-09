# Copy/Paste with floating selection

Add 'Copy' and 'Paste' tool buttons to the toolbar (after the crop button).

## Copy

Allows selecting a rectangular area by dragging the mouse (re-use crop tool area selection if possible). When the user releases the mouse button the area is stored internally (set a status message to confirm this). 
Don't use the system clipboard.

## Paste

Only enabled if an area was previously copied. Attaches the copied area to the mouse cursor (only visible inside the canvas, respects canvas zoom).
On mouse click it is dropped onto the image as a floating selection. 
While a floating selection is active:
- Draw, Smudge, Clone and Crop only affect the floating selection (clone samples from the floating selection)
- Drawing with held shift acts as an eraser, decreasing the opacity of the affected pixels. (Note: switch the pre-existing draw color picker function to the ctrl key. Shift modifier to erase is in line with mask erase)
The selection has a faint border with handles to manipulate it:
- center of each edge: moves the edge perpendicularly inwards/outwards and scales the image to fill the new dimensions (changing the aspect ratio)
- top right/bottom left corner: aspect ratio preserving scale (based on current aspect ratio; edges on opposite side retain their position)
- top left/bottom right corner: free rotation around centerpoint
Ok/Cancel buttons in the middle of the floating selection.
- Ok applies it to the canvas image
- Cancel discards the floating selection
## Decisions (agreed during review)

- The selection lives in its own offscreen canvas and is drawn only as an overlay; `view.base`
  is untouched until Apply composites it (so sync/undo/snapshots are unaffected).
- A new server-side view (generation landing, entity switch, segmentation, crop commit) discards
  the floating selection. Generate/save ignore it (only the committed base syncs).
- One selection at a time; a new Paste drop replaces the old one.
- Apply composites with per-pixel alpha, pushes ONE draw-undo step (Ctrl+Z removes the paste);
  Cancel discards with no history step. Selection edits (draw/smudge/clone/crop on it) get no
  undo entries of their own. Enter = Apply, Esc = Cancel (Esc aborts an in-flight rect drag first).
- Mask tool / morphology / segmentation keep acting on the canvas entity, never the selection.
- Crop with an active selection crops the SELECTION (canvas untouched, no new entity).
- Clone with an active selection samples the selection (source picked with ctrl-click, in the
  selection's frame). Smudge is clipped to the selection (it only touches the selection image).
- Shift+draw on the selection = eraser (reduces alpha, non-stacking max coverage). The draw
  colour picker moves from shift to ctrl (draw tool only; clone keeps ctrl-click, mask keeps
  shift-erase).
- Copy reads the client's current base (local edits included), clamped to the image like the
  crop rect, fully opaque. Copy works while a selection is active (copies from the base) and
  overwrites the stored area. No auto-switch to Paste.
- The copied area is in-memory only: it survives entity switches (copy from A, paste onto B),
  is lost on reload.
- Paste drops the area centred on the click, may hang off the canvas, clipped at Apply.
- Handles: edge midpoints move that edge in the selection's local (rotated) frame and rescale
  the image to fill the new rect (aspect changes); TR/BL scale with the aspect locked at drag
  start, opposite corner pinned; TL/BR rotate freely about the centre. No snapping.
  The image keeps its own resolution while dragging (stretched preview) and is resampled once
  at Apply. Min 8 px, max 8192 px per side.
- Hotkeys: Clone moves from C to O, Copy takes C, Paste is V; Ctrl+C / Ctrl+V are aliases.

## Follow-up changes (after the first implementation)

- Apply ends paste mode: it switches the toolbar to the Copy tool, so users cannot get stuck in
  paste mode.
- Paste mode with an active selection: dragging INSIDE the selection moves it (a click outside it
  drops a new copy). The cursor is `move` only inside the selection.
- Handle cursors: `ns-resize` (top/bottom edge), `ew-resize` (left/right edge), `nesw-resize`
  (TR/BL aspect corners), `grabbing` (TL/BR rotate corners).
- The Apply/Cancel pair is a zero-size anchor point at the selection centre with the buttons
  flanking it (a flex row shifts the gap off-centre once the labels differ in width, which made
  the centre unclickable/unpaintable); compact height to cover as little of the image as possible.
- The Apply/Cancel labels must not wrap (`white-space: nowrap`: the buttons shrink-to-fit inside
  the zero-size anchor, whose available width is 0). With the labels on one line the pair is wide,
  so on a small selection it would cover the edge/corner handles — `syncSelActions()` measures the
  fit and toggles `.compact`, which hides the labels (the glyphs stay, Enter/Esc and the tooltips
  carry the meaning).
- Flip buttons: `↔ H-flip` centred above and `↕ V-flip` centred below the apply/discard pair
  (same anchor point). They flip the floating selection's own pixels (H: left<->right, V: top<->bottom); the
  committed image is untouched until Apply, so flipping needs no undo step.
