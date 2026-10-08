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