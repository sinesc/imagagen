# Rework undo/redo system

- no longer track image selection in undo/redo

## Separate systems for mask and canvas draw operations

- system automatically selected depending on whether "brush" or "mask" mode is selected
- undo button labels change dynamically: "undo mask"/"redo mask" <-> "undo draw"/"redo draw" depending on selected "brush"/"mask" mode

### Mask

- tracks all changes to the mask: drawing/erasing via brush, expand, shrink, segment etc.
- image independent, switching images does nothing to the mask undo stack
- mask application by undo/redo is scaled to the current image size if it differs from original mask size (aspect-preserving scaling, touching image inner edges of image)

### Draw operations

- tracks changes to the current canvas image
- when canvas image is switched (selecting a different thumbnail) change history is preserved so that selecting the thumbnail for the previous canvas image also restores the history for that image
- each image gets its own undo/redo stack
- undo-history deleted when the corresponding image is evicted
- note: check current image eviction, I suspect we keep images until server is closed. they should be evicted when they are no longer on canvas or have thumbnails