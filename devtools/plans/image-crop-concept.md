# Image crop tool
- new toolbar button after 'mask'
- disables color, size, screen, opacity while selected
- while the user clicks and drags a rectangle should preview the crop area
- if crop starts or ends outside of the image the actual coordinates used should snap to the image/preview rectangle intersection points
- on crop, the not-yet-cropped canvas image should become the previous image in the thumbnail list and the cropped image should create a new thumbnail
- unlike image generation, the cropped image/thumbnail should be immediately selected
- pressing ESC while selecting the crop region should abort the action