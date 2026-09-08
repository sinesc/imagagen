# AI image generator

## Features

### Unified canvas

- one canvas to generate, edit and inpaint images
- default brush (circle) and customizable brushes (using greyscale png files as brush alpha from a "brushes" subdirectory)
- brush scale slider 0.1 to 10.0
- brush opacity slider 0.0 to 1.0
- mask tool: draw mask for AI inpainting
    - mask alpha (from brush) = inpainting alpha
    - button + input to expand/shrink mask by given pixels
    - button + dropdown to generate mask via given segmentation model
    - button to invert mask
- clone tool: useful for roughly "removing" or cloning areas of an image before inpainting, implemented as a plugin
- smudge tool: useful for smudging/blurring areas of an image to roughly fix/cover issues in an image before inpainting, implemented as a plugin
- each tool remembers the last brush and opacity it was used with
- unlimited undo/redo for canvas changes (mask changes and image modification via tools or image generation/inpaint)

### Image generation

- "generate" button, changes to "interrupt" while images are being generated
- model selector dropdown lists all model configurations:
    - each detected model ("models" subdirectory) without configuration files ("model-configs" subdirectory) gets an in-memory default configuration (so it shows up in the model selector)
    - the user can create multiple configurations per model (e.g. "My SDXL Model" and "My SDXL Model + DMD2 Lora")
    - configuration options: sampler, scheduler, CFG, LORAs. Each option separately for text to image and image to image (inpaint)
    - options are applied on model selection, can be overridden by user
- settings for image size, sampling steps, sampler, scheduler, CFG, seed ("-1" means random), LORAs, batch size, generation-mode, denoising-strength
- supported generation-modes:
    - "generate" creates a new image with the given dimensions (text to image)
    - "inpaint masked" operates on the minimal bounding box of the masked area at the resolution specified via image size. The result is scaled and blended via the mask alpha with the canvas image
    - "inpaint not masked" operates on the whole image after scaling it to the specified image size (if required). Scaling retains aspect ratio and the scaled image "touches" the inside of the target image size, e.g. going from 100x100 to 250x200 would scale the source to 200x200 (retaining the aspect ratio) and center the result on an empty 250x200 canvas before performing the inpaint. (This enables "outpainting".) The current mask should be scaled appropriately.
- batch size: support for generating mutiple images:
    - each image in a batch is created with a different seed (ascending from initial seed)
    - each generated image adds a clickable thumbnail above the canvas to allow selecting which generated variation is used in the canvas (also support cursor left/right key to switch between variants). The first generated image is automatically set in the canvas.
    - the canvas should be editable while images are generated. if the user edits the current canvas via e.g. the clone tool while the first image is still being generated then those changes will be overridden by the generation but should be restorable via undo
- setting preset dropdown and preset management button
    - button opens dialog to allow user to create/edit setting presets
    - presets have a name and configure one or more of the above image generation settings (each setting has a checkbox that determines whether the preset controls that setting, e.g. some preset might only change the image size to a specific value and leave all other settings untouched)
- prompt/negative prompt input fields

### Image import/metadata

- Generated images should contain metadata with generation settings
- "Import" button in the UI opens a file picker to select an image to import settings from. Alternatively dragging a file onto the button should work too.
- Should support importing settings from .png files created with "Forge"

## System

### Image generation queue

- supports distributing generation across multiple GPUs (one image per CPU but multiple images in parallel)
- queue dynamically assigns to whichever GPUs are currently available (i.e. do not pre-assign via modulo or similar, some GPUs will be slower/faster than others)

### Plugin system

- used to implement model support, ships with Stable Diffusion XL plugin
- used to implement canvas tools, ships with clone and smudge plugins

## Implementation

- pick most appropriate language
- web-based UI is acceptable
- implement all required tooling to enable you to verify/test/debug your implementation
- make a phased plan that priorizes MVP functionality
