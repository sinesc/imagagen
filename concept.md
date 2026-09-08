# AI image generator

## Features

### Unified canvas

- one canvas to generate, edit and inpaint images
- default brush (circle with 10px diameter) and customizable brushes using luminance of greyscale png files as brush alpha from a "brushes" subdirectory of the project folder
- brush scale slider 0.1 to 100.0 (likely not practical with linear scale, let's try logarithmic)
- brush opacity slider 0.0 to 1.0
- mask tool: draw mask for AI inpainting
    - mask alpha (from brush) = inpainting alpha
    - holding shift while drawing acts as a mask eraser
    - button + input to expand/shrink mask by given pixels
    - button + dropdown to generate mask via given segmentation model (from "models/segmentation" subdirectory)
    - button to clear the mask
- clone tool: useful for roughly "removing" or cloning areas of an image before inpainting, implemented as a plugin
- smudge tool: useful for smudging/blurring areas of an image to roughly fix/cover issues in an image before inpainting, implemented as a plugin
- each tool remembers the last brush and opacity it was used with
- unlimited undo/redo for canvas changes (mask and image modification via tools or image generation/inpaint)
- save button to save current image to "output" subdirectory, filename derived from prompt with a leading integer number derived from the highest leading integer found in the output directory, (e.g. output contains "123-photo-of-a-turtle.png" then the next saved image name could be "124-photo-of-a-frog.png")

### Image generation

- "generate" button, changes to "interrupt" while images are being generated (any already generated images are retained in thumbnails, partial images are not added)
- model selector dropdown lists all model configurations:
    - each detected model ("models/<model-type>/" subdirectory, e.g. "models/sdxl/") without configuration files ("model-configs" subdirectory) gets an in-memory default configuration (so it shows up in the model selector)
    - the user can create multiple configurations per model (e.g. "My SDXL Model" and "My SDXL Model + DMD2 Lora")
    - configuration options: sampler, scheduler, CFG, LORAs. Each option separately for text to image and image to image (inpaint)
    - configured values act as default values when the respective settings below are set to use the model default with the exception of "LORAs" which are always prepended to the selected LORAs
- settings for image size, sampling steps, sampler, scheduler, CFG, seed, LORAs, batch size, generation-mode, denoising-strength (disabled when not inpainting)
- "seed": setting this to "-1" means next generation starts with a random value (batch generations increment from that value)
- "sampler" and "scheduler" settings: these are dropdowns, the first option is "Use model default" and uses the default value configured for the model, followed by the supported samplers/schedulers. samplers and schedulers are implemented via plugins.
- "CFG" setting: this is a float, when set to 0.0 the configured model default is used.
- "LORAs" setting: this is a list of dropdowns and text-inputs (weight). The list should have a "+" button to add an entry and each list element should have a "-" button to remove the element. The selected LORAs are used **in addition** to any configured model default LORAs.
- supported generation-modes:
    - "generate" creates a new image with the given dimensions (text to image)
    - "inpaint masked" operates on the minimal aspect-preserving bounding of the masked area at the resolution specified via image size. (To clarify: the bounding box uses the aspect ratio of the specified image size and minimally encloses the masked area). The result is scaled and blended via the mask alpha with the canvas image
    - "inpaint not masked" operates on the whole image (with the masked content protected from changes) after scaling it to the specified image size (if required). Scaling retains aspect ratio and the scaled image "touches" the inside of the target image size, e.g. going from 100x100 to 250x200 would scale the source to 200x200 (retaining the aspect ratio) and center the result on an empty 250x200 canvas before performing the inpaint. (This enables "outpainting".)
- the mask is retained across image generations unless cleared by user. When "inpaint not masked" changes to size of the image the current canvas size is updated accordingly and the mask is scaled appropriately (e.g. when image is resized from 100x100 to 250x200 the mask should be scaled by 2x and centered by moving it 25 pixels to the right)
- batch size: support for generating mutiple images:
    - each image in a batch is created with a different seed (ascending from initial seed)
    - each generated image adds a clickable thumbnail above the canvas to allow selecting which generated variation is used in the canvas (also support cursor left/right key to switch between variants). The first generated image is automatically set in the canvas.
    - the canvas should be editable while images are generated. if the user edits the current canvas via e.g. the clone tool while the first image is still being generated then those changes will be overridden by the generation but should be restorable via undo
- setting preset dropdown and preset management button
    - button opens dialog to allow user to create/edit setting presets
    - presets have a name and configure one or more of the above image generation settings (each setting has a checkbox that determines whether the preset controls that setting, e.g. some preset might only change the image size to a specific value and leave all other settings untouched)
    - presets are stored in the "presets" subdirectory
- prompt/negative prompt input fields

### Image import/metadata

- Generated images should contain metadata with generation settings (included in the output file when saved, ideally in a format "Forge" can read)
- "Import" button in the UI opens a file picker to select an image to import settings from. Alternatively dragging a file onto the button should work too.
- Should support importing settings from .png files created with "Forge"

## System

### Image generation queue

- supports distributing generation across multiple GPUs (each GPU generates one image at a time)
- queue dynamically assigns to whichever GPUs are currently available (i.e. do not pre-assign via modulo or similar, some GPUs will be slower/faster than others)

### Plugin system

- used to implement model/LORA support, ships with Stable Diffusion XL plugin (includes LORA support)
- used to implement canvas tools, ships with clone and smudge plugins
- used to implement samplers, ships with "ancestral Euler" ("Euler a"), "DPM++ 2M", "DPM++ 2M SDE"
- used to implement schedulers, ships with "Karras", "Polyexponential"

## Implementation

- the folder containing this file is the project root
- sourcecode goes into `src` subdirectory
- pick most appropriate language
- web-based UI is acceptable
- implement all required tooling to enable you to verify/test/debug your implementation
- make a phased plan that priorizes MVP functionality
- the parallel folder "sd-webui-forge-classic" contains a python based inference webui that may contain useful code (e.g. sampler implementations)
- a segmentation implementation can be found in "sd-webui-forge-classic/extensions/a-person-mask-generator"
