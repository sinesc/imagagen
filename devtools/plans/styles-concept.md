Main UI
- similar to LoRAs UI: "+ Style" button, adds a dropdown for a style, each dropdown gets a `-` button to remove
- dropdowns sorted alphabetically by optgroup, style name 
- placement under the LoRAs UI
- edit button next to the "+ Style" button, cogwheel icon -> opens style configuration
- styles loaded from styles.csv, each style has a name, prompt, negative prompt and optional "optgroup" (dropdown optgroup for grouping styles)

Style dialog
- similar to preset dialog, new styles sub-dialog has inputs for name, group (optgroup), prompt, negative prompt 

Style application
- before passing prompt/negative prompt strings to image generation
- for each style in listed order `{prompt}` placeholder in style text is replaced with user's prompt (and any already applied styles). If `{prompt}` not present append to end of prompt instead

Image meta-data:
- save separate from prompt (not pre-applied)
- save style name but also include actual style prompt/negative prompt contents
- when loading meta data from image and a style does not exists (e.g. deleted, imported on different install, ...), add it as a temporary style to the UI (track those separately in memory only, use optgroup "Session", do not save them to the users styles.csv)
