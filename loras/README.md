# loras/

LoRAs live in one subfolder per model family. The folder names below are
fixed; inside a family folder everything is scanned recursively and
symlinked files and folders are followed.

| Folder | LoRAs for |
| --- | --- |
| `sdxl/` | Stable Diffusion XL checkpoints |
| `flux/` | FLUX.1 models |
| `zimage/` | Z-Image models |
| `krea2/` | Krea 2 models |

The format is the usual `.safetensors` (or `.ckpt`) LoRA file as
published on Civitai/Hugging Face — no conversion needed. LoRAs are
addressed by their path relative to the family folder, without the
extension: `loras/sdxl/test-dir/FuturisticXL.safetensors` →
`test-dir/FuturisticXL`.

A LoRA is only selectable together with a model of its own family (the UI
greys out / hides mismatched ones). Several LoRAs can be combined per
generation, each with its own strength and an enable toggle; a model
configuration can also carry a fixed LoRA set that is always applied.

## Optional per-LoRA settings

A `.json` file with the same basename as the `.safetensors`, stored next
to it, can carry per-LoRA settings (Forge-compatible keys). Besides the
usual *activation text* (always appended to the prompt), imagagen also
supports **optional activation text**: comma-separated tags that appear
as toggle buttons under the negative prompt, so you can switch parts of a
LoRA's trigger words on or off per generation. These settings can be
edited from the LoRA dialog in the UI instead of by hand.
