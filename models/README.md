# models/

Model weights used for generation live here. The top-level folder name
selects the **model family** and is fixed; everything inside a family
folder is scanned recursively, and symlinked files and folders are
followed — so you can organize (or share) your weights however you like.

| Folder | Contents | Accepted files |
| --- | --- | --- |
| `sdxl/` | Stable Diffusion XL checkpoints | `.safetensors` |
| `flux/` | FLUX.1 [dev] / [schnell] transformers | `.gguf` (e.g. Q4_K quants) or `.safetensors` (e.g. Kijai FP8 / bitsandbytes NF4) |
| `zimage/` | Z-Image (Turbo) checkpoints | `.safetensors` |
| `krea2/` | Krea 2 checkpoints | `.safetensors` |
| `segmentation/` | MediaPipe ImageSegmenter models for one-click mask generation | `.tflite` |
| `upscalers/` | Upscaler models for the Upscale dialog (RealESRGAN, ESRGAN, SwinIR, ...) | `.pth` / `.pt` |

## Naming

Models are addressed by their path **relative to the family folder,
without the extension**. A checkpoint at
`models/sdxl/test-dir/JuggernautXL9.0.safetensors` shows up in the UI as
`test-dir/JuggernautXL9.0`.

## Where to get the weights

Checkpoints are widely available on [Civitai](https://civitai.com) and
[Hugging Face](https://huggingface.co) — search for the family name plus
"SDXL checkpoint", "FLUX GGUF", "Z-Image", "Krea 2" etc. Pick the file
formats listed above (for Flux, plain fp16 diffusers *folders* are not
supported — use a single-file GGUF or FP8/NF4 `.safetensors`).

- **Segmentation**: any MediaPipe *image segmenter* `.tflite` works, e.g.
  the selfie multiclass model (16 MB):
  [selfie_multiclass_256x256.tflite](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite)
- **Upscalers**: the usual Forge/ComfyUI upscaler archives work as-is, e.g.
  RealESRGAN_x4plus. `Lanczos` and `Nearest` are always available in the
  dialog without any model file.

## Companion files

Flux, Z-Image and Krea 2 additionally need shared text-encoder / VAE
weights, which go in [`../infra/`](../infra/README.md). LoRAs go in
[`../loras/`](../loras/README.md).
