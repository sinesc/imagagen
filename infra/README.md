# infra/

Shared support weights — text encoders and VAEs (the encoder/decoder
pair around the actual model) — live here. Unlike `models/` and
`loras/`, these files are looked up by **exact file name**; place real
files or symlinks with the names below.

| File | Needed by | What it is |
| --- | --- | --- |
| `ae.safetensors` | Flux, Z-Image | The FLUX.1 AE VAE (byte-identical for both families); ships with `black-forest-labs/FLUX.1-dev` on Hugging Face and in every ComfyUI Flux pack. |
| `clip_l.safetensors` | Flux | CLIP-L text encoder (fp16), e.g. from `comfyanonymous/flux_text_encoders`. |
| `t5xxl.safetensors` | Flux | T5-XXL text encoder; tested with the `fp8_e4m3fn` variant. |
| `qwen_3_4b.safetensors` | Z-Image | Qwen3-4B text encoder; tested with the fp8-*scaled* variant. |
| `qwen_3vl_4b.safetensors` | Krea 2 | Qwen3-VL-4B text encoder (fp8). Its vision tower is also used for reference-image captioning. |
| `qwen_image_vae.safetensors` | Krea 2 | The Qwen-Image VAE. |
| `sdxl_vae.safetensors` | SDXL (optional) | The ft-MSE "Improved SDXL VAE". The stock SDXL VAE can overflow to NaN in fp16 for high-magnitude latents; the improved VAE is fp16-stable and renders colors better. If the file is present it is swapped in automatically, if absent the checkpoint's own VAE is used. |

You only need the files for the families you actually use: SDXL works
with just a checkpoint (the improved VAE is optional), Flux needs the
three files above it, Z-Image needs `ae` + `qwen_3_4b`, Krea 2 needs
`qwen_3vl_4b` + `qwen_image_vae`.

If you already have these files elsewhere (e.g. a shared ComfyUI
install), symlinks are the easiest way to reuse them:

```bash
ln -s /path/to/comfyui/models/text_encoders/t5xxl_fp8_e4m3fn.safetensors t5xxl.safetensors
```

