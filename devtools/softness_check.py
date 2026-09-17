"""Softness investigation: A/B/C comparison of sigma schedules.

A: current imagagen behavior  -- Karras over the clamped k-diffusion range [0.1, 10]
B: Forge default              -- Karras over the model's beta schedule range
                                 (SDXL: ~[0.038, ~14.4])
C: Forge "Zero Terminal SNR"  -- B, with the noisiest end rescaled to SNR ~ 0
                                 (Forge's rescale_zero_terminal_snr_sigmas)

Same seed/prompt/CFG/steps/sampler for all three. Saves individual PNGs, a
side-by-side composite, and prints a simple sharpness metric (Laplacian
variance).

Usage:  .venv3.13/bin/python devtools/softness_check.py
(No other server/model should be holding the GPU.)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from imagagen import kdiff  # noqa: E402
from imagagen.config import ensure_dirs, load_config  # noqa: E402
from imagagen.gpus import select_gpu  # noqa: E402

PROMPT = ("close-up portrait of a weathered lighthouse keeper, detailed face, "
          "fine skin texture, deep wrinkles, film photography, 85mm lens, "
          "dramatic side lighting")
NEGATIVE = ""
SEED = 42
STEPS = 20
CFG = 7.0
SIZE = 768


def rescale_ztsnr_desc(sigmas: torch.Tensor) -> torch.Tensor:
    """Forge's rescale_zero_terminal_snr_sigmas for descending sigmas.

    Mirrors backend/modules/k_prediction.py (Forge stores sigmas ascending;
    k-diffusion stores them descending): the cleanest end is kept fixed,
    the noisiest end is stretched to (almost) zero terminal SNR.
    """
    ab = 1.0 / (sigmas ** 2 + 1)
    ab_sqrt = ab.sqrt()
    s_noisy = ab_sqrt[0].clone()
    s_clean = ab_sqrt[-1].clone()
    ab_sqrt = (ab_sqrt - s_noisy) * s_clean / (s_clean - s_noisy)
    ab = ab_sqrt ** 2
    ab[0] = 4.8973451890853435e-08
    return ((1 - ab) / ab) ** 0.5


def laplacian_var(arr: np.ndarray) -> float:
    g = arr.astype(np.float32).mean(axis=2)
    c = g[1:-1, 1:-1]
    lap = (c + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
           - 5 * c)
    return float(lap.var())


def main() -> None:
    config = load_config()
    ensure_dirs(config)
    device = torch.device(f"cuda:{select_gpu(config.primary_gpu, config.ignore_gpus)}")
    print("device:", device, flush=True)

    import warnings

    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    model_path = config.root / "models" / "sdxl" / "epicrealismXL11.0.safetensors"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = StableDiffusionXLPipeline.from_single_file(
            str(model_path), torch_dtype=torch.float16)
        vae_path = config.root / "models" / "sdxl_vae.safetensors"
        if vae_path.is_file():
            pipe.vae = AutoencoderKL.from_single_file(str(vae_path),
                                                      torch_dtype=torch.float16)
    pipe.vae.enable_slicing()
    pipe.to(device)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    lh = lw = SIZE // 8
    with torch.inference_mode():
        pipe.text_encoder.to(device)
        pipe.text_encoder_2.to(device)
        prompt_embeds, _, pooled, _ = pipe.encode_prompt(
            prompt=PROMPT, device=device, num_images_per_prompt=1)
        _, neg_embeds, _, neg_pooled = pipe.encode_prompt(
            prompt=NEGATIVE, device=device, num_images_per_prompt=1)
        pipe.text_encoder.cpu()
        pipe.text_encoder_2.cpu()
        torch.cuda.empty_cache()

        time_ids = torch.tensor([SIZE, SIZE, 0, 0, SIZE, SIZE],
                                device=device).unsqueeze(0)
        unet = pipe.unet
        ac = pipe.scheduler.alphas_cumprod.to(device=device, dtype=torch.float32)
        log_sigmas = torch.log(torch.sqrt((1.0 - ac) / ac))

        def sigma_to_t(sigma):
            return (torch.log(sigma.float()) - log_sigmas).abs().argmin(-1)

        def model_fn(x, sigma):
            t = sigma_to_t(sigma)
            x_in = x / (sigma ** 2 + 1.0).sqrt()
            c = unet(x_in, t, encoder_hidden_states=prompt_embeds,
                     added_cond_kwargs={"time_ids": time_ids,
                                        "text_embeds": pooled}).sample
            uc = unet(x_in, t, encoder_hidden_states=neg_embeds,
                      added_cond_kwargs={"time_ids": time_ids,
                                         "text_embeds": neg_pooled}).sample
            eps = uc + CFG * (c - uc)
            return x - sigma * eps

        # kdiff samplers call model(x, sigma, **extra_args) -- absorb extras
        def model_wrap(x, sigma, **kw):
            return model_fn(x, sigma)

        # sigma ranges
        beta_min = float(log_sigmas.exp()[0])    # sigma at t=0 (cleanest)
        beta_max = float(log_sigmas.exp()[-1])   # sigma at t=999 (noisiest)
        print(f"beta schedule range: [{beta_min:.4f}, {beta_max:.4f}]", flush=True)

        gen = torch.Generator(device=device).manual_seed(SEED)
        base_noise = torch.randn((1, 4, lh, lw), device=device,
                                 dtype=torch.float16, generator=gen)

        def run(name, sigmas):
            sigmas = sigmas.to(device=device, dtype=torch.float32)
            print(f"\n{name}: sigmas[0]={sigmas[0].item():.4f} "
                  f"sigmas[-2]={sigmas[-2].item():.4f} "
                  f"sigmas[-1]={sigmas[-1].item():.4f} "
                  f"-> t first/last = "
                  f"{sigma_to_t(sigmas[0:1]).item()}/{sigma_to_t(sigmas[-1:]).item()}",
                  flush=True)
            x = (base_noise * sigmas[0].item()).to(torch.float16)
            samples = kdiff.sample_dpmpp_2m(model_wrap, x, sigmas,
                                            extra_args={"seed": SEED})
            latents = samples / pipe.vae.config.scaling_factor
            image = pipe.vae.decode(latents, return_dict=False)[0]
            image = (image / 2 + 0.5).clamp(0, 1)
            arr = (image.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)
            print(f"{name}: laplacian_var = {laplacian_var(arr):.1f}", flush=True)
            return arr

        variants = {
            "A_current_0.1_10": kdiff.get_sigmas_karras(STEPS, 0.1, 10.0),
            "B_forgedefault_betarange": kdiff.get_sigmas_karras(
                STEPS, beta_min, beta_max),
            "C_forgedefault_ztsnr": rescale_ztsnr_desc(
                kdiff.get_sigmas_karras(STEPS, beta_min, beta_max)),
        }

        out_dir = config.root / "output" / "softness"
        out_dir.mkdir(parents=True, exist_ok=True)
        images = []
        for name, sigmas in variants.items():
            arr = run(name, sigmas)
            p = out_dir / f"{name}.png"
            from PIL import Image
            Image.fromarray(arr).save(p)
            images.append((name, arr))
            torch.cuda.empty_cache()

        # side-by-side composite (scaled to half height for a single glance)
        h = SIZE // 2
        w = int(SIZE * 0.75)
        from PIL import Image, ImageDraw
        rows = [Image.fromarray(arr).resize((w, h)) for _, arr in images]
        composite = Image.new("RGB", (w * len(rows), h + 24))
        draw = ImageDraw.Draw(composite)
        for i, ((name, _), im) in enumerate(zip(images, rows, strict=True)):
            composite.paste(im, (i * w, 24))
            draw.text((i * w + 8, 6), name, fill=(255, 255, 0))
        cpath = out_dir / "composite.png"
        composite.save(cpath)
        print(f"\nsaved to {out_dir}\ncomposite: {cpath}")


if __name__ == "__main__":
    main()
