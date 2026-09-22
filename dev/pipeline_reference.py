"""Golden reference: step-by-step diff of our sampling loop vs the stock
diffusers SDXL pipeline.

This is the tool that caught the original four sampling bugs (timestep
mapping, unit-variance input scaling, eps->x0 conversion, noise init).
If the sampling loop is ever changed, re-run this: every step's latent
should match the stock pipeline's trajectory within fp16 noise
(max|diff| ~ < 0.05). A growing diff pinpoints which step the change broke.

Also saves both final images for visual (or ascii_view.py) comparison.

Usage (from the project root; needs GPU + cached HF files):
    .venv/bin/python dev/pipeline_reference.py
"""

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

MODEL = "models/sdxl/sd_xl_base_1.0.safetensors"
IMPROVED_VAE = "models/sdxl_vae.safetensors"
PROMPT = "a red cube on a wooden table, studio photo, soft lighting"
NEG = "blurry, low quality"
W = H = 512
STEPS = 25
CFG = 5.5
SEED = 42


def main() -> None:
    pipe = StableDiffusionXLPipeline.from_single_file(MODEL, torch_dtype=torch.float16)
    pipe.vae = AutoencoderKL.from_single_file(IMPROVED_VAE, torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe_state: dict[int, torch.Tensor] = {}

    def cb(pipe_, i, t, callback_kwargs):
        pipe_state[i] = callback_kwargs["latents"].clone()
        return callback_kwargs

    with torch.inference_mode():
        # 1) stock pipeline reference
        pipe(prompt=PROMPT, negative_prompt=NEG, width=W, height=H,
             num_inference_steps=STEPS, guidance_scale=CFG,
             generator=torch.Generator(device="cuda").manual_seed(SEED),
             callback_on_step_end=cb).images[0].save("/tmp/ref_pipeline.png")

        # 2) our loop, step by step
        x = torch.randn((1, 4, H // 8, W // 8), device="cuda", dtype=torch.float16,
                        generator=torch.Generator(device="cuda").manual_seed(SEED))
        x = x * pipe.scheduler.init_noise_sigma
        pe, _, pooled, _ = pipe.encode_prompt(prompt=PROMPT, device="cuda", num_images_per_prompt=1)
        _, ne, _, npool = pipe.encode_prompt(prompt=NEG, device="cuda", num_images_per_prompt=1)
        pipe.text_encoder.cpu()
        pipe.text_encoder_2.cpu()
        torch.cuda.empty_cache()

        time_ids = torch.tensor([H, W, 0, 0, H, W], device="cuda").unsqueeze(0)
        unet = pipe.unet
        ac = pipe.scheduler.alphas_cumprod.to(device="cuda", dtype=torch.float32)
        log_sigmas = torch.log(torch.sqrt((1.0 - ac) / ac))

        def sigma_to_t(sigma):
            return (torch.log(sigma.float()) - log_sigmas).abs().argmin(-1)

        def unet_fwd(emb, pool, xx, t):
            return unet(xx, t, encoder_hidden_states=emb,
                        added_cond_kwargs={"time_ids": time_ids, "text_embeds": pool}).sample

        sigmas = pipe.scheduler.sigmas.to("cuda")
        maxdiff = 0.0
        for i in range(STEPS):
            sigma = sigmas[i]
            x_in = x / (sigma ** 2 + 1.0).sqrt()      # unit-variance input
            c = unet_fwd(pe, pooled, x_in, sigma_to_t(sigma))
            uc = unet_fwd(ne, npool, x_in, sigma_to_t(sigma))
            x = x + (uc + CFG * (c - uc)) * (sigmas[i + 1] - sigma)
            d = (x - pipe_state[i]).abs().max().item()
            maxdiff = max(maxdiff, d)
            if i < 3 or i == STEPS - 1:
                print(f"step {i + 1:2d}: max|diff| = {d:.5f}")
        print(f"max diff over all steps: {maxdiff}")

        # 3) decode both conventions to images
        lat = x / pipe.vae.config.scaling_factor
        out = pipe.vae.decode(lat, return_dict=False)[0]
        out = (out / 2 + 0.5).clamp(0, 1)
        from PIL import Image
        a = (out.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype("uint8")
        Image.fromarray(a).save("/tmp/ref_mine.png")
        print("saved /tmp/ref_pipeline.png (stock) and /tmp/ref_mine.png (our loop)")
        print("compare with: python dev/ascii_view.py /tmp/ref_pipeline.png")


if __name__ == "__main__":
    main()
