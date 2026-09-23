# Flux img2img + inpaint

Make the img2img/inpaint generation modes work for Flux models. Today the
whole img2img/inpaint machinery exists and is family-agnostic; only the
backend half is missing for Flux: `FluxBackend._generate_impl` ignores
`request.init_image`, `request.init_mask` and `request.denoise` and always
runs pure txt2img.

## Background: how the modes reach the backend

- UI mode: "generate" | "inpaint_masked" (Inpaint) | "inpaint_not_masked"
  (Outpaint; with no mask and unchanged aspect ratio this **is** plain
  img2img). The denoise-strength slider and mask padding apply to the
  inpaint modes; the UI already sends `mode`/`denoise` unconditionally.
- `app.py` → `generation.with_inpaint(req, mode, canvas, denoise, padding)`
  snapshots the canvas and fills `req.init_image` (HxWx3 uint8, **exactly
  the request's width×height** — `with_inpaint` LANCZOS-resizes it),
  `req.init_mask` (HxW uint8, **255 = the inpaint/regenerate region**, 0 =
  protected; "inpaint not masked" passes the *inverted* mask) and
  `req.denoise`. It returns an `InpaintPlan` that post-processes the
  backend output back to canvas state afterwards — the backend always
  generates the raw request size, in all modes.
- Everything downstream (queue, source thumbnails, `InpaintPlan`, the
  `Denoising strength` PNG line, metadata) is already family-agnostic.
  **No changes needed outside `flux_backend.py`.**

## Backend contract (A1111/Forge parity, as implemented in
`sdxl_backend._generate_impl` — mirror it, flux conventions)

1. **Encode**: VAE-encode the init image to its deterministic latent
   (`latent_dist.mode()`, A1111 "Full" encode), converted into the solver
   space used by the decode path (`x = (latent − shift) · scaling`).
2. **Denoise truncation**: `t_enc = int(min(denoise, 0.999) · steps)`;
   sample `sigma_sched = sigmas[steps − 1 − t_enc :]` (denoise 1.0 → the
   full schedule). The shifted flow-matching grid is truncated by index,
   same arithmetic as SDXL.
3. **Init state** (flow-matching const `noise_scaling`,
   `sigma·noise + (1−sigma)·latent`): `x = init_latent + noise ·
   sigma_sched[0]` (noise = the seeded fp32 unit prior, as today).
4. **Mask lock** (init mask present): bilinear-interpolate the mask to the
   latent grid, `mask_t = 1 − m` (protected). The protected region starts
   at the clean init latent: `x = x·(1−mask_t) + init_latent·mask_t`.
   Per step, before the transformer call, replace the protected region with
   re-noised init: `xwork = x·(1−mask_t) + (init_latent + randn_like(x)·
   sigma)·mask_t` (fresh unseeded per-step noise — exactly SDXL's
   behaviour, so masked inpaint is *not* seed-reproducible, same as SDXL;
   maskless img2img stays deterministic). After the x0 conversion, hard-lock:
   `x0 = x0·(1−mask_t) + init_latent·mask_t`.
5. **Final lock** after sampling (solver space): `samples = samples·
   (1−mask_t) + init_latent·mask_t`, before the existing decode.
6. **16-px snap guard**: the backend snaps the request to multiples of 16
   (flux patchify), but `init_image` arrives at the un-snapped request size.
   When `(h, w) != init_image.shape[:2]`, LANCZOS-resize the init image to
   `(w, h)` before encoding (the mask needs no resize — it is interpolated
   to the latent grid anyway). SDXL (snap 8) never hits this because its
   encode grid always divides the request size; flux needs the guard.

Out of scope: SAG (UNet mid-block hook — not applicable to Flux), soft
inpainting (deferred below), LoRA, the fp8/nf4 safetensors checkpoints.

## Change

**`src/imagagen/flux_backend.py`** (the only file changed):

1. New `FluxBackend._encode_init(self, arr, device)`:
   ```python
   t = torch.from_numpy(np.ascontiguousarray(arr)).to(device).permute(2, 0, 1).float()[None] / 127.5 - 1.0
   lat = self.vae.encode(t.to(self.vae.dtype)).latent_dist.mode()
   return (lat - self.vae.config.shift_factor) * self.vae.config.scaling_factor
   ```
   (inverse of the decode line `latents = x / scaling_factor + shift_factor`).

2. In `_generate_impl`, replace the single
   `noise = noise * float(sigmas[0])` line with the img2img/inpaint block
   (structure mirrors `sdxl_backend` lines 939–1010):
   - generate the seeded fp32 unit prior as today;
   - `init_latent`/`mask_t`/`sigma_sched = sigmas` defaults; when
     `request.init_image is not None`: 16-snap-resize guard,
     `_encode_init`, `t_enc`/truncation,
     `noise = init_latent + noise * sigma_sched[0]`, then (if
     `request.init_mask`) build `mask_t` and blend the protected region to
     the clean init latent;
   - else (txt2img): `noise = noise * float(sigmas[0])` (unchanged).
3. In `model_fn`: compute `xwork` (per-step lock, item 4 above) and pass
   `xwork` — not `x` — to the transformer; return the locked x0:
   `x0 = xwork - sigma * v`, then
   `x0 = x0 * (1.0 - mask_t) + init_latent * mask_t` when `mask_t is not
   None`.
4. `self._steps_total = max(1, len(sigma_sched) - 1)`; sample with
   `kdiff.sample_euler(model_fn, noise, sigma_sched, extra_args={})`.
5. Final lock (item 5) right after sampling, before the existing
   `x / scaling_factor + shift_factor` decode.

## Deferred / follow-ups

- Soft inpainting for Flux (the `latent_blend` x0-blend path is
  family-agnostic and could be reused; Forge's post-sample pixel composite
  too) — secondary.
- SAG: not applicable (SDXL UNet mid-block self-attention hook).
- UI: the soft-inpaint and SAG checkboxes are silently ignored for Flux
  until the above land (same status as the already-deferred "hide the LoRA
  section for flux" item); gate them off for the flux family when that UI
  work happens.
- fp8/nf4 `.safetensors` Flux checkpoints (separate load path).

## Verification

1. New `dev/smoke_flux_img2img.py` (mirror of `dev/smoke_sdxl_inpaint.py`):
   load the flux model via `ModelRegistry(load_config(), gpus=[1])`, pick
   the entry with `family == "flux"`, then:
   - **[1] txt2img baseline** 512×512 ×15 (regression: unchanged path).
   - **[2] img2img**: `init_image=[1]`, no mask, denoise 0.75, seed 43 →
     output must differ from the init image (structure/PSNR check), save.
   - **[3] low strength**: same but denoise 0.1 → output stays close to the
     init image (strength direction sanity).
   - **[4] "inpaint masked" convention**: `init_mask` all-255, denoise 0.6 →
     whole image regenerated (differs from init).
   - **[5] "inpaint not masked" convention**: protect the left half
     (`init_mask` 0 on the left, 255 on the right) → left half of the
     output stays close to the init left half (per-pixel max/PSNR check),
     right half changes.
   - **[6] determinism**: maskless img2img, same seed twice → identical
     PNG (masked inpaint is intentionally not reproducible, item 4).
   Monitor peak VRAM (the encode pass adds a small transient working set on
   top of the ~13.1 GB sampling peak).
2. Run the unit + integration suites
   (`uv run --extra dev pytest tests/unit tests/integration`) — no
   regressions; the MockBackend mode tests continue to cover the API/mode
   plumbing.
3. Visual: one real "inpaint not masked" run in the UI on a canvas image
   with the flux model selected (denoise 0.75), and one masked inpaint with
   a drawn mask.
