# Reference-image support — phased plan

## Background (measured facts, research 2026-07)

Per-family capability for conditioning on a **reference image** (an image that
informs the generation but is not the img2img base):

| Family | Encoder | Native image path |
|---|---|---|
| Krea2 | Qwen3-VL-4B (`infra/qwen_3vl_4b.safetensors`) | **Yes, two paths.** (1) The checkpoint contains the full vision tower (315 `model.visual.*` keys, ~0.42 B params, fp8; we currently skip them at load). ComfyUI's `comfy/text_encoders/krea2.py` inherits `qwen3vl.py`, which injects image features at `vision_start/image_pad/vision_end` token positions (+ DeepStack injection at layers 5/11/17); the 12-tap conditioning then carries image-token embeddings through `txtfusion` unchanged. (2) `comfy/ldm/krea2/model.py::_forward` accepts `ref_latents` (reference VAE latents concatenated to the image-token sequence, indexed RoPE positions, optional timestep-zero) — Kontext/edit-style; works only if the checkpoint was trained with it. |
| Flux | T5-XXL + CLIP-Mistral | No encoder image path. Same `ref_latents` sequence-concat mechanism exists (that is Flux-Kontext), but our flux1-schnell checkpoint is not trained for it. Community IP-Adapter weights exist. |
| SDXL | CLIP ViT-bigG + ViT-L (text conditioning) | No native path. **IP-Adapter is the proven route**; Forge is vendored at `dev/implementations/stable-diffusion-webui-forge` (`extensions-builtin/sd_forge_ipadapter/`) as a port reference. SDXL checkpoints ship the CLIP-ViT **vision** tower (`conditioner.embedders.0.open_clip.model.visual.*`). |
| Z-Image | Qwen3-4B (text-only) | None. Checkpoint has no vision tower; the model was not trained with reference conditioning. Community IP-Adapter ports exist. |

**Chosen strategy:** a universal **VLM-caption tier** (works for every family
with one implementation, using the Qwen3-VL weights we already ship) plus a
**Krea2-native tier** (image tokens directly in the conditioning the DiT was
trained on). Captioning transfers semantics only (subject, style wording) —
not identity/palette/composition; that is accepted for phase 1–2 and is what
IP-Adapter (phase 3) would add later.

## Design decisions (resolved)

- **UI**: a second per-thumbnail checkbox, stacked **below the pin checkbox**
  in the top-right corner of each thumb ("reference image"). Single reference
  at a time: checking one clears the others. Checking also auto-pins the
  thumb (unpinned thumbs are dropped at submit; a reference must survive).
- **One reference per request** (v1). No strength slider in v1.
- **Caption policy (phase 1)**: caption is appended to the *positive* prompt
  after styles/wildcards; negative prompt untouched. Caption is deterministic
  (greedy decoding) and cached by image sha1, so generations stay
  reproducible; the caption text is recorded in the PNG metadata.
- **Where captioning runs**: on the generation worker thread of the GPU that
  will run the job, with the transformer parked (existing
  encode-swap/parking machinery). Krea2 reuses its resident encoder + a
  lazily-loaded vision tower (~0.45 GB).
- **Metadata**: a `Reference:` line in the parameters-chunk JSON (method +
  caption). PNG import records it as a style-like note only in v1 (the image
  itself cannot round-trip through the prompt text).

---

## Phase 0 — UI + plumbing (reference rides along, inert)

No backend behaviour change; the reference is carried, displayed, logged, and
recorded. Everything later phases need is in place.

1. **UI checkbox** (`src/imagagen/static/app.js`, `renderThumbs`):
   ```js
   const ref = el("input", { type: "checkbox", class: "thumb-ref",
     title: "Reference image: describe this image into the prompt of the next generation",
     "aria-label": "Reference image" });
   ref.checked = state.referenceImageId === t.image_id;
   ref.onchange = () => {
     state.referenceImageId = ref.checked ? t.image_id : null;
     if (ref.checked && !t.pinned) { /* pin it (same API as pinToggle) */ }
     renderThumbs(); // single-reference invariant: re-render resets siblings
   };
   // same hover-preview handlers as the pin checkbox (thumbPreviewShow/Hide)
   item.append(image, pin, ref);
   ```
   CSS (`style.css`): `.thumb-ref` positioned below `.thumb-pin` in the
   top-right corner (pin keeps its slot; ref gets `top: calc(corner + 1.4rem)`
   or a small flex column — verify with Playwright `bounding_box()` per the
   1.75rem UI convention).
2. **Request plumbing**:
   - `app.py` `GenerateBody`: `reference_image_id: str | None = None`.
   - `queue.py`: resolve via the canvas store (same path as
     `/api/canvas/image.png?id=`) to an `HxWx3 uint8` numpy array; 404 →
     clear + error status. Pass into
   - `backends/__init__.py` `GenerateRequest`: add
     `reference_image: np.ndarray | None = None` (in-process only, exactly
     like `init_image`; JSON callers leave it None).
   - `uistate.py`: persist `reference_image_id` (it is not a generation
     param, so it is allowed by the ui-state exclusions rule); on load, drop
     it if the image id no longer exists; `restoreQueueThumbs` re-checks the
     box when the id is in the restored set.
3. **Metadata**: `metadata.py` writes `Reference: caption (<method>)` when the
   backend reports one; `GenerateResult` gains `reference_caption: str | None`.
4. **MockBackend**: echo `reference_image is not None` into
   `reference_caption = "mock"` so Playwright tests can assert the round trip.
5. **Tests**: Playwright — check box on a thumb, submit, assert
   `reference_image_id` in the request body + single-reference invariant +
   auto-pin; unit test that `GenerateRequest.reference_image` round-trips
   through the queue.

Accept: full unit suite green; the checkbox persists across reload
(ui-state); generation output is unchanged (reference inert).

## Phase 1 — universal VLM captioning (all four families)

New `src/imagagen/encoders/qwen3vl_vision.py` (vision tower + preprocessing)
and `src/imagagen/reference.py` (captioner service + cache + application).

1. **Image preprocessing** — port `process_qwen2vl_images` from
   `dev/implementations/ComfyUI/comfy/text_encoders/qwen_vl.py` with the
   Qwen3-VL parameters used by `qwen3vl.py`: `patch_size=16`,
   `temporal_patch_size=2`, `merge_size=2`, `image_mean=[0.5]*3`,
   `image_std=[0.5]*3`, bilinear; min_pixels 3136, max_pixels ~12.8 M.
   A ≤448×448 image yields a 28×28 grid → **196 image tokens**.
2. **Vision tower** — port `Qwen3VLVisionModel` (from
   `comfy/text_encoders/qwen35.py::Qwen35VisionModel` +
   `qwen3vl.py::Qwen3VLVisionModel/Qwen3VLDeepstackMerger`):
   config `hidden_size=1024, intermediate_size=4096, depth=24, heads=16,
   patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
   out_hidden_size=2560, deepstack_visual_indexes=[5,11,17]`.
   Bidirectional (non-causal) attention with `cu_seqlens` (single image →
   one full block). Returns `merged` (n/4, 2560) + deepstack list (3 × n, 1024).
   Load `model.visual.*` keys from `infra/qwen_3vl_4b.safetensors` (fp8
   weights → the existing `Fp8Linear` cast-is-dequantize pattern; verify the
   stored dtype of visual projections at load time and reuse
   `Krea2Fp8Linear`-style handling if fp8).
3. **Token stream + injection** — caption template (fixed instruction;
   keep the shipped generation templates out of code/prompts — they contain
   the stop token):
   `<|im_start|>user\n<|vision_start|><|im_pad|>\n<|vision_end|>` +
   `Describe the image in detail: subject, clothing, art style, colors,
   composition, lighting.<|im_end|>\n<|im_start|>assistant\n`
   Tokenize; expand the single image-pad token (id 151655) to 196 copies
   between vision_start (151652) / vision_end (151653); at the embedding
   stage, scatter the `merged` rows into the text-embedding stream at the
   pad positions; after text-tower decoder layers 5/11/17 add the deepstack
   features at the same positions (Qwen3-VL DeepStack convention).
4. **Greedy generation** — reuse `Krea2Qwen3VLModel` (text tower) +
   `model.norm.weight` (currently skipped at load — load it for the caption
   path) + the **tied** lm_head (`logits = norm(h[-1]) @ embed_tokens.T`),
   greedy argmax decoding, stop at the end-of-text token (151645) /
   im_start (151644), cap 128 tokens. Deterministic: no sampling, no
   system prompt.
5. **Cache + service** (`src/imagagen/reference.py`):
   - `caption_reference(image_np) -> str`: sha1(image bytes) keyed JSON at
     `cache/reference_captions.json` (same location convention as the other
     caches).
   - Runs on the generation worker thread of the job's GPU (same executor
     as generation — the only thread that touches torch models): park the
     transformer (existing `_transformer_out` / parking pattern), load the
     text tower + vision tower (~4.6 GB peak, fits the 16 GB budget with
     the transformer parked), caption, unload. Krea2 reuses its resident
     text tower and loads only the vision tower lazily.
   - `apply_reference(request, caption)`:
     `request.prompt + " Referencing an image: " + caption + "."` —
     applied in each backend right before prompt encoding, after
     styles/wildcards. Report `report_phase(progress, PHASE_REFERENCE)`
     (new constant "phase:reference", UI label "Describing reference
     image").
6. **Backend wiring**: one call site per backend
   (`sdxl.py`, `flux.py`, `zimage`, `krea2`): if
   `request.reference_image is not None`, obtain the caption (cache hit =
   free) and append. `GenerateResult.reference_caption = caption`,
   `metadata.py` line `Reference: <caption> (caption)`.
7. **Preview endpoint**: `POST /api/reference/caption` (body
   `{image_id}`) → `{caption}; the UI fetches it on checkbox-check and
   puts the caption into the checkbox `title` tooltip, so the user sees
   what will be injected.

**Accept**: `dev/smoke_reference_caption.py` produces sensible captions for
3 sample images (pin the exact strings as regression fixtures); caption
generation is bit-stable across runs; SDXL + Krea2 generations with a
reference visibly follow the reference's subject while the no-reference
baseline is unchanged when the box is off; full unit suite green.

## Phase 2 — Krea2 native image conditioning

Image tokens inside the 12-tap conditioning the DiT was trained on.
Encoder-side only; the DiT consumes the conditioning stack unchanged.

1. `encoders/krea2.py`: stop skipping `model.visual.*` when a reference is
   in use (lazy load, keep resident); reuse the phase-1 vision tower and
   preprocessing.
2. Build the conditioning sequence with the image block **inside the Krea2
   chat template**: system turn, then the user turn starting with
   vision_start / 196 pad tokens / vision_end, then the user prompt text.
   The prefix strip must cut at the **second** im_start (the user turn) so
   the image tokens survive the strip — extend `template_prefix_cut`
   accordingly (the cut point is unchanged; the image block sits after it).
3. The flattened `(1, seq, 12*2560)` conditioning now contains image-token
   rows; `txtfusion`/`txtmlp` and the DiT are untouched (matches ComfyUI,
   where the image tokens flow through `encode_token_weights` unchanged).
4. **Verification (mandatory, per the Krea2 bisect lessons)**:
   - `dev/verify_krea2_reference.py`: run the vendored reference
     (`dev/implementations/krea-2/` + ComfyUI `qwen3vl.py` image path) on
     CPU/GPU against our port for one reference image + prompt; compare
     conditioning tensors (cos > 0.99) before rendering.
   - A/B render grid: prompt alone vs prompt + reference image; the
     reference content/style must appear. If the output is noise, bisect
     layer-by-layer (dev/scratch harness pattern from the int4 port).
5. UI: when the family is krea2, the checkbox tooltip says the image is
   used natively; PNG metadata method = `native`.

**Accept**: reference-bisect parity + visible A/B effect on both shipped
Krea2 checkpoints (int8 and int4).

## Phase 3 — SDXL IP-Adapter (optional, separate approval)

Identity/style transfer, the only non-text route for SDXL.

- Port Forge's `sd_forge_ipadapter` (`lib_ipadapter/IPAdapterPlus.py`) into
  the SDXL backend; image encoder = CLIP ViT-H/14 (ComfyUI-style
  `clip_vision` file, ~1 GB fp16) — do not reuse the in-checkpoint bigG
  tower (Forge's adapter is trained against ViT-H features).
- Injection into the UNet cross-attention K/V (per-block), weight slider
  0..2 in the UI next to the reference checkbox; VRAM budget check per the
  16 GB residency rules (adapter + ViT-H co-resident with SDXL).
- Metadata method = `ipadapter`.

## Phase 4 — Kontext-style reference latents (deferred)

`ref_latents` sequence concatenation (ComfyUI `flux/model.py` /
`krea2/model.py` pattern: patchified ref latents appended to the image
tokens with indexed RoPE positions + timestep-zero). Implement only when a
checkpoint trained for it is added (Flux-Kontext-class, Krea2-edit-class);
predictable garbage on schnell/Z-Image/Turbo.

## Test plan (summary)

- Unit: vision-tower shape/determinism tests; caption cache round-trip;
  `GenerateRequest.reference_image` queue plumbing; metadata `Reference:`
  line; MockBackend echo.
- Playwright (mock server): checkbox layout (bounding_box), single-reference
  invariant, auto-pin, request payload, ui-state persistence.
- Dev scripts: `dev/smoke_reference_caption.py` (phase 1),
  `dev/verify_krea2_reference.py` (phase 2).
- Use vision (read the PNG outputs) for qualitative A/B checks.

## Files touched (summary)

| Phase | Files |
|---|---|
| 0 | `static/app.js`, `static/style.css`, `app.py`, `queue.py`, `backends/__init__.py`, `uistate.py`, `metadata.py`, tests |
| 1 | `encoders/qwen3vl_vision.py` (new), `reference.py` (new), all four backends, `metadata.py`, `app.py` (caption endpoint), `dev/smoke_reference_caption.py` |
| 2 | `encoders/krea2.py`, `backends/krea2/*`, `dev/verify_krea2_reference.py` |
| 3 | `backends/sdxl.py`, new `ipadapter.py`, LoRA-style model dir for the adapter |
