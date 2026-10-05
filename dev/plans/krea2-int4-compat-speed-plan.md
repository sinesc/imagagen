# Krea 2 int4 residency, checkpoint compat and speed plan

Goals, in priority order: (1) downloaded checkpoints and LoRAs "just work";
(2) VRAM headroom for large-image generation/inpainting (3440x1440);
(3) faster generation; (4) quality. Pixel-identity with reference
implementations is explicitly NOT a goal — own-smoke comparison is the
quality bar.

State this plan builds on: `krea2_finetuned_int4.safetensors` loads since
5cb2be0, but `_load_krea2_state_dict` unpacks its nibble-packed
`convrot_w4a4` weights to int8, so the GPU-resident transformer is ~12.8
GiB — identical to the int8 checkpoint. The int4 packing only saves disk.
`Int8Linear` (krea2_transformer.py) keeps int8 `[out,in]` + f32
`weight_scale [out,1]` resident and dequantizes to bf16 per op; ConvRot
(group 256) rotates activations to cancel the stored weight rotation.
VAE tiling for Krea 2 already exists (krea2_backend: `use_tiling` above
2 MP, Flux tile sizes). `gguf_ops.Fp8Linear` exists (Flux). Turbo/raw is a
per-model `ModelConfig.turbo` flag, defaulting False for unconfigured
models (model_configs.py). torch is 2.13.0+cu129.

## Phase 1 — packed int4 resident on the GPU (VRAM, do first)

Keep the nibble-packed bytes resident and unpack per-op on the GPU.
Resident transformer drops to ~7 GiB (blocks ~3.4 GiB packed + txtfusion
bf16 ~0.7 GiB + plain bf16 sections ~1 GiB); parked RAM footprint and
PCIe swap traffic halve with it. Numerics stay bit-identical to today
(GPU unpack of exact int8 values; same dequantize, same bf16 GEMM).

### 1a. `Int4Linear(Int8Linear)` in `krea2_transformer.py`

Only the weight storage differs — inherit everything else (ConvRot
activation rotation, `set_lora_deltas` A-rotation, `weight_scale`
handling, `_apply` scale transport):

```python
class Int4Linear(Int8Linear):
    """int8 [out, in//2] nibble-packed weight (low nibble = even input
    column, signed nibbles); unpacks to int8 on the GPU per op, then the
    inherited per-row scale + bf16 GEMM path."""

    def _dequantize(self, w):
        # signed nibble extraction via arithmetic shifts on int8:
        # hi = w >> 4; lo = (w << 4) >> 4  (no int32 widening)
        hi = w >> 4
        lo = (w << 4) >> 4
        full = torch.stack([lo, hi], dim=-1).reshape(*w.shape[:-1], -1)
        return full.to(self.computation_dtype) * \
            self.weight_scale.to(self.computation_dtype)
```

Verify torch int8 `>>`/`<<` are arithmetic and stay int8 on CUDA (torch
may upcast int8 bitwise ops — if it does, cast back and measure; the
int16 path is the fallback). Transient per GEMM grows from ~2 B/weight
(bf16 copy) to ~3 B/weight (int8 unpack + bf16 copy); resident halves.

### 1b. Wire the layout choice through load

- `Krea2Transformer.__init__` gains `quant_cls=Int8Linear`; blocks are
  built with it (txtfusion/first/last/... stay `DeltaLinear`).
- `_load_krea2_state_dict` returns `(sd, layout)` where layout is
  `"int8" | "int4" | "bf16"`; for `"int4"` it NO LONGER unpacks the
  blocks weights — it keeps `weight [out, in/2]` int8 and only reshapes
  `weight_scale` to `(out, 1)`. txtfusion handling is unchanged (dequant
  + de-rotate to bf16 at load, scales dropped).
- `Krea2Backend.__init__` builds `Krea2Transformer(quant_cls=...)` from
  the layout. The bf16-cast exclusion uses `isinstance(mod, Int8Linear)`
  already, so Int4Linear is covered by inheritance.
- LoRA: nothing to do — `Int4Linear` inherits the plain-space delta
  contract (A rotated into ConvRot space at install).

### 1c. Tests & verification

- Unit (extend `tests/unit/test_krea2_checkpoint_loader.py` +
  `test_krea2_transformer.py`): pack→load→`Int4Linear.forward` ==
  `Int8Linear.forward` on the same logical weights; small-
  `Krea2Transformer(quant_cls=Int4Linear)` end-to-end load; the shift
  codec equals `_unpack_int4_row_major` on CPU.
- Live: load `krea2_finetuned_int4` and log the measured resident bytes
  (expect ~7 GiB vs the current ~12.8 GiB); `dev/smoke_krea2.py` render
  must be bit-identical to the current unpacked path (same weights, same
  GEMM).

## Phase 2 — checkpoint & LoRA compatibility (priority 1)

### 2a. Metadata-driven loader dispatch (krea2)

Generalize the layout detection in `_load_krea2_state_dict` to read the
per-tensor `comfy_quant` markers (currently skipped) and the
`_quantization_metadata` file metadata into one classifier:

| declared format | layout | modules |
|---|---|---|
| `int8_tensorwise` (+convrot) | int8 | `Int8Linear` |
| `convrot_w4a4` | int4 | `Int4Linear` |
| `float8_e4m3fn` | fp8 | `Fp8Linear`-style, NO ConvRot (weights unrotated; `convrot_group=None`) |

The fp8 row exists for the widely-distributed `krea2_*_fp8_scaled`
checkpoints (Comfy-Org): fp8 weights + one scalar `weight_scale` per
tensor, no rotation; txtfusion stays whatever the file declares. Add a
`Krea2Fp8Linear` (thin `Fp8Linear` subclass with the krea2
`DeltaLinear`-compatible delta contract — QuantLinear already provides
it) and a unit test per format on synthetic mini-checkpoints. Unknown
formats: raise with the observed marker JSON so the fix is obvious.

### 2b. Turbo auto-detection

`model_configs.py` default entries for unconfigured models:
`turbo = "turbo" in name.lower()` (krea2 family), so a CivitAI download
gets 8 steps / no-CFG / mu 1.15 with zero setup. Explicit `ModelConfig`
entries keep overriding. Unit test in the model-configs tests.

### 2c. LoRA rejection messages

`parse_krea2_lora` / `parse_flux_lora` / the SDXL loader: when ZERO keys
map, raise a named diagnosis instead of a generic mismatch — sniff
LoKr/LoHa (`lora_diff`, `dora` keys), GGUF-quantized LoRAs (header), and
say "unsupported LoRA format X" vs "no layers matched (is this a
Y-family LoRA?)". No new formats are added here — clear failure only.

### 2d. Compat check script

`dev/compat_check.py`: for every file under `models/`, read the
safetensors header only (no tensor load), classify the layout, map every
key against the family's module set, and print a per-file table
(family / layout / keys mapped / unmapped samples). Run manually after
adding any checkpoint; the fast header-only pass makes "does it just
work" a 10-second check.

## Phase 3 — wallpaper-size VRAM validation (3440x1440)

After Phase 1 lands, measure end-to-end on one 16 GiB card with the int4
checkpoint: t2i and one inpaint job at 3440x1440 (4.9 MP → tiled VAE
decode already engages at >2 MP; inpaint mostly rides the InpaintPlan
mask-box crop). Log peak VRAM at load, per-step, and at decode
(`torch.cuda.max_memory_allocated` + the existing vram probes). The
expected binding constraint is denoise activations, not weights: at
~78k combined tokens each bf16 hidden-state tensor is ~1 GB and a block
transiently holds several. If it does not fit, the follow-up (its own
plan then) is chunked/segmented attention or regional generation —
measure first, do not pre-build. Tune tile sizes only if decode is the
OOM site.

## Phase 4 — compute efficiency spike (flag-gated)

Per-step cost today is dominated by materializing a bf16 weight copy per
GEMM and running a bf16 GEMM. Spike: W8A8 tensor-core GEMM via
`torch._scaled_mm(a_int8, w_int8_rowmajor_t, scale_a_rowwise,
scale_w_rowwise, out_dtype=bf16)` — weights are already int8 + per-row
scales; activations get dynamic per-row int8 quantization (the ConvRot
rotation exists precisely to make activations quantizable). Steps:

1. Probe script: availability + speed of rowwise int8 `_scaled_mm` on
   the 6144/16384-wide shapes vs the current dequant+bf16 GEMM, and
   output deviation (cos, max-abs) on real activations.
2. If ≥20% step-time win and smoke renders look right: adopt behind an
   ini flag (`krea2-int8-gemm = off` default), Int8Linear/Int4Linear
   gain the alternate forward. Int4 unpacks to int8 on GPU and shares
   the same kernel.
3. If not: record the measurement in the memory files and close.

Cheap wins already covered: turbo auto-detection (2b) is a 1.5x step
reduction for turbo checkpoints. Explicitly deferred (revisit only if
Phase 4 underdelivers): true W4A4 MMA with activation quantization and
SVDQuant low-rank residuals — reference:
`dev/implementations/Krea-2-SVDQuant-ComfyUI` (reference-only, never
imported; copy needed pieces into `src/`).

## Commit sequence

1. `Int4Linear` + loader/transformer wiring + tests (Phase 1).
2. Loader dispatch + fp8 layout + turbo autodetect + LoRA messages +
   compat script (Phase 2, may split).
3. Wallpaper-size measurement: results into the memory files, config
   tweaks only if measured (Phase 3).
4. `_scaled_mm` spike as a dev/ script first; adoption only on evidence
   (Phase 4).
