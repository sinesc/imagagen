"""Functional test of the Z-Image Qwen3 scaled-fp8 dequantization.

The Kijai scaled-fp8 checkpoint stores, per linear: `weight` (float8_e4m3fn),
`scale_weight` (fp32) and `scale_input` (bf16). ComfyUI quantizes the
ACTIVATION with input_scale (q = x / input_scale, clamped to the fp8 range)
and multiplies the GEMM result by input_scale * weight_scale -- so the
activation scale CANCELS and the effective weight is `W_fp8 * scale_weight`.

imagagen's ScaledFp8Linear instead pre-MULTIPLIES the activation by
scale_input, which leaves every layer output scaled by an arbitrary
per-layer factor (0.003 ... 0.85 here). That makes the text conditioning
prompt-dependent but semantically meaningless -- which matches the
"seed dominates, prompt barely matters" symptom.

Run the encoder as a language model (its lm_head is tied to the embedding
matrix) to settle which reading is right: a correctly dequantized Qwen3
predicts the continuation of a sentence.

    .venv/bin/python dev/check_qwen_lm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen import qwen3_encoder as qe  # noqa: E402
from imagagen.qwen3_encoder import (COMPUTE_DTYPE,  # noqa: E402
                                    load_zit_encoder,
                                    zimage_prompt_template)
from transformers import Qwen2Tokenizer  # noqa: E402

DEVICE = "cuda:0"
ENC = ROOT / "infra" / "zit_encoder.safetensors"
CTX = "The capital of France is"


def lm_probe(model, tok, text=CTX):
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(DEVICE)
    with torch.inference_mode():
        x = torch.nn.functional.embedding(ids, model.embed_tokens.weight.to(COMPUTE_DTYPE))
        cos, sin = model._rope(x.shape[1], x.device, COMPUTE_DTYPE)
        for layer in model.layers:
            x = layer(x, cos, sin)
        h = model.norm(x)
        logits = h @ model.embed_tokens.weight.to(COMPUTE_DTYPE).t()
    top = logits[0, -1].topk(5)
    print(f"  ctx {text!r} -> next tokens: "
          + ", ".join(repr(tok.decode([i])) for i in top.indices.tolist()))
    print(f"    logit rms {logits.float().pow(2).mean().sqrt():.1f}  "
          f"hidden rms {h.float().pow(2).mean().sqrt():.3f}  "
          f"absmax {h.float().abs().max():.1f}")
    return tok.decode(top.indices[0].item())


def set_variant(model, variant):
    """variant: 'folded' (current code), 'weight-only', 'divided'."""
    for m in model.modules():
        if isinstance(m, qe.ScaledFp8Linear):
            if not hasattr(m, "_orig_si"):
                m._orig_si = m.scale_input.detach().float().clone()
            si = m._orig_si
            val = {"folded": si, "weight-only": torch.ones_like(si),
                   "divided": 1.0 / si}[variant].to(m.scale_input.dtype)
            m.scale_input.data = val


def main():
    tok = Qwen2Tokenizer.from_pretrained(str(ROOT / "res" / "tokenizer_zimage"))
    model = load_zit_encoder(str(ENC), DEVICE)

    # what do the fp8 weights look like under each convention?
    q = model.layers[0].self_attn.q_proj
    w8 = q.weight.float()
    print(f"layer0 q_proj: fp8 rms {w8.pow(2).mean().sqrt():.2f} absmax {w8.abs().max():.1f} "
          f"scale_weight {q.scale_weight.float().item():.5f} "
          f"scale_input {q.scale_input.float().item():.5f}")
    print(f"  dequantized weight rms: weight*scale_weight = "
          f"{(w8 * q.scale_weight.float()).pow(2).mean().sqrt():.5f}, "
          f"both scales = {(w8 * q.scale_weight.float() * q.scale_input.float()).pow(2).mean().sqrt():.7f}")

    for variant in ("folded", "weight-only", "divided"):
        set_variant(model, variant)
        print(f"\n[{variant}]")
        for text in (CTX, "Once upon a", "One plus one equals"):
            print("   ", repr(text), "->", repr(lm_probe(model, tok, text)))
        # also the chat-templated prompt the DiT actually sees
        tpl = zimage_prompt_template()
        set_variant(model, variant)
        lm_probe(model, tok, tpl.format("a red car"))


if __name__ == "__main__":
    main()
