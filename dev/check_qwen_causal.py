"""Focused check: is imagagen's Qwen3 attention causal?

Runs our Qwen3Model with the SDPA call patched to be causal / non-causal and
compares the first few layer outputs against the HF reference hidden states.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.encoders import qwen3 as qe  # noqa: E402
from imagagen.encoders.qwen3 import (  # noqa: E402
    COMPUTE_DTYPE,
    load_zit_encoder,
    zimage_prompt_template,  # noqa: E402
)

DEVICE = torch.device("cuda:0")
ENC = ROOT / "infra" / "zit_encoder.safetensors"
PROMPT = "a shiny red sports car on a wet city street at night"

CAUSAL = False


def patched_forward(self, x, cos, sin):
    """A copy of Qwen3Attention.forward with an explicit is_causal flag."""
    b, s, _ = x.shape
    hd = self.head_dim
    xq = self.q_norm(self.q_proj(x).view(b, s, self.n_heads, hd))
    xk = self.k_norm(self.k_proj(x).view(b, s, self.n_kv_heads, hd))
    xv = self.v_proj(x).view(b, s, self.n_kv_heads, hd)
    xq = qe._apply_rotary(xq, cos, sin)
    xk = qe._apply_rotary(xk, cos, sin)
    xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
    n_rep = self.n_heads // self.n_kv_heads
    if n_rep > 1:
        xk = xk.repeat_interleave(n_rep, dim=1)
        xv = xv.repeat_interleave(n_rep, dim=1)
    if CAUSAL:
        out = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
    else:
        # what the current implementation does: no mask at all
        out = F.scaled_dot_product_attention(xq, xk, xv)
    out = out.transpose(1, 2).reshape(b, s, self.n_heads * hd)
    return self.o_proj(out)


def run_mine(ids, layers=4):
    """Return the outputs of the first `layers` decoder layers."""
    model = load_zit_encoder(str(ENC), DEVICE)
    out = []
    with torch.inference_mode():
        x = F.embedding(ids.to(DEVICE), model.embed_tokens.weight.to(COMPUTE_DTYPE))
        cos, sin = model._rope(x.shape[1], x.device, COMPUTE_DTYPE)
        for i, layer in enumerate(model.layers):
            x = layer(x, cos, sin)
            out.append(x)
            if i + 1 >= layers:
                break
    model.cpu()
    del model
    torch.cuda.empty_cache()
    return out


def main():
    global CAUSAL
    from safetensors import safe_open
    from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model

    text = zimage_prompt_template().format(PROMPT)
    tok = Qwen2Tokenizer.from_pretrained(str(ROOT / "res" / "tokenizer_zimage"))
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids

    file = {}
    with safe_open(str(ENC), framework="pt") as f:
        for k in f.keys():
            if k != "scaled_fp8":
                file[k] = f.get_tensor(k)

    cfg = Qwen3Config(
        vocab_size=151936, hidden_size=2560, intermediate_size=9728,
        num_hidden_layers=36, num_attention_heads=32, num_key_value_heads=8,
        head_dim=128, max_position_embeddings=40960, rms_norm_eps=1e-6,
        rope_theta=1_000_000.0, attention_bias=False, tie_word_embeddings=False)
    ref = Qwen3Model(cfg)
    sd = {}
    for k, _v in ref.state_dict().items():
        key = k if k.startswith("model.") else "model." + k
        t = file.get(key)
        if t is None:
            continue
        if t.dtype == torch.float8_e4m3fn:
            base = key[:-len(".weight")]
            t = (t.to(torch.float32)
                 * file[base + ".scale_weight"].float()).to(torch.bfloat16)
        else:
            t = t.to(torch.bfloat16)
        sd[k] = t
    ref.load_state_dict(sd, strict=False)
    del sd
    ref.eval()
    ref.to(device=DEVICE, dtype=torch.bfloat16)
    with torch.inference_mode():
        hs = ref(ids.to(DEVICE), output_hidden_states=True).hidden_states
    ref.cpu()
    del ref
    torch.cuda.empty_cache()

    orig = qe.Qwen3Attention.forward
    for causal in (False, True):
        CAUSAL = causal
        qe.Qwen3Attention.forward = patched_forward
        mine = run_mine(ids, layers=6)
        qe.Qwen3Attention.forward = orig
        print(f"ours with is_causal={causal}:")
        for k, m in enumerate(mine, start=1):
            r = hs[k]
            d = (m.float() - r.float())
            c = torch.nn.functional.cosine_similarity(
                m.float().flatten()[None], r.float().flatten()[None], dim=1).item()
            print(f"   layer{k - 1} out vs hf hs[{k}]: cos {c:.6f} "
                  f"max|d| {d.abs().max().item():.3e} mean|d| {d.abs().mean().item():.3e}"
                  f" (ref mean|x| {r.float().abs().mean().item():.3e})")
        print()


if __name__ == "__main__":
    main()
