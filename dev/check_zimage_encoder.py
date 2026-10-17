"""Layer-by-layer comparison of imagagen's Qwen3-4B Z-Image text encoder
against HuggingFace transformers' reference `Qwen3Model`.

Our model's per-layer outputs are recomputed by driving `Qwen3Model`'s own
submodules (embeddings, rope, decoder layers) so every layer can be compared
against the corresponding HF hidden state; the first divergent layer pinpoints
the bug (rope layout / causal mask / norms / fp8 scales).

    .venv/bin/python dev/check_zimage_encoder.py [device]
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from imagagen.encoders.qwen3 import (  # noqa: E402
    COMPUTE_DTYPE,
    load_zit_encoder,
    zimage_prompt_template,  # noqa: E402
)

DEVICE = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
ENC = ROOT / "infra" / "zit_encoder.safetensors"
PROMPT = "a shiny red sports car on a wet city street at night"
FULL_RUN = "--full" in sys.argv


def rel(a, b):
    a = a.float()
    b = b.float()
    d = (a - b)
    return (d.abs().max().item(), d.abs().mean().item(),
            torch.nn.functional.cosine_similarity(
                a.flatten()[None], b.flatten()[None], dim=1).item(),
            b.abs().mean().item())


def main():
    from safetensors import safe_open
    from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model

    text = zimage_prompt_template().format(PROMPT)
    tok = Qwen2Tokenizer.from_pretrained(str(ROOT / "res" / "tokenizer_zimage"))
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
    print(f"tokens: {ids.shape[1]}")

    file = {}
    with safe_open(str(ENC), framework="pt") as f:
        for k in f.keys():
            if k != "scaled_fp8":
                file[k] = f.get_tensor(k)

    # --- ours: drive the layers one by one ----------------------------
    ours = load_zit_encoder(str(ENC), DEVICE)
    with torch.inference_mode():
        x = torch.nn.functional.embedding(
            ids.to(DEVICE), ours.embed_tokens.weight.to(COMPUTE_DTYPE)
        )
        cos_, sin_ = ours._rope(x.shape[1], x.device, COMPUTE_DTYPE)
        mine = [x]
        for layer in ours.layers:
            x = layer(x, cos_, sin_)
            mine.append(x)
        ours_normed = ours.norm(x)
    ours.cpu()
    del ours
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # --- the HF reference --------------------------------------------
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
            # only scale_weight belongs on the weight; scale_input is the
            # activation quantization step and cancels
            sw = file[base + ".scale_weight"].float()
            t = (t.to(torch.float32) * sw).to(torch.bfloat16)
        else:
            t = t.to(torch.bfloat16)
        sd[k] = t
    ref.load_state_dict(sd, strict=False)
    del sd, file
    ref.eval()
    ref.to(device=DEVICE, dtype=torch.bfloat16)
    with torch.inference_mode():
        out = ref(ids.to(DEVICE), output_hidden_states=True)
    hs = [h for h in out.hidden_states]

    print(f"hf hidden_states: {len(hs)}, ours: {len(mine)} (+1 normed)")
    n = min(len(hs), len(mine))
    print("layer-by-layer (ours[k] = output of layer k-1, same index as HF):")
    for k in range(n):
        mx, mean, c, refmag = rel(mine[k], hs[k])
        flag = "" if c > 0.9999 else ("   <-- DIVERGES" if k < 3 or c < 0.99 else "")
        print(f"  [{k:2d}] cos {c:.6f}  max|d| {mx:.3e}  mean|d| {mean:.3e}"
              f"  ref mean|x| {refmag:.3f}{flag}")
        if not FULL_RUN and k > 3 and c < 0.9999:
            print("  (rest suppressed; pass --full)")
            break
    mx, mean, c, refmag = rel(ours_normed, hs[-1])
    print(f"  final norm: cos {c:.6f} max|d| {mx:.3e} mean|d| {mean:.3e}")


if __name__ == "__main__":
    main()
