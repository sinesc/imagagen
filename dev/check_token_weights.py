"""Token-weight (emphasis) check for the SDXL CLIP encoding (GPU).

Verifies ``SdxlBackend._encode_prompt_chunked``'s Forge-style token
weights (``(green:1.2)``):

  1. a weightless prompt is byte-identical to diffusers' ``encode_prompt``
     (the pre-token-weight contract is intact)
  2. ``"(green:1.0)"`` encodes exactly like the stripped ``"green"``
     (a weight of 1.0 takes the unweighted whole-prompt path)
  3. the per-token scaling lands on exactly the right positions:
         enc("(green:1.2)")[0, 1, :] == 1.2 * enc("green")[0, 1, :]
     with BOS/EOS/pad untouched and the pooled output untouched; a
     weighted token at the 75/76 window boundary scales in window 2
  4. two weighted segments scale independently; "(x)" == "(x:1.1)" and a
     different weight gives different embeddings
  5. a long weighted prompt (>75 content tokens) encodes to 2 windows

Usage (from the project root; needs a GPU and models/sdxl/*.safetensors):
    .venv/bin/python dev/check_token_weights.py [gpu-index]
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from imagagen.config import load_config  # noqa: E402
from imagagen.models import ModelRegistry  # noqa: E402


def enc(backend, prompt):
    return backend._encode_prompt_chunked(prompt, backend.text_encoder_device)


def diff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("gpu", nargs="?", type=int)
    args = ap.parse_args()

    reg = ModelRegistry(load_config(), gpus=[args.gpu] if args.gpu is not None else None)
    sdxl = [m for m in reg.models if m.family == "sdxl"]
    if not sdxl:
        print("no SDXL model in models/sdxl/ -- nothing to check")
        return 1
    backend = reg.backend_for(sdxl[0].id)
    dev = backend.text_encoder_device
    ok = True

    def check(label: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok &= cond
        print(f"  {'ok  ' if cond else 'FAIL'} {label}" + (f"  ({detail})" if detail else ""))

    # --- 1. weightless contract: byte-identical to diffusers encode_prompt ---
    print("1. weightless prompt vs diffusers encode_prompt")
    emb, pooled = enc(backend, "a red panda on a mossy log")
    d_emb, _d_neg, d_pooled, _d_neg_pooled = backend.pipe.encode_prompt(
        "a red panda on a mossy log", "", device=dev, num_images_per_prompt=1)
    check("prompt_embeds byte-identical", torch.equal(emb, d_emb),
          f"max|diff|={diff(emb, d_emb):.3e}")
    check("pooled byte-identical", torch.equal(pooled, d_pooled),
          f"max|diff|={diff(pooled, d_pooled):.3e}")

    # --- 2. weight 1.0 -> stripped text, unweighted path ---
    print("2. weight 1.0 behaves like the stripped plain prompt")
    for weighted, plain in [("little (green:1.0) men", "little green men")]:
        e1, p1 = enc(backend, weighted)
        e2, p2 = enc(backend, plain)
        check(f"{weighted!r} == {plain!r}",
              torch.equal(e1, e2) and torch.equal(p1, p2),
              f"max|diff|={diff(e1, e2):.3e}")

    # --- 3. exact per-token scaling, right positions, pooled untouched ---
    print("3. per-token scaling positions")
    w12 = torch.tensor(1.2, dtype=torch.float16, device=dev)
    e_w, p_w = enc(backend, "(green:1.2)")
    e_p, p_p = enc(backend, "green")
    check("content token scaled by 1.2", torch.equal(e_w[0, 1], e_p[0, 1] * w12))
    check("BOS/EOS/pad untouched",
          torch.equal(e_w[0, 0], e_p[0, 0]) and torch.equal(e_w[0, 2:], e_p[0, 2:]))
    check("pooled untouched by the weight", torch.equal(p_w, p_p))

    # weighted token at the window-1/2 boundary (content index 74 = the
    # last slot of window 1)
    long_w = "a " * 74 + "(b:1.2)" + "c " * 40
    long_p = "a " * 74 + "b " + "c " * 40
    e_lw, _ = enc(backend, long_w)
    e_lp, _ = enc(backend, long_p)
    check("long weighted prompt is 2 windows", e_lw.shape[1] == 154,
          f"shape={tuple(e_lw.shape)}")
    check("boundary token (pos 75) scaled by 1.2",
          torch.equal(e_lw[0, 75], e_lp[0, 75] * w12))
    mask = torch.ones(154, dtype=torch.bool, device=dev)
    mask[75] = False
    check("every other position untouched", torch.equal(e_lw[0][mask], e_lp[0][mask]))

    # --- 4. independent segments; default emphasis; weight sensitivity ---
    print("4. segment semantics")
    e2w, _ = enc(backend, "(green:1.2) (blue:0.5)")
    e2p, _ = enc(backend, "green blue")
    w05 = torch.tensor(0.5, dtype=torch.float16, device=dev)
    check("both segments scaled independently",
          torch.equal(e2w[0, 1], e2p[0, 1] * w12)
          and torch.equal(e2w[0, 2], e2p[0, 2] * w05))
    e_def, _ = enc(backend, "(green)")
    e_exp, _ = enc(backend, "(green:1.1)")
    check("(x) == (x:1.1)", torch.equal(e_def, e_exp))
    e_a, _ = enc(backend, "(green:1.2)")
    # 1.2 vs 1.3: far enough apart to survive fp16 rounding
    e_b, _ = enc(backend, "(green:1.3)")
    check("different weights -> different embeddings", not torch.equal(e_a, e_b),
          f"max|diff|={diff(e_a, e_b):.3e}")
    e_plain, _ = enc(backend, "green")
    check("weighted prompt differs from plain", not torch.equal(e_w, e_plain),
          f"max|diff|={diff(e_w, e_plain):.3e}")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
