"""Compare imagagen's ZImageTransformer against the UNMODIFIED ComfyUI
reference NextDiT (dev/implementations/ComfyUI/comfy/ldm/lumina/model.py,
loaded through the tiny `comfy_ref_stub`).

Uses a small config so it runs on the CPU in fp32: any structural difference
(rope layout, position ids, padding, stream order, modulation, unpatchify)
shows up immediately, and the state-dict key sets must match exactly.

    .venv/bin/python dev/check_zimage_vs_ref.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dev"))

import comfy_ref_stub  # noqa: E402

ref = comfy_ref_stub.load_reference(
    str(ROOT / "dev" / "implementations" / "ComfyUI" / "comfy" / "ldm" / "lumina" / "model.py"))

from imagagen.backends.zimage.transformer import ZImageTransformer  # noqa: E402

# a small but structurally faithful config (axes_dims sum == dim // n_heads,
# ffn = int(8/3 * dim) rounded up to a multiple of 256). dim must be >= 256:
# the z_image_modulation adaLN takes min(dim, 256) inputs and the timestep
# embedding is 256-dim (head_dim 128 and axes [32, 48, 48] match the real
# checkpoint exactly).
DIM, HEADS = 384, 3
OURS_CFG = dict(
    patch_size=2, in_channels=16, dim=DIM, n_layers=2, n_refiner_layers=2,
    n_heads=HEADS, n_kv_heads=HEADS, ffn_hidden=int(8 / 3 * DIM),
    norm_eps=1e-5, cap_feat_dim=96, axes_dims=[32, 48, 48],
    rope_theta=256.0, pad_tokens_multiple=32, time_scale=1000.0)
REF_CFG = dict(
    patch_size=2, in_channels=16, dim=DIM, n_layers=2, n_refiner_layers=2,
    n_heads=HEADS, n_kv_heads=HEADS, multiple_of=256,
    ffn_dim_multiplier=8.0 / 3.0, norm_eps=1e-5, qk_norm=True,
    cap_feat_dim=96, axes_dims=[32, 48, 48], axes_lens=[1536, 512, 512],
    rope_theta=256.0, z_image_modulation=True, time_scale=1000.0,
    pad_tokens_multiple=32, dtype=torch.float32, device=torch.device("cpu"),
    operations=torch.nn)


def build():
    torch.manual_seed(0)
    ours = ZImageTransformer(**OURS_CFG)
    reference = ref.NextDiT(**REF_CFG)

    so, sr = dict(ours.state_dict()), dict(reference.state_dict())
    only_ours = sorted(set(so) - set(sr))
    only_ref = sorted(set(sr) - set(so))
    print(f"state dict: ours {len(so)} keys, ref {len(sr)} keys")
    if only_ours:
        print("  ONLY IN OURS:", only_ours[:10])
    if only_ref:
        print("  ONLY IN REF :", only_ref[:10])

    # copy the reference weights into ours (shape-compatible by construction)
    bad = []
    for k, v in sr.items():
        if k in so and so[k].shape == v.shape:
            ours.state_dict()[k].copy_(v)
        else:
            bad.append((k, tuple(v.shape)))
    if bad:
        print("  SHAPE/NAME PROBLEMS:", bad[:10])

    ours.eval()
    reference.eval()
    return ours, reference


def compare(ours, reference, hw=20, ncap=37, sigma=0.63, tag=""):
    torch.manual_seed(1)
    x = torch.randn(1, 16, hw, hw)
    context = torch.randn(1, ncap, OURS_CFG["cap_feat_dim"])
    t = torch.tensor([sigma])
    with torch.no_grad():
        out_ours = ours(x, t, context)
        out_ref = reference(x, t, context, None)[:, :, :hw, :hw]
    d = (out_ours - out_ref).abs()
    rel = d.max().item() / out_ref.abs().max().item()
    print(f"  {tag} shape ours {tuple(out_ours.shape)} ref {tuple(out_ref.shape)}: "
          f"max|diff| {d.max().item():.3e}  mean {d.mean().item():.3e}  "
          f"rel {rel:.3e}  ref rms {out_ref.pow(2).mean().sqrt().item():.4f}")
    return d.max().item()


def stage_probe(ours, reference):
    """Compare the reference's own intermediate tensors with ours (the rope
    position ids and the padded/refined streams) to locate a divergence."""
    from imagagen.backends.zimage import transformer as zt

    torch.manual_seed(1)
    x = torch.randn(1, 16, 20, 20)
    context = torch.randn(1, 37, OURS_CFG["cap_feat_dim"])
    t = torch.tensor([0.63])

    # --- reference stages
    with torch.no_grad():
        t_ref = reference.t_embedder((1.0 - t) * reference.time_scale, dtype=x.dtype)
        embeds, _mask, _sizes, cap_size, freqs, _tz = reference.patchify_and_embed(
            x, context, None, t_ref, None)
        print("  ref : joint", tuple(embeds.shape), "freqs", tuple(freqs.shape),
              "cap_len", cap_size)
    # --- our stages: spy on the rope embedder and the joint concat
    calls = []
    orig_embed = zt.EmbedND.forward

    def spy(self, ids):
        out = orig_embed(self, ids)
        calls.append((ids.clone(), out))
        return out

    zt.EmbedND.forward = spy
    with torch.no_grad():
        ours(x, t, context)
    zt.EmbedND.forward = orig_embed
    for i, (ids, emb) in enumerate(calls):
        print(f"  ours: rope call {i}: ids {tuple(ids.shape)} emb {tuple(emb.shape)} "
              f"t-axis {ids[0, :2, 0].tolist()}...{ids[0, -2:, 0].tolist()}")


def main():
    ours, reference = build()
    print("\nend-to-end comparison (fp32 cpu):")
    worst = 0.0
    for (hw, ncap, sigma) in [(20, 37, 0.63), (16, 32, 1.0), (18, 5, 0.25),
                              (32, 64, 0.87)]:
        worst = max(worst, compare(ours, reference, hw=hw, ncap=ncap, sigma=sigma,
                                   tag=f"hw={hw} ncap={ncap} sigma={sigma}:"))
    print(f"\nworst max|diff| = {worst:.3e}")
    print("\nstage probe:")
    stage_probe(ours, reference)


if __name__ == "__main__":
    main()
