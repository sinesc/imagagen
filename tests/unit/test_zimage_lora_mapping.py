"""Key mapping for Z-Image LoRAs (imagagen.backends.zimage.lora).

Two on-disk formats are tested:
- the *official naming* asset ``loras/zimage/ZIT-shine.
  safetensors`` (rank 32, 30 main blocks, ``to_q``/``to_k``/``to_v`` pairs,
  no alpha tensors): our transformer fuses q/k/v into one ``qkv`` GEMM, so
  each of those keys maps to a row slice of the fused weight;
- the *export naming* asset ``loras/zimage/lora_3784692__zit1.safetensors``
  (rank 64): verbatim fused module paths (``attention.qkv``),
  ``lora_down``/``lora_up`` pairs over ALL blocks (main, refiners,
  embedders, final layer) plus raw additive ``.diff`` (weight) / ``.diff_b``
  (bias) deltas for the RMSNorm scales and biases.
"""

from collections import Counter
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from imagagen.backends.zimage.lora import (
    DIM,
    FFN,
    OUT,
    THID,
    ParamDelta,
    map_base_key,
    parse_zimage_lora,
)

ROOT = Path(__file__).resolve().parents[2] / "loras" / "zimage"
OFFICIAL = ROOT / "ZIT-shine.safetensors"
EXPORT = ROOT / "lora_3784692__zit1.safetensors"


class TestMapBaseKey:
    def test_qkv_fusion_slices(self):
        assert map_base_key("layers.0.attention.to_q") == \
            ("layers.0.attention.qkv", 0, DIM)
        assert map_base_key("layers.3.attention.to_k") == \
            ("layers.3.attention.qkv", DIM, 2 * DIM)
        assert map_base_key("layers.29.attention.to_v") == \
            ("layers.29.attention.qkv", 2 * DIM, 3 * DIM)

    def test_fused_qkv_is_the_whole_weight(self):
        # the export naming LoRAs the fused qkv directly (all 3*DIM rows)
        for kind in ("layers", "noise_refiner", "context_refiner"):
            assert map_base_key(f"{kind}.0.attention.qkv") == \
                (f"{kind}.0.attention.qkv", 0, 3 * DIM)

    def test_to_out_is_the_full_output_projection(self):
        # the official `to_out.0` (trailing .0 stripped before mapping) is
        # the fused attention.out projection (full width)
        assert map_base_key("layers.1.attention.to_out") == \
            ("layers.1.attention.out", 0, DIM)
        assert map_base_key("layers.1.attention.out") == \
            ("layers.1.attention.out", 0, DIM)

    def test_ffn(self):
        assert map_base_key("layers.2.feed_forward.w1") == \
            ("layers.2.feed_forward.w1", 0, FFN)
        assert map_base_key("layers.2.feed_forward.w2") == \
            ("layers.2.feed_forward.w2", 0, DIM)
        assert map_base_key("layers.2.feed_forward.w3") == \
            ("layers.2.feed_forward.w3", 0, FFN)
        assert map_base_key("context_refiner.1.feed_forward.w1") == \
            ("context_refiner.1.feed_forward.w1", 0, FFN)

    def test_adaln(self):
        # the block adaLN (official `adaLN_modulation`, export
        # `adaLN_modulation.0` -- trailing .0 stripped before mapping) is
        # the single Linear(256 -> 4*DIM)
        assert map_base_key("layers.4.adaLN_modulation") == \
            ("layers.4.adaLN_modulation.0", 0, 4 * DIM)
        assert map_base_key("noise_refiner.0.adaLN_modulation") == \
            ("noise_refiner.0.adaLN_modulation.0", 0, 4 * DIM)

    def test_globals(self):
        # the export naming LoRAs the embedders / t embedder / final layer
        # with verbatim module paths
        assert map_base_key("x_embedder") == ("x_embedder", 0, DIM)
        assert map_base_key("cap_embedder.1") == ("cap_embedder.1", 0, DIM)
        # mlp.0 (trailing .0 stripped before mapping) -> the first Linear
        assert map_base_key("t_embedder.mlp") == \
            ("t_embedder.mlp.0", 0, THID)
        assert map_base_key("t_embedder.mlp.2") == \
            ("t_embedder.mlp.2", 0, 256)
        assert map_base_key("final_layer.linear") == \
            ("final_layer.linear", 0, OUT)
        assert map_base_key("final_layer.adaLN_modulation.1") == \
            ("final_layer.adaLN_modulation.1", 0, DIM)

    def test_unknown_keys_raise(self):
        with pytest.raises(ValueError, match="bogus"):
            map_base_key("layers.0.bogus")
        with pytest.raises(ValueError, match="x_pad_token"):
            map_base_key("x_pad_token")
        with pytest.raises(ValueError, match="bogus"):
            map_base_key("bogus")


class TestParseOfficialLora:
    """The official-naming asset: rank 32, no alpha, no direct deltas."""

    def test_pairs_all_mapped_rank32_scale1(self):
        assert OFFICIAL.is_file(), f"test asset missing: {OFFICIAL}"
        specs, deltas = parse_zimage_lora(OFFICIAL)
        # 30 main blocks x 8 mappable pairs (q/k/v/out + w1/w2/w3 + adaLN)
        assert len(specs) == 30 * 8
        assert deltas == []
        for s in specs:
            assert s.A.shape[0] == 32  # rank 32
            assert s.scale == pytest.approx(1.0)  # no .alpha tensors
            assert s.B.shape[0] == s.r1 - s.r0
            assert s.B.shape[1] == s.A.shape[0]
            assert not s.swap

    def test_module_coverage(self):
        specs, _ = parse_zimage_lora(OFFICIAL)
        mods = Counter(s.module for s in specs)
        # per block: one qkv (3 slices), one out, w1/w2/w3, adaLN
        assert sum(1 for m in mods if ".attention.qkv" in m) == 30
        assert sum(1 for m in mods if ".attention.out" in m) == 30
        assert sum(1 for m in mods if ".feed_forward." in m) == 90
        assert sum(1 for m in mods if ".adaLN_modulation.0" in m) == 30
        qkv = [s for s in specs if s.module.endswith(".qkv")]
        assert {s.r0 for s in qkv} == {0, DIM, 2 * DIM}
        assert all(s.r1 - s.r0 == DIM for s in qkv)


class TestParseExportLora:
    """The export-naming asset: fused paths, lora_down/up, .diff/.diff_b."""

    def test_every_key_is_consumed(self):
        assert EXPORT.is_file(), f"test asset missing: {EXPORT}"
        specs, deltas = parse_zimage_lora(EXPORT)
        # 208 A/B pairs: 30 main blocks x 6 (qkv/out/w1/w2/w3/adaLN) +
        # refiners (5 context x 2 + 6 noise x 2) + final (2) + embedders (4)
        assert len(specs) == 208
        # 243 direct deltas: 210 weight (.diff: 7/block x 30 main +
        # 6 x 2 refiners + the cap RMSNorm) + 33 bias (.diff_b)
        assert len(deltas) == 243
        assert sum(1 for d in deltas if d.attr == "weight") == 205
        assert sum(1 for d in deltas if d.attr == "bias") == 38
        for s in specs:
            assert s.A.shape[0] == 64  # rank 64
            assert s.scale == pytest.approx(1.0)
            assert s.B.shape[0] == s.r1 - s.r0
            assert s.B.shape[1] == s.A.shape[0]
        for d in deltas:
            assert d.attr in ("weight", "bias")

    def test_pair_targets(self):
        specs, _ = parse_zimage_lora(EXPORT)
        qkv = [s for s in specs if s.module.endswith(".attention.qkv")]
        assert len(qkv) == 34  # 30 main + 4 refiner blocks
        assert all((s.r0, s.r1) == (0, 3 * DIM) for s in qkv)
        final = [s for s in specs if s.module == "final_layer.linear"]
        assert [(s.r0, s.r1) for s in final] == [(0, OUT)]
        assert all(s.B.shape[0] == s.r1 - s.r0 for s in specs)

    def test_direct_delta_targets(self):
        _, deltas = parse_zimage_lora(EXPORT)
        by = {(d.module, d.attr) for d in deltas}
        # the block norms (weight deltas) and biases
        assert ("layers.0.attention.q_norm", "weight") in by
        assert ("layers.0.attention.k_norm", "weight") in by
        assert ("layers.0.attention_norm1", "weight") in by
        assert ("layers.0.ffn_norm2", "weight") in by
        assert ("layers.0.adaLN_modulation.0", "bias") in by
        assert ("noise_refiner.1.adaLN_modulation.0", "bias") in by
        # the global embedders / final layer
        assert ("cap_embedder.0", "weight") in by
        assert ("cap_embedder.1", "bias") in by
        assert ("x_embedder", "bias") in by
        assert ("t_embedder.mlp.0", "bias") in by
        assert ("t_embedder.mlp.2", "bias") in by
        assert ("final_layer.linear", "bias") in by
        assert ("final_layer.adaLN_modulation.1", "bias") in by
        # context refiners have NO adaLN -> no adaLN delta keys for them
        assert not any(m.startswith("context_refiner") and a == "bias"
                       and m.endswith("adaLN_modulation.0")
                       for m, a in by)


class TestParseEdgeCases:
    def test_unknown_key_raises_listed(self, tmp_path):
        # one mappable pair + one un-mappable key: the key must be named
        out = tmp_path / "bad.safetensors"
        save_file({
            "diffusion_model.layers.0.attention.to_q.lora_A.weight":
                torch.zeros(32, DIM),
            "diffusion_model.layers.0.attention.to_q.lora_B.weight":
                torch.zeros(DIM, 32),
            "diffusion_model.x_pad_token.lora_A.weight":
                torch.zeros(32, DIM),
        }, str(out))
        with pytest.raises(ValueError, match="x_pad_token"):
            parse_zimage_lora(out)

    def test_diff_keys_become_param_deltas(self, tmp_path):
        out = tmp_path / "diffs.safetensors"
        w = torch.randn(DIM)
        b = torch.randn(DIM)
        save_file({
            "diffusion_model.layers.0.attention.q_norm.diff":
                torch.randn(128),
            "diffusion_model.layers.0.attention_norm1.diff": w,
            "diffusion_model.x_embedder.diff_b": b,
            "diffusion_model.layers.0.attention.to_q.lora_A.weight":
                torch.zeros(32, DIM),
            "diffusion_model.layers.0.attention.to_q.lora_B.weight":
                torch.zeros(DIM, 32),
        }, str(out))
        specs, deltas = parse_zimage_lora(out)
        assert len(specs) == 1
        assert len(deltas) == 3
        by = {d.module: d for d in deltas}
        assert by["layers.0.attention.q_norm"].attr == "weight"
        assert torch.equal(by["layers.0.attention_norm1"].delta, w)
        assert by["x_embedder"].attr == "bias"
        assert torch.equal(by["x_embedder"].delta, b)
        assert all(isinstance(d, ParamDelta) for d in deltas)

    def test_incomplete_pair_raises(self, tmp_path):
        out = tmp_path / "half.safetensors"
        save_file({
            "diffusion_model.layers.0.attention.to_q.lora_A.weight":
                torch.zeros(32, DIM),
        }, str(out))
        with pytest.raises(ValueError, match="incomplete"):
            parse_zimage_lora(out)

    def test_alpha_is_honored(self, tmp_path):
        out = tmp_path / "alpha.safetensors"
        save_file({
            "diffusion_model.layers.0.attention.to_q.lora_A.weight":
                torch.zeros(32, DIM),
            "diffusion_model.layers.0.attention.to_q.lora_B.weight":
                torch.zeros(DIM, 32),
            "diffusion_model.layers.0.attention.to_q.alpha":
                torch.tensor(16.0),
        }, str(out))
        specs, deltas = parse_zimage_lora(out)
        assert deltas == []
        assert specs[0].scale == pytest.approx(16.0 / 32.0)
