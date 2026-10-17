"""Key mapping for standard Flux LoRAs (imagagen.flux_lora).

The mapping table is the port of Forge's ``flux_to_diffusers`` (reference
checkout), adapted to our module paths: representative double/single/global
keys -> expected (module, row slice); the trailing-``.0`` normalization;
and unknown keys raising with the key listed (the all-or-nothing apply
surfaces them to the job as a clear error).
"""

from collections import Counter
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from imagagen.flux_lora import H, M, alpha_key, lora_key, map_base_key, parse_lora

D = H  # 3072

METALSUIT = Path(__file__).resolve().parents[2] / \
    "loras/flux/MetalSuitFlux1.0.safetensors"


class TestDoubleBlocks:
    def test_qkv_fusion_slices(self):
        # fused qkv: q/k/v own the first 3H output rows
        assert map_base_key("transformer_blocks.0.attn.to_q") == \
            ("double_blocks.0.img_attn.qkv", 0, D, False)
        assert map_base_key("transformer_blocks.5.attn.to_k") == \
            ("double_blocks.5.img_attn.qkv", D, 2 * D, False)
        assert map_base_key("transformer_blocks.18.attn.to_v") == \
            ("double_blocks.18.img_attn.qkv", 2 * D, 3 * D, False)

    def test_txt_qkv_slices(self):
        assert map_base_key("transformer_blocks.0.attn.add_q_proj") == \
            ("double_blocks.0.txt_attn.qkv", 0, D, False)
        assert map_base_key("transformer_blocks.1.attn.add_k_proj") == \
            ("double_blocks.1.txt_attn.qkv", D, 2 * D, False)
        assert map_base_key("transformer_blocks.2.attn.add_v_proj") == \
            ("double_blocks.2.txt_attn.qkv", 2 * D, 3 * D, False)

    def test_pre_only_projections(self):
        # `to_out` is the full-width output projection (trailing .0 stripped)
        assert map_base_key("transformer_blocks.3.attn.to_out") == \
            ("double_blocks.3.img_attn.proj", 0, D, False)
        assert map_base_key("transformer_blocks.3.attn.to_add_out") == \
            ("double_blocks.3.txt_attn.proj", 0, D, False)

    def test_ff(self):
        assert map_base_key("transformer_blocks.4.ff.net.0") == \
            ("double_blocks.4.img_mlp.0", 0, M, False)
        assert map_base_key("transformer_blocks.4.ff.net.0.proj") == \
            ("double_blocks.4.img_mlp.0", 0, M, False)
        assert map_base_key("transformer_blocks.4.ff.net.2") == \
            ("double_blocks.4.img_mlp.2", 0, D, False)
        assert map_base_key("transformer_blocks.4.ff_context.net.0") == \
            ("double_blocks.4.txt_mlp.0", 0, M, False)
        assert map_base_key("transformer_blocks.4.ff_context.net.0.proj") == \
            ("double_blocks.4.txt_mlp.0", 0, M, False)
        assert map_base_key("transformer_blocks.4.ff_context.net.2") == \
            ("double_blocks.4.txt_mlp.2", 0, D, False)

    def test_modulation(self):
        # lin maps H -> 6H (double blocks: two shift/scale/gate triples)
        assert map_base_key("transformer_blocks.5.norm1.linear") == \
            ("double_blocks.5.img_mod.lin", 0, 6 * D, False)
        assert map_base_key("transformer_blocks.5.norm1_context.linear") == \
            ("double_blocks.5.txt_mod.lin", 0, 6 * D, False)


class TestSingleBlocks:
    def test_linear1_fusion_slices(self):
        P = "single_blocks.0"
        assert map_base_key("single_transformer_blocks.0.attn.to_q") == \
            (f"{P}.linear1", 0, D, False)
        assert map_base_key("single_transformer_blocks.1.attn.to_k") == \
            ("single_blocks.1.linear1", D, 2 * D, False)
        assert map_base_key("single_transformer_blocks.2.attn.to_v") == \
            ("single_blocks.2.linear1", 2 * D, 3 * D, False)

    def test_proj_mlp(self):
        # the mlp projection owns linear1 rows [3H:3H+M); BFL also names
        # it without the `attn.` prefix
        assert map_base_key("single_transformer_blocks.3.attn.proj_mlp") == \
            ("single_blocks.3.linear1", 3 * D, 3 * D + M, False)
        assert map_base_key("single_transformer_blocks.3.proj_mlp") == \
            ("single_blocks.3.linear1", 3 * D, 3 * D + M, False)

    def test_proj_out_and_modulation(self):
        assert map_base_key("single_transformer_blocks.37.proj_out") == \
            ("single_blocks.37.linear2", 0, D, False)
        # lin maps H -> 3H (one shift/scale/gate triple)
        assert map_base_key("single_transformer_blocks.37.norm.linear") == \
            ("single_blocks.37.modulation.lin", 0, 3 * D, False)


class TestGlobal:
    def test_embedders(self):
        assert map_base_key("x_embedder") == ("img_in", 0, D, False)
        assert map_base_key("context_embedder") == ("txt_in", 0, D, False)
        assert map_base_key("time_embedder.linear_1") == \
            ("time_in.in_layer", 0, D, False)
        assert map_base_key("time_embedder.linear_2") == \
            ("time_in.out_layer", 0, D, False)
        assert map_base_key("vector_embedder.linear_1") == \
            ("vector_in.in_layer", 0, D, False)
        assert map_base_key("vector_embedder.linear_2") == \
            ("vector_in.out_layer", 0, D, False)
        assert map_base_key("guidance_embedder.linear_1") == \
            ("guidance_in.in_layer", 0, D, False)
        assert map_base_key("guidance_embedder.linear_2") == \
            ("guidance_in.out_layer", 0, D, False)

    def test_norm_out_swaps_bhalves(self):
        # standard [scale, shift] row order, ours [shift, scale]
        assert map_base_key("norm_out.linear") == \
            ("final_layer.adaLN_modulation.1", 0, 2 * D, True)

    def test_final_proj_out(self):
        # the block-level `proj_out` is the final H -> 16 output projection
        assert map_base_key("proj_out") == ("final_layer.linear", 0, 16, False)


class TestKeySplitting:
    def test_lora_a_b_naming(self):
        assert lora_key(
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight"
        ) == ("transformer_blocks.0.attn.to_q", "A")
        assert lora_key(
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight"
        ) == ("transformer_blocks.0.attn.to_q", "B")

    def test_lora_down_up_naming(self):
        assert lora_key(
            "transformer.transformer_blocks.0.attn.to_q.lora_down.weight"
        ) == ("transformer_blocks.0.attn.to_q", "A")
        assert lora_key(
            "transformer.transformer_blocks.0.attn.to_q.lora_up.weight"
        ) == ("transformer_blocks.0.attn.to_q", "B")

    def test_trailing_dot0_is_stripped(self):
        a, b = (
            lora_key("transformer.transformer_blocks.0.attn.to_out.0"
                    ".lora_A.weight"),
            lora_key("transformer.transformer_blocks.0.attn.to_out"
                    ".lora_A.weight"),
        )
        assert a == b == ("transformer_blocks.0.attn.to_out", "A")

    def test_non_pair_keys(self):
        assert lora_key("transformer.time_embedder.linear_1.alpha") is None
        assert alpha_key("transformer.time_embedder.linear_1.alpha") == \
            "time_embedder.linear_1"
        assert alpha_key(
            "transformer.time_embedder.linear_1.lora_A.weight"
        ) is None


class TestUnknownKeys:
    def test_qk_norm_scale_raises_listed(self):
        # 1-D QK-norm scales have no weight-matrix target
        with pytest.raises(ValueError, match="attn.norm_q"):
            map_base_key("transformer_blocks.0.attn.norm_q")

    def test_single_norm_q_raises(self):
        with pytest.raises(ValueError, match="norm_q"):
            map_base_key("single_transformer_blocks.7.attn.norm_q")

    def test_garbage_raises(self):
        with pytest.raises(ValueError, match="bogus"):
            map_base_key("bogus")


class TestParseMetalSuit:
    """The shipped test LoRA: standard Flux LoRA, rank 16, no alpha."""

    def test_pairs_all_mapped_rank16_scale1(self):
        assert METALSUIT.is_file(), f"test asset missing: {METALSUIT}"
        specs = parse_lora(METALSUIT)
        assert len(specs) == 266  # 152 double-block + 114 single-block pairs
        for s in specs:
            assert s.A.shape[0] == 16  # rank 16
            assert s.scale == pytest.approx(1.0)  # no .alpha tensors
            assert s.B.shape[0] == s.r1 - s.r0
            assert s.B.shape[1] == s.A.shape[0]

    def test_module_coverage(self):
        specs = parse_lora(METALSUIT)
        mods = Counter(s.module for s in specs)
        # 4 unique double-block modules per block (img/txt qkv + proj)
        assert sum(1 for m in mods if m.startswith("double_blocks")) == 19 * 4
        # one unique module per single block (linear1)
        assert sum(1 for m in mods if m.startswith("single_blocks")) == 38
        # the single-block deltas all land in linear1's q/k/v rows
        single = [s for s in specs if s.module.startswith("single_blocks")]
        assert {s.r0 for s in single} == {0, D, 2 * D}
        assert all(s.r1 - s.r0 == D for s in single)

    def test_unknown_key_raises_listed(self, tmp_path):
        # one mappable pair + one un-mappable key: the key must be named
        out = tmp_path / "bad.safetensors"
        save_file({
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight":
                torch.zeros(16, 3072),
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight":
                torch.zeros(3072, 16),
            "transformer.transformer_blocks.0.attn.norm_q.lora_A.weight":
                torch.zeros(16, 3072),
        }, str(out))
        with pytest.raises(ValueError, match="attn.norm_q"):
            parse_lora(out)
