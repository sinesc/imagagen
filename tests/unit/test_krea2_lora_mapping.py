"""Key mapping for Krea 2 LoRAs (imagagen.krea2_lora).

The on-disk asset ``loras/krea2/rustic-ambience.safetensors`` ships the
diffusers naming under a ``transformer.`` prefix: 256 rank-16 bf16 pairs
over the 28 main blocks (``attn.to_q|to_k|to_v|to_gate|to_out.0``,
``ff.gate|up|down``) and the txtfusion layerwise + refiner blocks. No
alpha tensors (scale 1.0).
"""

from collections import Counter

import pytest
import torch

from imagagen.krea2_lora import (
    map_base_key,
    parse_krea2_lora,
)

ROOT = "loras/krea2/rustic-ambience.safetensors"


class TestMapBaseKey:
    def test_main_block_targets(self):
        assert map_base_key("transformer_blocks.0.attn.to_q") == \
            ("blocks.0.attn.wq", 0, 6144)
        assert map_base_key("transformer_blocks.1.attn.to_k") == \
            ("blocks.1.attn.wk", 0, 1536)
        assert map_base_key("transformer_blocks.2.attn.to_v") == \
            ("blocks.2.attn.wv", 0, 1536)
        assert map_base_key("transformer_blocks.3.attn.to_gate") == \
            ("blocks.3.attn.gate", 0, 6144)
        # the trailing .0 is stripped by the parser; the table key is to_out
        assert map_base_key("transformer_blocks.4.attn.to_out") == \
            ("blocks.4.attn.wo", 0, 6144)
        assert map_base_key("transformer_blocks.0.ff.gate") == \
            ("blocks.0.mlp.gate", 0, 16384)
        assert map_base_key("transformer_blocks.0.ff.up") == \
            ("blocks.0.mlp.up", 0, 16384)
        assert map_base_key("transformer_blocks.0.ff.down") == \
            ("blocks.0.mlp.down", 0, 6144)

    def test_txtfusion_targets(self):
        for group in ("layerwise_blocks", "refiner_blocks"):
            assert map_base_key(f"text_fusion.{group}.0.attn.to_q") == \
                (f"txtfusion.{group}.0.attn.wq", 0, 2560)
            assert map_base_key(f"text_fusion.{group}.0.attn.to_k") == \
                (f"txtfusion.{group}.0.attn.wk", 0, 2560)
            assert map_base_key(f"text_fusion.{group}.0.attn.to_gate") == \
                (f"txtfusion.{group}.0.attn.gate", 0, 2560)
            assert map_base_key(f"text_fusion.{group}.0.attn.to_out") == \
                (f"txtfusion.{group}.0.attn.wo", 0, 2560)
            assert map_base_key(f"text_fusion.{group}.0.ff.gate") == \
                (f"txtfusion.{group}.0.mlp.gate", 0, 6912)
            assert map_base_key(f"text_fusion.{group}.0.ff.down") == \
                (f"txtfusion.{group}.0.mlp.down", 0, 2560)

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError):
            map_base_key("x_embedder")
        with pytest.raises(ValueError):
            map_base_key("transformer_blocks.0.attn.to_rope")
        with pytest.raises(ValueError):
            map_base_key("text_fusion.other_blocks.0.attn.to_q")


class TestParseKrea2Lora:
    def test_real_asset(self):
        specs = parse_krea2_lora(ROOT)
        assert len(specs) == 256
        # every pair: rank 16, no alpha -> scale 1.0
        for s in specs:
            assert s.A.shape[0] == 16
            assert s.A.dtype == torch.bfloat16
            assert s.B.shape[0] == s.r1 - s.r0
            assert s.B.shape[1] == 16
            assert s.scale == 1.0
            assert s.r0 == 0
            assert s.swap is False
        # the expected key mix: 28 blocks * 8 targets + 16 txtfusion blocks
        # * 8 targets, spread across the two txtfusion groups
        blocks = [s for s in specs if s.module.startswith("blocks.")]
        txtfusion = [s for s in specs
                     if s.module.startswith("txtfusion.")]
        assert len(blocks) == 28 * 8
        # only the FIRST TWO blocks of each txtfusion group carry LoRAs
        # in this asset (2 groups x 2 blocks x 8 targets)
        assert len(txtfusion) == 2 * 2 * 8
        # spot-check a few concrete (mapped module, rows, A/B shapes)
        by_mod = {}
        for s in specs:
            by_mod.setdefault(s.module, []).append(s)
        assert by_mod["blocks.0.attn.wq"][0].A.shape == (16, 6144)
        assert by_mod["blocks.0.attn.wq"][0].B.shape == (6144, 16)
        assert by_mod["blocks.0.attn.wk"][0].B.shape == (1536, 16)
        assert by_mod["blocks.27.mlp.down"][0].A.shape == (16, 16384)
        assert by_mod["blocks.27.mlp.down"][0].B.shape == (6144, 16)
        lw = [s for s in txtfusion if ".layerwise_blocks." in s.module]
        rf = [s for s in txtfusion if ".refiner_blocks." in s.module]
        assert len(lw) == 2 * 8
        assert len(rf) == 2 * 8
        assert by_mod["txtfusion.layerwise_blocks.0.mlp.gate"][0].B.shape == \
            (6912, 16)
        assert by_mod["txtfusion.refiner_blocks.0.attn.wq"][0].A.shape == \
            (16, 2560)
        # all 28 main blocks x 8 targets are covered exactly once
        assert Counter(s.module for s in blocks) == Counter(
            f"blocks.{n}.{part}"
            for n in range(28)
            for part in ("attn.wq", "attn.wk", "attn.wv", "attn.gate",
                         "attn.wo", "mlp.gate", "mlp.up", "mlp.down")
        )

    def test_missing_pair_raises(self, tmp_path):
        from safetensors.torch import save_file

        p = tmp_path / "bad.safetensors"
        save_file({
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight":
                torch.zeros(16, 6144, dtype=torch.bfloat16),
            # no B
        }, str(p))
        with pytest.raises(ValueError, match="incomplete"):
            parse_krea2_lora(p)

    def test_unknown_key_raises(self, tmp_path):
        from safetensors.torch import save_file

        p = tmp_path / "bad.safetensors"
        save_file({
            "transformer.bogus_module.lora_A.weight":
                torch.zeros(16, 8, dtype=torch.bfloat16),
            "transformer.bogus_module.lora_B.weight":
                torch.zeros(8, 16, dtype=torch.bfloat16),
        }, str(p))
        with pytest.raises(ValueError, match="unknown Krea 2 LoRA key"):
            parse_krea2_lora(p)
