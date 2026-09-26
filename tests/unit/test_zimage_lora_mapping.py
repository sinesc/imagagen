"""Key mapping for Z-Image LoRAs (imagagen.zimage_lora).

The shipped LoRAs use the official Z-Image-Turbo naming
(``diffusion_model.layers.N.attention.to_q.lora_A.weight`` ...): our
transformer fuses q/k/v into one ``qkv`` GEMM, so each of those keys maps to
a row slice of the fused weight. The test asset is the shipped
``loras/zimage/ZIT shiny latex bodysuit.safetensors`` (rank 32, 30 main
blocks, no alpha tensors).
"""

from collections import Counter
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from imagagen.zimage_lora import DIM, FFN, map_base_key, parse_zimage_lora

LORA = Path(__file__).resolve().parents[2] / \
    "loras/zimage/ZIT shiny latex bodysuit.safetensors"


class TestMapBaseKey:
    def test_qkv_fusion_slices(self):
        assert map_base_key("layers.0.attention.to_q") == \
            ("layers.0.attention.qkv", 0, DIM)
        assert map_base_key("layers.3.attention.to_k") == \
            ("layers.3.attention.qkv", DIM, 2 * DIM)
        assert map_base_key("layers.29.attention.to_v") == \
            ("layers.29.attention.qkv", 2 * DIM, 3 * DIM)

    def test_to_out_is_the_full_output_projection(self):
        # the official `to_out.0` (trailing .0 stripped before mapping) is
        # the fused attention.out projection (full width)
        assert map_base_key("layers.1.attention.to_out") == \
            ("layers.1.attention.out", 0, DIM)

    def test_ffn(self):
        assert map_base_key("layers.2.feed_forward.w1") == \
            ("layers.2.feed_forward.w1", 0, FFN)
        assert map_base_key("layers.2.feed_forward.w2") == \
            ("layers.2.feed_forward.w2", 0, DIM)
        assert map_base_key("layers.2.feed_forward.w3") == \
            ("layers.2.feed_forward.w3", 0, FFN)

    def test_adaln(self):
        # the block adaLN (official `adaLN_modulation.0`, trailing .0
        # stripped before mapping) is the single Linear(256 -> 4*DIM)
        assert map_base_key("layers.4.adaLN_modulation") == \
            ("layers.4.adaLN_modulation.0", 0, 4 * DIM)

    def test_unknown_keys_raise(self):
        with pytest.raises(ValueError, match="refiner"):
            map_base_key("layers.0.noise_refiner")
        with pytest.raises(ValueError, match="x_embedder"):
            map_base_key("x_embedder")
        with pytest.raises(ValueError, match="bogus"):
            map_base_key("bogus")


class TestParseShippedLora:
    """The shipped test LoRA: official format, rank 32, no alpha."""

    def test_pairs_all_mapped_rank32_scale1(self):
        assert LORA.is_file(), f"test asset missing: {LORA}"
        specs = parse_zimage_lora(LORA)
        # 30 main blocks x 8 mappable pairs (q/k/v/out + w1/w2/w3 + adaLN)
        assert len(specs) == 30 * 8
        for s in specs:
            assert s.A.shape[0] == 32  # rank 32
            assert s.scale == pytest.approx(1.0)  # no .alpha tensors
            assert s.B.shape[0] == s.r1 - s.r0
            assert s.B.shape[1] == s.A.shape[0]
            assert not s.swap

    def test_module_coverage(self):
        specs = parse_zimage_lora(LORA)
        mods = Counter(s.module for s in specs)
        # per block: one qkv (3 slices), one out, w1/w2/w3, adaLN
        assert sum(1 for m in mods if ".attention.qkv" in m) == 30
        assert sum(1 for m in mods if ".attention.out" in m) == 30
        assert sum(1 for m in mods if ".feed_forward." in m) == 90
        assert sum(1 for m in mods if ".adaLN_modulation.0" in m) == 30
        qkv = [s for s in specs if s.module.endswith(".qkv")]
        assert {s.r0 for s in qkv} == {0, DIM, 2 * DIM}
        assert all(s.r1 - s.r0 == DIM for s in qkv)

    def test_unknown_key_raises_listed(self, tmp_path):
        # one mappable pair + one un-mappable key: the key must be named
        out = tmp_path / "bad.safetensors"
        save_file({
            "diffusion_model.layers.0.attention.to_q.lora_A.weight":
                torch.zeros(32, DIM),
            "diffusion_model.layers.0.attention.to_q.lora_B.weight":
                torch.zeros(DIM, 32),
            "diffusion_model.x_embedder.lora_A.weight":
                torch.zeros(32, 64),
        }, str(out))
        with pytest.raises(ValueError, match="x_embedder"):
            parse_zimage_lora(out)

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
        spec = parse_zimage_lora(out)[0]
        assert spec.scale == pytest.approx(16.0 / 32.0)
