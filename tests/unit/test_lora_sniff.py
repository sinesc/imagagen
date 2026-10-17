"""Phase 2c: every LoRA loader names an unsupported family (LoKr/LoHa/
DoRA/GaLore/GGUF) up front instead of drowning in key-mismatch noise."""


import pytest
import torch
from safetensors.torch import save_file

from imagagen.backends.krea2.lora import parse_krea2_lora
from imagagen.backends.sdxl import SdxlBackend
from imagagen.flux_lora import parse_lora, sniff_lora_format


def _write(tmp_path, name, tensors):
    p = tmp_path / name
    save_file({k: t.contiguous() for k, t in tensors.items()}, str(p))
    return p


@pytest.fixture
def lokr(tmp_path):
    return _write(tmp_path, "lokr.safetensors", {
        "transformer.blocks.0.attn.to_q.lora_A.weight":
            torch.randn(4, 16),
        "transformer.blocks.0.attn.to_q.lora_B.weight":
            torch.randn(16, 4),
        "transformer.blocks.0.attn.to_q.lokr_w1": torch.randn(8, 8),
        "transformer.blocks.0.attn.to_q.lokr_w2": torch.randn(16, 16),
    })


@pytest.fixture
def dora(tmp_path):
    return _write(tmp_path, "dora.safetensors", {
        "lora_unet_blocks_0_attn_to_q.lora_down.weight": torch.randn(4, 16),
        "lora_unet_blocks_0_attn_to_q.lora_up.weight": torch.randn(16, 4),
        "lora_unet_blocks_0_attn_to_q.dora_scale": torch.randn(1, 16),
    })


@pytest.fixture
def loha(tmp_path):
    return _write(tmp_path, "loha.safetensors", {
        "lora_unet_blocks_0_attn_to_q.loha_w1": torch.randn(4, 16),
        "lora_unet_blocks_0_attn_to_q.loha_w2": torch.randn(16, 4),
    })


@pytest.fixture
def gguf(tmp_path):
    p = tmp_path / "quant.gguf"
    p.write_bytes(b"GGUF" + b"\x00" * 64)
    return p


def test_sniff_names_each_family(lokr, dora, loha, gguf):
    assert "LoKr" in sniff_lora_format(lokr)
    assert "DoRA" in sniff_lora_format(dora)
    assert "LoHa" in sniff_lora_format(loha)
    assert "GGUF" in sniff_lora_format(gguf)
    assert sniff_lora_format(lokr, skip=("lokr",)) is None
    assert sniff_lora_format(dora, skip=("dora",)) is None


def test_flux_parser_rejects_by_name(lokr, dora, loha, gguf):
    for p, kind in ((lokr, "LoKr"), (dora, "DoRA"), (loha, "LoHa"),
                    (gguf, "GGUF")):
        with pytest.raises(ValueError, match=kind):
            parse_lora(p)


def test_krea2_parser_names_unsupported_families(loha, gguf):
    # LoKr and DoRA are SUPPORTED on Krea 2 -> the shared sniff must not
    # fire on them there (the krea2 parser handles them itself)
    with pytest.raises(ValueError, match="LoHa"):
        parse_krea2_lora(loha)
    with pytest.raises(ValueError, match="GGUF"):
        parse_krea2_lora(gguf)


def test_sdxl_loader_rejects_by_name(lokr, dora, gguf):
    # _lora_state_dict rejects before touching the pipeline, so a
    # pipe-less backend instance is enough
    be = object.__new__(SdxlBackend)
    for p, kind in ((lokr, "LoKr"), (dora, "DoRA"), (gguf, "GGUF")):
        with pytest.raises(ValueError, match=kind):
            be._lora_state_dict(str(p))
