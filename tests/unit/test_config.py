"""imagagen.ini parsing (GPU selection + VRAM options)."""

from __future__ import annotations

import pytest

from imagagen.config import load_config


def test_defaults_without_ini(tmp_path):
    c = load_config(tmp_path)
    assert c.primary_gpu is None
    assert c.ignore_gpus == frozenset()


def test_explicit_values(tmp_path):
    (tmp_path / "imagagen.ini").write_text(
        "[imagagen]\nprimary-gpu = 1\nignore-gpus = 3, 4, 5\n"
    )
    c = load_config(tmp_path)
    assert c.primary_gpu == 1
    assert c.ignore_gpus == frozenset({3, 4, 5})


def test_auto_and_empty(tmp_path):
    (tmp_path / "imagagen.ini").write_text(
        "[imagagen]\nprimary-gpu = auto\nignore-gpus =\n"
    )
    c = load_config(tmp_path)
    assert c.primary_gpu is None
    assert c.ignore_gpus == frozenset()


def test_comments_and_whitespace(tmp_path):
    (tmp_path / "imagagen.ini").write_text(
        "# comment\n[imagagen]\n; another comment\nprimary-gpu=0\n"
        "ignore-gpus = 1 , 2\n"
    )
    c = load_config(tmp_path)
    assert c.primary_gpu == 0
    assert c.ignore_gpus == frozenset({1, 2})


def test_missing_section_ignored(tmp_path):
    (tmp_path / "imagagen.ini").write_text("[other]\nprimary-gpu = 2\n")
    c = load_config(tmp_path)
    assert c.primary_gpu is None
    assert c.ignore_gpus == frozenset()


def test_bad_primary_raises(tmp_path):
    (tmp_path / "imagagen.ini").write_text("[imagagen]\nprimary-gpu = banana\n")
    with pytest.raises(ValueError, match="primary-gpu"):
        load_config(tmp_path)


def test_bad_ignore_raises(tmp_path):
    (tmp_path / "imagagen.ini").write_text("[imagagen]\nignore-gpus = 1, x\n")
    with pytest.raises(ValueError, match="ignore-gpus"):
        load_config(tmp_path)


def test_vram_options_defaults(tmp_path):
    c = load_config(tmp_path)
    assert c.unet_quantization == "off"
    assert c.text_encoder_device == "gpu"
    assert c.vae_tiling is False


def test_vram_options_explicit(tmp_path):
    (tmp_path / "imagagen.ini").write_text(
        "[imagagen]\n"
        "unet-quantization = nf4\n"
        "text-encoder-device = cpu\n"
        "vae-tiling = true\n"
    )
    c = load_config(tmp_path)
    assert c.unet_quantization == "nf4"
    assert c.text_encoder_device == "cpu"
    assert c.vae_tiling is True


def test_bad_quantization_raises(tmp_path):
    (tmp_path / "imagagen.ini").write_text("[imagagen]\nunet-quantization = int8\n")
    with pytest.raises(ValueError, match="unet-quantization"):
        load_config(tmp_path)


def test_bad_encoder_device_raises(tmp_path):
    (tmp_path / "imagagen.ini").write_text("[imagagen]\ntext-encoder-device = zzz\n")
    with pytest.raises(ValueError, match="text-encoder-device"):
        load_config(tmp_path)


def test_debug_flag(tmp_path):
    assert load_config(tmp_path).debug is False
    (tmp_path / "imagagen.ini").write_text("[imagagen]\ndebug = true\n")
    assert load_config(tmp_path).debug is True
    (tmp_path / "imagagen.ini").write_text("[imagagen]\ndebug = false\n")
    assert load_config(tmp_path).debug is False


def test_bad_debug_raises(tmp_path):
    (tmp_path / "imagagen.ini").write_text("[imagagen]\ndebug = maybe\n")
    with pytest.raises(ValueError, match="debug"):
        load_config(tmp_path)
