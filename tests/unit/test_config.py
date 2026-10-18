"""user/imagagen.ini parsing (GPU selection + VRAM options)."""

from __future__ import annotations

import pytest

from imagagen.config import load_config


def ini(p):
    f = p / "user" / "imagagen.ini"
    f.parent.mkdir(parents=True, exist_ok=True)
    return f


def test_defaults_without_ini(tmp_path):
    c = load_config(tmp_path)
    assert c.primary_gpu is None
    assert c.ignore_gpus == frozenset()


def test_explicit_values(tmp_path):
    ini(tmp_path).write_text(
        "[imagagen]\nprimary-gpu = 1\nignore-gpus = 3, 4, 5\n"
    )
    c = load_config(tmp_path)
    assert c.primary_gpu == 1
    assert c.ignore_gpus == frozenset({3, 4, 5})


def test_auto_and_empty(tmp_path):
    ini(tmp_path).write_text(
        "[imagagen]\nprimary-gpu = auto\nignore-gpus =\n"
    )
    c = load_config(tmp_path)
    assert c.primary_gpu is None
    assert c.ignore_gpus == frozenset()


def test_comments_and_whitespace(tmp_path):
    ini(tmp_path).write_text(
        "# comment\n[imagagen]\n; another comment\nprimary-gpu=0\n"
        "ignore-gpus = 1 , 2\n"
    )
    c = load_config(tmp_path)
    assert c.primary_gpu == 0
    assert c.ignore_gpus == frozenset({1, 2})


def test_missing_section_ignored(tmp_path):
    ini(tmp_path).write_text("[other]\nprimary-gpu = 2\n")
    c = load_config(tmp_path)
    assert c.primary_gpu is None
    assert c.ignore_gpus == frozenset()


def test_bad_primary_raises(tmp_path):
    ini(tmp_path).write_text("[imagagen]\nprimary-gpu = banana\n")
    with pytest.raises(ValueError, match="primary-gpu"):
        load_config(tmp_path)


def test_bad_ignore_raises(tmp_path):
    ini(tmp_path).write_text("[imagagen]\nignore-gpus = 1, x\n")
    with pytest.raises(ValueError, match="ignore-gpus"):
        load_config(tmp_path)


def test_vram_options_defaults(tmp_path):
    c = load_config(tmp_path)
    assert c.unet_quantization == "off"
    assert c.text_encoder_device == "gpu"
    assert c.vae_tiling is False


def test_vram_options_explicit(tmp_path):
    ini(tmp_path).write_text(
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
    ini(tmp_path).write_text("[imagagen]\nunet-quantization = int8\n")
    with pytest.raises(ValueError, match="unet-quantization"):
        load_config(tmp_path)


def test_bad_encoder_device_raises(tmp_path):
    ini(tmp_path).write_text("[imagagen]\ntext-encoder-device = zzz\n")
    with pytest.raises(ValueError, match="text-encoder-device"):
        load_config(tmp_path)


def test_debug_flag(tmp_path):
    assert load_config(tmp_path).debug is False
    ini(tmp_path).write_text("[imagagen]\ndebug = true\n")
    assert load_config(tmp_path).debug is True
    ini(tmp_path).write_text("[imagagen]\ndebug = false\n")
    assert load_config(tmp_path).debug is False


def test_bad_debug_raises(tmp_path):
    ini(tmp_path).write_text("[imagagen]\ndebug = maybe\n")
    with pytest.raises(ValueError, match="debug"):
        load_config(tmp_path)


def test_save_ui_state_flag(tmp_path):
    # enabled by default (even with no ini file at all)
    assert load_config(tmp_path).save_ui_state is True
    ini(tmp_path).write_text("[imagagen]\nsave-ui-state = false\n")
    assert load_config(tmp_path).save_ui_state is False
    ini(tmp_path).write_text("[imagagen]\nsave-ui-state = true\n")
    assert load_config(tmp_path).save_ui_state is True


def test_bad_save_ui_state_raises(tmp_path):
    ini(tmp_path).write_text("[imagagen]\nsave-ui-state = maybe\n")
    with pytest.raises(ValueError, match="save-ui-state"):
        load_config(tmp_path)


def test_mask_overlay_opacity_default(tmp_path):
    assert load_config(tmp_path).mask_overlay_opacity == 0.9


def test_mask_overlay_opacity_explicit(tmp_path):
    ini(tmp_path).write_text("[imagagen]\nmask-overlay-opacity = 0.5\n")
    assert load_config(tmp_path).mask_overlay_opacity == 0.5
    ini(tmp_path).write_text("[imagagen]\nmask-overlay-opacity = 1\n")
    assert load_config(tmp_path).mask_overlay_opacity == 1.0
    ini(tmp_path).write_text("[imagagen]\nmask-overlay-opacity = 0\n")
    assert load_config(tmp_path).mask_overlay_opacity == 0.0


def test_bad_mask_overlay_opacity_raises(tmp_path):
    for v in ("abc", "-0.1", "1.5"):
        ini(tmp_path).write_text(f"[imagagen]\nmask-overlay-opacity = {v}\n")
        with pytest.raises(ValueError, match="mask-overlay-opacity"):
            load_config(tmp_path)


def test_transformer_parking_delay_default(tmp_path):
    # 90 s even with no ini file / no section
    assert load_config(tmp_path).transformer_parking_delay == 90.0
    ini(tmp_path).write_text("[other]\n")
    assert load_config(tmp_path).transformer_parking_delay == 90.0


def test_transformer_parking_delay_values(tmp_path):
    ini(tmp_path).write_text("[imagagen]\ntransformer-parking-delay = 0\n")
    assert load_config(tmp_path).transformer_parking_delay == 0.0
    ini(tmp_path).write_text("[imagagen]\ntransformer-parking-delay = -1\n")
    assert load_config(tmp_path).transformer_parking_delay == -1.0
    ini(tmp_path).write_text("[imagagen]\ntransformer-parking-delay = 30.5\n")
    assert load_config(tmp_path).transformer_parking_delay == 30.5


def test_bad_transformer_parking_delay_raises(tmp_path):
    for v in ("abc", "-2"):
        ini(tmp_path).write_text(
            f"[imagagen]\ntransformer-parking-delay = {v}\n")
        with pytest.raises(ValueError, match="transformer-parking-delay"):
            load_config(tmp_path)


def test_prefetch_text_encoder_flag(tmp_path):
    # ENABLED by default (even with no ini file at all)
    assert load_config(tmp_path).prefetch_text_encoder is True
    ini(tmp_path).write_text("[imagagen]\nprefetch-text-encoder = false\n")
    assert load_config(tmp_path).prefetch_text_encoder is False
    ini(tmp_path).write_text("[imagagen]\nprefetch-text-encoder = true\n")
    assert load_config(tmp_path).prefetch_text_encoder is True


def test_bad_prefetch_text_encoder_raises(tmp_path):
    ini(tmp_path).write_text("[imagagen]\nprefetch-text-encoder = maybe\n")
    with pytest.raises(ValueError, match="prefetch-text-encoder"):
        load_config(tmp_path)
