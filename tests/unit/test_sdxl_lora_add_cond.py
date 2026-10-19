"""SDXL LoRA keys for the added-conditioning MLP (pooled text + size/crop).

That MLP is the one UNet module diffusers' kohya converter mishandles: its
SDXL index-strip heuristic eats the layer index of
``lora_unet_add_embedding_linear_0/_1/_2``, so ALL layers collapse onto the
same key ``add.embedding.linear`` (later weights overwrite earlier ones)
which matches no diffusers module. peft then reports

    Loading adapter weights from state_dict led to unexpected keys found in
    the model: add.embedding.linear.lora_A.<adapter>.weight, ...

and the weights are silently dropped -- measured with VibrantDreamXL1.0,
which carries exactly those two layers, so it rendered differently than in
A1111/Forge. ``normalize_add_cond_lora_keys`` fixes the RAW keys (the layer
information is gone after diffusers' conversion) by using the sgm spelling
``label_emb.0.N`` that diffusers maps to ``add_embedding.linear_N``.
"""

import pytest
import torch

from imagagen.backends.sdxl import normalize_add_cond_lora_keys, scale_lora_alphas

_SUFFIXES = (".alpha", ".lora_down.weight", ".lora_up.weight")


def _kohya_sd(prefix: str, idxs: tuple[str, ...], rank: int = 4) -> dict:
    """A minimal kohya/A1111-format LoRA state dict over one module family."""
    sd = {}
    for i in idxs:
        for suf in _SUFFIXES:
            key = f"lora_unet_{prefix}_{i}{suf}"
            sd[key] = (torch.tensor(float(rank * 4)) if suf == ".alpha"
                       else torch.ones(rank, 320))
    return sd


def _convert(sd: dict) -> tuple[dict, dict]:
    """diffusers' kohya -> diffusers key conversion (version canary below)."""
    from diffusers.loaders import lora_conversion_utils as lc

    fn = (getattr(lc, "_convert_non_diffusers_lora_to_diffusers", None)
          or getattr(lc, "_convert_kohya_lora_state_dict_to_diffusers", None))
    if fn is None:  # pragma: no cover - diffusers renamed the helper
        pytest.skip("diffusers kohya converter not found; key mapping changed")
    return fn(sd)


class TestNormalizeAddCondLoraKeys:
    def test_a1111_spelling_maps_to_both_layers(self):
        out = normalize_add_cond_lora_keys(
            _kohya_sd("add_embedding_linear", ("0", "2")))
        assert sorted(out) == [
            "lora_unet_label_emb_0_0.alpha",
            "lora_unet_label_emb_0_0.lora_down.weight",
            "lora_unet_label_emb_0_0.lora_up.weight",
            "lora_unet_label_emb_0_2.alpha",
            "lora_unet_label_emb_0_2.lora_down.weight",
            "lora_unet_label_emb_0_2.lora_up.weight",
        ]

    def test_diffusers_trainer_spelling_maps_too(self):
        # kohya-ss trains SDXL on the diffusers UNet, where the same two
        # Linears are named linear_1 / linear_2
        out = normalize_add_cond_lora_keys(
            _kohya_sd("add_embedding_linear", ("1", "2")))
        assert {k.split(".")[0] for k in out} == {
            "lora_unet_label_emb_0_0", "lora_unet_label_emb_0_2"}

    def test_other_layers_are_left_alone(self):
        sd = {
            "lora_unet_down_blocks_0_resnets_0_conv1.lora_down.weight":
                torch.ones(4, 320),
            "lora_unet_time_embedding_linear_1.lora_down.weight": torch.ones(4, 1280),
            "lora_te2_model_transformer_text_self_attn_out_projection_lora_down_weight":
                torch.ones(4, 1280),
        }
        assert normalize_add_cond_lora_keys(sd) is sd

    def test_no_match_returns_the_same_dict(self):
        sd = _kohya_sd("input_blocks_1_0_emb", ("0",))
        assert normalize_add_cond_lora_keys(sd) is sd

    def test_colliding_layers_are_left_unchanged(self, caplog):
        # a file that somehow carries BOTH namings of the first Linear must
        # not have them merged onto one module
        sd = {
            "lora_unet_add_embedding_linear_0.lora_down.weight": torch.ones(4, 320),
            "lora_unet_add_embedding_linear_1.lora_down.weight": torch.ones(4, 320),
        }
        assert normalize_add_cond_lora_keys(sd) is sd
        assert "same name" in caplog.text


class TestAgainstDiffusersConverter:
    def test_unrenamed_keys_collapse_onto_one_nonexistent_module(self):
        """Documents the upstream bug the normalizer guards against."""
        sd, _alphas = _convert(_kohya_sd("add_embedding_linear", ("0", "2")))
        down = [k for k in sd if k.endswith(".lora.down.weight")]
        assert down == ["unet.add.embedding.linear.lora.down.weight"]

    def test_renamed_keys_reach_the_real_unet_modules(self):
        sd, alphas = _convert(
            normalize_add_cond_lora_keys(
                _kohya_sd("add_embedding_linear", ("0", "2"))))
        assert {
            "unet.add_embedding.linear_1.lora.down.weight",
            "unet.add_embedding.linear_1.lora.up.weight",
            "unet.add_embedding.linear_2.lora.down.weight",
            "unet.add_embedding.linear_2.lora.up.weight",
        } <= set(sd)
        assert not [k for k in sd if ".add.embedding." in k]
        assert {
            "unet.add_embedding.linear_1.alpha",
            "unet.add_embedding.linear_2.alpha",
        } == set(alphas)

    def test_alphas_reach_the_add_cond_down_weights(self):
        # the dict-input load path drops alpha metadata, so imagagen
        # pre-scales: alpha 16 / rank 4 -> factor 4
        sd, alphas = _convert(
            normalize_add_cond_lora_keys(
                _kohya_sd("add_embedding_linear", ("0", "2"), rank=4)))
        scale_lora_alphas(sd, alphas)
        for key in ("unet.add_embedding.linear_1.lora.down.weight",
                    "unet.add_embedding.linear_2.lora.down.weight"):
            assert torch.allclose(sd[key], torch.full((4, 320), 4.0)), key
            # the up weights stay untouched
            up = key.replace(".down.", ".up.")
            assert torch.allclose(sd[up], torch.ones(4, 320)), up
