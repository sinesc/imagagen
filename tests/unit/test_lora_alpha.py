"""Regression tests for LoRA alpha pre-scaling and the sigma->timestep map.

Both fixes come from the DMD2 LoRA quality incident: diffusers 0.40's
dict-input load path drops alpha metadata (adapters ran rank/alpha times
too strong), and the timestep mapping must be the A1111/Forge
nearest-beta-sigma-index, not the ComfyUI-style sigma formula.
"""

import torch

from imagagen.backends.sdxl import scale_lora_alphas, sigma_to_timestep


class TestScaleLoraAlphas:
    def test_scales_lora_down_naming(self):
        w = torch.ones(4, 8)  # rank 4
        sd = {"unet.block.lora.down.weight": w}
        alphas = {"unet.block.alpha": 2.0}
        out = scale_lora_alphas(sd, alphas)
        assert out is sd  # in place
        # alpha/rank = 0.5
        assert torch.allclose(sd["unet.block.lora.down.weight"], w * 0.5)

    def test_scales_lora_a_naming(self):
        w = torch.ones(8, 16)  # rank 8
        sd = {"unet.block.lora_A.weight": w}
        alphas = {"unet.block.alpha": 8.0}
        scale_lora_alphas(sd, alphas)
        # alpha/rank = 1.0 -> unchanged values
        assert torch.allclose(sd["unet.block.lora_A.weight"], w)

    def test_3x3_conv_uses_first_dim_as_rank(self):
        w = torch.ones(64, 32, 3, 3)  # conv down: (rank, in, kh, kw)
        sd = {"unet.conv.lora.down.weight": w}
        alphas = {"unet.conv.alpha": 8.0}
        scale_lora_alphas(sd, alphas)
        assert torch.allclose(sd["unet.conv.lora.down.weight"], w * (8.0 / 64))

    def test_missing_down_key_is_ignored(self):
        sd = {}
        alphas = {"unet.block.alpha": 2.0}
        scale_lora_alphas(sd, alphas)  # must not raise
        assert sd == {}

    def test_none_alphas_is_noop(self):
        w = torch.ones(4, 8)
        sd = {"unet.block.lora.down.weight": w}
        scale_lora_alphas(sd, None)
        assert torch.allclose(sd["unet.block.lora.down.weight"], w)

    def test_dmd2_scale_values(self):
        # DMD2fp32XL1.0: rank 64, alpha 8 -> factor 0.125. Without the
        # fix these weights ran 8x too strong (the DMD2 mosaic bug).
        w = torch.randn(64, 320)
        sd = {"unet.down_blocks.0.resnets.0.conv1.lora.down.weight": w}
        alphas = {"unet.down_blocks.0.resnets.0.conv1.alpha": 8.0}
        scale_lora_alphas(sd, alphas)
        assert torch.allclose(
            sd["unet.down_blocks.0.resnets.0.conv1.lora.down.weight"], w * 0.125
        )

    def test_attention_processor_key_naming(self):
        # diffusers 0.40 keys transformer-block attention LoRAs through the
        # attention processor: the alpha key is ``<down_key>.alpha`` and the
        # state-dict key IS the down weight (no extra suffix). This is the
        # naming used by 560/788 DMD2 layers and all 560 EnvyZoom layers;
        # missing it left them unscaled (rank/alpha times too strong).
        w = torch.randn(64, 1280)
        up = torch.randn(1280, 64)
        key = ("unet.down_blocks.2.attentions.0.transformer_blocks.9"
               ".attn1.processor.to_k_lora.down.weight")
        up_key = key.replace(".to_k_lora.down.weight", ".to_k_lora.up.weight")
        sd = {key: w, up_key: up}
        alphas = {key + ".alpha": 8.0}
        scale_lora_alphas(sd, alphas)
        assert torch.allclose(sd[key], w * 0.125)  # down scaled
        assert torch.allclose(sd[up_key], up)  # up untouched

    def test_processor_style_does_not_break_regular_style(self):
        w_reg = torch.ones(4, 8)
        w_proc = torch.ones(4, 8)
        sd = {
            "unet.a.lora.down.weight": w_reg,
            "unet.b.processor.to_q_lora.down.weight": w_proc,
        }
        alphas = {
            "unet.a.alpha": 2.0,
            "unet.b.processor.to_q_lora.down.weight.alpha": 2.0,
        }
        scale_lora_alphas(sd, alphas)
        assert torch.allclose(sd["unet.a.lora.down.weight"], w_reg * 0.5)
        assert torch.allclose(
            sd["unet.b.processor.to_q_lora.down.weight"], w_proc * 0.5
        )


class TestSigmaToTimestep:
    @staticmethod
    def _schedule(n=1000):
        # synthetic log-spaced beta schedule, SDXL-like ends
        sigmas = torch.linspace(0.0292, 14.6, n)
        return sigmas, torch.log(sigmas)

    def test_exact_schedule_sigmas_map_to_their_index(self):
        sigmas, log_sigmas = self._schedule()
        for i in (0, 17, 500, 982, 999):
            t = sigma_to_timestep(sigmas[i], log_sigmas)
            assert t.item() == i

    def test_endpoints(self):
        sigmas, log_sigmas = self._schedule()
        assert sigma_to_timestep(sigmas[0], log_sigmas).item() == 0
        assert sigma_to_timestep(sigmas[-1], log_sigmas).item() == 999

    def test_intermediate_sigma_maps_nearest(self):
        sigmas, log_sigmas = self._schedule()
        # halfway (in log space) between indices 100 and 101 -> one of them
        t = sigma_to_timestep(sigmas[100] * 1.001, log_sigmas).item()
        assert t in (100, 101)

    def test_scalar_and_batch_shapes(self):
        _, log_sigmas = self._schedule()
        s = torch.tensor(1.0)
        assert sigma_to_timestep(s, log_sigmas).shape == ()
        s = torch.tensor([0.5, 1.0, 2.0])
        out = sigma_to_timestep(s, log_sigmas)
        assert out.shape == (3,)
        assert torch.all(out >= 0) and torch.all(out <= 999)

    def test_monotonic(self):
        _, log_sigmas = self._schedule()
        sigmas = torch.logspace(torch.tensor(0.0292).log10(), torch.tensor(14.6).log10(), 50)
        t = sigma_to_timestep(sigmas, log_sigmas)
        assert torch.all(t[1:] >= t[:-1])
