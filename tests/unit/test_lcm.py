"""Tests for the LCM path: the 50-slot schedule and the sample_lcm dynamics.

The schedule math mirrors Forge's LCMCompVisDenoiser
(modules/sd_samplers_lcm.py) built on the k-diffusion DiscreteSchedule:
50 original timesteps (19, 39, ..., 999) of the 1000-step beta schedule,
nearest-slot quantization in log space.
"""

import torch

from imagagen import kdiff
from imagagen.sdxl_backend import _LcmSchedule


def _sdxl_alphas_cumprod(n: int = 1000) -> torch.Tensor:
    """SDXL-shaped beta schedule (linear 0.00085 -> 0.012)."""
    betas = torch.linspace(0.00085, 0.012, n)
    return (1.0 - betas).cumprod(dim=0)


class TestLcmSchedule:
    def test_slots_match_forge_construction(self):
        ac = _sdxl_alphas_cumprod()
        lcm = _LcmSchedule(ac)
        # Forge: alphas_cumprod_valid[49 - x] = alphas_cumprod[999 - 20x]
        # -> our ascending slice ac[19::20] (t = 19, 39, ..., 999)
        expected = ((1.0 - ac[19::20]) / ac[19::20]).sqrt()
        assert lcm.sigmas.shape == (50,)
        assert torch.allclose(lcm.sigmas, expected)
        assert lcm.min_sigma == float(expected[0])
        assert lcm.max_sigma == float(expected[-1])
        assert lcm.min_sigma < 1.0 < lcm.max_sigma

    def test_exact_slot_sigmas_map_to_their_timestep(self):
        lcm = _LcmSchedule(_sdxl_alphas_cumprod())
        for i in (0, 1, 24, 49):
            assert lcm.timestep(lcm.sigmas[i]).item() == 19 + 20 * i

    def test_timesteps_always_on_the_50_slot_grid(self):
        lcm = _LcmSchedule(_sdxl_alphas_cumprod())
        s = torch.linspace(lcm.min_sigma, lcm.max_sigma, 500)
        t = lcm.timestep(s)
        grid = (19 + 20 * torch.arange(50)).float()
        assert torch.isin(t, grid).all()

    def test_x0_scaled_close_to_plain_x0(self):
        # with t' = 10 * t in [190, 9990] the CompVis rescaling is
        # (1 - 1e-4..1e-7) * (x - sigma*eps) + tiny * x
        lcm = _LcmSchedule(_sdxl_alphas_cumprod())
        x = torch.randn(1, 4, 8, 8)
        for sigma in (lcm.min_sigma, 1.0, lcm.max_sigma):
            eps = torch.randn_like(x)
            x0 = lcm.x0_scaled(x, torch.tensor(sigma), eps)
            assert torch.allclose(x0, x - sigma * eps, atol=1e-3)


class TestSampleLcm:
    def test_final_step_is_pure_denoise(self):
        states = []

        def model(x, sigma, **_):
            states.append(x.clone())
            return 0.0 * x  # x0 == 0

        x = torch.randn(1, 4, 2, 2)
        out = kdiff.sample_lcm(model, x, torch.tensor([4.0, 2.0, 0.0]),
                               extra_args={"seed": 1})
        assert out.abs().max().item() == 0.0  # ends exactly on the final x0
        # the re-noise after step 1 is unit-variance noise * sigma_next
        renoise = states[1] / 2.0
        assert abs(renoise.var().item() - 1.0) < 0.5

    def test_deterministic_per_seed(self):
        sigmas = torch.tensor([8.0, 4.0, 1.0, 0.0])
        x = torch.zeros(1, 4, 2, 2)
        same = kdiff.sample_lcm(lambda x, s, **_: x, x, sigmas, extra_args={"seed": 7})
        again = kdiff.sample_lcm(lambda x, s, **_: x, x, sigmas, extra_args={"seed": 7})
        other = kdiff.sample_lcm(lambda x, s, **_: x, x, sigmas, extra_args={"seed": 8})
        assert torch.equal(same, again)
        assert not torch.equal(same, other)

    def test_no_renoise_on_last_step(self):
        # with x0 == identity the final state must equal the input of the
        # last call (no sigma_0 noise added)
        last = {}

        def model(x, sigma, **_):
            last["x"] = x.clone()
            return x

        x = torch.randn(1, 4, 2, 2)
        sigmas = torch.tensor([2.0, 0.0])
        out = kdiff.sample_lcm(model, x, sigmas, extra_args={"seed": 3})
        assert torch.equal(out, last["x"])
