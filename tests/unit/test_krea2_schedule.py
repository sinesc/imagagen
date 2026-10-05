"""Tests for the Krea 2 sigma schedule (kdiff.get_sigmas_krea2).

The schedule is the reference ``sampling.py::timesteps``: the
``linspace(1, 0, n + 1)`` grid (n intervals, terminal 0 included) with
the exponential shift ``exp(mu) / (exp(mu) + (1/t - 1))`` where ``mu``
is linear in the image token count between (256, 0.5) and (6400, 1.15)
(turbo: pinned 1.15); the shift maps the terminal 0 to 0.
"""

import math

import torch

from imagagen import kdiff


def _reference(n: int, mu: float) -> torch.Tensor:
    t = torch.linspace(1.0, 0.0, n + 1)
    exp_mu = math.exp(mu)
    return exp_mu / (exp_mu + (1.0 / t - 1.0))


class TestGetSigmasKrea2:
    def test_shape_and_terminal_zero(self):
        for n in (1, 8, 28, 100):
            sigmas = kdiff.get_sigmas_krea2(n, 1024, "cpu")
            assert sigmas.shape == (n + 1,)
            assert float(sigmas[-1]) == 0.0

    def test_starts_at_full_noise(self):
        # t = 1 shifts to exactly 1.0 for any mu
        for n in (1, 8, 28):
            assert kdiff.get_sigmas_krea2(n, 1024, "cpu")[0].item() == 1.0

    def test_strictly_decreasing(self):
        sigmas = kdiff.get_sigmas_krea2(28, 1024, "cpu")
        assert (sigmas[1:] < sigmas[:-1]).all()

    def test_mu_matches_the_linear_resolution_formula(self):
        # image tokens 1024 -> mu = 1024 * m + b for m = 0.65/6144,
        # b = 0.5 - 256 * m
        m = (1.15 - 0.5) / (6400 - 256)
        b = 0.5 - 256 * m
        mu = 1024 * m + b
        sigmas = kdiff.get_sigmas_krea2(28, 1024, "cpu")
        torch.testing.assert_close(
            sigmas, _reference(28, mu), rtol=1e-6, atol=1e-9)

    def test_pinned_mu_overrides_the_resolution_formula(self):
        # turbo: mu = 1.15 regardless of the image token count
        a = kdiff.get_sigmas_krea2(8, 64, "cpu", mu=1.15)
        b = kdiff.get_sigmas_krea2(8, 4096, "cpu", mu=1.15)
        torch.testing.assert_close(a, b, rtol=1e-7, atol=1e-9)
        torch.testing.assert_close(
            a, _reference(8, 1.15), rtol=1e-6, atol=1e-9)

    def test_larger_images_spend_more_steps_at_high_noise(self):
        # a higher mu shifts the whole grid toward t ~ 1 (high noise)
        small = kdiff.get_sigmas_krea2(28, 256, "cpu")   # mu ~ 0.5
        large = kdiff.get_sigmas_krea2(28, 6400, "cpu")  # mu = 1.15
        # at the mid-point of the schedule the large image is still at a
        # much higher noise level
        assert float(large[14]) > float(small[14])
        # and the final step of the small image is at a lower noise level
        assert float(small[-2]) < float(large[-2])

    def test_small_image_mu_is_the_base_shift(self):
        # seq_len == base_seq_len -> mu == base_shift exactly
        sigmas = kdiff.get_sigmas_krea2(8, 256, "cpu")
        torch.testing.assert_close(
            sigmas, _reference(8, 0.5), rtol=1e-6, atol=1e-9)
