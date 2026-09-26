"""Tests for the Z-Image sigma schedule (kdiff.get_sigmas_zimage).

The schedule is the port of the ComfyUI reference's default (KSampler
"simple") schedule for Z-Image: a 1000-slot buffer of the linear-SNR shift
``sigma(t) = 3t / (1 + 2t)`` sampled at ``t = (999 - int(i * 1000/n)) /
1000``, with a terminal 0 appended.
"""

import torch

from imagagen import kdiff


def _shift(t: float, s: float = 3.0) -> float:
    return s * t / (1.0 + (s - 1.0) * t)


class TestGetSigmasZimage:
    def test_shape_and_terminal_zero(self):
        for n in (1, 8, 20, 100):
            sigmas = kdiff.get_sigmas_zimage(n)
            assert sigmas.shape == (n + 1,)
            assert float(sigmas[-1]) == 0.0

    def test_starts_at_full_noise(self):
        # t = 1 shifts to exactly 1.0 (3*1 / (1 + 2*1))
        for n in (1, 8, 20):
            assert kdiff.get_sigmas_zimage(n)[0].item() == 1.0

    def test_strictly_decreasing(self):
        sigmas = kdiff.get_sigmas_zimage(20)
        diffs = sigmas[1:] - sigmas[:-1]
        assert (diffs < 0).all()

    def test_n8_matches_the_official_linspace_grid(self):
        # for the standard 8-step Turbo count the buffer grid equals the
        # official shifted linspace(1, 1/8, 8)
        t = torch.linspace(1.0, 1.0 / 8, 8)
        expected = 3.0 * t / (1.0 + 2.0 * t)
        sigmas = kdiff.get_sigmas_zimage(8)
        assert torch.allclose(sigmas[:8], expected, atol=1e-6)

    def test_n20_matches_the_official_linspace_grid(self):
        # 20 divides 1000, so the buffer points equal 1 - i/20 exactly
        t = torch.linspace(1.0, 1.0 / 20, 20)
        expected = 3.0 * t / (1.0 + 2.0 * t)
        assert torch.allclose(kdiff.get_sigmas_zimage(20)[:20], expected)

    def test_non_divisor_steps_follow_the_buffer(self):
        # 7 does not divide 1000: t_i = (1000 - int(i * 1000/7)) / 1000
        expected = torch.tensor(
            [_shift((1000 - int(i * 1000.0 / 7)) / 1000.0) for i in range(7)])
        assert torch.allclose(kdiff.get_sigmas_zimage(7)[:7], expected)

    def test_shift_formula(self):
        # linear-SNR shift: identity at shift 1, 1 fixed point at t = 1
        assert _shift(0.5, 1.0) == 0.5
        assert _shift(1.0) == 1.0
        assert _shift(0.0) == 0.0
        assert _shift(0.5) == 1.5 / 2.0
