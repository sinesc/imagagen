"""Forge img2img_fix_steps schedule (kdiff.fix_steps_sigmas).

The point of the helper: img2img/inpaint keep the denoise-dependent start
sigma (strength of the change) AND run exactly the configured step count,
so a low denoise on a low-step (turbo) model no longer collapses to one or
two steps.
"""
import pytest
import torch

from imagagen import kdiff


def _karras(n):
    return kdiff.get_sigmas_karras(n, 0.03, 14.6, device="cpu")


def _flux_like(n):
    # linear 1 -> 0 flow schedule (n + 1 sigmas with a terminal 0, the shape
    # all flow backends use)
    return torch.linspace(1.0, 0.0, n + 1, device="cpu")


@pytest.mark.parametrize("make", [_karras, _flux_like])
@pytest.mark.parametrize("denoise", [1.0, 0.999, 0.8, 0.55, 0.3, 0.1, 0.05,
                                     0.01, 0.0])
@pytest.mark.parametrize("steps", [1, 4, 12, 20])
def test_step_count_is_exactly_configured(make, denoise, steps):
    sched = kdiff.fix_steps_sigmas(make, steps, denoise)
    assert len(sched) - 1 == steps, "executed steps must match the UI setting"
    assert float(sched[-1]) == 0.0
    assert bool((sched[:-1] > sched[1:]).all()), "schedule must descend"


@pytest.mark.parametrize("make", [_karras, _flux_like])
def test_start_sigma_tracks_strength_monotonically(make):
    starts = [float(kdiff.fix_steps_sigmas(make, 20, d)[0])
              for d in (0.05, 0.2, 0.4, 0.6, 0.8, 1.0)]
    assert all(a < b for a, b in zip(starts[:-1], starts[1:], strict=True)), starts


def test_full_denoise_matches_the_plain_schedule():
    plain = _karras(20)
    sched = kdiff.fix_steps_sigmas(_karras, 20, 1.0)
    assert torch.allclose(sched, plain)


def test_start_sigma_close_to_a1111_truncation():
    """The strength (start sigma) must stay close to A1111's truncation."""
    for steps, d in [(20, 0.5), (20, 0.8), (12, 0.55), (20, 0.2)]:
        plain = _karras(steps)
        t_enc = int(min(d, 0.999) * steps)
        a1111_start = float(plain[steps - 1 - t_enc])
        new_start = float(kdiff.fix_steps_sigmas(_karras, steps, d)[0])
        # same order of magnitude (within a factor of 2) and never higher
        assert new_start <= a1111_start * 1.05
        assert new_start > a1111_start / 2.0, (steps, d, a1111_start, new_start)


def test_zero_denoise_starts_at_the_lowest_noise_level():
    sched = kdiff.fix_steps_sigmas(_karras, 20, 0.0)
    assert len(sched) - 1 == 20
    assert float(sched[0]) < 0.1  # ~sigma_min: effectively no change


def test_schedule_length_is_capped():
    calls = []

    def make(n):
        calls.append(n)
        return _karras(n)

    for d in (1e-9, 1e-6, 0.01):
        sched = kdiff.fix_steps_sigmas(make, 20, d)
        assert max(calls) <= kdiff._FIX_STEPS_MAX_SCHEDULE
        assert len(sched) - 1 == 20
