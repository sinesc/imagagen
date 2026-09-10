"""Self-contained k-diffusion sampling routines.

Vendored (MIT, madebyollin/k-diffusion) and trimmed to what imagagen needs:
the ancestral/multistep samplers and the sigma schedules. Avoids the PyPI
`k_diffusion` package because it drags in torchvision/torchsde for features
we don't use.

Model contract (matches k-diffusion): the wrapped model is called as
``model(x, sigma)`` and returns the denoised x0 estimate for the noise level
``sigma``. Samplers consume a 1-D tensor of sigmas that ends in 0.
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------- sigmas ----


def append_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x.new_zeros([1])])


def get_sigmas_karras(n: int, sigma_min: float, sigma_max: float, rho: float = 7.0,
                      device: str = "cpu") -> torch.Tensor:
    """Noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n: int, sigma_min: float, sigma_max: float,
                           device: str = "cpu") -> torch.Tensor:
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n,
                            device=device).exp()
    return append_zero(sigmas)


def get_sigmas_polyexponential(n: int, sigma_min: float, sigma_max: float,
                               rho: float = 1.0, device: str = "cpu") -> torch.Tensor:
    ramp = torch.linspace(1, 0, n) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)


# ---------------------------------------------------------------- helpers ----


def to_d(x: torch.Tensor, sigma, denoised: torch.Tensor) -> torch.Tensor:
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / sigma


def get_ancestral_step(sigma_from, sigma_to, eta: float = 1.0):
    if not eta:
        return sigma_to, 0.
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def default_noise_sampler(x: torch.Tensor):
    """k-diffusion's noise sampler: deterministic per-step noise derived
    from the generation seed (STEP_SEED_OFFSET + step_index * batch).

    The offset keeps the step seeds clear of the initial-noise seed: a
    torch.randn with the same seed, shape, device and dtype reproduces the
    same draw, and step 0 would then re-inject the initial noise itself,
    inflating the latent variance and degrading the image.
    """

    def noise_sampler(shape, seed, offset=0):
        g = torch.Generator(device=x.device).manual_seed(seed)
        return torch.randn(shape, generator=g, device=x.device, dtype=x.dtype)

    return noise_sampler


# ---------------------------------------------------------------- euler ----


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, eta: float = 1.0,
                           s_noise: float = 1.0, noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    seed = extra_args.get("seed", 0)
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        d = to_d(x, sigmas[i], denoised)
        dt = sigma_down - sigmas[i]
        x = x + d * dt
        if sigmas[i + 1] > 0:
            noise = noise_sampler(x.shape, _STEP_SEED_OFFSET + seed + i * x.shape[0],
                                  offset=x.shape[0])
            x = x + noise * s_noise * sigma_up
    return x


# ---------------------------------------------------------------- dpm++ ----
# DPM-Solver++(2M) as used by A1111/Forge (verified: its first step is
# exactly the simple Euler step, so it is consistent with the verified
# Euler-a path in this codebase).


# per-step SDE noise seeds must not collide with the initial-noise seed
# (same seed + shape + device + dtype -> identical randn draw); see
# default_noise_sampler._STEP_SEED_OFFSET
_STEP_SEED_OFFSET = 1_000_000


def _dpmpp_2m_denoised(denoised, old_denoised, t, t_prev, t_next) -> torch.Tensor:
    """Midpoint-corrected denoised estimate (2M multistep term)."""
    if old_denoised is None:
        return denoised
    r = (t - t_prev) / (t_next - t)
    return (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised


def _sigma_from_t(t):
    return t.neg().exp()


def _t_from_sigma(sigma):
    return sigma.log().neg()


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None):
    """DPM-Solver++(2M) (https://arxiv.org/abs/2211.01095)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i],
                      "denoised": denoised})
        t, t_next = _t_from_sigma(sigmas[i]), _t_from_sigma(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (_sigma_from_t(t_next) / _sigma_from_t(t)) * x - (-h).expm1() * denoised
        else:
            denoised_d = _dpmpp_2m_denoised(denoised, old_denoised, t, _t_from_sigma(sigmas[i - 1]), t_next)
            x = (_sigma_from_t(t_next) / _sigma_from_t(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, eta: float = 1.0,
                        s_noise: float = 1.0, noise_sampler=None):
    """DPM-Solver++(2M) with ancestral SDE steps: a 2M ODE step down to
    sigma_down, then independent noise up to the scheduled sigma (the same
    ancestral decomposition as Euler a)."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    seed = extra_args.get("seed", 0)
    s_in = x.new_ones([x.shape[0]])
    old_denoised = None
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i],
                      "denoised": denoised})
        sigma_down, sigma_up = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
        t, t_down = _t_from_sigma(sigmas[i]), _t_from_sigma(sigma_down)
        if old_denoised is None or sigma_down == 0:
            denoised_d = denoised
        else:
            denoised_d = _dpmpp_2m_denoised(denoised, old_denoised, t, _t_from_sigma(sigmas[i - 1]), t_down)
        h = t_down - t
        x = (_sigma_from_t(t_down) / _sigma_from_t(t)) * x - (-h).expm1() * denoised_d
        if sigmas[i + 1] > 0 and sigma_up > 0:
            noise = noise_sampler(x.shape, _STEP_SEED_OFFSET + seed + i * x.shape[0],
                                  offset=x.shape[0])
            x = x + noise * s_noise * sigma_up
        old_denoised = denoised
    return x
