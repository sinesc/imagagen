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
    return lambda sigma, sigma_next: torch.randn_like(x)


# ---------------------------------------------------------------- euler ----


@torch.no_grad()
def sample_euler_ancestral(model, x, sigmas, extra_args=None, callback=None, eta: float = 1.0,
                           s_noise: float = 1.0, noise_sampler=None):
    """Ancestral sampling with Euler method steps."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
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
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * s_noise * sigma_up
    return x


# ---------------------------------------------------------------- dpmpp ----


class _DPMSolver:
    """DPM-Solver step kernels (https://arxiv.org/abs/2206.00927)."""

    def __init__(self, model, extra_args=None, eps_callback=None, info_callback=None):
        self.model = model
        self.extra_args = {} if extra_args is None else extra_args
        self.eps_callback = eps_callback
        self.info_callback = info_callback

    def t(self, sigma):
        return -sigma.log()

    def sigma(self, t):
        return t.neg().exp()

    def eps(self, eps_cache, key, x, t, *args, **kwargs):
        if key in eps_cache:
            return eps_cache[key], eps_cache
        sigma = self.sigma(t) * x.new_ones([x.shape[0]])
        eps = (x - self.model(x, sigma, *args, **self.extra_args, **kwargs)) / self.sigma(t)
        if self.eps_callback is not None:
            self.eps_callback()
        return eps, {key: eps, **eps_cache}

    def dpm_solver_1_step(self, x, t, t_next, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        x_1 = x - self.sigma(t_next) * h.expm1() * eps
        return x_1, eps_cache

    def dpm_solver_2_step(self, x, t, t_next, r1=1 / 2, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        x_2 = (x - self.sigma(t_next) * h.expm1() * eps
               - self.sigma(t_next) / (2 * r1) * h.expm1() * (eps_r1 - eps))
        return x_2, eps_cache

    def dpm_solver_3_step(self, x, t, t_next, r1=1 / 3, r2=2 / 3, eps_cache=None):
        eps_cache = {} if eps_cache is None else eps_cache
        h = t_next - t
        eps, eps_cache = self.eps(eps_cache, "eps", x, t)
        s1 = t + r1 * h
        s2 = t + r2 * h
        u1 = x - self.sigma(s1) * (r1 * h).expm1() * eps
        eps_r1, eps_cache = self.eps(eps_cache, "eps_r1", u1, s1)
        u2 = (x - self.sigma(s2) * (r2 * h).expm1() * eps
              - self.sigma(s2) * (r2 / r1) * (((r2 * h).expm1() / (r2 * h)) - 1) * (eps_r1 - eps))
        eps_r2, eps_cache = self.eps(eps_cache, "eps_r2", u2, s2)
        x_3 = (x - self.sigma(t_next) * h.expm1() * eps
               - self.sigma(t_next) / r2 * ((h.expm1() / h) - 1) * (eps_r2 - eps))
        return x_3, eps_cache


@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None):
    solver = _DPMSolver(model, extra_args=extra_args, info_callback=callback)
    ts = [solver.t(sigma) for sigma in sigmas]
    for i in range(len(sigmas) - 1):
        t, t_next = ts[i], ts[i + 1]
        x, _ = solver.dpm_solver_2_step(x, t, t_next)
    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, eta: float = 1.0,
                        s_noise: float = 1.0, noise_sampler=None, solver_type: str = "midpoint"):
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    solver = _DPMSolver(model, extra_args=extra_args, info_callback=callback)
    ts = [solver.t(sigma) for sigma in sigmas]
    for i in range(len(sigmas) - 1):
        t, t_next = ts[i], ts[i + 1]
        if i == 0:
            x, _ = solver.dpm_solver_1_step(x, t, t_next)
        elif solver_type == "midpoint":
            x, _ = solver.dpm_solver_3_step(x, t, t_next)
        else:
            x, _ = solver.dpm_solver_2_step(x, t, t_next)
        if sigmas[i + 1] > 0:
            sd, su = get_ancestral_step(sigmas[i], sigmas[i + 1], eta=eta)
            su = (sigmas[i + 1] ** 2 - sd ** 2) ** 0.5
            x = x + su * s_noise * noise_sampler(sigmas[i], sigmas[i + 1])
    return x
