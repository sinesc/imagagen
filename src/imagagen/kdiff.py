"""Self-contained k-diffusion sampling routines.

Vendored (MIT, madebyollin/k-diffusion) and trimmed to what imagagen needs:
the ancestral/multistep samplers and the sigma schedules. Avoids the PyPI
`k_diffusion` package (it drags in torchvision for features we don't use),
but DPM++ 2M SDE needs `torchsde`'s BrownianTree for its correlated SDE
noise (the same dependency Forge's k_diffusion uses).

Model contract (matches k-diffusion): the wrapped model is called as
``model(x, sigma)`` and returns the denoised x0 estimate for the noise level
``sigma``. Samplers consume a 1-D tensor of sigmas that ends in 0.
"""

from __future__ import annotations

import math
import warnings

import torch
import torchsde

# torchsde warns when a sigma query sits within fp rounding of the tree's
# rounded [sigma_min, sigma_max] bounds; the values are clamped, the noise is
# unaffected (Forge's path emits the same warnings)
warnings.filterwarnings(
    "ignore", message=r"^Should have (ta>=t0|tb<=t1) but got "
)

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
    ramp = torch.linspace(1, 0, n, device=device) ** rho
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


# per-step SDE/ancestral noise seeds must not collide with the initial-noise
# seed (same seed + shape + device + dtype -> identical randn draw)
_STEP_SEED_OFFSET = 1_000_000


def get_sigmas_flux(n: int, image_seq_len: int, device,
                    base_seq_len: int = 256, max_seq_len: int = 4096,
                    base_shift: float = 0.5, max_shift: float = 1.15) -> torch.Tensor:
    """FLUX (flow-matching) sigma schedule with the resolution-dependent shift.

    The base sigmas are ``linspace(1.0, 1/n, n)``; ``mu`` is a linear function
    of the image token count (``image_seq_len`` = ``(h//16)*(w//16)``) and an
    exponential time shift redistributes the steps (more steps spent at high
    noise). A terminal 0 is appended (the sampler integrates down to sigma 0).
    """
    t = torch.linspace(1.0, 1.0 / n, n, device=device)
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    exp_mu = math.exp(mu)
    t = exp_mu / (exp_mu + (1.0 / t - 1.0))
    return torch.cat([t, torch.zeros(1, device=device, dtype=t.dtype)])


def get_sigmas_krea2(n: int, image_seq_len: int, device,
                     base_seq_len: int = 256, max_seq_len: int = 6400,
                     base_shift: float = 0.5, max_shift: float = 1.15,
                     mu: float | None = None) -> torch.Tensor:
    """Krea 2 (flow-matching) sigma schedule with the resolution-dependent
    shift (the reference ``sampling.py::timesteps``).

    ``mu`` is a linear function of the image token count (``image_seq_len``
    = ``(h // 16) * (w // 16)``, the reference's minres 256 px -> 256
    tokens and maxres 1280 px -> 6400 tokens) and the exponential time
    shift (``exp(mu) / (exp(mu) + (1/t - 1))``) redistributes the steps
    (more steps spent at high noise). Pass an explicit ``mu`` to pin the
    shift regardless of resolution (the reference's ``--mu``; the
    distilled/turbo checkpoints were trained at the fixed ``mu = 1.15``).
    A terminal 0 is appended (the sampler integrates down to sigma 0).
    """
    # the reference's uniform grid INCLUDES the terminal 0 (n + 1 points,
    # n intervals); the shift maps 0 -> 0 (exp(mu) / (exp(mu) + inf))
    t = torch.linspace(1.0, 0.0, n + 1, device=device)
    if mu is None:
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
    exp_mu = math.exp(mu)
    return exp_mu / (exp_mu + (1.0 / t - 1.0))


def get_sigmas_zimage(n: int, device: str = "cpu", shift: float = 3.0) -> torch.Tensor:
    """Z-Image (flow-matching) sigma schedule with the linear-SNR shift.

    Port of the ComfyUI reference's default (KSampler "simple") schedule for
    Z-Image: a 1000-slot buffer of ``sigma(t) = shift*t / (1 + (shift-1)*t)``
    (ComfyUI ``time_snr_shift``) is sampled at ``t = (1000 - int(i * 1000/n))
    / 1000`` for ``i = 0..n-1`` (the buffer slot ``-(1 + int(i*ss))``), so
    the first sigma is exactly 1.0 (t=1 shifts to 1) and the grid equals the
    official ``linspace(1, 1/n, n)`` shifted schedule whenever n divides
    1000 (the standard step counts). A terminal 0 is appended (the sampler
    integrates down to sigma 0).
    """
    def _shift(t: float) -> float:
        return shift * t / (1.0 + (shift - 1.0) * t)

    ss = 1000 / n
    sigmas = torch.tensor(
        [_shift((1000 - int(i * ss)) / 1000.0) for i in range(n)],
        device=device,
    )
    return torch.cat([sigmas, torch.zeros(1, device=device, dtype=sigmas.dtype)])


# ---------------------------------------------------------------- euler ----


@torch.no_grad()
def sample_euler(model, x, sigmas, extra_args=None, callback=None):
    """Plain (non-ancestral) Euler step, flow-matching / x0 contract.

    The wrapped model is called as ``model(x, sigma)`` and returns the x0
    estimate; the velocity is ``v = (x - x0) / sigma`` and the step integrates
    ``x += v * (sigma_next - sigma)``. This is FLUX's sampler.
    """
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i],
                      "denoised": denoised})
        sigma = sigmas[i]
        x = x + (x - denoised) / sigma * (sigmas[i + 1] - sigma)
    return x


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


# ------------------------------------------------------------------- lcm ----


@torch.no_grad()
def sample_lcm(model, x, sigmas, extra_args=None, callback=None, noise_sampler=None):
    """LCM (A1111/Forge's `sample_lcm`): pure ancestral sampling. Every step
    jumps straight to the model's x0 estimate and re-noises to the next
    sigma (no ODE derivative, no eta/s_noise). The paired model wrap
    (sdxl_backend's LCM path) quantizes each sigma to one of the 50
    original LCM timesteps (19, 39, ..., 999) and applies the CompVis x0
    rescaling, so the distilled model only ever sees noise levels it was
    trained on."""
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    seed = extra_args.get("seed", 0)
    s_in = x.new_ones([x.shape[0]])
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i],
                      "denoised": denoised})
        x = denoised
        if sigmas[i + 1] > 0:
            noise = noise_sampler(x.shape, _STEP_SEED_OFFSET + seed + i * x.shape[0],
                                  offset=x.shape[0])
            x = x + noise * sigmas[i + 1]
    return x


# ---------------------------------------------------------------- dpm++ ----
# DPM-Solver++(2M) as used by A1111/Forge (verified: its first step is
# exactly the simple Euler step, so it is consistent with the verified
# Euler-a path in this codebase).


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


def _brownian_sort(a, b):
    return (a, b, 1) if a < b else (b, a, -1)


class BrownianTreeNoiseSampler:
    """k-diffusion's BrownianTreeNoiseSampler (vendored, MIT,
    madebyollin/k-diffusion -- identical to Forge's vendored copy).

    A single Brownian path over the full [sigma_min, sigma_max] interval
    (torchsde.BrownianTree, seeded from the generation seed): the noise
    injected on consecutive SDE steps is the increment of that one path, so
    adjacent increments are correlated (negative between neighbors) and
    largely cancel. Independent per-step noise instead accumulates and gives
    high-contrast, hard-edged images.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None):
        t0, t1, self.sign = _brownian_sort(
            torch.as_tensor(sigma_min), torch.as_tensor(sigma_max))
        w0 = torch.zeros_like(x)
        # the tree itself lives on CPU (k-diffusion's cpu=True default)
        self.tree = torchsde.BrownianTree(
            t0.detach().cpu(), w0.detach().cpu(), t1.detach().cpu(), entropy=seed)

    def __call__(self, sigma, sigma_next):
        t0, t1, sign = _brownian_sort(torch.as_tensor(sigma), torch.as_tensor(sigma_next))
        device, dtype = t0.device, t0.dtype
        t0, t1 = t0.detach().cpu().float(), t1.detach().cpu().float()
        w = self.tree(t0, t1).to(device=device, dtype=dtype) * (self.sign * sign)
        # normalize to unit variance (the raw increment has variance |t1 - t0|)
        return w / (torch.as_tensor(sigma_next) - torch.as_tensor(sigma)).abs().sqrt()


@torch.no_grad()
def sample_dpmpp_2m_sde(model, x, sigmas, extra_args=None, callback=None, eta: float = 1.0,
                        s_noise: float = 1.0, noise_sampler=None):
    """DPM-Solver++(2M)[SDE], ported from A1111/Forge's vendored k-diffusion
    (modules_forge/packages/k_diffusion/sampling.py, which mirrors ComfyUI's).

    One SDE update per step (instead of an ODE step to sigma_down followed
    by independent noise, as Euler a does): the 2M update is expressed in
    half-log-SNR time (lambda = -log sigma for the x0 model, sigma_data = 1)
    with the correlated Brownian-tree noise; the final step (sigma_next = 0)
    is a full denoise. solver_type is "midpoint" (Forge's default)."""
    if len(sigmas) <= 1:
        return x

    extra_args = {} if extra_args is None else extra_args
    seed = extra_args.get("seed", None)
    sigma_min = sigmas[sigmas > 0].min()
    sigma_max = sigmas.max()
    if noise_sampler is None:
        noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=seed)
    s_in = x.new_ones([x.shape[0]])

    old_denoised = None
    h = h_last = None
    for i in range(len(sigmas) - 1):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i],
                      "denoised": denoised})
        if sigmas[i + 1] == 0:
            # final step: full denoise, no noise
            x = denoised
            old_denoised = denoised
            continue
        lambda_s, lambda_t = sigmas[i].log().neg(), sigmas[i + 1].log().neg()
        h = lambda_t - lambda_s
        h_eta = h * (eta + 1)
        alpha_t = sigmas[i + 1] * lambda_t.exp()
        x = (sigmas[i + 1] / sigmas[i]) * (-h * eta).exp() * x \
            + alpha_t * (-h_eta).expm1().neg() * denoised
        if old_denoised is not None:
            r = h_last / h
            # 2M multistep correction (midpoint solver)
            x = x + 0.5 * alpha_t * (-h_eta).expm1().neg() * (1 / r) * (denoised - old_denoised)
        if eta > 0 and s_noise > 0:
            x = x + noise_sampler(sigmas[i], sigmas[i + 1]) * sigmas[i + 1] \
                * (-2 * h * eta).expm1().neg().sqrt() * s_noise
        old_denoised = denoised
        h_last = h
    return x
