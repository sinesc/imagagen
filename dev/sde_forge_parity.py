"""Verify imagagen's DPM++ 2M SDE sampler against Forge's vendored k-diffusion.

Extracts `sample_dpmpp_2m_sde` (+ its Brownian-tree noise sampler) verbatim
from sd-webui-forge-classic/modules_forge/packages/k_diffusion/sampling.py
and runs both implementations on an identical deterministic model; the
results must be bit-identical.

Usage (from the project root):
    .venv/bin/python dev/sde_forge_parity.py
"""

import sys
import types
from functools import partial
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FORGE = Path(
    "/workspaces/Projekte/sd-webui-forge-classic/modules_forge/packages/k_diffusion/sampling.py"
)
from imagagen import kdiff  # noqa: E402


def load_forge_impl():
    """Exec the Forge sampler + its helpers in an isolated namespace."""
    src = FORGE.read_text()
    import ast
    tree = ast.parse(src)
    want = {
        "sigma_to_half_log_snr", "offset_first_sigma_for_snr", "_is_const",
        "BatchedBrownianTree", "BrownianTreeNoiseSampler", "sample_dpmpp_2m_sde",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in want:
            nodes.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.FunctionDef) and node.name == "_is_const":
            nodes.append(ast.get_source_segment(src, node))
    ns = {
        "torch": torch,
        "torchsde": __import__("torchsde"),
        "partial": partial,
        "trange": lambda n, **kw: range(n),
        "math": __import__("math"),
    }
    for code in nodes:
        exec(compile(code, str(FORGE), "exec"), ns)
    return ns


class DummySampling:
    """Stands in for `model.inner_model.predictor` (non-const x0 model)."""
    prediction_type = "x0"


def make_model():
    """A deterministic pseudo-model: cheap, seeded, non-linear."""
    def model(x, sigma, **extra_args):
        s = sigma.view([1] * (x.dim() - 1))
        phase = torch.sin(0.011 * x) * 3.0
        return x - s * (0.8 * torch.tanh(phase) + 0.0001 * torch.cos(7.0 * x))
    return model


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ns = load_forge_impl()

    torch.manual_seed(0)
    x0 = torch.randn(1, 4, 32, 32, device=device, dtype=torch.float32) * 10.0
    sigmas = kdiff.get_sigmas_karras(20, 0.029, 14.614, device=device)
    model = make_model()

    class Wrap:
        def __init__(self, fn):
            self.fn = fn
            self.inner_model = types.SimpleNamespace(predictor=DummySampling())

        def __call__(self, x, sigma, **extra):
            return self.fn(x, sigma, **extra)

    # --- imagagen implementation (s_in convention: model sees sigma) ---
    out_ours = kdiff.sample_dpmpp_2m_sde(Wrap(model), x0.clone(), sigmas.clone(),
                                         extra_args={"seed": 3})

    # --- Forge implementation, called exactly as Forge calls it ---
    forge_sampler = ns["sample_dpmpp_2m_sde"]
    out_forge = forge_sampler(Wrap(model), x0.clone(), sigmas.clone(),
                              extra_args={"seed": 3})

    equal = torch.equal(out_ours, out_forge)
    diff = (out_ours - out_forge).abs()
    print(f"bit-identical: {equal}")
    print(f"max abs diff:  {diff.max().item():.3e}   "
          f"(magnitudes ~ {out_ours.abs().max().item():.1f})")
    print("PASS" if equal else "FAIL")
    return 0 if equal else 1


if __name__ == "__main__":
    sys.exit(main())
