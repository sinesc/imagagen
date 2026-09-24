"""GgufLinear runtime LoRA deltas (CPU, plain non-quantized weights).

The Flux backend applies LoRAs as runtime additive deltas on top of the
(quantized) base weight: ``out[..., r0:r1] += (x @ A.t()) @ B.t()``. These
tests exercise that forward path with float32 weights, including the
row-sliced case (fused qkv / linear1 slices), the concatenated multi-delta
case (one entry per module after the backend's merge), and the
empty/cleared case (zero overhead == base forward).
"""

import pytest
import torch
import torch.nn.functional as F

from imagagen.flux_lora import merge_module_deltas
from imagagen.gguf_ops import GgufLinear


def make_linear(in_f: int, out_f: int, seed: int = 0) -> GgufLinear:
    """A GgufLinear with a plain (non-quantized) float32 weight."""
    lin = GgufLinear(in_f, out_f)
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(out_f, in_f, generator=g)
    w.gguf_type = 1  # F16 marker: dequantize_tensor passes it through
    lin.weight = w
    return lin


def rand_ab(in_f: int, out_f: int, rank: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(rank, in_f, generator=g)
    B = torch.randn(out_f, rank, generator=g)
    return A, B


def test_forward_matches_manual_delta():
    lin = make_linear(32, 48)
    A, B = rand_ab(32, 48, 4, seed=1)
    lin.set_lora_deltas([(A, B, 0, 48)])
    x = torch.randn(2, 5, 32, generator=torch.Generator().manual_seed(2))
    out = lin(x)
    expected = F.linear(x, lin.weight) + (x @ A.t()) @ B.t()
    assert torch.allclose(out, expected, atol=1e-5)


def test_row_sliced_delta_leaves_other_rows_base():
    lin = make_linear(32, 96, seed=3)
    r0, r1 = 16, 80
    A, B = rand_ab(32, 96 - 0, 4, seed=4)  # B covers r1 - r0 = 64 rows
    B = B[: r1 - r0]
    lin.set_lora_deltas([(A, B, r0, r1)])
    x = torch.randn(3, 7, 32, generator=torch.Generator().manual_seed(5))
    base = F.linear(x, lin.weight)
    out = lin(x)
    assert torch.allclose(out[..., :r0], base[..., :r0], atol=1e-6)
    assert torch.allclose(out[..., r1:], base[..., r1:], atol=1e-6)
    expected = base[..., r0:r1] + (x @ A.t()) @ B.t()
    assert torch.allclose(out[..., r0:r1], expected, atol=1e-5)


def test_qkv_style_concatenated_delta():
    # the backend's merge concatenates q/k/v deltas into ONE fused-slice
    # delta: verify the merged (r0=0, r1=3H) delta equals the sum of the
    # three per-projection deltas through a real forward
    in_f, H = 32, 32  # small analog: fused out = 3H
    g = torch.Generator().manual_seed(7)
    Aq, Bq = rand_ab(in_f, H, 4, seed=8)
    Ak, Bk = rand_ab(in_f, H, 4, seed=9)
    Av, Bv = rand_ab(in_f, H, 4, seed=10)
    merged = merge_module_deltas({"m": {
        (0, H): [(Aq, Bq)], (H, 2 * H): [(Ak, Bk)], (2 * H, 3 * H): [(Av, Bv)],
    }})
    assert len(merged["m"]) == 1  # ONE delta after the merge
    lin = make_linear(in_f, 3 * H, seed=6)
    lin.set_lora_deltas(merged["m"])
    x = torch.randn(2, 4, in_f, generator=g)
    out = lin(x)
    # the three row blocks of the expected delta: one per projection
    delta = torch.cat([(x @ Aq.t()) @ Bq.t(), (x @ Ak.t()) @ Bk.t(),
                       (x @ Av.t()) @ Bv.t()], dim=-1)
    assert torch.allclose(out, F.linear(x, lin.weight) + delta, atol=1e-5)


def test_merge_same_slice_stacks_ranks():
    # two LoRAs on the same (module, slice): A stacks along the rank dim,
    # B along its rank dim (exact, no padding)
    A1, B1 = rand_ab(16, 32, 3, seed=1)  # rank 3
    A2, B2 = rand_ab(16, 32, 5, seed=2)  # rank 5
    merged = merge_module_deltas({"m": {(0, 32): [(A1, B1), (A2, B2)]}})
    A, B, r0, r1 = merged["m"][0]
    assert (r0, r1) == (0, 32)
    assert A.shape == (8, 16) and B.shape == (32, 8)
    lin = make_linear(16, 32, seed=3)
    lin.set_lora_deltas([(A, B, r0, r1)])
    x = torch.randn(1, 2, 16)
    expected = (F.linear(x, lin.weight)
                + (x @ A1.t()) @ B1.t() + (x @ A2.t()) @ B2.t())
    assert torch.allclose(lin(x), expected, atol=1e-5)


def test_two_disjoint_delta_entries():
    # two entries on the same module (non-adjacent slices) both apply
    lin = make_linear(16, 64, seed=11)
    A1, B1 = rand_ab(16, 32, 3, seed=12)  # B covers 32 rows (its slice)
    A2, B2 = rand_ab(16, 32, 3, seed=13)
    lin.set_lora_deltas([(A1, B1, 0, 32), (A2, B2, 32, 64)])
    x = torch.randn(1, 3, 16, generator=torch.Generator().manual_seed(14))
    out = lin(x)
    base = F.linear(x, lin.weight)
    assert torch.allclose(out[..., :32],
                          base[..., :32] + (x @ A1.t()) @ B1.t(), atol=1e-5)
    assert torch.allclose(out[..., 32:],
                          base[..., 32:] + (x @ A2.t()) @ B2.t(), atol=1e-5)


def test_empty_deltas_equals_base():
    lin = make_linear(16, 32)
    x = torch.randn(2, 2, 16)
    base = F.linear(x, lin.weight)
    assert torch.allclose(lin(x), base)
    lin.set_lora_deltas([])
    assert torch.allclose(lin(x), base)


def test_set_replaces_previous_and_clear_removes():
    lin = make_linear(16, 32, seed=21)
    A, B = rand_ab(16, 32, 2, seed=22)
    lin.set_lora_deltas([(A, B, 0, 32)])
    x = torch.randn(1, 1, 16)
    with_delta = lin(x)
    # replacing with an empty set (and: clearing) restores the base
    lin.set_lora_deltas([])
    base = F.linear(x, lin.weight)
    assert torch.allclose(lin(x), base)
    assert not torch.allclose(with_delta, lin(x))
    # a second set replaces the first entirely (no stacking)
    lin.set_lora_deltas([(2.0 * A, B, 0, 32)])
    assert torch.allclose(
        lin(x), base + 2.0 * (x @ A.t()) @ B.t(), atol=1e-5)


def test_deltas_are_registered_parameters():
    lin = make_linear(8, 8)
    A, B = rand_ab(8, 8, 2, seed=31)
    lin.set_lora_deltas([(A, B, 0, 8)])
    names = dict(lin.named_parameters(recurse=False, remove_duplicate=True))
    assert set(names) == {"dummy", "lora_A_0", "lora_B_0"}
    assert torch.allclose(lin.lora_deltas[0][0], A)
    assert torch.allclose(lin.lora_deltas[0][1], B)
    lin.clear_lora_deltas()
    assert lin.lora_deltas == []
    assert "lora_A_0" not in dict(
        lin.named_parameters(recurse=False, remove_duplicate=True))


def test_delta_dtype_cast_to_weight_dtype():
    # the backend hands over bf16 deltas (the on-disk dtype); the layer
    # must cast them to the weight's dtype
    lin = make_linear(16, 32, seed=41)
    A32, B32 = rand_ab(16, 32, 2, seed=42)
    A = A32.to(torch.bfloat16)
    B = B32.to(torch.bfloat16)
    lin.set_lora_deltas([(A, B, 0, 32)])
    assert lin.lora_deltas[0][0].dtype == torch.float32
    assert lin.lora_deltas[0][1].dtype == torch.float32
    x = torch.randn(1, 1, 16)
    out = lin(x)
    expected = F.linear(x, lin.weight) + (x @ A.float().t()) @ B.float().t()
    assert torch.allclose(out, expected, atol=1e-2)
