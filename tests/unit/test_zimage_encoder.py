"""Pure parts of the Z-Image Qwen3 encoder (imagagen.qwen3_encoder): the
prompt template (read from the ComfyUI reference source) and the scaled-fp8
linear / rotary math, without loading the 4.4 GB checkpoint.
"""

import torch
from torch.nn import functional as F

from imagagen.qwen3_encoder import (
    ScaledFp8Linear,
    _apply_rotary,
    zimage_prompt_template,
)


class TestPromptTemplate:
    def test_loads_from_the_reference_source(self):
        tpl = zimage_prompt_template()
        assert isinstance(tpl, str)
        assert "{}" in tpl  # the positional prompt placeholder

    def test_formats_the_prompt(self):
        tpl = zimage_prompt_template()
        out = tpl.format("hello world")
        assert "hello world" in out
        assert "{}" not in out

    def test_empty_prompt_still_formats(self):
        out = zimage_prompt_template().format("")
        assert "{}" not in out


class TestScaledFp8Linear:
    def test_matches_the_dequantized_matmul(self):
        g = torch.Generator().manual_seed(0)
        lin = ScaledFp8Linear(4, 3)
        lin.weight.data = torch.randn(3, 4, generator=g)
        lin.scale_weight.data = torch.tensor([0.013])
        lin.scale_input.data = torch.tensor([0.025])
        x = torch.randn(2, 4, generator=g)
        expected = F.linear(x * 0.025, lin.weight * 0.013)
        out = lin(x)
        assert torch.allclose(out, expected, atol=1e-7)

    def test_scales_are_applied_independently(self):
        lin = ScaledFp8Linear(2, 2)
        lin.weight.data = torch.ones(2, 2)
        lin.scale_weight.data = torch.tensor([2.0])
        lin.scale_input.data = torch.tensor([0.5])
        # y = (w * 2) @ (x * 0.5): x * 0.5 = [1.5, -0.5] -> each row is
        # 2 * (1.5 - 0.5) = 2.0
        x = torch.tensor([[3.0, -1.0]])
        out = lin(x)
        assert torch.allclose(out, torch.tensor([[2.0, 2.0]]), atol=1e-7)


class TestApplyRotary:
    def test_position_zero_is_the_identity(self):
        g = torch.Generator().manual_seed(0)
        x = torch.randn(1, 4, 8, 16, generator=g)
        # the (1, s, 1, d) shape _rope produces broadcasts over the heads
        cos = torch.ones(1, 4, 1, 16)
        sin = torch.zeros(1, 4, 1, 16)
        out = _apply_rotary(x, cos, sin)
        assert torch.allclose(out, x, atol=1e-6)

    def test_half_pair_rotation(self):
        # rotate_half pairs dim i with dim i + d/2: the value at dim 0
        # rotates into dim 2 under a 90-degree first-frequency turn
        x = torch.zeros(1, 1, 1, 4)
        x[0, 0, 0, 0] = 1.0
        cos = torch.tensor([[[[0.0, 1.0, 0.0, 1.0]]]])
        sin = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        out = _apply_rotary(x, cos, sin)
        assert torch.allclose(out, torch.tensor([[[[0.0, 0.0, 1.0, 0.0]]]]),
                              atol=1e-6)
