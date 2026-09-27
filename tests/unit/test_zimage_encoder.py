"""Pure parts of the Z-Image Qwen3 encoder (imagagen.qwen3_encoder): the
prompt template (read from the ComfyUI reference source) and the scaled-fp8
linear / rotary math, without loading the 4.4 GB checkpoint.
"""

import torch
from torch.nn import functional as F

from imagagen.qwen3_encoder import (
    COMPUTE_DTYPE,
    Qwen3Model,
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
    def test_only_the_weight_scale_dequantizes(self):
        """``scale_input`` is the reference's ACTIVATION quantization step:
        it cancels against the fp8 input quantization, so the dequantized
        matmul is ``F.linear(x, W_fp8 * scale_weight)``. Applying the input
        scale to the activation instead shrinks every layer output by an
        arbitrary per-layer factor and silently destroys the conditioning
        (dev/check_qwen_lm.py, dev/check_zimage_encoder.py)."""
        g = torch.Generator().manual_seed(0)
        lin = ScaledFp8Linear(4, 3)
        lin.weight.data = torch.randn(3, 4, generator=g)
        lin.scale_weight.data = torch.tensor([0.013])
        lin.scale_input.data = torch.tensor([0.025])
        x = torch.randn(2, 4, generator=g)
        out = lin(x)
        assert torch.allclose(out, F.linear(x, lin.weight * 0.013), atol=1e-7)
        assert not torch.allclose(out, F.linear(x * 0.025, lin.weight * 0.013),
                                  atol=1e-6)

    def test_scale_input_is_ignored(self):
        lin = ScaledFp8Linear(2, 2)
        lin.weight.data = torch.ones(2, 2)
        lin.scale_weight.data = torch.tensor([2.0])
        x = torch.tensor([[3.0, -1.0]])
        # each output is 2 * (3 - 1) = 4, whatever scale_input says
        for si in (1.0, 0.5, 0.003):
            lin.scale_input.data = torch.tensor([si])
            assert torch.allclose(lin(x), torch.full((1, 2), 4.0), atol=1e-7)


class TestQwen3CausalityAndLayer:
    """Structural checks on a tiny (randomly initialised) Qwen3 stack -- no
    checkpoint needed."""

    def _tiny(self):
        torch.manual_seed(0)
        model = Qwen3Model(vocab_size=32, hidden_size=16, intermediate_size=32,
                           num_layers=4, n_heads=2, n_kv_heads=2, head_dim=8)
        # the fp8 scales are uninitialized until a checkpoint is loaded;
        # identity scales keep the tiny model numerically sane
        for m in model.modules():
            if hasattr(m, "scale_weight"):
                m.scale_weight.data.fill_(1.0)
                m.scale_input.data.fill_(1.0)
        return model

    def test_attention_is_causal(self):
        model = self._tiny().eval()
        ids = torch.tensor([[3, 4, 5, 6, 7]])
        with torch.no_grad():
            base = self._layers(model, ids)
            # changing the LAST token must not move any earlier hidden state
            other = ids.clone()
            other[0, -1] = 9
            with torch.no_grad():
                changed = self._layers(model, other)
        for i, (a, b) in enumerate(zip(base, changed)):
            assert torch.equal(a[:, :-1], b[:, :-1]), f"layer {i} is not causal"
        assert not torch.equal(base[-1][:, -1], changed[-1][:, -1])

    def test_returns_the_penultimate_layer_output(self):
        """The reference target is comfy's ``layer_idx=-2``: the output of
        layer ``n-2`` (0-indexed), WITHOUT the final norm."""
        model = self._tiny().eval()
        ids = torch.tensor([[3, 4, 5, 6, 7]])
        with torch.no_grad():
            every = self._layers(model, ids)  # every[i] = output of layer i
            out = model(ids)
        assert torch.equal(out, every[len(model.layers) - 2])
        assert not torch.equal(out, every[-1])

    @staticmethod
    def _layers(model, ids):
        """Drive the decoder stack one layer at a time, exactly the way
        ``Qwen3Model.forward`` does (bf16 compute dtype)."""
        from torch.nn import functional as F

        x = F.embedding(ids, model.embed_tokens.weight.to(COMPUTE_DTYPE))
        cos, sin = model._rope(x.shape[1], x.device, COMPUTE_DTYPE)
        outs = []
        for layer in model.layers:
            x = layer(x, cos, sin)
            outs.append(x)
        return outs


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
