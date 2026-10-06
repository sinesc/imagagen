"""SDXL Forge-style chunked CLIP encoding (_encode_prompt_chunked).

Drives the encoder on a CPU-only stub backend (constructed with ``__new__``
so the multi-GB checkpoints are never loaded) with fake tokenizers/encoders.
Regression: an empty prompt must produce a single empty window
([BOS] + [EOS] + pad), not crash with
``not enough values to unpack (expected 2, got 0)`` (outpainting presets
with an empty prompt hit this in the wild).
"""

import types

import torch

from imagagen.sdxl_backend import SdxlBackend

DIM_L, DIM_G = 8, 4


class FakeTokenizer:
    """Maps each character to a token id; mirrors the CLIP framing ids."""

    def __init__(self, bos, eos, pad):
        self.bos_token_id, self.eos_token_id, self.pad_token_id = bos, eos, pad

    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(c) % 1000 + 10 for c in text]
        out = types.SimpleNamespace(
            input_ids=torch.tensor([ids], dtype=torch.long))
        return out


class FakeEncoder(torch.nn.Module):
    """Returns hidden_states whose penultimate entry is (1, seq, dim).

    ``out[0]`` is 2-D for CLIP-G (pooled) and 3-D for CLIP-L, matching the
    ndim check the backend uses to pick the pooled output.
    """

    def __init__(self, dim, pooled):
        super().__init__()
        self.dim, self._pooled = dim, pooled

    def forward(self, ids, output_hidden_states=True):
        # embeddings keyed off the token ids so tests can assert framing
        hidden = ids.unsqueeze(-1).to(torch.float32) * torch.ones(1, 1, self.dim)
        first = (torch.zeros(1, self.dim) if self._pooled
                 else torch.zeros(1, 1, self.dim))
        # [unused, hidden, hidden]: hidden_states[-2] picks the real hidden
        return _EncOut(first, [torch.zeros(1, 1, 1), hidden, hidden])


class _EncOut:
    def __init__(self, first, hidden_states):
        self._first = first
        self.hidden_states = hidden_states

    def __getitem__(self, idx):
        return self._first if idx == 0 else self.hidden_states[idx]


def _stub():
    b = SdxlBackend.__new__(SdxlBackend)
    b.pipe = types.SimpleNamespace(
        tokenizer=FakeTokenizer(bos=49406, eos=49407, pad=49407),
        text_encoder=FakeEncoder(DIM_L, pooled=False),
        tokenizer_2=FakeTokenizer(bos=49406, eos=49407, pad=0),
        text_encoder_2=FakeEncoder(DIM_G, pooled=True),
    )
    return b


class TestChunkedEncode:
    def test_empty_prompt_encodes_single_framed_window(self):
        b = _stub()
        embeds, pooled = b._encode_prompt_chunked("", "cpu")
        # one window of 77 tokens, CLIP-L and CLIP-G concatenated on features
        assert embeds.shape == (1, 77, DIM_L + DIM_G)
        assert pooled.shape == (1, DIM_G)

    def test_only_comment_prompt_also_empty(self):
        b = _stub()
        embeds, _ = b._encode_prompt_chunked("(())", "cpu")
        assert embeds.shape == (1, 77, DIM_L + DIM_G)

    def test_short_prompt_single_window(self):
        b = _stub()
        embeds, pooled = b._encode_prompt_chunked("cat", "cpu")
        assert embeds.shape == (1, 77, DIM_L + DIM_G)
        assert pooled.shape == (1, DIM_G)

    def test_long_prompt_splits_into_75_token_windows(self):
        b = _stub()
        prompt = "x" * 100  # 100 content tokens -> windows of 75 + 25
        embeds, _ = b._encode_prompt_chunked(prompt, "cpu")
        assert embeds.shape == (1, 77 + 77, DIM_L + DIM_G)

    def test_weighted_prompt_encodes(self):
        b = _stub()
        embeds, _ = b._encode_prompt_chunked("(cat:1.2)", "cpu")
        assert embeds.shape == (1, 77, DIM_L + DIM_G)
