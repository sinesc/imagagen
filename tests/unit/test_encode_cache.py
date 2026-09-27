"""The per-backend encoded-prompt cache (zimage ``_get_context`` / flux
``_get_encoding``).

An encoding is a deterministic function of the prompt (eval-mode encoders,
no dropout), so a batch of identical prompts should run the (expensive)
text encoder once and reuse the result for the rest of the batch. These
tests drive the cache helpers on CPU-only stub backends (constructed with
``__new__`` so the multi-GB checkpoints are never loaded) and check the
cache-hit count, LRU eviction, and -- for Flux -- that the negative prompt
is part of the key.
"""

from collections import OrderedDict

import torch

from imagagen.flux_backend import _ENC_CACHE_MAX, FluxBackend
from imagagen.zimage_backend import ZImageBackend


def _stub(cls):
    """A backend shell with just the attributes the encode helpers touch."""
    b = cls.__new__(cls)
    b._enc_cache = OrderedDict()
    b.device = "cpu"
    b.te_device = "cpu"  # short-circuits the transformer-swap branch (no CUDA)
    b.name = "test"
    b._encoder_bytes = 0
    b.transformer = torch.nn.Linear(2, 2)
    b._text_to = lambda d: None
    b._text_offload = lambda: None
    calls: list[str] = []

    def encode(prompt: str):
        calls.append(prompt)
        if cls is FluxBackend:
            # _encode_prompt -> (context, y)
            return _embed(prompt), _embed(prompt) + 1
        return _embed(prompt)

    b._encode_prompt = encode
    if cls is ZImageBackend:
        b.te = torch.nn.Linear(2, 2)
    else:
        b.t5 = torch.nn.Linear(2, 2)
    return b, calls


def _embed(prompt: str) -> torch.Tensor:
    # a deterministic stand-in that varies with the prompt
    return torch.ones(1, 3, 2) * (len(prompt) + 1)


class TestZImageContextCache:
    def test_same_prompt_encodes_once(self):
        b, calls = _stub(ZImageBackend)
        c1 = b._get_context("a prompt")
        c2 = b._get_context("a prompt")
        assert calls == ["a prompt"]  # the encoder ran exactly once
        assert torch.equal(c1, c2)
        assert c1.device.type == "cpu"

    def test_distinct_prompts_encode_separately(self):
        b, calls = _stub(ZImageBackend)
        b._get_context("one")
        b._get_context("two")
        assert calls == ["one", "two"]
        assert len(b._enc_cache) == 2

    def test_lru_eviction_caps_the_cache(self):
        b, _ = _stub(ZImageBackend)
        total = _ENC_CACHE_MAX + 3
        for i in range(total):
            b._get_context(f"p{i}")
        assert len(b._enc_cache) == _ENC_CACHE_MAX
        # the two oldest are evicted, the newest is retained
        assert "p0" not in b._enc_cache
        assert "p1" not in b._enc_cache
        assert f"p{total - 1}" in b._enc_cache

    def test_reuse_refreshes_lru_order(self):
        b, calls = _stub(ZImageBackend)
        for i in range(_ENC_CACHE_MAX):
            b._get_context(f"p{i}")
        # touch the oldest (a cache hit: no re-encode) so it becomes the
        # most-recently-used
        b._get_context("p0")
        assert calls == [f"p{i}" for i in range(_ENC_CACHE_MAX)]
        # now evict by adding a new key: p1 (the new oldest) must go, p0 stays
        b._get_context("new")
        assert "p1" not in b._enc_cache
        assert "p0" in b._enc_cache
        assert "new" in b._enc_cache


class TestFluxEncodingCache:
    def test_same_prompt_no_negative_encodes_once(self):
        b, calls = _stub(FluxBackend)
        (c1, y1, n1) = b._get_encoding("a prompt", None)
        (c2, y2, n2) = b._get_encoding("a prompt", None)
        assert calls == ["a prompt"]  # the encoder ran exactly once
        assert n1 is None and n2 is None
        assert torch.equal(c1, c2) and torch.equal(y1, y2)

    def test_negative_prompt_is_part_of_the_key(self):
        b, calls = _stub(FluxBackend)
        (c, y, neg) = b._get_encoding("p", None)
        assert neg is None
        (c2, y2, neg2) = b._get_encoding("p", "n")
        # different key -> a fresh encode of BOTH the prompt and negative
        assert calls == ["p", "p", "n"]
        assert neg2 is not None
        (c3, y3, neg3) = b._get_encoding("p", "n")
        # and the negative variant is now cached too
        assert calls == ["p", "p", "n"]
        assert torch.equal(c2, c3) and torch.equal(neg2[0], neg3[0])

    def test_lru_eviction_caps_the_cache(self):
        b, _ = _stub(FluxBackend)
        total = _ENC_CACHE_MAX + 2
        for i in range(total):
            b._get_encoding(f"p{i}", None)
        assert len(b._enc_cache) == _ENC_CACHE_MAX
        assert ("p0", None) not in b._enc_cache
        assert (f"p{total - 1}", None) in b._enc_cache
