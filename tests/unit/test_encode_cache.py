"""The shared per-model encoded-prompt cache (zimage ``_get_context`` /
flux ``_get_encoding``).

An encoding is a deterministic function of the prompt (eval-mode encoders,
no dropout), so a batch of identical prompts should run the (expensive)
text encoder once and reuse the result for the rest of the batch. These
tests drive the cache helpers on CPU-only stub backends (constructed with
``__new__`` so the multi-GB checkpoints are never loaded) and check the
cache-hit count, LRU eviction, and -- for Flux -- that the negative prompt
is part of the key.
"""

import threading

import torch

from imagagen.backends import EncCache, _ENC_CACHE_MAX
from imagagen.backends.flux import FluxBackend
from imagagen.backends.zimage import ZImageBackend


def _stub(cls):
    """A backend shell with just the attributes the encode helpers touch."""
    b = cls.__new__(cls)
    b._enc_cache = EncCache()
    b._lock = threading.Lock()
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

    def test_hit_offloads_a_co_resident_encoder(self):
        # The co-resident fit check can leave the ~4.1 GiB encoder on the GPU
        # (cold load / parked restore). A cache hit then skips the encode
        # AND the post-encode offload -- the hit path must drop the encoder
        # itself, or it sits next to the transformer through sampling and
        # the untiled 1024x1280 VAE decode OOMs. (Here self.device is the
        # stand-in for the GPU: the stub encoder's params live on it.)
        b, calls = _stub(ZImageBackend)
        b.device = torch.device("cpu")  # params live on self.device = co-resident
        b.te_device = "cuda:1"
        offloads: list[bool] = []
        b._text_offload = lambda: offloads.append(True)
        b._get_context("a prompt")  # miss: encode + offload
        assert offloads == [True]
        offloads.clear()
        b._get_context("a prompt")  # hit: no re-encode, but offload still
        assert offloads == [True]

    def test_hit_keeps_cpu_encoder_on_cpu(self):
        # te_device=cpu: the encoder never leaves system RAM; the miss path
        # still offloads once, the hit must add nothing
        b, calls = _stub(ZImageBackend)
        offloads: list[bool] = []
        b._text_offload = lambda: offloads.append(True)
        b._get_context("a prompt")
        b._get_context("a prompt")
        assert offloads == [True]

    def test_pre_encode_encodes_batch_in_one_session(self):
        # a wildcard batch: every unique prompt encoded exactly once, the
        # reserve bumps the LRU cap, and the later generations are hits
        b, calls = _stub(ZImageBackend)
        b.pre_encode([("a", ""), ("b", ""), ("a", "")], reserve=10)
        assert sorted(calls) == ["a", "b"]
        assert b._enc_cache.max == 10
        calls.clear()
        b._get_context("a")
        b._get_context("b")
        assert calls == []  # both were pre-encoded: no further encoder runs


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
        # (the order between the two texts is not contractual)
        assert sorted(calls) == ["n", "p", "p"]
        assert neg2 is not None
        (c3, y3, neg3) = b._get_encoding("p", "n")
        # and the negative variant is now cached too
        assert sorted(calls) == ["n", "p", "p"]
        assert torch.equal(c2, c3) and torch.equal(neg2[0], neg3[0])

    def test_lru_eviction_caps_the_cache(self):
        b, _ = _stub(FluxBackend)
        total = _ENC_CACHE_MAX + 2
        for i in range(total):
            b._get_encoding(f"p{i}", None)
        assert len(b._enc_cache) == _ENC_CACHE_MAX
        assert ("p0", None) not in b._enc_cache
        assert (f"p{total - 1}", None) in b._enc_cache

    def test_hit_offloads_co_resident_encoders(self):
        # Same guard as the zimage test: a hit must drop the T5/CLIP
        # encoders if the co-resident fit check left them on the GPU
        b, calls = _stub(FluxBackend)
        b.device = torch.device("cpu")  # params live on self.device = co-resident
        b.te_device = "cuda:1"
        offloads: list[bool] = []
        b._text_offload = lambda: offloads.append(True)
        b._get_encoding("a prompt", None)  # miss: encode + offload
        assert offloads == [True]
        offloads.clear()
        b._get_encoding("a prompt", None)  # hit: offload still required
        assert offloads == [True]

    def test_hit_keeps_cpu_encoders_on_cpu(self):
        # the miss path still offloads once, the hit must add nothing
        b, calls = _stub(FluxBackend)
        offloads: list[bool] = []
        b._text_offload = lambda: offloads.append(True)
        b._get_encoding("a prompt", None)
        b._get_encoding("a prompt", None)
        assert offloads == [True]

    def test_pre_encode_covers_both_key_variants(self):
        # a wildcard batch pre-encodes a pair under BOTH keys a generation
        # could look up -- (p, n) and (p, None) (distilled mode) -- from one
        # encoder session per unique text
        b, calls = _stub(FluxBackend)
        b.pre_encode([("p", "n")], reserve=8)
        assert sorted(calls) == ["n", "p"]
        (c1, y1, neg1) = b._get_encoding("p", "n")
        (c2, y2, neg2) = b._get_encoding("p", None)
        assert sorted(calls) == ["n", "p"]  # no further encoder runs
        assert neg1 is not None and neg2 is None
