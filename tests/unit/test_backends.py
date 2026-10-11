import numpy as np
import pytest

from imagagen.backends import GenerateRequest, MockBackend


def test_mock_deterministic():
    b = MockBackend()
    r1 = b.generate(GenerateRequest(prompt="x", width=32, height=16, seed=42))
    r2 = b.generate(GenerateRequest(prompt="x", width=32, height=16, seed=42))
    assert r1.seed == 42 == r2.seed
    assert np.array_equal(r1.image, r2.image)


def test_mock_seed_minus_one_random_but_valid():
    b = MockBackend()
    r = b.generate(GenerateRequest(prompt="x", width=8, height=8, seed=-1))
    assert 0 <= r.seed < 2**31


def test_mock_encodes_size_and_seed():
    b = MockBackend()
    a = b.generate(GenerateRequest(prompt="x", width=32, height=32, seed=1)).image
    bimg = b.generate(GenerateRequest(prompt="x", width=32, height=32, seed=2)).image
    c = b.generate(GenerateRequest(prompt="x", width=64, height=32, seed=1)).image
    assert a.shape == (32, 32, 3)
    assert c.shape == (32, 64, 3)
    assert not np.array_equal(a, bimg)  # different seed plane
    assert not np.array_equal(a, c)  # different geometry


def test_mock_progress_called():
    from imagagen.backends import PHASE_DENOISING, PHASE_ENCODING

    calls = []
    b = MockBackend()
    b.generate(
        GenerateRequest(prompt="x", width=8, height=8, steps=5, seed=0),
        progress=lambda f, m: calls.append((f, m)),
    )
    assert [f for f, m in calls if not m.startswith("phase:")] == [
        0.2, 0.4, 0.6, 0.8, 1.0]
    # the generation phases are reported too (encoding, then sampling)
    assert [m for f, m in calls if m.startswith("phase:")] == [
        PHASE_ENCODING, PHASE_DENOISING]


def test_mock_cancel_mid_flight_and_recovery():
    b = MockBackend()

    def progress(f, _m):
        if f >= 0.4:
            b.cancel()

    with pytest.raises(InterruptedError):
        b.generate(
            GenerateRequest(prompt="x", width=8, height=8, steps=5, seed=0), progress=progress
        )
    # a fresh generation after a cancel works fine
    r = b.generate(GenerateRequest(prompt="x", width=8, height=8, steps=2, seed=1))
    assert r.seed == 1


def test_mock_init_blending():
    b = MockBackend()
    init = np.zeros((16, 16, 3), dtype=np.uint8)
    init[..., 0] = 200
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[:, 8:] = 255  # right half gets generated
    r = b.generate(
        GenerateRequest(
            prompt="x",
            width=16,
            height=16,
            seed=7,
            init_image=init,
            init_mask=mask,
            denoise=1.0,
        )
    )
    # left half should be the untouched init image
    assert np.all(r.image[:, :8, 0] == 200)
    # right half should differ from init
    assert not np.all(r.image[:, 8:] == init[:, 8:])
