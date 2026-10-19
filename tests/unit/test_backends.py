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


# ---- init image/mask -> tensor (Pillow exports read-only buffers) --------

def test_device_tensor_accepts_readonly_pillow_arrays():
    """No "NumPy array is not writable" warning for img2img/inpaint inputs.

    Every init image/mask reaching a backend comes from
    ``np.asarray(Image...)`` (generation.resize_image / rotate_canvas), and
    Pillow >= 12 exports its pixel buffer READ-ONLY. ``np.ascontiguousarray``
    returns that same read-only object, so the old
    ``torch.from_numpy(np.ascontiguousarray(arr))`` warned on every job.
    """
    import warnings

    import torch
    from PIL import Image

    from imagagen.backends import device_tensor

    src = np.asarray(Image.fromarray(np.full((4, 4, 3), 7, np.uint8))
                     .resize((8, 8), Image.LANCZOS))
    assert not src.flags.writeable  # the Pillow buffer is read-only

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = device_tensor(src, torch.device("cpu"))
    assert not [w for w in caught if "writable" in str(w.message)], \
        [str(w.message) for w in caught]
    assert t.shape == (8, 8, 3) and t.dtype == torch.uint8
    assert t.numpy().flags.writeable


def test_device_tensor_always_copies():
    import torch

    from imagagen.backends import device_tensor

    src = np.full((4, 4), 5, np.uint8)
    t = device_tensor(src, torch.device("cpu"))
    src.fill(9)  # must not touch the tensor (it is moved to the GPU anyway)
    assert int(t[0, 0]) == 5


def test_device_tensor_makes_non_contiguous_input_c_contiguous():
    import torch

    from imagagen.backends import device_tensor

    src = np.arange(24, dtype=np.uint8).reshape(2, 4, 3).transpose(0, 2, 1)
    assert not src.flags.c_contiguous
    t = device_tensor(src, torch.device("cpu"))
    assert t.numpy().flags.c_contiguous
    assert np.array_equal(t.numpy(), src)
