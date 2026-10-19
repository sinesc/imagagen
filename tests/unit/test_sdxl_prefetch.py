"""Regression tests for the SDXL prompt-change prefetch.

``SdxlBackend.prepare_text_encode`` passed diffusers'
``silence_dtype_warnings`` kwarg to the CLIP text encoders. Those are
*transformers* models: ``PreTrainedModel.to`` forwards unknown kwargs
straight to ``torch.nn.Module.to``, which rejects them with a TypeError --
so every prefetch after the first generation (once the encoders were parked
to make room for sampling) raised, 27 tracebacks in a single session, and
the encode layout was never prepared. The pipeline-level ``unload()`` /
``load()`` calls are safe because ``PipelineMixin.to`` pops the kwarg before
recursing into the components.
"""

import threading

import torch

from imagagen.backends.sdxl import SdxlBackend


class _FakePipe:
    """Just enough pipeline for prepare_text_encode: two plain nn.Modules.

    nn.Module is the faithful stand-in here -- what matters is that
    ``to()`` rejects unknown kwargs, exactly like the real transformers
    CLIPTextModel does.
    """

    def __init__(self, with_te2: bool = True):
        self.text_encoder = torch.nn.Linear(4, 4)
        self.text_encoder_2 = torch.nn.Linear(4, 4) if with_te2 else None


def _backend(pipe) -> SdxlBackend:
    b = object.__new__(SdxlBackend)
    b._lock = threading.Lock()
    b.pipe = pipe
    # meta: a device the CPU-built encoders are NOT on, so the move really
    # runs, without needing a GPU (cpu -> meta needs no copy)
    b.device = torch.device("meta")
    b.text_encoder_device = "meta"
    return b


def test_prepare_text_encode_moves_both_encoders():
    b = _backend(_FakePipe())
    b.prepare_text_encode()
    assert next(b.pipe.text_encoder.parameters()).device.type == "meta"
    assert next(b.pipe.text_encoder_2.parameters()).device.type == "meta"


def test_prepare_text_encode_twice_is_a_noop():
    b = _backend(_FakePipe())
    b.prepare_text_encode()
    b.prepare_text_encode()  # already there: the guard path
    assert next(b.pipe.text_encoder.parameters()).device.type == "meta"


def test_prepare_text_encode_without_te2():
    b = _backend(_FakePipe(with_te2=False))
    b.prepare_text_encode()
    assert next(b.pipe.text_encoder.parameters()).device.type == "meta"


def test_prepare_text_encode_skips_when_encoders_run_on_cpu():
    b = _backend(_FakePipe())
    b.text_encoder_device = "cpu"
    b.prepare_text_encode()
    assert next(b.pipe.text_encoder.parameters()).device.type == "cpu"
