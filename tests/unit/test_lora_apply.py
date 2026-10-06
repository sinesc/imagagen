"""Regression tests for SdxlBackend LoRA apply/unload atomicity.

These run against a fake pipeline (no GPU / no real weights) that mimics the
two relevant diffusers behaviours:

* ``load_lora_weights`` refuses to reuse an adapter name that is still
  registered (``Adapter name ... already in use in the model``), and
* a failed injection can leave the previously-injected adapters in place.

The bug this guards: ``_apply_loras`` used to commit ``_active_loras`` only on
full success, but a multi-LoRA set that fails partway left ``imagagen_0``
injected while ``_active_loras`` stayed ``None``. The next load then skipped
the unload and re-hit ``imagagen_0`` -> "already in use", and kept failing
until the server was restarted.
"""

import pytest

from imagagen.backends.sdxl import SdxlBackend

A = "DMD2fp16XL1.0"
B = "DMD2fp32XL1.0"


class _FakeUnet:
    def __init__(self):
        self.peft_config = {}


class _FakePipe:
    """Mimics the diffusers pipeline surface used by ``_apply_loras``."""

    def __init__(self, fail_adapters=()):
        self.unet = _FakeUnet()
        self._fail = set(fail_adapters)
        self.set_adapters_calls = []

    def load_lora_weights(self, state_dict, adapter_name=None, **kwargs):
        # diffusers refuses to reuse a live adapter name.
        if adapter_name in self.unet.peft_config:
            raise ValueError(
                f"Adapter name {adapter_name} already in use in the model - "
                "please select a new adapter name."
            )
        if adapter_name in self._fail:
            # the name gets registered (as a partial inject would) then the
            # load blows up -- the earlier adapters stay in place.
            self.unet.peft_config[adapter_name] = object()
            raise RuntimeError(f"simulated load failure for {adapter_name}")
        self.unet.peft_config[adapter_name] = object()
        return None

    def unload_lora_weights(self):
        self.unet.peft_config = {}

    def set_adapters(self, tags, weights):
        self.set_adapters_calls.append((list(tags), list(weights)))


class _Stub:
    """Carries the state/attributes ``_apply_loras``/``_unload_loras`` need."""

    def __init__(self, pipe, missing=()):
        self.pipe = pipe
        self._active_loras = None
        self._missing = set(missing)

    def _lora_path(self, name):
        if name in self._missing:
            return None
        return f"/fake/{name}.safetensors"

    def _lora_state_dict(self, path):
        return {"unet.mid.lora_A.weight": 0}

    # bind the real methods under test
    _apply_loras = SdxlBackend._apply_loras
    _unload_loras = SdxlBackend._unload_loras


def make_stub(fail_adapters=(), missing=()):
    pipe = _FakePipe(fail_adapters)
    return _Stub(pipe, missing=missing), pipe


def test_normal_single_lora_applies_and_caches():
    stub, pipe = make_stub()
    stub._apply_loras([(A, 0.7)])
    assert stub._active_loras == ((A, 0.7),)
    assert list(pipe.unet.peft_config) == ["imagagen_0"]
    # second identical call is a cache hit: no reload, no set_adapters
    pipe.set_adapters_calls.clear()
    stub._apply_loras([(A, 0.7)])
    assert pipe.set_adapters_calls == []


def test_switching_lora_set_unloads_then_reloads():
    stub, pipe = make_stub()
    stub._apply_loras([(A, 0.7)])
    stub._apply_loras([(B, 0.5)])
    assert stub._active_loras == ((B, 0.5),)
    assert list(pipe.unet.peft_config) == ["imagagen_0"]  # B takes slot 0
    # and back to empty
    stub._apply_loras([])
    assert stub._active_loras is None
    assert pipe.unet.peft_config == {}


def test_partial_load_failure_leaves_pipeline_clean():
    """The regression: a set that fails on its 2nd LoRA must not strand
    imagagen_0, or the next load fails with 'already in use'."""
    stub, pipe = make_stub(fail_adapters={"imagagen_1"})

    with pytest.raises(RuntimeError, match="simulated load failure"):
        stub._apply_loras([(A, 0.7), (B, 0.7)])

    # the failure must have cleaned up: no leftover adapters, no stale cache
    assert pipe.unet.peft_config == {}
    assert stub._active_loras is None

    # the next load (even a subset) works instead of 'already in use'
    stub._apply_loras([(A, 0.7)])
    assert stub._active_loras == ((A, 0.7),)
    assert list(pipe.unet.peft_config) == ["imagagen_0"]


def test_recovers_from_stranded_adapter_state():
    """Even if adapters are present with _active_loras=None (the exact broken
    state), the next apply clears them before loading."""
    stub, pipe = make_stub()
    # simulate the broken state directly
    pipe.unet.peft_config["imagagen_0"] = object()
    stub._active_loras = None

    stub._apply_loras([(A, 0.7)])  # would raise 'already in use' pre-fix
    assert stub._active_loras == ((A, 0.7),)
    assert list(pipe.unet.peft_config) == ["imagagen_0"]


def test_missing_lora_raises_and_keeps_current_set():
    """A missing file must not leave the pipeline half-unloaded: the
    currently applied set stays active (path resolution precedes the
    unload)."""
    stub, pipe = make_stub(missing={"nope"})
    stub._apply_loras([(A, 0.7)])
    with pytest.raises(ValueError, match="LoRA not found"):
        stub._apply_loras([("nope", 1.0)])
    # the previously applied set is untouched
    assert stub._active_loras == ((A, 0.7),)
    assert list(pipe.unet.peft_config) == ["imagagen_0"]
