"""ModelRegistry.prefetch_text_encoder status strings.

A broken ``prepare_text_encode`` used to be indistinguishable from "the GPU
is busy": the API answered ``busy`` (HTTP 200) while the feature was raising
on every resident GPU -- 27 tracebacks in one session without a single
accurate status.
"""

import threading

from imagagen.backends import MockBackend
from imagagen.config import Config
from imagagen.models import ModelRegistry, _GpuState


class _PrefetchBackend(MockBackend):
    """MockBackend with an observable prepare_text_encode."""

    def __init__(self, *, boom: bool = False):
        super().__init__()
        self.model_id = "sdxl/model"
        self.boom = boom
        self.prepared = 0

    def encoding_cached(self, prompt: str, negative: str) -> bool:
        return False  # always a cache miss: the prepare path must run

    def prepare_text_encode(self) -> None:
        if self.boom:
            raise RuntimeError("prepare_text_encode is broken")
        self.prepared += 1


def _state(backend: _PrefetchBackend, gpu: int = 0) -> _GpuState:
    s = _GpuState(gpu)
    s.backend = backend
    s.model_id = backend.model_id
    return s


def _registry(tmp_path, *states: _GpuState) -> ModelRegistry:
    reg = ModelRegistry(Config(root=tmp_path))
    reg._gpu_states = {s.gpu: s for s in states}
    return reg


def test_prefetch_prepares_every_resident_gpu(tmp_path):
    a, b = _PrefetchBackend(), _PrefetchBackend()
    reg = _registry(tmp_path, _state(a, 0), _state(b, 1))
    assert reg.prefetch_text_encoder("sdxl/model", "a dog", "") == "prepared"
    assert (a.prepared, b.prepared) == (1, 1)


def test_prefetch_reports_failure_not_busy(tmp_path, caplog):
    reg = _registry(tmp_path, _state(_PrefetchBackend(boom=True), 0))
    assert reg.prefetch_text_encoder("sdxl/model", "a dog", "") == "failed"
    assert "prompt prefetch failed on GPU 0" in caplog.text


def test_prefetch_busy_when_a_gpu_is_generating(tmp_path):
    backend = _PrefetchBackend()
    s = _state(backend, 0)
    # the GPU lock is an RLock, so a same-thread acquire would succeed: hold
    # it from another thread, the way a worker inside generate() does
    held, stop = threading.Event(), threading.Event()

    def hold_lock():
        s.lock.acquire()
        held.set()
        stop.wait(5)
        s.lock.release()

    t = threading.Thread(target=hold_lock, daemon=True)
    t.start()
    assert held.wait(5)
    try:
        reg = _registry(tmp_path, s)
        assert reg.prefetch_text_encoder("sdxl/model", "a dog", "") == "busy"
    finally:
        stop.set()
        t.join(5)
    assert backend.prepared == 0


def test_prefetch_never_leaks_the_gpu_lock(tmp_path):
    s = _state(_PrefetchBackend(boom=True))
    _registry(tmp_path, s).prefetch_text_encoder("sdxl/model", "a dog", "")
    assert s.lock.acquire(blocking=False), "a failed prefetch leaked the lock"
    s.lock.release()


def test_prefetch_cached_and_not_resident(tmp_path):
    backend = _PrefetchBackend()
    backend.encoding_cached = lambda p, n: True
    reg = _registry(tmp_path, _state(backend))
    assert reg.prefetch_text_encoder("sdxl/model", "a dog", "") == "cached"
    assert reg.prefetch_text_encoder("other/model", "a dog", "") == "not-resident"
