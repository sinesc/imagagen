"""GPU selection logic (mocked torch.cuda, no real GPU needed)."""

from __future__ import annotations


def _mock_cuda(monkeypatch, count: int, free_bytes) -> None:
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda i: (free_bytes(i), 16 * 2**30))


def test_auto_picks_most_free(monkeypatch):
    from imagagen.gpus import select_gpu

    _mock_cuda(monkeypatch, 4, lambda i: (i + 1) * 10**9)
    assert select_gpu(None, frozenset()) == 3


def test_auto_respects_ignored(monkeypatch):
    from imagagen.gpus import select_gpu

    _mock_cuda(monkeypatch, 4, lambda i: (i + 1) * 10**9)
    assert select_gpu(None, frozenset({3})) == 2
    assert select_gpu(None, frozenset({2, 3})) == 1


def test_explicit_primary_wins(monkeypatch):
    from imagagen.gpus import select_gpu

    _mock_cuda(monkeypatch, 4, lambda i: (i + 1) * 10**9)
    assert select_gpu(0, frozenset()) == 0


def test_explicit_ignored_or_missing_returns_none(monkeypatch):
    from imagagen.gpus import select_gpu

    _mock_cuda(monkeypatch, 4, lambda i: (i + 1) * 10**9)
    assert select_gpu(1, frozenset({1})) is None
    assert select_gpu(9, frozenset()) is None


def test_all_ignored_returns_none(monkeypatch):
    from imagagen.gpus import select_gpu

    _mock_cuda(monkeypatch, 2, lambda i: (i + 1) * 10**9)
    assert select_gpu(None, frozenset({0, 1})) is None


def test_no_cuda(monkeypatch):
    import torch

    from imagagen.gpus import select_gpu

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert select_gpu(None, frozenset()) is None
    assert select_gpu(0, frozenset()) is None
