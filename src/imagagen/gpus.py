"""CUDA GPU discovery and selection."""

from __future__ import annotations


def available_gpus(ignored: frozenset[int] = frozenset()) -> list[int]:
    """Indices of visible CUDA devices (respects CUDA_VISIBLE_DEVICES),
    excluding any explicitly ignored indices.

    Returns an empty list when torch/CUDA is unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return [i for i in range(torch.cuda.device_count()) if i not in ignored]
    except Exception:
        return []


def _probe_free(gpu: int) -> int:
    """Free VRAM on ``gpu`` in bytes, or 0 when the probe fails.

    A GPU that cannot even initialize a CUDA context (almost fully
    occupied by another process) is reported as 0, the same as a fully
    occupied context.
    """
    try:
        import torch

        free, _ = torch.cuda.mem_get_info(gpu)
        return int(free)
    except Exception:
        return 0


def usable_gpus(ignored: frozenset[int] = frozenset()) -> list[int]:
    """Visible, non-ignored GPUs that can init a CUDA context.

    Unlike :func:`available_gpus`, this probes each device; a GPU that is
    already fully occupied by another process is excluded rather than
    later failing with a sticky CUDA OOM.
    """
    return [i for i in available_gpus(ignored) if _probe_free(i) > 0]


def select_gpu(primary: int | None = None,
               ignored: frozenset[int] = frozenset()) -> int | None:
    """Choose the GPU for generation.

    An explicit ``primary`` wins (when it is a usable, non-ignored
    device); otherwise the usable non-ignored GPU with the most free VRAM
    is picked. Returns ``None`` when no usable GPU exists (an explicit
    primary that is fully occupied is therefore an error, not a silent
    fallback that would OOM).
    """
    usable = usable_gpus(ignored)
    if primary is not None:
        return primary if primary in usable else None

    best: int | None = None
    best_free = -1
    for i in usable:
        free = _probe_free(i)
        if free > best_free:
            best, best_free = i, free
    return best


def gpu_order(primary: int | None = None,
              ignored: frozenset[int] = frozenset()) -> list[int]:
    """Ordered list of usable GPUs: primary first, then the rest by index.

    This is the single source of truth for the generation pool. When no
    usable GPU exists the list is empty (callers fall back to a CPU/mock
    worker); when the requested primary is unusable the most-free usable
    GPU becomes the head instead.
    """
    usable = usable_gpus(ignored)
    if not usable:
        return []
    p = select_gpu(primary, ignored)
    if p not in usable:
        p = max(usable, key=lambda g: _probe_free(g))
    return [p] + [g for g in usable if g != p]


def gpu_info(ignored: frozenset[int] = frozenset(),
             selected: int | None = None) -> list[dict]:
    """Human-readable info per visible GPU (name, free/total MiB, flags).

    Querying a GPU that cannot even initialize a CUDA context (almost
    fully occupied) raises a sticky OOM error, so each device is guarded
    individually.
    """
    out = []
    try:
        import torch

        for i in range(torch.cuda.device_count()):
            free = _probe_free(i)
            try:
                free_bytes, total = torch.cuda.mem_get_info(i)
                name: str | None = torch.cuda.get_device_name(i)
                free_mib: int | None = round(int(free_bytes) / 2**20)
                total_mib: int | None = round(total / 2**20)
                note: str | None = None
            except Exception:
                name = free_mib = total_mib = None
                note = "no context (out of VRAM)"
            if i not in ignored and free <= 0:
                note = note or "occupied (no usable VRAM)"
            out.append(
                {
                    "index": i,
                    "name": name,
                    "free_mib": free_mib,
                    "total_mib": total_mib,
                    "ignored": i in ignored,
                    "selected": i == selected,
                    **({"note": note} if note else {}),
                }
            )
    except Exception:
        pass
    return out
