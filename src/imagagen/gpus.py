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


def select_gpu(primary: int | None = None,
               ignored: frozenset[int] = frozenset()) -> int | None:
    """Choose the GPU for generation.

    An explicit ``primary`` wins (when it is a visible, non-ignored
    device); otherwise the visible non-ignored GPU with the most free
    VRAM is picked. Returns ``None`` when no usable GPU exists.
    """
    visible = available_gpus(ignored)
    if primary is not None:
        return primary if primary in visible else None

    import torch

    best: int | None = None
    best_free = -1
    for i in visible:
        try:
            free, _ = torch.cuda.mem_get_info(i)
        except Exception:
            continue
        if free > best_free:
            best, best_free = i, free
    return best


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
            try:
                free, total = torch.cuda.mem_get_info(i)
                name: str | None = torch.cuda.get_device_name(i)
                free_mib: int | None = round(free / 2**20)
                total_mib: int | None = round(total / 2**20)
                note: str | None = None
            except Exception:
                name = free_mib = total_mib = None
                note = "no context (out of VRAM)"
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
