"""CUDA GPU discovery."""

from __future__ import annotations


def available_gpus() -> list[int]:
    """Indices of visible CUDA devices (respects CUDA_VISIBLE_DEVICES).

    Returns an empty list when torch/CUDA is unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def gpu_info() -> list[dict]:
    """Human-readable info per visible GPU (name, free/total MiB)."""
    out = []
    try:
        import torch

        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            out.append(
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "free_mib": round(free / 2**20),
                    "total_mib": round(total / 2**20),
                }
            )
    except Exception:
        pass
    return out
