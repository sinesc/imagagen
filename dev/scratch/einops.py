"""Minimal einops.shim for the reference mmdit.py (scratch test only)."""
import re

import torch


def rearrange(x: torch.Tensor, pattern: str, **sizes):
    pattern = pattern.replace(" ", "")
    if pattern == "bnd(ij)->bndij":
        shp = x.shape  # (..., d/2, 4) -> (..., d/2, 2, 2)
        return x.view(*shp[:-1], 2, 2)
    if pattern == "BHLD->BL(HD)":
        b, h, l, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, l, h * d)
    if pattern == "two(d)->1two(d)" or pattern == "twod->1twod":
        return x.unsqueeze(0)
    if pattern == "BL(HD)->BHLD":
        b, l, hd = x.shape
        h = sizes["H"]
        return x.view(b, l, h, hd // h).permute(0, 2, 1, 3)
    if pattern == "(bl)nd->bldn":
        b, l = sizes["b"], sizes["l"]
        n, d = x.shape[1], x.shape[2]
        return x.view(b, l, n, d).permute(0, 1, 3, 2)
    raise NotImplementedError(pattern)
