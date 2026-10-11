"""Qwen3-VL vision tower + preprocessing for reference-image captioning.

Port of the ComfyUI reference's ``Qwen35VisionModel`` /
``Qwen3VLVisionModel`` (``comfy/text_encoders/qwen35.py`` +
``qwen3vl.py``) and ``process_qwen2vl_images``
(``comfy/text_encoders/qwen_vl.py``), kept verbatim in the repo (the
project does not depend on the untracked reference checkout). The
ComfyUI ``ops`` abstraction is replaced by plain torch modules and
``F.scaled_dot_product_attention``; the checkpoint's fp8 projections
(``model.visual.*`` in ``infra/qwen_3vl_4b.safetensors``) keep their
fp8 storage and are cast per op (the ``Fp8Linear`` cast-is-dequantize
pattern, cf. :mod:`imagagen.encoders.krea2`).

The tower runs in float32 (like the reference: the image is fed as
float32 and the ops cast the weights per op); the sequence is a single
image -> one full ``cu_seqlens`` block (bidirectional attention, no
windowing: the Qwen3.5/Qwen3-VL tower has no window index, unlike
Qwen2-VL).

``process_qwen2vl_images`` resizes an image to the model's grid
(multiples of ``patch_size * merge_size``, clamped to the min/max pixel
budget) and patchifies it into ``(grid_t*grid_h*grid_w, 1536)`` rows;
the tower merges each 2x2 patch block into one 2560-dim row, so a
448x448 image yields a 28x28 patch grid -> **196 image tokens** (the
number of image-pad tokens in the caption stream).

DeepStack (Qwen3-VL): the tower also returns three feature maps
(``deepstack_visual_indexes = [5, 11, 17]``), which the text tower adds
to the image-token hidden states after its decoder layers 0/1/2 (the
HF ``_deepstack_process`` convention).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

#: Qwen3-VL image preprocessing parameters (the reference's
#: ``Qwen3VL.preprocess_embed`` call: normalize to [-1, 1])
PATCH_SIZE = 16
TEMPORAL_PATCH_SIZE = 2
MERGE_SIZE = 2
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]
MIN_PIXELS = 3136
MAX_PIXELS = 12845056

#: qwen3vl_4b vision config (QWEN3VL_VISION + QWEN3VL_VISION_COMMON in
#: comfy/text_encoders/qwen3vl.py, out_hidden_size = text hidden 2560)
VISION_CONFIG = dict(
    hidden_size=1024,
    intermediate_size=4096,
    depth=24,
    deepstack_visual_indexes=[5, 11, 17],
    num_heads=16,
    patch_size=PATCH_SIZE,
    temporal_patch_size=TEMPORAL_PATCH_SIZE,
    in_channels=3,
    spatial_merge_size=MERGE_SIZE,
    num_position_embeddings=2304,
    out_hidden_size=2560,
)

#: Qwen3-VL chat special tokens (by id; see imagagen.encoders.krea2)
VISION_START_ID = 151652
VISION_END_ID = 151653
IMAGE_PAD_ID = 151655
IM_START_ID = 151644
IM_END_ID = 151645

#: caption-stream marker name -> token id (the tokenizer dir ships with
#: special=None, so the captioner inserts these ids directly; see
#: imagagen.encoders.caption_prompt)
MARKER_IDS = {
    "im_start": IM_START_ID,
    "im_end": IM_END_ID,
    "vision_start": VISION_START_ID,
    "vision_end": VISION_END_ID,
    "pad": IMAGE_PAD_ID,
}


def process_qwen2vl_images(
    images: torch.Tensor,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    patch_size: int = PATCH_SIZE,
    temporal_patch_size: int = TEMPORAL_PATCH_SIZE,
    merge_size: int = MERGE_SIZE,
    image_mean: list | None = None,
    image_std: list | None = None,
    interpolation: str = "bilinear",
):
    """(1, H, W, 3) float image (0..1) -> ((grid_h*grid_w, 1536) patches,
    image_grid_thw (1,3)). Verbatim port of the ComfyUI reference's
    ``comfy/text_encoders/qwen_vl.py::process_qwen2vl_images`` with the
    Qwen3-VL parameters (patch 16, mean/std 0.5)."""
    if image_mean is None:
        image_mean = IMAGE_MEAN
    if image_std is None:
        image_std = IMAGE_STD

    batch_size, height, width, channels = images.shape
    device = images.device

    images = images.permute(0, 3, 1, 2)

    grid_thw_list = []
    img = images[0]

    factor = patch_size * merge_size

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor

    img_resized = F.interpolate(
        img.unsqueeze(0),
        size=(h_bar, w_bar),
        mode=interpolation,
        align_corners=False,
    ).squeeze(0)
    normalized = img_resized.clone()
    for c in range(3):
        normalized[c] = (img_resized[c] - image_mean[c]) / image_std[c]

    grid_h = h_bar // patch_size
    grid_w = w_bar // patch_size
    grid_thw = torch.tensor([1, grid_h, grid_w], device=device, dtype=torch.long)

    pixel_values = normalized
    grid_thw_list.append(grid_thw)
    image_grid_thw = torch.stack(grid_thw_list)

    grid_t = 1
    channel = pixel_values.shape[0]
    pixel_values = pixel_values.unsqueeze(0).repeat(2, 1, 1, 1)

    patches = pixel_values.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )

    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )

    return flatten_patches, image_grid_thw


def qwen2vl_mrope_position_ids(index: int, size: int, grid_thw: torch.Tensor,
                               seq_len: int, device) -> torch.Tensor:
    """(3, seq_len) T/H/W MRoPE position ids for a stream with ONE image.

    Port of the ComfyUI reference's
    ``comfy/text_encoders/qwen_vl.py::qwen2vl_mrope_position_ids``
    (single-image specialization): text runs sequentially, the image
    span (vision_start .. vision_end, ``size`` tokens starting at
    ``index``) gets its grid positions, and the text after the image
    continues at ``start_next`` (the mrope convention shared with HF).
    """
    grid = grid_thw[0]
    position_ids = torch.zeros((3, seq_len), device=device)
    position_ids[:, :index] = torch.arange(0, index, device=device)
    end = size + index
    len_max = int(grid.max()) // 2
    start_next = len_max + index
    position_ids[:, end:] = torch.arange(start_next, start_next + (seq_len - end),
                                         device=device)
    position_ids[0, index:end] = index
    max_d = int(grid[1]) // 2
    position_ids[1, index:end] = (
        torch.arange(index, index + max_d, device=device)
        .unsqueeze(1).repeat(1, math.ceil((end - index) / max_d))
        .flatten(0)[:end - index]
    )
    max_d = int(grid[2]) // 2
    position_ids[2, index:end] = (
        torch.arange(index, index + max_d, device=device)
        .unsqueeze(0).repeat(math.ceil((end - index) / max_d), 1)
        .flatten(0)[:end - index]
    )
    return position_ids


class Fp8Linear(nn.Linear):
    """``nn.Linear`` with a plain ``F8_E4M3`` weight: the cast IS the
    dequantize (the checkpoint stores no scale tensors). The vision
    tower runs in float32, so this casts fp8 -> fp32 per op."""

    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype))


class VisionPatchEmbed(nn.Module):
    """Conv3d patchifier (kernel = stride = [temporal, patch, patch])."""

    def __init__(self, patch_size, temporal_patch_size, in_channels, embed_dim):
        super().__init__()
        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size,
                              stride=kernel_size, bias=True)

    def forward(self, x):
        x = x.view(-1, self.proj.in_channels,
                   self.proj.kernel_size[0], self.proj.kernel_size[1],
                   self.proj.kernel_size[2])
        x = F.conv3d(x, self.proj.weight.to(x.dtype),
                     self.proj.bias.to(x.dtype))
        return x.view(-1, self.proj.out_channels)


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, seqlen, device):
        inv_freq = 1.0 / (self.theta ** (
            torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
            / self.dim))
        seq = torch.arange(seqlen, device=device, dtype=torch.float)
        return torch.outer(seq, inv_freq)


def _apply_rope_split_half(q, k, cos, sin, neg_sin):
    """The reference's ``apply_rope`` with the (cos, sin, -sin) triple
    (the 2x2 rotation matrix in split-half layout):
    ``out1 = q1*cos1 + q2*sin``, ``out2 = q1*neg_sin + q2*cos2``."""
    half = sin.shape[-1]

    def rot(x):
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat((x1 * cos[..., :half] + x2 * neg_sin,
                          x1 * sin + x2 * cos[..., half:]), dim=-1)

    return rot(q), rot(k)


class VisionAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = Fp8Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = Fp8Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x, cu_seqlens, position_embeddings):
        seq_length = x.shape[0]
        query, key, value = (
            self.qkv(x).reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3).unbind(0)
        )
        cos, sin, neg_sin = position_embeddings
        query, key = _apply_rope_split_half(query, key, cos, sin, neg_sin)
        # bidirectional attention over the cu_seqlens blocks (a single
        # image is one full block); the output layout is (seq, heads*dim)
        # (the reference's optimized_attention convention)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outs = []
        for q, k, v in zip(
                torch.split(query, lengths, dim=0),
                torch.split(key, lengths, dim=0),
                torch.split(value, lengths, dim=0), strict=False):
            o = F.scaled_dot_product_attention(
                q.unsqueeze(0).transpose(1, 2),
                k.unsqueeze(0).transpose(1, 2),
                v.unsqueeze(0).transpose(1, 2))
            outs.append(o.transpose(1, 2).squeeze(0))
        attn = torch.cat(outs, dim=0).reshape(seq_length, -1)
        return self.proj(attn)


class VisionMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.linear_fc1 = Fp8Linear(hidden_size, intermediate_size, bias=True)
        self.linear_fc2 = Fp8Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x):
        return self.linear_fc2(F.gelu(self.linear_fc1(x), approximate="tanh"))


class VisionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = VisionAttention(hidden_size, num_heads)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

    def _norm(self, norm, x):
        return F.layer_norm(x, norm.normalized_shape,
                            norm.weight.to(x.dtype), norm.bias.to(x.dtype),
                            norm.eps)

    def forward(self, x, cu_seqlens, position_embeddings):
        x = x + self.attn(self._norm(self.norm1, x), cu_seqlens,
                          position_embeddings)
        return x + self.mlp(self._norm(self.norm2, x))


class VisionPatchMerger(nn.Module):
    """Main merger: pre-shuffle LayerNorm(hidden), fc1, gelu(tanh), fc2."""

    def __init__(self, hidden_size, spatial_merge_size, out_hidden_size):
        super().__init__()
        self.merge_dim = hidden_size * (spatial_merge_size ** 2)
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear_fc1 = Fp8Linear(self.merge_dim, self.merge_dim, bias=True)
        self.linear_fc2 = Fp8Linear(self.merge_dim, out_hidden_size, bias=True)

    def _norm(self, x):
        return F.layer_norm(x, self.norm.normalized_shape,
                            self.norm.weight.to(x.dtype),
                            self.norm.bias.to(x.dtype), self.norm.eps)

    def forward(self, x):
        x = self._norm(x).view(-1, self.merge_dim)
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class DeepstackMerger(nn.Module):
    """DeepStack merger: post-shuffle LayerNorm (applied after spatial
    merge), unlike the main merger (the reference's
    ``Qwen3VLDeepstackMerger``)."""

    def __init__(self, hidden_size, spatial_merge_size, out_hidden_size):
        super().__init__()
        self.merge_dim = hidden_size * (spatial_merge_size ** 2)
        self.norm = nn.LayerNorm(self.merge_dim, eps=1e-6)
        self.linear_fc1 = Fp8Linear(self.merge_dim, self.merge_dim, bias=True)
        self.linear_fc2 = Fp8Linear(self.merge_dim, out_hidden_size, bias=True)

    def _norm(self, x):
        return F.layer_norm(x, self.norm.normalized_shape,
                            self.norm.weight.to(x.dtype),
                            self.norm.bias.to(x.dtype), self.norm.eps)

    def forward(self, x):
        x = self._norm(x.view(-1, self.merge_dim))
        return self.linear_fc2(F.gelu(self.linear_fc1(x)))


class Qwen3VLVisionTower(nn.Module):
    """The Qwen3-VL vision tower (``model.visual.*``). ``forward`` takes
    the patchified image (float32) + grid_thw and returns
    ``(merged (n/4, 2560), [deepstack (n/4, 2560) x 3])``."""

    def __init__(self, config=None):
        super().__init__()
        config = {**VISION_CONFIG, **(config or {})}
        self.spatial_merge_size = config["spatial_merge_size"]
        self.patch_size = config["patch_size"]
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        self.num_position_embeddings = config["num_position_embeddings"]
        self.deepstack_visual_indexes = list(config["deepstack_visual_indexes"])

        self.patch_embed = VisionPatchEmbed(
            config["patch_size"], config["temporal_patch_size"],
            config["in_channels"], config["hidden_size"])
        self.pos_embed = nn.Embedding(self.num_position_embeddings,
                                      self.hidden_size)
        self.num_grid_per_side = int(self.num_position_embeddings ** 0.5)
        self.rotary_pos_emb = VisionRotaryEmbedding(
            self.hidden_size // self.num_heads // 2)
        self.blocks = nn.ModuleList([
            VisionBlock(config["hidden_size"], config["num_heads"],
                        config["intermediate_size"])
            for _ in range(config["depth"])
        ])
        self.merger = VisionPatchMerger(
            config["hidden_size"], self.spatial_merge_size,
            config["out_hidden_size"])
        self.deepstack_merger_list = nn.ModuleList([
            DeepstackMerger(config["hidden_size"], self.spatial_merge_size,
                            config["out_hidden_size"])
            for _ in self.deepstack_visual_indexes
        ])

    def rot_pos_emb(self, grid_thw):
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw, self.pos_embed.weight.device)
        device = freq_table.device
        total_tokens = sum(int(t * h * w) for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)
        offset = 0
        for num_frames, height, width in grid_thw_list:
            num_frames, height, width = int(num_frames), int(height), int(width)
            merged_h, merged_w = height // merge_size, width // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]
            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)
            num_tokens = coords.shape[0]
            pos_ids[offset:offset + num_tokens] = coords
            offset += num_tokens
        embeddings = freq_table[pos_ids]
        return embeddings.flatten(1)

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_thw_list = grid_thw.tolist()
        grid_ts = [int(row[0]) for row in grid_thw_list]
        grid_hs = [int(row[1]) for row in grid_thw_list]
        grid_ws = [int(row[2]) for row in grid_thw_list]
        device = self.pos_embed.weight.device
        weight_dtype = self.pos_embed.weight.dtype
        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]
        for _t, h, w in grid_thw_list:  # _t: frames (single image = 1)
            h, w = int(h), int(w)
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=device)
            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor
            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for j in range(4):
                idx_list[j].extend(indices[j].tolist())
                weight_list[j].extend(weights[j].tolist())
        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(
            weight_list, dtype=torch.float32, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device).to(torch.float32) \
            * weight_tensor[:, :, None]
        patch_pos_embeds = (pos_embeds[0] + pos_embeds[1]
                            + pos_embeds[2] + pos_embeds[3])
        patch_pos_embeds = patch_pos_embeds.split(
            [h * w for h, w in zip(grid_hs, grid_ws, strict=False)])
        merge_size = self.spatial_merge_size
        out = []
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs,
                                      grid_ws, strict=False):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size,
                               w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            out.append(pos_embed.to(weight_dtype))
        return torch.cat(out)

    def forward(self, pixel_values, grid_thw):
        x = self.patch_embed(pixel_values)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw).to(x.device)
        x = x + pos_embeds.to(x.dtype)
        rotary = self.rot_pos_emb(grid_thw).to(x.device)
        seq_len = x.shape[0]
        x = x.reshape(seq_len, -1)
        rotary = rotary.reshape(seq_len, -1)
        emb = torch.cat((rotary, rotary), dim=-1)
        cos = emb.cos().unsqueeze(-2)
        sin = emb.sin().unsqueeze(-2)
        sin_half = sin.shape[-1] // 2
        position_embeddings = (cos, sin[..., :sin_half], -sin[..., sin_half:])
        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        deepstack_features = []
        for layer_num, blk in enumerate(self.blocks):
            x = blk(x, cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings)
            if layer_num in self.deepstack_visual_indexes:
                deepstack_features.append(
                    self.deepstack_merger_list[
                        self.deepstack_visual_indexes.index(layer_num)](x))
        merged = self.merger(x)
        return merged, deepstack_features


class VisionHost(nn.Module):
    """Wrapper so the tower's parameter names match the checkpoint keys
    (``model.visual.*`` -> ``visual.*`` after the ``model.`` strip)."""

    def __init__(self):
        super().__init__()
        self.visual = Qwen3VLVisionTower()


def load_vision_tower(path, device="cpu"):
    """Load the ``model.visual.*`` keys of ``qwen_3vl_4b.safetensors``
    (fp8 projections, bf16 conv/embedding/norms) into a
    :class:`VisionHost`."""
    import safetensors.torch as st

    file = st.load_file(str(path))
    model = VisionHost()
    params = dict(model.named_parameters())
    used = set()
    for key, t in file.items():
        if not key.startswith("model.visual."):
            continue
        k = key[len("model."):]
        if k not in params:
            raise ValueError(f"unexpected Qwen3-VL vision key: {key!r}")
        params[k].data = t
        used.add(k)
    missing = [k for k in params if k not in used]
    if missing:
        raise ValueError(f"vision tower is missing keys: {missing}")
    model.to(device)
    return model
