"""CLIP-L text encoder for FLUX (the 768-dim pooled vector ``y``).

The shipped ``clip_l.safetensors`` is a bare ``CLIPTextModel`` (diffusers
layout, every key prefixed ``text_model.``) with no ``text_projection``.
Per the reference Flux engine it is used with ``text_projection=False`` and
``final_layer_norm=True``: the 768-dim ``y`` is the ``pooler_output`` (the
first-token embedding after the encoder's final layer norm).
"""

from __future__ import annotations

import torch
from transformers import CLIPTextConfig, CLIPTextModel


def load_clip_l(path: str, device):
    """Load ``clip_l.safetensors`` into a fp16 ``CLIPTextModel`` on ``device``."""
    import safetensors.torch as st

    cfg = CLIPTextConfig(
        vocab_size=49408,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=77,
        model_type="clip_text",
    )
    model = CLIPTextModel(cfg)
    file = st.load_file(path)
    remapped = {k[len("text_model."):]: v for k, v in file.items() if k.startswith("text_model.")}
    missing, unexpected = model.load_state_dict(remapped, strict=True), None
    model = model.to(device=device, dtype=torch.float16)
    model.eval()
    return model


def clip_pooled(model, input_ids) -> torch.Tensor:
    """(b, 77) token ids -> (b, 768) pooled vector ``y``."""
    with torch.no_grad():
        out = model(input_ids.to(torch.long))
    return out.pooler_output
