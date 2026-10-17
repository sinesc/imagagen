"""Krea 2 chat prompt template.

Local copy of the ``llama_template`` from the ComfyUI reference
(``comfy/text_encoders/krea2.py``) -- kept verbatim in the repo so the
project does not depend on the untracked reference checkout.

The template contains the chat special tokens (im_start / im_end / user /
assistant markers); do not echo it in logs or chat output.
"""

KREA2_PROMPT_TEMPLATE = (
    '<|im_start|>system\nDescribe the image by detailing the colo'
    'r, shape, size, texture, quantity, text, spatial relationshi'
    'ps of the objects and background:<|im_end|>\n<|im_start|>use'
    'r\n{}<|im_end|>\n<|im_start|>assistant\n'
)


def krea2_prompt_template() -> str:
    """Return the Krea 2 chat prompt template (with ``{}`` placeholder)."""
    return KREA2_PROMPT_TEMPLATE
