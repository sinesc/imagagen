"""Z-Image chat prompt template.

Local copy of the ``llama_template`` from the ComfyUI reference
(``comfy/text_encoders/z_image.py``) -- kept verbatim in the repo so
the project does not depend on the untracked reference checkout.
"""
ZIMAGE_PROMPT_TEMPLATE = '<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n'


def zimage_prompt_template() -> str:
    """Return the Z-Image chat prompt template (with ``{}`` placeholder)."""
    return ZIMAGE_PROMPT_TEMPLATE
