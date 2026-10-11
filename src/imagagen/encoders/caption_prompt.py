"""Fixed caption template for reference-image captioning.

The Qwen3-VL chat stream the captioner runs: a user turn with the
vision block (vision_start / image-pad / vision_end) and a fixed
instruction, then the assistant opener. The template is stored as
CAPTION_PARTS -- a list of (kind, text) tuples: kind is a chat
marker NAME ("im_start", "im_end", "vision_start", "vision_end",
"pad") whose token id is inserted directly, or "text" (the second
element is BPE-tokenized verbatim). The single "pad" part expands
to one image-pad token per merged vision row (see
:mod:`imagagen.reference`).

The markers are referenced by NAME only: the tokenizer dir ships
with special=None (only im_start/im_end split from text), so the
captioner assembles the token stream by parts. Do not echo the
assembled stream in logs or chat output.
"""

#: (kind, text); for marker kinds text is None
CAPTION_PARTS = [
    ("im_start", None),
    ("text", "user\n"),
    ("vision_start", None),
    ("pad", None),
    ("vision_end", None),
    ("text", "\nDescribe the image in detail: subject, clothing, "
              "art style, colors, composition, lighting."),
    ("im_end", None),
    ("text", "\n"),
    ("im_start", None),
    ("text", "assistant\n"),
]
