"""Vendored: madebyollin/taehv (MIT) -- Tiny AutoEncoder for video latents.

v918: brought in for the MiniMax H3 decoder (`taeh3`, 24 latent channels,
patch 2). ComfyUI Core carries the same architecture only from its
2026-09 nightlies (`is_h3` in comfy/taesd/taehv.py); the pack must not
depend on a Core newer than the one it was field-tested on (0.33.4,
rev 7a131a3a of 2026-08-24). See SOURCE.md for the pin.

Only `TAEHV`, `apply_model_with_memblocks` are used; the file is verbatim.
"""
from .taehv import TAEHV, apply_model_with_memblocks  # noqa: F401
