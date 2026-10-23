# -*- coding: ascii -*-
"""The Media Loader's placeholder frame -- made here, and recognised here (v1216).

Since v509 the Media Loader emits a built-in dummy frame when there is nothing to load (on_empty 'placeholder':
no selection, an empty folder, a file that vanished) -- a dark checkerboard with a thin accent border, so a graph
keeps running instead of stopping on a red error.

Frank's first first -> last frame run (23.10., PH_093901_FLF_SIT_A): both keyframe files were not found, the
loaders sent this frame, and MiniMax Keyframes pinned it at frame 0 and at the last frame. H3 obeyed: an iris
from the checkerboard into the room, the motion, an iris back -- and the orange border on both sides of the whole
clip. A full render on a picture nobody chose.

So the frame is made in ONE place (the loader calls make_np) and every node that PINS or REFERENCES a picture asks
is_placeholder() and refuses by name: the Cine Timeline (an anchor's source), MiniMax Keyframes, MiniMax Reference,
MiniMax Continue (anchors) and Control Fit. The test is exact -- the 512 x 512 frame equal to make_np() within
1/1000 -- so no real picture is ever taken for it: the loader hands the tensor on unchanged, and anything that
resized or re-encoded it is no longer the placeholder (and no longer the loader's empty state either).

Pure numpy (torch tensors are read through .detach().cpu().numpy()), no ComfyUI import. ASCII only.
"""
import numpy as np

SIZE = 512
CELL = 32
BORDER = 6
ACCENT = (0.90, 0.49, 0.38)     # the pack's accent
TOL = 1e-3

_REF = None


def make_np():
    """The placeholder frame [SIZE, SIZE, 3] float32 0..1 (v509, moved here from ph_media_loader in v1216)."""
    s = SIZE
    yy, xx = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
    checker = (((yy // CELL) + (xx // CELL)) % 2).astype(np.float32)
    rgb = np.empty((s, s, 3), dtype=np.float32)
    rgb[..., 0] = 0.10 + 0.05 * checker
    rgb[..., 1] = 0.10 + 0.05 * checker
    rgb[..., 2] = 0.11 + 0.05 * checker
    edge = (yy < BORDER) | (yy >= s - BORDER) | (xx < BORDER) | (xx >= s - BORDER)
    rgb[edge] = np.array(ACCENT, dtype=np.float32)
    return rgb


def _ref():
    global _REF
    if _REF is None:
        _REF = make_np()
    return _REF


def _frames(img):
    """IMAGE [B,H,W,C] / [H,W,C] (torch or numpy) -> numpy [B,H,W,C], or None."""
    if img is None:
        return None
    try:
        a = img.detach().to("cpu").float().numpy() if hasattr(img, "detach") else np.asarray(img, dtype=np.float32)
    except Exception:
        return None
    if a.ndim == 3:
        a = a[None]
    if a.ndim != 4:
        return None
    return a


def is_placeholder(img):
    """True when the picture (or EVERY frame of a batch) is the Media Loader's placeholder frame."""
    a = _frames(img)
    if a is None or a.shape[1] != SIZE or a.shape[2] != SIZE or a.shape[3] < 3:
        return False
    ref = _ref()
    return all(float(np.abs(f[..., :3] - ref).max()) <= TOL for f in a)


def refusal(node, slot):
    """The one sentence every node raises."""
    return ("[PLS] %s: %s is the Media Loader's placeholder (a dark checkerboard with an orange border) -- the "
            "loader found nothing to load: the file is missing, renamed or in another folder, or nothing is "
            "selected. Nothing was rendered on it; pick the picture in the Media Loader." % (node, slot))


def refuse(node, slot, img):
    """Raise ValueError(refusal) when img is the placeholder; else return img unchanged."""
    if is_placeholder(img):
        raise ValueError(refusal(node, slot))
    return img
