# -*- coding: ascii -*-
"""v1049 -- a FIRST FRAME and <Picture i> references in ONE MiniMax H3
conditioning (scene continuity in ref2va).

THE GAP, MEASURED IN CORE'S SOURCE (7a131a3a, comfy/model_base.py,
MiniMaxH3.extra_conds): with keyframes AND refs, the payload's
cond_video_latents is written from the keyframes and then OVERWRITTEN from
the refs -- while PackedLayout still lays out BOTH (keyframe rows right after
the text, then the reference rows). The keyframe's rows would be filled with
the wrong latents.

THE FIX: a wrapper around MiniMaxH3.extra_conds that, ONLY for keyframes we
mark (PLS_KEY), writes cond_video_latents = keyframe latents + reference
latents -- the order PackedLayout lays them out in. Everything else passes
through untouched; nothing is patched until a conditioning actually carries a
marked keyframe next to references (lazy). The same merge as AIMixer's
ComfyUI_MiniMaxH3_Director (Apache-2.0, h3_context_patches.py), written here
on its own.

Coexistence: when the Director's (or ComfyUI-H3-Motion-Context's) payload
patch is already installed, we do not stack a second wrapper; our keyframe
then also carries the Director's marker key, which makes ITS merge do the
same thing. The Director refuses to install its patch AFTER ours ("another
pack already patched"): use one pack's continuity per ComfyUI session.
"""
import logging

PLS_KEY = "pls_with_refs"
MARK = "_pls_h3_kf_refs_patch"
FOREIGN = ("_h3_director_continuity_payload_patch", "_h3_motion_context_payload_patch")
DIRECTOR_KF_KEY = "director_context_index"     # their merge looks for this key

log = logging.getLogger("PolyhedronLoRAStack.h3_kf_refs")


def merge_payload(out, kwargs):
    """The merge itself (pure): out = extra_conds' result dict."""
    kfs = kwargs.get("minimax_keyframes") or []
    refs = kwargs.get("minimax_refs") or []
    if not kfs or not refs or not any(isinstance(k, dict) and k.get(PLS_KEY) for k in kfs):
        return out
    cond = out.get("minimax_payload") if isinstance(out, dict) else None
    payload = getattr(cond, "cond", None)
    if not isinstance(payload, dict):
        log.warning("[PLS] H3 first frame + refs: the payload was not reachable -- core changed?")
        return out
    payload["cond_video_latents"] = ([k["latent"] for k in kfs if "latent" in k]
                                     + [r["latent"] for r in refs if "latent" in r])
    fc = kwargs.get("minimax_frame_count")
    if fc is not None:
        payload["frame_count"] = fc
    return out


def _wrap(orig):
    def _pls_extra_conds(self, **kwargs):
        return merge_payload(orig(self, **kwargs), kwargs)
    setattr(_pls_extra_conds, MARK, True)
    _pls_extra_conds._pls_orig = orig
    return _pls_extra_conds


def owner(cls):
    fn = getattr(cls, "extra_conds", None)
    if fn is None:
        return "none"
    if getattr(fn, MARK, False):
        return "ours"
    if any(getattr(fn, m, False) for m in FOREIGN):
        return "director"
    return "core"


def ensure(cls=None):
    """Install the wrapper (once). -> "ours" | "director" | "none"."""
    if cls is None:
        try:
            import comfy.model_base as mb
            cls = getattr(mb, "MiniMaxH3", None)
        except Exception:
            cls = None
    if cls is None:
        return "none"
    who = owner(cls)
    if who == "core":
        cls.extra_conds = _wrap(cls.extra_conds)
        print("[PLS] H3: first frame + <Picture> references in one conditioning enabled (MiniMaxH3.extra_conds wrapped)")
        return "ours"
    return who


def mark(keyframe, who):
    """Mark a keyframe dict for the merge (and for the Director's, when its
    patch is the installed one)."""
    keyframe[PLS_KEY] = True
    if who == "director":
        keyframe[DIRECTOR_KF_KEY] = int(keyframe.get("resolved_frame_index", 0))
    return keyframe
