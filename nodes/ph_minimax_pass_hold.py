# -*- coding: ascii -*-
"""Polyhedron MiniMax Second Pass Hold (v1180) -- the second pass of a continued scene keeps its join.

THE GAP IT CLOSES: a two-pass H3 run samples small, lifts the latent (H3 Latent Upscale) and samples again at
the target size. For a scene that CONTINUES the one before, the first pass held the join (MiniMax Continue
`hold`) -- but the lift drops every mask and every guide (they were made at the small size), so the second
pass re-renders the join from an enlarged copy of the small one and is free to drift away from the scene
before at full size. This node puts the scene before back at the target size:

  - the END OF THE SCENE BEFORE at full size -- its final frames (`tail`, encoded here at the lifted latent's
    size) or its final latent (`clip_latent`, cut on the chunk grid like Continue's K2L) -- copied to the
    start of the lifted latent and held: locked, then released over `release_frames` (ph_av_mask);
  - the SOUND FROZEN (`freeze_audio`): the first pass made it, the second pass leaves it exactly as it is --
    a second pass at a new size has nothing to add to the sound and can only smear it;
  - `sigmas` OUT: a schedule for a second pass with a held join, 0.50 0.30 0.14 0.06 0 by default (Easy Media's
    preset for its context second pass -- unmeasured here; the field decides). Wire it into the second
    sampler's sigmas, or leave it and keep your own.

Without a scene before (the first scene: no tail, no clip latent) only the sound is frozen -- said.

The idea: Easy Media's "separate low- and high-resolution context" (GPL: read, not copied). The library is
ph_av_mask; the cuts are MiniMax Continue's (snap_guide, latent_window, _resize).
"""

import torch

from . import ph_av_mask as AM
from . import uls_latent_math as M

DEFAULT_SIGMAS = "0.50, 0.30, 0.14, 0.06, 0"


def parse_sigmas(text):
    """The schedule from free text (the Sigma List's parser and contract)."""
    from .wan_sigma_schedule import parse_sigma_list, validate_sigma_list
    vals = parse_sigma_list(text)
    validate_sigma_list(vals)
    return vals


class ULSMiniMaxSecondPassHold:
    """The second pass of a continued scene: the scene before held at full size, the sound frozen."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT", {"tooltip": "The LIFTED H3 AV latent (H3 Latent Upscale's latent) -- the "
                                                 "second pass's input. Wire the output into the second sampler."}),
                "context_frames": ("INT", {"default": 22, "min": 5, "max": 362, "tooltip":
                    "How much of the scene before is held at the start, in frames -- the same as MiniMax Continue's "
                    "context_frames (snapped down to 5, 22, 39 ...)."}),
                "release_frames": ("INT", {"default": 16, "min": 0, "max": 64, "step": 4, "tooltip":
                    "Over how many of the held frames the picture is released again, counted back from the join "
                    "(rounded to whole tokens -- 1 or 4 frames each; the info says what it came to). 0 = held hard "
                    "up to the join."}),
                "freeze_audio": ("BOOLEAN", {"default": True, "tooltip":
                    "Keep the first pass's sound exactly as it is (its whole stream held). Off: the second pass "
                    "may rework the sound too."}),
                "sigmas": ("STRING", {"default": DEFAULT_SIGMAS, "tooltip":
                    "The schedule given out at `sigmas` for a second pass with a held join (Easy Media's preset). "
                    "Use it by wiring the output into the second sampler; your own schedule stays if you don't."}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Video VAE -- encodes `tail` at the lifted size."}),
                "tail": ("IMAGE", {"tooltip": "The scene before's last frames as the film shows them (Cine "
                                              "Timeline `tail`) -- any size, resized to the lifted latent."}),
                "clip_latent": ("LATENT", {"tooltip": "Or the scene before's FINAL latent (Clip Chain Source "
                                                      "clip_latent, stored after ITS second pass) -- same size "
                                                      "as this latent, no VAE."}),
            },
        }

    RETURN_TYPES = ("LATENT", "SIGMAS", "STRING")
    RETURN_NAMES = ("latent", "sigmas", "info")
    FUNCTION = "hold"
    CATEGORY = "Polyhedron/Cine"
    DESCRIPTION = ("The second pass of a continued scene: puts the end of the scene before back at full size at the "
                   "start of the lifted latent and holds it (released toward the join), freezes the first pass's "
                   "sound, and gives a schedule for the second pass. Between H3 Latent Upscale and the second "
                   "sampler.")

    def hold(self, latent, context_frames, release_frames, freeze_audio, sigmas, vae=None, tail=None, clip_latent=None):
        tag = "[PLS] MiniMax Second Pass Hold"
        from .ph_minimax_continue import _resize, latent_window, snap_guide
        v, a = AM.streams(latent, tag)
        frames = M.minimax_frames_from_latent_t(int(v.shape[2]))
        h, w = int(v.shape[3]) * 16, int(v.shape[4]) * 16
        sched = parse_sigmas(sigmas)
        lines = ["MiniMax Second Pass Hold: lifted latent %dx%d, %d frames" % (w, h, frames)]
        held_v, src = None, None
        if clip_latent is not None:
            cv, _ca = AM.streams(clip_latent, tag)
            if tuple(cv.shape[3:]) != tuple(v.shape[3:]):
                raise ValueError(tag + ": the clip latent is %dx%d, the lifted latent %dx%d -- give `tail` (frames are "
                                 "resized) or a clip latent stored after the second pass"
                                 % (int(cv.shape[4]) * 16, int(cv.shape[3]) * 16, w, h))
            n = snap_guide(min(int(context_frames), M.minimax_frames_from_latent_t(int(cv.shape[2]))))
            start, n_tok = latent_window(int(cv.shape[2]), n)
            held_v = cv[:, :, start:].clone()
            src = "its final latent (tokens %d..%d, no VAE)" % (start, start + n_tok - 1)
        elif tail is not None:
            if vae is None:
                raise ValueError(tag + ": `tail` is encoded at the lifted size -- wire `vae`")
            if int(tail.shape[0]) < 5:
                raise ValueError(tag + ": `tail` needs 5 frames or more")
            n = snap_guide(min(int(context_frames), int(tail.shape[0])))
            held_v = vae.encode(_resize(tail[-n:], w, h))
            src = "its last %d frames, encoded at %dx%d" % (n, w, h)
        out = latent
        if held_v is not None:
            rel, rel_f = AM.tokens_back(int(held_v.shape[2]), release_frames)
            out, rep = AM.hold_prefix(latent, held_v, None, video_release=rel, tag=tag)
            hv = rep["video"]
            lines.append("  held: the scene before at full size -- %s: %d token(s), %d locked, %d released over its "
                         "last %d frame(s)" % (src, hv["tokens"], hv["locked"], hv["release"], rel_f))
        else:
            lines.append("  no scene before (no tail, no clip latent) -- the picture is free; this is a first scene")
        if freeze_audio:
            vv, aa = AM.streams(out, tag)
            vm, am = AM.masks(out)
            am = torch.zeros_like(am)
            out = AM.with_streams(out, vv, aa, vm, am)
            lines.append("  sound: frozen -- the first pass's sound passes the second pass unchanged")
        else:
            lines.append("  sound: free -- the second pass may rework it")
        lines.append("  sigmas out: %s (%d steps) -- wire into the second sampler to use them"
                     % (" ".join("%g" % x for x in sched), len(sched) - 1))
        lines.append("  " + AM.core_line())
        info = "\n".join(lines)
        print("[PLS] " + info.replace("\n", "\n[PLS] "))
        return (out, torch.tensor(sched, dtype=torch.float32), info)


NODE_CLASS_MAPPINGS = {"ULSMiniMaxSecondPassHold": ULSMiniMaxSecondPassHold}
NODE_DISPLAY_NAME_MAPPINGS = {"ULSMiniMaxSecondPassHold": "\u2b21 Polyhedron MiniMax Second Pass Hold"}
