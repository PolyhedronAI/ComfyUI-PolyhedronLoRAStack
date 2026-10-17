#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v900 -- MiniMax H3: a single-frame decode gets TWO time tokens.

THE FAULT. Core's MiniMaxH3VideoVAE.decode() has a separate branch for one
latent frame that bypasses decode_temporal entirely, so the ViT decoder sees
exactly one time token. Position ids are length-normalised per axis --
2*(arange(0.5, size)/size) - 1 -- and for size 1 that collapses the time axis
onto exactly 0.0, a value no even token count produces (2 -> -0.5/+0.5;
4 -> -0.75..+0.75). The decoder gets a coordinate outside its training
distribution and lays a 16 px grid (one latent pixel) over the frame.

MEASURED, on Frank's own field renders, row-mean FFT power at period 16 px
against the surrounding band:

    single frame, 1024x1024, seven runs      x14 .. x34
      across er_sde/euler/dpmpp_2m, cfg 1/2/3/4, 8/20/30 steps,
      with and without LoRAs, on sdpa AND on sage3 fp4
    same prompt as 22 frames (cfg 4, sage3)  x4.3
    reference workflow                       x2.1

So it is not the sampler, not cfg, not the LoRAs, and not the attention
kernel -- every one of those was varied while the grid stayed put. Upstream
agrees: Comfy-Org/ComfyUI issue #15416 (2026-08-08, open) reports the same
16 px patch grid, notes the error is dtype-independent (fp16/bf16/fp32 within
0.1, hence algorithmic and not precision), and states padding to 2 tokens
works.

WHAT IS PINNED

  P1  Core's temporal arithmetic, run from its own constants: t=2 yields 5
      pixel frames (padding to 7 tokens internally), t=1 yields 1. This is
      the number the fix depends on, so it is computed here rather than
      quoted -- if Core's constants ever move, this fails loudly.
  P2  the position-id collapse is real: size 1 -> [0.0], and no even size
      produces 0.0. This is the REASON; if it stops holding, the fix's
      justification is gone and someone should know.
  P3  detection is asked of the OBJECT (class name), never a filename, and
      degrades to False rather than raising.
  P4  the trigger is exactly (auto) AND (H3 video VAE) AND (5-D latent with
      t == 1). Each of the three is pinned separately, in both directions.
  P5  padding repeats the token: content identical, count 2.
  P6  the lane drives it end to end -- a single-frame H3 decode returns ONE
      image out of the five, and the stock path is untouched with 'off'.
  P7  the console says what happened and names the escape hatch.
  P8  the canon is append-only: h3_still_fix sits AFTER mute_staging_logs, so
      saved workflows keep their widget positions (#577).

Script-style: exit 0 = pass.
"""

import importlib
import math
import os
import re
import sys
import types

NAME = "v900"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "nodes"))

PY = os.path.join(ROOT, "nodes", "ph_vae.py")

_fails = []
_checks = 0


def _need(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        _fails.append(msg)
        print("[%s] FAIL -- %s" % (NAME, msg))


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# P1 -- Core's temporal plan, computed from Core's constants.
# ---------------------------------------------------------------------------

def _core_frames(t_lat):
    clip_length, token_drop = 17, 3
    vae_ratio_t = math.prod((1, 2, 2, 1, 1, 1))
    frame_pre_padding = (-clip_length) % vae_ratio_t
    tokens_chunk = math.ceil(clip_length / vae_ratio_t)
    token_overlap = (-token_drop) % tokens_chunk

    if t_lat == 1:
        return 1

    pseudo = t_lat + token_drop
    pad = (-pseudo) % tokens_chunk
    pseudo += pad
    num_chunks = pseudo // tokens_chunk - int(token_drop > 0)
    if num_chunks < 1:
        pad += tokens_chunk
        num_chunks += 1
    z_len = t_lat + pad

    intra = clip_length % vae_ratio_t
    if intra == 0:
        pad_frames = pad * vae_ratio_t
    else:
        before = z_len - pad
        pad_frames = sum(
            (intra if (before + k) % tokens_chunk == 0 else vae_ratio_t)
            for k in range(pad))

    chunk_dec = tokens_chunk * vae_ratio_t
    split_count = int(token_drop > 0) + 1
    total, final_ov = 0, 0
    for i in range(num_chunks):
        s = i * tokens_chunk
        e = s + tokens_chunk + token_overlap
        clip_tok = max(0, min(e, z_len) - min(s, z_len))
        clip_fr = clip_tok * vae_ratio_t
        for j in range(split_count):
            fs = j * chunk_dec
            fe = min(fs + chunk_dec, clip_fr)
            cf = max(0, fe - fs - frame_pre_padding)
            if j == 0:
                total += cf
            else:
                final_ov = cf
    return total + final_ov - pad_frames


# ---------------------------------------------------------------------------
# Stubs: a fake torch that records shapes, and fake VAEs.
# ---------------------------------------------------------------------------

class _T(object):
    """Shape-carrying stand-in that also tracks WHICH frames it holds.

    Identity matters, not just count: [:1] and [-1:] produce the same SHAPE,
    so a shape-only check cannot tell "keep the first" from "keep the last".
    A mutation that swapped them stayed green until this list existed -- the
    v887 lesson again (two quantities that are coincidentally equal cannot
    pin their difference)."""

    def __init__(self, shape, tag="x", frames=None):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.tag = tag
        self.frames = (list(frames) if frames is not None
                       else list(range(self.shape[0])))

    def reshape(self, *a):
        # -1 must be RESOLVED. Leaving it literal made every frame count come
        # back as -1 and quietly passed a shape check that meant nothing.
        want = list(a)
        if -1 in want:
            known = 1
            for v in want:
                if v != -1:
                    known *= v
            total = 1
            for v in self.shape:
                total *= v
            want[want.index(-1)] = total // max(1, known)
        out = _T(want, self.tag)
        if want and want[0] == self.shape[0] * self.shape[1]:
            # [B,T,...] -> [B*T,...]: the frame identities carry over
            out.frames = list(range(want[0]))
        return out

    def __getitem__(self, k):
        if isinstance(k, slice):
            kept = self.frames[k]
            return _T((len(kept),) + self.shape[1:], self.tag, kept)
        raise TypeError(k)


def _install_torch(record):
    torch = types.ModuleType("torch")

    def cat(parts, dim=0):
        record.append(("cat", tuple(t.shape for t in parts), dim))
        shape = list(parts[0].shape)
        shape[dim] = sum(p.shape[dim] for p in parts)
        return _T(shape, parts[0].tag)

    torch.cat = cat
    torch.float32 = "float32"
    sys.modules["torch"] = torch

    # ph_vae imports comfy.model_management for the free-VRAM verdict; the
    # lane under test never consults it, so a stub is honest here.
    comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.get_free_memory = lambda *a, **k: float(1 << 34)
    mm.intermediate_device = lambda: "cpu"
    comfy.model_management = mm
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    return torch


class _FakeInner(object):
    pass


class MiniMaxH3VideoVAE(_FakeInner):
    """Named EXACTLY as Core names it -- that is the thing under test. My
    first draft called it _MiniMaxH3VideoVAE and every detection check went
    red, which is the guard working: the probe reads the real name."""
    pass


class _OtherVAE(_FakeInner):
    pass


class _VAE(object):
    """Decodes by Core's own frame plan, so the lane meets real counts."""

    def __init__(self, inner, calls):
        self.first_stage_model = inner
        self.calls = calls
        self.vae_dtype = "float16"

    def decode(self, z):
        t = z.shape[2]
        self.calls.append(("decode", z.shape))
        frames = _core_frames(t)
        return _T((z.shape[0], frames, z.shape[3] * 16, z.shape[4] * 16, 3),
                  "dec", None)

    def decode_tiled(self, z, **kw):
        return self.decode(z)

    def memory_used_decode(self, shape, dtype):
        return 0.0

    def spacial_compression_decode(self):
        return 16

    def temporal_compression_decode(self):
        return 4


def main():
    py = _read(PY)

    # ---------------- P1: the frame plan --------------------------------
    _need(_core_frames(1) == 1,
          "P1: one token is one frame in Core's own branch, got %d"
          % _core_frames(1))
    _need(_core_frames(2) == 5,
          "P1: TWO tokens must yield 5 pixel frames -- the whole fix rests "
          "on this number, got %d" % _core_frames(2))
    _need(_core_frames(7) == 22,
          "P1: 7 tokens are the 22-frame point Frank measured clean, got %d"
          % _core_frames(7))

    # ---------------- P2: the reason ------------------------------------
    def ids(size):
        return [2.0 * ((i + 0.5) / size) - 1.0 for i in range(size)]

    _need(ids(1) == [0.0],
          "P2: a single time token must land on exactly 0.0 -- that is the "
          "untrained coordinate the fix avoids, got %r" % (ids(1),))
    for size in (2, 4, 6, 8):
        _need(0.0 not in ids(size),
              "P2: no EVEN token count may produce 0.0, or the reasoning "
              "behind padding to two collapses; size %d gave %r"
              % (size, ids(size)))

    # ---------------- P3 / P4 / P5: detection and trigger ----------------
    record = []
    _install_torch(record)
    mod = importlib.import_module("ph_vae")
    importlib.reload(mod)

    h3 = _VAE(MiniMaxH3VideoVAE(), [])
    other = _VAE(_OtherVAE(), [])

    _need(mod._is_h3_video_vae(h3) is True,
          "P3: the H3 video VAE must be recognised by its class")
    _need(mod._is_h3_video_vae(other) is False,
          "P3: any other VAE must NOT be recognised")
    _need(mod._is_h3_video_vae(object()) is False,
          "P3: an object without first_stage_model must answer False, not "
          "raise -- a broken probe must degrade to 'fix off'")
    body = py.split("def _is_h3_video_vae")[1].split("\ndef ")[0]
    _need("first_stage_model" in body and "vae_name" not in body
          and ".safetensors" not in body,
          "P3: detection must read the OBJECT, never a filename")

    one = _T((1, 24, 1, 64, 64))
    two = _T((1, 24, 2, 64, 64))
    four_d = _T((1, 4, 64, 64))

    _need(mod._h3_still_needs_pad(h3, one, "auto") is True,
          "P4: auto + H3 + one time token is exactly the case that fires")
    _need(mod._h3_still_needs_pad(h3, one, "off") is False,
          "P4: 'off' must mean off -- that is the A/B lever")
    _need(mod._h3_still_needs_pad(other, one, "auto") is False,
          "P4: it must not touch a non-H3 VAE")
    _need(mod._h3_still_needs_pad(h3, two, "auto") is False,
          "P4: two tokens are already fine and must be left alone")
    _need(mod._h3_still_needs_pad(h3, _T((1, 24, 7, 64, 64)), "auto") is False,
          "P4: a real video latent must be left alone")
    _need(mod._h3_still_needs_pad(h3, four_d, "auto") is False,
          "P4: a 4-D image latent has no time axis and must be left alone")
    # ...and one where the shape ALONE would say yes. shape[-3] is 1 here, so
    # a probe that skips the ndim test fires on a latent that has no time
    # axis at all. A mutation doing exactly that survived until this case
    # existed.
    _need(mod._h3_still_needs_pad(h3, _T((1, 1, 64, 64)), "auto") is False,
          "P4: the probe must test ndim == 5, not just find a 1 somewhere "
          "in the shape -- a 4-D latent has no time axis to pad")

    padded = mod._h3_pad_to_two(one)
    _need(padded.shape == (1, 24, 2, 64, 64),
          "P5: padding must produce exactly two time tokens, got %r"
          % (padded.shape,))
    _need(record and record[-1][0] == "cat" and record[-1][2] == 2,
          "P5: the repeat must happen on the TIME axis (dim 2), got %r"
          % (record[-1:],))
    _need(record[-1][1] == ((1, 24, 1, 64, 64), (1, 24, 1, 64, 64)),
          "P5: the same token must be repeated, not zero-filled or shifted")

    # ---------------- P6: the lane, driven ------------------------------
    node = mod.ULSVAE()

    calls = []
    h3.calls = calls
    out, _dur = node._decode_lane(h3, one, "full", 512, 64, 64, 8,
                                  h3_still_fix="auto")
    _need(calls and calls[0][1][2] == 2,
          "P6: the VAE must receive TWO time tokens, got %r" % (calls[:1],))
    _need(out.shape[0] == 1,
          "P6: exactly ONE image may come back out of the five, got %r"
          % (out.shape,))
    _need(out.shape[1:] == (1024, 1024, 3),
          "P6: the image geometry must survive the fix, got %r" % (out.shape,))
    _need(out.frames == [4],
          "P6: it must be the LAST of the five frames, not the first -- "
          "Core's own choice in the single-frame branch and what the "
          "reference workflow does with ImageFromBatch(-1); got frame(s) %r"
          % (out.frames,))

    calls = []
    h3.calls = calls
    out_off, _ = node._decode_lane(h3, one, "full", 512, 64, 64, 8,
                                   h3_still_fix="off")
    _need(calls and calls[0][1][2] == 1,
          "P6: with 'off' the stock single-token path must run untouched, "
          "got %r" % (calls[:1],))
    _need(out_off.shape[0] == 1,
          "P6: the stock path still returns one image, got %r"
          % (out_off.shape,))

    calls = []
    h3.calls = calls
    vid, _ = node._decode_lane(h3, _T((1, 24, 7, 64, 64)), "full",
                              512, 64, 64, 8, h3_still_fix="auto")
    _need(calls and calls[0][1][2] == 7,
          "P6: a real video decode must pass through unchanged, got %r"
          % (calls[:1],))
    _need(vid.shape[0] == 22,
          "P6: all 22 frames must survive -- the fix must never trim a "
          "video, got %r" % (vid.shape,))

    calls = []
    other.calls = calls
    oth, _ = node._decode_lane(other, one, "full", 512, 64, 64, 8,
                               h3_still_fix="auto")
    _need(calls and calls[0][1][2] == 1,
          "P6: a non-H3 VAE must never be padded, got %r" % (calls[:1],))

    # ---------------- P7: what it says ----------------------------------
    # The console string is broken across source lines by concatenation, so
    # the check reads the pieces that actually sit contiguous in the file.
    # Searching for the assembled sentence found nothing and failed a working
    # build -- my error, not the code's.
    _need("decoding TWO " in py and "time tokens and keeping the last frame"
          in py,
          "P7: the run must say what it did")
    _need("#15416" in py,
          "P7: the upstream issue must be named, so the next reader can "
          "check whether it has been fixed")
    _need("h3_still_fix=off" in py,
          "P7: the console must name the escape hatch")
    _need("kept the" in py and "frame(s)" in py,
          "P7: the frame count actually decoded must be reported, not "
          "assumed -- it is the one number that proves which path ran")

    # ---------------- P8: canon is append-only --------------------------
    i = py.index("def INPUT_TYPES")
    canon = re.findall(r'"(\w+)":\s*\(', py[i:i + 7000])
    _need("h3_still_fix" in canon,
          "P8: the widget must exist in the canon")
    if "h3_still_fix" in canon and "mute_staging_logs" in canon:
        _need(canon.index("h3_still_fix") > canon.index("mute_staging_logs"),
              "P8: it must be APPENDED after mute_staging_logs -- inserting "
              "it earlier shifts every saved widgets_values (#577), got %r"
              % (canon,))
    _need(canon[:8] == ["vae", "mode", "tiling", "tile_size", "tile_overlap",
                        "temporal_size", "temporal_overlap",
                        "mute_staging_logs"],
          "P8: the existing canon prefix must be untouched, got %r"
          % (canon[:8],))

    if _fails:
        print("\n[%s] %d/%d FAILED" % (NAME, len(_fails), _checks))
        return 1
    print("[%s] OK -- single-frame H3 decodes get two tokens, everything "
          "else is untouched (%d checks)" % (NAME, _checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
