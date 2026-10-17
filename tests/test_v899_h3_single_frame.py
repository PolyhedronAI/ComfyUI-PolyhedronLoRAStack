#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v899 -- MiniMax H3: length 1 builds ONE frame, not five.

WHAT v898 GOT WRONG. It reported the five-frame floor as the MODEL's, on the
strength of a single function -- Core's temporal_shape, which begins
`align_frame_count(max(5, length))`. Frank did not accept it ("Es geht hier um
T2I und das ist genau ein Bild"), and he was right. Two independent sources say
a single frame is a supported state:

  comfy/ldm/minimax/vae.py
      decode_output_shape:  if t == 1: frames = 1
      CausalConv3d.forward: a dedicated branch for x.shape[2] == 1, truncating
      the temporal taps "instead of convolving zero frames" -- nobody writes
      that for a shape that cannot occur.

  ai-toolkit .../minimax_h3/src/pipeline.py
      is_video = num_frames > 1
      else:  # true single-frame generation (image mode)
          num_frames = 1 ; t_lat = 1
      and the module header: image datasets train as single latent frames; the
      17n+5 snapping fires only for num_frames > 1.

The five belongs to `EmptyMiniMaxH3LatentAV` -- Core's AUDIO+VIDEO node, whose
own schema declares `length` min=5, step=17. It is a NODE's floor, and our node
inherited it by mirroring that node faithfully. For video it is right. For T2I
it is the wrong path.

WHAT IS PINNED

  I1  the shape helper: (1, 1, 2). One pixel frame, ONE latent frame, and the
      ordinary audio length for a single frame, round(1/24 * 40).
  I2  the Core mirror is UNTOUCHED. minimax_temporal_shape(1) must still be
      (5, 2, 8) -- a mirror that "improves" its source is a drift (v749), which
      is exactly why the image shape got its own named function instead.
  I3  the node builds it: video [B, 24, 1, H/16, W/16], audio [B, 32, 2, 2],
      driven through the real _minimax_av with a stubbed torch/comfy.
  I4  video is bit-for-bit unchanged: 81 -> 90 frames / latent_t 27, and 2..4
      still become 5. The image branch may not leak into the video path.
  I5  the audio half is PRESENT even for a still. The toolkit drops it
      (with_audio and is_video); Core's model class reads `audio_src = x[1]`
      unconditionally -- the line three field crashes landed on in August.
  I6  the four size outputs stay correct on the image branch -- they are read
      from the TENSOR, and a new branch is exactly where that can rot.
  I7  the console says image mode, and no longer calls five the model's floor.
  I8  the pill agrees with the backend: 1 -> "1 frame", 2..4 -> five with the
      way out named, 5+ -> the grid.

Script-style: exit 0 = pass.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import types

NAME = "v899"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "nodes"))

JS = os.path.join(ROOT, "web", "js", "ph_empty_latent.js")
PY = os.path.join(ROOT, "nodes", "ph_empty_latent.py")

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


def _lift(src, sig):
    i = src.index(sig)
    j = src.index("{", i)
    depth, k = 0, j
    while k < len(src):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
        k += 1
    raise ValueError("unbalanced: " + sig)


def _lift_const(src, name):
    m = re.search(r"const\s+" + re.escape(name) + r"\s*=\s*(\[.*?\]|\{.*?\n\}|[^;\n]+);",
                  src, re.S)
    return "const %s = %s;" % (name, m.group(1))


# ---------------------------------------------------------------------------
# Drive the real node against a stubbed torch/comfy. Nothing about the shape
# arithmetic is stubbed -- only the tensor library it hands the numbers to.
# ---------------------------------------------------------------------------

class _FakeTensor(object):
    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def size(self):
        return self.shape


class _Nested(object):
    is_nested = True

    def __init__(self, parts):
        self._parts = list(parts)

    def unbind(self):
        return list(self._parts)


def _install_stubs():
    torch = types.ModuleType("torch")
    torch.zeros = lambda shape, **kw: _FakeTensor(shape)
    torch.float32 = "float32"
    sys.modules.setdefault("torch", torch)

    comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.intermediate_device = lambda: "cpu"
    nt = types.ModuleType("comfy.nested_tensor")
    nt.NestedTensor = lambda parts: _Nested(parts)
    comfy.model_management = mm
    comfy.nested_tensor = nt
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm
    sys.modules["comfy.nested_tensor"] = nt
    return torch


def _run_branch(node_cls, length, seconds=0.0, width=1024, height=1024):
    """Call the real _minimax_av and report what it built."""
    out = node_cls()._minimax_av(
        positive=None, negative=None, width=width, height=height,
        length=length, batch_size=1, noise_type="zeros", noise_seed=0,
        noise_strength=1.0, duration_seconds=seconds)
    lat = out[2]["samples"]
    video, audio = lat.unbind()
    return {
        "video": tuple(video.shape),
        "audio": tuple(audio.shape),
        "sizes": tuple(out[3:7]),
    }


def main():
    import uls_latent_math as M
    py = _read(PY)
    js = _read(JS)

    # ---------------- I1 / I2: the arithmetic -----------------------------
    _need(hasattr(M, "minimax_image_shape"),
          "I1: uls_latent_math must carry a NAMED single-frame shape")
    if hasattr(M, "minimax_image_shape"):
        got = M.minimax_image_shape()
        _need(got == (1, 1, 2),
              "I1: image shape must be (frames 1, latent_t 1, audio_t 2), got %r"
              % (got,))
        want_audio = int(round(1 / float(M.MINIMAX_FPS)
                               * M.MINIMAX_AUDIO_LATENT_FPS))
        _need(got[2] == want_audio,
              "I1: the audio length must be derived (round(1/fps * 40) = %d), "
              "not typed in" % want_audio)

    _need(M.minimax_temporal_shape(1) == (5, 2, 8),
          "I2: the CORE MIRROR must stay a mirror -- temporal_shape(1) is "
          "(5,2,8) in Core, and improving it here would be a drift. The image "
          "shape belongs in its own function, got %r"
          % (M.minimax_temporal_shape(1),))
    _need(M.minimax_temporal_shape(81) == (90, 27, 150),
          "I2: the video arithmetic must be untouched")

    # ---------------- I3 / I4 / I5 / I6: the node, driven -----------------
    _install_stubs()
    import importlib
    mod = importlib.import_module("ph_empty_latent")
    importlib.reload(mod)
    cls = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and hasattr(obj, "_minimax_av"):
            cls = obj
            break
    _need(cls is not None, "I3: the node class was not found")
    if cls is None:
        print("\n[%s] %d/%d FAILED" % (NAME, len(_fails), _checks))
        return 1

    img = _run_branch(cls, length=1)
    _need(img["video"] == (1, 24, 1, 64, 64),
          "I3: length 1 must build ONE latent frame [1,24,1,H/16,W/16], got %r"
          % (img["video"],))
    _need(img["audio"] == (1, 32, 2, 2),
          "I5: the audio half must still be there for a still -- Core's model "
          "reads x[1] unconditionally; got %r" % (img["audio"],))
    _need(img["sizes"] == (64, 64, 1024, 1024),
          "I6: the four size outputs must stay right on the new branch "
          "(latent w/h then pixel w/h), got %r" % (img["sizes"],))

    # ...and again at a size where the TENSOR and the WIDGET disagree. At 1024
    # both give 64, so that case cannot pin where the numbers come from -- the
    # v887 lesson, found here by a mutation that swapped the source and stayed
    # green. 1010 snaps up to 1024 (round(1010/32)*32), so the tensor says 64
    # while the raw widget would say 1010//16 = 63.
    odd = _run_branch(cls, length=1, width=1010, height=1010)
    _need(odd["sizes"] == (64, 64, 1024, 1024),
          "I6: the sizes must be read from the TENSOR, not mirrored back from "
          "the widgets -- at 1010 those differ (64 vs 63), got %r"
          % (odd["sizes"],))
    _need(odd["video"] == (1, 24, 1, 64, 64),
          "I6: the canvas snap must still apply on the image branch, got %r"
          % (odd["video"],))

    vid = _run_branch(cls, length=81)
    _need(vid["video"] == (1, 24, 27, 64, 64),
          "I4: video must be bit-for-bit unchanged (81 -> latent_t 27), got %r"
          % (vid["video"],))
    _need(vid["audio"] == (1, 32, 2, 150),
          "I4: the video audio length must be unchanged, got %r"
          % (vid["audio"],))

    for n in (2, 3, 4):
        low = _run_branch(cls, length=n)
        _need(low["video"] == (1, 24, 2, 64, 64),
              "I4: length %d is still video and still floors to 5 frames "
              "(latent_t 2), got %r -- the image branch must not leak"
              % (n, low["video"],))

    five = _run_branch(cls, length=5)
    _need(five["video"] == (1, 24, 2, 64, 64),
          "I4: length 5 is the first real video point, got %r" % (five["video"],))

    # seconds still win, and they never reach the image branch by accident
    sec = _run_branch(cls, length=1, seconds=3.375)
    _need(sec["video"] == (1, 24, 27, 64, 64),
          "I4: duration_seconds must still override length, got %r"
          % (sec["video"],))

    # ---------------- I7: what the console says ---------------------------
    _need("image mode: one latent frame" in py,
          "I7: the run must say when it built a still")
    _need("Core's video node has a floor of 5" in py,
          "I7: the floor must be attributed to Core's video node, not the model")
    _need("image_mode" in py and "frames_in == 1" in py,
          "I7: the branch must hang on the REQUEST being one frame")

    # ---------------- I8: the pill agrees ---------------------------------
    pill_js = "\n".join([
        _lift_const(js, "LATENT_TYPES"),
        _lift_const(js, "TYPE_GEOM"),
        'const MINIMAX_TYPE = "MiniMax H3 AV (video+audio)";',
        "const MINIMAX_FPS = 24;",
        _lift(js, "function minimaxAlign"),
        _lift(js, "function minimaxFramesFromSeconds"),
        _lift(js, "function _findWidget"),
        _lift(js, "function _framePill"),
        r"""
function mk(len, secs) {
    return { widgets: [
        { name: "latent_type", value: MINIMAX_TYPE },
        { name: "length", value: len },
        { name: "duration_seconds", value: secs },
    ] };
}
const out = {};
for (const n of [1, 2, 4, 5, 6, 22, 81]) out["n" + n] = _framePill(mk(n, 0));
out.sec = _framePill(mk(1, 3.375));
process.stdout.write(JSON.stringify(out));
""",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(pill_js)
        path = fh.name
    try:
        p = subprocess.run(["node", path], capture_output=True, text=True,
                           timeout=90)
    finally:
        os.unlink(path)
    _need(p.returncode == 0 and p.stdout.strip(),
          "I8: the pill harness did not run: %s"
          % (p.stderr or p.stdout)[-300:])
    if p.returncode == 0 and p.stdout.strip():
        o = json.loads(p.stdout)
        _need(o["n1"]["frames"] == 1 and o["n1"]["imageMode"] is True,
              "I8: the pill must show ONE frame for length 1, got %r" % (o["n1"],))
        _need("image mode" in (o["n1"]["note"] or ""),
              "I8: it must say WHY it is one -- image mode, got %r"
              % (o["n1"]["note"],))
        _need(o["n1"]["text"].endswith("1 frame"),
              "I8: singular, not '1 frames', got %r" % (o["n1"]["text"],))
        for n in ("n2", "n4"):
            _need(o[n]["frames"] == 5 and "floor" in (o[n]["note"] or "")
                  and "1" in (o[n]["note"] or ""),
                  "I8: %s must report five AND name the way out, got %r"
                  % (n, o[n],))
        _need(o["n5"]["frames"] == 5 and not o["n5"]["note"],
              "I8: five is exact -- no note, got %r" % (o["n5"],))
        _need(o["n6"]["frames"] == 22 and "grid" in (o["n6"]["note"] or ""),
              "I8: above the floor the grid governs, got %r" % (o["n6"],))
        # 81 is NOT a grid point (81 % 17 = 13), so it must carry the note --
        # my first draft of this check asserted the opposite and was simply
        # wrong about the arithmetic. 22 IS one, and must be silent.
        _need(o["n81"]["frames"] == 90 and "grid" in (o["n81"]["note"] or ""),
              "I8: 81 is off-grid and must say so, got %r" % (o["n81"],))
        _need(o["n22"]["frames"] == 22 and not o["n22"]["note"],
              "I8: 22 IS a grid point -- an exact value gets no note, got %r"
              % (o["n22"],))
        _need(o["sec"]["frames"] == 90 and o["sec"]["imageMode"] is False,
              "I8: seconds must override length in the pill too, got %r"
              % (o["sec"],))
        # the backend and the pill must not disagree about ANY of these
        for n, val in (("n1", 1), ("n2", 5), ("n4", 5), ("n5", 5),
                       ("n6", 22), ("n22", 22), ("n81", 90)):
            built = _run_branch(cls, length=int(n[1:]))
            lt = built["video"][2]
            want_lt = 1 if val == 1 else M.minimax_video_latent_t(val)
            _need(lt == want_lt,
                  "I8: pill says %d frames for length %s, the node builds "
                  "latent_t %d (expected %d) -- readout and run must agree"
                  % (val, n[1:], lt, want_lt))

    if _fails:
        print("\n[%s] %d/%d FAILED" % (NAME, len(_fails), _checks))
        return 1
    print("[%s] OK -- length 1 builds one frame, video is untouched (%d checks)"
          % (NAME, _checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
