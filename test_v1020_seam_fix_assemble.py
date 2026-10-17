# -*- coding: ascii -*-
"""Guard v1020 -- Cine C4: Seam Fix and Clip Assemble.

  F1  level_gains(): full correction at the join, identity at frame n, the
      target is the TREND of clip n (not its last frame), the start is the
      straightened opening of clip n+1 (one noisy first frame does not
      decide). A clip that already continues on trend gets gains of 1.
  F2  Seam Fix on a synthetic +12 grey step: the seam drops into 'seamless',
      the frames after fix_frames are untouched (identity -- nothing
      accumulates down a chain), black stays black (a gain, not an offset).
  F3  trim_head drops the overlap from frames AND audio by the same time; the
      audio is faded in (first sample 0, then rising); trimming everything is
      refused by name.
  F4  Clip Assemble joins three REAL clips with sound (PyAV-written) by packet
      copy: frame count = sum, duration = sum, audio present, making order
      (mtime) respected; the output lands in <folder>/assembled and never
      joins the reel; last-before-first and a foreign `last` are refused.
  F5  A clip of another size breaks the join by name.

Script-style: exit 0 = pass.
"""
import importlib
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _fail(m):
    FAILS.append(m)
    print("  FAIL  " + m)


def _ok(m):
    print("  ok    " + m)


try:
    import av
except Exception:
    print("  note  PyAV missing -- SKIPPED")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="v1020_"))
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: str(TMP)
fp.get_input_directory = lambda: str(TMP / "in")
fp.get_temp_directory = lambda: str(TMP / "temp")
comfy = types.ModuleType("comfy")
nt = types.ModuleType("comfy.nested_tensor")
nt.NestedTensor = object
comfy.nested_tensor = nt
sys.modules.update({"folder_paths": fp, "comfy": comfy, "comfy.nested_tensor": nt})
pkg = types.ModuleType("plsn1020")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1020"] = pkg
CH = importlib.import_module("plsn1020.ph_cine_chain")
C = sys.modules["plsn1020.cine_clip"]


def flat(level, n=16, h=16, w=16):
    return np.full((n, h, w, 3), level / 255.0, dtype=np.float32)


# ========================================================================= F1
prev = flat(100)
g = C.level_gains(prev, flat(112), n=12)
if g.shape == (12, 3) and abs(g[0, 0] - 100 / 112.0) < 1e-6 and abs(g[-1, 0] - (1 + (100 / 112.0 - 1) / 12)) < 1e-6:
    _ok("F1 full correction at the join (100/112), almost identity at the last corrected frame")
else:
    _fail("F1 gains %r %r" % (g[0], g[-1]))
ramp = np.stack([np.full((8, 8, 3), (100 + i) / 255.0) for i in range(10)])
on = np.stack([np.full((8, 8, 3), (110 + i) / 255.0) for i in range(12)])
if np.abs(C.level_gains(ramp, on) - 1.0).max() < 1e-6:
    _ok("F1 a clip that continues ON the trend of clip n gets gains of exactly 1")
else:
    _fail("F1 on-trend gains %r" % C.level_gains(ramp, on)[0])
noisy = flat(112)
noisy[0] = 200 / 255.0
g2 = C.level_gains(prev, noisy, n=12)
if abs(g2[0, 0] - 100 / 112.0) < 0.1:
    _ok("F1 one outlier first frame does not decide the correction (straightened opening)")
else:
    _fail("F1 outlier drove the gain to %r" % g2[0, 0])

# ========================================================================= F2
nxt = flat(112, n=30)
nxt[:, :4, :4, :] = 0.0                     # a black corner -- in BOTH clips (same content, other level)
nxt[:, 4:8, :4, :] = 40 / 255.0             # and a dark patch: a gain scales it, an offset would crush it
prev2 = prev.copy()
prev2[:, :4, :4, :] = 0.0
prev2[:, 4:8, :4, :] = 40 / 255.0
res = CH.ULSSeamFix().fix(torch.from_numpy(prev2), torch.from_numpy(nxt), 0, True, 12, 10.0)
frames, _aud, seam_after, info = res["result"]
fr = frames.numpy()
if seam_after <= C.SEAM_HOLDS and "before: seam" in info and "after:  seam" in info:
    _ok("F2 a +12 step is brought into 'seamless' (after %.2f), info reports before/after" % seam_after)
else:
    _fail("F2 seam after %.2f / %s" % (seam_after, info))
if np.array_equal(fr[12:], nxt[12:]):
    _ok("F2 frames from fix_frames on are untouched (nothing accumulates down a chain)")
else:
    _fail("F2 frames after the correction window changed")
dark = float(fr[0, 4:8, :4].mean()) * 255.0
if float(fr[0, :4, :4].max()) == 0.0 and abs(dark - 40 * fr[0, 8:, 8:].mean() / nxt[0, 8:, 8:].mean()) < 0.5:
    _ok("F2 black stays black, a dark patch scales with the rest (a gain, not an offset: %.2f)" % dark)
else:
    _fail("F2 black %r / dark patch %.2f -- not a gain" % (fr[0, :4, :4].max(), dark))

# ========================================================================= F3
sr = 32000
aud = {"waveform": torch.ones(1, 2, sr * 2), "sample_rate": sr}
r3 = CH.ULSSeamFix().fix(torch.from_numpy(prev), torch.from_numpy(flat(100, n=48)), 24, False, 12, 10.0, audio=aud)
f3, a3, _, i3 = r3["result"]
w = a3["waveform"].numpy()
if f3.shape[0] == 24 and w.shape[-1] == sr and w[0, 0, 0] == 0.0 and w[0, 0, 100] > 0 and w[0, 0, 400] == 1.0:
    _ok("F3 24 frames and 1.0 s of audio trimmed together; audio fades in over 10 ms")
else:
    _fail("F3 frames %d audio %d first %r" % (f3.shape[0], w.shape[-1], w[0, 0, :3]))
try:
    CH.ULSSeamFix().fix(torch.from_numpy(prev), torch.from_numpy(flat(100, n=10)), 10, True, 12, 0.0)
    _fail("F3 trimming everything was accepted")
except Exception as e:                      # a crash deeper down is a FAIL, not a pass
    _ok("F3 trimming the whole clip stops by name") if isinstance(e, ValueError) and "leaves nothing" in str(e) else _fail("F3 %r" % e)


# ========================================================================= F4
def clip(path, n, level, mt, size=(160, 96)):
    W, H = size
    with av.open(str(path), "w") as out:
        vs = out.add_stream("libx264", rate=24)
        vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
        a = out.add_stream("aac", rate=32000)
        a.layout = "stereo"
        for i in range(n):
            for p in vs.encode(av.VideoFrame.from_ndarray(np.full((H, W, 3), level, np.uint8), format="rgb24")):
                out.mux(p)
        for p in vs.encode():
            out.mux(p)
        wav = np.zeros((2, int(32000 * n / 24)), np.float32)
        for s0 in range(0, wav.shape[1], 1024):
            f = av.AudioFrame.from_ndarray(np.ascontiguousarray(wav[:, s0:s0 + 1024]), format="fltp", layout="stereo")
            f.sample_rate = 32000
            for p in a.encode(f):
                out.mux(p)
        for p in a.encode():
            out.mux(p)
    os.utime(str(path), (mt, mt))


chain = TMP / "chain"
chain.mkdir()
clip(chain / "c_b.mp4", 48, 60, 1e9)        # made FIRST although its name sorts second
clip(chain / "c_a.mp4", 24, 120, 1e9 + 10)
clip(chain / "c_c.mp4", 36, 180, 1e9 + 20)
r4 = CH.ULSClipAssemble().join(str(chain / "c_b.mp4"), "", "reel")
out = r4["result"][0]
with av.open(out) as c:
    levels = [float(f.to_ndarray(format="rgb24").mean()) for f in c.decode(video=0)]
with av.open(out) as c:
    has_audio = len(c.streams.audio) == 1
    dur = float(c.duration) / 1e6
if len(levels) == 108 and abs(dur - 4.5) < 0.1 and has_audio:
    _ok("F4 three real clips with sound joined: 108 frames, %.2f s, audio stream kept" % dur)
else:
    _fail("F4 frames %d dur %.2f audio %s" % (len(levels), dur, has_audio))
if abs(levels[0] - 60) < 6 and abs(levels[48] - 120) < 6 and abs(levels[72] - 180) < 6:
    _ok("F4 making order (mtime) respected: c_b, c_a, c_c")
else:
    _fail("F4 order levels %r %r %r" % (levels[0], levels[48], levels[72]))
if os.path.dirname(out) == str(chain / "assembled") and all(it["name"] != os.path.basename(out) for it in C.reel_items(str(chain / "c_b.mp4"), [])):
    _ok("F4 the joined video lands in assembled/ and never joins the chain's reel")
else:
    _fail("F4 out %s" % out)
r4b = CH.ULSClipAssemble().join(str(chain / "c_b.mp4"), str(chain / "c_a.mp4"), "part")
with av.open(r4b["result"][0]) as c:
    nb = sum(1 for _ in c.decode(video=0))
if nb == 72:
    _ok("F4 first..last range: c_b + c_a = 72 frames")
else:
    _fail("F4 range frames %d" % nb)
for first, last, needle in ((chain / "c_a.mp4", chain / "c_b.mp4", "BEFORE first"),
                            (chain / "c_b.mp4", TMP / "x.mp4", "not a video in the folder")):
    try:
        CH.ULSClipAssemble().join(str(first), str(last), "x")
        _fail("F4 accepted %s..%s" % (first.name, last.name))
    except ValueError as e:
        _ok("F4 refused by name: %s" % needle) if needle in str(e) else _fail("F4 %s" % e)

# ========================================================================= F5
clip(chain / "c_d.mp4", 24, 90, 1e9 + 30, size=(128, 96))
try:
    CH.ULSClipAssemble().join(str(chain / "c_b.mp4"), "", "bad")
    _fail("F5 a clip of another size was joined")
except ValueError as e:
    _ok("F5 a clip of another size breaks the join by name") if "only clips of one chain join" in str(e) else _fail("F5 %s" % e)

shutil.rmtree(TMP, ignore_errors=True)
print()
if FAILS:
    print("test_v1020_seam_fix_assemble: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1020_seam_fix_assemble: PASS (F1-F5)")
