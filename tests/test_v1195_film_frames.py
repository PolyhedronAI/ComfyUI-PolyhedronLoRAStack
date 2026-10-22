# -*- coding: ascii -*-
"""Guard v1195 -- film_frames: how many film frames (24 fps) a clip at another rate becomes -- the number ffmpeg's
fps filter emits (round HALF UP), one law in cine_edit used by clip_info.

(Frank 20.10., the neon film: frame 139/140 doubled at every seam -- the interpolated 48 fps clip of 281 frames
counted as 140 film frames (Python's round() takes .5 to EVEN) while ffmpeg's resample yields 141; the export
held a picture. "Das sollte allgemeingueltig funktionieren ... nicht nur hier.")

  F1 the law on paper: 281@48 -> 141, 283@48 -> 142, 61@60 -> 24, 49@12 -> 98, 73@16 -> 110, 250@25 -> 240,
     100@23.976 -> 100, 24 fps -> n unchanged, 0 / bad -> 0; it is floor(n*24/fps + 0.5), never banker's rounding
  F2 MEASURED against ffmpeg (fps=24 on synthetic clips at 48 / 60 / 16 / 25 / 12 fps, odd and even counts):
     the law equals the frame count ffmpeg writes, in every case
  F3 clip_info on a 281-frame 48 fps clip reports frames 141 (src_frames 281); the film plan for a 141-frame scene
     holds NO picture (hold 0, no "held" warning) -- before: 140 + 1 held
  F4 source_index still maps film frame 140 of that clip to its own frame 280 (the last), 0 -> 0
  Z  ASCII
Script-style: exit 0 = pass.
"""
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


import torch  # noqa: E402
sys.path.insert(0, str(ROOT / "nodes"))
E = importlib.import_module("cine_edit")

# ---- F1
cases = ((281, 48, 141), (283, 48, 142), (282, 48, 141), (61, 60, 24), (60, 60, 24), (49, 12, 98), (73, 16, 110),
         (250, 25, 240), (251, 25, 241), (300, 30, 240), (100, 24000 / 1001.0, 100), (141, 24, 141), (1, 48, 1), (2, 48, 1))
bad = [(n, f, E.film_frames(n, f), w) for n, f, w in cases if E.film_frames(n, f) != w]
_need(not bad, "F1 the law on paper (%s)" % (bad or "all %d cases" % len(cases)))
_need(E.film_frames(0, 48) == 0 and E.film_frames(10, 0) == 10 and E.film_frames(-3, 48) == 0, "F1 0 / negative -> 0; fps 0 = unknown -> the clip's own count")
_need(round(140.5) == 140 and E.film_frames(281, 48) == 141, "F1 Python rounds .5 to even; the law rounds half UP (the whole bug)")

# ---- F2
exe = shutil.which("ffmpeg")
probe = shutil.which("ffprobe")
TMP = Path(tempfile.mkdtemp(prefix="pls1195_"))
if exe and probe:
    bad = []
    for n, fps in ((281, 48), (61, 60), (73, 16), (251, 25), (49, 12), (7, 48), (127, 50)):
        src = TMP / ("c_%d_%s.mp4" % (n, fps))
        dst = TMP / ("o_%d_%s.mp4" % (n, fps))
        subprocess.run([exe, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=48x48:rate=%s" % fps, "-frames:v", str(n),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(src)], check=True)
        subprocess.run([exe, "-v", "error", "-y", "-i", str(src), "-vf", "fps=%d" % E.FPS, "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)], check=True)
        real = int(subprocess.run([probe, "-v", "error", "-select_streams", "v", "-count_frames", "-show_entries", "stream=nb_read_frames",
                                   "-of", "csv=p=0", str(dst)], capture_output=True, text=True).stdout.strip())
        if real != E.film_frames(n, fps):
            bad.append((n, fps, real, E.film_frames(n, fps)))
    _need(not bad, "F2 MEASURED: the law equals ffmpeg's fps=24 frame count on 7 clips (%s)" % (bad or "0 mismatches"))
else:
    _need(False, "F2 ffmpeg + ffprobe are needed to measure the law")

# ---- F3
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: str(TMP)
fp.get_temp_directory = lambda: str(TMP)
fp.get_input_directory = lambda: str(TMP)
fp.get_user_directory = lambda: str(TMP)
fp.models_dir = str(TMP / "models")
fp.add_model_folder_path = lambda *a, **k: None
fp.get_folder_paths = lambda f: [os.path.join(fp.models_dir, f)]
fp.get_filename_list = lambda *a, **k: []
sys.modules.setdefault("folder_paths", fp)
nh = types.ModuleType("node_helpers")
nh.conditioning_set_values = lambda cond, values: [[c[0], dict(c[1], **values)] for c in cond]
sys.modules.setdefault("node_helpers", nh)
cu = types.ModuleType("comfy.utils")
cu.common_upscale = lambda s_, w, h, m, c: torch.nn.functional.interpolate(s_, size=(h, w), mode="bilinear")
cm = types.ModuleType("comfy")
cm.utils = cu
sys.modules.setdefault("comfy", cm)
sys.modules.setdefault("comfy.utils", cu)
PKG = "plsff"
pkg = types.ModuleType(PKG)
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules[PKG] = pkg
try:
    PT = importlib.import_module(PKG + ".ph_cine_timeline")
    E = importlib.import_module(PKG + ".cine_edit")     # the package's own copy (one law, the same code)
    have_pt = True
except Exception as exc:
    have_pt = False
    print("  (ph_cine_timeline not importable here: %s)" % exc)
if have_pt and exe:
    clip = TMP / "scene1_48fps.mp4"
    subprocess.run([exe, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=64x36:rate=48", "-frames:v", "281",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
    ci = PT.clip_info(str(clip))
    _need(ci["frames"] == 141 and ci.get("src_frames") == 281 and abs(ci["fps"] - 48) < 0.01, "F3 clip_info: 281 @ 48 fps = 141 film frames (%s)" % {k: ci[k] for k in ("frames", "src_frames", "fps")})
    P = E.plan([], 141 / 24.0, [{"frames": ci["frames"]}], [], E.TRANSITIONS if hasattr(E, "TRANSITIONS") else {}, whole=False)
    seg = P["segments"][0]
    _need(seg["hold"] == 0 and seg["src_in"] == 0 and seg["src_out"] == 141 and not any("held" in w for w in P["warnings"]),
          "F3 the film plan for a 141-frame scene holds NO picture (src 0..141, warnings %s)" % P["warnings"])
    idx, fps, n = PT.source_index(str(clip), 140)
    _need(idx == 280 and n == 281 and PT.source_index(str(clip), 0)[0] == 0, "F4 source_index: film frame 140 -> the clip's frame 280 (the last), 0 -> 0 (%s)" % idx)
else:
    _need(False, "F3/F4 need ph_cine_timeline + ffmpeg")
shutil.rmtree(TMP, ignore_errors=True)

for p in ("nodes/cine_edit.py", "tests/test_v1195_film_frames.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "Z  ASCII %s" % p)
print("%s: %d check(s) failed" % ("test_v1195_film_frames", len(FAILS)) if FAILS else "test_v1195_film_frames: all checks passed")
sys.exit(1 if FAILS else 0)
