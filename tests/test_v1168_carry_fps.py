# -*- coding: ascii -*-
"""Guard v1168 -- the carry (and a @clip anchor) reads the clip at ITS frame rate.

Frank 18.10., the neon film (five scenes, carry, replace): "Wir haben hier Uebergabe-Fehler drin". Measured on his
export: every scene began on the MIDDLE frame of the scene before (film frame 70 of 141), at all four seams alike.
The newest take of a scene is the interpolated final -- 48 fps, 281 frames. The timeline counts film frames at
24 fps and asked that clip for frame 140 by NUMBER: its middle. Now the film frame goes through the clip's rate.

  S1 source_index: 24 fps clip -> the same number; 48 fps -> doubled; never past the clip's last frame; an
     unreadable clip -> 24 fps assumed
  S2 the carry on a real 48 fps clip (built with ffmpeg): the frame handed to the run is the clip's LAST picture,
     not its middle; the info says the rate
  S3 a @clip:K:last anchor goes the same way
  S  ASCII
"""
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1168_carry_fps"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


D = tempfile.mkdtemp(prefix="v1168cf_")
import importlib  # noqa: E402
import torch  # noqa: E402,F401
comfy = types.ModuleType("comfy")
nt = types.ModuleType("comfy.nested_tensor")
nt.NestedTensor = object
mm = types.ModuleType("comfy.model_management")
mm.unload_all_models = lambda: None
mm.soft_empty_cache = lambda force=False: None
capi = types.ModuleType("comfy_api")
ii = types.ModuleType("comfy_api.input_impl")
ii.VideoFromComponents = object
ut = types.ModuleType("comfy_api.util")
ut.VideoComponents = object
fp = types.ModuleType("folder_paths")
fp.get_temp_directory = lambda: D
fp.get_output_directory = lambda: os.path.join(D, "output")
fp.get_input_directory = lambda: os.path.join(D, "input")
fp.models_dir = D
fp.folder_names_and_paths = {}
fp.add_model_folder_path = lambda n, p: None
fp.get_filename_list = lambda n: []
sys.modules.update({"comfy": comfy, "comfy.nested_tensor": nt, "comfy.model_management": mm, "comfy_api": capi,
                    "comfy_api.input_impl": ii, "comfy_api.util": ut, "folder_paths": fp})
pkg = types.ModuleType("plsn1168")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1168"] = pkg
P = importlib.import_module("plsn1168.ph_cine_timeline")

ff = shutil.which("ffmpeg")
_need(bool(ff), "ffmpeg is here")
# a 48 fps clip of 281 frames whose frames are numbered by their brightness: frame n = gray n/281
os.makedirs(os.path.join(D, "output"), exist_ok=True)
clip = os.path.join(D, "output", "scene_1_48fps.mp4")
subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=48:d=5.854", "-vf",
                "geq=lum='255*N/280':cb=128:cr=128", "-frames:v", "281", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", clip],
               capture_output=True)
_need(os.path.isfile(clip), "a 48 fps test clip (281 frames)")

# ---- S1
i, fps, n = P.source_index(clip, 140)
_need(abs(fps - 48) < 0.01 and n == 281 and i == 280, "S1 film frame 140 on a 48 fps clip of 281 frames -> its frame 280 (the last), got %s" % ((i, fps, n),))
i2, _f, _n = P.source_index(clip, 70)
_need(i2 == 140, "S1 film frame 70 -> clip frame 140")
i3, _f, _n = P.source_index(clip, 999)
_need(i3 == 280, "S1 never past the clip's last frame")
i4, f4, n4 = P.source_index(os.path.join(D, "nope.mp4"), 140)
_need(i4 == 140 and f4 == 24.0 and n4 is None, "S1 an unreadable clip: 24 fps assumed, the number stays")

# ---- S2 / S3: the frame handed over
img = P.frame_index(clip, i)
lum = float(img[0, ..., 0].mean())
_need(lum > 0.9, "S2 the carry frame is the clip's LAST picture (brightness %.2f, the middle would be ~0.5)" % lum)
mid = P.frame_index(clip, 140)
_need(abs(float(mid[0, ..., 0].mean()) - 0.5) < 0.1, "S2 (for the record) frame 140 by number IS the middle (%.2f)" % float(mid[0, ..., 0].mean()))
src = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
_need('sidx, sfps, sn = source_index(r["carry"], r["carry_index"])' in src and 'carry_img = frame_index(r["carry"], sidx)' in src,
      "S2 build() carries through source_index")
_need('sidx, sfps, sn = source_index(path, idx)' in src and 'img = frame_index(path, sidx)' in src, "S3 the @clip anchors too")
_need('runs at %g fps' in src, "S2 the info says the clip's rate")
shutil.rmtree(D, ignore_errors=True)
_need(all(c < 128 for c in Path(__file__).read_bytes()), "S  ASCII")
print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
