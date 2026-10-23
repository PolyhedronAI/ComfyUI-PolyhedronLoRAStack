# -*- coding: ascii -*-
"""Guard v1218 -- Polyhedron Blend (layers over a base, Photoshop modes, clips) and keypoints into the Save node.

Frank 23.10.: "eine Blend-Node ... Open-Pose und Depth und Canny mergen und gemeinsam als ein Video oder Bild in
Controlnet einspeisen" -- "die klassischen Photoshop-Sachen" -- "alle Ausgaenge so umbauen, dass wir das alles in
der Save-Node reinbringen koennen".

  B1 over (black = clear): black leaves the base, a white line replaces it, opacity mixes, a grey anti-aliased edge
     lies on top at its brightness
  B2 the Photoshop modes are Relight Blend's own maths (multiply, screen, overlay, difference == blend_images);
     layers apply bottom to top (normal then multiply != multiply then normal); opacity 0 = off; relight_gain is
     not offered, OVER is the first mode
  B3 clips: one picture lies over every frame; a shorter clip holds its last frame or loops; length 'base' keeps the
     base's count, 'longest' takes the longest input
  B4 the node: a layer of another size is resized to the base; no layer -> the base passes through (said); info
     names every layer; preview on -> {"ui" animated, "result"}; off -> the plain tuple
  B5 Save + keypoints: wired alone -> ONE <name>_keypoints.json ({"format": "openpose", "frames": [...]}, numpy
     values written as numbers), nothing else; beside a mask -> the mask stills AND the JSON; not keypoints -> refused
     by name; the input is optional and comes after mask (a socket, no widget moved)
  B6 registration and baselines: __init__ maps ULSBlend; NODE_IDS + WIDGET_ORDER carry it
  S  ASCII
Script-style: exit 0 = pass.
"""
import importlib
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1218_blend_save"
FAILS = []
TMP = Path(tempfile.mkdtemp(prefix="pls1218_"))


def _need(c, m):
    print(("  ok    " if c else "  FAIL  ") + m)
    if not c:
        FAILS.append(m)


fp = types.ModuleType("folder_paths")
fp.get_temp_directory = lambda: str(TMP / "temp")
fp.get_output_directory = lambda: str(TMP / "out")
fp.get_filename_list = lambda f: []
fp.get_folder_paths = lambda f: [str(TMP / f)]


def _gsip(prefix, base, w, h):
    full = os.path.join(base, os.path.dirname(prefix))
    return full, os.path.basename(prefix), 1, os.path.dirname(prefix), prefix


fp.get_save_image_path = _gsip
sys.modules["folder_paths"] = fp
pkg = types.ModuleType("plsn1218")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1218"] = pkg
BL = importlib.import_module("plsn1218.ph_blend")
RB = importlib.import_module("plsn1218.ph_relight_blend")
SV = importlib.import_module("plsn1218.ph_save")

# ---------------------------------------------------------------- B1
print("B1 over (black = clear)")
base = np.full((1, 2, 2, 3), 0.5, np.float32)
lay = np.zeros((1, 2, 2, 3), np.float32)
lay[0, 0, 0] = 1.0
lay[0, 1, 1] = [0.4, 0.4, 0.4]
o = BL.blend_stack(base, [(lay, BL.OVER, 1.0)])
_need(np.allclose(o[0, 0, 1], 0.5) and np.allclose(o[0, 0, 0], 1.0), "B1 black keeps the base, white replaces it")
_need(np.allclose(o[0, 1, 1], 0.5 * 0.6 + 0.4 * 0.4), "B1 a grey edge lies on at its brightness (%.3f)" % o[0, 1, 1, 0])
o2 = BL.blend_stack(base, [(lay, BL.OVER, 0.5)])
_need(np.allclose(o2[0, 0, 0], 0.75), "B1 opacity 0.5 mixes the white line half over the base")

# ---------------------------------------------------------------- B2
print("B2 Photoshop modes, order, opacity")
rng = np.random.default_rng(1218)
b = rng.random((1, 4, 4, 3)).astype(np.float32)
s = rng.random((1, 4, 4, 3)).astype(np.float32)
ok = all(np.allclose(BL.blend_stack(b, [(s, m, 1.0)])[0], RB.blend_images(b[0], s[0], m, opacity=1.0), atol=1e-6)
         for m in ("multiply", "screen", "overlay", "difference", "soft_light", "luminosity"))
_need(ok, "B2 multiply / screen / overlay / difference / soft light / luminosity = Relight Blend's maths")
s2 = rng.random((1, 4, 4, 3)).astype(np.float32)
ab = BL.blend_stack(b, [(s, "normal", 0.5), (s2, "multiply", 1.0)])
ba = BL.blend_stack(b, [(s2, "multiply", 1.0), (s, "normal", 0.5)])
want = RB.blend_images(RB.blend_images(b[0], s[0], "normal", opacity=0.5), s2[0], "multiply", opacity=1.0)
_need(np.allclose(ab[0], want, atol=1e-6) and not np.allclose(ab, ba), "B2 layers apply bottom to top (order matters)")
_need(np.allclose(BL.blend_stack(b, [(s, "multiply", 0.0)]), b), "B2 opacity 0 = the layer is off")
_need(BL.MODES[0] == BL.OVER and "relight_gain" not in BL.MODES and "multiply" in BL.MODES and "hard_mix" in BL.MODES,
      "B2 OVER first, the Photoshop set, no relight_gain (%d modes)" % len(BL.MODES))

# ---------------------------------------------------------------- B3
print("B3 clips")
clip5 = np.stack([np.full((2, 2, 3), i / 10.0, np.float32) for i in range(5)])
lay3 = np.stack([np.full((2, 2, 3), 0.5 + i / 10.0, np.float32) for i in range(3)])
one = np.full((1, 2, 2, 3), 0.9, np.float32)
r = BL.blend_stack(clip5, [(one, "lighten", 1.0)])
_need(r.shape[0] == 5 and np.allclose(r[:, 0, 0, 0], 0.9), "B3 one picture lies over every frame")
h = BL.blend_stack(clip5, [(lay3, "normal", 1.0)], at_end="hold last frame")
lp = BL.blend_stack(clip5, [(lay3, "normal", 1.0)], at_end="loop")
_need(np.allclose(h[3:, 0, 0, 0], 0.7) and np.allclose(lp[3, 0, 0, 0], 0.5) and np.allclose(lp[4, 0, 0, 0], 0.6),
      "B3 a shorter clip holds its last frame, or loops")
lg = BL.blend_stack(one, [(lay3, "multiply", 1.0)], length="longest")
_need(lg.shape[0] == 3 and BL.blend_stack(one, [(lay3, "multiply", 1.0)]).shape[0] == 1,
      "B3 length: 'base' keeps the base's count, 'longest' takes the longest input")

# ---------------------------------------------------------------- B4
print("B4 the node")
node = BL.ULSBlend()
args = (BL.OVER, 1.0, "multiply", 1.0, "screen", 0.5, "base", "hold last frame")
img = torch.rand(4, 32, 48, 3)
res = node.blend(img, *args, False, layer_1=torch.rand(4, 16, 24, 3), layer_3=torch.rand(1, 32, 48, 3))
_need(isinstance(res, tuple) and tuple(res[0].shape) == (4, 32, 48, 3), "B4 a smaller layer is resized to the base")
_need("layer 1: 4 frame(s) 24x16 -> over (black = clear) 1.00" in res[1] and "layer 3: 1 frame(s) 48x32 -> screen 0.50" in res[1]
      and "layer 2" not in res[1], "B4 the info names every wired layer, not the unwired one")
p = node.blend(img, *args, False)
_need(torch.allclose(p[0], img) and "no layer wired" in p[1], "B4 no layer: the base passes through, said")
pv = node.blend(img, *args, True, layer_1=img)
_need(isinstance(pv, dict) and pv["ui"].get("animated") == [True] and len(pv["result"]) == 2, "B4 preview on: an animated WebP")
it = BL.ULSBlend.INPUT_TYPES()
_need(list(it["optional"]) == ["layer_1", "layer_2", "layer_3"] and list(it["required"])[-1] == "preview",
      "B4 three optional layer sockets; preview last")

# ---------------------------------------------------------------- B5
print("B5 Save + keypoints")
kp = [{"canvas_width": 64, "canvas_height": 48, "people": [{"pose_keypoints_2d": [np.float32(1.5), 2.0, 0.9]}]}
      for _ in range(3)]
W = ("pose/PH_kp", "auto", "png", 100, list(SV.U.PRESET_NAMES)[0], 17, 24.0, False, False, 0, False, True, True)
r = SV.ULSSave().save(*W, keypoints=kp)
path = r["result"][0]
files = sorted(os.listdir(TMP / "out" / "pose"))
d = json.load(open(path))
_need(files == ["PH_kp_00001_keypoints.json"] and d["format"] == "openpose" and len(d["frames"]) == 3
      and d["frames"][0]["people"][0]["pose_keypoints_2d"][0] == 1.5, "B5 keypoints alone: one JSON (numpy -> numbers)")
r2 = SV.ULSSave().save(*(("both/PH_m",) + W[1:]), mask=torch.ones(2, 8, 8), keypoints=kp[:2])
files = sorted(os.listdir(TMP / "out" / "both"))
_need(any(f.endswith("_keypoints.json") for f in files) and sum(f.endswith(".png") for f in files) == 2,
      "B5 beside a mask: the mask stills AND the JSON (%s)" % files)
try:
    SV.ULSSave().save(*W, keypoints=["nope"])
    _need(False, "B5 not-keypoints accepted")
except ValueError as e:
    _need("keypoints" in str(e), "B5 something else on the socket is refused by name")
opt = list(SV.ULSSave.INPUT_TYPES()["optional"])
_need(opt[-2:] == ["mask", "keypoints"] and SV.ULSSave.INPUT_TYPES()["optional"]["keypoints"][0] == "POSE_KEYPOINT",
      "B5 keypoints: an optional socket after mask")

# ---------------------------------------------------------------- B6
print("B6 registration")
init = (ROOT / "__init__.py").read_text(encoding="utf-8")
_need("from .nodes.ph_blend import ULSBlend" in init and 'NODE_CLASS_MAPPINGS["ULSBlend"] = ULSBlend' in init,
      "B6 __init__ maps ULSBlend")
ids = list(ROOT.glob("NODE_IDS_baseline_v*.txt"))
wo = list(ROOT.glob("WIDGET_ORDER_baseline_v*.txt"))
_need(len(ids) == 1 and "\nULSBlend\n" in ids[0].read_text(), "B6 NODE_IDS carries ULSBlend")
_need(len(wo) == 1 and "\nULSBlend\tmode_1,opacity_1,mode_2,opacity_2,mode_3,opacity_3,length,at_the_end,preview\n" in wo[0].read_text(),
      "B6 WIDGET_ORDER: ULSBlend's nine widgets")

_sv = (ROOT / "nodes" / "ph_save.py").read_text(encoding="utf-8")
_blk = _sv[_sv.find("# -- keypoints (v1218)"):_sv.find("# -- image ---")] + _sv[_sv.find('# v1218 (Frank 23.10.'):_sv.find('"hidden": {"prompt"')]
_need(len(_blk) > 500 and all(ord(c) < 128 for c in _blk), "S the v1218 lines of ph_save.py are ASCII (the file had non-ASCII before)")

# ---------------------------------------------------------------- S
print("S ASCII")
for f in ("nodes/ph_blend.py", "tests/test_v1218_blend_save.py"):
    try:
        (ROOT / f).read_bytes().decode("ascii")
        _need(True, "S %s" % f)
    except UnicodeDecodeError:
        _need(False, "S %s is not ASCII" % f)

shutil.rmtree(TMP, ignore_errors=True)
print()
print("%s: %s (%d fail)" % (NAME, "PASS" if not FAILS else "FAIL", len(FAILS)))
sys.exit(1 if FAILS else 0)
