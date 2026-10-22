# -*- coding: ascii -*-
"""Guard v1185 -- the motion head's length is the timeline's choice (head_frames 5 / 22 / 39, default 22).

  H1 motion_head: absent / None / junk / a value off the grid -> 22; 5, 22, 39 -> themselves
  H2 validate: head_frames off the grid is an error that names the grid; 5 / 22 / 39 pass
  H3 with_motion_head(head=5): the scene grows by 5 asked frames, the next shot moves back by 5/24 s, the grid cost is
     said in frames: 141 + 5 -> 158 (17 more), 141 + 22 -> 175 (34), 141 + 39 -> 192 (51)
  H4 run(): head_frames 5 -> r["motion"] 5, the info names head 5 and the grid's add; head_frames 39 -> a line that
     MiniMax Continue's context_frames must be >= 39; absent -> 22 and no such line (v1176 behaviour)
  H5 Continue follows a shorter tail: snap_guide(min(22, 5)) = 5 -- the Timeline's 5-frame tail needs no widget change
  H6 the editor offers HEAD_FRAMES 5/22/39, writes head_frames, disables it unless carry + motion
  Z  ASCII
Script-style: exit 0 = pass.
"""
import importlib
import json
import sys
import tempfile
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


fp = types.ModuleType("folder_paths")
fp.get_temp_directory = lambda: tempfile.gettempdir()
fp.get_output_directory = lambda: tempfile.gettempdir()
fp.get_input_directory = lambda: tempfile.gettempdir()
sys.modules.setdefault("folder_paths", fp)
nh = types.ModuleType("node_helpers")
nh.conditioning_set_values = lambda cond, values: [[c[0], dict(c[1], **values)] for c in cond]
sys.modules.setdefault("node_helpers", nh)
cu = types.ModuleType("comfy.utils")
cu.common_upscale = lambda s, w, h, m, c: torch.nn.functional.interpolate(s, size=(h, w), mode="bilinear")
cm = types.ModuleType("comfy")
cm.utils = cu
sys.modules.setdefault("comfy", cm)
sys.modules.setdefault("comfy.utils", cu)
PKG = "plshead"
pkg = types.ModuleType(PKG)
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules[PKG] = pkg
CO = importlib.import_module(PKG + ".ph_minimax_continue")
T = importlib.import_module(PKG + ".cine_timeline")
TL = importlib.import_module(PKG + ".ph_cine_timeline")
torch.set_num_threads(1)

# ---- H1
for val, want in ((None, 22), ("x", 22), (7, 22), (0, 22), (5, 5), (22, 22), (39, 39), ("39", 39)):
    _need(T.motion_head({"head_frames": val}) == want, "H1 head_frames %r -> %d" % (val, want))
_need(T.motion_head({}) == 22 and T.DEFAULT["head_frames"] == 22, "H1 absent -> 22; DEFAULT carries 22 (so load keeps the key)")
_need(T.HEAD_FRAMES == (5, 22, 39), "H1 the grid: 5 / 22 / 39")

# ---- H2 / H3 / H4
SC = {"at": 0.0, "cut": T.CUTS[0], "text": "A dancer spins under neon light."}
# scene 2 spans 5.875 s = 141 frames (Frank's Neon scenes); a third scene closes it, the whole sits on the grid (29.25 s)
base = dict(T.DEFAULT, seconds=29.25, render="scene", active=2, carry_mode="motion", project="v1185probe",
            soundscape="club bass",
            shots=[SC, dict(SC, at=5.875, scene=True, carry=True, text="She keeps spinning."),
                   dict(SC, at=8.875, text="The camera cuts to her face."),
                   dict(SC, at=11.75, scene=True, text="A new place.")])
bad = T.validate(T.load(json.dumps(dict(base, head_frames=17))))
_need(any("head_frames" in e and "5/22/39" in e for e in bad), "H2 head_frames 17 is refused and the grid named")
for h in (5, 22, 39):
    _need(not T.validate(T.load(json.dumps(dict(base, head_frames=h)))), "H2 head_frames %d passes" % h)

plain, k, n = T.resolve(T.load(json.dumps(base)))
_need(T.frames_for(plain["seconds"]) == 141, "H3 the plain scene 2: 5.875 s = 141 frames")
for h, grid in ((5, 158), (22, 175), (39, 192)):
    hm, _notes = T.with_motion_head(plain, head=h)
    _need(T.asked_frames(hm["seconds"]) == 141 + h and T.frames_for(hm["seconds"]) == grid
          and abs(hm["shots"][1]["at"] - (3.0 + h / 24.0)) < 1e-9 and hm["head"] == h,
          "H3 head %d: asked 141+%d, grid %d (+%d), the next shot back by %d/24 s" % (h, h, grid, grid - 141, h))

src = {"image_1": None}
for h, motion, grid in ((5, 5, 158), (None, 22, 175), (39, 39, 192)):
    mm = dict(base)
    if h is not None:
        mm["head_frames"] = h
    r, errs = TL.run(T.load(json.dumps(mm)), src, prev="/x/scene_1.mp4")
    lines = " | ".join(r["lines"]) if r else ""
    _need(not errs and r["motion"] == motion and r["frames"] == grid,
          "H4 head_frames %s -> motion %d, %d frames" % (h, motion, grid))
    _need(("head %d, the grid adds %d: 141 -> %d frames" % (motion, grid - 141 - motion, grid)) in lines,
          "H4 the info says the head and the grid's add (%d + %d)" % (motion, grid - 141 - motion))
    cap = any("context_frames" in x and "set it to %d or more" % motion in x for x in r["lines"])
    _need(cap == (motion > 22), "H4 the Continue cap line %s for head %d" % ("appears" if motion > 22 else "stays away", motion))

# ---- H5
_need(CO.snap_guide(min(22, 5)) == 5 and CO.snap_guide(min(22, 39)) == 22 and CO.snap_guide(min(39, 39)) == 39,
      "H5 Continue takes min(context_frames, tail) on the 17k+5 grid: 5-frame tail -> 5, 39 needs context_frames 39")

# ---- H6
js = (ROOT / "web/js/ph_cine_timeline.js").read_text(encoding="utf-8")
_need('export const HEAD_FRAMES = [[5, ' in js and '[22, ' in js and '[39, ' in js, "H6 the editor offers 5 / 22 / 39")
_need('mm.head_frames = Number(head.value)' in js and 'head.dataset.pls = "headframes"' in js, "H6 the editor writes head_frames")
_need('head.disabled = !cb.checked || sel.value !== "motion"' in js and "return [lab, sel, head];" in js,
      "H6 the choice is live only for a carrying scene in motion mode, and sits next to the carry mode")

# ---- Z
for p in ("nodes/cine_timeline.py", "nodes/ph_cine_timeline.py", "web/js/ph_cine_timeline.js", "tests/test_v1185_head_frames.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "Z  ASCII %s" % p)

print("%s: %d check(s) failed" % ("test_v1185_head_frames", len(FAILS)) if FAILS
      else "test_v1185_head_frames: all checks passed")
sys.exit(1 if FAILS else 0)
