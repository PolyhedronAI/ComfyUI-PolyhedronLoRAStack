"""Guard v1384 -- Control Fit stands a control figure on the figure of a reference picture, and names the cycle.

Frank, 24.11.: the walk cycles of one figure are led by ONE gait through the control line (a rig's walk, one camera
per direction) instead of drawn anew per seed. For that the control must show its figure exactly where the still
of that direction has it, and every direction must be picked on the same step.

  F   figure_numbers(): upright top (10th percentile), floor line (lowest row), torso centre; empty -> None
  M   size_mode 'match the reference figure': the control's figure lands on the reference's height, floor line and
      centre (1 px), ONE plan for all frames (a bob stays a bob, scaled); measured on the mask, without one on the
      non-black pixels; the reference by alpha or by its flat backdrop, of another size than the target; no
      reference -> a clear error; the three old modes untouched
  C   cycle_frames / contact_frame: one pass of a looped source in output frames; the contact = the earliest of the
      widest strides, two cycles in, a whole cycle after it; the mirrored walk gets the same frame; not looped -> 0 / -1
  N   node: 'reference' optional, two outputs appended, no widget added (baseline stands), the mode is the list's
      fourth value
  Z   ASCII; version stamps agree

Script-style: exit 0 = pass.
"""
import glob
import os
import re
import sys
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(c, m):
    if not c:
        FAILS.append(m)


fp = types.ModuleType("folder_paths")
fp.get_temp_directory = fp.get_output_directory = fp.get_input_directory = lambda: "/tmp"
fp.get_filename_list = lambda kind: []
sys.modules.setdefault("folder_paths", fp)
sys.path.insert(0, os.path.join(ROOT, "nodes"))
import torch  # noqa: E402
import ph_control as C  # noqa: E402

h, w = 160, 120


def walker(k, n=8, bob=True, x0=40, top=30, foot=129, flip=False):
    """Frame k of an n-frame walk: head + body + two legs spreading; the widest stride at k = 2 and k = 6 (the
    second a pixel narrower), the head 2 px lower there."""
    m = np.zeros((h, w), bool)
    spread = [2, 6, 12, 6, 2, 6, 11, 6][k % n]
    b = 2 if (bob and spread >= 11) else 0
    m[top + b:top + 12 + b, x0 + 4:x0 + 12] = True               # head
    m[top + 12 + b:80, x0:x0 + 16] = True                         # body
    m[80:foot + 1, x0 + 8 - spread - 2:x0 + 8 - spread + 2] = True
    m[80:foot + 1, x0 + 8 + spread - 2:x0 + 8 + spread + 2] = True
    return m[:, ::-1].copy() if flip else m


def clip(flip=False, bob=True):
    v = np.stack([walker(k, bob=bob, flip=flip) for k in range(8)])
    img = np.zeros((8, h, w, 3), np.float32)
    img[v] = (0.2, 0.9, 0.4)
    return torch.from_numpy(img), torch.from_numpy(v.astype(np.float32)), v


# F
img, msk, vis = clip()
f = C.figure_numbers(vis)
check(f["top"] == 30 and f["foot"] == 129 and f["height"] == 100 and abs(f["centre"] - 47.5) < 0.01,
      "F upright top, floor line, torso centre: %s" % f)
check(C.figure_numbers(np.zeros((3, 8, 8), bool)) is None, "F no figure: None")
check(C.figure_numbers(vis[2])["top"] == 32, "F one frame alone: its own top")

# the reference: a figure 600 px high on a 352 x 640 picture with a flat grey backdrop (half the target's size)
RH, RW = 640, 352
ref = np.full((RH, RW, 3), 0.85, np.float32)
ref[8:608, 150:230] = (0.1, 0.1, 0.1)                             # rows 8..607, centre 189.5
ref_t = torch.from_numpy(ref)[None]
WANT = {"height": 1200.0, "foot": 1215.0, "centre": 379.5}       # on 704 x 1280


def run(images, mask, reference, mode=C.SIZE_MATCH, frames=22, end="loop", src=24.0):
    return C.fit(images, src, 24.0, 0.0, frames, C.GRIDS[0], end, 704, 1280, mode, 32, mask, 0, 0, reference)


# M
out, m_out, rep = run(img, msk, ref_t)
got = C.figure_numbers((m_out[:8] > 0.5).numpy())
check(out.shape == (22, 1280, 704, 3) and m_out.shape == (22, 1280, 704), "M the target's size and count")
check(abs(got["height"] - WANT["height"]) <= 1 and abs(got["foot"] - WANT["foot"]) <= 1 and abs(got["centre"] - WANT["centre"]) <= 1,
      "M the figure stands on the reference's: %s" % got)
check(abs(rep["match"]["scale"] - 12.0) < 0.01 and rep["match"]["by"] == "its mask", "M scale 12, measured on the mask: %s" % rep["match"])
tops = [int(np.nonzero((m_out[k] > 0.5).numpy().any(1))[0][0]) for k in range(8)]
check(abs((tops[2] - tops[0]) - 24) <= 1 and tops[0] == tops[1], "M one plan for all frames: the 2 px bob is 24 px: %s" % tops)
lit = (out[0].amax(-1) > 0.45).numpy()                    # half of the figure colour = the mask's edge
check(abs(int(lit.sum()) - int((m_out[0] > 0.5).sum())) < 0.02 * lit.sum(), "M the picture went the mask's way")
check(torch.equal(out[8], out[0]) and torch.equal(out[21], out[5]) and torch.equal(out[15], out[7]) and not torch.equal(out[7], out[6]),
      "M looped: frame 8 = frame 0, every source frame in its place")
out2, m2, rep2 = run(img, None, ref_t)
g2 = C.figure_numbers((out2[:8].amax(-1) > 0.45).numpy())
check(m2 is None and rep2["match"]["by"] == "its non-black pixels" and abs(g2["height"] - 1200) <= 2 and abs(g2["foot"] - 1215) <= 1,
      "M without a mask: the non-black pixels: %s" % g2)
rgba = np.zeros((RH, RW, 4), np.float32); rgba[..., :3] = 0.1; rgba[8:608, 150:230, 3] = 1.0      # one flat colour: only the alpha tells
_o, m3, _r = run(img, msk, torch.from_numpy(rgba)[None])
g3 = C.figure_numbers((m3[:8] > 0.5).numpy())
check(abs(g3["height"] - 1200) <= 1 and abs(g3["centre"] - 379.5) <= 1, "M a reference with alpha reads the same: %s" % g3)
off = ref.copy(); off[:] = 0.85; off[100:400, 40:100] = (0.1, 0.1, 0.1)      # another figure: 300 px, left, higher
_o, m4, _r = run(img, msk, torch.from_numpy(off)[None])
g4 = C.figure_numbers((m4[:8] > 0.5).numpy())
check(abs(g4["height"] - 600) <= 1 and abs(g4["foot"] - 799) <= 1 and abs(g4["centre"] - 139.5) <= 1, "M another reference, another place: %s" % g4)
try:
    run(img, msk, None); check(False, "M no reference: refused")
except ValueError as e:
    check("needs the reference input" in str(e), "M a clear error: %s" % e)
try:
    run(img, msk, torch.full((1, 64, 64, 3), 0.5)); check(False, "M an empty reference: refused")
except ValueError as e:
    check("no figure found in the reference" in str(e), "M an empty reference named: %s" % e)
o_fill, _m, r_fill = run(img, msk, None, "fill (crop)")
check(o_fill.shape == (22, 1280, 704, 3) and r_fill["match"] is None and C.SIZE_MODES[:3] == ["fit (pad)", "fill (crop)", "stretch"]
      and C.SIZE_MODES[3] == C.SIZE_MATCH == "match the reference figure", "M the old modes stand; the new one is the fourth")

# C
check(rep["cycle_frames"] == 8, "C one pass of 8 source frames at 24 -> 24 fps = 8 frames")
_o, mm, rc = run(img, msk, ref_t, frames=56)
check(rc["cycle_frames"] == 8 and rc["contact_frame"] == 18, "C contact: frame 2 of the cycle (the widest; frame 6 is narrower), two cycles in: %s" % rc["contact_frame"])
_o, mm, rs = run(img, msk, ref_t, frames=56, src=12.0)
check(rs["cycle_frames"] == 16 and rs["contact_frame"] in (36, 37), "C at half the source rate the cycle is 16 frames: %s %s" % (rs["cycle_frames"], rs["contact_frame"]))
imf, mkf, _v = clip(flip=True)
reff = torch.from_numpy(ref[:, ::-1].copy())[None]
_o, mm, rf = run(imf, mkf, reff, frames=56)
check(rf["contact_frame"] == rc["contact_frame"], "C the mirrored walk starts on the same frame: %s" % rf["contact_frame"])
near = np.stack([walker(k) for k in range(8)]); near[6] = near[2]                      # two equal contacts
_o, mm, rn = run(img, torch.from_numpy(near.astype(np.float32)), ref_t, frames=56)
check(rn["contact_frame"] == 18, "C of two equal contacts the earliest: %s" % rn["contact_frame"])
_o, mm, r22 = run(img, msk, ref_t, frames=22)
check(r22["contact_frame"] == 10 and r22["contact_frame"] + 8 <= 22, "C a short clip: the last contact with a whole cycle after it: %s" % r22["contact_frame"])
_o, mm, rh = run(img, msk, ref_t, frames=22, end="hold last frame")
check(rh["cycle_frames"] == 0 and rh["contact_frame"] == -1, "C not looped: 0 / -1")
_o, mm, r5 = run(img, msk, ref_t, frames=56, src=20.0)
check(any("not a whole number" in l for l in r5["cycle_lines"]), "C a cycle that drifts against the frames is named")
_o, mm, rm = run(img, msk, None, "fill (crop)", frames=56)
check(rm["cycle_frames"] == 8 and rm["contact_frame"] == 18, "C the old modes name the cycle too (from the mask)")

# N
node = C.ULSControlFit
it = node.INPUT_TYPES()
check(list(it["optional"]) == ["fps", "mask", "reference"], "N 'reference' the last optional input")
check(node.RETURN_NAMES == ("images", "mask", "frame_count", "width", "height", "seconds", "info", "cycle_frames", "contact_frame")
      and node.RETURN_TYPES[7:] == ("INT", "INT") and len(node.OUTPUT_TOOLTIPS) == 9, "N two outputs appended")
bl = sorted(glob.glob(os.path.join(ROOT, "WIDGET_ORDER_baseline_v*.txt")))
check(len(bl) == 1 and "\nULSControlFit\ttarget_fps,source_fps,start,frames,grid,at_the_end,width,height,size_mode,multiple,block,block_index\n"
      in open(bl[-1], encoding="utf-8").read() and list(it["required"])[1:] == "target_fps,source_fps,start,frames,grid,at_the_end,width,height,size_mode,multiple,block,block_index".split(","),
      "N no widget added: the baseline stands")
o = node().fit(img, 24.0, 0.0, 0.0, 56, C.GRIDS[0], "loop", 704, 1280, C.SIZE_MATCH, 32, 0, 0, None, msk, ref_t)
check(len(o) == 9 and o[7] == 8 and o[8] == 18 and "matched to the reference figure" in o[6] and "scale 12.000" in o[6]
      and "contact_frame 18" in o[6], "N through the node: %s" % o[6])
on = node().fit(img, 24.0, 0.0, 0.0, 22, C.GRIDS[0], "hold last frame", 704, 1280, "fill (crop)", 32, 0, 0)
check(on[7] == 0 and on[8] == -1 and "matched" not in on[6], "N the old way: 0 / -1, no word of a match")

# Z
for rel in ("nodes/ph_control.py", "tests/test_v1384_control_match.py"):
    with open(os.path.join(ROOT, rel), "rb") as fh:
        check(all(ch < 128 for ch in fh.read()), "Z  ASCII-clean: " + rel)
_pv = re.search(r'PLUGIN_VERSION = "v(\d+)"', open(os.path.join(ROOT, "web", "js", "uls_compat.js")).read())
_pp = re.search(r'^version = "4\.(\d+)\.0"', open(os.path.join(ROOT, "pyproject.toml")).read(), re.M)
# PUBLIC CUT (v409, declared): the public build counts its own way -- vNNN = Major x 100 + Minor (test_v365 V).
check(_pv and _pp and int(_pv.group(1)) >= 409 and int(_pv.group(1)) - 400 == int(_pp.group(1)), "Z  version stamps agree")

if FAILS:
    print("FAIL test_v1384_control_match:")
    for f_ in FAILS:
        print("  - " + f_)
    sys.exit(1)
print("ok test_v1384_control_match")
