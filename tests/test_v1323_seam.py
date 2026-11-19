#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1323 -- harmonize_seam: the two-sided soft seam of 'mask to its ring'.

Frank's field run of 08.11. (PH_HARMONIZE_01 branch A): colour right, but the floor rectangle of the Qwen paint
still showed -- one light factor for the whole mask cannot fit every stretch of the seam (floor wanted x0.83, the
mask x0.88 because of the wall panels). His hand version had no edge because it blended across the seam.

WHAT IS PINNED (CPU, numpy)

  S1  a seam whose step CHANGES along the edge (left half too bright, right half too dark -- no global factor fits):
      the jump across the edge shrinks to a fraction, on both halves.
  S2  both sides move: inside towards outside, outside towards inside, meeting at the seam (no overshoot -- the
      sign of the step holds); pixels farther than `seam` px from the edge stay bit-equal (outside) / equal to
      the plain ring result (inside).
  S2b the edge itself (the 2-px gap, partly painted pixels) is bridged smoothly -- no 1-px line along the seam.
  S3  the room's own gradient is not read as a step: a smooth gradient running ACROSS an invisible seam stays
      (the 08.11. overshoot of a +-13 px measurement pinned via SEAM_BAND).
  S4  seam 0 = the v1322 result bit-equal; the result is still linear in amount (the preview's blend).
  S5  wiring: harmonize_seam appended after the v1322 four, default 32, INT 0..128, ring mode only (reference mode
      unchanged), described in the note, out of presets, CANON_DEFAULTS carry 32.

Script-style: exit 0 = pass.
"""

import importlib
import os
import re
import sys

NAME = "v1323"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "nodes"))
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)
    print(("  ok    " if cond else "  FAIL  ") + msg)


import numpy as np  # noqa: E402

H = importlib.import_module("ph_harmonize")
F = importlib.import_module("ph_filter")

Lab = lambda x: H.srgb_to_oklab(x)[..., 0]  # noqa: E731

# a floor: smooth vertical gradient, one warm light
yy, xx = np.mgrid[0:140, 0:200].astype(np.float32)
base = (0.42 + 0.0015 * yy)[..., None] * np.array([1.0, 0.78, 0.55], np.float32)
room = np.clip(base, 0, 1)[None]
mask = np.zeros((1, 140, 200), np.float32)
mask[0, 30:100, 40:160] = 1.0
painted = room.copy()
lin, gam = H._lin, H._gamma
inside = painted[0, 30:100, 40:160]
fac = np.where(xx[30:100, 40:160] < 100, 1.18, 0.86)[..., None]   # left too bright, right too dark
painted[0, 30:100, 40:160] = gam(lin(inside) * fac).astype(np.float32)


def jump(img, col):
    """|L just inside - L just outside| across the bottom edge (row 99 | row 100) at the given columns."""
    L = Lab(img[0])
    return float(np.abs(L[99, col] - L[100, col]).mean())


print("S1 a step that changes along the edge")
p0 = H.measure(painted, "mask to its ring", ring=8, mask=mask, seam=0)
r0 = H.apply(painted, p0, 1.0, mask)
p1 = H.measure(painted, "mask to its ring", ring=8, mask=mask, seam=32)
r1 = H.apply(painted, p1, 1.0, mask)
left, right = slice(55, 85), slice(115, 145)
for nm, c in (("left", left), ("right", right)):
    a, b = jump(r0, c), jump(r1, c)
    _need(b < 0.1 * a and b < 0.004, "%s half: jump %.4f -> %.4f" % (nm, a, b))

print("S1b two nearby stretches of seam with opposite steps (a 24-px strip): each keeps its own")
strip = np.zeros((1, 140, 200), np.float32)
strip[0, 60:84, 40:160] = 1.0
ps = room.copy()
grad = np.linspace(1.18, 0.86, 24, dtype=np.float32)[:, None, None]        # top too bright, bottom too dark
ps[0, 60:84, 40:160] = gam(lin(room[0, 60:84, 40:160]) * grad).astype(np.float32)
qs = H.measure(ps, "mask to its ring", ring=8, mask=strip, seam=32)
rs = H.apply(ps, qs, 1.0, strip)
Ls, Lp2 = Lab(rs[0]), Lab(ps[0])
jt0, jt1 = np.abs(Lp2[60, 70:130] - Lp2[59, 70:130]).mean(), np.abs(Ls[60, 70:130] - Ls[59, 70:130]).mean()
jb0, jb1 = np.abs(Lp2[83, 70:130] - Lp2[84, 70:130]).mean(), np.abs(Ls[83, 70:130] - Ls[84, 70:130]).mean()
_need(jt1 < 0.1 * jt0 and jb1 < 0.1 * jb0,
      "top %.4f -> %.4f, bottom %.4f -> %.4f (no mixing of the two edges)" % (jt0, jt1, jb0, jb1))

print("S2 both sides move, meet, and stop at the band")
Lr0, Lr1, Lp = Lab(r0[0]), Lab(r1[0]), Lab(painted[0])
_need(Lr1[100, left].mean() > Lp[100, left].mean() + 0.002, "left: the outside rises towards the bright inside")
_need(Lr1[99, left].mean() < Lr0[99, left].mean() - 0.002, "left: the inside comes down towards the outside")
_need((Lr1[99, left] - Lr1[100, left]).mean() * (Lr0[99, left] - Lr0[100, left]).mean() >= 0,
      "no overshoot: the step keeps its sign")
far = np.ones((140, 200), bool)
far[max(0, 30 - 33):100 + 33, max(0, 40 - 33):160 + 33] = False
assert far.sum() > 1000   # (08.11.: a negative slice once left this region empty)
_need(np.array_equal(r1[0][far], painted[0][far]), "outside farther than seam px: bit-equal")
_need(np.abs(r1[0, 64:66, 98:102] - r0[0, 64:66, 98:102]).max() < 1e-6,
      "deep inside (35 px from every edge): the plain ring result")

print("S2b the edge itself: no line along the seam (08.11.: partial mask pixels left a 1-px notch/bump)")
soft = np.zeros((1, 140, 200), np.float32)
soft[0, 30:100, 40:160] = 1.0
soft[0, 29, 40:160] = 0.4
soft[0, 100, 40:160] = 0.4
pp = H.measure(painted, "mask to its ring", ring=8, mask=soft, seam=32)
rr = H.apply(painted, pp, 1.0, soft)
prof = Lab(rr[0])[90:112, 115:145].mean(1)
d2 = np.abs(np.diff(prof, 2)).max()
_need(d2 < 0.006, "the profile across the soft bottom edge has no kink (max 2nd difference %.4f)" % d2)

prof2 = Lab(r1[0])[60:140, 60:80].mean(1)
d3 = np.abs(np.diff(prof2, 2)).max()
_need(d3 < 0.003, "no new edge where the band ends, 32 px out (max 2nd difference %.4f)" % d3)
_need(Lab(r1[0])[103, left].mean() - Lab(painted[0])[103, left].mean() >
      2.0 * (Lab(r1[0])[125, left].mean() - Lab(painted[0])[125, left].mean()) > -1,
      "the outside correction fades with the distance (3 px > 25 px)")

print("S3 the room's own gradient is no step")
q = H.measure(room, "mask to its ring", ring=8, mask=mask, seam=32)
rq = H.apply(room, q, 1.0, mask)
# measured 08.11.: a steep floor gradient (0.0015/px) leaves L 0.0034 -- the two 2-px bands sit ~7 px apart;
# far below a visible step (~0.01-0.02); extrapolating per side would make textured seams noisy
dl = np.abs(Lab(rq[0]) - Lab(room[0])).max()
_need(dl < 0.005, "a seamless picture stays as it is (max dL %.4f, known gradient bias)" % dl)
_need(H.SEAM_BAND <= 3, "SEAM_BAND thin (%d px) -- 08.11.: +-13 px read the floor gradient as a step" % H.SEAM_BAND)

print("S4 seam 0 = v1322, linear in amount")
_need(p0["seam"] == 0 and np.array_equal(r0, H.apply(painted, dict(p0), 1.0, mask)), "seam 0: the plain ring move")
half = H.apply(painted, p1, 0.5, mask)
_need(np.abs(half - (painted + 0.5 * (r1 - painted))).max() < 2e-6, "linear in amount with the seam on")
pr = H.measure(painted, "to reference", reference=room, mask=mask, seam=32)
_need("seam" not in pr, "reference mode does not carry a seam")
_need("seam softened 32 px" in H.describe(p1), "note: %s" % H.describe(p1))

print("S5 wiring")
canon = tuple(F.FILTER_CANON)
# RE-GROUNDED v1324 (DECLARED): harmonize_window appended after it -- the seam keeps its place 22
_need(canon[22] == "harmonize_seam" and canon[18:22] == ("harmonize", "harmonize_amount", "harmonize_ring",
                                                        "harmonize_tone"), "canon: appended after the v1322 four")
spec = F.ULSFilter.INPUT_TYPES()["required"]["harmonize_seam"]
_need(spec[0] == "INT" and spec[1]["default"] == 32 and spec[1]["min"] == 0 and spec[1]["max"] == 128,
      "INT 0..128, default 32")
_need("harmonize_seam" not in F.PRESET_KEYS, "out of presets")
JS = open(os.path.join(ROOT, "web", "js", "ph_filter.js"), encoding="utf-8").read()
d = re.search(r"const CANON_DEFAULTS = \{(.*?)\};", JS, re.S).group(1)
_need("harmonize_seam: 32" in d, "CANON_DEFAULTS carry 32 (Reset)")
src = open(os.path.join(ROOT, "nodes", "ph_filter.py"), encoding="utf-8").read()
# RE-GROUNDED v1324 (DECLARED): the window follows the seam in the call
_need("int(harmonize_seam)," in src or "int(harmonize_seam))" in src, "the node hands the seam to measure")

print("")
if _fails:
    print("test_%s: %d FAIL" % (NAME, len(_fails)))
    sys.exit(1)
print("test_%s: all pass" % NAME)
