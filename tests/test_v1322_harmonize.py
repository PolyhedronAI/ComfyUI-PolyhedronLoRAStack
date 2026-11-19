#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1322 -- Harmonize in the Polyhedron Filter (colour of one area carried onto another).

Frank 08.11. ("harmonize inside the mask" as a system gap, "darf auch gerne in eine bestehende Filter-Node"): the
Alien bridge needed it twice by hand -- the Qwen-painted back wall matched to its ring, the grey film stills matched
to the amber set photo.

WHAT IS PINNED (CPU, numpy)

  H1  Oklab both ways: srgb -> oklab -> srgb returns the colour (in gamut).
  H2  resolve: auto = reference first, then mask, else off; a mode without its input does not run and says why.
  H3  to reference: the image's Oklab mean lands on the reference's; amount is LINEAR (0.5 = halfway, the preview
      blends on that); one plan measured over the whole batch (two equal frames stay equal -- no flicker).
  H4  mask to its ring: a cast patch takes the ring's light, everything outside the soft mask stays bit-equal, the
      ring is measured RING_GAP px off the edge (a spilled seam is not "surroundings"); the soft edge blends;
      CONTENT KEPT -- a white door under a cool cast comes back to its own whiteness (gains undo the cast, the
      inside contrast survives); the ring's lamps (saturated, 6 % of it) do not tip the white balance (median).
  H5  colour only: lightness kept, colour moved; spread held in SPREAD_LIMITS.
  H6  honest skips: empty mask, a mask that fills the frame (no ring), a mask of another size (resized + said).
  H7  Filter node: unconnected = untouched output (early return holds); harmonize_to connected = reference colour,
      first in the pipeline; harmonize_mask alone = ring mode; ui carries the harmonized preview + the note;
      canon appended at the end; harmonize kept out of presets (py + js).
  H8  preview (js): the harmonized frame is loaded beside the source, blended live by harmonize_amount BEFORE the
      detail stage, 'off' = 0; both new knobs are live widgets; the note is drawn.

Script-style: exit 0 = pass.
"""

import importlib
import os
import re
import sys

NAME = "v1322"
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
rng = np.random.default_rng(1322)


def lab_mean(x, sel=None):
    lab = H.srgb_to_oklab(x).reshape(-1, 3)
    if sel is not None:
        lab = lab[sel.reshape(-1)]
    return lab.mean(0)


# ---------------------------------------------------------------- H1
print("H1 Oklab both ways")
x = rng.random((400, 3)).astype(np.float32)
_need(np.abs(H.oklab_to_srgb(H.srgb_to_oklab(x)) - x).max() < 1e-4, "roundtrip within 1e-4")

# ---------------------------------------------------------------- H2
print("H2 resolve")
_need(H.resolve("auto", True, True)[0] == "to reference", "auto + reference + mask -> to reference")
_need(H.resolve("auto", False, True)[0] == "mask to its ring", "auto + mask -> mask to its ring")
_need(H.resolve("auto", False, False) == ("off", None), "auto alone -> off, silent")
m, note = H.resolve("to reference", False, True)
_need(m == "off" and "harmonize_to" in (note or ""), "to reference without reference -> off + says why")
m, note = H.resolve("mask to its ring", True, False)
_need(m == "off" and "harmonize_mask" in (note or ""), "ring without mask -> off + says why")
_need(H.resolve("off", True, True)[0] == "off", "off stays off")

# ---------------------------------------------------------------- H3
print("H3 to reference")
grey = np.clip(0.5 + rng.normal(0, 0.05, (3, 48, 64, 3)), 0, 1).astype(np.float32)
grey[1] = grey[0]
amber = np.clip(np.array([0.85, 0.55, 0.22]) + rng.normal(0, 0.04, (2, 30, 40, 3)), 0, 1).astype(np.float32)
plan = H.measure(grey, "to reference", reference=amber)
full = H.apply(grey, plan, 1.0)
d = lab_mean(full) - lab_mean(amber)
_need(np.abs(d).max() < 0.01, "mean lands on the reference (max dLab %.4f)" % np.abs(d).max())
_need(np.array_equal(full[0], full[1]), "two equal frames stay equal (one plan for the batch)")
half = H.apply(grey, plan, 0.5)
lin = grey + 0.5 * (full - grey)
_need(np.abs(half - lin).max() < 1e-5, "amount is linear (0.5 = halfway, the preview's blend)")
_need(np.array_equal(H.apply(grey, plan, 0.0), grey), "amount 0 = untouched")
_need("warmer" in (H.describe(plan) or ""), "describe says warmer: %s" % H.describe(plan))
mixed = grey.copy()
mixed[2] = np.clip(mixed[2] * np.array([0.6, 0.8, 1.3], np.float32), 0, 1)   # one frame far bluer
pm = H.measure(mixed, "to reference", reference=amber)
_need(np.abs(pm["mu_s"] - lab_mean(mixed)).max() < 0.005,
      "the plan is measured over the WHOLE batch, not one frame (dLab %.4f)" % np.abs(pm["mu_s"] - lab_mean(mixed)).max())

# ---------------------------------------------------------------- H4
print("H4 mask to its ring")
room = np.zeros((1, 90, 90, 3), np.float32)
room[...] = (0.80, 0.50, 0.25)
room[0, 30:60, 30:60] = (0.62, 0.50, 0.42)            # the painted patch, cooler and darker
room[0, 28:30, 30:60] = (0.2, 0.9, 0.3)               # 2 px green spill right at the edge (inside the gap)
mask = np.zeros((1, 90, 90), np.float32)
mask[0, 30:60, 30:60] = 1.0
mask[0, 29, 30:60] = 0.5                              # one soft row
plan = H.measure(room, "mask to its ring", ring=8, mask=mask)
out = H.apply(room, plan, 1.0, mask)
inside = lab_mean(out[:, 32:58, 32:58]) - lab_mean(room[:, 0:20, 0:20])
_need(np.abs(inside).max() < 0.01, "inside takes the ring's light (max dLab %.4f)" % np.abs(inside).max())
outside = mask[0] == 0
_need(np.array_equal(out[0][outside], room[0][outside]), "outside the mask bit-equal")
full1 = H.apply(room, plan, 1.0, np.ones_like(mask))
_need(np.abs(out[0, 29, 40] - (room[0, 29, 40] + 0.5 * (full1[0, 29, 40] - room[0, 29, 40]))).max() < 1e-5
      and not np.array_equal(out[0, 29, 40], room[0, 29, 40]), "the soft row is blended by its mask value")
g = H.ring_of(mask > 0.5, 8)[0]
_need(not g[28, 40] and not g[29, 40] and g[27, 40] and g[20, 40] and not g[19, 40],
      "ring starts RING_GAP px off the edge and is 8 px wide (spill not measured)")
_need("white balance" in (H.describe(plan) or ""), "describe: %s" % H.describe(plan))
p_thin = H.measure(room, "mask to its ring", ring=2, mask=mask)
o_thin = H.apply(room, p_thin, 1.0, mask)
_need(p_thin["mode"] != "off" and np.abs(lab_mean(o_thin[:, 32:58, 32:58]) - lab_mean(room[:, 0:20, 0:20])).max() < 0.01,
      "the thinnest ring (2 px, the widget minimum) still measures the light across the gap")
xr = room.copy()
xr[0, 30:60, 30:60] = (0.10, 0.20, 0.95)                 # an absurd blue patch
px = H.measure(xr, "mask to its ring", ring=8, mask=mask)
_need(px["gains"].min() >= H.GAIN_LIMITS[0] - 1e-9 and px["gains"].max() <= H.GAIN_LIMITS[1] + 1e-9,
      "gains held on the safety rail %s" % (H.GAIN_LIMITS,))

# content kept: one warm light over grey albedos; the patch holds a WHITE door under a cool cast
lin = lambda c: np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
gam = lambda l: np.where(l <= 0.0031308, l * 12.92, 1.055 * np.power(np.clip(l, 0, 1), 1 / 2.4) - 0.055)
yy, xx = np.mgrid[0:120, 0:160].astype(np.float32)
albedo = 0.25 + 0.15 * np.sin(xx / 7.0) * np.cos(yy / 9.0)
albedo[40:80, 60:100] = 0.85                          # the white door, inside the mask
light = np.array([1.0, 0.72, 0.42], np.float32)
truth = gam(albedo[..., None] * light).astype(np.float32)[None]
cast = np.array([0.70, 0.85, 1.25], np.float32)
m2 = np.zeros((1, 120, 160), np.float32)
m2[0, 30:95, 45:115] = 1.0
painted = truth.copy()
painted[0, 30:95, 45:115] = gam(lin(truth[0, 30:95, 45:115]) * cast)
p2 = H.measure(painted, "mask to its ring", ring=8, mask=m2)
fixed = H.apply(painted, p2, 1.0, m2)
err = np.abs(H.srgb_to_oklab(fixed[0, 32:93, 47:113]) - H.srgb_to_oklab(truth[0, 32:93, 47:113])).reshape(-1, 3).mean(0)
was = np.abs(H.srgb_to_oklab(painted[0, 32:93, 47:113]) - H.srgb_to_oklab(truth[0, 32:93, 47:113])).reshape(-1, 3).mean(0)
_need(err.max() < 0.01 and was.max() > 0.03, "the cast is undone: dLab %s (was %s)" % (err.round(4), was.round(4)))
door = H.srgb_to_oklab(fixed[0, 45:75, 65:95])[..., 0].mean()
_need(abs(door - H.srgb_to_oklab(truth[0, 45:75, 65:95])[..., 0].mean()) < 0.01, "the white door stays white")

# lamps in the ring: 6 % saturated amber points must not tip the balance (median)
lamps = truth.copy()
rng2 = np.random.default_rng(7)
pick = rng2.random((120, 160)) < 0.06
pick[30:95, 45:115] = False
lamps[0][pick] = (1.0, 0.75, 0.05)
pl = H.measure(np.where(m2[..., None] > 0.5, painted, lamps), "mask to its ring", ring=8, mask=m2)
pc = H.measure(painted, "mask to its ring", ring=8, mask=m2)
_need(np.abs(pl["gains"] / pc["gains"] - 1).max() < 0.05,
      "lamps do not tip the white balance (gains %s vs %s)" % (pl["gains"].round(3), pc["gains"].round(3)))
pm = H.measure(painted, "mask to its ring", ring=8, mask=m2, tone="colour only")
_need(pm["k"] == 1.0 and not pm["light"], "colour only: no light factor")

# ---------------------------------------------------------------- H5
print("H5 colour only, spread limits")
plan = H.measure(grey, "to reference", tone="colour only", reference=amber)
co = H.apply(grey, plan, 1.0)
dl = lab_mean(co) - lab_mean(grey)
_need(abs(dl[0]) < 0.01 and abs(lab_mean(co)[2] - lab_mean(amber)[2]) < 0.01,
      "colour only: L kept (%+.4f), b moved to the reference" % dl[0])
_need((plan["k"] >= H.SPREAD_LIMITS[0] - 1e-9).all() and (plan["k"] <= H.SPREAD_LIMITS[1] + 1e-9).all(),
      "spread ratio held in SPREAD_LIMITS")
flat = np.full((1, 20, 20, 3), 0.5, np.float32)
p2 = H.measure(flat, "to reference", reference=amber)
_need(p2["k"].max() <= H.SPREAD_LIMITS[1] + 1e-9, "a flat source is not blown up past the limit")

# ---------------------------------------------------------------- H6
print("H6 honest skips")
p = H.measure(room, "mask to its ring", mask=np.zeros((1, 90, 90), np.float32))
_need(p["mode"] == "off" and "empty" in p["note"], "empty mask -> off + note")
p = H.measure(room, "mask to its ring", mask=np.ones((1, 90, 90), np.float32))
_need(p["mode"] == "off" and "ring" in p["note"], "full-frame mask -> no ring + note")
mm, note = H.fit_mask(np.ones((1, 45, 45), np.float32), 1, 90, 90)
_need(mm.shape == (1, 90, 90) and note and "resized" in note, "mask of another size resized + said")
mm, note = H.fit_mask(np.ones((2, 90, 90), np.float32), 4, 90, 90)
_need(mm.shape[0] == 4 and note and "last one holds" in note, "fewer masks than frames: the last holds + said")

# ---------------------------------------------------------------- H7
print("H7 Filter node")
try:
    import torch
except Exception:
    torch = None
node = F.ULSFilter()
base = dict(exposure=0.0, temperature=0.0, tint=0.0, contrast=0.0, gamma=1.0, shadows=0.0, highlights=0.0,
            saturation=0.0, vibrance=0.0, hue_shift=0.0, lut_name="none", lut_strength=1.0, sharpen_amount=0.0,
            sharpen_radius=1.0, preset="none")
F.ULSFilter._make_preview = lambda self, image: {"filename": "x.png", "probe": np.asarray(image)[0].copy()}
img = torch.from_numpy(grey) if torch is not None else grey
r = node.apply(img, **base)
_need(r["result"][0] is img, "nothing connected: output untouched (early return)")
_need("harmonized" not in r["ui"]["ph_filter"][0], "nothing connected: no harmonized preview")
ref = torch.from_numpy(amber) if torch is not None else amber
r = node.apply(img, harmonize_to=ref, **base)
o = np.asarray(r["result"][0])
_need(np.abs(lab_mean(o) - lab_mean(amber)).max() < 0.01, "harmonize_to connected: the reference colour")
it = r["ui"]["ph_filter"][0]
_need("harmonized" in it and "to the reference" in it.get("harmonize_note", ""), "ui: harmonized preview + note")
r = node.apply(img, harmonize_to=ref, harmonize="off", **base)
_need(r["result"][0] is img, "harmonize off: untouched")
r = node.apply(img, harmonize_to=ref, harmonize_amount=0.5, exposure=0.5, **{k: v for k, v in base.items()
                                                                              if k != "exposure"})
r2 = node.apply(torch.from_numpy(H.apply(grey, H.measure(grey, "to reference", reference=amber), 0.5))
                if torch is not None else H.apply(grey, H.measure(grey, "to reference", reference=amber), 0.5),
                exposure=0.5, **{k: v for k, v in base.items() if k != "exposure"})
_need(np.abs(np.asarray(r["result"][0]) - np.asarray(r2["result"][0])).max() < 1e-4,
      "harmonize runs FIRST (then the grade)")
rm = torch.from_numpy(room) if torch is not None else room
mk = torch.from_numpy(mask) if torch is not None else mask
r = node.apply(rm, harmonize_mask=mk, harmonize_seam=0, **base)   # v1323: the seam changes outside pixels
o = np.asarray(r["result"][0])
_need(np.abs(lab_mean(o[:, 32:58, 32:58]) - lab_mean(room[:, 0:20, 0:20])).max() < 0.01
      and np.abs(o[0][outside] - room[0][outside]).max() < 1e-6,
      "harmonize_mask alone: ring mode, outside untouched (the neutral grade after it: float noise only)")
canon = tuple(F.FILTER_CANON)
# RE-GROUNDED v1323 (DECLARED): harmonize_seam appended after the four -- the v1322 four keep their places
_need(canon[18:22] == ("harmonize", "harmonize_amount", "harmonize_ring", "harmonize_tone")
      and canon[17] == "detail_amount", "canon: four appended at the end, the old 18 untouched")
_need(not any(k.startswith("harmonize") for k in F.PRESET_KEYS), "py: harmonize stays out of presets")
req = list(F.ULSFilter.INPUT_TYPES()["required"])[1:]
_need(tuple(req) == canon, "INPUT_TYPES order = canon")
opt = F.ULSFilter.INPUT_TYPES()["optional"]
_need(opt["harmonize_to"][0] == "IMAGE" and opt["harmonize_mask"][0] == "MASK", "optional inputs typed literally")
_need(F.ULSFilter.INPUT_TYPES()["required"]["harmonize"][1]["default"] == "auto"
      and F.ULSFilter.INPUT_TYPES()["required"]["harmonize_ring"][1]["default"] == 8, "defaults auto, ring 8")

# ---------------------------------------------------------------- H8
print("H8 preview (js)")
JS = open(os.path.join(ROOT, "web", "js", "ph_filter.js"), encoding="utf-8").read()
_need('"harmonize", "harmonize_amount"' in JS.split("const LIVE_WIDGETS")[1].split(";")[0], "both knobs live")
_need('!k.startsWith("harmonize")' in JS.split("const PRESET_KEYS")[1].split(";")[0], "js: out of presets")
rec = JS.split("function _pfRecompute")[1].split("\nfunction ")[0]
_need(rec.find("_pfHarmData") >= 0 and rec.find("_pfHarmData") < rec.find("_pfDetail(node)"),
      "blend before the detail stage")
_need("hAmt * (h[i] / 255 - b0[f])" in rec, "blend is the linear one")
ha = JS.split("function _pfHarmAmount")[1].split("\nfunction ")[0]
_need('m.value === "off"' in ha and "return 0" in ha, "'off' = 0")
_need("_pfLoadHarm(node, item);" in JS and "item.harmonize_note" in JS
      and "_pfBadge(ctx, r.x + 6, r.y + 60" in JS, "loaded beside the source + the note drawn")
d = re.search(r"const CANON_DEFAULTS = \{(.*?)\};", JS, re.S).group(1)
_need('harmonize: "auto", harmonize_amount: 1.0, harmonize_ring: 8, harmonize_tone: "colour and light"' in d,  # v1323 seam follows
      "CANON_DEFAULTS carry the four defaults (Reset)")

print("")
if _fails:
    print("test_%s: %d FAIL" % (NAME, len(_fails)))
    sys.exit(1)
print("test_%s: all pass" % NAME)
