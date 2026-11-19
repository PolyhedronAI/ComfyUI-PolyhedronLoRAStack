#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1324 -- harmonize_window: 'to reference' that follows a drifting clip.

Frank's H3 bridge runs of 09.11.: warm at the start (the photo), grey from the middle on (b 0.009-0.016 against the
photo's 0.049). One correction for the whole clip warmed the grey part but turned the start too orange (b 0.076). A
window of 3 s brought the whole clip to b 0.039-0.054 (measured on his two scenes).

WHAT IS PINNED (CPU, numpy)

  W1  a drifting clip (warm -> grey): with a window every frame lands near the reference warmth; one global shift
      leaves the drift (start too warm, end too cool).
  W2  no flicker: content coming and going (a monitor) does not pump the rest; frame-to-frame noise in the clip does not come back as a jumping correction (the applied shift
      changes smoothly; its frame-to-frame change is far below the noise).
  W3  window 0 = the v1323 plan bit-equal; chunked apply (start offsets, as the Filter calls it) = one apply.
  W4  colour only keeps every frame's lightness; with a mask only the masked area is measured.
  W5  wiring: harmonize_window appended after harmonize_seam, INT 0..10000 default 0, the node hands it to measure and
      the chunk start to apply, the note says "window", out of presets, CANON_DEFAULTS carry 0.

Script-style: exit 0 = pass.
"""

import importlib
import os
import re
import sys

NAME = "v1324"
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
rng = np.random.default_rng(1324)

n = 60
t = np.linspace(0, 1, n, dtype=np.float32)[:, None, None, None]
warm = np.array([0.82, 0.60, 0.36], np.float32)
grey = np.array([0.62, 0.60, 0.58], np.float32)
tex = (0.85 + 0.3 * rng.random((1, 40, 56, 1))).astype(np.float32)
clip = np.clip(((1 - t) * warm + t * grey) * tex + rng.normal(0, 0.01, (n, 40, 56, 3)), 0, 1).astype(np.float32)
ref = np.clip(warm * tex[0] + 0.0, 0, 1)[None].astype(np.float32)


def bmean(x):
    return float(H.srgb_to_oklab(x)[..., 2].mean())


bt = bmean(ref)

print("W1 the correction follows the drift")
pg = H.measure(clip, "to reference", tone="colour only", reference=ref)
pw = H.measure(clip, "to reference", tone="colour only", reference=ref, window=12)
og, ow = H.apply(clip, pg, 1.0), H.apply(clip, pw, 1.0)
eg = max(abs(bmean(og[:5]) - bt), abs(bmean(og[-5:]) - bt))
ew = max(abs(bmean(ow[:5]) - bt), abs(bmean(ow[-5:]) - bt))
_need(ew < 0.006 and ew < 0.4 * eg, "start and end near the reference warmth: window %.4f, one shift %.4f" % (ew, eg))

print("W2 no flicker")
shift = np.array([bmean(ow[i:i + 1]) - bmean(clip[i:i + 1]) for i in range(n)])
jump = np.abs(np.diff(shift)).max()
noise = np.abs(np.diff([bmean(clip[i:i + 1]) for i in range(n)])).max()
_need(jump < 0.5 * noise + 1e-3 and jump < 0.004, "applied shift moves smoothly (max step %.4f, clip noise %.4f)" % (jump, noise))

# content that comes and goes (a blue monitor in and out of the frame every 3 frames): a per-frame correction
# would pump the colour of everything else; the window keeps the background steady
blue = clip.copy()
on = (np.arange(n) // 3) % 2 == 0
blue[on, 5:25, 30:52] = (0.15, 0.25, 0.85)
pb = H.measure(blue, "to reference", tone="colour only", reference=ref, window=24)
ob = H.apply(blue, pb, 1.0)
bg = np.array([bmean(ob[i:i + 1, 28:, :]) for i in range(n)])
pump = np.abs(np.diff(bg)).max()
_need(pump < 0.004, "a monitor coming and going does not pump the background (max step %.4f)" % pump)

print("W3 window 0 = v1323, chunks = one apply")
p0 = H.measure(clip, "to reference", tone="colour only", reference=ref, window=0)
_need("mu_s_t" not in p0 and np.array_equal(H.apply(clip, p0, 1.0), og), "window 0: the one-plan path, bit-equal")
_need("mu_s_t" not in H.measure(clip, "to reference", reference=ref, window=1), "window 1 = no window")
parts = np.concatenate([H.apply(clip[i:i + 7], pw, 1.0, None, i) for i in range(0, n, 7)])
_need(np.abs(parts - ow).max() < 1e-6, "chunked with start offsets = one apply")
half = H.apply(clip, pw, 0.5)
_need(np.abs(half - (clip + 0.5 * (ow - clip))).max() < 2e-6, "still linear in amount")

print("W4 colour only, mask")
L = lambda x: H.srgb_to_oklab(x)[..., 0]  # noqa: E731
_need(np.abs(L(ow).mean((1, 2)) - L(clip).mean((1, 2))).max() < 0.004, "colour only: every frame keeps its lightness")
m = np.zeros((n, 40, 56), np.float32)
m[:, :, :28] = 1.0
two = clip.copy()
two[:, :, 28:] = (0.2, 0.3, 0.8)                     # the unmasked half is something else entirely
pm = H.measure(two, "to reference", tone="colour only", reference=ref, mask=m, window=12)
om = H.apply(two, pm, 1.0, m)
_need(np.array_equal(om[:, :, 28:], two[:, :, 28:]), "with a mask the rest stays bit-equal")
em = max(abs(bmean(om[:5, :, :28]) - bt), abs(bmean(om[-5:, :, :28]) - bt))
_need(em < 0.008, "only the masked area is measured: it lands on the reference (%.4f)" % em)
_need("window" in H.describe(pw) and "12-frame" in H.describe(pw), "note: %s" % H.describe(pw))

print("W5 wiring")
canon = tuple(F.FILTER_CANON)
# RE-GROUNDED v1329: sharpen_masking appended behind it -- DECLARED; the pair stays adjacent
_i = canon.index("harmonize_window")
_need(canon[_i - 1:_i + 1] == ("harmonize_seam", "harmonize_window"), "canon: appended after harmonize_seam")
spec = F.ULSFilter.INPUT_TYPES()["required"]["harmonize_window"]
_need(spec[0] == "INT" and spec[1]["default"] == 0 and spec[1]["min"] == 0 and spec[1]["max"] == 10000,
      "INT 0..10000, default 0")
_need("harmonize_window" not in F.PRESET_KEYS, "out of presets")
src = open(os.path.join(ROOT, "nodes", "ph_filter.py"), encoding="utf-8").read()
_need("int(harmonize_window))" in src and "_process(src[i:i + step], dch, hch, i)" in src,
      "the node hands the window to measure and the chunk start to apply")
JS = open(os.path.join(ROOT, "web", "js", "ph_filter.js"), encoding="utf-8").read()
d = re.search(r"const CANON_DEFAULTS = \{(.*?)\};", JS, re.S).group(1)
_need("harmonize_window: 0" in d, "CANON_DEFAULTS carry 0 (Reset)")

print("")
if _fails:
    print("test_%s: %d FAIL" % (NAME, len(_fails)))
    sys.exit(1)
print("test_%s: all pass" % NAME)
