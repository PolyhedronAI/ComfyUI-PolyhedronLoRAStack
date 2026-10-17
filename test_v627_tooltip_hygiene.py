#!/usr/bin/env python3
"""
test_v627_tooltip_hygiene -- node tooltips stay future-proof: no model-version pins,
no internal build-version references, no time-bound wording, and (v814) no
wording bound to the sample picture it was measured on.

v627 (Frank's rule): tooltips describe BEHAVIOUR, not the version landscape of the day.
A tooltip that says "the 4.26 generation" or "rife426" goes stale the day a new
checkpoint ships; "(v292 convention)" is changelog material, not user guidance; "today"
rots silently. The concrete case was Karte 4: the Interpolate ckpt_name tooltip carried
a model-version recommendation (and its hover rendering hangs in ComfyUI's frontend --
shorter text is the bonus, the version pin was the offence).

This guard extracts EVERY "tooltip" string from nodes/*.py (concatenated pieces joined,
exactly like Python sees them) and rejects:

    rife\\d            -- a concrete RIFE checkpoint version in prose
    \\d.\\d+ generation -- "the 4.26 generation" style pins
    \\bv\\d{3}\\b        -- internal pack build references (v292, v312, ...)
    \\btoday\\b          -- time-bound wording

Widget DEFAULTS (e.g. default="rife426.pth") are functional filenames, not prose --
they are deliberately NOT scanned.

SELF-TEST: the checker is run against a synthetic offending tooltip first; if it fails
to flag it, the guard fails -- so a broken checker can never green-light a violation.
"""
import os
import re
import sys
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
NODES = os.path.join(os.path.dirname(HERE), "nodes")

RX_TT = re.compile(r'"tooltip":\s*((?:"(?:[^"\\]|\\.)*"\s*)+)')
RX_BAD = re.compile(r'rife\d|\d\.\d+ generation|\bv\d{3,4}\b|\btoday\b', re.I)  # v1001: 3-4 digits

# v814 -- THE SAME DOCTRINE, ONE LEVEL FURTHER (Frank's rule).
#
# v627 said: a tooltip describes BEHAVIOUR, not the version landscape of
# the day. The same argument applies to the PICTURE of the day. A tooltip
# that names the motif it was measured on ("the meadow", "the watch") or
# quotes a single measurement from it ("0.234 -> 0.000", "21/255") reads
# as evidence to the person who took the measurement and as noise to
# everyone else -- and it goes stale the moment the next picture behaves
# differently. The BEHAVIOUR is permanent, the sample is not. So: say what
# the setting does and when to reach for it; leave the numbers in the
# changelog, where they are dated and can be checked.
#
# Deliberately NOT banned: illustrative sizes ("1440x1440 -> 2048x2048"),
# which teach a shape rather than report a result, and model or format
# names, which are functional facts about what a wording expects.
RX_SAMPLE = [
    (re.compile(r'\b(meadow|bandana|graffiti|cel art|the watch)\b', re.I),
     "names the motif it was measured on"),
    (re.compile(r'\bFrank\b'),
     "names a person"),
    (re.compile(r'\d+\.\d+\s*->\s*\d+\.\d+'),
     "quotes a single measurement as a before/after pair"),
    (re.compile(r'\d+(?:\.\d+)?\s*(?:/\s*255\b|of\s*255\b)'),
     "quotes a measurement in raw 0-255 units"),
]

# ANCHOR EXEMPTION: these node files are sealed by the anchor composite -- changing a
# single tooltip byte in them breaks the anchor gate. Their strings are frozen with
# them (v627 measured this the direct way: two camera tooltip edits flipped the
# composite and were reverted). Polishing an anchor file's text is a deliberate
# anchor UPDATE with Frank's go, never a hygiene side effect.
ANCHOR_EXEMPT = {"wan_model_bridge.py", "ph_camera.py", "ph_cardinal_cameras.py",
                 "ph_mesh3d.py", "ph_mesh_render.py", "uls_merge_math.py",
                 "ph_viewport3d.py"}


def _fail(msg):
    print("[test_v627_tooltip_hygiene] FAIL: " + msg)
    sys.exit(1)


def _joined(block):
    return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', block))


def offenders(src):
    out = []
    for m in RX_TT.finditer(src):
        t = _joined(m.group(1))
        line = src[:m.start()].count("\n") + 1
        if RX_BAD.search(t):
            out.append((line, "version/time-bound wording: " + t[:80]))
        for rx, why in RX_SAMPLE:          # v814
            if rx.search(t):
                out.append((line, why + ": " + t[:80]))
    return out


# SELF-TEST -- a checker that cannot catch is worse than none.
_synthetic = '{"tooltip": "The 4.26 generation of rife426 is, as of today, the best (v292)."}'
if not offenders(_synthetic):
    _fail("the checker did not flag a synthetic offending tooltip -- it can no longer "
          "catch anything, so its green means nothing")

# v814 -- EVERY rule gets its own synthetic offender. One combined probe would
# stay green after three of the four rules were deleted.
_SAMPLE_PROBES = [
    '{"tooltip": "Measured on the meadow, this closes the rectangle."}',
    '{"tooltip": "Frank measured this on his own edit loop."}',
    '{"tooltip": "Tile deviation 0.234 -> 0.000 for a small colour cost."}',
    '{"tooltip": "The tone error in the sky was 21/255 across the seam."}',
]
for _probe in _SAMPLE_PROBES:
    if not offenders(_probe):
        _fail("the checker did not flag this synthetic sample-bound tooltip, so that "
              "rule can no longer catch anything: " + _probe)

# And it must NOT flag what is deliberately allowed -- a rule that fires on
# everything is as useless as one that fires on nothing.
_ALLOWED = [
    '{"tooltip": "Same-shape bases (1440x1440 -> 2048x2048) convert exactly."}',
    '{"tooltip": "klein = the official template, and the right default here."}',
]
for _ok in _ALLOWED:
    if offenders(_ok):
        _fail("the checker flagged a tooltip that is deliberately allowed (illustrative "
              "sizes and model/format names are functional facts): " + _ok)

bad = []
for f in sorted(glob.glob(os.path.join(NODES, "*.py"))):
    if os.path.basename(f) in ANCHOR_EXEMPT:
        continue   # sealed with the anchor composite -- see ANCHOR_EXEMPT above
    src = open(f, encoding="utf-8").read()
    for line, snippet in offenders(src):
        bad.append("%s:%d: %s" % (os.path.basename(f), line, snippet))

if bad:
    _fail("version/time-bound wording in tooltips (describe behaviour, not the version "
          "landscape of the day):\n  " + "\n  ".join(bad))

print("[test_v627_tooltip_hygiene] PASS -- all node tooltips free of model-version "
      "pins, internal build references, time-bound wording and sample-bound wording "
      "(motif names, person names, before/after measurements, raw 0-255 figures); the "
      "checker caught every synthetic offender and left the deliberately allowed "
      "wordings alone")
sys.exit(0)
