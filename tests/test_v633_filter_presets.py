#!/usr/bin/env python3
"""
test_v633_filter_presets -- the preset stage and the tip-panel fixes.

Presets are a FRONTEND feature by design: selecting one loads it over
GET /uls/filter/preset and writes the values onto the widgets (the
serialized widget values stay the single truth the run reads); saving POSTs
the current look. apply() deliberately ignores the preset selector --
honoring it in the backend too would double-apply the look. That makes the
sanitizer the load-bearing piece: EVERY byte that moves between disk and
widgets passes _sanitize_preset, so a hand-edited or downloaded preset can
neither smuggle unknown keys nor paths nor non-numeric junk.

  DRIVEN -- _sanitize_preset on a hostile dict: unknown keys dropped,
        numeric strings coerced to float, non-coercible values DROPPED (not
        guessed), lut_name reduced to a basename, non-dict input -> {}.
  STATIC -- GET+POST routes registered; the POST stem is regex-reduced; the
        GET path sanitizes through _sanitize_preset; the backend ignore is
        documented at the apply() site; JS: Save button + POST + load fetch
        + combo options push; the ignored selector never reaches
        _pfApplyParams. Tip placement after two rounds of screen feedback:
        NO hover machinery, NO tip panel, NO info glyph -- the scrub hint is
        painted in the title next to Reset, the full sentence opens the node
        DESCRIPTION (native tooltip) and every FLOAT field carries it in its
        own tooltip; the circled-info sign sits UNCOLORED inline at the
        start of the telegram scrub sentence, a full-width block right
        above the capitalized divider hint;
        fonts stay explicitly non-italic.

  Mutations: whitelist wound (unknown keys survive) -> driven breaks;
        basename wound (lut_name keeps its path) -> driven breaks; coercion
        wound (junk becomes 0.0 instead of dropped) -> driven breaks.
"""

# v372 (public build): the Filter routes live in their own module,
# nodes/ph_filter_routes.py -- uls_routes.py stays the Stack's file. Same
# source text, different path; the checks below are unchanged.
import os
import sys

import numpy as np  # noqa: F401  (lift environment parity with the node file)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_SRC = open(os.path.join(ROOT, "nodes", "ph_filter.py"), encoding="utf-8").read()
JS_SRC = open(os.path.join(ROOT, "web", "js", "ph_filter.js"), encoding="utf-8").read()
ROUTES_SRC = open(os.path.join(ROOT, "nodes", "ph_filter_routes.py"), encoding="utf-8").read()


def _fail(msg):
    print("[test_v633_filter_presets] FAIL: " + msg)
    sys.exit(1)


def _lift_pyfunc(src, name):
    out, inside = [], False
    for ln in src.splitlines(keepends=True):
        if not inside:
            if ln.startswith("def " + name + "("):
                inside = True
                out.append(ln)
        else:
            if ln.strip() == "" or ln.startswith((" ", "\t")):
                out.append(ln)
            else:
                break
    if not out:
        _fail("could not lift python function %s" % name)
    return "".join(out)


# ---------------------------------------------------------------------------
# STATIC: routes
# ---------------------------------------------------------------------------
for needle, msg in [
    ('("GET",  "/uls/filter/preset",  handle_filter_preset_get)', "preset GET route missing"),
    ('("POST", "/uls/filter/preset",  handle_filter_preset_post)', "preset POST route missing"),
    ('re.sub(r"[^A-Za-z0-9_\\-]+", "_"', "POST stem must be regex-reduced to a safe filename"),
    ("from .ph_filter import _sanitize_preset", "routes must sanitize through the node's own whitelist"),
]:
    if needle not in ROUTES_SRC:
        _fail(msg)

# ---------------------------------------------------------------------------
# STATIC: js preset machinery + tip-panel fixes
# ---------------------------------------------------------------------------
for needle, msg in [
    ('saveBtn.textContent = "Save preset"', "Save-preset button missing from the pane header"),
    ('method: "POST"', "preset save must POST"),
    ('"/uls/filter/preset?name="', "preset load fetch missing"),
    ("w.options.values.push(data.file)", "saved preset must become selectable immediately"),
    ("function _pfApplyParams(", "shared preset apply missing"),
]:
    if needle not in JS_SRC:
        _fail(msg)
if "onMouseMove" in JS_SRC or "onMouseLeave" in JS_SRC:
    _fail("hover machinery must be gone: the native node tooltip and the tip "
          "panel stacked on screen -- the panel is click-pinned only")
if "_pfInfoOn" in JS_SRC:
    _fail("stale hover state _pfInfoOn still present")
if "italic" in JS_SRC:
    _fail("fonts must be explicitly non-italic (screen finding)")
if "Every numeric field doubles as a slider" not in PY_SRC:
    _fail("the fields-are-sliders sentence must open the node DESCRIPTION "
          "(shows in the native tooltip -- the one surface proven on screen)")
if PY_SRC.count("Click-drag to scrub live.") != 13:
    _fail("every FLOAT control must carry the scrub sentence in its own "
          "field tooltip (expected 13)")
for needle, msg in [
    ('"\\u{1F6C8} Click-drag a value: scrubs it live. Click once: type it."',
     "the info sign must sit UNCOLORED inline at the start of the scrub sentence"),
    ("Drag the divider \\u00b7 hold A/B for the original.",
     "divider hint must be capitalized and punctuated"),
]:
    if needle not in JS_SRC:
        _fail(msg)
if "infoGlyph" in JS_SRC:
    _fail("standalone info-glyph element must be gone -- the sign lives inline "
          "in the sentence (uncolored, final cut on screen feedback)")

# apply() must document that the preset selector is ignored on purpose
if "deliberately NOT applied here" not in PY_SRC:
    _fail("the backend ignore of the preset selector must be documented at apply()")

# ---------------------------------------------------------------------------
# DRIVEN: sanitizer on hostile input
# ---------------------------------------------------------------------------
lifted = _lift_pyfunc(PY_SRC, "_sanitize_preset")
PREAMBLE = ("import os\nimport numpy as np\n"
            "FILTER_CANON = %r\nPRESET_KEYS = tuple(k for k in FILTER_CANON if k != \"preset\")\n")
CANON = ("exposure", "temperature", "tint", "contrast", "gamma", "shadows",
         "highlights", "saturation", "vibrance", "hue_shift", "lut_name",
         "lut_strength", "sharpen_amount", "sharpen_radius", "preset")

HOSTILE = {
    "exposure": "0.5",                       # numeric string -> float
    "gamma": "abc",                          # junk -> DROPPED
    "evil_key": 1.0,                         # unknown -> DROPPED
    "preset": "self.json",                   # the selector itself -> DROPPED
    "lut_name": "../../secrets/x.cube",      # path -> basename
    "vibrance": None,                        # None -> DROPPED
    "contrast": 0.25,
}
EXPECT = {"exposure": 0.5, "lut_name": "x.cube", "contrast": 0.25}


def _drive(src):
    env = {}
    exec(PREAMBLE % (CANON,) + src, env)
    f = env["_sanitize_preset"]
    if f(None) != {} or f([1, 2]) != {}:
        return False
    return f(dict(HOSTILE)) == EXPECT


if not _drive(lifted):
    _fail("sanitizer drove wrong values on the hostile fixture")

# ---------------------------------------------------------------------------
# MUTATIONS
# ---------------------------------------------------------------------------
mut = lifted.replace("        if k not in PRESET_KEYS:\n            continue\n", "")
if mut == lifted:
    _fail("whitelist mutation did not apply")
if _drive(mut):
    _fail("MUTATION NOT CAUGHT: whitelist wound survived (unknown keys leaked)")

mut = lifted.replace("os.path.basename(str(v))", "str(v)")
if mut == lifted:
    _fail("basename mutation did not apply")
if _drive(mut):
    _fail("MUTATION NOT CAUGHT: basename wound survived (paths leaked)")

mut = lifted.replace(
    "        except (TypeError, ValueError):\n            continue\n",
    "        except (TypeError, ValueError):\n            out[k] = 0.0\n            continue\n")
if mut == lifted:
    _fail("coercion mutation did not apply")
if _drive(mut):
    _fail("MUTATION NOT CAUGHT: coercion wound survived (junk became 0.0)")

print("[test_v633_filter_presets] PASS: sanitizer drove the hostile fixture, "
      "routes + save/load + tip-panel fixes pinned, 3 mutations caught")
