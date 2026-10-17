#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v898 -- the latent-type mirror, and the pill that says what governs.

THE WOUND (Frank, 28.08.: "aktuell kann man nur Sekunden eintragen ... ich
brauche hier auf jeden Fall einen Frame: 1"). The frame field was never
missing: `length` has sat at canon position 3 since the node was built, min 1,
and the backend reads it. It had gone INVISIBLE.

web/js/ph_empty_latent.js keeps its own copy of the type list, declared in its
own header as a mirror of uls_latent_math. Python carried NINE labels, the
mirror carried EIGHT -- "MiniMax H3 AV (video+audio)" was never added. So:

    TYPE_GEOM[type]  -> undefined
    VIDEO_TYPES      -> does not contain the type
    isVideo          -> false
    _setDisabled(length, true)  -> and since v888 that HIDES the row

One missing string, and the only field that can name a frame count vanished in
the one mode that needed it. Nothing was broken; something was simply not
mirrored. That is the failure mode this house has a rule for -- two places that
compute the same thing drift -- and the rule says the guard drives BOTH and
compares them, which is what M1/M2 do below.

THE SECOND FINDING, which the pill exists to speak: a single frame is NOT
available for this latent type, and never was. Core's own arithmetic
(comfy_extras/nodes_minimax_h3.py, read at the source) is

    frame_count = align_frame_count(max(5, length))     # while n % 17 != 5: n++

so five is the model's floor. length=1 comes back as 5. Until v898 that
happened silently on the node and was mentioned in the console under the wrong
name ("snapped onto the 17k+5 grid" -- the grid is not what moved a 1).

WHAT IS PINNED

  M1  the label lists agree, element for element AND in order.
  M2  every Python type has a geometry entry in the mirror, with the same
      spatial divisor and the same video flag.
  M3  MiniMax counts as a video type in the mirror -- the precise condition
      that hid the field.
  M4  the JS frame grid equals the Python one across a range, including the
      floor, the exact grid points and the values just under them.
  M5  the pill, RUN in node against the real file: which field governs, what
      the number becomes, and that the reason it gives is the right one --
      floor below five, grid above.
  M6  the pill never enters widgets_values (serialize:false) and is silent for
      every non-H3 type. A readout that renumbers a saved graph would be a
      far worse bug than the one it reports.
  M7  the backend names the right reason too, so console and pill cannot
      disagree.

Script-style: exit 0 = pass.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

NAME = "v898"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "nodes"))

JS = os.path.join(ROOT, "web", "js", "ph_empty_latent.js")
PY = os.path.join(ROOT, "nodes", "ph_empty_latent.py")

_fails = []
_checks = 0


def _need(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        _fails.append(msg)
        print("[%s] FAIL -- %s" % (NAME, msg))


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _lift(src, sig):
    """One top-level `function name(...) { ... }`, by brace balance."""
    i = src.index(sig)
    j = src.index("{", i)
    depth, k = 0, j
    while k < len(src):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
        k += 1
    raise ValueError("unbalanced: " + sig)


def _lift_const(src, name):
    m = re.search(r"const\s+" + re.escape(name) + r"\s*=\s*(\[.*?\]|\{.*?\n\}|[^;\n]+);",
                  src, re.S)
    if not m:
        raise ValueError("const not found: " + name)
    return "const %s = %s;" % (name, m.group(1))


def _node(js):
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(js)
        path = fh.name
    try:
        p = subprocess.run(["node", path], capture_output=True, text=True,
                           timeout=90)
        return p.returncode, p.stdout, p.stderr
    finally:
        os.unlink(path)


def main():
    import uls_latent_math as M
    js = _read(JS)
    py = _read(PY)

    # ---------------- M1: the label lists ---------------------------------
    m = re.search(r"const LATENT_TYPES = \[(.*?)\];", js, re.S)
    _need(m is not None, "M1: LATENT_TYPES not found in the mirror")
    js_labels = re.findall(r'"([^"]+)"', m.group(1)) if m else []
    py_labels = list(M.LATENT_TYPE_LABELS)
    _need(js_labels == py_labels,
          "M1: the type mirror has drifted.\n      python: %r\n      js    : %r\n"
          "      only in python: %r\n      only in js    : %r"
          % (py_labels, js_labels,
             [x for x in py_labels if x not in js_labels],
             [x for x in js_labels if x not in py_labels]))

    # ---------------- M2/M3: the geometry mirror --------------------------
    harness = "\n".join([
        _lift_const(js, "LATENT_TYPES"),
        _lift_const(js, "TYPE_GEOM"),
        'const VIDEO_TYPES = LATENT_TYPES.filter((t) => TYPE_GEOM[t] && TYPE_GEOM[t].video);',
        'process.stdout.write(JSON.stringify({geom: TYPE_GEOM, video: VIDEO_TYPES}));',
    ])
    rc, out, err = _node(harness)
    _need(rc == 0 and out.strip(),
          "M2: could not evaluate the mirror in node: %s" % (err or out)[-300:])
    if rc == 0 and out.strip():
        data = json.loads(out)
        geom, video = data["geom"], data["video"]
        for label in py_labels:
            key = M.LABEL_TO_KEY.get(label)
            spec = M.TYPE_SPEC.get(key)
            g = geom.get(label)
            _need(g is not None,
                  "M2: %r has no geometry in the mirror -- this is exactly how "
                  "the field disappeared" % label)
            if g is None or spec is None:
                continue
            _need(g["sdiv"] == spec[1],
                  "M2: %r spatial divisor: python %s, mirror %s"
                  % (label, spec[1], g["sdiv"]))
            _need(bool(g["video"]) == bool(spec[2]),
                  "M2: %r video flag: python %s, mirror %s"
                  % (label, spec[2], g["video"]))
        mm = "MiniMax H3 AV (video+audio)"
        _need(mm in video,
              "M3: %r must count as a video type -- otherwise `length` is "
              "hidden and there is no way to name a frame count" % mm)

    # ---------------- M4: the frame grid, both implementations ------------
    grid_js = "\n".join([
        "const MINIMAX_FPS = 24;",
        _lift(js, "function minimaxAlign"),
        _lift(js, "function minimaxFramesFromSeconds"),
        "const out = {align: {}, secs: {}};",
        "for (const n of %s) out.align[n] = minimaxAlign(n);"
        % json.dumps([0, 1, 2, 4, 5, 6, 17, 21, 22, 23, 38, 39, 40, 80, 81, 90, 200]),
        "for (const s of %s) out.secs[s] = minimaxFramesFromSeconds(s);"
        % json.dumps([0, 0.1, 1, 1.5, 3.375, 5, 10]),
        "process.stdout.write(JSON.stringify(out));",
    ])
    rc, out, err = _node(grid_js)
    _need(rc == 0 and out.strip(),
          "M4: the JS grid did not run: %s" % (err or out)[-300:])
    if rc == 0 and out.strip():
        got = json.loads(out)
        for k, v in got["align"].items():
            want = M.minimax_align_frames(int(k))
            _need(v == want,
                  "M4: align(%s): python %s, js %s" % (k, want, v))
        for k, v in got["secs"].items():
            want = M.minimax_frames_from_seconds(float(k))
            _need(v == want,
                  "M4: from_seconds(%s): python %s, js %s" % (k, want, v))
        _need(got["align"]["1"] == 5,
              "M4: minimaxAlign must keep MIRRORING Core -- align(1) is 5 there. "
              "v899 does not change this function; it routes ONE frame past it "
              "entirely (image mode), which is a different thing and is pinned "
              "in test_v899")

    # ---------------- M5/M6: the pill, driven -----------------------------
    pill_js = "\n".join([
        _lift_const(js, "LATENT_TYPES"),
        _lift_const(js, "TYPE_GEOM"),
        'const MINIMAX_TYPE = "MiniMax H3 AV (video+audio)";',
        "const MINIMAX_FPS = 24;",
        _lift(js, "function minimaxAlign"),
        _lift(js, "function minimaxFramesFromSeconds"),
        _lift(js, "function _findWidget"),
        _lift(js, "function _framePill"),
        # the widget's own constants, lifted rather than restated -- a bench
        # that retypes a value is testing its own copy of it
        _lift_const(js, "PILL_NAME"),
        _lift_const(js, "PILL_TYPE"),
        _lift_const(js, "PILL_H"),
        _lift(js, "function makePillWidget"),
        r"""
function mk(type, length, secs) {
    return { widgets: [
        { name: "latent_type", value: type },
        { name: "length", value: length },
        { name: "duration_seconds", value: secs },
    ] };
}
const out = {};
out.byLength   = _framePill(mk(MINIMAX_TYPE, 81, 0));
out.one        = _framePill(mk(MINIMAX_TYPE, 1, 0));
out.four       = _framePill(mk(MINIMAX_TYPE, 4, 0));
out.six        = _framePill(mk(MINIMAX_TYPE, 6, 0));
out.bySeconds  = _framePill(mk(MINIMAX_TYPE, 81, 3.375));
out.image      = _framePill(mk("Image", 81, 0));
out.wan        = _framePill(mk("WAN video", 81, 0));

// the widget itself
const w = makePillWidget();
out.serialize = w.serialize;
out.name = w.name;
const nH3 = mk(MINIMAX_TYPE, 81, 0), nImg = mk("Image", 81, 0);
w._node = nH3;  out.hH3 = w.computeSize(400)[1];
w._node = nImg; out.hImg = w.computeSize(400)[1];
// draw must survive a hostile context
let threw = false;
try {
    w._node = nH3;
    w.draw({ save(){}, restore(){}, measureText(){ throw new Error("boom"); },
             fillText(){}, set font(v){}, set fillStyle(v){},
             set textBaseline(v){}, set textAlign(v){} }, nH3, 400, 0);
} catch (e) { threw = true; }
out.drawThrew = threw;
process.stdout.write(JSON.stringify(out));
""",
    ])
    rc, out, err = _node(pill_js)
    _need(rc == 0 and out.strip(),
          "M5: the pill harness did not run: %s" % (err or out)[-400:])
    if rc == 0 and out.strip():
        o = json.loads(out)
        _need(o["byLength"] and o["byLength"]["bySeconds"] is False
              and o["byLength"]["frames"] == 90,
              "M5: with duration 0 the LENGTH field must govern (81 -> 90), "
              "got %r" % (o["byLength"],))
        _need(o["bySeconds"] and o["bySeconds"]["bySeconds"] is True
              and o["bySeconds"]["frames"] == 90,
              "M5: duration above 0 must win over length, got %r"
              % (o["bySeconds"],))
        # v899 re-grounded: ONE frame is a still, so the pill no longer reports
        # five here. The floor claim moved to values 2..4, where it is true and
        # where it belongs -- see test_v899 for the image branch itself.
        _need(o["four"] and o["four"]["frames"] == 5
              and "floor" in (o["four"]["note"] or ""),
              "M5: values 2..4 are the video floor, not the grid, got %r"
              % (o["four"],))
        _need(o["six"] and o["six"]["frames"] == 22
              and "grid" in (o["six"]["note"] or ""),
              "M5: above the floor the reason is the grid (6 -> 22), got %r"
              % (o["six"],))
        _need(o["byLength"]["seconds"] == 3.75,
              "M5: the pill must state the resulting duration (90/24), got %r"
              % o["byLength"]["seconds"])
        _need(o["image"] is None and o["wan"] is None,
              "M6: the pill must stay silent where the two-field question does "
              "not arise, got image=%r wan=%r" % (o["image"], o["wan"]))
        _need(o["serialize"] is False,
              "M6: the pill MUST carry serialize:false -- a readout that lands "
              "in widgets_values would renumber every saved graph (#577)")
        _need(o["hH3"] > 0 and o["hImg"] == 0,
              "M6: the pill must claim height only where it speaks (h3=%r, "
              "image=%r)" % (o["hH3"], o["hImg"]))
        _need(o["drawThrew"] is False,
              "M6: a readout must never break the canvas")

    # ---------------- M7: the backend names the same reason ---------------
    # v899 re-grounded these three. The promise was never "say the words
    # 'model floor'" -- it was "console and tooltip must tell the user the same
    # rule the pill shows". v898 stated that rule wrongly (the five is Core's
    # VIDEO node, not the model), so the wording changed and the promise did
    # not. What must hold: the two reasons stay told apart BY THE VALUE, and
    # both the console and the tooltip name the way out.
    _need("frames_in < 5" in py,
          "M7: the two reasons must be told apart by the value, not by wording")
    _need("floor of 5" in py and "length=1 for a still" in py,
          "M7: the console must name the video floor AND the way past it")
    _need("length=1 builds a true single-frame IMAGE latent" in py,
          "M7: the length tooltip must say what length 1 does, so the limit "
          "and its exit are readable before a run rather than after it")
    _need("model has a floor of FIVE frames" not in py,
          "M7: the v898 claim that FIVE is the MODEL's floor must not return -- "
          "it is the floor of Core's audio+video node, measured in v899")

    if _fails:
        print("\n[%s] %d/%d FAILED" % (NAME, len(_fails), _checks))
        return 1
    print("[%s] OK -- the type mirror agrees with Python and the pill tells "
          "the truth (%d checks)" % (NAME, _checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
