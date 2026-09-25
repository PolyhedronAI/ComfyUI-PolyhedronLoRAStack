#!/usr/bin/env python3
"""
test_v999_filter_auto -- the automatic (F2): one answer per batch, the same
arithmetic in both runtimes, and the two corrections to the Viewer's version.

  P1  GRADE PARITY -- _auto_grade (python) and _autoGrade (JS) over a stats
      fixture x all four stops: every value equal within 1e-9; off -> null.
  P2  PIXEL PARITY -- _apply_auto_np and _autoRGB over the 11^3 RGB grid with
      a strong correction: max |diff| < 1e-5.
  P3  ONE-SIDED -- a contrasty picture is never flattened, a vivid one never
      desaturated; a flat / dull one is lifted.
  P4  THE VIEWER'S SATURATION FAULT STAYS OUT -- a picture of saturated but
      balanced colours (mean colour grey) gets NO saturation boost at ultra.
  P5  ONE ANSWER PER BATCH -- the real ULSFilter.apply on a flickering batch:
      every frame gets the batch's correction (not its own), so frame 0 in
      the batch differs from frame 0 run alone; off -> the input returned
      untouched (fast path).
  P6  SAMPLING -- 100 frames -> 8 evenly spread indices incl. first and last.
  P7  PICKER AFTER THE AUTOMATIC -- the JS pick taken on _autoRGB(rgb) and
      fed through python _apply_auto_np + _grade_np comes out grey (1e-5).
  P8  WIRING -- ui item carries "auto"; sanitizer keeps a known stop, drops
      junk; CANON_DEFAULTS auto_mode "off"; auto_mode is a live widget; the
      badge text function is driven.
  P9  MUTATIONS -- three wounds, each must go red.
"""
import contextlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_PATH = os.path.join(ROOT, "nodes", "ph_filter.py")
JS_SRC = open(os.path.join(ROOT, "web", "js", "ph_filter.js"), encoding="utf-8").read()
PY_SRC = open(PY_PATH, encoding="utf-8").read()

FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)


def load_py(src):
    spec = importlib.util.spec_from_loader("phf_probe", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = PY_PATH
    exec(compile(src, PY_PATH, "exec"), mod.__dict__)
    return mod


def lift_js(src, name):
    i = src.find("function " + name + "(")
    j = src.find("\n}", i)
    if i < 0 or j < 0:
        raise SystemExit("[test_v999_filter_auto] FAIL: cannot lift " + name)
    return src[i:j + 2]


def js_block(src, start, end):
    i = src.find(start)
    j = src.find(end, i)
    return src[i:j + len(end)]


def run_js(src, body):
    consts = js_block(src, "const AUTO_FACTORS = {", "};") + "\n" + "\n".join(
        ln for ln in src.splitlines() if re.match(r"const AUTO_TARGET_\w+ = ", ln)) + "\n" + "\n".join(
        ln for ln in src.splitlines()
        if re.match(r"const (PICK_RADIUS|PICK_DARKEST|PICK_BLOWN|WB_LIMIT)\s*=", ln))
    code = consts + "\n" + "\n".join(lift_js(src, n) for n in (
        "_autoGrade", "_autoRGB", "_autoText", "_pickWhite")) + "\n" + body
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout)
    finally:
        os.unlink(path)


MODES = ("off", "neutral", "high", "ultra")
STATS = [
    [0.52, 0.43, 0.37, 0.43, 0.30, 0.026],   # warm, contrasty, dull (an H3 frame)
    [0.30, 0.34, 0.45, 0.25, 0.10, 0.020],   # cold, dark, flat, dull
    [0.50, 0.50, 0.50, 0.70, 0.12, 0.080],   # grey mean, bright, flat, vivid
    [0.005, 0.40, 0.40, 0.40, 0.005, 0.001],  # degenerate: dead red, no spread
]
GRID = [[r / 10, g / 10, b / 10] for r in range(11) for g in range(11) for b in range(11)]
STRONG = {"wb_r": 1.3, "wb_b": 0.8, "brightness": 0.06, "contrast": 1.4, "saturation": 1.5}
PICKS = [[0.52, 0.43, 0.37], [0.35, 0.40, 0.55], [0.6, 0.5, 0.45]]

BODY = """
const S = %s, M = %s, G = %s, A = %s, PK = %s;
const grades = S.map(s => M.map(m => _autoGrade(s, m)));
const px = G.map(([r, g, b]) => _autoRGB(r, g, b, A));
const au = _autoGrade(S[0], "ultra");
const picks = PK.map(([r, g, b]) => { const s = _autoRGB(r, g, b, au); return _pickWhite(s[0], s[1], s[2]); });
const text = _autoText("high", {wb_r: 1.04, wb_b: 0.93, brightness: -0.012, contrast: 1, saturation: 1.2});
console.log(JSON.stringify({grades, px, picks, text}));
""" % (json.dumps(STATS), json.dumps(MODES), json.dumps(GRID), json.dumps(STRONG), json.dumps(PICKS))


def evaluate(py_src, js_src):
    errs = []
    m = load_py(py_src)
    res = run_js(js_src, BODY)
    if res is None:
        return ["node driver failed"]
    # P1
    for si, s in enumerate(STATS):
        for mi, mode in enumerate(MODES):
            p, j = m._auto_grade(s, mode), res["grades"][si][mi]
            if (p is None) != (j is None):
                errs.append("P1 stats %d %s: py %r js %r" % (si, mode, p, j))
            elif p is not None and max(abs(p[k] - j[k]) for k in p) > 1e-9:
                errs.append("P1 stats %d %s differ: py %r js %r" % (si, mode, p, j))
    check_off = [m._auto_grade(s, "off") for s in STATS] + [m._auto_grade(None, "ultra"),
                                                           m._auto_grade(STATS[0], "bogus")]
    if any(v is not None for v in check_off):
        errs.append("P1 off / no stats / unknown stop must change nothing")
    # P2
    py_px = m._apply_auto_np(np.asarray(GRID, np.float32), STRONG).astype(np.float64)
    d = float(np.max(np.abs(py_px - np.asarray(res["px"]))))
    if d > 1e-5:
        errs.append("P2 pixel parity %.2e > 1e-5" % d)
    # P3
    contrasty = m._auto_grade([0.5, 0.5, 0.5, 0.45, 0.30, 0.08], "ultra")
    flat = m._auto_grade([0.5, 0.5, 0.5, 0.45, 0.10, 0.02], "ultra")
    if not (contrasty["contrast"] == 1.0 and contrasty["saturation"] == 1.0):
        errs.append("P3 a contrasty / vivid picture was flattened or desaturated: %r" % contrasty)
    if not (flat["contrast"] > 1.5 and flat["saturation"] > 1.2):
        errs.append("P3 a flat / dull picture was not lifted: %r" % flat)
    # P4: half pure red, half pure cyan-ish -> mean colour grey, per pixel vivid
    img = np.zeros((64, 64, 3), np.float32)
    img[:, :32] = [0.8, 0.2, 0.2]
    img[:, 32:] = [0.2, 0.8, 0.8]
    st = m._auto_stats(img)
    g4 = m._auto_grade(st, "ultra")
    if not (st[5] > 0.2 and g4["saturation"] == 1.0):
        errs.append("P4 a balanced vivid picture got a saturation boost: stats %r grade %r" % (st, g4))
    # P5
    base = np.asarray(GRID[:1331:7], np.float32)
    frames = np.stack([np.tile(base[None], (8, 1, 1)) * f for f in (0.6, 0.8, 1.0, 1.2)]).clip(0, 1)
    node = m.ULSFilter()
    kw = dict(exposure=0.0, temperature=0.0, tint=0.0, contrast=0.0, gamma=1.0,
              shadows=0.0, highlights=0.0, saturation=0.0, vibrance=0.0, hue_shift=0.0,
              lut_name="none", lut_strength=1.0, sharpen_amount=0.0, sharpen_radius=1.0,
              preset="none")
    out = node.apply(frames, auto_mode="high", **kw)
    got = np.asarray(out["result"][0])
    a_batch = m._auto_grade(m._auto_stats(frames), "high")
    want = m._grade_np(m._apply_auto_np(frames[0], a_batch), 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
    alone = np.asarray(node.apply(frames[:1], auto_mode="high", **kw)["result"][0])[0]
    if float(np.max(np.abs(got[0] - want))) > 1e-6:
        errs.append("P5 frame 0 did not get the batch's correction")
    if float(np.max(np.abs(got[0] - alone))) < 1e-3:
        errs.append("P5 frame 0 in the batch equals frame 0 alone -- per-frame, not per-batch")
    if "auto" not in out["ui"]["ph_filter"][0] or len(out["ui"]["ph_filter"][0]["auto"]) != 6:
        errs.append("P8 the ui item does not carry the six auto readings")
    off = node.apply(frames, auto_mode="off", **kw)["result"][0]
    if off is not frames:
        errs.append("P5 auto off with neutral sliders must return the input untouched")
    # P6
    big = np.zeros((100, 2, 2, 3), np.float32)
    big[np.arange(100), :, :, 0] = (np.arange(100) / 99.0)[:, None, None]
    s6 = m._auto_stats(big)
    if abs(s6[0] - np.mean([round(i * 99 / 7) / 99.0 for i in range(8)])) > 1e-6:
        errs.append("P6 frame sampling is not the 8 evenly spread frames")
    # P7
    au = m._auto_grade(STATS[0], "ultra")
    for rgb, pk in zip(PICKS, res["picks"]):
        if "reason" in pk:
            errs.append("P7 pick refused: %r" % pk)
            continue
        t, n = pk["exact"]
        o = m._grade_np(m._apply_auto_np(np.asarray([rgb], np.float32), au), 0.1, t, n, 0.1, 1.1,
                        0.1, 0, 0.2, 0.1, 15)[0]
        if float(o.max() - o.min()) > 1e-5:
            errs.append("P7 pick after the automatic not grey: %r" % (o,))
    # P8
    if m._sanitize_preset({"auto_mode": "high"}) != {"auto_mode": "high"} \
            or m._sanitize_preset({"auto_mode": "turbo"}) != {} \
            or m._sanitize_preset({"auto_mode": 3}) != {}:
        errs.append("P8 sanitizer: auto_mode must keep a known stop and drop the rest")
    # RE-GROUNDED v1000: sharpen_threshold appended behind it -- auto_mode
    # keeps its slot 15 (index), which is what the positional law needs
    if tuple(m.AUTO_MODES) != MODES or m.FILTER_CANON[15] != "auto_mode":
        errs.append("P8 AUTO_MODES / canon tail")
    if res["text"] != "auto high: wb 1.04/0.93 \u00b7 light -0.01 \u00b7 contrast 1.00 \u00b7 sat 1.20":
        errs.append("P8 badge text: %r" % res["text"])
    return errs


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a)


FAILS.extend(quiet(evaluate, PY_SRC, JS_SRC))
check('preset: "none", auto_mode: "off",' in JS_SRC, "P8 CANON_DEFAULTS must carry auto_mode off")
check(re.search(r'const LIVE_WIDGETS = GRADE_WIDGETS\.concat\(\[[^\]]*"auto_mode"', JS_SRC) is not None,
      "P8 auto_mode must be a live preview widget")
check("const s0 = rgb ? _autoRGB(rgb[0], rgb[1], rgb[2], _pfAuto(node).grade) : null;" in JS_SRC,
      "P8 the picker must pick on what the white balance receives")
# RE-GROUNDED v1001: the automatic now reads the stage -1 buffer b0 (the
# detail from the original runs before it); the order it pins is unchanged
check("const s0 = _autoRGB(b0[f], b0[f + 1], b0[f + 2], auto);\n"
      "        let rgb = _gradeRGB(s0[0], s0[1], s0[2], p);" in JS_SRC,
      "P8 the live preview must run the automatic before the sliders (static: needs a canvas)")
check("node._pfAutoStats = Array.isArray(item.auto) ? item.auto : null;" in JS_SRC,
      "P8 the preview must take the readings from the ui item")

MUT = [
    # one-sidedness is held twice (the condition AND the clamp floor 1.0):
    # the wound has to take out both, or it is no wound
    ("JS contrast flattens", "js",
     "if (spread > 0.01 && spread < AUTO_TARGET_SPREAD) {\n        contrast = lim(1 + (AUTO_TARGET_SPREAD / spread - 1) * f.spread, 1.0, 2.0);",
     "if (spread > 0.01) {\n        contrast = lim(1 + (AUTO_TARGET_SPREAD / spread - 1) * f.spread, 0.5, 2.0);"),
    ("PY chroma from the mean colour", "py",
     "chroma = np.abs(s - s.mean(axis=1, keepdims=True)).mean()",
     "chroma = float(np.abs(s.mean(axis=0) - s.mean()).mean())"),
    ("JS brightness before the gains", "js",
     "    r *= a.wb_r; b *= a.wb_b;\n    r += a.brightness; g += a.brightness; b += a.brightness;",
     "    r += a.brightness; g += a.brightness; b += a.brightness;\n    r *= a.wb_r; b *= a.wb_b;"),
]
caught = 0
for name, side, a, b in MUT:
    src = JS_SRC if side == "js" else PY_SRC
    if src.count(a) != 1:
        FAILS.append("P9 mutation anchor not unique: " + name)
        continue
    if side == "js":
        r = quiet(evaluate, PY_SRC, JS_SRC.replace(a, b))
    else:
        r = quiet(evaluate, PY_SRC.replace(a, b), JS_SRC)
    if r:
        caught += 1
    else:
        FAILS.append("P9 mutation survived: " + name)

if FAILS:
    for f in FAILS:
        print("[test_v999_filter_auto] FAIL: " + f)
    sys.exit(1)
print("[test_v999_filter_auto] OK -- 9 promises, grade + pixel parity, one answer per batch, "
      "%d/3 mutations caught" % caught)
