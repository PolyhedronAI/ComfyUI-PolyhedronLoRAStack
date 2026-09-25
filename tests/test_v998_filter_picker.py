#!/usr/bin/env python3
"""
test_v998_filter_picker -- the white picker (F1), measured through BOTH
runtimes.

The picker lives in web/js/ph_filter.js and only WRITES the existing
temperature/tint widgets. Its whole promise is "the spot I clicked comes
out neutral in the RUN". So this guard drives the real JS _pickWhite and
feeds its answer into the real python ground truth _grade_np, with every
other colour control off-neutral, and measures the grey:

  P1  NEUTRAL IN THE RUN -- over a grid of colour casts inside the range,
      the picked colour graded by _grade_np has max(rgb)-min(rgb) < TOL.
  P2  REFUSALS -- too dark, blown out, no picture: a reason, no values.
  P3  CLAMP -- a cast beyond the range is clamped to +-WB_LIMIT and says so.
  P4  _patchMean -- exact mean of a known fixture, edge patch respected.
  P5  _proxyPoint -- the letterbox: border -> null, corners map in.
  P6  WIRING -- python min/max == +-WB_LIMIT; canon unchanged (15 keys);
      the pick writes ONLY temperature/tint through _pfApplyParams; no
      document-level listener; the button exists.
  P7  MUTATIONS -- three wounds injected into the JS, each must go red.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_SRC = open(os.path.join(ROOT, "nodes", "ph_filter.py"), encoding="utf-8").read()
JS_SRC = open(os.path.join(ROOT, "web", "js", "ph_filter.js"), encoding="utf-8").read()

TOL_EXACT = 1e-5  # the arithmetic itself: float32 noise only
TOL_GREY = 2.0 / 255  # the widget values step in 0.01; measured worst 4.5e-3 with every
                      # other control pushed hard -- held under two 8-bit levels
OTHERS = {"exposure": 0.2, "contrast": 0.15, "gamma": 1.2, "shadows": 0.2,
          "highlights": -0.15, "saturation": 0.4, "vibrance": 0.3, "hue_shift": 30.0}
ORDER = ("exposure", "temperature", "tint", "contrast", "gamma",
         "shadows", "highlights", "saturation", "vibrance", "hue_shift")

FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)


def lift_py(src, name):
    out, inside = [], False
    for ln in src.splitlines(keepends=True):
        if not inside:
            if ln.startswith("def " + name + "("):
                inside = True
                out.append(ln)
        elif ln.strip() == "" or ln.startswith((" ", "\t")):
            out.append(ln)
        else:
            break
    return "".join(out)


def lift_js(src, name):
    i = src.find("function " + name + "(")
    j = src.find("\n}", i)
    if i < 0 or j < 0:
        raise SystemExit("[test_v998_filter_picker] FAIL: cannot lift " + name)
    return src[i:j + 2]


def js_consts(src):
    return "\n".join(ln for ln in src.splitlines()
                     if re.match(r"const (PICK_RADIUS|PICK_DARKEST|PICK_BLOWN|WB_LIMIT)\s*=", ln))


def run_js(src, body):
    code = js_consts(src) + "\n" + "\n".join(
        lift_js(src, n) for n in ("_fitRect", "_pickWhite", "_patchMean", "_proxyPoint")) + "\n" + body
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


def casts():
    vals = [0.2, 0.35, 0.5, 0.65, 0.8]
    return [(r, g, b) for r in vals for g in vals for b in vals]


BODY = """
const C = %s;
const picks = C.map(([r, g, b]) => _pickWhite(r, g, b));
const refuse = [_pickWhite(0.02, 0.5, 0.5), _pickWhite(0.5, 0.999, 0.5),
                _pickWhite(NaN, 0.5, 0.5), _pickWhite(-1, 0.5, 0.5)];
const clamp = _pickWhite(0.9, 0.5, 0.1);
const W = 4, H = 3, D = new Uint8ClampedArray(W * H * 4);
for (let i = 0; i < W * H; i++) { D[4*i] = i * 10; D[4*i+1] = 100; D[4*i+2] = 255 - i; D[4*i+3] = 255; }
const pm_mid = _patchMean(D, W, H, 1, 1, 1);
const pm_edge = _patchMean(D, W, H, 0, 0, 1);
const pp = [_proxyPoint(0, 50, 100, 50, 200, 200), _proxyPoint(0, 51, 100, 50, 200, 200),
            _proxyPoint(199, 148, 100, 50, 200, 200), _proxyPoint(100, 30, 100, 50, 200, 200),
            _proxyPoint(49, 10, 50, 100, 200, 200), _proxyPoint(50, 0, 50, 100, 200, 200),
            _proxyPoint(149, 199, 50, 100, 200, 200)];
console.log(JSON.stringify({picks, refuse, clamp, pm_mid, pm_edge, pp}));
""" % json.dumps(casts())


def grey_error(picks, exact=False):
    env = {"np": np}
    exec(lift_py(PY_SRC, "_grade_np"), env)
    worst = 0.0
    used = 0
    for (r, g, b), p in zip(casts(), picks):
        if "reason" in p or p.get("clamped"):
            continue
        t, n = p["exact"] if exact else (p["temperature"], p["tint"])
        params = dict(OTHERS, temperature=t, tint=n)
        out = env["_grade_np"](np.asarray([[r, g, b]], np.float32), *[params[k] for k in ORDER])[0]
        worst = max(worst, float(out.max() - out.min()))
        used += 1
    return worst, used


def evaluate(src):
    res = run_js(src, BODY)
    if res is None:
        return ["node driver failed"]
    errs = []
    worst_x, _u = grey_error(res["picks"], exact=True)
    if worst_x > TOL_EXACT:
        errs.append("P1 exact pick not neutral in the run: %.2e > %g" % (worst_x, TOL_EXACT))
    worst, used = grey_error(res["picks"])
    if used < 60:
        errs.append("P1 too few in-range casts (%d)" % used)
    if worst > TOL_GREY:
        errs.append("P1 rounded pick beyond two 8-bit levels: %.5f > %g" % (worst, TOL_GREY))
    for i, p in enumerate(res["refuse"]):
        if "reason" not in p or "temperature" in p:
            errs.append("P2 refusal %d gave values: %r" % (i, p))
    c = res["clamp"]
    if not (c.get("clamped") is True and abs(c.get("temperature", 0)) == 2.0):
        errs.append("P3 strong cast not clamped to the range: %r" % c)
    m = res["pm_mid"]
    exp_r = sum(i * 10 for i in (0, 1, 2, 4, 5, 6, 8, 9, 10)) / 9 / 255
    exp_b = sum(255 - i for i in (0, 1, 2, 4, 5, 6, 8, 9, 10)) / 9 / 255
    if not (abs(m[0] - exp_r) < 1e-9 and abs(m[1] - 100 / 255) < 1e-9 and abs(m[2] - exp_b) < 1e-9):
        errs.append("P4 patch mean wrong: %r" % m)
    e = res["pm_edge"]
    if abs(e[0] - (0 + 10 + 40 + 50) / 4 / 255) > 1e-9:
        errs.append("P4 edge patch not respecting the border: %r" % e)
    pp = res["pp"]
    if (pp[0] != [0, 0] or pp[1] != [0, 0] or pp[2] != [99, 49] or pp[3] is not None
            or pp[4] is not None or pp[5] != [0, 0] or pp[6] != [49, 99]):
        errs.append("P5 letterbox mapping wrong: %r" % pp)
    return errs, (worst_x, worst)


out = evaluate(JS_SRC)
if isinstance(out, list):
    FAILS.extend(out)
    worst = None
else:
    errs, worst = out
    FAILS.extend(errs)

# --- P6 wiring --------------------------------------------------------------
lim = float(re.search(r"const WB_LIMIT\s*=\s*([0-9.]+)", JS_SRC).group(1))
for key in ("temperature", "tint"):
    m = re.search(r'"%s": \("FLOAT", \{"default": 0\.0, "min": (-?[0-9.]+), "max": ([0-9.]+)' % key, PY_SRC)
    check(m is not None and float(m.group(1)) == -lim and float(m.group(2)) == lim,
          "P6 python %s range != +-WB_LIMIT" % key)
canon = re.search(r"FILTER_CANON = \((.*?)\)", PY_SRC, re.S).group(1)
check(tuple(re.findall(r'"(\w+)"', canon))[:15] == ("exposure", "temperature", "tint", "contrast",
      "gamma", "shadows", "highlights", "saturation", "vibrance", "hue_shift", "lut_name",
      "lut_strength", "sharpen_amount", "sharpen_radius", "preset"),
      "P6 the frozen fifteen changed (append-only; the picker adds no key)")
check("_pfApplyParams(node, { temperature: ans.temperature, tint: ans.tint })" in JS_SRC,
      "P6 the pick must write exactly temperature/tint through _pfApplyParams")
check("document.addEventListener" not in JS_SRC and "window.addEventListener" not in JS_SRC,
      "P6 no document/window listeners in ph_filter.js")
check('pickBtn.textContent = "Pick white"' in JS_SRC, "P6 Pick white button missing")
check("if (node._pfPicking) {\n                    doPick(ev);" in JS_SRC,
      "P6 an armed picker must take the click before the divider")

# --- P7 mutations -----------------------------------------------------------
MUT = [
    ("T formula sign", "let t = 4 * (b - r) / (r + b);", "let t = 4 * (r - b) / (r + b);"),
    ("blown check gone", "if (Math.max(r, g, b) >= PICK_BLOWN) {", "if (false) {"),
    ("tint target", "let n = 4 * (1 - (2 * r * b / (r + b)) / g);", "let n = 4 * (1 - (r + b) / 2 / g);"),
]
for name, a, b in MUT:
    if JS_SRC.count(a) != 1:
        FAILS.append("P7 mutation anchor not unique: " + name)
        continue
    r = evaluate(JS_SRC.replace(a, b))
    caught = isinstance(r, list) or bool(r[0])
    check(caught, "P7 mutation survived: " + name)

if FAILS:
    for f in FAILS:
        print("[test_v998_filter_picker] FAIL: " + f)
    sys.exit(1)
print("[test_v998_filter_picker] OK -- 7 promises, grey exact %.1e / widget-rounded %.1e, 3/3 mutations caught" % worst)
