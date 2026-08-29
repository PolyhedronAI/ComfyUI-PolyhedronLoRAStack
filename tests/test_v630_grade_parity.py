#!/usr/bin/env python3
"""
test_v630_grade_parity -- ONE pipeline, two runtimes, measured agreement.

The Polyhedron Filter's color math exists twice on purpose: _grade_np in
nodes/ph_filter.py is the ground truth the run applies to the full-res
IMAGE batch; _gradeRGB in web/js/ph_filter.js is its op-for-op mirror that
grades the in-node preview live. The whole feature stands on the promise
that what the sliders show is what the run produces. This guard MEASURES
that promise instead of believing it:

  PARITY -- both functions are lifted from source and DRIVEN over the same
        11x11x11 RGB grid (0..1 in 0.1 steps, 1331 samples) with every
        control off-neutral (including a hue rotation). Python runs the real
        numpy production code in float32, node runs the real JS in float64;
        max |diff| must stay under TOL_PARITY. Anything drifting past that
        would be visible against the preview's own 8-bit quantum (1/255).

  NEUTRAL -- the pipeline at default values must be the identity on the
        same grid (both sides, TOL_NEUTRAL). A grading node whose neutral
        position changes the image lies to every workflow that leaves the
        sliders alone.

  Mutations (inject the wound, prove the catch):
    1. JS white-balance gain 0.25 -> 0.35  -> parity must break.
    2. JS gamma exponent +0.1              -> neutral must break.
    3. PY contrast pivot 0.5 -> 0.4        -> parity must break.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_SRC = open(os.path.join(ROOT, "nodes", "ph_filter.py"), encoding="utf-8").read()
JS_SRC = open(os.path.join(ROOT, "web", "js", "ph_filter.js"), encoding="utf-8").read()

TOL_PARITY = 1e-3    # well under the preview's 8-bit quantum (1/255 ~ 3.9e-3)
TOL_NEUTRAL = 1e-5   # float32 matrix noise headroom, nothing more

PARAMS = {"exposure": 0.35, "temperature": -0.3, "tint": 0.2, "contrast": 0.15,
          "gamma": 1.3, "shadows": 0.25, "highlights": -0.2, "saturation": 0.3,
          "vibrance": 0.4, "hue_shift": 25.0}
NEUTRAL = {"exposure": 0.0, "temperature": 0.0, "tint": 0.0, "contrast": 0.0,
           "gamma": 1.0, "shadows": 0.0, "highlights": 0.0, "saturation": 0.0,
           "vibrance": 0.0, "hue_shift": 0.0}
ORDER = ("exposure", "temperature", "tint", "contrast", "gamma",
         "shadows", "highlights", "saturation", "vibrance", "hue_shift")


def _fail(msg):
    print("[test_v630_grade_parity] FAIL: " + msg)
    sys.exit(1)


def _grid():
    vals = [round(i * 0.1, 1) for i in range(11)]
    return [(r, g, b) for r in vals for g in vals for b in vals]


# ---------------------------------------------------------------------------
# lifting (house doctrine: run the code, do not read it)
# ---------------------------------------------------------------------------
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


def _lift_jsfunc(src, name):
    sig = "function " + name + "("
    i = src.find(sig)
    if i < 0:
        _fail("could not lift js function %s" % name)
    j = src.find("\n}", i)
    if j < 0:
        _fail("could not find end of js function %s" % name)
    return src[i:j + 2]


def _run_py(py_snippet, params):
    env = {"np": np}
    exec(py_snippet, env)
    f = env["_grade_np"]
    arr = np.asarray(_grid(), dtype=np.float32)
    out = f(arr, *[params[k] for k in ORDER])
    return np.asarray(out, dtype=np.float64)


def _run_js(js_snippet, params):
    driver = js_snippet + """
const P = %s;
const GRID = %s;
const out = [];
for (const [r, g, b] of GRID) out.push(_gradeRGB(r, g, b, P));
console.log(JSON.stringify(out));
""" % (json.dumps(params), json.dumps(_grid()))
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(driver)
        path = f.name
    try:
        r = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            _fail("node driver failed: %s" % (r.stderr or "")[:200])
        return np.asarray(json.loads(r.stdout), dtype=np.float64)
    finally:
        os.unlink(path)


def _max_diff(a, b):
    return float(np.max(np.abs(a - b)))


py_fn = _lift_pyfunc(PY_SRC, "_grade_np")
js_fn = _lift_jsfunc(JS_SRC, "_gradeRGB")

# --- PARITY ---------------------------------------------------------------
d = _max_diff(_run_py(py_fn, PARAMS), _run_js(js_fn, PARAMS))
if d > TOL_PARITY:
    _fail("parity broken: max |py - js| = %.6g > %g" % (d, TOL_PARITY))

# --- NEUTRAL --------------------------------------------------------------
grid = np.asarray(_grid(), dtype=np.float64)
dn_py = _max_diff(_run_py(py_fn, NEUTRAL), grid)
dn_js = _max_diff(_run_js(js_fn, NEUTRAL), grid)
if dn_py > TOL_NEUTRAL or dn_js > TOL_NEUTRAL:
    _fail("neutral is not the identity: py %.6g / js %.6g > %g"
          % (dn_py, dn_js, TOL_NEUTRAL))

# --- MUTATIONS ------------------------------------------------------------
mut = js_fn.replace("r *= 1 + 0.25 * p.temperature;", "r *= 1 + 0.35 * p.temperature;")
if mut == js_fn:
    _fail("js white-balance mutation did not apply")
if _max_diff(_run_py(py_fn, PARAMS), _run_js(mut, PARAMS)) <= TOL_PARITY:
    _fail("MUTATION NOT CAUGHT: js white-balance wound survived the parity check")

mut = js_fn.replace("Math.pow(Math.max(r, 0), p.gamma)",
                    "Math.pow(Math.max(r, 0), p.gamma + 0.1)")
if mut == js_fn:
    _fail("js gamma mutation did not apply")
if _max_diff(_run_js(mut, NEUTRAL), grid) <= TOL_NEUTRAL:
    _fail("MUTATION NOT CAUGHT: js gamma wound survived the neutral check")

mut = py_fn.replace("np.float32(0.5) + (x - np.float32(0.5))",
                    "np.float32(0.4) + (x - np.float32(0.4))")
if mut == py_fn:
    _fail("py contrast mutation did not apply")
if _max_diff(_run_py(mut, PARAMS), _run_js(js_fn, PARAMS)) <= TOL_PARITY:
    _fail("MUTATION NOT CAUGHT: py contrast wound survived the parity check")

print("[test_v630_grade_parity] PASS: 1331-sample parity %.2g, neutral identity "
      "(py %.2g / js %.2g), 3 mutations caught" % (d, dn_py, dn_js))
