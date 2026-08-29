#!/usr/bin/env python3
"""
test_v631_filter_lut -- the LUT stage: one .cube file, two parsers, two
trilinear engines, one measured truth. Plus the Reset contract.

The backend parses luts/*.cube with _parse_cube and grades through
_apply_lut_np; the live preview fetches the SAME file over /uls/filter/lut
and mirrors both steps in _parseCube/_lutRGB. This guard drives all four on
a fixture whose ground truth is ANALYTIC: a size-3 cube encoding the channel
rotation (r,g,b) -> (b,r,g). That mapping is linear, and trilinear
interpolation reproduces a linear function EXACTLY -- so both engines are
checked not just against each other but against the mathematical answer,
at full strength and at a 0.5 blend.

The Reset contract: the JS CANON_DEFAULTS map (what the painted Reset
chip writes) must equal the python INPUT_TYPES defaults. A reset that lands on
different values than a freshly created node is a lie with a button on it.

Pins:
  STATIC -- /uls/filter/lut route registered in ph_filter_routes.py with basename
        sanitization; Reset button present in ph_filter.js; the LUT honesty
        badge exists (preview must SAY when it grades without the LUT).
  DRIVEN -- py: _parse_cube on the fixture (size, domain, spot entries),
        _apply_lut_np vs analytic; js: _parseCube + _lutRGB vs analytic;
        cross parity py vs js; defaults map vs python defaults.
  Mutations: js trilinear axis swap -> analytic check breaks; py blend
        wound -> half-strength check breaks; js defaults wound -> defaults
        parity breaks.
"""

# v372 (public build): the Filter routes live in their own module,
# nodes/ph_filter_routes.py -- uls_routes.py stays the Stack's file. Same
# source text, different path; the checks below are unchanged.
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
ROUTES_SRC = open(os.path.join(ROOT, "nodes", "ph_filter_routes.py"), encoding="utf-8").read()

TOL = 1e-5   # float32 storage noise; the mapping itself is exact

FIXTURE = "\n".join(
    ["TITLE \"rot fixture\"", "# comment line", "LUT_3D_SIZE 3",
     "DOMAIN_MIN 0.0 0.0 0.0", "DOMAIN_MAX 1.0 1.0 1.0"]
    + ["%.1f %.1f %.1f" % (b, r, g)
       for b in (0.0, 0.5, 1.0) for g in (0.0, 0.5, 1.0) for r in (0.0, 0.5, 1.0)]
) + "\n"


def _fail(msg):
    print("[test_v631_filter_lut] FAIL: " + msg)
    sys.exit(1)


def _grid():
    vals = [round(i * 0.25, 2) for i in range(5)]
    return [(r, g, b) for r in vals for g in vals for b in vals]


def _analytic(grid, strength):
    out = []
    for r, g, b in grid:
        lr, lg, lb = b, r, g
        out.append([r * (1 - strength) + lr * strength,
                    g * (1 - strength) + lg * strength,
                    b * (1 - strength) + lb * strength])
    return np.asarray(out, dtype=np.float64)


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


def _run_node(code):
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            _fail("node driver failed: %s" % (r.stderr or "")[:200])
        return r.stdout
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# STATIC
# ---------------------------------------------------------------------------
if '"/uls/filter/lut"' not in ROUTES_SRC:
    _fail("LUT route missing from ph_filter_routes.py")
if "handle_filter_lut" not in ROUTES_SRC:
    _fail("LUT route handler missing")
if "os.path.basename(name)" not in ROUTES_SRC:
    _fail("LUT route must basename-sanitize the requested name")
if "function _pfReset(" not in JS_SRC:
    _fail("Reset logic missing (painted title chip since the controls moved "
          "above the fields; geometry is driven by the sharpen guard)")
if '"LUT not in preview"' not in JS_SRC:
    _fail("LUT honesty badge missing: the preview must SAY when it grades without the LUT")

# ---------------------------------------------------------------------------
# DRIVEN py: parser + trilinear vs analytic
# ---------------------------------------------------------------------------
env = {"np": np}
exec(_lift_pyfunc(PY_SRC, "_parse_cube"), env)
exec(_lift_pyfunc(PY_SRC, "_apply_lut_np"), env)

size, data, dmin, dmax = env["_parse_cube"](FIXTURE)
if size != 3 or data.shape != (3, 3, 3, 3):
    _fail("py parser: wrong size/shape")
if not (np.allclose(dmin, 0.0) and np.allclose(dmax, 1.0)):
    _fail("py parser: DOMAIN lines not honored")
# spot entries: line for (r=1, g=0, b=0) must be (0, 1, 0) -> data[b=0,g=0,r=2]
if not np.allclose(data[0, 0, 2], [0.0, 1.0, 0.0]):
    _fail("py parser: line ordering broken (r must run fastest)")

grid = np.asarray(_grid(), dtype=np.float32)
py_apply = env["_apply_lut_np"]


def _py_out(fn, strength, d=data):
    return np.asarray(fn(grid, size, d, dmin, dmax, strength), dtype=np.float64)


for s_ in (1.0, 0.5):
    dmaxdiff = float(np.max(np.abs(_py_out(py_apply, s_) - _analytic(_grid(), s_))))
    if dmaxdiff > TOL:
        _fail("py trilinear off analytic truth at strength %s: %.3g" % (s_, dmaxdiff))

# ---------------------------------------------------------------------------
# DRIVEN js: parser + trilinear vs analytic + cross parity
# ---------------------------------------------------------------------------
js_parse = _lift_jsfunc(JS_SRC, "_parseCube")
js_lut = _lift_jsfunc(JS_SRC, "_lutRGB")
js_coord = _lift_jsfunc(JS_SRC, "_lutCoord")


def _js_out(parse_fn, lut_fn, strength):
    driver = parse_fn + "\n" + lut_fn + "\n" + js_coord + """
const lut = _parseCube(%s);
const GRID = %s;
const out = [];
for (const [r, g, b] of GRID) out.push(_lutRGB(r, g, b, lut, %s));
console.log(JSON.stringify({size: lut.size, out: out}));
""" % (json.dumps(FIXTURE), json.dumps(_grid()), strength)
    res = json.loads(_run_node(driver))
    if res["size"] != 3:
        _fail("js parser: wrong size")
    return np.asarray(res["out"], dtype=np.float64)


for s_ in (1.0, 0.5):
    a = _js_out(js_parse, js_lut, s_)
    dj = float(np.max(np.abs(a - _analytic(_grid(), s_))))
    dx = float(np.max(np.abs(a - _py_out(py_apply, s_))))
    if dj > TOL:
        _fail("js trilinear off analytic truth at strength %s: %.3g" % (s_, dj))
    if dx > TOL:
        _fail("py/js LUT parity broken at strength %s: %.3g" % (s_, dx))

# ---------------------------------------------------------------------------
# DRIVEN: Reset defaults parity (JS map vs python INPUT_TYPES defaults)
# ---------------------------------------------------------------------------
mjs = re.search(r"const CANON_DEFAULTS = \{(.*?)\};", JS_SRC, re.S)
if not mjs:
    _fail("CANON_DEFAULTS map missing from ph_filter.js")
js_defaults = {}
for k, v in re.findall(r"(\w+):\s*(\"[^\"]*\"|[-\d.]+)", mjs.group(1)):
    js_defaults[k] = v.strip('"') if v.startswith('"') else float(v)

py_defaults = {}
mreq = re.search(r'"required"\s*:\s*\{(.*?)\n\s*\}\s*,?\s*\n\s*\}', PY_SRC, re.S)
if not mreq:
    _fail("required INPUT_TYPES block not found")
for name, dv in re.findall(r'"(\w+)":\s*\((?:[^()]*?)\{"default":\s*("?[\w.\-]+"?)', mreq.group(1)):
    py_defaults[name] = dv.strip('"') if dv.startswith('"') else float(dv)

if set(js_defaults) != set(py_defaults):
    _fail("defaults key sets differ: js-only %r, py-only %r"
          % (sorted(set(js_defaults) - set(py_defaults)),
             sorted(set(py_defaults) - set(js_defaults))))
for k in py_defaults:
    if js_defaults[k] != py_defaults[k]:
        _fail("default mismatch for %s: js %r vs py %r" % (k, js_defaults[k], py_defaults[k]))

# ---------------------------------------------------------------------------
# MUTATIONS
# ---------------------------------------------------------------------------
mut = js_lut.replace("const fr = cr - r0, fg = cg - g0, fb = cb - b0;",
                     "const fr = cg - g0, fg = cr - r0, fb = cb - b0;")
if mut == js_lut:
    _fail("js axis mutation did not apply")
a = _js_out(js_parse, mut, 1.0)
if float(np.max(np.abs(a - _analytic(_grid(), 1.0)))) <= TOL:
    _fail("MUTATION NOT CAUGHT: js trilinear axis wound survived the analytic check")

mut_py = _lift_pyfunc(PY_SRC, "_apply_lut_np").replace(
    "x * (1 - s) + lut_out * s", "x * (1 - s) + lut_out * s * 0.9")
env2 = {"np": np}
exec(mut_py, env2)
if float(np.max(np.abs(_py_out(env2["_apply_lut_np"], 0.5) - _analytic(_grid(), 0.5)))) <= TOL:
    _fail("MUTATION NOT CAUGHT: py blend wound survived the half-strength check")

mut_defaults = dict(js_defaults)
mut_defaults["gamma"] = 1.1
if mut_defaults == py_defaults:
    _fail("defaults mutation did not apply")
caught = any(mut_defaults[k] != py_defaults[k] for k in py_defaults)
if not caught:
    _fail("MUTATION NOT CAUGHT: defaults wound survived the parity check")

print("[test_v631_filter_lut] PASS: parser + trilinear exact vs analytic truth "
      "(both runtimes, strengths 1.0/0.5), defaults parity held, 3 mutations caught")
