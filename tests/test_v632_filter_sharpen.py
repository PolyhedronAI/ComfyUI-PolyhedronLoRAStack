#!/usr/bin/env python3
"""
test_v632_filter_sharpen -- the unsharp-mask stage and the painted title
controls (Reset chip + scrub hint ABOVE the fields; the circled-i glyph
was removed on screen feedback).

Sharpen is the first NEIGHBORHOOD op in the filter (everything before is
per-pixel), so the mirror contract gets its own drive: _sharpen_np (py) and
_gaussKernel/_sharpenBuf (js) are lifted from source and executed on the
same deterministic image (gradient + impulse). What is pinned:

  INVARIANTS -- amount 0 is a strict no-op (both sides); a constant image
        is a fixed point of the unsharp mask at any amount (blur of a
        constant is the constant); the impulse center must INCREASE
        (sharpening raises local contrast, direction sanity).
  PARITY -- py float32 vs js float64 on the fixture, max |diff| < 1e-3,
        including the replicate borders.
  TITLE CONTROLS -- non-canon DOM widgets must ride BELOW the canon, so
        "Reset above the fields" is PAINTED into the title bar with manual
        hit tests: _pfTitleLayout is pure and driven here (chips live
        inside the title strip, info left of Reset); the reset logic
        (_pfReset), the hit-test wiring and the tip lines are pinned
        statically. Still no document-level listeners.

  Mutations: js kernel width wound -> parity breaks; py amount wound ->
        parity breaks; layout margin wound -> driven layout breaks.
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

TOL = 1e-4   # measured true parity is ~1e-7; the smallest wound worth
             # catching (kernel 3-sigma -> 2-sigma) lands at ~8e-4
W, H = 9, 7
AMOUNT, RADIUS = 0.8, 1.2


def _fail(msg):
    print("[test_v632_filter_sharpen] FAIL: " + msg)
    sys.exit(1)


def _fixture():
    img = [[[((x + 1) * (y + 2) % 10) / 10.0,
             (x * 0.7 + y * 0.31) % 1.0,
             ((x * 3 + y * 5) % 11) / 11.0] for x in range(W)] for y in range(H)]
    img[3][4] = [1.0, 1.0, 1.0]   # impulse
    img[2][1] = [0.0, 0.0, 0.0]
    return img


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
# STATIC: painted title controls + discipline
# ---------------------------------------------------------------------------
for needle, msg in [
    ("function _pfTitleLayout(", "painted title layout helper missing"),
    ("function _pfReset(", "shared reset logic missing"),
    ("_pfHit(hits.reset, pos)", "Reset chip hit test missing"),
    ("const TITLE_HINT = ", "painted title scrub hint missing"),
    ("click-drag values", "scrub hint text drifted"),
    ('"sharpen_amount", "sharpen_radius"', "sharpen widgets missing from the live set"),
]:
    if needle not in JS_SRC:
        _fail(msg)
for gone in ("_pfInfoPin", "TIP_LINES", "hits.info"):
    if gone in JS_SRC:
        _fail("info-glyph remnant %r still present -- the glyph/panel was "
              "removed on screen feedback (it only raised the native tooltip); "
              "tips live in the title hint, the field tooltips and the "
              "description now" % gone)
if "resetBtn" in JS_SRC:
    _fail("DOM reset button still present -- Reset moved to the painted title bar")
if "document.addEventListener" in JS_SRC:
    _fail("document-level listeners are the v624 leak class -- keep events on node callbacks")

# ---------------------------------------------------------------------------
# DRIVEN: title layout geometry
# ---------------------------------------------------------------------------
js_layout = _lift_jsfunc(JS_SRC, "_pfTitleLayout")
out = json.loads(_run_node(js_layout + """
console.log(JSON.stringify(_pfTitleLayout(300, 30)));
"""))
reset = out["reset"]
if reset != [230, -23, 44, 16]:
    _fail("reset chip layout drifted: %r" % (reset,))
if "info" in out:
    _fail("layout still carries the removed info glyph")
if not (-30 <= reset[1] and reset[1] + reset[3] <= 0):
    _fail("reset chip must live inside the title strip")

# ---------------------------------------------------------------------------
# DRIVEN: sharpen -- invariants + parity
# ---------------------------------------------------------------------------
env = {"np": np}
exec(_lift_pyfunc(PY_SRC, "_gauss_kernel"), env)
exec(_lift_pyfunc(PY_SRC, "_sharpen_np"), env)
py_sharpen = env["_sharpen_np"]

img = np.asarray(_fixture(), dtype=np.float32)

# amount 0 -> strict no-op
if not np.array_equal(py_sharpen(img, 0.0, RADIUS), img):
    _fail("py: amount 0 must be a strict no-op")

# constant image -> fixed point
const = np.full((H, W, 3), 0.42, dtype=np.float32)
if float(np.max(np.abs(py_sharpen(const, AMOUNT, RADIUS) - const))) > 1e-6:
    _fail("py: a constant image must be a fixed point of the unsharp mask")

py_out = np.asarray(py_sharpen(img, AMOUNT, RADIUS), dtype=np.float64)
if not (py_out[3, 4] >= img[3, 4] - 1e-6).all() or float(py_out[3, 4].sum()) <= float(img[3, 4].sum()) - 1e-6:
    _fail("py: impulse center must not lose energy under sharpening")


def _js_sharpen(kernel_fn, sharpen_fn, amount):
    driver = kernel_fn + "\n" + sharpen_fn + """
const IMG = %s;
const W = %d, H = %d;
const buf = new Float32Array(W * H * 3);
let f = 0;
for (const row of IMG) for (const px of row) { buf[f++] = px[0]; buf[f++] = px[1]; buf[f++] = px[2]; }
const out = _sharpenBuf(buf, W, H, %s, %s);
console.log(JSON.stringify(Array.from(out)));
""" % (json.dumps(_fixture()), W, H, amount, RADIUS)
    flat = json.loads(_run_node(driver))
    return np.asarray(flat, dtype=np.float64).reshape(H, W, 3)


js_kernel = _lift_jsfunc(JS_SRC, "_gaussKernel")
js_sharp = _lift_jsfunc(JS_SRC, "_sharpenBuf")

js_out = _js_sharpen(js_kernel, js_sharp, AMOUNT)
d = float(np.max(np.abs(js_out - py_out)))
if d > TOL:
    _fail("sharpen parity broken: max |py - js| = %.6g > %g" % (d, TOL))

js_noop = _js_sharpen(js_kernel, js_sharp, 0.0)
# the js buffer stores float32; a strict no-op returns exactly those values
fix32 = np.asarray(_fixture(), dtype=np.float32).astype(np.float64)
if float(np.max(np.abs(js_noop - fix32))) > 0:
    _fail("js: amount 0 must be a strict no-op")

# ---------------------------------------------------------------------------
# MUTATIONS
# ---------------------------------------------------------------------------
mut = js_kernel.replace("Math.ceil(3 * sigma)", "Math.ceil(2 * sigma)")
if mut == js_kernel:
    _fail("js kernel mutation did not apply")
if float(np.max(np.abs(_js_sharpen(mut, js_sharp, AMOUNT) - py_out))) <= TOL:
    _fail("MUTATION NOT CAUGHT: js kernel-width wound survived the parity check")

mut_py = _lift_pyfunc(PY_SRC, "_sharpen_np").replace(
    "np.float32(float(amount)) * (x - blur)",
    "np.float32(float(amount) * 0.9) * (x - blur)")
env2 = {"np": np, "_gauss_kernel": env["_gauss_kernel"]}
exec(mut_py, env2)
if float(np.max(np.abs(np.asarray(env2["_sharpen_np"](img, AMOUNT, RADIUS), dtype=np.float64)
                       - js_out))) <= TOL:
    _fail("MUTATION NOT CAUGHT: py amount wound survived the parity check")

mut_layout = js_layout.replace("const chipW = 44, chipH = 16, margin = 26;",
                               "const chipW = 44, chipH = 16, margin = 6;")
if mut_layout == js_layout:
    _fail("layout mutation did not apply")
out2 = json.loads(_run_node(mut_layout + """
console.log(JSON.stringify(_pfTitleLayout(300, 30)));
"""))
if out2["reset"] == [230, -23, 44, 16]:
    _fail("MUTATION NOT CAUGHT: layout margin wound survived the driven check")

print("[test_v632_filter_sharpen] PASS: sharpen parity %.2g + invariants held, "
      "painted title controls pinned + layout driven, 3 mutations caught" % d)
