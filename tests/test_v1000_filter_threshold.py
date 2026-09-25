#!/usr/bin/env python3
"""
test_v1000_filter_threshold -- the unsharp-mask threshold (F3).

  P1  PARITY -- _sharpen_np and _sharpenBuf with a threshold, on a seeded
      synthetic picture (gradient, edges, noise): max |diff| < 1e-4.
  P2  OFF IS THE OLD MASK -- threshold 0 (python) and no threshold argument
      (JS, the pre-v1000 call) equal x + amount * (x - blur) exactly.
  P3  CORING -- flat grey with noise whose detail stays under the threshold
      comes out untouched (both runtimes).
  P4  EDGES STILL SHARPEN -- a big step keeps an overshoot, smaller than the
      plain mask's by amount * t.
  P5  MONOTONE -- flat-area noise after sharpening falls as the threshold
      rises (0, 2, 4, 8 levels).
  P6  WIRING -- canon slot 16; default 0 on both sides; live widget; the
      real apply() converts levels to full scale (/255); amount max 4.0; the
      preview passes the threshold.
  P7  MUTATIONS -- three wounds, each must go red.
  P8  PREVIEW RADIUS -- the proxy is sharpened with the radius scaled by
      proxy / run width (measured 3-6x closer to the run), driven + wired.
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
PY_SRC = open(PY_PATH, encoding="utf-8").read()
JS_SRC = open(os.path.join(ROOT, "web", "js", "ph_filter.js"), encoding="utf-8").read()
FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)


def load_py(src):
    spec = importlib.util.spec_from_loader("phf_thr", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = PY_PATH
    exec(compile(src, PY_PATH, "exec"), mod.__dict__)
    return mod


def lift_js(src, name):
    i = src.find("function " + name + "(")
    j = src.find("\n}", i)
    if i < 0 or j < 0:
        raise SystemExit("[test_v1000_filter_threshold] FAIL: cannot lift " + name)
    return src[i:j + 2]


def js_sharpen(src, img, amount, radius, threshold=None):
    h, w = img.shape[:2]
    call = ("_sharpenBuf(B, %d, %d, %r, %r)" % (w, h, amount, radius) if threshold is None
            else "_sharpenBuf(B, %d, %d, %r, %r, %r)" % (w, h, amount, radius, threshold))
    code = lift_js(src, "_gaussKernel") + "\n" + lift_js(src, "_sharpenBuf") + """
const B = new Float32Array(%s);
const O = %s;
console.log(JSON.stringify(Array.from(O)));
""" % (json.dumps([float(v) for v in img.reshape(-1)]), call)
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return None
        return np.asarray(json.loads(r.stdout), np.float64).reshape(img.shape)
    finally:
        os.unlink(path)


rng = np.random.default_rng(1000)
H, W = 24, 32
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
PIC = np.stack([xx / W, yy / H, 0.5 + 0.0 * xx], -1).astype(np.float32)
PIC[:, 16:] += 0.25
PIC += rng.normal(0, 0.01, PIC.shape).astype(np.float32)
PIC = np.clip(PIC, 0, 1).astype(np.float32)
T = 3 / 255.0
AMT, RAD = 2.5, 1.0


def blur_of(m, x, r):
    half, w = m._gauss_kernel(r)

    def ax(a, axis):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (half, half)
        p = np.pad(a, pad, mode="edge")
        out = np.zeros(a.shape, np.float32)
        for i in range(2 * half + 1):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i + a.shape[axis])
            out += w[i] * p[tuple(sl)]
        return out
    return ax(ax(x, 0), 1)


def evaluate(py_src, js_src):
    errs = []
    m = load_py(py_src)
    # P1
    py = m._sharpen_np(PIC, AMT, RAD, T).astype(np.float64)
    js = js_sharpen(js_src, PIC, AMT, RAD, T)
    if js is None:
        return ["node driver failed"]
    d = float(np.max(np.abs(py - js)))
    if d > 1e-4:
        errs.append("P1 parity with threshold %.2e > 1e-4" % d)
    # P2
    ref = np.clip(PIC + np.float32(AMT) * (PIC - blur_of(m, PIC, RAD)), 0, 1)
    d0 = float(np.max(np.abs(m._sharpen_np(PIC, AMT, RAD, 0.0) - ref)))
    dj = float(np.max(np.abs(js_sharpen(js_src, PIC, AMT, RAD) - ref)))
    if d0 > 1e-7 or dj > 1e-4:
        errs.append("P2 threshold off is not the old mask (py %.2e, js %.2e)" % (d0, dj))
    # P3: noise +-0.45 t -> |x - blur| <= 0.9 t < t
    flat = np.clip(0.5 + rng.uniform(-0.45 * T, 0.45 * T, (16, 16, 3)), 0, 1).astype(np.float32)
    c_py = float(np.max(np.abs(m._sharpen_np(flat, AMT, RAD, T) - flat)))
    c_js = float(np.max(np.abs(js_sharpen(js_src, flat, AMT, RAD, T) - flat)))
    if c_py > 1e-7 or c_js > 1e-6:
        errs.append("P3 detail under the threshold was sharpened (py %.2e, js %.2e)" % (c_py, c_js))
    # P4
    step = np.full((8, 16, 3), 0.2, np.float32)
    step[:, 8:] = 0.8
    over0 = float(m._sharpen_np(step, 1.0, RAD, 0.0)[:, 8:].max() - 0.8)
    overT = float(m._sharpen_np(step, 1.0, RAD, T)[:, 8:].max() - 0.8)
    if not (overT > 0 and abs((over0 - overT) - T) < 1e-5):
        errs.append("P4 edge overshoot %.4f vs plain %.4f (expected plain - t)" % (overT, over0))
    # P5
    noisy = np.clip(0.5 + rng.normal(0, 0.012, (32, 32, 3)), 0, 1).astype(np.float32)
    stds = [float(m._sharpen_np(noisy, AMT, RAD, lv / 255.0).std()) for lv in (0, 2, 4, 8)]
    if not all(a > b for a, b in zip(stds, stds[1:])):
        errs.append("P5 noise not falling with the threshold: %r" % stds)
    # P6 (drive the real apply)
    node = m.ULSFilter()
    kw = dict(exposure=0.0, temperature=0.0, tint=0.0, contrast=0.0, gamma=1.0,
              shadows=0.0, highlights=0.0, saturation=0.0, vibrance=0.0, hue_shift=0.0,
              lut_name="none", lut_strength=1.0, sharpen_amount=AMT, sharpen_radius=RAD,
              preset="none", auto_mode="off", sharpen_threshold=3.0)
    with contextlib.redirect_stdout(io.StringIO()):
        out = np.asarray(node.apply(PIC[None], **kw)["result"][0])[0]
    if float(np.max(np.abs(out - m._sharpen_np(PIC, AMT, RAD, T)))) > 1e-6:
        errs.append("P6 apply() does not turn levels into full scale (/255)")
    # RE-GROUNDED v1001: detail_amount appended behind it; the slot holds
    if m.FILTER_CANON[16] != "sharpen_threshold":
        errs.append("P6 canon slot 16")
    return errs


FAILS.extend(evaluate(PY_SRC, JS_SRC))
check('auto_mode: "off", sharpen_threshold: 0.0,' in JS_SRC, "P6 CANON_DEFAULTS sharpen_threshold 0")
# P8 the preview radius follows the proxy (driven + wired)
_pr = subprocess.run(["node", "-e", lift_js(JS_SRC, "_previewRadius") +
                      "; console.log(JSON.stringify([_previewRadius(1, 768, 1344), _previewRadius(2, 768, 768),"
                      " _previewRadius(1.5, 500, 0), _previewRadius(1, 768, 400)]))"],
                     capture_output=True, text=True, timeout=30)
try:
    _v = json.loads(_pr.stdout)
    check(abs(_v[0] - 768 / 1344) < 1e-12 and _v[1:] == [2, 1.5, 1],
          "P8 _previewRadius: %r" % (_v,))
except ValueError:
    FAILS.append("P8 _previewRadius driver failed")
check("_previewRadius(sharpR, src.width, node._pfSrcW)" in JS_SRC
      and "node._pfSrcW = Number(item.src_width) || 0;" in JS_SRC,
      "P8 the preview must scale the radius by proxy / run width")
check(re.search(r'const LIVE_WIDGETS = GRADE_WIDGETS\.concat\(\[[^\]]*"sharpen_threshold"', JS_SRC) is not None,
      "P6 sharpen_threshold must be a live preview widget")
check(re.search(r"_sharpenBuf\(fbuf, src\.width, src\.height, sharpA,\s*_previewRadius\(sharpR, src\.width, node\._pfSrcW\), sharpT\)", JS_SRC) is not None
      and "Number(wSharpT.value) / 255" in JS_SRC, "P6 the preview must pass the threshold in full scale")
check('"sharpen_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0,' in PY_SRC,
      "P6 sharpen_amount max 4.0 (a threshold needs 2-3)")
check('"sharpen_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 32.0, "step": 1.0,' in PY_SRC,
      "P6 sharpen_threshold widget: levels 0..32, default 0")

MUT = [
    ("JS coring sign", "js", "if (t > 0) d = d > t ? d - t : (d < -t ? d + t : 0);",
     "if (t > 0) d = d > t ? d + t : (d < -t ? d - t : 0);"),
    ("PY levels not converted", "py", "float(sharpen_threshold) / 255.0)", "float(sharpen_threshold))"),
    ("JS threshold ignored", "js", "const t = threshold > 0 ? threshold : 0;", "const t = 0;"),
]
caught = 0
for name, side, a, b in MUT:
    src = JS_SRC if side == "js" else PY_SRC
    if src.count(a) != 1:
        FAILS.append("P7 mutation anchor not unique: " + name)
        continue
    r = evaluate(PY_SRC, JS_SRC.replace(a, b)) if side == "js" else evaluate(PY_SRC.replace(a, b), JS_SRC)
    if r:
        caught += 1
    else:
        FAILS.append("P7 mutation survived: " + name)

if FAILS:
    for f in FAILS:
        print("[test_v1000_filter_threshold] FAIL: " + f)
    sys.exit(1)
print("[test_v1000_filter_threshold] OK -- 8 promises, parity + old mask kept + coring, %d/3 mutations caught" % caught)
