#!/usr/bin/env python3
"""
test_v1329_sharpen_masking -- edge masking for the unsharp mask (Polyhedron
Filter), like 'Masking' in Lightroom: contours sharpen, fine texture stays.

  M1  PARITY -- _sharpen_np and _sharpenBuf with masking 0.7 (and threshold),
      on a seeded synthetic picture (soft gradient, a contour, fine texture):
      max |diff| < 1e-4; the two edge masks agree to 1e-4.
  M2  OFF IS v1328 -- masking 0 equals the call without masking bit for bit
      (python) and the JS call without the argument (pre-v1329) equals 0.
  M3  TEXTURE STAYS -- fine low-amplitude texture on a flat area (skin pores)
      changes by less than 10 % of what the plain mask does to it.
  M4  CONTOURS STILL SHARPEN -- a hard step keeps at least 90 % of the plain
      mask's overshoot at masking 0.7 and 1.0.
  M5  MONOTONE -- texture change falls as masking rises (0, 0.3, 0.6, 1.0).
  M6  SCALE -- the same soft contour at half size with half the radius (the
      preview proxy) gives the same mask peak within 0.05.
  M7  WIRING -- canon slot 24 (the end), widget 0..1 default 0, apply()
      passes masking, CANON_DEFAULTS, live widget, the preview passes it,
      constants equal on both sides.
  M8  MUTATIONS -- five wounds, each must go red.
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
TAG = "[test_v1329_sharpen_masking]"
FAILS = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)


def load_py(src):
    spec = importlib.util.spec_from_loader("phf_mask", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = PY_PATH
    exec(compile(src, PY_PATH, "exec"), mod.__dict__)
    return mod


def lift_js(src, name):
    i = src.find("function " + name + "(")
    j = src.find("\n}", i)
    if i < 0 or j < 0:
        raise SystemExit(TAG + " FAIL: cannot lift " + name)
    return src[i:j + 2]


def js_consts(src):
    return "\n".join(re.findall(r"^const MASK_[A-Z_]+ = [0-9.]+;$", src, re.M))


def run_node(code):
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(["node", path], capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            return None
        return json.loads(r.stdout)
    finally:
        os.unlink(path)


def js_prelude(src):
    return (js_consts(src) + "\n" + lift_js(src, "_gaussKernel") + "\n"
            + lift_js(src, "_edgeMaskBuf") + "\n" + lift_js(src, "_sharpenBuf") + "\n")


def js_sharpen(src, img, amount, radius, threshold=None, masking=None):
    h, w = img.shape[:2]
    args = "B, %d, %d, %r, %r" % (w, h, amount, radius)
    if threshold is not None:
        args += ", %r" % threshold
        if masking is not None:
            args += ", %r" % masking
    code = js_prelude(src) + """
const B = new Float32Array(%s);
console.log(JSON.stringify(Array.from(_sharpenBuf(%s))));
""" % (json.dumps([float(v) for v in img.reshape(-1)]), args)
    out = run_node(code)
    return None if out is None else np.asarray(out, np.float64).reshape(img.shape)


def js_mask(src, img, radius, masking):
    h, w = img.shape[:2]
    code = js_prelude(src) + """
const B = new Float32Array(%s);
console.log(JSON.stringify(Array.from(_edgeMaskBuf(B, %d, %d, %r, %r))));
""" % (json.dumps([float(v) for v in img.reshape(-1)]), w, h, radius, masking)
    out = run_node(code)
    return None if out is None else np.asarray(out, np.float64).reshape(h, w)


rng = np.random.default_rng(1329)
H, W = 40, 48
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
# soft gradient + a hard contour at x = 24 + fine texture (pores) everywhere
PIC = np.stack([0.35 + 0.2 * xx / W, 0.3 + 0.15 * yy / H, 0.28 + 0.0 * xx], -1).astype(np.float32)
PIC[:, 24:] += 0.3
PIC += rng.normal(0, 0.006, (H, W, 1)).astype(np.float32)
PIC = np.clip(PIC, 0, 1).astype(np.float32)
AMT, RAD, T = 1.0, 1.1, 1 / 255.0

TEX = np.clip(0.6 + rng.normal(0, 0.008, (32, 32, 1)) + 0 * np.zeros((32, 32, 3)), 0, 1).astype(np.float32)
# coarser texture (a mix of pores and soft relief) for the monotone sweep
TEX2 = np.clip(0.55 + rng.normal(0, 0.04, (32, 32, 1)) + 0 * np.zeros((32, 32, 3)), 0, 1).astype(np.float32)
STEP = np.full((12, 24, 3), 0.25, np.float32)
STEP[:, 12:] = 0.8


def evaluate(py_src, js_src):
    errs = []
    m = load_py(py_src)
    # M1 parity
    py = m._sharpen_np(PIC, AMT, RAD, T, 0.7).astype(np.float64)
    js = js_sharpen(js_src, PIC, AMT, RAD, T, 0.7)
    if js is None:
        return ["node driver failed"]
    d = float(np.max(np.abs(py - js)))
    if d > 1e-4:
        errs.append("M1 parity with masking %.2e > 1e-4" % d)
    mp = m._edge_mask_np(PIC, RAD, 0.7).astype(np.float64)
    mj = js_mask(js_src, PIC, RAD, 0.7)
    if mj is None or float(np.max(np.abs(mp - mj))) > 1e-4:
        errs.append("M1 edge masks differ")
    # M2 off = v1328
    if not np.array_equal(m._sharpen_np(PIC, AMT, RAD, T, 0.0), m._sharpen_np(PIC, AMT, RAD, T)):
        errs.append("M2 masking 0 is not the v1328 mask (python)")
    j0 = js_sharpen(js_src, PIC, AMT, RAD, T)
    j1 = js_sharpen(js_src, PIC, AMT, RAD, T, 0.0)
    if j0 is None or j1 is None or float(np.max(np.abs(j0 - j1))) > 0.0:
        errs.append("M2 masking 0 is not the v1328 mask (JS)")
    # M3 texture stays
    plain = float(np.abs(m._sharpen_np(TEX, AMT, RAD, 0.0, 0.0) - TEX).mean())
    masked = float(np.abs(m._sharpen_np(TEX, AMT, RAD, 0.0, 0.7) - TEX).mean())
    if not (plain > 0 and masked < 0.1 * plain):
        errs.append("M3 texture changed %.2e at masking 0.7 vs plain %.2e" % (masked, plain))
    jt = js_sharpen(js_src, TEX, AMT, RAD, 0.0, 0.7)
    if jt is None or float(np.abs(jt - TEX).mean()) > 0.1 * plain:
        errs.append("M3 texture changed in JS")
    # M4 contours
    over0 = float(m._sharpen_np(STEP, 1.0, RAD, 0.0, 0.0)[:, 12:].max() - 0.8)
    for mk in (0.7, 1.0):
        over = float(m._sharpen_np(STEP, 1.0, RAD, 0.0, mk)[:, 12:].max() - 0.8)
        if not (over0 > 0 and over >= 0.9 * over0):
            errs.append("M4 contour overshoot %.4f at masking %.1f vs plain %.4f" % (over, mk, over0))
    # M5 monotone
    ch = [float(np.abs(m._sharpen_np(TEX2, AMT, RAD, 0.0, mk) - TEX2).mean()) for mk in (0.0, 0.3, 0.6, 1.0)]
    if not (all(a >= b for a, b in zip(ch, ch[1:])) and ch[0] > ch[1] > 0 and ch[0] > ch[-1]):
        errs.append("M5 texture change not falling with masking: %r" % ch)
    # M6 scale: soft contour (ramp 8 px, contrast 0.15 -- mid-transition at masking 1) at full size, radius 2 vs half size, radius 1
    def ramp(wd, width_px):
        x = np.arange(wd, dtype=np.float32)
        v = np.clip((x - wd / 2) / width_px + 0.5, 0, 1) * 0.15 + 0.4
        return np.repeat(np.repeat(v[None, :, None], 16, 0), 3, 2).astype(np.float32)
    big = float(m._edge_mask_np(ramp(64, 8.0), 2.0, 1.0).max())
    small = float(m._edge_mask_np(ramp(32, 4.0), 1.0, 1.0).max())
    if abs(big - small) > 0.05 or not (0.05 < big < 0.999):
        errs.append("M6 mask not scale-true: %.3f full vs %.3f half" % (big, small))
    # M7 apply() passes masking
    node = m.ULSFilter()
    kw = dict(exposure=0.0, temperature=0.0, tint=0.0, contrast=0.0, gamma=1.0,
              shadows=0.0, highlights=0.0, saturation=0.0, vibrance=0.0, hue_shift=0.0,
              lut_name="none", lut_strength=1.0, sharpen_amount=AMT, sharpen_radius=RAD,
              preset="none", auto_mode="off", sharpen_threshold=1.0, sharpen_masking=0.7)
    with contextlib.redirect_stdout(io.StringIO()):
        out = np.asarray(node.apply(PIC[None], **kw)["result"][0])[0]
    if float(np.max(np.abs(out - m._sharpen_np(PIC, AMT, RAD, T, 0.7)))) > 1e-6:
        errs.append("M7 apply() does not pass sharpen_masking")
    if m.FILTER_CANON[24] != "sharpen_masking" or len(m.FILTER_CANON) != 25:
        errs.append("M7 canon slot 24 (the end)")
    if (m.MASK_SIGMA_K, m.MASK_LO, m.MASK_HI) != (2.0, 0.04, 0.07):
        errs.append("M7 python constants")
    if js_consts(js_src) != "const MASK_SIGMA_K = 2.0;\nconst MASK_LO = 0.04;\nconst MASK_HI = 0.07;":
        errs.append("M7 JS constants")
    return errs


FAILS.extend(evaluate(PY_SRC, JS_SRC))
check("harmonize_window: 0,\n    sharpen_masking: 0.0,\n};" in JS_SRC, "M7 CANON_DEFAULTS sharpen_masking 0")
check(re.search(r'const LIVE_WIDGETS = GRADE_WIDGETS\.concat\(\[[^\]]*"sharpen_masking"', JS_SRC) is not None,
      "M7 sharpen_masking must be a live preview widget")
check(re.search(r"_previewRadius\(sharpR, src\.width, node\._pfSrcW\), sharpT, sharpM\)", JS_SRC) is not None
      and 'x.name === "sharpen_masking"' in JS_SRC, "M7 the preview must pass the masking")
check('"sharpen_masking": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,' in PY_SRC,
      "M7 sharpen_masking widget 0..1, default 0")
check("float(sharpen_threshold) / 255.0, float(sharpen_masking))" in PY_SRC, "M7 apply wiring")

MUT = [
    ("JS mask not applied", "js", "if (mask) d *= mask[(i / 3) | 0];", "if (mask) d *= 1;"),
    ("PY gradient not times sigma", "py", "g = np.sqrt(gx * gx + gy * gy) * np.float32(sigma)",
     "g = np.sqrt(gx * gx + gy * gy)"),
    ("JS lo/hi swapped", "js", "const lo = MASK_LO * masking, hi = MASK_HI * masking;",
     "const lo = MASK_HI * masking, hi = MASK_LO * masking;"),
    ("PY masking ignored in sharpen", "py", "    if float(masking) > 0.0:\n        d = d * _edge_mask_np",
     "    if float(masking) > 9.0:\n        d = d * _edge_mask_np"),
    ("PY mask sigma not scaled", "py", "sigma = MASK_SIGMA_K * max(float(radius), 0.1)", "sigma = MASK_SIGMA_K * 1.1"),
]
caught = 0
for name, side, a, b in MUT:
    src = JS_SRC if side == "js" else PY_SRC
    if src.count(a) != 1:
        FAILS.append("M8 mutation anchor not unique: " + name)
        continue
    r = evaluate(PY_SRC, JS_SRC.replace(a, b)) if side == "js" else evaluate(PY_SRC.replace(a, b), JS_SRC)
    if r:
        caught += 1
    else:
        FAILS.append("M8 mutation survived: " + name)

if FAILS:
    for f in FAILS:
        print(TAG + " FAIL: " + f)
    sys.exit(1)
print(TAG + " OK -- 8 promises, parity + v1328 kept + texture calm + contours sharp, %d/5 mutations caught" % caught)
