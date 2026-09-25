#!/usr/bin/env python3
"""
test_v1001_filter_detail -- detail from the original (F4).

  P1  THE VIEWER'S DEFINITION -- _detail_np with the full grid equals the
      Polyhedron Viewer's detail.reference (pv/detail.py v060, copied below
      verbatim as the fixed reference) exactly, single frame and batched.
  P2  PARITY -- _detail_np and _detailBuf, grid 0.6 and 0: max |diff| < 1e-5.
  P3  NO-OPS -- amount 0, and an original identical to the image, change
      nothing (both runtimes).
  P4  IT RECOVERS -- a blurred copy of a detailed picture, with the picture
      as the original, comes back closer to it (PSNR up, contours up).
  P5  THE GRID DECIDES ITSELF -- a blocky source reads a high block ratio and
      gets the full grid, a smooth one reads ~1 and none; the ramp is linear
      from BLOCK_CLEAN to BLOCK_FULL.
  P6  THE REAL apply() -- same size: the output IS _detail_np at the measured
      grid; one source frame serves a batch; a size or frame-count mismatch
      is skipped with a note and the input comes back untouched; no source ->
      untouched.
  P7  WIRING -- canon slot 17 default 1.0 on both sides; the optional input;
      a live widget; the preview runs the detail BEFORE the automatic; on a
      shrunk proxy the preview uses no grid (driven).
  P8  MUTATIONS -- four wounds, each must go red.
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

# --- the Viewer's definition (pv/detail.py, v060), verbatim -----------------
VIEWER = r'''
import numpy as np
CORE = 0.002
CAP = 0.15
GRID = 0.6
BOX = 5
EPS = 1e-6
def _soft(rgb):
    out = rgb
    for axis in (0, 1):
        for _ in range(2):
            n = out.shape[axis]
            lo = np.take(out, np.r_[0, np.arange(n - 1)], axis=axis)
            hi = np.take(out, np.r_[np.arange(1, n), n - 1], axis=axis)
            out = (lo + 2.0 * out + hi) * 0.25
    return out
def _box(x, r=BOX // 2):
    k = 2 * r + 1
    p = np.pad(x.astype(np.float64), ((r, r), (r, r), (0, 0)), mode="edge")
    c = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    return ((c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k])
            / (k * k)).astype(np.float32)
def grid_mask(height, width):
    gx = np.ones(width, np.float32)
    gx[(np.arange(width) % 8 == 7) | (np.arange(width) % 8 == 0)] -= GRID
    gy = np.ones(height, np.float32)
    gy[(np.arange(height) % 8 == 7) | (np.arange(height) % 8 == 0)] -= GRID
    return gy[:, None] * gx[None, :]
def reference(orig, ai, amount):
    orig = np.asarray(orig, np.float32)
    ai = np.asarray(ai, np.float32)
    if amount <= 0.0:
        return ai.copy()
    ho = orig - _soft(orig)
    ha = ai - _soft(ai)
    ho = np.sign(ho) * np.clip(np.abs(ho) - CORE, 0.0, CAP)
    ho = ho * grid_mask(*orig.shape[:2])[..., None]
    num = _box((ho * ha).sum(2, keepdims=True))
    den = np.sqrt(np.maximum(_box((ho * ho).sum(2, keepdims=True))
                             * _box((ha * ha).sum(2, keepdims=True)),
                             0.0)) + EPS
    agree = np.clip(num / den, 0.0, 1.0)
    surplus = np.where(((ho * ha) > 0) & (np.abs(ho) > np.abs(ha)),
                       ho - ha, 0.0)
    return np.clip(ai + amount * agree * surplus, 0.0, 1.0).astype(
        np.float32)
'''
VENV = {}
exec(VIEWER, VENV)


def check(ok, msg):
    if not ok:
        FAILS.append(msg)


def load_py(src):
    spec = importlib.util.spec_from_loader("phf_det", loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__dict__["__file__"] = PY_PATH
    exec(compile(src, PY_PATH, "exec"), mod.__dict__)
    return mod


def lift_js(src, name):
    i = src.find("function " + name + "(")
    j = src.find("\n}", i)
    if i < 0 or j < 0:
        raise SystemExit("[test_v1001_filter_detail] FAIL: cannot lift " + name)
    return src[i:j + 2]


def js_consts(src):
    return "\n".join(ln for ln in src.splitlines() if re.match(r"const DETAIL_\w+ = ", ln))


def run_js(src, body):
    code = js_consts(src) + "\n" + lift_js(src, "_detailBuf") + "\n" + lift_js(src, "_pfDetail") + "\n" + body
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


rng = np.random.default_rng(1001)
H, W = 24, 40
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
TRUTH = np.clip(np.stack([0.5 + 0.3 * np.sin(xx * 1.3) * np.cos(yy * 0.9),
                          0.5 + 0.25 * np.sin(xx * 0.7 + yy * 1.1),
                          0.4 + 0.2 * np.cos(xx * 1.9)], -1)
                + rng.normal(0, 0.02, (H, W, 3)), 0, 1).astype(np.float32)


def blur(x):
    return VENV["_soft"](VENV["_soft"](x))


AI = np.clip(blur(TRUTH) * 1.03 + 0.01, 0, 1).astype(np.float32)


def fl(a):
    return [float(v) for v in np.asarray(a, np.float32).reshape(-1)]


BODY = """
const O = new Float32Array(%s), A = new Float32Array(%s), W = %d, H = %d;
const g6 = Array.from(_detailBuf(O, A, W, H, 1.5, 0.6));
const g0 = Array.from(_detailBuf(O, A, W, H, 1.5, 0));
const z = Array.from(_detailBuf(O, A, W, H, 0, 0.6));
const same = Array.from(_detailBuf(A, A, W, H, 2.0, 0.6));
const wd = (v) => [{ name: "detail_amount", value: v }];
const fakeData = { width: 40, height: 24 };
const full = _pfDetail({ widgets: wd(1), _pfDetailItem: {}, _pfDetailData: fakeData,
                         _pfSrcData: fakeData, _pfSrcW: 40, _pfDetailGrid: 0.45 });
const shrunk = _pfDetail({ widgets: wd(1), _pfDetailItem: {}, _pfDetailData: fakeData,
                           _pfSrcData: fakeData, _pfSrcW: 1344, _pfDetailGrid: 0.45 });
const none = _pfDetail({ widgets: wd(1), _pfDetailItem: null });
const size = _pfDetail({ widgets: wd(1), _pfDetailItem: {}, _pfDetailData: { width: 10, height: 10 },
                         _pfSrcData: fakeData, _pfSrcW: 40 });
console.log(JSON.stringify({ g6, g0, z, same, full, shrunk, none, size }));
""" % (json.dumps(fl(TRUTH)), json.dumps(fl(AI)), W, H)


def psnr(a, b):
    return 10 * np.log10(1.0 / max(float(np.mean((a - b) ** 2)), 1e-12))


def lap(x):
    l = x.mean(2)
    return float(np.abs(4 * l[1:-1, 1:-1] - l[:-2, 1:-1] - l[2:, 1:-1] - l[1:-1, :-2] - l[1:-1, 2:]).mean())


def evaluate(py_src, js_src):
    errs = []
    m = load_py(py_src)
    # P1
    for amt in (0.5, 1.0, 2.0):
        d = float(np.max(np.abs(m._detail_np(TRUTH, AI, amt, m.DETAIL_GRID) - VENV["reference"](TRUTH, AI, amt))))
        if d != 0.0:
            errs.append("P1 not the Viewer's definition at amount %.1f (%.2e)" % (amt, d))
    b = m._detail_np(np.stack([TRUTH, TRUTH * 0.8]), np.stack([AI, AI * 0.8]), 1.0, m.DETAIL_GRID)
    if float(np.max(np.abs(b[1] - VENV["reference"](TRUTH * 0.8, AI * 0.8, 1.0)))) != 0.0:
        errs.append("P1 batched frame differs from the definition")
    # P2 / P3
    res = run_js(js_src, BODY)
    if res is None:
        return errs + ["node driver failed"]
    for key, g in (("g6", 0.6), ("g0", 0.0)):
        d = float(np.max(np.abs(np.asarray(res[key]).reshape(H, W, 3) - m._detail_np(TRUTH, AI, 1.5, g))))
        if d > 1e-5:
            errs.append("P2 parity at grid %.1f: %.2e" % (g, d))
    if float(np.max(np.abs(np.asarray(res["z"]).reshape(H, W, 3) - AI))) > 0 \
            or float(np.max(np.abs(m._detail_np(TRUTH, AI, 0.0) - AI))) > 0:
        errs.append("P3 amount 0 changed the picture")
    if float(np.max(np.abs(np.asarray(res["same"]).reshape(H, W, 3) - AI))) > 1e-7 \
            or float(np.max(np.abs(m._detail_np(AI, AI, 2.0) - AI))) > 1e-7:
        errs.append("P3 an original identical to the image changed it")
    # P4
    out = m._detail_np(TRUTH, AI, 1.0, 0.0)
    if not (psnr(out, TRUTH) > psnr(AI, TRUTH) + 0.3 and lap(out) > lap(AI) * 1.1):
        errs.append("P4 no recovery: PSNR %.2f -> %.2f, contour %.4f -> %.4f"
                    % (psnr(AI, TRUTH), psnr(out, TRUTH), lap(AI), lap(out)))
    # P5
    # a clean picture varies in BOTH directions (a first draft varied only
    # across: every vertical step was 0 and the ratio read 0.5, not ~1.0 --
    # the BLOCK_CLEAN mutation survived on it)
    gy, gx = np.mgrid[0:64, 0:64].astype(np.float32)
    smooth = np.clip(0.5 + 0.2 * np.sin(gx * 0.21) * np.cos(gy * 0.17)
                     + rng.normal(0, 0.01, (64, 64)), 0, 1)[..., None] + np.zeros((64, 64, 3))
    blocky = smooth.copy()
    for by in range(0, 64, 8):
        for bx in range(0, 64, 8):
            blocky[by:by + 8, bx:bx + 8] += rng.uniform(-0.04, 0.04)
    rs, rb = m._block_ratio(smooth.astype(np.float32)), m._block_ratio(np.clip(blocky, 0, 1).astype(np.float32))
    if not (0.95 < rs < 1.05 and m._grid_strength(rs) == 0.0 and rb > 1.2 and m._grid_strength(rb) == m.DETAIL_GRID):
        errs.append("P5 block ratio smooth %.3f / blocky %.3f" % (rs, rb))
    if abs(m._grid_strength((m.BLOCK_CLEAN + m.BLOCK_FULL) / 2) - m.DETAIL_GRID / 2) > 1e-12:
        errs.append("P5 the grid ramp is not linear")
    # the measured bounds themselves (clean LIVE1 <= 1.045, JPEG q60 >= 1.094)
    if not (m.BLOCK_CLEAN == 1.05 and m.BLOCK_FULL == 1.20 and m._grid_strength(1.045) == 0.0
            and m._grid_strength(1.094) > 0.0):
        errs.append("P5 the measured bounds moved: %r / %r" % (m.BLOCK_CLEAN, m.BLOCK_FULL))
    # P6
    node = m.ULSFilter()
    kw = dict(exposure=0.0, temperature=0.0, tint=0.0, contrast=0.0, gamma=1.0,
              shadows=0.0, highlights=0.0, saturation=0.0, vibrance=0.0, hue_shift=0.0,
              lut_name="none", lut_strength=1.0, sharpen_amount=0.0, sharpen_radius=1.0,
              preset="none", auto_mode="off", sharpen_threshold=0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        o1 = node.apply(AI[None], detail_amount=1.2, detail_source=TRUTH[None], **kw)
        g = m._grid_strength(m._block_ratio(TRUTH[None]))
        want = m._grade_np(m._detail_np(TRUTH[None], AI[None], 1.2, g), 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
        if float(np.max(np.abs(np.asarray(o1["result"][0]) - want))) > 1e-6:
            errs.append("P6 apply() is not _detail_np at the measured grid")
        if o1["ui"]["ph_filter"][0].get("detail_grid") != g:
            errs.append("P6 the ui item does not carry the measured grid")
        batch = np.stack([AI, AI * 0.9, AI * 0.8])
        o2 = np.asarray(node.apply(batch, detail_amount=1.0, detail_source=TRUTH[None], **kw)["result"][0])
        if float(np.max(np.abs(o2[2] - m._grade_np(m._detail_np(TRUTH, AI * 0.8, 1.0, g), 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)))) > 1e-6:
            errs.append("P6 one source frame does not serve every frame")
        srcs = np.stack([TRUTH, np.flip(TRUTH, 1).copy(), np.flip(TRUTH, 0).copy()])
        o2b = np.asarray(node.apply(batch, detail_amount=1.0, detail_source=srcs, **kw)["result"][0])
        g2 = m._grid_strength(m._block_ratio(srcs))
        if float(np.max(np.abs(o2b[2] - m._grade_np(m._detail_np(srcs[2], AI * 0.8, 1.0, g2), 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)))) > 1e-6:
            errs.append("P6 a per-frame source must give frame i its own source frame")
        # the torch path is the one ComfyUI runs (chunked, 32 frames; the
        # source sliced per chunk) -- numpy input takes the other branch
        try:
            import torch
        except Exception:
            torch = None
        if torch is None:
            errs.append("P6 torch missing: the chunked ComfyUI path cannot be driven")
        else:
            big = np.concatenate([batch] * 12)                       # 36 frames: two chunks
            bsrc = np.concatenate([srcs] * 12)
            ot = node.apply(torch.from_numpy(big), detail_amount=1.0,
                            detail_source=torch.from_numpy(bsrc), **kw)["result"][0].numpy()
            gt = m._grid_strength(m._block_ratio(bsrc))
            for k in (0, 2, 33, 35):
                w = m._grade_np(m._detail_np(bsrc[k], big[k], 1.0, gt), 0, 0, 0, 0, 1, 0, 0, 0, 0, 0)
                if float(np.max(np.abs(ot[k] - w))) > 1e-6:
                    errs.append("P6 torch path: frame %d did not get its own source frame" % k)
                    break
        o3 = node.apply(batch, detail_amount=1.0, detail_source=TRUTH[None, :20], **kw)
        if o3["result"][0] is not batch or "sizes must match" not in o3["ui"]["ph_filter"][0].get("detail_note", ""):
            errs.append("P6 a size mismatch must skip with a note and leave the input untouched")
        o4 = node.apply(batch, detail_amount=1.0, detail_source=np.stack([TRUTH, TRUTH]), **kw)
        if o4["result"][0] is not batch or "frames" not in o4["ui"]["ph_filter"][0].get("detail_note", ""):
            errs.append("P6 a frame-count mismatch must skip with a note")
        o5 = node.apply(batch, detail_amount=1.0, **kw)
        if o5["result"][0] is not batch:
            errs.append("P6 no source must leave the input untouched")
    # P7 (driven part)
    if not (res["full"]["state"] == "ready" and res["full"]["grid"] == 0.45
            and res["shrunk"]["grid"] == 0 and res["none"]["state"] == "none"
            and res["size"]["state"] == "size"):
        errs.append("P7 _pfDetail states: %r" % ({k: res[k] for k in ("full", "shrunk", "none", "size")},))
    if m.FILTER_CANON[17] != "detail_amount":
        errs.append("P7 canon slot 17")
    return errs


FAILS.extend(evaluate(PY_SRC, JS_SRC))
check("sharpen_threshold: 0.0, detail_amount: 1.0," in JS_SRC, "P7 CANON_DEFAULTS detail_amount 1.0")
check('"detail_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0,' in PY_SRC, "P7 widget default 1.0")
check('"optional": {\n                "detail_source": ("IMAGE",' in PY_SRC, "P7 the optional detail_source input")
check(re.search(r'const LIVE_WIDGETS = GRADE_WIDGETS\.concat\(\[[^\]]*"detail_amount"', JS_SRC) is not None,
      "P7 detail_amount must be a live preview widget")
i_det = JS_SRC.find("b0 = _detailBuf(ob, b0, src.width, src.height, det.amount, det.grid);")
i_auto = JS_SRC.find("const s0 = _autoRGB(b0[f], b0[f + 1], b0[f + 2], auto);")
check(0 < i_det < i_auto, "P7 the preview must run the detail BEFORE the automatic")
check("_pfLoadDetail(node, item);" in JS_SRC, "P7 the detail proxy must load with the preview")

MUT = [
    ("JS surplus without the 'further' test", "js",
     "(ho[i] * ha[i] > 0 && Math.abs(ho[i]) > Math.abs(ha[i])) ? ho[i] - ha[i] : 0;",
     "(ho[i] * ha[i] > 0) ? ho[i] - ha[i] : 0;"),
    ("PY core not subtracted", "py",
     "ho = np.sign(ho) * np.clip(np.abs(ho) - DETAIL_CORE, 0.0, DETAIL_CAP)",
     "ho = np.sign(ho) * np.clip(np.abs(ho), 0.0, DETAIL_CAP)"),
    ("PY grid ignored", "py",
     "    if float(grid) > 0.0:\n        ho = ho * _grid_mask_np(",
     "    if float(grid) > 9.0:\n        ho = ho * _grid_mask_np("),
    ("JS box 3x3", "js", "for (let dy = -2; dy <= 2; dy++) {", "for (let dy = -1; dy <= 1; dy++) {"),
]
caught = 0
for name, side, a, b in MUT:
    src = JS_SRC if side == "js" else PY_SRC
    if src.count(a) != 1:
        FAILS.append("P8 mutation anchor not unique: " + name)
        continue
    r = evaluate(PY_SRC, JS_SRC.replace(a, b)) if side == "js" else evaluate(PY_SRC.replace(a, b), JS_SRC)
    if r:
        caught += 1
    else:
        FAILS.append("P8 mutation survived: " + name)

if FAILS:
    for f in FAILS:
        print("[test_v1001_filter_detail] FAIL: " + f)
    sys.exit(1)
print("[test_v1001_filter_detail] OK -- 8 promises, the Viewer's definition exact, py/js parity, %d/4 mutations caught" % caught)
