#!/usr/bin/env python3
"""
test_v628_filter_preview -- Polyhedron Filter: frozen canon, preview
channel, neutral fast path, and frontend discipline.

Born with the stage-0 build (frozen canon + preview pipe + before/after
divider machinery, proven before any math depended on them); the pins keep
holding through the later stages. The color math itself is guarded by
test_v630_grade_parity. Invariants pinned here:

  STATIC (py) -- FILTER_CANON matches the frozen order exactly; the required
            INPUT_TYPES block declares the widgets in that same order; the ui
            payload uses the QUOTED key "ph_filter" (substring-trap lesson);
            the neutral fast path returns the input tensor untouched.
  STATIC (js) -- the DOM widget is serialize:false; the file contains NO
            ResizeObserver, NO prototype.onResize hook and NO document-level
            listeners (v624 leak class); pointer capture is present.
  DRIVEN (py) -- _preview_size lifted and executed: downscale-only long-edge
            fit to 768.
  DRIVEN (js) -- _clampFrac and _fitRect lifted and executed in node:
            clamping and centered letterbox math.

Each driven check is mutation-tested: the wound is injected into the lifted
source and the catch is proven. A guard that never fails is
indistinguishable from one that cannot fail.
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_SRC = open(os.path.join(ROOT, "nodes", "ph_filter.py"), encoding="utf-8").read()
JS_SRC = open(os.path.join(ROOT, "web", "js", "ph_filter.js"), encoding="utf-8").read()

CANON = ("exposure", "temperature", "tint", "contrast", "gamma", "shadows",
         "highlights", "saturation", "vibrance", "hue_shift", "lut_name",
         "lut_strength", "sharpen_amount", "sharpen_radius", "preset")


def _fail(msg):
    print("[test_v628_filter_preview] FAIL: " + msg)
    sys.exit(1)


# ---------------------------------------------------------------------------
# lifting helpers (house doctrine: run the code, do not read it)
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


def _run_js(code):
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        r = subprocess.run(["node", path], capture_output=True, text=True, timeout=30)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# STATIC (py)
# ---------------------------------------------------------------------------
m = re.search(r"FILTER_CANON\s*=\s*\((.*?)\)", PY_SRC, re.S)
if not m:
    _fail("FILTER_CANON tuple not found")
canon_decl = tuple(re.findall(r'"(\w+)"', m.group(1)))
if canon_decl != CANON:
    _fail("FILTER_CANON drifted: %r" % (canon_decl,))

# The required INPUT_TYPES block must declare the canon widgets in canon
# order (positional widgets_values law). Extract quoted keys that open a
# widget tuple inside the required block.
mreq = re.search(r'"required"\s*:\s*\{(.*?)\n\s*\}\s*,?\s*\n\s*\}', PY_SRC, re.S)
if not mreq:
    _fail("required INPUT_TYPES block not found")
keys = [k for k in re.findall(r'\n\s*"(\w+)"\s*:\s*\(', mreq.group(1))]
if keys and keys[0] == "image":
    keys = keys[1:]  # the image input precedes the widget canon
if tuple(keys) != CANON:
    _fail("INPUT_TYPES widget order drifted from canon: %r" % (keys,))

if '"ph_filter"' not in PY_SRC:
    _fail('quoted ui key "ph_filter" missing (substring-trap lesson: pin the QUOTED key)')
if '"result": (image,)' not in PY_SRC:
    _fail("neutral fast path missing: all-neutral controls must return the "
          "input tensor untouched (no numpy roundtrip)")

# ---------------------------------------------------------------------------
# STATIC (js)
# ---------------------------------------------------------------------------
if "serialize: false" not in JS_SRC:
    _fail("DOM widget must be serialize:false")
if "new ResizeObserver" in JS_SRC:
    _fail("ResizeObserver is banned in this pane (loop source class)")
if "prototype.onResize" in JS_SRC:
    _fail("onResize hook not allowed in this pane (loop source class)")
if "document.addEventListener" in JS_SRC:
    _fail("document-level listeners are the v624 leak class -- keep events on the canvas")
if "setPointerCapture" not in JS_SRC:
    _fail("pointer capture missing (divider drag must stay on the canvas)")
if "node.setSize(node.computeSize())" in JS_SRC:
    _fail("full-geometry reset wound present: setSize(computeSize()) snaps the "
          "node width back to LiteGraph's minimum on every run (re-run collapse)")
if "node.setSize([node.size[0]," not in JS_SRC:
    _fail("width-preserving height adjust missing: geometry writes must keep node.size[0]")

# ---------------------------------------------------------------------------
# DRIVEN (py): _preview_size
# ---------------------------------------------------------------------------
def _drive_preview_size(src_snippet):
    env = {}
    exec("PREVIEW_MAX_EDGE = 768\n" + src_snippet, env)
    f = env["_preview_size"]
    ok = (f(1536, 768) == (768, 384)
          and f(768, 1536) == (384, 768)
          and f(100, 100) == (100, 100)      # never upscale
          and f(768, 768) == (768, 768))     # boundary stays put
    return ok


lifted = _lift_pyfunc(PY_SRC, "_preview_size")
if not _drive_preview_size(lifted):
    _fail("_preview_size drove wrong values")

# mutation: widen the max edge -> downscale cases must break
mut = lifted.replace("max_edge=PREVIEW_MAX_EDGE", "max_edge=PREVIEW_MAX_EDGE * 10")
if mut == lifted:
    _fail("preview-size mutation did not apply")
if _drive_preview_size(mut):
    _fail("MUTATION NOT CAUGHT: max-edge wound survived the driven check")

# ---------------------------------------------------------------------------
# DRIVEN (js): _clampFrac + _fitRect
# ---------------------------------------------------------------------------
js_clamp = _lift_jsfunc(JS_SRC, "_clampFrac")
js_fit = _lift_jsfunc(JS_SRC, "_fitRect")
js_pane = _lift_jsfunc(JS_SRC, "_paneHeight")
JS_CONSTS = ("const PREVIEW_MIN_H = 140;\nconst PREVIEW_MAX_H = 900;\n"
             "const HEADER_H = 52;\nconst PANE_PAD = 8;\n")
driver = JS_CONSTS + """
%s
%s
%s
function eq(a, b) { return JSON.stringify(a) === JSON.stringify(b); }
let ok = true;
ok = ok && _clampFrac(-0.5) === 0;
ok = ok && _clampFrac(1.5) === 1;
ok = ok && _clampFrac(0.3) === 0.3;
ok = ok && Number.isNaN(NaN) && _clampFrac(NaN) === 0;
ok = ok && eq(_fitRect(1600, 900, 400, 300), { x: 0, y: 37, w: 400, h: 225 });
ok = ok && eq(_fitRect(900, 1600, 400, 300), { x: 116, y: 0, w: 168, h: 300 });
ok = ok && eq(_fitRect(0, 100, 400, 300), { x: 0, y: 0, w: 0, h: 0 });
ok = ok && _paneHeight(1000, 1000, 316) === 368;   // square at 300 inner + chrome
ok = ok && _paneHeight(2000, 1000, 316) === 218;   // 2:1 halves the image height
ok = ok && _paneHeight(1000, 4000, 316) === 900;   // tall portrait hits the ceiling
ok = ok && _paneHeight(4000, 1000, 66) === 140;    // tiny node hits the floor
console.log(ok ? "JSPASS" : "JSFAIL");
"""
rc, out = _run_js(driver % (js_clamp, js_fit, js_pane))
if rc != 0 or "JSPASS" not in out:
    _fail("js driven checks failed: %s" % out.strip()[:200])

# mutation: break the clamp lower bound -> driven must fail
mut_clamp = js_clamp.replace("if (!(f >= 0)) return 0;", "if (!(f >= 1)) return 0;")
if mut_clamp == js_clamp:
    _fail("js clamp mutation did not apply")
rc, out = _run_js(driver % (mut_clamp, js_fit, js_pane))
if "JSPASS" in out:
    _fail("MUTATION NOT CAUGHT: js clamp wound survived the driven check")

# mutation: uncenter the letterbox -> fit values must break
mut_fit = js_fit.replace("Math.floor((cw - w) / 2)", "0")
if mut_fit == js_fit:
    _fail("js fit mutation did not apply")
rc, out = _run_js(driver % (js_clamp, mut_fit, js_pane))
if "JSPASS" in out:
    _fail("MUTATION NOT CAUGHT: js letterbox wound survived the driven check")

# mutation: ignore the pane padding -> pane heights must break
mut_pane = js_pane.replace("nodeW - 2 * PANE_PAD", "nodeW")
if mut_pane == js_pane:
    _fail("js pane mutation did not apply")
rc, out = _run_js(driver % (js_clamp, js_fit, mut_pane))
if "JSPASS" in out:
    _fail("MUTATION NOT CAUGHT: js pane-height wound survived the driven check")

print("[test_v628_filter_preview] PASS: canon frozen, preview channel pinned, "
      "pane discipline held, 4 mutations caught")
