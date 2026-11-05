#!/usr/bin/env python3
"""
test_v1229_first_run_fit -- the CTE must spread open by itself once it is on screen.

Frank, 26.10.: "the CTE often does not spread open correctly on the first run and keeps
text hidden". The chain behind it: every programmatic refit (load, run, wiring) is deferred
one frame (v620) and, when a field is not laid out at that moment, KEEPS its stored height
instead of measuring blind (v622). Both stay right. What was missing is the way back: a node
that was off screen at that frame, or whose DOM fields the frontend mounted later, kept the
stored height until the user typed -- and a stored height from SHORTER text (a saved
workflow, a prompt changed outside the field) clips the rest under overflow:hidden.

v1229: _refit records the debt (_plsFitPending) when it kept a stored height, and the
node's own onDrawForeground (called only while the node is on screen) pays it through
_retryPendingFit -- one deferred refit per frame, at most FIT_RETRY_MAX per episode.

STATIC -- onDrawForeground calls _retryPendingFit(this); _refitNextFrame opens a fresh
          budget; _refit sets _plsFitPending.

DRIVEN -- lift the REAL _refit, _refitNextFrame and _retryPendingFit and run them in node:
  1. load: stale stored height 126, real content 420, field NOT laid out -> the deferred
     refit keeps 126 (v622 holds) and leaves the debt open;
  2. the node comes on screen (laid out) and is drawn -> the field measures to 420,
     the debt is cleared, and further draws schedule nothing;
  3. a field that can NEVER be measured: 500 draws cost at most FIT_RETRY_MAX refits.

MUTATIONS (each must turn the run red):
  M1 -- _refit never records the debt          -> step 2 stays clipped at 126
  M2 -- the retry budget check removed          -> step 3 refits on every draw
  M3 -- _retryPendingFit ignores the debt flag  -> step 2 keeps scheduling after the fix
"""
import os
import re
import sys
import json
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JS = open(os.path.join(ROOT, "web/js/ph_clip_encode.js"), encoding="utf-8").read()
TAG = "[test_v1229_first_run_fit]"


def _fail(msg):
    print(TAG + " FAIL: " + msg)
    sys.exit(1)


def _lift(sig, src=None):
    s = (src or JS)
    s = s[s.index(sig):]
    return s[:s.index("\n}") + 2]


def _const(name):
    m = re.search(r"^const " + re.escape(name) + r" = .*?;", JS, re.M)
    if not m:
        _fail("could not lift source constant `%s`" % name)
    return m.group(0)


# ---------------------------------------------------------------------------
# STATIC
# ---------------------------------------------------------------------------
for sig in ("function _refit(node)", "function _refitNextFrame(node)", "function _retryPendingFit(node)"):
    if sig not in JS:
        _fail("%s is gone" % sig)
REFIT = _lift("function _refit(node)")
NEXT = _lift("function _refitNextFrame(node)")
RETRY = _lift("function _retryPendingFit(node)")
if "node._plsFitPending = !allMeasured;" not in REFIT:
    _fail("_refit no longer records the measurement debt (_plsFitPending)")
if "_plsFitTries = 0" not in NEXT:
    _fail("_refitNextFrame no longer opens a fresh retry budget")
dfg = JS[JS.index("nodeType.prototype.onDrawForeground = function"):]
dfg = dfg[:dfg.index("\n        };")]
if "_retryPendingFit(this)" not in dfg:
    _fail("onDrawForeground no longer pays the pending measurement -- the first-run clip is back")
if "offsetParent" not in REFIT or "clientWidth" not in REFIT:
    _fail("_refit lost the v622 laid-out signal")


BODY = r"""
const vueFieldFor = () => null;
function _fitExt(node) {}
let Q = [];
globalThis.requestAnimationFrame = (fn) => { Q.push(fn); return Q.length; };
function flush() { const q = Q; Q = []; for (const fn of q) fn(); }
let LAID = false, CONTENT_H = 420;
function mkEl(h) {
    return { style: { height: h + "px", display: "" },
             get scrollHeight() { return LAID ? CONTENT_H : 8; },
             get offsetParent() { return LAID ? {} : null; },
             get clientWidth() { return LAID ? 400 : 0; } };
}
let REFITS = 0;
function mkNode(stored) {
    const pos1 = { name: "pos_1", value: "x ".repeat(200), element: mkEl(stored), _plsH: stored };
    const nd = {
        widgets: [ { name: "segments", value: 1 }, pos1 ],
        size: [400, 400],
        setSize([w, h]) { this.size = [w, h]; },
        setDirtyCanvas() {},
        computeSize() {
            let cur = 0;
            for (const w of _visibleFields(this)) cur += (w._plsH || FIELD_DEF_H);
            return [400, 100 + cur];
        },
    };
    return { nd, pos1 };
}
const _realRefit = _refit;
_refit = function (n) { REFITS++; return _realRefit(n); };
const draw = (n) => { _retryPendingFit(n); flush(); };

const out = {};
{   // 1 + 2: load off screen, then come on screen
    const r = mkNode(126);
    LAID = false; CONTENT_H = 420;
    _refitNextFrame(r.nd); flush();
    out.loadKept = r.pos1._plsH;
    out.loadPending = !!r.nd._plsFitPending;
    LAID = true;
    draw(r.nd);
    out.drawnH = r.pos1._plsH;
    out.drawnPending = !!r.nd._plsFitPending;
    REFITS = 0;
    for (let i = 0; i < 20; i++) draw(r.nd);
    out.afterFixRefits = REFITS;
}
{   // 3: never measurable -- bounded
    const r = mkNode(126);
    LAID = false;
    _refitNextFrame(r.nd); flush();
    REFITS = 0;
    for (let i = 0; i < 500; i++) draw(r.nd);
    out.neverRefits = REFITS;
}
console.log(JSON.stringify(out));
"""

PRELUDE = (
    'function _w(node, name){ return (node.widgets||[]).find(w => w.name === name); }\n'
    'const HIDDEN_PREFIX = "__pls_hidden__";\n'
)


def run(refit_src, retry_src, label):
    src = (
        _const("FIELD_MIN_H") + "\n" + _const("FIELD_MAX_H") + "\n"
        + _const("FIELD_DEF_H") + "\n" + _const("NEG_GAP") + "\n"
        + _const("FIELD_NAMES") + "\n" + _const("FIT_RETRY_MAX") + "\n"
        + PRELUDE
        + _lift("function _visibleFields(node)") + "\n"
        + _lift("function _contentH(w, el)") + "\n"
        + refit_src.replace("function _refit(node)", "let _refit = function (node)", 1) + ";\n"
        + NEXT + "\n" + retry_src + "\n"
        + BODY
    )
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as fh:
        fh.write(src)
        path = fh.name
    try:
        res = subprocess.run(["node", path], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    if res.returncode != 0:
        _fail("%s harness did not run: %s" % (label, res.stderr.strip()[:500]))
    return json.loads(res.stdout.strip().splitlines()[-1])


cap = int(re.search(r"const FIT_RETRY_MAX = (\d+);", JS).group(1))
real = run(REFIT, RETRY, "real")
if real["loadKept"] != 126:
    _fail("the off-screen load refit did not keep the stored height (%s) -- v622 broken" % real["loadKept"])
if not real["loadPending"]:
    _fail("the off-screen load refit left no measurement debt")
if real["drawnH"] < 400:
    _fail("drawing the node on screen did not spread the field to its text (_plsH %s, want ~420) "
          "-- the first-run clip" % real["drawnH"])
if real["drawnPending"]:
    _fail("the debt stayed open after a real measurement")
if real["afterFixRefits"] != 0:
    _fail("draws after the fix still scheduled %d refits" % real["afterFixRefits"])
if not (1 <= real["neverRefits"] <= cap + 1):
    _fail("a never-measurable field cost %d refits over 500 draws (cap %d)" % (real["neverRefits"], cap))

# --- mutations ---
m1 = run(REFIT.replace("node._plsFitPending = !allMeasured;", "node._plsFitPending = false;"), RETRY, "M1")
if m1["drawnH"] >= 400:
    _fail("MUTATION M1 NOT CAUGHT: without the recorded debt the field still spread open")
m2_src = RETRY.replace("if ((node._plsFitTries || 0) >= FIT_RETRY_MAX) return false;", "")
if m2_src == RETRY:
    _fail("mutation target for M2 not found")
m2 = run(REFIT, m2_src, "M2")
if m2["neverRefits"] <= cap + 1:
    _fail("MUTATION M2 NOT CAUGHT: without the budget a never-measurable field stayed bounded (%d)" % m2["neverRefits"])
m3_src = RETRY.replace("!node._plsFitPending || ", "")
if m3_src == RETRY:
    _fail("mutation target for M3 not found")
m3 = run(REFIT, m3_src, "M3")
if m3["afterFixRefits"] == 0:
    _fail("MUTATION M3 NOT CAUGHT: ignoring the debt flag did not keep scheduling after the fix")

print(TAG + " PASS -- off-screen load keeps 126 and owes a measurement; the first on-screen draw "
      "spreads the field to ~420 and clears the debt; a never-measurable field costs <= %d refits; "
      "M1/M2/M3 caught" % (cap + 1))
sys.exit(0)
