#!/usr/bin/env python3
"""
test_v622_keep_height -- the stored field height is the source of truth; an unlaid-out
refit must KEEP it, never re-measure it.

v620 deferred every programmatic refit through requestAnimationFrame so it would measure
AFTER layout. In Frank's frontend that defer did not fully hold: a refit still fired while
the textarea was unlaid-out, measured a collapsed scrollHeight, and squeezed an
already-correct field on RUN and RELOAD (his workaround -- touch/empty/refill a field --
worked precisely because typing happens while the element IS laid out).

v622 stops trusting an untrustworthy measurement. In _refit, a field's height is re-measured
ONLY when the element is actually laid out (a real offsetParent AND a non-zero clientWidth).
When it is not -- exactly the run/reload-before-layout moment -- _refit KEEPS the stored
_plsH instead of overwriting it. The measurement still happens on a live keystroke (laid out)
and on any laid-out refit, so the field still grows to its content; it just can no longer
collapse when measured blind.

Two guards, both must hold, mutation-tested (inject the wound, prove the catch):

  STATIC -- _refit carries the laid-out signal (offsetParent AND clientWidth). If the signal
            is gone, every refit measures unconditionally again -- the v619/v620 regression.

  DRIVEN -- lift and RUN _refit against a togglable-layout mock:
              * an already-expanded field (_plsH = 300) refit while UNLAID keeps _plsH = 300
                and does NOT collapse (Frank's exact run/reload moment);
              * the same field refit while LAID OUT measures to its new content height.
            Mutating the guard away (if (laidOut) -> if (true)) makes the UNLAID refit collapse
            the field to the floor -- caught here.
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


def _fail(msg):
    print("[test_v622_keep_height] FAIL: " + msg)
    sys.exit(1)


def _lift(sig):
    s = JS[JS.index(sig):]
    return s[:s.index("\n}") + 2]


def _const(name):
    m = re.search(r"^const " + re.escape(name) + r" = .*?;", JS, re.M)
    if not m:
        _fail("could not lift source constant `%s`" % name)
    return m.group(0)


# ---------------------------------------------------------------------------
# STATIC -- the laid-out signal must be present in _refit.
# ---------------------------------------------------------------------------
if "function _refit(node)" not in JS:
    _fail("_refit is gone")
REFIT = _lift("function _refit(node)")
if "offsetParent" not in REFIT or "clientWidth" not in REFIT:
    _fail("_refit no longer tests whether the field is laid out (offsetParent + clientWidth) "
          "before measuring -- an unlaid-out measurement will squeeze the field again")


# ---------------------------------------------------------------------------
# DRIVEN -- mock with togglable layout; run the REAL _refit, then a MUTANT.
# ---------------------------------------------------------------------------
BODY = """
// togglable layout: an unlaid-out textarea reports offsetParent=null, clientWidth=0, and a
// collapsed scrollHeight; once laid out it reports a parent, a real width, and CONTENT_H.
let LAID = false;
let CONTENT_H = 300;
const FLOOR_SCROLL = 8;   // an unlaid-out textarea (< FIELD_MIN_H -> clamps up to the floor)
function mkEl(startH) {
    return { style: { height: startH + "px", display: "" },
             get scrollHeight() { return LAID ? CONTENT_H : FLOOR_SCROLL; },
             get offsetParent() { return LAID ? {} : null; },
             get clientWidth() { return LAID ? 400 : 0; } };
}
function mkNode() {
    const pos1 = { name: "pos_1", value: "x ".repeat(120), element: mkEl(300), _plsH: 300 };
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

const out = {};
{   // UNLAID refit -- the run/reload-before-layout moment. Must KEEP the stored height.
    const r = mkNode();
    LAID = false;
    _refit(r.nd);
    out.keptPlsH = r.pos1._plsH;
    out.keptElH  = parseInt(r.pos1.element.style.height, 10);
}
{   // LAID-OUT refit with new content -- must MEASURE to it.
    const r = mkNode();
    LAID = true; CONTENT_H = 520;
    _refit(r.nd);
    out.measuredPlsH = r.pos1._plsH;
    out.measuredElH  = parseInt(r.pos1.element.style.height, 10);
}
console.log(JSON.stringify(out));
"""

PRELUDE = (
    'function _w(node, name){ return (node.widgets||[]).find(w => w.name === name); }\n'
    'const HIDDEN_PREFIX = "__pls_hidden__";\n'
)


def run(refit_src, label):
    src = (
        _const("FIELD_MIN_H") + "\n" + _const("FIELD_MAX_H") + "\n"
        + _const("FIELD_DEF_H") + "\n" + _const("NEG_GAP") + "\n"
        + _const("FIELD_NAMES") + "\n"
        + PRELUDE
        + _lift("function _visibleFields(node)") + "\n"
        + _lift("function _contentH(w)") + "\n"
        + refit_src + "\n"
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
        _fail("%s harness did not run: %s" % (label, res.stderr.strip()[:400]))
    return json.loads(res.stdout.strip().splitlines()[-1])


# 1) REAL _refit -- keep when unlaid, measure when laid out.
real = run(REFIT, "real")
if real["keptPlsH"] != 300:
    _fail("an UNLAID refit changed the stored height (_plsH %s, expected 300 kept) -- the "
          "run/reload squeeze is back" % real["keptPlsH"])
if real["keptElH"] < 200:
    _fail("an UNLAID refit collapsed the field element (height %spx) instead of keeping it"
          % real["keptElH"])
if real["measuredPlsH"] < 500:
    _fail("a LAID-OUT refit did not measure to new content (_plsH %s, expected ~520) -- "
          "keeping must not mean never measuring" % real["measuredPlsH"])
if real["measuredElH"] < 500:
    _fail("a LAID-OUT refit did not grow the field element to content (height %spx)"
          % real["measuredElH"])

# 2) MUTANT -- remove the guard (always measure); the UNLAID refit must now collapse.
MUT = REFIT.replace("if (laidOut) {", "if (true) {")
if MUT == REFIT:
    _fail("mutation target 'if (laidOut) {' not found -- harness out of sync with _refit")
mut = run(MUT, "mutant")
if mut["keptPlsH"] >= 200:
    _fail("MUTATION NOT CAUGHT: with the laid-out guard removed, an UNLAID refit still did not "
          "collapse the field (_plsH %s) -- the DRIVEN check does not prove the guard is what "
          "prevents the squeeze" % mut["keptPlsH"])

print("[test_v622_keep_height] PASS -- unlaid refit keeps _plsH (300), laid-out refit measures "
      "(~520); removing the laid-out guard collapses the unlaid refit (mutation caught)")
sys.exit(0)
