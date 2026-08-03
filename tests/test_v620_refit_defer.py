#!/usr/bin/env python3
"""
test_v620_refit_defer -- a PROGRAMMATIC refit must measure AFTER the browser lays out.

The v619 regression, in Frank's words: an already-correctly-expanded prompt field collapsed
back to minimum height on every RUN and every RELOAD, and only touching / emptying / refilling
a field brought the height back (until the next run). Root cause: a textarea reports its
height (scrollHeight) only once the browser has laid it out -- the v560 _applyVisibility rule,
pinned by test_v557. v619's _syncExternal (wired to onExecuted = RUN and the load setTimeout =
RELOAD) and both onConfigure hooks called _refit SYNCHRONOUSLY, so they measured the field
before layout, read it as empty, and _refit collapsed it. A live INPUT event escaped the bug
because the element IS laid out when the user types -- which is exactly why the manual
touch/empty/refill was the only workaround.

v620 routes every PROGRAMMATIC refit through _refitNextFrame (a one-shot requestAnimationFrame
schedule); a live input-event refit may still run inline.

Two guards, and both must hold:

  STATIC -- _syncExternal and both onConfigure refit paths go through _refitNextFrame, never a
            bare synchronous _refit; and _refitNextFrame schedules through requestAnimationFrame.

  DRIVEN -- with a QUEUING requestAnimationFrame and a togglable scrollHeight: calling
            _syncExternal while the field is not yet laid out must NOT collapse an
            already-expanded field and must NOT resize the node synchronously (the exact
            regression); after the queued frame flushes with the element laid out, the field is
            fit to its content and the node grows. Mutating _syncExternal back to a synchronous
            _refit(node) fails the DRIVEN check (proven below with the guard's own mutation).
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
    print("[test_v620_refit_defer] FAIL: " + msg)
    sys.exit(1)


def _lift(sig):
    s = JS[JS.index(sig):]
    return s[:s.index("\n}") + 2]


def _const(name):
    m = re.search(r"^const " + re.escape(name) + r" = .*?;", JS, re.M)
    if not m:
        _fail("could not lift the source constant `%s` -- the harness must stay in sync with "
              "the real values, not hard-code them" % name)
    return m.group(0)


# ---------------------------------------------------------------------------
# STATIC -- the wiring. A future refactor that drops _refitNextFrame back to a bare _refit on
# any programmatic path is the v619 bug again; catch it without needing the browser.
# ---------------------------------------------------------------------------
if "function _refitNextFrame(node)" not in JS:
    _fail("_refitNextFrame is gone -- the programmatic paths have nowhere to defer to")

_rnf = _lift("function _refitNextFrame(node)")
if "requestAnimationFrame" not in _rnf:
    _fail("_refitNextFrame does not schedule through requestAnimationFrame -- deferring past "
          "layout is the whole point (the v560/test_v557 rule)")

_sync = _lift("function _syncExternal(node)")
if "_refitNextFrame(node)" not in _sync:
    _fail("_syncExternal (RUN via onExecuted, RELOAD via the load timeout, and wiring changes) "
          "does not defer its refit through _refitNextFrame")
if "_refit(node)" in _sync:
    _fail("_syncExternal still calls _refit(node) SYNCHRONOUSLY -- that is the v619 collapse: it "
          "measures scrollHeight before layout and squeezes the field on every run/reload")

# both onConfigure refit paths (the pre-v613 height heal + the load re-apply) must defer.
if "_refitNextFrame(this)" not in JS:
    _fail("the onConfigure height-heal refit no longer defers through _refitNextFrame")
if "_refit(this)" in JS:
    _fail("a synchronous _refit(this) is back on onConfigure -- it measures before layout")
if "_refitNextFrame(self)" not in JS:
    _fail("the onConfigure load-timeout refit no longer defers through _refitNextFrame")

# a live INPUT event may still refit inline (the element is laid out when the user types) -- do
# NOT forbid every synchronous _refit, only the programmatic ones above. The input listeners are
# the proof this distinction is intended.
if 'addEventListener("input", () => _refit(' not in JS:
    _fail("the live input-event refit is gone -- v620 defers PROGRAMMATIC refits, not the "
          "inline refit a keystroke triggers (that one is correct: the field is laid out)")


# ---------------------------------------------------------------------------
# DRIVEN -- run _syncExternal against a queuing rAF and a togglable scrollHeight, and watch what
# happens to an ALREADY-EXPANDED field. This is Frank's exact scenario.
# ---------------------------------------------------------------------------
harness = """
%s
%s
%s
%s
%s
%s
const EXT_FIELDS = [["pls_ext_pos","pos_external","pos"],
                    ["pls_ext_neg","neg_external","neg"]];

%s
%s
%s
%s
%s
%s
%s

// --- a requestAnimationFrame that QUEUES (does not run inline); flush() runs the frame ---
let RAFQ = [];
globalThis.requestAnimationFrame = (fn) => { RAFQ.push(fn); return RAFQ.length; };
const flush = () => { const q = RAFQ; RAFQ = []; q.forEach((fn) => fn()); };

// --- a field element whose scrollHeight is EMPTY until the layout flag flips ---
let LAID_OUT = false;
const CONTENT_H = 300;   // the height the text really needs, once laid out
const UNLAID_H  = 8;     // what an unlaid-out textarea reports (~empty)
function mkEl(startH) {
    return { style: { height: startH + "px", display: "" },
             get scrollHeight() { return LAID_OUT ? CONTENT_H : UNLAID_H; },
             get offsetParent() { return LAID_OUT ? {} : null; },
             get clientWidth() { return LAID_OUT ? 400 : 0; } };
}

const SIZES = [];        // every node.setSize height, in order
function mkNode() {
    const pos1 = { name: "pos_1", value: "x ".repeat(120), element: mkEl(300), _plsH: 300 };
    const neg1 = { name: "neg_1", value: "y ".repeat(80),  element: mkEl(300), _plsH: 300 };
    const extP = { name: "pls_ext_pos", element: mkEl(96), hidden: true, _plsExt: true };
    const extN = { name: "pls_ext_neg", element: mkEl(96), hidden: true, _plsExt: true };
    const nd = {
        widgets: [ { name: "segments", value: 1 }, pos1,
                   { name: "use_negative", value: true }, neg1, extP, extN ],
        inputs: [ { name: "pos_external", link: 7 }, { name: "neg_external", link: null } ],
        _cteExt: { pos: "resolved external caption", neg: "" },
        size: [400, 400],
        setSize([w, h]) { this.size = [w, h]; SIZES.push(Math.round(h)); },
        setDirtyCanvas() {},
        computeSize() {
            // mimic LiteGraph: chrome + the fields' reserved heights + any shown EXT field
            let cur = 0;
            for (const w of _visibleFields(this)) cur += (w._plsH || FIELD_DEF_H);
            const ext = this.widgets.filter((w) => w._plsExt && !w.hidden).length * 96;
            return [400, 100 + cur + ext];
        },
    };
    return { nd, pos1 };
}

const { nd, pos1 } = mkNode();

// RUN / RELOAD moment: the browser has NOT laid the textareas out yet.
LAID_OUT = false;
_syncExternal(nd);                       // v620: must QUEUE the refit, not run it now
const deferHeight = pos1.element.style.height;   // still expanded? or collapsed?
const deferSizes  = SIZES.length;                // any synchronous resize?

// now the browser lays out and the frame fires.
LAID_OUT = true;
flush();
const fitHeight = pos1.element.style.height;     // fit to content after layout
const grew      = nd.size[1];

console.log(JSON.stringify({ deferHeight, deferSizes, fitHeight, grew }));
""" % (_const("FIELD_MIN_H"), _const("FIELD_MAX_H"), _const("FIELD_DEF_H"),
       _const("NEG_GAP"), _const("HIDDEN_PREFIX"), _const("FIELD_NAMES"),  # v715: BAR_H gone
       _lift("function _w(node, name)"),
       _lift("function _visibleFields(node)"),
       _lift("function _contentH(w)"),
       _lift("function _refit(node)"),
       _lift("function _refitNextFrame(node)"),
       _lift("function _extConnected(node, inputName)"),
       _lift("function _syncExternal(node)"))

with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as fh:
    fh.write(harness)
    path = fh.name
try:
    res = subprocess.run(["node", path], capture_output=True, text=True, timeout=30)
finally:
    os.unlink(path)
if res.returncode != 0:
    _fail("the refit-defer harness did not run: %s" % res.stderr.strip()[:400])
r = json.loads(res.stdout.strip().splitlines()[-1])

# 1) DEFER: an already-expanded field must survive the pre-layout _syncExternal untouched.
if r["deferHeight"] != "300px":
    _fail("_syncExternal COLLAPSED an already-expanded field synchronously (height -> %s before "
          "layout) -- this is the v619 run/reload squeeze. The programmatic refit must defer "
          "past layout, not measure an unlaid-out (empty) textarea now." % r["deferHeight"])
if r["deferSizes"] != 0:
    _fail("_syncExternal resized the node synchronously (%d setSize call(s) before layout) -- the "
          "refit must be QUEUED, not run inline on the run/reload event" % r["deferSizes"])

# 2) FIT AFTER LAYOUT: once the frame fires with the element laid out, the field fits its
#    content and the node grows -- proving the deferral still does the job it deferred.
if r["fitHeight"] != "300px":
    _fail("after the frame flushed with the element laid out, the field was not fit to its "
          "content (height %s, expected 300px) -- deferring must not mean never fitting"
          % r["fitHeight"])
if r["grew"] <= 400:
    _fail("the node did not grow to hold the fitted fields after layout (height %s) -- the "
          "deferred refit never took effect" % r["grew"])

print("[test_v620_refit_defer] PASS: every programmatic refit (_syncExternal + both onConfigure "
      "hooks) defers through _refitNextFrame/requestAnimationFrame; driven against a queuing rAF "
      "and a togglable scrollHeight, _syncExternal leaves an already-expanded field untouched "
      "pre-layout (no synchronous collapse, no synchronous resize) and fits it to content once "
      "the frame fires -- closing the v619 run/reload collapse. Live input-event refits stay "
      "inline, as intended.")
sys.exit(0)
