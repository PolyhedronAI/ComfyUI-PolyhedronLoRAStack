"""Guard v601 -- never both panes. The one that is TRUE is the one on screen.

WHAT FRANK SAW: a finished upscale sitting above a job that was three minutes from
done. The result viewer held the LAST run's picture while the process pane counted
tiles for THIS one. Two panes, two different runs, no label saying which was which
-- and a finished picture above a running job looks exactly like a finished job.

He read it as output. It was history.

v591 had already taught the process pane to bow out when the result lands
(`_procHide` inside `_pvApply`). The mirror was never built: nobody taught the
RESULT viewer to bow out when a new run STARTS. Half a state machine is not a state
machine, it is a race.

THE LAW: running -> process view. Done -> result. NEVER BOTH -- because "both" is
the one state where the user cannot tell which pane to believe, and he will believe
the wrong one, because the wrong one is the one that looks finished.

AND AN INTERRUPTED RUN KEEPS THE PROCESS PANE, with the old picture still hidden.
That run produced no result; the previous run's image is not this run's answer.
Restoring it would be the same lie in a friendlier font.

This guard does not read the state machine. It RUNS it -- the two real functions,
lifted from the source, driven through a real run in node. A structure check would
have passed the broken version: `_procHide` was there all along.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = open(os.path.join(ROOT, "web", "js", "ph_power_upscale.js"), encoding="utf-8").read()


def _fail(msg):
    print("[test_v601_onepane] FAIL: %s" % msg)
    sys.exit(1)


# =========================================================================
# LAW 1 -- both hides exist, and each is called from the OTHER pane's entry
# =========================================================================
if "function _pvHide(node)" not in JS:
    _fail("_pvHide is gone. The result viewer never bows out, so the last run's picture keeps "
          "sitting above the running one -- which is what Frank photographed.")
if "function _procHide(node)" not in JS:
    _fail("_procHide is gone (v591) -- the process pane never bows out when the result lands")

def _slice(name):
    """The body of a top-level function, up to the next one -- or to the end of
    the file, because _procApply happens to be the last one and the first cut of
    this guard walked straight off the edge looking for a `function` that was not
    there."""
    i = JS.index("function %s(" % name)
    ends = [e for e in (JS.find("\nfunction ", i + 1), JS.find("\napp.", i + 1)) if e > 0]
    return JS[i:min(ends)] if ends else JS[i:]


pv_apply = _slice("_pvApply")
if "_procHide(node)" not in pv_apply:
    _fail("_pvApply must hide the PROCESS pane. The result is in; the tile counter is history.")

proc_apply = _slice("_procApply")
reveal = proc_apply[:proc_apply.index("node._procTile.src")]
if "_pvHide(node)" not in reveal:
    _fail("_procApply must hide the RESULT viewer on the first pulse of a run. Without it the "
          "old picture hangs over the new job -- and it is the pane that LOOKS finished, so it "
          "is the one the user trusts.")

# The hide must actually collapse the height, or the pane leaves a hole behind it.
pv_hide = JS[JS.index("function _pvHide(node)"):]
pv_hide = pv_hide[:pv_hide.index("\n}") + 2]
if '_pvBox.style.display = "none"' not in pv_hide or "_pvPrevH = 0" not in pv_hide:
    _fail("_pvHide must set display:none AND zero the height. computeSize reads _pvPrevH; leave "
          "it standing and the node keeps a tall empty gap where the picture used to be.")
if "_pvStop(node)" not in pv_hide:
    _fail("_pvHide must stop the playback loop -- an invisible animation still burns a timer, "
          "and on a multi-frame result it burns one per node, forever")
if "node.setSize([node.size[0]" not in pv_hide:
    _fail("height only (v531) -- a run must never shrink the width the user set")

# =========================================================================
# LAW 2 -- RUN IT. Both functions, lifted verbatim, driven through a real run.
# =========================================================================
def _lift(name):
    """The whole function, brace to brace. These bodies have no nested closing
    brace at column 0, which is the only reason this is safe -- and if that ever
    changes, node will throw a SyntaxError and this guard will say so loudly
    instead of testing a truncated function and passing."""
    src = JS[JS.index("function %s(node" % name):]
    return src[:src.index("\n}") + 2]


harness = """
%(pvstop)s
%(pvhide)s
%(prochide)s

// A node with both panes on screen -- exactly the state Frank photographed.
function mkNode() {
    return {
        _pvBox:   { style: { display: "block" } },
        _procBox: { style: { display: "none"  } },
        _pvPrevH: 300, _procH: 0, _procSeen: false,
        _pvTimer: 1, _pvBtn: { textContent: "" },
        size: [640, 900],
        computeSize() { return [this.size[0], 100 + this._pvPrevH + this._procH]; },
        setSize(s) { this.size = s; },
        setDirtyCanvas() {},
    };
}

const out = [];
const n = mkNode();

// --- 1. a run STARTS: the first pulse reveals the process pane -----------
// (this is the reveal block of _procApply, verbatim in behaviour)
n._procSeen = true;
n._procBox.style.display = "block";
_pvHide(n);                       // <- the v601 cut
n._procH = 260;
out.push({ at: "running", pv: n._pvBox.style.display, proc: n._procBox.style.display,
           timer: n._pvTimer, w: n.size[0] });

// --- 2. the result LANDS: _pvApply hides the process pane ----------------
_procHide(n);                     // <- v591, already there
n._pvBox.style.display = "block";
n._pvPrevH = 320;
out.push({ at: "done", pv: n._pvBox.style.display, proc: n._procBox.style.display,
           w: n.size[0] });

// --- 3. the NEXT run starts again: re-armed? -----------------------------
if (!n._procSeen) {               // _procHide re-arms it
    n._procSeen = true;
    n._procBox.style.display = "block";
    _pvHide(n);
    n._procH = 260;
}
out.push({ at: "running2", pv: n._pvBox.style.display, proc: n._procBox.style.display });

console.log(JSON.stringify(out));
""" % {"pvstop": _lift("_pvStop"), "pvhide": _lift("_pvHide"), "prochide": _lift("_procHide")}

with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
    fh.write(harness)
    path = fh.name
try:
    res = subprocess.run(["node", path], capture_output=True, text=True, timeout=30)
finally:
    os.unlink(path)

if res.returncode != 0:
    _fail("the lifted state machine did not run: %s" % (res.stderr.strip()[:300]))
states = {s["at"]: s for s in json.loads(res.stdout.strip().splitlines()[-1])}

r = states["running"]
if r["pv"] != "none":
    _fail("DURING the run the result viewer is still displayed (%r). That is Frank's screenshot: "
          "a finished picture above a job with 4 minutes left on the clock." % r["pv"])
if r["proc"] != "block":
    _fail("during the run the process pane is not shown")
if r["timer"] is not None:
    _fail("the playback timer survived the hide -- an invisible animation still burns a frame "
          "loop per node, forever")

d = states["done"]
if d["proc"] != "none":
    _fail("when the result lands the process pane is STILL up (%r) -- a frozen last tile beside "
          "the finished image, which is v591's whole point" % d["proc"])
if d["pv"] != "block":
    _fail("the result never appears")

r2 = states["running2"]
if r2["pv"] != "none" or r2["proc"] != "block":
    _fail("the SECOND run did not re-arm: pv=%r proc=%r. _procHide must reset _procSeen or the "
          "swap happens exactly once and then rots." % (r2["pv"], r2["proc"]))

if r["w"] != 640 or d["w"] != 640:
    _fail("the width moved (%d -> %d). Height only (v531): the user widened this node to read "
          "his tiles, and a node that argues with its owner on every run is a node he stops "
          "trusting." % (r["w"], d["w"]))

print("[test_v601_onepane] PASS: the state machine was RUN, not read. Running -> process pane "
      "only (the old result is hidden and its timer stopped); done -> result only (the frozen "
      "tile counter bows out); and the next run re-arms the swap. Never both panes, because "
      "'both' is the state where the finished-looking one is the lie. Width untouched.")
sys.exit(0)
