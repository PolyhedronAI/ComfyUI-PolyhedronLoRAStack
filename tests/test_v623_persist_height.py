#!/usr/bin/env python3
"""
test_v623_persist_height -- the fitted field height survives a browser page reload (F5).

v622 remembers each prompt field's height in a runtime property (_plsH). That holds through
RUN and in-session reload, but a browser reload wipes ALL JavaScript state: on the fresh load
_wireField re-inits _plsH to FIELD_DEF_H (64) and the field renders collapsed. The only store
that survives a reload is the saved workflow file. v623 writes _plsH into it on serialize() and
re-applies it on load, before any refit.

Two named helpers carry it:
  _captureFieldHeights(node) -> { field: _plsH }   (serialize() stashes it under
                                                    o.pls_field_heights)
  _restoreFieldHeights(node, info)                 (onConfigure applies it BEFORE any refit)

Guards, both must hold, mutation-tested (inject the wound, prove the catch):

  STATIC -- the save side stashes the captured map (serialize -> o.pls_field_heights via
            _captureFieldHeights) and an onConfigure hook calls _restoreFieldHeights. If either
            end is gone the height is never persisted or never re-applied -- a reload collapses.

  DRIVEN -- capture a live node's heights, hand the map to a FRESH node whose _plsH was just
            reset to 64 (the exact cold-reload state), and prove restore lifts them back.
            Mutating the restore guard (if (!h) return; -> if (h) return;) makes restore a
            no-op and the field stays at 64 -- caught here.
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
    print("[test_v623_persist_height] FAIL: " + msg)
    sys.exit(1)


def _lift(sig):
    if sig not in JS:
        _fail("could not find `%s`" % sig)
    s = JS[JS.index(sig):]
    return s[:s.index("\n}") + 2]


def _const(name):
    m = re.search(r"^const " + re.escape(name) + r" = .*?;", JS, re.M)
    if not m:
        _fail("could not lift source constant `%s`" % name)
    return m.group(0)


# ---------------------------------------------------------------------------
# STATIC -- the save side stashes the map, the load side re-applies it.
# ---------------------------------------------------------------------------
if "function _captureFieldHeights(node)" not in JS:
    _fail("_captureFieldHeights is gone -- nothing writes the height into the workflow")
if "function _restoreFieldHeights(node, info)" not in JS:
    _fail("_restoreFieldHeights is gone -- nothing re-applies the saved height on load")

m = re.search(r"nodeType\.prototype\.serialize = function[\s\S]*?\n        \};", JS)
if not m:
    _fail("serialize() override not found -- cannot confirm the height is persisted")
SER = m.group(0)
if "pls_field_heights" not in SER or "_captureFieldHeights" not in SER:
    _fail("serialize() no longer writes _captureFieldHeights() into o.pls_field_heights -- the "
          "height is not persisted, so a reload cannot restore it")

if "_restoreFieldHeights(this, arguments[0])" not in JS:
    _fail("onConfigure no longer calls _restoreFieldHeights -- saved heights are never re-applied")


# ---------------------------------------------------------------------------
# DRIVEN -- capture -> (fresh node reset to 64) -> restore. Then a MUTANT.
# ---------------------------------------------------------------------------
PRELUDE = 'function _w(node, name){ return (node.widgets||[]).find(w => w.name === name); }\n'

BODY = """
const COLD = 64;   // what _wireField resets _plsH to on a cold (post-F5) load
function mkNode(h1, hn) {
    const pos1 = { name: "pos_1", _plsH: h1 };
    const neg1 = { name: "neg_1", _plsH: hn };
    return { widgets: [ { name: "segments", value: 1 }, pos1, neg1 ], pos1, neg1 };
}

const out = {};
{   // 1) a live node with fitted heights -> capture the map (what serialize() stashes).
    const src = mkNode(520, 300);
    const map = _captureFieldHeights(src);
    out.capPos = map["pos_1"];
    out.capNeg = map["neg_1"];

    // 2) a FRESH node after a cold reload: _wireField has already reset both to 64.
    //    Restore must lift them back to the saved 520 / 300.
    const cold = mkNode(COLD, COLD);
    _restoreFieldHeights(cold, { pls_field_heights: map });
    out.restPos = cold.pos1._plsH;
    out.restNeg = cold.neg1._plsH;
}
console.log(JSON.stringify(out));
"""


def run(restore_src, label):
    src = (
        _const("MAX_SEGMENTS") + "\n"
        + PRELUDE
        + _lift("function _captureFieldHeights(node)") + "\n"
        + restore_src + "\n"
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


RESTORE = _lift("function _restoreFieldHeights(node, info)")

# 1) REAL -- capture reads the fitted heights, restore lifts a cold (64) node back to them.
real = run(RESTORE, "real")
if real["capPos"] != 520 or real["capNeg"] != 300:
    _fail("capture did not read the fitted heights (pos %s neg %s, expected 520/300) -- the "
          "wrong values would be persisted into the workflow" % (real["capPos"], real["capNeg"]))
if real["restPos"] != 520 or real["restNeg"] != 300:
    _fail("restore did not re-apply the saved heights to a cold (64) node (pos %s neg %s, "
          "expected 520/300) -- the field stays collapsed after a reload"
          % (real["restPos"], real["restNeg"]))

# 2) MUTANT -- invert the restore guard so it early-returns; the cold node must stay at 64.
MUT = RESTORE.replace("if (!h) return;", "if (h) return;")
if MUT == RESTORE:
    _fail("mutation target 'if (!h) return;' not found -- harness out of sync with "
          "_restoreFieldHeights")
mut = run(MUT, "mutant")
if mut["restPos"] == 520:
    _fail("MUTATION NOT CAUGHT: with the restore guard inverted, the cold node still recovered "
          "its height (pos %s) -- the DRIVEN check does not prove restore is what survives the "
          "reload" % mut["restPos"])

print("[test_v623_persist_height] PASS -- capture reads fitted heights (520/300); restore lifts "
      "a cold (64) node back to them; inverting the restore guard leaves it collapsed (mutation "
      "caught)")
sys.exit(0)
