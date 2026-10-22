# -*- coding: ascii -*-
"""Guard v1190 -- the green NEW means "came in with the LATEST render".

(Frank 20.10., screenshot of the neon timeline mid-render: "zwei neue Clips sind hinzugekommen - die sollten als NEW
markiert werden ... die drei anderen Clips laufen erst, sind aber trotzdem mit NEW markiert, das ist falsch")

  N1 the history keeps the sweep stamp (register(..., sweep=)); an entry without one reads its own time
  N2 Frank's picture: 5 places rendered in sweep A; a new render B (sweep B) has reached places 1 and 2 ->
     is_new: 1, 2 True; 3, 4, 5 False (their clip is A's, however many takes lie under it)
  N3 the render finishes (B reaches 3..5) -> all five NEW; a later single-scene render C of place 4 -> only 4
  N4 a first-time place (one take, no earlier clip) in the latest render is NEW; a place rendered in A only, with
     two takes under it, is not; a clip with no history (a drop / scan) is never NEW
  N5 legacy: entries without a sweep (before v1190) compare by their own time -- the newest wins alone
  P1 the plan item carries "new" (take 0 only); the route reads sweep from the body
  J1 isNew (run in node): new + take 0; a pinned older take is not; null is not
  J2 the Run press stamps ONE sweep (seconds) on every timeline node before the first queue; the primary
     register call sends it
  S  ASCII of this guard
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1190_new_is_latest"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


sys.path.insert(0, str(ROOT / "nodes"))
import cine_assemble as A  # noqa: E402

d = tempfile.mkdtemp(prefix="v1190nl_")
root = os.path.join(d, "out")
os.makedirs(os.path.join(root, "day"))
folder = os.path.join(d, "proj")
os.makedirs(folder)
A.OUTPUT_ROOT = root


def mk(name):
    p = os.path.join(root, "day", name)
    open(p, "wb").write(b"x")
    return "day/" + name


def run(k, sweep, tag):
    rel = mk("PH_%s_neon_scene_%d_00001_.mp4" % (tag, k))
    A.register(folder, k, os.path.join(root, rel), primary=True, run_files=[rel], sweep=sweep)
    return rel


try:
    T0 = 1000.0
    # ---- N1 / N2  render A over 5 places, render B reaches 1 and 2
    a = {k: run(k, T0, "A%d" % k) for k in range(1, 6)}
    runs = A.read_runs(folder)
    _need(all(runs[k][-1].get("sweep") == T0 for k in range(1, 6)), "N1 the history keeps the sweep stamp")
    _need(A.sweep_of({"rel": "x", "files": [], "at": 7.0}) == 7.0 and A.sweep_of({"rel": "x", "files": [], "at": 7.0, "sweep": 9.0}) == 9.0,
          "N1 an entry without a sweep reads its own time")
    b = {k: run(k, T0 + 100, "B%d" % k) for k in (1, 2)}
    got = [A.is_new(folder, k, (b if k in b else a)[k]) for k in range(1, 6)]
    _need(got == [True, True, False, False, False], "N2 mid-render: the two places the new render reached are NEW, the three others are not (%s)" % got)
    _need(A.latest_sweep(A.read_runs(folder)) == T0 + 100, "N2 the latest sweep is render B")
    # ---- N3
    for k in (3, 4, 5):
        b[k] = run(k, T0 + 100, "B%d" % k)
    got = [A.is_new(folder, k, b[k]) for k in range(1, 6)]
    _need(got == [True] * 5, "N3 the render finished: all five NEW (%s)" % got)
    c4 = run(4, T0 + 200, "C4")
    got = [A.is_new(folder, k, c4 if k == 4 else b[k]) for k in range(1, 6)]
    _need(got == [False, False, False, True, False], "N3 a later single-scene render: only that place is NEW (%s)" % got)
    # ---- N4
    f6 = run(6, T0 + 200, "C6")                      # first-time place, same render as C4
    _need(A.is_new(folder, 6, f6) is True, "N4 a first-time place in the latest render is NEW")
    _need(A.is_new(folder, 1, b[1]) is False and len(A.read_list(folder)[1]) == 2,
          "N4 a place with two takes from earlier renders is not NEW")
    drop = mk("dropped_clip.mp4")
    A.register(folder, 2, os.path.join(root, drop))
    _need(A.is_new(folder, 2, drop) is False, "N4 a clip without a run history is never NEW")
    _need(A.is_new(folder, 2, None) is False and A.is_new(folder, 99, "nope") is False, "N4 nothing shown / no history -> no")
    # ---- N5 legacy entries: at only
    d2 = tempfile.mkdtemp(prefix="v1190leg_")
    root2 = os.path.join(d2, "out")
    os.makedirs(os.path.join(root2, "day"))
    f2 = os.path.join(d2, "proj")
    os.makedirs(f2)
    A.OUTPUT_ROOT = root2
    old_rel, new_rel = [], []
    for k in (1, 2):
        p = os.path.join(root2, "day", "leg_%d.mp4" % k)
        open(p, "wb").write(b"x")
        A.register(f2, k, p, primary=True, run_files=["day/leg_%d.mp4" % k])     # no sweep: its own time
        time.sleep(0.01)
    runs2 = A.read_runs(f2)
    _need(all("sweep" not in h for v in runs2.values() for h in v), "N5 legacy entries carry no sweep")
    _need(A.is_new(f2, 2, "day/leg_2.mp4") is True and A.is_new(f2, 1, "day/leg_1.mp4") is False,
          "N5 legacy: the newest registration alone is NEW")
    shutil.rmtree(d2, ignore_errors=True)
finally:
    A.OUTPUT_ROOT = None
    shutil.rmtree(d, ignore_errors=True)

# ---- P1
PY = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
_need(re.search(r'"new": bool\(i == 0 and rels and A\.is_new\(folder, k, rels\[0\]\)\)', PY) is not None,
      "P1 the plan item carries 'new' (take 0 only)")
_need('sweep=body.get("sweep") if isinstance(body.get("sweep"), (int, float)) else None' in PY
      and "run_files=run_files, sweep=sweep)" in PY, "P1 the route reads the sweep from the body and hands it on")

# ---- J1 / J2
JS = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
i = JS.find("export function isNew(")
fn = JS[i:JS.find("\n}", i) + 2]
node = shutil.which("node")
if node and fn:
    h = fn.replace("export ", "") + """
const T = (c) => isNew(c);
console.log(JSON.stringify([
  T({new: true, takes: ["a", "b"], take: 0}),
  T({new: true, takes: ["a"], take: 0}),
  T({new: true, takes: ["a", "b"], take: 1}),
  T({new: false, takes: ["a", "b"], take: 0}),
  T({replaced: true, takes: ["a", "b"], take: 0}),
  T(null)]));
"""
    td = tempfile.mkdtemp()
    f = os.path.join(td, "h.js")
    open(f, "w").write(h)
    r = subprocess.run([node, f], capture_output=True, text=True, timeout=60)
    shutil.rmtree(td, ignore_errors=True)
    got = json.loads(r.stdout.strip() or "null")
    _need(got == [True, True, False, False, False, False], "J1 isNew: the latest render's clip at take 0, a single take too; replaced alone is not (%s)" % got)
else:
    _need(False, "J1 node.js and isNew are needed")
lane = JS[JS.find("function _clipLane("):JS.find("function _clipLane(") + 12000]
_need("!sg.missing && isNew(c)" in lane and "isReplaced(c)" not in lane.split("box.title")[0].split("isNew(c)")[1][:400],
      "J2 the lane draws NEW by isNew")
split = JS[JS.find("const split = async function"):JS.find("const split = async function") + 1200]
_need("if (!splitting) { const sw = Date.now() / 1000; for (const n of tlNodes()) n._plsSweep = sw; }" in split
      and split.find("_plsSweep = sw") < split.find("if (!node || splitting) return orig.apply"),
      "J2 one sweep stamp (seconds) on every timeline node, before the first queue, once per Run press")
_need("sweep: sweepOfPid.get(pid) || (tlById(run.node) || {})._plsSweep || null" in JS and JS.find("primary: true, run_files: run.rels") < JS.find("sweep: sweepOfPid.get(pid)"),   # AMENDED v1198: stamped at queue time
      "J2 the primary register call sends the sweep")
_need(all(c < 128 for c in Path(__file__).read_bytes()), "S  ASCII")
print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
