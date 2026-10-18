# -*- coding: ascii -*-
"""Guard v1165 -- the green NEW on a timeline clip that replaced an earlier run's clip.

Frank 18.10.: "Ich haette gerne, dass man sehen kann, dass in der Timeline eine Datei ersetzt worden ist; das kann
ein kleines gruenes NEW sein am Clip". One run writes several files of a scene (raw, upscale, interpolated) -- all of
them are takes; that alone is no replacement. So the stack list now keeps a RUN HISTORY: the register call of a
run's clip (primary) carries every file the run wrote.

  H1 register(primary, run_files) writes the scene's run history (clip, its files, time); a writer that does not
     touch the history (migrate) keeps it
  H2 replaced(): one run with raw + final -> no; a file outside the shown clip's run (an earlier run, a run that
     broke off after saving) -> yes; a single take -> no
  H3 a list from before v1165 (no history): two clips of the run's own name (%date% tokens expanded) -> yes; a take
     older than LEGACY_RUN_GAP before the shown clip -> yes; one run's files only -> no
  P1 the plan item carries "replaced" (only when V1 shows the newest, take 0)
  J1 isReplaced (run in node): replaced + take 0 + more than one take; a pinned older take (new track mode) or a
     single take is not marked
  J2 the lane draws a green NEW (data-pls="newmark") on such a clip and says it in the tooltip; the register call
     of the run's clip sends run_files
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
NAME = "test_v1165_new_mark"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


sys.path.insert(0, str(ROOT / "nodes"))
import cine_assemble as A  # noqa: E402

d = tempfile.mkdtemp(prefix="v1165nm_")
root = os.path.join(d, "out")
os.makedirs(os.path.join(root, "day"))
folder = os.path.join(d, "proj")
os.makedirs(folder)
A.OUTPUT_ROOT = root


def mk(name, age=0.0):
    p = os.path.join(root, "day", name)
    open(p, "wb").write(b"x")
    t = time.time() - age
    os.utime(p, (t, t))
    return "day/" + name


STEM = "PH_%date:hhmmss%_neon_scene_2"
try:
    # ---- H1
    raw = mk("PH_130000_MINIMAX_00001_.mp4", 600)
    fin = mk("PH_131500_neon_scene_2_00001_.mp4", 0)
    A.register(folder, 2, os.path.join(root, raw))
    A.register(folder, 2, os.path.join(root, fin), primary=True, run_files=[raw, fin])
    runs = A.read_runs(folder)
    _need(list(runs) == [2] and runs[2][-1]["rel"] == fin and runs[2][-1]["files"] == sorted([raw, fin]),
          "H1 the run history: the clip and every file of its run")
    A._write_list(folder, A.read_list(folder), A.read_primary(folder))
    _need(A.read_runs(folder) == runs, "H1 a writer that does not touch the history keeps it")

    # ---- H2
    _need(A.replaced(folder, 2, [fin, raw], fin, STEM) is False, "H2 one run (raw + final) is no replacement")
    old = mk("PH_090000_MINIMAX_00001_.mp4", 100)
    _need(A.replaced(folder, 2, [fin, raw, old], fin, STEM) is True,
          "H2 a file outside the shown clip's run (a run that broke off after saving) -> replaced")
    _need(A.replaced(folder, 2, [fin], fin, STEM) is False, "H2 a single take -> no")

    # ---- H3 (no history: scene 3)
    a = mk("PH_100000_neon_scene_3_00001_.mp4", 1200)
    b = mk("PH_120000_neon_scene_3_00001_.mp4", 0)
    stem3 = "PH_%date:hhmmss%_neon_scene_3"
    _need(A.replaced(folder, 3, [b, a], b, stem3) is True, "H3 no history: two clips of the run's own name -> replaced")
    r3 = mk("PH_115000_MINIMAX_00001_.mp4", 600)
    _need(A.replaced(folder, 3, [b, r3], b, stem3) is False, "H3 no history: one run's files only -> no")
    g = mk("PH_080000_MINIMAX_00001_.mp4", A.LEGACY_RUN_GAP + 600)
    _need(A.replaced(folder, 3, [b, r3, g], b, stem3) is True, "H3 no history: a take older than the gap -> replaced")
finally:
    shutil.rmtree(d, ignore_errors=True)

# ---- P1
PY = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
_need(re.search(r'"replaced": bool\(i == 0 and rels and A\.replaced\(folder, k, rels, rels\[0\],', PY) is not None
      and "run_files=body.get(\"run_files\")" in PY, "P1 the plan item carries 'replaced' (take 0 only); the route passes run_files")

# ---- J1 / J2
JS = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
i = JS.find("export function isReplaced(")
fn = JS[i:JS.find("\n}", i) + 2]
node = shutil.which("node")
if node and fn:
    h = fn.replace("export ", "") + """
const T = (c) => isReplaced(c);
console.log(JSON.stringify([
  T({replaced: true, takes: ["a", "b"], take: 0}),
  T({replaced: true, takes: ["a", "b"], take: 1}),
  T({replaced: false, takes: ["a", "b"], take: 0}),
  T({replaced: true, takes: ["a"], take: 0}),
  T(null)]));
"""
    td = tempfile.mkdtemp()
    f = os.path.join(td, "h.js")
    open(f, "w").write(h)
    r = subprocess.run([node, f], capture_output=True, text=True, timeout=60)
    shutil.rmtree(td, ignore_errors=True)
    got = json.loads(r.stdout.strip() or "null")
    _need(got == [True, False, False, False, False], "J1 isReplaced: newest shown after an earlier run only (%s)" % got)
else:
    _need(False, "J1 node.js and isReplaced are needed")
lane = JS[JS.find("function _clipLane("):JS.find("function _clipLane(") + 12000]
_need('nw.dataset.pls = "newmark"' in lane and "background:#4caf50" in lane and '"NEW")' in lane
      and "!sg.missing && isReplaced(c)" in lane and "NEW: replaced" in lane,
      "J2 the lane draws the green NEW on a replaced clip and says it in the tooltip")
_need("primary: true, run_files: run.rels" in JS, "J2 the register call of the run's clip sends run_files")
_need(all(c < 128 for c in Path(__file__).read_bytes()), "S  ASCII")
print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
