# -*- coding: ascii -*-
"""Guard v1023 -- the timeline editor (web/js/ph_cine_timeline.js).

The prompt shown in the node comes from the backend's route. The few numbers
the drawing needs before the server answers are mirrored in JavaScript --
this guard holds them equal to nodes/cine_timeline.py and drives the pure
editing functions in node.js.

  J1  parity: FPS, the official cut phrases, the default timeline; the
      17k+5 grid and MM:SS.mmm for a sweep of values, JS == Python.
  J2  editing: split a shot in the middle (snapped to a frame), remove a shot
      (the next becomes shot 1 at 0), a cut keeps one frame of room to its
      neighbours, an anchor stays inside 0 .. last frame, a new anchor takes
      the first frame, then the last, then the middle; the role of an anchor
      (first / last / guide) matches the backend's split.
  J3  parseTimeline fills defaults, survives bad JSON, sorts shots.
  S   the wiring: the route path and the ui key match the backend, the state
      lives in the hidden `timeline` widget only, the field has a fixed
      height, no floating window.

Script-style: exit 0 = pass.
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "web" / "js" / "ph_cine_timeline.js"
FAILS = []


def _fail(m):
    FAILS.append(m)
    print("  FAIL  " + m)


def _ok(m):
    print("  ok    " + m)


def _need(c, m):
    (_ok if c else _fail)(m)


spec = importlib.util.spec_from_file_location("ct1023", str(ROOT / "nodes" / "cine_timeline.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
src = JS.read_text(encoding="utf-8")
node_py = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")

# ------------------------------------------------------------------ S static
_need('"/pls/cine/timeline"' in node_py and '"/pls/cine/timeline"' in src,
      "S the editor calls the route the backend registers (/pls/cine/timeline)")
_need('"pls_timeline"' in node_py and "message.pls_timeline" in src, "S both sides use the ui key pls_timeline")
_need('_w(node, "timeline")' in src and 'setHidden(_w(this, "timeline"), true)' in src and "serialize: false" in src,
      "S the state lives only in the hidden `timeline` widget; the view never serialises")
# v1030 RE-GROUNDED, not softened: the promise was "a field INSIDE the node
# whose height the node reserves -- no floating window". Since v1030 the
# preview part fits its text (Frank: spread like the CTE), so the reserved
# height is fieldH(node) = BASE_H + the fitted preview, still in px.
_need(re.search(r"w\.computeSize = \(width\) => \{ _widthCheck\(node\); return \[width, fieldH\(node\)\]; \};", src) is not None
      and "window.open" not in src and "position:fixed" not in src,
      "S a field inside the node, its height reserved through computeSize, no floating window")

if shutil.which("node") is None:
    print("  note  node.js not found -- J1-J3 SKIPPED")
else:
    tmp = Path(tempfile.mkdtemp(prefix="v1023js_"))
    body = (src.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
               .replace('import { api } from "../../scripts/api.js";', "const api = {};")
               .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (tmp / "m.mjs").write_text(body, encoding="utf-8")
    secs = [0.5, 1, 3.3, 4, 4.46, 5, 6, 7.99, 8, 10.2, 14.4, 15, 22.7]
    times = [0, 0.001, 0.0415, 2.5, 59.9995, 65.25, 125.1]
    (tmp / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const out = {};
out.fps = T.FPS; out.cuts = T.CUTS; out.def = T.DEFAULT;
out.frames = SECS.map((s) => T.framesFor(s));
out.labels = TIMES.map((s) => T.timeLabel(s));
// J2
const m = T.parseTimeline(JSON.stringify({ seconds: 8, shots: [{ at: 0, text: "a" }, { at: 4, text: "b" }] }));
const i = T.splitShot(m, 0);
out.split = [i, m.shots.map((s) => s.at)];
out.removeFirst = (() => { const x = JSON.parse(JSON.stringify(m)); const k = T.removeShot(x, 0); return [k, x.shots.map((s) => s.at)]; })();
T.moveCut(m, 1, 0); const lo = m.shots[1].at; T.moveCut(m, 1, 99); const hi = m.shots[1].at;
out.cut = [lo, hi];
m.anchors = [{ src: "a", at: 1 }];
T.moveAnchor(m, 0, -5); const a0 = m.anchors[0].at; T.moveAnchor(m, 0, 99); const a1 = m.anchors[0].at;
out.anchor = [a0, a1, T.lastFrameTime(m)];
const f = { seconds: 8, anchors: [] };
const s1 = T.freeSpot(f); f.anchors.push({ src: "a", at: s1 });
const s2 = T.freeSpot(f); f.anchors.push({ src: "b", at: s2 });
const s3 = T.freeSpot(f);
out.spots = [s1, s2, s3];
out.roles = [0, 3, 191 / 24, 8].map((at) => T.anchorRole({ seconds: 8 }, { at }));
// J3
out.bad = T.parseTimeline("{nope");
out.sorted = T.parseTimeline(JSON.stringify({ shots: [{ at: 2 }, { at: 0 }] })).shots.map((s) => s.at);
console.log(JSON.stringify(out));
""".replace("SECS", json.dumps(secs)).replace("TIMES", json.dumps(times)), encoding="utf-8")
    r = subprocess.run(["node", str(tmp / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(tmp, ignore_errors=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        _fail("J harness failed: %s" % r.stderr.strip()[-400:])
    else:
        d = json.loads(line[-1])
        _need(d["fps"] == T.FPS and d["cuts"] == T.CUTS and d["def"] == T.DEFAULT,
              "J1 FPS, the official cut phrases and the default timeline equal the backend's")
        _need(d["frames"] == [T.frames_for(s) for s in secs], "J1 the 17k+5 grid: JS == Python for %d lengths" % len(secs))
        _need(d["labels"] == [T.time_label(s) for s in times], "J1 MM:SS.mmm: JS == Python for %d times" % len(times))
        _need(d["split"] == [1, [0, 2, 4]], "J2 split shot 1 of 0..4 s -> a new cut at 2 s")
        _need(d["removeFirst"] == [0, [0, 4]], "J2 removing shot 1 makes the next one shot 1 at 0 %r" % d["removeFirst"])
        _need(abs(d["cut"][0] - 1 / 24.0) < 1e-9 and abs(d["cut"][1] - (4 - 1 / 24.0)) < 1e-9,
              "J2 a cut keeps one frame of room to both neighbours %r" % d["cut"])
        _need(d["anchor"][0] == 0 and abs(d["anchor"][1] - d["anchor"][2]) < 1e-9 and abs(d["anchor"][2] - 191 / 24.0) < 1e-9,
              "J2 an anchor stays inside 0 .. the last frame (191/24 s)")
        _need(abs(d["spots"][0]) < 1e-9 and abs(d["spots"][1] - 191 / 24.0) < 1e-9 and 3.9 < d["spots"][2] < 4.1,
              "J2 a new anchor: first frame, then last frame, then the middle %r" % d["spots"])
        be = [("first" if f == 0 else "last" if f == 191 else "guide") for f in [T.frame_of(a, 192) for a in (0, 3, 191 / 24.0, 8)]]
        _need(d["roles"] == be, "J2 anchor roles match the backend's split %r" % d["roles"])
        _need(d["bad"]["seconds"] == 6 and len(d["bad"]["shots"]) == 1 and d["sorted"] == [0, 2],
              "J3 bad JSON -> the default; shots come back sorted")

print()
if FAILS:
    print("test_v1023_cine_timeline_js: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1023_cine_timeline_js: PASS (J1-J3, S)")
