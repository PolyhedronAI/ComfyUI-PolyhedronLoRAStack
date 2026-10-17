# -*- coding: ascii -*-
"""Guard v1082 -- the basin made tight (the Timeline audit of 10.10., block B).

  T1  an unchanged re-run adds no second track: ComfyUI reports a cached run
      like a new one ("executed" still comes), so _autoTrack1 refuses a clip
      that is V1's pinned take or a track already -- driven in node: the
      pinned rel twice -> no layer, the status says so; a NEW rel -> a track.
  T2  "keep tail" with no clip at the last place: the JS handler refuses
      (unchecks, says "no clip yet") instead of storing 1e9; the server clamps
      a piece past its clip's end to the clip and treats a piece longer than
      PIECE_MAX with no clip as bad (the prompt's pieces are used, a warning).
  T3  "restore on open" off wipes tracks, cuts and takes but keeps the
      settings: wipeEdit driven (auto_track false and span stay, layers /
      pieces / takes / cuts go); onConfigure uses it.
  T4  a refused drop on "+ new track" (no video file, failed probe) leaves
      NO empty layer and NO edit write; a good drop opens the track with its
      clip in ONE write -- driven in node with stubbed routes.
  T5  "render all scenes" / the split: a refused queue call (false) stops
      the loop and the status says which scene was refused and how many were
      queued -- _renderAll driven with a queuePrompt stub refusing scene 2.
  S   ASCII.

Script-style: exit 0 = pass.
"""
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1082_basin_tight"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)
    return c


js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")

# ---- the node harness: the module with its three imports stubbed ------------
tmpd = Path(tempfile.mkdtemp(prefix="v1082js_"))
h = js.replace('import { app } from "../../scripts/app.js";', "const app = globalThis.__app;")
h = h.replace('import { api } from "../../scripts/api.js";', "const api = globalThis.__api;")
h = h.replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}; const refit = () => {};")
h = h.replace("function _render(node) {", "function _render(node) { if (globalThis.__noRender) return;", 1)   # no DOM in node
h += "\nexport { _autoTrack1, _dropNewTrack, _renderAll };\n"
(tmpd / "tl.mjs").write_text(h, encoding="utf-8")
(tmpd / "drive.mjs").write_text(r'''
const calls = [];
let queueAnswers = [];
globalThis.__app = { graph: { _nodes: [] }, registerExtension() {}, queuePrompt: async () => queueAnswers.shift() };
globalThis.__api = { fetchApi: async (url, opt) => { calls.push(url);
  if (url.startsWith("/pls/cine/probe")) { const b = JSON.parse(opt.body); if (b.src.path.endsWith("bad.mp4")) return { json: async () => ({ error: "unreadable" }) };
    return { json: async () => ({ name: "a.mp4", frames: 48, fps: 24, audio: true }) }; }
  if (url.startsWith("/pls/cine/media_dir")) return { json: async () => ({ dir: "/x", rel: "cine_media/fx" }) };
  if (url.startsWith("/uls/media/upload")) return { json: async () => ({ ok: true, names: [opt.body.get("file").name] }) };
  return { json: async () => ({ error: "stub" }) }; }, apiURL: (u) => u };
globalThis.console.info = () => {};
globalThis.requestAnimationFrame = (f) => 0; globalThis.__noRender = true;
const F = (name) => new File(["x"], name);
const M = await import("./tl.mjs");
const mk = (edit, model) => ({ properties: { [M.EDIT_KEY]: edit }, widgets: [{ name: "timeline", value: JSON.stringify(Object.assign({}, M.DEFAULT, model || {})) }],
  _plsTl: { state: { textContent: "" }, player: { t: 0 }, active: "base" } });
const out = {};
const layers = (n) => (n.properties[M.EDIT_KEY].layers || []).filter((l) => l.id !== "base").map((l) => l.src && l.src.path);
// T1
let n = mk({ takes: { "0": "fx/whole_00001_.mp4" } });
await M._autoTrack1(n, "fx/whole_00001_.mp4", 0, 1);
out.t1_same = { layers: layers(n), state: n._plsTl.state.textContent };
await M._autoTrack1(n, "fx/whole_00001_.mp4", 0, 1);
out.t1_again = layers(n);
// T3
out.t3 = [M.wipeEdit({ auto_track: false, span: 6.5, layers: [{ id: "L1" }], pieces: { whole: [1] }, takes: { "0": "x" }, cuts: [1] }),
          M.wipeEdit({ auto_track: true, layers: [{ id: "L1" }] }), M.wipeEdit(null)];
// T4
n = mk({}); calls.length = 0;
await M._dropNewTrack(n, { files: [F("x.png")], getData: () => "" }, "copy", 0);
out.t4_png = { layers: n.properties[M.EDIT_KEY].layers || null, writes: calls.filter((u) => u.startsWith("/pls/cine/edit")).length, state: n._plsTl.state.textContent };
n = mk({}); calls.length = 0;
await M._dropNewTrack(n, { files: [F("bad.mp4")], getData: () => "" }, "copy", 0);
out.t4_bad = { layers: n.properties[M.EDIT_KEY].layers || null, state: n._plsTl.state.textContent };
n = mk({}); calls.length = 0;
await M._dropNewTrack(n, { files: [F("a.mp4")], getData: () => "" }, "copy", 0);
out.t4_good = { layers: layers(n), active: n._plsTl.active, state: n._plsTl.state.textContent };
// T5
n = mk({}, { shots: [{ at: 0, text: "a" }, { at: 2, scene: true, cut: "the camera cuts to", text: "b" }, { at: 4, scene: true, cut: "the camera cuts to", text: "c" }], seconds: 6 });
queueAnswers = [true, false, true];
await M._renderAll(n);
out.t5 = { state: n._plsTl.state.textContent, left: queueAnswers.length };
console.log(JSON.stringify(out));
process.exit(0);   // the preview's debounce timer has no DOM here
''', encoding="utf-8")
r = subprocess.run(["node", str(tmpd / "drive.mjs")], capture_output=True, text=True, timeout=120)
ok = r.returncode == 0
if not ok:
    print(r.stderr[-1500:])
o = json.loads(r.stdout.strip().splitlines()[-1]) if ok else {}
shutil.rmtree(tmpd, ignore_errors=True)
# T1
_need(ok and o["t1_same"]["layers"] == [] and "V1 of this place already" in o["t1_same"]["state"],
      "T1 the pinned clip reported again adds no track, the status says so: %r" % (o.get("t1_same"),))
_need(ok and o["t1_again"] == [], "T1 ... nor on a third report")
# T3
_need(ok and o["t3"] == [{"auto_track": False, "span": 6.5}, {}, {}], "T3 wipeEdit keeps the explicit off and the span, drops the rest: %r" % (o.get("t3"),))
_need("this.properties[EDIT_KEY] = wipeEdit(this.properties[EDIT_KEY]);" in js, "T3 onConfigure wipes through wipeEdit")
# T4
_need(ok and not o["t4_png"]["layers"] and o["t4_png"]["writes"] == 0 and "no video file" in o["t4_png"]["state"],
      "T4 a non-video drop: no layer, no edit write: %r" % (o.get("t4_png"),))
_need(ok and not o["t4_bad"]["layers"] and "unreadable" in o["t4_bad"]["state"], "T4 a failed probe: no layer: %r" % (o.get("t4_bad"),))
_need(ok and len(o["t4_good"]["layers"]) == 1 and o["t4_good"]["layers"][0].endswith("a.mp4") and o["t4_good"]["active"] != "base",
      "T4 a good drop opens ONE track with its clip, active: %r" % (o.get("t4_good"),))
# T5
_need(ok and "scene 2 was refused" in o["t5"]["state"] and "2..3 not queued" in o["t5"]["state"] and "(1 queued)" in o["t5"]["state"] and o["t5"]["left"] == 1,
      "T5 render all stops at the refused scene and says so: %r" % (o.get("t5"),))
_need('if (res === false) { refused = k; break; }' in js and 'was refused (see the dialog) -- the remaining scenes are not queued' in js,
      "T5 the split path stops at a refused scene too")

# ---- T2: JS text + server clamp --------------------------------------------
_need('if (!Number.isFinite(fb)) { tail.checked = false; s2.state.textContent = "keep tail: the last place has no clip yet"; return; } pcs[j].b = fb;' in js
      and "Math.min(fK(pcs[j].k), 1e9)" not in js, "T2 keep tail refuses without a clip (no 1e9 any more)")
pkg = types.ModuleType("plsn1082")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1082"] = pkg
E = importlib.import_module("plsn1082.cine_edit")
_need(E.PIECE_MAX == 24 * 3600, "T2 PIECE_MAX is an hour of frames")
clips = [{"frames": 149, "fps": 24, "audio": False}]
stored = {"pieces": {"whole": [{"k": 0, "a": 0, "b": 1000000000, "speed": 1.0}]}}
_plan = lambda c, e: E.plan([], 6.0, c, [], ["cut"], edit=e, whole=True)
P = _plan(clips, stored)
_need(P["length"] == 149 and not any("does not fit" in w for w in P["warnings"]),
      "T2 a stored b past the clip is clamped to the clip (length %s, warnings %r)" % (P["length"], P["warnings"]))
P2 = _plan([None], stored)
_need(P2["length"] < E.PIECE_MAX and any("does not fit" in w for w in P2["warnings"]),
      "T2 with no clip the 1e9 piece is bad -> the prompt's pieces, a warning (length %s)" % P2["length"])
P3 = _plan(clips, {"pieces": {"whole": [{"k": 0, "a": 10, "b": 100, "speed": 1.0}]}})
_need(P3["length"] == 90 and not P3["warnings"], "T2 an ordinary cut list is untouched (length %s)" % P3["length"])

bad = [f for f in ("web/js/ph_cine_timeline.js", "nodes/cine_edit.py", "tests/test_v1082_basin_tight.py")
       if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not bad, "S ASCII only %r" % bad)
if FAILS:
    print("%s: %d FAILURE(S)" % (NAME, len(FAILS)))
    sys.exit(1)
print("%s: PASS (T1-T5, S)" % NAME)
