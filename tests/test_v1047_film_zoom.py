# -*- coding: ascii -*-
"""Guard v1047 -- the FILM row (free film length, zoom) and "new render ->
new track" that works for real (Frank 01.10.: "the total length free, e.g.
5 minutes, prominent; the view and the zoom follow"; "new clips in a new
track when the switch is set -- that did not work yesterday").

  F1  pure helpers (node.js): parseLen (5:00, 1:30.5, 300, 2m, 90s, h:mm:ss;
      junk -> NaN), fmtLen, tickStep (labels >= 56 px apart), zoomClamp
      (1 .. a 2-second window), runOf (= the server's resolve + clip_prefix,
      project safe-named), pickTrackFile (wired clip_prefix wins, else the
      run's last file).
  F2  the field: a FILM row between the toolbar and the tracks (length field,
      presets, zoom - fit +, view label, scroll bar); the span left the
      toolbar; the lanes sit in a scroller and get wider by the zoom; the
      heads are sticky; fieldH counts the row; the ruler's step follows the
      scale; a new length resets the zoom; the wheel is taken on the document
      in the capture phase (Nodes 2.0 swallows it above the node).
  F3  the run: a CACHED timeline sends no "executed" -- "execution_cached"
      names the run; the Saves' files are collected and ONE track is made
      when the run ends ("execution_success"); the tracks are added one
      after the other (two runs ending together lost one -- measured).
  F4  the server clamps the span to SPAN_MAX (an hour), both sides agree.
  S   ASCII.

Script-style: exit 0 = pass.
"""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
fld = js[js.index("function _field("):]

# ------------------------------------------------------------------ F1
if shutil.which("node") is None:
    print("  note  node.js not found -- F1 SKIPPED")
else:
    t2 = Path(tempfile.mkdtemp(prefix="v1047js_"))
    body = (js.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (t2 / "m.mjs").write_text(body, encoding="utf-8")
    (t2 / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const o = {};
o.parse = ["5:00", "1:30.5", "300", "2m", "90s", "0:01:05", "1,5", "abc", "", "1:75", "-3"].map((x) => { const v = T.parseLen(x); return Number.isFinite(v) ? v : null; });
o.fmt = [T.fmtLen(300), T.fmtLen(90.5), T.fmtLen(5), T.fmtLen(3725)];
o.tick = [T.tickStep(100), T.tickStep(20), T.tickStep(4), T.tickStep(0.5), T.tickStep(0.01)];
o.zc = [T.zoomClamp(0.3, 300), T.zoomClamp(1000, 300), T.zoomClamp(4, 300), T.zoomClamp(4, 3)];
o.zo = [T.zoomOf({ properties: {} }), T.zoomOf({ properties: { [T.ZOOM_KEY]: 8 } }), T.zoomOf({ properties: { [T.ZOOM_KEY]: "x" } })];
const m = (o2) => Object.assign({ seconds: 9, shots: [{ at: 0, text: "a" }, { at: 3, text: "b", scene: true }, { at: 6, text: "c", scene: true }], project: "ww1 cyborg!" }, o2);
o.run = [T.runOf(m({ render: "whole" })), T.runOf(m({ render: "all", active: 2 })), T.runOf(m({ render: "scene", active: 9 })), T.runOf({ seconds: 4, shots: [{ at: 0 }], project: "" })];
o.pick = [T.pickTrackFile(["a/x_00001_.mp4", "p/scene_2_00003_.mp4", "b/y_00001_.mp4"], "p/scene_2"),
          T.pickTrackFile(["a/x_00001_.mp4", "b/y_00001_.mp4"], "p/scene_2"), T.pickTrackFile([], "p"),
          T.pickTrackFile(["p/scene_20_00001_.mp4", "q.mp4"], "p/scene_2"),
          T.pickTrackFile(["2026-10-17/PH_101010_MINIMAX_00001_.mp4", "2026-10-17/PH_101011_fx_scene_2_00001_.mp4", "2026-10-17/PH_101012_INTERP_00001_.mp4"], T.DAY_PREFIX + "fx_scene_2"),
          T.pickTrackFile(["2026-10-17/PH_101011_fx_scene_20_00001_.mp4", "q.mp4"], T.DAY_PREFIX + "fx_scene_2")];
console.log(JSON.stringify(o));
""", encoding="utf-8")
    r = subprocess.run(["node", str(t2 / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(t2, ignore_errors=True)
    line = [x for x in r.stdout.splitlines() if x.startswith("{")]
    if r.returncode != 0 or not line:
        _need(False, "F1 harness failed: %s" % r.stderr.strip()[-400:])
    else:
        o = json.loads(line[-1])
        _need(o["parse"] == [300, 90.5, 300, 120, 90, 65, 1.5, None, None, None, None], "F1 parseLen %r" % o["parse"])
        _need(o["fmt"] == ["5:00", "1:30.50", "0:05", "62:05"], "F1 fmtLen %r" % o["fmt"])
        _need(o["tick"] == [1, 5, 15, 120, 600], "F1 tickStep: labels >= 56 px apart %r" % o["tick"])
        _need(o["zc"] == [1, 150, 4, 1.5] and o["zo"] == [1, 8, 1], "F1 zoomClamp / zoomOf %r %r" % (o["zc"], o["zo"]))
        # AMENDED v1152 (Frank 17.10.: the day folder): runOf names <day>/PH_<hhmmss>_<project>_...; the pick reads the tokens
        D = "%date:yyyy-MM-dd%/PH_%date:hhmmss%_"
        _need([x["clip_prefix"] for x in o["run"]] == [D + "ww1_cyborg_whole", D + "ww1_cyborg_scene_2", D + "ww1_cyborg_scene_3", D + "cine_whole"]
              and [x["scene"] for x in o["run"]] == [0, 2, 3, 0], "F1 runOf = resolve + clip_prefix, safe project %r" % o["run"])
        _need(o["pick"] == ["p/scene_2_00003_.mp4", "b/y_00001_.mp4", None, "q.mp4", "2026-10-17/PH_101011_fx_scene_2_00001_.mp4", "q.mp4"],
              "F1 pickTrackFile: wired wins (tokens included), else the last %r" % o["pick"])

# ------------------------------------------------------------------ F2
# AMENDED v1150: top bar, head row, refs, the WORK area (the selected shot left, the monitor right), then as before
_need("root.append(topBar, headRow, refsRow, work, state, tools, film, scroller, bar, status, prev, pop);" in fld and "work.append(left, sndRow, stack);" in fld, "F2 the FILM row between the toolbar and the tracks")   # AMENDED v1153
_need('film.append(_el("span", "font-weight:bold;color:#ffd27a;letter-spacing:1px;", "FILM"), span, preset, filmInfo,' in fld
      and "zOut, zSl, zIn, zFit, viewInfo, nav);" in fld, "F2 the row: length, presets, zoom - slider + fit, view, scroll bar")   # AMENDED v1153: the slider
_need("cutsBtn, tailLab,\n        sep(), autoLab, restLab, takesBtn, clearBtn);" in fld, "F2 the span left the toolbar")   # AMENDED v1200: + x tracks (v1198)
_need("scroller.appendChild(lanes);" in fld and 'const scroller = _el("div", "position:relative;overflow:hidden;flex:0 0 auto;");' in fld,
      "F2 the lanes sit in a scroller")
_need('"position:sticky;left:0;z-index:8;"' in js, "F2 the heads are sticky")
_need('["film", FILM_H, true],' in js, "F2 fieldH counts the FILM row")   # AMENDED v1150
_need('const w = z === 1 ? "100%" : "calc(" + GUT_W + "px + " + z + " * (100% - " + GUT_W + "px))";' in js
      and "_applyZoom(node);              // v1047" in js, "F2 the lanes get wider by the zoom, before the ruler measures")
_need("const stp = tickStep(rw / ax)" in js, "F2 the ruler's step follows the scale")
_need("node.properties[ZOOM_KEY] = 1;\n    if (st.scroller) st.scroller.scrollLeft = 0;\n    _saveEdit(node, e);" in js, "F2 a new length: fit, scrolled home")
_need('document.addEventListener("wheel", (ev) => {' in js and "{ capture: true, passive: false });" in js and "ev.stopPropagation();" in js,
      "F2 the wheel on the document, capture phase")
_need("if (pl.playing) _follow(node);" in js, "F2 playing: the view follows the playhead")

# ------------------------------------------------------------------ F3
# v1088 (DECLARED): the cached run is PROVISIONAL per timeline node until its replayed executed
_need('api.addEventListener("execution_cached", (ev) => {' in js and "rec.tls.set(String(n.id), freshRun(n._plsLastRun || runOf(_model(n)), n.id, true));" in js,
      "F3 a cached timeline: the run from execution_cached")
_need('api.addEventListener("execution_success", async (ev) => {' in js and "const pick = pickTrackFile(run.rels, run.clip_prefix);" in js
      and "await Promise.all(run.pending);" in js, "F3 ONE track per run, at its end, after every file is listed")
_need("_autoTrack(n, j.rel" not in js, "F3 no track per Save any more")
_need("const q = (node._plsAutoQ || Promise.resolve()).then(() => _autoTrack1(node, rel, scene, nFiles, sweep))" in js, "F3 tracks are added one after the other")   # AMENDED v1200: + the sweep (v1198)
_need("projectName(_model(n)) === run.project" in js, "F3 the project compared safe-named, like the server")

# ------------------------------------------------------------------ F4
spec = importlib.util.spec_from_file_location("ce1047", str(ROOT / "nodes" / "cine_edit.py"))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
_need(E.clean_edit({"span": 300}) == {"span": 300.0} and E.clean_edit({"span": 99999}) == {"span": 3600.0} and "span" not in E.clean_edit({"span": -1}),
      "F4 the server keeps the span, up to an hour")
_need("export const FILM_H = 32, ZOOM_KEY = \"pls_cine_zoom\", SPAN_MAX = 3600;" in js and E.SPAN_MAX == 3600.0, "F4 JS and server agree on SPAN_MAX")

bad = [f for f in ("web/js/ph_cine_timeline.js", "nodes/cine_edit.py", "tests/test_v1047_film_zoom.py")
       if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not bad, "S ASCII only %r" % bad)
if FAILS:
    print("test_v1047_film_zoom: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1047_film_zoom: PASS (F1-F4, S)")
