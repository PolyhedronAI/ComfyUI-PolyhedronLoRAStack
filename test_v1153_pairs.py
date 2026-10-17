# -*- coding: ascii -*-
"""Guard v1153 -- the tracks as pairs with a clamp; the work area's sound column; the small finds (Frank 17.10.:
"V and A not apart ... a button that sorts the tracks by type, but by default V1/A1 and V2/A2 held together ...
coupled, cut together; unlinked -- shown with a clamp like Photoshop's -- each alone"; "the box must be marked
as the audio prompt"; "after the second clip the timeline should refresh by itself"; "what is render all scenes?";
"the sound tracks with their waveforms"; "a zoom slider"; "the node stands on Reference, but we have none?").

  E1  clean_layers keeps a layer's clamp (link false) and its sound shift (aoff, int, clamped); the base and an
      empty layer carry neither
  E2  shift_sound: the asegments and the seams' a_at move by aoff as one; 0 = the track itself
  E3  the export: a shifted sound gets adelay (later) or atrim (earlier) at the end of its chain; unshifted, nothing
  E4  edit_plan shifts every layer's track and the row says link / aoff
  U1  the face (statics): pairs by default (An right under Vn), "V/A by type" = the old order; the clamp on every
      pair (V1/A1 = the A/V switch); the sound column with its two names between the shot and the monitor; the zoom
      slider; render all scenes hidden with one scene; the Reference hint; the second look after a render's track;
      the sound lanes 32 / 28
  U2  the numbers (node.js): layerLinked (base = the A/V switch, a layer = its own clamp), byType
  W   the waveform normalised (dynaudnorm, sqrt scale, 64 px) with a new cache key
  S   ASCII
"""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1153_pairs"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


spec = importlib.util.spec_from_file_location("pls1153_edit", str(ROOT / "nodes" / "cine_edit.py"))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
FPS = E.FPS

# ------------------------------------------------------------------ E1
L = E.clean_layers([
    {"id": "x", "src": {"kind": "render", "path": "a.mp4"}, "pieces": [{"k": 0, "a": 0, "b": 48, "speed": 1}], "link": False, "aoff": 12.4},
    {"id": "y", "src": {"kind": "render", "path": "b.mp4"}, "pieces": [{"k": 0, "a": 0, "b": 48, "speed": 1}], "link": True, "aoff": 0},
    {"id": "z", "src": {"kind": "render", "path": "c.mp4"}, "pieces": [{"k": 0, "a": 0, "b": 48, "speed": 1}], "aoff": -1e9},
    {"id": "e", "src": {"kind": "empty"}, "link": False, "aoff": 5},
    {"id": "base", "link": False, "aoff": 5}])
by = {x["id"]: x for x in L}
_need(by["x"].get("link") is False and by["x"].get("aoff") == 12, "E1 a layer's clamp and its shift (12.4 -> 12)")
_need("link" not in by["y"] and "aoff" not in by["y"], "E1 linked, unshifted: nothing stored")
_need(by["z"]["aoff"] == -E.AOFF_MAX, "E1 the shift is clamped")
_need("link" not in by["base"] and "aoff" not in by["base"] and "aoff" not in by["e"] and by["e"].get("link") is False,
      "E1 the base carries no clamp (it is the A/V switch); an empty layer no shift")

# ------------------------------------------------------------------ E2
tr = E.track([{"k": 0, "a": 0, "b": 24, "speed": 1.0}, {"k": 0, "a": 24, "b": 48, "speed": 1.0, "ao": 4}], {0: 48}, [], [])
t2 = E.shift_sound(tr, 10)
_need(E.shift_sound(tr, 0) is tr and "aoff" not in tr, "E2 aoff 0 = the track itself")
_need([(q["film_in"], q["film_out"]) for q in t2["asegments"]] == [(a + 10, b + 10) for a, b in [(q["film_in"], q["film_out"]) for q in tr["asegments"]]]
      and t2["aoff"] == 10 and t2["seams"][0]["a_at"] == tr["seams"][0]["a_at"] + 10 and t2["segments"] == tr["segments"],
      "E2 the sound moves as one (+10), the picture stays %r" % [(q["film_in"], q["film_out"]) for q in t2["asegments"]])
t3 = E.shift_sound(tr, -30)
_need(t3["asegments"][0]["film_in"] == -30 and t3["aoff"] == -30, "E2 earlier: before frame 0 (the flatten cuts it off)")
fl = E.flatten([{"video": True, "audio": True, "has_audio": True, "track": t3}], 48)
_need(fl["aruns"][0]["layer"] == 0 and fl["aruns"][0]["film_in"] == 0 and fl["aruns"][0]["film_out"] == 18 and fl["aruns"][1]["layer"] is None,
      "E2 flatten: the shifted sound covers 0..18 only, silence after %r" % fl["aruns"])

# ------------------------------------------------------------------ E3
def film_of(aoff):
    t = E.shift_sound(E.track([{"k": 0, "a": 0, "b": 48, "speed": 1.0}], {0: 48}, [], []), aoff)
    return E.stack_film([{"id": "L1", "name": "l", "video": True, "audio": True, "has_audio": True, "track": t, "paths": ["/x.mp4"]}], 48)


g_plus = E.stack_graph(film_of(12), (64, 64), True)[0]
g_minus = E.stack_graph(film_of(-12), (64, 64), True)[0]
g_zero = E.stack_graph(film_of(0), (64, 64), True)[0]
_need("adelay=500:all=1[l0ash]" in g_plus and "[l0ash]volume=volume=0" in g_plus, "E3 later: adelay 500 ms at the end of the chain, the mix reads it")
_need("atrim=start=0.500000,asetpts=PTS-STARTPTS[l0ash]" in g_minus, "E3 earlier: the head cut off (atrim 0.5 s)")
_need("ash" not in g_zero, "E3 unshifted: nothing added")

# ------------------------------------------------------------------ E4
py = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="ascii")
_need('tr = E.shift_sound(tr, L.get("aoff", 0))' in py and 'row["link"] = L.get("link", True) is not False' in py
      and 'row["aoff"] = int(L.get("aoff", 0) or 0)' in py, "E4 edit_plan shifts the layer's track; the row names link / aoff")

# ------------------------------------------------------------------ U1
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
_need('export const BYTYPE_KEY = "pls_cine_bytype";' in js and "export function byType(node)" in js
      and "export function layerLinked(node, lay)" in js, "U1 the order switch and the pair's clamp state")
_need("if (byType(node)) st.below.insertBefore(r.a, st.below.firstChild);   // A2 right under A1" in js
      and "else st.above.appendChild(r.a);" in js, "U1 pairs by default: An right under Vn; by type = the old order")
_need("function _bracket(node, row, hV, hA, linked, fn, title)" in js and 'b.dataset.pls = "clamp";' in js
      and "_bracket(node, st.v1Row, laneHeights(node).clip, laneHeights(node).audio, isLinked(node)," in js
      and "_bracket(node, vRow, H.clip, H.audio, linked, () => _saveEdit(node, setLayer(editOf(node), lay.id, { link: !linked }))," in js,
      "U1 the clamp on every pair -- V1/A1 toggles the A/V switch, a layer its own")
_need("const v1Row = _row(v1Head, clips); v1Row.style.position = \"relative\"; v1Row.dataset.pls = \"v1row\";" in js
      and "v1Row, _row(a1Head, audio), below, anchorsRow, tl);" in js, "U1 V1's row has a handle for its clamp")
_need('_saveEdit(node, setLayer(editOf(node), lay.id, { aoff: aoff + dF }));' in js
      # AMENDED v1154 (audit): the sound keeps its place by the picture's REAL move (movePiece clamps at the gaps)
      and "if (moved) _saveEdit(node, setLayer(setLayer(editOf(node), lay.id, { pieces: q }), lay.id, { aoff: aoff - moved }));" in js,
      "U1 unlinked: the sound lane drags the sound alone, a piece drag moves the picture alone")
_need('typeBtn.dataset.pls = "bytype";' in js and "tools.append(splitBtn, healBtn, delBtn, linkBtn, typeBtn, snapSel," in js, "U1 'V/A by type' in the toolbar")
_need('sndRow.dataset.pls = "soundcol";' in js and '"overall_soundscape");' in js and '"non_diegetic_music");' in js
      and "sndRow.append(sndLab, sound, musLab, music);" in js and "work.append(left, sndRow, stack);" in js
      and "SND_W = 300" in js and "TEXT_H = 246" in js, "U1 the sound column, each field under its name, between the shot and the monitor")
_need('zSl.dataset.pls = "zoomslider";' in js and "zOut, zSl, zIn, zFit, viewInfo, nav);" in js
      and "if (st.zSl && document.activeElement !== st.zSl)" in js, "U1 the zoom slider in the FILM row, following the zoom")
_need('st.all.style.display = sc.length < 2 ? "none" : "";' in js, "U1 render all scenes only with 2+ scenes")
_need('w.dataset.pls = "refhint";' in js and 'if (md === "ref") {' in js, "U1 Reference without pictures says: Text is the mode")
_need("setTimeout(() => { try { _stack(node); } catch (x) { /* view only */ } }, 1500); }" in js, "U1 a second look at the stack after a render's track")
_need("LANE_AUDIO_H = 32" in js and "LAYER_AUDIO_H = 28" in js, "U1 the sound lanes taller")
_need('state.addEventListener("mouseenter", () => { state.title = state.textContent; });' in js and "st.target.title = st.target.textContent;" in js,
      "U1 the status texts carry their whole text as the tooltip")

# ------------------------------------------------------------------ U2
if shutil.which("node") is None:
    print("  note  node.js not found -- U2 SKIPPED")
else:
    t2 = Path(tempfile.mkdtemp(prefix="v1153js_"))
    body = (js.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (t2 / "m.mjs").write_text(body, encoding="utf-8")
    (t2 / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const n0 = { properties: {} }, n1 = { properties: { [T.LINK_KEY]: false, [T.BYTYPE_KEY]: true } };
console.log(JSON.stringify({
  a: [T.layerLinked(n0, { id: T.BASE_ID }), T.layerLinked(n1, { id: T.BASE_ID }), T.layerLinked(n1, { id: "L1" }), T.layerLinked(n0, { id: "L1", link: false }), T.layerLinked(n0, null)],
  t: [T.byType(n0), T.byType(n1), T.byType(null)] }));
""", encoding="utf-8")
    r = subprocess.run(["node", str(t2 / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(t2, ignore_errors=True)
    line = [x for x in r.stdout.splitlines() if x.startswith("{")]
    if r.returncode != 0 or not line:
        _need(False, "U2 harness failed: %s" % r.stderr.strip()[-400:])
    else:
        o = json.loads(line[-1])
        _need(o["a"] == [True, False, True, False, True] and o["t"] == [False, True, False], "U2 layerLinked / byType %r" % o)

# ------------------------------------------------------------------ W
_need("dynaudnorm=f=150:g=15:p=0.95,showwavespic=s=%dx64:colors=0x80cbc4:scale=sqrt" in py and "|v1153" in py,
      "W the waveform normalised and taller, under a new cache key")

# ------------------------------------------------------------------ S
bad = []
for f in (ROOT / "tests" / "test_v1153_pairs.py", ROOT / "nodes" / "cine_edit.py", ROOT / "nodes" / "ph_cine_timeline.py",
          ROOT / "web" / "js" / "ph_cine_timeline.js"):
    try:
        f.read_bytes().decode("ascii")
    except UnicodeDecodeError:
        bad.append(f.name)
_need(not bad, "S ASCII only %r" % bad)

if FAILS:
    print("%s: %d FAILURE(S)" % (NAME, len(FAILS)))
    sys.exit(1)
print("%s: PASS (E1-E4, U1-U2, W, S)" % NAME)
