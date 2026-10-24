# -*- coding: ascii -*-
"""Guard v1150 -- the timeline as a FORM the CLIP Text Encode fills, and the timeline's new face (Frank 17.10.:
"everything that does not change in the prompt can just as well go into the CTE -- in the timeline only what
steers the clip ... the control areas bigger, above all the buttons ... the method decides what is visible";
"I want exactly the same result after the rebuild"; "nothing falls under the table just to look tidier").

  F1  CTE external_mode "fill {pos_N}" (appended last): {pos_N} = segment N as typed, {prompt} = all shown
      segments; the filled text is cleaned once like replace; a hidden segment's place stays empty; a wired text
      without a place falls back to append
  F2  the timeline's fill_text equals the CTE's _fill on every case (two seats, held equal here)
  F3  split_to_form ("-> CTE"): the head (triggers, look) and every rubric that does not steer a shot move into
      segments, the steering ones stay with a {pos_k} where each run stood -- compiled + filled + cleaned it is
      BYTE FOR BYTE the text of before; a shot 1 of steering rubrics only moves nothing but the head
  F4  run() with the CTE's segments: the check reads the filled prompt (the preview shows it), the info names the
      form; without them it says no CTE fills it
  F5  form_fill finds the CTE whose pos_external is our output 0 in the running graph; not in fill mode or a
      wired segment -> None
  F6  IS_CHANGED follows the filling CTE's segments (other segments, other key; the same graph, the same key)
  U1  the face (statics): the sections in order, mode buttons, "-> CTE" button and route, refs row with a hover
      picture inside the node, the toolbar keeps EVERY tool, the tracks keep every row, no floating window
  U2  the numbers (node.js): fieldH = the shown sections + their gaps; modeOf (ref2va -> ref; anchors -> keys;
      else the view setting); Text hides the anchors row (ANCH_H + a gap); the toolbar's height is clamped
  S   ASCII
"""
import importlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1150_timeline_form"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


sys.path.insert(0, str(ROOT / "nodes"))
import cine_timeline as T                                      # noqa: E402

py = (ROOT / "nodes" / "ph_clip_encode.py").read_text(encoding="ascii")
ns = {"re": re}
exec(py[py.index("MAX_SEGMENTS = 6"):py.index("# ---- v876")], ns)
compose, fill, clean = ns["_compose"], ns["_fill"], ns["_clean"]
FILL = ns["FILL_MODE"]

# F1
_need(ns["EXTERNAL_MODES"] == ["append", "prepend", "replace", FILL] and FILL == "fill {pos_N}",
      "F1 fill {pos_N} appended last to external_mode")
form = "A: [Shot 1] {pos_1}\n// CAM\nstatic camera,\n{pos_2}\nend {prompt}"
segs = ["// LOOK\nwarm light,", "// SUBJECT\na woman,"]
got = compose(segs, "comma", True, False, form, FILL, "//")
want = clean(form.replace("{pos_1}", segs[0]).replace("{pos_2}", segs[1]).replace("{prompt}", "\n".join(segs)),
             True, False, "//")
_need(got == want and "{pos" not in got and "warm light," in got and "static camera," in got,
      "F1 the places take the segments as typed, cleaned once like replace")
_need(compose(segs[:1], "comma", True, False, form, FILL, "//").count("a woman") == 0,
      "F1 a segment that is not shown leaves its place empty")
_need(compose(segs, "comma", True, False, "no places here", FILL, "//")
      == compose(segs, "comma", True, False, "no places here", "append", "//"), "F1 no place -> append")

# F2
for txt, parts in ((form, segs), ("{prompt}", ["x", "", "  ", "y"]), ("{pos_3}{pos_1}", ["a"]), ("plain", ["a"])):
    _need(T.fill_text(txt, parts) == fill(txt, parts)[0], "F2 timeline fill == CTE fill: %r" % txt[:20])

# F3
SHOT1 = ("// SCENE\na loft at golden hour,\n// LIGHT\nwarm sun,\n// CAMERA\nstatic camera, a medium wide shot,\n"
         "// SUBJECT\na 25 year old adult woman,\n// HAIR\nlong blonde hair,\n// MOOD\nlanguid,\n// MOTION\n"
         "She turns her head slowly,")
model = T.load(json.dumps({"seconds": 5.875, "triggers": "t1, t2,", "style": "photorealistic, film still,",
                           "target": "ref2va", "shots": [{"at": 0, "text": SHOT1},
                                                         {"at": 2.0, "text": "// SHOT\na close-up,\n// MOTION\nshe smiles,"}],
                           "soundscape": "a quiet room", "music": "N/A"}))
before = T.compile_prompt(model)["prompt"]
new, sg = T.split_to_form(model)
after = T.compile_prompt(new)["prompt"]
_need(len(sg) == 2 and sg[0].startswith("// TRIGGERS\nt1, t2,\n// LOOK\n") and sg[0].rstrip().endswith("warm sun,")
      and sg[1].startswith("// SUBJECT") and sg[1].rstrip().endswith("languid,"),
      "F3 two runs moved: head + SCENE + LIGHT, SUBJECT .. MOOD")
_need(new["shots"][0]["text"].split("\n")[0] == "{pos_1}" and "// CAMERA" in new["shots"][0]["text"]
      and "{pos_2}" in new["shots"][0]["text"] and "// MOTION" in new["shots"][0]["text"]
      and new["triggers"] == "" and new["style"] == "", "F3 Shot 1 keeps the steering rubrics and the places")
_need(T.fill_text(after, sg) == before, "F3 filled, the compiled prompt is byte for byte the one of before")
_need(compose(sg, "comma", True, False, after, FILL, "//") == compose([], "comma", True, False, before, "replace", "//"),
      "F3 through the CTE: the encoder sees the very text it saw with replace")
only = T.load(json.dumps({"seconds": 5.875, "triggers": "t1,", "style": "", "shots": [{"at": 0, "text": "// SHOT\nwide,\n// MOTION\nshe walks,"}]}))
n2, s2 = T.split_to_form(only)
_need(len(s2) == 1 and s2[0] == "// TRIGGERS\nt1," and T.fill_text(T.compile_prompt(n2)["prompt"], s2) == T.compile_prompt(only)["prompt"],
      "F3 steering rubrics only: just the head moves, still byte for byte")

# F4 / F5 / F6 -- the node module
import types                                                    # noqa: E402
TMP = Path(tempfile.mkdtemp(prefix="pls1150_"))
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: str(TMP / "output")
fp.get_input_directory = lambda: str(TMP / "input")
fp.get_temp_directory = lambda: str(TMP / "temp")
fp.models_dir = str(TMP)
fp.folder_names_and_paths = {}
fp.add_model_folder_path = lambda n, p: None
fp.get_filename_list = lambda n: []
sys.modules.setdefault("folder_paths", fp)
pkg = types.ModuleType("plsn1150")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1150"] = pkg
try:
    P = importlib.import_module("plsn1150.ph_cine_timeline")
except Exception as e:                                          # the node module needs comfy for its routes only
    P = None
    print("  note  ph_cine_timeline not importable here (%s) -- F4-F6 SKIPPED" % type(e).__name__)
if P is not None:
    r, errs = P.run(new, None, None, sg)
    r0, _e = P.run(new, None, None, None)
    _need(not errs and r.get("filled") == before and r["places"] == ["pos_1", "pos_2"]
          and any("filled from the CLIP Text Encode" in l for l in r["lines"]), "F4 with the segments: filled, named")
    _need(not r0.get("filled") and any("no such CTE" in l for l in r0["lines"]), "F4 without: the info says no CTE fills it")
    graph = {"7": {"class_type": "ULSCLIPTextEncode", "inputs": {"pos_external": ["3", 0], "external_mode": FILL,
                                                                   "segments": 2, "pos_1": sg[0], "pos_2": sg[1]}}}
    _need(P.form_fill(graph, "3") == sg and P.form_fill(graph, "4") is None, "F5 the CTE on our output 0 is found")
    g2 = json.loads(json.dumps(graph)); g2["7"]["inputs"]["external_mode"] = "replace"
    g3 = json.loads(json.dumps(graph)); g3["7"]["inputs"]["pos_2"] = ["9", 0]
    g5 = json.loads(json.dumps(graph)); g5["7"]["inputs"]["pos_external"] = ["3", 6]
    _need(P.form_fill(g2, "3") is None and P.form_fill(g3, "3") is None and P.form_fill(g5, "3") is None,
          "F5 not fill mode / a wired segment / another output of ours (info) -> None")
    tl = json.dumps(new)
    k1 = P.ULSCineTimeline.IS_CHANGED(tl, prompt=graph, unique_id="3")
    k1b = P.ULSCineTimeline.IS_CHANGED(tl, prompt=json.loads(json.dumps(graph)), unique_id="3")
    g4 = json.loads(json.dumps(graph)); g4["7"]["inputs"]["pos_2"] = sg[1] + " more"
    k2 = P.ULSCineTimeline.IS_CHANGED(tl, prompt=g4, unique_id="3")
    _need(k1 == k1b and k1 != k2 and "form:" in k1, "F6 IS_CHANGED follows the CTE's segments")
    _need('("POST", "/pls/cine/to_form", handle_to_form)' in (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="ascii"),
          "U1 the -> CTE route is registered")

# U1 statics
# AMENDED v1183: the preview keeps the fill it sent only when the answer is still the newest (previewSeq)
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
fld = js[js.index("function _field("):]
_need("root.append(topBar, headRow, refsRow, work, state, tools, film, scroller, bar, status, prev, pop);" in fld,
      "U1 the sections: top bar, head row, refs, work, state, toolbar, FILM, tracks, bar, status, preview")
_need("topBar.append(modeSeg, tgt," in fld and "all, run);" in fld and 'run.style.cssText = BTN_GO;' in fld,
      "U1 the top bar: mode, length, project, render choice, render all, the big Render")
_need("tools.append(splitBtn, healBtn, delBtn, linkBtn, typeBtn, snapSel, sep(), resetBtn, snapBtn, cutsBtn, tailLab,\n        sep(), autoLab, restLab, takesBtn, clearBtn);" in fld,   # AMENDED v1153: + V/A by type; AMENDED v1198: + x tracks; AMENDED v1221: + snap to scenes (v1219)
      "U1 the toolbar keeps EVERY tool (cut, heal, delete, A/V linked, snap, reset, cuts->prompt, keep tail, the takes switches)")
_need('lanes.append(_row(tc, ruler), _row(gut("Scenes"), scenes), _row(gut("Shots"), shots), _row(newHead, newLane), above,' in fld
      and "v1Row, _row(a1Head, audio), below, anchorsRow, tl);" in fld,   # AMENDED v1153
      "U1 the tracks keep every row (new track, V tracks, V1, A1, A tracks, anchors)")
_need('toCte.dataset.pls = "tocte";' in fld and '"/pls/cine/to_form"' in js and "fill: filled }" in js and "st.filled = filled;" in js,
      "U1 -> CTE button + route; the preview sends the CTE's segments")
_need('pop.dataset.pls = "refpop";' in fld and "st.pop.style.display = \"block\";" in js and "position:fixed" not in js,
      "U1 the refs show big on hover INSIDE the node (no floating window)")
_need('if (st.anchorsRow) st.anchorsRow.style.display = md === "text" ? "none" : "flex";' in js,
      "U1 Text hides the anchors row only")

# U2 numbers
if shutil.which("node") is None:
    print("  note  node.js not found -- U2 SKIPPED")
else:
    tmp = Path(tempfile.mkdtemp(prefix="v1150js_"))
    body = (js.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (tmp / "m.mjs").write_text(body, encoding="utf-8")
    (tmp / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const o = {};
const n1 = { _plsTl: { m: { triggers: "x", anchors: [] }, toolsH: 70, prevH: 222 }, properties: { pls_cine_prev_open: true, pls_cine_mode: "keys" } };
const S = T.sections(n1), on = S.filter((x) => x[2]);
o.sum = [T.fieldH(n1), on.reduce((a, x) => a + x[1], 0) + T.ROOT_GAP * (on.length - 1)];
o.keys = S.map((x) => x[0]);
o.tools = S.find((x) => x[0] === "tools")[1];
o.mode = [T.modeOf({}, { target: "ref2va", anchors: [] }), T.modeOf({}, { anchors: [{ at: 0 }] }), T.modeOf({ properties: { pls_cine_mode: "keys" } }, { anchors: [] }),
          T.modeOf({ properties: { pls_cine_mode: "zz" } }, { anchors: [] })];
o.lanes = [T.lanesH({}, "keys") - T.lanesH({}, "text"), T.ANCH_H + T.LANE_GAP];
o.th = [T.toolsHeight(0), T.toolsHeight(55.2), T.toolsHeight(1e6), T.TOOLS_H, T.TOOLS_MAX];
console.log(JSON.stringify(o));
""", encoding="utf-8")
    r = subprocess.run(["node", str(tmp / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(tmp, ignore_errors=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        _need(False, "U2 harness: %s" % r.stderr.strip()[-300:])
    else:
        d = json.loads(line[-1])
        _need(d["sum"][0] == d["sum"][1], "U2 fieldH = the shown sections + their gaps (%r)" % d["sum"])
        _need(d["keys"] == ["topbar", "headrow", "refs", "work", "state", "tools", "film", "scroller", "bar", "status", "prev"]
              and d["tools"] == 70, "U2 the sections in order; the toolbar's fitted height counts")
        _need(d["mode"] == ["ref", "keys", "keys", "text"], "U2 modeOf: ref2va -> ref, anchors -> keys, else the view (%r)" % d["mode"])
        _need(d["lanes"][0] == d["lanes"][1], "U2 Text hides exactly the anchors row (%r)" % d["lanes"])
        _need(d["th"][:3] == [d["th"][3], 56, d["th"][4]], "U2 the toolbar's height clamped to [TOOLS_H, TOOLS_MAX] (%r)" % d["th"])

shutil.rmtree(TMP, ignore_errors=True)
for rel in ("nodes/ph_clip_encode.py", "nodes/cine_timeline.py", "nodes/ph_cine_timeline.py", "web/js/ph_cine_timeline.js",
            "web/js/ph_clip_encode.js", "tests/test_v1150_timeline_form.py"):
    _need(all(b < 128 for b in (ROOT / rel).read_bytes()), "S ASCII: " + rel)

if FAILS:
    print("[%s] FAIL -- %d: %s" % (NAME, len(FAILS), "; ".join(FAILS)))
    sys.exit(1)
print("[%s] PASS -- the timeline is a form the CTE fills, byte for byte; the face keeps every tool" % NAME)
