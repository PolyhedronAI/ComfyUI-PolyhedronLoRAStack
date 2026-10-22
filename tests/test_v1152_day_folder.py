# -*- coding: ascii -*-
"""Guard v1152 -- the form's places head every run; the timeline writes into the day folder (Frank 17.10.:
"then build" on the scene-run head; "the whole lands in a folder outside again ... I want everything in the
folder of the day ... no task folders any more").

  H1  a scene run that does not begin with the timeline's Shot 1 carries Shot 1's places ({pos_1} {pos_2})
      at the head of its [Shot 1] -- INSIDE the description, before its own text; the whole run and scene 1
      are byte for byte as before; a timeline without places adds nothing; the info says so
  H2  filled like the CTE does it, the scene run's fixed parts stand after "[Shot 1] " (not before the
      description, as the append fallback put them)
  P1  clip_prefix = %date:yyyy-MM-dd%/PH_%date:hhmmss%_<project>_whole | _scene_k; JS runOf names the same;
      pickTrackFile reads the tokens (scene_2 is not scene_20)
  P2  the project's folder is input/cine_media/<project>/stack/; register() keeps paths relative to OUTPUT/
      (A.OUTPUT_ROOT); nothing is created under output/<project>/
  P3  an old output/<project>/ is taken over: its list, primaries and scene_k / whole clips -- incrementally
      (a clip that lands there later joins too), written only when something is added, nothing deleted
  P4  exports and the scene join go to the day folder as PH_<hhmmss>_<project>_film (both export seats, the
      assemble route); the run's ui names the project (the JS no longer parses it from the prefix)
  S   ASCII
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1152_day_folder"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


sys.path.insert(0, str(ROOT / "nodes"))
import cine_timeline as T                                      # noqa: E402

spec = importlib.util.spec_from_file_location("pls1152_assemble", str(ROOT / "nodes" / "cine_assemble.py"))
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)

FORM1 = "{pos_1}\n// CAMERA\nstatic camera,\n{pos_2}\n// MOTION\nshe lies still,"


def model(**kw):
    m = {"seconds": 9.0, "project": "boudoir feet", "render": "whole", "active": 1, "target": "fl2va",
         "triggers": "", "style": "", "anchors": [],
         "shots": [{"at": 0, "text": FORM1},
                   {"at": 2.0, "text": "// SHOT\nclose-up,\n// MOTION\nshe lifts her hand,"},
                   {"at": 4.5, "scene": True, "text": "// SHOT\nwide,\n// MOTION\nshe sits up,"},
                   {"at": 6.5, "text": "// SHOT\nher feet,\n// MOTION\nshe stretches,"}]}
    m.update(kw)
    return m


# ------------------------------------------------------------------ H1
m = model()
_need(T.validate(m) == [], "H1 the model validates")
whole = T.compile_prompt(T.resolve(m)[0])["prompt"]
_need("form_head" not in T.resolve(m)[0] and whole.startswith("integrated_multimodal_description: [Shot 1] " + FORM1),
      "H1 the whole run: Shot 1 as typed, no extra head")
s1 = T.compile_prompt(T.resolve(model(render="scene", active=1))[0])
_need(s1["prompt"].startswith("integrated_multimodal_description: [Shot 1] " + FORM1) and s1["prompt"].count("{pos_1}") == 1,
      "H1 scene 1 begins with the timeline's Shot 1 -- its places once, nothing added")
r2 = T.compile_prompt(T.resolve(model(render="scene", active=2))[0])
p2 = r2["prompt"]
_need(p2.startswith("integrated_multimodal_description: [Shot 1] {pos_1}\n{pos_2}\n// SHOT\nwide,"),
      "H1 scene 2: Shot 1's places head its [Shot 1], before its own text %r" % p2[:90])
_need(p2.count("{pos_1}") == 1 and p2.count("{pos_2}") == 1 and "static camera" not in p2 and "she lies still" not in p2,
      "H1 only the places travel -- not Shot 1's steering rubrics (CAMERA, MOTION)")
_need(any("the form's places {pos_1} {pos_2}" in ln for ln in r2["lines"]), "H1 the info names the places of the head")
mt = model(shots=[dict(s, text=s["text"].replace("{pos_1}\n", "").replace("{pos_2}\n", "")) for s in model()["shots"]])
r3 = T.compile_prompt(T.resolve(dict(mt, render="scene", active=2))[0])
_need("{pos_" not in r3["prompt"] and not any("form's places" in ln for ln in r3["lines"]),
      "H1 a timeline without places adds nothing")
sm = T.scene_model(model(), 2)
sm["shots"][0]["text"] = "{pos_2}\n" + sm["shots"][0]["text"]
_need(T.compile_prompt(sm)["prompt"].count("{pos_1}") == 0, "H1 a scene shot that already carries a place keeps its own")

# ------------------------------------------------------------------ H2
segs = ["// TRIGGERS\nph0t0r34l,\n// LOOK\nfilm still,", "// SUBJECT\na woman,"]
filled = T.fill_text(p2, segs)
_need(filled.startswith("integrated_multimodal_description: [Shot 1] // TRIGGERS\nph0t0r34l,") and "a woman,\n// SHOT\nwide," in filled,
      "H2 filled, the fixed parts stand INSIDE the description, after [Shot 1]")

# ------------------------------------------------------------------ P1
D = "%date:yyyy-MM-dd%/PH_%date:hhmmss%_"
_need(T.DAY_PREFIX == D and T.clip_prefix(m, 0) == D + "boudoir_feet_whole" and T.clip_prefix(m, 2) == D + "boudoir_feet_scene_2",
      "P1 clip_prefix: the day folder, PH_<hhmmss>_<project>_whole | _scene_k")
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
_need('export const DAY_PREFIX = "%date:yyyy-MM-dd%/PH_%date:hhmmss%_";' in js
      and 'clip_prefix: DAY_PREFIX + p + "_" + (k ? "scene_" + k : "whole")' in js, "P1 JS runOf names the same prefix")
if shutil.which("node") is None:
    print("  note  node.js not found -- P1 JS part SKIPPED")
else:
    t2 = Path(tempfile.mkdtemp(prefix="v1152js_"))
    body = (js.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (t2 / "m.mjs").write_text(body, encoding="utf-8")
    (t2 / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const m = { seconds: 9, shots: [{ at: 0 }, { at: 4.5, scene: true }], project: "boudoir feet", render: "scene", active: 2 };
const p = T.runOf(m).clip_prefix;
const day = ["2026-10-17/PH_101010_MINIMAX_00001_.mp4", "2026-10-17/PH_101011_boudoir_feet_scene_2_00001_.mp4", "2026-10-17/PH_101012_x_00001_.mp4"];
console.log(JSON.stringify({ p, a: T.pickTrackFile(day, p), b: T.pickTrackFile(["2026-10-17/PH_1_boudoir_feet_scene_20_00001_.mp4", "z.mp4"], p),
  c: T.prefixRx(p).test("2026-10-17/sub/PH_1_boudoir_feet_scene_2_00001_.mp4") }));
""", encoding="utf-8")
    r = subprocess.run(["node", str(t2 / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(t2, ignore_errors=True)
    line = [x for x in r.stdout.splitlines() if x.startswith("{")]
    if r.returncode != 0 or not line:
        _need(False, "P1 harness failed: %s" % r.stderr.strip()[-400:])
    else:
        o = json.loads(line[-1])
        _need(o["p"] == D + "boudoir_feet_scene_2" and o["a"] == "2026-10-17/PH_101011_boudoir_feet_scene_2_00001_.mp4"
              and o["b"] == "z.mp4" and o["c"] is False, "P1 pickTrackFile reads the tokens, one folder level each %r" % o)

# ------------------------------------------------------------------ P2 / P3
TMP = Path(tempfile.mkdtemp(prefix="v1152_"))
out, inp = TMP / "output", TMP / "input"
(out / "2026-10-17").mkdir(parents=True)
inp.mkdir()
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: str(out)
fp.get_input_directory = lambda: str(inp)
sys.modules["folder_paths"] = fp
src_node = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="ascii")
ns = {"os": os, "A": A, "T": T}
# AMENDED v1182: stack_items removed (with the dead /pls/cine/scenes route) -- the slice ends at register_clip
exec(src_node[src_node.index("def _output_dir():"):src_node.index("def register_clip")], ns)
exec(src_node[src_node.index("def register_clip"):src_node.index("async def handle_register")], ns)
pf = ns["project_folder"]("boudoir feet")
_need(pf == os.path.join(str(inp), "cine_media", "boudoir_feet", "stack") and A.OUTPUT_ROOT == str(out),
      "P2 the project's folder is input/cine_media/<project>/stack/, the root named output/")
(out / "2026-10-17" / "PH_101011_boudoir_feet_whole_00001_.mp4").write_bytes(b"w")
rel = ns["register_clip"]("boudoir feet", 0, {"filename": "PH_101011_boudoir_feet_whole_00001_.mp4", "subfolder": "2026-10-17", "type": "output"}, True)
_need(rel == "2026-10-17/PH_101011_boudoir_feet_whole_00001_.mp4" and A.read_list(pf) == {0: [rel]} and A.read_primary(pf)[0]["rel"] == rel,
      "P2 a day-folder clip is listed relative to output/ (and marked primary)")
_need(sorted(os.listdir(out)) == ["2026-10-17"], "P2 nothing is created under output/ but the day folder %r" % sorted(os.listdir(out)))
tk = A.scene_takes(pf, 1)
_need(tk.get(0) and tk[0][0] == os.path.realpath(str(out / "2026-10-17" / "PH_101011_boudoir_feet_whole_00001_.mp4")),
      "P2 the stack finds the clip where the Save wrote it")
# P3 an old project folder
old = out / "fox"
old.mkdir()
(old / "scene_1_00001_.mp4").write_bytes(b"1")
(out / "elsewhere").mkdir()
(out / "elsewhere" / "c_00001_.mp4").write_bytes(b"c")
json.dump({"v": 1, "scenes": {"2": ["elsewhere/c_00001_.mp4"]}, "primary": {"2": {"rel": "elsewhere/c_00001_.mp4", "at": 5.0}}},
          open(old / "stack.json", "w"))
fx = ns["project_folder"]("fox")
_need(A.read_list(fx) == {1: ["fox/scene_1_00001_.mp4"], 2: ["elsewhere/c_00001_.mp4"]} and A.read_primary(fx)[2]["rel"] == "elsewhere/c_00001_.mp4",
      "P3 the old folder's list, primary and named clips are taken over %r" % A.read_list(fx))
mt0 = os.path.getmtime(os.path.join(fx, A.STACK_LIST))
time.sleep(0.05)
ns["project_folder"]("fox")
_need(os.path.getmtime(os.path.join(fx, A.STACK_LIST)) == mt0, "P3 nothing new -- the list is not written again")
(old / "scene_1_00002_.mp4").write_bytes(b"2")
ns["project_folder"]("fox")
_need(A.read_list(fx)[1] == ["fox/scene_1_00001_.mp4", "fox/scene_1_00002_.mp4"], "P3 a clip that lands there later joins (incremental)")
_need(sorted(os.listdir(old)) == ["scene_1_00001_.mp4", "scene_1_00002_.mp4", "stack.json"], "P3 the old folder is left as it was")
_need(A.scene_takes(fx, 2).get(1, [None])[0].endswith("scene_1_00002_.mp4"), "P3 the old clips stay takes, the newest first")
shutil.rmtree(TMP, ignore_errors=True)
A.OUTPUT_ROOT = None

# ------------------------------------------------------------------ P4
_need(src_node.count("out = A.next_free(day_folder(), film_stem(model))") == 2 and 'os.path.join(d["folder"], "film")' not in src_node,
      "P4 both export seats write to the day folder as PH_<hhmmss>_<project>_film")
# AMENDED v1182: the /pls/cine/assemble route (the scene join that took out_dir=day_folder()) was dead and is gone;
# the library still writes to out_dir when it is given one
_need('"/pls/cine/assemble"' not in src_node and "out_dir or os.path.join(folder, \"assembled\")" in
      (ROOT / "nodes" / "cine_assemble.py").read_text(encoding="ascii"), "P4 the dead join route is gone; assemble() still takes out_dir")
ns2 = {"os": os, "T": T, "_output_dir": lambda: "/o"}
# AMENDED v1182: stack_items removed (with the dead /pls/cine/scenes route) -- the slice ends at register_clip
exec(src_node[src_node.index("def day_folder():"):src_node.index("def register_clip")], ns2)
_need(ns2["day_folder"]() == os.path.join("/o", time.strftime("%Y-%m-%d"))
      and re.match(r"^PH_\d{6}_boudoir_feet_film$", ns2["film_stem"]({"project": "boudoir feet"})) is not None,
      "P4 day_folder = output/<yyyy-MM-dd>, film_stem = PH_<hhmmss>_<project>_film")
_need('"clip_prefix": r["clip_prefix"], "project": T.project_name(model)}' in src_node
      and "project: t.project ? String(t.project) : projectOf(t.clip_prefix)" in js, "P4 the run's ui names the project; the JS reads it")

# ------------------------------------------------------------------ S
bad = []
for f in (ROOT / "tests" / "test_v1152_day_folder.py", ROOT / "nodes" / "cine_timeline.py", ROOT / "nodes" / "cine_assemble.py",
          ROOT / "nodes" / "ph_cine_timeline.py", ROOT / "web" / "js" / "ph_cine_timeline.js"):
    try:
        f.read_bytes().decode("ascii")
    except UnicodeDecodeError:
        bad.append(f.name)
_need(not bad, "S ASCII only %r" % bad)

if FAILS:
    print("%s: %d FAILURE(S)" % (NAME, len(FAILS)))
    sys.exit(1)
print("%s: PASS (H1-H2, P1-P4, S)" % NAME)
