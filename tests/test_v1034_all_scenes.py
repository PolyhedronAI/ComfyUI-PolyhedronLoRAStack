# -*- coding: ascii -*-
"""Guard v1034 -- three parts from one Run, and the stack without a cable.

Frank, 29.09.: a timeline of three scenes gave him one clip; the prompts of
scenes 2 and 3 never went through, and the stack stayed empty (his Save was
not wired to clip_prefix).

  A1  the model: render "all" is valid; a run with render "all" renders the
      ACTIVE scene (the editor sets active = 1..n per queued run) and says so.
  A2  the stack list: register() takes only video files under output/,
      stores them relative, dedupes, keeps the newest 20; scene_files() merges
      the list with the folder scan -- the newest file per scene wins; a
      listed file that is gone is ignored; no list = the old behaviour.
  A3  the route: register_clip() refuses non-output refs and path escapes.
  A4  the editor's pure parts: parseTimeline keeps "all", splitOrder (back
      of the queue 1..n, front n..1), videoRefs (only output videos),
      projectOf, stackHint for whole / scene / all / one scene.
  S   statics: the Run is split ONCE per queued run and awaited in order,
      the timeline restored in finally, never re-split; the executed
      listener pairs the timeline's scene with the Save's file by prompt id
      and posts /pls/cine/register; the route is registered; chip room.

Script-style: exit 0 = pass.
"""
import importlib
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
FAILS = []


def _fail(m):
    FAILS.append(m)
    print("  FAIL  " + m)


def _ok(m):
    print("  ok    " + m)


def _need(c, m):
    (_ok if c else _fail)(m)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("ct1034", "nodes/cine_timeline.py")
A = _load("ca1034", "nodes/cine_assemble.py")

# ------------------------------------------------------------------ A1
m = T.load(json.dumps({"seconds": 10, "render": "all", "active": 2,
                       "shots": [{"at": 0, "text": "a"}, {"at": 3.5, "text": "b", "scene": True},
                                 {"at": 7, "text": "c", "scene": True}]}))
_need(T.validate(m) == [] and "all" in T.RENDER_MODES, "A1 render 'all' is valid")
tgt, k, n = T.resolve(m)
# AMENDED v1152: the prefix names the day folder -- ..._<project>_scene_2
_need((k, n) == (2, 3) and T.clip_prefix(m, k).endswith("_scene_2"), "A1 an 'all' run renders the active scene (2 of 3) -> scene_2")
m5 = dict(m, active=5)
_need(any("does not exist" in e for e in T.validate(m5)), "A1 an 'all' run with a missing active scene is refused by name")

# the node's run() says it (lifted, not imported: the module needs the pack)
src_node = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
_need('if model.get("render") == "all":' in src_node and "all scenes, one run each" in src_node,
      "A1 the run's info line names 'all scenes, one run each'")

# ------------------------------------------------------------------ A2
tmp = Path(tempfile.mkdtemp(prefix="v1034_"))
out = tmp / "output"
fox, els = out / "fox", out / "elsewhere"
fox.mkdir(parents=True)
els.mkdir(parents=True)
(tmp / "outside.mp4").write_bytes(b"x")
(fox / "scene_1_00001_.mp4").write_bytes(b"1")
(els / "clip_00001_.mp4").write_bytes(b"2")
(els / "still.png").write_bytes(b"p")
time.sleep(0.02)
_need(A.scene_files(str(fox), 3).keys() == {1}, "A2 no list: the folder scan alone (the old behaviour)")
rel = A.register(str(fox), 2, str(els / "clip_00001_.mp4"))
_need(rel == "elsewhere/clip_00001_.mp4", "A2 stored relative to output/: %s" % rel)
bad = []
for what, path in (("outside output/", tmp / "outside.mp4"), ("a still", els / "still.png"), ("missing", els / "nope.mp4")):
    try:
        A.register(str(fox), 2, str(path))
        bad.append(what)
    except ValueError:
        pass
_need(not bad, "A2 refused: a file outside output/, a still, a missing file %r" % bad)
A.register(str(fox), 2, str(els / "clip_00001_.mp4"))
_need(A.read_list(str(fox)) == {2: ["elsewhere/clip_00001_.mp4"]}, "A2 registering twice keeps ONE entry")
sf = A.scene_files(str(fox), 3)
_need(set(sf) == {1, 2} and sf[2]["path"].endswith("clip_00001_.mp4"), "A2 the listed clip is scene 2 of the stack")
time.sleep(0.05)
(fox / "scene_2_00001_.mp4").write_bytes(b"newer")
sf = A.scene_files(str(fox), 3)
_need(sf[2]["name"] == "scene_2_00001_.mp4" and sf[2]["runs"] == 2, "A2 the NEWEST clip of a scene wins, wherever it lies (runs 2)")
os.remove(fox / "scene_2_00001_.mp4")
os.remove(els / "clip_00001_.mp4")
_need(2 not in A.scene_files(str(fox), 3), "A2 a listed clip that is gone is ignored")
# ... and does not count as a run next to a clip that is there (the tile's "x2")
(fox / "scene_2_00002_.mp4").write_bytes(b"there")
_need(A.scene_files(str(fox), 3)[2]["runs"] == 1, "A2 a gone listed clip is not counted as a run (runs 1, not 2)")
os.remove(fox / "scene_2_00002_.mp4")
(els / "c.mp4").write_bytes(b"c")
for i in range(25):
    (els / ("c%02d.mp4" % i)).write_bytes(b"c")
    A.register(str(fox), 3, str(els / ("c%02d.mp4" % i)))
_need(len(A.read_list(str(fox))[3]) == 20, "A2 the list keeps the newest 20 per scene")
# AMENDED v1154: a temp file of its own per write -- none of them is left
_need(not [x for x in os.listdir(fox) if x.endswith(".tmp")], "A2 the list is written atomically (no .tmp left)")

# ------------------------------------------------------------------ A3
# register_clip lifted from the node module: it only needs A, T and an output dir
code = src_node[src_node.index("def register_clip"):src_node.index("async def handle_register")]
# AMENDED v1182: stack_items removed (with the dead /pls/cine/scenes route) -- the slice ends at register_clip
pdef = src_node[src_node.index("def project_folder"):src_node.index("def register_clip")]
ns = {"os": os, "A": A, "T": T, "_output_dir": lambda: str(out)}
# AMENDED v1152: project_folder lives in input/cine_media/<project>/stack/ (folder_paths names input/)
_fp = types.ModuleType("folder_paths"); _fp.get_input_directory = lambda: str(out.parent / "input"); sys.modules["folder_paths"] = _fp
exec(pdef + "\n" + code, ns)
(els / "ok_00001_.mp4").write_bytes(b"o")
refused = []
for ref in ({"filename": "ok_00001_.mp4", "subfolder": "elsewhere", "type": "temp"},
            {"filename": "../../outside.mp4", "subfolder": "../..", "type": "output"},
            {"subfolder": "elsewhere", "type": "output"}, "nope"):
    try:
        ns["register_clip"]("fox", 1, ref)
        refused.append(ref)
    except ValueError:
        pass
_need(not refused, "A3 refused: a temp file, a path escape, no filename, not a ref %r" % refused)
_need(ns["register_clip"]("fox", 1, {"filename": "ok_00001_.mp4", "subfolder": "elsewhere", "type": "output"}) == "elsewhere/ok_00001_.mp4",
      "A3 an output video is listed")
shutil.rmtree(tmp, ignore_errors=True)
_need('("POST", "/pls/cine/register", handle_register)' in src_node, "A3 the route is registered")

# ------------------------------------------------------------------ A4 + S
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
setup = js[js.index("async setup()"):]
_need("if (!node || splitting) return orig.apply(this, arguments);" in setup and "splitting = true;" in setup
      and "res = await orig.call(this, number, batchCount);" in setup and "for (const k of splitOrder(n, number))" in setup
      and re.search(r"finally \{\s*const mm = _model\(node\); mm\.active = keep; _save\(node, mm, false\);\s*splitting = false;", setup) is not None
      and "split.__plsSplit = true;" in setup and "!orig.__plsSplit" in setup,
      "S one Run is split per scene, each awaited in order; restored in finally; never split twice")
_need('m.render === "all" && scenesOf(m).length > 1' in setup, "S only a timeline on 'all' with 2+ scenes splits")
_need('api.addEventListener("executed"' in setup and "d.output.pls_timeline" in setup and "runs.get(d.prompt_id)" in setup
      and "videoRefs(d.output)" in setup and '"/pls/cine/register"' in setup,
      "S the executed listener pairs the timeline's scene with the Save's file by prompt id")
# v1036 RE-GROUNDED: the transition chips moved from the scene lane to the
# seams of the clip lane -- nothing covers a scene label any more
_need("CHIP_ROOM" not in js and 'padding:2px 6px;box-sizing:border-box;cursor:pointer;"' in js
      and 'chip.dataset.pls = "seam";' in js, "S scene labels are free: the chips sit on the clip lane's seams")
# AMENDED v1150: the stack line sits in the status row
_need("st.stackLabel.textContent = h" in js and "status.append(chips, prevBtn, stackLabel, target);" in js, "S the stack line carries the hint")

if shutil.which("node") is None:
    print("  note  node.js not found -- A4 SKIPPED")
else:
    t2 = Path(tempfile.mkdtemp(prefix="v1034js_"))
    body = (js.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (t2 / "m.mjs").write_text(body, encoding="utf-8")
    (t2 / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const o = {};
o.parse = [T.parseTimeline(JSON.stringify({ render: "all" })).render, T.parseTimeline(JSON.stringify({ render: "bogus" })).render];
o.order = [T.splitOrder(3, 0), T.splitOrder(3, -1), T.splitOrder(0, 0)];
o.refs = T.videoRefs({ ph_save: [{ filename: "a.mp4", subfolder: "x", type: "output" }, { filename: "b.png", type: "output" }, { filename: "c.webm", type: "temp" }],
                       images: [{ filename: "d.MOV" }], text: ["e.mp4"], gifs: "no" });
o.proj = [T.projectOf("fox/scene_2"), T.projectOf(""), T.projectOf("whole")];
const m = (r, a) => ({ render: r, active: a, project: "fox" });
o.hint = [T.stackHint(m("whole", 1), 3), T.stackHint(m("scene", 2), 3), T.stackHint(m("all", 1), 3), T.stackHint(m("all", 1), 1)];
console.log(JSON.stringify(o));
""", encoding="utf-8")
    r = subprocess.run(["node", str(t2 / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(t2, ignore_errors=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        _fail("A4 harness failed: %s" % r.stderr.strip()[-400:])
    else:
        d = json.loads(line[-1])
        _need(d["parse"] == ["all", "whole"], "A4 parseTimeline keeps 'all', anything unknown is 'whole'")
        _need(d["order"] == [[1, 2, 3], [3, 2, 1], []], "A4 split order: back of the queue 1..n, front n..1 (so they still RUN 1..n)")
        _need(d["refs"] == [{"filename": "a.mp4", "subfolder": "x", "type": "output"}, {"filename": "d.MOV", "subfolder": "", "type": "output"}],
              "A4 videoRefs: output videos only (no stills, no temp files) %r" % d["refs"])
        _need(d["proj"] == ["fox", "cine", "whole"], "A4 projectOf reads the folder from clip_prefix")
        h = d["hint"]
        _need("whole = ONE clip" in h[0] and "scene 2 of 3 only" in h[1] and "Run = 3 runs" in h[2] and "clip lane above" in h[3],
              "A4 the stack line says what the next Run does (whole / scene / all / one scene)")

print()
if FAILS:
    print("test_v1034_all_scenes: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1034_all_scenes: PASS (A1-A4, S)")
