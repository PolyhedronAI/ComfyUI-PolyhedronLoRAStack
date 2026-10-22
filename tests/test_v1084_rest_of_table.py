# -*- coding: ascii -*-
"""Guard v1084 -- the Timeline audit's block C (the smaller logic finds).

  R1  h3_prompt.strip_rubrics: a '//' at a line start or after whitespace
      takes the rest of its line, an emptied line goes, the count is
      returned, "https://" is spared; the MiniMax Keyframes and Reference
      nodes strip before the resolve (text pins), the timeline's own check
      reads the stripped prompt -> no "line 3 is a // rubric" warning for a
      head with triggers (driven through run()).
  R2  the timeline's @tag line is honest: Reference resolves board tags,
      Keyframes only @first_frame / @last_frame -- a board tag is named as an
      error at Keyframes (driven).
  R3  adopt() keeps the project's fields across a storyboard take-over:
      project, triggers, target, carry_mode, render, active; transitions
      when the scene count matches (dropped with a note otherwise); the
      note names what was kept (driven with a real storyboard answer).
  R4  IS_CHANGED says what it could not read and returns NaN (the node
      re-runs) instead of "" (cached forever).
  R5  with_in: an abbreviation's or an initial's dot does not end the
      sentence ("Dr. Smith enters." / "J. Smith enters.").
  R6  cine_assemble.probe takes the VIDEO stream's length (a clip with 3.5 s
      of audio under 73 frames of video is 3.042 s, driven with ffmpeg).
  R7  (Frank 10.10., field: "keine neuen Spuren nach erfolgreichem Lauf")
      the pinned Core 7a131a3a replays the "executed" of a CACHED timeline
      node at the END of the prompt (execution.py: the cached-ui replay
      right before execution_success) -- after the Save's clips were
      registered; the handler started the run's bookkeeping over, the
      registered clips were gone, no pin, no track. Driven in node with the
      extension's own listeners and that event order: run 1 pins V1, run 2
      (timeline cached, replay after the Save) opens a track; the edit_plan
      root is the realpath register() uses.
  S   ASCII.

Script-style: exit 0 = pass.
"""
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1084_rest_of_table"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)
    return c


pkg = types.ModuleType("plsn1084")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1084"] = pkg
T = importlib.import_module("plsn1084.cine_timeline")
HP = importlib.import_module("plsn1084.h3_prompt")

# ------------------------------------------------------------------ R1
txt, n = HP.strip_rubrics("// TRIGGERS\nohwx,\n// LOOK\nLive-action, cinematic,\n\n[Shot 1] The camera. // a note\nSee https://x.y/z.")
_need(txt == "ohwx,\nLive-action, cinematic,\n\n[Shot 1] The camera.\nSee https://x.y/z." and n == 3,
      "R1 strip_rubrics: rubric lines gone, the trailing note cut, the URL spared, 3 counted: %r" % (txt,))
_need(HP.strip_rubrics("") == ("", 0) and HP.strip_rubrics("plain") == ("plain", 0), "R1 nothing to strip -> unchanged, 0")
kf = (ROOT / "nodes" / "ph_minimax_keyframes.py").read_text(encoding="utf-8")
rf = (ROOT / "nodes" / "ph_minimax_ref.py").read_text(encoding="utf-8")
_need("prompt, n_rub = HP.strip_rubrics(prompt)" in kf and kf.find("HP.strip_rubrics(prompt)") < kf.find("prompt, used, tag_errors = HP.resolve(")
      and "rubric line(s) stripped before the encode" in kf, "R1 Keyframes strips the rubrics before the resolve and says so")
_need("prompt, n_rub = HP.strip_rubrics(prompt)" in rf and rf.find("HP.strip_rubrics(prompt)") < rf.find("prompt, used, tag_errors = HP.resolve(prompt")
      and "rubric line(s) stripped before the encode" in rf, "R1 Reference strips the rubrics before the resolve and says so")

TMP = Path(tempfile.mkdtemp(prefix="v1084_"))
for d in ("output/fx", "input", "temp"):
    (TMP / d).mkdir(parents=True, exist_ok=True)


class _NT:
    is_nested = True


comfy = types.ModuleType("comfy"); nt = types.ModuleType("comfy.nested_tensor"); nt.NestedTensor = _NT; comfy.nested_tensor = nt
capi = types.ModuleType("comfy_api"); ii = types.ModuleType("comfy_api.input_impl"); ii.VideoFromComponents = object
ut = types.ModuleType("comfy_api.util"); ut.VideoComponents = object
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: str(TMP / "output"); fp.get_input_directory = lambda: str(TMP / "input"); fp.get_temp_directory = lambda: str(TMP / "temp")
sys.modules.update({"comfy": comfy, "comfy.nested_tensor": nt, "comfy_api": capi, "comfy_api.input_impl": ii, "comfy_api.util": ut, "folder_paths": fp})
TL = importlib.import_module("plsn1084.ph_cine_timeline")
m = T.load(json.dumps({"seconds": 6.0, "project": "fx", "triggers": "ohwx", "style": "Live-action, cinematic,",
                       "shots": [{"at": 0, "text": "A woman walks."}]}))
r, errs = TL.run(m, {}, None)
_need(not errs and "[Shot 1] // TRIGGERS" in r["prompt"] and not any("// rubric" in c for c in r["check"]),
      "R1 the prompt keeps its head, the check no longer warns about it: %r" % (r["check"],))

# ------------------------------------------------------------------ R2
m = T.load(json.dumps({"seconds": 6.0, "project": "fx", "shots": [{"at": 0, "text": "@hero walks past @first_frame."}]}))
r, errs = TL.run(m, {}, None)
line = [l for l in r["lines"] if "@tag(s)" in l]
_need(not errs and line and "MiniMax Keyframes only @first_frame / @last_frame" in line[0] and "1 other tag(s) here: an error at Keyframes" in line[0],
      "R2 the tag line names what resolves where, the board tag as an error at Keyframes: %r" % (line,))
m = T.load(json.dumps({"seconds": 6.0, "project": "fx", "shots": [{"at": 0, "text": "@first_frame walks."}]}))
r, errs = TL.run(m, {}, None)
line = [l for l in r["lines"] if "@tag(s)" in l]
_need(line and "other tag" not in line[0], "R2 only slot tags: no error named")

# ------------------------------------------------------------------ R3
old = T.load(json.dumps({"seconds": 6.0, "project": "myfilm", "triggers": "ohwx", "target": "ref2va", "carry_mode": "keyframe",
                         "render": "scene", "active": 2, "transitions": [{"kind": "dissolve", "seconds": 0.5}],
                         "shots": [{"at": 0, "text": "a"}, {"at": 3, "scene": True, "cut": "the camera cuts to", "text": "b"}]}))
board = json.dumps({"seconds": 6, "style": "Noir,", "shots": [{"at": 0, "text": "x"}, {"at": 2.5, "scene": True, "cut": "the camera cuts to", "text": "y"}]})
m2, adopted, notes = TL.adopt(old, board, [])
_need(adopted and m2["project"] == "myfilm" and m2["triggers"] == "ohwx" and m2["target"] == "ref2va" and m2["carry_mode"] == "keyframe"
      and m2["render"] == "scene" and m2["active"] == 2 and m2["style"] == "Noir," and len(m2["shots"]) == 2,
      "R3 the storyboard brings shots and style, the project's fields stay: %r" % ({k: m2.get(k) for k in TL.ADOPT_KEEPS},))
# a storyboard answer carries no scene flags -> one scene: the old transitions cannot fit
_need(m2["transitions"] == [] and any("transitions of the old timeline were dropped (2 scenes before, 1 now)" in n for n in notes)
      and any(n.startswith("kept from the timeline: project, triggers, target, carry_mode, render, active") for n in notes),
      "R3 the transitions are dropped with a note (a board has one scene), the note names what was kept: %r" % (notes,))
old1 = T.load(json.dumps({"seconds": 6.0, "project": "one", "shots": [{"at": 0, "text": "a"}]}))
m2b, _a, notes_b = TL.adopt(old1, board, [])
_need(m2b["project"] == "one" and not any("dropped" in n for n in notes_b), "R3 a one-scene timeline without transitions: nothing dropped")
fresh = T.load(json.dumps({"seconds": 6.0, "shots": [{"at": 0, "text": "a"}]}))
m4, _a, notes4 = TL.adopt(fresh, board, [])
_need(m4["project"] == "cine" and not any(n.startswith("kept from") for n in notes4), "R3 a default timeline keeps nothing to say")

# ------------------------------------------------------------------ R4
import io
import contextlib
import math
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    got = TL.ULSCineTimeline.IS_CHANGED("{not json")
_need(isinstance(got, float) and math.isnan(got) and "IS_CHANGED could not read" in buf.getvalue(),
      "R4 an unreadable timeline: NaN (re-run) and a console line: %r" % (buf.getvalue().strip(),))
got2 = TL.ULSCineTimeline.IS_CHANGED(json.dumps({"seconds": 6.0, "project": "fx", "shots": [{"at": 0, "text": "a"}]}))
_need(got2 == "", "R4 a readable timeline without carry or clip anchors: unchanged ('')")

# ------------------------------------------------------------------ R5
p = {"in": "the image resolves", "seconds": 0.5}
_need(T.with_in("Dr. Smith enters the room. He sits.", p) == "Dr. Smith enters the room. The image resolves within half a second. He sits.",
      "R5 'Dr.' does not end the sentence")
_need(T.with_in("J. Smith enters. He sits.", p) == "J. Smith enters. The image resolves within half a second. He sits.", "R5 an initial does not end the sentence")
_need(T.with_in("He enters the room. He sits.", p) == "He enters the room. The image resolves within half a second. He sits.", "R5 an ordinary sentence as before")

# ------------------------------------------------------------------ R6
A = importlib.import_module("plsn1084.cine_assemble")
ff = shutil.which("ffmpeg")
have_av = importlib.util.find_spec("av") is not None
if ff and have_av:
    clip, vid = TMP / "av.mp4", TMP / "v.mp4"
    subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x36:r=24", "-frames:v", "73", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid)], capture_output=True)
    # 73 frames of video (3.042 s) under 3.5 s of audio: the container says 3.5
    subprocess.run([ff, "-v", "error", "-y", "-i", str(vid), "-f", "lavfi", "-i", "sine=f=440:r=48000", "-t", "3.5", "-map", "0:v", "-map", "1:a",
                    "-c:v", "copy", "-c:a", "aac", str(clip)], capture_output=True)
    dur, has_a, w, h, fps = A.probe(str(clip))
    _need(has_a and abs(dur - 73 / 24.0) < 0.02, "R6 probe: 73 frames of video under 3.5 s of audio -> %.3f s (video), audio %s" % (dur, has_a))
else:
    print("  note  ffmpeg or PyAV missing -- R6 driven part skipped")
shutil.rmtree(TMP, ignore_errors=True)

# ------------------------------------------------------------------ R7
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
tmpd = Path(tempfile.mkdtemp(prefix="v1084js_"))
h = js.replace('import { app } from "../../scripts/app.js";', "const app = globalThis.__app;")
h = h.replace('import { api } from "../../scripts/api.js";', "const api = globalThis.__api;")
h = h.replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}; const refit = () => {};")
h = h.replace("function _render(node) {", "function _render(node) { if (globalThis.__noRender) return;", 1)
h = h.replace("async function _stack(node) {", "async function _stack(node) { if (globalThis.__noStack) return;", 1)
h += "\nexport { NODE };\n"
(tmpd / "tl.mjs").write_text(h, encoding="utf-8")
(tmpd / "drive.mjs").write_text(r'''
const listeners = {}; let ext = null; const reg = [];
globalThis.window = { addEventListener() {} };
globalThis.requestAnimationFrame = () => 0; globalThis.__noRender = true; globalThis.__noStack = true;
globalThis.__app = { graph: { _nodes: [] }, registerExtension(e) { ext = e; }, queuePrompt: async () => true };
globalThis.__api = { addEventListener: (n, f) => { (listeners[n] = listeners[n] || []).push(f); }, apiURL: (u) => u,
  fetchApi: async (url, opt) => {
    if (url.startsWith("/pls/cine/register")) { const b = JSON.parse(opt.body); reg.push(b); return { json: async () => ({ rel: b.file.subfolder + "/" + b.file.filename }) }; }
    if (url.startsWith("/pls/cine/probe")) return { json: async () => ({ name: "x.mp4", frames: 48, fps: 24, audio: true }) };
    return { json: async () => ({ error: "stub" }) }; } };
const M = await import("./tl.mjs");
await ext.setup();
const fire = async (n, detail) => { for (const f of listeners[n] || []) await f({ detail }); };
const node = { id: 7, type: "", properties: { [M.EDIT_KEY]: {} },
  widgets: [{ name: "timeline", value: JSON.stringify(Object.assign({}, M.DEFAULT, { project: "fx" })) }],
  _plsTl: { state: { textContent: "" }, player: { t: 0 }, active: "base" } };
node.type = M.NODE;
globalThis.__app.graph._nodes.push(node);
const tl = { pls_timeline: [{ clip_prefix: "fx/whole", scene: 0 }] };
const save = (name) => ({ images: [{ filename: name, subfolder: "2026-10-10", type: "output", format: "video/h264-mp4" }] });
const layers = () => (node.properties[M.EDIT_KEY].layers || []).filter((l) => l.id !== "base").map((l) => l.src && l.src.path);
// run 1: the timeline executes, the Save writes, success
await fire("executed", { prompt_id: "p1", node: "7", output: tl });
await fire("executed", { prompt_id: "p1", node: "9", output: save("a_00001_.mp4") });
await fire("execution_success", { prompt_id: "p1" });
const r1 = { takes: node.properties[M.EDIT_KEY].takes, layers: layers(), state: node._plsTl.state.textContent };
// run 2: the timeline is CACHED -- its "executed" is replayed at the END, after the Save (Core 7a131a3a)
await fire("execution_cached", { prompt_id: "p2", nodes: ["7"] });
await fire("executed", { prompt_id: "p2", node: "9", output: save("a_00002_.mp4") });
await fire("executed", { prompt_id: "p2", node: "7", output: tl });
await fire("execution_success", { prompt_id: "p2" });
const r2 = { takes: node.properties[M.EDIT_KEY].takes, layers: layers(), state: node._plsTl.state.textContent, reg: reg.map((b) => (b.primary ? "P:" : "") + b.file.filename) };
console.log(JSON.stringify({ r1, r2, edit: node.properties[M.EDIT_KEY] }));
process.exit(0);
''', encoding="utf-8")
r = subprocess.run(["node", str(tmpd / "drive.mjs")], capture_output=True, text=True, timeout=120)
shutil.rmtree(tmpd, ignore_errors=True)
okj = r.returncode == 0
if not okj:
    print(r.stderr[-1200:])
o = json.loads(r.stdout.strip().splitlines()[-1]) if okj else {}
_need(okj and o["r1"]["takes"] == {"0": "2026-10-10/a_00001_.mp4"} and o["r1"]["layers"] == [] and "first run of this place" in o["r1"]["state"],
      "R7 run 1 pins V1: %r" % (o.get("r1"),))
_need(okj and o["r2"]["layers"] == ["2026-10-10/a_00002_.mp4"] and o["r2"]["takes"] == {"0": "2026-10-10/a_00001_.mp4"}
      and o["r2"]["state"].startswith("new render -> new track: 2026-10-10/a_00002_.mp4"),
      "R7 run 2 (cached timeline, executed replayed AFTER the Save): a track on top, V1 keeps its pin: %r" % (o.get("r2"),))
_need(okj and o["r2"]["reg"] == ["a_00001_.mp4", "P:a_00001_.mp4", "a_00002_.mp4", "P:a_00002_.mp4"], "R7 both runs registered and marked primary: %r" % (o.get("r2", {}).get("reg"),))
py2 = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
# AMENDED v1152: the stack list moved to input/; register() takes its root from A.OUTPUT_ROOT = output/ -- edit_plan names the same
_need("root = os.path.realpath(_output_dir())   # v1084" in py2 and "A.OUTPUT_ROOT = _output_dir()" in py2,
      "R7 edit_plan's root is the realpath register() uses")

bad = [f for f in ("nodes/h3_prompt.py", "nodes/ph_cine_timeline.py",
                   "nodes/cine_timeline.py", "nodes/cine_assemble.py", "tests/test_v1084_rest_of_table.py")
       if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not bad, "S ASCII only %r" % bad)
if FAILS:
    print("%s: %d FAILURE(S)" % (NAME, len(FAILS)))
    sys.exit(1)
print("%s: PASS (R1-R7, S)" % NAME)
