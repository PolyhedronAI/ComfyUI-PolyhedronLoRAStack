# -*- coding: ascii -*-
"""Guard v1036 -- the video track under the prompt track (nodes/cine_edit.py,
the routes /pls/cine/edit + /pls/cine/export, the clip lane and the player in
web/js/ph_cine_timeline.js).

Frank, 29.09.: prompt transitions go INTO the run, video transitions onto
finished clips -- put both so they line up; play them; export; fine-tune the
lengths. A run is rounded up to 17k+5 frames, so a clip is longer than its
place: the plan puts every clip on its place and uses the overhang for the
blend, which sits CENTRED on its cut.

  E1  the plan of three scene clips: places, the centred blend (h1/h2), the
      film keeps the asked length, a held picture where the material ends.
  E2  the edit: the film's cuts (not the prompt's), MIN_SEG between cuts,
      the end, keep_tail, slip (clamped), a cut list of the wrong length is
      named and ignored; clean_edit drops junk.
  E3  a whole run: one clip, the asked length, held when it is too short.
  E4  the ffmpeg graph makes exactly the plan's frames for every pair of
      kinds (pure count), and ffmpeg itself agrees (real run, if found).
  E5  scene_takes: every take, newest first, the stack list included.
  J   the node's pure parts: mixAt (one clip / two in a blend, progress),
      srcTime (held at the last frame), dragCut (frames, room), cutsIntoPrompt,
      blendOps -- fade to black/white carry ffmpeg's own two-phase weights
      (checked against the formula measured on flat colours), wipe/slide
      edges; editOf returns a copy; axisOf.
  S   statics: the edit lives in node.properties (never the timeline widget),
      the handle's choice is H3 only, the chips sit on the clip lane, the
      routes are registered, the player is a canvas in the node, no floating
      window, BASE_H re-measured.

Script-style: exit 0 = pass.
"""
import importlib.util
import itertools
import json
import re
import os
import shutil
import subprocess
import sys
import tempfile
import time
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


E = _load("ce1036", "nodes/cine_edit.py")
A = _load("ca1036", "nodes/cine_assemble.py")
T = _load("ct1036", "nodes/cine_timeline.py")
K = T.TRANSITIONS
CL = [{"frames": 56, "audio": True}, {"frames": 56, "audio": True}, {"frames": 73, "audio": True}]
TR = [{"kind": "dissolve", "seconds": 0.5}, {"kind": "fade to black", "seconds": 0.5}]

# ------------------------------------------------------------------ E1
p = E.plan([2.0, 4.0], 6.0, CL, TR, K)
sg, sm = p["segments"], p["seams"]
_need(p["length"] == 144 and p["complete"], "E1 the film keeps the asked length (144 frames = 6.00 s)")
_need([(s["film_in"], s["film_out"]) for s in sg] == [(0, 54), (42, 102), (90, 144)],
      "E1 places: each blend centred on its cut (48, 96), 6 frames each side %r" % [(s["film_in"], s["film_out"]) for s in sg])
_need([(x["d"], x["h1"], x["h2"], x["at"], x["xfade"]) for x in sm] == [(12, 6, 6, 48, "fade"), (12, 6, 6, 96, "fadeblack")],
      "E1 seams: 0.5 s = 12 frames, h1/h2, at the prompt's cut, the ffmpeg name")
_need(sg[1]["hold"] == 4 and sg[1]["src_out"] == 56 and any("clip 2" in w and "held" in w and "under the blend" in w for w in p["warnings"]),
      "E1 clip 2 has 56 frames for 60 on the film: its last picture is held 4 frames, named")
# v1037 RE-GROUNDED: max_d is measured against the pieces as they sit (the
# clip after the first seam has exactly 6 frames before its cut -> 12)
_need(sm[0]["max_d"] == 12 and sm[1]["max_d"] == 4, "E1 max_d: how long a blend could be without a held picture %r" % [x["max_d"] for x in sm])
cut = E.plan([2.0, 4.0], 6.0, CL, [{"kind": "cut"}, {"kind": "cut"}], K)
_need([(s["film_in"], s["film_out"]) for s in cut["segments"]] == [(0, 48), (48, 96), (96, 144)] and not cut["warnings"],
      "E1 hard cuts: the clips sit exactly under their scenes, the overhang is cut away")
miss = E.plan([2.0, 4.0], 6.0, [CL[0], None, CL[2]], TR, K)
_need(not miss["complete"] and miss["segments"][1]["missing"] and any("scene 2 is not rendered" in w for w in miss["warnings"]),
      "E1 a scene without a clip: the plan says so, the film is not complete")

# ------------------------------------------------------------------ E2
e = E.plan([2.0, 4.0], 6.0, CL, [{"kind": "cut"}, {"kind": "cut"}], K, edit={"cuts": [1.5, 4.5]})
_need(e["cuts"] == [36, 108] and e["cuts_prompt"] == [48, 96] and e["segments"][1]["film_in"] == 36,
      "E2 the edit's cuts move the film's cuts; the prompt's stay as they were")
e = E.plan([2.0, 4.0], 6.0, CL, [{"kind": "cut"}, {"kind": "cut"}], K, edit={"cuts": [3.0, 1.0]})
_need(e["cuts"] == [72, 74], "E2 cuts keep MIN_SEG frames of room (3.0, 1.0 -> 72, 74) %r" % e["cuts"])
e = E.plan([2.0, 4.0], 6.0, CL, [{"kind": "cut"}, {"kind": "cut"}], K, edit={"cuts": [1.5]})
_need(e["cuts"] == [48, 96] and any("do not match" in w for w in e["warnings"]), "E2 a cut list of the wrong length is named and not used")
e = E.plan([2.0, 4.0], 6.0, CL, [{"kind": "cut"}, {"kind": "cut"}], K, edit={"end": 5.5})
_need(e["length"] == 132, "E2 the film's end (5.5 s -> 132 frames)")
e = E.plan([2.0, 4.0], 6.0, CL, [{"kind": "cut"}, {"kind": "cut"}], K, edit={"keep_tail": True})
_need(e["length"] == 96 + 73 and e["segments"][2]["hold"] == 0, "E2 keep_tail: the last clip runs out to its own end (169 frames)")
e = E.plan([2.0, 4.0], 6.0, CL, [{"kind": "cut"}, {"kind": "cut"}], K, edit={"slip": {"3": 0.5, "2": 99}})
_need(e["segments"][2]["src_in"] == 12 and e["segments"][1]["src_in"] == 54 and e["segments"][1]["hold"] > 0,
      "E2 slip: clip 3 starts 12 frames in; a slip past the material is clamped (and held)")
_need(E.clean_edit({"cuts": ["x"], "end": -1, "slip": {"a": 1, "2": -3, "3": 0.25}, "takes": {"1": 5, "2": "p/q.mp4"}, "zzz": 1})
      == {"slip": {"3": 0.25}, "takes": {"2": "p/q.mp4"}}, "E2 clean_edit keeps only well-typed keys")

# ------------------------------------------------------------------ E3
w = E.plan([], 6.0, [{"frames": 158}], [], K, whole=True)
_need(len(w["segments"]) == 1 and w["segments"][0]["k"] == 0 and w["length"] == 144 and w["segments"][0]["src_out"] == 144,
      "E3 a whole run: one clip, the asked 6.00 s of its 158 frames")
w = E.plan([], 6.0, [{"frames": 100}], [], K, whole=True)
_need(w["segments"][0]["hold"] == 44 and any("whole clip" in x for x in w["warnings"]), "E3 a whole clip too short for the length: held, named")

# ------------------------------------------------------------------ E4
bad = []
names = list(T.TRANSITION_NAMES)
for a, b in itertools.product(names, names):
    for ed in (None, {"cuts": [1.5, 4.5], "keep_tail": True, "slip": {"2": 0.2}}):
        q = E.plan([2.0, 4.0], 6.0, CL, [{"kind": a, "seconds": 0.5}, {"kind": b, "seconds": 1.0}], K, edit=ed)
        if E.filter_graph(q, True)[3] != q["length"]:
            bad.append((a, b, ed))
_need(not bad, "E4 the graph makes the plan's frames for all %d kind pairs x 2 edits %r" % (len(names) ** 2, bad[:2]))
exe = shutil.which("ffmpeg")
if exe is None:
    print("  note  ffmpeg not found -- E4 real run SKIPPED")
else:
    tmp = Path(tempfile.mkdtemp(prefix="v1036_"))
    paths = []
    for k, n, col in ((1, 56, "red"), (2, 56, "green"), (3, 73, "blue")):
        f = tmp / ("scene_%d_00001_.mp4" % k)
        subprocess.run([exe, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=%s:s=64x36:r=24" % col, "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
                        "-frames:v", str(n), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(f)], capture_output=True)
        paths.append(str(f))
    res = []
    for tr in ([{"kind": "cut"}, {"kind": "wipe left", "seconds": 0.5}], TR):
        q = E.plan([2.0, 4.0], 6.0, CL, tr, K)
        out = str(tmp / "o.mp4")
        r = subprocess.run(E.ffmpeg_command(exe, paths, out, q, True, preview=True), capture_output=True, text=True)
        n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", out],
                           capture_output=True, text=True).stdout.strip() if shutil.which("ffprobe") else str(q["length"])
        res.append((r.returncode, n))
    shutil.rmtree(tmp, ignore_errors=True)
    _need(res == [(0, "144"), (0, "144")], "E4 ffmpeg itself: cut + wipe and dissolve + fade to black (with a held picture) -> 144 frames %r" % res)

# ------------------------------------------------------------------ E5
tmp = Path(tempfile.mkdtemp(prefix="v1036t_"))
out = tmp / "output"
fox = out / "fox"
fox.mkdir(parents=True)
(out / "else").mkdir()
for name in ("scene_2_00001_.mp4", "scene_2_00002_.mp4", "scene_1_00001_.mp4"):
    (fox / name).write_bytes(b"x")
    time.sleep(0.02)
(out / "else" / "late.mp4").write_bytes(b"y")
A.register(str(fox), 2, str(out / "else" / "late.mp4"))
tk = A.scene_takes(str(fox), 3)
_need([os.path.basename(x) for x in tk.get(2, [])] == ["late.mp4", "scene_2_00002_.mp4", "scene_2_00001_.mp4"] and len(tk.get(1, [])) == 1,
      "E5 scene_takes: every take of a scene, newest first, the stack list included")
shutil.rmtree(tmp, ignore_errors=True)

# ------------------------------------------------------------------ J
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
if shutil.which("node") is None:
    print("  note  node.js not found -- J SKIPPED")
else:
    t2 = Path(tempfile.mkdtemp(prefix="v1036js_"))
    body = (js.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (t2 / "m.mjs").write_text(body, encoding="utf-8")
    (t2 / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const plan = JSON.parse(process.argv[2]);
const D = { plan };
const o = {};
o.mix = [1.0, 1.75, 2.0, 2.2, 3.0].map((t) => T.mixAt(D, t));
o.src = [T.srcTime(plan.segments[1], 1.75), T.srcTime(plan.segments[1], 4.2)];
// v1037: dragCut became trimSeam on the pieces (guarded in test_v1037)
const m = T.parseTimeline(JSON.stringify({ seconds: 6, shots: [{ at: 0, text: "a" }, { at: 2, text: "b", scene: true }, { at: 4, text: "c", scene: true }] }));
o.into = T.cutsIntoPrompt(m, { plan: Object.assign({}, plan, { place_cuts: [36, 108] }) }).shots.map((s) => s.at);
o.ops = {};
for (const k of ["fade", "fadeblack", "fadewhite", "wipeleft", "wiperight", "slideleft"]) o.ops[k] = [0.1, 0.25, 0.5, 0.9].map((p) => T.blendOps(k, p, 100));
const node = { properties: { pls_cine_edit: { cuts: [1] } } };
const c = T.editOf(node); c.cuts.push(2);
o.copy = node.properties.pls_cine_edit.cuts.length;
o.axis = [T.axisOf(m, null), T.axisOf(m, { plan: { length: 200 } })];
o.ss = [T.smoothstep(0.2, 1, 0.1), T.smoothstep(0.2, 1, 0.6), T.smoothstep(0.2, 1, 2)];
console.log(JSON.stringify(o));
""", encoding="utf-8")
    r = subprocess.run(["node", str(t2 / "h.mjs"), json.dumps(p)], capture_output=True, text=True, timeout=60)
    shutil.rmtree(t2, ignore_errors=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        _fail("J harness failed: %s" % r.stderr.strip()[-400:])
    else:
        d = json.loads(line[-1])
        mx = d["mix"]
        _need([len(x["parts"]) for x in mx] == [1, 2, 2, 2, 1] and mx[0]["parts"][0]["i"] == 0 and mx[4]["parts"][0]["i"] == 1,
              "J mixAt: one clip outside a blend, two inside (1.75-2.25 s)")
        _need(abs(mx[1]["p"] - 0.0) < 1e-9 and abs(mx[2]["p"] - 0.5) < 1e-9 and mx[2]["kind"] == "fade"
              and abs(mx[2]["parts"][0]["w"] - 0.5) < 1e-9, "J mixAt: progress 0 at the blend's start, 0.5 on the cut")
        _need(abs(d["src"][0] - 0.5 / 24) < 1e-9 and abs(d["src"][1] - (55.5 / 24)) < 1e-9,
              "J srcTime: the first frame at the clip's film start; held at its last frame past its material")
        _need(d["into"] == [0, 1.5, 4.5], "J cutsIntoPrompt: the scene starts move to the film's cuts %r" % d["into"])

        def ff(kind, p):
            P = 1 - p
            ss = lambda e0, e1, x: (lambda t: t * t * (3 - 2 * t))(min(1, max(0, (x - e0) / (e1 - e0))))
            wa, wb = P * ss(0.8, 1, P), (1 - P) * (1 - ss(0.2, 1, P))
            return wa, wb
        okf = True
        for kind in ("fadeblack", "fadewhite"):
            for p0, ops in zip([0.1, 0.25, 0.5, 0.9], d["ops"][kind]):
                wa, wb = ff(kind, p0)
                got = {o[1]: o[2] for o in ops}
                okf &= abs(got["a"] - wa) < 1e-9 and abs(got["b"] - wb) < 1e-9
                if kind == "fadewhite":
                    okf &= abs(got["#fff"] - (1 - wa - wb)) < 1e-9
        _need(okf, "J fade to black/white: ffmpeg's two-phase weights (measured on flat colours, +-1 of 255)")
        _need(d["ops"]["fade"][2] == [["a", 0, 1], ["b", 0, 0.5]] and d["ops"]["wipeleft"][1] == [["a", 0, 1], ["bclip", 75, 100]]
              and d["ops"]["wiperight"][1] == [["a", 0, 1], ["bclip", 0, 25]] and d["ops"]["slideleft"][1] == [["a", -25, 1], ["b", 75, 1]],
              "J dissolve / wipes / slide: the edges ffmpeg draws (measured: B enters from the right on wipeleft)")
        _need(d["copy"] == 1, "J editOf hands out a copy (the node's edit is only written by _saveEdit)")
        _need(abs(d["axis"][0] - 158 / 24) < 1e-9 and abs(d["axis"][1] - 200 / 24) < 1e-9, "J axisOf: the grid, or the film when longer")
        _need(abs(d["ss"][0]) < 1e-9 and abs(d["ss"][1] - 0.5) < 1e-9 and abs(d["ss"][2] - 1) < 1e-9, "J smoothstep")

# ------------------------------------------------------------------ S
node_py = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
_need('("POST", "/pls/cine/edit", handle_edit)' in node_py and '("POST", "/pls/cine/export", handle_export)' in node_py
      and '"/pls/cine/edit"' in js and '"/pls/cine/export"' in js, "S the routes are registered and called")
_need('node.properties[EDIT_KEY] = e;' in js and 'export const EDIT_KEY = "pls_cine_edit";' in js
      and "function _saveEdit(node, e) {" in js and "_save(node" not in js[js.index("function _saveEdit(node, e) {"):js.index("function _clipLane(")],
      "S the cutting-room edit lives in node.properties -- a fine cut never touches the timeline widget (no re-render)")
_need('grp("inside the clip -- H3 paints it (one run)", [H3_CUT].concat(H3T_NAMES));' in js and 'grp("between two clips' not in js,
      "S the handle offers only what goes into the run")
# v1039 RE-GROUNDED: the film's cuts and its end moved onto the cut layer (one
# line through picture AND sound, Frank 29.09.); the chips stay on the clip lane
_need('chip.dataset.pls = "seam";' in js and 'part(0, picH, at, col, "cutgrip")' in js and 'part(0, picH, end, "#b0bec5", "endgrip")' in js,
      "S the transition chips, the film's cuts and its end sit on the clip lane")
_need('_el("canvas", "height:" + MON_H + "px;' in js and "requestAnimationFrame(tick)" in js and "cancelAnimationFrame(pl.raf)" in js,
      "S the player is a canvas inside the node, its loop stops on pause and on removal")
_need("window.open" not in js and "position:fixed" not in js, "S no floating window")
# v1037 RE-GROUNDED: + the audio lane (611); the v1036 measurement note stays
# AMENDED v1150: BASE_H is gone -- every section has its own px height and fieldH adds them (no measured rest)
_need("export function sections(node, m)" in js and "export function fieldH(node, m)" in js and "BASE_H =" not in js,
      "S the field's height is the sum of its sections (no measured BASE_H)")
_need('"-frames:v", str(p["length"])' in (ROOT / "nodes" / "cine_edit.py").read_text(encoding="utf-8"),
      "S the export is pinned to the plan's frame count")

print()
if FAILS:
    print("test_v1036_video_track: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1036_video_track: PASS (E1-E5, J, S)")
