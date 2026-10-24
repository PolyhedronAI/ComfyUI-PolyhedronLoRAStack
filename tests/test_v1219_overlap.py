# -*- coding: ascii -*-
"""Guard v1219 -- the overlap at the scene seams (Frank 24.10.: "Dadurch, dass wir hier mit ueberlappenden Frames
arbeiten, kann es sein, dass wir trotzdem oefters Unstimmigkeiten bekommen, gerade wenn geschnitten oder verschoben
wird") and the way back ("die Clips wieder korrekt an die Schneidegrenze zuruecksetzen").

  O1  the plan: an EDITED scene cut list never shows a motion context's repeated head -- a trim / slip into it is
      held at the head (named), `free` (Alt) passes; a blend set later takes no head frames (lands on the prompt's
      placement); a piece cut on another take moves by the change of the head; a list made for another scene count
      falls back to the prompt; the prompt's pieces carry their take + head; the plan hands the guard out; the
      note names a repeated head as such.
  O2  the compiler / the run: an anchor ON a boundary marked `both` is the last frame of the scene before AND the
      first of the scene after (unmarked: as before); an own first anchor that wins over the carry warns unless
      `both`; carry_ends (only in a queued run) moves the carried frame, on another take it is ignored and named;
      two anchors on one frame of a scene stop the run by name; `both` off a boundary is named.
  O3  MiniMax Continue warns when it pins fewer tail frames than the timeline lengthened the scene by.
  O4  snap (pure): three clips of a track in disorder go back to their scenes in order; a clip with no place stays;
      cut_at_frame / insert_clip equal the editor's cutAtFrame / insertClip (node.js); scene_of reads the names.
  O5  snap on real clips (the neon film, re-encoded): V1 + a track whose scene 2 / 3 clips lie swapped and shifted
      -> each at its scene's start, scene 2's own repeated head (22) measured on the track's file and left out.
  O6  the editor (node.js): anchorRoleIn == the backend's scene split for a sweep; a boundary anchor follows its
      cut, `both` goes when the anchor leaves the boundary; guardHeads holds / frees; carryEnds; carryInject gives
      the ONE run its cut end only (edit source, the needed scene), never for a prompt cut or a whole render.
  S   ASCII; the snap route; the snap buttons (toolbar, track heads); the cut end never written into the widget.

Script-style: exit 0 = pass.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


import importlib  # noqa: E402

TD = tempfile.mkdtemp(prefix="v1219_")
IN_D, OUT_D = os.path.join(TD, "input"), os.path.join(TD, "output")
os.makedirs(IN_D)
os.makedirs(OUT_D)
fp = types.ModuleType("folder_paths")
fp.get_temp_directory = lambda: os.path.join(TD, "temp")
fp.get_output_directory = lambda: OUT_D
fp.get_input_directory = lambda: IN_D
sys.modules["folder_paths"] = fp
try:
    import torch  # noqa: F401
    nh = types.ModuleType("node_helpers")
    nh.conditioning_set_values = lambda cond, values: [[c[0], dict(c[1], **values)] for c in cond]
    sys.modules.setdefault("node_helpers", nh)
    cu = types.ModuleType("comfy.utils")
    cm = types.ModuleType("comfy")
    cm.utils = cu
    sys.modules.setdefault("comfy", cm)
    sys.modules.setdefault("comfy.utils", cu)
except Exception:
    pass
PKG = "pls1219"
pkg = types.ModuleType(PKG)
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules[PKG] = pkg
T = importlib.import_module(PKG + ".cine_timeline")
E = importlib.import_module(PKG + ".cine_edit")
TL = importlib.import_module(PKG + ".ph_cine_timeline")
JS = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
CO_SRC = (ROOT / "nodes" / "ph_minimax_continue.py").read_text(encoding="utf-8")

# ------------------------------------------------------------------ O1 the plan
ST, CL, CUT = [4.0], [{"frames": 101}, {"frames": 124}], [{"kind": "cut"}]
HELD = [(0, 0, 101), (22, 0, 124, True)]
TK = {1: "d/s1.mp4", 2: "d/s2.mp4"}
p0 = E.plan(ST, 8.0, CL, CUT, T.TRANSITIONS, held=HELD, takes=TK)
_need([(x["k"], x["a"], x["b"]) for x in p0["pieces"]] == [(1, 0, 96), (2, 22, 118)]
      and p0["pieces"][1]["t"] == "d/s2.mp4" and p0["pieces"][1]["h0"] == 22,
      "O1 the prompt's pieces: scene 2 after its repeated head, stamped with take + head %s" % p0["pieces"])
_need(p0["guard"]["2"] == {"head": 22, "repeat": True, "frozen_head": 0, "tail": 0} and not p0["guard"]["1"]["repeat"],
      "O1 the plan hands the guard out %s" % p0["guard"])
_need(p0["held"] == ["scene 2: the first 22 left out (the repeated head -- the end of scene 1 again)"],
      "O1 the note names a repeated head as such %s" % p0["held"])
ED = {"pieces": {"scenes": [{"k": 1, "a": 0, "b": 96, "speed": 1.0}, {"k": 2, "a": 2, "b": 118, "speed": 1.0}]}}
p1 = E.plan(ST, 8.0, CL, CUT, T.TRANSITIONS, edit=ED, held=HELD, takes=TK)
_need(p1["source"] == "edit" and p1["pieces"][1]["a"] == 22 and any("repeated head" in w and "Alt" in w for w in p1["warnings"]),
      "O1 a trim 20 frames into the repeated head is held at it, named %s" % p1["warnings"][:1])
EDF = json.loads(json.dumps(ED))
EDF["pieces"]["scenes"][1]["free"] = True
p1f = E.plan(ST, 8.0, CL, CUT, T.TRANSITIONS, edit=EDF, held=HELD, takes=TK)
_need(p1f["pieces"][1]["a"] == 2 and not p1f["warnings"], "O1 a piece cut past the head on purpose (free, Alt) passes")
EDB = {"pieces": {"scenes": [{"k": 1, "a": 0, "b": 96, "speed": 1.0},
                             {"k": 2, "a": 22, "b": 118, "speed": 1.0, "tr": {"kind": "dissolve", "seconds": 0.5}}]}}
p2 = E.plan(ST, 8.0, CL, CUT, T.TRANSITIONS, edit=EDB, held=HELD, takes=TK)
pd = E.plan(ST, 8.0, CL, [{"kind": "dissolve", "seconds": 0.5}], T.TRANSITIONS, held=HELD, takes=TK)
_need(p2["segments"][1]["src_in"] == 22 and [(x["a"], x["b"]) for x in p2["pieces"]] == [(x["a"], x["b"]) for x in pd["pieces"]],
      "O1 a dissolve set on the edited seam takes no head frame -- the piece lands where the prompt's blend puts it %s"
      % [(x["a"], x["b"]) for x in p2["pieces"]])
EDT = {"pieces": {"scenes": [{"k": 1, "a": 0, "b": 96, "speed": 1.0, "t": "d/s1.mp4", "h0": 0},
                             {"k": 2, "a": 22, "b": 118, "speed": 1.0, "t": "d/s2_old.mp4", "h0": 22}]}}
p3 = E.plan(ST, 8.0, [{"frames": 101}, {"frames": 107}], CUT, T.TRANSITIONS, edit=EDT,
            held=[(0, 0, 101), (5, 0, 107, True)], takes=TK)
_need((p3["pieces"][1]["a"], p3["pieces"][1]["b"]) == (5, 90) and p3["pieces"][1]["t"] == "d/s2.mp4"
      and any("a new take" in w and "-17" in w for w in p3["warnings"]),
      "O1 a new take with a head of 5: the cut list moves by -17 to it, named %s" % p3["warnings"][:1])
p4 = E.plan(ST, 8.0, CL, CUT, T.TRANSITIONS, edit=dict(ED, pieces_n=3), held=HELD, takes=TK)
_need(p4["source"] == "prompt" and any("made for 3 scene(s)" in w for w in p4["warnings"]),
      "O1 a cut list made for another scene count: the prompt's pieces, named")
_need(E.clean_edit({"pieces_n": 2, "pieces": ED["pieces"]}).get("pieces_n") == 2
      and E._clean_piece({"k": 1, "a": 0, "b": 4, "t": "x", "h0": 3, "free": 1}) == {"k": 1, "a": 0, "b": 4, "speed": 1.0, "t": "x", "h0": 3, "free": True},
      "O1 the edit keeps pieces_n, a piece its take, head and free mark")
pw = E.plan([], 6.0, [{"frames": 141}], [], T.TRANSITIONS, whole=True)
_need(pw["guard"] == {} and "t" not in pw["pieces"][0], "O1 a whole render: no guard, nothing stamped (as before)")

# ------------------------------------------------------------------ O2 compiler / run
BASE = {"seconds": 8.0, "shots": [{"at": 0, "text": "A stands"}, {"at": 4.0, "text": "B sits", "scene": True, "carry": True}],
        "anchors": [{"src": "image_1", "at": 0.0}, {"src": "image_2", "at": 4.0}], "render": "scene", "active": 1}


def roles(m, k):
    sm = T.scene_model(m, k)
    f, l, g = T.split_anchors(sm, T.frames_for(sm["seconds"]))
    return (f and f["src"], l and l["src"], [x[0] for x in g])


m0 = T.load(json.dumps(BASE))
_need(roles(m0, 1) == ("image_1", None, []) and roles(m0, 2) == ("image_2", None, []),
      "O2 unmarked boundary anchor: the first frame of scene 2 only (as before)")
mb = T.load(json.dumps(BASE))
mb["anchors"][1]["both"] = True
_need(roles(mb, 1) == ("image_1", "image_2", []) and roles(mb, 2) == ("image_2", None, []),
      "O2 marked 'both': the last frame of scene 1 AND the first of scene 2 -- the clips meet on one picture")
_need("both" not in T.scene_model(mb, 1)["anchors"][1], "O2 the scene's own anchors carry no 'both' key")
PREV = "/x/PH_1_cine_scene_1.mp4"
for both, want in ((False, "WARN: carry: scene 2 starts on image_2, but scene 1 ends free"),
                   (True, "carry: scene 1 ends on the same picture (image_2 marked 'both scenes')")):
    mm = T.load(json.dumps(dict(BASE, active=2)))
    mm["anchors"][1]["both"] = both
    r, e = TL.run(mm, {"image_1", "image_2"}, prev=PREV)
    _need(r is not None and any(l.startswith(want) for l in r["lines"]),
          "O2 an own first anchor wins over the carry -- %s" % ("named as the join" if both else "WARN: scene 1 ends free"))
mc = T.load(json.dumps(dict(BASE, active=2, anchors=[], carry_ends={"1": {"f": 80, "t": "2026-10-24/PH_1_cine_scene_1.mp4"}})))
_need(mc.get("carry_ends") == {"1": {"f": 80, "t": "2026-10-24/PH_1_cine_scene_1.mp4"}} and "carry_ends" not in T.DEFAULT,
      "O2 load() keeps carry_ends of a queued run; it is not part of the saved timeline (DEFAULT)")
r, e = TL.run(mc, set(), prev=PREV)
_need(r["carry_index"] == 80 and r["carry_cut"] and any("where your cut ends it" in l for l in r["lines"]),
      "O2 carry_ends: scene 2 begins where the CUT ends scene 1 (frame 81)")
r, e = TL.run(mc, set(), prev="/x/PH_2_cine_scene_1.mp4")
_need(r["carry_index"] == T.film_last_index(mc, 1) and any("made on PH_1_cine_scene_1.mp4" in l for l in r["lines"]),
      "O2 carry_ends cut on another take: the prompt's end, named")
md = T.load(json.dumps(dict(BASE, active=1, anchors=[{"src": "image_1", "at": 95 / 24.0}, {"src": "image_2", "at": 4.0, "both": True}])))
_need(not T.validate(md, {"image_1", "image_2"}), "O2 (frames 95 and 96 of the whole timeline: no clash there)")
r, e = TL.run(md, {"image_1", "image_2"}, prev=None)
_need(r is None and e and "land on the same frame" in e[0] and "of scene 1" in e[0],
      "O2 a 'both' anchor and scene 1's own last anchor on ONE frame of scene 1 stop the run by name %s" % e[:1])
mo = T.load(json.dumps(dict(BASE, active=1, anchors=[{"src": "image_1", "at": 2.0, "both": True}])))
r, e = TL.run(mo, {"image_1"}, prev=None)
_need(any(l.startswith("note: anchor image_1 is marked 'both scenes' but sits at 00:02.000") for l in r["lines"]),
      "O2 'both' off a boundary is named")

# ------------------------------------------------------------------ O3 Continue
_need("if int(tail.shape[0]) > n:" in CO_SRC and "K2 pins %d of the %d tail frames" in CO_SRC,
      "O3 MiniMax Continue warns when it pins fewer tail frames than the timeline planned")

# ------------------------------------------------------------------ O4 snap, pure
PLACES = {1: {"at": 0}, 2: {"at": 96}, 3: {"at": 192}}
mess = [{"k": 1, "a": 22, "b": 118, "speed": 1.0}, E.gap_piece(10), {"k": 0, "a": 0, "b": 96, "speed": 1.0},
        {"k": 2, "a": 22, "b": 110, "speed": 1.0}, {"k": 3, "a": 0, "b": 30, "speed": 1.0}]
files = [{"place": 1, "a": 0, "b": 96}, {"place": 2, "a": 22, "b": 118}, {"place": 3, "a": 22, "b": 110}, None]
sp, done = E.snap_layer(mess, files, PLACES)
st, acc = [], 0
for p in sp:
    st.append(acc)
    acc += E._length(p)
lay = [(p["k"], p["a"], p["b"], s) for p, s in zip(sp, st) if not E.is_gap(p)]
_need(lay[:3] == [(0, 0, 96, 0), (1, 22, 118, 96), (2, 22, 110, 192)] and [d[1] for d in done] == [1, 2, 3],
      "O4 three clips in disorder: back to scenes 1, 2, 3 at 0 / 4 / 8 s with their own windows %s" % lay[:3])
_need([k for k, _a, _b, _s in lay].count(0) == 1 and [k for k, _a, _b, _s in lay].count(1) == 1
      and [k for k, _a, _b, _s in lay].count(2) == 1,
      "O4 a snapped clip leaves nothing behind at its old place (each file once) %s" % lay)
_need((3, 0, 30) in [(k, a, b) for k, a, b, _s in lay] and lay[-1][3] == 290,
      "O4 a clip with no place stays where it lay (film frame 290)")
_need(E.scene_of("2026-10-24/PH_093901_cine_scene_3_00002_.mp4") == 3 and E.scene_of("x/PH_1_cine_whole_00001_.mp4") is None
      and E.scene_of("x/PH_1_cine_whole_00001_.mp4", whole=True) == 0 and E.scene_of("dropped.mp4") is None
      and E.scene_of("stack/scene_12_00001_.webm") == 12,
      "O4 scene_of reads ..._scene_<k>_ / _whole_ from the file name")
CASES = [([{"k": 0, "a": 0, "b": 50, "speed": 1.0}], {"k": 1, "a": 5, "b": 25, "speed": 1.0}, 10),
         ([{"k": 0, "a": 0, "b": 50, "speed": 1.0}, E.gap_piece(7), {"k": 1, "a": 3, "b": 40, "speed": 0.5}], {"k": 2, "a": 0, "b": 33, "speed": 1.0}, 40),
         ([], {"k": 0, "a": 0, "b": 12, "speed": 1.0}, 30),
         ([{"k": 0, "a": 0, "b": 20, "speed": 2.0}], {"k": 1, "a": 0, "b": 9, "speed": 1.0}, 3)]
py_ins = [E.insert_clip(q, p, at) for q, p, at in CASES]
py_cut = [E.cut_at_frame(q, at) for q, _p, at in CASES]

# ------------------------------------------------------------------ O5 snap on real clips
SRC = Path(os.environ.get("PLS_NEON") or "/nonexistent")   # public v396: the neon clips are not part of the repo
ff = shutil.which("ffmpeg")
if ff and (SRC / "scene_1_00001_.webm").is_file():
    try:
        s1, s2, s3 = (str(SRC / ("scene_%d_00001_.webm" % k)) for k in (1, 2, 3))
        three = {"seconds": 15.0, "render": "scene", "active": 2, "carry_mode": "motion", "head_frames": 22, "project": "o5",
                 "shots": [{"at": 0.0, "text": "A dancer spins under neon light."},
                           {"at": 5.0, "text": "She keeps spinning.", "scene": True, "carry": True},
                           {"at": 10.0, "text": "Out.", "scene": True}]}
        m5 = T.load(json.dumps(three))
        stack = TL.project_folder("o5")
        os.makedirs(stack, exist_ok=True)
        e1 = T.film_last_index(m5, 1)
        shutil.copy(s1, os.path.join(stack, "scene_1_00001_.webm"))
        c2 = os.path.join(stack, "scene_2_00001_.mp4")
        subprocess.run([ff, "-v", "error", "-y", "-i", s1, "-i", s2, "-filter_complex",
                        "[0:v]fps=24,trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS[a];[1:v]fps=24,setpts=PTS-STARTPTS[b];"
                        "[a][b]concat=n=2:v=1:a=0[v]" % (e1 - 21, e1 + 1), "-map", "[v]", "-c:v", "libx264",
                        "-crf", "23", "-pix_fmt", "yuv420p", c2], check=True, timeout=600)
        shutil.copy(s3, os.path.join(stack, "scene_3_00001_.webm"))
        day = os.path.join(OUT_D, "2026-10-24")
        os.makedirs(day)
        t2, t3 = os.path.join(day, "PH_120000_o5_scene_2_00002_.mp4"), os.path.join(day, "PH_120000_o5_scene_3_00002_.webm")
        # the track's scene 2: the same continued clip re-encoded once more (another take, the same head)
        subprocess.run([ff, "-v", "error", "-y", "-i", c2, "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", t2],
                       check=True, timeout=600)
        shutil.copy(s3, t3)
        n2, n3 = int(TL.clip_info(t2)["frames"]), int(TL.clip_info(t3)["frames"])
        rel2, rel3 = "2026-10-24/" + os.path.basename(t2), "2026-10-24/" + os.path.basename(t3)
        # disorder: scene 3's clip first at 1 s, scene 2's after it, both cut
        trk = [E.gap_piece(24), {"k": 1, "a": 0, "b": 60, "speed": 1.0}, {"k": 0, "a": 5, "b": 70, "speed": 1.0}]
        edit = {"layers": [{"id": "t1", "name": "render 12:00", "src": {"kind": "render", "path": rel2, "at": 24},
                            "more": [{"kind": "render", "path": rel3, "at": 0}], "pieces": trk},
                           {"id": "base", "src": {"kind": "base"}}],
                "pieces": {"scenes": [{"k": 1, "a": 0, "b": 50, "speed": 1.0}, {"k": 3, "a": 0, "b": 40, "speed": 1.0}]}}
        d = TL.snap_edit(m5, edit, "all")
        plain = TL.edit_plan(m5, {}, strips=False)["plan"]
        at = {}
        a0 = 0
        for pc in plain["pieces"]:
            at.setdefault(pc["k"], (a0, pc))
            a0 += E._length(pc)
        pt = d["layers"].get("t1", {}).get("pieces") or []
        pos, a0 = {}, 0
        for pc in pt:
            if not E.is_gap(pc):
                pos[pc["k"]] = (a0, pc["a"], pc["b"])
            a0 += E._length(pc)
        _need(d["base"] and plain["pieces"][1]["a"] == 22,
              "O5 V1 back to the prompt's cut (its scene 2 after the measured repeated head: a = %d)" % plain["pieces"][1]["a"])
        _need(pos.get(0, (None,))[0] == at[2][0] and pos[0][1] == 22 and pos[0][2] - pos[0][1] == E._length(at[2][1]),
              "O5 the track's scene 2 clip at scene 2's start (%s s), its OWN repeated head (22) left out, as long as V1's: %s"
              % (at[2][0] / 24.0, pos.get(0)))
        _need(pos.get(1, (None,))[0] == at[3][0] and pos[1][1] == at[3][1]["a"] and pos[1][2] <= n3,
              "O5 the track's scene 3 clip at scene 3's start (%s s), its frozen first frame(s) left out at the cut like "
              "V1's (a = %d): %s" % (at[3][0] / 24.0, at[3][1]["a"], pos.get(1)))
        _need(d["layers"]["t1"]["src"]["at"] == at[2][0] and d["layers"]["t1"]["more"][0]["at"] == at[3][0],
              "O5 the files' 'laid at' follow (a later track reset lands on the scenes too)")
        lk = json.loads(json.dumps(edit))
        lk["layers"][0]["lock"] = True
        dl = TL.snap_edit(m5, lk, "t1")
        _need(not dl["layers"] and not dl["base"] and any("locked" in x for x in dl["lines"]), "O5 a locked track is left alone")
    except Exception as e:
        _need(False, "O5 %s: %s" % (type(e).__name__, e))
else:
    # public v396: the real-clip probe needs the neon test clips, which are not in the repo. Skipped (and said so)
    # when PLS_NEON does not point at them -- O4 / O6 above check the same snap on synthetic clips.
    print("  skip  O5 snap on real clips: set PLS_NEON to the neon clips (scene_1/2/3_00001_.webm) to run it")

# ------------------------------------------------------------------ O6 the editor (node.js)
if shutil.which("node") is None:
    _need(False, "O6 needs node.js")
else:
    tmp = Path(tempfile.mkdtemp(prefix="v1219js_"))
    body = (JS.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (tmp / "m.mjs").write_text(body, encoding="utf-8")
    TLM = {"seconds": 15.0, "render": "scene", "active": 1,
           "shots": [{"at": 0, "text": "a"}, {"at": 5.0, "text": "b", "scene": True, "carry": True},
                     {"at": 7.5, "text": "b2"}, {"at": 10.0, "text": "c", "scene": True}]}
    sweep = [0.0, 1.0, 4.0, 4.95, 4.9583, 5.0, 6.0, 9.9583, 10.0, 12.0, 14.9]
    (tmp / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const out = {};
const M = TLM;
out.roles = SWEEP.map((at) => { const r = T.anchorRoleIn(M, { at, both: true }); return [r.role, r.scene]; });
out.rolesPlain = SWEEP.map((at) => { const r = T.anchorRoleIn(M, { at }); return [r.role, r.scene]; });
out.bf = [...T.boundaryFrames(M)].sort((a, b) => a - b);
const m2 = T.parseTimeline(JSON.stringify(M)); m2.anchors = [{ src: "x", at: 5.0, both: true }, { src: "y", at: 2.0 }];
T.moveCut(m2, 1, 5.5); out.follow = m2.anchors.map((a) => a.at);
T.moveAnchor(m2, 0, 6.0); out.bothGone = !("both" in m2.anchors[0]);
const G = { "2": { head: 22, repeat: true } };
out.gh = T.guardHeads([{ k: 1, a: 0, b: 96, speed: 1 }, { k: 2, a: 2, b: 118, speed: 1 }], G, false);
out.ghAlt = T.guardHeads([{ k: 2, a: 2, b: 118, speed: 1 }], G, true);
out.ends = T.carryEnds(T.parseTimeline(JSON.stringify(M)), [{ k: 1, a: 0, b: 100, speed: 1 }, { k: 2, a: 22, b: 140, speed: 1 }, { k: 3, a: 0, b: 50, speed: 1 }], (k) => "d/s" + k + ".mp4");
const plan = { plan: { source: "edit", pieces: [{ k: 1, a: 0, b: 100, speed: 1 }, { k: 2, a: 22, b: 140, speed: 1 }, { k: 3, a: 0, b: 50, speed: 1 }] },
               clips: [{ k: 1, path: "/p1", takes: ["d/s1.mp4"], take: 0 }, { k: 2, path: "/p2", takes: ["d/s2.mp4"], take: 0 }, { k: 3, path: "/p3", takes: ["d/s3.mp4"], take: 0 }] };
const txt = (o) => JSON.stringify(Object.assign({}, M, o));
out.inj2 = JSON.parse(T.carryInject(txt({ active: 2 }), plan) || "null");
out.inj3 = T.carryInject(txt({ active: 3 }), plan);
out.inj3clip = JSON.parse(T.carryInject(txt({ active: 3, anchors: [{ src: "@clip:2:last", at: 10.0 }] }), plan) || "null");
out.injPrompt = T.carryInject(txt({ active: 2 }), { plan: Object.assign({}, plan.plan, { source: "prompt" }), clips: plan.clips });
out.injWhole = T.carryInject(txt({ active: 2, render: "whole" }), plan);
out.ins = INS.map(([q, p, at]) => T.insertClip(q, p, at));
out.cut = INS.map(([q, _p, at]) => T.cutAtFrame(q, at));
console.log(JSON.stringify(out));
""".replace("TLM", json.dumps(TLM)).replace("SWEEP", json.dumps(sweep)).replace("INS", json.dumps(CASES)), encoding="utf-8")
    r = subprocess.run(["node", str(tmp / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(tmp, ignore_errors=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        _need(False, "O6 node.js run failed: %s" % (r.stderr[-600:],))
    else:
        o = json.loads(line[-1])
        mT = T.load(json.dumps(TLM))

        def py_role(at, both):
            m = json.loads(json.dumps(mT))
            m["anchors"] = [{"src": "q", "at": at, "both": both}]
            found = []
            for k in range(1, len(T.scenes(m)) + 1):
                sm = T.scene_model(m, k)
                f, l, g = T.split_anchors(sm, T.frames_for(sm["seconds"]))
                if f:
                    found.append(("first", k))
                if l:
                    found.append(("last", k))
                if g:
                    found.append(("guide", k))
            if len(found) == 2 and found[0] == ("last", found[1][1] - 1) and found[1][0] == "first":
                return ["both", found[1][1]]
            return list(found[0]) if found else None

        want_b = [py_role(at, True) for at in sweep]
        want_p = [py_role(at, False) for at in sweep]
        _need(o["roles"] == want_b and o["rolesPlain"] == want_p,
              "O6 anchorRoleIn == the backend's scene split (%d times, with and without 'both')" % len(sweep)
              + ("" if o["roles"] == want_b and o["rolesPlain"] == want_p else " JS %s / %s PY %s / %s" % (o["roles"], o["rolesPlain"], want_b, want_p)))
        _need(o["bf"] == sorted(T.boundary_frames(mT)), "O6 boundaryFrames == boundary_frames %s" % o["bf"])
        _need(o["follow"] == [5.5, 2.0] and o["bothGone"], "O6 a boundary anchor follows its cut; 'both' goes off the boundary")
        _need(o["gh"]["held"] == 1 and o["gh"]["pieces"][1]["a"] == 22 and o["gh"]["pieces"][0]["a"] == 0
              and o["ghAlt"]["pieces"][0]["a"] == 2 and o["ghAlt"]["pieces"][0]["free"] is True,
              "O6 guardHeads: held at the repeated head; with Alt kept and marked free")
        _need(o["ends"] == {"1": {"f": 99, "t": "d/s1.mp4"}}, "O6 carryEnds: only where the next scene carries %s" % o["ends"])
        _need(o["inj2"] and o["inj2"]["carry_ends"] == {"1": {"f": 99, "t": "d/s1.mp4"}} and o["inj3"] is None
              and o["inj3clip"] and o["inj3clip"]["carry_ends"] == {"2": {"f": 139, "t": "d/s2.mp4"}}
              and o["injPrompt"] is None and o["injWhole"] is None,
              "O6 carryInject: scene 2's run gets scene 1's cut end, scene 3 (no carry) nothing, @clip:2:last scene 2's; "
              "a prompt cut or a whole render nothing")
        _need(o["ins"] == py_ins and o["cut"] == py_cut, "O4 insert_clip / cut_at_frame == the editor's insertClip / cutAtFrame (%d cases)" % len(CASES)
              + ("" if o["ins"] == py_ins else " JS %s PY %s" % (o["ins"], py_ins)))

# ------------------------------------------------------------------ S
TLPY = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
_need('("POST", "/pls/cine/snap", handle_snap)' in TLPY and '"/pls/cine/snap"' in JS, "S the snap route: registered, called by the editor")
_need('snapBtn.dataset.pls = "snapall";' in JS and "resetBtn, snapBtn, cutsBtn" in JS and '}, "snap"),' in JS,
      "S snap to scenes in the toolbar and in every track head")
ci = JS[JS.index("export function carryInject"):JS.index("export function carryInject") + 2400]
sv = JS[JS.index("function _savePieces(node, pieces)"):JS.index("function _clearCuts(node, e)")]
_need("_save(" not in ci and "carry_ends" not in sv.replace("NOT written", "") and "nd.inputs.timeline = t2" in JS,
      "S the cut end goes into the queued prompt only -- never into the timeline widget (a fine cut re-renders nothing)")
for f in ("nodes/cine_edit.py", "nodes/cine_timeline.py", "nodes/ph_cine_timeline.py", "nodes/ph_minimax_continue.py",
          "web/js/ph_cine_timeline.js", "tests/test_v1219_overlap.py"):
    raw = (ROOT / f).read_bytes()
    _need(all(b < 128 for b in raw), "S  ASCII %s" % f)

shutil.rmtree(TD, ignore_errors=True)
if FAILS:
    print("test_v1219_overlap: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1219_overlap: all checks passed")
