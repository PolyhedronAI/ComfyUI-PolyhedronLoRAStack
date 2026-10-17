# -*- coding: ascii -*-
"""Guard v1043 -- the LAYERS of the cutting room, stage E1: the model and the
export (nodes/cine_edit.py, nodes/ph_cine_timeline.py, the pure mirrors in
web/js/ph_cine_timeline.js).

Frank, 30.09.: "layers you open yourself, or every new render in a NEW
layer; show, hide, lock, delete, move up and down; upper layers win when
playing; where parts of a clip were cut out, the lower clips shine through
-- the sound too. Cut areas can be deleted and are empty then; the others
can be moved left and right or stretched."

  G1  emptiness is a piece: gap_piece / is_gap; norm_layer (gaps re-based,
      merged, none at the end); a gap cleans to a = GAP_A, speed 1; place_clip.
  G2  delete_piece (a gap of the same length) and move_piece (the gap before
      gives what the gap after takes; clamped at the next piece; a gap made at
      the head; the last piece grows the layer).
  G3  track() with gaps: a transition next to a gap is a cut, no split edit
      on a gap seam, gap segments carry no material and are not "missing";
      an old edit plans exactly as before (plan == track + the prompt's keys).
  G4  clean_layers: TOP FIRST, exactly one base, a file layer needs a path
      and pieces, a stack of only the default base is no stack.
  G5  flatten(): the topmost layer switched on with material wins, picture
      and sound on their own; a layer without sound lets the sound below
      through; nothing = None (black / silence); vcover/acover.
  G6  the browser's mirrors give the same numbers (node.js): normLayer,
      placeClip, deletePiece, movePiece, flattenRuns.
  G7  the server: stack_of() returns None for a plain film (same plan, same
      export as v1042); a stack with a render layer (a gap in it), a file
      from input/ (30 fps, square) and a file read in place -> the film;
      layer_path refuses a render layer outside output/ and a non-video.
  G8  the export, REAL ffmpeg over the server's path: 144 frames; the colour
      of each run where flatten() says (base / blue / green / black); the
      sound per interval (440 Hz base, 880 Hz the green layer, silence in
      the hole); the 30 fps clip lands on the right source frame.
  S   statics: ASCII, the plain film still goes through ffmpeg_command, the
      stack through stack_command, get_running_loop.

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


E = _load("ce1043", "nodes/cine_edit.py")
T = _load("ct1043", "nodes/cine_timeline.py")
K = T.TRANSITIONS
G = E.GAP


def shape(ps):
    return [("gap", E._length(p)) if E.is_gap(p) else (p["k"], p["a"], p["b"]) for p in ps]


# ------------------------------------------------------------------ G1
g = E.gap_piece(12)
_need(E.is_gap(g) and g["a"] == E.GAP_A and E._length(g) == 12 and g["speed"] == 1.0, "G1 gap_piece: 12 frames of nothing at a = GAP_A")
n = E.norm_layer([E.gap_piece(5), E.gap_piece(7), {"k": 0, "a": 0, "b": 20, "speed": 1}, E.gap_piece(3), E.gap_piece(4)])
_need(shape(n) == [("gap", 12), (0, 0, 20)], "G1 norm_layer: two gaps merge, the gap at the end goes %r" % shape(n))
c = E._clean_piece({"k": G, "a": 7, "b": 31, "speed": 2.0})
_need(c == {"k": G, "a": E.GAP_A, "b": E.GAP_A + 12, "speed": 1.0}, "G1 a gap cleans to a = GAP_A, speed 1, its film length kept %r" % c)
_need(shape(E.place_clip(72, 36)) == [("gap", 36), (0, 0, 72)] and shape(E.place_clip(50, 0)) == [(0, 0, 50)],
      "G1 place_clip: the whole clip from its frame, a gap before it")

# ------------------------------------------------------------------ G2
P3 = [{"k": 0, "a": 0, "b": 24, "speed": 1}, {"k": 0, "a": 24, "b": 48, "speed": 1}, {"k": 0, "a": 48, "b": 72, "speed": 1}]
d1 = E.delete_piece(P3, 1)
_need(shape(d1) == [(0, 0, 24), ("gap", 24), (0, 48, 72)], "G2 delete: the middle piece becomes 24 frames of nothing")
_need(shape(E.delete_piece(P3, 2)) == [(0, 0, 24), (0, 24, 48)], "G2 delete the last piece: the layer ends earlier")
_need(shape(E.delete_piece(d1, 1)) == shape(d1), "G2 deleting a gap changes nothing")
L = [E.gap_piece(10), {"k": 0, "a": 0, "b": 20, "speed": 1}, E.gap_piece(6), {"k": 0, "a": 30, "b": 40, "speed": 1}]
_need(shape(E.move_piece(L, 1, -4)) == [("gap", 6), (0, 0, 20), ("gap", 10), (0, 30, 40)], "G2 move left: the gap before gives, the gap after takes")
_need(shape(E.move_piece(L, 1, -40)) == [(0, 0, 20), ("gap", 16), (0, 30, 40)], "G2 move left past the head: clamped at frame 0")
_need(shape(E.move_piece(L, 1, 40)) == [("gap", 16), (0, 0, 20), (0, 30, 40)], "G2 move right: clamped at the next piece")
_need(shape(E.move_piece(L, 3, 25)) == [("gap", 10), (0, 0, 20), ("gap", 31), (0, 30, 40)], "G2 the last piece moves on: the layer grows")
_need(shape(E.move_piece(P3, 0, 5)) == shape(P3), "G2 the first piece against its neighbour cannot move right")
_need(shape(E.move_piece(P3, 1, 5)) == shape(P3), "G2 a piece between two pieces cannot move")
_need(shape(E.move_piece(P3, 2, -5)) == shape(P3), "G2 no gap before it: it cannot move left")
_need(shape(E.move_piece(P3, 2, 6)) == [(0, 0, 24), (0, 24, 48), ("gap", 6), (0, 48, 72)], "G2 the last piece moves right: a gap opens before it")

# ------------------------------------------------------------------ G3
w = []
tr = E.track([{"k": 0, "a": 0, "b": 48, "speed": 1}, dict(E.gap_piece(24), tr={"kind": "dissolve", "seconds": 0.5}),
              {"k": 0, "a": 72, "b": 120, "speed": 1, "tr": {"kind": "dissolve", "seconds": 0.5}, "ao": 8}], {0: 144}, K, w)
_need([s["kind"] for s in tr["seams"]] == ["cut", "cut"] and all(s["d"] == 0 for s in tr["seams"]),
      "G3 a transition next to a gap is a cut (nothing to blend with)")
_need(all(s.get("ao", 0) == 0 for s in tr["seams"]), "G3 no split edit on a gap seam")
gs = tr["segments"][1]
_need(gs.get("gap") and not gs["missing"] and (gs["film_in"], gs["film_out"]) == (48, 72) and tr["asegments"][1].get("gap"),
      "G3 the gap segment: film 48-72, no material, not missing, silent")
_need(tr["complete"] and tr["length"] == 120, "G3 a layer with a gap is complete, 120 frames")
_need(E.track([], {0: 10}, K, [])["length"] == 0, "G3 an empty layer has no length (no crash)")
CL = [{"frames": 56}, {"frames": 56}, {"frames": 73}]
p_old = E.plan([2.0, 4.0], 6.0, CL, [{"kind": "dissolve", "seconds": 0.5}, {"kind": "cut"}], K)
w2 = []
t_same = E.track(p_old["pieces"], {1: 56, 2: 56, 3: 73}, K, w2)
_need(all(t_same[k] == p_old[k] for k in ("length", "segments", "asegments", "seams", "cuts", "place_cuts")),
      "G3 plan() = track() + the prompt's keys: an old edit plans exactly as before")
_need([(x["k"], x["a"], x["b"]) for x in p_old["pieces"]] == [(1, 0, 48), (2, 6, 54), (3, 0, 48)], "G3 the v1037 prompt pieces unchanged")

# ------------------------------------------------------------------ G4
_need(E.clean_layers(None) == [] and E.clean_layers([{"id": "base", "src": {"kind": "base"}}]) == [],
      "G4 no stack / only the default base = no stack (the plain film)")
cl = E.clean_layers([{"id": "a1", "name": "fox", "src": {"kind": "file", "path": "/x/fox.mp4"}, "pieces": [E.gap_piece(5), {"k": 0, "a": 0, "b": 9}]},
                     {"id": "zz", "src": {"kind": "file"}, "pieces": [{"k": 0, "a": 0, "b": 9}]},
                     {"id": "a1", "src": {"kind": "file", "path": "/y.mp4"}, "pieces": [{"k": 0, "a": 0, "b": 9}]},
                     {"id": "bad", "src": {"kind": "weird", "path": "/y.mp4"}, "pieces": [{"k": 0, "a": 0, "b": 9}]}])
_need([x["id"] for x in cl] == ["a1", "base"] and cl[0]["video"] and cl[0]["audio"] and not cl[0]["lock"]
      and shape(cl[0]["pieces"]) == [("gap", 5), (0, 0, 9)], "G4 top first; no path / a double id / an unknown kind dropped; base added at the bottom")
cl2 = E.clean_layers([{"id": "base", "src": {"kind": "base"}, "audio": False}])
_need(len(cl2) == 1 and cl2[0]["audio"] is False, "G4 a base alone with its sound off IS a stack (the switch counts)")
ce = E.clean_edit({"layers": cl, "span": 5.5, "pieces": {"whole": [{"k": 0, "a": 0, "b": 20}, {"k": G, "a": 0, "b": 12}]}})
_need(ce["span"] == 5.5 and len(ce["layers"]) == 2 and ce["pieces"]["whole"][1]["k"] == G, "G4 clean_edit keeps span, layers and gap pieces")

# ------------------------------------------------------------------ G5
base_p = E.norm_layer([{"k": 0, "a": 0, "b": 120, "speed": 1}, E.gap_piece(12), {"k": 0, "a": 132, "b": 144, "speed": 1}])
A = E.delete_piece([E.gap_piece(36), {"k": 0, "a": 0, "b": 24, "speed": 1}, {"k": 0, "a": 24, "b": 48, "speed": 1},
                    {"k": 0, "a": 48, "b": 72, "speed": 1}], 2)
B = E.norm_layer([E.gap_piece(24), {"k": 0, "a": 0, "b": 24, "speed": 1}, E.gap_piece(48), {"k": 0, "a": 24, "b": 48, "speed": 1}])
tb, ta, tbb = E.track(base_p, {0: 144}, K, []), E.track(A, {0: 72}, K, []), E.track(B, {0: 72}, K, [])
ENT = [dict(id="B", video=True, audio=True, has_audio=False, track=tbb), dict(id="A", video=True, audio=True, has_audio=True, track=ta),
       dict(id="base", video=True, audio=True, has_audio=True, track=tb)]
fl = E.flatten(ENT, 144)
V = [(r["film_in"], r["film_out"], r["layer"]) for r in fl["vruns"]]
AU = [(r["film_in"], r["film_out"], r["layer"]) for r in fl["aruns"]]
_need(V == [(0, 24, 2), (24, 48, 0), (48, 60, 1), (60, 84, 2), (84, 96, 1), (96, 120, 0), (120, 132, None), (132, 144, 2)],
      "G5 picture: the top wins, the deleted part shows the base, the base's hole is black %r" % V)
_need(AU == [(0, 36, 2), (36, 60, 1), (60, 84, 2), (84, 108, 1), (108, 120, 2), (120, 132, None), (132, 144, 2)],
      "G5 sound: the blue layer has none -- the sound below comes through %r" % AU)
off = [dict(x) for x in ENT]
off[1]["video"] = False
V2 = [(r["film_in"], r["film_out"], r["layer"]) for r in E.flatten(off, 144)["vruns"]]
AU2 = [(r["film_in"], r["film_out"], r["layer"]) for r in E.flatten(off, 144)["aruns"]]
_need(V2[2] == (48, 96, 2) and AU2 == AU, "G5 the green layer's picture off: the base shows, its sound still plays")
off[1]["video"], off[2]["audio"] = True, False
AU3 = [(r["film_in"], r["film_out"], r["layer"]) for r in E.flatten(off, 144)["aruns"]]
_need(AU3 == [(0, 36, None), (36, 60, 1), (60, 84, None), (84, 108, 1), (108, 144, None)], "G5 the base's sound off: silence under it")
_need(fl["vcover"][1] == [(36, 60), (84, 108)], "G5 vcover: a layer's own material (the export's overlay)")

# ------------------------------------------------------------------ G6
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
if shutil.which("node") is None:
    print("  note  node.js not found -- G6 SKIPPED")
else:
    t2 = Path(tempfile.mkdtemp(prefix="v1043js_"))
    body = (js.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (t2 / "m.mjs").write_text(body, encoding="utf-8")
    CASES = {"P3": P3, "L": L, "base": base_p, "A0": [E.gap_piece(36), {"k": 0, "a": 0, "b": 24, "speed": 1}, {"k": 0, "a": 24, "b": 48, "speed": 1},
                                                    {"k": 0, "a": 48, "b": 72, "speed": 1}], "B": B,
             "messy": [E.gap_piece(5), E.gap_piece(7), {"k": 0, "a": 0, "b": 20, "speed": 1}, E.gap_piece(3)]}
    OPS = [("norm", "messy", 0, 0), ("del", "P3", 1, 0), ("del", "P3", 2, 0), ("del", "A0", 2, 0),
           ("mv", "L", 1, -4), ("mv", "L", 1, -40), ("mv", "L", 1, 40), ("mv", "L", 3, 25), ("mv", "P3", 1, 5),
           ("mv", "P3", 2, 6), ("mv", "P3", 0, 5), ("mv", "P3", 2, -5), ("place", None, 72, 36), ("place", None, 50, 0)]
    (t2 / "cases.json").write_text(json.dumps({"cases": CASES, "ops": OPS,
                                               "stack": [{"video": x["video"], "audio": x["audio"], "has_audio": x["has_audio"],
                                                          "track": {"segments": x["track"]["segments"], "asegments": x["track"]["asegments"]}}
                                                         for x in ENT]}), encoding="utf-8")
    (t2 / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
import { readFileSync } from "fs";
const D = JSON.parse(readFileSync(new URL("./cases.json", import.meta.url)));
const out = [];
for (const [op, c, j, d] of D.ops) {
    const P = c ? D.cases[c] : null;
    if (op === "norm") out.push(T.normLayer(P));
    else if (op === "del") out.push(T.deletePiece(P, j));
    else if (op === "mv") out.push(T.movePiece(P, j, d));
    else out.push(T.placeClip(j, d));
}
const fl = T.flattenRuns(D.stack, 144);
console.log(JSON.stringify({ ops: out, fl: { vruns: fl.vruns, aruns: fl.aruns, vcover: fl.vcover, acover: fl.acover } }));
""", encoding="utf-8")
    r = subprocess.run(["node", str(t2 / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(t2, ignore_errors=True)
    line = [x for x in r.stdout.splitlines() if x.startswith("{")]
    if r.returncode != 0 or not line:
        _fail("G6 harness failed: %s" % r.stderr.strip()[-400:])
    else:
        got = json.loads(line[-1])
        want = []
        for op, c, j, d in OPS:
            P = CASES.get(c) if c else None
            want.append(E.norm_layer(P) if op == "norm" else E.delete_piece(P, j) if op == "del"
                        else E.move_piece(P, j, d) if op == "mv" else E.place_clip(j, d))
        norm = lambda ps: [[int(p["k"]), int(p["a"]), int(p["b"]), float(p["speed"])] for p in ps]
        bad = [i for i, (a, b) in enumerate(zip(got["ops"], want)) if norm(a) != norm(b)]
        _need(not bad and len(got["ops"]) == len(want), "G6 the browser's layer ops = Python's (%d ops)%s" % (len(want), "" if not bad else " -- differ at %r" % bad))
        jf = json.loads(json.dumps({"vruns": fl["vruns"], "aruns": fl["aruns"], "vcover": fl["vcover"], "acover": fl["acover"]}))
        _need(got["fl"] == jf, "G6 flattenRuns = flatten (picture, sound, covers)")

# ------------------------------------------------------------------ G7 / G8 the server
ff = shutil.which("ffmpeg")
if ff is None:
    print("  note  ffmpeg not found -- G7/G8 SKIPPED")
else:
    TMP = Path(tempfile.mkdtemp(prefix="v1043_"))
    for d in ("output/cine", "input/cine_media/cine", "temp", "elsewhere"):
        (TMP / d).mkdir(parents=True, exist_ok=True)

    class _NT:
        is_nested = True

    comfy = types.ModuleType("comfy")
    nt = types.ModuleType("comfy.nested_tensor")
    nt.NestedTensor = _NT
    comfy.nested_tensor = nt
    capi = types.ModuleType("comfy_api")
    ii = types.ModuleType("comfy_api.input_impl")
    ii.VideoFromComponents = object
    ut = types.ModuleType("comfy_api.util")
    ut.VideoComponents = object
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: str(TMP / "output")
    fp.get_input_directory = lambda: str(TMP / "input")
    fp.get_temp_directory = lambda: str(TMP / "temp")
    fp.base_path = str(TMP)
    sys.modules.update({"comfy": comfy, "comfy.nested_tensor": nt, "comfy_api": capi,
                        "comfy_api.input_impl": ii, "comfy_api.util": ut, "folder_paths": fp})
    pkg = types.ModuleType("plsn1043")
    pkg.__path__ = [str(ROOT / "nodes")]
    sys.modules["plsn1043"] = pkg
    TL = importlib.import_module("plsn1043.ph_cine_timeline")
    EE = sys.modules["plsn1043.cine_edit"]

    def mk(args, out):
        r = subprocess.run([ff, "-v", "error", "-y"] + args + [str(out)], capture_output=True, text=True)
        return r.returncode == 0

    ok = (mk(["-f", "lavfi", "-i", "testsrc2=s=320x180:r=24", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
              "-frames:v", "144", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"], TMP / "output/cine/whole_00001_.mp4")
          and mk(["-f", "lavfi", "-i", "color=c=0x00c000:s=200x200:r=30", "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000",
                  "-frames:v", "90", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac"], TMP / "input/cine_media/cine/green.mp4")
          and mk(["-f", "lavfi", "-i", "color=c=0x0000e0:s=320x180:r=24", "-frames:v", "72", "-c:v", "libx264", "-pix_fmt", "yuv420p"],
                 TMP / "elsewhere/blue.mp4"))
    _need(ok, "G7 test clips made (base 24 fps + 440 Hz, green 30 fps square + 880 Hz, blue without sound)")
    model = T.load(json.dumps({"seconds": 6.0, "project": "cine", "render": "whole",
                               "shots": [{"at": 0, "cut": "the camera cuts to", "text": "x"}]}))
    # G7 a plain film: no stack, the v1042 plan and export path
    d0 = TL.edit_plan(model, {}, strips=False)
    _need("film" not in d0 and d0["plan"]["length"] == 144, "G7 a plain film: no stack -- the plan as before (144 frames)")
    stack = [{"id": "L2", "name": "blue", "src": {"kind": "file", "path": str(TMP / "elsewhere/blue.mp4")}, "pieces": B},
             {"id": "L1", "name": "green", "src": {"kind": "file", "path": "cine_media/cine/green.mp4"}, "pieces": A},
             {"id": "base", "src": {"kind": "base"}}]
    ed = {"pieces": {"whole": base_p}, "layers": stack}
    d1 = TL.edit_plan(model, ed, strips=False)
    film = d1.get("film") or {}
    _need(film.get("length") == 144 and [r["id"] for r in film.get("layers", [])] == ["L2", "L1", "base"] and film.get("complete"),
          "G7 the stack: three layers top first, the film 144 frames, complete")
    gl = (film.get("layers") or [{}, {}])[1].get("clip") or {}
    _need(gl.get("frames") == 72 and gl.get("src_frames") == 90 and abs(gl.get("fps", 0) - 30) < 0.01,
          "G7 the 30 fps clip counts 72 film frames (90 source frames at 30 fps)")
    FV = [(r["film_in"], r["film_out"], r["layer"]) for r in film.get("flat", {}).get("vruns", [])]
    _need(FV == V, "G7 the server's stack = flatten() of G5")
    refused = []
    for src in ({"kind": "render", "path": "../elsewhere/blue.mp4"}, {"kind": "file", "path": str(TMP / "temp")},
                {"kind": "file", "path": str(TMP / "output/cine/nope.mp4")}):
        try:
            TL.layer_path(src)
            refused.append(False)
        except ValueError:
            refused.append(True)
    (TMP / "elsewhere/notes.txt").write_text("x")
    try:
        TL.layer_path({"kind": "file", "path": str(TMP / "elsewhere/notes.txt")})
        refused.append(False)
    except ValueError:
        refused.append(True)
    _need(all(refused), "G7 layer_path refuses: a render layer outside output/, a folder, a missing file, a non-video %r" % refused)
    d_only = TL.edit_plan(model, {"span": 5.0}, strips=False)
    _need((d_only.get("film") or {}).get("length") == 120 and [r["id"] for r in d_only["film"]["layers"]] == ["base"],
          "G7 an edit with only a span is a stack of one: the film is the span (120 frames)")
    d_span = TL.edit_plan(model, dict(ed, span=5.0), strips=False)
    _need(d_span["film"]["length"] == 120 and any("past the edit span" in w for w in d_span["film"]["warnings"]) is False,
          "G7 span 5 s: the film is 120 frames (the base's own length does not count as past it)")
    # G8 the export
    try:
        out, lines = TL.export_film(model, ed, preview=False)
    except Exception as e:
        out, lines = None, ["%s" % e]
    _need(out and Path(out).is_file() and "3 layer(s)" in lines[0], "G8 export over the server's path: %s" % (lines[0] if lines else "?"))
    if out and Path(out).is_file():
        nfr = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries", "stream=nb_read_frames",
                              "-of", "csv=p=0", out], capture_output=True, text=True).stdout.strip()
        _need(nfr == "144", "G8 144 frames exported (%s)" % nfr)
        raw = subprocess.run([ff, "-v", "error", "-i", out, "-vf", "scale=32:18", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True).stdout
        fsz = 32 * 18 * 3

        def px(f, x, y):
            o = f * fsz + (y * 32 + x) * 3
            return tuple(raw[o:o + 3])

        def kind(f):
            # the whole middle row: the layers are flat colours, the base
            # (testsrc2) is not -- one pixel can sit on a blue bar of it
            row = [px(f, x, 9) for x in range(32)]
            mid = row[10:22]
            if all(max(c) < 30 for c in row):
                return "black"
            if all(c[2] > 150 and c[0] < 60 and c[1] < 60 for c in row):
                return "blue"
            if all(c[1] > 120 and c[0] < 60 and c[2] < 60 for c in mid):
                return "green" + ("|bars" if max(row[1]) < 30 and max(row[30]) < 30 else "")
            return "base"
        seen = [kind(f) for f in (12, 36, 54, 72, 90, 108, 126, 138)]
        _need(seen == ["base", "blue", "green|bars", "base", "green|bars", "blue", "black", "base"],
              "G8 the picture where flatten() says it (the square clip in black bars) %r" % seen)
        import numpy as np
        pcm = np.frombuffer(subprocess.run([ff, "-v", "error", "-i", out, "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
                                           capture_output=True).stdout, np.float32)

        def tone(t0, t1):
            # which of the two tones sound here (a mix would show both)
            x = pcm[int(t0 * 48000):int(t1 * 48000)]
            if not len(x) or float(np.sqrt((x ** 2).mean())) < 1e-3:
                return 0
            fq = np.fft.rfftfreq(len(x), 1 / 48000.0)
            sp = np.abs(np.fft.rfft(x))
            at = lambda hz: float(sp[np.abs(fq - hz) < 6].max())
            e4, e8 = at(440), at(880)
            return "+".join(str(h) for h, e in ((440, e4), (880, e8)) if e > 0.05 * max(e4, e8))
        tones = [tone(a, b) for a, b in ((0.2, 1.3), (1.6, 2.4), (2.6, 3.4), (3.6, 4.4), (4.6, 4.9), (5.1, 5.4), (5.6, 5.9))]
        _need(tones == ["440", "880", "440", "880", "440", 0, "440"],
              "G8 the sound: base 440 / green 880 -- never both, the blue lets it through, silence in the hole %r" % tones)
    shutil.rmtree(TMP, ignore_errors=True)

# ------------------------------------------------------------------ S
tl = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
ce_src = (ROOT / "nodes" / "cine_edit.py").read_text(encoding="utf-8")
_need("if d.get(\"film\") is not None:\n        return _export_stack(model, d, preview)" in tl and "cmd = E.ffmpeg_command(exe, paths, out, p, audio" in tl
      and "cmd = E.stack_command(exe, stack, out, size, audio" in tl, "S a plain film still goes through ffmpeg_command, a stack through stack_command")
_need("get_event_loop" not in tl, "S get_running_loop only (v577 rule)")
bad = [f for f in ("nodes/cine_edit.py", "nodes/ph_cine_timeline.py", "web/js/ph_cine_timeline.js", "tests/test_v1043_layers.py")
       if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not bad, "S ASCII only %r" % bad)
_need("GAP = -1" in ce_src and "export const GAP = -1, GAP_A = 100000;" in js and "GAP_A = 100000" in ce_src, "S GAP / GAP_A the same on both sides")

if FAILS:
    print("test_v1043_layers: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1043_layers: PASS (G1-G8, S)")
