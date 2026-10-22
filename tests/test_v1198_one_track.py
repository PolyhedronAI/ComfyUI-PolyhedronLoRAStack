# -*- coding: ascii -*-
"""Guard v1198 -- the timeline: "new render -> one track per render" and a clip dragged onto another track.

(Frank 21.10., the neon timeline, five scenes rendered at once: "er legt mir jetzt jedes Mal eine neue Ebene an, das
KANN gewollt sein, aber ich haette gerne noch die Einstellung, dass die anderen Teile dann auch auf dem EINEN neuen
Track landen, damit sie alle zusammenbleiben" -- "kann drinbleiben ... einfach als neue Option"; and "dass man einen
Clip eines Tracks in einen anderen Track reinziehen kann ... zurzeit kann man ihn nur auf dem selben Track verschieben")

  C1 clean_edit keeps auto_mode "render" (only while the switch is on) and render_tracks (strings, at most 32);
     clean_layers keeps a track's MORE files (kind file/render, path, at; at most 64) and src.at; a piece naming a
     file the track does not have becomes an empty piece of the same length
  J1 (node.js) AUTO_CHOICES = track / render / replace; autoMode: missing -> track, auto_mode render -> render,
     auto_track false -> replace (whatever auto_mode says); wipeEdit keeps the render choice
  J2 insertClip: onto an empty layer at 30 = gap 30 + the clip; past the end = a gap up to it; ONTO material it
     overwrites exactly its span (the pieces left and right keep their frames and their place); into a gap of the
     same size = the gap filled; cutAtFrame splits a gap and a piece at the frame
  J3 addSource: an empty track takes the file as its first (k 0); a file it has is reused (same k); a new one is
     appended (k = count); dropOnto lays the piece with that k; resetMulti lays every file whole where it was laid
  J4 the sweep scenario: three scenes of ONE render laid onto one track one after another -> one track, three files,
     each at its scene's frame; the track's pieces k 0, 1, 2 in film order
  J5 srcOfClip: an output/ clip -> render + its path below output/; another -> file + its path; clipK reads row.clips
     (and row.clip for a one-file track)
  S1 the server, REAL files: a track of two files (24 fps + 30 fps) -> the film complete; the picture runs where the
     pieces say; the row carries clips + paths_k; a missing second file names itself and leaves the first playing
  S2 the export over the server's path: 144 frames, the colour of each run from the RIGHT file (red / green / black)
  W  wiring: the render branch of _autoTrack1 (the render's track by its sweep stamp, render_tracks, a new track named
     "render hh:mm:ss" otherwise); the sweep stamped at QUEUE time (api.queuePrompt wrapped, prompt_id -> stamp) and
     used for the register call and the auto track; both drags (a track piece: move / Ctrl = copy; V1's piece: a copy)
     go through _laneUnder + _moveAcross; the layer drawing, the player and the chain read each piece's OWN file
  R1 render_groups (Frank's 47 tracks): runs with a sweep stamp group by it -- one render per Run press, newest first,
     ONE clip per run (its pick; the backup / raw files it wrote counted as extras, never listed); a scene run twice
     in one render gives its newer clip
  R2 runs without a sweep (v1165..v1189) group by time: a scene number that does not grow, or 45 min of quiet, starts
     the next render
  R3 clips of the list WITHOUT any history (before v1165): files within 120 s are one run, its pick = the file named
     for its scene; the render is marked legacy ("by time")
  J6 takePlan: renders newest first in -> OLDEST first out (the newest ends on top), files V1 shows / a track has are
     skipped, an emptied render drops; clearTracks keeps V1 (its pins) and drops render_tracks; renderStamp
     "dd.mm. hh:mm:ss"; blurAfterChange: ticks / sliders / choices yes, text fields no
  W2 takes -> tracks asks /pls/cine/renders, builds one track per render (_renderTrack + _renderInto) and saves ONCE;
     the "x tracks" button sits in the toolbar; the timeline gives the focus back after a control's change; _saveEdit
     closes a still open undo bracket before it records
  Z  ASCII
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


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load("ce1198", "nodes/cine_edit.py")
JS = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")

# ---- C1
ce = E.clean_edit({"auto_mode": "render", "render_tracks": {"1000.5": "La", "x": 3, "": "Lb"}})
_need(ce.get("auto_mode") == "render" and ce.get("render_tracks") == {"1000.5": "La"}, "C1 auto_mode render + render_tracks kept (%s)" % ce)
_need("auto_mode" not in E.clean_edit({"auto_mode": "render", "auto_track": False}) and "auto_mode" not in E.clean_edit({"auto_mode": "bogus"}),
      "C1 auto_mode dropped when the switch is off (replace) or unknown")
rt = {str(i): "L%d" % i for i in range(40)}
_need(len(E.clean_edit({"render_tracks": rt})["render_tracks"]) == 32 and "39" in E.clean_edit({"render_tracks": rt})["render_tracks"],
      "C1 render_tracks: at most 32, the newest kept")
lay = E.clean_layers([{"id": "L1", "src": {"kind": "render", "path": "day/a.mp4", "at": 0},
                       "more": [{"kind": "render", "path": "day/b.mp4", "at": 72}, {"kind": "zip", "path": "x"}, {"kind": "file", "path": ""}],
                       "pieces": [{"k": 0, "a": 0, "b": 48, "speed": 1}, {"k": 1, "a": 0, "b": 24, "speed": 1}, {"k": 5, "a": 0, "b": 10, "speed": 1}]},
                      {"id": "base", "src": {"kind": "base"}}])
L1 = lay[0] if lay else {}
_need(L1.get("more") == [{"kind": "render", "path": "day/b.mp4", "at": 72}] and L1["src"].get("at") == 0,
      "C1 more files kept (the bad ones dropped), src.at kept")
_need([p["k"] for p in L1.get("pieces", [])] == [0, 1] and E.is_gap(L1["pieces"][-1]) is False and len(L1["pieces"]) == 2,
      "C1 a piece naming a file the track does not have -> empty (and an empty tail goes): %s" % L1.get("pieces"))
lay2 = E.clean_layers([{"id": "L1", "src": {"kind": "render", "path": "a"}, "more": [{"kind": "render", "path": "m%d" % i} for i in range(80)],
                        "pieces": [{"k": 0, "a": 0, "b": 4, "speed": 1}]}])
_need(len(lay2[0]["more"]) == 64, "C1 at most 64 more files")

# ---- J (node)
node = shutil.which("node")
TMPJ = Path(tempfile.mkdtemp(prefix="v1198js_"))
J = None
if node:
    body = (JS.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (TMPJ / "m.mjs").write_text(body, encoding="utf-8")
    (TMPJ / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const G = (n) => T.gapPiece(n), C = (k, a, b) => ({ k, a, b, speed: 1 });
const lens = (q) => q.map((p) => [T.isGap(p) ? "g" : p.k, T.pieceLen(p), T.isGap(p) ? 0 : p.a, T.isGap(p) ? 0 : p.b]);
const out = {};
out.choices = T.AUTO_CHOICES.map((x) => x[0]);
out.modes = [T.autoMode({}), T.autoMode({ auto_mode: "render" }), T.autoMode({ auto_track: false, auto_mode: "render" }), T.autoMode({ auto_track: true })];
out.wipe = T.wipeEdit({ auto_mode: "render", layers: [1], span: 9 });
out.i_empty = lens(T.insertClip([], C(0, 0, 48), 30));
out.i_past = lens(T.insertClip([C(0, 0, 24)], C(1, 0, 10), 50));
out.i_over = lens(T.insertClip([C(0, 0, 100)], C(1, 0, 20), 40));
out.i_gap = lens(T.insertClip([C(0, 0, 24), G(30), C(0, 50, 60)], C(2, 0, 30), 24));
out.i_head = lens(T.insertClip([G(10), C(0, 0, 20)], C(1, 0, 15), 0));
out.cut_gap = lens(T.cutAtFrame([G(30), C(0, 0, 10)], 12));
out.cut_pc = lens(T.cutAtFrame([C(0, 5, 45)], 15));
const e0 = { layers: [{ id: "L1", src: { kind: "empty" }, pieces: [], video: true, audio: true, lock: false }, { id: "base", src: { kind: "base" } }] };
const a1 = T.addSource(T.layerOf(e0, "L1"), { kind: "render", path: "day/a.mp4", at: 0 });
out.add_empty = [a1.k, a1.lay.src];
const a2 = T.addSource(a1.lay, { kind: "render", path: "day/a.mp4" });
const a3 = T.addSource(a1.lay, { kind: "file", path: "/x/b.mp4", at: 70 });
out.add_more = [a2.k, a3.k, a3.lay.more];
// J4 the sweep: scenes at 0 / 72 / 140, files of 72 / 68 / 50 frames, laid one after another on ONE track
let e = e0;
e = T.dropOnto(e, "L1", { kind: "render", path: "day/s1.mp4", at: 0 }, C(0, 0, 72), 0);
e = T.dropOnto(e, "L1", { kind: "render", path: "day/s3.mp4", at: 140 }, C(0, 0, 50), 140);
e = T.dropOnto(e, "L1", { kind: "render", path: "day/s2.mp4", at: 72 }, C(0, 0, 68), 72);
const L = T.layerOf(e, "L1");
out.sweep = { src: L.src.path, more: (L.more || []).map((x) => [x.path, x.at]), pieces: lens(L.pieces), n: T.stackLayers(e).length };
const fr = { 0: 72, 1: 50, 2: 68 };
out.reset = lens(T.resetMulti(L, (k) => fr[k]));
out.src_out = T.srcOfClip({ view: { filename: "a.mp4", subfolder: "2026-10-21/PH", type: "output" }, path: "/o/2026-10-21/PH/a.mp4" });
out.src_in = T.srcOfClip({ view: { filename: "b.mp4", subfolder: "", type: "input" }, path: "/i/b.mp4" });
out.plan = T.takePlan([{ stamp: 3, runs: [{ k: 1, rel: "c1" }, { k: 2, rel: "c2" }] }, { stamp: 2, runs: [{ k: 1, rel: "b1" }] }, { stamp: 1, runs: [{ k: 1, rel: "a1" }, { k: 2, rel: "a2" }] }],
    new Set(["c1", "c2", "a2"])).map((g) => [g.stamp, g.runs.map((r) => r.rel)]);
out.clear = T.clearTracks({ layers: [{ id: "L1" }, { id: "base", lock: true }], takes: { 1: "x" }, render_tracks: { 1: "L1" }, span: 5 });
out.clear0 = T.clearTracks({ layers: [{ id: "L1" }], takes: { 1: "x" } });
out.stamp = /^\d\d\.\d\d\. \d\d:\d\d:\d\d$/.test(T.renderStamp(1761000000));
out.blur = [{ tagName: "SELECT" }, { tagName: "INPUT", type: "checkbox" }, { tagName: "INPUT", type: "range" }, { tagName: "INPUT", type: "text" }, { tagName: "TEXTAREA" }, null].map((t) => T.blurAfterChange(t));
out.clipk = [T.clipK({ clips: [{ frames: 1 }, { frames: 2 }] }, 1).frames, T.clipK({ clip: { frames: 7 } }, 0).frames, T.clipK({ clip: { frames: 7 } }, 1)];
console.log(JSON.stringify(out));
""", encoding="utf-8")
    r = subprocess.run([node, str(TMPJ / "h.mjs")], capture_output=True, text=True, timeout=60)
    try:
        J = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        print(r.stdout[-800:], r.stderr[-1500:])
shutil.rmtree(TMPJ, ignore_errors=True)
if J:
    _need(J["choices"] == ["track", "render", "replace"], "J1 AUTO_CHOICES: track / render / replace (%s)" % J["choices"])
    _need(J["modes"] == ["track", "render", "replace", "track"], "J1 autoMode (%s)" % J["modes"])
    _need(J["wipe"].get("auto_mode") == "render" and "layers" not in J["wipe"] and J["wipe"].get("span") == 9, "J1 wipeEdit keeps the render choice and the span")
    _need(J["i_empty"] == [["g", 30, 0, 0], [0, 48, 0, 48]], "J2 onto an empty layer at 30: gap 30 + the clip (%s)" % J["i_empty"])
    _need(J["i_past"] == [[0, 24, 0, 24], ["g", 26, 0, 0], [1, 10, 0, 10]], "J2 past the end: a gap up to it (%s)" % J["i_past"])
    _need(J["i_over"] == [[0, 40, 0, 40], [1, 20, 0, 20], [0, 40, 60, 100]], "J2 onto material: exactly its span overwritten, left and right keep their frames (%s)" % J["i_over"])
    _need(J["i_gap"] == [[0, 24, 0, 24], [2, 30, 0, 30], [0, 10, 50, 60]], "J2 into a gap of the same size: the gap filled (%s)" % J["i_gap"])
    _need(J["i_head"] == [[1, 15, 0, 15], [0, 15, 5, 20]], "J2 at frame 0 over a gap and a piece: the piece trimmed at its head (%s)" % J["i_head"])
    _need(J["cut_gap"] == [["g", 12, 0, 0], ["g", 18, 0, 0], [0, 10, 0, 10]] and J["cut_pc"] == [[0, 15, 5, 20], [0, 25, 20, 45]],
          "J2 cutAtFrame splits a gap and a piece (%s / %s)" % (J["cut_gap"], J["cut_pc"]))
    _need(J["add_empty"] == [0, {"kind": "render", "path": "day/a.mp4", "at": 0}], "J3 an empty track takes the file as k 0 (%s)" % J["add_empty"])
    _need(J["add_more"] == [0, 1, [{"kind": "file", "path": "/x/b.mp4", "at": 70}]], "J3 a file it has is reused, a new one appended (%s)" % J["add_more"])
    sw = J["sweep"]
    _need(sw["n"] == 2 and sw["src"] == "day/s1.mp4" and sw["more"] == [["day/s3.mp4", 140], ["day/s2.mp4", 72]],
          "J4 ONE track, three files in the order they came (%s)" % sw)
    _need(sw["pieces"] == [[0, 72, 0, 72], [2, 68, 0, 68], [1, 50, 0, 50]], "J4 each scene at its frame: 0 / 72 / 140, k in film order 0, 2, 1 (%s)" % sw["pieces"])
    _need(J["reset"] == sw["pieces"], "J3 resetMulti: every file whole where it was laid (%s)" % J["reset"])
    _need(J["src_out"] == {"kind": "render", "path": "2026-10-21/PH/a.mp4"} and J["src_in"] == {"kind": "file", "path": "/i/b.mp4"},
          "J5 srcOfClip: output/ -> render + its path there; else file + where it lies")
    _need(J["clipk"] == [2, 7, None], "J5 clipK: row.clips by k, row.clip for k 0 of a one-file track")
    _need(J["plan"] == [[1, ["a1"]], [2, ["b1"]]], "J6 takePlan: oldest first out, shown / tracked files skipped, an emptied render drops (%s)" % J["plan"])
    _need(J["clear"] == {"layers": [{"id": "base", "lock": True}], "takes": {"1": "x"}, "span": 5} and J["clear0"] == {"takes": {"1": "x"}},
          "J6 clearTracks: V1 (and its pins) stay, every track and render_tracks go (%s / %s)" % (J["clear"], J["clear0"]))
    _need(J["stamp"] is True and J["blur"] == [True, True, True, False, False, False], "J6 renderStamp dd.mm. hh:mm:ss; blurAfterChange (%s)" % J["blur"])
else:
    _need(False, "J node.js run failed")

# ---- S1 / S2 the server, real files
ff = shutil.which("ffmpeg")
if ff:
    TMP = Path(tempfile.mkdtemp(prefix="v1198_"))
    for d in ("output/day", "output/cine", "input", "temp"):
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
    pkg = types.ModuleType("plsn1198")
    pkg.__path__ = [str(ROOT / "nodes")]
    sys.modules["plsn1198"] = pkg
    TL = importlib.import_module("plsn1198.ph_cine_timeline")
    T = sys.modules.get("plsn1198.cine_timeline") or importlib.import_module("plsn1198.cine_timeline")

    def mk(args, out):
        return subprocess.run([ff, "-v", "error", "-y"] + args + [str(out)], capture_output=True, text=True).returncode == 0

    ok = (mk(["-f", "lavfi", "-i", "testsrc2=s=320x180:r=24", "-frames:v", "144", "-c:v", "libx264", "-pix_fmt", "yuv420p"], TMP / "output/cine/whole_00001_.mp4")
          and mk(["-f", "lavfi", "-i", "color=c=0xe00000:s=320x180:r=24", "-frames:v", "48", "-c:v", "libx264", "-pix_fmt", "yuv420p"], TMP / "output/day/red.mp4")
          and mk(["-f", "lavfi", "-i", "color=c=0x00c000:s=320x180:r=30", "-frames:v", "90", "-c:v", "libx264", "-pix_fmt", "yuv420p"], TMP / "output/day/green.mp4"))
    _need(ok, "S1 test clips (base 144 @ 24, red 48 @ 24, green 90 @ 30 = 72 film frames)")
    model = T.load(json.dumps({"seconds": 6.0, "project": "cine", "render": "whole", "shots": [{"at": 0, "cut": "the camera cuts to", "text": "x"}]}))
    gap = E.gap_piece
    track = {"id": "LR", "name": "render 07:02:54", "src": {"kind": "render", "path": "day/red.mp4", "at": 0},
             "more": [{"kind": "render", "path": "day/green.mp4", "at": 72}],
             "pieces": [{"k": 0, "a": 0, "b": 48, "speed": 1}, gap(24), {"k": 1, "a": 0, "b": 72, "speed": 1}]}
    base_p = [{"k": 0, "a": 0, "b": 144, "speed": 1}]
    ed = E.clean_edit({"pieces": {"whole": base_p}, "layers": [track, {"id": "base", "src": {"kind": "base"}}]})
    d1 = TL.edit_plan(model, ed, strips=False)
    film = d1.get("film") or {}
    row = (film.get("layers") or [{}])[0]
    V = [(r["film_in"], r["film_out"], r["layer"]) for r in film.get("flat", {}).get("vruns", [])]
    _need(film.get("complete") and V == [(0, 48, 0), (48, 72, 1), (72, 144, 0)], "S1 the film complete, the track's picture where its pieces say (%s)" % V)
    _need(len(row.get("clips") or []) == 2 and (row["clips"][1] or {}).get("frames") == 72 and [p.endswith(("red.mp4", "green.mp4")) for p in row.get("paths_k") or []] == [True, True],
          "S1 the row carries clips + paths_k (the 30 fps file = 72 film frames)")
    bad = json.loads(json.dumps(ed))
    bad["layers"][0]["more"][0]["path"] = "day/nope.mp4"
    db = TL.edit_plan(model, bad, strips=False)
    rb = db["film"]["layers"][0]
    _need("file 2" in (rb.get("error") or "") and (rb.get("clip") or {}).get("frames") == 48 and not db["film"]["complete"],
          "S1 a missing second file names itself, the first still plays (%s)" % rb.get("error"))
    try:
        out, lines = TL.export_film(model, ed, preview=False)
    except Exception as e:
        out, lines = None, ["%s" % e]
    _need(out and Path(out).is_file(), "S2 export over the server's path: %s" % (lines[0] if lines else "?"))
    if out and Path(out).is_file():
        raw = subprocess.run([ff, "-v", "error", "-i", out, "-vf", "scale=32:18", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
        fsz = 32 * 18 * 3
        nfr = len(raw) // fsz

        def col(f):
            o = f * fsz + (9 * 32 + 16) * 3
            r_, g_, b_ = raw[o:o + 3]
            return "red" if r_ > 150 and g_ < 80 else "green" if g_ > 120 and r_ < 80 else "other"
        got = [col(f) for f in (5, 40, 100, 140)]
        _need(nfr == 144 and got == ["red", "red", "green", "green"], "S2 144 frames; red from file 1, green from file 2 where they lie (%d, %s)" % (nfr, got))
        mid = col(60)
        _need(mid == "other", "S2 the gap between them shows the render below (%s)" % mid)
    shutil.rmtree(TMP, ignore_errors=True)
else:
    _need(False, "S ffmpeg is needed")

# ---- R1..R3 render_groups
A = _load("ca1198", "nodes/cine_assemble.py")
import os  # noqa: E402
RT = Path(tempfile.mkdtemp(prefix="v1198rg_"))
(RT / "out" / "day").mkdir(parents=True)
PF = RT / "proj"
PF.mkdir()
A.OUTPUT_ROOT = str(RT / "out")


def put(name, mt=None):
    f = RT / "out" / "day" / name
    f.write_bytes(b"x")
    if mt is not None:
        os.utime(f, (mt, mt))
    return "day/" + name


def run1(k, sweep, tag):
    rel = put("PH_%s_neon_scene_%d_00001_.mp4" % (tag, k))
    bak = put("PH_%s_UPSCALE_BAK_PH_00001_.mp4" % tag)
    raw = put("PH_%s_MINIMAX_00001_.mp4" % tag)
    for f in (bak, raw):
        A.register(str(PF), k, str(RT / "out" / f))
    A.register(str(PF), k, str(RT / "out" / rel), primary=True, run_files=[rel, bak, raw], sweep=sweep)
    return rel


ra = {k: run1(k, 1000.0, "a%d" % k) for k in (1, 2, 3)}
rb = {k: run1(k, 2000.0, "b%d" % k) for k in (1, 2, 3)}
rb2 = run1(2, 2000.0, "b2x")                        # scene 2 again inside render B: the newer one counts
G = A.render_groups(str(PF), 3)
gr = [(g["sweep"], [(r["k"], r["rel"]) for r in g["runs"]]) for g in G["renders"]]
_need(gr == [(2000.0, [(1, rb[1]), (2, rb2), (3, rb[3])]), (1000.0, [(1, ra[1]), (2, ra[2]), (3, ra[3])])],
      "R1 two renders newest first, one pick per run, scene 2 run twice -> the newer (%s)" % gr)
_need(G["extras"] == 14 and not any("BAK" in r["rel"] or "MINIMAX" in r["rel"] for g in G["renders"] for r in g["runs"]),
      "R1 the backup / raw saves are never listed (counted: %d)" % G["extras"])
# R2: a second project, history without sweep
PF2 = RT / "proj2"
PF2.mkdir()
hist = {}
T0 = 5000.0
for i, (k, dt) in enumerate(((1, 0), (2, 300), (3, 600), (1, 900), (2, 1200), (2, 1500), (3, 1500 + 46 * 60))):
    rel = put("PH_r2_%d_neon_scene_%d_00001_.mp4" % (i, k))
    hist.setdefault(k, []).append({"rel": rel, "files": [rel], "at": T0 + dt})
A._write_list(str(PF2), {k: [h["rel"] for h in v] for k, v in hist.items()}, {}, hist)
G2 = A.render_groups(str(PF2), 3)
g2 = [[r["k"] for r in g["runs"]] for g in G2["renders"]]
_need(g2 == [[3], [2], [1, 2], [1, 2, 3]], "R2 without a sweep: scene order + 45 min quiet split the renders (%s)" % g2)
# R3: a list without any history
PF3 = RT / "proj3"
PF3.mkdir()
L3, T3 = {}, 9000.0
for run, base in ((0, T3), (1, T3 + 3600)):
    for k in (1, 2):
        b0 = base + k * 400
        names = [put("PH_l%d%d_UPSCALE_BAK_PH_00001_.mp4" % (run, k), b0), put("PH_l%d%d_neon_scene_%d_00001_.mp4" % (run, k, k), b0 + 20),
                 put("PH_l%d%d_MINIMAX_00001_.mp4" % (run, k), b0 + 40)]
        L3.setdefault(k, []).extend(names)
A._write_list(str(PF3), L3, {}, {})
G3 = A.render_groups(str(PF3), 2)
g3 = [(g["legacy"], [os.path.basename(r["rel"]) for r in g["runs"]]) for g in G3["renders"]]
_need(g3 == [(True, ["PH_l11_neon_scene_1_00001_.mp4", "PH_l12_neon_scene_2_00001_.mp4"]), (True, ["PH_l01_neon_scene_1_00001_.mp4", "PH_l02_neon_scene_2_00001_.mp4"])]
      and G3["extras"] == 8, "R3 no history: files within 120 s = one run, the scene-named file is its pick, marked legacy (%s, extras %d)" % (g3, G3["extras"]))
shutil.rmtree(RT, ignore_errors=True)

# ---- W wiring
a1 = JS[JS.find("async function _autoTrack1("):JS.find("/* ------------------------------------------------------------ v1036 the player")]
_need('autoMode(e) === "render" && sweep' in a1 and "(e.render_tracks || {})[key]" in a1 and "_renderInto(node, rel, _sceneAt(node, scene), e, id)" in a1
      and '"render " + stamp' in a1 and "e2.render_tracks = rt" in a1, "W the render branch: its track by the sweep stamp, else ONE new track remembered")
_need(a1.find('autoMode(e) === "render"') > a1.find("if (!(e.takes && e.takes[key]))"), "W the first run of a place still goes to V1 first")
_need("api.queuePrompt = wrapQ" in JS and "sweepOfPid.set(pid, sw)" in JS and "sweep: sweepOfPid.get(pid) ||" in JS
      and "_autoTrack(n, pick, run.scene, run.rels.length, sweepOfPid.get(pid)" in JS, "W the sweep stamped at QUEUE time, used by register and the auto track")
lr = JS[JS.find("function _layerRows("):JS.find("/* a track's height: pulled at its row's bottom edge")]
_need("_laneUnder(node, e.clientX, e.clientY, lay.id)" in lr and "_moveAcross(node, { kind: sx.kind, path: sx.path, at: sx.at }, pcs[j]" in lr
      and "copy ? null : { id: lay.id, j, pcs }" in lr, "W a track piece: dragged onto another track = moved (Ctrl = a copy)")
_need("const c = clipK(row, sg.k) || {}, url = c.strip" in lr and "const c = clipK(row, sg.k) || {};    // v1198" in lr, "W the drawing reads each piece's own file (picture + sound)")
cl = JS[JS.find("function _clipLane("):JS.find("function _clipLane(") + 14000]
_need("_laneUnder(node, e.clientX, e.clientY, BASE_ID)" in cl and "_moveAcross(node, src, pcs[i]" in cl and ", to, null);" in cl, "W V1's piece onto a track = a copy")
_need("const url = _clipURL(clipK(r, sg.k))" in JS and "((clipK(r, k) || {}).frames || Infinity)" in JS, "W the player and the chain read each piece's own file")
mv = JS[JS.find("function _moveAcross("):JS.find("function _saveLayerPieces(")]
_need("deletePiece(from.pcs, from.j)" in mv and mv.count("_saveEdit(node, e)") == 1, "W the move empties the old place and saves ONCE (one undo step)")

tt = JS[JS.find("async function _takesToTracks("):JS.find("export function clearTracks(")]
_need('"/pls/cine/renders"' in tt and "_renderTrack(node, r.rel, at, e," in tt and "_renderInto(node, r.rel, at, e, id)" in tt and tt.count("_saveEdit(node, e)") == 1,
      "W2 takes -> tracks: one track per render from /pls/cine/renders, saved ONCE")
_need("sep(), autoLab, restLab, takesBtn, clearBtn);" in JS and '("POST", "/pls/cine/renders", handle_renders)' in (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8"),
      "W2 the x tracks button in the toolbar; the renders route registered")
_need('root.addEventListener("change", (ev) => { const t = ev.target; if (blurAfterChange(t))' in JS, "W2 the timeline gives the focus back after a control's change")
se = JS[JS.find("function _saveEdit(node, e) {"):JS.find("function _saveEdit(node, e) {") + 600]
_need("if (node._plsTl && node._plsTl.bracket) _end(node);" in se and se.find("_end(node)") < se.find("node.properties[EDIT_KEY] = e") < se.find("commitEdit()"),
      "W2 _saveEdit closes an open bracket BEFORE it writes the edit, then records it (one undo step of its own)")
for p in ("nodes/cine_assemble.py", "nodes/cine_edit.py", "nodes/ph_cine_timeline.py", "web/js/ph_cine_timeline.js", "tests/test_v1198_one_track.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "Z ASCII %s" % p)
print("%s: %d check(s) failed" % ("test_v1198_one_track", len(FAILS)) if FAILS else "test_v1198_one_track: all checks passed")
sys.exit(1 if FAILS else 0)
