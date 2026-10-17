# -*- coding: ascii -*-
"""Guard v1037 -- the film as a list of PIECES: split, heal, trim, stretch,
slip (nodes/cine_edit.py + web/js/ph_cine_timeline.js), the audio lane,
the export per piece.

Frank, 29.09.: "a handle to spread the video a little longer; a click on
the clip shows its first frame; the playhead amber; a razor to make three or
four clips out of one, a plaster to heal it, then pull the pieces longer or
shorter; and an audio lane under the video."

  P1  pieces from the prompt: one per scene (the v1036 placement: a = where
      it shows at its cut), or the whole run; the plan hands them back.
  P2  explicit pieces: lengths from speed, the running cuts, a split seam
      (same clip, touching) as a cut, a transition on it, holds at head and
      tail (named), place_cuts only where the clip changes.
  P3  a cut list that does not fit the clips any more is named and not used;
      clean_edit keeps only well-formed pieces (speed clamped 0.25..4).
  P4  ffmpeg: tempo (setpts + framerate / fps, atempo chain), a held first
      picture (tpad start + adelay); every piece ends as exactly its frames;
      a real run with a split, a slowed and a faster piece = the plan's count.
  P5  the editor's piece ops (node.js): splitAt (and its edges), healableAt
      (+ tolerance), heal (the film keeps its length), trimSeam (one gives,
      one takes, inside the material), stretch (speed, ripple), slipPiece,
      trimEnd, srcTime at a speed and after a held head.
  S   statics: one ffmpeg input per piece (the v1037 export bug), the razor
      and the plaster at the playhead, the stretch knob, click on a clip =
      its first frame, the amber playhead, the end frame not black, the
      audio lane, get_running_loop (v577 rule), BASE_H 611.

Script-style: exit 0 = pass.
"""
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import tempfile
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


E = _load("ce1037", "nodes/cine_edit.py")
T = _load("ct1037", "nodes/cine_timeline.py")
K = T.TRANSITIONS
CL = [{"frames": 56}, {"frames": 56}, {"frames": 73}]

# ------------------------------------------------------------------ P1
p = E.plan([2.0, 4.0], 6.0, CL, [{"kind": "dissolve", "seconds": 0.5}, {"kind": "cut"}], K)
_need(p["source"] == "prompt" and [(x["k"], x["a"], x["b"]) for x in p["pieces"]] == [(1, 0, 48), (2, 6, 54), (3, 0, 48)]
      and p["pieces"][1]["tr"]["kind"] == "dissolve", "P1 pieces from the prompt: one per scene, a = where it shows at its cut")
w = E.plan([], 6.0, [{"frames": 158}], [], K, whole=True)
_need([(x["k"], x["a"], x["b"]) for x in w["pieces"]] == [(0, 0, 144)], "P1 the whole run: one piece")

# ------------------------------------------------------------------ P2
PC = [{"k": 0, "a": 0, "b": 48, "speed": 1}, {"k": 0, "a": 48, "b": 96, "speed": 0.8, "tr": {"kind": "dissolve", "seconds": 0.5}},
      {"k": 0, "a": 96, "b": 150, "speed": 1.5}]
q = E.plan([], 6.0, [{"frames": 158}], [], K, edit={"pieces": {"whole": PC}}, whole=True)
_need(q["source"] == "edit" and q["cuts"] == [48, 108] and q["length"] == 144 and [s["speed"] for s in q["segments"]] == [1.0, 0.8, 1.5],
      "P2 lengths from speed (48 / 60 / 36), the running cuts, the film 144 frames %r" % q["cuts"])
s1 = q["segments"][1]
_need(q["seams"][0]["d"] == 12 and s1["film_in"] == 42 and abs(s1["src_in"] - (48 - 6 * 0.8)) < 1e-9 and q["seams"][1]["d"] == 0,
      "P2 a dissolve on a split seam: centred, the piece after starts 6 film frames earlier in its own material")
_need(q["place_cuts"] == [], "P2 place_cuts: none where the clip stays the same")
h = E.plan([], 6.0, [{"frames": 158}], [], K, edit={"pieces": {"whole": [{"k": 0, "a": 2, "b": 50, "speed": 1},
                                                                            {"k": 0, "a": 60, "b": 158, "speed": 1, "tr": {"kind": "dissolve", "seconds": 1}}]}}, whole=True)
_need(h["segments"][0]["hold_tail"] == 0 and h["segments"][1]["hold_tail"] == 0 and h["segments"][1]["hold_head"] == 0
      and h["segments"][0]["src_out"] == 62, "P2 a blend takes the material around the cut (no hold where there is some)")
h2 = E.plan([], 6.0, [{"frames": 158}], [], K, edit={"pieces": {"whole": [{"k": 0, "a": 100, "b": 158, "speed": 1},
                                                                             {"k": 0, "a": 0, "b": 60, "speed": 1, "tr": {"kind": "dissolve", "seconds": 1}}]}}, whole=True)
_need(h2["segments"][0]["hold_tail"] == 12 and h2["segments"][1]["hold_head"] == 12
      and sum(1 for x in h2["warnings"] if "held" in x) == 2, "P2 no material past the end / before the start: the picture is held, named twice")
sc = E.plan([2.0, 4.0], 6.0, CL, [], K, edit={"pieces": {"scenes": [{"k": 1, "a": 0, "b": 48, "speed": 1}, {"k": 2, "a": 0, "b": 20, "speed": 1},
                                                                     {"k": 2, "a": 20, "b": 48, "speed": 1}, {"k": 3, "a": 0, "b": 48, "speed": 1}]}})
_need(sc["place_cuts"] == [48, 96] and sc["cuts"] == [48, 68, 96], "P2 place_cuts only where the clip changes (48, 96), the split at 68 is not one")

# ------------------------------------------------------------------ P3
bad = E.plan([2.0, 4.0], 6.0, CL, [], K, edit={"pieces": {"scenes": [{"k": 1, "a": 0, "b": 48, "speed": 1}, {"k": 5, "a": 0, "b": 48, "speed": 1}]}})
_need(bad["source"] == "prompt" and any("does not fit" in x and "5" in x for x in bad["warnings"]), "P3 a cut list for other clips is named and not used")
bad = E.plan([2.0, 4.0], 6.0, CL, [], K, edit={"pieces": {"scenes": [{"k": 1, "a": 60, "b": 70, "speed": 1}]}})
_need(bad["source"] == "prompt", "P3 a piece starting past its clip's material is not used")
ce = E.clean_edit({"pieces": {"whole": [{"k": 0, "a": 0, "b": 10, "speed": 9}], "scenes": [{"k": 1, "a": 5, "b": 5}]}})
_need(ce == {"pieces": {"whole": [{"k": 0, "a": 0, "b": 10, "speed": 4.0}]}}, "P3 clean_edit: speed clamped to 4, an empty piece drops its list %r" % ce)

# ------------------------------------------------------------------ P4
g = E.filter_graph(q, True)[0]
_need("setpts=PTS/0.800000,framerate=fps=24" in g and "setpts=PTS/1.500000,fps=24" in g and "atempo=0.800000" in g and "atempo=1.500000" in g,
      "P4 tempo: slower = framerate blend, faster = fps drop, the sound by atempo")
_need(E.atempo_chain(0.25) == [0.5, 0.5] and E.atempo_chain(4) == [2.0, 2.0] and E.atempo_chain(3) == [2.0, 1.5], "P4 the atempo chain stays inside 0.5 .. 2")
g2 = E.filter_graph(h2, True)[0]
_need("tpad=start_mode=clone:start=12" in g2 and "adelay=500:all=1" in g2, "P4 a held first picture: tpad at the start, the sound delayed (silent)")
_need(all(",trim=end_frame=%d," % (s["film_out"] - s["film_in"]) in g for s in q["segments"]), "P4 every piece ends as exactly its film frames")
exe = shutil.which("ffmpeg")
if exe is None:
    print("  note  ffmpeg not found -- P4 real run SKIPPED")
else:
    tmp = Path(tempfile.mkdtemp(prefix="v1037_"))
    src = tmp / "whole.mp4"
    subprocess.run([exe, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x36:r=24", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
                    "-frames:v", "158", "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(src)], capture_output=True)
    res = []
    for plan_ in (q, h2):
        out = tmp / "o.mp4"
        r = subprocess.run(E.ffmpeg_command(exe, [str(src)] * len(plan_["segments"]), str(out), plan_, True, preview=True), capture_output=True, text=True)
        n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(out)],
                           capture_output=True, text=True).stdout.strip() if shutil.which("ffprobe") else str(plan_["length"])
        res.append((r.returncode, n, str(plan_["length"])))
    shutil.rmtree(tmp, ignore_errors=True)
    _need(all(x[0] == 0 and x[1] == x[2] for x in res), "P4 ffmpeg: a split, slowed and faster film, and one with held pictures = the plan's frames %r" % res)

# ------------------------------------------------------------------ P5
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
if shutil.which("node") is None:
    print("  note  node.js not found -- P5 SKIPPED")
else:
    t2 = Path(tempfile.mkdtemp(prefix="v1037js_"))
    body = (js.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (t2 / "m.mjs").write_text(body, encoding="utf-8")
    (t2 / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const P = [{ k: 0, a: 0, b: 144, speed: 1 }];
const fK = () => 158;
const o = {};
const s1 = T.splitAt(P, 48), s2 = T.splitAt(s1, 96);
o.split = s2.map((p) => [p.a, p.b]);
o.edge = [T.splitAt(P, 1), T.splitAt(P, 143), T.splitAt(P, 200)];
o.heal = [T.healableAt(s2, 50), T.healableAt(s2, 60), T.healableAt(s2, 94)];
o.healed = T.heal(s2, 1).map((p) => [p.a, p.b]);
const tr = T.trimSeam(s2, 1, 60, fK);
o.trim = tr.map((p) => [p.a, p.b]);
o.trimClamp = T.trimSeam(s2, 2, 500, fK).map((p) => [p.a, p.b]);
const st = T.stretch(s2, 0, 60);
o.stretch = [st[0].speed, T.starts(st), T.pieceLen(st[0])];
const slow = st;
o.healSpeed = (() => { const h = T.heal(slow, 1); return [h[0].speed, T.pieceLen(h[0]), T.pieceLen(slow[0]) + T.pieceLen(slow[1])]; })();
o.slip = T.slipPiece(s2, 2, 30, fK).map((p) => [p.a, p.b]);
o.end = T.trimEnd(s2, 150, fK).map((p) => [p.a, p.b]);
o.endClamp = T.trimEnd(s2, 400, fK)[2].b;
const seg = { film_in: 10, film_out: 70, src_in: 0, src_out: 40, speed: 0.8, hold_head: 6 };
o.src = [T.srcTime(seg, 12 / 24), T.srcTime(seg, 26 / 24), T.srcTime(seg, 69 / 24)];
o.noheal = [T.healableAt([{ k: 1, a: 0, b: 48, speed: 1 }, { k: 2, a: 0, b: 48, speed: 1 }], 48),
            T.healableAt([{ k: 1, a: 0, b: 40, speed: 1 }, { k: 1, a: 44, b: 90, speed: 1 }], 40)];
o.touch = T.splitAt([{ k: 1, a: 10, b: 60, speed: 0.5 }], 20);
console.log(JSON.stringify(o));
""", encoding="utf-8")
    r = subprocess.run(["node", str(t2 / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(t2, ignore_errors=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        _fail("P5 harness failed: %s" % r.stderr.strip()[-400:])
    else:
        d = json.loads(line[-1])
        _need(d["split"] == [[0, 48], [48, 96], [96, 144]], "P5 the razor twice: one clip -> three pieces that touch %r" % d["split"])
        _need(d["edge"] == [None, None, None], "P5 no split one frame from an edge or past the end")
        _need(d["heal"] == [1, -1, 2], "P5 healableAt: the split within 6 frames of the playhead, none between them %r" % d["heal"])
        _need(d["healed"] == [[0, 96], [96, 144]], "P5 heal: one piece again")
        _need(d["noheal"] == [-1, -1], "P5 no heal where two scenes meet, nor where the pieces do not touch (a gap in the material)")
        _need(d["trim"] == [[0, 60], [60, 96], [96, 144]], "P5 trimSeam: the seam to 60 -- one gives, the other takes, they touch")
        _need(d["trimClamp"][1][1] == d["trimClamp"][2][0] and d["trimClamp"][2][1] - d["trimClamp"][2][0] == 2,
              "P5 trimSeam is clamped: the piece after keeps two frames %r" % d["trimClamp"])
        _need(abs(d["stretch"][0] - 0.8) < 1e-9 and d["stretch"][1] == [0, 60, 108] and d["stretch"][2] == 60,
              "P5 stretch piece 1 to 60 frames: x0.80, the pieces after it move (ripple)")
        _need(d["healSpeed"][1] == d["healSpeed"][2], "P5 heal of pieces at different speeds keeps the film's length %r" % d["healSpeed"])
        _need(d["slip"] == [[0, 48], [48, 96], [110, 158]], "P5 slip: a and b move together, inside the material")
        _need(d["end"] == [[0, 48], [48, 96], [96, 150]] and d["endClamp"] == 158, "P5 trimEnd: the last piece takes more, up to its material")
        want = [(0 + 0.5) / 24, (int(10 * 0.8) + 0.5) / 24, (39 + 0.5) / 24]
        _need(all(abs(a - b) < 1e-9 for a, b in zip(d["src"], want)), "P5 srcTime: a held head, then the material at x0.8, the last frame held %r" % d["src"])
        _need(d["touch"] and d["touch"][0]["b"] == d["touch"][1]["a"] == 20 and d["touch"][1]["speed"] == 0.5,
              "P5 a split of a slowed piece cuts its material at the playhead's source frame")

# ------------------------------------------------------------------ S
node_py = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
_need('paths = [by_k[sg["k"]]["path"] for sg in p["segments"]]' in node_py,
      "S the export gives ffmpeg one input per PIECE (v1037 bug: 'Invalid file index 3')")
_need("get_event_loop(" not in node_py and "get_running_loop()" in node_py, "S the routes use get_running_loop (the v577 rule)")
_need('item["wave"]' in node_py and "showwavespic" in node_py, "S the audio lane gets the clips' waveforms")
# v1038 (Frank 29.09.: "unten split, oben cut -- nicht einheitlich"): the razor is called "cut"
_need('mk("\\u2702 cut", () => _split(node)' in js and 'mk("\\u271a heal", () => _heal(node)' in js, "S the razor and the plaster sit at the player")
# v1038: the knob became the diamond ON each cut's line (and the film's end) -- still every piece
_need('kn.dataset.pls = "stretch";' in js and "if (done) _savePieces(node, stretch(pcs, j, len));" in js
      # v1039: the diamond lives on the cut layer and carries its line along
      and "_diamond(node, L, pct, segH, pcs, st0, j, pic, snd, ao);" in js and "_diamond(node, L, pct, segH, pcs, st0, pcs.length - 1, ep, es, 0);" in js,
      "S the stretch knob on every piece")
_need('st.sel = { kind: "clip", i }; _playerSeek(node, st0[i] / FPS);' in js, "S a click on a clip = its first frame in the player")
_need("border-left:2px solid #ffb300;" in js and "#ff5252" not in js, "S the playhead is amber")
_need("Math.min(pl.t || 0, filmLen(D) / FPS - 0.5 / FPS)" in js, "S at the very end the player shows the last frame (was black; v1044: the stack's length)")
# AMENDED v1153: V1's row has a handle (v1Row) for the pair's clamp
_need('box.dataset.pls = "audioclip";' in js and "v1Row, _row(a1Head, audio), below," in js, "S the audio lane sits under the clips (v1045: in their rows, the heads left)")
# AMENDED v1150: the audio lane is a row of lanesH (LH.audio), the field adds the lanes
_need("const rows = [RULER_H, SCENES_H, SHOTS_H, LH.clip, LH.audio]" in js, "S the audio lane counts in the lanes' height")
_need("_save(node, mm); _render(node); _stack(node);      // v1037" in js, "S whole <-> scenes reloads the clip lane")

print()
if FAILS:
    print("test_v1037_pieces: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1037_pieces: PASS (P1-P5, S)")
