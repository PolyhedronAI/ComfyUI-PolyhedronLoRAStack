# -*- coding: ascii -*-
"""Guard v1038 -- the cutting desk: the playhead is dragged, one word for the
cut, the cut as a line with its speed anchor, lanes that can be pulled
taller (pictures in their aspect), picture and sound LINKED by default and
unlinked on demand (J-/L-cut).

Frank, 29.09.: "the amber line must be draggable, and the clip plays live
where it stands; below it says split, above cut -- not consistent; the cut
must be easy to see and gets its anchor right away to squeeze or stretch the
video (and audio!); the orange squares are of no real use; I want to pull
the video taller at the right to see the frames better; by default a cut
goes through video AND audio and both move together; an option lets you
split them apart, to pull the audio over the second video clip for a clean
audio transition. Look at how Premiere, Resolve and the like do it."

  J1  the sound's cut (cine_edit.py): ao on the piece after a seam, clamped
      inside the material (J: the next clip's frames before its a; L: the
      clip before's frames after its b) and between the neighbour cuts
      (MIN_SEG and both blends); every clamp is named; clean_edit keeps an
      int ao and drops a zero one.
  J2  the sound's segments: without ao they ARE the picture's (film window,
      source window, pad == hold); with ao the sound still adds up to the
      film (the offsets telescope) over a grid of edits with speeds and
      blends, and every piece's sound keeps its picture's source line (sync).
  J3  ffmpeg, measured: an L-cut plays the first clip's tone under the second
      picture, a J-cut the second clip's tone under the first picture,
      picture and sound end together; ao = 0 builds the v1037 graph.
  J4  the editor (node.js), mirrored against Python: aoRange / soundCuts over
      random edits, setSoundCut (clamped, 0 drops the key), trimCut (linked:
      the sound's cut goes along; unlinked: it stays), audioMixAt (the blend
      sits on the sound's cut), asrcTime, tilePlan (the pictures keep their
      aspect, cover the box, show the nearest tile), snapHead, laneHeights /
      laneExtra (clamped), linkedFor (Alt flips), speedLabel (percent).
  S   statics: the button says cut; the playhead has a grip and the ruler
      scrubs (snap, Shift free, dragging pauses); the player mixes picture
      and sound separately; each cut is a line with a grip and a diamond, the
      film's end too; the old knob is gone; the lanes have height grips and
      the node grows with them (fieldH); lane heights and the link switch
      live in node.properties, not in the timeline JSON; the strip is 24
      pictures at 96 px and says so; the export's sound comes from the sound's
      segments and must end with the picture; ASCII only; no floating window.

Script-style: exit 0 = pass.
"""
import importlib.util
import json
import random
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


E = _load("ce1038", "nodes/cine_edit.py")
T = _load("ct1038", "nodes/cine_timeline.py")
K = T.TRANSITIONS
DIS = [k for k in K if k not in ("cut",)][0]
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
node_py = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
edit_py = (ROOT / "nodes" / "cine_edit.py").read_text(encoding="utf-8")


def _pl(pcs, clips, trs=None):
    return E.plan([2.0], 4.0, clips, trs or [{"kind": "cut"}], K, edit={"pieces": {"scenes": pcs}})


CL2 = [{"frames": 96}, {"frames": 96}]
# ------------------------------------------------------------------ J1
base = [{"k": 1, "a": 10, "b": 58, "speed": 1.0}, {"k": 2, "a": 10, "b": 58, "speed": 1.0}]
L = [dict(base[0]), dict(base[1], ao=12)]
p = _pl(L, CL2)
_need(p["seams"][0]["ao"] == 12 and p["seams"][0]["a_at"] == 60 and not p["warnings"], "J1 an L-cut of 0.5 s sits 12 frames after the picture's cut")
J = [dict(base[0]), dict(base[1], ao=-12)]
p = _pl(J, CL2)
_need(p["seams"][0]["ao"] == -10 and any("sound's cut" in w for w in p["warnings"]),
      "J1 a J-cut is clamped to the next clip's material before its a (10 frames) -- named")
p = _pl([dict(base[0]), dict(base[1], ao=500)], CL2)
_need(p["seams"][0]["ao"] == 38, "J1 an L-cut is clamped to the first clip's material after its b (96 - 58)")
p = _pl([dict(base[0], b=58), dict(base[1], a=40, b=46, ao=500)], CL2)
_need(p["seams"][0]["a_at"] <= p["length"] - E.MIN_SEG, "J1 the sound's cut stays MIN_SEG before the film's end")
sp = [{"k": 1, "a": 10, "b": 58, "speed": 2.0}, {"k": 2, "a": 20, "b": 58, "speed": 0.5, "ao": -100}]
p = _pl(sp, CL2)
_need(p["seams"][0]["ao"] == -min(40, 24 - E.MIN_SEG), "J1 at a speed the J-limit is a / speed frames (and the piece before keeps MIN_SEG)")
ce = E.clean_edit({"pieces": {"scenes": [dict(base[0], ao=0), dict(base[1], ao=7.4)]}})
_need("ao" not in ce["pieces"]["scenes"][0] and ce["pieces"]["scenes"][1]["ao"] == 7, "J1 clean_edit: an int ao, a zero one dropped")

# ------------------------------------------------------------------ J2
p0 = E.plan([2.0, 4.0], 6.0, [{"frames": 56}, {"frames": 56}, {"frames": 73}],
            [{"kind": DIS, "seconds": 0.5}, {"kind": "cut"}], K)
same = all(a["film_in"] == s["film_in"] and a["film_out"] == s["film_out"] and a["src_in"] == s["src_in"]
           and a["src_out"] == s["src_out"] and a["pad_head"] == s["hold_head"] and a["pad_tail"] == s["hold_tail"]
           for a, s in zip(p0["asegments"], p0["segments"]))
_need(same and len(p0["asegments"]) == len(p0["segments"]), "J2 without an offset the sound's segments ARE the picture's")
rng = random.Random(1038)
bad_len = bad_sync = 0
for _ in range(400):
    n = rng.randint(2, 4)
    fr = [rng.choice([56, 73, 90, 124]) for _ in range(n)]
    pcs = []
    for k in range(n):
        a = rng.randint(0, 20)
        b = rng.randint(a + 10, fr[k])
        pc = {"k": k + 1, "a": a, "b": b, "speed": rng.choice([1.0, 1.0, 0.5, 1.5, 2.0])}
        if k:
            pc["ao"] = rng.randint(-40, 40)
            if rng.random() < 0.5:
                pc["tr"] = {"kind": DIS, "seconds": rng.choice([0.25, 0.5])}
        pcs.append(pc)
    starts_ = [2.0 * (k + 1) for k in range(n - 1)]
    q = E.plan(starts_, 6.0, [{"frames": f} for f in fr], [{"kind": "cut"}] * (n - 1), K, edit={"pieces": {"scenes": pcs}})
    tot = sum(x["film_out"] - x["film_in"] for x in q["asegments"]) - sum(sm["d"] for sm in q["seams"] if sm["d"] >= 2 and sm["xfade"])
    vt = sum(x["film_out"] - x["film_in"] for x in q["segments"]) - sum(sm["d"] for sm in q["seams"] if sm["d"] >= 2 and sm["xfade"])
    if tot != q["length"] or vt != q["length"]:
        bad_len += 1
    st0 = [0]
    for c in q["cuts"]:
        st0.append(c)
    for j, (pc, a) in enumerate(zip(q["pieces"], q["asegments"])):
        f = (a["film_in"] + a["film_out"]) / 2.0
        want = pc["a"] + (f - st0[j]) * pc["speed"]
        got = a["src_in"] + (f - a["film_in"] - a["pad_head"]) * pc["speed"]
        if a["pad_head"] == 0 and a["pad_tail"] == 0 and abs(want - got) > 1.0:
            bad_sync += 1
    # the sound's cuts keep their order and room
    A = [0] + [sm["a_at"] for sm in q["seams"]] + [q["length"]]
    if any(A[i + 1] - A[i] < E.MIN_SEG for i in range(len(A) - 1)):
        bad_len += 1
_need(bad_len == 0, "J2 over 400 edits the sound adds up to the film and its cuts keep their order (%d bad)" % bad_len)
_need(bad_sync == 0, "J2 every piece's sound keeps its picture's source line -- sync (%d bad)" % bad_sync)

# ------------------------------------------------------------------ J3
ff = shutil.which("ffmpeg")
try:
    import numpy as np
except Exception:
    np = None
if not ff or not shutil.which("ffprobe") or np is None:
    print("  note  ffmpeg/ffprobe/numpy missing -- J3 SKIPPED")
else:
    tmp = Path(tempfile.mkdtemp(prefix="pls1038_"))
    for f, col in ((440, "red"), (880, "blue")):
        subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "color=c=%s:s=96x54:r=24:d=4" % col, "-f", "lavfi", "-i",
                        "sine=frequency=%d:sample_rate=48000:duration=4" % f, "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", str(tmp / ("c%d.mp4" % f))], check=True)

    def _freq(path, t0, t1):
        raw = subprocess.run([ff, "-v", "error", "-i", str(path), "-ss", str(t0), "-to", str(t1), "-ac", "1", "-ar", "48000", "-f", "f32le", "-"],
                             capture_output=True).stdout
        x = np.frombuffer(raw, np.float32)
        s = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        return int(round(float(np.argmax(s) * 48000 / len(x)), -1))

    def _col(path, t):
        raw = subprocess.run([ff, "-v", "error", "-ss", str(t), "-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True).stdout
        a = np.frombuffer(raw, np.uint8).reshape(-1, 3).mean(0)
        return "red" if a[0] > a[2] else "blue"

    def _dur(path, st):
        return subprocess.run(["ffprobe", "-v", "error", "-select_streams", st, "-show_entries", "stream=duration", "-of", "csv=p=0", str(path)],
                              capture_output=True, text=True).stdout.strip()

    res = {}
    for name, pcs in (("L", L), ("J", J)):
        q = _pl(pcs, CL2)
        out = tmp / ("o_%s.mp4" % name)
        r = subprocess.run(E.ffmpeg_command(ff, [str(tmp / "c440.mp4"), str(tmp / "c880.mp4")], str(out), q, True, preview=True),
                           capture_output=True, text=True)
        res[name] = (r.returncode, out)
    ok_l = res["L"][0] == 0 and _col(res["L"][1], 2.25) == "blue" and _freq(res["L"][1], 2.1, 2.4) == 440 and _freq(res["L"][1], 2.6, 2.9) == 880
    ok_j = res["J"][0] == 0 and _col(res["J"][1], 1.8) == "red" and _freq(res["J"][1], 1.65, 1.95) == 880 and _freq(res["J"][1], 1.0, 1.4) == 440
    _need(ok_l, "J3 L-cut: the second picture from 2.0 s, the first clip's tone until 2.5 s (measured)")
    _need(ok_j, "J3 J-cut: the second clip's tone from 1.58 s under the first picture (measured)")
    _need(all(_dur(o, "v:0") == _dur(o, "a:0") == "4.000000" for _, o in res.values()), "J3 picture and sound end together (4.000 s)")
    shutil.rmtree(tmp, ignore_errors=True)
# ao = 0: the graph's sound part is the v1037 one (the picture's windows)
g0 = E.filter_graph(p0, True)[0]
_need("atrim=start=0.250000" in g0 or "atrim=start=" in g0, "J3 the sound graph builds without an offset")
seg_based = p0["segments"]
old_trim = ["atrim=start=%.6f:end=%.6f" % (int(s["src_in"]) / 24.0, max(int(s["src_in"]) + 1, int(-(-s["src_out"] // 1))) / 24.0)
            for s in seg_based]
_need(all(t in g0 for t in old_trim), "J3 without an offset the sound cuts the same source windows as v1037")

# ------------------------------------------------------------------ J4
if shutil.which("node") is None:
    print("  note  node.js not found -- J4 SKIPPED")
else:
    cases = []
    rng = random.Random(7)
    for _ in range(300):
        n = rng.randint(2, 4)
        fr = [rng.choice([56, 73, 90, 124]) for _ in range(n)]
        pcs = []
        for k in range(n):
            a = rng.randint(0, 20)
            b = rng.randint(a + 10, fr[k])
            pc = {"k": k + 1, "a": a, "b": b, "speed": rng.choice([1.0, 0.5, 1.5, 2.0])}
            if k:
                pc["ao"] = rng.randint(-40, 40)
                if rng.random() < 0.5:
                    pc["tr"] = {"kind": DIS, "seconds": rng.choice([0.25, 0.5])}
            pcs.append(pc)
        q = E.plan([2.0 * (k + 1) for k in range(n - 1)], 6.0, [{"frames": f} for f in fr], [{"kind": "cut"}] * (n - 1), K,
                   edit={"pieces": {"scenes": pcs}})
        cases.append({"pieces": q["pieces"], "d": [sm["d"] for sm in q["seams"]], "frames": {str(k + 1): fr[k] for k in range(n)},
                      "a_at": [sm["a_at"] for sm in q["seams"]]})
    Lp = _pl(L, CL2)
    t2 = Path(tempfile.mkdtemp(prefix="pls1038js_"))
    body = (js.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (t2 / "m.mjs").write_text(body, encoding="utf-8")
    (t2 / "cases.json").write_text(json.dumps({"cases": cases, "Lplan": {"plan": Lp}}), encoding="utf-8")
    (t2 / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
import { readFileSync } from "fs";
const X = JSON.parse(readFileSync(new URL("./cases.json", import.meta.url)));
const out = {};
let bad = 0;
for (const c of X.cases) {
    const fK = (k) => c.frames[String(k)] ?? Infinity;
    const got = T.soundCuts(c.pieces.map((p) => Object.assign({}, p)), c.d, fK);
    if (JSON.stringify(got) !== JSON.stringify(c.a_at)) bad++;
}
out.parity_bad = bad;
// setSoundCut / trimCut on the L case
const P = X.Lplan.plan, pcs = P.pieces, d = P.seams.map((s) => s.d), fK = (k) => P.frames[String(k)];
const s1 = T.setSoundCut(pcs, 1, 1000, d, fK);
out.set_clamped = s1[1].ao;
const s0 = T.setSoundCut(pcs, 1, T.endsOf(pcs)[0], d, fK);
out.set_zero_drops = !("ao" in s0[1]);
const lk = T.trimCut(pcs, 1, T.endsOf(pcs)[0] + 6, fK, true, d);
out.linked_keeps_ao = lk[1].ao;
const ul = T.trimCut(pcs, 1, T.endsOf(pcs)[0] + 6, fK, false, d);
out.unlinked_sound_stays = T.soundCuts(ul, d, fK)[0];
out.linked_sound_moves = T.soundCuts(lk, d, fK)[0];
// the sound's mix on the L plan: at 2.25 s the first clip is heard, the picture is the second
out.amix = T.audioMixAt(X.Lplan, 2.25).parts;
out.pmix = T.mixAt(X.Lplan, 2.25).parts;
out.asrc = T.asrcTime(P.asegments[0], 2.25);
// tiles: aspect kept, box covered, nearest tile
const tl = T.tilePlan(500, 90, 16 / 9, 24, 4, (x) => x / 500 * 96);
out.tiles = { n: tl.length, tw: tl[0].tw, last: tl[tl.length - 1], cover: tl.reduce((a, t) => a + t.w, 0), idx: tl.map((t) => t.idx) };
out.snap = [T.snapHead(1.97, [0, 2, 4], 0.05), T.snapHead(1.9, [0, 2, 4], 0.05)];
const nd = (v) => ({ properties: v === undefined ? {} : { pls_cine_lanes: v } });
out.lanes = [T.laneHeights(nd()), T.laneHeights(nd({ clip: 999, audio: 5 })), T.laneExtra(nd({ clip: 100, audio: 40 }))];
const lnk = (v) => ({ properties: v === undefined ? {} : { pls_cine_linked: v } });
out.linked = [T.isLinked(lnk()), T.isLinked(lnk(false)), T.linkedFor(lnk(), true), T.linkedFor(lnk(false), true), T.linkedFor(lnk(), false)];
out.speed = [T.speedLabel(1), T.speedLabel(1.5), T.speedLabel(0.25)];
console.log(JSON.stringify(out));
""", encoding="utf-8")
    r = subprocess.run(["node", str(t2 / "h.mjs")], capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        _fail("J4 harness failed: %s" % r.stderr.strip()[-400:])
    else:
        o = json.loads(r.stdout.strip().splitlines()[-1])
        _need(o["parity_bad"] == 0, "J4 aoRange/soundCuts == Python over 300 random edits (%d differ)" % o["parity_bad"])
        _need(o["set_clamped"] == 38 and o["set_zero_drops"], "J4 setSoundCut clamps to the material; back on the picture's cut drops the key")
        _need(o["linked_keeps_ao"] == 12 and o["linked_sound_moves"] == 66 and o["unlinked_sound_stays"] == 60,
              "J4 trimCut: linked the sound's cut goes along (66), unlinked it stays (60)")
        _need([x["i"] for x in o["amix"]] == [0] and [x["i"] for x in o["pmix"]] == [1],
              "J4 at 2.25 s of the L-cut: the second picture, the first clip's sound")
        _need(abs(o["asrc"] - (58 + 6 + 0.5) / 24.0) < 1e-6, "J4 asrcTime: the sound runs on in its own clip (source 64)")
        t = o["tiles"]
        _need(abs(t["tw"] - 160) < 1e-9 and t["n"] == 4 and abs(t["cover"] - 500) < 1e-6 and t["last"]["w"] == 20
              and t["idx"] == sorted(t["idx"]) and max(t["idx"]) <= 23, "J4 tilePlan: 90 px high = 160 px wide (16:9), the box covered, tiles in order")
        _need(o["snap"] == [2, 1.9], "J4 snapHead: within the tolerance onto the cut, outside free")
        # AMENDED v1153: the sound lane is 32 (the waveform readable)
        _need(o["lanes"][0] == {"clip": 44, "audio": 32} and o["lanes"][1] == {"clip": 240, "audio": 32} and o["lanes"][2] == 64,
              "J4 laneHeights clamped (44..240 / 32..120), laneExtra = what they add")
        _need(o["linked"] == [True, False, False, True, True], "J4 linked by default; Alt flips it for one drag")
        _need(o["speed"] == ["100%", "150%", "25%"], "J4 the speed in percent, as editors show it")
    shutil.rmtree(t2, ignore_errors=True)

# ------------------------------------------------------------------ S
cutb = js[js.index("function _cutButtons("):js.index("function _linkButton(")]
# the UI's words (labels, titles), not the comments -- a guard must not punish its own documentation
_need('mk("\\u2702 cut", () => _split(node)' in js and '"\\u2702 split"' not in js and not any("split" in lit for lit in re.findall(r'"([^"\\]*)"', cutb))
      and '"heal the cut at the playhead"' in js and '"split the selected shot' not in js,
      "S one word: the button, its title and heal say cut")
_need('headGrip.addEventListener("pointerdown", (ev) => _scrub(node, ev));' in js
      and 'ruler.addEventListener("pointerdown", (ev) => _scrub(node, ev));' in js, "S the playhead has a grip; the ruler scrubs")
scrub = js[js.index("function _scrub("):js.index("function _playerToggle(")]
_need("snapHead(t, _headTargets(st)" in scrub and "if (!e.shiftKey)" in scrub and "if (pl.playing) _playerToggle(node);" in scrub
      and "_playerSeek(node, t);" in scrub and 'addEventListener("pointermove", mv)' in scrub
      and "if (e.buttons === 0) { up(); return; }" in scrub and "try { el.setPointerCapture(ev.pointerId); } catch" in scrub,
      "S scrubbing: the player follows every move, snaps to the cuts, Shift = free, dragging pauses")
sync = js[js.index("function _syncChain("):js.index("function _playerSeek(")]   # v1044: per layer
_need("audioMixAt({ plan: P }, tt)" in sync and "v.volume = Math.max(0, Math.min(1, snd.get(i) || 0));" in sync and "asrcTime(AS[i], tt)" in sync,
      "S the player: picture and sound mixed apart (a J-/L-cut is heard live)")
lane = js[js.index("function _clipLane("):js.index("function _cutButtons(")]
# v1039 RE-GROUNDED: the grips moved onto the cut layer (same functions, same drags)
_need('part(0, picH, at, col, "cutgrip")' in lane and "trimCut(pcs, j + 1, f, fK, linkedFor(node, alt), dArr)" in lane,
      "S each cut is a line with a grip; the drag honours A/V linked (Alt flips)")
_need('part(sndTop, sndH, aat, ao ? "#ffcc80" : col, "soundcut")' in lane and "setSoundCut(pcs, j + 1, f, dArr, fK)" in lane and "trimCut(pcs, j + 1, f - ao, fK, true, dArr)" in lane,
      "S the sound's cut has its own grip: unlinked = the sound alone, linked = both")
_need("rotate(45deg)" in lane and "#ffb300" in lane and "cursor:col-resize" in lane and "\\u2194" not in lane
      and "margin-left:-13px" not in lane, "S the speed anchor is a diamond on the line; the old knob is gone")
# v1039 RE-GROUNDED: the corner grips became the dividers in the gap below each lane
_need("_dividers(node);" in lane and 'd.dataset.pls = which === "clip" ? "clipheight" : "audioheight";' in lane
      and "const LH = laneHeights(node);" in js[js.index("export function lanesH("):], "S the lanes have height grips; the node grows with them")   # AMENDED v1150
_need("node.properties[LANE_KEY]" in js and "node.properties[LINK_KEY]" in js
      and "pls_cine_lanes" not in json.dumps(T.DEFAULT) and "pls_cine_linked" not in json.dumps(T.DEFAULT),
      "S lane heights and the link switch are view settings (node.properties), not the timeline JSON")
_need("STRIP_TILES, STRIP_H = 24, 96" in node_py and 'item["strip_tiles"]' in node_py and 'item["strip_step"]' in node_py
      and "scale=-2:%d,tile=%dx1" in node_py and "tilePlan(boxW, th, c.w / c.h, c.strip_tiles" in lane and "th = segH - 2" in lane,
      "S the strip: 24 pictures at 96 px, laid out in their aspect")
_need('q = asegs[i]' in edit_py and "the sound makes %d frames, the picture %d" in edit_py, "S the export's sound comes from the sound's segments and ends with the picture")
# the banner moves with every build: v1038 or later
_need(int(re.search(r'ph_cine_timeline\.js v(\d+) loaded', js).group(1)) >= 1038, "S the banner says v1038 or later")
nonascii = [f for f in ("web/js/ph_cine_timeline.js", "nodes/cine_edit.py", "nodes/ph_cine_timeline.py", "tests/test_v1038_cutting_desk.py")
            if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not nonascii, "S ASCII only (%s)" % ", ".join(nonascii))
_need("position:fixed" not in js and "window.open" not in js, "S no floating window")

if FAILS:
    print("test_v1038_cutting_desk: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1038_cutting_desk: PASS (J1-J4, S)")
