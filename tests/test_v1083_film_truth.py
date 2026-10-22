# -*- coding: ascii -*-
"""Guard v1083 -- the film's last frame is ONE truth (Timeline audit of
10.10., block A: the untested image-to-video / carry path).

The grid lifts a render past the asked length (6.0 s = 144 frames -> 158).
Measured on the table: the carry took the CLIP's last frame while the film
cut 13 frames earlier (a visible jump at the seam), the last keyframe sat in
the overhang the film never showed, whether an anchor near a scene's end was
keyframe or guide depended on the span lying on the grid, pin mode claimed a
carry it did not output, @clip:K:2.5 landed one frame off (seek), the
rounding of a half frame differed between Python and the browser.

  F1  asked_frames is half-up like the browser (158.5 -> 159, Python's
      round says 158); frames_for = the grid of that; the JS framesFor
      agrees on the same seconds (driven in node).
  F2  film_frames: the asked length, unless a shot or an anchor lands on or
      after it -> the grid (the overhang is shown); a grid-exact length is
      the grid.
  F3  scene_model lifts an anchor on a scene's last asked frame to the
      scene's LAST keyframe for every scene (span 2.5 s: 60 -> 73, an anchor
      at t1 - 1/24 is the last keyframe, not a guide at 59); an anchor in
      the middle stays a guide; an anchor at t1 of a non-last scene still
      belongs to the next scene.
  F4  plan(tails=...): a place in tails is shown to its clip's end (whole:
      158 of 158, else 144); a middle scene in tails lengthens the film and
      warns; without tails nothing changes.
  F5  run(): pin mode with an own first-frame anchor -> not carried, the
      line says so, no carry frame; keyframe mode as before; a carried run
      carries carry_index = film_last_index of the scene before.
  F6  frame_index(path, idx) picks frame idx by NUMBER (a clip whose grey
      equals the frame number: 59 and 60 differ, past the end = the last);
      build() reads the carry frame and @clip frames through it (text pins);
      frame_at (the seek) is gone.
  F7  the register wait: constants, the loop before "starts free", the
      console lines (text pins); edit_plan hands the tails to plan().
  S   ASCII.

Script-style: exit 0 = pass.
"""
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1083_film_truth"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)
    return c


pkg = types.ModuleType("plsn1083")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1083"] = pkg
T = importlib.import_module("plsn1083.cine_timeline")
E = importlib.import_module("plsn1083.cine_edit")
FPS = T.FPS

# ------------------------------------------------------------------ F1
_need(T.asked_frames(158.5 / FPS) == 159 and T.asked_frames(6.0) == 144 and T.frames_for(158.5 / FPS) == 175,
      "F1 asked_frames half-up (158.5 -> 159), frames_for the grid of it (175)")
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
tmpd = Path(tempfile.mkdtemp(prefix="v1083js_"))
i0 = js.find("export function alignFrames"); i1 = js.find("export function duration")
(tmpd / "h.mjs").write_text("const FPS = 24;\n" + js[i0:i1].replace("export ", "")
                            + "\nconsole.log(JSON.stringify([framesFor(158.5 / 24), framesFor(6.0), framesFor(2.5)]));\n", encoding="utf-8")
r = subprocess.run(["node", str(tmpd / "h.mjs")], capture_output=True, text=True, timeout=60)
shutil.rmtree(tmpd, ignore_errors=True)
got = json.loads(r.stdout.strip()) if r.returncode == 0 else None
_need(got == [T.frames_for(158.5 / FPS), T.frames_for(6.0), T.frames_for(2.5)] == [175, 158, 73],
      "F1 the browser's framesFor agrees frame for frame: %r" % (got,))

# ------------------------------------------------------------------ F2
m0 = {"seconds": 6.0, "shots": [{"at": 0, "text": "a"}], "anchors": []}
_need(T.film_frames(m0) == 144, "F2 nothing in the overhang: the asked 144 frames")
_need(T.film_frames(dict(m0, shots=[{"at": 0, "text": "a"}, {"at": 6.2, "text": "b"}])) == 158, "F2 a shot after the asked length: the grid (158)")
_need(T.film_frames(dict(m0, anchors=[{"src": "img", "at": 157 / FPS}])) == 158, "F2 a last-frame anchor: the grid")
_need(T.film_frames(dict(m0, anchors=[{"src": "img", "at": 143 / FPS}])) == 144, "F2 a guide on the asked last frame (143) is inside the asked length")
_need(T.film_frames(dict(m0, anchors=[{"src": "img", "at": 3.0}])) == 144, "F2 a guide in the middle: the asked length")
_need(T.film_frames(dict(m0, seconds=73 / FPS)) == 73, "F2 a grid-exact length: the grid")

# ------------------------------------------------------------------ F3
two = T.load(json.dumps({"seconds": 6.0, "shots": [{"at": 0, "text": "a"}, {"at": 2.5, "scene": True, "cut": "the camera cuts to", "text": "b"}],
                         "anchors": [{"src": "img", "at": 2.5 - 1 / FPS}, {"src": "img2", "at": 1.0}, {"src": "img3", "at": 2.5}]}))
s1 = T.scene_model(two, 1)
grid1 = T.frames_for(s1["seconds"])
_need(grid1 == 73 and T.asked_frames(s1["seconds"]) == 60, "F3 scene 1: 2.5 s = 60 asked, 73 on the grid")
ats = {a["src"]: a["at"] for a in s1["anchors"]}
_need(abs(ats.get("img", -1) - (grid1 - 1) / FPS) < 1e-9, "F3 the anchor at t1 - 1/24 is the scene's LAST keyframe (frame %d), not a guide: %r" % (grid1 - 1, ats.get("img")))
_need(abs(ats.get("img2", -1) - 1.0) < 1e-9, "F3 an anchor in the middle stays where it is")
_need("img3" not in ats, "F3 an anchor AT t1 of a non-last scene belongs to the next scene")
first, last, guides = T.split_anchors(s1, grid1)
_need(last is not None and last["src"] == "img" and [g[0] for g in guides] == ["img2"], "F3 split_anchors: img is the last keyframe, img2 a guide")
s2 = T.scene_model(two, 2)
_need(any(a["src"] == "img3" and abs(a["at"]) < 1e-9 for a in s2["anchors"]), "F3 scene 2 gets img3 on its frame 0")
_need(T.film_frames(s1) == 73 and T.film_frames(s2) == T.asked_frames(s2["seconds"]), "F3 scene 1 is shown to its grid (its last keyframe), scene 2 to its asked length")

# ------------------------------------------------------------------ F4
clips = [{"frames": 158, "fps": 24, "audio": False}]
P0 = E.plan([], 6.0, clips, [], T.TRANSITIONS, edit={}, whole=True)
P1 = E.plan([], 6.0, clips, [], T.TRANSITIONS, edit={}, whole=True, tails={0})
_need(P0["length"] == 144 and P1["length"] == 158, "F4 whole: 144 without tails, 158 with (got %s / %s)" % (P0["length"], P1["length"]))
two_clips = [{"frames": 73, "fps": 24, "audio": False}, {"frames": 90, "fps": 24, "audio": False}]
Q0 = E.plan([2.5], 6.0, two_clips, [], T.TRANSITIONS, edit={}, whole=False)
Q1 = E.plan([2.5], 6.0, two_clips, [], T.TRANSITIONS, edit={}, whole=False, tails={1})
_need(Q0["length"] == 144 and Q0["place_cuts"] == [60], "F4 scenes without tails: cut at 60, length 144 (%s %s)" % (Q0["length"], Q0["place_cuts"]))
_need(Q1["length"] == 157 and Q1["place_cuts"] == [73] and any("scene 1 is shown to its last frame" in w for w in Q1["warnings"]),
      "F4 scene 1 in tails: shown to 73, the film longer, a warning (%s %s %r)" % (Q1["length"], Q1["place_cuts"], Q1["warnings"]))
Q2 = E.plan([2.5], 6.0, [None, two_clips[1]], [], T.TRANSITIONS, edit={}, whole=False, tails={1})
_need(Q2["length"] == 144, "F4 a tail place with no clip: nothing to show longer")

# ------------------------------------------------------------------ F5 (+F6/F7 text pins) -- the node module
TMP = Path(tempfile.mkdtemp(prefix="v1083_"))
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
TL = importlib.import_module("plsn1083.ph_cine_timeline")
base = {"seconds": 6.0, "project": "fx", "render": "scene", "active": 2,
        "shots": [{"at": 0, "text": "a"}, {"at": 2.5, "scene": True, "cut": "the camera cuts to", "text": "b", "carry": True}]}
# pin mode + own first anchor on scene 2
m = T.load(json.dumps(dict(base, carry_mode="pin", anchors=[{"src": "img", "at": 2.5}])))
r, errs = TL.run(m, {"img": None}, "/x/fx/scene_1_00001_.mp4")
_need(not errs and r["carry"] is None and r.get("carry_index") is None and any("own first-frame anchor -- that one wins" in l for l in r["lines"]),
      "F5 pin mode + own first anchor: NOT carried, the line says so (%r)" % ([l for l in r["lines"] if "carry" in l],))
m = T.load(json.dumps(dict(base, carry_mode="pin")))
r, errs = TL.run(m, {}, "/x/fx/scene_1_00001_.mp4")
_need(not errs and r["carry"] == "/x/fx/scene_1_00001_.mp4" and r["carry_index"] == T.film_last_index(m, 1) == 59
      and any("frame the film shows LAST of scene 1 (frame 60 of" in l for l in r["lines"]),
      "F5 pin mode, no own anchor: carried on the film's last frame of scene 1 (index 59) (%r)" % ([l for l in r["lines"] if "carry" in l],))
m = T.load(json.dumps(dict(base, carry_mode="keyframe", anchors=[{"src": "img", "at": 2.5 - 1 / FPS}])))
r, errs = TL.run(m, {"img": None}, "/x/fx/scene_1_00001_.mp4")
_need(not errs and r["carry"] and r["carry_index"] == 72, "F5 scene 1 with a last keyframe is shown to its grid: the carry is frame 73 (index 72)")
m = T.load(json.dumps(dict(base, carry_mode="keyframe", anchors=[{"src": "img", "at": 2.5}])))
r, errs = TL.run(m, {"img": None}, "/x/fx/scene_1_00001_.mp4")
_need(not errs and r["carry"] is None and any("that one wins" in l for l in r["lines"]), "F5 keyframe mode + own first anchor: as before, not carried")

# ------------------------------------------------------------------ F6
ff = shutil.which("ffmpeg")
if ff:
    clip = TMP / "c.mp4"
    subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", "nullsrc=s=64x36:r=24,geq=lum='16+3*N':cb=128:cr=128", "-frames:v", "73",
                    "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", str(clip)], capture_output=True)
    g = lambda i: round(float(TL.frame_index(str(clip), i)[0, :, :, 0].mean() * 255))
    # yuv limited range: grey 16+3N decodes to about 3N*255/219 -- an ABSOLUTE value per frame number
    want = lambda i: round(3 * i * 255 / 219.0)
    got = {i: g(i) for i in (0, 20, 59, 60, 72)}
    _need(all(abs(got[i] - want(i)) <= 1 for i in got) and g(99) == got[72],
          "F6 frame_index picks by NUMBER: %r (expected about %r), past the end = the last" % (got, {i: want(i) for i in got}))
else:
    print("  note  ffmpeg missing -- F6 driven part skipped")
py = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
# AMENDED v1168: the film's index goes through the clip's own rate first (source_index) -- the neon film's seams
_need('sidx, sfps, sn = source_index(r["carry"], r["carry_index"])' in py and 'carry_img = frame_index(r["carry"], sidx)' in py,
      "F6 build() reads the carry frame by the film's index, turned into the clip's own frame (v1168)")
_need('idx = T.film_last_index(model, k) if which == "last" else (0 if which == "first" else T.asked_frames(which))' in py
      and "img = frame_index(path, sidx)" in py, "F6 @clip:K:last / :S go through frame_index (via source_index, v1168)")
_need("def frame_at(" not in py, "F6 the seek-based frame_at is gone")

# ------------------------------------------------------------------ F7
_need("REGISTER_WAIT_TRIES, REGISTER_WAIT_STEP = 6, 0.5" in py and "for _i in range(REGISTER_WAIT_TRIES):" in py
      and "prev = carry_source(model, k0)" in py[py.find("for _i in range(REGISTER_WAIT_TRIES):"):][:400]
      and "clip registered after" in py and "starts free" in py, "F7 build() waits for the run before's register, says what happened")
# AMENDED v1172: plan() also gets the frozen frames at the cuts (held=held) after the tails
_need("tails.add(k)" in py and "T.TRANSITIONS, edit=ed, whole=whole, tails=tails, held=held)" in py, "F7 edit_plan hands the tails to plan()")
shutil.rmtree(TMP, ignore_errors=True)

bad = [f for f in ("nodes/cine_timeline.py", "nodes/cine_edit.py", "nodes/ph_cine_timeline.py", "tests/test_v1083_film_truth.py")
       if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not bad, "S ASCII only %r" % bad)
if FAILS:
    print("%s: %d FAILURE(S)" % (NAME, len(FAILS)))
    sys.exit(1)
print("%s: PASS (F1-F7, S)" % NAME)
