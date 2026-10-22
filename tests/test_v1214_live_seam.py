# -*- coding: ascii -*-
"""Guard v1214 -- the live monitor ends a piece on its last frame, never on the one after the cut.

Frank 22.10. (CONTINUE fox, scene 2 continues scene 1 by motion): "Ich bekomme bei der Live-Wiedergabe in der Timeline
einen 'Hacker', wo sich die Frames ueberlagern ... Das ist im Export dann wieder bereinigt".
Measured in the browser (two barcoded clips, scene 2 repeats 22 frames of scene 1): the plan was right (scene 2 from
source frame 22), but the monitor's clock ran half a frame behind the video it reads -- at the cut scene 1 went on to
source frame 108 (the grid overhang, outside the film) in 4 of 5 runs before scene 2 took over at its frame 22, which
continues frame 107: the same moment twice. With the fix: 0 of 7, also at CPU throttle 4.

  C1 masterTime counts the frame the video shows: a video at the START of a frame is on that frame (the old -0.5
     put it on the frame before); mid-frame stays on it; at src_out the film time is the piece's film_out (the next
     piece shows); paused / loading: no clock
  C2 pastOut: a playing video at or past its piece's src_out is past its material; before it, paused, or no video: not
  C3 the monitor keeps its last picture while the one piece on screen is past its material
  S  ASCII
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1214_live_seam"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)


JS = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
node = shutil.which("node")
if node:
    D = tempfile.mkdtemp(prefix="v1214_")
    mod = os.path.join(D, "tl.mjs")
    open(mod, "w").write(JS.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
                         .replace('import { api } from "../../scripts/api.js";', "const api = {};")
                         .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    h = os.path.join(D, "h.mjs")
    open(h, "w").write(
        "const m = await import(process.argv[2]);\n"
        "const sg = { film_in: 0, film_out: 108, src_in: 0, src_out: 108, speed: 1, hold_head: 0 };\n"
        "const sg2 = { film_in: 108, film_out: 216, src_in: 22, src_out: 130, speed: 1, hold_head: 0 };\n"
        "const play = (ct, x) => Object.assign({ paused: false, seeking: false, readyState: 4, currentTime: ct }, x || {});\n"
        "const plan = { plan: { segments: [sg, sg2], seams: [{ at: 108, h1: 0, h2: 0, d: 0 }] } };\n"
        "const tEnd = m.masterTime(sg, play(108 / 24));\n"
        "console.log(JSON.stringify({\n"
        "  start: m.masterTime(sg, play(107 / 24)), mid: m.masterTime(sg, play(107.5 / 24)), end: tEnd,\n"
        "  endShows: m.mixAt(plan, tEnd).parts.map((q) => q.i), s2: m.masterTime(sg2, play(23 / 24)),\n"
        "  paused: m.masterTime(sg, play(1, { paused: true })), loading: m.masterTime(sg, play(1, { readyState: 2 })),\n"
        "  past: m.pastOut(sg, play(108 / 24)), past2: m.pastOut(sg, play(110 / 24)), inside: m.pastOut(sg, play(107.9 / 24)),\n"
        "  pausedPast: m.pastOut(sg, play(110 / 24, { paused: true })), none: m.pastOut(sg, null) }));\n")
    r = subprocess.run([node, h, "file://" + mod], capture_output=True, text=True, timeout=60)
    try:
        J = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        J = None
        print(r.stdout[-1500:], r.stderr[-1500:])
    shutil.rmtree(D, ignore_errors=True)
    f = lambda t: None if t is None else int(t * 24 + 1e-9)
    _need(J is not None and f(J["start"]) == 107 and f(J["mid"]) == 107,
          "C1 a video at the start (or middle) of source frame 107 is on film frame 107 (got %s / %s)"
          % (J and f(J["start"]), J and f(J["mid"])))
    _need(J is not None and abs(J["end"] - 108 / 24.0) < 1e-9 and J["endShows"] == [1],
          "C1 at src_out the film time is film_out -- the next piece shows")
    _need(J is not None and f(J["s2"]) == 109, "C1 a piece with src_in 22: source frame 23 is film frame 109")
    _need(J is not None and J["paused"] is None and J["loading"] is None, "C1 paused / loading: no clock")
    _need(J is not None and J["past"] and J["past2"] and not J["inside"] and not J["pausedPast"] and not J["none"],
          "C2 pastOut: at / past src_out while playing; not inside, not paused, not without a video")
else:
    _need(False, "node.js is needed for the player's pure parts")

_need("(v.currentTime * FPS - sg.src_in) / sp) / FPS;" in JS and "v.currentTime * FPS - 0.5 - sg.src_in" not in JS,
      "C1 the clock without the half-frame lag")
_need("|| (pv.length === 1 && pastOut(pv[0].sg, pv[0].v))) { _playerLabel(node); return; }" in JS,
      "C3 the monitor keeps its last picture while the piece on screen is past its material")

me = Path(__file__).read_bytes()
_need(all(b < 128 for b in me), "S this guard is ASCII")

if FAILS:
    print("%s: %d FAIL" % (NAME, len(FAILS)))
    for x in FAILS:
        print("  -", x)
    sys.exit(1)
print("%s: all green" % NAME)
