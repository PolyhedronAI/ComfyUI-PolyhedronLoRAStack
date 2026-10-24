# -*- coding: ascii -*-
"""Guard v1211 -- cuts are held, not dissolved; the pin zone never copies.

(Frank 22.10.: the fox T2V run doubled to 48 fps painted a ghost dissolve on both cuts -- cut_guard was 0. "ich will
jetzt nicht noch zig-mal den Fuchs t2v-Workflow laufen lassen ... go". And the parked Media Loader card: "load from
where it is" must never copy into input.)

  K1 _cut_pairs on Frank's MEASURED fox clip (141 frames, 24 fps, RGB MAE 0-1 per pair, cuts asked at 1.75 s and
     3.50 s): holds exactly pairs 41 and 85 -- the second one 2 frames after the asked time, where the model painted it
  K2 fast motion without a cut (a whole window at 0.05) holds nothing; the default threshold 0.1 leaves it alone too
  K3 a painted dissolve (a ramp, no single jump) holds nothing; a flat still holds nothing (floor)
  K4 a cut time outside the clip is skipped and said; no cuts / no fps -> nothing
  K5 the timeline: cut_seconds takes the shots after the first inside the clip; the node's 12th output `cuts` =
     {seconds, fps, frames}; the eleven outputs before it unchanged
  K6 the Interpolate: cuts pin appended after video, cut_guard default 0.1, held pairs overwrite the verdicts,
     the findings reach the readout
  K7 the Media Loader: the pin branch ends in the picker and returns BEFORE the upload; no message says it copies
Script-style: exit 0 = pass.
"""
import ast
import importlib
import json
import os
import re
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


SRC = ROOT / "nodes" / "ph_interpolate.py"
TREE = ast.parse(SRC.read_text(encoding="utf-8"))


def lift(names):
    want, body, found = set(names), [], set()
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            body.append(node); found.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in want:
                    body.append(node); found.add(t.id)
    assert not want - found, sorted(want - found)
    ns = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(SRC), "exec"), ns)
    return ns


ns = lift(["_cut_pairs", "_gate_verdict", "CUT_WINDOW_S", "CUT_RATIO", "CUT_FLOOR", "CUT_GUARD_DEFAULT"])
cut_pairs, gate = ns["_cut_pairs"], ns["_gate_verdict"]

# ---- K1 Frank's fox, measured 22.10. on PH_134546_FOX_T2V_UPSCALE_BAK_00001_.mp4
FOX = [0.006, 0.0008, 0.0006, 0.0009, 0.0039, 0.0008, 0.0009, 0.0016, 0.0041, 0.0019, 0.0023, 0.0029, 0.0043, 0.0023, 0.0023, 0.0025, 0.0047, 0.0052, 0.0026, 0.0026, 0.0033, 0.0049, 0.002, 0.0019, 0.0022, 0.0033, 0.0016, 0.0014, 0.0017, 0.0022, 0.0012, 0.0012, 0.0016, 0.0031, 0.0037, 0.0035, 0.0035, 0.0032, 0.0084, 0.0043, 0.0067, 0.2075, 0.0133, 0.0073, 0.0068, 0.0071, 0.0101, 0.0072, 0.0085, 0.0083, 0.0131, 0.024, 0.0183, 0.0174, 0.0147, 0.0146, 0.0103, 0.0096, 0.0084, 0.011, 0.0094, 0.0102, 0.0095, 0.0094, 0.0079, 0.0087, 0.0081, 0.0084, 0.0086, 0.0086, 0.008, 0.0068, 0.0094, 0.0074, 0.0086, 0.0092, 0.0082, 0.0082, 0.0085, 0.008, 0.0079, 0.007, 0.0088, 0.0095, 0.01, 0.2094, 0.0085, 0.0077, 0.009, 0.0107, 0.005, 0.0054, 0.0074, 0.0101, 0.006, 0.0053, 0.0071, 0.01, 0.0076, 0.012, 0.0093, 0.0098, 0.0089, 0.0064, 0.0065, 0.0048, 0.0079, 0.0038, 0.0042, 0.0026, 0.0052, 0.0023, 0.0033, 0.0034, 0.0051, 0.0024, 0.0034, 0.0032, 0.0069, 0.0049, 0.0051, 0.0053, 0.0054, 0.0071, 0.0032, 0.0022, 0.0033, 0.0051, 0.0025, 0.0019, 0.0033, 0.0042, 0.0024, 0.0026, 0.0043, 0.0063, 0.0049, 0.003, 0.0026, 0.0021]
held, notes = cut_pairs(FOX, [1.75, 3.5], 24.0)
_need(len(FOX) == 140 and held == {41, 85}, "K1 fox: cuts asked at 1.75 s / 3.50 s held at pairs 41|42 and 85|86 (%s)" % sorted(held))
_need(len(notes) == 2 and "frames 42|43" in notes[0] and "frames 86|87" in notes[1] and all("held" in n for n in notes),
      "K1 the readout names where each cut was found")
_need(not [i for i, m in enumerate(FOX) if gate(m, 0.0, 0.1) == "hold" and i not in (41, 85)]
      and all(gate(FOX[i], 0.0, 0.1) == "hold" for i in (41, 85)),
      "K1 the default threshold 0.1 holds the same two pairs on the fox and nothing else")

# ---- K2 fast motion
pan = [0.05] * 60
h2, n2 = cut_pairs(pan, [1.0], 24.0)
_need(h2 == set() and "no hard cut" in n2[0], "K2 a whip pan at 0.05 everywhere: no hold near the asked cut")
_need(gate(0.05, 0.0, ns["CUT_GUARD_DEFAULT"]) == "interp" and ns["CUT_GUARD_DEFAULT"] == 0.1,
      "K2 the default threshold leaves 0.05 motion alone")

# ---- K3 dissolve / still
ramp = [0.004] * 20 + [0.012, 0.018, 0.022, 0.024, 0.022, 0.018, 0.012] + [0.004] * 20
h3, _n3 = cut_pairs(ramp, [1.0], 24.0)
still = [0.0005] * 40
still[23] = 0.003
h3b, _ = cut_pairs(still, [1.0], 24.0)
_need(h3 == set() and h3b == set(), "K3 a painted dissolve and a still's grain: nothing held")
spike = [0.004] * 47
spike[25] = 0.09
h3c, _ = cut_pairs(spike, [1.0], 24.0)
_need(h3c == {25}, "K3 a soft cut (0.09, under the 0.1 threshold) next to calm frames IS found by its neighbourhood")

# ---- K4 edges
h4, n4 = cut_pairs(FOX, [9.0], 24.0)
_need(h4 == set() and "outside" in n4[0], "K4 a cut beyond the clip is skipped and said")
_need(cut_pairs(FOX, [], 24.0) == (set(), []) and cut_pairs(FOX, [1.75], 0) == (set(), [])
      and cut_pairs([], [1.75], 24.0) == (set(), []), "K4 no cuts / no fps / no pairs -> nothing")

# ---- K5 the timeline
import torch  # noqa: E402,F401  (the node package imports it)
fp = types.ModuleType("folder_paths")
fp.get_temp_directory = lambda: tempfile.gettempdir()
fp.get_output_directory = lambda: tempfile.gettempdir()
fp.get_input_directory = lambda: tempfile.gettempdir()
sys.modules.setdefault("folder_paths", fp)
PKG = "plscuts"
pkg = types.ModuleType(PKG)
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules[PKG] = pkg
TL = importlib.import_module(PKG + ".ph_cine_timeline")
_need(TL.cut_seconds({"shots": [{"at": 0}, {"at": 3.5}, {"at": 1.75}, {"at": 9.0}, {"at": "x"}]}, 141) == [1.75, 3.5],
      "K5 cut_seconds: shots after the first, inside the clip, sorted")
_need(TL.ULSCineTimeline.RETURN_NAMES == ("prompt", "first_frame", "last_frame", "anchors", "seconds", "frames", "info",
                                          "clip_prefix", "scene", "tail", "tail_audio", "cuts")
      and TL.ULSCineTimeline.RETURN_TYPES[-1] == "PLS_CUTS" and len(TL.ULSCineTimeline.RETURN_TYPES) == 12,
      "K5 cuts appended as output 12, the eleven before unchanged")
model = {"v": 1, "seconds": 5.875, "shots": [{"at": 0, "cut": "the camera cuts to", "text": "a fox in the snow"},
                                            {"at": 1.75, "cut": "the camera cuts to", "text": "a close-up of the fox"},
                                            {"at": 3.5, "cut": "the camera cuts to", "text": "the fox trots away"}],
         "anchors": [], "soundscape": "wind", "music": "N/A", "render": "whole", "active": 1, "project": "guard_cuts",
         "transitions": [], "carry_mode": "pin", "target": "fl2va"}
res = TL.ULSCineTimeline().build(json.dumps(model))["result"]
_need(len(res) == 12 and res[11] == {"seconds": [1.75, 3.5], "fps": 24.0, "frames": res[5]} and res[5] == 141,
      "K5 the node hands out {seconds [1.75, 3.5], fps 24, frames 141}")

# ---- K6 the Interpolate (static -- the node needs the model to run)
txt = SRC.read_text(encoding="utf-8")
opt = re.search(r'"optional": \{(.*?)\n            \},', txt, re.S).group(1)
_need(opt.index('"frames"') < opt.index('"video"') < opt.index('"cuts": ("PLS_CUTS"'),
      "K6 cuts pin APPENDED after frames and video")
_need('"default": CUT_GUARD_DEFAULT' in txt and "cut_guard=CUT_GUARD_DEFAULT" in txt and "video=None, cuts=None)" in txt,
      "K6 cut_guard default 0.1 in the widget and the signature; cuts in the signature")
# AMENDED v1222: the decisions moved into the pure _verdicts() (a wired pin switches the threshold off)
body = txt[txt.index("def _verdicts("):txt.index("return verdicts, notes, not wired")]
_need("held, notes = _cut_pairs(maes, cuts.get(\"seconds\") or [], float(src_fps))" in body and 'verdicts[i] = "hold"' in body
      and "verdicts, cut_notes, _theta_on = _verdicts(maes, static_skip, cut_guard, cuts, float(source_fps))" in txt,
      "K6 the found pairs overwrite the verdicts")
_need('info += " | " + "; ".join(cut_notes)' in txt, "K6 the findings reach interp_info")

# ---- K7 the Media Loader pin zone
js = (ROOT / "web" / "js" / "ph_media_loader.js").read_text(encoding="utf-8")
od = js[js.index("    async _onDrop(ev, intent) {"):js.index("    // v1211: the pin zone's way out")]
pin_exit = od.index("        if (wantPin) {\n")
upload = od.index('"/uls/media/upload?folder="')
tail = od[pin_exit:upload]
_need(pin_exit < upload and "this._pinPick();" in tail and "return;" in tail.split("this._pinPick();", 1)[1][:40],
      "K7 every pin drop ends in the picker and returns before the upload")
_need("copying into" not in js and "a copy goes to input instead" not in js and "copies into input instead" not in js,
      "K7 no message promises a copy for the pin zone any more")
pp = js[js.index("    _pinPick() {"):js.index("    _dropSelect(name, tries = 0) {")]
_need("openFolderPicker(this.folder" in pp and "pushRecentFolder(p)" in pp and "upload" not in pp,
      "K7 _pinPick opens the picker (native dialog inside), remembers the folder, uploads nothing")

print("test_v1211_cuts: %s" % ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
sys.exit(1 if FAILS else 0)
