# -*- coding: ascii -*-
"""Guard v1035 -- H3 transitions, written as text.

Frank, 29.09.: the wipe formula alone gave a hard cut (P1_D); the formula
PLUS a description of the image gave the blend (PH_103202: blur to white
peaks on the cut's frame, fade through black 3.71 .. 5.46 s). The timeline
now writes both from a choice at the cut.

  P1  parity: the presets, names, formulas and limits are the same in
      nodes/cine_timeline.py and web/js/ph_cine_timeline.js.
  P2  no H3 transition -> the prompt is byte-equal to v1034 (golden).
  P3  a preset sets the formula, puts "During the last <span> of the shot,
      <out>." at the END of the shot before and "<In> within <span>." behind
      the FIRST sentence of the shot after; the last shot gets no out.
  P4  the span in words (within / last), two decimals otherwise.
  P5  a shot that starts a scene ignores its h3t (whole run: note line;
      scene run: its first shot takes none).
  P6  validate names a bad kind, a length outside 0.25 .. 2 s, a custom
      formula that is not one of the list.
  P7  H3 custom: its own formula ("(none)" = no lead), its own sentences,
      an empty sentence writes nothing.
  P8  the editor's pure parts: an H3 choice clears the scene flag, an ffmpeg
      choice drops the h3t and makes a scene, "H3 cut" drops it; custom is
      prefilled from the preset it came from; the length is clamped; a
      custom formula outside the list is refused; h3tOf is null on a scene.
  S   statics: two groups in the choice, the formula select locked by a
      preset, the band on the handle, the custom row within the fixed edit
      height, the info line.

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


spec = importlib.util.spec_from_file_location("ct1035", str(ROOT / "nodes" / "cine_timeline.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")

FOX = [{"at": 0, "text": "A wide shot of a red fox in the snow. The fox lifts its head."},
       {"at": 2, "text": "an extreme close-up of the fox's face. Amber eyes, frost on the whiskers."},
       {"at": 4, "text": "a low tracking shot: the fox trots toward the lens"}]


def model(shots, **kw):
    d = {"seconds": 6, "shots": shots, "soundscape": "Wind."}
    d.update(kw)
    return T.load(json.dumps(d))


def body(m):
    p = T.compile_prompt(m)["prompt"]
    return p.split("integrated_multimodal_description: ", 1)[1].split("\n\noverall_soundscape", 1)[0]


# ------------------------------------------------------------------ P2
GOLD = ("integrated_multimodal_description: [Shot 1] Live-action, cinematic, A wide shot of a red fox in the snow. The fox lifts its head.\n\n"
        "[Shot 2] At 00:02.000, the camera cuts to an extreme close-up of the fox's face. Amber eyes, frost on the whiskers.\n\n"
        "[Shot 3] At 00:04.000, the camera cuts to a low tracking shot: the fox trots toward the lens\n\n"
        "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A")
_need(T.compile_prompt(model(FOX))["prompt"] == GOLD, "P2 no H3 transition: the prompt is byte-equal to v1034")
m = model([dict(FOX[0]), dict(FOX[1], h3t=None), dict(FOX[2])])
_need(T.validate(m) == [] and T.compile_prompt(m)["prompt"] == GOLD, "P2 h3t null = none (valid, same prompt)")

# ------------------------------------------------------------------ P3
sh = [dict(s) for s in FOX]
sh[1]["h3t"] = {"kind": "H3 blur to white", "seconds": 0.5}
sh[2]["h3t"] = {"kind": "H3 fade through black", "seconds": 1}
m = model(sh)
_need(T.validate(m) == [], "P3 a timeline with two presets is valid")
b = body(m).split("\n\n")
B, K = T.H3T_PRESETS["H3 blur to white"], T.H3T_PRESETS["H3 fade through black"]
_need(b[0] == "[Shot 1] Live-action, cinematic, A wide shot of a red fox in the snow. The fox lifts its head. "
      "During the last half second of the shot, " + B["out"] + ".", "P3 the OUT sentence ends the shot before")
_need(b[1] == "[Shot 2] At 00:02.000, the shot cross-dissolves to an extreme close-up of the fox's face. "
      + B["in"] + " within half a second. Amber eyes, frost on the whiskers. During the last second of the shot, " + K["out"] + ".",
      "P3 formula at the cut, IN behind the first sentence, then the next OUT: %r" % b[1][:120])
_need(b[2] == "[Shot 3] At 00:04.000, the shot fades to black, then fades in on a low tracking shot: the fox trots toward the lens. "
      + K["in"] + " within one second.", "P3 a shot without a closing mark gets one; the last shot takes no OUT")
_need("the camera cuts to" not in body(m), "P3 the preset's formula replaces the shot's cut phrase")
lines = T.compile_prompt(m)["lines"]
_need(any("<- H3 blur to white at the cut, 0.50 s" in l and "decides the exact length" in l for l in lines),
      "P3 the info line names the transition and says H3 decides the length")

# ------------------------------------------------------------------ P4
W = [T.span_words(x) for x in (0.25, 0.5, 1, 0.75, 1.5, 2)]
L = [T.span_words(x, last=True) for x in (0.25, 0.5, 1, 0.75)]
_need(W == ["a quarter of a second", "half a second", "one second", "0.75 seconds", "1.5 seconds", "2 seconds"]
      and L == ["quarter second", "half second", "second", "0.75 seconds"], "P4 the span in words %r %r" % (W, L))

# ------------------------------------------------------------------ P5
sh = [dict(s) for s in FOX]
sh[1].update(scene=True, h3t={"kind": "H3 blur to white", "seconds": 0.5})
m = model(sh)
_need(B["out"] not in body(m) and "the camera cuts to an extreme" in body(m)
      and any("starts a scene" in l and "is not used" in l for l in T.compile_prompt(m)["lines"]),
      "P5 whole run: a scene cut ignores its h3t and says so")
m2 = model(sh, render="scene", active=2)
tgt, k, n = T.resolve(m2)
_need(T.validate(m2) == [] and B["in"] not in body(tgt) and body(tgt).startswith("[Shot 1] Live-action, cinematic, an extreme close-up"),
      "P5 scene run: the scene's first shot takes no h3t")
m3 = model(sh, render="scene", active=1)
tgt, k, n = T.resolve(m3)
_need(B["out"] not in body(tgt), "P5 scene run: the scene before gets no OUT from the next scene's h3t")

# ------------------------------------------------------------------ P6
bad = [dict(s) for s in FOX]
bad[1]["h3t"] = {"kind": "H3 wipe", "seconds": 0.5}
bad[2]["h3t"] = {"kind": "H3 fade through black", "seconds": 3}
e = T.validate(model(bad))
_need(any("shot 2" in x and "H3 wipe" in x for x in e) and any("shot 3" in x and "outside 0.25" in x for x in e),
      "P6 a bad kind and a length outside 0.25 .. 2 s are named %r" % e)
bad[2]["h3t"] = {"kind": "H3 custom", "seconds": 0.5, "formula": "the shot wipes to"}
_need(any("formula" in x for x in T.validate(model(bad))), "P6 a custom formula outside the list is named")

# ------------------------------------------------------------------ P7
sh = [dict(s) for s in FOX]
sh[1]["h3t"] = {"kind": "H3 custom", "seconds": 0.75, "formula": "(none)", "out": "the snow thickens into a white wall", "in": ""}
bb = body(model(sh)).split("\n\n")
_need(bb[0].endswith("During the last 0.75 seconds of the shot, the snow thickens into a white wall.")
      and bb[1] == "[Shot 2] At 00:02.000, an extreme close-up of the fox's face. Amber eyes, frost on the whiskers.",
      "P7 custom: its sentence and formula ('(none)' = no lead); an empty IN writes nothing")

# ------------------------------------------------------------------ P1 + P8
if shutil.which("node") is None:
    print("  note  node.js not found -- P1/P8 SKIPPED")
else:
    t2 = Path(tempfile.mkdtemp(prefix="v1035js_"))
    bodyjs = (js.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
                .replace('import { api } from "../../scripts/api.js";', "const api = {};")
                .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (t2 / "m.mjs").write_text(bodyjs, encoding="utf-8")
    (t2 / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const o = {};
o.presets = T.H3T_PRESETS; o.names = T.H3T_NAMES; o.formulas = T.H3T_FORMULAS; o.lim = [T.H3T_MIN, T.H3T_MAX, T.H3T_DEFAULT];
const mk = () => T.parseTimeline(JSON.stringify({ seconds: 6, shots: [{ at: 0, text: "a" }, { at: 2, text: "b", scene: true }, { at: 4, text: "c", scene: true }],
    transitions: [{ kind: "dissolve", seconds: 1 }, { kind: "wipe left", seconds: 0.5 }] }));
let m = mk();
T.setCutChoice(m, 1, "H3 blur to white", 9);
o.a = { scene: !!m.shots[1].scene, h3t: m.shots[1].h3t, tr: m.transitions, choice: T.cutChoice(m, 1) };
T.setCutChoice(m, 1, "H3 custom", 0.1);
o.b = JSON.parse(JSON.stringify(m.shots[1].h3t));   // a snapshot: the next edits change the same object
T.setH3Custom(m, 1, "formula", "the shot wipes to"); o.c1 = m.shots[1].h3t.formula;
T.setH3Custom(m, 1, "out", "x"); o.c2 = m.shots[1].h3t.out;
T.setCutChoice(m, 1, "H3 fade through black", 0.5); T.setCutChoice(m, 1, "H3 custom", 0.5);
o.d = JSON.parse(JSON.stringify(m.shots[1].h3t));
T.setCutChoice(m, 1, "fade to white", 0.7);
o.e = { scene: !!m.shots[1].scene, h3t: m.shots[1].h3t || null, choice: T.cutChoice(m, 1) };
T.setCutChoice(m, 2, "H3 cross-dissolve", 0.5); T.setCutChoice(m, 2, T.H3_CUT, 0.5);
o.f = { scene: !!m.shots[2].scene, h3t: m.shots[2].h3t || null, choice: T.cutChoice(m, 2).kind };
const s = { scene: true, h3t: { kind: "H3 blur to white", seconds: 0.5 } };
o.g = [T.h3tOf(s), T.h3tOf({ h3t: { kind: "nope" } }), T.h3tOf({ h3t: { kind: "H3 blur to white", seconds: 1 } })];
console.log(JSON.stringify(o));
""", encoding="utf-8")
    r = subprocess.run(["node", str(t2 / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(t2, ignore_errors=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        _fail("P harness failed: %s" % r.stderr.strip()[-400:])
    else:
        d = json.loads(line[-1])
        _need(d["presets"] == T.H3T_PRESETS and d["names"] == list(T.H3T_NAMES) and d["formulas"] == list(T.H3T_FORMULAS)
              and d["lim"] == [T.H3T_MIN, T.H3T_MAX, T.H3T_DEFAULT], "P1 presets, names, formulas, limits: JS == Python")
        _need(d["a"]["scene"] is False and d["a"]["h3t"] == {"kind": "H3 blur to white", "seconds": 2}
              and d["a"]["tr"] == [{"kind": "wipe left", "seconds": 0.5}] and d["a"]["choice"] == {"kind": "H3 blur to white", "seconds": 2},
              "P8 an H3 choice clears the scene flag (the next boundary keeps its transition), length clamped to 2 %r" % d["a"])
        B = T.H3T_PRESETS["H3 blur to white"]
        _need(d["b"] == {"kind": "H3 custom", "seconds": 0.25, "formula": B["formula"], "out": B["out"], "in": B["in"]},
              "P8 custom is prefilled from the preset it came from, length clamped to 0.25 %r" % d["b"])
        _need(d["c1"] == B["formula"] and d["c2"] == "x", "P8 a custom formula outside the list is refused; a sentence is taken")
        K = T.H3T_PRESETS["H3 fade through black"]
        _need(d["d"]["formula"] == K["formula"] and d["d"]["out"] == K["out"], "P8 custom after another preset takes THAT preset's sentences")
        _need(d["e"]["scene"] is True and d["e"]["h3t"] is None and d["e"]["choice"] == {"kind": "fade to white", "seconds": 0.7},
              "P8 an ffmpeg choice drops the h3t and makes the cut a scene %r" % d["e"])
        _need(d["f"] == {"scene": False, "h3t": None, "choice": T.__dict__.get("H3_CUT", "H3 cut (inside the clip)")},
              "P8 'H3 cut' drops the h3t and the scene flag %r" % d["f"])
        _need(d["g"][0] is None and d["g"][1] is None and d["g"][2] == {"kind": "H3 blur to white", "seconds": 1},
              "P8 h3tOf: null on a scene cut and on an unknown kind")

# ------------------------------------------------------------------ S
# v1036 RE-GROUNDED: the handle offers only what goes INTO the run; the
# ffmpeg transitions moved to the seams of the clip lane
_need('grp("inside the clip -- H3 paints it (one run)", [H3_CUT].concat(H3T_NAMES));' in js
      and 'grp("between two clips' not in js and '"scene cut: " + c.kind + " (on the clip lane)"' in js,
      "S the handle's choice is H3 only; a scene cut names its seam's transition (disabled)")
_need('c.value = f; c.disabled = true; c.title = "set by " + ht.kind' in js, "S a preset locks the formula select and shows its formula")
_need("repeating-linear-gradient(135deg,\" + H3T_COL" in js and 'pointer-events:none' in js, "S the handle carries a band of the asked length")
# AMENDED v1150: the edit box is the work area's left column, the text TEXT_H in it
# AMENDED v1153: the sound moved beside the monitor -- the edit box takes the whole WORK_H, the text 246
_need('height:" + (custom ? TEXT_H - CUSTOM_ROW_H - 3 : TEXT_H) + "px;' in js and "export const CUSTOM_ROW_H = 22, TEXT_H = 246;" in js
      and 'height:" + WORK_H + "px;flex:0 0 auto;overflow:hidden;' in js
      and 26 + 4 + 26 + 4 + 246 <= 312, "S the custom row fits the fixed edit height (the text gives up its room)")
_need(re.search(r"ph_cine_timeline\.js v10\d\d loaded", js) is not None, "S the module announces its version")

print()
if FAILS:
    print("test_v1035_h3_transitions: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1035_h3_transitions: PASS (P1-P8, S)")
