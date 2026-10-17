# -*- coding: ascii -*-
"""Guard v1023 -- Cine C6: the timeline (model, compiler, node, route).

  T1  the grid: Core's 17k+5 at 24 fps (8 s -> 192, 5 s -> 124, 4 s -> 107),
      MM:SS.mmm and S.SS exactly as the official text writes them.
  T2  validation by name: shot 1 not at 0, cuts not forward, a cut past the
      end, an unofficial cut phrase, no shot, an anchor outside the clip, two
      anchors on one frame, an anchor source that is not there, bad JSON.
  T3  the compiled prompt, verbatim against 10_H3_OFFIZIELL.md / 60_VORLAGEN:
      T2VA without alignment line; I2VA, FL2VA, L2VA sentences word for word
      (last frame = frame_count - 1, its shot number); style after [Shot 1];
      [Shot N] At MM:SS.mmm, <cut> ...; '(none)' drops the phrase; field
      order; music N/A by default; no soundscape -> no field + WARN.
  T4  anchors split by what the model can see: frame 0 / last frame ->
      Keyframes, the rest -> Continue guides (named in info, not in the text).
  T5  the H3 check on a compiled prompt stays clean -- the alignment line's
      "(from [Shot 1])" and Ref2VA's "(appears in [Shot 1], [Shot 2])" are
      references, not markers (the v1017 check counted them), while inline
      shots still count.
  T6  the node: first/last frames and guides come from the board by @tag and
      from wired slots; a missing source stops the run by name; @tags in the
      text are left for the Keyframes node.
  T7  MiniMax Continue takes the timeline's anchors at their frames, with
      Add Guide's fit rule.
  T8  the preview route answers with the backend's own compiler.

Script-style: exit 0 = pass.
"""
import asyncio
import importlib
import json
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _fail(m):
    FAILS.append(m)
    print("  FAIL  " + m)


def _ok(m):
    print("  ok    " + m)


def _need(c, m):
    (_ok if c else _fail)(m)


comfy = types.ModuleType("comfy")
cu = types.ModuleType("comfy.utils")
cu.common_upscale = lambda s, w, h, m, c: F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)
comfy.utils = cu
nh = types.ModuleType("node_helpers")
nh.conditioning_set_values = lambda cond, v: [[c[0], dict(c[1], **v)] for c in cond]
ce = types.ModuleType("comfy_extras")
core_h3 = types.ModuleType("comfy_extras.nodes_minimax_h3")
core_h3.MiniMaxH3AddGuide = object
ce.nodes_minimax_h3 = core_h3
sys.modules.update({"comfy": comfy, "comfy.utils": cu, "node_helpers": nh,
                    "comfy_extras": ce, "comfy_extras.nodes_minimax_h3": core_h3})
pkg = types.ModuleType("plsn1023")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1023"] = pkg
TL = importlib.import_module("plsn1023.ph_cine_timeline")
T = sys.modules["plsn1023.cine_timeline"]
HP = sys.modules["plsn1023.h3_prompt"]
CO = importlib.import_module("plsn1023.ph_minimax_continue")
# v1061: the guide verdict reads the H3 MODEL code only (no comfy_extras guess); the stub world sets it.
CO.model_has_guides = lambda: True
M = sys.modules["plsn1023.uls_latent_math"]


def model(**kw):
    m = json.loads(json.dumps(T.DEFAULT))
    m.update(kw)
    return m


# ===================================================================== T1
_need((T.frames_for(8), T.frames_for(5), T.frames_for(4), T.frames_for(15)) == (192, 124, 107, 362),
      "T1 grid: 8 s -> 192, 5 s -> 124, 4 s -> 107, 15 s -> 362 frames (Core's align up)")
_need((T.time_label(2.5), T.time_label(65.25), T.time_label(0)) == ("00:02.500", "01:05.250", "00:00.000")
      and T.mark(191 / 24.0) == "7.96", "T1 MM:SS.mmm and S.SS as the official text writes them")

# ===================================================================== T2
S2 = [{"at": 0, "cut": T.CUTS[0], "text": "a"}, {"at": 2.0, "cut": T.CUTS[0], "text": "b"}]
cases = [
    (model(shots=[{"at": 0.5, "text": "a"}]), "[Shot 1] starts at 0", "shot 1 not at 0"),
    (model(shots=[S2[0], {"at": 3.0, "text": "b"}, {"at": 2.0, "text": "c"}]), "cuts go forward", "cuts backwards"),
    (model(seconds=5, shots=[S2[0], {"at": 6.0, "text": "b"}]), "starts after the clip ends", "cut past the end"),
    (model(shots=[S2[0], {"at": 2.0, "cut": "the camera whips to", "text": "b"}]), "not one of the official phrases", "unofficial cut"),
    (model(shots=[]), "at least [Shot 1]", "no shot"),
    (model(anchors=[{"src": "image_1", "at": 9.0}]), "outside the clip", "anchor outside"),
    (model(anchors=[{"src": "image_1", "at": 1.0}, {"src": "image_2", "at": 1.01}]), "the same frame", "two anchors one frame"),
    (model(anchors=[{"src": "@ghost", "at": 1.0}]), "not on the board and not wired", "unknown source"),
]
for m, needle, label in cases:
    errs = T.validate(m, {"image_1", "image_2"})
    _need(any(needle in e for e in errs), "T2 refused by name: %s" % label + ("" if any(needle in e for e in errs) else " %r" % errs))
try:
    T.load("{nope")
    _fail("T2 bad JSON accepted")
except ValueError as e:
    _need("not JSON" in str(e), "T2 bad JSON refused by name")
_need(T.validate(model(shots=S2, anchors=[{"src": "image_1", "at": 0}]), {"image_1"}) == [],
      "T2 a clean timeline validates")

# ===================================================================== T3
r = T.compile_prompt(model(seconds=8, shots=S2, soundscape="Wind."))
_need(r["prompt"] == ("integrated_multimodal_description: [Shot 1] Live-action, cinematic, a\n\n"
                      "[Shot 2] At 00:02.000, the camera cuts to b\n\n"
                      "overall_soundscape: Wind.\n\nnon_diegetic_music: N/A"),
      "T3 T2VA: no alignment line, style after [Shot 1], [Shot 2] At 00:02.000, cut phrase, fields in order, music N/A")
I2 = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
r = T.compile_prompt(model(seconds=8, shots=S2, anchors=[{"src": "image_1", "at": 0}]))
_need(r["prompt"].startswith(I2 + "\n\nintegrated_multimodal_description: "), "T3 I2VA sentence word for word (10_H3_OFFIZIELL section 2)")
FL = ("How the reference pictures align with the target video -- Picture 1 (from Shot 1) aligns with the "
      "0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the 7.96-second mark of the "
      "target video.")
r = T.compile_prompt(model(seconds=8, shots=S2, anchors=[{"src": "image_1", "at": 0}, {"src": "image_2", "at": 191 / 24.0}]))
_need(r["prompt"].startswith(FL + "\n\n"), "T3 FL2VA sentence word for word, last frame 191/24 = 7.96 s in Shot 2")
L2 = ("How the reference pictures align with the target video -- <Picture 1> (from [Shot 2]) aligns with the "
      "7.96-second mark of the target video.")
r = T.compile_prompt(model(seconds=8, shots=S2, anchors=[{"src": "image_2", "at": 191 / 24.0}]))
_need(r["prompt"].startswith(L2 + "\n\n"), "T3 L2VA sentence word for word")
r = T.compile_prompt(model(seconds=8, shots=[S2[0], {"at": 2.0, "cut": "(none)", "text": "Meanwhile b"}]))
_need("[Shot 2] At 00:02.000, Meanwhile b" in r["prompt"], "T3 cut '(none)' writes the text as it is")
r = T.compile_prompt(model(seconds=8, shots=S2, soundscape="", music=""))
_need("overall_soundscape" not in r["prompt"] and r["prompt"].endswith("non_diegetic_music: N/A")
      and any("no overall_soundscape" in l for l in r["lines"]), "T3 empty soundscape: no field, a WARN line; empty music -> N/A")
r = T.compile_prompt(model(seconds=3, shots=[S2[0]]))
_need(any("outside the official 4-15 s" in l for l in r["lines"]), "T3 a clip under 4 s is reported")

# ===================================================================== T4
m4 = model(seconds=8, shots=S2, anchors=[{"src": "image_1", "at": 0}, {"src": "image_2", "at": 4.0},
                                         {"src": "image_3", "at": 191 / 24.0}])
r = T.compile_prompt(m4)
_need(r["first"]["src"] == "image_1" and r["last"]["src"] == "image_3" and r["guides"] == [("image_2", 96)],
      "T4 frame 0 and the last frame are keyframes, frame 96 is a guide")
_need("image_2" not in r["prompt"] and any("guide: image_2 at frame 96" in l and "does not name it" in l for l in r["lines"]),
      "T4 the guide is named in the info, never in the text")

# ===================================================================== T5
rep = HP.check(r["prompt"], {"Picture": 2, "Video": 0, "Audio": 0}, r["seconds"])
_need(rep == [], "T5 the check on a compiled FL2VA prompt is clean: no shot problem, and the bare "
      "'Picture 1/2' of the official wording count as named %r" % (rep,))
refs_txt = "<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - x.\ndetailed_description: [Shot 1] a. [Shot 2] At 00:01.000, b."
_need([m.group(1) for m in HP.shot_markers(refs_txt)] == ["1", "2"], "T5 '(appears in [Shot 1], [Shot 2])' is a reference, not a marker")
_need([m.group(1) for m in HP.shot_markers("[Shot 1] a. [Shot 3] At 00:01.000, b.")] == ["1", "3"],
      "T5 inline shots still count (the CTE strips line breaks)")

# ===================================================================== T6
img = lambda v: torch.full((1, 32, 48, 3), v)
refs = {"images": [{"tag": "fox", "image": img(0.1), "mp": 0}, {"tag": "end", "image": img(0.9), "mp": 0}]}
tl6 = json.dumps(model(seconds=8, shots=[{"at": 0, "cut": T.CUTS[0], "text": "@fox runs"}],
                       anchors=[{"src": "@fox", "at": 0}, {"src": "image_2", "at": 3.0}, {"src": "@end", "at": 191 / 24.0}],
                       soundscape="Wind."))
out = TL.ULSCineTimeline().build(tl6, refs=refs, image_2=img(0.5))
prompt, first, last, anchors, seconds, frames, info = out["result"][:7]
_need(float(first.mean()) == float(img(0.1).mean()) and float(last.mean()) == float(img(0.9).mean()),
      "T6 first/last frames come from the board by @tag")
_need(len(anchors) == 1 and anchors[0]["frame"] == 72 and anchors[0]["src"] == "image_2"
      and abs(float(anchors[0]["image"].mean()) - 0.5) < 1e-6, "T6 the guide comes from the wired slot, at frame 72")
_need(frames == 192 and abs(seconds - 8.0) < 1e-9 and "@fox runs" in prompt and "MiniMax Keyframes only @first_frame / @last_frame" in info   # v1084 (DECLARED): the honest tag line
      and "unresolved tag" not in info, "T6 frames/seconds out; @tags left for the Keyframes node, said in the info, "
      "not reported as an error")
_need("pls_timeline" in out["ui"], "T6 the node shows the prompt that ran (ui pls_timeline)")
try:
    TL.ULSCineTimeline().build(tl6, refs=refs)
    _fail("T6 a missing source was accepted")
except ValueError as e:
    _need("image_2 is not on the board and not wired" in str(e), "T6 a missing source stops the run by name")

# ===================================================================== T7
class RecVae:
    def __init__(self):
        self.n = []

    def encode(self, x):
        self.n.append(x.shape[0])
        return torch.full((1, 24, 1, x.shape[1] // 16, x.shape[2] // 16), float(x.mean()))


class Nested:
    is_nested = True

    def __init__(self, *t):
        self.t = t

    def unbind(self):
        return list(self.t)


lat = {"samples": Nested(torch.zeros(1, 24, M.minimax_video_latent_t(192), 48, 84), torch.zeros(1, 32, 2, 320))}
rv = RecVae()
try:
    o7 = CO.ULSMiniMaxContinue().build([[torch.zeros(1), {}]], lat, CO.MODES[2], 22, True, -1, vae=rv, anchors=anchors)
    k7 = o7[0][0][1]["minimax_keyframes"]
    _need(len(k7) == 1 and k7[0]["resolved_frame_index"] == 72 and "anchor image_2: 1 image at frame 72" in o7[3],
          "T7 Continue places the timeline's anchor at frame 72 and names it")
except Exception as e:          # a refusal here is a FAIL by name, not a crash
    _fail("T7 Continue refused the timeline's anchors: %s" % e)
try:
    CO.ULSMiniMaxContinue().build([[torch.zeros(1), {}]], lat, CO.MODES[2], 22, True, -1, vae=RecVae(),
                                  anchors=[{"src": "clip", "frame": 190, "image": torch.rand(22, 32, 48, 3)}])
    _fail("T7 a 22-frame anchor at 190 of 192 was accepted")
except ValueError as e:
    _need("does not fit" in str(e), "T7 Add Guide's fit rule holds for timeline anchors")

# ===================================================================== T8
import importlib.util as _iu
try:
    if _iu.find_spec("aiohttp") is None:
        raise ImportError("aiohttp")

    class Req:
        def __init__(self, body):
            self.body = body

        async def json(self):
            return self.body

    resp = asyncio.run(TL.handle_timeline_preview(Req({"timeline": tl6, "sources": ["@fox", "@end", "image_2"]})))
    d = json.loads(resp.text)
    want = T.compile_prompt(T.load(tl6))["prompt"]
    _need(d["errors"] == [] and d["prompt"] == want and d["frames"] == 192,
          "T8 the preview route returns exactly the backend's compiled prompt")
    resp = asyncio.run(TL.handle_timeline_preview(Req({"timeline": tl6, "sources": ["@fox"]})))
    _need(any("not on the board" in e for e in json.loads(resp.text)["errors"]), "T8 the route reports errors by name")
except ImportError:
    print("  note  aiohttp missing -- T8 SKIPPED")

print()
if FAILS:
    print("test_v1023_cine_timeline: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1023_cine_timeline: PASS (T1-T8)")
