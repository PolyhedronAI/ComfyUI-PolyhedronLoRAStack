# -*- coding: ascii -*-
"""Guard v1215 -- first frame -> last frame: Transition Check, the Keyframes pair line, the FL2VA rules.

Frank 23.10.: "First Frame to Last Frame ... A) Model steht B) Model hat sich auf einen Stuhl gesetzt". The model
side (first / last keyframes, guides in between, pose control) was in the pack already; v1215 adds the measuring.

  T1  render_check on synthetic clips of one scene (a figure standing, then sitting on a chair):
      exact cross-fade -> dissolve, cross-fade + noise -> dissolve, the figure moving -> motion (also with noise
      and eased), held and switched at the end -> snap, half fade half move -> mixed, first = last -> still,
      4 frames -> short; a passage crowded into the last 15 % is named
  T2  pair_check: B made from A (only the figure changed) is a pair to move through; an unrelated picture is far;
      a brightness shift is named; a size mismatch is fitted, not refused
  T3  anchor_hit: 0 % on the exact keyframes, off when the clip ends elsewhere
  T4  the node: three outputs + ui text; the pair alone works; nothing wired raises by name; report is an IMAGE
  T5  registration: __init__ import + both mappings, NODE_IDS and WIDGET_ORDER baselines carry the node
  T6  Keyframes: with both frames the info carries 'pair first -> last', with one frame it does not
  T7  Timeline fl2va_rules: both keyframes -> the official line; 2 shots -> WARN; a guide -> its line;
      ref2va or one keyframe -> nothing; compile_prompt carries them;
      an anchor 0.25 s or less before the end is named as a guide meant as the last frame
  S   ASCII

Script-style: exit 0 = pass.
"""
import importlib
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1215_transition"
FAILS = []


def _need(c, m):
    print(("  ok    " if c else "  FAIL  ") + m)
    if not c:
        FAILS.append(m)


def _stubs():
    comfy = types.ModuleType("comfy")
    cu = types.ModuleType("comfy.utils")
    cu.common_upscale = lambda s, w, h, m, c: F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)
    comfy.utils = cu
    nh = types.ModuleType("node_helpers")
    nh.conditioning_set_values = lambda cond, v: [[c[0], dict(c[1], **v)] for c in cond]
    sys.modules.update({"comfy": comfy, "comfy.utils": cu, "node_helpers": nh})
    pkg = types.ModuleType("plsn1215")
    pkg.__path__ = [str(ROOT / "nodes")]
    sys.modules["plsn1215"] = pkg


_stubs()
T = importlib.import_module("plsn1215.ph_transition")
KF = importlib.import_module("plsn1215.ph_minimax_keyframes")
M = importlib.import_module("plsn1215.uls_latent_math")
TL = importlib.import_module("plsn1215.cine_timeline")

# ------------------------------------------------------------------ one scene, a figure standing / sitting
torch.manual_seed(1215)
H, W, N = 432, 768, 124
BG = F.interpolate(torch.rand(1, 3, 12, 20), size=(H, W), mode="bicubic", align_corners=False)[0].clamp(0, 1) * 0.6 + 0.2


def scene(a):
    """a = 0: the figure stands left of the chair; a = 1: it sits on it. In between: on the way."""
    f = BG.clone()
    f[:, 300:380, 400:470] = torch.tensor([0.4, 0.25, 0.1])[:, None, None]          # the chair
    x, y0, h = int(250 + 155 * a), int(100 + 120 * a), int(280 - 160 * a)
    f[:, y0:y0 + h, x:x + 60] = torch.tensor([0.9, 0.8, 0.7])[:, None, None]       # the figure
    return f


A, B = scene(0.0), scene(1.0)


def clip(fn, n=N):
    return torch.stack([fn(t / float(n - 1)) for t in range(n)]).movedim(1, -1)


def img(x):
    return x[None].movedim(1, -1)


def noisy(x):
    return (x + 0.03 * torch.randn_like(x)).clamp(0, 1)


def ease(a):
    return 3 * a * a - 2 * a ** 3


cases = {
    "cross-fade": ("dissolve", clip(lambda a: (1 - a) * A + a * B)),
    "cross-fade + noise": ("dissolve", clip(lambda a: noisy((1 - a) * A + a * B))),
    "motion": ("motion", clip(scene)),
    "motion + noise": ("motion", clip(lambda a: noisy(scene(a)))),
    "eased motion": ("motion", clip(lambda a: scene(ease(a)))),
    "hold and snap": ("snap", clip(lambda a: A if a < 0.95 else B)),
    "half fade half move": ("mixed", clip(lambda a: 0.5 * ((1 - a) * A + a * B) + 0.5 * scene(a))),
    "first = last": ("still", clip(lambda a: A)),
    "four frames": ("short", clip(scene, 4)),
}
print("T1 render_check")
for name, (want, frames) in cases.items():
    r = T.render_check(frames)
    _need(r["kind"] == want, "T1 %-20s -> %s (got %s, score %.3f, between %d)"
          % (name, want, r["kind"], r["score"], r["between"]))
r = T.render_check(cases["cross-fade + noise"][1])
_need(0.0 <= r["score"] < T.DISSOLVE_MAX, "T1 the noisy cross-fade stays under the fade line (%.3f)" % r["score"])
r = T.render_check(cases["motion"][1])
_need(r["score"] >= T.MOTION_MIN, "T1 the motion stays over the motion line (%.3f)" % r["score"])
late = clip(lambda a: scene(max(0.0, (a - 0.85) / 0.15)))
r = T.render_check(late)
_need(r["kind"] in ("motion", "snap") and ("crowded" in r["verdict"] or r["kind"] == "snap"),
      "T1 a passage in the last 15 %% is named (%s)" % r["verdict"][:90])

print("T2 pair_check")
p = T.pair_check(img(A), img(B))
_need(not p["far"] and 0.02 < p["changed"] < 0.3, "T2 B made from A: %.0f%% changed, not far" % (p["changed"] * 100))
q = T.pair_check(img(A), torch.rand(1, H, W, 3))
_need(q["far"] and "dissolve" in q["verdict"], "T2 an unrelated picture is far: %s" % q["verdict"][:60])
d = T.pair_check(img(A), img((B * 0.6).clamp(0, 1)))
_need("light differs" in d["verdict"], "T2 a darker B names the light (%+.2f)" % d["light"])
s = T.pair_check(img(A), torch.rand(1, 300, 300, 3))
_need("changed" in s, "T2 a square B on a 16:9 A is fitted, not refused")

print("T3 anchor_hit")
hit = dict(T.anchor_hit(cases["motion"][1], img(A), img(B)))
_need(hit["first"] < 1.0 and hit["last"] < 1.0, "T3 exact keyframes: %.1f%% / %.1f%%" % (hit["first"], hit["last"]))
off = dict(T.anchor_hit(clip(lambda a: scene(0.5 * a)), img(A), img(B)))
_need(off["last"] > 30.0, "T3 a clip that stops half way is off its last keyframe (%.0f%%)" % off["last"])

print("T4 the node")
node = T.ULSTransitionCheck()
o = node.check(T.PIXEL_LEVEL, images=cases["cross-fade"][1], first_frame=img(A), last_frame=img(B))
res = o["result"]
_need(len(res) == 3 and isinstance(res[0], str) and isinstance(res[1], float) and res[2].dim() == 4
      and res[2].shape[-1] == 3, "T4 info / score / report (IMAGE %s)" % (tuple(res[2].shape),))
_need("DISSOLVE" in res[0] and "pair A -> B" in res[0] and "last frame:" in res[0] and o["ui"]["text"] == [res[0]],
      "T4 the info carries pair, clip verdict and anchor hit; ui text = info")
o2 = node.check(T.PIXEL_LEVEL, first_frame=img(A), last_frame=img(B))["result"]
_need(o2[1] == -1.0 and "pair A -> B" in o2[0] and "clip:" not in o2[0], "T4 the pair alone: no clip verdict, score -1")
try:
    node.check(T.PIXEL_LEVEL)
    _need(False, "T4 nothing wired was accepted")
except ValueError as e:
    _need("Transition Check" in str(e), "T4 nothing wired raises by name")
_need(T.ULSTransitionCheck.OUTPUT_NODE is True, "T4 an output node (runs with nothing wired behind it)")

print("T5 registration")
init = (ROOT / "__init__.py").read_text(encoding="utf-8")
_need("from .nodes.ph_transition import ULSTransitionCheck" in init
      and 'NODE_CLASS_MAPPINGS["ULSTransitionCheck"] = ULSTransitionCheck' in init
      and 'NODE_DISPLAY_NAME_MAPPINGS["ULSTransitionCheck"]' in init, "T5 __init__ imports and maps the node")
ids = list(ROOT.glob("NODE_IDS_baseline_v*.txt"))
wo = list(ROOT.glob("WIDGET_ORDER_baseline_v*.txt"))
_need(len(ids) == 1 and "\nULSTransitionCheck\n" in ids[0].read_text(), "T5 NODE_IDS baseline carries it")
_need(len(wo) == 1 and "\nULSTransitionCheck\tpixel_level\n" in wo[0].read_text(), "T5 WIDGET_ORDER: pixel_level")

print("T6 Keyframes pair line")


class Nested:
    is_nested = True

    def __init__(self, v):
        self._v = v

    def unbind(self):
        return [self._v]


class Clip:
    def tokenize(self, prompt, images=None, **kw):
        return {}

    def encode_from_tokens_scheduled(self, t):
        return [[torch.zeros(1), {}]]


class Vae:
    def encode(self, x):
        return torch.zeros(1, 24, 1, int(x.shape[1]) // 16, int(x.shape[2]) // 16)


lat = {"samples": Nested(torch.zeros(1, 24, M.minimax_video_latent_t(124), 768 // 16, 1344 // 16))}
two = KF.ULSMiniMaxKeyframes().build(Clip(), Vae(), lat, "she sits down", KF.FRAME_FIT[0],
                                     first_frame=img(A), last_frame=img(B))["result"][2]
one = KF.ULSMiniMaxKeyframes().build(Clip(), Vae(), lat, "she sits down", KF.FRAME_FIT[0],
                                     first_frame=img(A))["result"][2]
_need("pair first -> last:" in two and "move through" in two, "T6 both frames: the pair line is in the info")
_need("pair first -> last" not in one, "T6 one frame: no pair line")

print("T7 Timeline FL2VA rules")
fa, la = {"src": "image_1", "at": 0}, {"src": "image_2", "at": 5}
r1 = TL.fl2va_rules(fa, la, [], [{"at": 0}], False)
_need(len(r1) == 1 and "describe the WAY" in r1[0], "T7 both keyframes, one shot: the official line")
r2 = TL.fl2va_rules(fa, la, [("image_3", 60)], [{"at": 0}, {"at": 2}], False)
_need(len(r2) == 3 and "WARN: FL2VA with 2 shots" in r2[1] and "image_3 at 2.50 s" in r2[2],
      "T7 two shots WARN + the guide line")
_need(TL.fl2va_rules(fa, la, [], [{"at": 0}], True) == [] and TL.fl2va_rules(fa, None, [], [{"at": 0}], False) == [],
      "T7 ref2va / one keyframe: nothing")
model = {"seconds": 5, "shots": [{"at": 0, "cut": TL.CUTS[0], "text": "she sits down"}],
         "anchors": [{"src": "image_1", "at": 0}, {"src": "image_2", "at": 5.2}], "soundscape": "a quiet room",
         "music": "N/A", "target": "fl2va"}
c = TL.compile_prompt(model)
_need(c["first"] is not None and c["last"] is not None and any("FL2VA (official)" in x for x in c["lines"]),
      "T7 compile_prompt carries the rule line")
_need("FL2VA (official)" not in c["prompt"], "T7 nothing of it reaches the prompt")
short = dict(model, anchors=[{"src": "image_1", "at": 0}, {"src": "image_2", "at": 5}])
cs = TL.compile_prompt(short)
_need(cs["last"] is None and any("a guide, not the last keyframe" in x and "5.12 s" in x for x in cs["lines"]),
      "T7 an anchor at 5 s on a 5.17 s clip is named as a guide that was meant as the last frame")
_need(not any("a guide, not the last keyframe" in x for x in c["lines"]), "T7 the real last frame: no such WARN")

print("S ASCII")
for f in ("nodes/ph_transition.py", "tests/test_v1215_transition.py"):
    try:
        (ROOT / f).read_bytes().decode("ascii")
        _need(True, "S %s is ASCII" % f)
    except UnicodeDecodeError:
        _need(False, "S %s is not ASCII" % f)

print()
print("%s: %s (%d fail)" % (NAME, "PASS" if not FAILS else "FAIL", len(FAILS)))
sys.exit(1 if FAILS else 0)
