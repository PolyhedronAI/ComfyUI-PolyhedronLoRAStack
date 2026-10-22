# -*- coding: ascii -*-
"""Guard v1017 -- Cine C1: @tags and the H3 prompt check.

  P1  resolve(): slot names and aliases become the LIVE label; an unknown
      tag, an alias to a non-slot and a tag on an unwired slot are errors by
      name; '@' inside a word (mail@host) is not a tag; a tag is reported once.
  P2  check(): each rule fires on its bad case and stays silent on the clean
      prompt -- label out of range (ERROR), wired but unnamed (HINT), weight
      bracket, // rubric, single-slash head, shot numbering / missing time /
      time on shot 1 / non-increasing / past the clip end, unclosed <d>,
      quotes in dialogue, length.
  P3  Reference node: @fox (alias) and @video_1 resolve to the numbers the
      tokenizer will use -- including a soundtrack taking <Audio 1> before a
      standalone audio -- the tokenizer receives the RESOLVED text, the new
      prompt_text output carries it, the info carries the tag map and the
      check; an unknown tag raises BEFORE anything is encoded.
  P4  Keyframes node: @goal is <Picture 1> with only a last frame and
      <Picture 2> with both; prompt_text is resolved; an unwired tag raises.
  P5  Widget order: `tags` is the LAST widget of both nodes (#577 append).

Script-style: exit 0 = pass.
"""
import importlib
import sys
import types
from fractions import Fraction
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _fail(msg):
    FAILS.append(msg)
    print("  FAIL  " + msg)


def _ok(msg):
    print("  ok    " + msg)


def _stubs():
    comfy = types.ModuleType("comfy")
    cu = types.ModuleType("comfy.utils")
    cu.common_upscale = lambda s, w, h, m, c: F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)
    comfy.utils = cu
    nh = types.ModuleType("node_helpers")
    nh.conditioning_set_values = lambda cond, v: [[c[0], dict(c[1], **v)] for c in cond]
    sys.modules.update({"comfy": comfy, "comfy.utils": cu, "node_helpers": nh})
    pkg = types.ModuleType("plsn1017")
    pkg.__path__ = [str(ROOT / "nodes")]
    sys.modules["plsn1017"] = pkg


_stubs()
HP = importlib.import_module("plsn1017.h3_prompt")
MR = importlib.import_module("plsn1017.ph_minimax_ref")
KF = importlib.import_module("plsn1017.ph_minimax_keyframes")
M = sys.modules["plsn1017.uls_latent_math"]

# ========================================================================= P1
slots = ["image_1", "image_2", "image_3", "video_1"]
live = {"image_1": "<Picture 1>", "image_3": "<Picture 2>", "video_1": "<Video 1>"}
al, al_err = HP.parse_aliases("fox = image_1\n# comment\nforest: image_3\n\n// x\nprev=@video_1")
if al == {"fox": "image_1", "forest": "image_3", "prev": "video_1"} and not al_err:
    _ok("P1 aliases parse (=, :, @, comments, blanks)")
else:
    _fail("P1 aliases %r %r" % (al, al_err))
txt, used, err = HP.resolve("@fox walks to @forest; see @prev and @image_3. mail@host @fox", live, slots, al)
if txt == "<Picture 1> walks to <Picture 2>; see <Video 1> and <Picture 2>. mail@host <Picture 1>" and not err:
    _ok("P1 aliases and slot names resolve to the live labels, mail@host untouched")
else:
    _fail("P1 resolved %r err %r" % (txt, err))
if [u[0] for u in used] == ["@fox", "@forest", "@prev", "@image_3"]:
    _ok("P1 each tag is reported once, in order of first use")
else:
    _fail("P1 used %r" % used)
_, _, e1 = HP.resolve("@wolf", live, slots, {})
_, _, e2 = HP.resolve("@image_2", live, slots, {})
_, _, e3 = HP.resolve("x", live, slots, {"bad": "image_9"})
if e1 and "unknown tag @wolf" in e1[0] and e2 and "not wired" in e2[0] and e3 and "not a slot" in e3[0]:
    _ok("P1 unknown tag / unwired slot / alias to a non-slot are errors by name")
else:
    _fail("P1 errors %r %r %r" % (e1, e2, e3))
_, bad = HP.parse_aliases("fox image_1")
if bad and "not 'name = slot'" in bad[0]:
    _ok("P1 a malformed tags line is an error by line number")
else:
    _fail("P1 malformed line %r" % bad)

# ========================================================================= P2
LIVE = {"Picture": 2, "Video": 1, "Audio": 0}
clean = ("[Shot 1] <Picture 1> stands in <Picture 2>, <Video 1> sets the pace. "
         "[Shot 2] At 00:02.500, she turns. (S1) says: <d>[English] Hello there.</d>")
if not [f for f in HP.check(clean, LIVE, 5.0) if f[0] != HP.HINT]:
    _ok("P2 the clean prompt raises no error and no warning")
else:
    _fail("P2 clean prompt: %r" % HP.check(clean, LIVE, 5.0))


def fires(text, level, needle, live=LIVE, dur=5.0):
    return any(f[0] == level and needle in f[1] for f in HP.check(text, live, dur))


cases = [
    ("<Picture 3> smiles", HP.ERROR, "<Picture 3> refers to nothing", "label out of range"),
    ("<Picture 1> only", HP.HINT, "<Picture 2> is wired but never named", "wired but unnamed"),
    ("a (glowing:1.3) sword", HP.WARN, "weight bracket", "weight bracket"),
    ("// SUBJECT\na fox", HP.WARN, "// rubric", "rubric reaches the model"),
    ("/ SUBJECT\na fox", HP.WARN, "single '/'", "broken rubric head"),
    ("[Shot 1] a. [Shot 3] At 00:01.000, b.", HP.WARN, "without gaps", "shot numbering gap"),
    ("[Shot 1] a. [Shot 2] b.", HP.WARN, "has no 'At MM:SS.mmm'", "missing shot time"),
    ("[Shot 1] At 00:00.000, a.", HP.WARN, "[Shot 1] carries a time", "time on shot 1"),
    ("[Shot 1] a. [Shot 2] At 00:03.000, b. [Shot 3] At 00:02.000, c.", HP.WARN, "is not after", "non-increasing times"),
    ("[Shot 1] a. [Shot 2] At 00:06.000, b.", HP.WARN, "after the clip ends", "shot past the clip end"),
    ("(S1) says: <d>[English] hi", HP.WARN, "not closed", "unclosed dialogue"),
    ('(S1) says: <d>[English] "hi"</d>', HP.HINT, "double quotes inside", "quotes in dialogue"),
    (" ".join(["word"] * 600), HP.WARN, "tokens (estimate", "length beyond ~700 tokens"),
    (" ".join(["word"] * 510), HP.HINT, "the official norm is 350-500", "length above the official norm"),
]
for text, level, needle, what in cases:
    if fires(text, level, needle):
        _ok("P2 %s -> %s" % (what, level))
    else:
        _fail("P2 %s did not fire: %r" % (what, HP.check(text, LIVE, 5.0)))


# ========================================================================= P3
class FakeNested:
    is_nested = True

    def __init__(self, v):
        self._v = v

    def unbind(self):
        return [self._v]


class Clip:
    def __init__(self):
        self.prompt = None
        self.calls = 0

    def tokenize(self, prompt, minimax_ref_items=None, images=None, **kw):
        self.prompt = prompt
        self.calls += 1
        return {}

    def encode_from_tokens_scheduled(self, t):
        return [[torch.zeros(1), {}]]


class Vae:
    def __init__(self):
        self.calls = 0

    def encode(self, x):
        self.calls += 1
        n, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        return torch.zeros(1, 24, M.minimax_video_latent_t(n) if n > 1 else 1, h // 16, w // 16)


class AVae:
    audio_sample_rate = 32000

    def encode(self, w):
        return torch.zeros(1, 32, 2, int(w.shape[1]) // 800)


class Vid:
    def __init__(self, n):
        self.f = torch.rand(n, 360, 640, 3)

    def get_components(self):
        return types.SimpleNamespace(images=self.f, audio=None, frame_rate=Fraction(24))


def aud(sec):
    return {"waveform": torch.zeros(1, 2, int(sec * 32000)), "sample_rate": 32000}


lt = M.minimax_video_latent_t(124)
lat = {"samples": FakeNested(torch.zeros(1, 24, lt, 768 // 16, 1344 // 16))}
clip = Clip()
res = MR.ULSMiniMaxReference().build(
    clip, Vae(), lat, "@fox meets @image_4. @prev plays, @song hums, @ost ticks.",
    "match", 0.0, 0.0, 0.0,
    image_2=torch.rand(1, 256, 256, 3), image_4=torch.rand(1, 256, 256, 3),
    audio_vae=AVae(), video_1=Vid(60), video_audio_1=aud(3.0), audio_1=aud(3.0),
    tags="fox = image_2\nprev = video_1\nsong = audio_1\nost = video_audio_1")
want = "<Picture 1> meets <Picture 2>. <Video 1> plays, <Audio 2> hums, <Audio 1> ticks."
if clip.prompt == want:
    _ok("P3 the tokenizer receives the resolved prompt (soundtrack = <Audio 1>, standalone = <Audio 2>)")
else:
    _fail("P3 tokenizer got %r" % clip.prompt)
if len(res) == 4 and res[3] == want:
    _ok("P3 prompt_text carries the resolved prompt")
else:
    _fail("P3 prompt_text %r" % (res[3] if len(res) > 3 else res,))
if "tags: @fox -> <Picture 1>" in res[2] and "prompt check:" in res[2]:
    _ok("P3 info carries the tag map and the check")
else:
    _fail("P3 info lacks tag map/check")
clip2, vae2 = Clip(), Vae()
try:
    MR.ULSMiniMaxReference().build(clip2, vae2, lat, "@wolf", "match", 0.0, 0.0, 0.0,
                                   image_1=torch.rand(1, 256, 256, 3))
    _fail("P3 an unknown tag was accepted")
except ValueError as e:
    if "@wolf" in str(e) and clip2.calls == 0 and vae2.calls == 0:
        _ok("P3 an unknown tag stops the run by name BEFORE any encode")
    else:
        _fail("P3 raised but encoded first (clip %d, vae %d): %s" % (clip2.calls, vae2.calls, e))

# ========================================================================= P4
def kf(first, last, prompt, tags=""):
    c = Clip()
    out = KF.ULSMiniMaxKeyframes().build(
        c, Vae(), lat, prompt, KF.FRAME_FIT[0],
        first_frame=first, last_frame=last, tags=tags)
    return c, out


img = torch.rand(1, 384, 672, 3)
c1, o1 = kf(None, img, "walk to @goal", "goal = last_frame")
c2, o2 = kf(img, img, "from @first_frame to @goal", "goal = last_frame")
r1 = o1["result"] if isinstance(o1, dict) else o1
r2 = o2["result"] if isinstance(o2, dict) else o2
if c1.prompt == "walk to <Picture 1>" and c2.prompt == "from <Picture 1> to <Picture 2>":
    _ok("P4 @goal is <Picture 1> with only a last frame, <Picture 2> with both")
else:
    _fail("P4 got %r / %r" % (c1.prompt, c2.prompt))
if r2[3] == "from <Picture 1> to <Picture 2>":
    _ok("P4 Keyframes prompt_text is the resolved prompt")
else:
    _fail("P4 prompt_text %r" % (r2[3],))
try:
    kf(None, None, "@first_frame")
    _fail("P4 an unwired keyframe tag was accepted")
except ValueError as e:
    if "not wired" in str(e):
        _ok("P4 a tag on an unwired frame stops the run by name")
    else:
        _fail("P4 wrong refusal: %s" % e)

# ========================================================================= P5
for cls, name in ((MR.ULSMiniMaxReference, "Reference"), (KF.ULSMiniMaxKeyframes, "Keyframes")):
    it = cls.INPUT_TYPES()
    widgets = [k for sec in ("required", "optional") for k, v in it.get(sec, {}).items()
               if isinstance(v[0], list) or v[0] in ("STRING", "FLOAT", "INT", "BOOLEAN")]
    # AMENDED v1201: the Reference appended ref_noise behind tags (the #577 law) -- tags stays the widget before it
    if widgets[-1] == "tags" or (name == "Reference" and widgets[-2:] == ["tags", "ref_noise"]):
        _ok("P5 `tags` is the last widget of %s (Reference: + ref_noise, v1201)" % name)
    else:
        _fail("P5 %s widget tail %r" % (name, widgets[-3:]))

print()
if FAILS:
    print("test_v1017_h3_prompt: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1017_h3_prompt: PASS (P1-P5)")
