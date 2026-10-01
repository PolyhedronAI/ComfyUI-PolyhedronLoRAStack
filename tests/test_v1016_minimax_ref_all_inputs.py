# -*- coding: ascii -*-
"""Guard v1016 -- Polyhedron MiniMax Reference takes every input H3 takes.

MiniMax's README (Ref2VA): <= 9 images; <= 3 videos, 2-15 s each, <= 15 s
together; <= 3 audios, same durations; <= 12 files. Core's
MiniMaxH3ReferenceToVideo exposes the counts (9/3/3/3) and checks none of the
durations. Until v1015 this node took three images and nothing else.

The promises, DRIVEN (fake clip / vae / audio vae / VIDEO objects):

  A1  INPUT_TYPES: image_1..9, megapixels_4..9 (optional FLOAT, AFTER
      megapixels_3 in the widget order), audio_vae, video_1..3 (VIDEO),
      video_audio_1..3 (AUDIO), audio_1..3 (AUDIO). image_1..3 stay first.
  A2  plan_tags mirrors Core's token order: all images, then per video its
      soundtrack <Audio j> BEFORE the <Video k>, then standalone audio.
  A3  build() with 2 images + a 30 fps video with soundtrack + a 24 fps video
      + 1 audio: tokenizer items in Core's order, DiT block kinds in Core's
      order, the 30 fps clip RESAMPLED to 24 (Core would not), trimmed to the
      17k+5 grid, Qwen frames every 12th frame with half-second stamps, and the
      info output names every live tag.
  A4  an audio without audio_vae is refused BY NAME; a soundtrack without its
      video is reported as IGNORED.
  A5  limit_notes: clip outside 2-15 s, total over 15 s, more than 12 files,
      audio alone -- each produces a LIMIT line; a clean set produces none.
  A6  resample_indices / ref_video_frames: identity at 24 fps, 30 -> 24 keeps
      the duration, the trim lands on 17k+5, fewer than 5 frames is None.
  A7  CORE PARITY (skipped when Core is not reachable): the same references
      through Core's own node yield the same ref_items type order and the same
      block kinds, latent_t, latent_h, latent_w and ref_audio_t.

Script-style: exit 0 = pass.
"""
import importlib
import os
import subprocess
import sys
import textwrap
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


# --------------------------------------------------------------------------
# load the module with stubbed Core pieces
# --------------------------------------------------------------------------

def _stub_modules():
    comfy = types.ModuleType("comfy")
    cu = types.ModuleType("comfy.utils")

    def common_upscale(samples, width, height, method, crop):
        return F.interpolate(samples, size=(height, width), mode="bilinear",
                             align_corners=False)
    cu.common_upscale = common_upscale
    comfy.utils = cu
    nh = types.ModuleType("node_helpers")

    def conditioning_set_values(cond, values):
        return [[c[0], dict(c[1], **values)] for c in cond]
    nh.conditioning_set_values = conditioning_set_values
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = cu
    sys.modules["node_helpers"] = nh


def _load():
    _stub_modules()
    pkg = types.ModuleType("plsn1016")
    pkg.__path__ = [str(ROOT / "nodes")]
    sys.modules["plsn1016"] = pkg
    return importlib.import_module("plsn1016.ph_minimax_ref")


MR = _load()
M = sys.modules["plsn1016.uls_latent_math"]


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeNested:
    is_nested = True

    def __init__(self, video):
        self._v = video

    def unbind(self):
        return [self._v]


def latent_for(w, h, frames):
    lt = M.minimax_video_latent_t(frames)
    return {"samples": FakeNested(torch.zeros(1, 24, lt, h // 16, w // 16))}


class FakeClip:
    def __init__(self):
        self.items = None

    def tokenize(self, prompt, minimax_ref_items=None, **kw):
        self.items = list(minimax_ref_items or [])
        return {"t": prompt}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1), {}]]


class FakeVae:
    def __init__(self):
        self.shapes = []

    def encode(self, x):
        self.shapes.append(tuple(x.shape))
        n, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        return torch.zeros(1, 24, M.minimax_video_latent_t(n) if n > 1 else 1,
                           h // 16, w // 16)


class FakeAudioVae:
    audio_sample_rate = 32000

    def encode(self, w):
        return torch.zeros(1, 32, 2, int(w.shape[1]) // 800)


class FakeVideo:
    def __init__(self, n, fps, h=360, w=640):
        self.frames = torch.rand(n, h, w, 3)
        self.fps = fps

    def get_components(self):
        return types.SimpleNamespace(images=self.frames, audio=None,
                                     frame_rate=Fraction(self.fps))


def audio(sec, sr=44100):
    return {"waveform": torch.zeros(1, 2, int(sec * sr)), "sample_rate": sr}


def img(h=512, w=512):
    return torch.rand(1, h, w, 3)


# ========================================================================= A1
it = MR.ULSMiniMaxReference.INPUT_TYPES()
opt = list(it["optional"].keys())
req = list(it["required"].keys())
want_imgs = ["image_%d" % n for n in range(1, 10)]
if opt[:3] == ["image_1", "image_2", "image_3"]:
    _ok("A1 image_1..3 stay first among the optional inputs")
else:
    _fail("A1 optional order starts %r" % (opt[:3],))
missing = [n for n in want_imgs + ["audio_vae"]
           + ["video_%d" % n for n in (1, 2, 3)]
           + ["video_audio_%d" % n for n in (1, 2, 3)]
           + ["audio_%d" % n for n in (1, 2, 3)]
           + ["megapixels_%d" % n for n in range(4, 10)] if n not in opt]
if missing:
    _fail("A1 missing inputs %r" % missing)
else:
    _ok("A1 9 images, 3 videos, 3 soundtracks, 3 audios, audio_vae, megapixels_4..9")
types_ok = (all(it["optional"]["video_%d" % n][0] == "VIDEO" for n in (1, 2, 3))
            and all(it["optional"]["video_audio_%d" % n][0] == "AUDIO" for n in (1, 2, 3))
            and all(it["optional"]["audio_%d" % n][0] == "AUDIO" for n in (1, 2, 3))
            and it["optional"]["audio_vae"][0] == "VAE"
            and all(it["optional"]["megapixels_%d" % n][0] == "FLOAT" for n in range(4, 10)))
if types_ok:
    _ok("A1 socket types VIDEO / AUDIO / VAE / FLOAT as declared")
else:
    _fail("A1 a socket carries the wrong type")
widget_order = [k for k in req if it["required"][k][0] in ("FLOAT", "STRING") or isinstance(it["required"][k][0], list)] \
    + [k for k in opt if it["optional"][k][0] == "FLOAT"]
if widget_order[-9:] == ["megapixels_%d" % n for n in range(1, 10)]:
    _ok("A1 megapixels_4..9 follow megapixels_3 in the widget order (append law)")
else:
    _fail("A1 widget order tail %r" % (widget_order[-9:],))

# ========================================================================= A2
tags = MR.plan_tags(2, [True, False], 1)
want = [("image", "<Picture 1>"), ("image", "<Picture 2>"),
        ("soundtrack", "<Audio 1>"), ("video", "<Video 1>"),
        ("video", "<Video 2>"), ("audio", "<Audio 2>")]
if tags == want:
    _ok("A2 plan_tags: images, soundtrack BEFORE its video, standalone audio last")
else:
    _fail("A2 plan_tags %r" % (tags,))

# ========================================================================= A3
clip, vae, avae = FakeClip(), FakeVae(), FakeAudioVae()
lat = latent_for(1344, 768, 124)
v30 = FakeVideo(90, 30)          # 3.0 s at 30 fps
v24 = FakeVideo(60, 24)          # 2.5 s at 24 fps
cond, lat_out, info, _pt = MR.ULSMiniMaxReference().build(  # v1017: + prompt_text
    clip, vae, lat, "p", "match", 0.0, 0.0, 0.0,
    image_1=img(), image_3=img(1024, 768), audio_vae=avae,
    video_1=v30, video_audio_1=audio(3.0), video_2=v24, audio_1=audio(4.0))
kinds = [i["type"] for i in clip.items]
if kinds == ["image", "image", "audio", "video", "video", "audio"]:
    _ok("A3 tokenizer items in Core's order (image, image, soundtrack, video, video, audio)")
else:
    _fail("A3 tokenizer item order %r" % kinds)
blocks = cond[0][1].get("minimax_refs", [])
bk = [b["kind"] for b in blocks]
if bk == ["image", "image", "video_audio", "video", "audio"]:
    _ok("A3 DiT blocks in Core's order, the soundtracked video is 'video_audio'")
else:
    _fail("A3 block kinds %r" % bk)
vid_shapes = [s for s in vae.shapes if s[0] > 1]
# 90 @ 30 fps = 3.0 s -> 72 frames @ 24 -> trim to 17k+5 -> 56
# 60 @ 24 fps -> 60 -> 56
if [s[0] for s in vid_shapes] == [56, 56]:
    _ok("A3 30 fps clip resampled to 24 (72 frames) and both trimmed to 56 = 17*3+5")
else:
    _fail("A3 video frame counts at the vae %r" % ([s[0] for s in vid_shapes],))
qv = [i for i in clip.items if i["type"] == "video"]
if qv and qv[0]["data"].shape[0] == 5 and qv[0]["timestamps"] == [0.0, 0.5, 1.0, 1.5, 2.0]:
    _ok("A3 Qwen sees every 12th frame with half-second stamps (Core's rule)")
else:
    _fail("A3 Qwen frames %r" % ([(i["data"].shape[0], i["timestamps"]) for i in qv],))
needles = ("<Picture 1>  image_1", "<Picture 2>  image_3", "<Video 1>  video_1",
           "30.00 fps -> 24 (resampled)", "<Audio 1>  video_audio_1  soundtrack of <Video 1>",
           "<Video 2>  video_2", "<Audio 2>  audio_1", "2 image(s), 2 video(s), 2 audio(s)")
miss = [n for n in needles if n not in info]
if miss:
    _fail("A3 info lacks %r" % miss)
else:
    _ok("A3 info names every live tag, the slot behind it and the fps resample")
if lat_out is lat:
    _ok("A3 the latent passes through unchanged (one source of truth)")
else:
    _fail("A3 the latent was replaced")

# ========================================================================= A4
try:
    MR.ULSMiniMaxReference().build(FakeClip(), FakeVae(), lat, "p", "match",
                                   0.0, 0.0, 0.0, audio_1=audio(3.0))
    _fail("A4 an audio without audio_vae was accepted")
except Exception as e:           # a crash further down is NOT a refusal by name
    if isinstance(e, ValueError) and "audio_vae" in str(e):
        _ok("A4 an audio without audio_vae is refused by name")
    else:
        _fail("A4 no refusal by name, got %s: %s" % (type(e).__name__, e))
_, _, info2, _ = MR.ULSMiniMaxReference().build(
    FakeClip(), FakeVae(), lat, "p", "match", 0.0, 0.0, 0.0,
    image_1=img(), audio_vae=avae, video_audio_2=audio(3.0))
if "IGNORED: video_audio_2 is wired but video_2 is not" in info2:
    _ok("A4 a soundtrack without its video is reported as IGNORED")
else:
    _fail("A4 orphan soundtrack not reported")
try:
    MR.ULSMiniMaxReference().build(FakeClip(), FakeVae(), lat, "p", "match",
                                   0.0, 0.0, 0.0, video_1=FakeVideo(3, 24))
    _fail("A4 a 3-frame video was accepted")
except ValueError as e:
    if "video_1" in str(e) and "at least 5" in str(e):
        _ok("A4 a video under five frames is refused by name")
    else:
        _fail("A4 short-video refusal: %s" % e)

# ========================================================================= A5
clean = MR.limit_notes(3, [("<Video 1>", 5.0)], [("<Audio 1>", 4.0)])
if not clean:
    _ok("A5 a clean set produces no LIMIT line")
else:
    _fail("A5 clean set produced %r" % clean)
cases = (
    (MR.limit_notes(1, [("<Video 1>", 1.0)], []), "1.00 s", "clip under 2 s"),
    (MR.limit_notes(1, [("<Video 1>", 16.0)], []), "16.00 s", "clip over 15 s"),
    (MR.limit_notes(1, [("<Video 1>", 9.0), ("<Video 2>", 9.0)], []), "add up to 18.00 s", "total over 15 s"),
    (MR.limit_notes(9, [("<Video %d>" % i, 3.0) for i in (1, 2)], [("<Audio %d>" % i, 3.0) for i in (1, 2)]), "13 reference files", "more than 12 files"),
    (MR.limit_notes(0, [], [("<Audio 1>", 3.0)]), "audio without any image or video", "audio alone"),
)
for notes, needle, what in cases:
    if any(needle in n and n.startswith("LIMIT:") for n in notes):
        _ok("A5 LIMIT line for %s" % what)
    else:
        _fail("A5 no LIMIT line for %s: %r" % (what, notes))

# ========================================================================= A6
if MR.resample_indices(10, 24.0) == list(range(10)):
    _ok("A6 24 fps is the identity")
else:
    _fail("A6 24 fps not identity")
r = MR.resample_indices(90, 30.0)
if len(r) == 72 and r[0] == 0 and r[-1] <= 89 and r == sorted(r):
    _ok("A6 30 -> 24 fps keeps the duration (90 -> 72 frames, monotone)")
else:
    _fail("A6 30 -> 24: len %d" % len(r))
if MR.ref_video_frames(72, 124) == 56 and MR.ref_video_frames(300, 124) == 124 \
        and MR.ref_video_frames(4, 124) is None and MR.ref_video_frames(5, 124) == 5:
    _ok("A6 trim: 72 -> 56, capped at the clip, 5 is the floor, 4 is None")
else:
    _fail("A6 trim values %r" % ([MR.ref_video_frames(x, 124) for x in (72, 300, 4, 5)],))

# ========================================================================= A7
core = os.environ.get("PLS_CORE_ROOT")
if not core:
    for up in ROOT.parents:
        if (up / "comfy_extras" / "nodes_minimax_h3.py").exists():
            core = str(up)
            break
if not core or not (Path(core) / "comfy_extras" / "nodes_minimax_h3.py").exists():
    print("  note  Core not reachable -- A7 parity SKIPPED "
          "(set PLS_CORE_ROOT=<ComfyUI> to run it)")
else:
    script = textwrap.dedent(r'''
        import sys, types, importlib, json
        sys.argv = [sys.argv[0], "--cpu"]      # Core's cli_args: no GPU here
        sys.path.insert(0, CORE)
        import comfy.options
        comfy.options.enable_args_parsing()    # main.py does this; else --cpu is ignored
        import torch
        from fractions import Fraction
        import comfy_extras.nodes_minimax_h3 as C
        pkg = types.ModuleType("plsn_par"); pkg.__path__ = [NODES]
        sys.modules["plsn_par"] = pkg
        MR = importlib.import_module("plsn_par.ph_minimax_ref")
        M = sys.modules["plsn_par.uls_latent_math"]
        class Clip:
            def tokenize(self, p, minimax_ref_items=None, images=None, **k):
                self.items = [i["type"] for i in (minimax_ref_items or [])]; return {}
            def encode_from_tokens_scheduled(self, t): return [[torch.zeros(1), {}]]
        class Vae:
            def encode(self, x):
                n, h, w = x.shape[0], x.shape[1], x.shape[2]
                return torch.zeros(1, 24, M.minimax_video_latent_t(n) if n > 1 else 1, h // 16, w // 16)
        class AVae:
            audio_sample_rate = 32000
            def encode(self, w): return torch.zeros(1, 32, 2, int(w.shape[1]) // 800)
        class Vid:
            def __init__(self, f): self.f = f
            def get_components(self):
                return types.SimpleNamespace(images=self.f, audio=None, frame_rate=Fraction(24))
        torch.manual_seed(0)
        i1 = torch.rand(1, 512, 512, 3); i2 = torch.rand(1, 1024, 768, 3)
        f1 = torch.rand(80, 360, 640, 3); f2 = torch.rand(60, 480, 480, 3)
        a0 = {"waveform": torch.zeros(1, 2, 3 * 44100), "sample_rate": 44100}
        a1 = {"waveform": torch.zeros(1, 2, 4 * 44100), "sample_rate": 44100}
        def summ(cond, items):
            bl = cond[0][1].get("minimax_refs", [])
            return {"items": items,
                    "blocks": [[b["kind"], b.get("latent_t"), b.get("latent_h"),
                                b.get("latent_w"), b.get("ref_audio_t")] for b in bl]}
        cc = Clip()
        # keywords, not positions: Core master (0.37, 25.09.) made vae/audio_vae
        # optional and moved them behind ref_image_size -- positional calls
        # handed the prompt to `width`. Keywords hold on the pin and on master.
        out = C.MiniMaxH3ReferenceToVideo.execute(clip=cc, vae=Vae(), audio_vae=AVae(), prompt="p",
            width=1344, height=768, length=124, ref_image_size="match",
            ref_images={"ref_image_1": i1, "ref_image_2": i2},
            ref_videos={"ref_video_1": f1, "ref_video_2": f2},
            ref_video_audios={"ref_video_audio_1": a0},
            ref_audios={"ref_audio_1": a1})
        core_s = summ(out.result[0] if hasattr(out, "result") else out[0], cc.items)
        oc = Clip()
        lt = M.minimax_video_latent_t(124)
        class N:
            is_nested = True
            def unbind(self): return [torch.zeros(1, 24, lt, 768 // 16, 1344 // 16)]
        cond, _, _, _ = MR.ULSMiniMaxReference().build(oc, Vae(), {"samples": N()}, "p", "match",
            0.0, 0.0, 0.0, image_1=i1, image_2=i2, audio_vae=AVae(),
            video_1=Vid(f1), video_audio_1=a0, video_2=Vid(f2), audio_1=a1)
        ours_s = summ(cond, oc.items)
        print(json.dumps({"core": core_s, "ours": ours_s}))
    ''').replace("CORE", repr(core)).replace("NODES", repr(str(ROOT / "nodes")))
    env = dict(os.environ)
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/pyc")
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                           text=True, timeout=600, env=env, cwd=core)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if r.returncode != 0 or not line:
            _fail("A7 parity harness failed: %s" % (r.stderr.strip().splitlines()[-3:],))
        else:
            import json
            d = json.loads(line[-1])
            if d["core"]["items"] == d["ours"]["items"]:
                _ok("A7 tokenizer item order identical to Core's: %s" % d["ours"]["items"])
            else:
                _fail("A7 item order core %s ours %s" % (d["core"]["items"], d["ours"]["items"]))
            if d["core"]["blocks"] == d["ours"]["blocks"]:
                _ok("A7 DiT blocks identical to Core's (kind, latent_t, h, w, ref_audio_t)")
            else:
                _fail("A7 blocks core %s ours %s" % (d["core"]["blocks"], d["ours"]["blocks"]))
    except subprocess.TimeoutExpired:
        _fail("A7 parity harness timed out")

print()
if FAILS:
    print("test_v1016_minimax_ref_all_inputs: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1016_minimax_ref_all_inputs: PASS (A1-A7)")
