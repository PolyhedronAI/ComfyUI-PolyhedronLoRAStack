# -*- coding: ascii -*-
"""Guard v1019 -- Cine C2: Polyhedron Reference Board.

Driven with REAL files in a stub input folder (a PNG, an mp4 with a stereo
track, an mp4 without sound, a WAV) written here.

  B1  validate(): bad tag, slot-name tag, duplicate, soundtrack collision,
      unknown role / retention, over the 9/3/3 limit -> errors by name; a good
      board -> none.
  B2  build(): images load as IMAGE, videos as VideoFromFile, own soundtrack
      and audio as AUDIO dicts; definitions carry the official labels and
      markers in @tag form; info names every tile.
  B3  A file name leaving its folder ('../') is refused.
  B4  A missing file and 'own sound' on a silent video stop by name.
  B5  apply_refs(): board media fill the FREE slots in board order (a wired
      image_1 stays the wire's), a board budget reaches its slot, a video's
      soundtrack lands in video_audio_n, no free slot -> error by name.
  B6  Reference node with refs + a wire: the tokenizer gets the board image
      as <Picture 2>, @fox resolves without a tags line, info says where each
      tile went; a board tag that is also a tags line stops by name.
  B7  definitions_block(): <Subject k> numbered in board order for subject /
      scene / style, picture phrases for frame roles, the soundtrack line.

Script-style: exit 0 = pass.
"""
import importlib
import json
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _fail(m):
    FAILS.append(m)
    print("  FAIL  " + m)


def _ok(m):
    print("  ok    " + m)


try:
    import av
    from PIL import Image
except Exception:
    print("  note  PyAV/Pillow missing -- SKIPPED")
    sys.exit(0)

TMP = Path(tempfile.mkdtemp(prefix="v1019_"))
IN = TMP / "input"
(IN / "pls_board").mkdir(parents=True)
fp = types.ModuleType("folder_paths")
fp.get_input_directory = lambda: str(IN)
fp.get_output_directory = lambda: str(TMP / "output")
fp.get_temp_directory = lambda: str(TMP / "temp")


class VideoFromFile:
    def __init__(self, path):
        self.path = path

    def get_components(self):
        frames = []
        with av.open(self.path) as c:
            for fr in c.decode(video=0):
                frames.append(fr.to_ndarray(format="rgb24"))
        from fractions import Fraction
        return types.SimpleNamespace(images=torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0),
                                     audio=None, frame_rate=Fraction(24))


comfy = types.ModuleType("comfy")
cu = types.ModuleType("comfy.utils")
cu.common_upscale = lambda s, w, h, m, c: F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)
comfy.utils = cu
nh = types.ModuleType("node_helpers")
nh.conditioning_set_values = lambda cond, v: [[c[0], dict(c[1], **v)] for c in cond]
capi = types.ModuleType("comfy_api")
ii = types.ModuleType("comfy_api.input_impl")
ii.VideoFromFile = VideoFromFile
sys.modules.update({"folder_paths": fp, "comfy": comfy, "comfy.utils": cu, "node_helpers": nh,
                    "comfy_api": capi, "comfy_api.input_impl": ii})
pkg = types.ModuleType("plsn1019")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1019"] = pkg
RB = importlib.import_module("plsn1019.ph_reference_board")
MR = importlib.import_module("plsn1019.ph_minimax_ref")
HP = sys.modules["plsn1019.h3_prompt"]
M = sys.modules["plsn1019.uls_latent_math"]


def write_mp4(path, n=48, sound=True):
    with av.open(str(path), "w") as out:
        vs = out.add_stream("libx264", rate=24)
        vs.width, vs.height, vs.pix_fmt = 160, 96, "yuv420p"
        a = out.add_stream("aac", rate=32000) if sound else None
        if a is not None:
            a.layout = "stereo"
        for i in range(n):
            for p in vs.encode(av.VideoFrame.from_ndarray(np.full((96, 160, 3), 40 + i, np.uint8), format="rgb24")):
                out.mux(p)
        for p in vs.encode():
            out.mux(p)
        if a is not None:
            wav = np.zeros((2, int(32000 * n / 24)), np.float32)
            for s0 in range(0, wav.shape[1], 1024):
                fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(wav[:, s0:s0 + 1024]), format="fltp", layout="stereo")
                fr.sample_rate = 32000
                for p in a.encode(fr):
                    out.mux(p)
            for p in a.encode():
                out.mux(p)


def write_wav(path, sec=3.0):
    with av.open(str(path), "w") as out:
        a = out.add_stream("pcm_s16le", rate=32000)
        a.layout = "mono"
        x = (np.sin(np.arange(int(32000 * sec)) / 10.0) * 8000).astype(np.int16)[None]
        for s0 in range(0, x.shape[1], 1024):
            fr = av.AudioFrame.from_ndarray(np.ascontiguousarray(x[:, s0:s0 + 1024]), format="s16", layout="mono")
            fr.sample_rate = 32000
            for p in a.encode(fr):
                out.mux(p)
        for p in a.encode():
            out.mux(p)


Image.fromarray(np.full((300, 400, 3), 120, np.uint8)).save(IN / "pls_board" / "fox.png")
write_mp4(IN / "pls_board" / "walk.mp4", sound=True)
write_mp4(IN / "pls_board" / "silent.mp4", sound=False)
write_wav(IN / "pls_board" / "voice.wav")


def ref(name):
    return {"filename": name, "subfolder": "pls_board", "type": "input"}


GOOD = [
    {"kind": "image", "file": ref("fox.png"), "tag": "fox", "role": "subject", "desc": "a red fox with a white tail tip", "retention": "fully_preserved", "mp": 0.5},
    {"kind": "image", "file": ref("fox.png"), "tag": "start", "role": "first frame", "desc": "the fox at the forest edge", "retention": "fully_preserved"},
    {"kind": "video", "file": ref("walk.mp4"), "tag": "prev", "role": "continuation", "desc": "the fox walking left", "retention": "fully_preserved", "sound": "own"},
    {"kind": "audio", "file": ref("voice.wav"), "tag": "narr", "role": "voice", "desc": "a calm male narrator", "retention": "reference"},
]

# ========================================================================= B1
if RB.validate(GOOD) == []:
    _ok("B1 a good board validates clean")
else:
    _fail("B1 good board: %r" % RB.validate(GOOD))
bad = [
    ([dict(GOOD[0], tag="9fox")], "is not a name"),
    ([dict(GOOD[0], tag="image_2")], "is a slot name"),
    ([GOOD[0], dict(GOOD[1], tag="fox")], "is already tile 1"),
    ([GOOD[2], dict(GOOD[3], tag="prev_sound")], "prev_sound"),
    ([dict(GOOD[0], role="hero")], "role 'hero' is not one of"),
    ([dict(GOOD[3], retention="fully_preserved")], "retention 'fully_preserved' is not one of"),
    ([dict(GOOD[0], tag="t%d" % i) for i in range(10)], "10 image tiles"),
]
for board, needle in bad:
    errs = RB.validate(board)
    if any(needle in e for e in errs):
        _ok("B1 refused by name: ...%s" % needle)
    else:
        _fail("B1 expected %r in %r" % (needle, errs))

# ========================================================================= B2
refs, definitions, info = RB.ULSReferenceBoard().build(json.dumps(GOOD))
if len(refs["images"]) == 2 and refs["images"][0]["image"].shape == (1, 300, 400, 3) and refs["images"][0]["mp"] == 0.5:
    _ok("B2 images load as IMAGE [1,H,W,3] with their budget")
else:
    _fail("B2 images %r" % [(i["tag"], tuple(i["image"].shape)) for i in refs["images"]])
v = refs["videos"][0]
if isinstance(v["video"], VideoFromFile) and v["sound"] is not None and v["sound"]["waveform"].shape[1] == 2 and v["sound_tag"] == "prev_sound":
    _ok("B2 video as VideoFromFile, its own stereo soundtrack as @prev_sound")
else:
    _fail("B2 video %r" % v)
a = refs["audios"][0]["audio"]
if a["sample_rate"] == 32000 and abs(a["waveform"].shape[-1] / 32000.0 - 3.0) < 0.05:
    _ok("B2 audio as AUDIO dict, 3.0 s")
else:
    _fail("B2 audio %r" % (a["waveform"].shape,))
for needle in ("<Subject 1> is a red fox with a white tail tip, as shown in @fox.",
               "@start is the first frame of [Shot 1], showing the fox at the forest edge.",
               "@prev is the source video whose ending the target video continues: the fox walking left.",
               "@prev_sound is the soundtrack of @prev.",
               "@narr is a calm male narrator (voice).",
               "// retention_analysis", "<Subject 1>: fully_preserved - a red fox with a white tail tip",
               "@narr: reference - a calm male narrator"):
    if needle not in definitions:
        _fail("B2 definitions lack %r" % needle)
        break
else:
    _ok("B2 definitions: official labels and markers, in @tag form")
if all(("@" + t) in info for t in ("fox", "start", "prev", "narr")):
    _ok("B2 info names every tile")
else:
    _fail("B2 info %s" % info)

# ========================================================================= B3/B4
try:
    RB.file_path({"filename": "../../etc/passwd", "subfolder": "", "type": "input"})
    _fail("B3 a path leaving the folder was accepted")
except ValueError as e:
    _ok("B3 a file name leaving its folder is refused") if "refused" in str(e) else _fail("B3 %s" % e)
try:
    RB.ULSReferenceBoard().build(json.dumps([dict(GOOD[0], file=ref("gone.png"))]))
    _fail("B4 a missing file was accepted")
except ValueError as e:
    _ok("B4 a missing file stops by name") if "@fox" in str(e) and "gone" in str(e) else _fail("B4 %s" % e)
try:
    RB.ULSReferenceBoard().build(json.dumps([dict(GOOD[2], file=ref("silent.mp4"))]))
    _fail("B4 own sound on a silent video was accepted")
except ValueError as e:
    _ok("B4 'own sound' on a silent video stops by name") if "no audio track" in str(e) else _fail("B4 %s" % e)

# ========================================================================= B5
opt = {"image_1": torch.zeros(1, 8, 8, 3)}
mps = {n: 0.0 for n in range(1, 10)}
al, notes = RB.apply_refs(refs, opt, mps)
if al == {"fox": "image_2", "start": "image_3", "prev": "video_1", "prev_sound": "video_audio_1", "narr": "audio_1"} \
        and mps[2] == 0.5 and opt.get("video_audio_1") is v["sound"] and opt["image_1"].shape == (1, 8, 8, 3):
    _ok("B5 board fills FREE slots in order (wired image_1 kept), budget and soundtrack land")
else:
    _fail("B5 aliases %r mps2 %r" % (al, mps[2]))
full = {"image_%d" % n: torch.zeros(1) for n in range(1, 10)}
try:
    RB.apply_refs(refs, full, dict(mps))
    _fail("B5 no free slot was accepted")
except ValueError as e:
    _ok("B5 no free image slot stops by name") if "@fox" in str(e) else _fail("B5 %s" % e)


# ========================================================================= B6
class FakeNested:
    is_nested = True

    def __init__(self, v):
        self._v = v

    def unbind(self):
        return [self._v]


class Clip:
    prompt = None

    def tokenize(self, prompt, minimax_ref_items=None, **kw):
        self.prompt = prompt
        return {}

    def encode_from_tokens_scheduled(self, t):
        return [[torch.zeros(1), {}]]


class Vae:
    def encode(self, x):
        n, h, w = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
        return torch.zeros(1, 24, M.minimax_video_latent_t(n) if n > 1 else 1, h // 16, w // 16)


class AVae:
    audio_sample_rate = 32000

    def encode(self, w):
        return torch.zeros(1, 32, 2, int(w.shape[1]) // 800)


lt = M.minimax_video_latent_t(124)
lat = {"samples": FakeNested(torch.zeros(1, 24, lt, 768 // 16, 1344 // 16))}
clip = Clip()
out = MR.ULSMiniMaxReference().build(clip, Vae(), lat, "@image_1 meets @fox at @start; @prev goes on, @narr speaks.",
                                     "match", 0.0, 0.0, 0.0, image_1=torch.rand(1, 64, 64, 3),
                                     audio_vae=AVae(), refs=refs)
want = "<Picture 1> meets <Picture 2> at <Picture 3>; <Video 1> goes on, <Audio 2> speaks."
if clip.prompt == want:
    _ok("B6 board tags resolve without a tags line; the wired image stays <Picture 1>")
else:
    _fail("B6 tokenizer got %r" % clip.prompt)
if "board @fox -> image_2" in out[2] and "board @prev -> video_1 (+ @prev_sound -> video_audio_1)" in out[2]:
    _ok("B6 info says where each tile went")
else:
    _fail("B6 info %s" % out[2])
try:
    MR.ULSMiniMaxReference().build(Clip(), Vae(), lat, "@fox", "match", 0.0, 0.0, 0.0,
                                   audio_vae=AVae(), refs=refs, tags="fox = image_5")
    _fail("B6 a doubled tag was accepted")
except ValueError as e:
    _ok("B6 a board tag that is also a tags line stops by name") if "board tile AND a tags line" in str(e) else _fail("B6 %s" % e)

# ========================================================================= B7
d, k = HP.definitions_block([
    {"kind": "image", "tag": "a", "role": "scene", "desc": "a snowy forest."},
    {"kind": "image", "tag": "b", "role": "last frame", "desc": "the fox in the den"},
    {"kind": "image", "tag": "c", "role": "style", "desc": "etched ink"},
])
if d[0] == "<Subject 1> is the environment a snowy forest, as shown in @a." \
        and d[1] == "@b is the final frame of the video, showing the fox in the den." \
        and d[2].startswith("<Subject 2> is the visual style etched ink") and k[1].startswith("@b (last frame): fully_preserved"):
    _ok("B7 <Subject k> in board order, frame phrase for picture roles, trailing dots cleaned")
else:
    _fail("B7 %r %r" % (d, k))

shutil.rmtree(TMP, ignore_errors=True)
print()
if FAILS:
    print("test_v1019_reference_board: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1019_reference_board: PASS (B1-B7)")
