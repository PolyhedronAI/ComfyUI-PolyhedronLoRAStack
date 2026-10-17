# -*- coding: ascii -*-
"""Guard v1112 -- the Media Loader decodes only what is wired, and checks RAM first
(Frank, 12.10.2026: a 1080p music video into the voice-swap workflow -> "Unable to allocate
46.3 GiB for an array with shape (2000, 1080, 1920, 3)" after 14 s of decode, though only
`audio` was wired).

  R1  _audio_only: true only when `audio` is wired and nothing that needs frames; no "outs"
      (old frontend, API prompt) -> false = decode as before.
  R2  _check_frame_budget: refuses BEFORE decoding when the frames do not fit, naming the
      size, the free RAM and the levers; passes when they fit or RAM is unreadable.
  R3  OpenCV decode straight into one float32 block: the frames are bit-identical to the old
      uint8 -> astype -> /255 result, also when the container under-reports its frame count.
  R4  a real video with sound, only `audio` wired: no frame decoded (the decoder is never
      called), the audio comes in FULL length (not cut to the frame cap), from the video's own
      track when nothing is paired; the paired audio wins when there is one.
  R5  a frame output wired too -> the decoder runs (old behaviour); the cache follows the
      wiring (IS_CHANGED differs when only "outs" differ).
  R6  the frontend writes the wired slots into the PROMPT's media_ref (serializeValue), the
      saved workflow keeps the plain value; the helper run in node.

Script-style: exit 0 = pass.
"""
import importlib
import json
import os
import subprocess
import sys
import tempfile
import types

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def need(c, m):
    print(("  ok    " if c else "  FAIL  ") + m)
    if not c:
        FAILS.append(m)


TMP = tempfile.mkdtemp(prefix="pls1112_")
fp = types.ModuleType("folder_paths")
fp.get_input_directory = lambda: TMP
fp.get_output_directory = lambda: TMP
fp.get_temp_directory = lambda: TMP
sys.modules.setdefault("folder_paths", fp)
pkg = types.ModuleType("pls1112")
pkg.__path__ = [os.path.join(ROOT, "nodes")]
sys.modules["pls1112"] = pkg
M = importlib.import_module("pls1112.ph_media_loader")

# ------------------------------------------------------------ R1
need(M._audio_only({"outs": [3]}) and M._audio_only({"outs": [3, 9, 11]}), "R1 audio (+ filename / video_path) -> audio only")
need(not M._audio_only({"outs": [0, 3]}) and not M._audio_only({"outs": [3, 5]}) and not M._audio_only({"outs": [4]}),
     "R1 image / frame_count / video_audio wired -> frames needed")
need(not M._audio_only({}) and not M._audio_only({"outs": "3"}) and not M._audio_only({"outs": []})
     and not M._audio_only({"outs": ["x"]}), "R1 no or malformed outs -> decode as before")

# ------------------------------------------------------------ R2
real_free = M._free_ram
M._free_ram = lambda: 16 * 1024 ** 3
try:
    M._check_frame_budget(2000, 1080, 1920, 3, "/x/Cherry.mp4")
    need(False, "R2 2000 x 1080p on 16 GB free -> refused")
except RuntimeError as e:
    s = str(e)
    need("Cherry.mp4" in s and "49.8 GB" in s and "2000 x 1920x1080" in s and "frame_load_cap" in s
         and "audio output" in s, "R2 refused before decoding, naming size, file and levers: %s" % s[:90])
need(M._check_frame_budget(81, 720, 1280, 3, "/x/a.mp4") > 0, "R2 81 x 720p fits -> passes")
M._free_ram = lambda: 0
need(M._check_frame_budget(2000, 1080, 1920, 3, "/x/a.mp4") > 0, "R2 RAM unreadable -> no check")
M._free_ram = real_free

# ------------------------------------------------------------ a test video with sound (PyAV)
import av  # noqa: E402

VID = os.path.join(TMP, "song clip.mp4")
W, H, FPS, NF, SR, SECS = 64, 48, 10, 30, 22050, 6.0
with av.open(VID, "w") as c:
    vs = c.add_stream("mpeg4", rate=FPS)
    vs.width, vs.height, vs.pix_fmt = W, H, "yuv420p"
    as_ = c.add_stream("aac", rate=SR)
    as_.layout = "mono"
    for i in range(NF):
        img = np.zeros((H, W, 3), np.uint8)
        img[:, :, 0] = (i * 8) % 256
        img[:, : (i * 2) % W, 1] = 200
        for p in vs.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            c.mux(p)
    for p in vs.encode():
        c.mux(p)
    t = np.arange(int(SR * SECS)) / SR
    pcm = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    pts = 0
    for k in range(0, len(pcm), 1024):
        fr = av.AudioFrame.from_ndarray(pcm[None, k:k + 1024], format="flt", layout="mono")
        fr.sample_rate, fr.pts = SR, pts
        pts += fr.samples
        for p in as_.encode(fr):
            c.mux(p)
    for p in as_.encode():
        c.mux(p)

# ------------------------------------------------------------ R3
import cv2  # noqa: E402


def old_way(path):
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(fr, axis=0).astype(np.float32) / 255.0


ref_stack = old_way(VID)
img, mask, n, fps = M._load_video_cv2(VID, 0, 0, 0.0)
need(n == ref_stack.shape[0] and img.shape == (n, H, W, 3) and np.array_equal(img.numpy(), ref_stack),
     "R3 preallocated decode == the old result, bit for bit (%d frames)" % n)
real_vc = cv2.VideoCapture


class Under:
    """the container claims fewer frames than it has -> the rest goes the list way"""
    def __init__(self, *a):
        self.c = real_vc(*a)

    def get(self, prop):
        v = self.c.get(prop)
        return 7 if prop == cv2.CAP_PROP_FRAME_COUNT else v

    def __getattr__(self, k):
        return getattr(self.c, k)


M.cv2.VideoCapture = Under
img2, _, n2, _ = M._load_video_cv2(VID, 0, 0, 0.0)
M.cv2.VideoCapture = real_vc
need(n2 == n and np.array_equal(img2.numpy(), ref_stack), "R3 under-reported count: block + rest, still identical")
src = open(os.path.join(ROOT, "nodes", "ph_media_loader.py"), encoding="utf-8").read()
need("np.stack(frames, axis=0).astype(np.float32) / 255.0" not in src, "R3 no decoder holds the frames three times any more")

# ------------------------------------------------------------ R4
calls = []
real_lv = M._load_video
M._load_video = lambda *a, **k: (calls.append(a), real_lv(*a, **k))[1]
node = M.ULSMediaLoader()
base = {"folder": TMP, "file": "song clip.mp4", "kind": "video", "mtime": 0, "vtrimStart": 0, "vtrimEnd": 0}
out = node.load(json.dumps(dict(base, outs=[3])), 0, 0, 0.0, "", "", "7")
aud = out[3]
dur = aud["waveform"].shape[-1] / float(aud["sample_rate"]) if aud else 0
need(not calls, "R4 only audio wired -> the frame decoder is never called")
need(aud is not None and abs(dur - SECS) < 0.15, "R4 the video's own track in full length: %.2f s (frames would be %.1f s)"
     % (dur, NF / float(FPS)))
need(len(out) == 12 and out[9] == "song clip.mp4" and out[11] == os.path.realpath(VID), "R4 the 12-output contract holds")
WAV = os.path.join(TMP, "other.wav")
import wave  # noqa: E402
with wave.open(WAV, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(8000)
    w.writeframes((np.sin(np.arange(8000 * 2) / 5.0) * 9000).astype("<i2").tobytes())
out = node.load(json.dumps(dict(base, outs=[3], audio={"folder": TMP, "file": "other.wav"})), 0, 0, 0.0, "", "", "7")
need(not calls and out[3]["sample_rate"] == 8000 and out[3]["waveform"].shape[-1] == 16000,
     "R4 a paired audio wins over the own track")

# ------------------------------------------------------------ R5
out = node.load(json.dumps(dict(base, outs=[0, 3])), 0, 0, 0.0, "", "", "7")
need(len(calls) == 1 and out[0].shape[0] == n, "R5 image wired too -> the decoder runs (%d frames)" % out[0].shape[0])
calls.clear()
out = node.load(json.dumps(base), 0, 0, 0.0, "", "", "7")
need(len(calls) == 1, "R5 no outs (old frontend) -> decoded as before")
M._load_video = real_lv
a = M.ULSMediaLoader.IS_CHANGED(json.dumps(dict(base, outs=[3])))
b = M.ULSMediaLoader.IS_CHANGED(json.dumps(dict(base, outs=[0, 3])))
need(a != b, "R5 rewiring changes the cache key")

# ------------------------------------------------------------ R6
js = open(os.path.join(ROOT, "web", "js", "ph_media_loader.js"), encoding="utf-8").read()
need("rw.serializeValue = async () => plsWithWiredOuts(rw.value, node);" in js,
     "R6 serializeValue feeds the prompt; widget.value (the saved workflow) stays plain")
i = js.find("function plsWithWiredOuts")
j = js.find("\n}\n", i)
fn = js[i:j + 2]
probe = fn + """
const node = {outputs: [{links: []}, {links: null}, {links: []}, {links: [5]}, {links: []}, {links: []},
                        {links: []}, {links: []}, {links: []}, {links: [9, 10]}]};
const r = JSON.parse(plsWithWiredOuts(JSON.stringify({file: "a.mp4"}), node));
console.log(JSON.stringify([r.outs, r.file, plsWithWiredOuts("", node), plsWithWiredOuts("x", node)]));
"""
pf = os.path.join(TMP, "probe.js")
open(pf, "w").write(probe)
res = subprocess.run(["node", pf], capture_output=True, text=True)
need(res.returncode == 0 and json.loads(res.stdout) == [[3, 9], "a.mp4", "", "x"],
     "R6 the helper writes the wired slots, keeps the rest, passes non-JSON: %s" % (res.stdout.strip() or res.stderr[:120]))

if FAILS:
    print("[v1112] FAIL -- %d check(s)" % len(FAILS))
    sys.exit(1)
print("[v1112] PASS -- the Media Loader decodes only what is wired, and checks RAM first")
sys.exit(0)
