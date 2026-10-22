# -*- coding: ascii -*-
"""Guard v1178 -- the AV mask foundation: a held context in the latent, a locked soundtrack, the handoff rules.

(Frank 19.10.: "Bau uns das Fundament fuer alles weitere" -- after the Easy Media audit: the per-stream noise
masks on the H3 AV latent that the hard continuation, the audio lock, the drift control and the second pass
all stand on. Easy Media / ComfyUI-H3-Motion-Context are GPL: ideas read, no code taken.)

  A1 the time axis: 22 frames = 7 tokens (= uls_latent_math) = 37 audio frames; the ramp locks, then releases
     i/(r+1) INSIDE the prefix; frames_of_tokens inverts video_tokens on the grid
  A2 hold_prefix: the copy lands at the start of both streams, the masks are locked / ramped / free, a mask
     already on the latent stays (minimum), a prefix as long as the clip or of another size is refused by name
  A3 lock_audio: the window from `start`, 1 - strength, a sound past the end is cut and counted, the picture
     stream untouched
  A4 MiniMax Continue `hold`: K2L copies clip n's last 7 tokens and its sound BY TIME (the same cut as the guide)
     to the start, the guide still appended; K2 copies the encoded tail and the re-encoded last 22/24 s; "guide
     only" passes the SAME latent object; no context = nothing to hold, said; widgets appended after the old
  A5 MiniMax Audio Lock: the waveform laid into the clip (lead silence, silence / loop after, cut past the end),
     the latent's sound held from start_frame at strength, the picture free, the info names cut and Core
  A6 on BOTH Cores (tiny H3, CPU, a real sampler run): a latent from hold_prefix + lock_audio comes out with the
     held rows unchanged (< 1e-5) and the free rows changed; Core master runs held rows at their own timestep
     (its model code says so), the pinned Core does not -- core_per_row tells them apart
  A7 the handoff rules: a cut within 2 s of the join, an empty Shot 1, a cut in the last second before a
     continued join, a last shot that ends at rest -- each said; a clean scene says nothing; the timeline calls it
  A8 registration, both baselines, ASCII
Script-style: exit 0 = pass.
"""
import importlib
import math
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


# ---- fakes for the pure parts (the Core probe in A6 uses the real ones)
class FakeNested:
    is_nested = True

    def __init__(self, parts):
        self.parts = tuple(parts)

    def unbind(self):
        return self.parts


cnt = types.ModuleType("comfy.nested_tensor")
cnt.NestedTensor = FakeNested
cm = sys.modules.get("comfy") or types.ModuleType("comfy")
cm.nested_tensor = cnt
cu = types.ModuleType("comfy.utils")
cu.common_upscale = lambda s, w, h, m, c: torch.nn.functional.interpolate(s, size=(h, w), mode="bilinear")
cm.utils = cu
sys.modules["comfy"] = cm
sys.modules["comfy.nested_tensor"] = cnt
sys.modules.setdefault("comfy.utils", cu)
fp = types.ModuleType("folder_paths")
fp.get_temp_directory = lambda: tempfile.gettempdir()
fp.get_output_directory = lambda: tempfile.gettempdir()
fp.get_input_directory = lambda: tempfile.gettempdir()
sys.modules.setdefault("folder_paths", fp)
nh = types.ModuleType("node_helpers")
nh.conditioning_set_values = lambda cond, values: [[c[0], dict(c[1], **values)] for c in cond]
sys.modules.setdefault("node_helpers", nh)
PKG = "pls1178"
pkg = types.ModuleType(PKG)
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules[PKG] = pkg
AM = importlib.import_module(PKG + ".ph_av_mask")
M = importlib.import_module(PKG + ".uls_latent_math")
CO = importlib.import_module(PKG + ".ph_minimax_continue")
AL = importlib.import_module(PKG + ".ph_minimax_audio_lock")
T = importlib.import_module(PKG + ".cine_timeline")


def av(frames, w=192, h=128, fill=0.0):
    fc, lt, at = M.minimax_temporal_shape(frames)
    v = torch.full((1, 24, lt, h // 16, w // 16), fill)
    a = torch.full((1, 32, 2, at), fill)
    return {"samples": FakeNested((v, a))}


# ---- A1
_need(AM.video_tokens(22) == 7 == M.minimax_video_latent_t(22) and AM.video_tokens(5) == 2 and AM.video_tokens(1) == 1
      and AM.frames_of_tokens(7) == 22 and AM.audio_steps(22) == 37,   # AMENDED v1181: 22 (cyclic grid), not 25
      "A1 22 frames = 7 tokens (uls_latent_math agrees), 37 audio frames; 7 tokens cover exactly 22 frames")
vals, locked, r = AM.ramp(7, 4)
_need(locked == 3 and r == 4 and [round(x, 2) for x in vals] == [0, 0, 0, 0.2, 0.4, 0.6, 0.8],
      "A1 ramp(7, 4): 3 locked, then 0.2 0.4 0.6 0.8 -- the release ends INSIDE the prefix")
_need(AM.ramp(3, 9)[1:] == (0, 3) and AM.ramp(5, 0)[0] == [0.0] * 5, "A1 a release longer than the prefix is capped; 0 = all locked")

# ---- A2
lat = av(124)
v0, a0 = AM.streams(lat)
pv = torch.full((1, 24, 7, 8, 12), 3.0)
pa = torch.full((1, 32, 2, 37), 5.0)
out, rep = AM.hold_prefix(lat, pv, pa, video_release=4, audio_release=4)
ov, oa = out["samples"].unbind()
vm, am = out["noise_mask"].unbind()
_need(bool((ov[:, :, :7] == 3).all()) and bool((ov[:, :, 7:] == 0).all()) and bool((oa[..., :37] == 5).all())
      and bool((oa[..., 37:] == 0).all()) and bool((v0 == 0).all()),
      "A2 the copy lands at the start of both streams; the rest and the input latent untouched")
_need(tuple(vm.shape) == (1, 1, 37, 8, 12) and tuple(am.shape) == (1, 1, 2, 207)
      and [round(float(x), 2) for x in vm[0, 0, :8, 0, 0]] == [0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
      and float(am[..., :33].max()) == 0.0 and abs(float(am[0, 0, 0, 36]) - 0.8) < 1e-6 and float(am[..., 37:].min()) == 1.0,
      "A2 masks: picture 3 locked + 0.2..0.8 then free; sound 33 locked + release over 4, then free")
_need(rep["video"]["frames"] == 22 and rep["audio"]["steps"] == 37, "A2 the report counts tokens / frames / steps")
pre = dict(lat)
pre["noise_mask"] = FakeNested((torch.ones(1, 1, 37, 8, 12), torch.zeros(1, 1, 2, 207)))
out2, _r = AM.hold_prefix(pre, pv, None)
_v2, am2 = out2["noise_mask"].unbind()
_need(float(am2.max()) == 0.0, "A2 a mask already on the latent stays (minimum: a locked sound is not set free)")
for bad, why in ((torch.zeros(1, 24, 37, 8, 12), "as long as the clip"), (torch.zeros(1, 24, 7, 4, 4), "another size")):
    try:
        AM.hold_prefix(lat, bad, None)
        _need(False, "A2 a held picture %s is refused" % why)
    except ValueError as e:
        _need("must be" in str(e) or "same size" in str(e), "A2 a held picture %s is refused by name" % why)

# ---- A3
sz = torch.full((1, 32, 2, 300), 7.0)
out3, rep3 = AM.lock_audio(lat, sz, start=10, strength=0.75)
ov3, oa3 = out3["samples"].unbind()
vm3, am3 = out3["noise_mask"].unbind()
_need(rep3["steps"] == 197 and rep3["cut"] == 103 and bool((oa3[..., 10:] == 7).all()) and bool((oa3[..., :10] == 0).all())
      and abs(float(am3[..., 10:].max()) - 0.25) < 1e-6 and float(am3[..., :10].min()) == 1.0 and float(vm3.min()) == 1.0,
      "A3 lock_audio: from frame 10 at 1 - 0.75, 103 frames past the end cut and counted, the picture free")

# ---- A4 Continue
CO.core_has_guides = lambda: True


class FakeVae:
    def encode(self, x):
        t = x.shape[0]
        return torch.full((1, 24, 2 if t <= 5 else (t - 5) // 17 * 5 + 2, x.shape[1] // 16, x.shape[2] // 16), 9.0)


class FakeAudioVae:
    audio_sample_rate = 32000

    def encode(self, w):
        n = w.shape[1]
        return torch.arange(int(math.ceil(n / 800.0)), dtype=torch.float32).view(1, 1, 1, -1).expand(1, 32, 2, -1).clone()


CO._encode_audio = lambda avae, audio: (avae.encode(audio["waveform"][0]),
                                        int(math.ceil(audio["waveform"].shape[-1] / 800.0)))
node = CO.ULSMiniMaxContinue()
cond = [[torch.zeros(1), {}]]
tgt = av(158)
clip = av(124)
cv, ca = clip["samples"].unbind()
cv.copy_(torch.arange(cv.shape[2], dtype=torch.float32).view(1, 1, -1, 1, 1).expand_as(cv))
ca.copy_(torch.arange(ca.shape[3], dtype=torch.float32).view(1, 1, 1, -1).expand_as(ca))
K2L, K2, ANC = CO.MODES
o, lo, ov4, info = node.build(cond, tgt, K2L, 22, True, -1, 0, CO.HOLDS[1], 16, clip_latent=clip)
hv, ha = lo["samples"].unbind()
hvm, ham = lo["noise_mask"].unbind()
s0, s1, _off = CO.audio_tail_cut(124, 22, 207)
_need([int(x) for x in hv[0, 0, :8, 0, 0]] == [30, 31, 32, 33, 34, 35, 36, 0] and (s0, s1) == (170, 207)
      and [int(x) for x in ha[0, 0, 0, :3]] == [170, 171, 172] and int(ha[0, 0, 0, 36]) == 206 and int(ha[0, 0, 0, 37]) == 0,
      "A4 K2L hold: clip n's tokens 30..36 and its sound 170..206 (the guide's cut by time) at the start")
_need(len(o[0][1]["minimax_keyframes"]) == 1 and ov4 == 22 and "hold: the context is copied" in info
      and "(frames 0..21): 2 locked, 5 released over its last 17 frame(s)" in info and "37 audio frame(s)" in info,
      "A4 the guide still appended, overlap 22, the info names the hold (v1181: release 16 = the last 5 tokens, 17 "
      "frames; 2 locked; 37 audio frames)")
o5, lo5, _ov5, info5 = node.build(cond, tgt, K2L, 22, True, -1, 0, CO.HOLDS[0], 16, clip_latent=clip)
_need(lo5 is tgt and "hold" not in info5, "A4 'guide only' passes the SAME latent object (as before v1178)")
_o6, lo6, _ov6, info6 = node.build(cond, tgt, K2L, 22, True, -1, 0, CO.HOLDS[1], 0, clip_latent=clip)
_need(float(lo6["noise_mask"].unbind()[0][:, :, :7].max()) == 0.0, "A4 release_frames 0: the picture held hard up to the join")
tail = torch.rand(30, 128, 192, 3)
wav = {"waveform": torch.rand(1, 2, 32000 * 2), "sample_rate": 32000}
_o7, lo7, ov7, info7 = node.build(cond, tgt, K2, 22, True, -1, 18, CO.HOLDS[1], 16, vae=FakeVae(), audio_vae=FakeAudioVae(),
                                  tail=tail, tail_audio=wav)
hv7, ha7 = lo7["samples"].unbind()
_need(bool((hv7[:, :, :7] == 9).all()) and bool((hv7[:, :, 7:] == 0).all()) and int(ha7[0, 0, 0, 36]) == 36
      and int(ha7[0, 0, 0, 37]) == 0 and ov7 == 22,
      "A4 K2 hold: the encoded tail (7 tokens) and the last 22/24 s re-encoded (37 frames) at the start")
_o8, lo8, ov8, info8 = node.build(cond, tgt, K2, 22, True, -1, 0, CO.HOLDS[1], 16, vae=FakeVae(), audio_vae=FakeAudioVae())
_need(lo8 is tgt and "nothing to hold" in info8 and ov8 == 0, "A4 no context (the first scene): nothing to hold, said")
wo = next(ROOT.glob("WIDGET_ORDER_baseline_v*.txt")).read_text(encoding="utf-8")
_need("\nULSMiniMaxContinue\tmode,context_frames,with_audio,anchor_frame,audio_context,hold,release_frames\n" in wo,
      "A4 hold / release_frames appended after the old widgets (old workflows: guide only, as before)")

# ---- A5 Audio Lock
sr = 1000
w = torch.ones(1, 2, 3000)
pl, cut = AL.place_waveform(w, sr, 1.0, 5.0, "free")
_need(pl.shape[-1] == 4000 and float(pl[..., :1000].abs().max()) == 0 and cut == 0.0, "A5 place: 1 s lead silence, the 3 s sound, nothing after (free)")
pl2, _c2 = AL.place_waveform(w, sr, 1.0, 5.0, "silence")
pl3, _c3 = AL.place_waveform(w[..., :1500], sr, 0.0, 5.0, "loop")
pl4, cut4 = AL.place_waveform(w, sr, 3.0, 5.0, "free")
_need(pl2.shape[-1] == 5000 and float(pl2[..., 4000:].abs().max()) == 0 and pl3.shape[-1] == 5000
      and float(pl3.min()) == 1.0 and pl4.shape[-1] == 5000 and abs(cut4 - 1.0) < 1e-9,
      "A5 silence pads to the clip's end, loop repeats to it, a sound past the end is cut (1.0 s, counted)")
lock = AL.ULSMiniMaxAudioLock()
lt8 = av(124)
aud = {"waveform": torch.ones(1, 2, 32000 * 3), "sample_rate": 32000}
ref = types.ModuleType(PKG + ".ph_minimax_ref")   # the node encodes lazily through this module
ref.encode_ref_audio = lambda avae, audio: (
    torch.full((1, 32, 2, int(math.ceil(audio["waveform"].shape[-1] * 40 / 32000.0))), 4.0), 0)
sys.modules[PKG + ".ph_minimax_ref"] = ref
lo9, au9, info9 = lock.lock(lt8, None, aud, 1.0, 24, "free")
v9, a9 = lo9["samples"].unbind()
vm9, am9 = lo9["noise_mask"].unbind()
_need(bool((a9[..., 40:160] == 4).all()) and bool((a9[..., :40] == 0).all()) and float(am9[..., 40:160].max()) == 0.0
      and float(am9[..., :40].min()) == 1.0 and float(vm9.min()) == 1.0 and au9["waveform"].shape[-1] == 32000 * 4,
      "A5 the sound from frame 24 (audio frame 40) held at strength 1, before it free, the picture free; audio out laid in")
_need("free -- H3 carries on" in info9 and "Core" in info9 and "kept exactly" in info9, "A5 the info names the free rest and the Core")
try:
    lock.lock(lt8, None, aud, 1.0, 124, "free")
    _need(False, "A5 a start past the clip is refused")
except ValueError as e:
    _need("past the clip" in str(e), "A5 a start past the clip is refused by name")

# ---- A6 both Cores, a real sampler run on a tiny H3
PROBE = r'''
import sys, os, json, types, importlib
sys.argv = ["x", "--cpu"]
import comfy.options; comfy.options.enable_args_parsing()
import torch, inspect
torch.set_num_threads(1)   # v1180: one thread -- the probe runs next to other work; a lone NaN was seen under load
import comfy.supported_models as SM, comfy.model_patcher, comfy.sample, comfy.samplers, comfy.nested_tensor as NT
import comfy.ldm.minimax.model as mm
pkg = types.ModuleType("plsprobe"); pkg.__path__ = [sys.argv_nodes]; sys.modules["plsprobe"] = pkg
AM = importlib.import_module("plsprobe.ph_av_mask")
torch.manual_seed(0)
cfg = {"image_model": "minimax_h3", "hidden_size": 128, "num_layers": 2, "token_refiner_num_layers": 1,
       "num_attention_heads": 1, "attention_head_dim": 128, "ffn_hidden_size": 128, "text_dim": 128,
       "time_embed_hidden_size": 128, "time_embed_dim": 32}
mc = SM.MiniMaxH3(cfg); mc.set_inference_dtype(torch.float32, None)
m = mc.get_model({}, device="cpu")
with torch.no_grad():
    for p in m.parameters():
        p.copy_(torch.randn_like(p) * 0.02)
mp = comfy.model_patcher.ModelPatcher(m, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
v = torch.randn(1, 24, 12, 4, 6); a = torch.randn(1, 32, 2, 67)
lat = {"samples": NT.NestedTensor((v, a))}
held_v = torch.randn(1, 24, 7, 4, 6); held_a = torch.randn(1, 32, 2, 37)
lat, rep = AM.hold_prefix(lat, held_v, held_a, video_release=4, audio_release=4)
lat, rep2 = AM.lock_audio(lat, torch.randn(1, 32, 2, 20), start=45, strength=1.0)
x0v, x0a = [t.clone() for t in lat["samples"].unbind()]
noise = comfy.sample.prepare_noise(lat["samples"], 1)
cond = [[torch.randn(1, 8, 128), {}]]
out = comfy.sample.sample_custom(mp, noise, 1.0, comfy.samplers.sampler_object("euler"), torch.tensor([1.0, 0.6, 0.3, 0.0]),
                                 cond, cond, lat["samples"], noise_mask=lat["noise_mask"], seed=1)
ov, oa = out.unbind()
print(json.dumps({"v_held": float((ov[:, :, :3] - x0v[:, :, :3]).abs().max()),
                  "v_free": float((ov[:, :, 7:] - x0v[:, :, 7:]).abs().mean()),
                  "a_held": float((oa[..., :33] - x0a[..., :33]).abs().max()),
                  "a_lock": float((oa[..., 45:65] - x0a[..., 45:65]).abs().max()),
                  "a_free": float((oa[..., 37:45] - x0a[..., 37:45]).abs().mean()),
                  "per_row": AM.per_row_in_source(inspect.getsource(mm)), "core_per_row": AM.core_per_row()}))
'''.replace("sys.argv_nodes", repr(str(ROOT / "nodes")))
CORES = [("master", os.environ.get("PLS_CORE_MASTER") or "/home/claude/core_master",
          os.environ.get("PLS_CORE_PY") or "/home/claude/venv_master/bin/python", True),
         ("pin", os.environ.get("PLS_CORE_PIN") or "/home/claude/core_pin", sys.executable, False)]
import json  # noqa: E402
for name, core, py, want_rows in CORES:
    if not (os.path.isfile(os.path.join(core, "comfy", "ldm", "minimax", "model.py")) and os.path.isfile(py)):
        print("  note  Core %s not found -- A6 %s SKIPPED (PLS_CORE_MASTER / PLS_CORE_PIN / PLS_CORE_PY)" % (name, name))
        continue
    env = dict(os.environ, PYTHONPATH=core)
    rr = subprocess.run([py, "-c", PROBE], cwd=core, env=env, capture_output=True, text=True, timeout=900)
    try:
        J = json.loads(rr.stdout.strip().splitlines()[-1])
    except Exception:
        J = None
        print(rr.stdout[-800:], rr.stderr[-2000:])
    _need(J is not None, "A6 %s: the tiny H3 sampled a latent with held rows" % name)
    if J:
        _need(J["v_held"] < 1e-5 and J["a_held"] < 1e-5 and J["a_lock"] < 1e-5 and J["v_free"] > 0.1 and J["a_free"] > 0.1,
              "A6 %s: held picture / sound / locked sound unchanged (%.1e %.1e %.1e), free rows changed (%.2f %.2f)"
              % (name, J["v_held"], J["a_held"], J["a_lock"], J["v_free"], J["a_free"]))
        _need(J["per_row"] is want_rows and J["core_per_row"] is want_rows,
              "A6 %s: the model runs held rows at their own timestep = %s (core_per_row agrees)" % (name, want_rows))

# ---- A7 the handoff rules
sc = {"seconds": 6.0 + 22 / 24.0, "head": 22, "shots": [{"at": 0, "text": "she"}, {"at": 22 / 24.0 + 1.0, "text": "cut"}]}
n7 = T.context_notes(sc, True, False)
_need(len(n7) == 2 and "after the join" in n7[0] and "freeze" in n7[1], "A7 incoming: a cut 1.0 s after the join and an empty Shot 1 are said")
ok7 = {"seconds": 6.0 + 22 / 24.0, "head": 22, "shots": [{"at": 0, "text": "she keeps walking, breathing hard"},
                                                         {"at": 22 / 24.0 + 2.5, "text": "the camera cuts to her hands"}]}
_need(T.context_notes(ok7, True, True) == [], "A7 a clean continued scene, continued again, says nothing")
out7 = {"seconds": 6.0, "shots": [{"at": 0, "text": "he runs"}, {"at": 5.5, "text": "the car slows and stops at the gate"}]}
n8 = T.context_notes(out7, False, True)
_need(len(n8) == 2 and "last 1.0 s" in n8[0] and "'slows', 'stops'" in n8[1].replace("stops', 'slows", "slows', 'stops"),
      "A7 outgoing: a cut in the last second and a last shot that ends at rest are said")
src = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
_need("T.context_notes(target, carried and motion, outgoing)" in src, "A7 the timeline checks every continued / continuing scene")

# ---- A8
init = (ROOT / "__init__.py").read_text(encoding="utf-8")
ids = next(ROOT.glob("NODE_IDS_baseline_v*.txt")).read_text(encoding="utf-8")
_need('NODE_CLASS_MAPPINGS["ULSMiniMaxAudioLock"] = ULSMiniMaxAudioLock' in init and "ULSMiniMaxAudioLock" in ids
      and "\nULSMiniMaxAudioLock\tstrength,start_frame,after\n" in wo,
      "A8 Audio Lock registered, in NODE_IDS and WIDGET_ORDER")
for p in ("nodes/ph_av_mask.py", "nodes/ph_minimax_audio_lock.py", "nodes/ph_minimax_continue.py", "nodes/cine_timeline.py",
          "nodes/ph_cine_timeline.py", "tests/test_v1178_av_mask.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "A8 ASCII %s" % p)

if FAILS:
    print("[v1178] FAIL -- %d check(s)" % len(FAILS))
    sys.exit(1)
print("[v1178] PASS -- the AV mask foundation: hold, lock, handoff rules")
sys.exit(0)
