# -*- coding: ascii -*-
"""Guard v1180 -- Second Pass Hold: the second pass of a continued scene keeps its join at full size.

(Frank 19.10.: "bau so viele Stufen alleine wie es geht" -- stage 3 on the v1178 mask foundation; the idea of
Easy Media's separate low- / high-resolution context, read, not copied.)

  P1 the schedule: 0.50 0.30 0.14 0.06 0 by default, parsed and checked like the Sigma List (a rising list refused)
  P2 from `tail`: the frames resized to the LIFTED size and encoded there, copied to the start and held
     (3 locked + 4 released at 22 frames), the sound frozen (its whole mask 0), the info names all of it
  P3 from `clip_latent`: its last tokens on the chunk grid, no VAE; a clip latent of another size refused by name
  P4 no scene before: only the sound frozen, said; freeze off and nothing held = the SAME latent object
  P5 on BOTH Cores (tiny H3, CPU): a lifted latent with the join held and the sound frozen, sampled on the
     0.5 ... 0 schedule: the held rows and the WHOLE sound come out unchanged, the free picture rows changed
  P6 registration, both baselines, ASCII
Script-style: exit 0 = pass.
"""
import importlib
import json
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
SEEN = {}


def _up(s, w, h, m, c):
    SEEN["size"] = (w, h)
    return torch.nn.functional.interpolate(s, size=(h, w), mode="bilinear")


cu.common_upscale = _up
cm.utils = cu
sys.modules["comfy"] = cm
sys.modules["comfy.nested_tensor"] = cnt
sys.modules["comfy.utils"] = cu
fp = types.ModuleType("folder_paths")
fp.get_temp_directory = lambda: tempfile.gettempdir()
sys.modules.setdefault("folder_paths", fp)
nh = types.ModuleType("node_helpers")
nh.conditioning_set_values = lambda cond, values: [[c[0], dict(c[1], **values)] for c in cond]
sys.modules.setdefault("node_helpers", nh)
PKG = "pls1180"
pkg = types.ModuleType(PKG)
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules[PKG] = pkg
AM = importlib.import_module(PKG + ".ph_av_mask")
PH = importlib.import_module(PKG + ".ph_minimax_pass_hold")
M = importlib.import_module(PKG + ".uls_latent_math")


def av(frames, w, h, fill=0.0):
    fc, lt, at = M.minimax_temporal_shape(frames)
    return {"samples": FakeNested((torch.full((1, 24, lt, h // 16, w // 16), fill), torch.full((1, 32, 2, at), fill)))}


class FakeVae:
    def encode(self, x):
        t = x.shape[0]
        SEEN["enc"] = tuple(x.shape)
        SEEN["first"] = round(float(x[0, 0, 0, 0]), 4)
        return torch.full((1, 24, (t - 5) // 17 * 5 + 2, x.shape[1] // 16, x.shape[2] // 16), 9.0)


# ---- P1
_need(PH.parse_sigmas(PH.DEFAULT_SIGMAS) == [0.5, 0.3, 0.14, 0.06, 0.0], "P1 the default schedule 0.50 0.30 0.14 0.06 0")
try:
    PH.parse_sigmas("0.1, 0.5, 0")
    _need(False, "P1 a rising schedule is refused")
except ValueError:
    _need(True, "P1 a rising schedule is refused (the Sigma List's contract)")

node = PH.ULSMiniMaxSecondPassHold()
lifted = av(124, 384, 256)
tail = torch.arange(30, dtype=torch.float32).view(30, 1, 1, 1).expand(30, 96, 160, 3) / 100.0   # frame k = k/100
# ---- P2
out, sg, info = node.hold(lifted, 22, 16, True, PH.DEFAULT_SIGMAS, vae=FakeVae(), tail=tail)
v, a = out["samples"].unbind()
vm, am = out["noise_mask"].unbind()
_need(SEEN.get("size") == (384, 256) and SEEN.get("enc") == (22, 256, 384, 3) and SEEN.get("first") == 0.08
      and bool((v[:, :, :7] == 9).all())
      and bool((v[:, :, 7:] == 0).all()),
      "P2 the LAST 22 frames (from frame 8) resized to the LIFTED 384x256 and encoded there; 7 tokens at the start")
_need([round(float(x), 2) for x in vm[0, 0, :8, 0, 0]] == [0, 0, 0.17, 0.33, 0.5, 0.67, 0.83, 1.0] and float(am.max()) == 0.0,   # AMENDED v1181
      "P2 held 2 locked + 5 released (release 16 = the last 5 tokens, 17 frames), then free; the sound frozen")
_need(isinstance(sg, torch.Tensor) and [round(float(x), 2) for x in sg] == [0.5, 0.3, 0.14, 0.06, 0.0]
      and "frozen" in info and "encoded at 384x256" in info and "2 locked, 5 released over its last 17 frame(s)" in info,
      "P2 sigmas out, and the info names the source, the hold and the frozen sound")
# ---- P3
clip = av(124, 384, 256)
cv, _ca = clip["samples"].unbind()
cv.copy_(torch.arange(cv.shape[2], dtype=torch.float32).view(1, 1, -1, 1, 1).expand_as(cv))
out3, _s3, info3 = node.hold(lifted, 22, 0, True, PH.DEFAULT_SIGMAS, clip_latent=clip)
v3, _a3 = out3["samples"].unbind()
vm3, _am3 = out3["noise_mask"].unbind()
_need([int(x) for x in v3[0, 0, :8, 0, 0]] == [30, 31, 32, 33, 34, 35, 36, 0] and float(vm3[:, :, :7].max()) == 0.0
      and "tokens 30..36, no VAE" in info3, "P3 from the final latent: tokens 30..36 at the start, held hard (release 0)")
try:
    node.hold(lifted, 22, 16, True, PH.DEFAULT_SIGMAS, clip_latent=av(124, 192, 128))
    _need(False, "P3 a clip latent of another size is refused")
except ValueError as e:
    _need("give `tail`" in str(e), "P3 a clip latent of another size is refused by name (tail is the way)")
# ---- P4
out4, _s4, info4 = node.hold(lifted, 22, 16, True, PH.DEFAULT_SIGMAS)
vm4, am4 = out4["noise_mask"].unbind()
_need(float(vm4.min()) == 1.0 and float(am4.max()) == 0.0 and "first scene" in info4,
      "P4 no scene before: the picture free, only the sound frozen, said")
out5, _s5, info5 = node.hold(lifted, 22, 16, False, PH.DEFAULT_SIGMAS)
_need(out5 is lifted and "sound: free" in info5, "P4 freeze off and nothing to hold: the SAME latent object")

# ---- P5 both Cores
PROBE = r'''
import sys, json, types, importlib
NODES = sys.argv[1]; sys.argv = ["x", "--cpu"]
import comfy.options; comfy.options.enable_args_parsing()
import torch
torch.set_num_threads(1)   # v1180: one thread -- the probe runs next to other work; a lone NaN was seen under load
import comfy.supported_models as SM, comfy.model_patcher, comfy.sample, comfy.samplers, comfy.nested_tensor as NT
pkg = types.ModuleType("plsprobe"); pkg.__path__ = [NODES]; sys.modules["plsprobe"] = pkg
PH = importlib.import_module("plsprobe.ph_minimax_pass_hold")
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
lifted = {"samples": NT.NestedTensor((torch.randn(1, 24, 12, 4, 6), torch.randn(1, 32, 2, 67)))}
prev = {"samples": NT.NestedTensor((torch.randn(1, 24, 12, 4, 6), torch.randn(1, 32, 2, 67)))}
lat, sig, info = PH.ULSMiniMaxSecondPassHold().hold(lifted, 22, 16, True, PH.DEFAULT_SIGMAS, clip_latent=prev)
x0v, x0a = [t.clone() for t in lat["samples"].unbind()]
noise = comfy.sample.prepare_noise(lat["samples"], 1)
nan_in = [bool(torch.isnan(t).any()) for t in list(lat["samples"].unbind()) + list(lat["noise_mask"].unbind()) + list(noise.unbind())]
out = comfy.sample.sample_custom(mp, noise, 1.0, comfy.samplers.sampler_object("euler"),
                                 sig, [[torch.randn(1, 8, 128), {}]], [[torch.randn(1, 8, 128), {}]], lat["samples"],
                                 noise_mask=lat["noise_mask"], seed=1)
ov, oa = out.unbind()
pv = prev["samples"].unbind()[0]
print(json.dumps({"nan_in": nan_in, "nan_param": any(bool(torch.isnan(q).any()) for q in m.parameters()),
                  "held": float((ov[:, :, :2] - pv[:, :, 5:7]).abs().max()),   # AMENDED v1181: 2 locked rows
                  "audio": float((oa - x0a).abs().max()),
                  "free": float((ov[:, :, 7:] - x0v[:, :, 7:]).abs().mean())}))
'''
CORES = [("master", os.environ.get("PLS_CORE_MASTER") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")),
          os.environ.get("PLS_CORE_PY") or sys.executable),
         ("pin", os.environ.get("PLS_CORE_PIN") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")), sys.executable)]
for name, core, py in CORES:
    if not (os.path.isfile(os.path.join(core, "comfy", "ldm", "minimax", "model.py")) and os.path.isfile(py)):
        print("  note  Core %s not found -- P5 %s SKIPPED (PLS_CORE_MASTER / PLS_CORE_PIN / PLS_CORE_PY)" % (name, name))
        continue
    rr = subprocess.run([py, "-c", PROBE, str(ROOT / "nodes")], cwd=core, env=dict(os.environ, PYTHONPATH=core,
                        PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, timeout=900)
    try:
        J = json.loads(rr.stdout.strip().splitlines()[-1])
    except Exception:
        J = None
        print(rr.stdout[-800:], rr.stderr[-2000:])
    _need(J is not None and J["held"] < 1e-5 and J["audio"] < 1e-5 and J["free"] > 0.05,
          "P5 %s: a second pass on 0.5..0 -- the held join and the whole sound unchanged, the free picture changed %s"
          % (name, J))

# ---- P6
init = (ROOT / "__init__.py").read_text(encoding="utf-8")
ids = next(ROOT.glob("NODE_IDS_baseline_v*.txt")).read_text(encoding="utf-8")
wo = next(ROOT.glob("WIDGET_ORDER_baseline_v*.txt")).read_text(encoding="utf-8")
_need('NODE_CLASS_MAPPINGS["ULSMiniMaxSecondPassHold"] = ULSMiniMaxSecondPassHold' in init and "\nULSMiniMaxSecondPassHold\n" in ids
      and "\nULSMiniMaxSecondPassHold\tcontext_frames,release_frames,freeze_audio,sigmas\n" in wo,
      "P6 registered, in NODE_IDS and WIDGET_ORDER")
for p in ("nodes/ph_minimax_pass_hold.py", "tests/test_v1180_pass_hold.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "P6 ASCII %s" % p)

if FAILS:
    print("[v1180] FAIL -- %d check(s)" % len(FAILS))
    sys.exit(1)
print("[v1180] PASS -- the second pass of a continued scene keeps its join")
sys.exit(0)
