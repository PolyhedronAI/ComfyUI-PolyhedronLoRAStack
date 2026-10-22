# -*- coding: ascii -*-
"""Guard v1179 -- Drift Control: the held prefix masked anew at every sampler step.

(Frank 19.10.: "bau so viele Stufen alleine wie es geht" -- stage 2 on the v1178 mask foundation; the idea of
ethanfel's Drift-Control AV via Easy Media's context_swap, read, not copied.)

  D1 the weights: 1 on the far side, tapering to 0 at the join; the ratio sigma_next / sigma on the sampler's own
     schedule (duplicates dropped, 0 at the last step)
  D2 the live mask: written into the packed video part only, the sound part untouched, other shapes passed
     through, every call logged
  D3 install: a clone with the mask function and the apply-model wrapper; a model that already carries a live
     mask is refused by name
  D4 the node: the prefix read from the latent's hold (MiniMax Continue), prefix_frames overrides, nothing held
     is refused by name, the info names weights, Core and wiring
  D5 on BOTH Cores (tiny H3, CPU, a real 4-step run, hold vs hold + drift): the ratios 0.75 / 0.67 / 0.5 / 0;
     the model is fed a different far prefix at every step but the last, the join row the same at every step;
     the prefix comes out as the copied clip; Core master is TOLD the live mask (weights x ratio, its own
     quantisation), the pinned Core is not (core_per_row); the sound stays as held
  D6 registration, both baselines, ASCII
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
pe = types.ModuleType("comfy.patcher_extension")


class _WMP:
    APPLY_MODEL = "apply_model"


pe.WrappersMP = _WMP
cm.patcher_extension = pe
sys.modules["comfy"] = cm
sys.modules["comfy.nested_tensor"] = cnt
sys.modules["comfy.patcher_extension"] = pe
fp = types.ModuleType("folder_paths")
fp.get_temp_directory = lambda: tempfile.gettempdir()
sys.modules.setdefault("folder_paths", fp)
nh = types.ModuleType("node_helpers")
nh.conditioning_set_values = lambda cond, values: [[c[0], dict(c[1], **values)] for c in cond]
sys.modules.setdefault("node_helpers", nh)
PKG = "pls1179"
pkg = types.ModuleType(PKG)
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules[PKG] = pkg
AM = importlib.import_module(PKG + ".ph_av_mask")
DR = importlib.import_module(PKG + ".ph_minimax_drift")
M = importlib.import_module(PKG + ".uls_latent_math")

# ---- D1
_need(AM.drift_weights(7, 4) == [1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0] and AM.drift_weights(3, 9) == [2 / 3.0, 1 / 3.0, 0.0],
      "D1 weights: far 1, taper over 4 to 0 at the join; a taper longer than the prefix is capped")
sig = torch.tensor([1.0, 0.75, 0.75, 0.5, 0.0])
_need(AM.schedule(sig) == [1.0, 0.75, 0.5, 0.0] and abs(AM.drift_ratio(1.0, sig) - 0.75) < 1e-9
      and abs(AM.drift_ratio(0.75, sig) - 2 / 3.0) < 1e-9 and AM.drift_ratio(0.5, sig) == 0.0 and AM.drift_ratio(0.0, sig) == 0.0,
      "D1 the ratio sigma_next / sigma on the schedule (duplicates dropped, 0 at the last step)")

# ---- D2
st = AM.DriftState(7, 4)
vshape, ashape = (1, 24, 12, 2, 3), (1, 32, 2, 20)
inner = types.SimpleNamespace(latent_shapes=[vshape, ashape])
nv, na = 24 * 12 * 2 * 3, 32 * 2 * 20
packed = torch.ones(1, 1, nv + na)
packed[..., nv:nv + 5] = 0.0
packed[..., :nv].reshape(vshape)[:, :, :7] = 0.0   # AMENDED v1181: drift touches held rows only -- hold 7
packed0 = packed.clone()
out = st.mask_function(torch.tensor([1.0]), packed, {"sigmas": sig, "model": types.SimpleNamespace(inner_model=inner)})
vid = out[..., :nv].reshape(vshape)
_need([round(float(x), 4) for x in vid[0, 0, :8, 0, 0]] == [0.75, 0.75, 0.75, 0.5625, 0.375, 0.1875, 0.0, 1.0]
      and bool((vid[:, 5, :7] == vid[:, 0, :7]).all()) and float(out[..., nv:nv + 5].max()) == 0.0
      and float(out[..., nv + 5:].min()) == 1.0 and bool(torch.equal(packed, packed0)),
      "D2 step at sigma 1: the prefix = weights x 0.75 on every channel, the sound part untouched, the input not mutated")
_need(list(st.log) == [(1.0, 0.75)], "D2 the call is logged (sigma, ratio)")
other = types.SimpleNamespace(latent_shapes=[(1, 16, 4, 4)])
o2 = st.mask_function(torch.tensor([0.5]), packed, {"sigmas": sig, "model": types.SimpleNamespace(inner_model=other)})
_need(o2 is packed, "D2 a pack that is not the H3 AV pair passes through unchanged")


# ---- D3
class FakePatcher:
    def __init__(self, opts=None):
        self.model_options = dict(opts or {})
        self.wrappers = {}

    def clone(self):
        c = FakePatcher(self.model_options)
        return c

    def set_model_denoise_mask_function(self, f):
        self.model_options["denoise_mask_function"] = f

    def add_wrapper_with_key(self, kind, key, fn):
        self.wrappers[(kind, key)] = fn


base = FakePatcher()
pm, st3 = AM.install_drift(base, 7, 4)
_need(pm is not base and callable(pm.model_options.get("denoise_mask_function")) and ("apply_model", AM.DRIFT_KEY) in pm.wrappers
      and "denoise_mask_function" not in base.model_options and pm.model_options[AM.DRIFT_KEY] is st3,
      "D3 a clone carries the mask function and the apply-model wrapper; the input model untouched")
try:
    AM.install_drift(pm, 7, 4)
    _need(False, "D3 a model that already carries a live mask is refused")
except ValueError as e:
    _need("one per model" in str(e), "D3 a model that already carries a live mask is refused by name")


# ---- D4 the node  (AMENDED v1181: taper 16 = the last 5 tokens -> 1 1 0.8 0.6 0.4 0.2 0)
def av(frames, w=192, h=128):
    fc, lt, at = M.minimax_temporal_shape(frames)
    return {"samples": FakeNested((torch.zeros(1, 24, lt, h // 16, w // 16), torch.zeros(1, 32, 2, at)))}


lat = av(124)
held, _r = AM.hold_prefix(lat, torch.ones(1, 24, 7, 8, 12), torch.ones(1, 32, 2, 37), video_release=4)
node = DR.ULSMiniMaxDriftControl()
m4, l4, info4 = node.patch(base, held, 16, 0)
_need(l4 is held and m4.model_options[AM.DRIFT_KEY].prefix == 7 and "prefix 7 token(s) = frames 0..21 (from the latent's hold)" in info4
      and "1.00 1.00 0.80 0.60 0.40 0.20 0.00" in info4 and "wire `model` into the sampler" in info4,
      "D4 the prefix read from the hold (7 tokens), weights and wiring in the info, the latent passed on")
m5, _l5, info5 = node.patch(base, held, 8, 39)
_need(m5.model_options[AM.DRIFT_KEY].prefix == 7 and "cut to the hold's 7" in info5 and "0.67 0.33 0.00" in info5,   # AMENDED v1181
      "D4 prefix_frames 39 is cut to the hold (7 tokens, said); taper 8 frames = the last 3 tokens (9 frames)")
m6, _l6, info6 = node.patch(base, lat, 16, 0)   # AMENDED v1181: nothing held passes unpatched (one workflow, every scene)
_need(m6 is base and "passes unpatched" in info6, "D4 a latent with nothing held passes the model unpatched, said")

# ---- D5 both Cores
PROBE = r'''
import sys, json, types, importlib
NODES = sys.argv[1]; sys.argv = ["x", "--cpu"]
import comfy.options; comfy.options.enable_args_parsing()
import torch
torch.set_num_threads(1)   # v1180: one thread -- the probe runs next to other work; a lone NaN was seen under load
import comfy.supported_models as SM, comfy.model_patcher, comfy.sample, comfy.samplers, comfy.nested_tensor as NT
import comfy.ldm.minimax.model as mm
pkg = types.ModuleType("plsprobe"); pkg.__path__ = [NODES]; sys.modules["plsprobe"] = pkg
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
seen, xs = [], []
_f = mm.MiniMaxH3Model.forward
def f(self, x, timestep, context, transformer_options={}, **kw):
    dm = kw.get("denoise_mask")
    seen.append(None if dm is None else [round(float(v), 4) for v in dm[0, 0, :8, 0, 0]])
    xs.append(x[0][:, :, :7].detach().clone())
    return _f(self, x, timestep, context, transformer_options, **kw)
mm.MiniMaxH3Model.forward = f
def run(drift):
    torch.manual_seed(0)
    lat = {"samples": NT.NestedTensor((torch.randn(1, 24, 12, 4, 6), torch.randn(1, 32, 2, 67)))}
    hv = torch.randn(1, 24, 7, 4, 6); ha = torch.randn(1, 32, 2, 37)
    lat, _ = AM.hold_prefix(lat, hv, ha, video_release=0, audio_release=4)
    model, st = AM.install_drift(mp, AM.held_prefix(lat), 4) if drift else (mp, None)
    out = comfy.sample.sample_custom(model, comfy.sample.prepare_noise(lat["samples"], 1), 1.0,
                                     comfy.samplers.sampler_object("euler"), torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
                                     [[torch.randn(1, 8, 128), {}]], [[torch.randn(1, 8, 128), {}]], lat["samples"],
                                     noise_mask=lat["noise_mask"], seed=1)
    ov, oa = out.unbind()
    return ov, oa, hv, st
seen.clear(); xs.clear(); ov0, oa0, hv0, _ = run(False); s0, x0 = list(seen), list(xs)
seen.clear(); xs.clear(); ov1, oa1, hv1, st = run(True); s1, x1 = list(seen), list(xs)
print(json.dumps({"log": [[round(a, 4), round(b, 4)] for a, b in st.log], "per_row": st.per_row,
                  "seen_hold": s0, "seen_drift": s1,
                  "far_seen": [round(float((a[:, :, :3] - b[:, :, :3]).abs().mean()), 4) for a, b in zip(x0, x1)],
                  "join_seen": [float((a[:, :, 6] - b[:, :, 6]).abs().max()) for a, b in zip(x0, x1)],
                  "prefix_out": float((ov1[:, :, :7] - hv1[:, :, :7]).abs().max()),
                  "audio": float((oa1[..., :33] - oa0[..., :33]).abs().max())}))
'''
CORES = [("master", os.environ.get("PLS_CORE_MASTER") or "/home/claude/core_master",
          os.environ.get("PLS_CORE_PY") or "/home/claude/venv_master/bin/python", True),
         ("pin", os.environ.get("PLS_CORE_PIN") or "/home/claude/core_pin", sys.executable, False)]
for name, core, py, want in CORES:
    if not (os.path.isfile(os.path.join(core, "comfy", "ldm", "minimax", "model.py")) and os.path.isfile(py)):
        print("  note  Core %s not found -- D5 %s SKIPPED (PLS_CORE_MASTER / PLS_CORE_PIN / PLS_CORE_PY)" % (name, name))
        continue
    rr = subprocess.run([py, "-c", PROBE, str(ROOT / "nodes")], cwd=core, env=dict(os.environ, PYTHONPATH=core,
                        PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, timeout=900)
    try:
        J = json.loads(rr.stdout.strip().splitlines()[-1])
    except Exception:
        J = None
        print(rr.stdout[-800:], rr.stderr[-2000:])
    _need(J is not None, "D5 %s: hold and hold + drift sampled on a tiny H3" % name)
    if not J:
        continue
    _need([r for _s, r in J["log"]] == [0.75, 0.6667, 0.5, 0.0], "D5 %s: the ratios per step 0.75 0.67 0.5 0" % name)
    # master injects the held rows CLEAN (its scale_latent_inpaint), so the first step differs too; the pin
    # injects them noised to sigma -- at sigma 1 that is the noise itself, the same in both runs
    first = 0 if want else 1
    _need(all(d > 0.1 for d in J["far_seen"][first:3]) and J["far_seen"][3] == 0.0 and max(J["join_seen"]) == 0.0
          and (want or J["far_seen"][0] == 0.0),
          "D5 %s: the model is fed a different far prefix at steps %d-3 (%s), the same at the last; the join row the "
          "same at every step" % (name, first + 1, J["far_seen"]))
    _need(J["prefix_out"] < 1e-5 and J["audio"] == 0.0, "D5 %s: the prefix comes out as the copied clip, the sound as held" % name)
    if want:
        exp = [[0.75, 0.75, 0.75, 0.5625, 0.375, 0.1875, 0.0, 1.0], [0.668, 0.668, 0.668, 0.5, 0.3359, 0.168, 0.0, 1.0],
               [0.5, 0.5, 0.5, 0.375, 0.25, 0.125, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        _need(J["per_row"] is True and J["seen_drift"] == exp and J["seen_hold"] == [exp[3]] * 4,
              "D5 master: the model is TOLD the live mask each step (weights x ratio, quantised to 1/256), hold alone "
              "tells it the still mask")
    else:
        _need(J["per_row"] is False and J["seen_drift"] == [None] * 4, "D5 pin: the model is not told (sampler side only)")

# ---- D6
init = (ROOT / "__init__.py").read_text(encoding="utf-8")
ids = next(ROOT.glob("NODE_IDS_baseline_v*.txt")).read_text(encoding="utf-8")
wo = next(ROOT.glob("WIDGET_ORDER_baseline_v*.txt")).read_text(encoding="utf-8")
_need('NODE_CLASS_MAPPINGS["ULSMiniMaxDriftControl"] = ULSMiniMaxDriftControl' in init and "\nULSMiniMaxDriftControl\n" in ids
      and "\nULSMiniMaxDriftControl\ttaper_frames,prefix_frames\n" in wo, "D6 registered, in NODE_IDS and WIDGET_ORDER")
for p in ("nodes/ph_av_mask.py", "nodes/ph_minimax_drift.py", "tests/test_v1179_drift.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "D6 ASCII %s" % p)

if FAILS:
    print("[v1179] FAIL -- %d check(s)" % len(FAILS))
    sys.exit(1)
print("[v1179] PASS -- drift control: the held prefix masked anew at every step")
sys.exit(0)
