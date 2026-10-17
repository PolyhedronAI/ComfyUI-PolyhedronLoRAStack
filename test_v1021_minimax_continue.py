# -*- coding: ascii -*-
"""Guard v1021 -- Cine C5: Polyhedron MiniMax Continue.

  N1  pure rules: AddGuide's snap (<5 -> 1, else 17k+5), its index rule and
      its "does not fit" refusal; the latent window of the last 17m+5 frames
      starts on the 17-frame chunk grid (else refused); the audio window is
      cut by TIME (5/3 audio latent frames per pixel frame).
  N2  refusals by name: a Core without MiniMaxH3AddGuide (the pin), a latent
      that is not H3 AV, K2L without clip_latent or with another size, K2
      with a short tail or no vae, an anchor
      clip that does not fit.
  N3  K2L driven: the guide IS clip n's last 7 tokens (no VAE called), the
      sound its audio latent from frame (124-22)*5/3 on, index 0, overlap 22;
      keyframes and references already in the conditioning survive.
  N4  K2 driven: the LAST 22 frames go to the VAE at the target size, the
      sound is the last 22/24 s, cropped to the target's audio length.
  N5  anchors: -1 = last frame, a 30-frame clip snaps to 22, a guide inside
      the context is reported.
  N6  Core half (only where Core has MiniMaxH3AddGuide, e.g. master 0.37):
      K2 and an anchor clip are IDENTICAL to Core's own Add Guide (index,
      video latent, audio latent), and Core's real encode_temporal shows the
      K2L claim: encode(clip)[-7 tokens] == encode(last 22 frames).

Script-style: exit 0 = pass.
"""
import importlib
import os
import subprocess
import sys
import textwrap
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


def _raises(fn, needle, label):
    try:
        fn()
    except ValueError as e:
        if needle in str(e):
            _ok(label)
        else:
            _fail("%s -- wrong message: %s" % (label, e))
        return
    except Exception as e:            # a crash is a FAIL, not a refusal
        _fail("%s -- crashed instead: %r" % (label, e))
        return
    _fail("%s -- accepted" % label)


# -------------------------------------------------------------- stubs + load
comfy = types.ModuleType("comfy")
cu = types.ModuleType("comfy.utils")
cu.common_upscale = lambda s, w, h, m, c: F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)
comfy.utils = cu
nh = types.ModuleType("node_helpers")
nh.conditioning_set_values = lambda cond, v: [[c[0], dict(c[1], **v)] for c in cond]
ce = types.ModuleType("comfy_extras")
core_h3 = types.ModuleType("comfy_extras.nodes_minimax_h3")
core_h3.MiniMaxH3AddGuide = object          # "master": guides exist
ce.nodes_minimax_h3 = core_h3
sys.modules.update({"comfy": comfy, "comfy.utils": cu, "node_helpers": nh,
                    "comfy_extras": ce, "comfy_extras.nodes_minimax_h3": core_h3})
pkg = types.ModuleType("plsn1021")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1021"] = pkg
CO = importlib.import_module("plsn1021.ph_minimax_continue")
# v1061: the guide verdict reads the H3 MODEL code only (no comfy_extras guess); the stub world sets it.
CO.model_has_guides = lambda: True          # "master": guides exist
M = sys.modules["plsn1021.uls_latent_math"]


class Nested:
    is_nested = True

    def __init__(self, *t):
        self.t = t

    def unbind(self):
        return list(self.t)


def av_latent(frames, w=1344, h=768, fill=None):
    t = M.minimax_video_latent_t(frames)
    a = int(round(frames / 24.0 * 40))
    v = torch.zeros(1, 24, t, h // 16, w // 16)
    au = torch.zeros(1, 32, 2, a)
    if fill:
        v += torch.arange(t, dtype=torch.float32).view(1, 1, t, 1, 1)
        au += torch.arange(a, dtype=torch.float32).view(1, 1, 1, a)
    return {"samples": Nested(v, au)}


class Boom:
    def encode(self, x):
        raise AssertionError("the VAE was called")


class RecVae:
    def __init__(self):
        self.seen = []

    def encode(self, x):
        self.seen.append(x)
        n = x.shape[0]
        t = 1 if n == 1 else M.minimax_video_latent_t(n)
        return torch.full((1, 24, t, x.shape[1] // 16, x.shape[2] // 16), float(x.mean()))


class RecAVae:
    audio_sample_rate = 32000

    def __init__(self):
        self.seen = []

    def encode(self, w):
        self.seen.append(w)
        return torch.ones(1, 32, 2, int(w.shape[1]) // 800)


def cond(**extra):
    return [[torch.zeros(1, 4, 8), dict(extra)]]


NODE = CO.ULSMiniMaxContinue()
K2L, K2, ANC = CO.MODES

# ========================================================================= N1
if (CO.snap_guide(4), CO.snap_guide(5), CO.snap_guide(30), CO.snap_guide(39)) == (1, 5, 22, 39):
    _ok("N1 snap: 4 -> 1, 5, 30 -> 22, 39 (AddGuide's rule)")
else:
    _fail("N1 snap %r" % ([CO.snap_guide(x) for x in (4, 5, 30, 39)],))
if CO.resolve_index(-1, 73, 1) == 72 and CO.resolve_index(20, 73, 22) == 20:
    _ok("N1 index: -1 -> 72 of 73, a 22-frame clip at 20 fits")
else:
    _fail("N1 index")
_raises(lambda: CO.resolve_index(-1, 73, 22), "does not fit", "N1 a 22-frame clip at -1 does not fit, by name")
if CO.latent_window(37, 22) == (30, 7) and CO.latent_window(22, 22) == (15, 7):
    _ok("N1 latent window: last 22 frames = last 7 tokens, starting on the chunk grid")
else:
    _fail("N1 window %r %r" % (CO.latent_window(37, 22), CO.latent_window(22, 22)))
_raises(lambda: CO.latent_window(21, 22), "off the 17-frame chunk grid", "N1 an off-grid latent is refused")
_raises(lambda: CO.latent_window(5, 39), "needs", "N1 a window longer than the latent is refused")
if CO.audio_window(124, 22, 207) == (170, 207):
    _ok("N1 audio window by time: (124-22) * 5/3 = 170 .. 207")
else:
    _fail("N1 audio window %r" % (CO.audio_window(124, 22, 207),))

# ========================================================================= N2
tgt = av_latent(73)
CO.model_has_guides = lambda: False         # the pin
_raises(lambda: NODE.build(cond(), tgt, K2L, 22, True, -1, clip_latent=av_latent(124)),
        "has no MiniMaxH3AddGuide", "N2 the pinned Core is refused by name")
CO.model_has_guides = lambda: True
_raises(lambda: NODE.build(cond(), {"samples": torch.zeros(1, 4, 8, 8)}, K2L, 22, True, -1),
        "not a MiniMax H3 AV latent", "N2 a non-H3 latent is refused")
_raises(lambda: NODE.build(cond(), tgt, K2L, 22, True, -1), "needs `clip_latent`",
        "N2 K2L without clip_latent is refused")
_raises(lambda: NODE.build(cond(), tgt, K2L, 22, True, -1, clip_latent=av_latent(124, w=768, h=1344)),
        "keep the size of the chain", "N2 K2L with another size is refused")
_raises(lambda: NODE.build(cond(), tgt, K2, 22, True, -1, vae=RecVae(), tail=torch.rand(4, 48, 64, 3)),
        "5 frames or more", "N2 K2 with a 4-frame tail is refused")
_raises(lambda: NODE.build(cond(), tgt, K2, 22, True, -1, tail=torch.rand(22, 48, 64, 3)),
        "wire `vae`", "N2 K2 without vae is refused")
# v1051: 'anchor only' with nothing to add passes through (one workflow serves
# a first/last-frame run and a run with guides) -- said in info, not refused
_pt = NODE.build(cond(), tgt, ANC, 22, True, -1)
(_ok if (_pt[0] is not None and _pt[2] == 0 and "nothing to add" in _pt[3]) else _fail)(
    "N2 'anchor only' without an anchor passes through, said in info")
_raises(lambda: NODE.build(cond(), tgt, ANC, 22, True, -1, vae=RecVae(), anchor=torch.rand(22, 48, 64, 3)),
        "does not fit", "N2 an anchor clip at -1 does not fit, by name")

# ========================================================================= N3
clip = av_latent(124, fill=True)
prev_kf = {"resolved_frame_index": 72, "latent": torch.zeros(1)}
refs = [{"kind": "image"}]
out, lat, overlap, info = NODE.build(cond(minimax_keyframes=[prev_kf], minimax_refs=refs), tgt, K2L, 22,
                                     True, -1, vae=Boom(), clip_latent=clip)
kfs = out[0][1]["minimax_keyframes"]
g = kfs[-1]
cv, ca = clip["samples"].t
if torch.equal(g["latent"], cv[:, :, 30:37]) and g["resolved_frame_index"] == 0:
    _ok("N3 K2L guide = clip n's last 7 tokens (30..36) at frame 0, no VAE called")
else:
    _fail("N3 K2L latent tokens %r idx %r" % (g["latent"][0, 0, :, 0, 0].tolist(), g["resolved_frame_index"]))
if torch.equal(g.get("audio_latent", torch.zeros(1)), ca[..., 170:207]):
    _ok("N3 K2L sound = the clip's audio latent 170..206 (by time, not re-encoded)")
else:
    _fail("N3 K2L audio %r" % (None if "audio_latent" not in g else g["audio_latent"][0, 0, 0, :3].tolist(),))
if overlap == 22 and lat is tgt and "trim_head 22" in info and "no VAE" in info:
    _ok("N3 overlap 22 -> Seam Fix trim_head; latent passed on unchanged")
else:
    _fail("N3 overlap %r / %s" % (overlap, info))
if kfs[0] is prev_kf and len(kfs) == 2 and out[0][1].get("minimax_refs") is refs:
    _ok("N3 keyframes and references already in the conditioning survive")
else:
    _fail("N3 existing conditioning lost: %d keyframes, refs %r" % (len(kfs), out[0][1].get("minimax_refs")))
tiny = av_latent(5)                   # 8 audio latent frames; a 5-frame tail is 9 by time
g3 = NODE.build(cond(), tiny, K2L, 5, True, -1, clip_latent=clip)[0][0][1]["minimax_keyframes"][0]
if g3["audio_latent"].shape[-1] == 8 and torch.equal(g3["audio_latent"], ca[..., 198:206]):
    _ok("N3 the sound is cropped to the target's audio length (9 -> 8), from its start")
else:
    _fail("N3 audio not cropped: %d frames" % g3["audio_latent"].shape[-1])
o2 = NODE.build(cond(), tgt, K2L, 22, False, -1, clip_latent=clip)[0]
if "audio_latent" not in o2[0][1]["minimax_keyframes"][0]:
    _ok("N3 with_audio off carries no sound")
else:
    _fail("N3 with_audio off still carried sound")

# ========================================================================= N4
tail = torch.stack([torch.full((48, 64, 3), i / 100.0) for i in range(30)])
wav = torch.arange(3 * 32000, dtype=torch.float32).view(1, 1, -1).repeat(1, 2, 1)
rv, ra = RecVae(), RecAVae()
out4, _, ov4, i4 = NODE.build(cond(), tgt, K2, 22, True, -1, vae=rv, audio_vae=ra, tail=tail,
                              tail_audio={"waveform": wav, "sample_rate": 32000})
x = rv.seen[0]
if tuple(x.shape) == (22, 768, 1344, 3) and abs(float(x[0].mean()) - 0.08) < 1e-4 and ov4 == 22:
    _ok("N4 K2 encodes the LAST 22 frames (first = frame 8) at the target size 1344x768")
else:
    _fail("N4 K2 frames %r first %.4f" % (tuple(x.shape), float(x[0].mean())))
w = ra.seen[0]
n_s = int(round(22 / 24.0 * 32000))
if w.shape[1] == n_s and float(w[0, 0, 0]) == float(3 * 32000 - n_s):
    _ok("N4 K2 sound = the last 22/24 s of tail_audio (%d samples)" % n_s)
else:
    _fail("N4 K2 audio len %d first %r" % (w.shape[1], float(w[0, 0, 0])))
short = av_latent(5)
# a 5-frame target cannot hold a 22-frame context -- refused by the fit rule
_raises(lambda: NODE.build(cond(), short, K2, 22, True, -1, vae=RecVae(), tail=tail),
        "does not fit", "N4 a context longer than the target is refused")

# ========================================================================= N5
o5, _, ov5, i5 = NODE.build(cond(), tgt, ANC, 22, True, -1, vae=RecVae(), anchor=torch.rand(1, 48, 64, 3))
if o5[0][1]["minimax_keyframes"][0]["resolved_frame_index"] == 72 and ov5 == 0:
    _ok("N5 anchor image at -1 lands on frame 72 of 73, no overlap")
else:
    _fail("N5 anchor idx %r" % o5[0][1]["minimax_keyframes"][0]["resolved_frame_index"])
rv5 = RecVae()
o5b, _, _, i5b = NODE.build(cond(), tgt, K2L, 22, True, 10, vae=rv5, clip_latent=clip,
                            anchor=torch.rand(30, 48, 64, 3))
k5 = o5b[0][1]["minimax_keyframes"]
if len(k5) == 2 and k5[1]["resolved_frame_index"] == 10 and rv5.seen[0].shape[0] == 22 and "WARN" in i5b:
    _ok("N5 a 30-frame anchor snaps to 22, lands at 10, inside the context -> WARN")
else:
    _fail("N5 anchor clip: %d guides, info %s" % (len(k5), i5b))

# ========================================================================= N6
core = os.environ.get("PLS_CORE_ROOT")
if not core:
    for up in ROOT.parents:
        if (up / "comfy_extras" / "nodes_minimax_h3.py").exists():
            core = str(up)
            break
src = Path(core or "/nonexistent") / "comfy_extras" / "nodes_minimax_h3.py"
if not src.exists():
    print("  note  Core not reachable -- N6 parity SKIPPED (set PLS_CORE_ROOT=<ComfyUI>)")
elif "class MiniMaxH3AddGuide" not in src.read_text(encoding="utf-8", errors="replace"):
    print("  note  this Core has no MiniMaxH3AddGuide (the pin) -- N6 parity SKIPPED; "
          "the node refuses here (N2)")
else:
    script = textwrap.dedent(r'''
        import sys, types, importlib, json
        sys.argv = [sys.argv[0], "--cpu"]
        sys.path.insert(0, CORE)
        import comfy.options
        comfy.options.enable_args_parsing()
        import torch
        import comfy.nested_tensor
        import comfy_extras.nodes_minimax_h3 as C
        import comfy.ldm.minimax.vae as V
        pkg = types.ModuleType("plsn_c5"); pkg.__path__ = [NODES]
        sys.modules["plsn_c5"] = pkg
        CO = importlib.import_module("plsn_c5.ph_minimax_continue")
        M = sys.modules["plsn_c5.uls_latent_math"]
        class Vae:
            def encode(self, x):
                n = x.shape[0]
                t = 1 if n == 1 else M.minimax_video_latent_t(n)
                z = torch.zeros(1, 24, t, x.shape[1] // 16, x.shape[2] // 16)
                z += x.mean(dim=(1, 2, 3))[:t].view(1, 1, -1, 1, 1) if n >= t else x.mean()
                return z
        class AVae:
            audio_sample_rate = 32000
            def encode(self, w):
                n = int(w.shape[1]) // 800
                return w[:, :n * 800, :1].reshape(1, 1, n, 800).mean(-1).view(1, 1, 1, n).repeat(1, 32, 2, 1)
        lat, fc = C._empty_av_latent(1344, 768, 73)
        torch.manual_seed(1)
        tail = torch.rand(22, 90, 160, 3)
        aud = {"waveform": torch.rand(1, 2, int(round(22 / 24.0 * 32000))), "sample_rate": 32000}
        pos = [[torch.zeros(1, 4, 8), {}]]
        res = {}
        core = C.MiniMaxH3AddGuide.execute(pos, lat, 0, vae=Vae(), audio_vae=AVae(), image=tail, audio=aud)
        ck = (core.result[0] if hasattr(core, "result") else core[0])[0][1]["minimax_keyframes"][0]
        ours = CO.ULSMiniMaxContinue().build(pos, lat, CO.MODES[1], 22, True, -1, vae=Vae(),
                                             audio_vae=AVae(), tail=tail, tail_audio=aud)[0]
        ok = ours[0][1]["minimax_keyframes"][0]
        res["k2_idx"] = ck["resolved_frame_index"] == ok["resolved_frame_index"]
        res["k2_video"] = bool(torch.equal(ck["latent"], ok["latent"]))
        res["k2_audio"] = bool(torch.equal(ck["audio_latent"], ok["audio_latent"]))
        anc = torch.rand(30, 90, 160, 3)
        core2 = C.MiniMaxH3AddGuide.execute(pos, lat, 20, vae=Vae(), image=anc)
        ck2 = (core2.result[0] if hasattr(core2, "result") else core2[0])[0][1]["minimax_keyframes"][0]
        ok2 = CO.ULSMiniMaxContinue().build(pos, lat, CO.MODES[2], 22, True, 20, vae=Vae(),
                                            anchor=anc)[0][0][1]["minimax_keyframes"][0]
        res["anchor"] = ck2["resolved_frame_index"] == ok2["resolved_frame_index"] and bool(torch.equal(ck2["latent"], ok2["latent"]))
        # Core's REAL chunk loop with a chunk-local stand-in for the CNN
        vae = V.MiniMaxH3VideoVAE.__new__(V.MiniMaxH3VideoVAE)
        torch.nn.Module.__init__(vae)
        vae.clip_length, vae.token_drop = 17, 3
        def enc(x):     # 17 frames -> 5 tokens: frame 0, then groups of 4 (FRAME_PER_TOKEN)
            g = [x[:, :, :1]] + [x[:, :, 1 + 4 * j:5 + 4 * j] for j in range(4)]
            return torch.cat([s.mean(2, keepdim=True) for s in g], 2)
        vae._adaptive_encode = enc
        vae._normalize_pixels = lambda x: x
        clip = torch.rand(1, 3, 124, 4, 4)
        full = vae.encode_temporal(clip, "cpu")
        tail_z = vae.encode_temporal(clip[:, :, -22:], "cpu")
        s, n = CO.latent_window(full.shape[2], 22)
        res["chunk_T"] = [int(full.shape[2]), int(tail_z.shape[2]), s, n]
        res["k2l_exact"] = bool(torch.allclose(full[:, :, s:], tail_z))
        print(json.dumps(res))
    ''').replace("CORE", repr(core)).replace("NODES", repr(str(ROOT / "nodes")))
    env = dict(os.environ)
    env.setdefault("PYTHONPYCACHEPREFIX", "/tmp/pyc")
    try:
        r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                           timeout=600, env=env, cwd=core)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        if r.returncode != 0 or not line:
            _fail("N6 Core harness failed: %s" % (r.stderr.strip().splitlines()[-4:],))
        else:
            import json
            d = json.loads(line[-1])
            if d["k2_idx"] and d["k2_video"] and d["k2_audio"]:
                _ok("N6 K2 identical to Core's Add Guide (index, video latent, audio latent)")
            else:
                _fail("N6 K2 vs Core: %r" % d)
            if d["anchor"]:
                _ok("N6 anchor clip identical to Core's Add Guide at frame 20")
            else:
                _fail("N6 anchor vs Core: %r" % d)
            if d["k2l_exact"] and d["chunk_T"] == [37, 7, 30, 7]:
                _ok("N6 Core's encode_temporal: encode(124 frames)[tokens 30..36] == encode(last 22 frames)")
            else:
                _fail("N6 K2L claim does not hold on this Core: %r" % d)
    except subprocess.TimeoutExpired:
        _fail("N6 Core harness timed out")

print()
if FAILS:
    print("test_v1021_minimax_continue: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1021_minimax_continue: PASS (N1-N6)")
