# -*- coding: ascii -*-
"""Guard v1194 -- the RESTORE path of Power Upscale: a restoration model (Core's SeedVR2 family) wired into 'model'.

The run uses a TINY SeedVR2 built from Core's own classes (supported_models.SeedVR2 -> model_base.SeedVR2 -> NaDiT,
2 layers, dim 64) and Core's REAL SeedVR2 video VAE (250 M parameters, deterministic init), on the CPU. The chain
is Core's: nodes_seedvr + common_ksampler. What the guard judges is OUR plumbing -- when the path runs, what it
reads, what it refuses, the sizes, the chunking, the line it shows.

  R1 restore_family: Core's SeedVR2 -> 'seedvr2'; an ordinary model (no conditioning buffers) -> None; vae_matches
     accepts Core's SeedVR2 VAE and names anything else
  R2 chunk_plan: 0 = auto, 4n+1 manual, anything else refused by name
  R3 restore(): 9 frames, 5 per chunk, overlap 1 -> 2 chunks, finite output at the frames' size (odd size: the pad
     and the crop back), the info carries the numbers; a wrong VAE is refused by name; (v1197) every piece of the
     library-level chain is MEASURED equal to Core's own node (pad, chunk, merge, conditioning, colour match)
  R4 the node end to end: upscale_by 2.0 + pixel_stage 'fit only' -> 2x canvas, restored, path 'restore', verdict
     'restore: ...', final_upscale_by not read (canvas = upscale_by, said); positive/negative not read; a bad
     restore_frames refused before any work; an ordinary (tile) model keeps path 'tile'
  R5 the widgets: restore_color/overlap/frames APPENDED after pixel_speed (python), ORDER_CANON / DISPLAY_ORDER /
     CANON_DEFAULTS 30-32 / CANON_RANGES 31-32 (JS); the tooltips name no model file; ASCII
Script-style: exit 0 = pass.
"""
import importlib
import os
import re
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


torch.set_num_threads(2)
torch.manual_seed(0)
CORE = os.environ.get("PLS_CORE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, CORE)
sys.path.insert(0, str(ROOT / "nodes"))
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
fp = types.ModuleType("folder_paths")
for _n in ("get_temp_directory", "get_output_directory", "get_input_directory", "get_user_directory"):
    setattr(fp, _n, lambda: "/tmp")
fp.get_filename_list = lambda *a, **k: []
fp.get_full_path = lambda *a, **k: None
fp.models_dir = "/tmp/models"
fp.add_model_folder_path = lambda *a, **k: None
fp.get_folder_paths = lambda f: ["/tmp/models/" + f]
sys.modules.setdefault("folder_paths", fp)
import comfy.supported_models as SM  # noqa: E402
import comfy.model_patcher as MP  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.ldm.seedvr.vae as sv  # noqa: E402
R = importlib.import_module("ph_restore")
R._rungs_path = lambda: None          # AMENDED v1209: no learned rungs from the shared user directory
PU = importlib.import_module("ph_power_upscale")


def _init(module, cond=False):
    with torch.no_grad():
        for n_, p in module.named_parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, std=0.02)
            elif "bias" in n_:
                p.zero_()
            else:
                p.fill_(1.0)
        for n_, b in module.named_buffers():
            if (cond and "conditioning" in n_) or not torch.isfinite(b).all():
                b.normal_(std=0.02) if cond and "conditioning" in n_ else b.zero_()


def tiny_restore_model():
    cfg = {"image_model": "seedvr2", "vid_dim": 64, "heads": 2, "head_dim": 32, "num_layers": 2, "mm_layers": 1,
           "norm_eps": 1e-5, "mlp_type": "swiglu", "vid_out_norm": True, "rope_dim": 32}
    mc = SM.SeedVR2(cfg)
    mc.set_inference_dtype(torch.float32, None)
    mc.custom_operations = None
    bm = mc.get_model({}, device=torch.device("cpu"))
    bm.diffusion_model.to(torch.float32)
    _init(bm.diffusion_model, cond=True)
    return MP.ModelPatcher(bm, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def restore_vae():
    m = sv.VideoAutoencoderKLWrapper()
    _init(m)
    sd = m.state_dict()
    del m                              # the sandbox has 7 GB: one copy of the 250 M parameters, not two
    return comfy.sd.VAE(sd=sd, device=torch.device("cpu"), dtype=torch.float32)


class _PlainModel:              # an ordinary model: no conditioning buffers, not Core's class
    class _Inner:
        diffusion_model = torch.nn.Conv2d(3, 3, 1)
    model = _Inner()


class _PlainVAE:
    first_stage_model = torch.nn.Conv2d(3, 3, 1)


model = tiny_restore_model()
vae = restore_vae()

# ---- R1
_need(R.restore_family(model) == "seedvr2" and R.restore_family(_PlainModel()) is None and R.restore_family(None) is None,
      "R1 the family is read off Core's model class; an ordinary model is none")
_need(R.vae_matches(vae) == (True, "SeedVR2 video VAE") and R.vae_matches(_PlainVAE()) == (False, "Conv2d"),
      "R1 the family's own VAE is accepted, anything else named")

# ---- R2
_need(R.chunk_plan(0) == {"chunking_mode": "auto"} and R.chunk_plan(9) == {"chunking_mode": "manual", "frames_per_chunk": 9},
      "R2 chunk_plan: auto / 4n+1 manual")
try:
    R.chunk_plan(8)
    _need(False, "R2 8 must be refused")
except ValueError as e:
    _need("4n+1" in str(e), "R2 a count that is not 4n+1 is refused by name")

# ---- R3
frames = torch.rand(9, 38, 50, 3)          # small: Core's real VAE on the CPU costs GBs per 100 px of edge
# v1197: the chain is on Core's LIBRARY level -- spies on the module's pieces and on comfy.sample.sample
import comfy.sample  # noqa: E402
CALLS = {"sample": 0, "merge": 0, "post": [], "pre": 0}
_ss, _mg, _cm, _pr = comfy.sample.sample, R.merge_latent, R.color_match, R.pad_frames


def _spy_ss(*a, **k):
    CALLS["sample"] += 1
    CALLS["noise"] = CALLS.get("noise", 0.0) + float(a[1].abs().sum())      # the noise handed to the sampler
    return _ss(*a, **k)


def _spy_mg(*a, **k):
    CALLS["merge"] += 1
    return _mg(*a, **k)


def _spy_cm(dec, ref, color):
    CALLS["post"].append(color)
    return _cm(dec, ref, color)


def _spy_pr(*a, **k):
    CALLS["pre"] += 1
    return _pr(*a, **k)


comfy.sample.sample = _spy_ss
R.merge_latent = _spy_mg
R.color_match = _spy_cm
R.pad_frames = _spy_pr
out, info = R.restore(model, vae, frames, 7, color="lab", overlap=1, frames_per_chunk=5)
_need(CALLS.get("noise", 0.0) > 0.0 and {k: v for k, v in CALLS.items() if k != "noise"} == {"sample": 2, "merge": 1, "post": ["lab"], "pre": 1},
      "R3 the pieces are CALLED: pad once, comfy.sample.sample once per chunk (2) WITH noise, one merge, one colour match with the colour (%s)" % CALLS)
# EQUIVALENCE to Core's own node chain (the guard may import Core's node modules; the node may not -- v1063 law):
# the same pad, chunking, crossfade, conditioning and colour transfer, measured
import comfy_extras.nodes_seedvr as NS  # noqa: E402
pad_core = NS.SeedVR2Preprocess.execute(frames).args[0]
_need(torch.equal(R.pad_frames(frames), pad_core), "R3 pad_frames == Core's Preprocess (bit for bit)")
f7 = torch.rand(7, 30, 44, 3)                              # not 4n+1: the last frame repeated to 9
_need(torch.equal(R.pad_frames(f7), NS.SeedVR2Preprocess.execute(f7).args[0]) and R.pad_frames(f7).shape[1] == 9,
      "R3 pad_frames == Core's Preprocess on a 7-frame clip (repeated to 4n+1 = 9)")
lat = torch.randn(1, 16, 15, 4, 5)
ch_core = NS.SeedVR2TemporalChunk.execute({"samples": lat}, 6, {"chunking_mode": "manual", "frames_per_chunk": 33}).args
ch_ours, ov_ours = R.split_latent(lat, 33, 6)
_need(ov_ours == ch_core[1] and len(ch_ours) == len(ch_core[0]) and all(torch.equal(a, b["samples"]) for a, b in zip(ch_ours, ch_core[0])),
      "R3 split_latent == Core's TemporalChunk (%d chunks, overlap %d)" % (len(ch_ours), ov_ours))
# sampled chunks DIFFER in their overlap (slices of one latent would not, and any weights would merge them alike)
ch_sampled = [c + 0.3 * (i + 1) * torch.randn_like(c) for i, c in enumerate(ch_ours)]
mg_core = NS.SeedVR2TemporalMerge.execute([{"samples": c} for c in ch_sampled], [ov_ours]).args[0]["samples"]
_need(torch.allclose(R.merge_latent(ch_sampled, ov_ours), mg_core), "R3 merge_latent == Core's TemporalMerge (the Hann crossfade over differing chunks)")
pc, nc = NS.SeedVR2Conditioning.execute(model, {"samples": lat[:, :, :2]}).args
po, no = R.conditioning(model, lat[:, :, :2])
_need(torch.equal(po[0][0], pc[0][0]) and torch.equal(po[0][1]["condition"], pc[0][1]["condition"]) and torch.equal(no[0][0], nc[0][0]),
      "R3 conditioning == Core's Conditioning (buffers + latent with its mask)")
dec = torch.rand(5, 48, 64, 3)
ref = torch.rand(5, 38, 50, 3)
for c in ("lab", "wavelet", "adain", "none"):
    pp = NS.SeedVR2PostProcessing.execute(dec, ref, c).args[0]
    ours = R.color_match(dec, ref, c)
    _need(tuple(ours.shape) == tuple(pp.shape) and torch.allclose(ours, pp, atol=1e-5), "R3 color_match(%s) == Core's PostProcessing" % c)
_need(info["latent_out"] == info["latent"], "R3 the merged latent has the input latent's length (the overlap is crossfaded, not concatenated)")
_need(tuple(out.shape) == (9, 38, 50, 3) and bool(torch.isfinite(out).all()), "R3 restore(): finite frames at the frames' own (odd) size")
_need(info["chunks"] == 2 and info["frames_per_chunk"] == 5 and info["overlap"] == 1 and info["latent"] == (1, 16, 3, 6, 8),      # 38x50 padded to 48x64 by Core, /8
      "R3 9 frames at 5 per chunk, overlap 1 -> 2 chunks over a 3-frame latent (%s)" % info)
_need("restore: seedvr2 over 9 frame(s) at 50x38 -- 2 chunk(s) of 5 frames (overlap 1 latent)" in R.verdict(info, "after refine")
      and "; after refine;" in R.verdict(info, "after refine"), "R3 the verdict line (AMENDED v1199: it names where the restore ran)")
try:
    R.restore(model, _PlainVAE(), frames, 7)
    _need(False, "R3 a wrong VAE must be refused")
except ValueError as e:
    _need("OWN video VAE" in str(e) and "Conv2d" in str(e), "R3 a wrong VAE is refused by name")
try:
    R.restore(model, vae, frames, 7, color="sepia")
    _need(False, "R3 an unknown colour mode must be refused")
except ValueError as e:
    _need("restore_color" in str(e), "R3 an unknown colour mode is refused")

del out
import gc  # noqa: E402
gc.collect()

# ---- R4  AMENDED v1199: the restoration model is PICKED on the node (restore_model); wired into 'model' it is refused
#          with the way to the pick -- the end-to-end run through the node lives in test_v1199_restore_pick (P3)
node = PU.ULSPowerUpscale()
cond = [[torch.zeros(1, 4, 8), {}]]
img = torch.rand(5, 20, 28, 3)
try:
    node.upscale(model, cond, cond, vae, None, False, 2.0, 0.2, 3, 1.0, 1.3, 0.25, 5, 1.9, 7, "euler", "simple", 1024, 64, 0.0,
                 image=img, resize_method="bicubic", pixel_stage="fit only", result_preview=False, process_preview="Off")
    _need(False, "R4 a restoration model on 'model' must be refused (v1199)")
except ValueError as e:
    _need("pick it in 'restore_model'" in str(e), "R4 (v1199) a restoration model on 'model' is refused with the way to restore_model")
SRC = (ROOT / "nodes" / "ph_power_upscale.py").read_text(encoding="utf-8")
_need('"pls_pu_path": ["joint" if _joint is not None else "tile"]' in SRC and "_joint = _joint_streams(model, model_low)\n" in SRC,
      "R4 (v1199) the path stays joint / tile; the restore is a stage next to the refine")

# ---- R5
req = list(PU.ULSPowerUpscale.INPUT_TYPES()["required"])
_need(req[req.index("pixel_speed"):req.index("pixel_speed") + 4] == ["pixel_speed", "restore_color", "restore_overlap", "restore_frames"],
      "R5 the three dials are APPENDED after pixel_speed (AMENDED v1199: restore_model / restore_order follow them)")
JS = (ROOT / "web" / "js" / "ph_power_upscale.js").read_text(encoding="utf-8")
canon = JS[JS.find("const ORDER_CANON = ["):JS.find("];", JS.find("const ORDER_CANON = ["))]
disp = JS[JS.find("const DISPLAY_ORDER = ["):JS.find("];", JS.find("const DISPLAY_ORDER = ["))]
_need(('"restore_color", "restore_overlap", "restore_frames",   // v1194 appended (the restore path\'s three dials)') in canon   # AMENDED v1199: the pick follows
      and '"restore_color", "restore_overlap", "restore_frames",' in disp,
      "R5 ORDER_CANON and DISPLAY_ORDER end with the three dials")
_need('const DOM_TAIL = ["pls_pu_result", "pls_pu_process", "pls_weights_line"];' in JS,
      "R5 the door's line is the THIRD DOM tail of the Power Upscale (two tails left a null in the core and scrambled every dial)")
WJS = (ROOT / "web" / "js" / "ph_world_weights.js").read_text(encoding="utf-8")
_need('ULSPowerUpscale: "restore"' in WJS and 'ULSPowerUpscale: { kind: "pick", widget: "restore_model", companion: /vae/i }' in WJS,   # AMENDED v1204
      "R5 the door line on the Power Upscale; it offers only the picked model + its VAE (v1204)")
_need('30: "lab",' in JS and "31: 1," in JS and "32: 0," in JS and "31: [0, 64]," in JS and "32: [0, 16384]," in JS,
      "R5 CANON_DEFAULTS 30-32 and CANON_RANGES 31-32")
tips = " ".join(v[1].get("tooltip", "") for k, v in PU.ULSPowerUpscale.INPUT_TYPES()["required"].items() if k.startswith("restore_"))
_need(not re.search(r"\.safetensors|seedvr|SeedVR|3b|7b", tips), "R5 the dials' tooltips name no model file or size (general by construction)")
for p in ("nodes/ph_restore.py", "tests/test_v1194_restore.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "R5 ASCII %s" % p)

print("%s: %d check(s) failed" % ("test_v1194_restore", len(FAILS)) if FAILS else "test_v1194_restore: all checks passed")
sys.exit(1 if FAILS else 0)
