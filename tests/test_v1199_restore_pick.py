# -*- coding: ascii -*-
"""Guard v1199 -- the restoration model is PICKED on Power Upscale (restore_model) and runs NEXT TO the refine.

(Frank 21.10.: "Entweder holen wir uns das Modell mit in die Node oder wir machen das extern wie bei ESRGAN, dann
gibt es aber keinen Check und keinen Download" -- "machen wir erst mal so ... go": in the node.)

  P1 the pick: choices() = 'off' first, then the door's registry checkpoints (missing ones '... not installed') and a
     family checkpoint placed by hand; the family VAE is never offered as a model; resolve('off') = None; resolve of a
     missing checkpoint / a missing VAE is refused by name with the way to the Download; with both on disk ->
     (model path, vae path); vae_path prefers the registry VAE, else a tagged one
  P2 load(): Core's library (comfy.sd.load_diffusion_model, comfy.utils.load_torch_file + comfy.sd.VAE) -- a model
     that is no restoration model is refused, a VAE that is not the family's is refused, a second load of the same
     pick is the resident one (no second load)
  P3 the node end to end (tiny SeedVR2 + Core's real SeedVR2 VAE on the CPU, an ordinary model on 'model'):
     refine_order off + restore_model -> 2x canvas, restored, the line names the restore ("no refine ran"), path
     'tile'; 'before refine' with no refine running says so and restores at the end
  P4 refusals before any work: a restoration model on 'model' (-> "pick it in 'restore_model'"), a pick that is not
     installed, a bad restore_frames
  P5 where the restore runs (statics): 'before refine' = right after the input for the tile stages / the joint
     refine before pixel, inside the after-pixel joint block before _joint_refine; 'after refine' = behind
     everything; the joint cache is neither read nor written while the restore runs before the refine
  P6 the widgets: restore_model + restore_order APPENDED after restore_frames (python), ORDER_CANON / DISPLAY_ORDER /
     CANON_DEFAULTS 33-34 (JS); the restore dials hide while restore_model is off (RESTORE_ONLY, the pick's callback);
     the door notes say "restore_model", never "Load Model into 'model'"; the tooltips name no model file; ASCII
  P7 the door line FOLLOWS the pick (node.js; AMENDED v1204: neededModels): the picked checkpoint + the family VAE needed
     (orange while missing, the Download fetches exactly them), 'off' leaves every file optional; the line re-reads
     on a change of the pick and relabels the pick after the Download; with 'off' it says "restore off"
Script-style: exit 0 = pass.
"""
import contextlib
import importlib
import io
import os
import re
import shutil
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


torch.set_num_threads(2)
torch.manual_seed(0)
CORE = os.environ.get("PLS_CORE_ROOT") or "/home/claude/core_pin"
sys.path.insert(0, CORE)
sys.path.insert(0, str(ROOT / "nodes"))
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
TMP = Path(tempfile.mkdtemp(prefix="v1199rs_"))
for d in ("diffusion_models", "vae"):
    (TMP / d).mkdir()
fp = types.ModuleType("folder_paths")
for _n in ("get_temp_directory", "get_output_directory", "get_input_directory", "get_user_directory"):
    setattr(fp, _n, lambda: str(TMP))
fp.models_dir = str(TMP)
fp.add_model_folder_path = lambda *a, **k: None
fp.get_folder_paths = lambda f: [str(TMP / f)]
fp.get_filename_list = lambda f: sorted(os.listdir(TMP / f)) if (TMP / f).is_dir() else []
fp.get_full_path = lambda f, n: str(TMP / f / n) if (TMP / f / n).is_file() else None
sys.modules["folder_paths"] = fp
import comfy.supported_models as SM  # noqa: E402
import comfy.model_patcher as MP  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.ldm.seedvr.vae as sv  # noqa: E402
R = importlib.import_module("ph_restore")
WW = importlib.import_module("ph_world_weights")
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



# ---- P1 the pick
DIT = "seedvr2_3b_int8_convrot.safetensors"
VAEF = "seedvr2_ema_vae_fp16.safetensors"
ch = R.choices()
_need(ch[0] == "off" and len(ch) == 4 and all(WW.strip(c) in WW.REGISTRY and "not installed" in c for c in ch[1:])
      and not any("vae" in c for c in ch), "P1 nothing on disk: off + the three registry checkpoints, 'not installed', no VAE (%s)" % ch)
(TMP / "diffusion_models" / DIT).write_bytes(b"x" * 2048)
(TMP / "diffusion_models" / "my_SeedVR2_custom.safetensors").write_bytes(b"x" * 1024)
(TMP / "diffusion_models" / "wan_other.safetensors").write_bytes(b"x")
(TMP / "diffusion_models" / "seedvr2_vae_copy.safetensors").write_bytes(b"x")   # a family VAE filed among the models
ch = R.choices()
names = [WW.strip(c) for c in ch]
_need(names[1] == DIT and "not installed" not in ch[1] and "my_SeedVR2_custom.safetensors" in names and "wan_other.safetensors" not in names
      and "seedvr2_vae_copy.safetensors" not in names,
      "P1 on disk: the registry file with its size, a family file placed by hand joins, another model / a family VAE does not (%s)" % ch)
_need(R.resolve("off") is None and R.resolve("") is None, "P1 resolve('off') = None")
try:
    R.resolve(ch[1])
    _need(False, "P1 a missing VAE must be refused")
except ValueError as e:
    _need(VAEF in str(e) and "Download" in str(e), "P1 the VAE missing: refused by name, the way to the Download (%s)" % e)
(TMP / "vae" / "x_seedvr_vae.safetensors").write_bytes(b"v")
_need(R.vae_path().endswith("x_seedvr_vae.safetensors"), "P1 no registry VAE: a tagged one serves")
(TMP / "vae" / VAEF).write_bytes(b"v")
_need(R.vae_path().endswith(VAEF), "P1 the registry VAE first")
mp, vp = R.resolve(ch[1])
_need(mp.endswith(DIT) and vp.endswith(VAEF), "P1 resolve -> (model path, vae path)")
missing = [c for c in ch if "not installed" in c][0]
try:
    R.resolve(missing)
    _need(False, "P1 a missing checkpoint must be refused")
except ValueError as e:
    _need("not installed" in str(e) and "Download" in str(e), "P1 a missing checkpoint: refused by name, the way to the Download")

# ---- P2 load through the library (the calls patched: Core's detection needs a real checkpoint)
model = tiny_restore_model()
vae = restore_vae()
CALLS = {"dm": 0, "vae": 0}
_orig = (comfy.sd.load_diffusion_model, comfy.utils.load_torch_file, comfy.sd.VAE)



WHICH = {"m": model, "v": vae}


def _ldm(path, model_options=None):
    CALLS["dm"] += 1
    return WHICH["m"]


def _vae(sd=None, metadata=None, **k):
    CALLS["vae"] += 1
    return WHICH["v"]


comfy.sd.load_diffusion_model = _ldm
comfy.utils.load_torch_file = lambda p, return_metadata=False, **k: ({}, None) if return_metadata else {}
comfy.sd.VAE = _vae
m1, v1 = R.load((mp, vp))
m2, v2 = R.load((mp, vp))
_need(m1 is model and v1 is vae and CALLS == {"dm": 1, "vae": 1}, "P2 loaded through comfy.sd once, the second pick is resident (%s)" % CALLS)
WHICH["m"] = _PlainModel()
try:
    R.load((mp + "x", vp))
    _need(False, "P2 a model that is no restoration model must be refused")
except ValueError as e:
    _need("no restoration model" in str(e), "P2 not a restoration model: refused")
WHICH.update(m=model, v=_PlainVAE())
try:
    R.load((mp + "y", vp))
    _need(False, "P2 a foreign VAE must be refused")
except ValueError as e:
    _need("not the family's video VAE" in str(e), "P2 a foreign VAE: refused")
WHICH["v"] = vae
R.load((mp, vp))

# ---- P3 the node end to end
import gc  # noqa: E402
node = PU.ULSPowerUpscale()
cond = [[torch.zeros(1, 4, 8), {}]]
img = torch.rand(5, 20, 28, 3)
pick = [c for c in R.choices() if WW.strip(c) == DIT][0]
ARGS = (False, 2.0, 0.2, 3, 1.0, 1.3, 0.25, 5, 1.9, 7, "euler", "simple", 1024, 64, 0.0)
KW = dict(image=img, resize_method="bicubic", pixel_stage="fit only", result_preview=False, process_preview="Off",
          refine_order="off", final_upscale_by=2.0, restore_color="lab", restore_overlap=1, restore_frames=5, restore_model=pick)
r = node.upscale(_PlainModel(), cond, cond, _PlainVAE(), None, *ARGS, **KW)
o = r["result"][0]
v = r["ui"]["pls_pu_verdict"][0]
_need(tuple(o.shape) == (5, 40, 56, 3) and bool(torch.isfinite(o).all()), "P3 2x canvas, restored, finite (%s)" % (tuple(o.shape),))
_need("restore: seedvr2 over 5 frame(s) at 56x40" in v and "no refine ran" in v and r["ui"]["pls_pu_path"] == ["tile"],
      "P3 the line names the restore and where it ran, path 'tile' (%s)" % v[:160])
del r
gc.collect()
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    r = node.upscale(_PlainModel(), cond, cond, _PlainVAE(), None, *ARGS, **dict(KW, restore_order="before refine"))
_need("no refine runs this time; the restore runs at the end" in buf.getvalue() and "no refine ran" in r["ui"]["pls_pu_verdict"][0],
      "P3 'before refine' with no refine running: said, restored at the end")
del r
gc.collect()

# ---- P4 refusals
for kw, mdl, want, what in ((dict(KW), model, "pick it in 'restore_model'", "a restoration model on 'model'"),
                            (dict(KW, restore_model=missing), _PlainModel(), "not installed", "a pick not installed"),
                            (dict(KW, restore_frames=6), _PlainModel(), "4n+1", "a bad restore_frames")):
    try:
        node.upscale(mdl, cond, cond, _PlainVAE(), None, *ARGS, **kw)
        _need(False, "P4 %s must be refused" % what)
    except ValueError as e:
        _need(want in str(e), "P4 %s: refused before any work (%s)" % (what, str(e)[:90]))
comfy.sd.load_diffusion_model, comfy.utils.load_torch_file, comfy.sd.VAE = _orig

# ---- P5 where the restore runs
SRC = (ROOT / "nodes" / "ph_power_upscale.py").read_text(encoding="utf-8")
a = SRC.find("        cur = frames\n        if _rs_before and not (_joint_on and _order == \"after pixel\"):")
b = SRC.find("        if _joint_on and _order == \"before pixel\":", a)
c = SRC.find("        if _joint_on and _order == \"after pixel\":\n            # v1004: the hires-fix -- the refine on the finished canvas.\n            if _rs_before:")
d = SRC.find("cur, _jr_info = _joint_refine(model, positive, negative, vae, cur, latent,", c)
e_ = SRC.find("        if _restore is not None and not _rs_before:")
f = SRC.find("        # v1008: WHERE the pixel model sat this run")
_need(0 < a < b and "_restore_pass(cur, \"before refine, at the input size\")" in SRC[a:b], "P5 before the refine on the input: right after the input, ahead of the joint 'before pixel'")
_need(0 < c < d and "_restore_pass(cur, \"before refine, on the finished canvas\")" in SRC[c:d], "P5 before the joint refine after pixel: inside its block, ahead of _joint_refine")
_need(d < e_ < f, "P5 after the refine: behind everything, before the line is made")
_need("if _rs_before and _jr_hit is not None:" in SRC and SRC.count("if _jr_hit is None and not _rs_before:") == 2,
      "P5 the joint cache is neither read nor written while the restore runs before the refine")
_need("_RS.restore_family(model) is not None" in SRC and "_joint = _joint_streams(model, model_low)\n" in SRC, "P5 the one way: a restoration model on 'model' is refused, the joint probe always runs")

# ---- P6 widgets
import inspect  # noqa: E402
_need(inspect.getfullargspec(PU.ULSPowerUpscale.VALIDATE_INPUTS).args[1:] == ["restore_model"] and PU.ULSPowerUpscale.VALIDATE_INPUTS(restore_model="x \u00b7 3.2 GB") is True,
      "P6 VALIDATE_INPUTS takes ONLY restore_model out of Core's membership check (a saved label whose size changed still runs)")
_need("pick the model in the node's restore_model" in (ROOT / "web" / "js" / "ph_world_weights.js").read_text(encoding="utf-8"), "P6 the door's text points at restore_model")
req = list(PU.ULSPowerUpscale.INPUT_TYPES()["required"])
_need(req[-6:] == ["pixel_speed", "restore_color", "restore_overlap", "restore_frames", "restore_model", "restore_order"],
      "P6 restore_model + restore_order APPENDED (%s)" % req[-6:])
it = PU.ULSPowerUpscale.INPUT_TYPES()["required"]
_need(it["restore_model"][1]["default"] == "off" and it["restore_order"][0] == ["after refine", "before refine"], "P6 defaults: off / after refine")
JS = (ROOT / "web" / "js" / "ph_power_upscale.js").read_text(encoding="utf-8")
_need(JS.count('"restore_model", "restore_order",') == 2 and '33: "off",' in JS and '34: "after refine",' in JS,
      "P6 ORDER_CANON + DISPLAY_ORDER appended, CANON_DEFAULTS 33-34")
_need('const RESTORE_ONLY = ["restore_color", "restore_overlap", "restore_frames", "restore_order"];' in JS
      and 'const rOff = !rm || ["", "off"].includes(' in JS
      and 'const rmw = _findWidget(self, "restore_model");' in JS and "for (const name of RESTORE_ONLY)" in JS,
      "P6 the restore dials hide while restore_model is off; the pick's callback re-applies")
WSRC = (ROOT / "nodes" / "ph_world_weights.py").read_text(encoding="utf-8")
_need("Load Model into 'model'" not in WSRC and "Load VAE into 'vae'" not in WSRC and "restore_model" in WSRC, "P6 the door notes point at restore_model")
tips = " ".join(v[1].get("tooltip", "") for k, v in it.items() if k.startswith("restore_"))
_need(not re.search(r"\.safetensors|seedvr|SeedVR|3b|7b", tips), "P6 the tooltips name no model file (general by construction)")
for p in ("nodes/ph_restore.py", "tests/test_v1199_restore_pick.py"):
    _need(all(ch_ < 128 for ch_ in (ROOT / p).read_bytes()), "P6 ASCII %s" % p)
# the two Power Upscale files carry box-drawing comments since long before; every v1199 line is ASCII
for p in ("nodes/ph_power_upscale.py", "web/js/ph_power_upscale.js"):
    bad = [ln for ln in (ROOT / p).read_text(encoding="utf-8").splitlines() if ("v1199" in ln or "restore_model" in ln or "RESTORE_ONLY" in ln or "_rs_" in ln) and any(ord(x) > 127 for x in ln)]
    _need(not bad, "P6 ASCII: every v1199 line of %s (%s)" % (p, bad[:1]))
# ---- P7 the door line follows the pick  (AMENDED v1204: pickRequired / SET_PICK became neededModels / OPTION_NEEDS --
#          the line now shows ONLY what is needed; the pick + its VAE when picked, nothing when 'off')
WJ = (ROOT / "web" / "js" / "ph_world_weights.js").read_text(encoding="utf-8")
node_js = shutil.which("node")
if node_js:
    i = WJ.find("export function stripLabel(v)"); j = WJ.find("\n}", i) + 2
    k = WJ.find("const TAIL = "); k2 = WJ.find("\n", k) + 1
    i2 = WJ.find("export function neededModels("); j2 = WJ.find("\n}", i2) + 2
    code = WJ[k:k2] + WJ[i:j] + WJ[i2:j2] + r"""
const M = [{ name: "a_3b.safetensors", optional: true }, { name: "fam_vae.safetensors", optional: true }, { name: "b_7b.safetensors", optional: true }];
const P = { kind: "pick", widget: "restore_model", companion: /vae/i };
const n = (L) => L.map((m) => m.name);
console.log(JSON.stringify([n(neededModels(M, P, "a_3b.safetensors \u00b7 3.2 GB \u00b7 not installed")), n(neededModels(M, P, "off")),
                            n(neededModels(M, P, "my_own.safetensors \u00b7 1 GB"))]));
"""
    (TMP / "p7.mjs").write_text(code.replace("export function", "function"), encoding="utf-8")
    import subprocess  # noqa: E402
    out = subprocess.run([node_js, str(TMP / "p7.mjs")], capture_output=True, text=True, timeout=60)
    try:
        r = __import__("json").loads(out.stdout.strip())
    except Exception:
        r = None
        print(out.stdout, out.stderr[-600:])
    _need(r == [["a_3b.safetensors", "fam_vae.safetensors"], [], []],
          "P7 the pick + the VAE needed, 'off' nothing, a hand-placed pick nothing to fetch (%s)" % r)
else:
    _need(False, "P7 node.js needed")
_need('ULSPowerUpscale: { kind: "pick", widget: "restore_model", companion: /vae/i }' in WJ
      and "const need = neededModels(models, opt, ow ? ow.value : undefined);" in WJ
      and "try { updateSet(node); } catch (e) { /* never break a pick */ }" in WJ and "ow.value = fresh.label;" in WJ,
      "P7 the line follows the pick, re-reads on a change, relabels after the Download")
shutil.rmtree(TMP, ignore_errors=True)
print("%s: %d check(s) failed" % ("test_v1199_restore_pick", len(FAILS)) if FAILS else "test_v1199_restore_pick: all checks passed")
sys.exit(1 if FAILS else 0)
