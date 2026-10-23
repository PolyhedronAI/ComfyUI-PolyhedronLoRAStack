# -*- coding: ascii -*-
"""Guard v1216 -- the control line at the model door, and no render on the Media Loader's placeholder.

Frank 23.10. (first Control field run, Control Map pose_model '(none)'): "Auf jeden Fall braucht es wohl Modelle
... ausserdem muessen hier wohl auch offiziell Download-Optionen mit rein" -- and, from the first first -> last frame
run (PH_093901: both keyframe files missing, the loader's checkerboard pinned at both ends): "die Korrekturen von
oben gleich mit rein".

  P1 the placeholder: made in ONE place (uls_placeholder.make_np == the loader's frame); is_placeholder True for
     the loader's tensor (also as a batch), False for a real picture, a resized placeholder, None, a 2D array
  P2 the refusals: MiniMax Keyframes refuses a placeholder first / last frame by name BEFORE any encode; Control
     Fit refuses placeholder images; a real picture passes both; Timeline, Continue and Reference carry the refusal
     at the point where the picture is taken (source scan)
  P3 the registry: set 'control' (4 H3 + 2 Qwen-Image 2.1 Fun ControlNet Union files, models/model_patches; one
     recommended per family) and set 'sdpose' (1 file, models/checkpoints): https on huggingface.co, sha256 64 hex,
     bytes > 0, the file name at the end of the URL
  P4 the combos and the resolution: door_choices lists the door's files first (labelled, 'not installed' when
     missing), then the folder's own files plain, a door file already on disk not twice; resolve_file strips the
     label, refuses a missing door file with the way to the Download, finds a door file on disk in a subfolder;
     the nodes' INPUT_TYPES use them (default = the recommended H3 union / SDPose)
  P5 VALIDATE_INPUTS: Control Map frees exactly pose_model + depth_model, Apply Control exactly control_model
  P6 the line (node.js): resolvePick -- Apply Control follows control_model; Control Map pose -> sdpose_model of
     set sdpose, depth -> depth_model of set da3, canny / gray / keypoints -> nothing; pickWidgetNames binds the mode
     too
  S  ASCII
Script-style: exit 0 = pass.
"""
import importlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1216_control_door"
FAILS = []


def _need(c, m):
    print(("  ok    " if c else "  FAIL  ") + m)
    if not c:
        FAILS.append(m)


TMP = Path(tempfile.mkdtemp(prefix="pls1216_"))
MODELS = TMP / "models"
fp = types.ModuleType("folder_paths")
fp.models_dir = str(MODELS)
fp.add_model_folder_path = lambda *a, **k: None
fp.get_folder_paths = lambda f: [str(MODELS / f)]


def _list(f):
    d = MODELS / f
    if not d.is_dir():
        return []
    return sorted(str(p.relative_to(d)).replace("\\", "/") for p in d.rglob("*") if p.is_file())


def _full(f, n):
    p = MODELS / f / n
    return str(p) if p.is_file() else None


def _full_raise(f, n):
    p = _full(f, n)
    if not p:
        raise FileNotFoundError(n)
    return p


fp.get_filename_list = _list
fp.get_full_path = _full
fp.get_full_path_or_raise = _full_raise
comfy = types.ModuleType("comfy")
cu = types.ModuleType("comfy.utils")
cu.common_upscale = lambda s, w, h, m, c: F.interpolate(s, size=(h, w), mode="bilinear", align_corners=False)
comfy.utils = cu
nh = types.ModuleType("node_helpers")
nh.conditioning_set_values = lambda cond, v: [[c[0], dict(c[1], **v)] for c in cond]
sys.modules.update({"folder_paths": fp, "comfy": comfy, "comfy.utils": cu, "node_helpers": nh})
pkg = types.ModuleType("plsn1216")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1216"] = pkg
PH = importlib.import_module("plsn1216.uls_placeholder")
WW = importlib.import_module("plsn1216.ph_world_weights")
CT = importlib.import_module("plsn1216.ph_control")
KF = importlib.import_module("plsn1216.ph_minimax_keyframes")
M = importlib.import_module("plsn1216.uls_latent_math")
SRC = {n: (ROOT / "nodes" / n).read_text(encoding="utf-8") for n in
       ("ph_media_loader.py", "ph_cine_timeline.py", "ph_minimax_continue.py", "ph_minimax_ref.py")}

# ---------------------------------------------------------------- P1
print("P1 the placeholder")
ref = PH.make_np()
loader_src = SRC["ph_media_loader.py"]
_need("from .uls_placeholder import make_np" in loader_src and "return make_np()" in loader_src,
      "P1 the loader's _placeholder_media_np returns uls_placeholder.make_np() (one place)")
_need(ref.shape == (512, 512, 3) and abs(float(ref[0, 0, 0]) - 0.90) < 1e-6 and abs(float(ref[256, 256, 0]) - 0.10) < 0.06,
      "P1 the frame: 512 x 512, orange border, dark checker inside")
ph_t = torch.from_numpy(ref).unsqueeze(0)
_need(PH.is_placeholder(ph_t) and PH.is_placeholder(ph_t.repeat(3, 1, 1, 1)) and PH.is_placeholder(ref),
      "P1 the loader's tensor is recognised (single, batch, numpy)")
real = torch.rand(1, 512, 512, 3)
framed = real.clone()
framed[:, :6] = framed[:, -6:] = torch.tensor([0.90, 0.49, 0.38])
resized = F.interpolate(ph_t.movedim(-1, 1), size=(768, 768), mode="bilinear").movedim(1, -1)
_need(not PH.is_placeholder(real) and not PH.is_placeholder(framed) and not PH.is_placeholder(resized)
      and not PH.is_placeholder(None) and not PH.is_placeholder(np.zeros((4, 4))),
      "P1 a real picture, an orange-framed picture, a resized placeholder, None, a 2D array: not the placeholder")
_need(not PH.is_placeholder(torch.cat([ph_t, real])), "P1 a batch with one real frame is not the placeholder")
_need(not PH.is_placeholder((ph_t + 0.02 * torch.randn_like(ph_t)).clamp(0, 1)) and PH.is_placeholder(ph_t + 0.0005),
      "P1 exact within 1/1000: a re-encoded (noisy) placeholder is not the loader's empty state, float dust is")

# ---------------------------------------------------------------- P2
print("P2 the refusals")


class Nested:
    is_nested = True

    def __init__(self, v):
        self._v = v

    def unbind(self):
        return [self._v]


class Clip:
    calls = 0

    def tokenize(self, prompt, images=None, **kw):
        Clip.calls += 1
        return {}

    def encode_from_tokens_scheduled(self, t):
        return [[torch.zeros(1), {}]]


class Vae:
    calls = 0

    def encode(self, x):
        Vae.calls += 1
        return torch.zeros(1, 24, 1, int(x.shape[1]) // 16, int(x.shape[2]) // 16)


lat = {"samples": Nested(torch.zeros(1, 24, M.minimax_video_latent_t(124), 768 // 16, 1344 // 16))}
pic = torch.rand(1, 432, 768, 3)
for slot, kw in (("first_frame", {"first_frame": ph_t, "last_frame": pic}),
                 ("last_frame", {"first_frame": pic, "last_frame": ph_t})):
    Clip.calls = Vae.calls = 0
    try:
        KF.ULSMiniMaxKeyframes().build(Clip(), Vae(), lat, "she sits down", KF.FRAME_FIT[0], **kw)
        _need(False, "P2 Keyframes accepted a placeholder %s" % slot)
    except ValueError as e:
        _need(slot in str(e) and "placeholder" in str(e) and Clip.calls == 0 and Vae.calls == 0,
              "P2 Keyframes refuses a placeholder %s by name, nothing encoded" % slot)
out = KF.ULSMiniMaxKeyframes().build(Clip(), Vae(), lat, "she sits down", KF.FRAME_FIT[0], first_frame=pic, last_frame=pic)
_need(isinstance(out, dict) and len(out["result"]) == 4, "P2 Keyframes runs on real pictures")
FIT = (24.0, 0.0, 0.0, 22, CT.GRIDS[0], CT.END[0], 256, 256, CT.SIZE_MODES[1], 32)
try:
    CT.ULSControlFit().fit(ph_t, *FIT)
    _need(False, "P2 Control Fit accepted placeholder images")
except ValueError as e:
    _need("Control Fit" in str(e) and "placeholder" in str(e), "P2 Control Fit refuses placeholder images by name")
r = CT.ULSControlFit().fit(torch.rand(30, 64, 64, 3), *FIT)
_need(tuple(r[0].shape[:1]) == (22,), "P2 Control Fit runs on a real clip")
_need("from .uls_placeholder import refuse" in SRC["ph_cine_timeline.py"]
      and 'refuse("Cine Timeline", "the %s %s" % (which, a["src"]), sources[a["src"]])' in SRC["ph_cine_timeline.py"]
      and 'refuse("Cine Timeline", "the anchor %s" % a["src"], a["image"])' in SRC["ph_cine_timeline.py"],
      "P2 the Cine Timeline refuses a placeholder first / last frame / anchor source")
_need('refuse("MiniMax Continue", "anchor", anchor)' in SRC["ph_minimax_continue.py"]
      and 'refuse("MiniMax Continue", "the anchor %s" % a.get("src", "?"), a.get("image"))' in SRC["ph_minimax_continue.py"],
      "P2 MiniMax Continue refuses a placeholder anchor")
_need('refuse("MiniMax Reference", "image_%d" % n, img)' in SRC["ph_minimax_ref.py"]
      and 'refuse("MiniMax Reference", "first_frame", opt.get("first_frame"))' in SRC["ph_minimax_ref.py"],
      "P2 MiniMax Reference refuses a placeholder reference / first frame")
_need("placeholder" in PH.refusal("X", "y") and "Media Loader" in PH.refusal("X", "y"), "P2 the sentence names the loader")

# ---------------------------------------------------------------- P3
print("P3 the registry")
ctl = {n: s for n, s in WW.REGISTRY.items() if s["set"] == "control"}
sdp = {n: s for n, s in WW.REGISTRY.items() if s["set"] == "sdpose"}
_need(len(ctl) == 6 and all(s["folder"] == "model_patches" for s in ctl.values()), "P3 control: 6 files in model_patches")
_need(len([n for n in ctl if n.startswith("minimax_h3_fun")]) == 4 and len([n for n in ctl if n.startswith("qwen_image_2.1")]) == 2,
      "P3 control: 4 H3 + 2 Qwen-Image 2.1")
rec = sorted(n for n, s in ctl.items() if s.get("recommended"))
_need(rec == ["minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors",
              "qwen_image_2.1_fun_controlnet_union_int8_convrot.safetensors"], "P3 one recommended per family (int8)")
_need(list(sdp) == ["sdpose_wholebody_fp16.safetensors"] and sdp["sdpose_wholebody_fp16.safetensors"]["folder"] == "checkpoints",
      "P3 sdpose: the wholebody fp16 in checkpoints")
ok = all(s["url"].startswith("https://huggingface.co/") and s["url"].endswith("/" + n) and len(s["sha256"]) == 64
         and all(c in "0123456789abcdef" for c in s["sha256"]) and s["bytes"] > 0 for n, s in list(ctl.items()) + list(sdp.items()))
_need(ok, "P3 https huggingface.co, the file at the end of the URL, sha256 64 hex, bytes > 0")
_need("control" in WW.SETS and "sdpose" in WW.SETS, "P3 both sets in SETS")

# ---------------------------------------------------------------- P4
print("P4 combos and resolution")
(MODELS / "model_patches").mkdir(parents=True)
(MODELS / "model_patches" / "my_union.safetensors").write_bytes(b"x" * 10)
sub = MODELS / "model_patches" / "fun"
sub.mkdir()
(sub / "minimax_h3_fun_controlnet_union_pruned_bf16.safetensors").write_bytes(b"y" * 2048)
ch = CT.door_choices("model_patches", "control")
_need(len(ch) == 7 and WW.strip(ch[0]) == "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors"
      and ch[0].endswith("not installed") and ch[-1] == "my_union.safetensors",
      "P4 door files first (missing ones marked), the folder's own file plain at the end (%d)" % len(ch))
_need(sum(1 for c in ch if "minimax_h3_fun_controlnet_union_pruned_bf16" in c) == 1
      and any(c.startswith("minimax_h3_fun_controlnet_union_pruned_bf16.safetensors \u00b7 2 KB") for c in ch),
      "P4 a door file already on disk (in a subfolder) is listed once, with its real size")
try:
    CT.resolve_file("model_patches", ch[0], "Apply Control", "control_model")
    _need(False, "P4 a missing door file resolved")
except ValueError as e:
    _need("not installed" in str(e) and "Download" in str(e) and "huggingface.co" in str(e),
          "P4 a missing door file is refused with the way to the Download")
p = CT.resolve_file("model_patches", "minimax_h3_fun_controlnet_union_pruned_bf16.safetensors \u00b7 2 KB", "A", "c")
_need(p == str(sub / "minimax_h3_fun_controlnet_union_pruned_bf16.safetensors"), "P4 a door file on disk is found in its subfolder")
_need(CT.resolve_file("model_patches", "my_union.safetensors", "A", "c").endswith("my_union.safetensors"),
      "P4 a hand-placed file resolves through ComfyUI")
try:
    CT.resolve_file("model_patches", "(none)", "Apply Control", "control_model")
    _need(False, "P4 '(none)' resolved")
except ValueError as e:
    _need("no file picked" in str(e), "P4 '(none)' is refused by name")
ac = CT.ULSApplyControl.INPUT_TYPES()["required"]["control_model"]
cm = CT.ULSControlMap.INPUT_TYPES()["required"]
_need(WW.strip(ac[1]["default"]) == "minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors",
      "P4 Apply Control lists the door and defaults to the recommended H3 union")
_need(WW.strip(cm["pose_model"][1]["default"]) == "sdpose_wholebody_fp16.safetensors"
      and any("depth_anything_3" in v for v in cm["depth_model"][0]),
      "P4 Control Map: pose defaults to the door's SDPose, depth lists the door's DA3 files")

# ---------------------------------------------------------------- P5
print("P5 VALIDATE_INPUTS")
a1 = inspect.signature(CT.ULSControlMap.VALIDATE_INPUTS).parameters
a2 = inspect.signature(CT.ULSApplyControl.VALIDATE_INPUTS).parameters
_need(list(a1) == ["pose_model", "depth_model"] and list(a2) == ["control_model"],
      "P5 exactly the model combos leave Core's membership check (no **kwargs)")
_need(CT.ULSControlMap.VALIDATE_INPUTS() is True and CT.ULSApplyControl.VALIDATE_INPUTS() is True, "P5 they pass")

# ---------------------------------------------------------------- P6
print("P6 the line follows the pick")
WJ = (ROOT / "web" / "js" / "ph_world_weights.js").read_text(encoding="utf-8")
node_js = shutil.which("node")
if node_js:
    i = WJ.find("const PICK_NODES = "); j = WJ.find("};", i) + 2
    k = WJ.find("export function resolvePick("); k2 = WJ.find("\n}", WJ.find("export function pickWidgetNames(")) + 2
    code = WJ[i:j] + "\n" + WJ[k:k2] + r"""
const cm = PICK_NODES.ULSControlMap, ac = PICK_NODES.ULSApplyControl;
const at = (mode) => resolvePick(cm, () => mode);
const s = (p) => p ? p.widget + "/" + p.set : null;
console.log(JSON.stringify([s(at("pose (SDPose)")), s(at("depth (Depth Anything 3)")), s(at("canny")), s(at("gray")),
  s(at("keypoints (draw only)")), s(resolvePick(ac, () => "")), pickWidgetNames(cm), pickWidgetNames(ac),
  resolvePick(undefined, () => "")]));
"""
    (TMP / "p6.mjs").write_text(code.replace("export function", "function"), encoding="utf-8")
    out = subprocess.run([node_js, str(TMP / "p6.mjs")], capture_output=True, text=True, timeout=60)
    try:
        r = json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        r = None
    _need(r == ["pose_model/sdpose", "depth_model/da3", None, None, None, "control_model/control",
                ["mode", "pose_model", "depth_model"], ["control_model"], None],
          "P6 pose -> SDPose, depth -> DA3, the rest -> no line; Apply Control -> its pick; the mode is bound (%s %s)"
          % (r, out.stderr[-200:]))
else:
    _need(False, "P6 node.js needed")
_need("if (line && !pk && !line.busy) { line.hide(); return; }" in WJ, "P6 a mode that needs no model hides the line")

# ---------------------------------------------------------------- S
print("S ASCII")
for f in ("nodes/uls_placeholder.py", "nodes/ph_control.py", "nodes/ph_world_weights.py", "web/js/ph_world_weights.js",
          "tests/test_v1216_control_door.py"):
    try:
        (ROOT / f).read_bytes().decode("ascii")
        _need(True, "S %s" % f)
    except UnicodeDecodeError:
        _need(False, "S %s is not ASCII" % f)

shutil.rmtree(TMP, ignore_errors=True)
print()
print("%s: %s (%d fail)" % (NAME, "PASS" if not FAILS else "FAIL", len(FAILS)))
sys.exit(1 if FAILS else 0)
