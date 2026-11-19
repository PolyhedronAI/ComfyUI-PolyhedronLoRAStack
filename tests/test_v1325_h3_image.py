#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1325 -- MiniMax H3 Image (Core 0.45.0, 10.11.2026) carried by the EXISTING nodes, no node of its own.

Frank, 13.11.: our nodes support the model instead of new nodes, unless it cannot be done otherwise.

WHAT IS PINNED

  H1  h3_image (library, no node): the variant comes from the VAE CLASS (MiniMaxH3ImageVAE -> H3 Image,
      PH3ImageVAE -> p-h3-image, anything else None); the zeroed negative (pooled too); a Core without the node is
      refused by name; the IMAGE VAE list equals Image Reference's.
  H2  against the REAL Core 0.45.0 (when /home/claude/core_new or PLS_CORE_NEW imports):
      a) the H3 tokenizer encodes a text with NO pictures exactly like a plain tokenize -- so the CLIP Text Encode
         IS Core's text-to-image conditioning;
      b) Core's latent for a size == the Empty Latent's type "MiniMax H3 Image" for the same size;
      c) the delegate hands Core the size it is given and 0 x 0 = 2048 x 2048.
  H3  Empty Latent: type "MiniMax H3 Image (24ch, one frame)" builds [B, 24, 1, H/16, W/16] on the 32 grid, appended
      last in Python AND in the JS mirror, no length row; the AV type is unchanged.
  H4  Image Reference: H3 tokenizer + H3 IMAGE VAE -> family minimax_h3_image (appended last), Core's recipe through
      h3_image; a wired latent sets the canvas, unwired Core decides; the video VAE keeps minimax_h3. (inpaint_mask:
      amended v1339, carried now -- see test_v1339.)
  H5  Sampler: sigma_shift leaves a model on its own distilled grid (MiniMaxH3ImageModelSampling) untouched and says
      so; any other model is still patched.
  H6  Load CLIP 'auto' resolves MiniMaxH3 and its subclass MiniMaxH3Image to 'minimax'.
  H7  Apply Control names the H3 Image model as its own family (a video Fun ControlNet is refused on it).
  H8  no node of its own: nothing registered, no JS file; the library is ASCII-clean.
  H9  Nodes 2.0 under Frontend 1.59: the control row ("control after generate") goes AFTER the one-track box the
      frontend wraps each widget row in (inside it, it was drawn over the seed row -- measured, same y); with the
      older flat rows it goes after the row as before.

Script-style: exit 0 = pass.
"""

import importlib.util
import json
import os
import re
import subprocess
import sys
import types

NAME = "v1325"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODES = os.path.join(ROOT, "nodes")
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)
    print(("  ok    " if cond else "  FAIL  ") + msg)


for _c in (os.environ.get("PLS_CORE_ROOT"), "/home/claude/core_pin"):
    if _c and os.path.isdir(os.path.join(_c, "comfy")) and _c not in sys.path:
        sys.path.insert(0, _c)
        break

import torch  # noqa: E402

sys.argv = [sys.argv[0], "--cpu"]          # Core's cli_args: no GPU here
try:
    import comfy.options
    comfy.options.enable_args_parsing()    # main.py does this; else --cpu is ignored
except Exception:
    pass

pkg = types.ModuleType("plspack")
pkg.__path__ = [NODES]
sys.modules["plspack"] = pkg


def load(name):
    spec = importlib.util.spec_from_file_location("plspack." + name, os.path.join(NODES, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["plspack." + name] = m
    spec.loader.exec_module(m)
    return m


HI = load("h3_image")

# ---- H1 the library ---------------------------------------------------------------
_need(HI.variant_for("MiniMaxH3ImageVAE") == HI.VARIANT_H3, "H1 image VAE -> H3 Image")
_need(HI.variant_for("PH3ImageVAE") == HI.VARIANT_PH3, "H1 p-h3 VAE -> p-h3-image")
_need(HI.variant_for("MiniMaxH3VideoVAE") is None and HI.variant_for("") is None, "H1 the video VAE / none -> None")
_need(HI.vae_class(None) == "" and HI.vae_class(types.SimpleNamespace(
    first_stage_model=type("MiniMaxH3ImageVAE", (), {})())) == "MiniMaxH3ImageVAE", "H1 vae_class reads the model class")
cond = [[torch.ones(1, 4, 8), {"pooled_output": torch.ones(1, 3), "x": 1}]]
z = HI.zero_out(cond)
_need(float(z[0][0].abs().sum()) == 0 and float(z[0][1]["pooled_output"].abs().sum()) == 0 and z[0][1]["x"] == 1
      and float(cond[0][0].sum()) > 0, "H1 negative zeroed (pooled too), the input untouched")
_orig = HI.core_class
HI.core_class = lambda variant: None
try:
    HI.encode("CLIP", "a fox", 0, 0)
    _need(False, "H1 a Core without the node must be refused")
except RuntimeError as e:
    _need("0.45.0" in str(e) and "MiniMaxH3ImageConditioning" in str(e), "H1 an old Core is refused by name")
HI.core_class = _orig

# ---- H2 the real Core ---------------------------------------------------------------
core_new = os.environ.get("PLS_CORE_NEW", "/home/claude/core_new")
probe = r"""
import sys, json, torch
sys.path.insert(0, %r); sys.path.insert(0, %r)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options; comfy.options.enable_args_parsing()
import comfy_extras.nodes_minimax_h3 as M
import h3_image as HI
out = {}
try:
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer
    tok = MiniMaxH3Tokenizer()
    t = "a red fox in deep snow, low sun from the left, frost on the whiskers"
    a = tok.tokenize_with_weights(t)
    b = tok.tokenize_with_weights(t, images=[], picture_label="Picture %%d: ")
    out["tok_equal"] = (a == b)
    out["tok_len"] = len(list(a.values())[0][0])
except Exception as e:
    out["tok_err"] = repr(e)[:200]
class Clip:
    def tokenize(self, text, **kw): return {"kw": sorted(kw)}
    def encode_from_tokens_scheduled(self, tok): return [[torch.ones(1, 3, 4), {}]]
for w, h in ((0, 0), (6272, 2688), (2496, 1664)):
    pos, lat = HI.encode(Clip(), "a fox", w, h, None, None, HI.VARIANT_H3)
    out["%%dx%%d" %% (w, h)] = list(lat["samples"].shape)
print("JSON" + json.dumps(out))
""" % (core_new, NODES)
r = None
pys = [sys.executable] + [p for p in ("/home/claude/venv_new/bin/python",) if os.path.isfile(p)]
if os.path.isfile(os.path.join(core_new, "comfy_extras", "nodes_minimax_h3.py")):
    for py in pys:
        r = subprocess.run([py, "-c", probe], capture_output=True, text=True, timeout=600, cwd=core_new)
        if "JSON" in r.stdout:
            break
CORE = None
if r is not None and "JSON" in r.stdout:
    CORE = json.loads(r.stdout.split("JSON", 1)[1].strip().splitlines()[0])
    if "tok_err" in CORE:
        print("  SKIP  H2a tokenizer not loadable here: %s" % CORE["tok_err"])
    else:
        _need(CORE["tok_equal"] and CORE["tok_len"] > 5,
              "H2a no pictures: the H3 tokenizer gives the plain text tokens (%d) -> the CTE is Core's t2i encode"
              % CORE["tok_len"])
    _need(CORE["0x0"] == [1, 24, 1, 128, 128], "H2c 0 x 0 = 2048 x 2048 -> %r" % CORE["0x0"])
    _need(CORE["6272x2688"] == [1, 24, 1, 168, 392], "H2c the size reaches Core as given")
else:
    print("  SKIP  H2 no Core 0.45.0 importable here (%s)" % ((r.stderr or "")[-200:] if r else "not found"))

# ---- H3 Empty Latent -----------------------------------------------------------------
LM = load("uls_latent_math")
EL = load("ph_empty_latent")
lab = "MiniMax H3 Image (24ch, one frame)"
_need(LM.LATENT_TYPE_LABELS[-1] == lab and LM.canonical_type(lab) == "minimax_h3_image", "H3 the type is appended last")
_need(not LM.is_video_family(lab), "H3 not a video row (no length)")
js_el = open(os.path.join(ROOT, "web", "js", "ph_empty_latent.js"), encoding="utf-8").read()
m = re.search(r"const LATENT_TYPES = (\[.*?\]);", js_el)
_need(m and json.loads(m.group(1)) == LM.LATENT_TYPE_LABELS, "H3 the JS combo mirror equals Python")
_need('"%s": { sdiv: 16, tdiv: 0, video: false }' % lab in js_el, "H3 the JS geometry row")
out = EL.ULSEmptyLatent().generate(None, None, lab, 2500, 1660, 1, 2, "zeros", 0, 1.0, 0.0)
_need(tuple(out[2]["samples"].shape) == (2, 24, 1, 104, 156) and out[3:] == (156, 104, 2496, 1664),
      "H3 [B, 24, 1, H/16, W/16] on the 32 grid -> %r %r" % (tuple(out[2]["samples"].shape), out[3:]))
if CORE is not None:
    for w, h in ((6272, 2688), (2496, 1664)):
        el = list(EL.ULSEmptyLatent().generate(None, None, lab, w, h, 1, 1, "zeros", 0, 1.0, 0.0)[2]["samples"].shape)
        _need(el == CORE["%dx%d" % (w, h)], "H2b Empty Latent %dx%d == Core's latent %r" % (w, h, el))
av = EL.ULSEmptyLatent().generate(None, None, "MiniMax H3 AV (video+audio)", 1024, 1024, 1, 1, "zeros", 0, 1.0, 0.0)
_need(type(av[2]["samples"]).__name__ == "NestedTensor", "H3 the AV type is unchanged (joint latent)")

# ---- H4 Image Reference ----------------------------------------------------------------
sys.modules["plspack.h3_image"] = HI
IR = load("ph_image_ref")
_need(IR.FAMILIES[-1] == IR.FAMILY_MINIMAX_IMAGE == "minimax_h3_image", "H4 the family is appended last")
_need(tuple(IR.MINIMAX_IMAGE_VAES) == HI.IMAGE_VAES, "H4 Image Reference and h3_image name the same IMAGE VAEs")
_need(IR.resolve_family("auto", ["MiniMaxH3Tokenizer"], "MiniMaxH3ImageVAE")[0] == IR.FAMILY_MINIMAX_IMAGE,
      "H4 H3 tokenizer + image VAE -> minimax_h3_image")
_need(IR.resolve_family("auto", ["PH3ImageTokenizer", "MiniMaxH3Tokenizer"], "PH3ImageVAE")[0] == IR.FAMILY_MINIMAX_IMAGE,
      "H4 p-h3 tokenizer (an H3 subclass) + p-h3 VAE -> minimax_h3_image")
_need(IR.resolve_family("auto", ["MiniMaxH3Tokenizer"], "MiniMaxH3VideoVAE")[0] == IR.FAMILY_MINIMAX,
      "H4 the video VAE keeps minimax_h3")
_need(IR.resolve_family("auto", ["MiniMaxH3Tokenizer"])[0] == IR.FAMILY_MINIMAX, "H4 the old two-argument call is unchanged")
SEEN = {}


class FakeCore:
    @classmethod
    def execute(cls, clip, prompt, width, height, vae=None, images=None):
        SEEN.update(prompt=prompt, width=width, height=height, keys=list(images or {}))
        w, h = (width or 2048), (height or 2048)
        return types.SimpleNamespace(args=([[torch.ones(1, 4, 8), {}]],
                                           {"samples": torch.zeros(1, 24, 1, h // 16, w // 16)}))


class VaeImg:
    first_stage_model = type("MiniMaxH3ImageVAE", (), {})()


HI.core_class = lambda variant: FakeCore
clip = types.SimpleNamespace(tokenizer=type("MiniMaxH3Tokenizer", (), {})())
pos, neg, lat, info, text = IR.ULSImageReference().build(
    clip, "make the fox white", "ignored", vae=VaeImg(), image_3=torch.zeros(1, 8, 8, 3), image_1=torch.zeros(1, 8, 8, 3))
_need(SEEN.get("prompt") == "make the fox white" and SEEN.get("keys") == ["image_1", "image_3"]
      and (SEEN["width"], SEEN["height"]) == (0, 0), "H4 Core's recipe gets the prompt and the pictures in slot order, canvas left to Core")
_need("minimax_h3_image" in info and "image_1" in info and float(neg[0][0].abs().sum()) == 0,
      "H4 info names the family, negative zeroed")
SEEN.clear()
wired = {"samples": torch.zeros(1, 24, 1, 104, 156)}
pos, neg, lat, info, text = IR.ULSImageReference().build(
    clip, "make the fox white", "", vae=VaeImg(), latent=wired, image_1=torch.zeros(1, 8, 8, 3))
_need((SEEN["width"], SEEN["height"]) == (2496, 1664) and "wired latent" in info,
      "H4 a wired latent sets the canvas -> %r" % ((SEEN.get("width"), SEEN.get("height")),))
# AMENDED v1339 (Frank 17.11., Go): inpaint_mask is now CARRIED for the image family (image_1 encoded at the
# canvas, the mask as noise mask, the edit conditioning kept) -- pinned in tests/test_v1339_h3_image_inputs.py.
HI.core_class = _orig

# ---- H5 Sampler ---------------------------------------------------------------------------
src_s = open(os.path.join(NODES, "uls_sampler.py"), encoding="utf-8").read()
ns = {}
for fn in ("_sampling_name", "_owns_sigma_grid", "_is_av_sampling", "_apply_av_shift", "_apply_sigma_shift"):
    exec(re.search(r"^def %s\(.*?(?=^\S)" % fn, src_s, re.S | re.M).group(0), ns)
ns["GRID_SAMPLINGS"] = eval(re.search(r"^GRID_SAMPLINGS = (.*)$", src_s, re.M).group(1))


class Model:
    def __init__(self, ms):
        self.ms = ms
        self.cloned = False

    def get_model_object(self, k):
        return self.ms

    def clone(self):
        self.cloned = True
        raise RuntimeError("patched")


mdl = Model(type("MiniMaxH3ImageModelSampling", (), {})())
_need(ns["_apply_sigma_shift"](mdl, 5.0) is mdl and not mdl.cloned, "H5 a grid model is left alone at sigma_shift 5")
other = Model(type("ModelSamplingDiscreteFlow", (), {})())
try:
    ns["_apply_sigma_shift"](other, 5.0)
    _need(False, "H5 another model must still be patched")
except RuntimeError:
    _need(other.cloned, "H5 another model is still patched")
_need(ns["_apply_sigma_shift"](other, 0.0) is other, "H5 shift 0 stays a no-op")

# ---- H6 Load CLIP auto ------------------------------------------------------------------------
src_b = open(os.path.join(NODES, "ph_basics.py"), encoding="utf-8").read()
nb = {}
exec(re.search(r"^_MODEL_CLASS_CLIP_TYPE = \{.*?^\}", src_b, re.S | re.M).group(0), nb)
H3 = type("MiniMaxH3", (), {})
H3I = type("MiniMaxH3Image", (H3,), {})
hit = next((nb["_MODEL_CLASS_CLIP_TYPE"][c.__name__] for c in H3I.__mro__ if c.__name__ in nb["_MODEL_CLASS_CLIP_TYPE"]), None)
_need(nb["_MODEL_CLASS_CLIP_TYPE"].get("MiniMaxH3") == "minimax" and hit == "minimax",
      "H6 MiniMaxH3 and its image subclass resolve to minimax")

# ---- H7 Apply Control --------------------------------------------------------------------------
src_c = open(os.path.join(NODES, "ph_control.py"), encoding="utf-8").read()
nc = {}
exec(re.search(r"^def model_family\(.*?(?=^\S)", src_c, re.S | re.M).group(0), nc)


def dm_model(cls_name):
    cls = type(cls_name, (), {})
    cls.__module__ = "comfy.ldm.minimax.model"
    return types.SimpleNamespace(get_model_object=lambda k: cls())


_need(nc["model_family"](dm_model("MiniMaxH3ImageModel")) == "MiniMax H3 Image", "H7 the image DiT is its own family")
_need(nc["model_family"](dm_model("MiniMaxH3Model")) == "video", "H7 the video DiT stays video")

# ---- H8 no node of its own -----------------------------------------------------------------------
init = open(os.path.join(ROOT, "__init__.py"), encoding="utf-8").read()
_need("ULSMiniMaxImage" not in init and "h3_image" not in init, "H8 nothing new is registered")
_need(not os.path.exists(os.path.join(ROOT, "web", "js", "ph_minimax_image.js"))
      and not os.path.exists(os.path.join(NODES, "ph_minimax_image.py")), "H8 no node module, no JS of its own")
raw = open(os.path.join(NODES, "h3_image.py"), "rb").read()
_need(all(b < 128 for b in raw), "H8 nodes/h3_image.py is pure ASCII")

# ---- H9 the control row under Frontend 1.59 ---------------------------------------------------------
vp = open(os.path.join(ROOT, "web", "js", "uls_vue_parity.js"), encoding="utf-8").read()
mm = re.search(r"export function controlAnchor\(row\) \{.*?\n\}\n", vp, re.S)
_need(mm is not None and "controlAnchor(anchor).insertAdjacentElement(\"afterend\", row)" in vp,
      "H9 applyControlRows inserts after controlAnchor(anchor)")
if mm:
    harness = mm.group(0).replace("export function", "function") + r"""
const cls = (names) => ({ contains: (c) => names.includes(c) });
const wrap = { classList: cls(["col-span-full", "grid", "grid-rows-subgrid"]), querySelectorAll: () => [1] };
const wrap2 = { classList: cls(["grid-rows-subgrid"]), querySelectorAll: () => [1, 2] };
const flat = { classList: cls(["grid"]), querySelectorAll: () => [1, 2, 3] };
const r159 = { parentElement: wrap }, rOld = { parentElement: flat }, rTwo = { parentElement: wrap2 };
console.log(JSON.stringify([controlAnchor(r159) === wrap, controlAnchor(rOld) === rOld, controlAnchor(rTwo) === rTwo, controlAnchor(null) === null]));
"""
    import tempfile
    tf = os.path.join(tempfile.mkdtemp(prefix="v1325h9_"), "h.mjs")
    open(tf, "w").write(harness)
    rr = subprocess.run(["node", tf], capture_output=True, text=True, timeout=60)
    got = json.loads(rr.stdout.strip().splitlines()[-1]) if rr.returncode == 0 else None
    _need(got == [True, True, True, True],
          "H9 1.59 box -> after the box; flat rows / a shared box -> after the row -> %r %s" % (got, rr.stderr[-200:]))

if _fails:
    print("[%s] FAIL -- %d" % (NAME, len(_fails)))
    sys.exit(1)
print("[%s] PASS" % NAME)
