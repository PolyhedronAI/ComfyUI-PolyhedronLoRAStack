"""Guard v1078 -- Polyhedron Image Reference (ULSImageReference), one node for
every family that takes pictures next to its prompt.

WHAT THIS PINS (driven wherever a mechanism exists):

  R1  fit_area equals Core's own formulas for both Qwen recipes, on real
      sizes: 2.1 (sqrt(res^2 * ratio) rounded to /32, nodes_qwen.py
      TextEncodeQwenImage21), Edit vision 384^2 (plain rounding) and Edit
      VAE 1024^2 at /8 (TextEncodeQwenImageEditPlus). side 0 only snaps.
  R2  family resolution: measured on the tokenizer MRO names; the 2.1
      tokenizer (a subclass of the Edit one? no -- of Qwen3VL) and the Edit
      tokenizer resolve apart; an unknown tokenizer falls back to
      reference_latents AND says so; an explicit choice wins; a bad choice
      raises by name.
  R3  DRIVEN: the 2.1 recipe against a recording clip + vae -- every image
      reaches tokenize() as RGB with alpha over WHITE (a half-transparent
      red pixel becomes pink), the VAE gets FOUR channels when it takes
      four, keep_vision is False with a VAE and True without, both
      conditionings carry one reference_latents entry per image, the
      latent is 64 ch at /16 on the FIRST reference's fitted size, the
      negative is encoded with the same images.
  R4  DRIVEN: the Edit recipe -- vision copies at the 384^2 area, VAE copies
      at the 1024^2 /8 area, the 'Picture i:' prefix once per image in front
      of the user text, the edit llama template, latent 16 ch at /8.
  R5  DRIVEN: reference_latents -- plain tokenize (no images kwarg), one
      reference latent per image, the VAE's own channel count and ratio
      decide the latent; without a VAE it raises by name.
  R6  DRIVEN: minimax_h3 hands the images to ULSMiniMaxReference.build with
      the SAME image kwargs and returns its conditioning/latent; without
      vae or latent it raises by name (the canvas comes from the wire).
  R7  the JS half: N_IMAGES and the pin names mirror the backend; tidy()
      driven against a fake node -- fresh node shows image_1 only, wiring
      grows one spare, a hole stays, a wired pin is never dropped, links
      keep their target_slot after a reorder, idempotent.
  R8  registration: the node_id is registered with a display name; the
      Empty Latent family table carries qwen_image21 (64, 16) with its label
      appended LAST; the JS latent mirror carries the same label.

Script-style: exit 0 = pass.
"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

NAME = "v1078"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MOD = os.path.join(ROOT, "nodes", "ph_image_ref.py")


def _fail(msg):
    print("[%s] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


# ---- module under test, as a member of a throwaway package ---------------
pkg = types.ModuleType("plspack")
pkg.__path__ = [os.path.join(ROOT, "nodes")]
sys.modules["plspack"] = pkg
spec = importlib.util.spec_from_file_location("plspack.ph_image_ref", MOD)
IR = importlib.util.module_from_spec(spec)
sys.modules["plspack.ph_image_ref"] = IR
spec.loader.exec_module(IR)

# ---- R1: sizes equal Core's formulas -------------------------------------
import math


def core_21(w, h, res):
    ratio = w / h
    return (round(math.sqrt(res * res * ratio) / 32) * 32,
            round(math.sqrt(res * res / ratio) / 32) * 32)


def core_edit_vl(w, h):
    s = math.sqrt(384 * 384 / (w * h))
    return round(w * s), round(h * s)


def core_edit_vae(w, h):
    s = math.sqrt(1024 * 1024 / (w * h))
    return round(w * s / 8.0) * 8, round(h * s / 8.0) * 8


for (w, h) in [(1024, 1024), (1024, 1280), (204, 256), (1920, 1080), (640, 1536), (333, 777)]:
    for res in (1024, 1536, 2048):
        _need(IR.fit_area(w, h, res, 32) == core_21(w, h, res),
              "R1: 2.1 size for %dx%d @%d is %r, Core says %r"
              % (w, h, res, IR.fit_area(w, h, res, 32), core_21(w, h, res)))
    _need(IR.fit_area(w, h, 384, 1) == core_edit_vl(w, h),
          "R1: Edit vision size for %dx%d" % (w, h))
    _need(IR.fit_area(w, h, 1024, 8) == core_edit_vae(w, h),
          "R1: Edit VAE size for %dx%d" % (w, h))
_need(IR.fit_area(1000, 500, 0, 32) == (992, 512), "R1: side 0 only snaps to the grid")
_need(IR.fit_area(10, 10, 0, 32) == (32, 32), "R1: never below one grid")
try:
    IR.fit_area(0, 10, 1024, 32)
    _fail("R1: an empty image must raise")
except ValueError:
    pass

# ---- R2: family resolution ------------------------------------------------
_need(IR.resolve_family("auto", ["QwenImage21Tokenizer", "Qwen3VLTokenizer", "object"])[0] == IR.FAMILY_QWEN21,
      "R2: the 2.1 tokenizer must resolve to qwen_image_2.1")
_need(IR.resolve_family("auto", ["QwenImageTokenizer", "SD1Tokenizer", "object"])[0] == IR.FAMILY_QWEN_EDIT,
      "R2: the Edit tokenizer must resolve to qwen_image_edit")
_need(IR.resolve_family("auto", ["MiniMaxH3Tokenizer", "SD1Tokenizer"])[0] == IR.FAMILY_MINIMAX,
      "R2: the H3 tokenizer must resolve to minimax_h3")
fam, how = IR.resolve_family("auto", ["Flux2Tokenizer", "object"])
_need(fam == IR.FAMILY_REFLAT and "reference latents" in how and "Flux2Tokenizer" in how,
      "R2: an unknown tokenizer falls back to reference_latents and SAYS so, got %r/%r" % (fam, how))
_need(IR.resolve_family("qwen_image_edit", ["QwenImage21Tokenizer"]) == ("qwen_image_edit", "set"),
      "R2: an explicit choice wins")
try:
    IR.resolve_family("nonsense", [])
    _fail("R2: a bad explicit family must raise")
except ValueError as e:
    _need("nonsense" in str(e), "R2: the error must name the bad family")
_need(IR.edit_prefix(2) == "Picture 1: <|vision_start|><|image_pad|><|vision_end|>"
      "Picture 2: <|vision_start|><|image_pad|><|vision_end|>", "R2: the Edit prefix")
_need(IR.edit_prefix(0) == "", "R2: no images, no prefix")

# ---- stand-ins for the driven runs ---------------------------------------
import torch

comfy = types.ModuleType("comfy")
cu = types.ModuleType("comfy.utils")


def _common_upscale(samples, w, h, method, crop):
    return torch.nn.functional.interpolate(samples, size=(h, w), mode="bilinear",
                                           align_corners=False)


cu.common_upscale = _common_upscale
mm = types.ModuleType("comfy.model_management")
mm.intermediate_device = lambda: torch.device("cpu")
comfy.utils = cu
comfy.model_management = mm
sys.modules["comfy"] = comfy
sys.modules["comfy.utils"] = cu
sys.modules["comfy.model_management"] = mm


class Tok:
    pass


class QwenImage21Tokenizer(Tok):
    pass


class QwenImageTokenizer(Tok):
    pass


class MiniMaxH3Tokenizer(Tok):
    pass


class Flux2Tokenizer(Tok):
    pass


class Clip:
    def __init__(self, tok):
        self.tokenizer = tok
        self.calls = []

    def tokenize(self, text, **kw):
        rec = {"text": text, "kw": {k: v for k, v in kw.items() if k != "images"},
               "images": [i.clone() for i in kw.get("images", [])]}
        self.calls.append(rec)
        return rec

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 4, 8), {"tokens": tokens["text"]}]]


class Vae:
    def __init__(self, channels=3, latent_channels=16, ratio=8):
        self.output_channels = channels
        self.latent_channels = latent_channels
        self.downscale_ratio = ratio
        self.seen = []

    def encode(self, px):
        self.seen.append(tuple(px.shape))
        return torch.zeros(1, self.latent_channels, px.shape[1] // self.downscale_ratio,
                           px.shape[2] // self.downscale_ratio)


def rgba(w, h, alpha=0.5):
    img = torch.zeros(1, h, w, 4)
    img[..., 0] = 1.0          # red
    img[..., 3] = alpha
    return img


def rgb(w, h):
    img = torch.zeros(1, h, w, 3)
    img[..., 1] = 1.0
    return img


Node = IR.ULSImageReference

# ---- R3: the 2.1 recipe ----------------------------------------------------
clip = Clip(QwenImage21Tokenizer())
vae = Vae(channels=4, latent_channels=64, ratio=16)
pos, neg, lat, info, text = Node().build(clip, "turn her to the left", "blurry",
                                          family="auto", ref_size=1024, vae=vae,
                                          image_1=rgba(1024, 1280, 0.5), image_2=rgb(512, 512))
_need(len(clip.calls) == 2, "R3: positive and negative = two tokenize calls")
p = clip.calls[0]
_need(len(p["images"]) == 2 and all(i.shape[-1] == 3 for i in p["images"]),
      "R3: the vision tower sees RGB copies of both images")
px = p["images"][0][0, 0, 0]
_need(abs(px[0] - 1.0) < 1e-4 and abs(px[1] - 0.5) < 1e-4 and abs(px[2] - 0.5) < 1e-4,
      "R3: alpha over WHITE -- half-transparent red must be pink, got %r" % (px.tolist(),))
_need(tuple(p["images"][0].shape[1:3]) == (core_21(1024, 1280, 1024)[1], core_21(1024, 1280, 1024)[0]),
      "R3: vision copy at Core's 2.1 size, got %r" % (tuple(p["images"][0].shape),))
_need(p["kw"].get("keep_vision") is False and p["kw"].get("prevent_empty_text") is True,
      "R3: with a VAE keep_vision is False and empty text is prevented, got %r" % (p["kw"],))
_need(vae.seen[0][-1] == 4, "R3: a four-channel VAE gets FOUR channels (alpha kept), got %r" % (vae.seen,))
_need(len(pos[0][1]["reference_latents"]) == 2 and len(neg[0][1]["reference_latents"]) == 2,
      "R3: one reference latent per image on BOTH conditionings")
fw, fh = core_21(1024, 1280, 1024)
_need(tuple(lat["samples"].shape) == (1, 64, fh // 16, fw // 16),
      "R3: latent 64ch at /16 on the first reference, got %r" % (tuple(lat["samples"].shape),))
_need(clip.calls[1]["text"] == "blurry" and len(clip.calls[1]["images"]) == 2,
      "R3: the negative is encoded with the same images")
_need("qwen_image_2.1" in info and "auto from QwenImage21Tokenizer" in info, "R3: info names family and how: %r" % info)
# without a VAE: vision kept, no reference latents, latent still sized
clip2 = Clip(QwenImage21Tokenizer())
pos2, neg2, lat2, info2, _ = Node().build(clip2, "x", "", image_1=rgb(640, 640))
_need(clip2.calls[0]["kw"].get("keep_vision") is True, "R3: no VAE -> keep_vision True")
_need("reference_latents" not in pos2[0][1], "R3: no VAE -> no reference latents")
_need(tuple(lat2["samples"].shape) == (1, 64, 64, 64), "R3: latent from the FITTED size (640 -> 1024 at ref_size 1024, as Core) without a VAE")

# ---- R4: the Edit recipe ---------------------------------------------------
clip = Clip(QwenImageTokenizer())
vae = Vae(channels=3, latent_channels=16, ratio=8)
pos, neg, lat, info, text = Node().build(clip, "make the bag blue", "", vae=vae,
                                          image_1=rgb(1024, 1280), image_2=rgb(300, 300))
p = clip.calls[0]
vl_w, vl_h = core_edit_vl(1024, 1280)
_need(tuple(p["images"][0].shape[1:3]) == (vl_h, vl_w), "R4: vision copy at the 384^2 area, got %r" % (tuple(p["images"][0].shape),))
va_w, va_h = core_edit_vae(1024, 1280)
_need(vae.seen[0][1:3] == (va_h, va_w) and vae.seen[0][-1] == 3,
      "R4: VAE copy at the 1024^2 /8 area with three channels, got %r" % (vae.seen,))
_need(p["text"] == IR.edit_prefix(2) + "make the bag blue", "R4: the Picture prefix precedes the user text")
_need(p["kw"].get("llama_template") == IR.EDIT_TEMPLATE, "R4: the edit llama template is passed")
_need(text == p["text"], "R4: prompt_text is the text that reached the encoder")
_need(tuple(lat["samples"].shape) == (1, 16, va_h // 8, va_w // 8), "R4: latent 16ch at /8, got %r" % (tuple(lat["samples"].shape),))

# ---- R5: reference_latents -------------------------------------------------
clip = Clip(Flux2Tokenizer())
vae = Vae(channels=3, latent_channels=128, ratio=16)
pos, neg, lat, info, text = Node().build(clip, "a fox", "", vae=vae, ref_size=1024,
                                          image_1=rgb(1000, 1000))
_need("images" not in clip.calls[0]["kw"] and not clip.calls[0]["images"],
      "R5: plain tokenize, no images kwarg for a text-only tokenizer")
_need(len(pos[0][1]["reference_latents"]) == 1, "R5: one reference latent")
_need(tuple(lat["samples"].shape) == (1, 128, 64, 64), "R5: the VAE's channels and ratio decide, got %r" % (tuple(lat["samples"].shape),))
_need("reference latents only" in info, "R5: the info says the fallback was taken")
try:
    Node().build(Clip(Flux2Tokenizer()), "a fox", "", image_1=rgb(64, 64))
    _fail("R5: reference_latents without a VAE must raise")
except ValueError as e:
    _need("VAE" in str(e), "R5: the error must name the VAE")

# ---- R6: minimax_h3 delegates ---------------------------------------------
mmref = types.ModuleType("plspack.ph_minimax_ref")
SEEN = {}


class ULSMiniMaxReference:
    def build(self, clip, vae, latent, prompt, ref_image_size, m1, m2, m3, **opt):
        SEEN.update({"keys": sorted(opt), "mode": ref_image_size, "prompt": prompt})
        return ([[torch.zeros(1), {"h3": True}]], latent, "h3-info", prompt)


mmref.ULSMiniMaxReference = ULSMiniMaxReference
sys.modules["plspack.ph_minimax_ref"] = mmref
clip = Clip(MiniMaxH3Tokenizer())
lat_in = {"samples": torch.zeros(1, 24, 5, 8, 8)}
pos, neg, lat, info, text = Node().build(clip, "<Picture 1> walks", "", vae=Vae(), latent=lat_in,
                                          image_1=rgb(64, 64), image_3=rgb(64, 64))
_need(SEEN["keys"] == ["image_1", "image_3"] and SEEN["mode"] == "match",
      "R6: the same image kwargs reach MiniMax Reference, got %r" % (SEEN,))
_need(pos[0][1].get("h3") is True and lat is lat_in, "R6: its conditioning and the wired latent come back")
_need(clip.calls and clip.calls[-1]["text"] == "" and not clip.calls[-1]["images"],
      "R6: the negative is a plain text encode")
try:
    Node().build(Clip(MiniMaxH3Tokenizer()), "x", "", vae=Vae(), image_1=rgb(64, 64))
    _fail("R6: minimax_h3 without the latent must raise")
except ValueError as e:
    _need("latent" in str(e), "R6: the error must name the latent")

# ---- R7: the JS half -------------------------------------------------------
JS = Path(ROOT) / "web" / "js" / "ph_image_ref.js"
VIS = Path(ROOT) / "web" / "js" / "ph_widget_vis.js"
_need(JS.is_file(), "R7: web/js/ph_image_ref.js missing")
tmp = Path(tempfile.mkdtemp(prefix="v1078js_"))
src = JS.read_text(encoding="utf-8").replace(
    'import { app } from "../../scripts/app.js";',
    'const app = { registerExtension() {} };')
(tmp / "ph_image_ref.mjs").write_text(src, encoding="utf-8")
shutil.copy(VIS, tmp / "ph_widget_vis.js")
harness = r"""
import * as IR from "./ph_image_ref.mjs";
globalThis.LiteGraph = { INPUT: 1 };
const out = { n: IR.N_IMAGES, order: IR.DISPLAY_ORDER, types: IR.PIN_TYPE };
let LID = 100;
function mkNode(names, links) {
  const g = { links: new Map() };
  const node = { graph: g, size: [300, 200], inputs: [], widgets: [],
    addInput(name, type) { this.inputs.push({ name, type, link: null }); },
    removeInput(i) { this.inputs.splice(i, 1); },
    setSize(s) { this.size = s; }, computeSize() { return [300, 20 * this.inputs.length]; },
    setDirtyCanvas() {} };
  for (const n of names) node.inputs.push({ name: n, type: IR.PIN_TYPE[n] || "STRING", link: null });
  for (const n of (links || [])) wire(node, n);
  return node;
}
function wire(node, name) { const p = node.inputs.find((i) => i.name === name); const id = LID++; p.link = id;
  node.graph.links.set(id, { id, target_slot: node.inputs.indexOf(p) }); }
const names = (n) => n.inputs.map((i) => i.name);
const slotsOk = (n) => n.inputs.every((p, i) => p.link == null || n.graph.links.get(p.link).target_slot === i);
const fresh = mkNode(IR.DISPLAY_ORDER); IR.tidy(fresh); out.fresh = names(fresh);
wire(fresh, "image_1"); IR.tidy(fresh); out.grow = names(fresh);
const h = mkNode(IR.DISPLAY_ORDER); IR.tidy(h); wire(h, "image_1"); IR.tidy(h); wire(h, "image_2"); IR.tidy(h);
wire(h, "image_3"); IR.tidy(h); h.inputs.find((i) => i.name === "image_2").link = null; IR.tidy(h); out.hole = names(h);
const old = mkNode(["clip", "image_3", "vae", "latent", "prompt"], ["image_3", "prompt"]); IR.tidy(old);
out.old = { names: names(old), slots: slotsOk(old), wired: old.inputs.filter((i) => i.link != null).map((i) => i.name) };
const before = JSON.stringify(names(old)); IR.tidy(old); out.idem = before === JSON.stringify(names(old)) && slotsOk(old);
console.log(JSON.stringify(out));
"""
(tmp / "h.mjs").write_text(harness, encoding="utf-8")
r = subprocess.run(["node", str(tmp / "h.mjs")], capture_output=True, text=True, timeout=60)
_need(r.returncode == 0, "R7: node harness failed: " + (r.stderr or "")[-600:])
js = json.loads(r.stdout.strip().splitlines()[-1])   # the module logs a load line first
_need(js["n"] == IR.N_IMAGES and js["order"] == ["clip", "vae", "latent"] + IR.IMAGE_KEYS,
      "R7: the JS pin list must mirror the backend, got %r" % (js["order"],))
_need(js["fresh"] == ["clip", "vae", "latent", "image_1"], "R7: a fresh node shows image_1 only, got %r" % (js["fresh"],))
_need(js["grow"] == ["clip", "vae", "latent", "image_1", "image_2"], "R7: wiring grows ONE spare, got %r" % (js["grow"],))
_need(js["hole"] == ["clip", "vae", "latent", "image_1", "image_2", "image_3", "image_4"],
      "R7: a hole stays visible, got %r" % (js["hole"],))
_need(js["old"]["names"] == ["clip", "vae", "latent", "image_1", "image_2", "image_3", "image_4", "prompt"]
      and js["old"]["slots"] and js["old"]["wired"] == ["image_3", "prompt"],
      "R7: an old save heals into DISPLAY_ORDER with links intact, got %r" % (js["old"],))
_need(js["idem"], "R7: tidy must be idempotent")

# ---- R8: registration + latent family -------------------------------------
init = open(os.path.join(ROOT, "__init__.py"), encoding="utf-8").read()
_need('NODE_CLASS_MAPPINGS["ULSImageReference"] = ULSImageReference' in init
      and 'NODE_DISPLAY_NAME_MAPPINGS["ULSImageReference"]' in init,
      "R8: ULSImageReference must be registered with a display name")
lm_spec = importlib.util.spec_from_file_location("plspack.uls_latent_math",
                                                 os.path.join(ROOT, "nodes", "uls_latent_math.py"))
LM = importlib.util.module_from_spec(lm_spec)
lm_spec.loader.exec_module(LM)
_need(LM.TYPE_SPEC.get("qwen_image21", (None,))[:2] == (64, 16), "R8: qwen_image21 = 64 channels at /16")
# v1325: minimax_h3_image was appended after it -- what this pins is that qwen_image21 joined at the END of its day
# and never moved (labels serialize by value): its slot stays 9 (declared re-grounding).
_need(LM.TYPE_ORDER.index("qwen_image21") == 9 and LM.TYPE_ORDER[:9][-1] == "minimax_h3",
      "R8: the new family was appended LAST and keeps its slot (labels serialize by value)")
label = LM.TYPE_LABELS["qwen_image21"]
el = open(os.path.join(ROOT, "web", "js", "ph_empty_latent.js"), encoding="utf-8").read()
_need(('"%s"' % label) in el and ('"%s": { sdiv: 16' % label) in el,
      "R8: the JS latent mirror must carry %r with sdiv 16" % label)
_need(LM.ALIASES.get("qwen_image_2.1") == "qwen_image21", "R8: the family spelling resolves")

print("[%s] PASS -- sizes equal Core's formulas, family measured on the tokenizer, "
      "2.1 / Edit / reference-latents / H3 recipes driven, JS progression driven, "
      "registered, latent family appended" % NAME)
