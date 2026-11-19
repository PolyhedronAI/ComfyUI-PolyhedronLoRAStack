#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1339 -- Image Reference, family minimax_h3_image: input pictures before Core, and inpainting.

Frank, 17.11.: "check whether our Reference node is equipped for H3 Image -- input pictures". Found and built:

  I1  alpha: Core's _resize/_resize_to CUT a fourth channel; we lay it over WHITE first (h3_image.flatten_alpha,
      the rule of our Qwen route). Three channels pass untouched. The info names the pins.
  I2  a batch on a pin: one frame (Core takes [:1] too) -- the info says so.
  I3  inpaint_mask is CARRIED (was refused in v1325): the edit conditioning stays, the latent becomes image_1
      VAE-encoded at the canvas size (resized when needed), the mask its noise mask (grown, feathered outward);
      the info reports the share. Without the VAE or image_1 it is refused by name (the general rule).
  I4  ph_image_ref.js against a node that behaves like the LIVE frontend (measured 17.11.): assigning
      node.inputs re-reads every pin's link from the links' target_slot. A wired widget socket (prompt) must
      keep its wire when image_2 appears; the old order (swap first, slots after) is shown to lose it.
  I5  ASCII.

Script-style: exit 0 = pass.
"""

import importlib.util
import os
import sys
import types

NAME = "v1339"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODES = os.path.join(ROOT, "nodes")
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)
        print("  FAIL " + msg)
    else:
        print("  ok   " + msg)


core = os.environ.get("PLS_CORE_ROOT") or "/home/claude/core_pin"
if os.path.isdir(core):
    sys.path.insert(0, core)
import torch  # noqa: E402

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
IR = load("ph_image_ref")

# ---- I1 ---------------------------------------------------------------------------------------------------
print("I1 alpha over white")
rgba = torch.zeros(1, 4, 4, 4)
rgba[..., 0] = 1.0                        # hidden red under ...
rgba[:, :2, :, 3] = 1.0                   # ... an opaque top half and a transparent bottom half
flat = HI.flatten_alpha(rgba)
_need(tuple(flat.shape) == (1, 4, 4, 3), "four channels -> three")
_need(torch.allclose(flat[0, 0, 0], torch.tensor([1.0, 0.0, 0.0])), "opaque part keeps its colour")
_need(torch.allclose(flat[0, 3, 0], torch.tensor([1.0, 1.0, 1.0])), "transparent part becomes white, not the hidden colour")
rgb = torch.rand(1, 4, 4, 3)
_need(HI.flatten_alpha(rgb) is rgb, "three channels pass untouched")

# ---- route through the node with a fake Core ----------------------------------------------------------------
SEEN = {}


class FakeCore:
    @classmethod
    def execute(cls, clip, prompt, width, height, vae=None, images=None):
        SEEN.update(prompt=prompt, width=width, height=height, images=dict(images or {}))
        w, h = (width or 128), (height or 96)
        return types.SimpleNamespace(args=([[torch.ones(1, 4, 8), {}]],
                                           {"samples": torch.zeros(1, 24, 1, h // 16, w // 16)}))


class VaeImg:
    first_stage_model = type("MiniMaxH3ImageVAE", (), {})()

    def __init__(self):
        self.seen = None

    def encode(self, px):
        self.seen = px
        return torch.full((1, 24, 1, int(px.shape[1]) // 16, int(px.shape[2]) // 16), 0.5)


HI.core_class = lambda variant: FakeCore
clip = types.SimpleNamespace(tokenizer=type("MiniMaxH3Tokenizer", (), {})())

print("I1/I2 through the node")
vae = VaeImg()
img4 = torch.zeros(2, 96, 128, 4)
img4[..., 3] = 0.0                        # fully transparent, hidden black
pos, neg, lat, info, text = IR.ULSImageReference().build(clip, "edit <Picture 1>", "", vae=vae, image_1=img4)
got = SEEN["images"]["image_1"]
_need(tuple(got.shape) == (1, 96, 128, 3) and float(got.min()) == 1.0,
      "Core gets one RGB frame, the transparent area white")
_need("alpha laid over white: image_1" in info, "info names the alpha pin")
_need("first frame only: image_1" in info, "info names the batch pin")
_need("noise_mask" not in lat, "no mask -> plain canvas latent as before")

# ---- I3 ---------------------------------------------------------------------------------------------------
print("I3 inpainting")
SEEN.clear()
vae = VaeImg()
mask = torch.zeros(1, 96, 128)
mask[:, 40:56, 60:80] = 1.0
img = torch.rand(1, 96, 128, 3)
pos, neg, lat, info, text = IR.ULSImageReference().build(
    clip, "give <Picture 1> glasses", "", vae=vae, image_1=img, inpaint_mask=mask,
    inpaint_grow=0, inpaint_feather=0)
_need("image_1" in SEEN["images"], "the edit conditioning keeps image_1 as <Picture 1>")
_need(tuple(lat["samples"].shape) == (1, 24, 1, 6, 8) and float(lat["samples"].mean()) == 0.5,
      "latent = image_1 encoded at the canvas (5-D, one frame)")
_need(vae.seen is not None and tuple(vae.seen.shape) == (1, 96, 128, 3), "encoded at the canvas size, RGB")
nm = lat["noise_mask"]
_need(tuple(nm.shape[-2:]) == (96, 128) and float(nm[0, 0, 48, 70]) == 1.0 and float(nm[0, 0, 5, 5]) == 0.0,
      "noise mask: inside 1, outside 0, at pixel size")
_need("inpaint:" in info and "repainted" in info, "info reports the inpaint and its share")

print("I3 canvas differs from image_1 -> resized before encoding")
SEEN.clear()
vae = VaeImg()
wired = {"samples": torch.zeros(1, 24, 1, 12, 16)}            # 256 x 192 canvas
try:
    pos, neg, lat, info, text = IR.ULSImageReference().build(
        clip, "x", "", vae=vae, latent=wired, image_1=img, inpaint_mask=mask)
    _need(tuple(vae.seen.shape) == (1, 192, 256, 3) and tuple(lat["noise_mask"].shape[-2:]) == (192, 256),
          "image_1 and mask brought to the 256 x 192 canvas")
except ImportError as e:
    print("  skip (no comfy.utils: %s)" % e)

print("I3 refusals")
for kw, what in (({"image_1": img}, "no VAE"), ({"vae": VaeImg(), "image_2": img}, "no image_1")):
    try:
        IR.ULSImageReference().build(clip, "x", "", inpaint_mask=mask, **kw)
        _need(False, "refused without " + what)
    except ValueError as e:
        # without the image VAE the family cannot be told from the tokenizer (it falls to minimax_h3), so the
        # refusal names inpaint_mask either way
        _need("inpaint_mask" in str(e), "refused by name: " + what)

# ---- I4 -------------------------------------------------------------------------------------------------------
print("I4 JS: wired prompt socket survives a new image pin")
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

JS = Path(ROOT) / "web" / "js" / "ph_image_ref.js"
tmp = Path(tempfile.mkdtemp(prefix="v1339js_"))
src = JS.read_text(encoding="utf-8").replace('import { app } from "../../scripts/app.js";',
                                             'const app = { registerExtension() {} };')
(tmp / "ph_image_ref.mjs").write_text(src, encoding="utf-8")
shutil.copy(Path(ROOT) / "web" / "js" / "ph_widget_vis.js", tmp / "ph_widget_vis.js")
# the old order: the list swapped BEFORE the links learn their slots (AMENDED v1340: the loop now reads all ids
# first; the probe finds the loop by its first line)
old_js = src.replace("""    const links = node.graph && node.graph.links;
    if (links) {
        // ALL ids first""", """    node.inputs = next;
    const links = node.graph && node.graph.links;
    if (links) {
        // ALL ids first""", 1)
assert old_js != src, "v1339 probe: the reorder block was not found"
(tmp / "old.mjs").write_text(old_js, encoding="utf-8")
harness = r"""
import * as NEW from "./ph_image_ref.mjs";
import * as OLD from "./old.mjs";
globalThis.LiteGraph = { INPUT: 1 };
function run(IR) {
  const links = new Map();
  let arr = [];
  const node = { graph: { links }, size: [300, 200], widgets: [],
    get inputs() { return arr; },
    // the live frontend: a new pin list re-reads each pin's link from the links' target_slot
    set inputs(v) { arr = v; for (const p of arr) p.link = null;
      for (const [id, l] of links) if (arr[l.target_slot]) arr[l.target_slot].link = id; },
    addInput(name, type) { arr.push({ name, type, link: null }); },
    removeInput(i) { arr.splice(i, 1); for (const l of links.values()) if (l.target_slot > i) l.target_slot--; },
    setSize(s) { this.size = s; }, computeSize() { return [300, 20 * arr.length]; }, setDirtyCanvas() {} };
  for (const n of ["clip", "vae", "latent", "image_1", "inpaint_mask", "prompt", "negative_prompt"])
    arr.push({ name: n, type: "X", link: null });
  const wire = (name, id) => { const i = arr.findIndex((p) => p.name === name); arr[i].link = id; links.set(id, { id, target_slot: i }); };
  wire("prompt", 1); IR.tidy(node);
  wire("image_1", 2); IR.tidy(node);
  const by = Object.fromEntries(arr.map((p) => [p.name, p.link]));
  const slots = arr.every((p, i) => p.link == null || links.get(p.link).target_slot === i);
  return { prompt: by.prompt, mask: by.inpaint_mask, image_1: by.image_1, image_2: by.image_2 === undefined ? "absent" : by.image_2, slots };
}
console.log(JSON.stringify({ now: run(NEW), old: run(OLD) }));
"""
(tmp / "h.mjs").write_text(harness, encoding="utf-8")
r = subprocess.run(["node", str(tmp / "h.mjs")], capture_output=True, text=True, timeout=60)
_need(r.returncode == 0, "JS harness ran " + (r.stderr or "")[-300:])
if r.returncode == 0:
    import json  # noqa: E402
    js = json.loads(r.stdout.strip().splitlines()[-1])
    now, old = js["now"], js["old"]
    _need(now["prompt"] == 1 and now["mask"] is None and now["image_1"] == 2 and now["image_2"] is None
          and now["slots"], "now: prompt keeps its wire, inpaint_mask stays free, image_2 appears unwired: %r" % now)
    _need(old["mask"] == 1 or not old["slots"], "the probe sees the old bug (swap first: inpaint_mask carries the prompt wire): %r" % old)

# ---- I5 -------------------------------------------------------------------------------------------------------
print("I5 ASCII")
for f in ("h3_image.py", "ph_image_ref.py"):
    _need(all(ord(ch) < 128 for ch in open(os.path.join(NODES, f)).read()), f + " ASCII-clean")

print("%s: %s" % (NAME, "PASS" if not _fails else "FAIL %d" % len(_fails)))
sys.exit(1 if _fails else 0)
