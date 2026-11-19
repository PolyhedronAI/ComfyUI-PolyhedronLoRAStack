#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1343 -- the blank picture on p-h3-image (Frank 18.11.: compare P-H3-Image with H3 Image + blank).

MEASURED in Core 0.45.0, p-h3-image differs from MiniMax H3 Image in three ways; each is pinned here.

  P1  label: PH3ImageConditioning tokenizes without picture_label -> the tokenizer default "<Picture %d>: ".
      Our joint encode uses exactly that label; H3 Image keeps "Picture %d: " (picture_block untouched).
  P2  sink: Core's TextConnector replaces the FIRST row of every encode by a sink (minimax.py). So p-h3-image
      gets ONE joint encode (picture + prompt in one tokenize call), and prime_joint puts those rows IN PLACE
      of the text rows inside the window -- never picture rows laid in front of separately encoded text.
  P3  steps: steps_for(p-h3-image) = 8; with ModelSamplingAV at shift 1.0 (Core's ModelSamplingMiniMaxH3
      1.0/1.0) and the simple scheduler, Core's closed cond window selects exactly steps_covered(.., 8).
  P4  Image Reference routes by variant: p-h3-image -> joint path with the prompt, H3 Image -> v1338 path;
      the info names 8 steps and the joint encode.
  P5  ASCII-clean.

Script-style: exit 0 = pass.
"""

import importlib.util
import os
import sys
import types

NAME = "v1343"
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


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(NODES, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


import torch  # noqa: E402

HI = _load("h3_image")
core = os.environ.get("PLS_CORE_ROOT") or "/home/claude/core_now"

# ---- P1 -------------------------------------------------------------------------------------------------------
print("P1 label")
_need(HI.PH3_PICTURE_LABEL == "<Picture %d>: ", "p-h3-image label is <Picture %d>: ")
_need(HI.PICTURE_LABEL == "Picture %d: ", "H3 Image label unchanged")
try:
    src = open(os.path.join(core, "comfy", "text_encoders", "minimax.py")).read()
    _need('picture_label="<Picture %d>: "' in src, "Core tokenizer default label is <Picture %d>: ")
    ext = open(os.path.join(core, "comfy_extras", "nodes_minimax_h3.py")).read()
    body = ext[ext.index("class PH3ImageConditioning"):]
    body = body[body.index("def execute"):body.index("return io.NodeOutput")]
    _need("clip.tokenize(prompt, images=vision_images)" in body, "PH3ImageConditioning passes no label")
    _need("sink" in src and "torch.cat([sink, x[:, 1:]]" in src, "Core text connector puts a sink on row 0")
except (OSError, ValueError) as e:
    print("  skip (no Core source: %s)" % e)


class _FakeClip:
    def __init__(self, rows):
        self.calls = []
        self.rows = rows

    def tokenize(self, t, **kw):
        self.calls.append((t, kw))
        return {"t": t}

    def encode_from_tokens_scheduled(self, tok):
        return [[self.rows.clone(), {"minimax_token_tags": torch.zeros(self.rows.shape[1], dtype=torch.long)}]]


# ---- P2 -------------------------------------------------------------------------------------------------------
print("P2 one joint encode")
joint = torch.full((1, 9, 2), 3.0)
fc = _FakeClip(joint)
blk = HI.joint_block(fc, "a cafe portrait", "mid grey")
_need(len(fc.calls) == 1, "exactly one encode")
t, kw = fc.calls[0]
_need(t == "a cafe portrait" and kw.get("picture_label") == "<Picture %d>: " and len(kw["images"]) == 1,
      "prompt and picture in the SAME tokenize call, p-h3 label")
_need(tuple(kw["images"][0].shape) == (1, 512, 512, 3) and abs(float(kw["images"][0].mean()) - 0.5) < 1e-6,
      "blank 512 px, tone 0.5")
text = torch.arange(10, dtype=torch.float32).view(1, 5, 2)
cond = [[text, {"minimax_token_tags": torch.ones(5, dtype=torch.long), "keep": 1}]]
out = HI.prime_joint(cond, blk, 0.0, 0.25)
_need(len(out) == 2, "window from 0: joint + plain tail")
c0, d0 = out[0]
_need(torch.equal(c0, joint) and c0.shape[1] == 9, "joint rows REPLACE the text rows (not laid in front)")
_need(d0["minimax_token_tags"].shape[0] == 9 and d0["keep"] == 1, "tags from the joint encode, dict kept")
_need(d0["start_percent"] == 0.0 and d0["end_percent"] == 0.25, "window set")
_need(out[1][0] is text and out[1][1]["start_percent"] == 0.25, "plain text for the rest")
_need("start_percent" not in cond[0][1], "input dict not mutated")
_need(len(HI.prime_joint(cond, blk, 0.2, 0.5)) == 3, "window inside: head + tail")

# ---- P3 -------------------------------------------------------------------------------------------------------
print("P3 steps")
_need(HI.steps_for(HI.VARIANT_PH3) == 8 and HI.steps_for(HI.VARIANT_H3) == 6, "8 steps p-h3, 6 H3 Image")
_need(HI.steps_covered(0.0, 0.25, 8) == ([1, 2, 3], [3]), "0-0.25 on 8 steps -> 1,2,3 (3 on the edge)")
try:
    sys.path.insert(0, core)
    _argv = sys.argv
    sys.argv = [_argv[0], "--cpu"]          # Core parses argv on import; no GPU in the sandbox
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy.samplers as csm
    finally:
        sys.argv = _argv
    import comfy.model_sampling as cms

    class _MS(cms.ModelSamplingAV, cms.CONST):
        pass

    ms = _MS(None)
    ms.set_parameters(shift=1.0, audio_shift=1.0)
    sig = [float(x) for x in csm.simple_scheduler(ms, 8)][:-1]

    def core_steps(a, b):
        ts, te = ms.percent_to_sigma(a), ms.percent_to_sigma(b)
        return [i + 1 for i, s in enumerate(sig) if not (s > ts + 1e-6) and not (s < te - 1e-6)]

    for a, b in ((0.0, 0.25), (0.0, 0.4), (0.0, 0.2), (0.1, 0.6)):
        _need(core_steps(a, b) == HI.steps_covered(a, b, 8)[0],
              "Core window %.3f-%.3f = steps %s" % (a, b, core_steps(a, b)))
except Exception as e:   # noqa: BLE001
    print("  skip (no Core importable: %s)" % e)

# ---- P4 -------------------------------------------------------------------------------------------------------
print("P4 Image Reference routes by variant")
pkg = types.ModuleType("plsnodes")
pkg.__path__ = [NODES]
sys.modules["plsnodes"] = pkg
try:
    spec = importlib.util.spec_from_file_location("plsnodes.ph_image_ref", os.path.join(NODES, "ph_image_ref.py"))
    IR = importlib.util.module_from_spec(spec)
    sys.modules["plsnodes.ph_image_ref"] = IR
    spec.loader.exec_module(IR)
    node_cls = [v for v in vars(IR).values() if isinstance(v, type) and hasattr(v, "_blank_prime")][0]
    fc = _FakeClip(joint)
    pos, tail = node_cls._blank_prime(HI, fc, cond, [], "white", 0.0, 0.25, HI.VARIANT_PH3, "the prompt")
    _need(fc.calls[0][0] == "the prompt" and fc.calls[0][1]["picture_label"] == "<Picture %d>: ",
          "p-h3-image: joint encode of the prompt")
    _need(torch.equal(pos[0][0], joint), "p-h3-image: joint rows in the window")
    _need("of 8" in tail and "joint encode" in tail and "step 3 on the edge" in tail, "info: %s" % tail.strip())
    pic = torch.full((1, 3, 2), -1.0)
    fc2 = _FakeClip(pic)
    pos2, tail2 = node_cls._blank_prime(HI, fc2, cond, [], "white", 0.0, 0.25, HI.VARIANT_H3, "the prompt")
    _need(fc2.calls[0][0] == "" and fc2.calls[0][1]["picture_label"] == "Picture %d: ", "H3 Image: v1338 path")
    _need(pos2[0][0].shape[1] == 8 and "of 6" in tail2 and "joint" not in tail2, "H3 Image: rows in front, 6 steps")
    pos3, tail3 = node_cls._blank_prime(HI, fc2, cond, [("image_1", text)], "white", 0.0, 0.25, HI.VARIANT_PH3, "x")
    _need(pos3 is cond and "NOT used" in tail3, "with a reference wired: not used")
except Exception as e:   # noqa: BLE001
    _need(False, "Image Reference importable: %r" % e)

# ---- P5 -------------------------------------------------------------------------------------------------------
print("P5 ASCII")
for f in ("h3_image.py", "ph_image_ref.py"):
    _need(all(ord(ch) < 128 for ch in open(os.path.join(NODES, f)).read()), f + " ASCII-clean")

print("%s: %s" % (NAME, "PASS" if not _fails else "FAIL %d" % len(_fails)))
sys.exit(1 if _fails else 0)
