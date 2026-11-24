#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1338 -- blank picture prime in Image Reference (family minimax_h3_image).

Frank, 17.11.: the field video's workflow carries KJNodes' MiniMax H3 Prompt Weight; foreign nodes are not
allowed, so the idea of its mode "text with empty image" is built into OUR node (own implementation; KJNodes
is GPL-3, nothing copied).

WHAT IS PINNED

  B1  h3_image pure parts: blank_window clamps and refuses an empty window; steps_covered on the 6-step grid
      (0-0.25 -> steps 1,2; 0-0.4 -> 1,2,3; an edge exactly on a step is reported as shared).
  B2  steps_covered agrees with REAL Core (PLS_CORE_ROOT / core_now): MiniMaxH3ImageModelSampling.percent_to_sigma
      on the H3 Image grid + Core's closed cond window (comfy/samplers.py) select exactly those steps.
  B3  prime_blank_picture with a fake encoder: the picture rows stand IN FRONT of the text rows, the token tags
      are concatenated in the same order, the window is set, the plain text covers the rest of the run, the
      text rows themselves are untouched.
  B4  Image Reference: blank_picture / blank_start / blank_end are the LAST three widgets (append-only law);
      default off; with a reference wired the prime is not used and the info says so.
  B5  ASCII-clean.

Script-style: exit 0 = pass.
"""

import importlib.util
import os
import re
import sys

NAME = "v1338"
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

# ---- B1 ---------------------------------------------------------------------------------------------------
print("B1 pure parts")
_need(HI.blank_window(0.0, 0.25) == (0.0, 0.25), "window 0-0.25 kept")
_need(HI.blank_window(-1, 2) == (0.0, 1.0), "window clamped to 0..1")
_need(HI.blank_window(0.3, 0.3) is None and HI.blank_window(0.5, 0.2) is None, "empty window refused")
_need(HI.steps_covered(0.0, 0.25) == ([1, 2], []), "0-0.25 -> steps 1,2")
_need(HI.steps_covered(0.0, 0.4) == ([1, 2, 3], []), "0-0.4 -> steps 1,2,3")
_need(HI.steps_covered(0.0, 1.0 / 3.0) == ([1, 2, 3], [3]), "edge on step 3 reported as shared")
_need(HI.BLANK_TONES[0] == "off" and set(HI.TONE_VALUE) == set(HI.BLANK_TONES[1:]), "tones: off + three values")

# ---- B2 real Core -------------------------------------------------------------------------------------------
print("B2 against real Core")
core = os.environ.get("PLS_CORE_ROOT") or "/home/claude/core_now"
try:
    sys.path.insert(0, core)
    from comfy.ldm.minimax.sampling import MiniMaxH3ImageModelSampling

    class _MC:
        unet_config = {"pdd_sigmas": [1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.2320, 0.0]}
        sampling_settings = {}

    ms = MiniMaxH3ImageModelSampling(_MC())
    grid = [1.0, 0.9675, 0.8875, 0.75, 0.5328, 0.2320]   # the six steps run at these sigmas

    def core_steps(start, end):
        ts, te = ms.percent_to_sigma(start), ms.percent_to_sigma(end)
        # comfy/samplers.py: skipped when sigma > timestep_start or sigma < timestep_end
        return [i + 1 for i, s in enumerate(grid) if not (s > ts + 1e-6) and not (s < te - 1e-6)]

    for a, b in ((0.0, 0.25), (0.0, 0.4), (0.0, 1.0 / 3.0), (0.1, 0.6), (0.0, 0.05)):
        _need(core_steps(a, b) == HI.steps_covered(a, b)[0],
              "Core window %.3f-%.3f = steps %s" % (a, b, core_steps(a, b)))
except Exception as e:   # noqa: BLE001
    print("  skip (no Core importable: %s)" % e)

# ---- B3 prime with a fake encoder ----------------------------------------------------------------------------
print("B3 prime_blank_picture")

text = torch.arange(10, dtype=torch.float32).view(1, 5, 2)
ttags = torch.ones(5, dtype=torch.long)
pic = torch.full((1, 3, 2), -1.0)
ptags = torch.tensor([1, 0, 1])
cond = [[text, {"minimax_token_tags": ttags, "keep": 7}]]
out = HI.prime_blank_picture(cond, [pic, {"minimax_token_tags": ptags}], 0.0, 0.25)
_need(len(out) == 2, "window from 0: primed + plain tail (2 entries)")
c0, d0 = out[0]
_need(tuple(c0.shape) == (1, 8, 2) and torch.equal(c0[:, :3], pic) and torch.equal(c0[:, 3:], text),
      "picture rows in front, text rows untouched behind")
_need(torch.equal(d0["minimax_token_tags"], torch.cat([ptags, ttags])), "tags concatenated in row order")
_need(d0["start_percent"] == 0.0 and d0["end_percent"] == 0.25 and d0["keep"] == 7, "window set, dict kept")
c1, d1 = out[1]
_need(c1 is text and d1["start_percent"] == 0.25 and d1["end_percent"] == 1.0, "plain text for the rest")
_need("start_percent" not in cond[0][1], "input dict not mutated")
out2 = HI.prime_blank_picture(cond, [pic, {"minimax_token_tags": ptags}], 0.2, 0.5)
_need(len(out2) == 3 and out2[1][1]["end_percent"] == 0.2, "window inside: plain head + plain tail")


class _FakeClip:
    def __init__(self):
        self.calls = []

    def tokenize(self, t, **kw):
        self.calls.append((t, kw))
        return {"t": t, "kw": kw}

    def encode_from_tokens_scheduled(self, tok):
        return [[pic.clone(), {"minimax_token_tags": ptags.clone()}]]


fc = _FakeClip()
blk = HI.picture_block(fc, "mid grey")
t, kw = fc.calls[0]
_need(t == "" and kw.get("picture_label") == "Picture %d: " and len(kw.get("images")) == 1,
      "picture block: empty text, one picture, Core's label")
img = kw["images"][0]
_need(tuple(img.shape) == (1, 512, 512, 3) and abs(float(img.mean()) - 0.5) < 1e-6, "512 px, tone 0.5")

# ---- B4 Image Reference --------------------------------------------------------------------------------------
print("B4 Image Reference")
src = open(os.path.join(NODES, "ph_image_ref.py")).read()
opt = src[src.index('"optional": {'):src.index("RETURN_TYPES")]
keys = re.findall(r'^\s{16}"(\w+)": \(', opt, re.M)
# AMENDED v1341: the socket prompt_in (forceInput, no widget) is appended after them -- widgets only here
keys = [k for k in keys if k != "prompt_in"]
# AMENDED v1354: blank_strength / blank_colour / blank_anchor appended after them (the law) -- pinned in test_v1354
i = keys.index("blank_picture") if "blank_picture" in keys else -1
_need(keys[i:i + 3] == ["blank_picture", "blank_start", "blank_end"] and keys[i + 3:] in
      ([], ["blank_strength", "blank_colour", "blank_anchor"]),
      "three new widgets appended in order, only v1354's after them: %s" % keys[-7:])
_need('"default": "off"' in opt[opt.index('"blank_picture"'):], "blank_picture defaults to off")
_need("NOT used: %d reference(s) wired" in src, "with references the prime is refused in the info")
_need("does not take it" in src, "other families say so in the info")

# ---- B5 -----------------------------------------------------------------------------------------------------
print("B5 ASCII")
for f in ("h3_image.py", "ph_image_ref.py"):
    _need(all(ord(ch) < 128 for ch in open(os.path.join(NODES, f)).read()), f + " ASCII-clean")

print("%s: %s" % (NAME, "PASS" if not _fails else "FAIL %d" % len(_fails)))
sys.exit(1 if _fails else 0)
