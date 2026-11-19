#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1341 -- Image Reference: a wired text gets its own socket row (prompt_in).

Frank, 17.11. (screenshot): with the prompt FIELD's socket wired (CLIP Text Encode positive_text -> prompt) its dot
sat on top of the clip row; in the sandbox frontend 1.59.1 it sat on the negative field. A hidden field has no row
of its own any more. Now the text comes through prompt_in, a plain socket (forceInput) in the socket list.

  T1  prompt_in is the LAST optional entry, a STRING with forceInput (a socket, not a widget -- saved graphs keep
      every widget slot).
  T2  wired, it replaces the prompt field (H3 Image route: Core gets the wired text); unwired, the field counts.
  T3  ASCII.

Measured live (17.11., frontend 1.59.1, classic and Nodes 2.0): prompt_in wired draws on its own row
(classic y 134, Vue y 132) below inpaint_mask, the prompt field below it -- no two dots on one row.

Script-style: exit 0 = pass.
"""

import importlib.util
import os
import re
import sys
import types

NAME = "v1341"
HERE = os.path.dirname(os.path.abspath(__file__))
NODES = os.path.join(os.path.dirname(HERE), "nodes")
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

print("T1 the socket")
opt = IR.ULSImageReference.INPUT_TYPES()["optional"]
_need(list(opt)[-1] == "prompt_in", "prompt_in is the last optional entry")
_need(opt["prompt_in"][0] == "STRING" and opt["prompt_in"][1].get("forceInput") is True, "STRING, forceInput (a socket)")

print("T2 the wired text wins")
SEEN = {}


class FakeCore:
    @classmethod
    def execute(cls, clip, prompt, width, height, vae=None, images=None):
        SEEN["prompt"] = prompt
        return types.SimpleNamespace(args=([[torch.ones(1, 4, 8), {}]], {"samples": torch.zeros(1, 24, 1, 6, 8)}))


class VaeImg:
    first_stage_model = type("MiniMaxH3ImageVAE", (), {})()


HI.core_class = lambda variant: FakeCore
clip = types.SimpleNamespace(tokenizer=type("MiniMaxH3Tokenizer", (), {})())
out = IR.ULSImageReference().build(clip, "field text", "", vae=VaeImg(), prompt_in="wired text")
_need(SEEN["prompt"] == "wired text" and out[4] == "wired text", "wired: Core and prompt_text get the wired text")
IR.ULSImageReference().build(clip, "field text", "", vae=VaeImg())
_need(SEEN["prompt"] == "field text", "unwired: the field counts")

print("T3 ASCII")
src = open(os.path.join(NODES, "ph_image_ref.py")).read()
_need(all(ord(c) < 128 for c in src), "ph_image_ref.py ASCII-clean")
_need(re.search(r'"prompt_in": \("STRING", \{"forceInput": True', src) is not None, "literal in INPUT_TYPES (the gate reads it)")

print("%s: %s" % (NAME, "PASS" if not _fails else "FAIL %d" % len(_fails)))
sys.exit(1 if _fails else 0)
