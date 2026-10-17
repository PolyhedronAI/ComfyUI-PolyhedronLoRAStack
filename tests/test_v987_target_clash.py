#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v987 -- two merged layers never land on one weight.

v986 merges a group that mixes key naming by meeting the layers through
_canonical_base (the two spellings Core's key map knows by name). Any OTHER
pair of spellings that Core maps to the same weight (diffusers-style names,
an unknown prefix) would still become two merged layers -- and load_lora
writes patch_dict[target] per layer, so the second silently replaces the
first: a LoRA lost while the console says 'merged'. v987 checks the merged
dict against the model's own key map before the hand-off and runs the group
SEQ when two layers share a target.

LIVE (sandbox ComfyUI, 22.09.2026, CHANGELOG_v987): a lora_B/A LoRA on
diffusion_model.blocks.0.attn.qkv_proj + a kohya LoRA spelled
lora_unet_transformer_blocks_0_attn_to_qkv (both mapped to one weight) ->
SEQ, both LoRAs through Core's loader; the canonical pair (lora_unet_blocks_
0_attn_qkv_proj) -> one merged patch.

  X  RUN _target_clashes: no clash for distinct targets; a clash is reported
     with the target and both spellings; unmapped bases are ignored; tuple
     targets (Core's sliced keys) compare by value
  H  the Stack calls it after the key map is built and BEFORE both hand-offs
     (bypass and load_lora), and falls back to SEQ carrying {_fb}
"""
import ast
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lift as _LIFT  # noqa: E402

FAILED = []


def _need(cond, msg):
    print(("ok  : " if cond else "FAIL: ") + msg)
    if not cond:
        FAILED.append(msg)


SRC = (ROOT / "nodes" / "uls_stack_node.py").read_text(encoding="utf-8")
lift, missing = _LIFT.close_over(SRC, ["_target_clashes"], provided=set())
_need(not missing, "X  lifts closed (%s)" % (sorted(missing) or "none missing"))
ns = {}
exec(compile(lift, "<v987>", "exec"), ns)
TC = ns["_target_clashes"]
UP, DN = ".lora_up.weight", ".lora_down.weight"
# two EQUAL but distinct tuple objects, built at run time (literal tuples would
# be one shared constant, and an identity compare would pass by accident)
km = {"a": "W1", "b": "W2", "c": "W1",
      "s1": tuple(["W3", tuple([0, 0, 8])]), "s2": tuple(["W3", tuple([0, 0, 8])])}
md = {"a" + UP: 0, "a" + DN: 0, "a.alpha": 0, "b" + UP: 0, "b" + DN: 0}
_need(TC(md, UP, km) == [], "X  distinct targets: no clash")
md["c" + UP] = 0
_need(TC(md, UP, km) == [("W1", "a", "c")], "X  a shared target is reported with both spellings")
_need(TC({"zz" + UP: 0, "a" + UP: 0}, UP, km) == [], "X  a base Core does not map is ignored")
_need(len(TC({"s1" + UP: 0, "s2" + UP: 0}, UP, km)) == 1, "X  tuple targets compare by value")

fn = re.search(r"def _apply_concat_or_dare\(.*?(?=\ndef )", SRC, re.S).group(0)
i_map = fn.find("full_keymap = {**model_keymap, **clip_keymap}")
i_chk = fn.find("_clash = _target_clashes(merged_td, out_up_suffix, full_keymap)")
i_byp = fn.find('if handoff == "bypass":', i_map)
i_load = fn.find("loaded = comfy.lora.load_lora(merged_td, full_keymap)")
_need(0 < i_map < i_chk < i_byp < i_load,
      "H  checked after the key map, before the bypass AND the load_lora hand-off")
seg = fn[i_chk:i_byp]
_need("return _apply_seq(" in seg and "{_fb}" in seg, "H  a clash runs SEQ and says so ({_fb})")

print()
if FAILED:
    print("test_v987_target_clash: %d failure(s)" % len(FAILED))
    sys.exit(1)
print("test_v987_target_clash: 0 failure(s)")
