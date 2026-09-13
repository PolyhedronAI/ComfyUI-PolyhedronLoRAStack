#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v914 -- merge-path hygiene: convert like Core, refuse what the merge drops.

THE FINDING (audit 04.09.): the CONCAT / DARE / bypass path rebuilds each
layer's delta as up @ down and hands a SYNTHETIC dict to Core. Two things the
SEQ path gets for free through Core's LoraLoader were missing here:

  1  comfy.lora_convert.convert_lora -- BFL control / Wan Fun / USO key
     shapes. The merge read the raw dict and saw alien names for those.
  2  everything load_lora reads BEYOND up/down/alpha was dropped SILENTLY:
     dora_scale (a DoRA applied as plain LoRA is a different, larger delta),
     .diff / .diff_b, LoRA bias, norm scales, LyCORIS families. Only the conv
     'mid' had a guard (v250).

WHAT IS PINNED, and how:

  A  `_foreign_keys` (uls_merge_policy), pure: a plain LoRA reports nothing;
     every family in FOREIGN_SUFFIXES is named; a mixed dict names all of them
     sorted; a model key that merely CONTAINS a family word is not matched.
  B  the refusal block inside `_apply_concat_or_dare`, cut out and DRIVEN with
     injected values: a group with one foreign LoRA falls to SEQ and names the
     LoRA and its families; under bypass the message says BAKED; a clean group
     passes through; the old mid test still fires on its own.
  C  `_convert_lora_like_core`, lifted closed and driven with a fake
     comfy.lora_convert: converted dict wins, an empty / non-dict result and a
     raising converter both hand back the raw dict, an absent module too.
  D  the load loop calls the converter on what _cached_load_torch_file
     returned -- checked on the AST of `_apply_concat_or_dare`, not as text.

No torch, no comfy.
"""
import ast
import contextlib
import io
import os
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lift as _LIFT  # noqa: E402
import uls_merge_policy as MP  # noqa: E402

FAILED = []


def _fail(msg):
    FAILED.append(msg)
    print("FAIL: {}".format(msg))


def _ok(msg):
    print("ok  : {}".format(msg))


def _need(cond, msg):
    _ok(msg) if cond else _fail(msg)


# --- A: the pure detector -----------------------------------------------------
PLAIN = {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": 0,
         "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": 0,
         "diffusion_model.blocks.0.attn.qkv_proj.alpha": 0,
         "lora_unet_blocks_1_fc1.lora_down.weight": 0,
         "lora_unet_blocks_1_fc1.lora_up.weight": 0}
_need(MP._foreign_keys(PLAIN) == [], "A  a plain up/down/alpha LoRA (both conventions) reports nothing")
missed = []
for suf, label in MP.FOREIGN_SUFFIXES:
    td = dict(PLAIN)
    td["diffusion_model.blocks.0.x" + suf] = 0
    if MP._foreign_keys(td) != [label]:
        missed.append((suf, MP._foreign_keys(td)))
_need(not missed, "A  every family in FOREIGN_SUFFIXES is named on its own: %s" % (missed or "all"))
td = dict(PLAIN)
td["a.dora_scale"] = 0
td["b.diff_b"] = 0
td["c.lokr_w1"] = 0
_need(MP._foreign_keys(td) == ["DoRA scale", "LoKr", "bias diff"], "A  a mixed dict names all families, sorted")
td = dict(PLAIN)
td["diffusion_model.blocks.0.diffusion.lora_A.weight"] = 0     # contains 'diff', is not '.diff'
td["diffusion_model.oft_blocks_used.lora_B.weight"] = 0        # contains 'oft_blocks', suffix is a LoRA
_need(MP._foreign_keys(td) == [], "A  suffix match only: a key that merely contains a family word is clean")

# --- B: the refusal block, cut out and driven ----------------------------------
SRC = (ROOT / "nodes" / "uls_stack_node.py").read_text(encoding="utf-8")
start = SRC.index("    foreign = [(valid_names[i], _foreign_keys(td)) for i, td in enumerate(raw)]")
end = SRC.index("return _apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)", start)
end = SRC.index("\n", end)
BLOCK = textwrap.dedent(SRC[start:end])


def _drive_block(raw, names, mode, handoff, mid=False):
    calls = []

    def _apply_seq(loader, model, clip, n, w, wc):
        calls.append(list(n))
        return "SEQ"

    ns = {"_foreign_keys": MP._foreign_keys, "_has_mid_tensor": lambda td: mid,
          "_short_name": lambda n, w=40: n, "_apply_seq": _apply_seq, "print": print,
          "raw": raw, "valid_names": names, "valid_weights": [1.0] * len(names),
          "valid_clip_weights": [1.0] * len(names), "mode": mode, "handoff": handoff,
          "_fb": " (BAKED, not bypass)" if handoff == "bypass" else "",   # v915: as the function defines it
          "loader": None, "model": "M", "clip": None}
    out = io.StringIO()
    fn = "def _blk():\n" + textwrap.indent(BLOCK, "    ") + "\n    return 'PASS'\n"
    exec(compile(fn, "<refusal block>", "exec"), ns)
    with contextlib.redirect_stdout(out):
        res = ns["_blk"]()
    return res, calls, out.getvalue()


clean = dict(PLAIN)
dora = dict(PLAIN)
dora["diffusion_model.blocks.0.attn.qkv_proj.dora_scale"] = 0
res, calls, txt = _drive_block([clean, dora], ["clean.safetensors", "dora.safetensors"], "CONCAT", "patch")
_need(res == "SEQ" and calls == [["clean.safetensors", "dora.safetensors"]],
      "B  one foreign LoRA sends the WHOLE group to SEQ")
_need("dora.safetensors: DoRA scale" in txt and "clean.safetensors" not in txt.split("falling back")[1],
      "B  the message names the offending LoRA and its family, not the clean one")
_need("BAKED" not in txt, "B  under patch the message does not mention BAKED")
res, calls, txt = _drive_block([clean, dora], ["clean.safetensors", "dora.safetensors"], "CONCAT", "bypass")
_need(res == "SEQ" and "(BAKED, not bypass)" in txt, "B  under bypass the message says the fallback is BAKED")
res, calls, txt = _drive_block([clean, dict(PLAIN)], ["a", "b"], "DARE", "patch")
_need(res == "PASS" and calls == [] and txt == "", "B  a clean group passes through silently")
res, calls, txt = _drive_block([clean, dict(PLAIN)], ["a", "b"], "DARE", "patch", mid=True)
_need(res == "SEQ", "B  the v250 mid test still fires on its own")

# --- C: the converter, lifted closed and driven ---------------------------------
# v930: the seam now consults the foreign-schema registry and the header
# metadata reader. Both are stubbed to do nothing here, so this guard's
# promises about CORE's converter are unchanged -- v930's own guard drives the
# new half.
CONV_SRC, MISSING = _LIFT.close_over(
    SRC, ["_convert_lora_like_core"],
    ["print", "convert_foreign_lora", "safetensors_metadata"])
_need(not MISSING, "C  the lift is closed" if not MISSING else
      "C  the lift is short of %s -- a GUARD fault" % ", ".join(MISSING))


def _drive_conv(behaviour, raw):
    import types
    if behaviour == "absent":
        sys.modules.pop("comfy.lora_convert", None)
        sys.modules["comfy"] = types.ModuleType("comfy")   # package without the submodule
    else:
        mod = types.ModuleType("comfy.lora_convert")
        if behaviour == "convert":
            mod.convert_lora = lambda sd: {"converted": 1}
        elif behaviour == "empty":
            mod.convert_lora = lambda sd: {}
        elif behaviour == "notdict":
            mod.convert_lora = lambda sd: None
        elif behaviour == "raise":
            def _r(sd):
                raise RuntimeError("boom")
            mod.convert_lora = _r
        pkg = types.ModuleType("comfy")
        pkg.lora_convert = mod
        sys.modules["comfy"] = pkg
        sys.modules["comfy.lora_convert"] = mod
    ns = {"print": print,
          "convert_foreign_lora": lambda sd, metadata=None, log=None: (sd, None),
          "safetensors_metadata": lambda path: {}}
    exec(compile(CONV_SRC, "<lifted converter>", "exec"), ns)
    with contextlib.redirect_stdout(io.StringIO()):
        return ns["_convert_lora_like_core"](raw)


RAW = {"x.lora_A.weight": 1}
_need(_drive_conv("convert", RAW) == {"converted": 1}, "C  a converted dict wins")
_need(_drive_conv("empty", RAW) is RAW, "C  an empty result hands back the raw dict")
_need(_drive_conv("notdict", RAW) is RAW, "C  a non-dict result hands back the raw dict")
_need(_drive_conv("raise", RAW) is RAW, "C  a raising converter hands back the raw dict")
_need(_drive_conv("absent", RAW) is RAW, "C  an absent module hands back the raw dict")
sys.modules.pop("comfy", None)
sys.modules.pop("comfy.lora_convert", None)

# --- D: the load loop calls the converter (AST) ---------------------------------
tree = ast.parse(SRC)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_concat_or_dare")
loads, converts = [], []
for node in ast.walk(fn):
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
            and node.targets[0].id == "td":
        src = ast.unparse(node.value)
        if "_cached_load_torch_file(" in src:
            loads.append(node.lineno)
        if "_convert_lora_like_core(" in src:
            converts.append(node.lineno)
_need(len(loads) == 1 and len(converts) == 1 and converts[0] > loads[0],
      "D  in _apply_concat_or_dare, td is loaded once and converted right after (AST)")

# --- verdict -----------------------------------------------------------------
if FAILED:
    print("\n{} check(s) FAILED".format(len(FAILED)))
    sys.exit(1)
print("\nall checks passed")
