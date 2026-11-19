#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1342 -- MiniMax Reference / Keyframes: prompt_in + tags_in sockets; carry on Core 989d93d.

Frank, 18.11.: the open points of v1341.
  T1  both nodes: prompt_in and tags_in are the LAST optional entries, STRING with forceInput (sockets -- the
      widget order is untouched, saved graphs keep every slot).
  T2  wired, they replace the fields -- the substitution sits at the top of build() in both nodes, before the
      latent and the tag lines are read (unwired: None, the fields count).
  T3  h3_kf_refs: a Core whose merge sits in MiniMaxH3._cond_latents (Core 989d93d, 2026-11-10) is
      "core-native" -- no wrapper over a Core that already lays keyframe + reference latents side by side
      (the wrapper would drop the audio side and the None filter Core added). Checked on a stand-in class
      and, when PLS_CORE_ROOT is a Core with the helper, on its real source.
  T4  ASCII.

Measured live (18.11., frontend 1.59.1, classic + Nodes 2.0): prompt_in / tags_in wired draw on their own rows
(Reference y 194/214, Keyframes y 114/134); the pin reorder keeps the wires, also after a reload.

Script-style: exit 0 = pass.
"""

import importlib.util
import os
import sys
import types

NAME = "v1342"
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
pkg = types.ModuleType("plspack")
pkg.__path__ = [NODES]
sys.modules["plspack"] = pkg


def load(name):
    spec = importlib.util.spec_from_file_location("plspack." + name, os.path.join(NODES, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules["plspack." + name] = m
    spec.loader.exec_module(m)
    return m


print("T1 the sockets")
for name, cls_name in (("ph_minimax_ref", "ULSMiniMaxReference"), ("ph_minimax_keyframes", "ULSMiniMaxKeyframes")):
    src = open(os.path.join(NODES, name + ".py")).read()
    opt = src[src.index('"optional": {'):src.index("RETURN_TYPES")]
    import re
    keys = re.findall(r'^\s{16}"(\w+)": \(', opt, re.M)
    _need(keys[-2:] == ["prompt_in", "tags_in"], "%s: prompt_in, tags_in appended last %r" % (cls_name, keys[-3:]))
    _need(opt.count('"forceInput": True') >= 2 and '"prompt_in": ("STRING", {"forceInput": True' in opt
          and '"tags_in": ("STRING", {"forceInput": True' in opt, "%s: STRING sockets (forceInput)" % cls_name)
    _need("prompt_in" in src[src.index("    def build"):] and "tags_in" in src[src.index("    def build"):],
          "%s: build() reads both" % cls_name)

print("T2 the wired text wins (both nodes, before anything reads the fields)")
src = open(os.path.join(NODES, "ph_minimax_keyframes.py")).read()
body = src[src.index("    def build"):]
_need(body.index("prompt = str(prompt_in)") < body.index("samples = latent.get")
      and body.index("tags = str(tags_in)") < body.index("HP.parse_aliases(tags"),
      "Keyframes: prompt_in / tags_in replace the fields before the latent and the tags are read")
ref_src = open(os.path.join(NODES, "ph_minimax_ref.py")).read()
rb = ref_src[ref_src.index("    def build"):]
_need(rb.index('prompt = str(opt["prompt_in"])') < rb.index("samples = latent.get")
      and 'opt["tags"] = str(opt["tags_in"])' in rb, "Reference: both substitutions at the top of build()")

print("T3 carry on a Core with the _cond_latents helper")
KR = load("h3_kf_refs")


class HelperH3:
    def extra_conds(self, **kw):
        payload = {}
        payload.update(self._cond_latents(kw.get("k"), kw.get("r")))
        return payload

    @staticmethod
    def _cond_latents(keyframes, refs):
        return {"cond_video_latents": [kf["latent"] for kf in keyframes if kf.get("latent") is not None]
                + [r["latent"] for r in refs if "latent" in r]}


_need(KR.owner(HelperH3) == "core-native", "helper that merges -> core-native (no wrapper)")
mb = os.path.join(core, "comfy", "model_base.py")
if os.path.isfile(mb):
    h3 = open(mb, encoding="utf-8").read()
    h3 = h3[h3.index("class MiniMaxH3(BaseModel):"):] if "class MiniMaxH3(BaseModel):" in h3 else ""
    if "self._cond_latents(" in h3:
        _need(KR._NATIVE_HELPER_MERGE in h3, "this Core (%s) merges in its helper -- recognised" % os.path.basename(core))
    else:
        print("  note  this Core has no _cond_latents helper (older kind) -- C4 of test_v1049 covers it")

print("T4 ASCII")
for f in ("ph_minimax_ref.py", "ph_minimax_keyframes.py", "h3_kf_refs.py"):
    s = open(os.path.join(NODES, f), encoding="utf-8").read()
    body = s[s.index("v1342"):] if "v1342" in s else ""
    _need(all(ord(c) < 128 for c in body.split("NODE_DISPLAY_NAME_MAPPINGS")[0]), f + " v1342 lines ASCII-clean")

print("%s: %s" % (NAME, "PASS" if not _fails else "FAIL %d" % len(_fails)))
sys.exit(1 if _fails else 0)
