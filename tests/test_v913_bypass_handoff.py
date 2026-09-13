#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v913 -- LoRAs as a forward hook (Core bypass), base weights untouched.

THE FINDING (audit 04.09.): on an int8 / nvfp4 target every LoRA bake makes
Core dequantize, add, and REQUANTIZE the layer with a recalculated scale --
in every mode, SEQ included. ai-toolkit refuses to merge into such a model for
that very reason and loads its assistant LoRA live. Core has carried a bypass
path since v0.32 (comfy/weight_adapter/bypass.py, sd.load_bypass_lora_for_models):
the LoRA runs as a hook, output = base(x) + up(down(x)) * scale, and the base
weight is never touched. Measured 04.09. with Core's real machinery: a CONCAT
adapter through the bypass equals baked SEQ to 5e-7; two groups on the same
layer accumulate to rank 32+16 and match the baked sum to 1.8e-7.

THE TRAP: BypassInjectionManager keeps ONE adapter per module key, and
ModelPatcher.set_injections keeps ONE list per key -- a second group applied
on its own would silently replace the first. So the pack keeps ONE synthetic
LoRA per model (_BypassState, an attachment that survives clone()) and
rebuilds the complete hook set on every apply.

WHAT IS PINNED, and how:

  A  the pure decision `_apply_decision` (uls_merge_policy): auto = bypass
     exactly on a quantized target; patch / bypass are absolute; unknown ->
     auto; under bypass SEQ folds into CONCAT with a note, DARE stays.
  B  `_BypassState`: same base rank-concatenates, other bases add a layer,
     `add` never mutates its receiver, alpha in the tensor dict equals the
     rank, a clone shares the (immutable) state, a shape mismatch is skipped.
  C  `_handoff_bypass`, lifted CLOSED and driven against a fake comfy: the
     second call carries BOTH groups' layers under the pack's own injection
     key, the parent patcher is untouched, the synthetic dict carries factors
     and alpha only, an empty mapping returns None (caller falls back).
  D  `apply_lora_set`, lifted CLOSED and driven: bypass + ONE LoRA goes to the
     merge (not to SEQ); auto goes BAKED everywhere since v917 (field A/B 05.09.), on a plain
     target stays baked; patch on a quantized target stays baked.

torch is used for B (real tensors); no comfy. E drives the JS helpers in node.
"""
import io
import os
import pathlib
import sys
import contextlib

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lift as _LIFT  # noqa: E402


class _NoPaths(object):
    """v929: every lookup answers None, so the payload gate finds nothing."""

    def get_full_path(self, kind, name):
        return None

import uls_merge_policy as MP  # noqa: E402

FAILED = []


def _fail(msg):
    FAILED.append(msg)
    print("FAIL: {}".format(msg))


def _ok(msg):
    print("ok  : {}".format(msg))


def _need(cond, msg):
    _ok(msg) if cond else _fail(msg)


# --- A: the pure decision ----------------------------------------------------
_need(MP._apply_decision("auto", True, "CONCAT")[0] is False, "A  auto + quantized -> BAKED (v917: the field decided)")
_need(MP._apply_decision("auto", False, "CONCAT")[0] is False, "A  auto + plain -> baked")
_need(MP._apply_decision("patch", True, "CONCAT")[0] is False, "A  patch + quantized -> baked")
_need(MP._apply_decision("bypass", False, "CONCAT")[0] is True, "A  bypass + plain -> bypass")
_need(MP._apply_decision("nonsense", True, "CONCAT")[0] is False
      and MP._apply_decision("nonsense", False, "CONCAT")[0] is False, "A  unknown -> auto (baked)")
_need(MP._apply_decision(None, True, "CONCAT")[0] is False, "A  None -> auto (baked)")
ub, mode, notes = MP._apply_decision("bypass", False, "SEQ")
_need(mode == "CONCAT" and any("SEQ folds" in n for n in notes), "A  SEQ folds into CONCAT under bypass, with a note")
ub, mode, notes = MP._apply_decision("bypass", False, "DARE")
_need(mode == "DARE", "A  DARE stays DARE under bypass (v912 decides that, not this)")
ub, mode, notes = MP._apply_decision("patch", True, "SEQ")
_need(mode == "SEQ" and len(notes) == 1 and notes[0].startswith("BAKED"), "A  baked SEQ stays SEQ and SAYS baked (v917)")
ub, mode, notes = MP._apply_decision("auto", True, "CONCAT")
_need(any(n.startswith("BAKED (auto)") and "BYPASS" in n for n in notes), "A  auto on quantized says BAKED and names the hand switch (v917)")
_need(MP.BYPASS_KEY == "polyhedron_bypass", "A  the pack's own injection key, not Core's 'bypass_lora'")

# --- B: the state ------------------------------------------------------------
SRC = (ROOT / "nodes" / "uls_stack_node.py").read_text(encoding="utf-8")
STATE_SRC, MISSING = _LIFT.close_over(SRC, ["_BypassState"], ["print"])
_need(not MISSING, "B  _BypassState lift is closed" if not MISSING else
      "B  the lift is short of %s -- a GUARD fault" % ", ".join(MISSING))
ns = {"print": print}
exec(compile(STATE_SRC, "<lifted _BypassState>", "exec"), ns)
S = ns["_BypassState"]
UP, DOWN = ".lora_B.weight", ".lora_A.weight"


def _td(base, out, rank, inn, seed):
    g = torch.Generator().manual_seed(seed)
    return {base + UP: torch.randn(out, rank, generator=g), base + DOWN: torch.randn(rank, inn, generator=g),
            base + ".alpha": torch.tensor(float(rank))}


s0 = S()
s1 = s0.add(dict(_td("l1", 8, 4, 6, 1), **_td("l2", 8, 3, 6, 2)), UP, DOWN)
s2 = s1.add(_td("l1", 8, 5, 6, 3), UP, DOWN)
_need(len(s0) == 0 and len(s1) == 2, "B  first add: two layers, receiver untouched")
_need(s1.layers["l1"][0].shape[1] == 4 and s2.layers["l1"][0].shape[1] == 9,
      "B  same base rank-concatenates (4 -> 4+5), previous state keeps 4")
_need(s2.layers["l2"][0].shape[1] == 3, "B  other base carried over unchanged")
td = s2.tensor_dict()
_need(float(td["l1.alpha"]) == 9.0 and float(td["l2.alpha"]) == 3.0, "B  alpha == rank (scale folds to 1.0)")
_need(torch.equal(td["l1" + UP][:, :4], s1.layers["l1"][0]), "B  first group's factors sit first in the concat")
_need(s2.on_model_patcher_clone() is s2, "B  a clone shares the immutable state")
out = io.StringIO()
with contextlib.redirect_stdout(out):
    s3 = s2.add(_td("l1", 7, 2, 6, 4), UP, DOWN)   # out 7 != 8
_need(s3.layers["l1"][0].shape[1] == 9 and "shape mismatch" in out.getvalue(),
      "B  a shape-mismatched base is skipped, loudly")

# --- C: the hand-off, driven against a fake comfy ---------------------------
HAND_SRC, MISSING = _LIFT.close_over(SRC, ["_handoff_bypass"],
                                     ["comfy", "BYPASS_KEY", "print", "_BypassState"])
_need(not MISSING, "C  _handoff_bypass lift is closed" if not MISSING else
      "C  the lift is short of %s -- a GUARD fault" % ", ".join(MISSING))


class _FakeAdapter:
    def __init__(self, key):
        self.key = key


class _FakeManager:
    def __init__(self):
        self.keys = []

    def add_adapter(self, key, adapter, strength=1.0):
        self.keys.append((key, strength))

    def create_injections(self, model):
        return ["INJ:" + k for k, _ in self.keys]

    def get_hook_count(self):
        return len(self.keys)


class _FakePatcher:
    def __init__(self, sd_keys, parent=None):
        self.attachments = {}
        self.injections = {}
        self.patches = []
        self.model = type("M", (), {"state_dict": lambda s: {k: None for k in sd_keys}})()
        self._sd = sd_keys

    def clone(self):
        n = _FakePatcher(self._sd)
        n.attachments = dict(self.attachments)
        n.injections = {k: list(v) for k, v in self.injections.items()}
        n.patches = list(self.patches)
        return n

    def get_attachment(self, k):
        return self.attachments.get(k)

    def set_attachments(self, k, v):
        self.attachments[k] = v

    def set_injections(self, k, v):
        self.injections[k] = v

    def add_patches(self, p, a, b):
        self.patches.append(p)
        return list(p.keys())


def _fake_load_lora(td, keymap, log_missing=True):
    out = {}
    for base in {k[:-len(UP)] for k in td if k.endswith(UP)}:
        if base in keymap:
            out[keymap[base]] = _FakeAdapter(base)
    if "REGULAR" in td:
        out["diffusion_model.norm.weight"] = ("diff", (td["REGULAR"],))
    return out


def _hand(model, td, keymap):
    fake_wa = type("wa", (), {})()
    fake_wa.BypassInjectionManager = _FakeManager
    fake_wa.WeightAdapterBase = _FakeAdapter
    fake_comfy = type("c", (), {})()
    fake_comfy.lora = type("l", (), {"load_lora": staticmethod(_fake_load_lora)})()
    sys.modules["comfy"] = fake_comfy
    sys.modules["comfy.weight_adapter"] = fake_wa
    ns = {"comfy": fake_comfy, "BYPASS_KEY": MP.BYPASS_KEY, "print": print, "_BypassState": S}
    exec(compile(HAND_SRC, "<lifted _handoff_bypass>", "exec"), ns)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        res = ns["_handoff_bypass"](model, None, td, UP, DOWN, keymap, {}, "CONCAT", 1, ["x"])
    return res, out.getvalue()


KM = {"diffusion_model.l1": "diffusion_model.l1.weight", "diffusion_model.l2": "diffusion_model.l2.weight"}
p0 = _FakePatcher(["diffusion_model.l1.weight", "diffusion_model.l2.weight", "diffusion_model.norm.weight"])
r1, _ = _hand(p0, _td("diffusion_model.l1", 8, 4, 6, 1), KM)
p1 = r1[0]
r2, txt2 = _hand(p1, _td("diffusion_model.l2", 8, 3, 6, 2), KM)
p2 = r2[0]
inj2 = p2.injections.get(MP.BYPASS_KEY, [])
_need(sorted(inj2) == ["INJ:diffusion_model.l1.weight", "INJ:diffusion_model.l2.weight"],
      "C  second call carries BOTH groups' hooks under the pack's key (nothing replaced)")
_need(len(p1.injections.get(MP.BYPASS_KEY, [])) == 1 and p0.injections == {} and p0.attachments == {},
      "C  parent patchers keep their own state (p1 one hook, p0 nothing)")
_need(p2.get_attachment(MP.BYPASS_KEY) is not p1.get_attachment(MP.BYPASS_KEY)
      and len(p1.get_attachment(MP.BYPASS_KEY)) == 1 and len(p2.get_attachment(MP.BYPASS_KEY)) == 2,
      "C  the attachment is replaced by a NEW state, the old one is not mutated")
_need("hooks=2" in txt2 and "layers=2" in txt2, "C  the console reports layers and hooks")
keys = set(p2.get_attachment(MP.BYPASS_KEY).tensor_dict())
_need(all(k.endswith((UP, DOWN, ".alpha")) for k in keys),
      "C  the synthetic dict carries factors and alpha only (no diff/bias can ride in)")
r4, _ = _hand(p2, _td("diffusion_model.unknown", 8, 2, 6, 6), {})
_need(r4 is None, "C  no mapped adapter -> None (caller falls back to SEQ)")
r5, txt5 = _hand(_FakePatcher(["diffusion_model.l1.weight"]), _td("diffusion_model.l2", 8, 3, 6, 2), KM)
_need(r5 is not None and "unmapped=1" in txt5, "C  an adapter without a module is counted, not hooked")

# --- D: apply_lora_set, driven -------------------------------------------------
sys.modules.pop("comfy", None)
sys.modules.pop("comfy.weight_adapter", None)
PROVIDED = ["_apply_seq", "_apply_concat_or_dare", "_joint_latent_parts",
            "_joint_merge_downgrade", "JOINT_MERGE_REASON", "print",
            "_apply_decision", "_target_is_quantized",
            # v929: the payload gate consults these. Stubbed to answer nothing,
            # so no file is refused here and this guard's promises are unchanged.
            "folder_paths", "payload_family", "safetensors_header_names", "os",
            # v935: the whole-file payload handlers, injected EMPTY for the same
            # reason -- the PDD path is driven in test_v935, not here.
            "_PAYLOAD_HANDLERS"]

APPLY_SRC, MISSING = _LIFT.close_over(SRC, ["apply_lora_set"], PROVIDED)
_need(not MISSING, "D  apply_lora_set lift is closed" if not MISSING else
      "D  the lift is short of %s -- a GUARD fault" % ", ".join(MISSING))


def _drive(apply, quant, mode, names):
    calls = {"seq": 0, "merge": []}

    def _apply_seq(loader, model, clip, n, w, wc=None):
        calls["seq"] += 1
        return model, clip, []

    def _apply_concat_or_dare(loader, model, clip, n, w, **kw):
        calls["merge"].append(kw)
        return model, clip, []

    ns = {"_apply_seq": _apply_seq, "_apply_concat_or_dare": _apply_concat_or_dare,
          "_joint_latent_parts": lambda m: 1,
          "_joint_merge_downgrade": MP._joint_merge_downgrade,
          "JOINT_MERGE_REASON": MP.JOINT_MERGE_REASON, "print": print,
          "_apply_decision": MP._apply_decision,
          "_target_is_quantized": lambda m: quant,
          "folder_paths": _NoPaths(), "os": os,              # v929
          "payload_family": MP.payload_family,               # v929
          "safetensors_header_names": MP.safetensors_header_names,
          "_PAYLOAD_HANDLERS": {}}                            # v935
    exec(compile(APPLY_SRC, "<lifted apply_lora_set>", "exec"), ns)
    with contextlib.redirect_stdout(io.StringIO()):
        ns["apply_lora_set"](None, "MODEL", None, names, [1.0] * len(names), mode=mode, apply=apply)
    return calls


one, two = ["a.safetensors"], ["a.safetensors", "b.safetensors"]
c = _drive("bypass", False, "SEQ", one)
_need(c["seq"] == 0 and len(c["merge"]) == 1 and c["merge"][0]["handoff"] == "bypass"
      and c["merge"][0]["mode"] == "CONCAT", "D  bypass + ONE LoRA + SEQ -> merge path, handoff=bypass, CONCAT")
c = _drive("auto", True, "SEQ", two)
_need(c["seq"] == 1 and len(c["merge"]) == 0, "D  auto on a quantized target -> baked SEQ (v917)")
c = _drive("auto", False, "SEQ", two)
_need(c["seq"] == 1 and c["merge"] == [], "D  auto on a plain target + SEQ -> baked SEQ")
c = _drive("patch", True, "CONCAT", two)
_need(c["seq"] == 0 and c["merge"][0]["handoff"] == "patch", "D  patch on a quantized target -> baked merge")
c = _drive("patch", True, "SEQ", one)
_need(c["seq"] == 1 and c["merge"] == [], "D  patch + one LoRA -> SEQ as before v913")

# --- E: frontend plumbing ------------------------------------------------------
# The pure helpers are DRIVEN in node; the config writers are counted (that part
# is plumbing a driven test would need the whole ComfyUI app for -- named as
# such, not dressed up as behaviour).
import re
import subprocess
import tempfile
JS = (ROOT / "web" / "js" / "uls_node.js").read_text(encoding="utf-8")
m = re.search(r"const APPLY_STEPS = .*?function applyNext\(v\) \{.*?\}\n", JS, re.S)
_need(m is not None, "E  the apply helpers exist in uls_node.js")
if m is not None:
    prog = m.group(0) + """
const out = [];
out.push(applyNorm(undefined), applyNorm("nonsense"), applyNorm("bypass"));
let v = "auto"; for (let i = 0; i < 4; i++) { out.push(v); v = applyNext(v); }
out.push(applyNext("zzz"));
console.log(JSON.stringify(out));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(prog)
    try:
        got = subprocess.run(["node", f.name], capture_output=True, text=True, timeout=30).stdout.strip()
    finally:
        os.unlink(f.name)
    _need(got == '["auto","auto","bypass","auto","bypass","patch","auto","bypass"]',
          "E  driven in node: unknown -> auto, cycle auto -> bypass -> patch -> auto, next(unknown) = bypass (got %s)" % got)
_need(JS.count("apply: applyNorm(this._uls.apply),") == 4,
      "E  all four config writers (Stack + Engine, sync + serialize) carry `apply` (counted)")
_need(JS.count("this._uls.apply = applyNorm(d.apply);") == 2,
      "E  both onConfigure readers restore `apply` (counted)")
_need('which === "apply"' in JS and JS.count("uls.apply = applyNext(uls.apply);") == 2,
      "E  Stack popup dispatcher, Stack header pill (v915) and Engine pill all write the value (counted)")

# --- verdict -----------------------------------------------------------------
if FAILED:
    print("\n{} check(s) FAILED".format(len(FAILED)))
    sys.exit(1)
print("\nall checks passed")
