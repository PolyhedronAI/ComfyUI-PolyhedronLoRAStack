#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v985 -- the Merge Analyzer tells the truth about the run, in names a reader
can tell apart.

Field, 21.09.2026 (Frank's H3 stack, 13 + 6 LoRAs): the overview read
'[subject] CONCAT +TRIM' and '[scene] CONCAT +TRIM'. Both groups ran SEQ: five
LoRAs used the other key naming (kohya .lora_up/.lora_down against
.lora_B/.lora_A), and the Stack sends a mixed group to SEQ -- no merge, no
TRIM, no cap, no bake. Only the Overlap depth's 'left out' lines hinted at
it. And every name was cut to its first 24..34 characters, which for
`polyhedron_minimax_h3_image_lora__<what>` is exactly the shared part: the
pair table read 'polyhedron_minimax_h3_im <-> polyhedron_minimax_h3_im'.

  N  RUN: name_shortener -- Frank's names come out distinct and readable
     (the shared prefix dropped, leading separators too), a name that shares
     nothing stays whole, two names that would collide get less cut until
     they differ, a width cut ends in an ellipsis, one name alone is untouched
  B  RUN: _concat_blocker on real dicts -- clean -> None; mixed -> 'mixed'
     with the convention per LoRA; LoHA-like -> 'unrecognised'; DoRA and a
     conv mid -> 'foreign'
  E  the Stack decides with it: _apply_concat_or_dare calls _concat_blocker
     and routes 'unrecognised' / 'mixed' by its answer; its 'foreign' branch
     tests the same thing (driven on every fixture)
  A  RUN: the Analyzer in a stub package on real files -- a mixed group reads
     'CONCAT +TRIM -> runs SEQ', the odd rows are marked with their naming,
     the reason and the consequence are spelled out; a clean group is not
     touched; a blocked group is not counted as a Resolve group; a missing
     file is named; names are the distinct ones everywhere (overview, pair
     table, left-out lines, which now name the convention)

RE-GROUNDED IN v986: a group that mixes key naming is no longer blocked --
the merge reads each LoRA in its own naming and meets the layers by
_canonical_base (test_v986_mixed_naming). So B/E/A below now pin: the mix is
REPORTED (not blocked), the Analyzer says 'merged per layer' and marks the
odd ones, and the 'runs SEQ' path is exercised with a LoRA that truly blocks
(DoRA keys). Nothing is left out of the overlap for its naming any more.

MUTATION PROBE: see CHANGELOG_v985.md.
"""

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import re
import sys
import tempfile
import types
from collections import OrderedDict

ROOT = pathlib.Path(__file__).resolve().parent.parent
NODES = ROOT / "nodes"
sys.path.insert(0, str(NODES))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILED = []


def _need(cond, msg):
    if cond:
        print("ok  : " + msg)
    else:
        FAILED.append(msg)
        print("FAIL: " + msg)


try:
    import torch
    from safetensors.torch import save_file, load_file
except Exception:
    print("SKIP: torch / safetensors not installed")
    sys.exit(0)

import _lift as _LIFT  # noqa: E402
import uls_overlap_math as OV  # noqa: E402
import uls_merge_math as MM  # noqa: E402
import uls_merge_policy as MP  # noqa: E402

ELL = "\u2026"
P = "polyhedron_minimax_h3_image_lora_"

# --- N: names ------------------------------------------------------------------
frank = [P + "_liza_l8za.safetensors", P + "_perfe8ct.safetensors",
         P + "v2__glamgir8ls.safetensors", P + "v2_00001800__nippl8es.safetensors",
         P + "_gasmask_ga8s.safetensors", P + "_robot_r0b8t.safetensors",
         P + "_maelstrom2_ma8elstrom.safetensors", P + "oxyg8en.safetensors",
         "Minimax H3_Motion_Repair.safetensors",
         "polyhedron_minimax_h3_image_barnacles_barn8cles.safetensors"]
sh = OV.name_shortener(frank)
d24 = [sh(n, 24) for n in frank]
_need(len(set(d24)) == len(frank), "N  all ten of Frank's names are distinct at width 24: %s" % d24)
_need(sh(frank[0], 34) == ELL + "liza_l8za" and sh(frank[4], 24) == ELL + "gasmask_ga8s",
      "N  the shared prefix AND the separators after it are dropped (%r)" % sh(frank[0], 34))
_need(sh("Minimax H3_Motion_Repair.safetensors", 34) == "Minimax H3_Motion_Repair",
      "N  a name that shares nothing stays whole")
_need(not any(d.startswith(ELL + "_") for d in d24), "N  no display starts with a separator")
two = [P + "v2__glamgir8ls.safetensors", P + "_glamgir8ls.safetensors",
       P + "v2_00001800__nippl8es.safetensors"]
s2 = OV.name_shortener(two)
_need(len({s2(n, 34) for n in two}) == 3 and "v2" in s2(two[0], 34),
      "N  two names that would read the same get less cut until they differ (%s)"
      % [s2(n, 34) for n in two])
long_ = OV.name_shortener(["a_" + "x" * 60, "b_y"])
_need(long_("a_" + "x" * 60, 20).endswith(ELL) and len(long_("a_" + "x" * 60, 20)) == 20,
      "N  a width cut ends in an ellipsis and keeps the width")
_need(OV.name_shortener([P + "_one.safetensors"])(P + "_one.safetensors", 60) == P + "_one",
      "N  one name alone is not cut")
_need(OV.name_shortener(["sub/dir/" + P + "_a.safetensors", P + "_b.safetensors"])(
      "sub/dir/" + P + "_a.safetensors") == ELL + "a", "N  a subfolder path is shortened by its file name")
_need(OV.name_shortener([])("x.safetensors") == "x", "N  an unknown name still shows its stem")


# --- B: _concat_blocker on real dicts ------------------------------------------
torch.manual_seed(985)


def mk(conv, blocks=(0, 1), r=4, extra=None):
    td = {}
    for b in blocks:
        if conv == "kohya":
            base, up, dn = "lora_unet_blocks_%d_attn_qkv" % b, ".lora_up.weight", ".lora_down.weight"
        elif conv == "loha":
            base, up, dn = "lora_unet_blocks_%d_attn_qkv" % b, ".hada_w1_a", ".hada_w1_b"
        else:
            base, up, dn = "diffusion_model.blocks.%d.attn.qkv" % b, ".lora_B.weight", ".lora_A.weight"
        td[base + up] = torch.randn(24, r) * 0.1
        td[base + dn] = torch.randn(r, 24) * 0.1
        td[base + ".alpha"] = torch.tensor(float(r))
        if extra:
            td[base + extra] = torch.ones(1)
    return td


tmp = tempfile.mkdtemp()
lora_dir = os.path.join(tmp, "loras")
os.makedirs(lora_dir)
FILES = {
    P + "_liza_l8za.safetensors": mk("wan", (0, 1)),
    P + "_gasmask_ga8s.safetensors": mk("wan", (0, 2)),
    P + "v2__glamgir8ls.safetensors": mk("kohya", (0, 1)),
    "Minimax H3_Motion_Repair.safetensors": mk("kohya", (1, 2)),
    P + "_fog_fo8g.safetensors": mk("wan", (0, 1)),
    P + "_ocean_oc8an.safetensors": mk("wan", (1, 2)),
    "odd_loha.safetensors": mk("loha"),
    "odd_dora.safetensors": mk("wan", extra=".dora_scale"),
    P + "_hair_h8ir.safetensors": mk("wan", (0, 1)),
}
for fn, td in FILES.items():
    save_file({k: v.contiguous() for k, v in td.items()}, os.path.join(lora_dir, fn))


def gp(n):
    p = os.path.join(lora_dir, n)
    return p if os.path.exists(p) else None


SRC = (NODES / "uls_stack_node.py").read_text(encoding="utf-8")
lift, missing = _LIFT.close_over(
    SRC, ["_sort_active_rows", "_short_name", "_row_clip_weight", "_group_mult", "_group_scaled",
          "_group_cap_factor", "_group_effective", "_cap_text", "_concat_blocker", "_convention_label", "_naming_mix"],
    provided={"math", "os", "GROUP_ORDER", "folder_paths", "OrderedDict", "_cached_load_torch_file",
              "_convert_lora_like_core", "_detect_convention", "_resolve_pick_device",
              "_check_interrupt", "INTERRUPT_EXC", "_ov_measure", "_ov_cap", "_foreign_keys",
              "_has_mid_tensor"})
_need(not missing, "B  stack helpers lift closed (%s)" % (sorted(missing) or "none missing"))
fp = types.ModuleType("folder_paths")
fp.get_full_path = lambda kind, name: gp(name)
sys.modules["folder_paths"] = fp
pkg = types.ModuleType("plsv985")
pkg.__path__ = [str(NODES)]
sys.modules["plsv985"] = pkg
sn = types.ModuleType("plsv985.uls_stack_node")
ns = sn.__dict__
import math  # noqa: E402
ns.update(math=math, os=os, OrderedDict=OrderedDict, folder_paths=fp,
          GROUP_ORDER=["\u2014", "acc", "style", "scene", "motion", "subject", "detail", "custom"])
exec(compile(lift, "<v985 stack helpers>", "exec"), ns)
LOADS = []


def _load(path):
    LOADS.append(os.path.basename(path))
    return load_file(path)


ns.update(_cached_load_torch_file=_load, _convert_lora_like_core=lambda td, path=None: td,
          _detect_convention=MM._detect_convention, _collect_factor_keys=MM._collect_factor_keys,
          _has_mid_tensor=MM._has_mid_tensor, _resolve_sign_elect=MM._resolve_sign_elect,
          _trim_channel_indices=MM._trim_channel_indices, _trim_keep_fraction=MM._trim_keep_fraction,
          _resolve_pick_device=lambda *a, **k: "cpu", _check_interrupt=lambda: None,
          INTERRUPT_EXC=(KeyboardInterrupt,), _ov_measure=OV.measure_overlap, _ov_cap=OV.cap_factor,
          _apply_concat_or_dare=lambda *a, **k: None,   # v988: the Analyzer imports it (merge check)
          _foreign_keys=MP._foreign_keys, _canonical_base=MP._canonical_base)
sys.modules["plsv985.uls_stack_node"] = sn
CB = ns["_concat_blocker"]


def L(*names):
    return [load_file(gp(n)) for n in names]


W1, W2, K1 = P + "_liza_l8za.safetensors", P + "_gasmask_ga8s.safetensors", P + "v2__glamgir8ls.safetensors"
_need(CB([W1, W2], L(W1, W2)) is None, "B  two LoRAs of one naming: the merge runs")
b = CB([W1, W2, K1], L(W1, W2, K1))
mix = ns["_naming_mix"]([W1, W2, K1], L(W1, W2, K1))
_need(b is None and [lab.split(" ")[0] for _n, lab in mix] == ["WAN/FLUX", "WAN/FLUX", "kohya"],
      "B  v986: a mixed group is NOT blocked; the mix is reported per LoRA (%s)" % (mix,))
_need(ns["_naming_mix"]([W1, W2], L(W1, W2)) == [], "B  one naming: no mix reported")
b = CB([W1, "odd_loha.safetensors"], L(W1, "odd_loha.safetensors"))
_need(b and b[0] == "unrecognised" and b[1][0][0] == "odd_loha.safetensors",
      "B  a LoRA in no known layout: 'unrecognised', named")
b = CB([W1, "odd_dora.safetensors"], L(W1, "odd_dora.safetensors"))
_need(b and b[0] == "foreign" and b[1] == [("odd_dora.safetensors", "DoRA scale")],
      "B  a DoRA LoRA: 'foreign', named with its family (%s)" % (b,))
mid = load_file(gp(W2))
mid["diffusion_model.blocks.0.attn.qkv.lora_mid.weight"] = torch.ones(4, 4)
b = CB([W1, "mid"], [load_file(gp(W1)), mid])
_need(b and b[0] == "foreign", "B  a conv mid tensor: 'foreign'")

# --- E: the Stack decides with it ---------------------------------------------
fn = re.search(r"def _apply_concat_or_dare\(.*?(?=\ndef )", SRC, re.S)
body = fn.group(0) if fn else ""
_need("_blk = _concat_blocker(valid_names, raw)" in body,
      "E  _apply_concat_or_dare asks _concat_blocker")
_need('unrecognised = [n for n, _l in _blk[1]] if _bk == "unrecognised" else []' in body
      and 'if _bk == "mixed":' not in body and "_mix = _naming_mix(valid_names, raw)" in body,
      "E  its 'unrecognised' fallback follows the answer; a mix is only reported (v986)")
_need(body.count("_detect_convention(td) for td in raw") == 1 and "len({c for c in convs}) > 1" not in body,
      "E  no second copy of the mixed-naming test left in the Stack")
same = True
for grp in ([W1, W2], [W1, "odd_dora.safetensors"], [W2, "odd_dora.safetensors"]):
    raw = L(*grp)
    fk = [(n, MP._foreign_keys(td)) for n, td in zip(grp, raw)]
    stack_says = bool([x for x in fk if x[1]] or any(MM._has_mid_tensor(td) for td in raw))
    blk = CB(grp, raw)
    same &= stack_says == bool(blk and blk[0] == "foreign")
raw = [load_file(gp(W1)), mid]
same &= bool(MM._has_mid_tensor(mid) or MP._foreign_keys(mid)) == (CB([W1, "m"], raw)[0] == "foreign")
_need(same, "E  the Stack's 'foreign' test and the blocker's agree on every fixture")

# --- A: the Analyzer on real files --------------------------------------------
probe = types.ModuleType("plsv985.ph_joint_probe")
probe._joint_latent_parts = lambda model: 1
sys.modules["plsv985.ph_joint_probe"] = probe
for mod in ("uls_merge_math", "uls_merge_policy", "uls_overlap_math", "uls_resolve_inspector"):
    spec = importlib.util.spec_from_file_location("plsv985." + mod, str(NODES / (mod + ".py")))
    m = importlib.util.module_from_spec(spec)
    sys.modules["plsv985." + mod] = m
    spec.loader.exec_module(m)
RI = sys.modules["plsv985.uls_resolve_inspector"]

rows = [{"name": n, "group": "subject", "weight": 0.2, "on": True}
        for n in (W1, W2, K1, "Minimax H3_Motion_Repair.safetensors")]
rows += [{"name": n, "group": "scene", "weight": 0.2, "on": True}
         for n in (P + "_fog_fo8g.safetensors", P + "_ocean_oc8an.safetensors", P + "_gone_g0ne.safetensors")]
rows += [{"name": n, "group": "detail", "weight": 0.2, "on": True}
         for n in (P + "_hair_h8ir.safetensors", "odd_dora.safetensors")]
cfg = {"rows": rows, "group_modes": {"subject": "CONCAT", "scene": "CONCAT", "detail": "CONCAT"},
       "group_trim": {"subject": True, "scene": True, "detail": True},
       "group_resolve": {"subject": True, "scene": True, "detail": True}}
with contextlib.redirect_stdout(io.StringIO()):
    rep, _e, _a, act = RI.ULSResolveInspector().analyze(json.dumps(cfg), "Overview")
def _sect(name):
    if "[%s]" % name not in rep:
        return ""
    t = rep.split("[%s]" % name)[1]
    return re.split(r"\n  \[|\u2500{10}", t)[0]


subj, scen, detl = _sect("subject"), _sect("scene"), _sect("detail")
_need(subj and "\u2192 runs SEQ" not in subj.splitlines()[0]
      and "key naming mixed (2 \u00d7 WAN/FLUX, 2 \u00d7 kohya) -- merged per layer" in subj,
      "A  v986: a mixed group merges and says so (%r)" % subj.splitlines()[0][:70])
_need(scen and "\u2192 runs SEQ" not in scen.splitlines()[0],
      "A  a clean group is not touched (%r)" % scen.splitlines()[0][:60])
marked = [ln for ln in subj.splitlines() if ln.rstrip().endswith("[kohya]")]
_need(len(marked) == 2 and "glamgir8ls" in marked[0] and "Motion_Repair" in marked[1],
      "A  the odd rows are marked with their naming")
_need(detl and "CONCAT +TRIM +RESOLVE \u2192 runs SEQ" in detl.splitlines()[0]
      and "carry keys a merge cannot hold" in detl and "no merge, no TRIM" in detl
      and "no energy cap, no bake" in detl and "give them their own group" in detl
      and detl.count("[keys]") == 1,
      "A  a truly blocked group says it runs SEQ, why, the consequence and the way out")
_need(("\u2022 " + ELL + "liza_l8za") in subj and "polyhedron_minimax_h3_image_lora__" not in rep,
      "A  the overview shows the distinct names")
_need("not loadable: " + ELL + "gone_g0ne" in scen, "A  a missing file in a merge group is named")
_rl = [ln for ln in rep.splitlines() if "Resolve groups with" in ln]
_need(act is True and _rl and ": 2  (" in _rl[0] and "detail" not in _rl[0],
      "A  a blocked group is not counted as a Resolve group (%s)" % _rl[:1])

with contextlib.redirect_stdout(io.StringIO()):
    rep2, *_x = RI.ULSResolveInspector().analyze(json.dumps(cfg), RI.DEPTH_OVERLAP)
pairs = [ln for ln in rep2.splitlines() if "<->" in ln]
_need(pairs and any(re.search(ELL + r"\w+ <-> " + ELL + r"\w+", ln) for ln in pairs)
      and not any("polyhedron_minimax" in ln for ln in pairs),
      "A  the pair table reads in distinct names (%s)" % (pairs[:1],))
lo = [ln for ln in rep2.splitlines() if "left out:" in ln]
_need(not any("other key naming" in ln for ln in lo)
      and any(("left out: " + ELL + "gone_g0ne") in ln for ln in lo)
      and "Key naming mixed" in rep2
      and any("Motion_Repair" in ln and "share" in ln for ln in rep2.splitlines()),
      "A  v986: nothing is left out for its naming -- the kohya LoRAs are measured; "
      "a missing file is named distinctly (%s)" % lo[:1])

print()
if FAILED:
    print("test_v985_analyzer_honest: %d failure(s)" % len(FAILED))
    sys.exit(1)
print("test_v985_analyzer_honest: 0 failure(s)")
