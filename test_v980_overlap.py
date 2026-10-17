#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v980 -- Merge Analyzer: 'Overlap & energy' and what really runs on the model.

THE FINDING (21.09.): on MiniMax H3 the Stack turns DARE into CONCAT and
switches RESOLVE off (v912), but the Analyzer had no model input and reported
the modes as SET. And with only the plain sum left, the question that decides
a 19-LoRA stack -- which LoRAs pull together, which fight, which barely add
anything -- had no instrument.

WHAT IS PINNED, and how:

  A  uls_overlap_math.measure_overlap, DRIVEN on random LoRAs against the
     brute force (full deltas built, inner products summed): every G[i, j]
     to 1e-5 relative. Alpha, a missing alpha, the model weight, the CLIP
     weight on a text-encoder layer, several ranks, partial layer overlap.
  B  a layer whose shape disagrees is skipped AND counted; block and kind
     parsing for ComfyUI and kohya keys.
  C  summarize: identical LoRAs -> cosine +1, a negated copy -> -1 and a
     negative net; ratio = total / independent; the report names 'pull the
     same way', 'push against each other', 'quiet', 'works against the rest'.
  D  the Analyzer itself, loaded in a stub package (no comfy): not connected
     -> modes as set; joint model -> 'DARE->CONCAT (joint)', RESOLVE off, no
     Resolve group left; plain model -> as set; Overlap depth loads every
     active LoRA through the converter, ticks the interrupt, names what it
     leaves out, prints the shared-blocks note on a joint model.
  E  schema: DEPTHS append-only (the two old names first, unchanged), the
     socket 'model' is the LAST optional entry, the V3 form reads the SAME
     list (AST: options=list(_OV.DEPTHS)) and passes model through.

Needs torch. No comfy.
"""
import ast
import os
import pathlib
import sys
import types
import importlib.util

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
except Exception:
    print("SKIP: torch not installed")
    sys.exit(0)

import _lift as _LIFT  # noqa: E402
import uls_overlap_math as OV  # noqa: E402
from uls_merge_math import _detect_convention  # noqa: E402

torch.manual_seed(980)


def mk(rank, alpha, blocks, out=36, inn=28, kind="attn.qkv_proj",
       prefix="diffusion_model.blocks.", dtype=torch.bfloat16):
    td = {}
    for b in blocks:
        base = "%s%d.%s" % (prefix, b, kind)
        td[base + ".lora_B.weight"] = torch.randn(out, rank, dtype=dtype)
        td[base + ".lora_A.weight"] = torch.randn(rank, inn, dtype=dtype)
        if alpha is not None:
            td[base + ".alpha"] = torch.tensor(float(alpha))
    return td


def full(td, w, wc):
    d = {}
    for k in td:
        if not k.endswith(".lora_B.weight"):
            continue
        base = k[:-len(".lora_B.weight")]
        B = td[k].float()
        A = td[base + ".lora_A.weight"].float()
        s = (td[base + ".alpha"].item() / A.shape[0]) if base + ".alpha" in td else 1.0
        ww = wc if base.startswith("text_encoder") else w
        d[base] = ww * s * (B.reshape(B.shape[0], -1) @ A.reshape(A.shape[0], -1))
    return d


def brute(tds, ws, wcs):
    F = [full(t, w, c) for t, w, c in zip(tds, ws, wcs)]
    n = len(F)
    return [[sum(float((F[i][b] * F[j][b]).sum()) for b in F[i] if b in F[j])
             for j in range(n)] for i in range(n)]


def entries(tds, ws, wcs):
    return [dict(td=t, conv=_detect_convention(t), weight=w, clip_weight=c)
            for t, w, c in zip(tds, ws, wcs)]


# --- A: exact against the brute force -----------------------------------------
te = mk(4, 2, [0], prefix="text_encoder.encoder.block.", kind="mlp.fc1")
L0 = mk(8, 4, [0, 1, 2])
L1 = mk(16, None, [1, 2, 3])
L2 = mk(4, 8, [2])
L3 = dict(mk(8, 8, [5, 6], kind="mlp.fc1"))
L3.update(te)
tds = [L0, L1, L2, L3]
ws = [0.3, -0.2, 1.1, 0.7]
wcs = [0.3, -0.2, 1.1, 0.05]
m = OV.measure_overlap(entries(tds, ws, wcs), torch)
ref = brute(tds, ws, wcs)
worst = 0.0
for i in range(4):
    for j in range(4):
        worst = max(worst, abs(ref[i][j] - m["G"][i][j]) / max(1.0, abs(ref[i][j])))
_need(worst < 1e-5, "A  every G[i,j] equals the brute-force inner product (worst rel %.1e)" % worst)
_need(abs(m["G"][0][1] - m["G"][1][0]) < 1e-6 * max(1.0, abs(m["G"][0][1])), "A  G is symmetric")
_need(m["layers"] == [3, 3, 1, 3], "A  layer count per LoRA: %s" % m["layers"])

# --- B: shape mismatch, block and kind parsing -----------------------------------
bad = mk(8, 8, [1], out=40)
mb = OV.measure_overlap(entries([L0, bad], [1.0, 1.0], [1.0, 1.0]), torch)
_need(mb["skipped_shape"] == 1 and mb["layers"][1] == 0,
      "B  a layer with another shape is skipped and counted (%d)" % mb["skipped_shape"])
_need(OV.block_index("diffusion_model.blocks.17.attn.qkv_proj") == 17
      and OV.block_index("lora_unet_blocks_4_mlp_fc1") == 4
      and OV.block_index("diffusion_model.token_refiner.blocks.1.attn") == 1
      and OV.block_index("diffusion_model.final_layer.video_out") is None,
      "B  block index read from ComfyUI and kohya keys")
_need([OV.module_kind(x) for x in (
    "diffusion_model.blocks.1.attn.qkv_proj", "diffusion_model.blocks.1.mlp.fc2",
    "diffusion_model.blocks.1.adaln_proj.linear", "diffusion_model.token_refiner.blocks.0.mlp.fc1",
    "diffusion_model.condition_proj")] == ["attn", "mlp", "adaln", "refiner", "other"],
    "B  module kinds attn / mlp / adaln / refiner / other")

# --- C: summarize and the report wording -----------------------------------------
X = mk(8, 8, [0, 1])
negX = {k: (-v if k.endswith(".lora_B.weight") else v) for k, v in X.items()}
tiny = mk(8, 8, [7])
mc = OV.measure_overlap(entries([X, X, negX, tiny], [1.0, 1.0, 0.5, 0.01], [1.0] * 4), torch)
sc = OV.summarize(mc["G"], ["s", "s", "s", "d"])
cos = {(i, j): c for i, j, c in sc["pairs"]}
_need(abs(cos[(0, 1)] - 1.0) < 1e-6, "C  identical LoRAs: cosine +1 (%.6f)" % cos[(0, 1)])
_need(abs(cos[(0, 2)] + 1.0) < 1e-6, "C  negated copy: cosine -1 (%.6f)" % cos[(0, 2)])
_need(sc["net"][2] < 0, "C  the negated copy has a negative net contribution (%.3f)" % sc["net"][2])
_need(abs(sc["ratio"] - sc["total"] / sc["indep"]) < 1e-12, "C  ratio = total / independent")
# Sharpened after the v980 mutation round: the fixture above lands at ratio
# ~1.0 (1.5^2 / 2.25), where a ratio and its inverse agree -- M6 slipped
# through. Two identical LoRAs reinforce to exactly (1+1)^2 / 2 = 2.
m2 = OV.measure_overlap(entries([X, X], [1.0, 1.0], [1.0, 1.0]), torch)
s2 = OV.summarize(m2["G"], ["s", "s"])
_need(abs(s2["ratio"] - 2.0) < 1e-6, "C  two identical LoRAs reinforce: ratio 2.00 (%.4f)" % s2["ratio"])
mo = OV.measure_overlap(entries([X, negX], [1.0, 0.5], [1.0, 1.0]), torch)
so = OV.summarize(mo["G"], ["s", "s"])
_need(abs(so["ratio"] - 0.25 / 1.25) < 1e-6, "C  X and -0.5X cancel: ratio 0.20 (%.4f)" % so["ratio"])
tot = sum(sum(r) for r in mc["G"])
_need(abs(sc["total"] - tot) < 1e-9 * max(1.0, tot), "C  total = sum of G")
_need(abs(sc["group"]["s"]["ratio"] - (1.5 ** 2) / (1 + 1 + 0.25)) < 1e-4,
      "C  group ratio of X, X, -0.5X = 1.5^2 / 2.25 (%.4f)" % sc["group"]["s"]["ratio"])
rep = "\n".join(OV.overlap_lines(["x", "x2", "negx", "tiny"], ["s", "s", "s", "d"],
                                 [1.0, 1.0, 0.5, 0.01], mc, sc, lambda n, w: n))
for word in ("pull the same way", "push against each other", "<- quiet",
             "works against the rest", "[s] 3 LoRAs", "Quiet LoRAs"):
    _need(word in rep, "C  report says %r" % word)

# --- D: the Analyzer, in a stub package ------------------------------------------
STACK_SRC = (NODES / "uls_stack_node.py").read_text(encoding="utf-8")
lift_src, missing = _LIFT.close_over(STACK_SRC, ["_sort_active_rows", "_short_name",
                                                 "_row_clip_weight",
                                                 # v981/v983: the Analyzer folds in strength + cap
                                                 "_group_mult", "_group_scaled",
                                                 "_group_effective", "_cap_text",
                                                 # v985: the Analyzer asks the Stack's own fallback decision
                                                 "_concat_blocker", "_convention_label", "_naming_mix"],
                                     provided={"math", "os", "GROUP_ORDER", "folder_paths", "OrderedDict", "_cached_load_torch_file", "_convert_lora_like_core", "_detect_convention", "_resolve_pick_device", "_check_interrupt", "INTERRUPT_EXC", "_ov_measure", "_ov_cap",
                                               "_foreign_keys", "_has_mid_tensor", "_canonical_base"})
_need(not missing, "D  stack helpers lift closed (%s)" % (sorted(missing) or "none missing"))

STORE = {}
CONVERTED = []
TICKS = [0]


def _load(path):
    return STORE[path]


def _conv(td, path=None):
    CONVERTED.append(path)
    return td


def _tick():
    TICKS[0] += 1


pkg = types.ModuleType("plsv980")
pkg.__path__ = [str(NODES)]
sys.modules["plsv980"] = pkg
fp = types.ModuleType("folder_paths")
fp.get_full_path = lambda kind, name: name if name in STORE else None
sys.modules["folder_paths"] = fp
import math as _math  # noqa: E402
import uls_merge_math as MM  # noqa: E402
sn = types.ModuleType("plsv980.uls_stack_node")
ns = sn.__dict__
ns.update(math=_math, os=os, OrderedDict=__import__("collections").OrderedDict,
          GROUP_ORDER=["\u2014", "acc", "style", "scene", "motion", "subject", "detail", "custom"])
exec(lift_src, ns)
ns.update(_detect_convention=MM._detect_convention, _collect_factor_keys=MM._collect_factor_keys,
          _has_mid_tensor=MM._has_mid_tensor, _resolve_sign_elect=MM._resolve_sign_elect,
          _trim_channel_indices=MM._trim_channel_indices, _trim_keep_fraction=MM._trim_keep_fraction,
          _cached_load_torch_file=_load, _resolve_pick_device=lambda *a, **k: "cpu",
          _check_interrupt=_tick, INTERRUPT_EXC=(KeyboardInterrupt,),
          _convert_lora_like_core=_conv,
          # v983: the cap's own imports (the cap is off in these fixtures)
          folder_paths=fp, OrderedDict=__import__("collections").OrderedDict,
          _ov_measure=OV.measure_overlap, _ov_cap=OV.cap_factor,
          _apply_concat_or_dare=lambda *a, **k: None,   # v988: the Analyzer imports it (merge check)
          _foreign_keys=__import__("uls_merge_policy")._foreign_keys,
          _canonical_base=__import__("uls_merge_policy")._canonical_base)   # v985: _concat_blocker
sys.modules["plsv980.uls_stack_node"] = sn
probe = types.ModuleType("plsv980.ph_joint_probe")
probe._joint_latent_parts = lambda model: model.parts
sys.modules["plsv980.ph_joint_probe"] = probe
for mod in ("uls_merge_policy", "uls_overlap_math", "uls_merge_math"):
    spec = importlib.util.spec_from_file_location("plsv980." + mod, str(NODES / (mod + ".py")))
    mm = importlib.util.module_from_spec(spec)
    sys.modules["plsv980." + mod] = mm
    spec.loader.exec_module(mm)
spec = importlib.util.spec_from_file_location("plsv980.uls_resolve_inspector",
                                              str(NODES / "uls_resolve_inspector.py"))
RI = importlib.util.module_from_spec(spec)
sys.modules["plsv980.uls_resolve_inspector"] = RI
spec.loader.exec_module(RI)

STORE.update({"a.safetensors": X, "b.safetensors": X, "c.safetensors": negX,
              "d.safetensors": tiny})
CFG = ('{"rows": ['
       '{"name": "a.safetensors", "group": "subject", "weight": 1.0, "on": true},'
       '{"name": "b.safetensors", "group": "subject", "weight": 1.0, "on": true},'
       '{"name": "c.safetensors", "group": "subject", "weight": 0.5, "on": true},'
       '{"name": "d.safetensors", "group": "scene", "weight": 0.01, "on": true},'
       '{"name": "gone.safetensors", "group": "scene", "weight": 0.3, "on": true}],'
       '"group_modes": {"subject": "DARE", "scene": "CONCAT"},'
       '"group_resolve": {"subject": true}}')


class M:
    def __init__(self, parts):
        self.parts = parts


node = RI.ULSResolveInspector()
r0, *_x, act0 = node.analyze(CFG, "Overview")
_need("not connected" in r0 and "DARE [CHAN] +RESOLVE" in r0 and act0 is True,
      "D  no model: modes shown as set, RESOLVE group counted")
r1, *_x, act1 = node.analyze(CFG, "Overview", model=M(2))
_need("joint audio/video" in r1 and "DARE\u2192CONCAT (joint)" in r1
      and "(RESOLVE off: joint)" in r1 and act1 is False,
      "D  joint model: DARE->CONCAT, RESOLVE off, no Resolve group left")
r2, *_x = node.analyze(CFG, "Overview", model=M(1))
_need("plain -- modes run as set" in r2 and "DARE [CHAN] +RESOLVE" in r2,
      "D  plain model: modes run as set")
CONVERTED.clear()
TICKS[0] = 0
r3, *_x = node.analyze(CFG, RI.DEPTH_OVERLAP, model=M(2))
# v985: the overview now loads the merge groups too (the merge-path check), so
# a LoRA may pass the converter twice -- the claim is EVERY one, via the converter.
_need(sorted(set(CONVERTED)) == ["a.safetensors", "b.safetensors", "c.safetensors", "d.safetensors"],
      "D  Overlap loads every active LoRA through the converter")
_need(TICKS[0] >= 3, "D  the interrupt check ticks per layer (%d)" % TICKS[0])
_need("left out: gone" in r3 and "file not found" in r3, "D  a missing file is named, not dropped silently")
_need("SAME DiT blocks" in r3, "D  joint model: the shared-blocks note is printed")
_need("pull the same way" in r3 and "push against each other" in r3,
      "D  the report carries the pair verdicts")
r4, *_x = node.analyze(CFG, RI.DEPTH_OVERLAP)
_need("SAME DiT blocks" not in r4, "D  no model: no joint note")

# --- E: schema --------------------------------------------------------------------
it = RI.ULSResolveInspector.INPUT_TYPES()
depths = it["required"]["analysis_depth"][0]
_need(depths[:2] == ["Overview", "Deep analysis"] and depths[2] == "Overlap & energy",
      "E  depth list append-only: %s" % depths)
opt = list(it["optional"].keys())
_need(opt[-1] == "model" and it["optional"]["model"][0] == "MODEL",
      "E  'model' is the LAST optional entry and a MODEL socket (%s)" % opt)
V3 = (NODES / "uls_resolve_inspector_v3.py").read_text(encoding="utf-8")
tree = ast.parse(V3)
opts_ok = model_in = model_pass = False
for n in ast.walk(tree):
    if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "Input":
        args = [a.value for a in n.args if isinstance(a, ast.Constant)]
        if args[:1] == ["analysis_depth"]:
            for kw in n.keywords:
                if kw.arg == "options" and ast.unparse(kw.value) == "list(_OV.DEPTHS)":
                    opts_ok = True
        if args[:1] == ["model"] and ast.unparse(n.func) == "io.Model.Input":
            model_in = True
    if isinstance(n, ast.keyword) and n.arg == "model" and ast.unparse(n.value) == "model":
        model_pass = True
_need(opts_ok, "E  V3 reads options=list(_OV.DEPTHS)")
_need(model_in and model_pass, "E  V3 declares io.Model.Input('model') and passes model through")

print()
print("test_v980_overlap: %d failure(s)" % len(FAILED))
sys.exit(1 if FAILED else 0)
