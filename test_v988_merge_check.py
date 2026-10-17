#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v988 -- the Merge Analyzer measures the Stack's merge against SEQ.

Field, 22.09.2026: Frank's A/B (SEQ against CONCAT, same stack) did not look
the same. Whether that is the seed, bf16 rounding cascading through an 8-step
distilled sampler, or a merge fault, a picture cannot say. The new depth
'Merge check (vs SEQ)' builds each group twice on the connected model --
every LoRA through ComfyUI's own loader (what SEQ runs), and the Stack's
merge -- and compares the patches per weight, in factor space.

LIVE (sandbox ComfyUI, 22.09.2026, CHANGELOG_v988): Core's real load_lora
and convert_lora against the Stack's real _apply_concat_or_dare, mixed
naming, alpha present and missing, bf16 files: CONCAT 0.14% / 0.16% (bf16
storage) -> EQUAL; the same with alpha/rank deliberately dropped from the
merge -> 78% / 100%, every weight named -> DIFFERS.

  F  RUN adapter_factors: Core's LoRAAdapter.weights and the legacy tuple
     read alike; alpha None -> scale 1.0 (Core's rule); mid / DoRA / other
     adapters are refused, not guessed
  C  RUN compare: equal patches -> 0; a reference split over two LoRAs
     against one merged factor pair of the same sum -> 0; a missing LoRA
     -> the exact relative error; a weight only SEQ patches -> MISSING with
     the LoRA's name; one only the merge patches -> extra; the factor-space
     result equals the full-matrix result
  L  check_lines: EQUAL below 1% with nothing missing; DIFFERS names the
     worst weights
  A  the Analyzer: the depth is APPENDED (saved workflows store the name),
     the reference goes through comfy.lora_convert + comfy.lora.load_lora with
     the model's key map, the merged side through the Stack's own
     _apply_concat_or_dare with a capture model that keeps only keys the
     model has (as Core's add_patches does), a SEQ fallback is reported not
     compared, a LoRA Core maps to 0 weights is named
  N  the Stack's console shows the distinct names (group lines and the merge
     line); skipped-row matching still uses the old short name the errors carry
"""
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))
FAILED = []


def _need(cond, msg):
    print(("ok  : " if cond else "FAIL: ") + msg)
    if not cond:
        FAILED.append(msg)


try:
    import torch
except Exception:
    print("SKIP: torch not installed")
    sys.exit(0)
import uls_merge_check as MC  # noqa: E402
import uls_overlap_math as OV  # noqa: E402

torch.manual_seed(988)


class Ad:
    def __init__(self, up, down, alpha=None, mid=None, dora=None):
        self.weights = (up, down, alpha, mid, dora, None)


def rnd(o, r, i):
    return torch.randn(o, r), torch.randn(r, i)


# --- F ---------------------------------------------------------------------
B, A = rnd(12, 3, 10)
f = MC.adapter_factors(Ad(B, A, torch.tensor(6.0)), torch)
_need(f is not None and abs(f[2] - 2.0) < 1e-9, "F  alpha/rank read from Core's adapter (6/3 = 2)")
_need(MC.adapter_factors(Ad(B, A, None), torch)[2] == 1.0, "F  alpha None -> scale 1.0, Core's rule")
_need(MC.adapter_factors(("lora", (B, A, 3.0, None, None, None)), torch)[2] == 1.0,
      "F  the legacy ('lora', tuple) form reads alike")
_need(MC.adapter_factors(Ad(B, A, None, mid=torch.ones(1)), torch) is None
      and MC.adapter_factors(Ad(B, A, None, dora=torch.ones(1)), torch) is None
      and MC.adapter_factors(("diff", (B,)), torch) is None,
      "F  conv-mid, DoRA and non-LoRA patches are refused, not guessed")

# --- C ---------------------------------------------------------------------
B1, A1 = rnd(16, 4, 14)
B2, A2 = rnd(16, 2, 14)
ref = {"W": [("a", 0.5, Ad(B1, A1, torch.tensor(2.0))), ("b", 0.3, Ad(B2, A2, None))]}
Bm = torch.cat([B1 * (0.5 * 0.5), B2 * 0.3], 1)
Am = torch.cat([A1, A2], 0)
merged = {"W": [(1.0, Ad(Bm, Am, torch.tensor(6.0)))]}           # alpha = rank -> 1.0
r = MC.compare(ref, merged, torch)
_need(MC.group_rel(r) < 1e-6, "C  two LoRAs against their concatenated merge: 0 (%.2e)" % MC.group_rel(r))
merged_missing_b = {"W": [(1.0, Ad(B1 * 0.25, A1, torch.tensor(4.0)))]}
r = MC.compare(ref, merged_missing_b, torch)
full_ref = 0.25 * B1 @ A1 + 0.3 * B2 @ A2
want = float((0.25 * B1 @ A1 - full_ref).norm() / full_ref.norm())
_need(abs(MC.group_rel(r) - want) < 1e-5,
      "C  a LoRA left out: the exact relative error, factor space == full matrix (%.5f vs %.5f)"
      % (MC.group_rel(r), want))
r = MC.compare({"W": ref["W"], "X": [("c", 1.0, Ad(B1, A1, None))]}, merged, torch)
_need(r["missing"] == {"X": ["c"]}, "C  a weight only SEQ patches -> MISSING, with the LoRA's name")
r = MC.compare(ref, {"W": merged["W"], "Y": [(1.0, Ad(B1, A1, None))]}, torch)
_need(r["extra"] == ["Y"], "C  a weight only the merge patches -> extra")
r = MC.compare({"W": [("a", 1.0, Ad(B1, A1, None))]}, {"W": [(1.0, Ad(torch.randn(9, 4), A1, None))]}, torch)
_need(r["unreadable"] and "shape" in r["unreadable"][0][1], "C  a shape disagreement is reported, not crashed")

# --- L ---------------------------------------------------------------------
sh = OV.name_shortener(["x_a", "x_b"])
ok = MC.check_lines("g", MC.compare(ref, merged, torch), sh)
bad = MC.check_lines("g", MC.compare(ref, merged_missing_b, torch), sh)
_need("EQUAL to SEQ" in ok[0], "L  equal -> EQUAL (%s)" % ok[0].strip())
_need("DIFFERS from SEQ" in bad[0] and any(" W" in ln for ln in bad[1:]),
      "L  a difference -> DIFFERS and the weight is named")

# --- A ---------------------------------------------------------------------
ovs = (ROOT / "nodes" / "uls_overlap_math.py").read_text(encoding="utf-8")
m = re.search(r"^DEPTHS = \[(.*?)\]", ovs, re.M)
_need(m and m.group(1).split(",")[-1].strip() == "DEPTH_MERGE"
      and m.group(1).startswith('"Overview", "Deep analysis", DEPTH_OVERLAP'),
      "A  the depth is APPENDED; the old names keep their places")
ri = (ROOT / "nodes" / "uls_resolve_inspector.py").read_text(encoding="utf-8")
blk = re.search(r"def _merge_check_block\(.*?(?=\ndef )", ri, re.S)
blk = blk.group(0) if blk else ""
_need("comfy.lora_convert.convert_lora(_cached_load_torch_file(path))" in blk
      and "comfy.lora.load_lora(td, keymap, log_missing=False)" in blk
      and "comfy.lora.model_lora_keys_unet(model.model, {})" in blk,
      "A  the reference is Core's own SEQ path with the model's key map")
_need("_apply_concat_or_dare(ld, cap, None, names" in blk and "_CaptureModel(model, keys)" in blk,
      "A  the merged side is the Stack's own merge, into a capture model")
cap = re.search(r"class _CaptureModel.*?(?=\nclass |\ndef )", ri, re.S).group(0)
_need("if key in self._keys:" in cap, "A  the capture keeps only keys the model has (Core's add_patches rule)")
_need("fell back to SEQ here -- nothing to compare" in blk and "maps 0 weights" in blk,
      "A  a SEQ fallback is reported, a LoRA Core cannot map is named")
_need("Connect the model input" in blk, "A  without the model it says what it needs")

# --- N ---------------------------------------------------------------------
sn = (ROOT / "nodes" / "uls_stack_node.py").read_text(encoding="utf-8")
_need("_nm = _ov_names([r.get(\"name\", \"\") for _g, _rs, _w in ordered for r in _rs])" in sn
      and 'disp = _nm(row.get("name", ""), 35)' in sn
      and 'short = _short_name(row.get("name", ""), 35)   # matches the error texts' in sn,
      "N  the console lists distinct names; error matching keeps the old short name")
_need(sn.count("shorts = [_sn(n, 24) for n in valid_names]") == 2 and "_sn = _ov_names(valid_names)" in sn,
      "N  the merge line names its LoRAs distinctly")

print()
if FAILED:
    print("test_v988_merge_check: %d failure(s)" % len(FAILED))
    sys.exit(1)
print("test_v988_merge_check: 0 failure(s)")
