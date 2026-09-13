# -*- coding: utf-8 -*-
"""
test_v302_clip_weight.py
════════════════════════
Guards the v302 per-row CLIP strength feature. Three parts, none of which
need ComfyUI/torch (pattern follows test_v267):

  [1] Pure functions, AST-extracted and exec'd standalone:
        • _row_clip_weight — wClip read, non-numeric/missing → fallback
        • _is_te_base      — TE-prefix classification (graceful-miss design)
  [2] Source wiring (text + AST):
        • _apply_seq signature carries clip_weights and calls
          load_lora(m, c, name, w, wc) — separate strengths
        • all 10 SEQ fallbacks inside the merge pass valid_clip_weights
        • merge build picks weight via _is_te_base(base)
        • DARE/RESOLVE seed still derived from (valid_names, valid_weights)
          ONLY — clip weights must NOT change existing masks
        • apply_lora_set keeps rows where EITHER strength is non-zero
        • frontend serializes wClip in both mappers; backward-compat default
  [3] Behavioural check of the combined-zero filter logic (pure re-impl).
"""
import os
import re
import ast
import sys
import math
import types

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "nodes", "uls_stack_node.py")
JS  = os.path.join(HERE, "..", "web", "js", "uls_node.js")

failures = []


def check(label, cond):
    mark = "✓" if cond else "✗"
    print(f"  {mark} {label}")
    if not cond:
        failures.append(label)


with open(SRC, encoding="utf-8") as f:
    src = f.read()
# v348: _is_te_base + _TE_KEY_PREFIXES moved to uls_merge_math.py; _row_clip_weight
# stayed in uls_stack_node.py — concatenate both so every picked helper is found.
_MM = os.path.join(HERE, "..", "nodes", "uls_merge_math.py")
if os.path.exists(_MM):
    with open(_MM, encoding="utf-8") as f:
        src += "\n" + f.read()
tree = ast.parse(src)

# ── [1] Extract + exec the pure helpers ────────────────────────────────────
ns = {"math": math}
wanted = {"_safe_weight", "_row_clip_weight", "_is_te_base"}
picked = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        picked.append(node)
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name) and t.id == "_TE_KEY_PREFIXES":
                picked.append(node)
mod = ast.Module(body=picked, type_ignores=[])
exec(compile(mod, "<v302-helpers>", "exec"), ns)

print("[1] pure helpers")
rcw = ns["_row_clip_weight"]
check("wClip read",              rcw({"wClip": 0.8}, 1.0) == 0.8)
check("missing → fallback",      rcw({}, 0.5) == 0.5)
check("non-numeric → fallback",  rcw({"wClip": "x"}, 0.3) == 0.3)
check("None → fallback",         rcw({"wClip": None}, 0.7) == 0.7)
check("zero is a valid value",   rcw({"wClip": 0}, 1.0) == 0.0)
check("negative allowed",        rcw({"wClip": -0.4}, 1.0) == -0.4)

ite = ns["_is_te_base"]
check("kohya te",        ite("lora_te_text_model_encoder_layers_0_mlp_fc1") is True)
check("sdxl te1",        ite("lora_te1_text_model_encoder_layers_0_self_attn_q_proj") is True)
check("sdxl te2",        ite("lora_te2_text_model_encoder_layers_0_self_attn_q_proj") is True)
check("diffusers te",    ite("text_encoder.layers.0.mlp.fc1") is True)
check("cascade prior",   ite("lora_prior_te_something") is True)
check("unet not te",     ite("lora_unet_down_blocks_0_attentions_0") is False)
check("wan not te",      ite("diffusion_model.blocks.0.self_attn.q") is False)
check("empty not te",    ite("") is False)

# ── [2] Source wiring ──────────────────────────────────────────────────────
print("[2] wiring")
check("seq signature has clip_weights",
      "def _apply_seq(loader, model, clip, names: list, weights: list,\n"
      "               clip_weights: list = None)" in src)
check("seq calls load_lora(m, c, name, w, wc)",
      "loader.load_lora(m, c, name, w, wc)" in src)
check("old linked call gone",
      "loader.load_lora(m, c, name, w, w)" not in src)
n_fb = src.count("_apply_seq(loader, model, clip, valid_names, valid_weights, valid_clip_weights)")
# public v374: an eleventh fallback came with the internal v913-v935 line
# (Baked apply / PDD); every one of them still passes valid_clip_weights.
check(f"all 11 merge fallbacks pass clip weights (found {n_fb})", n_fb == 11)
check("no fallback left without clip weights",
      "_apply_seq(loader, model, clip, valid_names, valid_weights)" not in src)
check("merge picks weight via _is_te_base",
      "valid_clip_weights[li] if _is_te_base(base) else valid_weights[li]" in src)
check("DARE seed unchanged (model weights only)",
      src.count("_dare_seed(valid_names, valid_weights)") == 2
      and "_dare_seed(valid_names, valid_clip_weights)" not in src)
check("OOM retry carries clip_weights",
      re.search(r'force_resolve_device="cpu",\s*clip_weights=clip_weights', src) is not None)   # v374: handoff= follows since v913
check("apply_lora_set keeps CLIP-only rows",
      "abs(float(w)) >= 1e-6 or abs(float(wc)) >= 1e-6" in src)
check("stack site collects grp_clip", "grp_clip = [round(_row_clip_weight(r, w), 4)" in src)
check("stack site passes clip_weights", "clip_weights=grp_clip" in src)
check("engine site collects active_clip",
      "active_clip.append(round(_row_clip_weight(row, w), 4))" in src)
check("engine site passes clip_weights", "clip_weights=active_clip," in src)

with open(JS, encoding="utf-8") as f:
    js = f.read()
check("frontend serializes wClip (4 mappers)",  # v309: +2 Engine mappers
      js.count("wClip: r.wClip,") == 4)
check("stack shift-click edits CLIP", "\"CLIP Strength\"" in js)  # v314 label
check("two-line cell rendered", js.count("row.wClip.toFixed(2)") == 2)

# ── [3] combined-zero filter behaviour (pure re-impl of the new rule) ──────
print("[3] filter behaviour")
def survives(w, wc):
    return abs(float(w)) >= 1e-6 or abs(float(wc)) >= 1e-6
check("model-only row survives", survives(0.8, 0.0))
check("clip-only row survives",  survives(0.0, 0.8))
check("dead row dropped",        not survives(0.0, 0.0))
check("linked default survives", survives(1.0, 1.0))

print("=" * 56)
if failures:
    print(f"RESULT: {len(failures)} CHECK(S) FAILED")
    sys.exit(1)
print("RESULT: ALL CHECKS PASS")
sys.exit(0)
