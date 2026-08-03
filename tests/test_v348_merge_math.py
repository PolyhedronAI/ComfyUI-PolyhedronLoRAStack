# -*- coding: utf-8 -*-
"""
test_v348_merge_math.py
═══════════════════════
Real numeric tests for the merge math, now possible because v348 moved it into
nodes/uls_merge_math.py (a ComfyUI-free module). These exercise the ACTUAL
production helpers with torch — not a reimplementation — so a regression in the
DARE mask, the Trim channel selection, the convention detection, or the TIES
RESOLVE sign-election surfaces as a failing assertion rather than a silent
wrong-merge.

Requires torch. Where torch is absent (e.g. a CI box without it) the whole file
SKIPS with exit 0, matching the existing v259/v261/v264 convention.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "nodes"))

try:
    import torch
except Exception:
    print("SKIP: torch not importable in this environment")
    sys.exit(0)

import uls_merge_math as M

failures = []
def check(label, cond):
    print(f"  {'✓' if cond else '✗'} {label}")
    if not cond:
        failures.append(label)


# ── density / keep-fraction scalars ──────────────────────────────────────────
print("[1] DARE density & Trim keep-fraction")
ds = [M._dare_density(n) for n in (1, 2, 4, 8, 16, 32)]
check("density within (0, 1]", all(0.0 < d <= 1.0 for d in ds))
check("density non-increasing with n", all(a >= b - 1e-9 for a, b in zip(ds, ds[1:])))
ks = [M._trim_keep_fraction(n) for n in (1, 2, 4, 8, 16, 32)]
check("keep-fraction within [0.5, 1.0]", all(0.5 - 1e-9 <= k <= 1.0 + 1e-9 for k in ks))
check("keep-fraction non-increasing with n", all(a >= b - 1e-9 for a, b in zip(ks, ks[1:])))

# ── text-encoder base detection ──────────────────────────────────────────────
print("[2] _is_te_base")
check("TE prefixes detected", all(M._is_te_base(p + "_block.0") for p in M._TE_KEY_PREFIXES))
check("non-TE rejected", not M._is_te_base("diffusion_model.blocks.0"))

# ── deterministic seed ───────────────────────────────────────────────────────
print("[3] _dare_seed determinism")
s1 = M._dare_seed(["a.safetensors", "b.safetensors"], [1.0, 0.5])
s2 = M._dare_seed(["a.safetensors", "b.safetensors"], [1.0, 0.5])
s3 = M._dare_seed(["a.safetensors", "b.safetensors"], [1.0, 0.6])
check("same inputs → same seed", s1 == s2)
check("different weights → different seed", s1 != s3)
check("seed is int", isinstance(s1, int))

# ── convention detection / key collection / mid guard ────────────────────────
print("[4] convention, factor keys, mid guard")
kohya = {"base.lora_up.weight": 0, "base.lora_down.weight": 0, "base.alpha": 0}
wan   = {"x.lora_B.weight": 0, "x.lora_A.weight": 0}
check("kohya convention detected", M._detect_convention(kohya) == (".lora_up.weight", ".lora_down.weight"))
check("WAN/FLUX convention detected", M._detect_convention(wan) == (".lora_B.weight", ".lora_A.weight"))
check("unknown layout → None", M._detect_convention({"foo.weight": 0, "bar.bias": 0}) is None)
trips = M._collect_factor_keys(kohya, (".lora_up.weight", ".lora_down.weight"))
check("factor keys: one triple with alpha",
      trips == [("base", "base.lora_up.weight", "base.lora_down.weight", "base.alpha")])
check("mid tensor detected", M._has_mid_tensor({"c.lora_mid.weight": 0}) is True)
check("no mid → False", M._has_mid_tensor(kohya) is False)

# ── DARE mask: rescale, channel/element behaviour, determinism ────────────────
print("[5] _dare_mask_apply")
B = torch.ones(6, 8)                                  # [out, rank]
# density outside (0,1) is a no-op
check("density=1.0 is a no-op", torch.equal(M._dare_mask_apply(B, 1.0, "channel", None), B))
check("density=0.0 is a no-op", torch.equal(M._dare_mask_apply(B, 0.0, "channel", None), B))

g = torch.Generator().manual_seed(123)
out_ch = M._dare_mask_apply(B.clone(), 0.5, "channel", g)
# every rank channel (column) is either entirely 0 or entirely original/density (=2.0)
col_ok = True
for r in range(8):
    col = out_ch[:, r]
    if not (torch.all(col == 0) or torch.allclose(col, torch.full_like(col, 2.0))):
        col_ok = False
check("channel mask drops WHOLE channels, survivors scaled 1/density", col_ok)

g2 = torch.Generator().manual_seed(123)
out_ch2 = M._dare_mask_apply(B.clone(), 0.5, "channel", g2)
check("channel mask deterministic (same seed)", torch.equal(out_ch, out_ch2))

g3 = torch.Generator().manual_seed(7)
out_el = M._dare_mask_apply(B.clone(), 0.5, "element", g3)
# every entry is either 0 or 2.0; element-wise mask should not zero whole columns identically
vals_ok = torch.all((out_el == 0) | torch.isclose(out_el, torch.tensor(2.0)))
check("element mask: each entry 0 or original/density", bool(vals_ok))

# ── Trim channel selection keeps the strongest ───────────────────────────────
print("[6] _trim_channel_indices keeps strongest channels")
rank = 8
Bt = torch.zeros(4, rank)
At = torch.zeros(rank, 4)
# make channel magnitude strictly increasing with index
for r in range(rank):
    Bt[:, r] = float(r + 1)
    At[r, :] = float(r + 1)
keep_half = M._trim_channel_indices(Bt, At, 0.5)
check("keep=0.5 selects about half", keep_half is not None and abs(keep_half.numel() - rank // 2) <= 1)
kept = set(keep_half.tolist())
strongest = set(range(rank - keep_half.numel(), rank))   # the top-magnitude indices
check("keep=0.5 keeps the strongest channels", kept == strongest)
keep_all = M._trim_channel_indices(Bt, At, 1.0)
check("keep=1.0 keeps all (None or full)", keep_all is None or keep_all.numel() == rank)

# ── RESOLVE (TIES sign-election): determinism, shape, sign agreement ──────────
print("[7] _resolve_sign_elect (TIES)")
torch.manual_seed(0)
out_dim, in_dim, r = 6, 5, 3
B0 = torch.randn(out_dim, r)
A0 = torch.randn(r, in_dim)
delta0 = B0 @ A0
# two identical LoRAs: every sign agrees → reconstructed delta ≈ delta0
res = M._resolve_sign_elect([B0.clone(), B0.clone()], [A0.clone(), A0.clone()],
                            out_dim, in_dim, seed=1, device="cpu", use_fp16=False)
check("identical pair → non-None result", res is not None)
if res is not None:
    Bo, Ao = res
    check("output shapes [out, R] / [R, in]", Bo.shape[0] == out_dim and Ao.shape[1] == in_dim)
    recon = (Bo.float() @ Ao.float())
    agree = (torch.sign(recon) == torch.sign(delta0)).float().mean().item()
    check("reconstructed delta keeps the agreed sign (>0.9)", agree > 0.9)
    res_again = M._resolve_sign_elect([B0.clone(), B0.clone()], [A0.clone(), A0.clone()],
                                      out_dim, in_dim, seed=1, device="cpu", use_fp16=False)
    check("RESOLVE deterministic (same seed → identical)",
          res_again is not None and torch.allclose(res_again[0], Bo) and torch.allclose(res_again[1], Ao))

# majority: 2 agree (+), 1 opposes (−) → elected sign follows the majority
res_maj = M._resolve_sign_elect([B0.clone(), B0.clone(), (-B0).clone()],
                                [A0.clone(), A0.clone(), A0.clone()],
                                out_dim, in_dim, seed=2, device="cpu", use_fp16=False)
check("majority case → non-None", res_maj is not None)
if res_maj is not None:
    rmaj = (res_maj[0].float() @ res_maj[1].float())
    agree_maj = (torch.sign(rmaj) == torch.sign(delta0)).float().mean().item()
    check("majority sign wins (>0.85)", agree_maj > 0.85)

# ── [8] version triple (maintenance) ─────────────────────────────────────────
print("[8] version triple v365")
def _read(*p):
    with open(os.path.join(HERE, "..", *p), encoding="utf-8") as f:
        return f.read()
check("__init__ banner v365", "Polyhedron Suite  v365" in _read("__init__.py"))
check("uls_compat PLUGIN_VERSION v365", 'PLUGIN_VERSION = "v365"' in _read("web", "js", "uls_compat.js"))

print("=" * 56)
if failures:
    print(f"RESULT: {len(failures)} CHECK(S) FAILED")
    sys.exit(1)
print("RESULT: ALL CHECKS PASS")
sys.exit(0)
