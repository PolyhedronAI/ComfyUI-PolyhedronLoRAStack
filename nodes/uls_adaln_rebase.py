# -*- coding: utf-8 -*-
"""
uls_adaln_rebase.py
═══════════════════
v931 -- dense adaln LoRA factors, expressed in a curve-form model's coordinates.

THE MISMATCH
------------
Core builds an H3 block one of two ways (comfy/ldm/minimax/model.py):

    if self.use_adaln_curves:                 # pruned / curve-form checkpoints
        register_buffer("adaln_t_table", [rows, k])
    else:
        self.time_embedder = TimeEmbedder(...)

and its AdalnProj consumes `silu(t_emb)` in the dense case, the table row in
the curve case (`apply_silu = not use_adaln_curves`). A LoRA trained on the
dense model therefore has `adaln_proj.linear.lora_A` of width 2688 -- the full
time-embedding -- while a pruned model feeds that module an 8-wide curve
coordinate. The shapes do not meet. Core's loader logs a mismatch per module
and drops all ~50 of them: the run proceeds, quietly missing that part.

THE FIX
-------
A fitted basis expresses the dense signal in table coordinates:

    silu(t_emb(t))  ~=  c + V @ table(t)          V [2688, k], c [2688]

The module's delta is W = B @ A * (alpha/rank), applied to that signal:

    W @ silu(t_emb)  ~=  W @ c  +  (W @ V) @ table

The right-hand term is a rank-r delta again, with A' = A @ V of width k -- the
SAME B, the SAME alpha, only A narrowed. The left-hand term does not depend on
t at all: it is a constant added to the module's output, i.e. a BIAS delta.
Core carries that natively as `.diff_b`, and AdalnProj's linear does have a
bias, so nothing needs inventing.

Dropping the constant term is the mistake that looks like it works: the
t-dependent part is the visible one, and without the bias the module sits at a
wrong offset for every t. Both halves are produced here, and the guard drives a
model with and without it.

WHICH BASIS
-----------
Never by filename, always by the table itself: a basis carries the sha256 of
the adaln_t_table it was fitted against, and it is used only on a model whose
own table hashes the same. The two published trunks differ by 1.85% relative --
close enough that a fit against the wrong one succeeds numerically. The
checksum is the only thing that tells them apart.
"""

import hashlib
import os

BASIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         os.pardir, "assets", "adaln_basis")

ADALN_MODULE = "adaln_proj.linear"


def table_sha256(table):
    """Identity of a curve table: sha256 over its float32 bytes."""
    import numpy as np
    try:
        import torch
        if isinstance(table, torch.Tensor):
            table = table.detach().to(torch.float32).contiguous().cpu().numpy()
    except Exception:
        pass
    return hashlib.sha256(
        np.ascontiguousarray(table, dtype=np.float32).tobytes()).hexdigest()


def available_bases():
    """[(path, metadata)] for every basis shipped with the pack. Never raises."""
    out = []
    try:
        names = sorted(os.listdir(BASIS_DIR))
    except OSError:
        return out
    for name in names:
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(BASIS_DIR, name)
        try:
            from .uls_merge_policy import safetensors_metadata
        except ImportError:
            from uls_merge_policy import safetensors_metadata
        out.append((path, safetensors_metadata(path)))
    return out


def basis_for_table(table):
    """The shipped basis fitted against THIS table, or None.

    Matching is by checksum only. A model whose table matches nothing gets no
    basis and no rebase -- it is not a defect, it is an unknown build, and
    guessing a basis for it would be silently wrong.
    """
    want = table_sha256(table)
    for path, meta in available_bases():
        if meta.get("table_sha256") == want:
            return path, meta
    return None


def load_basis(path):
    """(c, V) as float32 torch tensors."""
    import comfy.utils
    sd = comfy.utils.load_torch_file(path, safe_load=True)
    return sd["c"], sd["V"]


def model_curve_table(model):
    """The model's adaln_t_table, or None when it is not a curve-form build.

    Reads the buffer off the diffusion model. Never raises: an unreadable model
    answers None, which means "no answer", never "dense".
    """
    for attr in ("model", "diffusion_model"):
        try:
            model = getattr(model, attr, model)
        except Exception:
            return None
    try:
        table = getattr(model, "adaln_t_table", None)
    except Exception:
        return None
    if table is None:
        return None
    try:
        if table.ndim != 2:
            return None
    except Exception:
        return None
    return table


def rebase_factors(A, B, c, V, scale):
    """(A_curve, bias_delta) for one dense adaln module.

    A       [r, dense]    the LoRA's down factor, dense-width
    B       [out, r]      its up factor, untouched
    c, V    the fitted basis
    scale   alpha/rank -- folded into the BIAS only. The t-dependent half keeps
            A/B/alpha as they are, so core applies the same scale to it that it
            would have applied to the dense pair. Folding scale into both would
            square it.
    """
    import torch
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    c32 = c.to(torch.float32)
    V32 = V.to(torch.float32)
    A_curve = A @ V32                      # [r, k]
    bias = (B @ (A @ c32)) * float(scale)  # [out]
    return A_curve, bias


def rebase_lora_adaln(sd, c, V, log=None):
    """Rewrite every dense adaln module in `sd` into curve coordinates.

    Returns (new dict, count). Modules that are already narrow, or that carry
    no matching factor pair, are left exactly as they are.
    """
    import torch

    dense = int(c.shape[0])
    k = int(V.shape[1])
    out = dict(sd)
    done = 0

    bases = set()
    for key in sd:
        if ADALN_MODULE + ".lora_A.weight" in key:
            bases.add(key[:-len(".lora_A.weight")])

    for base in sorted(bases):
        A = sd.get(base + ".lora_A.weight")
        B = sd.get(base + ".lora_B.weight")
        if A is None or B is None:
            continue
        if int(A.shape[1]) == k:
            continue                       # already in curve coordinates
        if int(A.shape[1]) != dense:
            if log:
                log("adaln %s has width %d, expected %d or %d -- left alone"
                    % (base, A.shape[1], dense, k))
            continue
        alpha = sd.get(base + ".alpha")
        rank = int(A.shape[0])
        scale = (float(alpha) / rank) if alpha is not None else 1.0
        A_curve, bias = rebase_factors(A, B, c, V, scale)
        out[base + ".lora_A.weight"] = A_curve.to(A.dtype)
        out[base + ".diff_b"] = bias.to(B.dtype)
        done += 1

    if log and done:
        log("adaln rebase: %d module(s) moved from %d-wide dense onto the "
            "model's %d-wide curve (each keeps its B and alpha; the constant "
            "term rides along as a bias delta)" % (done, dense, k))
    return out, done
