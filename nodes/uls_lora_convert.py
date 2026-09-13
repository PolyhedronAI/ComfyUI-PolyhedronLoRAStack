# -*- coding: utf-8 -*-
"""
uls_lora_convert.py
═══════════════════
v930 -- LoRA key schemas core does not know, converted to ComfyUI's own.

Core ships `comfy/lora_convert.py`: 43 lines covering BFL Flux control, Wan Fun
and USO. Everything else it hands through untouched, and a LoRA whose modules
carry foreign names simply matches nothing -- it loads, applies to zero layers
and says almost nothing. MiniMax-H3 is trained in the Diffusers layout, so
EVERY H3 LoRA that has not been repacked by hand lands in that hole.

This module is the table core does not have. Adding a schema is one entry plus
its target map; nothing here is specific to one file.

WHY THE FUSIONS ARE WHAT THEY ARE (measured, not assumed)
---------------------------------------------------------
Core's Attention keeps ONE projection for q, k and v (`attn.qkv_proj`,
Linear(hidden, inner*3)), while Diffusers keeps three. Three rank-r LoRAs must
therefore become one rank-3r LoRA whose delta is their vertical stack:

    A_fused = [A_q ; A_k ; A_v]                     (3r, in)
    B_fused = blockdiag(B_q, B_k, B_v)              (3*out, 3r)
    alpha_fused = 3 * alpha        -- keeps alpha/rank, hence the scale, fixed

That B is block diagonal and A is concatenated is not a convention we picked:
it is the only arrangement whose product reproduces the three separate deltas,
and `_delta_of` in the guard proves it numerically rather than by inspection.

Core's MLP keeps ONE fc1 of width ffn*2 and splits it as

    gate, up = x.chunk(2, dim=-1)        (comfy/ops.py, _swiglu_eager)

i.e. GATE FIRST. Diffusers' feed-forward emits [value; gate]. The two halves of
fc1's OUTPUT rows must therefore be swapped. This one is a convention, not an
identity -- no arithmetic can detect getting it backwards, the model would just
be quietly wrong. It is pinned against two independent witnesses: core's own
source, and the metadata of the published Comfy-Org turbo LoRAs, which record
`swi_glu_mapping: Diffusers [value;gate] -> ComfyUI [gate;value]` and whose
shapes corroborate the qkv arrangement above.

WHAT IS NOT DONE HERE
---------------------
The adaln factors are carried across unchanged. On a curve-form (pruned)
checkpoint they still do not fit -- their input is the dense 2688-wide signal,
the model's is an 8-wide curve coordinate. Rebasing them is a separate concern
with its own inputs (the fitted basis) and lives in its own module.
"""

_KEY_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".lora_down", ".lora_up",
                 ".lora_A.weight", ".lora_B.weight", ".alpha")

# H3, Diffusers layout -> core. Order matters only for readability.
H3_SIMPLE_TARGETS = (
    ("attn.to_out.0", "attn.out_proj", None),
    ("ff.net.0.proj", "mlp.fc1", "swiglu"),
    ("ff.net.2", "mlp.fc2", None),
    ("adaln_proj.linear", "adaln_proj.linear", None),
)

H3_QKV_PARTS = ("attn.to_q", "attn.to_k", "attn.to_v")
H3_QKV_TARGET = "attn.qkv_proj"

H3_PREFIXES = (
    ("transformer_blocks.", "blocks."),
    ("token_refiner.refiner_blocks.", "token_refiner.blocks."),
)

COMFY_PREFIX = "diffusion_model."


def _split_key(key):
    """(module, suffix) for a factor key, or (None, None) if it is not one."""
    for suf in _KEY_SUFFIXES:
        if key.endswith(suf):
            return key[:-len(suf)], suf
    return None, None


def _canon_suffix(suf):
    """lora_down/lora_up (with or without .weight) -> A/B; alpha stays alpha."""
    if "lora_down" in suf or "lora_A" in suf:
        return "A"
    if "lora_up" in suf or "lora_B" in suf:
        return "B"
    return "alpha"


def detect_h3_diffusers(sd):
    """True when this dict is an H3 LoRA in the Diffusers layout.

    THREE conditions, all required, and the third is the load-bearing one.

    `transformer_blocks.N.attn.to_q` and `ff.net.0.proj` are NOT H3 markers --
    Flux, SD3, Hunyuan and most other Diffusers transformers spell their
    attention and feed-forward exactly the same way. Recognising on those alone
    would convert a foreign LoRA into H3 keys: it would apply to nothing, or
    worse, to the wrong thing. What distinguishes H3 is `adaln_proj`; the other
    architectures call that module `norm1.linear` or `adaLN_modulation`.

    The cost of this narrowness is stated plainly: an H3 LoRA that trains only
    attention and MLP, with no adaln at all, is NOT recognised and travels
    through unconverted. That is the right way round -- a missed conversion is
    visible (the LoRA does nothing), a wrong one is not.
    """
    has_prefix = False
    has_module = False
    has_adaln = False
    for k in sd:
        if k.startswith("transformer_blocks.") or k.startswith("token_refiner.refiner_blocks."):
            has_prefix = True
        if ".attn.to_q" in k or ".ff.net.0.proj" in k or ".attn.to_out.0" in k:
            has_module = True
        if ".adaln_proj." in k:
            has_adaln = True
        if has_prefix and has_module and has_adaln:
            return True
    return False


def _rename_prefix(module):
    for src, dst in H3_PREFIXES:
        if module.startswith(src):
            return dst + module[len(src):]
    return module


def _scale_of(metadata, rank):
    """alpha for a module, from the file's metadata. None when unknown.

    These files carry no per-module .alpha tensor at all: the ratio lives in
    the header (lora_alpha / lora_rank). Core needs a per-module alpha, so one
    is synthesised -- with the SAME ratio, never a guessed 1.0.
    """
    if not metadata:
        return None
    for key in ("lora_alpha", "alpha"):
        if key in metadata:
            try:
                return float(metadata[key])
            except (TypeError, ValueError):
                return None
    return None


def convert_h3_diffusers(sd, metadata=None, log=None):
    """Return (converted dict, notes). Never raises; unknown keys travel as-is.

    torch is imported lazily so the module stays importable without it.
    """
    import torch

    notes = []
    out = {}
    groups = {}          # module -> {"A":t, "B":t, "alpha":t}
    passthrough = {}

    for k, v in sd.items():
        module, suf = _split_key(k)
        if module is None:
            passthrough[k] = v
            continue
        groups.setdefault(module, {})[_canon_suffix(suf)] = v

    file_alpha = _scale_of(metadata, None)

    # --- qkv: gather the three parts per block ------------------------------
    qkv_bases = {}
    for module in groups:
        for part in H3_QKV_PARTS:
            if module.endswith("." + part) or module == part:
                base = module[:-len(part)].rstrip(".")
                qkv_bases.setdefault(base, {})[part] = module

    fused = 0
    for base, parts in sorted(qkv_bases.items()):
        if len(parts) != 3:
            notes.append("qkv group %s has %d of 3 parts -- left unfused"
                         % (base or "<root>", len(parts)))
            continue
        As, Bs, alphas, ranks = [], [], [], []
        ok = True
        for part in H3_QKV_PARTS:
            g = groups.get(parts[part], {})
            if "A" not in g or "B" not in g:
                ok = False
                break
            As.append(g["A"])
            Bs.append(g["B"])
            ranks.append(g["A"].shape[0])
            if "alpha" in g:
                alphas.append(float(g["alpha"]))
        if not ok:
            notes.append("qkv group %s is missing a factor -- left unfused" % base)
            continue
        if len(set(ranks)) != 1:
            notes.append("qkv group %s mixes ranks %s -- left unfused" % (base, ranks))
            continue

        r = ranks[0]
        a_cat = torch.cat(As, dim=0)                       # (3r, in)
        out_dims = [b.shape[0] for b in Bs]
        b_blk = torch.zeros((sum(out_dims), 3 * r), dtype=Bs[0].dtype)
        row = 0
        for i, b in enumerate(Bs):
            b_blk[row:row + b.shape[0], i * r:(i + 1) * r] = b.to(b_blk.dtype)
            row += b.shape[0]

        target = COMFY_PREFIX + _rename_prefix(base + "." + H3_QKV_TARGET if base
                                               else H3_QKV_TARGET)
        out[target + ".lora_A.weight"] = a_cat
        out[target + ".lora_B.weight"] = b_blk
        base_alpha = alphas[0] if alphas else file_alpha
        if base_alpha is not None:
            out[target + ".alpha"] = torch.tensor(base_alpha * 3.0)
        fused += 1
        for part in H3_QKV_PARTS:
            groups.pop(parts[part], None)

    # --- the simple one-to-one modules --------------------------------------
    simple = 0
    swapped = 0
    for module in sorted(groups):
        g = groups[module]
        mapped = None
        action = None
        for src, dst, act in H3_SIMPLE_TARGETS:
            if module.endswith("." + src) or module == src:
                mapped = module[:-len(src)] + dst if module != src else dst
                action = act
                break
        if mapped is None:
            for suf, t in (("A", ".lora_A.weight"), ("B", ".lora_B.weight"),
                           ("alpha", ".alpha")):
                if suf in g:
                    passthrough[module + t] = g[suf]
            continue

        a = g.get("A")
        b = g.get("B")
        if a is None or b is None:
            notes.append("%s is missing a factor -- carried through unchanged" % module)
            continue

        if action == "swiglu":
            rows = b.shape[0]
            if rows % 2:
                notes.append("%s has an odd output width (%d) -- NOT swapped"
                             % (module, rows))
            else:
                half = rows // 2
                b = torch.cat([b[half:], b[:half]], dim=0)
                swapped += 1

        target = COMFY_PREFIX + _rename_prefix(mapped)
        out[target + ".lora_A.weight"] = a
        out[target + ".lora_B.weight"] = b
        alpha = g.get("alpha")
        if alpha is not None:
            out[target + ".alpha"] = alpha
        elif file_alpha is not None:
            out[target + ".alpha"] = torch.tensor(file_alpha)
        simple += 1

    out.update(passthrough)
    notes.insert(0, "H3 Diffusers -> ComfyUI: %d qkv groups fused, %d modules mapped, "
                    "%d SwiGLU halves swapped" % (fused, simple, swapped))
    if log:
        for n in notes:
            log(n)
    return out, notes


# name -> (detector, converter). A new schema is one line.
LORA_CONVERTERS = (
    ("MiniMax-H3 (Diffusers)", detect_h3_diffusers, convert_h3_diffusers),
)


def convert_foreign_lora(sd, metadata=None, log=None):
    """Run the first converter that recognises this dict. Returns (sd, name).

    Unrecognised dicts come back untouched with name None -- this is a hand
    through, never a guess.
    """
    for name, detect, convert in LORA_CONVERTERS:
        try:
            if not detect(sd):
                continue
        except Exception:
            continue
        try:
            out, _notes = convert(sd, metadata=metadata, log=log)
            return out, name
        except Exception as ex:
            if log:
                log("%s conversion failed (%s) -- using the raw dict"
                    % (name, type(ex).__name__))
            return sd, None
    return sd, None
