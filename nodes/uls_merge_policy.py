# -*- coding: utf-8 -*-
"""
uls_merge_policy.py
═══════════════════
v912 -- WHICH merge cleanups may run on WHICH model. Pure: no torch, no comfy,
so it is unit-testable in isolation (tests/test_v912_joint_merge_downgrade.py).

Kept out of uls_merge_math.py on purpose: that module is one of the ten anchor
files pinned since v598 (ANCHORS_baseline_v598.txt) and must stay byte-identical.
The math stays there; the policy that decides whether the math may run lives
here.
"""

JOINT_MERGE_REASON = ("joint audio/video model is CFG-distilled: merge noise is "
                      "not guarded by guidance")


def _joint_merge_downgrade(mode, resolve, trim, is_joint):
    """v912 -- decide how the merge cleanups apply on a JOINT model.

    DARE (random Bernoulli mask + 1/density rescale) and RESOLVE (TIES sign
    election + disjoint mean) were calibrated on WAN, where CFG > 1 pulls the
    sample back toward the prompt. MiniMax H3 samples at CFG 1 -- it is
    distilled -- so whatever noise the cleanup writes into the delta reaches the
    output unfiltered. Measured 04.09. against the SEQ sum (random rank-32
    sources, 2/4/8 LoRAs): DARE raised the delta energy by 2/9/25 % at 25/41/73 %
    deviation; RESOLVE shrank it to 0.69/0.46/0.32 of the sum (TIES averages,
    additive LoRAs vanish). CONCAT itself equals SEQ to bf16 rounding.

    Returns (mode, resolve, notes). On a plain model everything passes through
    unchanged; on a joint model DARE becomes CONCAT and RESOLVE is switched off,
    each with a note the caller prints. TRIM is deterministic (weakest channels
    only) and is kept, but named in a note so the console says what still runs.
    """
    mode = (mode or "SEQ").upper()
    notes = []
    if not is_joint:
        return mode, bool(resolve), notes
    if mode == "DARE":
        notes.append("DARE -> CONCAT on a joint model: " + JOINT_MERGE_REASON)
        mode = "CONCAT"
    if resolve:
        notes.append("RESOLVE switched off on a joint model: " + JOINT_MERGE_REASON)
        resolve = False
    if trim and mode != "SEQ":
        notes.append("TRIM stays (deterministic magnitude trim, no random mask)")
    return mode, bool(resolve), notes


# -- v913: HOW a group is applied -- baked into the weights, or as a hook -----
APPLY_MODES = ("auto", "patch", "bypass")
BYPASS_KEY = "polyhedron_bypass"      # injection AND attachment key on the ModelPatcher


def _apply_decision(apply, is_quantized, mode):
    """v913 -- decide whether this group goes BAKED (Core patches the weight:
    dequantize, add, requantize) or BYPASS (Core's forward hook, base weight
    untouched -- comfy/weight_adapter/bypass.py, since v0.32).

    `apply`: "auto" | "patch" | "bypass" (unknown -> "auto"). Auto picks bypass
    exactly when the target carries quantized weights: on int8/nvfp4 every bake
    resamples the layer's scale (ai-toolkit refuses to merge into such a model
    for that very reason), on a bf16 model the bake is exact and the hook only
    costs a matmul per step.

    Measured 04.09. (v915, Core's real QuantizedTensor, one 1024x1024 layer, a
    rank-32 delta at 5 % of |W|, against fp32 truth): the LoRA delta reaches the
    output under bypass to 0.4 %; baked into bf16 it loses 6.7 % of itself to
    rounding at W's scale, baked into int8-convrot 25 % to the requant. Total
    output error: bf16 base 0.33 % (bypass) vs 0.29 % (baked) -- a wash; int8
    base 0.93 % (bypass) vs 1.5 % (baked). Hence (v913-v916) auto = bypass on quantized
    targets only. One published counter-measurement exists (ComfyUI-VDN-H3:
    adapters co-trained with a linear branch, validated as MERGED weights,
    degrade under bypass on an 8-step DMD checkpoint) -- a co-adapted special
    case; the field A/B on Frank's own LoRAs decides the default for good.

    Under bypass SEQ folds into CONCAT: the hook holds ONE adapter per layer
    (BypassInjectionManager keeps one adapter per module key), and measured
    04.09. a CONCAT adapter through Core's bypass equals baked SEQ to 5e-7.
    DARE / TRIM / RESOLVE keep whatever the caller decided (v912 runs before).

    v917 -- THE FIELD DECIDED: auto = BAKED everywhere. Frank's A/B of 05.09.
    on MiniMax H3 int8-convrot (same noise seed, 3 turbo LoRAs in the Engine +
    11 image LoRAs in two Stack groups): baked visibly better, 26.5 s/step
    against 30.2 s under bypass (~10 % per step, the hooks), identical load
    time (step 1: 33.0 s bypass vs 34.0/34.6 s baked). Measurement 7's
    int8 edge for bypass (0.93 % vs 1.5 % total error) did not survive the
    field on distilled adapters. Bypass stays one click away on the pill.
    Not field-measured: a plain style LoRA on a quantized WAN -- that A/B is
    on the field list; if it goes the other way, a model-aware rule gets
    built from THAT measurement, not from this docstring. `is_quantized`
    is still taken so the note can say the hand switch exists.

    Pure. Returns (use_bypass, mode, notes).
    """
    apply = (apply or "auto").lower()
    if apply not in APPLY_MODES:
        apply = "auto"
    mode = (mode or "SEQ").upper()
    notes = []
    use_bypass = apply == "bypass"
    if use_bypass:
        notes.append("BYPASS: LoRAs run as a forward hook, base weights untouched")
        if mode == "SEQ":
            notes.append("SEQ folds into one CONCAT adapter per layer under bypass (same delta)")
            mode = "CONCAT"
    elif apply == "auto":
        # v917: baked was silent before -- a run's apply mode could only be
        # read from the ABSENCE of bypass lines (05.09.). Say it.
        notes.append("BAKED (auto): LoRAs merged into the weights"
                     + (" -- quantized target, the Apply pill offers BYPASS "
                        "as the hand switch" if is_quantized else ""))
    else:
        notes.append("BAKED: LoRAs merged into the weights")
    return use_bypass, mode, notes


# -- v914: keys the merged dict cannot carry -----------------------------------
# The CONCAT / DARE / bypass path rebuilds each layer's delta as up @ down and
# nothing else. Core's native loader (comfy/lora.py load_lora) reads more than
# that; whatever it reads that we drop would be applied SILENTLY WRONG:
#   dora_scale      DoRA -- the factors are trained under weight decomposition,
#                   applying them as plain LoRA is a different (larger) delta
#   .diff / .diff_b full-weight or bias deltas (norm / bias training)
#   .lora_B.bias / .lora_up.bias   bias half of a LoRA
#   .w_norm / .set_weight          norm scales (BFL / USO conversions)
#   .lora_mid.weight               CP/Tucker mid (already guarded since v250)
#   hada_* / lokr_* / oft_* / glora a1..b2   LyCORIS families (convention
#                   detection catches most; the mixed case slips through)
# One tuple, one test, one message -- so a new family lands in one place.
FOREIGN_SUFFIXES = (
    (".dora_scale", "DoRA scale"),
    (".diff", "full-weight diff"),
    (".diff_b", "bias diff"),
    (".lora_B.bias", "LoRA bias"),
    (".lora_up.bias", "LoRA bias"),
    (".w_norm", "norm scale"),
    (".set_weight", "set-weight"),
    (".lora_mid.weight", "conv mid"),
    (".hada_w1_a", "LoHa"),
    (".lokr_w1", "LoKr"),
    (".lokr_w1_a", "LoKr"),
    (".oft_diag", "OFT"),
    (".oft_blocks", "OFT"),
    (".a1.weight", "GLoRA"),
)


def _foreign_keys(td):
    """v914 -- which families of keys in this LoRA the merged dict would drop.

    Pure, suffix match only (no tensor is read). Returns a sorted list of
    family labels, empty when the dict is a plain up/down(/alpha) LoRA. A
    caller that sees a non-empty list routes the group to SEQ, where Core's
    native loader applies every key it understands.
    """
    found = set()
    for k in td:
        for suf, label in FOREIGN_SUFFIXES:
            if k.endswith(suf):
                found.add(label)
                break
    return sorted(found)


# ─── v929: payloads that are NOT plain LoRAs ─────────────────────────────────
#
# A LoRA loader reads factor keys (lora_A/lora_B, lora_up/lora_down, alpha) and
# ignores everything else. Some published files carry a SECOND half that is not
# expressible as factors -- dropping it leaves the file half-applied, running,
# plausible-looking and WRONG. _foreign_keys above catches the families Core's
# own loader still handles (DoRA, LyCORIS, diff/bias) and routes them to SEQ.
# This table is the other case: halves NO loader can carry. Each family is
# either HANDLED -- uls_stack_node has a whole-file path for it that applies
# both halves or refuses the file (v935: the PDD entry below) -- or REFUSED
# with the reason and the folder it belongs in. Never half.
#
# Match is on EXACT top-level tensor names, never a suffix: a real LoRA names
# every tensor `<module>.lora_A.weight`-style, so a bare `proj_out.weight`
# cannot collide with one. ALL listed names must be present -- one stray match
# never convicts a file.
#
# First entry (the case that prompted this, 10.09.): the MiniMax-H3 PDD Acc
# releases (alibaba-pai) ship a rank-64 trunk LoRA PLUS a Parallel Decoding
# Distillation head bank -- 32 per-interval copies of the final-layer video and
# audio projections. The trunk applies through any loader; the head bank does
# not, and without it the distillation is simply absent. v929 refused it; since
# v935 the Stack/Engine apply it whole (uls_stack_node._apply_pdd). The files
# stay in models/loras -- Frank's decision, 10.09.
PAYLOAD_SIGNATURES = (
    ("MiniMax-H3 PDD Acc",
     frozenset({"proj_out.weight", "proj_out.bias",
                "audio_proj_out.weight", "audio_proj_out.bias"}),
     "models/loras",
     "the parallel-decoding head bank rides beside the trunk and is applied "
     "with it by the Stack/Engine -- both halves or neither"),
)


def payload_family(keys):
    """v929 -- is this key set a known non-LoRA payload? Pure, no I/O.

    `keys` is an iterable of top-level tensor names. Returns
    (label, home_folder, reason) for the first signature whose names are ALL
    present, else None. Order of the table is the order of the answer.
    """
    have = set(keys or ())
    for label, names, home, reason in PAYLOAD_SIGNATURES:
        if names <= have:
            return (label, home, reason)
    return None


def safetensors_metadata(path):
    """v930 -- the __metadata__ block of a safetensors header, or {}.

    Header only. These files record rank and alpha there and carry no
    per-module .alpha tensor at all, so a converter that ignores the metadata
    has to invent a scale. Never raises; {} means "no answer", as with the
    names below.
    """
    import json
    import struct
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) != 8:
                return {}
            (n,) = struct.unpack("<Q", head)
            if n <= 0 or n > (64 << 20):
                return {}
            blob = fh.read(n)
            if len(blob) != n:
                return {}
        meta = json.loads(blob.decode("utf-8")).get("__metadata__")
    except Exception:
        return {}
    return meta if isinstance(meta, dict) else {}


def safetensors_header_names(path, _max_header=64 << 20):
    """v929 -- top-level tensor names from a safetensors HEADER, or None.

    Reads the 8-byte length prefix and the JSON header only -- no tensor data,
    so this is cheap enough to run on every file of every run. Returns None for
    anything it cannot read that way (a .pt/.ckpt pickle, a truncated file, an
    unreadable path): NONE MEANS "NO ANSWER", never "clean". Never raises -- a
    diagnostic must not be able to kill a run.
    """
    import json
    import struct
    try:
        with open(path, "rb") as fh:
            head = fh.read(8)
            if len(head) != 8:
                return None
            (n,) = struct.unpack("<Q", head)
            if n <= 0 or n > _max_header:
                return None
            blob = fh.read(n)
            if len(blob) != n:
                return None
        meta = json.loads(blob.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    return frozenset(k for k in meta if k != "__metadata__")


# ---------------------------------------------------------------------------
# v986 -- one layer, however its LoRA spells it
# ---------------------------------------------------------------------------
# Two LoRAs may name the SAME model weight differently: kohya writes
# `lora_unet_blocks_0_attn_qkv_proj` + .lora_up/.lora_down, the ComfyUI /
# Diffusers-converted form writes `diffusion_model.blocks.0.attn.qkv_proj` +
# .lora_B/.lora_A. Core maps both to one weight (comfy/lora.py,
# model_lora_keys_unet: key_map["lora_unet_" + X.replace(".", "_")] and
# key_map["diffusion_model." + X] point at the same key). Until v986 the
# merge collected layers by their SPELLING, so a mixed group could not be
# merged at all (it fell back to SEQ), and two spellings of one weight in a
# merged dict would have let the later entry overwrite the earlier in
# load_lora -- a LoRA silently lost.
#
# _canonical_base is that same equivalence, and nothing more: it strips the two
# prefixes Core treats as one and writes the rest with underscores. Every
# other spelling (text encoder, lycoris_, ...) passes through unchanged, i.e.
# is grouped exactly as before.

_CANON_PREFIXES = (("diffusion_model.", True), ("lora_unet_", False))


def _canonical_base(base):
    """A grouping key: equal for two bases that Core maps to one UNet weight."""
    for pre, dotted in _CANON_PREFIXES:
        if base.startswith(pre):
            rest = base[len(pre):]
            return "unet:" + (rest.replace(".", "_") if dotted else rest)
    return base


def _kohya_base(base):
    """The kohya spelling of a UNet base (`lora_unet_` + underscores) -- exact in
    this direction, unlike the reverse (an underscore in a module name cannot
    be told from a dot). Other bases are returned as they are."""
    c = _canonical_base(base)
    return "lora_unet_" + c[len("unet:"):] if c.startswith("unet:") else base


def _merged_naming(convs):
    """(output convention, spell) for a merged/baked layer set. One naming in
    the group: its own, spelling untouched -- the pre-v986 output, key for key.
    Mixed naming: kohya (.lora_up/.lora_down), every UNet layer spelled the
    kohya way, because only that direction translates without the model. The
    written dict then holds ONE naming; Core reads it like any kohya LoRA."""
    cs = [c for c in convs if c is not None]
    if len(set(cs)) <= 1:
        return (cs[0] if cs else None), (lambda b: b)
    k = next((c for c in cs if c[0].startswith(".lora_up")), cs[0])
    return k, _kohya_base
