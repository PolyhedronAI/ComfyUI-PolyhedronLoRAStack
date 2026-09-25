# -*- coding: ascii -*-
"""
uls_overlap_math.py
===================
v980 -- how the LoRAs of a stack relate to each other, measured on the real
factors. Pure: torch only (passed in), no comfy, no folder_paths, so the whole
measurement is guard-driven in isolation (tests/test_v980_overlap.py).

WHY. On a joint, CFG-distilled model (MiniMax H3) the Stack turns DARE into
CONCAT and switches RESOLVE off (v912, measured). What is left is the plain
sum of every LoRA's delta. Whether 19 LoRAs work together or fight is then a
property of the deltas themselves -- and that can be measured instead of
guessed from renders.

WHAT. For every pair of LoRAs i, j the Frobenius inner product of their deltas,
summed over all layers:

    G[i, j] = sum_layers < w_i s_i B_i A_i , w_j s_j B_j A_j >_F

computed EXACTLY in factor space, never forming a full delta:

    < B1 A1 , B2 A2 >_F = trace(A1^T B1^T B2 A2) = sum( (B1^T B2) * (A1 A2^T) )

With all sources of one layer stacked along the rank axis this is one
(R x R) Gram product per layer. From G:

    energy of LoRA i          G[i, i]           (= ||delta_i||^2 over all layers)
    cosine of i and j         G[i, j] / sqrt(G[i, i] G[j, j])
    energy of the whole sum   sum(G)            (= ||sum_i delta_i||^2)
    interaction ratio         sum(G) / trace(G) (1.0 = independent,
                                                 > 1 reinforce, < 1 cancel)
    net contribution of i     sum_j G[i, j] / sum(G)

Weights and alpha/rank are folded in exactly as the merge folds them
(uls_stack_node._apply_concat_or_dare): text-encoder layers scale with the
CLIP weight, everything else with the model weight; a missing alpha means
scale 1.0. Layers whose shapes disagree with the first source are skipped and
counted, as the merge skips them.
"""

import re

try:  # package import (ComfyUI) / flat import (guards, tools)
    from .uls_merge_math import _collect_factor_keys, _is_te_base
    from .uls_merge_policy import _canonical_base          # v986
except ImportError:
    from uls_merge_math import _collect_factor_keys, _is_te_base
    from uls_merge_policy import _canonical_base           # v986


# -- v980: the Analyzer's depth list and tips live HERE (pure module) so the
# legacy node and its V3 form read ONE list without the V3 file importing the
# legacy backend at schema time.
DEPTH_OVERLAP = "Overlap & energy"
DEPTH_MERGE = "Merge check (vs SEQ)"
DEPTHS = ["Overview", "Deep analysis", DEPTH_OVERLAP, DEPTH_MERGE]   # APPEND-ONLY: saved workflows store the name
DEPTH_TIP = ("Overview = instant (selection/modes only). "
             "Deep analysis = loads the LoRAs + SVD per layer (slower), measures "
             "Resolve fidelity. Overlap & energy = loads every active LoRA and "
             "measures, exactly and in factor space, how much each one adds and "
             "which ones pull the same way or against each other. Merge check "
             "= builds every group twice on the connected model -- each LoRA "
             "through ComfyUI's own loader (what SEQ runs) and the Stack's "
             "merge -- and says whether the weights are the same, per weight "
             "if not. Needs the model input.")
MODEL_TIP = ("Optional: the model the Stack patches. Connected, the report shows "
             "what really runs on it -- on a joint audio/video model (MiniMax H3) "
             "the Stack turns DARE into CONCAT and switches RESOLVE off.")


_BLOCK_RE = re.compile(r"(?:^|[._])blocks[._](\d+)(?=[._])")


def module_kind(base):
    """attn / mlp / adaln / refiner / other -- the coarse role of a layer, read
    from its key. Works for ComfyUI keys ('diffusion_model.blocks.3.attn.
    qkv_proj') and kohya keys ('lora_unet_blocks_3_attn_qkv_proj')."""
    b = base.lower()
    if "token_refiner" in b or "refiner" in b:
        return "refiner"
    if "adaln" in b:
        return "adaln"
    if "attn" in b or "qkv" in b or "out_proj" in b:
        return "attn"
    if "mlp" in b or "fc1" in b or "fc2" in b or "ffn" in b:
        return "mlp"
    return "other"


def block_index(base):
    """The DiT block number a layer key sits in, or None."""
    m = _BLOCK_RE.search(base)
    return int(m.group(1)) if m else None


def lora_factors(td, conv, weight, clip_weight, base, keys, torch):
    """(B2 [out, r], A2 [r, in_flat]) in fp32 with w * alpha/rank folded into
    B, or None when the layer cannot be read. `keys` = (up, down, alpha)."""
    uk, dk, ak = keys
    try:
        B = td[uk]
        A = td[dk]
    except Exception:
        return None
    rank = int(A.shape[0])
    if rank <= 0 or int(B.shape[1]) != rank:
        return None
    alpha = None
    if ak is not None:
        try:
            alpha = td[ak].item() if hasattr(td[ak], "item") else float(td[ak])
        except Exception:
            alpha = None
    scale = float(alpha / rank) if alpha is not None else 1.0
    w = clip_weight if _is_te_base(base) else weight
    B2 = B.float().reshape(B.shape[0], rank) * (float(w) * scale)
    A2 = A.float().reshape(rank, -1)
    return B2, A2


def measure_overlap(loras, torch, dev="cpu", tick=None):
    """loras: list of dicts {td, conv, weight, clip_weight}. Returns
    {G (n x n list of floats), layers [n], blocks [n] (sorted ints),
     kinds [n] (dict kind -> count), n_bases, skipped_shape}.
    `tick` (optional) is called once per layer -- the caller hangs its
    interrupt check there."""
    n = len(loras)
    G = torch.zeros((n, n), dtype=torch.float64)
    layers = [0] * n
    blocks = [set() for _ in range(n)]
    kinds = [dict() for _ in range(n)]
    per_base = {}
    # v986: each LoRA in its own key naming, layers grouped by the WEIGHT they
    # patch (_canonical_base: Core's equivalence of the kohya and the
    # diffusion_model. spelling) -- the grouping the merge itself uses.
    for i, lo in enumerate(loras):
        for base, uk, dk, ak in _collect_factor_keys(lo["td"], lo["conv"]):
            per_base.setdefault(_canonical_base(base), []).append((i, base, (uk, dk, ak)))
    skipped = 0
    for _ck, srcs in per_base.items():
        if tick is not None:
            tick()
        Bs, As, owners = [], [], []
        shape = None
        for i, base, keys in srcs:
            lo = loras[i]
            f = lora_factors(lo["td"], lo["conv"], lo["weight"], lo["clip_weight"],
                             base, keys, torch)
            if f is None:
                continue
            B2, A2 = f
            sh = (int(B2.shape[0]), int(A2.shape[1]))
            if shape is None:
                shape = sh
            elif sh != shape:
                skipped += 1
                continue
            Bs.append(B2)
            As.append(A2)
            owners.append((i, int(A2.shape[0])))
            layers[i] += 1
            bi = block_index(base)
            if bi is not None:
                blocks[i].add(bi)
            k = module_kind(base)
            kinds[i][k] = kinds[i].get(k, 0) + 1
        if not Bs:
            continue
        B = torch.cat(Bs, dim=1).to(dev)
        A = torch.cat(As, dim=0).to(dev)
        H = (B.t() @ B) * (A @ A.t())
        H = H.double().cpu()
        offs = []
        o = 0
        for i, r in owners:
            offs.append((i, o, o + r))
            o += r
        for (i, a0, a1) in offs:
            for (j, b0, b1) in offs:
                G[i, j] += H[a0:a1, b0:b1].sum()
    return {
        "G": G.tolist(),
        "layers": layers,
        "blocks": [sorted(b) for b in blocks],
        "kinds": kinds,
        "n_bases": len(per_base),
        "skipped_shape": skipped,
    }


def summarize(G, groups):
    """Pure-python figures from G. groups: list (len n) of group labels.
    Returns {energy [n], share [n], net [n], total, indep, ratio,
             pairs [(i, j, cos)] sorted by |cos| desc,
             group {label: {n, energy, indep, ratio}}}."""
    n = len(G)
    energy = [float(G[i][i]) for i in range(n)]
    indep = sum(energy)
    total = sum(float(G[i][j]) for i in range(n) for j in range(n))
    share = [(e / indep) if indep > 0 else 0.0 for e in energy]
    net = [(sum(float(G[i][j]) for j in range(n)) / total) if total > 0 else 0.0
           for i in range(n)]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = (energy[i] * energy[j]) ** 0.5
            cos = float(G[i][j]) / d if d > 0 else 0.0
            pairs.append((i, j, cos))
    pairs.sort(key=lambda p: -abs(p[2]))
    grp = {}
    for label in dict.fromkeys(groups):
        idx = [i for i in range(n) if groups[i] == label]
        ge = sum(float(G[i][j]) for i in idx for j in idx)
        gi = sum(energy[i] for i in idx)
        grp[label] = {"n": len(idx), "energy": ge, "indep": gi,
                      "ratio": (ge / gi) if gi > 0 else 1.0}
    return {"energy": energy, "share": share, "net": net, "total": total,
            "indep": indep, "ratio": (total / indep) if indep > 0 else 1.0,
            "pairs": pairs, "group": grp}


# -- report --------------------------------------------------------------------
COS_SAME = 0.30       # guide values for the wording, not physics
COS_AGAINST = -0.30
QUIET_SHARE = 0.02


def _span(bl):
    if not bl:
        return "-"
    if len(bl) == bl[-1] - bl[0] + 1:
        return "%d-%d" % (bl[0], bl[-1]) if len(bl) > 1 else str(bl[0])
    return "%d-%d (%d)" % (bl[0], bl[-1], len(bl))


def _kinds(k):
    order = ("attn", "mlp", "adaln", "refiner", "other")
    return " ".join("%s %d" % (x, k[x]) for x in order if k.get(x))


def overlap_lines(names, groups, weights, meas, summ, short, top=8):
    """The report block. `short(name, width)` shortens a LoRA name."""
    L = []
    n = len(names)
    L.append("  Stack as a whole: interaction ratio %.2f  "
             "(1.00 = independent, > 1 reinforce, < 1 cancel)" % summ["ratio"])
    L.append("  Layers read: %d, skipped for a shape mismatch: %d"
             % (meas["n_bases"], meas["skipped_shape"]))
    L.append("")
    L.append("  Per LoRA  (share = own energy, net = incl. its overlaps):")
    order = sorted(range(n), key=lambda i: -summ["share"][i])
    for i in order:
        flag = ""
        if summ["share"][i] < QUIET_SHARE:
            flag = "  <- quiet"
        if summ["net"][i] < 0:
            flag += "  <- works against the rest"
        L.append("   %-30s [%s] x%-5g share %5.1f%%  net %+6.1f%%%s"
                 % (short(names[i], 30), groups[i], weights[i],
                    100 * summ["share"][i], 100 * summ["net"][i], flag))
        L.append("      layers %d, blocks %s, %s"
                 % (meas["layers"][i], _span(meas["blocks"][i]),
                    _kinds(meas["kinds"][i]) or "-"))
    L.append("")
    L.append("  Per group:")
    for label, g in summ["group"].items():
        L.append("   [%s] %d LoRA%s  ratio %.2f  (%.1f%% of the stack's own energy)"
                 % (label, g["n"], "" if g["n"] == 1 else "s", g["ratio"],
                    100 * (g["indep"] / summ["indep"]) if summ["indep"] > 0 else 0.0))
    L.append("")
    L.append("  Strongest pairs (cosine of the full deltas):")
    shown = 0
    for i, j, c in summ["pairs"]:
        if shown >= top:
            break
        if c >= COS_SAME:
            what = "pull the same way -- one of them may be enough"
        elif c <= COS_AGAINST:
            what = "push against each other"
        else:
            what = "mostly independent"
        L.append("   %+.2f  %s <-> %s  (%s)"
                 % (c, short(names[i], 24), short(names[j], 24), what))
        shown += 1
    if not summ["pairs"]:
        L.append("   (one LoRA -- no pairs)")
    quiet = [names[i] for i in range(n) if summ["share"][i] < QUIET_SHARE]
    if quiet:
        L.append("")
        L.append("  Quiet LoRAs (< %d%% of the stack's energy): %d -- candidates for an"
                 % (int(QUIET_SHARE * 100), len(quiet)))
        L.append("  A/B without them (same seed). The measure is energy, not effect:")
        L.append("  a small, well-aimed delta can still matter.")
    return L


# -- v983: the energy cap -------------------------------------------------------
def cap_factor(G):
    """(factor, ratio) of the overlap-neutral energy cap for one group's G.

    ratio = sum(G) / trace(G): 1.0 when the LoRAs are independent, > 1 when
    they pull the same way and pile up, < 1 when they cancel. The cap scales
    the WHOLE group by sqrt(1 / ratio) exactly when ratio > 1, so that
    afterwards

        ||sum_i s delta_i||^2 = s^2 sum(G) = trace(G)

    -- the group carries the energy its LoRAs would have if they were
    independent. It never boosts (ratio <= 1 -> factor 1.0), and it needs no
    tuned threshold: the reference is the group's own measured energy.
    One scalar for the group, so it works in SEQ, CONCAT and bypass alike and
    keeps the ratio between the group's LoRAs as set."""
    n = len(G)
    indep = sum(float(G[i][i]) for i in range(n))
    total = sum(float(G[i][j]) for i in range(n) for j in range(n))
    if indep <= 0 or total <= 0:
        return 1.0, 1.0
    ratio = total / indep
    if ratio <= 1.0:
        return 1.0, ratio
    return (1.0 / ratio) ** 0.5, ratio


# ---------------------------------------------------------------------------
# v985 -- names a reader can tell apart
# ---------------------------------------------------------------------------
# The report cut every name to its first 24..34 characters. Frank's LoRAs are
# named `polyhedron_minimax_h3_image_lora__<what>`, so every line kept the
# shared 34 characters and lost exactly the part that says WHICH LoRA it is
# (field report 21.09.2026: 'polyhedron_minimax_h3_im <-> polyhedron_minimax_h3_im').
# Now each name drops the longest prefix it shares with ANOTHER name in the
# same report (cut back to a separator), and keeps the rest. Names that share
# nothing (Minimax H3_Motion_Repair) stay whole. Two names that would read the
# same after the cut get less cut until they differ. Display only -- no
# number changes.

ELLIPSIS = "\u2026"
_SEPS = "_- ."
MIN_STRIP = 8            # shorter shared prefixes are not worth an ellipsis


def _stem(name):
    s = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    return s[:-len(".safetensors")] if s.endswith(".safetensors") else s


def _shared_cut(a, b):
    """Length of the common prefix of a and b, cut back to just after a
    separator, and never the whole of a."""
    m = min(len(a), len(b))
    n = 0
    while n < m and a[n] == b[n]:
        n += 1
    if n >= len(a):
        n = len(a) - 1
    while n > 0 and a[n - 1] not in _SEPS:
        n -= 1
    return n


def _back_off(s, k):
    """The previous separator boundary before k (0 = no cut)."""
    k -= 1
    while k > 0 and s[k - 1] not in _SEPS:
        k -= 1
    return max(0, k)


def _rest(s, k):
    """What stays after cutting k characters: leading separators dropped
    too (`lora__liza` -> `liza`, not `_liza`), never empty."""
    r = s[k:]
    return r.lstrip(_SEPS) or r


def distinct_names(names):
    """{stem: display} for every name in the report."""
    stems = []
    for nm in names:
        st = _stem(nm)
        if st not in stems:
            stems.append(st)
    cut = {}
    for s in stems:
        best = max((_shared_cut(s, o) for o in stems if o != s), default=0)
        cut[s] = best if best >= MIN_STRIP else 0
    for _ in range(64):                      # collisions: cut less
        seen = {}
        for s in stems:
            seen.setdefault(_rest(s, cut[s]), []).append(s)
        clash = [grp for grp in seen.values() if len(grp) > 1]
        if not clash:
            break
        for grp in clash:
            for s in grp:
                cut[s] = _back_off(s, cut[s]) if cut[s] else 0
                if cut[s] < MIN_STRIP:
                    cut[s] = 0
    return {s: (ELLIPSIS + _rest(s, cut[s])) if cut[s] else s for s in stems}


def name_shortener(names):
    """short(name, width) for one report: the distinct part, then cut to
    width (an ellipsis at the end marks a cut)."""
    disp = distinct_names(names)

    def short(name, width=34):
        d = disp.get(_stem(name))
        if d is None:
            d = _stem(name)
        width = max(4, int(width))
        return d if len(d) <= width else d[:width - 1] + ELLIPSIS
    return short
