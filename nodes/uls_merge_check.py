# -*- coding: ascii -*-
"""
uls_merge_check.py
==================
v988 -- does the Stack's merge give the SAME weights as SEQ? Measured on the
user's real files and the connected model, not argued.

Field, 22.09.2026: Frank's A/B (same stack, SEQ against CONCAT) did not look
the same. A picture difference on an 8-step distilled video model can come
from the seed, from bf16 rounding cascading through the sampler, or from a
real merge fault. This module separates the third from the first two.

THE TWO SIDES, each built by the code that really runs:
  reference -- every LoRA through Core's OWN loader path, exactly what SEQ
               runs (comfy.lora_convert.convert_lora + comfy.lora.load_lora
               with the model's key map), times its effective weight;
  merged    -- the Stack's _apply_concat_or_dare, handed a capture model that
               records the patches it would add.
Both come out as Core patches {target weight: adapter}. Per target the
difference ||merged - sum(reference)||_F is computed in FACTOR SPACE
(<B1 A1, B2 A2>_F = sum((B1^T B2) * (A1 A2^T)), the v980 identity) -- no
full-size delta is ever formed, so 200+ layers of a 15 GB model cost seconds.

Pure: torch and the patch dicts are passed in. The Analyzer drives it.
"""

# Below this the difference is storage rounding (the merged factors are kept
# in the LoRA's own dtype, bf16 -> ~3e-3 relative). Above it, a layer is named.
EQUAL_REL = 0.01


def adapter_factors(p, torch):
    """(B [out, r], A [r, in_flat], scale) of one Core LoRA patch, fp32, or
    None when the patch is not a plain LoRA (diff, set, LoKr ...). Reads the
    same fields Core's calculate_weight reads: alpha None -> scale 1.0."""
    w = getattr(p, "weights", None)
    if w is None and isinstance(p, tuple) and len(p) == 2 and p[0] == "lora":
        w = p[1]
    if not isinstance(w, (tuple, list)) or len(w) < 3:
        return None
    if len(w) > 3 and w[3] is not None:          # conv mid: not a plain product
        return None
    if len(w) > 4 and w[4] is not None:          # DoRA: not additive
        return None
    up, down, alpha = w[0], w[1], w[2]
    if up is None or down is None:
        return None
    B = up.float().flatten(1)
    A = down.float().flatten(1)
    if int(B.shape[1]) != int(A.shape[0]):
        return None
    scale = (float(alpha) / A.shape[0]) if alpha is not None else 1.0
    return B, A, scale


def _inner(B1, A1, B2, A2):
    return float(((B1.t() @ B2) * (A1 @ A2.t())).sum())


def compare(reference, merged, torch, dev="cpu", tick=None):
    """reference: {target: [(name, strength, patch), ...]} -- SEQ's patches.
    merged:    {target: [(strength, patch), ...]}          -- the merge's.
    Returns {rows: [(target, rel, e_ref)], e_ref, e_err, missing: {target:
    [names]}, extra: [targets], unreadable: [(target, why)]}."""
    rows, missing, extra, unreadable = [], {}, [], []
    e_ref_tot = 0.0
    e_err_tot = 0.0
    for t in sorted(set(reference) | set(merged)):
        if tick is not None:
            tick()
        Bs, As = [], []
        bad = None
        for name, s, p in reference.get(t, []):
            f = adapter_factors(p, torch)
            if f is None:
                bad = "reference patch of %s is not a plain LoRA" % name
                break
            B, A, sc = f
            Bs.append(B * (s * sc))
            As.append(A)
        Bm, Am = [], []
        for s, p in merged.get(t, []):
            f = adapter_factors(p, torch)
            if f is None:
                bad = "merged patch is not a plain LoRA"
                break
            B, A, sc = f
            Bm.append(B * (s * sc))
            Am.append(A)
        if bad:
            unreadable.append((t, bad))
            continue
        if not Bs:
            extra.append(t)
            continue
        if not Bm:
            missing[t] = [n for n, _s, _p in reference[t]]
            continue
        Br = torch.cat(Bs, 1).to(dev)
        Ar = torch.cat(As, 0).to(dev)
        Bg = torch.cat(Bm, 1).to(dev)
        Ag = torch.cat(Am, 0).to(dev)
        if Br.shape[0] != Bg.shape[0] or Ar.shape[1] != Ag.shape[1]:
            unreadable.append((t, "shape %s vs %s" % (tuple(Bg.shape[:1]) + tuple(Ag.shape[1:]),
                                                    tuple(Br.shape[:1]) + tuple(Ar.shape[1:]))))
            continue
        e_r = _inner(Br, Ar, Br, Ar)
        e_g = _inner(Bg, Ag, Bg, Ag)
        x = _inner(Br, Ar, Bg, Ag)
        err = max(0.0, e_r + e_g - 2.0 * x)
        e_ref_tot += e_r
        e_err_tot += err
        rows.append((t, (err / e_r) ** 0.5 if e_r > 0 else 0.0, e_r))
    return {"rows": rows, "e_ref": e_ref_tot, "e_err": e_err_tot,
            "missing": missing, "extra": extra, "unreadable": unreadable}


def group_rel(res):
    """sqrt(sum err^2 / sum ref^2) over the group -- one number."""
    return (res["e_err"] / res["e_ref"]) ** 0.5 if res["e_ref"] > 0 else 0.0


def check_lines(label, res, short, what="CONCAT", worst=5):
    """Report lines for one comparison."""
    L = []
    rel = group_rel(res)
    n = len(res["rows"])
    if not n and not res["missing"]:
        L.append("     %s: nothing to compare" % what)
        return L
    verdict = ("EQUAL to SEQ (difference is storage rounding)" if rel < EQUAL_REL
               and not res["missing"] and not res["extra"] and not res["unreadable"]
               else "DIFFERS from SEQ")
    L.append("     %s vs SEQ: %d weights, deviation %.2f%% -> %s"
             % (what, n, rel * 100.0, verdict))
    bad = sorted((r for r in res["rows"] if r[1] >= EQUAL_REL), key=lambda r: -r[1])
    for t, r, _e in bad[:worst]:
        L.append("       %6.2f%%  %s" % (r * 100.0, t))
    if len(bad) > worst:
        L.append("       ... %d more weights above %.0f%%" % (len(bad) - worst, EQUAL_REL * 100))
    for t, names in list(res["missing"].items())[:worst]:
        L.append("       MISSING in the merge: %s  (from %s)"
                 % (t, ", ".join(short(n, 24) for n in names)))
    if len(res["missing"]) > worst:
        L.append("       ... %d more weights missing" % (len(res["missing"]) - worst))
    for t in res["extra"][:worst]:
        L.append("       ONLY in the merge (SEQ does not patch it): %s" % t)
    for t, why in res["unreadable"][:worst]:
        L.append("       not compared: %s -- %s" % (t, why))
    return L
