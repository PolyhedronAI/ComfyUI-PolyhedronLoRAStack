"""
Polyhedron Merge Analyzer — Backend (v266)
═══════════════════════════════════════════
Passive analysis node for the CONCAT / DARE / Resolve(TIES) merge. Reads the
Stack's `uls_config_out` (same wiring as the Inspector), shows the LIVE selected
LoRAs per group, and — on demand — measures how faithfully Resolve's low-rank
re-pack reproduces the true sign-elected delta (audit finding B-1).

No model patching, no merge-path changes — purely informational. It reuses the
SHIPPED merge functions from uls_stack_node.py, so the analysis reflects exactly
what the real merge does (no re-implementation, no drift).

Two depths (widget):
  • "Overview"      — instant: groups, LoRAs, weights, mode, Trim/Resolve state.
                       Flags which groups actually use Resolve with ≥2 LoRAs.
  • "Deep analysis" — slower (loads the LoRAs + truncated-SVD per layer): for
                       each Resolve group, energy retained at 1×/2×/4× sum_rank,
                       cosine, and the amplitude ratio ‖repacked‖/‖true‖.

Wire the STRING `report` into a "Show Text" node (exactly like the Inspector /
Token Counter). One Analyzer per Stack — drop two for the WAN HIGH/LOW dual setup.

v266: live console progress during the deep analysis (throttled `[PLS] ANALYZE`
lines, analogous to the v260 merge logging) + trim-aware report wording (audit
A-7): with Trim active all metrics measure the TRIMMED delta, so the report no
longer recommends an amplitude scalar — the B-1 render A/B showed washed-out
results are a Trim-strength issue, not a re-pack issue. Measurement math is
untouched; all numbers are identical to v264/v265.
"""

import json
import time
import hashlib

import folder_paths

# Reuse the SHIPPED merge helpers — single source of truth, zero drift.
from .uls_stack_node import (
    _sort_active_rows, _short_name,
    _detect_convention, _collect_factor_keys, _has_mid_tensor,
    _resolve_sign_elect, _trim_channel_indices, _trim_keep_fraction,
    _cached_load_torch_file, _resolve_pick_device, _check_interrupt, INTERRUPT_EXC,
)

# Re-pack ranks probed in the deep analysis: m × sum_rank (capped by min_dim).
_CANDIDATE_MULTIPLES = (1, 2, 4)


# ─── Deep analysis of ONE resolve group ────────────────────────────────────

def _true_resolved_delta(bs, as_, out, inn, torch):
    """Pass 1+2 of _resolve_sign_elect, BEFORE the SVD re-pack — i.e. the true
    full-rank resolved delta. Mirrors uls_stack_node.py:
    γ = sign(ΣΔ); disjoint mean num / den.clamp(min=1)."""
    deltas = [bs[i].reshape(out, bs[i].shape[1]).float() @ as_[i].reshape(as_[i].shape[0], inn).float()
              for i in range(len(bs))]
    Ssum = sum(deltas)
    gamma = torch.sign(Ssum)
    num = torch.zeros(out, inn, device=Ssum.device)
    den = torch.zeros(out, inn, device=Ssum.device)
    for W in deltas:
        agree = (torch.sign(W) == gamma) & (gamma != 0)
        num += torch.where(agree, W, torch.zeros_like(W))
        den += agree.float()
    return num / den.clamp(min=1.0)


def _analyze_group(names, weights, trim_keep, max_layers, dev, torch, label=""):
    """Measure Resolve re-pack fidelity for one group. Returns a result dict.
    Never raises for data issues — returns {'error': msg} instead.
    `label` is the group name, used only for the live console progress."""
    raw, nm, ws = [], [], []
    for name, w in zip(names, weights, strict=True):
        if not name or name == "None":
            continue
        path = folder_paths.get_full_path("loras", name)
        if not path:
            continue
        try:
            td = _cached_load_torch_file(path)
        except Exception:
            td = None
        if td:
            raw.append(td); nm.append(name); ws.append(float(w))
    if len(raw) < 2:
        return {"error": "fewer than 2 loadable LoRAs"}

    convs = [_detect_convention(td) for td in raw]
    keep = [i for i, (c, td) in enumerate(zip(convs, raw, strict=True)) if c is not None and not _has_mid_tensor(td)]
    if len({convs[i] for i in keep}) > 1:
        from collections import Counter
        majority = Counter(convs[i] for i in keep).most_common(1)[0][0]
        keep = [i for i in keep if convs[i] == majority]
    if len(keep) < 2:
        return {"error": "<2 compatible LoRAs after convention guards (production: SEQ)"}

    base_to_sources = {}
    for li in keep:
        for base, uk, dk, ak in _collect_factor_keys(raw[li], convs[li]):
            base_to_sources.setdefault(base, []).append((li, uk, dk, ak))
    multi = {b: s for b, s in base_to_sources.items() if len(s) >= 2}
    if not multi:
        return {"n_total": len(base_to_sources), "n_multi": 0, "rows": [],
                "note": "no shared layers with ≥2 sources — nothing to resolve"}

    def _proxy(sources):
        t = 0.0
        for (li, uk, dk, ak) in sources:
            t += float(raw[li][uk].float().norm()) * float(raw[li][dk].float().norm())
        return t
    ranked = sorted(multi.items(), key=lambda kv: _proxy(kv[1]), reverse=True)
    measure = ranked[: max(1, max_layers)]

    rows = []
    agg = {m: [0.0, 0.0] for m in _CANDIDATE_MULTIPLES}     # [Σ energie·gewicht, Σ gewicht]
    cur_rel_w = cur_cos_w = amp_w = wsum = 0.0

    _t_cum = 0.0
    for _li, (base, sources) in enumerate(measure, 1):
        _check_interrupt()                     # v265: red X (Cancel) aborts a long deep analysis
        _t0 = time.perf_counter()              # v266: per-layer wall time for the progress line
        bs, as_ = [], []
        out_dim = in_dim = None
        bad = False
        for (li, uk, dk, ak) in sources:
            B = raw[li][uk]; A = raw[li][dk]
            if B.ndim != 2 or A.ndim != 2:
                # Conv sources are skipped here: the deep analysis covers 2-D
                # (linear) layers only — WAN/FLUX LoRAs are all-linear. The
                # MERGE itself handles conv factors via trailing dims as usual.
                bad = True; break
            o = B.shape[0]; iflat = 1
            for s in A.shape[1:]:
                iflat *= s
            if out_dim is None:
                out_dim, in_dim = o, iflat
            elif o != out_dim or iflat != in_dim:
                bad = True; break
            rank = A.shape[0]
            alpha = None
            if ak is not None:
                try:
                    alpha = raw[li][ak].item() if hasattr(raw[li][ak], "item") else float(raw[li][ak])
                except Exception:
                    alpha = None
            scale = (alpha / rank) if (alpha is not None and rank > 0) else 1.0
            Bf = B.float() * (ws[li] * scale)
            Af = A.float()
            if trim_keep is not None:
                ki = _trim_channel_indices(Bf, Af, trim_keep)
                if ki is not None and ki.numel() < rank:
                    Bf = Bf.index_select(1, ki).contiguous()
                    Af = Af.index_select(0, ki).contiguous()
            bs.append(Bf.to(dev)); as_.append(Af.to(dev))
        if bad or len(bs) < 2:
            continue

        sum_rank = sum(b.shape[1] for b in bs)
        min_dim = min(out_dim, in_dim)
        Wt = _true_resolved_delta(bs, as_, out_dim, in_dim, torch)
        wnorm = float(Wt.norm())
        if wnorm < 1e-12:
            continue
        sv = torch.linalg.svdvals(Wt.float())
        e_tot = float((sv ** 2).sum())
        eff_rank = int((sv > 1e-6 * sv[0]).sum())
        energy_at = {}
        for m in _CANDIDATE_MULTIPLES:
            r = max(1, min(m * sum_rank, min_dim))
            energy_at[m] = (float((sv[:r] ** 2).sum()) / e_tot) if e_tot > 0 else 1.0
            agg[m][0] += energy_at[m] * wnorm
            agg[m][1] += wnorm
        res = _resolve_sign_elect(bs, as_, out_dim, in_dim, seed=0, device=dev, use_fp16=False)
        if res is None:
            continue
        Wa = (res[0].reshape(out_dim, sum_rank).float().to(dev) @
              res[1].reshape(sum_rank, in_dim).float().to(dev))
        rel = float((Wt - Wa).norm() / Wt.norm())
        cos = float(torch.nn.functional.cosine_similarity(Wt.flatten(), Wa.flatten(), dim=0))
        amp = float(Wa.norm() / Wt.norm())
        cur_rel_w += rel * wnorm; cur_cos_w += cos * wnorm; amp_w += amp * wnorm; wsum += wnorm
        rows.append((base, out_dim, in_dim, sum_rank, eff_rank, energy_at, cos, amp))
        # v266: throttled live progress (analogous to the v260 merge logging) —
        # the SVD loop used to print NOTHING until the very end. Diagnostic
        # only: clock reads + prints, no tensor math is touched. flush=True
        # pushes the line out DURING the loop instead of buffering it.
        _dt = time.perf_counter() - _t0
        _t_cum += _dt
        if len(rows) == 1 or len(rows) % 5 == 0 or _li == len(measure):
            print(f"[PLS]   ANALYZE [{label}] layer {len(rows)}/{len(measure)}  {dev}  "
                  f"layer={_dt:.2f}s  cum={_t_cum:.1f}s", flush=True)
        del Wt, Wa, sv, bs, as_
        if dev == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        return {"n_total": len(base_to_sources), "n_multi": len(multi), "rows": [],
                "note": "no measurable layers (cancellation / shape mismatch)"}

    out = {
        "n_total": len(base_to_sources), "n_multi": len(multi),
        "measured": len(rows), "rows": rows,
        "e": {m: (agg[m][0] / agg[m][1] if agg[m][1] else 1.0) for m in _CANDIDATE_MULTIPLES},
        "cos": cur_cos_w / wsum if wsum else 1.0,
        "rel": cur_rel_w / wsum if wsum else 0.0,
        "amp": amp_w / wsum if wsum else 1.0,
    }
    return out


# ─── Node ──────────────────────────────────────────────────────────────────

class ULSResolveInspector:
    """
    ⬡ Polyhedron Merge Analyzer

    Reads the Stack's `uls_config_out`, shows the live selected LoRAs per group
    and (on demand) measures Resolve's low-rank re-pack fidelity (audit B-1).
    Passive — no model patching. Wire `report` into a "Show Text" node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "uls_config_out": ("STRING", {
                    "default": '{"rows":[]}',
                    "multiline": False,
                    "forceInput": True,
                }),
                "analysis_depth": (["Overview", "Deep analysis"], {
                    "default": "Overview",
                    "tooltip": "Overview = instant (selection/modes only). "
                               "Deep analysis = loads the LoRAs + SVD per layer "
                               "(slower), measures Resolve fidelity.",
                }),
            },
            "optional": {
                "max_layers": ("INT", {
                    "default": 24, "min": 1, "max": 200, "step": 1,
                    "tooltip": "Deep analysis: how many of the largest conflict layers "
                               "are fully measured (speed/memory).",
                }),
                "device": (["auto", "cpu"], {
                    "default": "auto",
                    "tooltip": "auto = GPU if free (like the real Resolve path), "
                               "else CPU. 'cpu' forces CPU.",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT",          "FLOAT",                "BOOLEAN")
    RETURN_NAMES = ("report", "energy_1x_pct",  "amplitude_ratio",        "resolve_active")
    FUNCTION     = "analyze"
    CATEGORY     = "Polyhedron/Utils"
    OUTPUT_NODE  = False
    DESCRIPTION  = ("Analyzes the Stack's CONCAT/DARE/Resolve merge. "
                    "Shows the live-selected LoRAs; 'Deep analysis' measures the "
                    "Resolve re-pack fidelity (energy 1×/2×/4×, amplitude). "
                    "report → Show Text. One per Stack (HIGH/LOW).")

    def analyze(self, uls_config_out, analysis_depth="Overview",
                max_layers=24, device="auto"):
        try:
            cfg = json.loads(uls_config_out) if uls_config_out and uls_config_out.strip() else {}
        except Exception:
            cfg = {}

        rows          = cfg.get("rows", []) if isinstance(cfg.get("rows"), list) else []
        group_modes   = cfg.get("group_modes", {}) if isinstance(cfg.get("group_modes"), dict) else {}
        group_dare    = cfg.get("group_dare", {}) if isinstance(cfg.get("group_dare"), dict) else {}
        group_trim    = cfg.get("group_trim", {}) if isinstance(cfg.get("group_trim"), dict) else {}
        group_resolve = cfg.get("group_resolve", {}) if isinstance(cfg.get("group_resolve"), dict) else {}
        group_trim_amt= cfg.get("group_trim_amount", {}) if isinstance(cfg.get("group_trim_amount"), dict) else {}
        legacy_dare   = str(cfg.get("dare_variant", "channel")).lower()
        mult          = cfg.get("mult", 1.0)
        flat_mode     = bool(cfg.get("flatMode", False))
        custom_order  = cfg.get("groupOrder", {}) if isinstance(cfg.get("groupOrder"), dict) else {}

        try:
            mult_f = float(mult)
        except Exception:
            mult_f = 1.0

        ordered = _sort_active_rows(rows, flat_mode=flat_mode, custom_order=custom_order or None)

        L = ["═══ Polyhedron Merge Analyzer ═══",
             f"  Groups active : {len(ordered)}",
             f"  Global mult   : ×{mult_f:.2f}",
             "─────────────────────────────────"]

        if not ordered:
            L.append("  (no active LoRA rows — connect uls_config_out from the Stack)")
            return ("\n".join(L), 100.0, 1.0, False)

        # --- Overview: per group, mode + switches + LoRAs ---
        resolve_groups = []   # (group, names, weights, trim_keep_or_None)
        for group, grp_rows, grp_weights in ordered:
            n = len(grp_rows)
            mode = (group_modes.get(group) or "SEQ").upper()
            if mode not in ("SEQ", "CONCAT", "DARE"):
                mode = "SEQ"
            variant = str(group_dare.get(group, legacy_dare)).lower()
            if variant not in ("channel", "element"):
                variant = "channel"
            trim    = bool(group_trim.get(group, False))    and mode != "SEQ"
            resolve = bool(group_resolve.get(group, False)) and mode != "SEQ"
            trim_keep = None
            if trim:
                _ta = group_trim_amt.get(group, None)
                trim_keep = float(_ta) if isinstance(_ta, (int, float)) else _trim_keep_fraction(n)

            tag = mode
            if mode == "DARE":
                tag += f" [{variant[:4].upper()}]"
            if trim:
                tag += " +TRIM"
            if resolve:
                tag += " +RESOLVE"
            grp_label = f"[{group}]" if group != "—" else "[—]"
            flag = "   ← Resolve active" if (resolve and n >= 2) else ""
            L.append(f"  {grp_label} {tag}  ({n} LoRA{'s' if n != 1 else ''}){flag}")
            for r, w in zip(grp_rows, grp_weights, strict=True):
                L.append(f"     • {_short_name(r.get('name',''), 34):<34} ×{w}")

            if resolve and n >= 2:
                names = [r.get("name", "None") for r in grp_rows]
                resolve_groups.append((group, names, list(grp_weights), trim_keep))

        L.append("─────────────────────────────────")
        if resolve_groups:
            L.append(f"  Resolve groups with ≥2 LoRAs: {len(resolve_groups)}  "
                     f"({', '.join(g for g, *_ in resolve_groups)})")
        else:
            L.append("  No Resolve group with ≥2 LoRAs — nothing to measure "
                     "for CONCAT/DARE fidelity here.")

        # --- Overview only: done here ---
        if analysis_depth != "Deep analysis":
            if resolve_groups:
                L.append("  → For the fidelity measurement, set 'analysis_depth' to 'Deep analysis'.")
            L.append("─────────────────────────────────")
            return ("\n".join(L), 100.0, 1.0, bool(resolve_groups))

        # --- Deep analysis ---
        if not resolve_groups:
            L.append("─────────────────────────────────")
            return ("\n".join(L), 100.0, 1.0, False)

        try:
            import torch
        except Exception:
            L.append("  ✗ PyTorch unavailable — deep analysis not possible.")
            L.append("─────────────────────────────────")
            return ("\n".join(L), 100.0, 1.0, True)

        dev = "cpu" if device == "cpu" else _resolve_pick_device()
        L.append("")
        L.append("═══ Deep analysis: Resolve re-pack fidelity ═══")
        L.append(f"  Device: {dev} (fp32 for the measurement) | top {int(max_layers)} largest-contribution layers each")

        all_e1, all_amp, w_all = 0.0, 0.0, 0.0
        for group, names, weights, trim_keep in resolve_groups:
            L.append("  " + "─" * 33)
            L.append(f"  [{group}]  ({len([n for n in names if n and n!='None'])} LoRAs)"
                     f"{'  +TRIM keep=%.2f' % trim_keep if trim_keep is not None else ''}")
            try:
                res = _analyze_group(names, weights, trim_keep, int(max_layers), dev, torch, label=group)
            except INTERRUPT_EXC:
                raise                       # v265: a Cancel (red X) aborts the deep analysis
            except Exception as ex:
                L.append(f"     ✗ Analysis failed: {ex}")
                continue
            if "error" in res:
                L.append(f"     ⚠ {res['error']}")
                continue
            if not res.get("rows"):
                L.append(f"     Layers total {res.get('n_total','?')}, "
                         f"conflict-capable {res.get('n_multi',0)} — {res.get('note','')}")
                continue

            e = res["e"]
            L.append(f"     Layers: {res['n_total']} total, {res['n_multi']} with ≥2 sources, "
                     f"{res['measured']} measured")
            L.append(f"     Energy retained:  1×={e[1]*100:.0f}%   2×={e.get(2,e[1])*100:.0f}%   "
                     f"4×={e.get(4,e[1])*100:.0f}%   (1× = current rank)")
            L.append(f"     Cosine {res['cos']:.3f}  |  Amplitude (repacked/true) {res['amp']:.2f}")
            if trim_keep is not None:
                L.append(f"     ⚠ Trim is ON (keep={trim_keep:.2f}) — all metrics measure the TRIMMED")
                L.append("       delta. Washed-out renders usually mean too much Trim, not")
                L.append("       re-pack loss: calibrate Trim strength first (B-1 render A/B).")

            # energy-weighted aggregation across groups (roughly by layer count)
            wgt = float(res["measured"])
            all_e1 += e[1] * wgt; all_amp += res["amp"] * wgt; w_all += wgt

            # short per-group verdict
            gain4 = (e.get(4, e[1]) - e[1]) * 100
            if e[1] >= 0.90:
                L.append("     → Faithful. Higher rank not worth it.")
            elif gain4 < 8:
                L.append(f"     → Higher rank barely helps (+{gain4:.0f}pp at 4×); "
                         f"the residual lives in the detail tail.")
            else:
                L.append(f"     → Higher rank could help noticeably (+{gain4:.0f}pp at 4×).")

        L.append("─────────────────────────────────")
        agg_e1 = (all_e1 / w_all * 100) if w_all else 100.0
        agg_amp = (all_amp / w_all) if w_all else 1.0
        L.append(f"  Total across all Resolve groups: ~{agg_e1:.0f}% energy (1×), "
                 f"amplitude ~{agg_amp:.2f}")
        L.append("  Note: high cosine = direction preserved; the residual sits in the")
        L.append("  detail tail. Re-pack rank is settled (flat recovery curve, B-1).")
        if any(_tk is not None for _g, _n, _w, _tk in resolve_groups):
            L.append("  Trim was ON during this measurement — the analysis is trim-blind,")
            L.append("  so an amplitude scalar derived from these numbers would correct")
            L.append("  the wrong thing. Calibrate Trim strength first.")
        L.append("─────────────────────────────────")
        return ("\n".join(L), round(agg_e1, 1), round(agg_amp, 3), True)

    @classmethod
    def IS_CHANGED(cls, uls_config_out="", analysis_depth="Overview",
                   max_layers=24, device="auto", **kw):
        # Recompute only when selection/mode/depth change — so the expensive
        # deep analysis does not run on every queue, only when something changes.
        h = hashlib.sha1()
        for part in (str(uls_config_out), str(analysis_depth), str(max_layers), str(device)):
            h.update(part.encode("utf-8", "replace")); h.update(b"|")
        return h.hexdigest()
