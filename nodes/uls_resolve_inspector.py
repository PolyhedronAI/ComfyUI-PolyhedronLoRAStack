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
    _convert_lora_like_core,
    _group_effective, _cap_text,         # v981 strength + v983 energy cap
    _concat_blocker, _convention_label,  # v985 the Stack's own SEQ-fallback decision
    _naming_mix, _canonical_base,        # v986 mixed key naming merges per layer
    _apply_concat_or_dare,               # v988 the merge itself, for the merge check
)
# v1010: the green bar over the measured layers -- its own import, so a
# harness that stubs uls_stack_node keeps working (silent stand-in then).
try:
    from .uls_stack_node import _merge_progress
except Exception:
    class _SilentBar:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def tick(self, n=1):
            pass

    def _merge_progress(*_a, **_k):
        return _SilentBar()
from . import uls_merge_check as _MC     # v988
# v980: what the Stack REALLY runs on the connected model, and the overlap depth.
from .uls_merge_policy import _joint_merge_downgrade
from . import uls_overlap_math as _OV

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
    # v986: mixed key naming is merged per layer (each LoRA in its own naming,
    # layers met by _canonical_base) -- measured the same way here.
    if len(keep) < 2:
        return {"error": "<2 compatible LoRAs after convention guards (production: SEQ)"}

    base_to_sources = {}
    for li in keep:
        for base, uk, dk, ak in _collect_factor_keys(raw[li], convs[li]):
            base_to_sources.setdefault(_canonical_base(base), []).append((li, uk, dk, ak))
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
    # v1010: the green bar over the measured layers (the console has its own
    # per-layer line since v266, so the instrument stays quiet).
    _aprog = _merge_progress(len(measure), "analyze", label="Merge Analyzer", quiet=True)
    _aprog.__enter__()
    for _li, (base, sources) in enumerate(measure, 1):
        _check_interrupt()                     # v265: red X (Cancel) aborts a long deep analysis
        _aprog.tick()
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

    _aprog.__exit__(None, None, None)
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

# ─── v980: depth list, model line, overlap block ───────────────────────────

DEPTH_OVERLAP, DEPTHS, DEPTH_TIP, MODEL_TIP = (    # v980: one home, uls_overlap_math
    _OV.DEPTH_OVERLAP, _OV.DEPTHS, _OV.DEPTH_TIP, _OV.MODEL_TIP)
DEPTH_MERGE = _OV.DEPTH_MERGE                      # v988
JOINT_SHARED_NOTE = (
    "  Joint model: video AND audio tokens run through the SAME DiT blocks,\n"
    "  so every block LoRA also acts on the audio stream -- image LoRAs too.")


def _model_is_joint(model):
    """True / False from the Stack's own joint probe, None when no model is
    connected or the probe cannot answer."""
    if model is None:
        return None
    try:
        from .ph_joint_probe import _joint_latent_parts
        return _joint_latent_parts(model) >= 2
    except Exception:
        return None


def _model_line(is_joint):
    if is_joint is None:
        return "not connected -- modes shown as set, not as run"
    if is_joint:
        return "joint audio/video (DARE runs as CONCAT, RESOLVE off)"
    return "plain -- modes run as set"


def _merge_path(grp_rows, ws, cs):
    """v985 -- what the Stack's CONCAT/DARE path will do with this group:
    load the LoRAs exactly as _apply_concat_or_dare loads them (same skip of
    zero weights and missing files, cache, Core's conversion) and ask
    _concat_blocker, the function the Stack itself decides with.
    Returns (blocker_or_None, n_loadable, [missing names], naming mix)."""
    names, raw, missing = [], [], []
    for r, w, wc in zip(grp_rows, ws, cs, strict=True):
        name = r.get("name", "None")
        if (abs(w) < 1e-6 and abs(wc) < 1e-6) or not name or name == "None":
            continue
        path = folder_paths.get_full_path("loras", name)
        if not path:
            missing.append(name)
            continue
        try:
            td = _cached_load_torch_file(path)
            td = _convert_lora_like_core(td, path) if td else td
        except INTERRUPT_EXC:
            raise
        except Exception:
            missing.append(name)
            continue
        if td:
            names.append(name)
            raw.append(td)
    if len(raw) < 2:
        return None, len(raw), missing, []
    return _concat_blocker(names, raw), len(raw), missing, _naming_mix(names, raw)


def _merge_path_lines(blk, n_ok, missing, short, mix=()):
    """(row marks {name: label}, lines) for a merge group: why it runs SEQ,
    or -- v986 -- that its mixed key naming is merged per layer."""
    marks, L = {}, []
    if mix and not blk and n_ok >= 2:
        from collections import Counter
        cnt = Counter(lab for _n, lab in mix)
        major = cnt.most_common(1)[0][0]
        marks = {n: lab.split(" ")[0] for n, lab in mix if lab != major}
        L.append("     key naming mixed (" +
                 ", ".join(f"{c} × {lab.split(' ')[0]}" for lab, c in cnt.most_common()) +
                 ") -- merged per layer, each LoRA in its own naming")
    if n_ok < 2:
        L.append("     ⚠ fewer than 2 loadable LoRAs -- the Stack runs this group as SEQ")
    elif blk:
        kind, items = blk
        if kind == "unrecognised":
            marks = {n: "not plain" for n, _l in items}
            L.append(f"     ⚠ {len(items)} LoRA(s) in no known LoRA layout (LyCORIS/LoHA/LoKr?)")
        else:
            marks = {n: "keys" for n, _l in items}
            L.append(f"     ⚠ {len(items)} LoRA(s) carry keys a merge cannot hold:")
            for n, lab in items:
                L.append(f"       {short(n, 30)}: {lab}")
        L.append("       -> the Stack runs this group as SEQ: no merge, no TRIM,")
        L.append("          no energy cap, no bake. Marked rows are the odd ones out;")
        L.append("          give them their own group and the rest merges.")
    for n in missing:
        L.append(f"     ⚠ not loadable: {short(n, 30)}")
    return marks, L


class _CaptureModel:
    """v988 -- stands in for the ModelPatcher in the merge check: the real
    model's key map, and a record of the patches instead of patching.
    add_patches keeps only keys the model has, exactly as Core's does."""

    def __init__(self, real, keys, patches=None):
        self.model = real.model
        self._real = real
        self._keys = keys
        self.patches = dict(patches or {})

    def clone(self):
        return _CaptureModel(self._real, self._keys, {k: list(v) for k, v in self.patches.items()})

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        out = []
        for k, p in patches.items():
            key = k if isinstance(k, str) else k[0]
            if key in self._keys:
                self.patches.setdefault(k, []).append((float(strength_patch), p))
                out.append(k)
        return out


class _NoSeqLoader:
    """If the merge falls back to SEQ inside the check, record it instead."""

    def __init__(self):
        self.calls = []

    def load_lora(self, m, c, name, w, wc):
        self.calls.append(name)
        return m, c


def _merge_check_block(ordered, cfg, model, is_joint, device, short):
    """v988 -- per group: SEQ's patches (Core's own loader path) against the
    Stack's merge, on the connected model, in factor space."""
    import contextlib
    import io
    L = ["", "═══ Merge check: the Stack's merge vs SEQ ═══"]
    if model is None or not hasattr(model, "model"):
        L.append("  Connect the model input -- the check needs the model's own key map.")
        return L
    try:
        import torch
        import comfy.lora
        import comfy.lora_convert
    except Exception as ex:
        L.append(f"  ✗ ComfyUI/PyTorch unavailable -- {ex}")
        return L
    try:
        keys = set(model.model.state_dict().keys())
        keymap = comfy.lora.model_lora_keys_unet(model.model, {})
    except Exception as ex:
        L.append(f"  ✗ could not read the model's key map -- {ex}")
        return L
    dev = "cpu" if device == "cpu" else _resolve_pick_device()
    L.append(f"  Device: {dev} | reference = each LoRA through ComfyUI's own loader "
             f"(what SEQ runs); merged = the Stack's CONCAT")
    group_modes = cfg.get("group_modes", {}) if isinstance(cfg.get("group_modes"), dict) else {}
    group_trim = cfg.get("group_trim", {}) if isinstance(cfg.get("group_trim"), dict) else {}
    group_resolve = cfg.get("group_resolve", {}) if isinstance(cfg.get("group_resolve"), dict) else {}
    group_trim_amt = cfg.get("group_trim_amount", {}) if isinstance(cfg.get("group_trim_amount"), dict) else {}
    t0 = time.perf_counter()
    for group, grp_rows, grp_weights in ordered:
        if len(grp_rows) < 2:
            continue
        ws, cs, _gm, _cap = _group_effective(cfg, group, grp_rows, grp_weights)
        names = [r.get("name", "None") for r in grp_rows]
        L.append(f"  [{group}]  ({len(names)} LoRAs)")
        # --- reference: Core's path, per LoRA (comfy.sd.load_lora_for_models)
        ref = {}
        for name, w in zip(names, ws, strict=True):
            _check_interrupt()
            if abs(w) < 1e-6 or not name or name == "None":
                continue
            path = folder_paths.get_full_path("loras", name)
            if not path:
                L.append(f"     ⚠ not found: {short(name, 30)}")
                continue
            try:
                td = comfy.lora_convert.convert_lora(_cached_load_torch_file(path))
                loaded = comfy.lora.load_lora(td, keymap, log_missing=False)
            except INTERRUPT_EXC:
                raise
            except Exception as ex:
                L.append(f"     ⚠ {short(name, 30)}: ComfyUI's loader failed -- {ex}")
                continue
            hit = 0
            for k, p in loaded.items():
                key = k if isinstance(k, str) else k[0]
                if key in keys:
                    ref.setdefault(k, []).append((name, float(w), p))
                    hit += 1
            if hit == 0:
                L.append(f"     ⚠ {short(name, 30)}: ComfyUI's own loader maps 0 weights "
                         f"-- under SEQ this LoRA does NOTHING")

        def _run(mode, trim, resolve, trim_amount):
            cap = _CaptureModel(model, keys)
            ld = _NoSeqLoader()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                out = _apply_concat_or_dare(ld, cap, None, names, list(ws), mode=mode,
                                            dare_variant="channel", trim=trim, resolve=resolve,
                                            trim_amount=trim_amount, clip_weights=list(cs),
                                            handoff="patch")
            m = out[0] if isinstance(out, tuple) else out
            notes = [ln.replace("[PLS]", "").strip() for ln in buf.getvalue().splitlines()
                     if "⚠" in ln or "✗" in ln or "falling back" in ln]
            if ld.calls or not isinstance(m, _CaptureModel):
                return None, notes
            return m.patches, notes

        merged, notes = _run("CONCAT", False, False, None)
        if merged is None:
            L.append("     the merge fell back to SEQ here -- nothing to compare:")
            L.extend("       " + n for n in notes[:4])
            continue
        res = _MC.compare(ref, merged, torch, dev=dev, tick=_check_interrupt)
        L.extend(_MC.check_lines(group, res, short, what="CONCAT"))
        # --- the group's own settings, when they change the weights by design
        mode = (group_modes.get(group) or "SEQ").upper()
        trim = bool(group_trim.get(group, False)) and mode != "SEQ"
        resolve = bool(group_resolve.get(group, False)) and mode != "SEQ"
        if mode == "DARE" or resolve:
            mode2, resolve2, _n = _joint_merge_downgrade(mode, resolve, trim, bool(is_joint))
        else:
            mode2, resolve2 = ("CONCAT" if mode != "SEQ" else "SEQ"), resolve
        if mode2 != "SEQ" and (trim or resolve2 or mode2 == "DARE"):
            ta = group_trim_amt.get(group)
            m2, _n2 = _run(mode2, trim, resolve2,
                           float(ta) if isinstance(ta, (int, float)) else None)
            if m2 is not None:
                r2 = _MC.compare(ref, m2, torch, dev=dev, tick=_check_interrupt)
                why = [x for x, on in (("TRIM", trim), ("RESOLVE", resolve2),
                                       ("DARE", mode2 == "DARE")) if on]
                tag = mode2 + "".join(" +" + x for x in why if x != "DARE")
                L.append(f"     {tag} (your setting) vs SEQ: deviation "
                         f"{_MC.group_rel(r2) * 100:.1f}% -- {' + '.join(why)} changes "
                         f"the weights ON PURPOSE; this is how much")
    L.append(f"  ({time.perf_counter() - t0:.1f} s)")
    return L


def _overlap_block(ordered, is_joint, device, cfg=None, short=None):
    """Load every active LoRA the way the merge loads it (cache + Core's
    conversion + v930 schemas), then measure G in factor space. Never raises
    for data issues -- a line says what was left out and why."""
    L = ["", "═══ Overlap & energy ═══"]
    short = short or _short_name
    try:
        import torch
    except Exception:
        L.append("  ✗ PyTorch unavailable -- measurement not possible.")
        return L
    entries = []
    left_out = []
    for group, grp_rows, grp_weights in ordered:
        # v981/v983: the weights the Stack really applies (strength + cap)
        ws, cs, _gm, _cap = _group_effective(cfg or {}, group, grp_rows, grp_weights)
        for r, w, wc in zip(grp_rows, ws, cs, strict=True):
            name = r.get("name", "None")
            if not name or name == "None":
                continue
            path = folder_paths.get_full_path("loras", name)
            if not path:
                left_out.append((name, "file not found"))
                continue
            try:
                td = _cached_load_torch_file(path)
                td = _convert_lora_like_core(td, path) if td else td
            except INTERRUPT_EXC:
                raise
            except Exception as ex:
                left_out.append((name, f"load failed: {ex}"))
                continue
            conv = _detect_convention(td) if td else None
            if conv is None:
                left_out.append((name, "not a plain LoRA (LyCORIS?)"))
                continue
            if _has_mid_tensor(td):
                left_out.append((name, "conv mid tensor"))
                continue
            entries.append({"name": name, "group": group, "weight": float(w),
                            "clip_weight": float(wc),
                            "td": td, "conv": conv})
    # v986: no convention filter any more -- measure_overlap reads each LoRA in
    # its own key naming and meets the layers by _canonical_base, exactly as
    # the merge now does. (v980-v985 left the minority naming out.)
    _mixed = len({e["conv"] for e in entries}) > 1
    if is_joint:
        L.append(JOINT_SHARED_NOTE)
    if len(entries) < 1:
        L.append("  No LoRA could be measured.")
    else:
        dev = "cpu" if device == "cpu" else _resolve_pick_device()
        L.append(f"  Device: {dev} | {len(entries)} LoRA(s) | weights and alpha folded "
                 f"in as the merge folds them")
        if _mixed:
            L.append("  Key naming mixed (kohya + lora_A/lora_B): layers met per weight, "
                     "as the merge meets them")
        t0 = time.perf_counter()
        meas = _OV.measure_overlap(entries, torch, dev=dev, tick=_check_interrupt)
        summ = _OV.summarize(meas["G"], [e["group"] for e in entries])
        L.extend(_OV.overlap_lines([e["name"] for e in entries],
                                   [e["group"] for e in entries],
                                   [e["weight"] for e in entries],
                                   meas, summ, short))
        L.append(f"  ({time.perf_counter() - t0:.1f} s)")
    for name, why in left_out:
        L.append(f"  ⚠ left out: {short(name, 34)} -- {why}")
    return L


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
                "analysis_depth": (DEPTHS, {
                    "default": "Overview",
                    "tooltip": DEPTH_TIP,
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
                # v980: a SOCKET, appended last -- no widgets_values slot moves.
                "model": ("MODEL", {"tooltip": MODEL_TIP}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT",          "FLOAT",                "BOOLEAN")
    RETURN_NAMES = ("report", "energy_1x_pct",  "amplitude_ratio",        "resolve_active")
    FUNCTION     = "analyze"
    CATEGORY     = "Polyhedron/Utils"
    OUTPUT_NODE  = False
    DESCRIPTION  = ("Analyzes the Stack's CONCAT/DARE/Resolve merge. "
                    "Shows the live-selected LoRAs; 'Deep analysis' measures the "
                    "Resolve re-pack fidelity (energy 1×/2×/4×, amplitude); "
                    "'Overlap & energy' measures what each LoRA adds and which "
                    "pull together or apart. Connect 'model' to see what really "
                    "runs on it. report → Show Text. One per Stack.")

    def analyze(self, uls_config_out, analysis_depth="Overview",
                max_layers=24, device="auto", model=None):
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

        is_joint = _model_is_joint(model)          # v980: None = not connected
        L = ["═══ Polyhedron Merge Analyzer ═══",
             f"  Groups active : {len(ordered)}",
             (f"  Global mult   : ×{mult_f:.2f} saved -- NOT applied (no slider; "
              f"use the group strength)" if abs(mult_f - 1.0) > 1e-9 else
              "  Global mult   : none (use the group strength)"),
             f"  Model         : {_model_line(is_joint)}",
             "─────────────────────────────────"]

        if not ordered:
            L.append("  (no active LoRA rows — connect uls_config_out from the Stack)")
            return ("\n".join(L), 100.0, 1.0, False)

        # v985: names a reader can tell apart -- one shortener for the report
        short = _OV.name_shortener([r.get("name", "") for _g, rows_, _w in ordered
                                    for r in rows_])

        # --- Overview: per group, mode + switches + LoRAs ---
        resolve_groups = []   # (group, names, weights, trim_keep_or_None)
        for group, grp_rows, grp_weights in ordered:
            n = len(grp_rows)
            # v981/v983: the weights the Stack runs with -- the SAME function
            grp_weights, grp_clip, gm, cap = _group_effective(cfg, group, grp_rows, grp_weights)
            mode = (group_modes.get(group) or "SEQ").upper()
            if mode not in ("SEQ", "CONCAT", "DARE"):
                mode = "SEQ"
            variant = str(group_dare.get(group, legacy_dare)).lower()
            if variant not in ("channel", "element"):
                variant = "channel"
            trim    = bool(group_trim.get(group, False))    and mode != "SEQ"
            resolve = bool(group_resolve.get(group, False)) and mode != "SEQ"
            # v980: show what RUNS, not what is set. Same policy call as the
            # Stack (uls_stack_node.apply_lora_set) -- one source of truth.
            set_mode, set_resolve = mode, resolve
            if is_joint and n >= 2:
                mode, resolve, _n = _joint_merge_downgrade(mode, resolve, trim, True)
            trim_keep = None
            if trim:
                _ta = group_trim_amt.get(group, None)
                trim_keep = float(_ta) if isinstance(_ta, (int, float)) else _trim_keep_fraction(n)

            tag = mode
            if mode == "DARE":
                tag += f" [{variant[:4].upper()}]"
            if set_mode != mode:
                tag = f"{set_mode}→{mode} (joint)"
            if trim:
                tag += " +TRIM"
            if resolve:
                tag += " +RESOLVE"
            elif set_resolve:
                tag += " (RESOLVE off: joint)"
            # v985: would the run really merge? Ask the Stack's own decision.
            marks, path_lines, blocked = {}, [], False
            if mode != "SEQ" and n >= 2:
                blk, n_ok, missing, mix = _merge_path(grp_rows, grp_weights, grp_clip)
                blocked = bool(blk) or n_ok < 2
                marks, path_lines = _merge_path_lines(blk, n_ok, missing, short, mix)
                if blocked:
                    tag += " → runs SEQ"
                    resolve = False
            grp_label = f"[{group}]" if group != "—" else "[—]"
            flag = "   ← Resolve active" if (resolve and n >= 2) else ""
            gtxt = (f"  group ×{gm:g}" if gm != 1.0 else "") + _cap_text(cap)
            L.append(f"  {grp_label} {tag}  ({n} LoRA{'s' if n != 1 else ''}){gtxt}{flag}")
            for r, w in zip(grp_rows, grp_weights, strict=True):
                nm = r.get('name', '')
                mk = f"  [{marks[nm]}]" if nm in marks else ""
                L.append(f"     • {short(nm, 34):<34} ×{w}{mk}")
            L.extend(path_lines)

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

        # --- v988: Merge check ---
        if analysis_depth == DEPTH_MERGE:
            L.extend(_merge_check_block(ordered, cfg, model, is_joint, device, short))
            L.append("─────────────────────────────────")
            return ("\n".join(L), 100.0, 1.0, bool(resolve_groups))

        # --- v980: Overlap & energy ---
        if analysis_depth == DEPTH_OVERLAP:
            L.extend(_overlap_block(ordered, is_joint, device, cfg, short))
            L.append("─────────────────────────────────")
            return ("\n".join(L), 100.0, 1.0, bool(resolve_groups))

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
                   max_layers=24, device="auto", model=None, **kw):
        # Recompute only when selection/mode/depth change — so the expensive
        # deep analysis does not run on every queue, only when something changes.
        h = hashlib.sha1()
        for part in (str(uls_config_out), str(analysis_depth), str(max_layers), str(device)):
            h.update(part.encode("utf-8", "replace")); h.update(b"|")
        return h.hexdigest()
