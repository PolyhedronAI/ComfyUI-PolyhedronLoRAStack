"""
Polyhedron Noise Schedule Nodes
================================
Generates SIGMAS tensors for WAN Flow-Matching samplers using named
sigma-curve schedules — fully independent of kijai's WanVideoWrapper.

Plug SIGMAS output into WanVideoScheduler's sigmas-input to override
the internal sigma curve while keeping any solver (dpm++, res_multistep…).

Nodes:
  • ULSWanSigmaSchedule       — single schedule, one SIGMAS output
  • ULSWanSplitNoiseSchedule  — split HIGH/LOW with seamless handoff at split_step

v204 — Dual rescales LOW tail to handoff, eliminates plateaus
"""

import math
import re

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Helper: validate and normalize inputs
# ---------------------------------------------------------------------------

def _validate(n, smin, smax):
    """Ensure n ≥ 1 and sigma_min < sigma_max."""
    n = max(1, int(n))
    if smin >= smax:
        smin, smax = min(smin, smax), max(smin, smax)
        if smin >= smax:
            smax = smin + 1e-3
    smin = max(1e-9, float(smin))
    smax = max(float(smax), smin + 1e-6)   # no upper clamp — SDXL needs >1.0
    return n, smin, smax


def _finalize_raw(sigs, smax, smin):
    """
    Common post-processing — RAW version returns numpy array WITHOUT terminal 0.
    Used internally when we need to slice/stitch curves before adding the zero.
    """
    sigs = np.asarray(sigs, dtype=np.float32)
    sigs = np.maximum.accumulate(sigs[::-1])[::-1].copy()
    if len(sigs) >= 2:
        sigs[0]  = smax   # no clamp — allow k-diffusion range (>1.0)
        sigs[-1] = smin
    elif len(sigs) == 1:
        sigs[0]  = smax
    return sigs


def _to_tensor_with_zero(sigs_np):
    """Append terminal 0.0 and return float32 tensor."""
    return torch.tensor(np.concatenate([sigs_np, [0.0]]), dtype=torch.float32)


def _finalize(sigs, smax, smin):
    """Convenience: raw + tensor with zero (back-compat)."""
    return _to_tensor_with_zero(_finalize_raw(sigs, smax, smin))


# ---------------------------------------------------------------------------
# Sigma schedule implementations (RAW — return numpy arrays without terminal 0)
# ---------------------------------------------------------------------------

def _raw_bong_tangent(n, smin, smax):
    n, smin, smax = _validate(n, smin, smax)
    if n == 1:
        return _finalize_raw(np.array([smax]), smax, smin)
    t = np.linspace(0.0, 1.0, n)
    angle = (1.0 - t) * (np.pi / 2.0 - 0.08)
    t_tan = np.tan(angle)
    t_tan = t_tan / t_tan.max()
    sigs = smin + t_tan * (smax - smin)
    return _finalize_raw(sigs, smax, smin)


def _raw_beta57(n, smin, smax):
    n, smin, smax = _validate(n, smin, smax)
    if n == 1:
        return _finalize_raw(np.array([smax]), smax, smin)
    try:
        from scipy.stats import beta as _sb
    except ImportError:
        raise ImportError(
            "[PolyhedronSigma] scipy required for beta57.\n"
            "  Fix: .\\python_embeded\\python.exe -m pip install scipy"
        ) from None
    timesteps = 1.0 - np.linspace(0.0, 1.0, n)
    sigs = smin + (smax - smin) * _sb.ppf(timesteps, 0.5, 0.7)
    return _finalize_raw(sigs, smax, smin)


def _raw_karras(n, smin, smax, rho):
    n, smin, smax = _validate(n, smin, smax)
    if n == 1:
        return _finalize_raw(np.array([smax]), smax, smin)
    ramp = np.linspace(0.0, 1.0, n)
    min_inv_rho = smin ** (1.0 / rho)
    max_inv_rho = smax ** (1.0 / rho)
    sigs = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return _finalize_raw(sigs, smax, smin)


def _raw_exponential(n, smin, smax, rho):
    n, smin, smax = _validate(n, smin, smax)
    if n == 1:
        return _finalize_raw(np.array([smax]), smax, smin)
    log_min = np.log(smin)
    log_max = np.log(smax)
    t = np.linspace(0.0, 1.0, n) ** rho
    sigs = np.exp(log_max + t * (log_min - log_max))
    return _finalize_raw(sigs, smax, smin)


def _raw_linear(n, smin, smax):
    n, smin, smax = _validate(n, smin, smax)
    sigs = np.linspace(smax, smin, n)
    return _finalize_raw(sigs, smax, smin)


def _raw_cosine(n, smin, smax):
    n, smin, smax = _validate(n, smin, smax)
    if n == 1:
        return _finalize_raw(np.array([smax]), smax, smin)
    t = np.linspace(0.0, 1.0, n)
    sigs = smin + 0.5 * (smax - smin) * (1.0 + np.cos(np.pi * t))
    return _finalize_raw(sigs, smax, smin)


def _raw_sgm_uniform(n, smin, smax):
    n, smin, smax = _validate(n, smin, smax)
    if n == 1:
        return _finalize_raw(np.array([smax]), smax, smin)
    sigs = np.linspace(smax, smin, n + 1)[:-1]
    return _finalize_raw(sigs, smax, smin)


def _raw_laplace(n, smin, smax, rho):
    n, smin, smax = _validate(n, smin, smax)
    if n == 1:
        return _finalize_raw(np.array([smax]), smax, smin)
    mu = 0.5 * (smax + smin)
    scale = (smax - smin) / (2.0 * max(rho, 0.1))
    t = np.linspace(0.0, 1.0, n)
    half = 0.5

    def _laplace_ppf(p):
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return mu - scale * np.sign(p - half) * np.log(1.0 - 2.0 * np.abs(p - half))

    sigs = _laplace_ppf(1.0 - t)
    return _finalize_raw(sigs, smax, smin)


# ---------------------------------------------------------------------------
# Dispatch table — maps name → (raw_fn, uses_rho)
# ---------------------------------------------------------------------------

_SCHEDULES = {
    "bong_tangent": {"fn": lambda n, smin, smax, rho: _raw_bong_tangent(n, smin, smax),    "uses_rho": False},
    "beta57":       {"fn": lambda n, smin, smax, rho: _raw_beta57(n, smin, smax),          "uses_rho": False},
    "karras":       {"fn": lambda n, smin, smax, rho: _raw_karras(n, smin, smax, rho),     "uses_rho": True},
    "exponential":  {"fn": lambda n, smin, smax, rho: _raw_exponential(n, smin, smax, rho),"uses_rho": True},
    "linear":       {"fn": lambda n, smin, smax, rho: _raw_linear(n, smin, smax),          "uses_rho": False},
    "cosine":       {"fn": lambda n, smin, smax, rho: _raw_cosine(n, smin, smax),          "uses_rho": False},
    "sgm_uniform":  {"fn": lambda n, smin, smax, rho: _raw_sgm_uniform(n, smin, smax),     "uses_rho": False},
    "laplace":      {"fn": lambda n, smin, smax, rho: _raw_laplace(n, smin, smax, rho),    "uses_rho": True},
}

SIGMA_SCHEDULE_NAMES = list(_SCHEDULES.keys())


def _compute_raw(schedule_name, n, smin, smax, rho):
    """Compute raw sigma array (no terminal 0) for a named schedule."""
    return _SCHEDULES[schedule_name]["fn"](int(n), float(smin), float(smax), float(rho))


# ---------------------------------------------------------------------------
# Node 1: Single Noise Schedule (with passthrough outputs for sync)
# ---------------------------------------------------------------------------

class ULSWanSigmaSchedule:
    """
    ⬡ Polyhedron Noise Schedule

    Generates a SIGMAS tensor using a named sigma-curve schedule,
    completely independent of kijai's WanVideoWrapper.

    Outputs:
      • sigmas    — feed to WanVideoScheduler.sigmas
      • steps     — passthrough INT for sync (connect to Sampler/Scheduler steps)
      • sigma_max — passthrough FLOAT
      • sigma_min — passthrough FLOAT
      • rho       — passthrough FLOAT
    """

    # v578: the REAL mechanism, not a marker typed into the label.
    # ComfyUI hides DEPRECATED nodes from the search box but keeps them
    # fully alive in existing workflows - which is exactly what a retired
    # node needs, and exactly the road any future node_id rename must take
    # (register the new id, keep the old one here with this flag).
    # An older ComfyUI that does not know the flag simply ignores it.
    DEPRECATED = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigma_schedule": (SIGMA_SCHEDULE_NAMES, {"default": "karras"}),
                "steps":     ("INT",   {"default": 20,    "min": 1, "max": 300,
                                        "tooltip": "Number of sigma steps"}),
                "sigma_max": ("FLOAT", {"default": 1.0,   "min": 0.001, "max": 1.0, "step": 0.001,
                                        "tooltip": "Max sigma — use 1.0 for WAN flow-matching"}),
                "sigma_min": ("FLOAT", {"default": 0.002, "min": 0.0001, "max": 0.999, "step": 0.0001,
                                        "tooltip": "Min sigma — use 0.002 for WAN flow-matching"}),
                "rho":       ("FLOAT", {"default": 7.0,   "min": 0.1, "max": 20.0, "step": 0.1,
                                        "tooltip": "Shape param — karras/exponential/laplace"}),
            },
        }

    RETURN_TYPES = ("SIGMAS", "INT",   "FLOAT",     "FLOAT",     "FLOAT")
    RETURN_NAMES = ("sigmas", "steps", "sigma_max", "sigma_min", "rho")
    FUNCTION     = "compute"
    CATEGORY     = "Polyhedron/Sigma"
    DESCRIPTION  = (
        "Single sigma-curve node. INT/FLOAT outputs render inline next to widgets — "
        "wire them into samplers and schedulers for instant sync."
    )

    def compute(self, sigma_schedule, steps, sigma_max, sigma_min, rho):
        if sigma_min >= sigma_max:
            print(f"[PolyhedronSigma] ⚠ sigma_min ({sigma_min}) >= sigma_max ({sigma_max}) — swapping")
            # v263: actually swap here (matching the Dual node), so the
            # passthrough FLOAT outputs below report the SAME order the curve
            # was built with. _validate() also swaps internally for the curve
            # itself; this only fixes the previously-inconsistent passthrough.
            sigma_min, sigma_max = sigma_max, sigma_min

        raw = _compute_raw(sigma_schedule, steps, sigma_min, sigma_max, rho)
        sigmas = _to_tensor_with_zero(raw)

        uses_rho = _SCHEDULES[sigma_schedule].get("uses_rho", False)
        rho_str = f"rho={rho}" if uses_rho else "rho=n/a"
        print(f"[PolyhedronSigma] {sigma_schedule} | steps={steps} | {rho_str} | "
              f"σ [{sigmas[0]:.4f} → {sigmas[-2]:.4f}] | len={len(sigmas)}")

        return (sigmas, int(steps), float(sigma_max), float(sigma_min), float(rho))


# ---------------------------------------------------------------------------
# Node 2: Dual Sigma Curve — HIGH/LOW with seamless handoff
# ---------------------------------------------------------------------------

def split_curves(schedule_high, schedule_low, total_steps, split_step,
                 sigma_max, sigma_min, rho_high, rho_low):
    """v984: the Dual Sigma Curve's math, pure (no printing) -- the node's
    compute() and the curve preview route both call THIS, so the drawn curve
    is the run's curve. Returns (sigmas_high, sigmas_low, split_step used,
    swapped). Moved verbatim out of compute(); the v984 guard pins the output
    bit-identical to the pre-move node over 752 parameter combinations."""
    split_step = max(1, min(int(split_step), int(total_steps) - 1))
    swapped = False
    if sigma_min >= sigma_max:
        swapped = True
        sigma_min, sigma_max = sigma_max, sigma_min

    raw_high = _compute_raw(schedule_high, total_steps, sigma_min, sigma_max, rho_high)
    raw_low  = _compute_raw(schedule_low,  total_steps, sigma_min, sigma_max, rho_low)

    # Both outputs are FULL-length lists.
    # The sampler does its own start_step/end_step slicing.
    # We only enforce the SEAMLESS HANDOFF at index split_step:
    #   sigmas_low[split_step] = sigmas_high[split_step]
    # This guarantees that HIGH's last sigma == LOW's first sigma at the handoff.
    sigmas_high_np = raw_high.copy()
    sigmas_low_np  = raw_low.copy()

    # Strategy: re-scale LOW's tail so that LOW[split_step] = HIGH[split_step]
    # while preserving the shape of the LOW curve after split_step.
    #
    # The LOW curve naturally has sigma_max at index 0 and sigma_min at the end.
    # We rescale only the part [split_step..end] linearly so that:
    #   - LOW[split_step] = HIGH[split_step] (the handoff sigma)
    #   - LOW[end] = sigma_min (preserved)
    # This avoids plateaus AND preserves the LOW curve's shape characteristic.

    handoff = float(sigmas_high_np[split_step])
    original_at_split = float(raw_low[split_step])

    if original_at_split > sigma_min:
        # Rescale tail: map [original_at_split, sigma_min] → [handoff, sigma_min]
        scale = (handoff - sigma_min) / (original_at_split - sigma_min)
        for i in range(split_step, len(sigmas_low_np)):
            sigmas_low_np[i] = sigma_min + scale * (raw_low[i] - sigma_min)

    # v267 (audit A-8): pin the handoff EXACTLY. The rescale formula above
    # is algebraically exact but leaves 1 float32 ULP (measured 5.96e-08
    # max over 256 schedule combos); direct assignment makes
    # HIGH[split] == LOW[split] bit-equal. Cosmetic — the sampler never
    # saw the ULP — and it also covers the rescale-skipped branch
    # (original_at_split <= sigma_min).
    sigmas_low_np[split_step] = handoff

    # Use HIGH values for [0..split_step-1] so the prefix is monotonic with handoff
    for i in range(split_step):
        sigmas_low_np[i] = sigmas_high_np[i]

    # Pin the very last value to sigma_min for numerical safety
    sigmas_low_np[-1] = sigma_min

    # Final safety: ensure endpoints are pinned
    sigmas_high_np[0]  = sigma_max
    sigmas_high_np[-1] = sigma_min
    sigmas_low_np[0]   = sigma_max
    sigmas_low_np[-1]  = sigma_min

    sigmas_high = _to_tensor_with_zero(sigmas_high_np)
    sigmas_low  = _to_tensor_with_zero(sigmas_low_np)
    return sigmas_high, sigmas_low, split_step, swapped


class ULSWanSplitNoiseSchedule:
    """
    ⬡ Polyhedron Dual Sigma Curve

    Generates TWO sigma curves for HIGH/LOW dual-pass sampling.
    Different schedules per pass — seamless handoff at split_step guaranteed.

    What it outputs (v984: the old text described a slicing this node has
    not done for a long time, and promised outputs it never had):
      * both outputs are FULL-length lists (total_steps + 1 values incl. the
        terminal 0); each sampler does its own start/end slicing;
      * HIGH is its own schedule unchanged;
      * LOW takes HIGH's values up to split_step, carries HIGH's sigma AT
        split_step exactly (the seamless handoff), and after it keeps its
        own schedule's SHAPE, rescaled to run from that sigma to sigma_min.
    split_step is clamped to 1 .. total_steps-1. The math lives in
    split_curves() so the node's curve preview draws exactly this.

    sigma_max / sigma_min:
      Flow-matching (WAN, FLUX, SD3): max=1.0,   min=0.002
      k-diffusion   (SDXL, SD 1.5):  max=14.61, min=0.029
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule_high": (SIGMA_SCHEDULE_NAMES, {
                    "default": "karras",
                    "tooltip": "Sigma curve for HIGH pass (structure phase)"
                }),
                "schedule_low":  (SIGMA_SCHEDULE_NAMES, {
                    "default": "bong_tangent",
                    "tooltip": "Sigma curve for LOW pass (detail phase)"
                }),
                "total_steps":   ("INT",   {
                    "default": 20, "min": 2, "max": 300,
                    "tooltip": "Total steps across both passes"
                }),
                "split_step":    ("INT",   {
                    "default": 8, "min": 1, "max": 299,
                    "tooltip": "Where HIGH ends and LOW begins"
                }),
                "sigma_max":     ("FLOAT", {
                    "default": 1.0, "min": 0.0001, "max": 1000.0, "step": 0.001,
                    "tooltip": "Flow-matching (WAN/FLUX/SD3): 1.0 — k-diffusion (SDXL/SD1.5): 14.61"
                }),
                "sigma_min":     ("FLOAT", {
                    "default": 0.002, "min": 0.00001, "max": 100.0, "step": 0.0001,
                    "tooltip": "Flow-matching (WAN/FLUX/SD3): 0.002 — k-diffusion (SDXL/SD1.5): 0.029"
                }),
                "rho_high":      ("FLOAT", {
                    "default": 7.0, "min": 0.1, "max": 20.0, "step": 0.1,
                    "tooltip": "Shape param for HIGH schedule (karras/exponential/laplace only)"
                }),
                "rho_low":       ("FLOAT", {
                    "default": 7.0, "min": 0.1, "max": 20.0, "step": 0.1,
                    "tooltip": "Shape param for LOW schedule (karras/exponential/laplace only)"
                }),
            },
        }

    RETURN_TYPES  = ("SIGMAS",      "SIGMAS")
    RETURN_NAMES  = ("sigmas_high", "sigmas_low")
    FUNCTION      = "compute"
    CATEGORY      = "Polyhedron/Sigma"
    DESCRIPTION   = (
        "Dual sigma-curve for HIGH/LOW dual-pass sampling. "
        "Different schedules per pass with seamless handoff at split_step. "
        "Both outputs are full-length lists; set each sampler's start/end "
        "step to split_step yourself."
    )

    def compute(self, schedule_high, schedule_low, total_steps, split_step,
                sigma_max, sigma_min, rho_high, rho_low):
        sigmas_high, sigmas_low, split_step, swapped = split_curves(
            schedule_high, schedule_low, total_steps, split_step,
            sigma_max, sigma_min, rho_high, rho_low)
        if swapped:
            print(f"[PolyhedronDual] ⚠ sigma_min ({sigma_min}) >= sigma_max ({sigma_max}) — swapping")

        print(
            f"[PolyhedronDual] HIGH '{schedule_high}' (full curve, used 0..{split_step}) "
            f"σ_split={sigmas_high[split_step]:.4f} | "
            f"LOW '{schedule_low}' (full curve, used {split_step}..{total_steps}) "
            f"σ_split={sigmas_low[split_step]:.4f} | "
            f"handoff diff={abs(sigmas_high[split_step]-sigmas_low[split_step]):.6f} | "
            f"both lists len={len(sigmas_high)}"
        )
        return (sigmas_high, sigmas_low)


# ---------------------------------------------------------------------------
# Node 3: Universal Sigma Curve
# ---------------------------------------------------------------------------

def universal_curve(sigma_schedule, steps, sigma_max, sigma_min, rho):
    """v984: the Sigma Curve's math, pure (no printing) -- compute() and the
    curve preview route both call THIS. Returns (sigmas, swapped)."""
    swapped = False
    if sigma_min >= sigma_max:
        swapped = True
        sigma_min, sigma_max = sigma_max, sigma_min
    raw = _compute_raw(sigma_schedule, steps, sigma_min, sigma_max, rho)
    return _to_tensor_with_zero(raw), swapped


def _pv_num(data, key, default, lo, hi, cast=float):
    """One preview input: the widget's value, clamped to the widget's range
    (INPUT_TYPES), so a hand-edited request cannot ask for a 10^6-step curve."""
    try:
        v = cast(data.get(key, default))
    except (TypeError, ValueError):
        v = cast(default)
    if v != v:          # NaN
        v = cast(default)
    return max(lo, min(hi, v))


def _pv_sched(data, key, default):
    s = str(data.get(key) or default)
    if s not in _SCHEDULES:
        raise ValueError("unknown schedule '%s'" % s)
    return s


def curve_preview(data):
    """v984: the curve nodes' preview -- what the RUN outputs for these widget
    values, from the same function compute() calls (universal_curve /
    split_curves). The frontend draws this and never calculates.

    data: {node: "curve"|"dual", + the node's widget values by name}.
    Returns {ok, node, sigmas | sigmas_high+sigmas_low+split_step, swapped}
    or {ok: False, error}."""
    data = data if isinstance(data, dict) else {}
    kind = str(data.get("node") or "curve")
    try:
        smax = _pv_num(data, "sigma_max", 1.0, 0.0001, 1000.0)
        smin = _pv_num(data, "sigma_min", 0.002, 0.00001, 100.0)
        if kind == "dual":
            total = _pv_num(data, "total_steps", 20, 2, 300, int)
            hi, lo, split, swapped = split_curves(
                _pv_sched(data, "schedule_high", "karras"),
                _pv_sched(data, "schedule_low", "bong_tangent"),
                total, _pv_num(data, "split_step", 8, 1, 299, int), smax, smin,
                _pv_num(data, "rho_high", 7.0, 0.1, 20.0),
                _pv_num(data, "rho_low", 7.0, 0.1, 20.0))
            return {"ok": True, "node": "dual", "sigmas_high": [float(x) for x in hi],
                    "sigmas_low": [float(x) for x in lo], "split_step": int(split),
                    "steps": total, "swapped": swapped}
        if kind != "curve":
            raise ValueError("unknown node kind '%s'" % kind)
        steps = _pv_num(data, "steps", 20, 1, 300, int)
        sig, swapped = universal_curve(
            _pv_sched(data, "sigma_schedule", "karras"), steps, smax, smin,
            _pv_num(data, "rho", 7.0, 0.1, 20.0))
        return {"ok": True, "node": "curve", "sigmas": [float(x) for x in sig],
                "steps": steps, "swapped": swapped}
    except Exception as e:
        return {"ok": False, "error": str(e)}


class ULSUniversalSigmaCurve:
    """
    ⬡ Polyhedron Sigma Curve

    Universal sigma-curve node — works with any model, any sampler, any pass.

    Use one per pass. Two for HIGH/LOW dual-pass setups.

    sigma_max / sigma_min:
      Flow-matching models (WAN, FLUX, SD3):  max=1.0,   min=0.002
      k-diffusion models  (SDXL, SD 1.5):    max=14.61, min=0.029
      Any future model:   set the values your model expects.

    One output, SIGMAS: steps + 1 values ending in the terminal 0. (v984: the
    old text promised a 'steps' pass-through output this node never had.)
    The math lives in universal_curve() so the node's curve preview draws
    exactly this.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigma_schedule": (SIGMA_SCHEDULE_NAMES, {
                    "default": "karras",
                    "tooltip": "Sigma curve shape — affects how steps are distributed across the noise range"
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 300,
                    "tooltip": "Number of steps -- the curve has steps + 1 values, ending in 0."
                }),
                "sigma_max": ("FLOAT", {
                    "default": 1.0, "min": 0.0001, "max": 1000.0, "step": 0.001,
                    "tooltip": "Flow-matching (WAN/FLUX/SD3): 1.0 — k-diffusion (SDXL/SD1.5): 14.61"
                }),
                "sigma_min": ("FLOAT", {
                    "default": 0.002, "min": 0.00001, "max": 100.0, "step": 0.0001,
                    "tooltip": "Flow-matching (WAN/FLUX/SD3): 0.002 — k-diffusion (SDXL/SD1.5): 0.029"
                }),
                "rho": ("FLOAT", {
                    "default": 7.0, "min": 0.1, "max": 20.0, "step": 0.1,
                    "tooltip": "Shape param — only affects karras, exponential, laplace"
                }),
            },
        }

    RETURN_TYPES  = ("SIGMAS",)
    RETURN_NAMES  = ("sigmas",)
    FUNCTION      = "compute"
    CATEGORY      = "Polyhedron/Sigma"
    DESCRIPTION   = (
        "Universal sigma-curve for any model. "
        "Set sigma_max/sigma_min to match your model family. "
        "The node draws the curve it outputs."
    )

    def compute(self, sigma_schedule, steps, sigma_max, sigma_min, rho):
        sigmas, swapped = universal_curve(sigma_schedule, steps, sigma_max, sigma_min, rho)
        if swapped:
            print(f"[PolyhedronSigma] ⚠ sigma_min ({sigma_min}) >= sigma_max ({sigma_max}) — swapping")
            sigma_min, sigma_max = sigma_max, sigma_min

        uses_rho = _SCHEDULES[sigma_schedule].get("uses_rho", False)
        rho_str  = f"rho={rho}" if uses_rho else "rho=n/a"
        print(
            f"[PolyhedronSigma] {sigma_schedule} | "
            f"steps={steps} | {rho_str} | "
            f"σ_max={sigma_max} σ_min={sigma_min} | "
            f"σ [{sigmas[0]:.4f} → {sigmas[-2]:.4f}]"
        )
        return (sigmas,)


# ---------------------------------------------------------------------------
# Node 4: Sigma List — an explicit, hand-written sigma grid (v971)
# ---------------------------------------------------------------------------
# Why this exists as its own node rather than a ninth "schedule" in the curve
# node: a named curve is COMPUTED from parameters, an explicit grid is GIVEN.
# Folding the second into the first would have to append a widget to a node
# that is wired into every saved workflow -- ComfyUI serialises widget values
# BY INDEX, so that renumbers all of them (guard #577). A separate node leaves
# every stored graph untouched.
#
# Distilled models (HyperFlow, AYS, turbo recipes) publish their trained grid
# as a list of numbers. This node takes ANY such list; it knows nothing about
# any particular model.

_SIGMA_SPLIT = re.compile(r"[,;\s]+")


def parse_sigma_list(text):
    """Parse a free-form sigma list.

    Tolerant about separators ON PURPOSE: comma, semicolon, space and newline
    all work, in any mix. A list published in a paper, a README or a metadata
    header gets pasted as-is. (A stricter split on ", " is a known foot-gun:
    it turns a missing space into a crash.)
    """
    parts = [p for p in _SIGMA_SPLIT.split(str(text).strip()) if p]
    if len(parts) < 2:
        raise ValueError(
            "Sigma List: need at least two values (a grid of N steps has N+1 "
            "points), got %d." % len(parts))
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except ValueError:
            raise ValueError(
                "Sigma List: '%s' is not a number. Separate values with "
                "commas, spaces or newlines." % p)
    return out


def validate_sigma_list(vals):
    """The contract every sampler assumes: finite, non-negative, strictly
    decreasing. Checked here so a typo fails loudly at the node instead of
    silently deforming the run."""
    for v in vals:
        if not math.isfinite(v):
            raise ValueError("Sigma List: values must be finite, got %r." % v)
        if v < 0.0:
            raise ValueError("Sigma List: values must be >= 0, got %r." % v)
    for i in range(1, len(vals)):
        if vals[i] >= vals[i - 1]:
            raise ValueError(
                "Sigma List: values must be strictly decreasing -- position %d "
                "(%r) is not below position %d (%r)." % (i, vals[i], i - 1, vals[i - 1]))
    return vals


def shift_sigma_list(vals, shift):
    """The exponential shift  s*x / (1 + (s-1)*x)  used by flow-matching
    schedulers. Maps 0 to 0 and 1 to 1, so a valid grid stays valid.

    Only defined on a normalised grid: a k-diffusion range (sigma_max 14.61)
    is not in [0, 1] and shifting it would be meaningless, so that is refused
    rather than quietly producing nonsense.
    """
    if shift <= 0.0:
        raise ValueError("Sigma List: shift must be positive, got %r." % shift)
    if shift == 1.0:
        return list(vals)
    for v in vals:
        if v > 1.0:
            raise ValueError(
                "Sigma List: shift only applies to a normalised grid (all "
                "values <= 1.0), but found %r. Leave shift at 1.0 for "
                "k-diffusion ranges." % v)
    return [shift * v / (1.0 + (shift - 1.0) * v) for v in vals]


# ---------------------------------------------------------------------------
# v972: named sigma grids.
# ---------------------------------------------------------------------------
# A published grid is a MEASUREMENT someone else made; typing it by hand is
# where a digit gets lost. Each entry is (grid, suggested_shift, note). The
# suggested shift is documentation, NOT applied automatically -- the `shift`
# widget stays the single place that decides, so what the node does is always
# what the graph shows.
#
# NOTHING here knows about any particular node pack. These are grids.
SIGMA_PRESETS = {
    # name: (grid, shift, enforce_terminal_zero, note)
    "custom": (None, None, None, "the widgets decide"),
    "hyperflow_8step_h3_raw": (
        (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0),
        12.0, True,
        "HyperFlow 8-step for MiniMax-H3. RAW grid, shifted here with the "
        "model's video shift (12.0). Recipe: euler, 8 steps, cfg 1.0."),
    "wan22_lightning_4step_raw": (
        (1.0, 0.75, 0.5, 0.25, 0.0),
        5.0, True,
        "lightx2v Wan 2.2 Lightning, 4 steps. RAW grid, shifted here with 5.0 -- "
        "reproduces the list in their published workflow to 1e-7. Recipe: euler, "
        "4 steps, split_step 2 across the HIGH/LOW experts."),
    "ays_sd15_10step": (
        (14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029),
        1.0, True,
        "NVIDIA Align-Your-Steps, SD 1.5. k-diffusion range, no shift."),
    "ays_sdxl_10step": (
        (14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029),
        1.0, True,
        "NVIDIA Align-Your-Steps, SDXL. k-diffusion range, no shift."),
    "ays_svd_10step": (
        (700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002),
        1.0, True,
        "NVIDIA Align-Your-Steps, SVD. k-diffusion range, no shift."),
}

SIGMA_PRESET_NAMES = list(SIGMA_PRESETS.keys())
PRESET_CUSTOM = "custom"


def preset_entry(name):
    """The full recipe behind a preset name as (grid, shift, enforce, note), or
    None for `custom` and for any name this build does not know. Unknown names
    fall back rather than raising: a workflow saved against a later build must
    still run, and the console line says which source won."""
    entry = SIGMA_PRESETS.get(str(name))
    if entry is None or entry[0] is None:
        return None
    return entry


def preset_grid(name):
    """Just the grid. Kept because a grid is what most callers want."""
    entry = preset_entry(name)
    return None if entry is None else entry[0]


def resolve_sigma_request(sigmas_text, shift, enforce_terminal_zero, preset):
    """Resolve widget settings into the grid a RUN would use (v976).

    The node's compute() and the preview route both need this, and a second
    copy would be a second truth. Raises the same refusals compute() raises;
    the route turns them into text, the node lets them stop the run.

    Returns (values, effective_shift, source, note)."""
    entry = preset_entry(preset)
    if entry is None:
        vals = validate_sigma_list(parse_sigma_list(sigmas_text))
        eff_shift = float(shift)
        eff_zero = bool(enforce_terminal_zero)
        source, note = "text", ""
    else:
        grid, p_shift, p_zero, note = entry
        vals = validate_sigma_list(list(grid))
        eff_shift = float(p_shift)
        eff_zero = bool(p_zero)
        source = "preset=%s" % preset
    vals = shift_sigma_list(vals, eff_shift)
    if eff_zero and vals[-1] != 0.0:
        vals = vals + [0.0]
    return vals, eff_shift, source, note


class ULSSigmaList:
    """
    ⬡ Polyhedron Sigma List

    An explicit sigma grid, typed in rather than computed.

    Distilled and few-step recipes ship a TRAINED grid -- a fixed list of
    numbers -- instead of a curve shape. Paste it here and wire the SIGMAS
    output straight into the sampler, in place of a scheduler.

    Separators are free: commas, spaces or newlines, in any mix.

    shift applies the flow-matching exponential shift  s*x/(1+(s-1)*x)  so a
    published RAW grid can be entered verbatim and shifted here (H3 video
    shift is 12.0), instead of pasting pre-multiplied numbers.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sigmas_text": ("STRING", {
                    "default": "1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.0",
                    "multiline": True,
                    "tooltip": "The sigma grid, highest first, strictly decreasing. "
                               "Commas, spaces or newlines all separate."
                }),
                "shift": ("FLOAT", {
                    "default": 1.0, "min": 0.0001, "max": 1000.0, "step": 0.01,
                    "tooltip": "Flow-matching exponential shift. 1.0 = off (paste an "
                               "already-shifted grid). MiniMax-H3 video shift is 12.0. "
                               "IGNORED while a preset other than 'custom' is selected."
                }),
                "enforce_terminal_zero": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Append a final 0.0 when the list does not end at zero. "
                               "Samplers expect the grid to reach zero. IGNORED while a "
                               "preset other than 'custom' is selected."
                }),
                # v972: APPENDED, never inserted. ComfyUI restores widget values
                # BY INDEX -- putting this above 'sigmas_text', where it belongs
                # semantically, would renumber every saved workflow (guard #577).
                "preset": (SIGMA_PRESET_NAMES, {
                    "default": PRESET_CUSTOM,
                    "tooltip": "A published recipe, measured once and kept here. Anything but "
                               "'custom' OVERRIDES the widgets above -- grid, shift and the "
                               "terminal-zero rule all come from the preset, and the console "
                               "names every value it replaced."
                }),
            },
        }

    RETURN_TYPES = ("SIGMAS", "INT")
    RETURN_NAMES = ("sigmas", "steps")
    FUNCTION     = "compute"
    CATEGORY     = "Polyhedron/Sigma"
    DESCRIPTION  = (
        "An explicit sigma grid, typed in rather than computed from a curve "
        "shape. For distilled recipes that publish a trained grid. The 'steps' "
        "output is the step count this grid implies (points - 1)."
    )

    def compute(self, sigmas_text, shift, enforce_terminal_zero, preset=PRESET_CUSTOM):
        # v973: a PRESET IS A PRESET. It carries a whole recipe -- grid, shift
        # and the terminal-zero rule -- and it overrides the widgets, all of
        # them. Half a preset (grid here, shift by hand) is a trap: it looks
        # applied while running the wrong curve, which is exactly what happened
        # in the field on the day v972 shipped.
        entry = preset_entry(preset)
        overrides = []
        if entry is not None:
            if abs(float(entry[1]) - float(shift)) > 1e-9:
                overrides.append("shift %s -> %s" % (shift, entry[1]))
            if bool(entry[2]) != bool(enforce_terminal_zero):
                overrides.append("enforce_terminal_zero %s -> %s"
                                 % (bool(enforce_terminal_zero), bool(entry[2])))
        vals, eff_shift, src, note = resolve_sigma_request(
            sigmas_text, shift, enforce_terminal_zero, preset)
        sigmas = torch.tensor(vals, dtype=torch.float32)
        steps = len(vals) - 1
        print(
            "[PolyhedronSigma] list | source=%s | points=%d steps=%d | shift=%s | "
            "sigma [%.4f -> %.4f]%s" % (src, len(vals), steps, eff_shift,
                                        vals[0], vals[-2],
                                        (" | preset overrides: " + ", ".join(overrides))
                                        if overrides else "")
        )
        if note:
            print("[PolyhedronSigma]   %s" % note)
        return (sigmas, steps)
