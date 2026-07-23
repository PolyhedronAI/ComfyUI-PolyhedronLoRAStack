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

class ULSWanSplitNoiseSchedule:
    """
    ⬡ Polyhedron Dual Sigma Curve

    Generates TWO sigma curves for HIGH/LOW dual-pass sampling.
    Different schedules per pass — seamless handoff at split_step guaranteed.

    Internal mechanics:
      1. Compute both curves independently (total_steps each)
      2. Slice HIGH: [0 .. split_step]
      3. Slice LOW:  [split_step .. total_steps]
      4. Force LOW[0] = HIGH[-1]  ← the seamless handoff
      5. Append terminal 0 to each

    sigma_max / sigma_min:
      Flow-matching (WAN, FLUX, SD3): max=1.0,   min=0.002
      k-diffusion   (SDXL, SD 1.5):  max=14.61, min=0.029

    The 'total_steps' output passes through for downstream sync.
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
        "Wire total_steps and split_step outputs to all samplers/schedulers."
    )

    def compute(self, schedule_high, schedule_low, total_steps, split_step,
                sigma_max, sigma_min, rho_high, rho_low):
        split_step = max(1, min(int(split_step), int(total_steps) - 1))

        if sigma_min >= sigma_max:
            print(f"[PolyhedronDual] ⚠ sigma_min ({sigma_min}) >= sigma_max ({sigma_max}) — swapping")
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

class ULSUniversalSigmaCurve:
    """
    ⬡ Polyhedron Sigma Curve

    Universal sigma-curve node — works with any model, any sampler, any pass.

    Use one per pass. Two for HIGH/LOW dual-pass setups.

    sigma_max / sigma_min:
      Flow-matching models (WAN, FLUX, SD3):  max=1.0,   min=0.002
      k-diffusion models  (SDXL, SD 1.5):    max=14.61, min=0.029
      Any future model:   set the values your model expects.

    The 'steps' output passes through to all downstream samplers/schedulers
    so one change here keeps everything in sync.
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
                    "tooltip": "Number of steps. Also passed through as output for downstream sync."
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
        "Wire the 'steps' output to samplers and schedulers for single-point sync."
    )

    def compute(self, sigma_schedule, steps, sigma_max, sigma_min, rho):
        if sigma_min >= sigma_max:
            print(f"[PolyhedronSigma] ⚠ sigma_min ({sigma_min}) >= sigma_max ({sigma_max}) — swapping")
            sigma_min, sigma_max = sigma_max, sigma_min

        raw    = _compute_raw(sigma_schedule, steps, sigma_min, sigma_max, rho)
        sigmas = _to_tensor_with_zero(raw)

        uses_rho = _SCHEDULES[sigma_schedule].get("uses_rho", False)
        rho_str  = f"rho={rho}" if uses_rho else "rho=n/a"
        print(
            f"[PolyhedronSigma] {sigma_schedule} | "
            f"steps={steps} | {rho_str} | "
            f"σ_max={sigma_max} σ_min={sigma_min} | "
            f"σ [{sigmas[0]:.4f} → {sigmas[-2]:.4f}]"
        )
        return (sigmas,)
