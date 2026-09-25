"""
⬡ Polyhedron Power Upscale (v514) -- the MoE-aware tiled upscaler.

One node replaces the chained double Ultimate-SD-Upscale setup for Wan 2.2
style expert pairs, mirroring the ⬡ Polyhedron Sampler's Single / High+Low
switch. CLEAN-ROOM: no code from the Ultimate-SD-Upscale ecosystem (GPL) --
the tile geometry lives in uls_tile_math.py (pure, guard-proved) and the
heavy lifting is DELEGATED to ComfyUI core, exactly like the sampler:
model-based pixel upscaling to comfy_extras' ImageUpscaleWithModel (spandrel:
ESRGAN, RealESRGAN, SwinIR, DAT, SPAN, ...), sampling to comfy.sample.sample,
VAE work to the wired VAE, and the WAN flow-matching sigma shift to the
sampler's own _apply_sigma_shift (v491 mechanism, imported, not copied).

MoE as a STAGE CHAIN (not a sigma-boundary split): at refine-strength
denoises (~0.2) the schedule starts far below the Wan handoff sigma, so a
boundary split would hand the HIGH expert zero steps. The correct
translation is the chain the community wires by hand:

  Single    : ESRGAN(upscale_model) x upscale_by -> tiled refine(model)
  High + Low: stage H  ESRGAN(upscale_model)      x upscale_by
                        -> tiled refine(model,     denoise, steps, cfg)
              stage L  ESRGAN(upscale_model_low*) x upscale_by_low
                        -> tiled refine(model_low*, denoise_low, steps_low,
                                        cfg_low)
              (* optional inputs; each falls back to the H input)

Seam-free by construction: tiles overlap and are stitched as a weighted sum
with separable smoothstep feathers whose accumulated weight is EXACTLY 1
everywhere (proved numerically in test_v514) -- no seam-fix pass exists
because none is needed.

Video thinking (v514): frames are only "single images" on the ESRGAN pass.
The refine crops each tile across the WHOLE frame stack and encodes it through
the wired VAE, so a Wan VAE yields ONE 5D video latent per tile and the
video model itself keeps time consistent inside the tile. On top, tile noise
seeds derive from (seed, tile index) and NEVER from the frame number, so the
noise structure per tile is stable over time. The green VIDEO input accepts
a native comfy_api VIDEO; its audio and frame rate ride through losslessly
into the VIDEO output (frames swapped for the upscaled ones). IMAGE in ->
VIDEO out is None, mirroring the Media Loader's convention.
"""

import base64
import inspect
import io
import json
import math
import os
import threading
import time
import uuid

import torch

import comfy.model_management   # v565: explicit - it was only ever reachable
import comfy.sample             # transitively, through comfy.sample
import comfy.samplers
import comfy.utils

import folder_paths

try:  # package load (ComfyUI) vs direct module load (tools)
    from . import uls_tile_math
    from .ph_runclock import _fmt_clock, _RunClock  # noqa: F401 (v576 re-export)
    from .uls_sampler import (_apply_sigma_shift, _resolve_low_shift, _low_or,
                              SAME_AS_HIGH, _current_node_id,
                              _latent_parts, _joint_video_half)   # v1004
    from .ph_logmute import MuteStagingLogs as _MuteInfoLogs
except ImportError:  # pragma: no cover
    import os as _os
    import sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    import uls_tile_math
    from ph_runclock import _fmt_clock, _RunClock  # noqa: F401 (v576 re-export)
    from uls_sampler import (_apply_sigma_shift, _resolve_low_shift, _low_or,
                             SAME_AS_HIGH, _current_node_id,
                             _latent_parts, _joint_video_half)   # v1004
    from ph_logmute import MuteStagingLogs as _MuteInfoLogs

# ComfyUI's VIDEO type (optional) -- the exact Media Loader pattern: the
# comfy_api paths are version-stable back-compat shims; absent -> the VIDEO
# output is simply None and a wired VIDEO input raises a clear error.
try:
    from comfy_api.input_impl import VideoFromComponents
    from comfy_api.util import VideoComponents
    from fractions import Fraction
    _HAS_VIDEO_API = True
except Exception:  # pragma: no cover
    VideoFromComponents = VideoComponents = None
    _HAS_VIDEO_API = False

# Model-based pixel upscaling: delegated to core (tiled + OOM-backoff there).
try:
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
    _MODEL_UPSCALER = ImageUpscaleWithModel()
except Exception:  # pragma: no cover
    _MODEL_UPSCALER = None


def _lanczos_to(image, width, height):
    """[N,H,W,C] -> [N,height,width,C] via core common_upscale (lanczos)."""
    t = image.movedim(-1, 1)
    t = comfy.utils.common_upscale(t, int(width), int(height), "lanczos", "disabled")
    return t.movedim(1, -1)


# ── v588: TRUE lanczos on the GPU ────────────────────────────────────────────
# comfy's "lanczos" is PIL: per frame, on the CPU, through an 8-bit uint8
# round trip (comfy/utils.py, read 2026-07-14). The kernel itself is the
# REFERENCE - _fit_method above measures every other kernel against "PIL's
# lanczos is a windowed sinc that low-passes correctly" - so the cure is not
# a different filter, it is the SAME filter where the frames already are.
# Separable lanczos-3 as two dense matmuls, float32 end to end, weight
# matrices cached per (in, out, device). On a shrink the kernel support
# stretches by the ratio (that IS the antialias); edges clamp (replicate);
# every output row's weights sum to exactly 1 - the same choices PIL makes,
# minus the 8-bit stop.
_LANCZOS_A = 3


def _lanczos_kernel(x):
    """The lanczos-3 window: a*sin(pi*x)*sin(pi*x/a)/(pi*x)^2 inside |x|<a,
    1 at 0, 0 outside. Pure - the guard exec()s this and pins real values."""
    x = float(x)
    if x == 0.0:
        return 1.0
    if abs(x) >= float(_LANCZOS_A):
        return 0.0
    px = math.pi * x
    return _LANCZOS_A * math.sin(px) * math.sin(px / _LANCZOS_A) / (px * px)


def _lanczos_weights_1d(n_in, n_out):
    """One axis of the resampling: for every output index j a tap list
    [(i, w), ...] with i clamped to [0, n_in-1] (replicate edges) and the
    weights normalised to sum 1. Pure python on purpose: the guard exec()s
    this without torch and pins shapes, sums, stretch and clamping."""
    n_in, n_out = int(n_in), int(n_out)
    scale = n_out / float(n_in)
    s = min(1.0, scale)              # a shrink stretches the kernel = antialias
    support = _LANCZOS_A / s
    rows = []
    for j in range(n_out):
        center = (j + 0.5) / scale - 0.5
        lo = int(math.floor(center - support))
        hi = int(math.ceil(center + support))
        taps = {}
        for i in range(lo, hi + 1):
            w = _lanczos_kernel((i - center) * s)
            if w != 0.0:
                ci = min(max(i, 0), n_in - 1)
                taps[ci] = taps.get(ci, 0.0) + w
        total = sum(taps.values())
        rows.append([(i, w / total) for i, w in sorted(taps.items())])
    return rows


_LANCZOS_MATS = {}   # (n_in, n_out, device) -> [n_out, n_in] float32 matrix


def _lanczos_matrix(n_in, n_out, device):
    key = (int(n_in), int(n_out), str(device))
    m = _LANCZOS_MATS.get(key)
    if m is None:
        if len(_LANCZOS_MATS) >= 8:   # video runs use 2 per pass - 8 is plenty
            _LANCZOS_MATS.clear()
        m = torch.zeros((int(n_out), int(n_in)), dtype=torch.float32)
        for j, taps in enumerate(_lanczos_weights_1d(n_in, n_out)):
            for i, w in taps:
                m[j, i] = w
        m = m.to(device)
        _LANCZOS_MATS[key] = m
    return m


def _lanczos_gpu_to(frames, tw, th):
    """[N,H,W,C] -> [N,th,tw,C]: separable lanczos-3 on the GPU, float32,
    self-sized chunks (~2 GB in flight), OOM halves and retries. Results
    collected on the CPU - the _resize_chunked contract."""
    dev = comfy.model_management.get_torch_device()
    n = int(frames.shape[0])
    sh, sw = int(frames.shape[1]), int(frames.shape[2])
    tw, th = int(tw), int(th)
    mh = _lanczos_matrix(sh, th, dev)          # [th, sh]
    mw = _lanczos_matrix(sw, tw, dev)          # [tw, sw]
    per_frame = (sh * sw + th * sw + th * tw) * 3 * 4      # in + mid + out
    chunk = max(1, min(n, int(2_000_000_000 // max(1, per_frame))))
    out, i = [], 0
    while i < n:
        j = min(n, i + chunk)
        try:
            t = frames[i:j, :, :, :3].to(dev, torch.float32)      # [b,H,W,C]
            t = torch.einsum("oh,bhwc->bowc", mh, t)              # H pass
            t = torch.einsum("ow,bhwc->bhoc", mw, t)              # W pass
            out.append(t.clamp_(0.0, 1.0).cpu())
            del t
            i = j
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or chunk <= 1:
                raise
            chunk = max(1, chunk // 2)
            torch.cuda.empty_cache()
            print(f"[PLS] Power Upscale: lanczos (gpu) chunk -> {chunk} "
                  f"(OOM backoff, same recipe as the pixel pass)")
    return torch.cat(out, dim=0) if len(out) > 1 else out[0]


_RESIZE_PER_BATCH = 32   # v555: internal chunk for the GPU fit (VRAM guard)


def _fit_to(image, width, height, resize_method, per_batch, say=False,
            involuntary=False):
    """One fit onto an exact canvas, through the v559/v565 downscale guard.

    involuntary=True marks a fit whose shrink the user never asked for - see
    _fit_method. The Power Upscale pixel stage sets it; an explicit resize
    (Fast Upscale) does not."""
    if int(image.shape[2]) == int(width) and int(image.shape[1]) == int(height):
        return image
    if str(resize_method) == _NO_RESIZE:
        # v568: no filter is allowed to touch these pixels. The canvas was
        # DERIVED from the pixel stage, so the only remainder is the VAE's /8
        # grid - at most 7 px, taken by crop/pad, never by resampling.
        if say:
            print(f"[PLS] Power Upscale: fit=none "
                  f"({int(image.shape[2])}x{int(image.shape[1])} -> "
                  f"{int(width)}x{int(height)}: /8 crop-pad only, "
                  f"zero interpolation - the model owns the pixels)")
        return _crop_to(image, width, height)
    method, note = _fit_method(int(image.shape[2]), int(image.shape[1]),
                               width, height, resize_method,
                               involuntary=involuntary)
    if note and say:
        print(f"[PLS] Power Upscale: fit={method} ({note})")
    if method == "nvidia_rtx_vsr":
        return _vsr_resize(image, width, height)
    if method == "lanczos (cpu)":
        return _lanczos_to(image, width, height)
    if method == "lanczos (gpu)":
        return _lanczos_gpu_to(image, width, height)
    return _resize_chunked(image, width, height, method, "gpu", per_batch)


# v565: the ESRGAN tile we ASK for first. A tile that covers the frame makes
# comfy take its own single-tile branch (utils.py: "handle entire input fitting
# in a single tile") - no out_div, no mask, no div_() at all. Core's OOM backoff
# (tile //= 2) sits behind it, so an over-large ask costs one retry, never a run.
_ESRGAN_TILE_CAP = 1024
_ESRGAN_OVERLAP = 32
# v565: the GPU output buffer is scale^2 times the input chunk. At 4x on a 1536
# frame that is 453 MB PER FRAME - per_batch=8 would ask torch.empty() for 3.6 GB
# in ONE allocation, before a single tile has run, and no tile backoff can rescue
# an allocation that big. So the chunk comes from a budget, not from the widget.
_ESRGAN_OUT_BUDGET = 1_500_000_000


def _model_card(um):
    """A fingerprint of a wired UPSCALE_MODEL. Pure; never raises.

    v580 -- Frank, after three clean runs: "generell frage ich mich, ob die
    externen Upscale-Modelle ueberhaupt greifen". They do. He could not tell,
    because this node read exactly TWO facts off the object it was handed
    (`scale`, `supports_half`) and never said either of them out loud.

    ComfyUI does not pass the FILENAME through -- `UpscaleModelLoader` hands over
    a bare spandrel descriptor. But the descriptor knows what it is, and that is
    enough to recognise your own model: architecture, factor, size, precision.
    A 16.7M-param ESRGAN x4 is not a 0.6M-param Compact x2, and you will know
    which one you wired the moment you read the line.
    """
    if um is None:
        return "(unwired)"
    bits = []
    arch = None
    try:
        a = getattr(um, "architecture", None)
        arch = getattr(a, "name", None) or getattr(a, "id", None) or (a if isinstance(a, str) else None)
    except Exception:
        arch = None
    if not arch:
        arch = type(um).__name__
    bits.append(str(arch))
    try:
        sc = float(getattr(um, "scale", 0.0) or 0.0)
        if sc > 0:
            bits.append(f"x{sc:g}")
    except Exception:
        pass
    try:
        inner = getattr(um, "model", None)
        if inner is not None and hasattr(inner, "parameters"):
            n = sum(int(p.numel()) for p in inner.parameters())
            if n:
                bits.append(f"{n / 1e6:.1f}M params" if n >= 1e5 else f"{n} params")
    except Exception:
        pass
    try:
        bits.append("fp16 ok" if bool(getattr(um, "supports_half", False)) else "fp32 only")
    except Exception:
        pass
    try:
        ci, co = getattr(um, "input_channels", None), getattr(um, "output_channels", None)
        if ci and co:
            bits.append(f"{int(ci)}->{int(co)} ch")
    except Exception:
        pass
    return "  |  ".join(bits)


def _say_model_wires(um_high, um_low, pixel_stage, resize_method):
    """The line that answers 'do my upscale models actually fire?'. Never raises."""
    try:
        stages_have_model = str(pixel_stage) not in ("fit only", "model final")
        final_um = um_low if um_low is not None else um_high
        final_wire = "L wire" if um_low is not None else ("H wire" if um_high is not None else None)
        print("[PLS] Power Upscale: pixel models on the wires:")
        print(f"[PLS]     H wire : {_model_card(um_high)}")
        print(f"[PLS]     L wire : {_model_card(um_low)}")
        if stages_have_model:
            print(f"[PLS]     stages : pixel_stage='{pixel_stage}' -> the stages DO run a "
                  f"pixel model (H for stage-H, L for stage-L; L never inherits H)")
        else:
            print(f"[PLS]     stages : NO model, by design -- pixel_stage='{pixel_stage}' puts "
                  f"it behind the last decode (a pixel model in front of a VAE "
                  f"round-trip is nearly worthless; behind the last one it is "
                  f"everything)")
        if str(pixel_stage) == "fit only":
            print("[PLS]     final  : none -- pixel_stage='fit only' runs no model at all")
        elif final_um is None:
            print("[PLS]     final  : none -- no upscale model is wired")
        else:
            print(f"[PLS]     final  : {final_wire}, {_model_card(final_um)}"
                  f"{'  (resize_method=none -> its output IS the file)' if str(resize_method) == _NO_RESIZE else ''}")
    except Exception as e:
        print(f"[PLS] Power Upscale: could not read the upscale models ({type(e).__name__}: {e}) "
              f"-- telemetry only, the pass itself is unaffected.")


class _TileGrind(Exception):
    """v572: raised from INSIDE the model call when a single tile forward runs
    grind-slow - the WDDM paging that torch's memory counters cannot see. The
    retry loop catches it BEFORE raise_non_oom, halves the tile and redoes the
    CURRENT chunk: the failed attempt costs seconds, not the 242 s Frank's
    chunk 2 paid while v570's after-the-fact watch could only take notes."""


def _grind_verdict(call_s, base_s, armed):
    """v572: the pure grind predicate, exec'd in isolation by the guard.

    A tile forward is grinding when it runs more than 3x the measured
    baseline for this tile size AND takes over 1.0 s absolute (3x of a
    40 ms call is noise, not paging). No baseline yet, or detector
    disarmed (tile at floor / not cuda) -> never fires."""
    if not armed or base_s is None:
        return False
    return float(call_s) > 1.0 and float(call_s) > 3.0 * float(base_s)


def _watch_verdict(dt_s, base_s, peak_b, free_b):
    """v573: the outer watch's pure verdict, exec'd in isolation by the guard.

    'slow'  -> wall clock ran > 2x chunk 1. The one spill signal the driver
               cannot hide (v570's lesson) - THE backoff trigger.
    'tight' -> the peak exceeded the free reading. Under pool quiet (v572)
               free reads ~0 because the POOL holds our blocks, so this is
               NARRATIVE, never a verdict - measured false-SPILLED on a
               22.5 s chunk 2 that ran exactly as fast as chunk 1, costing
               an unnecessary backoff.
    'ok'    -> neither. Precedence: slow wins - a genuinely slow chunk with
               an empty free reading must still back off."""
    if float(dt_s) > 2.0 * float(base_s):
        return "slow"
    if float(peak_b) > float(free_b):
        return "tight"
    return "ok"


def _even_dial(w, h, model_scale, resize_method, dial, span=100):
    """The nearest final_upscale_by ON THE WIDGET'S OWN 0.01 GRID whose canvas
    lands EVEN on both edges - or None if nothing within `span` notches does.

    v593. The v591 note computed 1076/768 = 1.40104..., printed it rounded to
    two decimals ("1.40", the shape the widget has) and then printed the pixel
    count from the UNROUNDED value. So it told Frank

        "on 768x768, 1.40 gives 1076px"

    while his dial was already at 1.40 and giving 1075 - the very number the
    note was warning about. A suggestion the user cannot type is not a
    suggestion; a suggestion that reproduces the fault it warns about is worse
    than silence, because he would have followed it.

    Two laws, and they are the whole fix:

      1. Search ON the grid the dial actually has (0.01, two decimals). What is
         printed must be typeable, exactly as printed.
      2. Ask _final_canvas - the SAME function the pass itself uses - what that
         value produces. Never re-derive the arithmetic beside it. A copy of a
         calculation drifts from the calculation, and the copy is always the
         one that lies.

    Returns (dial_value, width, height) or None. None is a real answer: under
    resize_method='none' the dial is ignored entirely, so no value on any grid
    can move the canvas - and the note must say THAT instead of inventing one.
    """
    # v593: 'none' means the model's own factor IS the canvas (the size law
    # ignores final_by there, and says so). No notch can move it. Without this
    # the search finds the FIRST candidate whose canvas happens to be even -
    # which under 'none' is every candidate, because they all produce the same
    # canvas - and hands back a dial change that changes nothing. Caught by the
    # guard's own sweep, which is what a guard is for.
    if str(resize_method) == _NO_RESIZE:
        return None
    for d in range(1, int(span) + 1):
        for s in (1, -1):
            v = round(float(dial) + s * d * 0.01, 2)
            if v < 0.25 or v > 8.0:
                continue
            cw, ch = _final_canvas(w, h, model_scale, resize_method, v)[:2]
            if cw >= 2 and ch >= 2 and not (int(cw) & 1) and not (int(ch) & 1):
                return (v, int(cw), int(ch))
    return None


def _esrgan_resident(image, upscale_model, width, height, resize_method,
                     per_batch, on_chunk=None, fit_s=None):
    """v564: the model is loaded ONCE and stays resident for the whole pass.
    v565: and the whole pass STAYS ON THE GPU.

    v564 fixed the wrong ping-pong. Moving the model WEIGHTS once instead of 17
    times was worth 89 ms/frame (3448 -> 3359, 2.6%). The TENSORS ping-ponged on
    every single frame, because comfy.utils.tiled_scale declares

        def tiled_scale(..., output_device="cpu", pbar=None)

    and we passed NEITHER. Core's own ImageUpscaleWithModel passes both. So per
    frame at 3072x3072 the mask build, the `o.add_(ps_view * mask_view)`, the
    `out.div_(out_div)` and our own clamp all ran on the CPU, and the result then
    had to be uploaded BACK to the GPU for the fit. MEASURED on Frank's 65-frame
    run: ~8.3 G CPU ops and ~18.3 GB over the PCIe bus, of which ~8.2 GB was a
    pure return trip. That - not the weights - was the 3358 ms/frame.

    v565: output_device IS the compute device, the tile covers the frame (one
    model call per frame instead of four, and comfy's single-tile branch skips
    the entire blend apparatus), the pass is interruptible, and the GPU output
    buffer is PAID FOR in free_memory instead of being wished away. The v560
    fusion (fit each chunk immediately) stays; only the fitted result - the
    small tensor - comes back to the CPU.

    Raises on any API drift - the caller then falls back to the core node."""
    import comfy.model_management as mm
    import comfy.utils as cu

    device = mm.get_torch_device()
    scale = float(getattr(upscale_model, "scale", 4.0))
    n = int(image.shape[0])
    src_h, src_w = int(image.shape[1]), int(image.shape[2])
    chunk = max(1, min(int(per_batch), n))
    # v565: the buffer moved to the GPU, so it has to be SIZED, not assumed.
    out_frame = int(src_h * scale) * int(src_w * scale) * 3 * 4
    fits = max(1, int(_ESRGAN_OUT_BUDGET // max(1, out_frame)))
    if fits < chunk:
        print(f"[PLS] Power Upscale: per_batch {chunk} -> {fits} for the pixel "
              f"pass (the {scale:.0f}x output buffer is {out_frame / 1e6:.0f} MB "
              f"per frame; {_ESRGAN_OUT_BUDGET / 1e9:.1f} GB budget). This does "
              f"not change the result - only how much of it is in flight.")
        chunk = fits
    # v566: the tile is chosen BEFORE the memory ask, because the ask depends on
    # it. v565 grew the tile to 1024 but left comfy's activation constant CALIBRATED
    # TO 512 - so free_memory cleared room for a 512 forward while the 1024 forward
    # wanted ~4x. On Windows that is not an OOM: the driver spills to system RAM
    # over PCIe and grinds. MEASURED as the per-pixel curve 3.2 / 5.2 / 7.3 us/px
    # over tiles 512 / 768 / 1024 on the same model and card.
    # v587: ask for the FRAME first. _ESRGAN_TILE_CAP clamped the first ask at
    # 1024, so a 1104 source ran a 2x2 grid - four model calls per frame plus
    # the whole blend apparatus, for a frame that FIT in one call (measured
    # 2026-07-13: activation est 10.0 GB vs 15.7 GB free; the pass paid 4x the
    # forward pixels and the feather for nothing). The cap was guarding against
    # a spill that the v566 estimate + the backoff right below already handle,
    # loudly and per-card. So on CUDA the first ask IS the frame edge and the
    # estimate talks it down if the card disagrees; without CUDA there is no
    # estimate, so the old cap stays as the first ask. The cap also remains the
    # backoff's business as usual (tile //= 2 lives behind every ask).
    is_cuda = (getattr(device, "type", str(device)) == "cuda")
    edge = max(src_w, src_h)
    tile = max(128, edge if is_cuda else min(int(_ESRGAN_TILE_CAP), edge))
    need = mm.module_size(upscale_model.model)
    need += (tile * tile * 3) * image.element_size() * max(scale, 1.0) * 384.0
    need += chunk * image[0].nelement() * image.element_size()
    # v565: the output buffer now lives on the GPU. It is roughly scale^2 times
    # the input chunk - at 4x that is 16x. Wishing it away is how you OOM.
    need += chunk * out_frame
    if is_cuda:
        try:
            free0 = torch.cuda.mem_get_info(device)[0]
        except Exception:
            is_cuda = False
    mm.free_memory(need, device)          # ONCE, not per chunk
    if is_cuda:
        try:
            free1, total1 = torch.cuda.mem_get_info(device)
            # The measured peak driver of an ESRGAN forward is the pair of
            # upsample feature maps at output resolution (64ch, fp32, in+out).
            # This estimate separated Frank's clean 768 run from the grinding
            # 1024 run on the same 16 GB card - it has earned the job.
            act = (tile * scale) ** 2 * 64 * 4 * 2
            while tile > 256 and act + chunk * out_frame > free1 * 0.85:
                tile = max(256, tile // 2)
                act = (tile * scale) ** 2 * 64 * 4 * 2
                print(f"[PLS] Power Upscale: esrgan tile -> {tile} up front "
                      f"(activation est {act / 1e9:.1f} GB + buffers must fit "
                      f"{free1 / 1e9:.1f} GB free; spilling to system RAM is "
                      f"slower than smaller tiles)")
            torch.cuda.reset_peak_memory_stats(device)
            print(f"[PLS] Power Upscale: vram free {free1 / 1e9:.1f}/"
                  f"{total1 / 1e9:.1f} GB after free_memory (was "
                  f"{free0 / 1e9:.1f} GB; asked {need / 1e9:.1f} GB; activation "
                  f"est {act / 1e9:.1f} GB @ tile {tile})")
        except Exception:
            is_cuda = False
    upscale_model.to(device)
    # v587: the fit gets its own stopwatch. The pass total blends model forward
    # and fit into one number; whether the fit is a rounding error or a real
    # bite depends on the kernel (lanczos = per-frame CPU through comfy's
    # 8-bit PIL round trip; bicubic/area = chunked GPU, antialiased since
    # v568). Measure, then SAY it - the dial stays the user's.
    #
    # v590 (Frank's field crash, 2026-07-14): the stopwatch is OWNED by the
    # CALLER and handed in. v587 declared it HERE and read it in _esrgan_pass -
    # two scopes, one name. Every path carrying an upscale model (model + fit,
    # model only, model final) died on `NameError: fit_s` at the line that was
    # supposed to SPEAK the measurement. A dict, not a float, was always the
    # right shape - it just never crossed the call. Now it does. The default
    # keeps this function callable on its own.
    fit_s = {"s": 0.0} if fit_s is None else fit_s
    try:
        chunks = (n + chunk - 1) // chunk   # v565: the TRUE count, after the clamp

        # v568: fp16 where the model says it is safe. Core casts every tile to
        # fp32 (we copied that), but spandrel carries `supports_half` and Blackwell
        # runs fp16 convs on tensor cores - up to 2x on the RRDB trunk, which is
        # where the whole pixel pass lives. Autocast keeps fp32 master weights, so
        # nothing is quantised: only the matmuls drop. Some ESRGAN weights still
        # overflow, so chunk 1 is CHECKED and a single non-finite value disarms
        # half for the rest of the pass, loudly, with that chunk redone in fp32.
        half = {"on": bool(getattr(upscale_model, "supports_half", False))
                and is_cuda}

        # v572: the in-chunk grind detector. Chunk 2 of Frank's run ground for
        # 241.8 s while the v570 watch - correct but AFTER the fact - could
        # only take notes and fix chunk 3. The detector times every tile
        # forward (synchronized: CUDA is async, an unsynced stopwatch times
        # the launch, not the work) and raises _TileGrind the moment ONE call
        # runs grind-slow. The redo of the current chunk costs seconds.
        # Baseline = the first synced call at the current tile size; any
        # backoff resets it. Disarmed at the 256 floor (nowhere left to go).
        grind = {"armed": is_cuda and tile > 256, "base": None, "hit": 0.0}

        def _model_fn(a):
            # v566: the interrupt lives INSIDE the model call, not only at the
            # chunk boundary. A 92 s chunk answered Frank's six clicks with
            # silence; per tiled_scale call the answer comes in seconds. The
            # exception is not an OOM, so raise_non_oom re-raises it cleanly.
            mm.throw_exception_if_processing_interrupted()
            if grind["armed"]:
                torch.cuda.synchronize(device)
            t0c = time.monotonic()
            if half["on"]:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    r = upscale_model(a.float())
            else:
                r = upscale_model(a.float())
            if grind["armed"]:
                torch.cuda.synchronize(device)
                dc = time.monotonic() - t0c
                if _grind_verdict(dc, grind["base"], grind["armed"]):
                    grind["hit"] = dc
                    raise _TileGrind()
                if grind["base"] is None:
                    grind["base"] = max(1e-6, dc)
            return r

        out, mid = [], None
        # v570: THE SPILL WATCH. Frank's v569 run measured chunk 1 at 32 s and
        # every later chunk at ~100 s - and the telemetry was BLIND to it: the
        # peak/free check ran on chunk 1 only, and the one chunk that fit said
        # "fits, no spill" for a pass that then ground for minutes. Two truths
        # the old check missed: WDDM never OOMs, it spills to system RAM over
        # PCIe (torch's allocated counter cannot see driver-side paging, so
        # peak<free does NOT prove residency); and free VRAM is not static
        # (browser, DWM and the preview nibble between chunks). So every chunk
        # reads free BEFORE it runs, resets the peak counter, and is TIMED -
        # the wall clock is the one spill detector the driver cannot hide
        # from. A chunk that spilled, or ran > 2x chunk 1, says so with
        # numbers, and the REMAINING chunks drop the ESRGAN tile (floor 256;
        # activation falls ~4x per halving). Smaller tiles beat spilling.
        t_c1, spill_backoffs, pool_said = None, 0, False
        for k, (i, j) in enumerate(_chunks(n, chunk)):
            mm.throw_exception_if_processing_interrupted()   # v565: cancellable
            free_i = None
            if is_cuda:
                try:
                    free_i = torch.cuda.mem_get_info(device)[0]
                    torch.cuda.reset_peak_memory_stats(device)
                except Exception:
                    free_i = None
            t_ck = time.monotonic()
            in_img = image[i:j].movedim(-1, -3).to(device)
            while True:
                try:
                    s = cu.tiled_scale(in_img, _model_fn,
                                       tile_x=tile, tile_y=tile,
                                       overlap=_ESRGAN_OVERLAP,
                                       upscale_amount=scale,
                                       output_device=device)   # v565: THE fix
                    break
                except _TileGrind:
                    # v572: caught BEFORE raise_non_oom - this is paging, not
                    # OOM. Halve, reset the baseline, redo THIS chunk.
                    was = tile
                    tile = max(256, tile // 2)
                    grind["armed"] = is_cuda and tile > 256
                    grind["base"] = None
                    _free()
                    print(f"[PLS] Power Upscale: tile forward ran "
                          f"{grind['hit']:.1f}s at tile {was} (grind: >3x the "
                          f"measured baseline - WDDM paging mid-chunk) -> "
                          f"tile {tile}, redoing THIS chunk now instead of "
                          f"donating it to the driver")
                except Exception as exc:
                    mm.raise_non_oom(exc)   # core's contract: only OOM retries
                    tile //= 2
                    grind["armed"] = is_cuda and tile > 256
                    grind["base"] = None
                    _free()
                    if tile < 128:
                        raise
                    print(f"[PLS] Power Upscale: esrgan tile -> {tile} "
                          f"(OOM backoff, core's recipe)")
            part = torch.clamp(s.movedim(-3, -1), min=0.0, max=1.0)
            if half["on"] and i == 0 and not bool(torch.isfinite(part).all()):
                # v568: this model overflows in fp16. Disarm and redo THIS chunk -
                # a silent NaN frame is worse than a slow one.
                half["on"] = False
                print("[PLS] Power Upscale: fp16 produced non-finite values - "
                      "back to fp32 for the whole pass (the model claims "
                      "supports_half, but the weights say otherwise)")
                del s, part
                _free()
                _was_armed = grind["armed"]
                grind["armed"] = False   # v572: the redo runs OUTSIDE the
                # retry net - a _TileGrind here would escape to the core
                # fallback (the slow road). One redo, chunk 1 only: unarmed.
                s = cu.tiled_scale(in_img, _model_fn, tile_x=tile, tile_y=tile,
                                   overlap=_ESRGAN_OVERLAP, upscale_amount=scale,
                                   output_device=device)
                grind["armed"] = _was_armed
                part = torch.clamp(s.movedim(-3, -1), min=0.0, max=1.0)
            if mid is None:
                mid = (int(part.shape[2]), int(part.shape[1]))
            _tf0 = time.monotonic()
            fit = _fit_to(part, width, height, resize_method, _RESIZE_PER_BATCH,
                          say=(i == 0), involuntary=True).cpu()
            fit_s["s"] += time.monotonic() - _tf0
            if on_chunk is not None:
                on_chunk(i, j, fit, chunks)   # v565: the pixel stage is not silent
            out.append(fit)
            del in_img, s, part, fit
            # v572: POOL QUIET - no empty_cache between chunks. ComfyUI runs
            # cudaMallocAsync + DynamicVRAM hooks (Frank's boot log), and
            # v565-v571 handed the pool back to the driver after EVERY chunk.
            # Chunk 2 then had to re-commit everything, and WDDM's second
            # grant is the one that grinds - both measured runs show the same
            # signature (chunk 1 fast, chunk 2 crawling, post-backoff steady).
            # The dels above drop our references so the pool can REUSE the
            # same-size blocks for the next chunk; _free() stays where it
            # belongs: backoffs, the fp16 redo, and the finally.
            dt_ck = time.monotonic() - t_ck
            if t_c1 is None:
                t_c1 = max(1e-6, dt_ck)
            if is_cuda and free_i is not None:
                # v573: THE CLOCK DECIDES, THE MEMORY NARRATES. v572's pool
                # quiet made free~0 the HEALTHY steady state (the pool holds
                # our blocks between chunks) - and the crude peak>free
                # verdict promptly called a 22.5 s chunk "SPILLED" and paid
                # an unnecessary backoff. The wall clock is the one signal
                # the driver cannot fake in either direction; peak/free/pool
                # stay in the line so the numbers stay readable, but only
                # 'slow' (and the in-chunk _TileGrind) may touch the tile.
                try:
                    peak = torch.cuda.max_memory_allocated(device)
                    resv = torch.cuda.memory_reserved(device)
                    verdict = _watch_verdict(dt_ck, t_c1, peak, free_i)
                    if k == 0 or verdict == "slow" or (verdict == "tight"
                                                       and not pool_said):
                        line = (f"[PLS] Power Upscale: chunk {k + 1}/{chunks}"
                                f" peak {peak / 1e9:.1f} GB (pool holds "
                                f"{resv / 1e9:.1f} GB) vs "
                                f"{free_i / 1e9:.1f} GB free, {dt_ck:.1f}s")
                        if verdict == "slow":
                            line += (f" -> {dt_ck / t_c1:.1f}x chunk 1 - "
                                     f"the wall clock IS the spill verdict "
                                     f"(v573); backing off")
                        elif verdict == "tight":
                            line += (" -> free reads near zero because the "
                                     "POOL holds our blocks (v572 pool "
                                     "quiet) and the clock is healthy - "
                                     "narrative, not a verdict")
                            pool_said = True
                        else:
                            line += " -> fits, healthy"
                        print(line)
                    if verdict == "slow" and tile > 256:
                        tile = max(256, tile // 2)
                        grind["armed"] = is_cuda and tile > 256
                        grind["base"] = None   # v572: new size, new baseline
                        spill_backoffs += 1
                        _free()
                        print(f"[PLS] Power Upscale: esrgan tile -> {tile} "
                              f"for the remaining chunks (activation drops "
                              f"~4x per halving; smaller tiles beat "
                              f"spilling)")
                except Exception:
                    pass
        calls = cu.get_tiled_scale_steps(src_w, src_h, tile_x=tile, tile_y=tile,
                                         overlap=_ESRGAN_OVERLAP)
        info = (f"resident, tile={tile} -> {calls} call/frame, "
                f"{'fp16' if half['on'] else 'fp32'}, buffers on {device}"
                + (f", {spill_backoffs} spill backoff(s)"
                   if spill_backoffs else ""))
        return ((torch.cat(out, dim=0) if len(out) > 1 else out[0]),
                mid, info, chunk)   # v571: the CLAMPED chunk, for honest telemetry
    finally:
        upscale_model.to("cpu")           # ONCE
        _free()


def _esrgan_pass(image, upscale_model, width, height,
                 resize_method="lanczos (cpu)", per_batch=8, on_chunk=None,
                 final=False):
    """Stage pixel upscale: the wired UPSCALE_MODEL via core (spandrel), then
    an exact fit onto the /8-snapped stage canvas. v555: the fit method is
    selectable - "lanczos (cpu)" IS the v514 behaviour (byte-identical
    default); the torch methods run chunked on the GPU; nvidia_rtx_vsr goes
    through the Maxine capsule (the stage canvas is already /8-snapped, so
    the SDK contract holds by construction). With no model the stage falls
    back to a plain fit (still a valid, honest resize).

    v565: `on_chunk(i, j, frames)` fires after EVERY finished chunk, on both
    paths. Until v564 this whole pass was structurally mute - no progress, no
    preview, no interrupt - so a 218 s pixel stage looked exactly like a hang.
    It was not hanging. It was working, silently, in the wrong place."""
    n = int(image.shape[0])
    # v563: a stale save (or a hand-edited graph) could still carry per_batch=0,
    # which used to mean "whole batch" - the 14.7 GB path. Never again.
    per_batch = 8 if not per_batch or int(per_batch) < 1 else int(per_batch)
    if upscale_model is None or _MODEL_UPSCALER is None:
        # v565: 'fit only' is chunked too - not for VRAM (a resize is cheap) but
        # so the bar moves, the console counts and the pixel view fills from the
        # first chunk. A fast pass that LOOKS frozen is still a bad pass.
        t0 = time.monotonic()
        chunks = (n + per_batch - 1) // per_batch
        out = []
        for i, j in _chunks(n, per_batch):
            comfy.model_management.throw_exception_if_processing_interrupted()
            part = _fit_to(image[i:j], width, height, resize_method,
                           _RESIZE_PER_BATCH, say=(i == 0))
            if on_chunk is not None:
                on_chunk(i, j, part, chunks)
            out.append(part)
            del part
        image = torch.cat(out, dim=0) if len(out) > 1 else out[0]
        dur = time.monotonic() - t0
        print(f"[PLS] Power Upscale: fit-only {n}f -> {width}x{height} in "
              f"{dur:.1f}s ({dur / max(1, n) * 1000.0:.0f} ms/frame, "
              f"method={resize_method}, no upscale model in this stage)")
        return image

    # v560: ESRGAN and the fit are FUSED per chunk. The old path upscaled the
    # WHOLE batch first and only then fit it down - with 129 frames and a 4x
    # model that intermediate is ~14.7 GB, which is exactly the multi-minute
    # stall (and the OOM) Frank measured. Each chunk is now upscaled AND fit
    # immediately, so the peak holds per_batch frames at the intermediate size,
    # never the whole batch. The result is bit-for-bit the same: both ESRGAN and
    # the fit are per-frame operations.
    t0 = time.monotonic()
    src_w, src_h = int(image.shape[2]), int(image.shape[1])
    path = "resident"
    eff_chunk = min(int(per_batch), n)   # what actually flies per batch
    # v590: the fit stopwatch lives in the scope that READS it (below). Both
    # feed sites mutate this one object: the resident path through the dict we
    # hand it, the core fallback right here. This is the scope law the v587
    # guard THOUGHT it was pinning - it counted the string and never looked at
    # the function it lived in. test_v587 now reads the AST; test_v590_names
    # scans the whole tree for the same disease.
    fit_s = {"s": 0.0}
    try:
        image, mid, path, eff_chunk = _esrgan_resident(image, upscale_model,
                                                       width, height,
                                                       resize_method, per_batch,
                                                       on_chunk=on_chunk,
                                                       fit_s=fit_s)
    except Exception as exc:
        print(f"[PLS] Power Upscale: resident ESRGAN path unavailable "
              f"({exc!r}) - falling back to the core node per chunk")
        path = "core/chunk (buffers on the CPU - the slow road)"
        chunks = (n + per_batch - 1) // per_batch
        out, mid = [], None
        for i, j in _chunks(n, per_batch):
            comfy.model_management.throw_exception_if_processing_interrupted()
            (part,) = _MODEL_UPSCALER.upscale(upscale_model, image[i:j])
            if mid is None:
                mid = (int(part.shape[2]), int(part.shape[1]))
            _tf0 = time.monotonic()
            fit = _fit_to(part, width, height, resize_method, _RESIZE_PER_BATCH,
                          say=(i == 0), involuntary=True).cpu()
            fit_s["s"] += time.monotonic() - _tf0
            if on_chunk is not None:
                on_chunk(i, j, fit, chunks)
            out.append(fit)
            del part, fit
        image = torch.cat(out, dim=0) if len(out) > 1 else out[0]
    dur = time.monotonic() - t0
    peak = (min(int(eff_chunk) or n, n) * mid[0] * mid[1] * 3 * 4 / 1e9) if mid else 0.0
    print(f"[PLS] Power Upscale: esrgan {n}f {src_w}x{src_h} -> {mid[0]}x{mid[1]} "
          f"-> fit {width}x{height} in {dur:.1f}s "
          f"({dur / max(1, n) * 1000.0:.0f} ms/frame, peak ~{peak:.1f} GB "
          f"@ {int(eff_chunk)}f in flight"
          + (f" [per_batch {int(per_batch)} budget-clamped]"
             if int(eff_chunk) < min(int(per_batch), n) else "")
          + f", model {path})")
    if fit_s["s"] > 0.5 and mid:
        _share = fit_s["s"] / max(1e-6, dur) * 100.0
        print(f"[PLS] Power Upscale: of that, the fit ({mid[0]}x{mid[1]} -> "
              f"{int(width)}x{int(height)}) took {fit_s['s']:.1f}s "
              f"({_share:.0f}%, {fit_s['s'] / max(1, n) * 1000.0:.0f} ms/frame, "
              f"method={resize_method})"
              + (". lanczos runs per frame on the CPU through comfy's 8-bit "
                 "PIL round trip; bicubic and area run chunked on the GPU, "
                 "antialiased since v568 - same duty at a fraction of the "
                 "time. The dial stays yours."
                 if str(resize_method) == "lanczos (cpu)" and _share >= 15.0
                 else ""))
    if mid and (mid[0] * mid[1]) > 2.0 * (int(width) * int(height)):
        f = (mid[0] * mid[1] / max(1.0, float(width * height))) ** 0.5
        model_x = mid[0] / max(1.0, float(src_w))          # the model's own factor
        if final:
            # v569: this shrink is the SUPERSAMPLE the user chose - the pass sits
            # behind the last decode, no VAE follows, and the antialiased kernel
            # (v568) carries part of the model's detail down into the file. The
            # stage-NOTE below would name a wall that is not in this room.
            print(f"[PLS] Power Upscale: final pass supersampled {f:.1f}x back "
                  f"onto the dialled canvas ({width}x{height}) through an "
                  f"antialiased kernel - the model's detail lands in the FILE, "
                  f"no VAE follows this pass. resize_method='none' would keep "
                  f"the raw {mid[0]}x{mid[1]} instead.")
            return image
        stage_x = float(width) / max(1.0, float(src_w))    # what the stage asked for
        print(f"[PLS] Power Upscale: NOTE the {model_x:.0f}x model built {f:.1f}x "
              f"more pixels than this {stage_x:.2f}x stage needs, and it cost "
              f"{dur:.0f}s. The waste is NOT the issue - the VAE is. MEASURED: "
              f"after the fit the model's output is ~5x sharper than a plain "
              f"resize (it grips), but that detail sits at ~2.5px, and the Wan VAE "
              f"compresses 8x SPATIALLY - nothing finer than ~8px survives the "
              f"refine's encode, and the sampler repaints what does. Below ~2x "
              f"stage factor a pixel model therefore changes the OUTPUT by almost "
              f"nothing. Three real options: put the model AFTER the refine "
              f"(pixel_stage='model final' - the last stage's wire runs ONCE "
              f"behind the final decode, straight into the file), or set "
              f"upscale_by={model_x:.2f} so the model's factor IS the stage, or "
              f"set pixel_stage='fit only' and keep the {dur:.0f}s.")
    return image




# ── v555: the shared resize machine (moved here FROM ph_fast_upscale so the
# Power Upscale can use it too - the import direction Fast -> PU already
# exists, so ONE source of truth without a cycle) ─────────────────────
# UI label -> comfy.utils.common_upscale method name.
_METHODS = {
    "bicubic": "bicubic",
    "bilinear": "bilinear",
    "area": "area",
    "nearest-exact": "nearest-exact",
    "lanczos (cpu)": "lanczos",
    "lanczos (gpu)": None,    # its own path (_lanczos_gpu_to) - v588: the SAME
                              # windowed sinc, float32 on the card, no 8-bit stop
    "nvidia_rtx_vsr": None,   # its own path (_vsr_resize)
    # v568 (Frank's third ask - and he was right every time): NO interpolation,
    # anywhere. The canvas becomes whatever the PIXEL STAGE produced: the model's
    # own factor with a model wired, the input size without one. Only the VAE's
    # /8 grid is enforced, and by CROPPING (<= 7 px), never by resampling. This
    # is the "models pure, no admixture" switch - in Fast Upscale it hands you
    # the raw model output, in Power Upscale it makes the model the sole author
    # of size.
    "none": None,             # its own path (_crop_to, no resampling at all)
}
_NO_RESIZE = "none"


def _crop_to(frames, tw, th):
    """v568: the ONLY geometry `none` is allowed to do - centre-crop, or
    edge-replicate pad, by at most a /8 remainder. No filter, no resampling,
    no opinion. The caller announces it when it bites."""
    h, w = int(frames.shape[1]), int(frames.shape[2])
    tw, th = int(tw), int(th)
    if w > tw:
        x = (w - tw) // 2
        frames = frames[:, :, x:x + tw, :]
    elif w < tw:
        frames = torch.cat(
            [frames, frames[:, :, -1:, :].repeat(1, 1, tw - w, 1)], dim=2)
    if h > th:
        y = (h - th) // 2
        frames = frames[:, y:y + th, :, :]
    elif h < th:
        frames = torch.cat(
            [frames, frames[:, -1:, :, :].repeat(1, th - h, 1, 1)], dim=1)
    return frames


def _fit_method(src_w, src_h, dst_w, dst_h, method, involuntary=False):
    """v559/v565: pick the method that ACTUALLY fits, and say when we overrule.

    v559 wrote the right sentence and then did not act on it. Its own docstring
    said "bicubic/bilinear ring" - and its code overruled nvidia_rtx_vsr alone
    and waved every ringing kernel through. Frank ran `fit=bicubic` behind a 4x
    model in front of a 2x stage: a 2:1 DOWNSCALE with an interpolator.

    A shrink is not an upscale with a minus sign. bicubic, bilinear and
    nearest-exact are INTERPOLATORS: they sample the source at the new grid
    positions and never look at the pixels in between - and common_upscale calls
    torch.nn.functional.interpolate WITHOUT antialias. Discarding three of every
    four pixels without averaging them first is textbook aliasing. `area` is the
    box filter that averages exactly the pixels it throws away; PIL's `lanczos`
    is a windowed sinc that low-passes correctly in both directions. Ultimate SD
    Upscale - the node this one replaces - fits with PIL LANCZOS for precisely
    this reason. So: 7.4 minutes of ESRGAN edge detail, sampled away by the fit
    that was supposed to keep it.

    nvidia_rtx_vsr is not even a filter, it is a super-resolution EFFECT. A
    smaller target is the wrong tool, never a preference - overruled on ANY
    shrink (the v559 law, unchanged).

    involuntary=True marks a fit whose shrink the USER never chose: the Power
    Upscale pixel stage, where the downscale is a SIDE EFFECT of the model factor
    overshooting the stage. There a ringing kernel is overruled to `area`, loudly.
    On an EXPLICIT resize (Fast Upscale: the user typed the target size) the
    choice is KEPT and warned about, loudly. We correct our own arithmetic; we do
    not overrule a human.

    Returns (method, note) - note is None when there is nothing to say.
    """
    method = str(method)
    if method == "none":          # literal, not _NO_RESIZE: this function is
        return method, None       # extracted and exec'd in isolation by the guards
    if (int(dst_w) * int(dst_h)) >= (int(src_w) * int(src_h)):
        return method, None                       # an upscale, or exactly equal
    f = ((src_w * src_h) / max(1.0, float(dst_w * dst_h))) ** 0.5
    if method == "nvidia_rtx_vsr":
        return "area", (f"vsr is an upscaler, but the target is {f:.1f}x "
                        f"smaller - using area (the right downscale filter)")
    if method in ("bicubic", "bilinear"):
        # v568 AMENDS v565. v565 coerced these to `area` on a shrink because
        # F.interpolate does not antialias by default. The coercion was right
        # about the aliasing and WRONG about the cure: `area` is a box filter,
        # it antialiases by BLURRING. Measured on a 3.6x supersample downscale
        # of real ESRGAN output, `area` keeps 1.24x LESS detail than a windowed
        # kernel - it blurred away exactly the detail the pixel pass was paid to
        # build. _resize_chunked now passes antialias=True, which scales the
        # kernel support to the ratio: no aliasing AND no blur. Kernel kept.
        return method, None
    if method == "nearest-exact":
        # No antialias exists for nearest - it can only alias on a shrink.
        if involuntary:
            return "area", (f"nearest-exact cannot antialias and this fit is a "
                            f"{f:.1f}x downscale you never asked for - using area")
        return method, (f"WARNING nearest-exact cannot antialias and this is a "
                        f"{f:.1f}x downscale - it WILL alias. Pick bicubic "
                        f"(antialiased since v568), area, or lanczos.")
    return method, None                           # area / lanczos: already right


def _final_canvas(w, h, model_scale, resize_method, final_by=1.0):
    """v569/v582: the size law of the FINAL pixel pass (pixel_stage='model final').

    The pass runs BEHIND the last decode, so no VAE follows and no /8 grid
    applies - its detail goes straight to the file. Two outcomes, decided by
    resize_method:

      'none'      -> the output IS the model result: (w*scale, h*scale).
                     final_by is IGNORED here (the caller announces it): 'none'
                     means no kernel exists that could reach any other canvas.
                     With a 1x restoration model that is a pure detail pass at
                     unchanged size - the smoother, no growth.
      any kernel  -> the model overshoots at its NATIVE factor (baked into the
                     weights; a 4x net builds 4x pixels, always), then the
                     antialiased fit (v568) lands it on the USER'S canvas:
                     (w*final_by, h*final_by). final_by=1.0 is the classic
                     supersample back onto the stage canvas; any other value
                     puts the size in the user's hand and leaves the model
                     exactly one job - its imprint. Swapping a 2x for a 4x for
                     an 8x model changes the imprint, never the canvas.

    AMENDED IN v582 (1st amendment): the kernel path used to pin the canvas to
    (w, h), unconditionally. Between that and 'none' the file size was decided
    by everything EXCEPT the user: the model's own factor under 'none', the
    stage plan under a kernel. The dial the user actually reaches for did not
    exist. final_by=1.0 reproduces the old law bit for bit.

    Pure on purpose: the guard exec()s this in isolation. 'none' is the literal
    _NO_RESIZE value (the comment may name the constant; the code may not).
    Returns (target_w, target_h, grows) - grows is True when the pass changes
    the canvas size."""
    w, h = int(w), int(h)
    s = float(model_scale)
    if str(resize_method) == "none":
        tw, th = int(round(w * s)), int(round(h * s))
        return tw, th, (tw != w or th != h)
    f = float(final_by)
    tw, th = int(round(w * f)), int(round(h * f))
    return tw, th, (tw != w or th != h)


def _chunks(n, per_batch):
    step = n if int(per_batch) <= 0 else int(per_batch)
    for i in range(0, n, step):
        yield i, min(i + step, n)


def _resize_chunked(frames, tw, th, ui_method, device_choice, per_batch):
    """Resize on the chosen device, in sub-batches, results collected on the
    CPU. lanczos is PIL-backed = CPU-only: a gpu selection fails LOUD (the
    KJ lesson) instead of silently crawling."""
    method = _METHODS[ui_method]
    if ui_method == "lanczos (gpu)":
        if device_choice == "cpu":
            raise ValueError("Fast Upscale: 'lanczos (gpu)' runs its matmul "
                             "kernels on the GPU - pick 'lanczos (cpu)' for "
                             "cpu, or device=gpu.")
        return _lanczos_gpu_to(frames, tw, th)
    if method == "lanczos":
        if device_choice == "gpu":
            raise ValueError("Fast Upscale: 'lanczos (cpu)' is a PIL path "
                             "and cannot run on the GPU - pick bicubic/"
                             "bilinear/area for gpu, or device=cpu.")
        return _lanczos_to(frames, tw, th)
    dev = (comfy.model_management.get_torch_device()
           if device_choice == "gpu" else torch.device("cpu"))
    out = []
    for i, j in _chunks(int(frames.shape[0]), per_batch):
        t = frames[i:j].movedim(-1, 1).to(dev)
        # v568: a SHRINK through bicubic/bilinear is antialiased PROPERLY.
        # F.interpolate does not antialias by default and comfy's common_upscale
        # never asks it to - which is why v559/v565 overruled those kernels to
        # `area` on a shrink. But `area` is a BOX filter: it antialiases by
        # BLURRING. Measured on a 3.6x supersample downscale of real ESRGAN
        # output: `area` keeps 1.24x LESS detail than a windowed kernel.
        # antialias=True scales the kernel support to the ratio - correct
        # antialiasing AND detail preservation, on the GPU. The overrule can go.
        shrink = (int(tw) * int(th)) < (int(t.shape[3]) * int(t.shape[2]))
        if shrink and method in ("bicubic", "bilinear"):
            t = torch.nn.functional.interpolate(
                t, size=(int(th), int(tw)), mode=method,
                antialias=True, align_corners=False)
        else:
            t = comfy.utils.common_upscale(t, int(tw), int(th), method,
                                           "disabled")
        out.append(t.movedim(1, -1).cpu())
    return torch.cat(out, dim=0)


def _vsr_resize(frames, tw, th):
    """NVIDIA Maxine VideoSuperRes, re-implemented after studying the KJ
    path: the SDK API is SINGLE-FRAME (no batch call exists), frames go in
    as CHW CUDA contiguous, results come back over DLPack. The ULTRA
    context is opened ONCE per run and ALWAYS closed (try/finally - the
    lifecycle KJ also honours). Import failure speaks plainly."""
    try:
        import nvvfx
    except ImportError as exc:
        raise RuntimeError(
            "Fast Upscale: resize_method 'nvidia_rtx_vsr' needs the NVIDIA "
            "Maxine nvvfx library and a compatible RTX GPU (import failed: "
            f"{exc}). Pick another method, or install nvidia-vfx.") from exc
    ctx = nvvfx.VideoSuperRes(nvvfx.effects.QualityLevel.ULTRA)
    sr = ctx.__enter__()
    try:
        sr.output_width = int(tw)    # _target already snapped to /8
        sr.output_height = int(th)
        sr.load()
        out = []
        for k in range(int(frames.shape[0])):   # per-frame: the Maxine API
            f = frames[k].movedim(-1, 0).cuda().contiguous()
            r = sr.run(f).image
            out.append(torch.from_dlpack(r).clone().movedim(0, -1).cpu())
        return torch.stack(out, dim=0)
    finally:
        ctx.__exit__(None, None, None)


_VAE_TILES = ["Off", "512", "640", "768"]


def _vae_ops(vae, vae_tiling):
    """(encode, decode, label) for this stage. v562: optional SPATIAL VAE tiling.

    NEVER temporal. A Wan VAE compresses time 4:1, so a temporal tile would cut
    ACROSS that compression and stitch two independently decoded time windows -
    stutter seams, exactly the artefact the spatial feather exists to avoid.
    `tile_t` is therefore pinned to None (whole stack) wherever it exists.

    Comfy takes PIXEL tiles on encode and LATENT tiles on decode; the signature
    is READ at runtime (inspect) instead of guessed, and anything unexpected
    falls back to the plain path with one honest line."""
    if str(vae_tiling) == "Off":
        return vae.encode, vae.decode, "off"
    enc = getattr(vae, "encode_tiled", None)
    dec = getattr(vae, "decode_tiled", None)
    if enc is None or dec is None:
        print("[PLS] Power Upscale: this VAE has no tiled path - using the plain "
              "encode/decode (vae_tiling ignored)")
        return vae.encode, vae.decode, "off (unsupported)"
    t = int(vae_tiling)

    def _kw(fn, tile, overlap):
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return {}
        kw = {}
        for name, val in (("tile_x", tile), ("tile_y", tile),
                          ("overlap", overlap)):
            if name in params:
                kw[name] = val
        for name in ("tile_t", "overlap_t"):   # the temporal ban, made explicit
            if name in params:
                kw[name] = None
        return kw

    def _encode(x):
        try:
            return enc(x, **_kw(enc, t, max(16, t // 8)))
        except Exception as exc:
            print(f"[PLS] Power Upscale: tiled encode failed ({exc!r}) - plain encode")
            return vae.encode(x)

    def _decode(x):
        lt = max(8, t // 8)
        try:
            return dec(x, **_kw(dec, lt, max(2, lt // 8)))
        except Exception as exc:
            print(f"[PLS] Power Upscale: tiled decode failed ({exc!r}) - plain decode")
            return vae.decode(x)

    return _encode, _decode, f"tiled({t})"


def _vram_note(n, tw, th):
    """v562: say it BEFORE the stage, not after the OOM."""
    try:
        if not torch.cuda.is_available():
            return
        free, _total = torch.cuda.mem_get_info()
        need = n * tw * th * 3 * 4 * 2 / 1e9      # decode buffer + one copy
        if need > (free / 1e9) * 0.8:
            print(f"[PLS] Power Upscale: WARNING the decode buffer for {n} frames "
                  f"at {tw}x{th} is roughly {need:.1f} GB, but only "
                  f"{free / 1e9:.1f} GB is free. Set vae_tiling (512), lower "
                  f"per_batch, or lower tile_size.")
    except Exception:
        pass


def _free():
    """v561: hand the freed blocks back between tiles/stages. A tiled video run
    allocates and releases GB-sized tensors; without this the allocator
    fragments and the NEXT tile OOMs on memory it technically has."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _grid_advice(sw, sh, tile, overlap, nx, ny):
    """v566: turn the coverage warning into NUMBERS the user can act on.

    Frank's live case: tile 1072 on a 1104 canvas. 32 px past the tile edge
    tipped the grid from 1 tile to 4, with 97% overlap - coverage 3.8x, a
    refine 4x more expensive for zero quality. The old warning only offered
    'one big tile', which the s/step measurements show is often the SLOWER
    choice (attention is quadratic in tokens). This helper computes:

      snug  - the smallest /8 tile that keeps the CURRENT grid, coverage ~1.0x.
              plan_axis puts n1 = ceil((size-t)/(t-ov)) + 1 tiles on an axis,
              so n stays put while t >= (size + (n-1)*ov) / n.
      edge  - the largest canvas this tile carries at the SAME tile count:
              n*t - (n-1)*ov. A 1072 tile at 2x2 carries up to 2080 - so a
              1104 canvas pays exactly what a 2080 canvas would.

    Pure integers, no torch - extracted and RUN by test_v566."""
    ov = max(0, min(int(overlap), int(tile) - 1))

    def snug(size, n):
        if n <= 1:
            return int(size)
        t = -(-(int(size) + (n - 1) * ov) // n)       # ceil division
        return ((t + 7) // 8) * 8                      # /8 snap, upward

    s = max(snug(int(sw), int(nx)), snug(int(sh), int(ny)))
    edge_w = int(nx) * int(tile) - (int(nx) - 1) * ov
    edge_h = int(ny) * int(tile) - (int(ny) - 1) * ov
    return s, edge_w, edge_h


def _refine_tiles(model, positive, negative, vae, image, grid, seed, steps,
                  cfg, sampler_name, scheduler, denoise, clock,
                  stage_tag, tile_probe=None, vae_encode=None, vae_decode=None,
                  sig_run=None, rsec=None, peaks=None):
    """Feather-tiled img2img refine over the WHOLE frame stack per tile.
    Each tile: crop [N,th,tw,3] -> vae.encode (a Wan VAE returns ONE 5D
    video latent -> the model keeps time consistent) -> comfy.sample.sample
    (the common_ksampler contract, tile-seeded frame-independently) ->
    vae.decode -> weighted accumulation. The accumulated feather weight is
    exactly 1 everywhere (uls_tile_math invariant), so the division is pure
    float hygiene, never a seam fix.

    v567: the sampler no longer works in silence. Frank's 11:38 run had 462
    mute seconds between 'low tile 1/1' and its done line - the callback fed
    the bar and the probe but never SPOKE. Now every step prints its measured
    duration plus the stage and run ETA, the encode/decode phases feed the
    same clock, and the tick counters are gone (the clock owns the bar).

    v1008: `sig_run` (a list of sigmas) replaces the scheduler: the tile is
    sampled with sample_custom on EXACTLY that run (the wired curve's tail,
    _sigma_run); an empty run means denoise 0 -- the tile is a VAE round trip
    and says so. `rsec` names the rates section: every tile's encode / steps /
    decode run inside a heartbeat phase (HUD + a console tick every 15 s,
    never silent), are learned per megapixel-frame and remembered. `peaks`
    (a dict) collects the measured VRAM peak per phase for the stage line."""
    n, height, width = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    tw, th = grid["tile_w"], grid["tile_h"]
    if sig_run is not None:
        steps = max(0, len(sig_run) - 1)
    else:
        steps = max(1, int(steps))
    node_id = _current_node_id() if rsec is not None else None
    t_mpf = float(tw * th * n) / 1e6
    t_est = {"enc": None, "step": None, "dec": None}
    if rsec is not None:
        _r = _rates_load(rsec)
        for _k in t_est:
            t_est[_k] = (_r[_k] * t_mpf) if _r.get(_k) else None
    peaks = peaks if peaks is not None else {}

    def _peak(kind, ph):
        v = getattr(ph, "peak_gb", None)
        if v is not None:
            peaks[kind] = v if peaks.get(kind) is None else max(peaks[kind], v)
    # v561: ONE tile covering the whole canvas needs no accumulator at all - the
    # feather is 1.0 everywhere, so acc/wacc/multiply/divide are pure waste. On a
    # 129-frame 848x848 stage that is a 1.1 GB CPU tensor plus 278M CPU multiplies
    # plus 278M CPU divides, for a result identical to the decoded tile itself.
    single = (len(grid["tiles"]) == 1 and tw == width and th == height)
    acc = wacc = None
    if not single:
        acc = torch.zeros((n, height, width, 3), dtype=torch.float32)
        wacc = torch.zeros((1, height, width, 1), dtype=torch.float32)
    for t in grid["tiles"]:
        x, y = t["x"], t["y"]
        crop = image[:, y:y + th, x:x + tw, :3]
        # v565: the refine has NEVER been broken down. Frank's interrupted run
        # measured stage=high at 1259.9 s, of which the pixel pass was 218.3 s -
        # so 1041.6 s (83 %) happened inside THIS loop and nobody knows where.
        # Guessing which quarter to optimise is exactly how v564 fixed the wrong
        # ping-pong. Measure first, cut second.
        ti = t["iy"] * grid["nx"] + t["ix"] + 1   # 1-based reading order
        _pk = dict(stage=stage_tag, label=f"{stage_tag} tile {ti}", tile=ti,
                   tiles=len(grid["tiles"]), rect=(x, y, tw, th), announce=False)
        tt = [time.monotonic()]
        if rsec is not None:
            with _Phase("encode", t_est["enc"], clock, node_id, (width, height), **_pk) as _pe:
                latent = (vae_encode or vae.encode)(crop)
            _peak("encode", _pe)
        else:
            latent = (vae_encode or vae.encode)(crop)
        tt.append(time.monotonic())
        clock.measure(f"enc:{stage_tag}", tt[1] - tt[0])
        tseed = uls_tile_math.tile_seed(seed, t["ix"], t["iy"], grid["nx"])
        noise = comfy.sample.prepare_noise(latent, tseed, None)
        print(f"[PLS] Power Upscale:   {stage_tag} tile {ti}/{len(grid['tiles'])} "
              f"({tw}x{th} @ {x},{y})")
        stw = {"last": time.monotonic(), "phase": None}   # v567: per-step stopwatch

        def _open_step(i, _ti=ti, _s=stw, _pk=_pk):
            # v1008: the v1007 heartbeat for the tile path -- the callback
            # fires at a step's END, so the phase of step i+1 opens there.
            if _s["phase"] is not None:
                _s["phase"].__exit__(None, None, None)
                _peak("sample", _s["phase"])
                _s["phase"] = None
            if rsec is not None and i <= steps:
                ph = _Phase("sample", t_est["step"], clock, node_id, (width, height),
                            step=i, steps=steps, **_pk)
                ph.__enter__()
                _s["phase"] = ph

        def _cb(step, x0, x_, total, _ti=ti, _x=x, _y=y, _s=stw):
            now = time.monotonic()
            dt = now - _s["last"]
            _s["last"] = now
            # v567: the step SPEAKS. measure() also pushes the time bar, so
            # console, bar and pane move on the same heartbeat.
            clock.measure(f"step:{stage_tag}", dt)
            seta = clock.eta(stage_tag)
            reta = clock.eta()
            print(f"[PLS] Power Upscale:   {stage_tag} tile {_ti} "
                  f"step {step + 1}/{steps} {dt:.1f}s (stage eta "
                  f"{_fmt_clock(seta or 0)}, run eta ~{_fmt_clock(reta or 0)})")
            if tile_probe is not None:   # v550 (encapsulated in _make_tile_probe)
                tile_probe(stage_tag, _ti, len(grid["tiles"]), step + 1, steps,
                           (_x, _y, tw, th), (width, height), x0,
                           clock.elapsed(), reta)
            _open_step(step + 2)

        _open_step(1)
        try:
            if sig_run is not None and steps == 0:
                # v1008: denoise 0 on the wired curve -- nothing to sample. The
                # tile is a VAE round trip, and the console says so ONCE.
                if ti == 1:
                    print(f"[PLS] Power Upscale:   {stage_tag}: denoise 0 on the wired "
                          f"curve -> no step; the stage is a VAE round trip only")
                samples = latent
            elif sig_run is not None:
                # v1008: the wired curve reaches the TILE refine too -- exactly
                # the run _sigma_run cut from it, no scheduler in between.
                samples = comfy.sample.sample_custom(
                    model, noise, float(cfg), comfy.samplers.sampler_object(sampler_name),
                    torch.tensor([float(v) for v in sig_run], dtype=torch.float32),
                    positive, negative, latent, noise_mask=None, callback=_cb,
                    disable_pbar=True, seed=tseed,
                )
            else:
                samples = comfy.sample.sample(
                    model, noise, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent, denoise=float(denoise),
                    disable_noise=False, start_step=None, last_step=None,
                    force_full_denoise=False, noise_mask=None, callback=_cb,
                    disable_pbar=True, seed=tseed,
                )
        finally:
            _open_step(steps + 99)   # closes whatever phase is open, opens none
        tt.append(time.monotonic())
        if rsec is not None:
            with _Phase("decode", t_est["dec"], clock, node_id, (width, height), **_pk) as _pd:
                px = (vae_decode or vae.decode)(samples)
            _peak("decode", _pd)
        else:
            px = (vae_decode or vae.decode)(samples)
        tt.append(time.monotonic())
        if rsec is not None:
            # v1008: learn per tile -- the next tile (and the next run of this
            # size class) opens with real numbers
            _rates_learn(rsec, t_mpf, tt[1] - tt[0],
                         (tt[2] - tt[1]) / max(1, steps) if steps else None,
                         tt[3] - tt[2],
                         peak_gb=max([v for v in peaks.values() if v is not None] or [0.0]) or None)
            _r = _rates_load(rsec)
            for _k in t_est:
                t_est[_k] = (_r[_k] * t_mpf) if _r.get(_k) else None
        clock.measure(f"dec:{stage_tag}", tt[3] - tt[2])
        if ti == 1:
            # v562 (Fable's catch): Comfy returns the decode on
            # intermediate_device() - CPU unless --gpu-only - so the v561
            # "feather on the GPU" is a no-op in the default setup. MEASURE it.
            print(f"[PLS] Power Upscale:   decode device={px.device} "
                  f"(the feather runs there)")
        del latent, samples, noise            # v561: free the GPU before the fit
        if px.dim() == 5:  # some VAEs return [B,T,H,W,C] -- flatten to frames
            px = px.reshape(-1, px.shape[-3], px.shape[-2], px.shape[-1])
        if int(px.shape[1]) != th or int(px.shape[2]) != tw:
            px = _lanczos_to(px, tw, th)  # guard a non-roundtripping VAE
            # (deliberately stays lanczos: a rare per-tile corrective fit
            #  must never spin up a VSR context)
        px = px[:, :, :, :3].to(torch.float32)

        def _phases(_tt=tt, _ti=ti, _st=steps):
            enc, smp, dec = _tt[1] - _tt[0], _tt[2] - _tt[1], _tt[3] - _tt[2]
            bl = time.monotonic() - _tt[3]
            print(f"[PLS] Power Upscale:   {stage_tag} tile {_ti} done: "
                  f"encode {enc:.1f}s + sample {smp:.1f}s "
                  f"({smp / max(1, _st):.1f}s/step) + decode {dec:.1f}s "
                  f"+ blend {bl:.1f}s = {time.monotonic() - _tt[0]:.1f}s")

        if single:                            # v561: the fast path - no blending
            out = px.cpu()
            del px
            _free()
            _phases()
            clock.push()
            return out
        # v561: weight ON THE DEVICE px already lives on. The old code copied the
        # decoded tile to the CPU FIRST and multiplied there - 278M float ops per
        # tile on the slowest hardware in the box.
        w2 = torch.from_numpy(uls_tile_math.feather_2d(grid, t)).to(
            dtype=torch.float32)[None, :, :, None]
        acc[:, y:y + th, x:x + tw, :] += (px * w2.to(px.device)).cpu()
        wacc[:, y:y + th, x:x + tw, :] += w2
        del px
        _free()
        _phases()
        clock.push()
    return acc / wacc.clamp_min(1e-8)


# ── v549: result preview (PURE ui) ─────────────────────────────────────────────────
# Small temp JPEGs of the finished frames, shown inside the node as a flipbook
# (the ph_save v532/v533 pattern; the JS side is a lean second instance - we
# deliberately do NOT refactor ph_save.js, which is live-OK at v536). The graph
# outputs are untouched, so a run is byte-identical with the toggle On or Off.
# Fully encapsulated: any failure logs ONE line and returns [] - a preview can
# never cost a finished upscale.
_PREVIEW_MAX_EDGE = 896     # long edge of a preview JPEG
_PREVIEW_MAX_FRAMES = 48    # flipbook budget (evenly sampled beyond that)
_PREVIEW_JPEG_Q = 87


def _emit_result_preview(frames, frame_rate, enabled):
    if not enabled:
        return []
    try:
        from PIL import Image
        n = int(frames.shape[0])
        if n <= _PREVIEW_MAX_FRAMES:
            take = list(range(n))
        else:
            take = [round(i * (n - 1) / (_PREVIEW_MAX_FRAMES - 1))
                    for i in range(_PREVIEW_MAX_FRAMES)]
        sub = "pls_pu"
        root = os.path.join(folder_paths.get_temp_directory(), sub)
        os.makedirs(root, exist_ok=True)
        stem = uuid.uuid4().hex[:8]
        fps = float(frame_rate) if frame_rate else 12.0
        lanczos = getattr(Image, "Resampling", Image).LANCZOS  # Pillow 10 safe
        entries = []
        for k, idx in enumerate(take):
            arr = (frames[idx, :, :, :3].clamp(0.0, 1.0).cpu().numpy()
                   * 255.0).astype("uint8")
            pil = Image.fromarray(arr)
            pil.thumbnail((_PREVIEW_MAX_EDGE, _PREVIEW_MAX_EDGE), lanczos)
            name = f"PLS_pu_{stem}_{k:04d}.jpg"
            pil.convert("RGB").save(os.path.join(root, name), "JPEG",
                                    quality=_PREVIEW_JPEG_Q)
            entries.append({"filename": name, "subfolder": sub, "type": "temp",
                            "kind": "image", "fps": fps,
                            "frame": int(idx), "frames": n})
        return entries
    except Exception as exc:  # never let a preview cost a finished upscale
        print(f"[PLS] Power Upscale: result preview skipped ({exc})")
        return []


# ── v550: process view (per-step tile probe) ──────────────────────────────────────
# Streams the tile being refined RIGHT NOW into the node: latent2rgb via the
# model's own latent_rgb_factors (the sampler's decode formula: einsum +
# (x + 1) / 2 -> 0..255, see nodes/uls_sampler.py), throttled, JPEG-b64 over a
# send_sync event. Fully encapsulated: the first failure DISARMS the probe with
# one log line - a preview can never cost a render.
_PROBE_MIN_INTERVAL = 0.12   # min seconds between frames (last step always ships)
_PROBE_MAX_EDGE = 512
_PROBE_JPEG_Q = 80
# v594: the preview may not eat the render. After a few decodes the probe
# compares what IT has spent against the stage clock it already carries (v567)
# and disarms sharp mode if it is past this share. A preview that costs a
# measurable slice of the pass is not a preview, it is a second pass.
_SHARP_MAX_SHARE = 0.05
_SHARP_MIN_SAMPLES = 3


def _sharp_frame(vae, x0):
    """v594: ONE latent frame through the REAL vae - the picture the sampler is
    actually building, at pixel resolution.

    latent2rgb reads the latent's own grid. On Frank's 768 tile that grid is
    96x96 (the WAN vae is /8), and PIL's thumbnail() only ever scales DOWN - so
    the probe shipped a 96px jpeg, the v592 pane blew it up 5.2x, and it looked
    exactly like what it was. No kernel can put back what was never sampled;
    the only way to a sharp preview is to actually decode one.

    Returns [H, W, C] in 0..1. Shape-tolerant: comfy's video vaes hand back
    [B,T,H,W,C] or [B,H,W,C] depending on the family, and a preview must not
    care which.
    """
    lat = x0[:, :, :1] if x0.dim() == 5 else x0[:1]   # ONE latent frame, not 17
    px = vae.decode(lat)
    if px.dim() == 5:                                 # [B,T,H,W,C]
        px = px[0]
    return px[0]                                      # [H,W,C]

_PROBE_EVENT = "polyhedron.pu_tile"


def _offer(clock, pil):
    """v885: hand ONE finished preview frame to the node's own progress slot.

    Core's PreviewImageTuple is (format, PIL.Image, max_edge); the clock keeps
    it for exactly one push (see ph_runclock.offer_preview), so the send rate
    is the PROBE's throttle, never the clock's. Wrapped like every other line
    in this file: a preview may never cost a render.
    """
    if clock is None:
        return
    try:
        clock.offer_preview(("JPEG", pil, _PROBE_MAX_EDGE))
    except Exception:
        pass


def _emit_input_preview(frames, clock=None):
    """v885 -- THE THIRD DOOR into the same pane: what came IN.

    The v565 door (pixel) cannot fire until a chunk is finished; the v550 door
    (tile) cannot fire until the refine has started. Both are LATE by
    construction, so until the first chunk lands the node shows nothing at all
    -- and 'nothing' reads as 'broken', not as 'working'. Frank, 26.08.: "Da
    ist ueberhaupt nichts mehr zu sehen, bis alles fertig ist, weder ein
    Eingangsbild noch diese Box-Chunk-Mechanik."

    This needs no model, no VAE and no latent_rgb_factors -- the input frames
    are already RGB. One frame, once, before any pass. Same encapsulation as
    its two siblings: a failure says itself once and costs nothing.
    """
    node_id = _current_node_id()
    if node_id is None:
        print("[PLS] Power Upscale: input view unavailable (no node id)")
        return
    try:
        from PIL import Image
        h, w = int(frames.shape[1]), int(frames.shape[2])
        n = int(frames.shape[0])
        arr = (frames[0, :, :, :3].to(torch.float32).clamp(0.0, 1.0).cpu()
               .numpy() * 255.0).astype("uint8")
        pil = Image.fromarray(arr)
        pil.thumbnail((_PROBE_MAX_EDGE, _PROBE_MAX_EDGE),
                      getattr(Image, "Resampling", Image).LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, "JPEG", quality=_PROBE_JPEG_Q)
        _offer(clock, pil)
        from server import PromptServer
        PromptServer.instance.send_sync(_PROBE_EVENT, {
            "node": str(node_id), "stage": "input",
            "elapsed": 0, "eta": None,
            "tile": 1, "tiles": 1,
            "step": 1, "steps": n,
            "rect": [0, 0, w, h],     # the source IS the whole canvas
            "canvas": [w, h],
            "jpeg": base64.b64encode(buf.getvalue()).decode("ascii"),
        })
        print(f"[PLS] Power Upscale: input view sent ({w}x{h}, {n} frame"
              f"{'' if n == 1 else 's'}) - the node is alive before the first "
              f"pass finishes.")
    except Exception as exc:
        print(f"[PLS] Power Upscale: input view unavailable ({exc})")


def _make_tile_probe(model, vae=None, sharp=False, clock=None):
    node_id = _current_node_id()   # imported from uls_sampler (house pattern)
    if node_id is None:
        # v552: this used to be a SILENT no-op - the one place a broken chain
        # could hide. Every exit now states itself (measure > believe).
        print("[PLS] Power Upscale: process view unavailable (no node id)")
        return None
    try:
        fmt = model.model.latent_format
        fac = torch.tensor(fmt.latent_rgb_factors, dtype=torch.float32)
        bias = getattr(fmt, "latent_rgb_factors_bias", None)
        bias = (torch.tensor(bias, dtype=torch.float32)
                if bias is not None else torch.zeros(3))
    except Exception as exc:
        print(f"[PLS] Power Upscale: process view unavailable ({exc})")
        return None
    state = {"t": 0.0, "dead": False, "sent": False, "x0w": False,
             # v594: sharp mode's own books. It pays for itself or it leaves.
             "sharp": bool(sharp) and vae is not None,
             "vae_s": 0.0, "vae_n": 0, "said": False, "coarse": False}
    if sharp and vae is None:
        print("[PLS] Power Upscale: process view: 'vae (sharp)' asked for but no "
              "VAE on the wire - falling back to latent2rgb (/8, coarse).")

    print(f"[PLS] Power Upscale: process view armed "
          f"(node={node_id}, event={_PROBE_EVENT}, "
          f"{'vae (sharp)' if state['sharp'] else 'latent2rgb'})")

    def probe(stage, tile, tiles, step, steps, rect, canvas, x0,
              elapsed=0.0, eta=None):   # v567: the pane shares the clock
        if state["dead"]:
            return
        if x0 is None:   # v552: the second former silent path - say it ONCE
            if not state["x0w"]:
                state["x0w"] = True
                print("[PLS] Power Upscale: process view idle (sampler "
                      "callback carries no x0)")
            return
        now = time.monotonic()
        if step < steps and (now - state["t"]) < _PROBE_MIN_INTERVAL:
            return   # throttle mid-run; the final step of a tile always ships
        state["t"] = now
        try:
            from PIL import Image
            if state["sharp"]:
                # v594: a real decode of ONE latent frame. Measured on every
                # event, audited against the stage clock the probe already
                # carries (v567) - if the preview starts eating the render, it
                # leaves. That decision is the code's, not a promise in a doc.
                _t0 = time.monotonic()
                img = _sharp_frame(vae, x0)
                _dt = time.monotonic() - _t0
                state["vae_s"] += _dt
                state["vae_n"] += 1
                arr = (img.to(torch.float32).clamp(0.0, 1.0).cpu().numpy()
                       * 255.0).astype("uint8")
                if not state["said"]:
                    state["said"] = True
                    print(f"[PLS] Power Upscale: process view: vae (sharp) "
                          f"decode {_dt:.2f}s for one frame -> "
                          f"{arr.shape[1]}x{arr.shape[0]} (latent2rgb would be "
                          f"/8 of that). The cost is audited every event; over "
                          f"{int(_SHARP_MAX_SHARE * 100)}% of the pass and it "
                          f"drops back on its own.")
                _share = (state["vae_s"] / elapsed) if elapsed and elapsed > 0 else 0.0
                if (state["vae_n"] >= _SHARP_MIN_SAMPLES
                        and _share > _SHARP_MAX_SHARE):
                    state["sharp"] = False
                    print(f"[PLS] Power Upscale: process view: sharp decodes "
                          f"have cost {state['vae_s']:.1f}s of {elapsed:.0f}s "
                          f"({_share * 100:.0f}%, {state['vae_n']} events) - "
                          f"past the {int(_SHARP_MAX_SHARE * 100)}% budget. "
                          f"Falling back to latent2rgb for the rest of the run: "
                          f"the picture gets coarser, the render does not get "
                          f"slower. A preview that costs a slice of the pass is "
                          f"a second pass.")
            else:
                lat = x0
                if lat.dim() == 5:
                    lat = lat[:, :, 0]          # [B,C,T,H,W] -> first temporal frame
                lat = lat[0].to(torch.float32).cpu()            # [C,H,W]
                rgb = torch.einsum("chw,cf->hwf", lat, fac) + bias
                arr = (((rgb + 1.0) / 2.0).clamp(0.0, 1.0)
                       * 255.0).numpy().astype("uint8")
                if not state["coarse"]:
                    # v594: say the resolution OUT LOUD. The pane will scale this
                    # to the node's width, and a viewer who does not know the
                    # source is 96px reads the softness as a bug in the render.
                    state["coarse"] = True
                    print(f"[PLS] Power Upscale: process view: latent2rgb source "
                          f"is {arr.shape[1]}x{arr.shape[0]} (the latent's own "
                          f"grid, /8) - the pane scales it up, so it will look "
                          f"soft. That is the map, not the render. "
                          f"process_preview='vae (sharp)' decodes the real frame.")
            pil = Image.fromarray(arr)
            pil.thumbnail((_PROBE_MAX_EDGE, _PROBE_MAX_EDGE),
                          getattr(Image, "Resampling", Image).LANCZOS)
            buf = io.BytesIO()
            pil.save(buf, "JPEG", quality=_PROBE_JPEG_Q)
            # v885: the SAME frame also goes to the node's own progress slot,
            # so the picture is visible without opening the pane (the courtesy
            # Core's KSampler has always had). One-shot -- see
            # _RunClock.offer_preview.
            _offer(clock, pil)
            from server import PromptServer
            PromptServer.instance.send_sync(_PROBE_EVENT, {
                "node": str(node_id), "stage": stage,
                "elapsed": int(elapsed),
                "eta": (None if eta is None else int(eta)),
                "tile": int(tile), "tiles": int(tiles),
                "step": int(step), "steps": int(steps),
                "rect": [int(v) for v in rect],
                "canvas": [int(v) for v in canvas],
                "jpeg": base64.b64encode(buf.getvalue()).decode("ascii"),
            })
            if not state["sent"]:   # v552: prove the send side ONCE per run
                state["sent"] = True
                print(f"[PLS] Power Upscale: process view: first frame sent "
                      f"(stage={stage} tile={tile}/{tiles} step={step}/{steps})")
        except Exception as exc:   # first failure disarms - never cost a render
            state["dead"] = True
            print(f"[PLS] Power Upscale: process view disarmed ({exc})")

    return probe


def _make_pixel_probe(clock=None):
    """v565: the SECOND door into the same pane - the pixel stage.

    The v550 probe hangs off the sampler callback, so it cannot fire until the
    refine has started. On Frank's run that was 218 seconds in: the node showed
    nothing, the bar showed nothing, and the pixel stage was doing all the work.
    The complaint was "the preview doesn't come" - the truth was "there is
    nothing to preview yet, and nobody says so".

    Pixel frames are ALREADY RGB, so this door needs no model and no
    latent_rgb_factors: it ships the first frame of each finished chunk over the
    SAME event, tagged stage="pixel". Same encapsulation as v550/v552 - the first
    failure disarms with one line and can never cost a render."""
    node_id = _current_node_id()
    if node_id is None:
        print("[PLS] Power Upscale: pixel view unavailable (no node id)")
        return None
    state = {"dead": False, "sent": False}
    print(f"[PLS] Power Upscale: pixel view armed "
          f"(node={node_id}, event={_PROBE_EVENT})")

    def pixel(stage, chunk, chunks, frames_done, frames_total, frames,
              elapsed=0.0, eta=None):   # v567: same clock, same pane
        if state["dead"]:
            return
        try:
            from PIL import Image
            h, w = int(frames.shape[1]), int(frames.shape[2])
            arr = (frames[0, :, :, :3].to(torch.float32).clamp(0.0, 1.0).cpu()
                   .numpy() * 255.0).astype("uint8")
            pil = Image.fromarray(arr)
            pil.thumbnail((_PROBE_MAX_EDGE, _PROBE_MAX_EDGE),
                          getattr(Image, "Resampling", Image).LANCZOS)
            buf = io.BytesIO()
            pil.save(buf, "JPEG", quality=_PROBE_JPEG_Q)
            _offer(clock, pil)   # v885: also into the node's progress slot
            from server import PromptServer
            PromptServer.instance.send_sync(_PROBE_EVENT, {
                "node": str(node_id), "stage": "pixel",
                "elapsed": int(elapsed),
                "eta": (None if eta is None else int(eta)),
                "tile": int(chunk), "tiles": int(chunks),
                "step": int(frames_done), "steps": int(frames_total),
                "rect": [0, 0, w, h],      # the pixel stage owns the whole canvas
                "canvas": [w, h],
                "jpeg": base64.b64encode(buf.getvalue()).decode("ascii"),
            })
            if not state["sent"]:   # prove the send side ONCE per run (v552)
                state["sent"] = True
                print(f"[PLS] Power Upscale: pixel view: first frame sent "
                      f"(stage={stage} chunk={chunk}/{chunks} "
                      f"frame={frames_done}/{frames_total})")
        except Exception as exc:   # first failure disarms - never cost a render
            state["dead"] = True
            print(f"[PLS] Power Upscale: pixel view disarmed ({exc})")

    return pixel


def _resolve_input(image, video):
    """Exactly one of image / video. A VIDEO unpacks losslessly into
    (frames, audio, frame_rate); a plain IMAGE rides as (frames, None, 0)."""
    if (image is None) == (video is None):
        raise ValueError(
            "⬡ Power Upscale: connect exactly ONE input — either 'image' "
            "or the green 'video' pin (not both, not neither).")
    if video is None:
        return image, None, 0
    if not _HAS_VIDEO_API:
        raise RuntimeError(
            "⬡ Power Upscale: a VIDEO is wired but this ComfyUI build has "
            "no comfy_api VIDEO support — update ComfyUI or feed frames "
            "into 'image' instead.")
    comps = video.get_components()
    return comps.images, comps.audio, comps.frame_rate


def _build_video(frames, audio, frame_rate):
    """VIDEO output: the upscaled frames with the ORIGINAL audio + fps.
    None when no VIDEO API (mirrors the Media Loader convention)."""
    if not _HAS_VIDEO_API or frames is None or int(frames.shape[0]) == 0:
        return None
    rate = frame_rate if frame_rate else Fraction(16)
    if not isinstance(rate, Fraction):
        rate = Fraction(rate)
    return VideoFromComponents(VideoComponents(images=frames, frame_rate=rate,
                                               audio=audio))


def _reject_none_conditioning(positive, negative):
    """v881: refuse a None CONDITIONING before any work is done.

    THE WOUND, from Frank's field log (26.08.), the second ten-minute run in a
    row lost to a late failure. After the pixel pass the refine died in CORE
    with `TypeError: 'NoneType' object is not iterable` at
    comfy/sampler_helpers.py:72, `for c in cond:` -- a CONDITIONING input was
    None.

    WHERE None COMES FROM: our OWN CLIP Text Encode, since v876. Its gate
    skips an encode nobody reads and returns None on that output, by design and
    loudly logged. That was half a promise: a node that MAY return None obliges
    every node reading it to check. This is the other half.

    ComfyUI validates that a required input is WIRED, never that the value on
    the wire is usable -- so a None sails straight through into the sampler.
    """
    for tag, cond in (("positive", positive), ("negative", negative)):
        if cond is None:
            raise ValueError(
                "\u2b21 Power Upscale: '%s' carries None instead of a "
                "CONDITIONING. The most likely source is a Polyhedron CLIP "
                "Text Encode whose encode was SKIPPED because nothing read "
                "that output when it last ran -- its result is then cached as "
                "None. Touch that node (change a character and change it "
                "back) so it re-runs, or wire a conditioning that is actually "
                "produced." % tag)
        try:
            iter(cond)
        except TypeError:
            raise ValueError(
                "\u2b21 Power Upscale: '%s' is not a CONDITIONING list "
                "(got %s). The sampler iterates it on the first line."
                % (tag, type(cond).__name__))


# v889: the three-witness joint probe MOVED to nodes/ph_joint_probe.py, because
# ph_basics needs the same answer (Load VAE must not refuse a 24ch video VAE
# against an H3 model that reports 32). Re-exported here so every caller in this
# file, and test_v880 which pins them, keep working unchanged -- same shape as
# the v576 ph_runclock re-export. THE PROBE ITSELF DID NOT CHANGE.
try:
    from .ph_joint_probe import (_joint_latent_parts,  # noqa: F401
                                 _joint_streams)       # noqa: F401
except ImportError:  # pragma: no cover - direct-run fallback, as elsewhere here
    from ph_joint_probe import (_joint_latent_parts,   # noqa: F401
                                _joint_streams)        # noqa: F401


# ── v1008: ONE refine vocabulary for EVERY model ────────────────────────────
# Frank, 25.09.: "Warum heisst das eigentlich H3 Refine? Das sollte fuer alle
# Modelle brauchbar bleiben ... auch WAN 2.2 ... ebenso SD1." He was right:
# v1004-v1007 built the exact start, the cache, the heartbeat and the order
# choice for the JOINT branch only, under an H3 name. v1008 moves them to
# where every model reaches them:
#   refine_order  after pixel / before pixel / off -- read by EVERY model.
#                 Tile path: 'before pixel' = the pixel model moves behind the
#                 last decode (what pixel_stage='model final' always did);
#                 'off' = no model pass, the pixel path delivers the canvas.
#                 Joint path: as v1004 (the whole-clip refine before/after).
#   audio_stream  keep / denoise -- joint (video+audio) models only.
#   sigmas        read by the tile refine too (exact start on a flow curve,
#                 step slicing on any other), so a Wan lightx2v or an SD
#                 Turbo curve reaches the refine it was built for.
#   caches        the joint latent cache (v1006) + a PIXEL cache for the tile
#                 path's first model pass (RAM-budgeted, never lossy).
#   clock         the v1007 heartbeat, plan and learned rates for both paths,
#                 plus MEASURED peak VRAM per phase (weak cards over long
#                 clips: the node says where the memory went and which dial
#                 relieves it, from the last run's numbers -- never a guess).
#
# THE JOINT BRANCH (history v1004, unchanged in substance): the tile refine
# cannot serve a joint model -- its forward reads the audio half of a packed
# latent and its packed layout hangs on the geometry of the WHOLE clip. So a
# joint model is refined as ONE pass over the whole frame stack as ONE video
# latent, packed with the run's own audio latent (wired in through 'latent'),
# conditioned by the Reference node's output, sampled from the denoise point
# down. Measured on Core 7a131a3a before a line was written:
#   - KSampler.sample (samplers.py:1280) packs a nested latent and its noise
#     flat, runs the ordinary path and unpacks;
#   - denoise_mask is handled PER STREAM (samplers.py:1297-1314), so the audio
#     stream can be frozen with a zero mask while the video is refined;
#   - the VAE's own spatial factor and frame law are READ from it
#     (downscale_ratio / upscale_ratio, sd.py) -- 16x and 17k+5 <-> 5k+2 for
#     the H3 video VAE; the canvas is snapped first so the VAE's own crop never
#     moves the picture; a frame count off the law is said by number.
# No sigma shift is laid on a joint model: _apply_sigma_shift would replace
# ModelSamplingAV (which carries the audio clock, witness C of the joint
# probe) with a plain flow sampler. The curve carries its own shift.
_REFINE_ORDERS = ("after pixel", "before pixel", "off")
_AUDIO_STREAM_MODES = ("keep", "denoise")
_JOINT_SNAP_DEFAULT = 16   # a joint VAE that does not state its factor (H3's is 16)


def _sigma_tail(vals, denoise):
    """The sigmas a refine runs from a FLOW curve. v1006: the refine STARTS
    EXACTLY at sigma = denoise -- that value is inserted as the first point,
    followed by every curve point strictly below it. So the dial never says one
    thing and the run another: 0.30 on a shift-12 Hyperflow curve (1, .994,
    .984, .966, .923, .835, .697, .469, 0) is [0.30, 0] -- ONE step with 30 %
    noise. A start that is not a curve point is said by the caller (off-grid
    for a distilled LoRA). denoise >= 1 runs the whole curve; denoise <= 0 is
    no refine (empty). Pure -- the guard drives it without torch."""
    vals = [float(v) for v in vals]
    if len(vals) < 2:
        return list(vals)
    d = float(denoise)
    if d >= 1.0:
        return list(vals)
    if d <= 0.0:
        return []
    below = [v for v in vals if v < d - 1e-9]
    if not below or below[-1] > 1e-9:
        below.append(0.0)          # the terminal is always the terminal
    return [d] + below


def _sigma_on_grid(vals, denoise):
    """True when the dialled start IS a point of the curve (within 1e-6)."""
    d = float(denoise)
    return any(abs(float(v) - d) < 1e-6 for v in vals)


def _is_flow_curve(vals):
    """A flow-matching curve lives in [0, 1]: sigma IS the noise fraction
    (x = s*noise + (1-s)*image). An eps/v curve (SD1, SDXL) starts near 14.6,
    where 'denoise 0.30' is NOT a sigma. Pure."""
    vals = [float(v) for v in vals]
    return bool(vals) and max(vals) <= 1.0 + 1e-6


def _sigma_run(vals, denoise):
    """(run, how) -- the sigmas a refine runs from ANY wired curve.
    how = 'exact' (flow curve: _sigma_tail, the start IS denoise), 'slice'
    (any other curve: the last round(n * denoise) steps -- the same reading
    Core gives denoise on its own schedules, where sigma is not a noise
    fraction), 'whole' (denoise >= 1 or a one-value list) or 'none'
    (denoise <= 0: nothing to run). Pure."""
    vals = [float(v) for v in vals]
    if len(vals) < 2:
        return list(vals), "whole"
    d = float(denoise)
    if d >= 1.0:
        return list(vals), "whole"
    if d <= 0.0:
        return [], "none"
    if _is_flow_curve(vals):
        return _sigma_tail(vals, d), "exact"
    n = len(vals) - 1
    k = max(1, min(n, int(round(n * d))))
    return vals[-(k + 1):], "slice"


def _vae_spatial(vae, default=8):
    """The VAE's spatial factor, READ from it (Core's downscale_ratio: an int,
    or (time_law, h, w) for a video VAE). `default` when it states none."""
    r = getattr(vae, "downscale_ratio", None)
    try:
        if isinstance(r, (tuple, list)) and len(r) >= 2:
            return max(1, int(r[1]))
        if isinstance(r, (int, float)) and not isinstance(r, bool) and r > 0:
            return max(1, int(r))
    except Exception:
        pass
    return int(default)


def _vae_frame_law(vae, n):
    """(latent frames, frames back) for n input frames on this VAE, READ from
    its own time law (downscale_ratio[0] / upscale_ratio[0]; H3: 17k+5 <->
    5k+2, Wan: 4k+1). (n, n) when the VAE has no time law (a still VAE)."""
    n = int(n)
    d = getattr(vae, "downscale_ratio", None)
    u = getattr(vae, "upscale_ratio", None)
    try:
        if (isinstance(d, (tuple, list)) and isinstance(u, (tuple, list))
                and callable(d[0]) and callable(u[0])):
            t = int(d[0](n))
            return t, int(u[0](t))
    except Exception:
        pass
    return n, n


def _snap_to(w, h, snap):
    """The largest /snap canvas inside (w, h), never below one cell -- the VAE
    would crop to it anyway (centre crop); snapping first keeps the crop at 0."""
    snap = max(1, int(snap))
    return max(snap, int(w) // snap * snap), max(snap, int(h) // snap * snap)


def _patch_count(model):
    """How many weight patches (LoRAs, accelerators) the wired model carries.
    0 means the raw loader output is on the wire -- the refine then runs
    WITHOUT the sampler's Turbo/Hyperflow chain, which is what happened in
    Frank's 24.09. workflow. Read from the patcher, never inferred."""
    try:
        return len(getattr(model, "patches", {}) or {})
    except Exception:
        return -1


_PIXEL_WHERE = {"behind": "pixel model behind the last decode",
                "front": "pixel model in front of the VAE",
                "none": "no pixel model"}


def _joint_verdict(mode, on, sigmas_n, steps_run, start_sigma, audio_mode,
                   patches, frames_in, frames_out, why="", where="none"):
    """ONE line for the frontend: what the run did with the joint model. Pure."""
    if not on or steps_run <= 0:
        return ("Joint refine OFF (%s) -- pixel path only · %s"
                % (why or mode, _PIXEL_WHERE.get(where, where)))
    curve = ("%d-step curve" % sigmas_n) if sigmas_n > 0 else "scheduler"
    tail = ""
    if frames_out != frames_in:
        tail = " · %d frames in, %d out (VAE frame law)" % (frames_in, frames_out)
    pch = ("%d patches" % patches) if patches > 0 else "NO patches (raw model!)"
    return ("Joint refine %s: %d steps from σ %.2f (%s) · audio %s · "
            "model carries %s · %s%s" % (mode, steps_run, start_sigma, curve,
                                             "kept" if audio_mode == "keep" else "denoised",
                                             pch, _PIXEL_WHERE.get(where, where), tail))


def _tile_verdict(order, n_stages, tiles, steps, start_sigma, curve_n, where, why=""):
    """ONE line for the frontend: what the run did with an ordinary model. Pure."""
    if n_stages <= 0:
        return ("Refine OFF (%s) -- pixel path only · %s"
                % (why or order, _PIXEL_WHERE.get(where, where)))
    curve = ((" from σ %.2f (%d-step curve)" % (start_sigma, curve_n))
             if curve_n > 0 else " (scheduler)")
    return ("Tile refine %s: %d stage%s · %d tile%s · %d step%s%s · %s"
            % (order, n_stages, "" if n_stages == 1 else "s", tiles,
               "" if tiles == 1 else "s", steps, "" if steps == 1 else "s", curve,
               _PIXEL_WHERE.get(where, where)))


# ── v1006/v1008: the caches -- ONE entry each, keyed on CONTENT ─────────────
# Frank's 24.09. field run: ESRGAN 351 s + encode 127 s before the first refine
# step, and every denoise/cfg/seed experiment on the SAME clip paid them again
# (ComfyUI re-runs the whole node when any dial moves). What a refine starts
# from depends on nothing but the input frames, the pixel wire and the canvas
# -- so it is remembered on the CPU under a fingerprint of exactly those:
#   _LATENT_CACHE  the joint path's video latent (~20 MB) -- a hit skips the
#                  pixel pass AND the encode (v1006).
#   _PIXEL_CACHE   the tile path's FIRST model pass (v1008) -- a hit skips the
#                  ESRGAN pass of stage 1. Full frames are big (121 frames at
#                  1872x1072 are 2.9 GB in float32), so the store is BUDGETED
#                  against free system RAM and kept at full precision or not
#                  at all -- the cache never trades quality for a hit.
# Nothing else is cached: the sampling and the decode depend on the dials. The
# fingerprint reads the frames (shape, dtype, stripe sums and a strided byte
# sample) -- never their id -- so a new clip of the same size can never hit.
_LATENT_CACHE = {"key": None, "vid_lat": None, "note": ""}
_PIXEL_CACHE = {"key": None, "frames": None, "note": ""}
_PIXEL_CACHE_SHARE = 0.35   # at most this share of the FREE system RAM


def _content_fingerprint(frames):
    """Content fingerprint of a frame stack, cheap and deterministic."""
    import hashlib
    try:
        f = frames
        h = hashlib.md5()
        h.update(repr((tuple(f.shape), str(f.dtype))).encode())
        n = int(f.shape[0])
        for i in range(0, n, max(1, n // 8)):
            h.update(repr(float(f[i].double().sum().item())).encode())
        sample = f[::max(1, n // 6), ::13, ::11, :].contiguous().to(torch.float32).cpu().numpy()
        h.update(sample.tobytes())
        return h.hexdigest()
    except Exception as exc:  # pragma: no cover - a fingerprint must never break a run
        return "nofp-%s-%s" % (type(exc).__name__, time.monotonic())


def _wire_id(um):
    """The pixel wire as a key part: class, scale and the loaded object."""
    return ("none" if um is None else
            "%s:%s:%s" % (type(getattr(um, "model", um)).__name__,
                          getattr(um, "scale", "?"), id(um)))


def _cache_key(frames, vae, um, canvas, resize_method):
    """The fingerprint the cached LATENT is filed under: frames + VAE + the
    pixel wire + the canvas it lands on. `um` may be None (no pixel pass)."""
    vae_id = "%s:%s" % (type(getattr(vae, "first_stage_model", vae)).__name__,
                        getattr(vae, "latent_channels", "?"))
    return "|".join([_content_fingerprint(frames), vae_id, _wire_id(um),
                     "%dx%d" % (int(canvas[0]), int(canvas[1])), str(resize_method)])


def _pixel_cache_key(frames, um, canvas, resize_method, kind):
    """The fingerprint a cached PIXEL pass is filed under: frames + the wire +
    the canvas + the fit + the pass kind. No VAE -- a pixel pass never meets one."""
    return "|".join([_content_fingerprint(frames), _wire_id(um),
                     "%dx%d" % (int(canvas[0]), int(canvas[1])), str(resize_method),
                     str(kind)])


def _latent_cache_get(key):
    if key is not None and _LATENT_CACHE["key"] == key and _LATENT_CACHE["vid_lat"] is not None:
        return _LATENT_CACHE["vid_lat"]
    return None


def _latent_cache_put(key, vid_lat, note=""):
    try:
        _LATENT_CACHE["key"] = key
        _LATENT_CACHE["vid_lat"] = vid_lat.detach().to("cpu").clone()
        _LATENT_CACHE["note"] = note
    except Exception:
        _LATENT_CACHE["key"] = None
        _LATENT_CACHE["vid_lat"] = None


def _ram_available():
    """Free system RAM in bytes, or None when it cannot be read."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return None


def _pixel_cache_get(key):
    if key is not None and _PIXEL_CACHE["key"] == key and _PIXEL_CACHE["frames"] is not None:
        return _PIXEL_CACHE["frames"]
    return None


def _pixel_cache_put(key, frames, note="", avail=None):
    """(stored, why). Full precision or nothing: the stack is kept as it is
    (a CPU clone) when it fits _PIXEL_CACHE_SHARE of the free RAM; otherwise
    the OLD entry is dropped too (it would only hold RAM for a clip that is
    gone) and the reason is returned for the console."""
    try:
        need = int(frames.numel()) * int(frames.element_size())
    except Exception:
        return False, "size unreadable"
    avail = _ram_available() if avail is None else avail
    _PIXEL_CACHE["key"] = None
    _PIXEL_CACHE["frames"] = None
    if avail is None:
        return False, "free RAM unreadable (no psutil) -- not stored"
    if need > _PIXEL_CACHE_SHARE * avail:
        return False, ("%.1f GB would exceed %d %% of the %.1f GB free RAM -- not stored"
                       % (need / 1e9, int(_PIXEL_CACHE_SHARE * 100), avail / 1e9))
    try:
        _PIXEL_CACHE["frames"] = frames.detach().to("cpu").clone()
        _PIXEL_CACHE["key"] = key
        _PIXEL_CACHE["note"] = note
        return True, "%.1f GB kept in RAM" % (need / 1e9)
    except Exception as exc:
        _PIXEL_CACHE["key"] = None
        _PIXEL_CACHE["frames"] = None
        return False, "store failed (%s)" % type(exc).__name__


# ── v1007/v1008 clock -- v1010: MOVED to nodes/ph_progress.py ───────────────
# The heartbeat, the learned rates, the plan, the VRAM peaks and the memory
# note are the same code, at a shared address, so every long-running node of
# the suite tells time the way this one does (Frank, 25.09.: "fuer unsere
# Nodes, die ein wenig laenger brauchen ... dass man sieht, was gerade
# laeuft"). Re-exported here, the v576 ph_runclock pattern: every caller in
# this file keeps its names.
try:
    from .ph_progress import (_HEARTBEAT_S, _RATES_EMA, _RATES, _rates_section,  # noqa: F401
                              _rates_load, _rates_learn, _phase_plan, _vram_total_gb,
                              _vram_peak_reset, _vram_peak_gb, _memory_note, _Phase,
                              _peak_line)
except ImportError:  # pragma: no cover - direct-run fallback, as elsewhere here
    from ph_progress import (_HEARTBEAT_S, _RATES_EMA, _RATES, _rates_section,  # noqa: F401
                             _rates_load, _rates_learn, _phase_plan, _vram_total_gb,
                             _vram_peak_reset, _vram_peak_gb, _memory_note, _Phase,
                             _peak_line)


def _joint_refine(model, positive, negative, vae, frames, latent, sigmas, seed,
                  steps, cfg, sampler_name, scheduler, denoise, audio_mode,
                  clock, probe=None, mute=True, cached=None, vae_tiling="Off"):
    """The whole-clip refine on a joint (video+audio) model. Returns
    (frames_out, info) -- info carries steps_run / start_sigma / sigmas_n /
    frames_out for the verdict. Raises only on a broken wire it cannot heal
    (said by name); everything else is measured and printed."""
    import comfy.samplers as _cs
    from comfy.nested_tensor import NestedTensor
    snap = _vae_spatial(vae, _JOINT_SNAP_DEFAULT)
    if cached is not None:
        # v1006: (vid_lat, n_in, sw, sh) from the cache -- no frames needed
        vid_lat_c, n_in, sw, sh = cached
        h_in, w_in = sh, sw
        src = None
    else:
        vid_lat_c = None
        n_in, h_in, w_in = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        sw, sh = _snap_to(w_in, h_in, snap)
        src = frames
    if src is not None and (sw, sh) != (w_in, h_in):
        print(f"[PLS] Power Upscale: joint refine: canvas {w_in}x{h_in} is not /{snap} "
              f"-> fitted to {sw}x{sh} (the VAE would centre-crop it otherwise)")
        src = _lanczos_to(frames, sw, sh)
    t_lat, n_back = _vae_frame_law(vae, n_in)
    if n_back != n_in:
        print(f"[PLS] Power Upscale: joint refine: {n_in} frames encode to {t_lat} latent "
              f"frames and decode to {n_back} (the VAE's frame law) -- the last "
              f"{n_in - n_back} frame(s) do not survive the round trip")
    # -- the audio stream comes from the run's own latent --
    parts = _latent_parts(latent["samples"]) if latent is not None else []
    if len(parts) < 2:
        raise ValueError("⬡ Power Upscale: the joint refine needs the run's packed "
                         "AV latent on 'latent' (the Sampler's LATENT output) to "
                         "carry the audio stream -- got %s." %
                         ("nothing" if latent is None else "a single-stream latent"))
    audio_lat = parts[1]
    # -- the sigmas: the wired curve's run, or Core's own slicing --
    if sigmas is not None:
        curve = [float(v) for v in sigmas.flatten().tolist()]
        run, how = _sigma_run(curve, denoise)
        sig = torch.tensor(run, dtype=torch.float32)
        sigmas_n = max(0, len(curve) - 1)
        if not run:
            raise ValueError("⬡ Power Upscale: joint refine: denoise %.2f leaves "
                             "no step on the curve." % float(denoise))
        print(f"[PLS] Power Upscale: joint refine: curve {sigmas_n} steps, denoise "
              f"{float(denoise):.2f} -> {len(run) - 1} step(s) from σ "
              f"{run[0]:.3f} ({', '.join('%.3f' % v for v in run)})"
              + ("" if how != "exact" or _sigma_on_grid(curve, denoise) else
                 f" -- the start is NOT a point of the curve (nearest: "
                 f"{min(curve, key=lambda v: abs(v - float(denoise))):.3f}); a "
                 f"distilled LoRA was trained on its grid, so judge the result by eye")
              + (" -- step slicing (not a flow curve: sigma is not a noise fraction)"
                 if how == "slice" else ""))
    else:
        ks = _cs.KSampler(model, steps=max(1, int(steps)),
                          device=comfy.model_management.get_torch_device(),
                          sampler=sampler_name, scheduler=scheduler,
                          denoise=float(denoise), model_options=model.model_options)
        sig = ks.sigmas
        sigmas_n = 0
        print(f"[PLS] Power Upscale: joint refine: scheduler={scheduler} steps={int(steps)} "
              f"denoise {float(denoise):.2f} -> {max(0, len(sig) - 1)} step(s) from "
              f"σ {float(sig[0]):.3f}")
    steps_run = max(0, int(sig.shape[-1]) - 1)
    if steps_run == 0:
        raise ValueError("⬡ Power Upscale: the joint refine has no step to run "
                         "(denoise %.2f on this curve) -- the decision above should "
                         "have turned it off." % float(denoise))
    # -- v1007: the PLAN, before the first silent second --
    node_id = _current_node_id()
    mpf = float(sw * sh * n_in) / 1e6
    rsec = _rates_section("joint", model)
    rates = _rates_load(rsec)
    plan_lines, est = _phase_plan(mpf, steps_run, vid_lat_c is not None, rates)
    print(f"[PLS] Power Upscale: === JOINT REFINE (video+audio model) === {n_in} frames "
          f"{sw}x{sh} ({mpf:.0f} megapixel-frames) -> ONE video latent, cfg={float(cfg):.2f} "
          f"sampler={sampler_name} audio={audio_mode} model patches={_patch_count(model)}")
    for ln in plan_lines:
        print(f"[PLS] Power Upscale:   {ln}")
    total_gb = _vram_total_gb()
    mnote = _memory_note(rates, mpf, total_gb,
                         "refine_order='before pixel' (refines at the input size), "
                         "vae_tiling 512, fewer frames per clip")
    if mnote:
        print(f"[PLS] Power Upscale: {mnote}")
    v_enc, v_dec, v_label = _vae_ops(vae, vae_tiling)
    # -- encode the whole clip as ONE video latent --
    tt = [time.monotonic()]
    enc_s = None
    peaks = []
    if vid_lat_c is not None:
        vid_lat = vid_lat_c
        tt.append(time.monotonic())
        print(f"[PLS] Power Upscale: joint encode: video latent {tuple(vid_lat.shape)} "
              f"REUSED from the cache (same frames, pixel wire and canvas as the "
              f"previous run -- the encode is skipped) + audio latent "
              f"{tuple(audio_lat.shape)}")
    else:
        with _Phase("encode", est["enc"], clock, node_id, (sw, sh)) as _pe:
            with _MuteInfoLogs(mute, label="Power Upscale"):
                vid_lat = v_enc(src[:, :, :, :3])
        peaks.append(("encode", _pe.peak_gb))
        tt.append(time.monotonic())
        enc_s = tt[1] - tt[0]
        clock.measure("enc:joint", enc_s)
        if vid_lat.dim() == 4:          # a still-VAE would answer [N,C,H,W]; a video VAE answers 5D
            vid_lat = vid_lat.movedim(0, 1).unsqueeze(0)
        print(f"[PLS] Power Upscale: joint encode done in {_fmt_clock(enc_s)} (vae={v_label}) "
              f"-> video latent {tuple(vid_lat.shape)} + audio latent {tuple(audio_lat.shape)}"
              + (f" (planned ~{_fmt_clock(est['enc'])})" if est["enc"] else ""))
    vid_lat_keep = vid_lat.detach().to("cpu")   # v1006: handed back for the cache
    packed = NestedTensor([vid_lat, audio_lat.to(vid_lat.dtype)])
    # -- noise + the per-stream denoise mask --
    noise = comfy.sample.prepare_noise(packed, int(seed), None)
    mask = None
    if str(audio_mode) == "keep":
        mask = NestedTensor([torch.ones((1, 1) + tuple(vid_lat.shape[2:]), dtype=torch.float32),
                             torch.zeros((1, 1) + tuple(audio_lat.shape[2:]), dtype=torch.float32)])
    sampler = _cs.sampler_object(sampler_name)
    stw = {"last": time.monotonic(), "phase": None, "peak": None}

    def _open_step(i):
        # v1007: every step gets its own heartbeat; the callback fires at a
        # step's END, so the phase for step i+1 opens there.
        if stw["phase"] is not None:
            stw["phase"].__exit__(None, None, None)
            pk = stw["phase"].peak_gb
            if pk is not None:
                stw["peak"] = pk if stw["peak"] is None else max(stw["peak"], pk)
            stw["phase"] = None
        if i <= steps_run:
            ph = _Phase("sample", est["step"], clock, node_id, (sw, sh),
                        step=i, steps=steps_run)
            ph.__enter__()
            stw["phase"] = ph

    def _cb(step, x0, x_, total):
        now = time.monotonic()
        dt = now - stw["last"]
        clock.measure("step:joint", dt)
        stw["last"] = now
        print(f"[PLS] Power Upscale:   joint sample step {step + 1}/{steps_run} done in "
              f"{_fmt_clock(dt)}"
              + (f" (planned ~{_fmt_clock(est['step'])})" if est["step"] else "")
              + f" -- stage left ~{_fmt_clock(clock.eta('joint') or 0)}, run left "
              f"~{_fmt_clock(clock.eta() or 0)}")
        if probe is not None:
            probe("joint", 1, 1, step + 1, steps_run, (0, 0, sw, sh), (sw, sh),
                  _joint_video_half(x0), clock.elapsed(), clock.eta())
        _open_step(step + 2)

    print(f"[PLS] Power Upscale: joint sample begin -> {steps_run} step(s) on "
          f"{tuple(vid_lat.shape)} from σ {float(sig[0]):.3f}")
    _open_step(1)
    try:
        with _MuteInfoLogs(mute, label="Power Upscale"):
            out = comfy.sample.sample_custom(model, noise, float(cfg), sampler, sig,
                                             positive, negative, packed,
                                             noise_mask=mask, callback=_cb,
                                             disable_pbar=True, seed=int(seed))
    finally:
        _open_step(steps_run + 99)   # closes whatever phase is open, opens none
    peaks.append(("sample", stw["peak"]))
    tt.append(time.monotonic())
    vid_out = _joint_video_half(out)
    with _Phase("decode", est["dec"], clock, node_id, (sw, sh)) as _pd:
        with _MuteInfoLogs(mute, label="Power Upscale"):
            px = v_dec(vid_out)
    peaks.append(("decode", _pd.peak_gb))
    tt.append(time.monotonic())
    clock.measure("dec:joint", tt[3] - tt[2])
    del noise, packed, out, vid_lat
    _free()
    if px.dim() == 5:
        px = px.reshape(-1, px.shape[-3], px.shape[-2], px.shape[-1])
    px = px[:, :, :, :3].to(torch.float32).cpu()
    print(f"[PLS] Power Upscale: joint refine done: encode {tt[1] - tt[0]:.1f}s + "
          f"sample {tt[2] - tt[1]:.1f}s ({(tt[2] - tt[1]) / max(1, steps_run):.1f}s/step) "
          f"+ decode {tt[3] - tt[2]:.1f}s -> {int(px.shape[2])}x{int(px.shape[1])} "
          f"x{int(px.shape[0])} frames")
    pl = _peak_line("joint refine", peaks, total_gb)
    if pl:
        print(pl)
    top = max([v for _k, v in peaks if v is not None] or [0.0]) or None
    # v1007: learn -- the next run of this size class opens with real numbers
    learned = _rates_learn(rsec, mpf, enc_s, (tt[2] - tt[1]) / max(1, steps_run),
                           tt[3] - tt[2], peak_gb=top)
    print(f"[PLS] Power Upscale: joint rates learned (s per megapixel-frame, {rsec}): "
          f"encode {learned['enc'] if learned['enc'] is None else round(learned['enc'], 3)} · "
          f"step {round(learned['step'], 3) if learned['step'] else None} · "
          f"decode {round(learned['dec'], 3) if learned['dec'] else None} "
          f"({learned['runs']} run(s); remembered in the user directory)")
    clock.push()
    return px, {"steps_run": steps_run, "start_sigma": float(sig[0]) if steps_run else 0.0,
                "sigmas_n": sigmas_n, "frames_out": int(px.shape[0]),
                "vid_lat": vid_lat_keep, "n_in": n_in, "canvas": (sw, sh),
                "cached": vid_lat_c is not None}


class ULSPowerUpscale:
    """⬡ Polyhedron Power Upscale — MoE-aware tiled upscaler (see module doc)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model refining the tiles. "
                                               "In High + Low this is the HIGH-noise expert "
                                               "(structure); the LOW expert plugs into 'model_low'."}),
                "positive": ("CONDITIONING", {"tooltip": "Conditioning to include (shared by both stages)."}),
                "negative": ("CONDITIONING", {"tooltip": "Conditioning to exclude (shared by both stages)."}),
                "vae": ("VAE", {"tooltip": "The VAE encoding/decoding every tile. A Wan VAE turns the "
                                           "whole frame stack of a tile into ONE 5D video latent, so the "
                                           "video model itself keeps time consistent inside the tile."}),
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Pixel upscale model for the stage pre-pass "
                                                              "(ESRGAN family via core/spandrel). In High + Low "
                                                              "this serves the HIGH stage; the LOW stage uses "
                                                              "'upscale_model_low' when wired, else this one."}),
                # ── Mode — the fundamental Single/Dual decision, kept at the top ──
                "dual_moe": ("BOOLEAN", {"default": False,
                                         "label_on": "High + Low", "label_off": "Single",
                                         "tooltip": "Upscaling architecture. "
                                                    "Single: one stage — ESRGAN x upscale_by, then a tiled "
                                                    "refine on 'model'. "
                                                    "High + Low: the mixture-of-experts STAGE CHAIN — stage H "
                                                    "(the HIGH-noise expert, 'model') shapes structure at "
                                                    "upscale_by/denoise/steps/cfg, then stage L (the LOW-noise "
                                                    "expert, 'model_low') polishes detail at the *_low values, "
                                                    "each stage with its own upscale model. This is the correct "
                                                    "MoE translation for refine-strength denoises: a sigma-"
                                                    "boundary split (the sampler's Handoff) would hand the HIGH "
                                                    "expert zero steps below the boundary."}),
                "upscale_by": ("FLOAT", {"default": 1.10, "min": 1.0, "max": 8.0, "step": 0.05, "round": 0.01,
                                         "tooltip": "Stage-H size factor (Single: the whole factor). The stage "
                                                    "canvas snaps to the VAE /8 grid."}),
                "denoise": ("FLOAT", {"default": 0.19, "min": 0.0, "max": 1.0, "step": 0.01,
                                      "tooltip": "Stage-H refine strength. ~0.2 keeps the input and adds "
                                                 "detail; higher re-imagines."}),
                "steps": ("INT", {"default": 3, "min": 1, "max": 10000,
                                  "tooltip": "Stage-H steps per tile (Lightning/distilled experts need few)."}),
                "cfg": ("FLOAT", {"default": 1.6, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01,
                                  "tooltip": "Stage-H CFG. Distilled / Lightning LoRAs live near 1."}),
                "upscale_by_low": ("FLOAT", {"default": 1.30, "min": 1.0, "max": 8.0, "step": 0.05, "round": 0.01,
                                             "tooltip": "High + Low: stage-L size factor (total = product of both)."}),
                "denoise_low": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "High + Low: stage-L refine strength (detail polish)."}),
                "steps_low": ("INT", {"default": 5, "min": 1, "max": 10000,
                                      "tooltip": "High + Low: stage-L steps per tile."}),
                "cfg_low": ("FLOAT", {"default": 1.9, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01,
                                      "tooltip": "High + Low: stage-L CFG."}),
                # ── Universal controls ──
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": True,
                                 "tooltip": "Base seed. Every tile derives its noise seed from (seed, tile "
                                            "index) — deliberately NOT from the frame number, so a video "
                                            "batch keeps the same noise structure per tile over time "
                                            "(less flicker)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,
                                 {"tooltip": "Sampling algorithm — the full list, no per-model subset."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,
                              {"tooltip": "Sigma schedule — the full list."}),
                "tile_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 8,
                                      "tooltip": "Tile edge for the refine (constant per axis — uniform tiles "
                                                 "are what diffusion models like). Clamped to the canvas."}),
                "tile_overlap": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 8,
                                         "tooltip": "Target overlap between neighbouring tiles. Tiles are "
                                                    "stitched as a weighted sum with smoothstep feathers whose "
                                                    "accumulated weight is EXACTLY 1 everywhere — the reason "
                                                    "this node has no seam-fix family: none is needed. When a "
                                                    "canvas can't host the requested overlap safely, the planner "
                                                    "narrows it (correct beats wide)."}),
                "sigma_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 20.0, "step": 0.01, "round": 0.01,
                                          "tooltip": "Flow-matching sigma shift, the sampler's own mechanism "
                                                     "(0 = off). Wan 2.2 wants its native shift (typically 8.0) "
                                                     "or the tile schedules are mis-scaled. Applied to both "
                                                     "experts in High + Low."}),
                # ── v546: per-stage sampler / scheduler — appended LAST in required for
                #    serialised index stability. UNLIKE the Sampler these are legal
                #    UNCONDITIONALLY: this node is a STAGE CHAIN, not a sigma-boundary
                #    split. Stage L starts on the finished, decoded, re-upscaled image of
                #    stage H and runs its OWN fresh partial denoise with its OWN schedule
                #    (steps_low / denoise_low) — there is no seam to keep continuous, so
                #    no handoff mode gates them. Default "same as high". ──
                "sampler_low": ([SAME_AS_HIGH] + list(comfy.samplers.KSampler.SAMPLERS),
                                {"default": SAME_AS_HIGH,
                                 "tooltip": "High + Low: sampling ALGORITHM for stage L "
                                            "('sampler_name' drives stage H). The stages are "
                                            "independent runs, so this is always honoured. "
                                            "Ancestral / SDE samplers (*_a, *_sde) inject "
                                            "fresh noise per step — at refine strengths that "
                                            "changes the grain."}),
                "scheduler_low": ([SAME_AS_HIGH] + list(comfy.samplers.KSampler.SCHEDULERS),
                                  {"default": SAME_AS_HIGH,
                                   "tooltip": "High + Low: sigma SCHEDULE for stage L. Always "
                                              "honoured — stage L builds its own schedule from "
                                              "steps_low / denoise_low anyway, so a second "
                                              "scheduler breaks nothing (no handoff, no seam)."}),
                # ── v549: result preview. PURE ui - the tensors returned to the graph
                #    are byte-identical with this On or Off; the toggle only decides
                #    whether small temp JPEGs are written and shown in the node. ──
                "result_preview": ("BOOLEAN", {"default": True,
                                               "label_on": "On", "label_off": "Off",
                                               "tooltip": "Show the finished frames inside the node "
                                                          "(small temp JPEGs; multi-frame runs play as "
                                                          "a flipbook at the source frame rate). Never "
                                                          "touches the outputs."}),
                # ── v550: process view. Off = the probe is never built; latent2rgb
                #    streams the tile being refined into a collapsible pane. ──
                # v885: the DEFAULT moved Off -> latent2rgb (Frank's go). A NEW
                #    node now shows its work; latent2rgb is the free path (a
                #    linear projection of a latent that already exists). The
                #    HEAL is deliberately NOT moved: a save from before v550
                #    ran with no process view at all, and _healPreV550 keeps
                #    filling "Off" so such a workflow stays bit-for-bit what it
                #    was. Consequence, said out loud rather than discovered:
                #    an EXISTING workflow carries its own stored "Off" and is
                #    unaffected - it needs one flip of the widget.
                "process_preview": (["Off", "latent2rgb", "vae (sharp)"],
                                    {"default": "latent2rgb",
                                     "tooltip": "Watch the refine LIVE: the tile being "
                                                "sampled right now, a minimap locating it "
                                                "on the stage canvas, and stage/tile/step "
                                                "counters. Off = zero overhead (the probe "
                                                "is never built). latent2rgb = free, but "
                                                "it reads the LATENT's grid (/8 - a 768 "
                                                "tile is 96px, and it looks it). "
                                                "vae (sharp) = a real decode of one frame "
                                                "per event: full resolution, at the cost "
                                                "of one vae forward - measured against the "
                                                "stage clock and dropped back to latent2rgb "
                                                "on its own if it starts eating the pass."}),
                # ── v553: silence ComfyUI's per-tile model staging chatter for the
                #    duration of each stage. Pure logging - the run is untouched. ──
                "mute_staging_logs": ("BOOLEAN", {"default": True,
                                                  "label_on": "On", "label_off": "Off",
                                                  "tooltip": "Silence ComfyUI's per-tile staging INFO "
                                                             "lines ('prepared for dynamic VRAM "
                                                             "loading', 'x models unloaded') while a "
                                                             "stage runs. Restored byte-exactly after. "
                                                             "Turn Off when debugging VRAM."}),
                # ── v555: the stage-fit method after the ESRGAN pass. The default
                #    "lanczos (cpu)" IS the historic behaviour - byte-identical. ──
                "resize_method": (list(_METHODS.keys()),
                                  {"default": "lanczos (cpu)",
                                   "tooltip": "Fit onto the stage canvas after the "
                                              "pixel upscale. 'lanczos (cpu)' = the "
                                              "historic path, byte-identical. bicubic/"
                                              "bilinear/area/nearest run chunked on "
                                              "the GPU (fast). nvidia_rtx_vsr = Maxine "
                                              "VideoSuperRes (needs nvvfx + RTX)."}),
                # ── v560: OOM guard for the pixel pass. Appended LAST. ──
                "per_batch": ("INT", {"default": 8, "min": 1, "max": 256,
                                      "tooltip": "Frames per chunk for the ESRGAN pass "
                                                 "and its fit (the upscale + fit are "
                                                 "FUSED per chunk, so the peak holds "
                                                 "only this many frames at the "
                                                 "intermediate size). 129 frames through "
                                                 "a 4x model is a ~15 GB tensor in one "
                                                 "go - this is the OOM guard. There is no "
                                                 "0 any more: it meant 'whole batch', which "
                                                 "is exactly the 14.7 GB path this node removed. "
                                                 "For the old behaviour set 256."}),
                # ── v562: spatial VAE tiling. Appended LAST. ───────────────────
                "vae_tiling": (_VAE_TILES, {"default": "Off",
                                            "tooltip": "Spatial VAE tiling for each "
                                                       "tile's encode/decode. Off = the "
                                                       "historic path (byte-identical). A "
                                                       "size (512/640/768) cuts the VAE "
                                                       "peak on big canvases, at the cost "
                                                       "of a tiny internal blend at the VAE "
                                                       "tile borders. NEVER temporal: a Wan "
                                                       "VAE compresses time 4:1, so time "
                                                       "tiles would stutter. The joint "
                                                       "(video+audio) refine's whole-clip "
                                                       "encode/decode reads it too -- a weak "
                                                       "card's first lever against a VAE OOM."}),
                # ── v564: the explicit either/or. Appended LAST.
                #    v566 added 'model only' - the canvas IS the model factor.
                #    v568 REMOVED 'model (high only)': it exists for free now,
                #    because the WIRES decide (leave upscale_model_low unwired and
                #    stage L runs a plain fit). An old save carrying that value is
                #    healed to the default by the v563 _sanitize net.
                #    v569 ADDS 'model final' (appended - value lists grow at the
                #    tail like widgets do): the stages run pure fit and the LAST
                #    stage's wire runs ONCE behind the final decode, where no VAE
                #    can erase its detail. That is the measured place for a pixel
                #    model as a REFINER - eyes, buttons, hair - at refine factors
                #    like 1.10/1.30. ─────────────────────────────────────────────
                "pixel_stage": (["model + fit", "fit only", "model only",
                                 "model final"],
                                {"default": "model + fit",
                                 "tooltip": "How each stage reaches its canvas. 'model + fit' = the wired UPSCALE_MODEL "
                                            "runs first, the fit lands it on the exact stage size (the classic recipe). "
                                            "'fit only' IGNORES the wired model; resize_method does the whole upscale -- "
                                            "seconds instead of minutes, and the refine paints the detail anyway. 'model "
                                            "only' = the canvas IS the model factor; upscale_by is ignored (loudly), the "
                                            "fit only corrects to the /8 snap. 'model final' = the stages run pure fit and "
                                            "the last stage's pixel model runs ONCE behind the final decode -- the measured"
                                            " place for it: no VAE follows, its detail goes straight to the file "
                                            "(resize_method='none' keeps the raw model result; a kernel supersamples it "
                                            "back to the dialled canvas). Measured: a pixel model in FRONT of a VAE round "
                                            "trip is nearly free of effect -- the /8 compression cannot carry its fine "
                                            "detail. Switch without unplugging. refine_order='before pixel' is the same move for every model (it turns 'model + fit' / 'model only' into 'model final'); 'off' skips the refine."}),
                # ── v582: the user's dial for the FINAL pass canvas. The pixel
                #    model always computes its NATIVE factor (baked into the
                #    weights); this factor decides where the fit lands it. ────
                "final_upscale_by": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 8.0,
                                               "step": 0.05, "round": 0.01,
                                               "tooltip": "Canvas of the FINAL pixel pass (pixel_stage='model final'), as a factor of the"
                                                          " stage result. The model always computes its native factor; the kernel then "
                                                          "lands it on THIS canvas -- the model's imprint reaches the file while the size"
                                                          " stays yours (swap a 2x/4x/8x model: the imprint changes, the size never). 1.0"
                                                          " = the stage canvas (classic supersample refine); below 1.0 = supersampled "
                                                          "DOWNscale (sprite work). Ignored (loudly) when resize_method='none', which "
                                                          "keeps the raw model result."}),
                # ── v851: per-stage sigma shift — appended LAST in required for
                #    serialised index stability (#577). Mirrors the Sampler's v839
                #    widget INCLUDING its -1 sentinel, and resolves through the SAME
                #    function (_resolve_low_shift, imported — not copied).
                #    v894 CORRECTS THE v851 CLAIM. v851 said this dial "can never be
                #    inert, and needs no honesty line", reasoning that the stages are
                #    independent runs. That is true of High + Low and FALSE of Single:
                #    plan_stages(False, ...) yields ONE stage tagged 'single', and
                #    model_low is only ever read where the tag is 'low'. So in Single
                #    the dial reaches nothing -- and until v894 it still built a second
                #    shifted model clone on every run, memory and time for a model the
                #    stage never touches. The dial is now a DUAL_ONLY twin like every
                #    other *_low: hidden in Single by the frontend, skipped by the
                #    backend, and said out loud once when a Single run carries a dialled
                #    value. The lesson is the general one: a promise is re-grounded, not
                #    talked around. ──
                "sigma_shift_low": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20.0,
                                              "step": 0.01, "round": 0.01,
                                              "tooltip": "Flow-matching sigma shift for stage L (the LOW-noise "
                                                         "expert). -1 = 'same as high': stage L follows "
                                                         "sigma_shift, so both stages are shifted alike. "
                                                         "0 = OFF for stage L only -- its model_sampling is left "
                                                         "alone even when sigma_shift is set. Any positive value "
                                                         "gives stage L its OWN shift (e.g. sigma_shift 8.0 with "
                                                         "sigma_shift_low 5.0). HIGH + LOW ONLY: Single runs one "
                                                         "stage and never reads a low expert, so this dial does "
                                                         "nothing there -- the row is hidden in Single, and a "
                                                         "dialled value on a Single run is reported once in the "
                                                         "console instead of quietly building a clone. Works with "
                                                         "or without a wired "
                                                         "'model_low' -- without one, stage L falls back to the "
                                                         "'model' input and gets its own shifted copy of it."}),
                # ── v1004 appended these two LAST in required (#577); v1008
                #    RENAMED them (h3_refine -> refine_order, h3_audio ->
                #    audio_stream) and made the first one read by EVERY model.
                #    Same slots, same values: a save stores values by POSITION,
                #    so every v1004-v1007 workflow loads unchanged (the browser
                #    probe A-D proves it on a live frontend). The defaults
                #    reproduce v1001 bit for bit: 'after pixel' leaves the tile
                #    path exactly as pixel_stage says, and the joint refine
                #    still needs the optional 'latent' wire, which an old save
                #    does not carry. ──
                "refine_order": (list(_REFINE_ORDERS), {"default": "after pixel",
                                                        "tooltip": "WHERE the model refine sits against the pixel model (ESRGAN), for EVERY "
                                                                   "model. 'after pixel' = ESRGAN + fit first, then the refine on that canvas "
                                                                   "(tile path: exactly what pixel_stage says; video+audio models: the refine on "
                                                                   "the big canvas). 'before pixel' = the refine first, then the pixel model "
                                                                   "BEHIND the last decode, where its detail reaches the file untouched (tile "
                                                                   "path: the same as pixel_stage='model final'; video+audio models: the refine "
                                                                   "at the input size -- also the light option for weak cards). 'off' = no model "
                                                                   "pass at all: ESRGAN + fit deliver the canvas (final_upscale_by). Measured law: "
                                                                   "a pixel model in FRONT of a VAE round trip is nearly free of effect -- the "
                                                                   "VAE's compression erases most of its detail."}),
                "audio_stream": (list(_AUDIO_STREAM_MODES), {"default": "keep",
                                                              "tooltip": "Video+audio (joint) models only, e.g. MiniMax H3: what the "
                                                                         "refine does with the AUDIO stream of the packed latent. "
                                                                         "'keep' freezes it (denoise mask 0 -- the run's audio comes "
                                                                         "out bit-identical); 'denoise' lets the model touch it with "
                                                                         "the same sigmas as the video. Ordinary models never read it."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Frame input [N,H,W,C] — stills or an unpacked video. Wire "
                                               "EXACTLY one of image / video."}),
                "video": ("VIDEO", {"tooltip": "Native VIDEO input (green). Its frames are upscaled; its "
                                               "audio and frame rate ride through losslessly into the VIDEO "
                                               "output. Wire EXACTLY one of image / video."}),
                "model_low": ("MODEL", {"tooltip": "High + Low: the LOW-noise expert for stage L. Unwired -> "
                                                   "stage L falls back to 'model'."}),
                "upscale_model_low": ("UPSCALE_MODEL", {"tooltip": "High + Low: stage-L pixel upscale model - "
                                                                  "and, with pixel_stage='model final', the "
                                                                  "model of the FINAL pass behind the last "
                                                                  "decode. Unwired -> stage L runs a plain fit "
                                                                  "(the H model is NOT inherited) and a "
                                                                  "'model final' run says so and skips the "
                                                                  "final pass."}),
                # ── v1004: two wires; v1008: 'sigmas' serves every model ──
                "latent": ("LATENT", {"tooltip": "Video+audio (joint) models only: the packed AV latent of the run "
                                                 "(the Sampler's LATENT output). Its AUDIO stream rides into the "
                                                 "refine; the video stream is re-encoded from the frames. Unwired -> "
                                                 "no joint refine (pixel path only, said out loud). Ordinary models "
                                                 "ignore this wire."}),
                "sigmas": ("SIGMAS", {"tooltip": "The sampler's own sigma curve (e.g. a Hyperflow / lightx2v / Turbo "
                                                 "list), for EVERY model. The refine runs its tail: on a flow curve "
                                                 "(values in 0..1: Wan, Flux, H3) it starts EXACTLY at sigma = denoise "
                                                 "(0.30 = 30 % noise, whatever the curve's grid); on any other curve "
                                                 "(SD1/SDXL, sigma is not a noise fraction) it runs the last "
                                                 "round(steps x denoise) steps. 'steps' and 'scheduler' are not read "
                                                 "while this is wired (said in the console). Unwired -> 'scheduler' + "
                                                 "'steps' with Core's denoise slicing."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("image", "video")
    FUNCTION = "upscale"
    CATEGORY = "Polyhedron/Upscaling"
    DESCRIPTION = ("MoE-aware tiled upscaler: one node in place of a chained double upscale "
                   "setup, mirroring the sampler's Single / High+Low split. Tile size is "
                   "chosen against free VRAM rather than a fixed guess, the pixel model can "
                   "sit in front of the stages or behind the last decode (which is where it "
                   "earns its time) - and there the output canvas is YOUR factor, not the "
                   "model's native one, so swapping pixel models changes their imprint, "
                   "never the file size. Every stage reports its own seconds so the slow "
                   "one is visible instead of merely suspected.")

    def upscale(self, model, positive, negative, vae, upscale_model, dual_moe,
                upscale_by, denoise, steps, cfg, upscale_by_low, denoise_low,
                steps_low, cfg_low, seed, sampler_name, scheduler, tile_size,
                tile_overlap, sigma_shift, image=None, video=None,
                model_low=None, upscale_model_low=None, sampler_low=SAME_AS_HIGH, scheduler_low=SAME_AS_HIGH,
                result_preview=True, process_preview="latent2rgb", mute_staging_logs=True,
                resize_method="lanczos (cpu)", per_batch=8, vae_tiling="Off",
                pixel_stage="model + fit", final_upscale_by=1.0,
                sigma_shift_low=-1.0, refine_order="after pixel", audio_stream="keep",
                latent=None, sigmas=None):
        t_all = time.monotonic()   # v553/v567: wall clock of THIS function,
        # started before input resolve so 'total=' and the ComfyUI badge
        # disagree only by executor overhead OUTSIDE the node body.
        # v880: the cheapest check in the node, and it runs FIRST.
        _joint = _joint_streams(model, model_low)
        if _joint is None:
            _reject_none_conditioning(positive, negative)
        else:
            print("\u2b21 Power Upscale: the model on '%s' denoises %d "
                  "latent streams jointly (video + audio) -- MiniMax H3 and "
                  "its kin. It cannot refine image tiles, so the tile REFINE "
                  "stages are OFF for this run and the PIXEL path (final "
                  "ESRGAN + fit) delivers the canvas. With refine_order on and "
                  "'latent' wired, the whole-clip JOINT refine runs with it "
                  "(see the lines below). For a TILE refine wire an ordinary "
                  "image/video model (e.g. Wan) with its matching VAE and "
                  "conditioning." % _joint)
        # ── v1008: ONE order dial for every model. Decided ONCE, up front,
        #    and every 'no' is said by name (the v552/v885 rule). ──
        _order = str(refine_order or "after pixel")
        if _order not in _REFINE_ORDERS:
            print(f"[PLS] Power Upscale: refine_order={refine_order!r} is not a mode "
                  f"-- running 'after pixel' (the default), loudly.")
            _order = "after pixel"
        # the tile path: 'before pixel' IS pixel_stage='model final' (the
        # pixel model moves behind the last decode); 'off' drops the stages.
        # Rebinding pixel_stage here keeps every reader below on ONE truth.
        if _joint is None and _order == "before pixel":
            if str(pixel_stage) in ("model + fit", "model only"):
                print(f"[PLS] Power Upscale: refine_order='before pixel' -> the tile "
                      f"refine runs first and the pixel model runs ONCE behind the last "
                      f"decode (pixel_stage '{pixel_stage}' acts as 'model final' this "
                      f"run: stages grow by fit to upscale_by, final_upscale_by sets the "
                      f"file canvas).")
                pixel_stage = "model final"
            elif str(pixel_stage) == "fit only":
                print("[PLS] Power Upscale: refine_order='before pixel' with "
                      "pixel_stage='fit only' -- there is no pixel model to move; "
                      "the order changes nothing this run.")
        # ── v1004: is the JOINT refine ON for this run? ──
        _joint_why = ""
        _joint_on = False
        if _joint is not None:
            if _order == "off":
                _joint_why = "refine_order=off"
            elif latent is None:
                _joint_why = "'latent' not wired (the audio stream has no source)"
            elif len(_latent_parts(latent.get("samples"))) < 2:
                _joint_why = "'latent' is not a packed AV latent (no audio stream)"
            else:
                _joint_on = True
                _reject_none_conditioning(positive, negative)
            if not _joint_on:
                print(f"[PLS] Power Upscale: joint refine OFF -- {_joint_why}. The pixel "
                      f"path runs alone (final ESRGAN + fit); denoise/steps/cfg "
                      f"reach nothing this run.")
            else:
                _pc = _patch_count(model)
                if _pc == 0:
                    print("[PLS] Power Upscale: NOTE - the wired 'model' carries "
                          "NO weight patches: the refine runs on the raw loader output, "
                          "without the Turbo/Hyperflow/LoRA chain your sampler uses. "
                          "Wire the END of that chain (e.g. the NAG output) into "
                          "'model' if the refine should see the same model.")
                if sigmas is None:
                    print(f"[PLS] Power Upscale: NOTE - no 'sigmas' wired: the joint "
                          f"refine builds its own {scheduler} schedule over {int(steps)} "
                          f"steps with Core's denoise slicing. An accelerator that "
                          f"expects its own curve (Hyperflow, Turbo) wants that "
                          f"curve here too.")
                if abs(float(cfg) - 1.0) > 1e-9:
                    print(f"[PLS] Power Upscale: NOTE - cfg={float(cfg):.2f} on the joint "
                          f"refine: any value but 1.0 runs a SECOND forward per step "
                          f"(the negative), doubling the step time -- 121 s instead of "
                          f"~60 s at 1872x1072 in the 24.09. field run. Distilled "
                          f"curves (Hyperflow, Turbo) are made for cfg 1.0.")
                if float(denoise) <= 0.0:
                    _joint_on = False
                    _joint_why = "denoise=0 (nothing to refine)"
                    print("[PLS] Power Upscale: joint refine OFF -- denoise=0.")
                if float(denoise) >= 0.7:
                    print(f"[PLS] Power Upscale: NOTE - denoise={float(denoise):.2f} "
                          f"starts the joint refine with {float(denoise) * 100:.0f} % noise "
                          f"on the picture -- that is a re-render, not a detail pass. "
                          f"A detail pass lives around 0.20-0.35.")
        # v884: ONE source of truth for "does the final pass run". v883
        # emptied the stages on a joint model but the ONLY carrier of
        # final_upscale_by outside the stages is the 'model final' block --
        # Frank's field run (pixel_stage='model + fit') therefore delivered
        # the INPUT SIZE, and the amber chunk tiles stayed dark because no
        # pass fired a clock event. The downgrade now routes delivery through
        # the existing final pass, whatever the pixel_stage dial says; the
        # v571 law ("one condition, two places, same spelling") collapses to
        # one NAME read in both places.
        # v1008: refine_order='off' on an ordinary model is the same shape --
        # no stage runs, so the final pass IS the delivery. _final_delivers
        # names those runs: there the pass also runs as a plain FIT when no
        # pixel model is wired (or pixel_stage='fit only'), so the canvas the
        # user dialled (final_upscale_by) is still the canvas he gets.
        _tile_off = (_joint is None and _order == "off")
        _final_delivers = (_joint is not None) or _tile_off
        _final_runs = (str(pixel_stage) == "model final") or (_joint is not None) or _tile_off
        frames, audio, frame_rate = _resolve_input(image, video)
        # ── v851: per-stage sigma shift. The LOW value resolves through the
        #    Sampler's own _resolve_low_shift (-1 sentinel = "same as high", 0 =
        #    OFF for stage L, positive = its own). Two things make this trickier
        #    here than in the Sampler:
        #    (1) ORDER. The HIGH patch REBINDS 'model', so stage L's source must
        #        be captured BEFORE it - otherwise a "low" shift would be laid on
        #        top of an already-shifted model.
        #    (2) THE FALLBACK. Without a wired 'model_low', stage L runs on the
        #        'model' input (v568 law). To give that stage its own shift we
        #        must build a SECOND clone from the RAW model; leaving it to the
        #        loop would silently hand it the HIGH-shifted one.
        #    When the sentinel is untouched AND there is no wired low expert we
        #    deliberately build nothing: the loop then uses the high-shifted
        #    'model' exactly as it did before v851, bit for bit.
        #    (3) v894: THE WHOLE LOW HALF IS DUAL-ONLY. Single plans one stage
        #        tagged 'single' and reads model_low nowhere, so every clone built
        #        for stage L there is paid for and never touched. The gate is on
        #        _dual, not on the dial, and the HIGH shift stays unconditional -
        #        it is the one that reaches the single stage.
        if _joint is not None and (float(sigma_shift or 0.0) > 0
                                   or float(sigma_shift_low or -1.0) > 0):
            # v1004: a sigma shift is NEVER laid on a joint model. The patch
            # would replace ModelSamplingAV -- the sampling object that carries
            # the audio clock (witness C of the joint probe) -- with a plain
            # flow sampler. The dials are zeroed HERE, before the shift block,
            # so that block stays the closed, guard-driven unit it is (v851).
            print(f"[PLS] Power Upscale: sigma_shift={float(sigma_shift or 0):g} "
                  f"is NOT applied to a joint (video+audio) model -- it would "
                  f"replace the AV sampling and its audio clock. The model's own "
                  f"sampling (or the wired curve) sets the schedule.")
            sigma_shift, sigma_shift_low = 0.0, -1.0
        _dual = bool(dual_moe)
        _hi_shift = float(sigma_shift or 0.0)
        _low_raw = float(sigma_shift_low if sigma_shift_low is not None else -1.0)
        _low_dialled = _low_raw >= 0.0          # -1 is the "same as high" sentinel
        _low_shift = _resolve_low_shift(_hi_shift, sigma_shift_low)
        _low_src = model_low if model_low is not None else model   # BEFORE the rebind
        _low_wired = model_low is not None
        _low_same = (_low_shift == _hi_shift)
        if _hi_shift > 0:
            model = _apply_sigma_shift(model, _hi_shift)
        if _dual and (_low_wired or not _low_same):
            if _low_shift > 0:
                model_low = _apply_sigma_shift(_low_src, _low_shift)
            elif not _low_wired and _hi_shift > 0:
                # stage L is OFF while stage H is shifted and no low expert is
                # wired -> hand the loop the RAW model instead of the shifted one.
                model_low = _low_src
        if not _dual and _low_dialled:
            # v894 honesty line: the dial is set, the run is Single, nothing was
            # built for it. Say so once - the v552/v885 rule against silent
            # degradation cuts both ways.
            print(f"[PLS] Power Upscale: sigma_shift_low={_low_raw:g} is ignored - "
                  f"dual_moe is OFF, so there is no stage L to shift. Set "
                  f"dual_moe to use it (the row is hidden in Single).")
        if _hi_shift > 0 or (_dual and _low_shift and _low_shift > 0):
            print(f"[PLS] Power Upscale: sigma shift H={_hi_shift:g}"
                  f"{'' if not _dual else (' L=%g' % _low_shift)}"
                  f"{' (same as high)' if (_dual and _low_same) else ''}"
                  f"{'' if (not _dual or _low_wired) else ' [stage L runs on the model input]'}")

        stages = uls_tile_math.plan_stages(bool(dual_moe), upscale_by, denoise,
                                           steps, cfg, upscale_by_low,
                                           denoise_low, steps_low, cfg_low)
        stages = uls_tile_math.drop_refine_stages(stages, (_joint is not None) or _tile_off)
        if _tile_off:
            print("[PLS] Power Upscale: refine_order='off' -- NO model pass this run: the "
                  "pixel path (ESRGAN + fit, or a plain fit under pixel_stage='fit only') "
                  "delivers the canvas at final_upscale_by. upscale_by, denoise, steps, "
                  "cfg and the tile dials reach nothing.")
        # ── v1008: a wired curve reaches the TILE refine too. Each stage cuts
        #    its OWN run from it at its own denoise (_sigma_run: exact start on
        #    a flow curve, step slicing on any other); the stage's step count
        #    becomes the run's length, so the clock is posted with the truth. ──
        _curve = None
        if sigmas is not None and _joint is None and stages:
            _curve = [float(v) for v in sigmas.flatten().tolist()]
            for st in stages:
                run, how = _sigma_run(_curve, st["denoise"])
                st["sig_run"], st["sig_how"] = run, how
                st["steps"] = max(0, len(run) - 1)
                print(f"[PLS] Power Upscale: stage={st['tag']}: curve {len(_curve) - 1} steps, "
                      f"denoise {float(st['denoise']):.2f} -> {st['steps']} step(s)"
                      + (f" from \u03c3 {run[0]:.3f} ({', '.join('%.3f' % v for v in run)})"
                         if run else " (nothing to sample -- a VAE round trip)")
                      + {"exact": ("" if _sigma_on_grid(_curve, st["denoise"]) else
                                   " -- the start is NOT a point of the curve; a distilled "
                                   "LoRA was trained on its grid, judge by eye"),
                         "slice": " -- step slicing (not a flow curve: sigma is not a noise fraction)",
                         "whole": " -- the whole curve",
                         "none": ""}[how])
            print("[PLS] Power Upscale: 'sigmas' is wired -> the tile refine runs the "
                  "curve's run; 'steps'/'steps_low' and 'scheduler'/'scheduler_low' are "
                  "NOT read this run (the sampler dials still are).")
            if float(sigma_shift or 0.0) > 0 or (bool(dual_moe) and float(sigma_shift_low or -1.0) > 0):
                print("[PLS] Power Upscale: NOTE - sigma_shift is set AND a curve is wired: "
                      "the curve IS the schedule (it carries its own shift); the shift only "
                      "re-labels the model's sampling and does not move a single sigma.")
        elif sigmas is not None and _joint is None and _tile_off:
            print("[PLS] Power Upscale: 'sigmas' is wired but refine_order='off' -- no "
                  "refine reads it this run.")
        # v566: 'model only' - the canvas IS the model factor (Frank's law: the
        # model and the filter are alternatives for GROWING; the fit stays only
        # as the /8 snap corrective). The factor swap happens HERE, before the
        # grids are planned, because the canvas is the contract with the refine.
        # v566/v568: the canvas may be DERIVED instead of dialled. Two triggers:
        #   pixel_stage='model only'  - the model's factor IS the stage factor
        #   resize_method='none'      - no filter may touch the pixels at all, so
        #                               the canvas is whatever the pixel stage
        #                               produced (the model's factor with a model
        #                               wired, the INPUT SIZE without one - i.e. a
        #                               pure refine that does not grow).
        # It happens HERE, before the grids are planned, because the canvas is the
        # contract with the refine. The wires decide which model (v568): stage L
        # uses upscale_model_low or nothing - it never inherits the H model.
        # v580: say what is on the wires BEFORE anything derives a canvas from
        # them. "Greifen die Upscale-Modelle ueberhaupt?" must never again be a
        # question you can only answer by reading the source.
        _say_model_wires(upscale_model, upscale_model_low, pixel_stage, resize_method)
        _derive = (str(pixel_stage) == "model only"
                   or str(resize_method) == _NO_RESIZE)
        if _derive:
            for st in stages:
                um_st = (upscale_model_low if st["tag"] == "low"
                         else upscale_model)
                if str(pixel_stage) in ("fit only", "model final"):
                    # v569: 'model final' stages are PURE FIT - the model moved
                    # behind the last decode, so no stage canvas may follow it.
                    um_st = None
                msc = (float(getattr(um_st, "scale", 0.0) or 0.0)
                       if um_st is not None else 0.0)
                if msc <= 0.0:
                    if str(resize_method) == _NO_RESIZE:
                        # v580: SAY WHY there is no model. 'has no pixel model'
                        # read like a defect and was in fact the receipt for the
                        # user's own setting -- it sent Frank looking for a broken
                        # wire that was never broken. Absence by design and absence
                        # by omission are different facts and must not share a line.
                        why = ("pixel_stage='%s' took the model OUT of this stage "
                               "BY DESIGN (it runs behind the last decode, where a "
                               "pixel model is worth something)" % pixel_stage
                               if str(pixel_stage) in ("fit only", "model final")
                               else "no upscale model is wired to this stage")
                        print(f"[PLS] Power Upscale: stage={st['tag']}: {why}. With "
                              f"resize_method='none' the stage does NOT grow either "
                              f"(x1.00, ignoring upscale_by x{float(st['factor']):.2f}) "
                              f"-> a pure refine at the input size. Legitimate, and "
                              f"said on purpose.")
                        st["factor"] = 1.0
                    else:
                        print(f"[PLS] Power Upscale: stage={st['tag']} has no "
                              f"usable upscale model for pixel_stage='model only' "
                              f"- falling back to upscale_by "
                              f"x{float(st['factor']):.2f}, loudly.")
                    continue
                if abs(msc - float(st["factor"])) > 1e-6:
                    print(f"[PLS] Power Upscale: stage={st['tag']} canvas follows "
                          f"the model: x{msc:.2f} (ignoring upscale_by "
                          f"x{float(st['factor']):.2f} - that is what "
                          f"{'resize_method=none' if str(resize_method) == _NO_RESIZE else chr(39) + 'model only' + chr(39)} means)")
                st["factor"] = msc
        # Plan every stage canvas + grid up front so ONE progress bar can span
        # all tiles x stages (the sampler's shared-bar pattern).
        # v570: each stage also declares its pixel KIND for the clock. v569
        # posted every pixel pass as kind 'pix' - so in 'model final' the
        # FINAL pass (a 4x model forward) rate-borrowed from the stage fits
        # (63 ms/chunk) and the early ETA was a fairy tale until chunk 1
        # measured (Frank watched the bar jump). Kinds mean OPERATIONS:
        # 'fit' = a plain resize, 'pix' = a model forward. Borrowing stays
        # rung-2 legal WITHIN a kind (model+fit stages really do predict the
        # final pass); across kinds it is now impossible by construction.
        w, h = int(frames.shape[2]), int(frames.shape[1])
        plans = []
        for st in stages:
            um_st = (upscale_model_low if st["tag"] == "low"
                     else upscale_model)
            if str(pixel_stage) in ("fit only", "model final"):
                um_st = None   # mirrors the stage loop's null-set exactly
            st["pix_kind"] = "pix" if um_st is not None else "fit"
            # v571: a model pass costs INPUT pixels (the RRDB trunk runs at
            # input resolution; the upsampler tail is cheap). A fit costs
            # roughly OUTPUT pixels (interpolate writes the target). The v570
            # posts weighed both by target canvas - which over-weighed the
            # final 4x pass 16x and fed the rung-3 blend a fairy tale the
            # tilde only half-excused (measured: 'run eta ~1:07:54' against a
            # real ~6 min). Weights now follow the operation's own physics.
            st["pix_in"] = float(w * h)          # canvas BEFORE this stage
            w, h = uls_tile_math.scaled_size(w, h, st["factor"])
            grid = uls_tile_math.plan_grid(w, h, int(tile_size), int(tile_overlap))
            plans.append((st, w, h, grid))
        # v567: the bar is a TIME bar now. v565's tick budget counted pixel
        # chunks and sampler steps as equals - 18 chunks at ~62 ms and 8 steps
        # at 43-93 s. Half a second in it showed 35%; the truth was 0.07%.
        # The clock keeps posts weighted by tile pixel area and learns real
        # rates live (see _RunClock). No card constants; it calibrates on
        # itself, and the first measured step arrives within the first minute.
        n_frames = int(frames.shape[0])
        pb = 8 if not per_batch or int(per_batch) < 1 else int(per_batch)
        pix_chunks = (n_frames + pb - 1) // pb
        pbar = comfy.utils.ProgressBar(1000)   # the clock owns value AND total
        clock = _RunClock(pbar)
        for st, _w, _h, g in plans:
            tile_px = float(g["tile_w"] * g["tile_h"])
            clock.post(f"{st['pix_kind']}:{st['tag']}", pix_chunks,
                       st["pix_in"] if st["pix_kind"] == "pix"
                       else float(_w * _h))
            clock.post(f"enc:{st['tag']}", len(g["tiles"]), tile_px)
            clock.post(f"step:{st['tag']}",
                       len(g["tiles"]) * max(1, int(st["steps"])), tile_px)
            clock.post(f"dec:{st['tag']}", len(g["tiles"]), tile_px)
        # v569: the final pass is a run phase like any other - post it up front
        # so the ETA covers it from second one. Weight = the MODEL's output area
        # (the forward dominates its cost); the 'pix' kind rate-borrows from the
        # stage passes until its own first chunk measures (rung 2, declared ~).
        if _final_runs:
            _um_fin = upscale_model_low if bool(dual_moe) else upscale_model
            if _final_delivers and str(pixel_stage) == "fit only":
                _um_fin = None   # v1008: 'fit only' means no pixel model, on every path
            # v571: the gate mirrors the final-pass block EXACTLY (wired or
            # not) - v570 gated the post on scale > 0, so a scale-less model
            # would have run the pass and then KeyError'd the clock on its
            # first measure. One condition, two places, same spelling. The
            # weight is the pass's INPUT (the last stage canvas): ESRGAN cost
            # lives on input pixels; the old x scale^2 over-weighed a 4x pass
            # 16x into the rung-3 blend.
            if _um_fin is not None:
                clock.post("pix:final", pix_chunks, float(w * h))
            elif _final_delivers and str(resize_method) != _NO_RESIZE:
                clock.post("fit:final", pix_chunks, float(w * h))   # v1008: the plain-fit delivery
        # ── v1004: the JOINT refine is a run phase like any other -- posted up
        #    front so the ETA covers it. Its step count is known now (the
        #    curve's run, or 'steps' under Core's slicing); its weight is the
        #    canvas it runs on times the frames (one latent, all frames). ──
        _jr_info = None
        _jr_frames_in = n_frames
        _jr_key = None
        _jr_hit = None
        _jr_snap = _vae_spatial(vae, _JOINT_SNAP_DEFAULT) if _joint_on else _JOINT_SNAP_DEFAULT
        if _joint_on:
            if sigmas is not None:
                _jr_steps = max(1, len(_sigma_run(
                    [float(v) for v in sigmas.flatten().tolist()], denoise)[0]) - 1)
            else:
                _jr_steps = max(1, int(steps))
            if _order == "before pixel":
                _hw, _hh = _snap_to(int(frames.shape[2]), int(frames.shape[1]), _jr_snap)
            else:
                _hw, _hh = w, h   # the last planned canvas (final pass may grow it)
                if _final_runs:
                    try:
                        _fby0 = float(final_upscale_by)
                    except (TypeError, ValueError):
                        _fby0 = 1.0
                    if 0.25 <= _fby0 <= 8.0 and str(resize_method) != _NO_RESIZE:
                        _hw, _hh = int(round(w * _fby0)), int(round(h * _fby0))
                _hw, _hh = _snap_to(_hw, _hh, _jr_snap)
            # v1006: the cache is asked NOW, so a hit posts neither the pixel
            # pass nor the encode to the clock (they will not run).
            _um_key = _um_fin if _final_runs else None
            if _order == "after pixel":
                _jr_key = _cache_key(frames, vae, _um_key, (_hw, _hh), resize_method)
            else:
                _jr_key = _cache_key(frames, vae, None, (_hw, _hh), "input")
            _jr_hit = _latent_cache_get(_jr_key)
            if _jr_hit is not None:
                print(f"[PLS] Power Upscale: joint cache HIT -- the video latent of this exact "
                      f"input{' + pixel pass' if _order == 'after pixel' else ''} is "
                      f"still in memory; skipping "
                      f"{'the pixel pass and ' if _order == 'after pixel' else ''}the "
                      f"encode. Only the sampling and the decode run.")
                if _order == "after pixel":
                    clock.resize("pix:final" if _um_key is not None else "fit:final", 0)
            _jr_px = float(_hw * _hh) * float(n_frames)
            if _jr_hit is None:
                clock.post("enc:joint", 1, _jr_px)
            clock.post("step:joint", _jr_steps, _jr_px)
            clock.post("dec:joint", 1, _jr_px)
        clock.push()
        # v550: built ONCE - latent_rgb factors are a property of the latent
        # FAMILY, identical across a Wan expert pair. Off -> None -> zero cost.
        # v594: the probe takes the VAE now - 'vae (sharp)' decodes ONE latent
        # frame per event instead of reading the /8 latent2rgb map. The mode
        # pays for itself against the stage clock or it disarms itself.
        probe = (_make_tile_probe(model, vae=vae,
                                  sharp=(str(process_preview) == "vae (sharp)"),
                                  clock=clock)
                 if str(process_preview) != "Off" else None)
        # v565: the pixel door needs no model at all - the frames are RGB already.
        pixel_probe = (_make_pixel_probe(clock=clock)
                       if str(process_preview) != "Off" else None)
        # v885: Off still means OFF - the v550 promise ("zero overhead, the
        # probe is never built") is not quietly broken for a nicer picture.
        # But it says so ONCE, because a silent Off is indistinguishable from a
        # broken preview, and that cost Frank a whole run of guessing.
        if str(process_preview) == "Off":
            print("[PLS] Power Upscale: process view is Off - no live picture "
                  "this run (not a fault). Set process_preview to 'latent2rgb' "
                  "for the free /8 map, or 'vae (sharp)' for a real decode.")
        else:
            # v885: the third door - the INPUT, before any pass has finished.
            _emit_input_preview(frames, clock=clock)

        cur = frames
        if _joint_on and _order == "before pixel":
            # v1004: refine at the INPUT size, then the pixel path grows it.
            _c = None
            if _jr_hit is not None:
                _c = (_jr_hit, int(cur.shape[0])) + _snap_to(int(cur.shape[2]), int(cur.shape[1]), _jr_snap)
            cur, _jr_info = _joint_refine(model, positive, negative, vae, cur, latent,
                                          sigmas, seed, steps, cfg, sampler_name,
                                          scheduler, denoise, audio_stream, clock,
                                          probe=probe, mute=mute_staging_logs, cached=_c,
                                          vae_tiling=vae_tiling)
            if _jr_hit is None:
                _latent_cache_put(_jr_key, _jr_info["vid_lat"], "before pixel")
            n_frames = int(cur.shape[0])
        # v1008: the tile path's books -- rates section, per-stage VRAM peaks,
        # and what the verdict will say about the run
        _tile_rsec = _rates_section("tile", model) if (_joint is None and plans) else None
        _tile_tiles = 0
        _tile_steps = 0
        _tile_start = None
        for st, sw, sh, grid in plans:
            is_low = (st["tag"] == "low")
            m = (model_low if (is_low and model_low is not None) else model)
            # v568: THE WIRES ARE THE TRUTH. The MoE expert falls back to the H
            # model (right - one expert pair, two sigma ranges). The PIXEL model
            # does NOT: v566 inherited the H upscaler into stage L, which sent a
            # 4x ESRGAN over the LARGER L frames (848 -> 3392, 310 s measured) to
            # build material the fit immediately threw away. One wire, one stage.
            um = (upscale_model_low if is_low else upscale_model)
            if (is_low and upscale_model_low is None
                    and str(pixel_stage) in ("model + fit", "model only")):
                print("[PLS] Power Upscale: stage=low has no upscale_model_low "
                      "wired -> its pixel pass is a plain fit. (v568: the H model "
                      "is NOT inherited any more - it used to run over the larger "
                      "L frames for material the fit threw away.)")
            if str(pixel_stage) in ("fit only", "model final") and um is not None:
                # v564: the explicit either/or - the wired model is deliberately
                # ignored, resize_method does the whole upscale.
                # v569: 'model final' nulls it too - the stages run pure fit and
                # the LAST stage's wire runs once behind the final decode, where
                # its detail cannot be erased by an encode.
                um = None
            # v546: stage L may run its own sampler AND scheduler. No gate needed --
            # the stages are independent runs (see the module docstring); nothing is
            # continued across them.
            samp = _low_or(sampler_low, sampler_name) if is_low else sampler_name
            sched = _low_or(scheduler_low, scheduler) if is_low else scheduler
            if st.get("sig_how") is not None:
                sched = "curve"   # v1008: the wired curve IS the schedule (never read below)
            # v553: the plan line moved to BEGIN (all fields are known up
            # front), a per-tile line lives in _refine_tiles, and DONE carries
            # the MEASURED stage duration - a 300 s run now narrates itself.
            # v560: coverage = how many times the canvas is actually sampled.
            # A tile_size just below the canvas produces near-identical tiles
            # (848 canvas + 768 tile => offsets 0 and 80 => 3.3x the work).
            cov = (len(grid["tiles"]) * grid["tile_w"] * grid["tile_h"]
                   / max(1.0, float(sw * sh)))
            if cov > 1.6:
                snug, ew, eh = _grid_advice(sw, sh, int(tile_size),
                                            int(tile_overlap),
                                            grid["nx"], grid["ny"])
                print(f"[PLS] Power Upscale: WARNING tile_size={int(tile_size)} on a "
                      f"{sw}x{sh} canvas samples {cov:.1f}x the area "
                      f"({len(grid['tiles'])} tiles, offsets overlap heavily). "
                      f"Two clean choices: tile_size >= {max(sw, sh)} "
                      f"would use ONE tile (1.0x), or tile_size {snug} keeps the "
                      f"{grid['nx']}x{grid['ny']} grid snug. Note: this "
                      f"{int(tile_size)} tile carries canvases up to {ew}x{eh} at "
                      f"the SAME cost - a canvas just past a tile edge (like this "
                      f"one) pays the full next grid.")
            v_enc, v_dec, v_label = _vae_ops(vae, vae_tiling)
            _vram_note(int(cur.shape[0]), grid["tile_w"], grid["tile_h"])
            print(f"[PLS] Power Upscale: stage={st['tag']} begin -> {sw}x{sh} "
                  f"grid={grid['nx']}x{grid['ny']} tiles={len(grid['tiles'])} "
                  f"coverage={cov:.1f}x vae={v_label} "
                  f"denoise={st['denoise']:.2f} steps={st['steps']} cfg={st['cfg']:.2f} "
                  f"sampler={samp} sched={sched} fit={resize_method} "
                  f"pixel={'fit only' if um is None else (('model only' if str(pixel_stage) == 'model only' else 'model+fit') + ' (' + ('low' if is_low else 'high') + ' model)')}"
                  f"{' expert=low' if (is_low and model_low is not None) else (' expert=high(fallback)' if is_low else '')}")
            t0 = time.monotonic()
            # v567: the chunk narration feeds the clock and formats honestly -
            # a 62 ms chunk prints '62ms/chunk', not '0.0s'. Sub-100ms chunks
            # fold into one summary line instead of a nine-line litany (the
            # first chunk always prints, so a run never LOOKS frozen). The
            # probe still fires per chunk - folding is a console courtesy.
            pix = {"k": 0, "t0": time.monotonic(), "last": time.monotonic(),
                   "n": pix_chunks}

            def _on_chunk(i, j, part, chunks, _tag=st["tag"], _n=n_frames,
                          _p=pix, _key=f"{st['pix_kind']}:{st['tag']}"):
                key = _key   # v570: kinds mean operations (fit vs pix)
                if _p["k"] == 0 and int(chunks) != _p["n"]:
                    # The pass may clamp its own chunk for VRAM (v565), so the
                    # true count is only known once it runs. Correct the plan.
                    clock.resize(key, int(chunks))
                    _p["n"] = int(chunks)
                now = time.monotonic()
                dt = now - _p["last"]
                _p["last"] = now
                _p["k"] += 1
                clock.measure(key, dt)   # also pushes the time bar
                per = (now - _p["t0"]) / max(1, _p["k"])
                eta = per * max(0, _p["n"] - _p["k"])
                per_s = f"{per * 1000:.0f}ms" if per < 0.1 else f"{per:.1f}s"
                if _p["k"] == 1 or per >= 0.1:
                    print(f"[PLS] Power Upscale:   {_tag} pixel chunk "
                          f"{_p['k']}/{_p['n']} ({j}/{_n} frames) "
                          f"{per_s}/chunk eta {eta:.0f}s")
                elif _p["k"] == _p["n"]:
                    print(f"[PLS] Power Upscale:   {_tag} pixel chunks "
                          f"2-{_p['n']} folded ({now - _p['t0']:.1f}s total, "
                          f"{per_s}/chunk - too fast to narrate one by one)")
                if pixel_probe is not None:
                    pixel_probe(_tag, _p["k"], _p["n"], j, _n, part,
                                clock.elapsed(), clock.eta())

            # ── v1008: the PIXEL cache -- the first stage's model pass depends on
            #    nothing but the input, the wire, the canvas and the fit, so a
            #    denoise/cfg/seed experiment on the same clip does not pay for
            #    ESRGAN again. Full precision or not at all (RAM-budgeted). ──
            _pc_key = None
            _pc_hit = None
            if um is not None and st is plans[0][0]:
                _pc_key = _pixel_cache_key(frames, um, (sw, sh), resize_method,
                                           str(pixel_stage))
                _pc_hit = _pixel_cache_get(_pc_key)
            st_peaks = {}
            if _tile_rsec is not None:
                _pm = float(grid["tile_w"] * grid["tile_h"] * int(cur.shape[0])) / 1e6
                _rr = _rates_load(_tile_rsec)
                _pl, _pe = _phase_plan(_pm, int(st["steps"]), False, _rr)
                print(f"[PLS] Power Upscale: stage={st['tag']} plan per tile "
                      f"({grid['tile_w']}x{grid['tile_h']} x {int(cur.shape[0])} frames, "
                      f"{_pm:.0f} megapixel-frames, {_tile_rsec}):")
                for _ln in _pl:
                    print(f"[PLS] Power Upscale:   {_ln}")
                _mn = _memory_note(_rr, _pm, _vram_total_gb(),
                                   "a smaller tile_size, vae_tiling 512, fewer frames per clip")
                if _mn:
                    print(f"[PLS] Power Upscale: {_mn}")
            with _MuteInfoLogs(mute_staging_logs, label="Power Upscale"):
                if _pc_hit is not None:
                    cur = _pc_hit
                    clock.resize(f"{st['pix_kind']}:{st['tag']}", 0)
                    print(f"[PLS] Power Upscale: stage={st['tag']} pixel cache HIT -- the "
                          f"model pass of this exact input, wire, canvas and fit is still in "
                          f"RAM; the ESRGAN pass is skipped ({sw}x{sh} x{int(cur.shape[0])} frames)")
                else:
                    cur = _esrgan_pass(cur, um, sw, sh, resize_method, per_batch,
                                       on_chunk=_on_chunk)
                    if _pc_key is not None:
                        _ok, _why = _pixel_cache_put(_pc_key, cur, st["tag"])
                        print(f"[PLS] Power Upscale: stage={st['tag']} pixel cache: "
                              f"{'stored' if _ok else 'skipped'} ({_why})")
                t_pix = time.monotonic()
                cur = _refine_tiles(m, positive, negative, vae, cur, grid, seed,
                                    st["steps"], st["cfg"], samp, sched,
                                    st["denoise"], clock, st["tag"],
                                    tile_probe=probe,
                                    vae_encode=v_enc, vae_decode=v_dec,
                                    sig_run=st.get("sig_run"), rsec=_tile_rsec,
                                    peaks=st_peaks)
            _tile_tiles += len(grid["tiles"])
            _tile_steps += int(st["steps"])
            if _tile_start is None and st.get("sig_run"):
                _tile_start = float(st["sig_run"][0])
            _pkl = _peak_line(f"stage={st['tag']}",
                              [(k, st_peaks.get(k)) for k in ("encode", "sample", "decode")],
                              _vram_total_gb())
            if _pkl:
                print(_pkl)
            dur = time.monotonic() - t0
            _free()   # v561: between stages, hand the blocks back
            # v565: the old line divided the WHOLE stage by the tile count, which
            # folded a 218 s pixel pass into a "315.0s/tile" that described nothing.
            # Two costs, two numbers.
            pix_dur = t_pix - t0
            ref_dur = dur - pix_dur
            print(f"[PLS] Power Upscale: stage={st['tag']} done in {dur:.1f}s "
                  f"(pixel {pix_dur:.1f}s + refine {ref_dur:.1f}s = "
                  f"{ref_dur / max(1, len(grid['tiles'])):.1f}s/tile)"
                  f"{' [single-tile fast path: no blend]' if len(grid['tiles']) == 1 and grid['tile_w'] == sw and grid['tile_h'] == sh else ''}")

        # ── v569: THE FINAL PASS (pixel_stage='model final') ─────────────────
        # The measured law (v568): a pixel model in front of a VAE round trip
        # is nearly free of effect - the Wan VAE compresses 8x spatially and
        # erases what the model builds; the sampler repaints the rest. BEHIND
        # the last decode nothing follows but the file, so this is where the
        # wire earns its seconds: eyes, buttons, hair - the refiner role.
        # One pass on the finished canvas, reusing the whole stack: fp16
        # (v568), resident (v564), budget-clamped output buffer (v565 - a 4x
        # model on 65 frames stays a per_batch clamp, never a 15 GB tensor),
        # chunked + interruptible (v565/v566). The wires are the truth (v568):
        # in High + Low the pass belongs to the LAST stage and takes its wire
        # (upscale_model_low); in Single it takes upscale_model.
        if _final_runs and _jr_hit is not None and _order == "after pixel":
            print("[PLS] Power Upscale: final pass SKIPPED (joint cache hit -- its "
                  "result is the cached latent's source)")
        elif _final_runs:
            um_fin = upscale_model_low if bool(dual_moe) else upscale_model
            wire = "upscale_model_low" if bool(dual_moe) else "upscale_model"
            if _final_delivers and str(pixel_stage) == "fit only":
                um_fin = None   # v1008: mirrors the clock post exactly
            if um_fin is None and _final_delivers:
                # v1008: the final pass IS the delivery here (a joint model, or
                # refine_order='off') -- without a pixel model it is a plain fit
                # onto the dialled canvas, never a silent input-size file.
                fw, fh = int(cur.shape[2]), int(cur.shape[1])
                try:
                    fby = float(final_upscale_by)
                except (TypeError, ValueError):
                    fby = float("nan")
                if not (0.25 <= fby <= 8.0):
                    fby = 1.0
                tw, th = int(round(fw * fby)), int(round(fh * fby))
                if _joint_on and _order == "after pixel":
                    tw, th = _snap_to(tw, th, _jr_snap)
                if str(resize_method) == _NO_RESIZE or (tw, th) == (fw, fh):
                    print(f"[PLS] Power Upscale: final pass: no pixel model "
                          f"({'pixel_stage=fit only' if str(pixel_stage) == 'fit only' else wire + ' not wired'})"
                          f" and nothing to fit ({'resize_method=none' if str(resize_method) == _NO_RESIZE else 'canvas unchanged'})"
                          f" -> the frames pass as they are ({fw}x{fh}).")
                else:
                    print(f"[PLS] Power Upscale: final pass begin -> plain fit "
                          f"({'pixel_stage=fit only' if str(pixel_stage) == 'fit only' else wire + ' not wired'}) "
                          f"{fw}x{fh} -> {tw}x{th} (final_upscale_by={fby:.2f}, {resize_method})")
                    t_ff = time.monotonic()
                    ffin = {"last": t_ff, "n": pix_chunks, "k": 0}

                    def _on_fitfin(i, j, part, chunks, _p=ffin):
                        if _p["k"] == 0 and int(chunks) != _p["n"]:
                            clock.resize("fit:final", int(chunks))
                            _p["n"] = int(chunks)
                        now3 = time.monotonic()
                        clock.measure("fit:final", now3 - _p["last"])
                        _p["last"] = now3
                        _p["k"] += 1
                        if pixel_probe is not None:
                            pixel_probe("final", _p["k"], _p["n"], j, n_frames, part,
                                        clock.elapsed(), clock.eta())

                    cur = _esrgan_pass(cur, None, tw, th, resize_method, per_batch,
                                       on_chunk=_on_fitfin)
                    print(f"[PLS] Power Upscale: final fit done in "
                          f"{time.monotonic() - t_ff:.1f}s -> {tw}x{th}")
            elif um_fin is None:
                print(f"[PLS] Power Upscale: pixel_stage='model final' but "
                      f"{wire} is not wired -> NO final pass (the wires are "
                      f"the truth). This run behaved exactly like 'fit only'.")
            else:
                fsc = float(getattr(um_fin, "scale", 0.0) or 0.0)
                fw, fh = int(cur.shape[2]), int(cur.shape[1])
                # v583: the belt per_batch wears (v563), same reason - a
                # conserved ''/0/junk from an old save or a raw API prompt must
                # never reach the size law. 0 is the measured field artefact
                # (Number('') on a stale frontend); below the widget's own min
                # it was never a dial position, so it falls to 1.0 (the old
                # law), loudly, instead of clamping to 0.25 silently.
                try:
                    fby = float(final_upscale_by)
                except (TypeError, ValueError):
                    fby = float("nan")
                if not (0.25 <= fby <= 8.0):
                    print(f"[PLS] Power Upscale: final_upscale_by="
                          f"{final_upscale_by!r} is outside [0.25, 8.0] - a "
                          f"conserved artefact, not a dial position. Running "
                          f"the old law (1.0).")
                    fby = 1.0
                if str(resize_method) == _NO_RESIZE and abs(fby - 1.0) > 1e-9:
                    # v582: 'none' has no kernel to reach any other canvas -
                    # the dial is ignored, and it is SAID (the house pattern,
                    # same as upscale_by under 'none' in the stages).
                    print(f"[PLS] Power Upscale: final_upscale_by={fby:.2f} is "
                          f"IGNORED under resize_method='none' - 'none' means "
                          f"the raw model result IS the file (x{fsc:.2f} here). "
                          f"Pick a kernel to put the canvas in your hand.")
                tw, th, grows = _final_canvas(fw, fh, fsc, resize_method, fby)
                if _joint_on and _order == "after pixel" and str(resize_method) != _NO_RESIZE:
                    # v1004: the joint VAE's grid (H3: /16) -- land the canvas
                    # there NOW so the refine's encode never centre-crops it.
                    _sw, _sh = _snap_to(tw, th, _jr_snap)
                    if (_sw, _sh) != (tw, th):
                        print(f"[PLS] Power Upscale: final canvas {tw}x{th} snapped to "
                              f"{_sw}x{_sh} (/{_jr_snap}) for the joint refine that follows")
                        tw, th = _sw, _sh
                if (str(resize_method) != _NO_RESIZE and fsc > 0.0
                        and fby > fsc + 1e-9):
                    # v582: the canvas asks for MORE pixels than the model
                    # builds - the tail of that growth is a plain (antialiased)
                    # resize, not model detail. Legal, but said out loud.
                    print(f"[PLS] Power Upscale: final_upscale_by={fby:.2f} "
                          f"exceeds the model's own x{fsc:.2f} - the model "
                          f"builds x{fsc:.2f}, the kernel stretches the rest. "
                          f"Above the model factor there is no new detail to "
                          f"carry, only interpolation.")
                if bool(dual_moe) and upscale_model is not None:
                    print(f"[PLS] Power Upscale: final pass takes the L wire "
                          f"({wire}); the H wire only serves stage pre-passes, "
                          f"which are pure fit in this mode.")
                print(f"[PLS] Power Upscale: final pass begin -> {wire} "
                      f"x{fsc:.2f} on {fw}x{fh}"
                      + (f" -> {tw}x{th} (resize_method='none': the output IS "
                         f"the model result - no VAE follows, no /8 grid "
                         f"applies)" if str(resize_method) == _NO_RESIZE
                         else f" -> {tw}x{th} (final_upscale_by={fby:.2f}: the "
                              f"USER'S canvas - the model computes x{fsc:.2f}, "
                              f"the {resize_method} fit lands it here, "
                              f"antialiased since v568)"))
                # v591: the canvas is decided HERE, and a video encoder will
                # meet it 18 minutes from now. H.264 subsamples chroma 2x2 and
                # will not open on an odd edge - Frank's 1075x1075 (768 * 1.40)
                # killed a finished run at the save. ph_save now crops one pixel
                # rather than dying, but a warning at the DECISION beats a crop
                # at the exit: say it while the dial is still worth turning.
                #
                # v593: ...and the value it names must be one he can TYPE. The
                # v591 line printed a rounded dial next to unrounded pixels and
                # recommended the number he already had. _even_dial searches the
                # widget's own grid and asks _final_canvas - the same function
                # this pass just used - what each candidate really produces.
                if (int(tw) & 1) or (int(th) & 1):
                    _et, _eb = int(tw) - (int(tw) & 1), int(th) - (int(th) & 1)
                    _sug = _even_dial(fw, fh, fsc, resize_method, fby)
                    print(f"[PLS] Power Upscale: NOTE - {int(tw)}x{int(th)} has "
                          f"an odd edge. H.264 (and every yuv420/422 format) "
                          f"cannot encode odd dimensions; a video save will "
                          f"crop this to {_et}x{_eb}. Images and "
                          f"resize_method='none' are unaffected."
                          + (f" To land even, set final_upscale_by="
                             f"{_sug[0]:.2f} -> {_sug[1]}x{_sug[2]} (verified "
                             f"against the same size law this pass just ran)."
                             if _sug else
                             " No value on the dial's grid can move this canvas "
                             "(the dial is not in play here) - the save's crop "
                             "is the whole answer.")
                          + " Your dial stays yours - this is a note, not a "
                          "change.")
                t_fin = time.monotonic()
                fin = {"k": 0, "t0": t_fin, "last": t_fin, "n": pix_chunks}

                def _on_final(i, j, part, chunks, _p=fin, _n=n_frames):
                    key = "pix:final"
                    if _p["k"] == 0 and int(chunks) != _p["n"]:
                        # The pass may clamp its own chunk for VRAM (v565);
                        # correct the plan, keep truth.
                        clock.resize(key, int(chunks))
                        _p["n"] = int(chunks)
                    now2 = time.monotonic()
                    dt = now2 - _p["last"]
                    _p["last"] = now2
                    _p["k"] += 1
                    clock.measure(key, dt)
                    per = (now2 - _p["t0"]) / max(1, _p["k"])
                    eta = per * max(0, _p["n"] - _p["k"])
                    per_s = (f"{per * 1000:.0f}ms" if per < 0.1
                             else f"{per:.1f}s")
                    if _p["k"] == 1 or per >= 0.1:
                        print(f"[PLS] Power Upscale:   final pixel chunk "
                              f"{_p['k']}/{_p['n']} ({j}/{_n} frames) "
                              f"{per_s}/chunk eta {eta:.0f}s")
                    elif _p["k"] == _p["n"]:
                        print(f"[PLS] Power Upscale:   final pixel chunks "
                              f"2-{_p['n']} folded ({now2 - _p['t0']:.1f}s "
                              f"total, {per_s}/chunk)")
                    if pixel_probe is not None:
                        pixel_probe("final", _p["k"], _p["n"], j, _n, part,
                                    clock.elapsed(), clock.eta())

                with _MuteInfoLogs(mute_staging_logs, label="Power Upscale"):
                    cur = _esrgan_pass(cur, um_fin, tw, th, resize_method,
                                       per_batch, on_chunk=_on_final,
                                       final=True)
                print(f"[PLS] Power Upscale: final pass done in "
                      f"{time.monotonic() - t_fin:.1f}s -> "
                      f"{int(cur.shape[2])}x{int(cur.shape[1])} "
                      f"(this detail never meets a VAE)")

        if _joint_on and _order == "after pixel":
            # v1004: the hires-fix -- the refine on the finished canvas.
            _c = None
            if _jr_hit is not None:
                _c = (_jr_hit, n_frames, int(_hw), int(_hh))
            cur, _jr_info = _joint_refine(model, positive, negative, vae, cur, latent,
                                          sigmas, seed, steps, cfg, sampler_name,
                                          scheduler, denoise, audio_stream, clock,
                                          probe=probe, mute=mute_staging_logs, cached=_c,
                                          vae_tiling=vae_tiling)
            if _jr_hit is None:
                _latent_cache_put(_jr_key, _jr_info["vid_lat"], "after pixel")
        # v1008: WHERE the pixel model sat this run -- the one fact behind
        # Frank's 25.09. question ("arbeiten wir doppelt?"), on the node.
        _fin_wire = upscale_model_low if bool(dual_moe) else upscale_model
        if str(pixel_stage) == "fit only":
            _where = "none"
        elif _joint is not None:
            _where = ("none" if _fin_wire is None else
                      ("front" if (_joint_on and _order == "after pixel") else "behind"))
        elif _final_runs:   # the tile path: 'model final' (or 'before pixel') or 'off'
            _where = "behind" if _fin_wire is not None else "none"
        else:
            _where = ("front" if (upscale_model is not None or
                                  (bool(dual_moe) and upscale_model_low is not None))
                      else "none")
        if _joint is not None:
            _verdict = _joint_verdict(_order, _joint_on,
                                      (_jr_info or {}).get("sigmas_n", 0),
                                      (_jr_info or {}).get("steps_run", 0),
                                      (_jr_info or {}).get("start_sigma", 0.0),
                                      str(audio_stream), _patch_count(model),
                                      _jr_frames_in, int(cur.shape[0]), why=_joint_why,
                                      where=_where)
            if _jr_info and _jr_info.get("cached"):
                _verdict += " \u00b7 latent from cache"
        else:
            _verdict = _tile_verdict(_order, 0 if _tile_off else len(plans), _tile_tiles,
                                     _tile_steps, _tile_start or 0.0,
                                     (len(_curve) - 1) if _curve else 0, _where,
                                     why="refine_order=off" if _tile_off else "")
        if _verdict:
            print(f"[PLS] Power Upscale: {_verdict}")
        video_out = _build_video(cur, audio, frame_rate) if video is not None else None
        preview = _emit_result_preview(cur, frame_rate if video is not None else None,
                                       bool(result_preview))
        print(f"[PLS] Power Upscale: done mode={'High + Low' if dual_moe else 'Single'} "
              f"frames={int(cur.shape[0])} out={int(cur.shape[2])}x{int(cur.shape[1])} "
              f"video={'yes (audio+fps passthrough)' if video_out is not None else 'none'} "
              f"tile_seeds=frame-independent "
              f"preview={('%d jpeg' % len(preview)) if preview else 'off'} "
              f"total={time.monotonic() - t_all:.1f}s"
              f" (node wall clock; the ComfyUI badge adds executor overhead "
              f"outside this function)")
        return {"ui": {"pls_pu_preview": preview,
                       # v1004: ONE line the frontend shows -- the node says
                       # what it did. v1008: for EVERY model, plus the path
                       # ('joint' / 'tile') the frontend hides unread dials by.
                       "pls_pu_verdict": [_verdict],
                       "pls_pu_path": ["joint" if _joint is not None else "tile"]},
                "result": (cur, video_out)}
