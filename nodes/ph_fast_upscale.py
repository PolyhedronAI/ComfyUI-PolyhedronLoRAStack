"""Polyhedron Fast Upscale (v554)

The FAST path next to Power Upscale: [optional ESRGAN] -> resize. No
diffusion refine - 65 frames in seconds, not minutes. Born from dissecting
KJNodes' ImageResizeKJv2 (GPL-3.0: STUDIED and RE-IMPLEMENTED, never copied;
see docs). The two mechanisms worth having:

  1. GPU residency: comfy's common_upscale is torch interpolate - moved onto
     model_management.get_torch_device() it flies. lanczos is the exception
     (a PIL path), so it is honestly labelled "lanczos (cpu)" and refuses a
     gpu device selection (fail loud, the KJ lesson).
  2. NVIDIA Maxine VideoSuperRes via the nvvfx library: a per-frame API
     (no batching), CHW CUDA frames in, DLPack out, target snapped to /8.
     Hard import with a clear message when the library is missing.

House pattern: _resolve_input / _build_video / _MODEL_UPSCALER /
_MuteInfoLogs is IMPORTED from ph_power_upscale - one source
of truth, ph_power_upscale.py stays byte-identical to its v553 baseline.
Telemetry follows v553: a begin line with the full plan, a done line with
the MEASURED duration and ms/frame.
"""
import time

import torch

try:  # package load (ComfyUI) vs direct module load (tools)
    from .ph_power_upscale import (_METHODS, _MODEL_UPSCALER, _MuteInfoLogs,
                                   _NO_RESIZE,
                                   _build_video, _chunks, _fit_method,
                                   _resize_chunked,
                                   _resolve_input, _vsr_resize)
except ImportError:  # pragma: no cover
    import os as _os
    import sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from ph_power_upscale import (_METHODS, _MODEL_UPSCALER, _MuteInfoLogs,
                                  _NO_RESIZE,
                                  _build_video, _chunks, _fit_method,
                                  _resize_chunked,
                                  _resolve_input, _vsr_resize)



def _target(size_mode, upscale_by, width, height, src_w, src_h,
            divisible_by, resize_method):
    """Pure math -> (tw, th). factor: scale both sides; exact: as given, a
    zero side keeps the aspect from the other (both zero -> error). Snap
    DOWN to divisible_by (the KJ convention), floored at divisible_by
    itself; nvidia_rtx_vsr then snaps to /8 nearest (the Maxine contract).
    """
    if str(size_mode) == "factor":
        f = float(upscale_by)
        if f <= 0:
            raise ValueError("Fast Upscale: upscale_by must be > 0.")
        tw, th = round(src_w * f), round(src_h * f)
    else:
        tw, th = int(width), int(height)
        if tw <= 0 and th <= 0:
            raise ValueError("Fast Upscale: exact mode needs width or "
                             "height > 0 (a zero side keeps the aspect).")
        if tw <= 0:
            tw = round(src_w * th / src_h)
        if th <= 0:
            th = round(src_h * tw / src_w)
    d = max(1, int(divisible_by))
    tw = max(d, (tw // d) * d)
    th = max(d, (th // d) * d)
    if str(resize_method) == "nvidia_rtx_vsr":
        tw = max(8, round(tw / 8) * 8)
        th = max(8, round(th / 8) * 8)
    return int(tw), int(th)


# ── v559: keep_proportion (the Resize-v2 gap) ─────────────────────────────
# stretch (the v554 behaviour) distorts when the aspect ratios differ. crop
# COVERS the target and cuts the overhang; pad CONTAINS the image and fills the
# rest. Both are pure math here, so the guard can execute them.
def _proportion_plan(src_w, src_h, dst_w, dst_h, keep):
    """-> (resize_w, resize_h). stretch: the target itself. crop: scale by the
    LARGER ratio (cover), then cut. pad: scale by the SMALLER ratio (contain),
    then fill."""
    if str(keep) == "stretch":
        return int(dst_w), int(dst_h)
    rw, rh = float(dst_w) / max(1, src_w), float(dst_h) / max(1, src_h)
    r = max(rw, rh) if str(keep) == "crop" else min(rw, rh)
    return max(1, round(src_w * r)), max(1, round(src_h * r))


def _offset(outer, inner, position, axis):
    """Where the inner rect sits inside the outer one (crop AND pad share it)."""
    slack = max(0, int(outer) - int(inner))
    p = str(position)
    if axis == "x":
        return 0 if p == "left" else (slack if p == "right" else slack // 2)
    return 0 if p == "top" else (slack if p == "bottom" else slack // 2)


def _parse_color(text):
    """'0, 0, 0' / '255,255,255' -> a 0..1 RGB triple. Anything unparseable is
    black (fail soft - a pad colour must never abort a render)."""
    try:
        parts = [float(p) for p in str(text).replace(";", ",").split(",")]
        if len(parts) == 1:
            parts = parts * 3
        if len(parts) < 3:
            return (0.0, 0.0, 0.0)
        return tuple(min(1.0, max(0.0, v / 255.0)) for v in parts[:3])
    except Exception:
        return (0.0, 0.0, 0.0)


def _crop_or_pad(frames, dst_w, dst_h, keep, position, color, is_mask=False):
    """Cut the overhang (crop) or fill the rest (pad). Frames are [N,H,W,C]
    (or [N,H,W] for a mask). A padded mask border is 0 = not part of the
    source, which is the honest answer."""
    if str(keep) == "stretch":
        return frames
    h, w = int(frames.shape[1]), int(frames.shape[2])
    if str(keep) == "crop":
        x = _offset(w, dst_w, position, "x")
        y = _offset(h, dst_h, position, "y")
        return frames[:, y:y + int(dst_h), x:x + int(dst_w), ...]
    x = _offset(dst_w, w, position, "x")
    y = _offset(dst_h, h, position, "y")
    if is_mask:
        canvas = torch.zeros((frames.shape[0], int(dst_h), int(dst_w)),
                             dtype=frames.dtype, device=frames.device)
        canvas[:, y:y + h, x:x + w] = frames
        return canvas
    c = int(frames.shape[3])
    canvas = torch.empty((frames.shape[0], int(dst_h), int(dst_w), c),
                         dtype=frames.dtype, device=frames.device)
    for ch in range(c):
        canvas[..., ch] = color[ch] if ch < 3 else 1.0
    canvas[:, y:y + h, x:x + w, :] = frames
    return canvas


def _esrgan_chunked(frames, upscale_model, per_batch):
    """The wired UPSCALE_MODEL via core (spandrel), fed in sub-batches so a
    65-frame 4x intermediate never has to exist as ONE tensor. No model
    upscaler in this build -> honest fallback (the ph_power_upscale
    convention), stated in the log."""
    if _MODEL_UPSCALER is None:
        print("[PLS] Fast Upscale: core model upscaler unavailable - "
              "skipping the ESRGAN pass (plain resize)")
        return frames
    out = []
    for i, j in _chunks(int(frames.shape[0]), per_batch):
        (part,) = _MODEL_UPSCALER.upscale(upscale_model, frames[i:j])
        out.append(part.cpu())
    return torch.cat(out, dim=0)


class ULSFastUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_mode": (["factor", "exact"],
                              {"default": "factor",
                               "tooltip": "factor: scale both sides by "
                                          "upscale_by (the Power Upscale "
                                          "way). exact: hit width x height "
                                          "(the Resize-v2 way; a zero side "
                                          "keeps the aspect)."}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.05,
                                         "max": 8.0, "step": 0.05,
                                         "tooltip": "factor mode: scale for "
                                                    "both sides."}),
                "width": ("INT", {"default": 1024, "min": 0, "max": 8192,
                                  "step": 8,
                                  "tooltip": "exact mode: target width "
                                             "(0 = derive from height)."}),
                "height": ("INT", {"default": 1024, "min": 0, "max": 8192,
                                   "step": 8,
                                   "tooltip": "exact mode: target height "
                                              "(0 = derive from width)."}),
                "resize_method": (list(_METHODS.keys()),
                                  {"default": "bicubic",
                                   "tooltip": "bicubic/bilinear/area/nearest "
                                              "run on gpu OR cpu. lanczos is "
                                              "PIL = cpu-only. nvidia_rtx_vsr "
                                              "= Maxine VideoSuperRes "
                                              "(needs nvvfx + RTX; target "
                                              "snaps to /8)."}),
                "device": (["gpu", "cpu"],
                           {"default": "gpu",
                            "tooltip": "Where the torch resize runs. "
                                       "lanczos refuses gpu (fail loud); "
                                       "nvidia_rtx_vsr always runs on the "
                                       "GPU regardless."}),
                "divisible_by": ("INT", {"default": 16, "min": 1, "max": 64,
                                         "tooltip": "Snap the target DOWN to "
                                                    "a multiple (Wan-friendly "
                                                    "default 16)."}),
                "per_batch": ("INT", {"default": 32, "min": 0, "max": 512,
                                      "tooltip": "Frames per sub-batch for "
                                                 "ESRGAN + resize (VRAM "
                                                 "guard). 0 = whole batch. "
                                                 "nvidia_rtx_vsr is per-frame "
                                                 "by SDK design."}),
                "mute_staging_logs": ("BOOLEAN",
                                      {"default": True,
                                       "label_on": "On", "label_off": "Off",
                                       "tooltip": "Silence ComfyUI's model "
                                                  "staging INFO lines while "
                                                  "this node runs (restored "
                                                  "byte-exactly after)."}),
                # ── v559: the Resize-v2 gap. Appended LAST (serialisation law). ──
                "keep_proportion": (["stretch", "crop", "pad"],
                                    {"default": "stretch",
                                     "tooltip": "What to do when the aspect ratios "
                                                "differ. stretch = distort to the "
                                                "target (the stretch behaviour). crop = "
                                                "cover the target and cut the "
                                                "overhang. pad = fit inside and fill "
                                                "the rest with pad_color."}),
                "crop_position": (["center", "top", "bottom", "left", "right"],
                                  {"default": "center",
                                   "tooltip": "Where the image sits - used for BOTH "
                                              "the crop cut and the pad placement."}),
                "pad_color": ("STRING", {"default": "0, 0, 0",
                                         "tooltip": "Fill colour for pad, as R, G, B "
                                                    "(0-255). Unparseable = black "
                                                    "(fail soft)."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Frame input [N,H,W,C]. Wire "
                                               "EXACTLY one of image/video."}),
                "video": ("VIDEO", {"tooltip": "Native VIDEO input; audio + "
                                               "frame rate ride through into "
                                               "the VIDEO output. Wire "
                                               "EXACTLY one of image/video."}),
                "upscale_model": ("UPSCALE_MODEL",
                                  {"tooltip": "Optional ESRGAN pass BEFORE "
                                              "the resize (core spandrel "
                                              "path, sub-batched)."}),
                "mask": ("MASK", {"tooltip": "Optional mask - resized with the "
                                             "SAME geometry (crop/pad included). "
                                             "A padded border is 0: it was never "
                                             "part of the source."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "INT", "INT", "MASK")
    RETURN_NAMES = ("image", "video", "width", "height", "mask")
    FUNCTION = "upscale"
    CATEGORY = "Polyhedron/Upscaling"
    DESCRIPTION = ("Plain resize and model-based upscaling in one node, for IMAGE, VIDEO "
                   "and MASK together - the mask follows the image instead of drifting out "
                   "of register. Works through long batches in chunks so a clip does not "
                   "have to fit in VRAM all at once, and can snap the result to the divisor "
                   "the samplers downstream expect.")

    def upscale(self, size_mode, upscale_by, width, height, resize_method,
                device, divisible_by, per_batch, mute_staging_logs,
                keep_proportion="stretch", crop_position="center",
                pad_color="0, 0, 0",
                image=None, video=None, upscale_model=None, mask=None):
        frames, audio, frame_rate = _resolve_input(image, video)
        n = int(frames.shape[0])
        src_h, src_w = int(frames.shape[1]), int(frames.shape[2])
        tw, th = _target(size_mode, upscale_by, width, height, src_w, src_h,
                         divisible_by, resize_method)
        if resize_method == "nvidia_rtx_vsr" and device == "cpu":
            raise ValueError("Fast Upscale: nvidia_rtx_vsr runs on the GPU "
                             "- set device=gpu or pick another method.")
        color = _parse_color(pad_color)
        print(f"[PLS] Fast Upscale: begin -> {src_w}x{src_h} \u2192 {tw}x{th} "
              f"method={resize_method} device={device} frames={n} "
              f"esrgan={'yes' if upscale_model is not None else 'no'} "
              f"keep={keep_proportion}"
              f"{'/' + str(crop_position) if keep_proportion != 'stretch' else ''} "
              f"mask={'yes' if mask is not None else 'no'} "
              f"per_batch={int(per_batch)}")
        t0 = time.monotonic()
        with _MuteInfoLogs(mute_staging_logs, label="Fast Upscale"):
            if upscale_model is not None:
                frames = _esrgan_chunked(frames, upscale_model, per_batch)
            cur_h, cur_w = int(frames.shape[1]), int(frames.shape[2])
            if str(resize_method) == _NO_RESIZE:
                # v568: THE PURE MODEL PASS. No filter, no crop, no pad - width /
                # height / keep_proportion are ignored (loudly) and the output is
                # EXACTLY what the model produced. This is the node to hang behind
                # a Power Upscale: the model's detail goes straight to the file and
                # never through a VAE, where an 8x spatial compression would erase
                # it. With no model wired this is a pass-through.
                print(f"[PLS] Fast Upscale: resize_method='none' -> output is "
                      f"{cur_w}x{cur_h}, exactly what "
                      f"{'the upscale model produced' if upscale_model is not None else 'came in'}"
                      f" (width={int(tw)} height={int(th)} keep_proportion="
                      f"'{keep_proportion}' are IGNORED - no filter touches these "
                      f"pixels)")
                tw, th = cur_w, cur_h
                rw, rh = cur_w, cur_h
                if mask is not None:
                    m = mask if mask.dim() == 3 else mask.unsqueeze(0)
                    m = _resize_chunked(m.unsqueeze(-1), rw, rh, "bilinear",
                                        device, per_batch)
                    mask_out = m.squeeze(-1).clamp(0.0, 1.0)
                else:
                    mask_out = torch.zeros((n, int(th), int(tw)),
                                           dtype=frames.dtype)
            else:
                # v559: the fit runs on the plan (cover / contain / exact), and the
                # method is checked against the DIRECTION of that fit (vsr upscales).
                rw, rh = _proportion_plan(cur_w, cur_h, tw, th, keep_proportion)
                method, note = _fit_method(cur_w, cur_h, rw, rh, resize_method)
                if note:
                    print(f"[PLS] Fast Upscale: method={method} ({note})")
                if cur_w != rw or cur_h != rh:
                    if method == "nvidia_rtx_vsr":
                        frames = _vsr_resize(frames, rw, rh)
                    else:
                        frames = _resize_chunked(frames, rw, rh, method,
                                                 device, per_batch)
                frames = _crop_or_pad(frames, tw, th, keep_proportion,
                                      crop_position, color)
                if mask is not None:
                    m = mask if mask.dim() == 3 else mask.unsqueeze(0)
                    m = m.unsqueeze(-1)                  # [N,H,W,1] for the resize
                    m = _resize_chunked(m, rw, rh, "bilinear", device, per_batch)
                    m = _crop_or_pad(m, tw, th, keep_proportion, crop_position,
                                     color)
                    mask_out = m.squeeze(-1).clamp(0.0, 1.0)
                else:
                    mask_out = torch.zeros((n, int(th), int(tw)),
                                           dtype=frames.dtype)
        dur = time.monotonic() - t0
        print(f"[PLS] Fast Upscale: done in {dur:.1f}s "
              f"({dur / max(1, n) * 1000.0:.0f} ms/frame) out={tw}x{th}")
        video_out = (_build_video(frames, audio, frame_rate)
                     if video is not None else None)
        return (frames, video_out, int(tw), int(th), mask_out)

