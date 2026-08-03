"""
ph_media_loader.py — ⬡ Polyhedron Media Loader (ULSMediaLoader)

A Load-Image/Video node that loads from a PINNED server folder (chosen via the
node's "Choose folder" dialog and persisted in the workflow, so it survives a
reload) and shows a THUMBNAIL grid of that folder's media. The frontend
(web/js/ph_media_loader.js) drives a server-side folder browser + thumbnail/list
routes (nodes/uls_routes.py: /pls/media/*) and writes the chosen item into this
node's `media_ref` widget; the backend just decodes whatever `media_ref` points at.

Outputs:
  image        — IMAGE [N,H,W,3] (N=1 for a still; N=frames for a video clip)
  mask         — MASK  [N,H,W]   (alpha-derived for stills; a grayscale still
                                  without alpha carries its luminance as the
                                  mask, white -> 1; zeros for video)
  video        — VIDEO           (single-file video losslessly with its own audio;
                                  an image batch or a lone still as a silent clip;
                                  None only when there are no frames)
  frame_count  — INT             (1 for a still)
  fps          — FLOAT           (0.0 for a still; native or forced for video)
  width        — INT             (decoded frame width)
  height       — INT             (decoded frame height)
  filename     — STRING          (the loaded file's name; empty for sequence mode)
  video_path   — STRING          (v639: the loaded file's FULL path — feeds the
                                  Batch Pipeline Source as its navigator; empty in
                                  modes without a single source file)
  audio        — AUDIO           (Stufe B: the paired audio decoded to
                                  {waveform,sample_rate}; None if none paired)
  video_audio  — VIDEO           (Stufe B: the visual's frames muxed with the
                                  paired audio — wire to SaveVideo for an mp4 with
                                  a sound track; None if no audio paired)

Audio (v457 Stufe A -> v459 Stufe B): audio is a browsable/previewable media kind
in the grid (faux-waveform tile, hover-play, click-to-select, play in the
Selection) and can be PAIRED with a visual (v458). Stufe B (v459) adds the graph
outputs: `audio` (the decoded AUDIO dict, via the same PyAV path as core
LoadAudio) and `video_audio` (the visual's frames muxed with the paired audio as
a native comfy_api VIDEO). The paired audio rides in media_ref as ref['audio'];
an audio-only selection still emits a tiny black placeholder frame on the IMAGE
output so the graph never crashes, while the AUDIO output carries the sound.

Dependencies: Pillow for images (already an optional pack dep). Video decoding
needs OpenCV (`opencv-python`, import name cv2) and degrades gracefully: if a
video is selected and cv2 is absent, the node raises a clear, actionable error
instead of failing obscurely. Images never need cv2.

`media_ref` is JSON: {"folder": <abs dir>, "file": <name>, "kind": "image"|"video"|"audio"}.
The folder is an absolute path the user pinned; this node is a LOCAL tool, so it
reads from anywhere the user points it (the file is realpath-checked to exist).
"""

import os
import json
import time

import numpy as np
import torch

# Pillow is the image decoder. Kept import-guarded so a missing Pillow yields a
# clear message rather than an import-time crash of the whole pack.
try:
    from PIL import Image, ImageOps
    _HAS_PIL = True
except Exception:
    Image = None
    ImageOps = None
    _HAS_PIL = False

# OpenCV is the (optional) video decoder. Absent -> images still work; video
# selection raises a clear error. Mirrors the pack's trimesh/Pillow pattern.
try:
    import cv2  # opencv-python
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False

# PyAV (av) is the (optional) alpha-aware video decoder. OpenCV throws the alpha
# channel away, so a per-frame mask from a transparent video (VP9-alpha WEBM,
# ProRes 4444, transparent GIF) needs PyAV. Absent -> video still loads via
# OpenCV with a blank mask. PyAV ships with recent ComfyUI.
try:
    import av  # PyAV
    _HAS_AV = True
except Exception:
    av = None
    _HAS_AV = False

# ComfyUI's VIDEO type (optional). VideoFromFile wraps a media file losslessly
# (audio rides along — get_components() recovers it), VideoFromComponents builds
# a VIDEO from an image batch (silent). The comfy_api.input_impl /
# comfy_api.util paths are version-stable back-compat shims, so this single
# import works on current ComfyUI; absent -> the VIDEO output is simply None.
try:
    from comfy_api.input_impl import VideoFromFile, VideoFromComponents
    from comfy_api.util import VideoComponents
    from fractions import Fraction
    _HAS_VIDEO_API = True
except Exception:
    VideoFromFile = VideoFromComponents = VideoComponents = None
    _HAS_VIDEO_API = False

from .ph_media_util import (pix_fmt_has_alpha, rgb_and_mask_from_rgba,
                            rgb_and_mask_from_still,
                            select_frames, frames_target_and_offenders,
                            order_names, select_slice)

# ── Video timing constants (fps rates + still-video framing) ────────────────
# D4 (v483): the clip-rate fps constants and the paired still-video frame cap,
# gathered here from their formerly scattered definitions so the whole timing
# model reads in one place. Values and names are byte-for-byte unchanged from
# v482; every consumer still references these names — no magic-number fps literal
# bypasses them (verified module-wide).
#
#   _BATCH_VIDEO_FPS         synthesized image-batch video has no native rate;
#                            the silent-batch fallback when force_fps is 0. 16
#                            matches the WAN T2V/I2V convention in this stack.
#                            v478: a multi-frame batch falls back to THIS (16),
#                            NOT _DEFAULT_CLIP_FPS (24) — pairing audio must not
#                            re-time the same frames 16->24 (faster/shorter).
#                            See load()/_emit (kind == "batch").
#   _DEFAULT_CLIP_FPS        fallback output fps when a clip — or a still expanded
#                            to an audio length — has no native rate and none is
#                            forced. Numerically equals the force_fps widget
#                            default (24.0), but kept as a SEPARATE constant on
#                            purpose: the internal fallback and the UI default are
#                            conceptually distinct knobs (do not couple blindly).
#   _STILL_VIDEO_FPS         a still paired with audio is held for the audio's
#   _STILL_VIDEO_MAX_FRAMES  duration as a static "cover" (not a one-frame flash)
#                            at this low rate, capped so a long track can't blow
#                            up RAM (the video-visual path never tiles — it uses
#                            the real frames).
_BATCH_VIDEO_FPS = 16
_DEFAULT_CLIP_FPS = 24
_STILL_VIDEO_FPS = 2
_STILL_VIDEO_MAX_FRAMES = 600
# ────────────────────────────────────────────────────────────────────────────


_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".gif")  # gif: either path

# v457 (Stufe A): audio is a third browsable media kind. This set drives KIND
# DETECTION + grid display only — every entry shows as a tile and is selectable;
# whether the BROWSER can actually play it on hover/in the Selection is a separate
# matter (mp3/wav/ogg/flac/m4a/aac/opus play; .aiff/.wma typically show but stay
# silent — a documented limitation, not a bug). Audio has no graph output in
# Stufe A; load() routes it to a graceful placeholder (see _audio_placeholder).
_AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus",
               ".aiff", ".aif", ".wma")

# Image extensions eligible for the image-batch path. .gif decodes to its first
# frame via Pillow here (the same way _load_image treats a still gif).
_BATCH_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif")


def _is_batch_image(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in _BATCH_IMAGE_EXTS


def _is_proc_media(name: str) -> bool:
    """Batch-processing iterates over BOTH stills and videos (unlike the image-batch
    path, which is stills only). A file counts if it's a loadable image or video."""
    ext = os.path.splitext(name)[1].lower()
    return ext in _IMAGE_EXTS or ext in _VIDEO_EXTS


def _batch_cfg(batch_config):
    """Parse the Batch panel's batch_config JSON into a normalized dict with safe
    defaults. Single source of truth for both load() and IS_CHANGED.

    v528 (unified batch): batch_config is now the ONE config for both batch modes.
    New fields — `mode` ("" legacy | "frames" | "proc"), an optional explicit
    `selection` (the checked-files set; None = legacy rule pipeline), and the
    proc-side fields (start_at / wrap / reset_seq) so the proc mode can run
    entirely from here. proc_config stays readable for old workflows (see
    _load_strict's precedence)."""
    try:
        c = json.loads(batch_config) if batch_config else {}
        if not isinstance(c, dict):
            c = {}
    except Exception:
        c = {}
    sel = c.get("selection")
    if isinstance(sel, dict):
        sel = sel.get("names")
    if isinstance(sel, list):
        sel = [str(n) for n in sel if str(n)]
        sel = sel if sel else None
    else:
        sel = None
    return {
        "enabled": bool(c.get("enabled", False)),
        "mode": str(c.get("mode", "")) if c.get("mode") in ("frames", "proc") else "",
        "source": str(c.get("source", "")),
        "sort_mode": str(c.get("sort_mode", "name (natural)")),
        "name_filter": str(c.get("name_filter", "*")) or "*",
        "every_nth": int(c.get("every_nth", 1) or 1),
        "resize_method": str(c.get("resize_method", "none (strict)")),
        "selection": sel,
        "wrap": bool(c.get("wrap", False)),
        "start_at": int(c.get("start_at", 0) or 0),
        "reset_seq": int(c.get("reset_seq", 0) or 0),
    }

# Cap a single decoded clip so an accidental multi-thousand-frame video can't
# exhaust VRAM/RAM silently. The user's frame_load_cap (when >0) wins under this.
_HARD_FRAME_CAP = 2000

# Hard ceiling on how many files the batch-processing cursor will walk. Large
# enough that a real working folder is never truncated, finite for safety.
_PROC_LIST_CAP = 1_000_000


def _proc_cfg(proc_config):
    """Parse the Batch Processing panel's proc_config JSON. Single source of truth
    for both load() and IS_CHANGED. `enabled` drives the third (iterative) mode."""
    try:
        c = json.loads(proc_config) if proc_config else {}
        if not isinstance(c, dict):
            c = {}
    except Exception:
        c = {}
    return {
        "enabled": bool(c.get("enabled", False)),
        "source": str(c.get("source", "")),
        "sort_mode": str(c.get("sort_mode", "name (natural)")),
        "name_filter": str(c.get("name_filter", "*")) or "*",
        "wrap": bool(c.get("wrap", False)),
        "start_at": int(c.get("start_at", 0) or 0),
        "reset_seq": int(c.get("reset_seq", 0) or 0),
    }


def _selection_names(folder, selection, sort_mode, kind_check):
    """Materialize an explicit checked-files set against reality: intersect the
    stored names with what actually exists in `folder` (and passes `kind_check`),
    then order by sort_mode. Vanished files silently drop out — the set is a
    wish-list, the disk is the truth."""
    try:
        existing = {e.name for e in os.scandir(folder)
                    if e.is_file() and kind_check(e.name)}
    except OSError as ex:
        raise RuntimeError(f"[PLS] MediaLoader: cannot read "
                               f"folder '{folder}': {ex}") from ex
    names = [n for n in selection if n in existing]
    times = {}
    if sort_mode in ("mtime (oldest first)", "created"):
        for n in names:
            try:
                st = os.stat(os.path.join(folder, n))
                times[n] = (st.st_mtime if sort_mode.startswith("mtime")
                            else getattr(st, "st_ctime", st.st_mtime))
            except OSError:
                times[n] = 0.0
    return order_names(names, sort_mode, times=times)


def _proc_resolve_files(folder, sort_mode, name_filter, selection=None):
    """Ordered list of the folder's media files (stills + videos) the cursor walks.
    v528: an explicit selection (the checked set) wins over the rule pipeline;
    otherwise unchanged — the image-batch ordering pipeline with no skip/nth and
    an effectively unbounded cap (per-file frame caps still apply on decode)."""
    if selection:
        return _selection_names(folder, selection, sort_mode, _is_proc_media)
    names = [e.name for e in os.scandir(folder)
             if e.is_file() and _is_proc_media(e.name)]
    times = {}
    if sort_mode in ("mtime (oldest first)", "created"):
        for n in names:
            try:
                st = os.stat(os.path.join(folder, n))
                times[n] = (st.st_mtime if sort_mode.startswith("mtime")
                            else getattr(st, "st_ctime", st.st_mtime))
            except OSError:
                times[n] = 0.0
    return select_frames(names, sort_mode, name_filter, 0, 1, 0,
                         _PROC_LIST_CAP, times=times)


# Batch-processing cursor state — one entry per node instance (keyed by the
# hidden UNIQUE_ID), mirroring uls_sampler's per-node live-preview registry. In
# process only: a ComfyUI restart sensibly starts each batch fresh. {idx, seq}
# remembers the position and which reset_seq it belongs to, so bumping reset_seq
# (Reset / jump-to-file in the panel) re-homes the cursor on the next run.
_PROC_CURSOR = {}


def _proc_pick(node_id, total, pcfg):
    """Return (index to emit THIS run, is_last, loop) and store the cursor for the
    NEXT run. Honors reset_seq (re-home to start_at) and wrap (loop vs clamp at
    the end). `loop` counts the pass number (1 on the first sweep; increments
    each time wrap carries the cursor past the end) — the batch_info pin shows it.
    `total` must be >= 1 (the caller raises on an empty folder)."""
    nid = str(node_id)
    start = min(max(0, pcfg["start_at"]), total - 1)
    st = _PROC_CURSOR.get(nid)
    if st is None or st.get("seq") != pcfg["reset_seq"]:
        idx, loop = start, 1
    else:
        idx = min(max(0, int(st.get("idx", start))), total - 1)
        loop = max(1, int(st.get("loop", 1)))
    is_last = idx >= total - 1
    wrapped = pcfg["wrap"] and is_last
    nxt = ((idx + 1) % total) if pcfg["wrap"] else min(idx + 1, total - 1)
    _PROC_CURSOR[nid] = {"idx": nxt, "seq": pcfg["reset_seq"],
                         "loop": loop + (1 if wrapped else 0)}
    return idx, is_last, loop


def _clear_proc_cursor(node_id):
    _PROC_CURSOR.pop(str(node_id), None)


def _kind_for(name: str) -> str:
    ext = os.path.splitext(name)[1].lower()
    if ext in _VIDEO_EXTS and ext != ".gif":
        return "video"
    if ext in _AUDIO_EXTS:
        return "audio"
    return "image"


def _to_image_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 -> 1xHxWx3 float32 in [0,1]."""
    arr = rgb_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr[None]))


def _decode_image_rgba(path: str):
    """Open an image, honour EXIF orientation, split to (rgb, mask).

    Returns rgb [H,W,3] float32 in [0,1] and mask [H,W] float32 (ComfyUI alpha
    convention: opaque -> 0, transparent -> 1). Shared by the single-image path
    and the image-batch path so both decode identically."""
    if not _HAS_PIL:
        raise RuntimeError("[PLS] MediaLoader: Pillow is required to load images "
                           "(pip install pillow).")
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # honour camera orientation
    rgba = np.array(img.convert("RGBA")).astype(np.float32) / 255.0       # [H,W,4]
    # v654: a grayscale still without alpha IS a mask (white -> 1, the Save's
    # mode-L round-trip); real alpha keeps the LoadImage convention (opaque -> 0).
    rgb, mask, _ = rgb_and_mask_from_still(rgba)
    return rgb, mask


def _load_image(path: str):
    rgb, mask = _decode_image_rgba(path)
    image_t = torch.from_numpy(np.ascontiguousarray(rgb[None]))          # [1,H,W,3]
    mask_t = torch.from_numpy(np.ascontiguousarray(mask[None]))          # [1,H,W]
    return image_t, mask_t, 1, 0.0


def _audio_placeholder():
    """v457 (Stufe A): audio has no graph output yet — a dedicated AUDIO socket
    arrives in Stufe B. When an audio file is the active selection and the graph
    is queued, return a tiny black frame [1,64,64,3] + empty mask [1,64,64] so the
    node degrades gracefully instead of feeding audio bytes into the image decoder.
    Matches _load_image's shape contract; n=1, fps=0.0; the caller's _make_video
    call (mode "image") yields video=None, so the VIDEO output is empty too."""
    h = w = 64
    image_t = torch.zeros((1, h, w, 3), dtype=torch.float32)             # [1,H,W,3]
    mask_t = torch.zeros((1, h, w), dtype=torch.float32)                 # [1,H,W]
    return image_t, mask_t, 1, 0.0


class _NothingToLoad(RuntimeError):
    """v509: 'there is simply NOTHING to load' -- an EMPTY state, not a defect.
    load() turns this into the built-in placeholder when on_empty=='placeholder';
    misconfiguration (mode on without a source folder, unreadable folder,
    malformed selection JSON) and decode failures keep raising hard."""


_PLACEHOLDER_SIZE = 512


def _placeholder_media_np():
    """v509: deterministic built-in dummy frame (pure numpy -- CI-testable
    without torch): a dark checkerboard with a thin accent border, instantly
    readable as 'no media here'. Emitted when there is nothing to load and
    on_empty=='placeholder' (Frank: the node should show/emit SOMETHING sane
    instead of a red error when the folder / selection is simply empty)."""
    s = _PLACEHOLDER_SIZE
    yy, xx = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
    checker = (((yy // 32) + (xx // 32)) % 2).astype(np.float32)
    rgb = np.empty((s, s, 3), dtype=np.float32)
    rgb[..., 0] = 0.10 + 0.05 * checker
    rgb[..., 1] = 0.10 + 0.05 * checker
    rgb[..., 2] = 0.11 + 0.05 * checker
    b = 6
    edge = (yy < b) | (yy >= s - b) | (xx < b) | (xx >= s - b)
    rgb[edge] = np.array([0.90, 0.49, 0.38], dtype=np.float32)   # the pack's accent
    return rgb


def _placeholder_media():
    """torch edge of _placeholder_media_np -- same contract as _load_image:
    image [1,H,W,3], mask [1,H,W] (empty -- nothing is 'subject'), n=1, fps=0.0
    (a still; _emit's mode-'image' path yields video=None, mirroring the
    _audio_placeholder contract above)."""
    rgb = _placeholder_media_np()
    image_t = torch.from_numpy(rgb).unsqueeze(0)
    mask_t = torch.zeros((1, rgb.shape[0], rgb.shape[1]), dtype=torch.float32)
    return image_t, mask_t, 1, 0.0


def _slice_frames(stack, native_fps, trim_start, trim_end):
    """v484 (D1 Stufe 2): keep the [trim_start, total - trim_end] window (seconds) of a
    decoded frame stack [N,...], mapped to frame indices via native_fps. Clamped so at
    least one frame survives; zero trim (or no usable fps / <=1 frame) -> unchanged.
    Mirrors _slice_audio's window semantics so the video- and audio-trim handles agree."""
    nf = float(native_fps or 0.0)
    ts = max(0.0, float(trim_start or 0.0))
    te = max(0.0, float(trim_end or 0.0))
    n = int(stack.shape[0])
    if nf <= 0.0 or n <= 1 or (ts <= 0.0 and te <= 0.0):
        return stack
    s = max(0, int(round(ts * nf)))
    e = n - max(0, int(round(te * nf)))
    if e - s < 1:
        e = min(n, s + 1)
    s = min(s, max(0, e - 1))
    return stack[s:e]


def _load_video(path: str, frame_load_cap: int, frame_skip: int, force_fps: float,
                vtrim_start: float = 0.0, vtrim_end: float = 0.0):
    """Decode a video to an IMAGE batch [N,H,W,3] + a MASK batch [N,H,W].

    Alpha-bearing video (VP9-alpha WEBM, ProRes 4444, transparent GIF) is
    decoded via PyAV so the alpha channel becomes a real per-frame mask. Plain
    video uses the proven OpenCV path (blank mask); PyAV also serves as a
    fallback decoder when OpenCV is absent.
    """
    if _HAS_AV and _video_has_alpha(path):
        return _load_video_av(path, frame_load_cap, frame_skip, force_fps, vtrim_start, vtrim_end)
    if _HAS_CV2:
        return _load_video_cv2(path, frame_load_cap, frame_skip, force_fps, vtrim_start, vtrim_end)
    if _HAS_AV:
        return _load_video_av(path, frame_load_cap, frame_skip, force_fps, vtrim_start, vtrim_end)
    raise RuntimeError(
        "[PLS] MediaLoader: this file is a video and neither OpenCV nor PyAV is "
        "installed. Install one:  pip install opencv-python   (or  pip install av  "
        "for alpha-aware decoding) — or pick an image instead."
    )


def _video_has_alpha(path: str) -> bool:
    """Cheap probe (no full decode): does the video stream's pixel format carry
    an alpha channel? Returns False on any error or without PyAV."""
    if not _HAS_AV:
        return False
    try:
        container = av.open(path)
        try:
            vstream = container.streams.video[0]
            pix = vstream.codec_context.pix_fmt
        finally:
            container.close()
    except Exception:
        return False
    return pix_fmt_has_alpha(pix)


def _load_video_av(path: str, frame_load_cap: int, frame_skip: int, force_fps: float,
                   vtrim_start: float = 0.0, vtrim_end: float = 0.0):
    """PyAV decoder: yields RGBA frames so transparent video produces a real
    per-frame mask (opaque -> 0, transparent -> 1). For a non-alpha source
    ffmpeg fills alpha = 255, so the mask is blank — same as the OpenCV path."""
    skip = max(0, int(frame_skip))
    want = int(frame_load_cap) if int(frame_load_cap) > 0 else _HARD_FRAME_CAP
    want = min(want, _HARD_FRAME_CAP)

    frames = []
    native_fps = 0.0
    container = av.open(path)
    try:
        vstream = container.streams.video[0]
        try:
            native_fps = float(vstream.average_rate) if vstream.average_rate else 0.0
        except Exception:
            native_fps = 0.0
        idx = 0
        for frame in container.decode(vstream):
            if len(frames) >= want:
                break
            if idx >= skip:
                frames.append(frame.to_ndarray(format="rgba"))          # [H,W,4] uint8
            idx += 1
    finally:
        container.close()

    if not frames:
        raise RuntimeError(f"[PLS] MediaLoader: video '{os.path.basename(path)}' yielded no frames "
                           f"(frame_skip={skip} too large?).")

    stack = np.stack(frames, axis=0).astype(np.float32) / 255.0          # [N,H,W,4]
    stack = _slice_frames(stack, native_fps, vtrim_start, vtrim_end)     # v484: video trim window
    rgb, mask = rgb_and_mask_from_rgba(stack)                            # [N,H,W,3], [N,H,W]
    image_t = torch.from_numpy(np.ascontiguousarray(rgb))
    mask_t = torch.from_numpy(np.ascontiguousarray(mask))
    n = stack.shape[0]
    fps = float(force_fps) if float(force_fps) > 0 else native_fps
    return image_t, mask_t, n, fps


def _load_video_cv2(path: str, frame_load_cap: int, frame_skip: int, force_fps: float,
                    vtrim_start: float = 0.0, vtrim_end: float = 0.0):
    # Proven OpenCV path. cv2 drops alpha, so the mask is blank [N,H,W].
    # Force the FFMPEG backend: the default Windows backend (MSMF) can pad the
    # decoded width to an aligned size (e.g. 720 -> 728); FFMPEG returns the
    # canonical dimensions. Fall back to the default backend if FFMPEG can't open.
    cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"[PLS] MediaLoader: could not open video '{os.path.basename(path)}'.")
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)

    skip = max(0, int(frame_skip))
    want = int(frame_load_cap) if int(frame_load_cap) > 0 else _HARD_FRAME_CAP
    want = min(want, _HARD_FRAME_CAP)

    frames = []
    idx = 0
    while len(frames) < want:
        ok, frame = cap.read()
        if not ok:
            break
        if idx >= skip:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()

    if not frames:
        raise RuntimeError(f"[PLS] MediaLoader: video '{os.path.basename(path)}' yielded no frames "
                           f"(frame_skip={skip} too large?).")

    stack = np.stack(frames, axis=0).astype(np.float32) / 255.0          # [N,H,W,3]
    stack = _slice_frames(stack, native_fps, vtrim_start, vtrim_end)     # v484: video trim window
    image_t = torch.from_numpy(np.ascontiguousarray(stack))
    n, h, w = stack.shape[0], stack.shape[1], stack.shape[2]
    mask_t = torch.zeros((n, h, w), dtype=torch.float32)
    fps = float(force_fps) if float(force_fps) > 0 else native_fps
    return image_t, mask_t, n, fps


def _coerce_to_target(rgb: np.ndarray, mask: np.ndarray, target_wh, method: str):
    """Resize one frame's (rgb [H,W,3], mask [H,W]) to the target (W,H) per method.

    method ∈ {"resize to first", "pad to first", "center crop to first"}. The
    target is always the FIRST frame's size, so every frame becomes stackable.
    LANCZOS for downscale quality; pad fills black with an opaque (0) mask so the
    letterbox isn't read as transparent. No-op when the frame already matches."""
    tw, th = int(target_wh[0]), int(target_wh[1])
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if (w, h) == (tw, th):
        return rgb, mask
    rgb_im = Image.fromarray((rgb * 255.0 + 0.5).astype(np.uint8), "RGB")
    mask_im = Image.fromarray((mask * 255.0 + 0.5).astype(np.uint8), "L")
    if method == "resize to first":
        rgb_im = rgb_im.resize((tw, th), Image.LANCZOS)
        mask_im = mask_im.resize((tw, th), Image.LANCZOS)
    elif method == "center crop to first":
        scale = max(tw / w, th / h)                       # cover
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        rgb_im = rgb_im.resize((nw, nh), Image.LANCZOS)
        mask_im = mask_im.resize((nw, nh), Image.LANCZOS)
        left, top = (nw - tw) // 2, (nh - th) // 2
        rgb_im = rgb_im.crop((left, top, left + tw, top + th))
        mask_im = mask_im.crop((left, top, left + tw, top + th))
    else:                                                 # "pad to first" (letterbox/fit)
        scale = min(tw / w, th / h)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        rgb_im = rgb_im.resize((nw, nh), Image.LANCZOS)
        mask_im = mask_im.resize((nw, nh), Image.LANCZOS)
        rgb_canvas = Image.new("RGB", (tw, th), (0, 0, 0))
        mask_canvas = Image.new("L", (tw, th), 0)         # opaque pad
        ox, oy = (tw - nw) // 2, (th - nh) // 2
        rgb_canvas.paste(rgb_im, (ox, oy))
        mask_canvas.paste(mask_im, (ox, oy))
        rgb_im, mask_im = rgb_canvas, mask_canvas
    rgb2 = np.asarray(rgb_im).astype(np.float32) / 255.0
    mask2 = np.asarray(mask_im).astype(np.float32) / 255.0
    return rgb2, mask2


def _load_image_batch(folder, sort_mode, name_filter, frame_skip, every_nth,
                      frame_load_cap, resize_method, force_fps, selection=None):
    """Decode every image in `folder` (filtered + ordered) into one IMAGE batch
    [N,H,W,3] + MASK batch [N,H,W] — the still-image analogue of _load_video.

    Order/selection is the pure pipeline in ph_media_util (filter -> sort ->
    skip -> every-nth -> cap). v528: an explicit checked-files selection wins
    over the filter; skip / every-nth / cap still slice ON the set (select 500,
    every 2nd -> 250). Non-uniform sizes are either refused ('none (strict)',
    listing the offenders) or coerced to the first frame via resize_method.
    fps = force_fps if set, else 0.0 (a still sequence)."""
    if not _HAS_PIL:
        raise RuntimeError("[PLS] MediaLoader: Pillow is required for image-batch "
                           "loading (pip install pillow).")
    if selection:
        base = _selection_names(folder, selection, sort_mode, _is_batch_image)
        names = select_slice(base, frame_skip, every_nth, frame_load_cap,
                             _HARD_FRAME_CAP)
        if not names:
            raise _NothingToLoad(f"[PLS] MediaLoader: the checked selection left "
                               f"nothing to load in '{folder}' ({len(selection)} "
                               f"checked, none present after skip/step).")
    else:
        try:
            all_names = [e.name for e in os.scandir(folder)
                         if e.is_file() and _is_batch_image(e.name)]
        except OSError as ex:
            raise RuntimeError(f"[PLS] MediaLoader: cannot read "
                               f"folder '{folder}': {ex}") from ex
        if not all_names:
            raise _NothingToLoad(f"[PLS] MediaLoader: image-batch found no images in "
                               f"'{folder}'.")

        times = None
        if sort_mode in ("mtime (oldest first)", "created"):
            times = {}
            for n in all_names:
                try:
                    st = os.stat(os.path.join(folder, n))
                    times[n] = (st.st_mtime if sort_mode.startswith("mtime")
                                else getattr(st, "st_ctime", st.st_mtime))
                except OSError:
                    times[n] = 0.0

        names = select_frames(all_names, sort_mode, name_filter, frame_skip,
                              every_nth, frame_load_cap, _HARD_FRAME_CAP, times=times)
        if not names:
            raise _NothingToLoad(f"[PLS] MediaLoader: image-batch filter/skip/step left "
                               f"nothing to load (filter='{name_filter}', "
                               f"{len(all_names)} image(s) in folder).")

    rgbs, masks, dims = [], [], []
    for n in names:
        p = os.path.realpath(os.path.join(folder, n))
        rgb, mask = _decode_image_rgba(p)                # [H,W,3], [H,W]
        rgbs.append(rgb)
        masks.append(mask)
        dims.append((rgb.shape[1], rgb.shape[0]))        # (w, h)

    target, offenders = frames_target_and_offenders(dims)
    if offenders:
        if resize_method == "none (strict)":
            shown = ", ".join(f"{names[i]} ({dims[i][0]}x{dims[i][1]})"
                              for i in offenders[:8])
            more = "" if len(offenders) <= 8 else f" … (+{len(offenders) - 8} more)"
            raise RuntimeError(
                f"[PLS] MediaLoader: image-batch needs uniform frame size. Target "
                f"is {target[0]}x{target[1]} (first frame '{names[0]}'); "
                f"{len(offenders)} file(s) differ: {shown}{more}. Choose a "
                f"resize_method, tighten name_filter, or fix the folder.")
        for i in offenders:
            rgbs[i], masks[i] = _coerce_to_target(rgbs[i], masks[i], target,
                                                  resize_method)

    stack_rgb = np.stack(rgbs, axis=0).astype(np.float32)    # [N,H,W,3]
    stack_mask = np.stack(masks, axis=0).astype(np.float32)  # [N,H,W]
    image_t = torch.from_numpy(np.ascontiguousarray(stack_rgb))
    mask_t = torch.from_numpy(np.ascontiguousarray(stack_mask))
    n = int(stack_rgb.shape[0])
    fps = float(force_fps) if float(force_fps) > 0 else 0.0
    return image_t, mask_t, n, fps


def _make_video(mode, path, image_t, fps, audio=None):
    """Build the optional VIDEO output. Returns None when the comfy_api VIDEO
    type is unavailable, or there are no frames to build from.
      • "video" -> VideoFromFile(path): the original file, lossless, audio intact
        (get_components() recovers the file's own audio track downstream).
      • "batch" -> VideoFromComponents from the [N,H,W,3] batch, at force_fps
        (>0) else _BATCH_VIDEO_FPS. `audio` is None for a silent batch; a TRIMMED
        video (v485) passes its own sliced audio here so the re-encoded clip keeps sound.
      • "image" / anything else -> None (no-video fallback: no frames / VIDEO api off).
        A lone still is routed to "batch" by _video_for (v488), not here."""
    if not _HAS_VIDEO_API:
        return None
    if mode == "video":
        return VideoFromFile(path) if path else None
    if mode == "batch":
        if image_t is None or int(image_t.shape[0]) == 0:
            return None
        rate = Fraction(fps) if (fps and float(fps) > 0) else Fraction(_BATCH_VIDEO_FPS)
        return VideoFromComponents(VideoComponents(images=image_t, frame_rate=rate, audio=audio))
    return None


# ── Stufe B: audio output + muxed video ─────────────────────────────────────
# (Timing constants _STILL_VIDEO_FPS / _STILL_VIDEO_MAX_FRAMES / _DEFAULT_CLIP_FPS
#  now live in the video-timing block near the top of the module — gathered there
#  in D4/v483; values unchanged.)


def _f32_pcm(wav):
    """int PCM -> float32 in [-1, 1]; float passes through. Mirrors ComfyUI
    core's f32_pcm (comfy_extras.nodes_audio)."""
    if wav.dtype.is_floating_point:
        return wav
    if wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    if wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    return wav.float()


def _load_audio(path):
    """Decode an audio file to ComfyUI's AUDIO dict {waveform:[1,C,S], sample_rate}.
    Vendored from ComfyUI core's PyAV loader so the output matches core LoadAudio
    exactly — no torchaudio-backend gaps for m4a / aac / opus."""
    if not _HAS_AV:
        raise RuntimeError("PyAV (av) unavailable — cannot decode audio")
    with av.open(path) as af:
        if not af.streams.audio:
            raise ValueError("no audio stream in file")
        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n_channels = stream.channels
        frames = []
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)
        if not frames:
            raise ValueError("no audio frames decoded")
        wav = _f32_pcm(torch.cat(frames, dim=1))
    return {"waveform": wav.unsqueeze(0), "sample_rate": int(sr)}


def _audio_ref_from(media_ref):
    """The paired audio {folder, file, mtime, trimStart, trimEnd} carried in
    media_ref, or None. Present either as ref['audio'] (a visual paired with an
    audio) or as the whole ref when an audio-only selection was made
    (ref['kind'] == 'audio'). trimStart/trimEnd (seconds) are the v464 trim window."""
    def _f(v):
        try:
            return max(0.0, float(v))
        except (TypeError, ValueError):
            return 0.0
    try:
        ref = json.loads(media_ref) if media_ref else None
    except Exception:
        return None
    if not isinstance(ref, dict):
        return None
    a = ref.get("audio")
    if isinstance(a, dict) and a.get("folder") and a.get("file"):
        return {"folder": str(a["folder"]), "file": str(a["file"]), "mtime": a.get("mtime", 0),
                "trimStart": _f(a.get("trimStart")), "trimEnd": _f(a.get("trimEnd"))}
    if ref.get("kind") == "audio" and ref.get("folder") and ref.get("file"):
        return {"folder": str(ref["folder"]), "file": str(ref["file"]), "mtime": ref.get("mtime", 0),
                "trimStart": _f(ref.get("trimStart")), "trimEnd": _f(ref.get("trimEnd"))}
    return None


def _audio_stamp(media_ref):
    """(path, mtime, trimStart, trimEnd) of the paired audio for IS_CHANGED, so
    swapping it, editing it on disk, or moving a trim handle re-runs the node —
    including in image-batch mode, where the raw media_ref string is not otherwise
    part of the cache key."""
    a = _audio_ref_from(media_ref)
    if not a:
        return ("", 0, 0.0, 0.0)
    try:
        p = os.path.join(a["folder"], a["file"])
        return (a["folder"] + "|" + a["file"], os.path.getmtime(p) if os.path.isfile(p) else 0,
                a.get("trimStart", 0.0), a.get("trimEnd", 0.0))
    except Exception:
        return ("", 0, 0.0, 0.0)


def _decode_paired_audio(media_ref):
    """Parse + decode the paired audio to an AUDIO dict, or None (gracefully — a
    missing/undecodable audio must never break the graph). v464: the [start, end]
    trim window from the node UI is applied right after decode."""
    a = _audio_ref_from(media_ref)
    if not a:
        return None
    path = os.path.realpath(os.path.join(a["folder"], a["file"]))
    if not os.path.isfile(path):
        print(f"[PLS] MediaLoader: paired audio not found on disk: {path}")
        return None
    try:
        return _slice_audio(_load_audio(path), a.get("trimStart", 0.0), a.get("trimEnd", 0.0))
    except Exception as e:
        print(f"[PLS] MediaLoader: paired audio decode failed ({e}); AUDIO output is None.")
        return None


def _companion_audio_windowed(media_ref, trim_start, trim_end, seconds):
    """v487 (D1 Stufe 3): the paired companion audio sliced to the VIDEO-trim
    [start, total-end] window (NOT its own v464 audio-trim) and fitted to `seconds`. Used for
    the muxed `video_audio` output of a TRIMMED video: the video trim is the master clip lever
    for the muxed clip and the companion audio follows the SAME window, so `video_audio` is
    A/V-aligned like the plain `video` socket (v485/v486). The standalone `audio` output keeps
    the independent v464 trim. None (gracefully) when there is no companion or it cannot be
    decoded -- the caller then falls back to the v464-trimmed `aud`, never breaking the graph."""
    a = _audio_ref_from(media_ref)
    if not a:
        return None
    path = os.path.realpath(os.path.join(a["folder"], a["file"]))
    if not os.path.isfile(path):
        return None
    try:
        return _fit_audio(_slice_audio(_load_audio(path), trim_start, trim_end), seconds)
    except Exception:
        return None


def _make_muxed_video(image_t, fps, audio):
    """Stufe B: a NEW VIDEO = the visual's frames + the paired audio, built via the
    native comfy_api VIDEO type. Wire it to SaveVideo for an mp4 with a sound
    track. None when the VIDEO api / frames / audio are unavailable. A single
    still is held for (about) the audio's duration; a video/batch uses its real
    frames (no tiling)."""
    if not _HAS_VIDEO_API or image_t is None or audio is None:
        return None
    if int(image_t.shape[0]) == 0:
        return None
    rate = Fraction(fps) if (fps and float(fps) > 0) else Fraction(_BATCH_VIDEO_FPS)
    if int(image_t.shape[0]) == 1:
        try:
            sr = int(audio["sample_rate"]); samples = int(audio["waveform"].shape[-1])
            secs = (samples / sr) if sr else 0.0
            sfps = float(fps) if (fps and float(fps) > 0) else _STILL_VIDEO_FPS
            count = max(1, min(_STILL_VIDEO_MAX_FRAMES, int(round(secs * sfps))))
            image_t = image_t.expand(count, -1, -1, -1)
            rate = Fraction(sfps)
        except Exception:
            pass
    return VideoFromComponents(VideoComponents(images=image_t, frame_rate=rate, audio=audio))


# ── v464: audio trim is the single clip-length lever ────────────────────────
# The paired audio is trimmed to a [start, end] window in the node UI; its
# resulting duration drives the clip. A still becomes a real N-frame clip of that
# length; a real video keeps its own frames and the audio is capped (tail dropped)
# or silence-padded to the video. No paired audio -> pass-through (still = 1 frame).
def _video_for(kind, path, image_t, fps, own_audio=None, trimmed=False):
    """Pick the right VIDEO build for the (possibly trim-expanded) visual: a real
    source file stays VideoFromFile; any multi-frame tensor builds from components
    — so a still expanded to the audio's length becomes a real clip here.
    `own_audio` is the audio to embed into a components-built video: a trimmed video's own
    track sliced to the same window (v485), or a still's paired audio when the still was
    expanded to the audio's length (v486). VideoFromFile can't cut, so a trimmed video
    (trimmed=True) re-encodes from the sliced frames instead of the lossless passthrough.
    v488: a lone still (no audio, 1 frame) now also builds a silent 1-frame components video
    instead of None, so Save Video never receives None; an audio-only pick has no visual and
    stays None."""
    if kind == "video":
        if trimmed and image_t is not None and int(image_t.shape[0]) >= 1:
            return _make_video("batch", None, image_t, fps, audio=own_audio)
        return _make_video("video", path, image_t, fps)
    # v488: any non-audio visual with >=1 frame builds a real VIDEO from components, so a lone
    # still becomes a silent 1-frame video instead of None. Save Video (core nodes_video.py)
    # calls video.get_dimensions() with no None-check and crashes on None; emitting a 1-frame
    # video keeps the `video` socket always-a-VIDEO when frames exist. Audio-only has no visual
    # -> stays None. `own_audio` is None for a bare still, so the 1-frame clip is silent.
    if kind != "audio" and image_t is not None and int(image_t.shape[0]) >= 1:
        return _make_video("batch", None, image_t, fps, audio=own_audio)
    return _make_video("image", path, image_t, fps)


def _audio_duration(audio):
    """Length of an AUDIO dict in seconds (0.0 on anything unexpected)."""
    try:
        sr = int(audio["sample_rate"])
        return (int(audio["waveform"].shape[-1]) / sr) if sr > 0 else 0.0
    except Exception:
        return 0.0


def _slice_audio(audio, trim_start, trim_end):
    """Keep the [trim_start, total - trim_end] window (seconds), clamped so at least
    one sample survives. Zero trim (or anything odd) -> unchanged."""
    if audio is None:
        return audio
    ts = max(0.0, float(trim_start or 0.0))
    te = max(0.0, float(trim_end or 0.0))
    if ts <= 0 and te <= 0:
        return audio
    try:
        sr = int(audio["sample_rate"]); wav = audio["waveform"]
        total = int(wav.shape[-1])
        i0 = min(total - 1, int(round(ts * sr)))
        i1 = min(total, max(i0 + 1, total - int(round(te * sr))))
        if i0 <= 0 and i1 >= total:
            return audio
        return {"waveform": wav[..., i0:i1], "sample_rate": sr}
    except Exception:
        return audio


def _fit_audio(audio, seconds):
    """Trim (or silence-pad) an AUDIO dict to exactly `seconds`. None / <=0 -> raw."""
    if audio is None or not seconds or float(seconds) <= 0:
        return audio
    try:
        sr = int(audio["sample_rate"]); wav = audio["waveform"]
        target = max(1, int(round(float(seconds) * sr)))
        cur = int(wav.shape[-1])
        if cur >= target:
            wav = wav[..., :target]
        else:
            pad = torch.zeros(wav.shape[0], wav.shape[1], target - cur, dtype=wav.dtype)
            wav = torch.cat([wav, pad], dim=-1)
        return {"waveform": wav, "sample_rate": sr}
    except Exception:
        return audio


def _fit_clip(image_t, mask_t, n, fps, fps_timing, is_still, audio):
    """Reconcile the visual + the (already-trimmed) paired audio.
      • no audio          -> pass-through (still stays 1 frame, video unchanged).
      • still + audio      -> a real clip of the audio's length (still expands).
      • real video + audio -> keep the video's frames; cap/pad the audio to it.
    Returns (image_t, mask_t, n, out_fps, audio)."""
    a_dur = _audio_duration(audio)
    if audio is None or a_dur <= 0:
        return image_t, mask_t, n, float(fps), audio
    rate = float(fps_timing) if (fps_timing and float(fps_timing) > 0) else float(_DEFAULT_CLIP_FPS)
    if is_still:
        target = max(1, int(round(a_dur * rate)))
        capped = min(target, _HARD_FRAME_CAP)
        if capped < target:
            print(f"[PLS] MediaLoader: audio {a_dur:.2f}s @ {rate:g}fps = {target} frames "
                  f"exceeds the {_HARD_FRAME_CAP}-frame cap; clipped to {capped} "
                  f"({capped / rate:.2f}s). Trim the audio shorter to fit.")
        # a single frame -> a real clip of `capped` frames (expand = a view, cheap)
        image_t = image_t[:1].expand(capped, *([-1] * (image_t.dim() - 1)))
        mask_t = mask_t[:1].expand(capped, *([-1] * (mask_t.dim() - 1)))
        n = capped
        out_fps = rate
        audio = _fit_audio(audio, n / rate)        # exact match (absorbs rounding)
    else:
        # real video wins: keep its frames, fit the audio to the video's duration
        out_fps = float(fps) if fps > 0 else rate
        audio = _fit_audio(audio, (n / out_fps) if out_fps > 0 else a_dur)
    return image_t, mask_t, n, out_fps, audio


def _video_trim_of(media_ref):
    """(vtrimStart, vtrimEnd) in seconds from the visual pick's media_ref, or (0.0, 0.0).
    The v484 video-trim window (head, tail) the frontend serializes for the `video` output."""
    try:
        r = json.loads(media_ref) if media_ref else {}
        return float(r.get("vtrimStart") or 0.0), float(r.get("vtrimEnd") or 0.0)
    except Exception:
        return 0.0, 0.0


def _emit(media_ref, image_t, mask_t, n, fps, kind, path, file, eff_force_fps,
          batch_info=""):
    """Finalize one decoded clip into the 12-output contract — the single tail shared by all
    three load() paths. Callers pass the decoded tensors plus the per-path discriminators; every
    per-path difference is derived from `kind` here, so the W/H, audio-pairing, timing and mux
    rules live in exactly one place (was triplicated in load() before AUDIT B1).

    v528: `batch_info` (STRING, appended LAST) is the live batch counter for a wired
    text node — "Batch 0023 / 1334" per proc firing, "Batch 1334 frames (one pass)"
    for the image batch, "Single: <name>" otherwise. Appending keeps every existing
    workflow's link indices stable.

    kind: "image"/"video"/"audio" (single), the decoded "image"/"video" (batch-processing — never
    "audio", `_is_proc_media` excludes it), or the literal "batch" (image-batch; path=None, file="").
    Returns the contract tuple:
      (IMAGE, MASK, VIDEO, AUDIO, VIDEO(muxed), frame_count, fps, width, height, filename,
      batch_info, video_path).
    """
    # W/H come straight from the decoded tensor [N,H,W,3], captured BEFORE _fit_clip re-times.
    h, w = int(image_t.shape[1]), int(image_t.shape[2])
    aud = _decode_paired_audio(media_ref)
    is_still = (kind != "audio") and int(image_t.shape[0]) == 1
    # v478: a multi-frame image-batch with force_fps=0 has no native rate — fall back to
    # _BATCH_VIDEO_FPS (16, the silent-batch rate _make_video uses), NOT _DEFAULT_CLIP_FPS (24),
    # else pairing audio would re-time the SAME frames 16->24 (faster, shorter). A 1-frame batch
    # (a still) still expands at the clip default. Only "batch" needs this; single/proc videos
    # carry fps>0 and proc stills are 1 frame, so their fallback is unaffected.
    if kind == "batch":
        fps_timing = fps if fps > 0 else (eff_force_fps if eff_force_fps > 0
                                          else (_DEFAULT_CLIP_FPS if is_still else _BATCH_VIDEO_FPS))
    else:
        fps_timing = fps if fps > 0 else (eff_force_fps if eff_force_fps > 0 else _DEFAULT_CLIP_FPS)
    # v464: the trim window is already applied in _decode_paired_audio, so an audio-only pick
    # emits the trimmed segment as-is and skips _fit_clip; everything else fits (a still expands
    # to the audio's length, a real video keeps its frames and the audio is capped/padded to it).
    if kind != "audio":
        image_t, mask_t, n, fps, aud = _fit_clip(image_t, mask_t, n, fps, fps_timing, is_still, aud)
    # v485: a video trim means the `video` output must be re-encoded from the sliced frames
    # (VideoFromFile can't cut) — carry the file's OWN audio, sliced to the SAME window and
    # fitted to the trimmed duration, so `video` stays a self-contained, aligned clip. The
    # separate paired-audio trim still drives the standalone `audio` output; v487 (below) aligns
    # a TRIMMED `video_audio` to the video window instead of that separate audio trim.
    _vts, _vte = _video_trim_of(media_ref)
    _vtrimmed = (kind == "video") and (_vts > 0.0 or _vte > 0.0)
    _own_audio = None
    if _vtrimmed and path:
        try:
            _own_audio = _fit_audio(_slice_audio(_load_audio(path), _vts, _vte),
                                    (n / fps) if fps > 0 else 0.0)
        except Exception:
            _own_audio = None
    elif is_still and aud is not None:
        # v486: a still only becomes a video because of the paired audio (a lone still -> no
        # VIDEO). _fit_clip already expanded it to the audio's length and fitted `aud`, so the
        # plain `video` output carries that audio too — mirrors the trimmed-video own-audio path.
        _own_audio = aud
    video = _video_for(kind, path, image_t, fps, _own_audio, _vtrimmed)
    # v487 (D1 Stufe 3): pick the audio for the muxed `video_audio`. A TRIMMED video aligns the
    # muxed clip to the video-trim window (the paired companion audio sliced to the SAME
    # [start, total-end] window the frames use, fitted to the trimmed duration), falling back to
    # the v464-trimmed `aud` if that companion decode fails; every other case (still, untrimmed
    # video, image-batch) muxes the v464-trimmed `aud` exactly as before. The video trim is the
    # master clip lever for `video_audio`; the audio follows it. The standalone `audio` output
    # keeps the independent v464 trim regardless.
    _mux_audio = ((_companion_audio_windowed(media_ref, _vts, _vte, (n / fps) if fps > 0 else 0.0) or aud)
                  if (_vtrimmed and aud is not None) else aud)
    # the muxed VIDEO needs real visual frames -> skip it for an audio-only pick (the placeholder
    # frame isn't a visual); the AUDIO output still carries it.
    # v490: a bare still has no paired audio, so _make_muxed_video returns None -> core's Save Video
    # crashed on None.get_dimensions(). Fall back to the silent `video` clip already built above,
    # mirroring the v488 fix on the `video` socket. `or video` binds tighter than the conditional's
    # `else`, so this parses as `None if audio else (mux or video)` -- audio-only stays None via the
    # guard, and Still+Audio (v486) / video+companion (v487) yield a non-None mux so `or video` never
    # fires there (no regression). Appended (not wrapped) so it does not disturb the v459/v487/v488
    # source guards that pin the muxed-video line.
    vmux = None if kind == "audio" else _make_muxed_video(image_t, fps, _mux_audio) or video
    return (image_t, mask_t, video, aud, vmux, int(n), float(fps), w, h, file,
            str(batch_info or ""),
            # v639 (APPENDED LAST): the file's FULL path -- the Batch Pipeline
            # Source consumes it as video_path (convert the widget to an input
            # and wire it). Empty in modes without a single source file
            # (image-batch / sequence), exactly like `filename`.
            os.path.realpath(path) if path else "")


class ULSMediaLoader:
    """Load media from a pinned folder. Two modes (load_mode):
      • single      — decode the clicked file (image or video) → [N,H,W,3].
      • image batch — decode EVERY image in the folder as one ordered batch.
    The frontend supplies the pin/selection via `media_ref` (JSON
    {folder, file, kind}); batch mode uses only `folder`."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Set by web/js/ph_media_loader.js when the user clicks a thumbnail.
                "media_ref": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Managed by the node UI: the pinned folder + chosen file. "
                               "Pick a folder and a thumbnail above.",
                }),
                # Visible knobs (apply to both the video and the image-batch
                # path). Kept in `required` so the JS-managed hidden widgets
                # (media_ref, batch_config) hide reliably; order is unchanged, so
                # older graphs keep their positional widget values.
                "frame_load_cap": ("INT", {
                    "default": 0, "min": 0, "max": _HARD_FRAME_CAP, "step": 1,
                    "tooltip": "Video only: max frames to load (0 = all, capped for safety).",
                }),
                "frame_skip": ("INT", {
                    "default": 0, "min": 0, "max": 100000, "step": 1,
                    "tooltip": "Video only: skip this many frames at the start.",
                }),
                "force_fps": ("FLOAT", {
                    "default": 24.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Output fps (default 24). 0 = the file's native fps. With "
                               "keep_input_fps on, the native fps is used regardless. Sets the "
                               "play rate, and the frame count when a still is expanded to the "
                               "trimmed audio's length.",
                }),
                # Image-batch config (JSON) — written by the "Batch…" panel in
                # web/js/ph_media_loader.js; collapses the former per-knob widgets
                # into one serialized value so the node stays decluttered.
                "batch_config": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Managed by the node's Batch panel: a small JSON "
                               "{enabled, source, sort_mode, name_filter, every_nth, "
                               "resize_method}. Empty / enabled=false = single mode.",
                }),
                # Batch-processing config (JSON) — written by the "Batch
                # Processing" panel. APPENDED last in `required` so existing
                # graphs keep their positional widget values (v423-lock safe);
                # the frontend hides it like media_ref / batch_config.
                "proc_config": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Managed by the node's Batch Processing panel: a "
                               "small JSON {enabled, source, sort_mode, name_filter, "
                               "wrap, start_at, reset_seq}. enabled=true streams the "
                               "folder one file per run (iterative mode).",
                }),
                # v464: keep_input_fps — the fps lever for expanding a still to the
                # trimmed audio's length. APPENDED last in `required` so existing
                # graphs keep their positional widget values (v423-lock safe — a
                # required widget, no `optional` block); the frontend renders it
                # right after force_fps (the hidden JSON widgets take no space).
                "keep_input_fps": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "On: ignore force_fps and use the source's native fps (a "
                               "still/sequence falls back to 24). Off: force_fps applies.",
                }),
                # v509: the empty-state lever. 'placeholder' (default) emits a
                # built-in dummy frame when there is simply NOTHING to load;
                # 'error' restores the strict pre-v509 behaviour.
                "on_empty": (["placeholder", "error"], {
                    "default": "placeholder",
                    "tooltip": "When there is NOTHING to load (no selection, empty or "
                               "empty-filtered folder, vanished file): emit a built-in "
                               "placeholder frame instead of a hard error. Misconfiguration "
                               "and decode failures always error."}),
            },
            # UNIQUE_ID lets load()/IS_CHANGED key the per-node batch-processing
            # cursor. ComfyUI injects it at runtime, so older graphs need nothing
            # (backward-compatible). Only UNIQUE_ID goes here — ComfyUI silently
            # discards custom STRING widgets in 'hidden', so proc_config stays in
            # 'required' (same rule as ph_viewport3d / uls_stack_node).
            "hidden": {
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "VIDEO", "AUDIO", "VIDEO", "INT", "FLOAT", "INT", "INT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("image", "mask", "video", "audio", "video_audio", "frame_count", "fps",
                    "width", "height", "filename", "batch_info", "video_path")
    FUNCTION = "load"
    CATEGORY = "Polyhedron/IO"

    @classmethod
    def IS_CHANGED(cls, media_ref="", frame_load_cap=0, frame_skip=0, force_fps=0.0,
                   batch_config="", proc_config="", node_id=None,
                   keep_input_fps=False, on_empty="placeholder"):
        # Batch-processing (iterative) mode must re-run on EVERY queue so the
        # cursor advances to the next file — independent of any widget value (a
        # frontend control_after_generate increment is NOT reliable on current
        # ComfyUI). A monotonic timestamp guarantees the node is always dirty.
        # v528 precedence mirrors _load_strict: an explicit unified mode wins;
        # a mode-less batch_config falls back to the legacy proc_config flag.
        cfg = _batch_cfg(batch_config)
        proc_on = (cfg["mode"] == "proc" and cfg["enabled"]) if cfg["mode"] \
            else _proc_cfg(proc_config)["enabled"]
        if proc_on:
            return time.time()
        # Stufe B: the paired audio rides in media_ref. Stamp it (path + mtime) so
        # swapping or editing the audio re-runs the node in every non-proc mode.
        astamp = _audio_stamp(media_ref)
        # Single mode keys on the picked file's mtime; image-batch keys on the
        # (name, mtime, size) of the exact set of files it would load, so adding,
        # removing or editing any frame — or editing the checked selection —
        # re-reads the source folder.
        frames_on = (cfg["mode"] == "frames" and cfg["enabled"]) if cfg["mode"] \
            else cfg["enabled"]
        if frames_on and cfg["source"] and os.path.isdir(cfg["source"]):
            folder = cfg["source"]
            try:
                if cfg["selection"]:
                    base = _selection_names(folder, cfg["selection"],
                                            cfg["sort_mode"], _is_batch_image)
                    chosen = select_slice(base, frame_skip, cfg["every_nth"],
                                          frame_load_cap, _HARD_FRAME_CAP)
                else:
                    names = [e.name for e in os.scandir(folder)
                             if e.is_file() and _is_batch_image(e.name)]
                    times = {}
                    if cfg["sort_mode"] in ("mtime (oldest first)", "created"):
                        for n in names:
                            try:
                                st = os.stat(os.path.join(folder, n))
                                times[n] = (st.st_mtime if cfg["sort_mode"].startswith("mtime")
                                            else getattr(st, "st_ctime", st.st_mtime))
                            except OSError:
                                times[n] = 0.0
                    chosen = select_frames(names, cfg["sort_mode"], cfg["name_filter"],
                                           frame_skip, cfg["every_nth"], frame_load_cap,
                                           _HARD_FRAME_CAP, times=times)
                sig = []
                for n in chosen:
                    try:
                        st = os.stat(os.path.join(folder, n))
                        sig.append((n, st.st_mtime, st.st_size))
                    except OSError:
                        sig.append((n, 0, 0))
                return ("image batch", folder, tuple(sig), cfg["sort_mode"],
                        cfg["name_filter"], cfg["every_nth"], cfg["resize_method"],
                        tuple(cfg["selection"] or ()),
                        frame_skip, frame_load_cap, force_fps, keep_input_fps, astamp)
            except Exception:
                return ("image batch", folder, batch_config, frame_skip,
                        frame_load_cap, force_fps, keep_input_fps, astamp)
        # single mode (unchanged behaviour)
        try:
            ref = json.loads(media_ref) if media_ref else {}
        except Exception:
            ref = {}
        folder = ref.get("folder", "")
        try:
            p = os.path.join(folder, ref.get("file", ""))
            stamp = (os.path.getmtime(p)
                     if (folder and ref.get("file") and os.path.isfile(p)) else 0)
        except Exception:
            stamp = 0
        return (media_ref, stamp, astamp, frame_load_cap, frame_skip, force_fps,
                keep_input_fps, batch_config)

    def load(self, media_ref="", frame_load_cap=0, frame_skip=0, force_fps=0.0,
             batch_config="", proc_config="", node_id=None,
             keep_input_fps=False, on_empty="placeholder"):
        """v509: the on_empty gate around the strict loader. _NothingToLoad (an
        EMPTY state -- no selection, empty or empty-filtered folder, vanished
        file) becomes the built-in placeholder frame; 'error' restores the old
        strict behaviour. Real defects always raise (see _NothingToLoad)."""
        try:
            return self._load_strict(media_ref, frame_load_cap, frame_skip,
                                     force_fps, batch_config, proc_config,
                                     node_id, keep_input_fps)
        except _NothingToLoad as e:
            if str(on_empty) != "placeholder":
                raise
            print(f"{e} -> emitting the built-in placeholder (on_empty=placeholder).")
            image_t, mask_t, n, fps = _placeholder_media()
            return _emit("", image_t, mask_t, n, fps, "image", None, "", 0.0,
                         batch_info="Single: (placeholder)")

    def _load_strict(self, media_ref="", frame_load_cap=0, frame_skip=0, force_fps=0.0,
                     batch_config="", proc_config="", node_id=None,
                     keep_input_fps=False):
        ucfg = _batch_cfg(batch_config)          # v528: the unified config
        pcfg_legacy = _proc_cfg(proc_config)     # legacy workflows (pre-unified)
        # keep_input_fps overrides force_fps to the source's native rate; the decode
        # helpers then return native fps (force_fps=0 already meant native).
        eff_force_fps = 0.0 if keep_input_fps else float(force_fps)

        # ── v528 mode precedence ────────────────────────────────────────────
        # A batch_config carrying an explicit `mode` is the unified world and wins.
        # Without one (old workflow), the legacy rule holds: proc_config.enabled
        # -> proc; batch_config.enabled -> frames. Old graphs run untouched.
        if ucfg["mode"]:
            proc_on = ucfg["mode"] == "proc" and ucfg["enabled"]
            frames_on = ucfg["mode"] == "frames" and ucfg["enabled"]
            pcfg = ucfg                          # proc fields live in the unified cfg
            selection = ucfg["selection"]
        else:
            proc_on = pcfg_legacy["enabled"]
            frames_on = (not proc_on) and ucfg["enabled"]
            pcfg = pcfg_legacy
            selection = None                     # legacy = rule pipeline only

        # v536 DIAG: Bug B (Frames-Batch returns N=1 right after a restart) is still
        # OPEN. The v529 serialize "fix" was removed after measurement disproved the
        # transport hypothesis: a populated batch_config already transports (this line
        # confirmed 11 frames on an explicit selection), and the selection=None rule
        # pipeline loads the whole folder. This line stays to capture the TRUE failing
        # case -- a cold restart with selection=None and no tile click -- so we can see
        # exactly what reaches the backend (empty vs enabled/mode present). Drop once
        # the real cause is found.
        print(f"[PLS v536 DIAG] batch_config={batch_config!r} | frames_on={frames_on} "
              f"proc_on={proc_on} sel={ucfg['selection'] and len(ucfg['selection'])}")

        # ── batch-processing: stream ONE file per run (iterative mode) ─────────
        # The selection/folder media (stills + videos) is walked one file at a
        # time by a per-node cursor; Auto-Queue sweeps it. Each file decodes
        # exactly like single mode (a video yields its frames), plus `filename`
        # for downstream save naming and `batch_info` as the live counter.
        if proc_on:
            folder = pcfg["source"]
            if not folder:
                raise ValueError("[PLS] MediaLoader: Batch Processing is on but no source "
                                 "folder is set. Open the Batch panel and choose one.")
            folder_rp = os.path.realpath(folder)
            if not os.path.isdir(folder_rp):
                raise FileNotFoundError(f"[PLS] MediaLoader: Batch Processing source folder "
                                        f"not found: {folder_rp}")
            files = _proc_resolve_files(folder_rp, pcfg["sort_mode"],
                                        pcfg["name_filter"], selection=selection)
            total = len(files)
            if total == 0:
                raise _NothingToLoad(f"[PLS] MediaLoader: Batch Processing found no matching media "
                                     f"in {folder_rp} (filter='{pcfg['name_filter']}'"
                                     f"{', ' + str(len(selection)) + ' checked' if selection else ''})")
            idx, is_last, loop = _proc_pick(node_id, total, pcfg)
            file = files[idx]
            path = os.path.realpath(os.path.join(folder_rp, file))
            if not os.path.isfile(path):
                raise FileNotFoundError(f"[PLS] MediaLoader: batch-processing file vanished: {path}")
            kind = _kind_for(file)
            if kind == "video":
                image_t, mask_t, n, fps = _load_video(path, frame_load_cap, frame_skip, eff_force_fps)
            else:
                image_t, mask_t, n, fps = _load_image(path)
            width = len(str(total))
            binfo = f"Batch {idx + 1:0{width}d} / {total}"
            if pcfg["wrap"] and loop > 1:
                binfo += f" (loop {loop})"
            print(f"[PLS] MediaLoader: batch-processing '{folder_rp}' -> [{idx + 1}/{total}] "
                  f"'{file}' ({kind}, {image_t.shape[2]}x{image_t.shape[1]})"
                  f"{' — last' if is_last else ''}")
            return _emit(media_ref, image_t, mask_t, n, fps, kind, path, file,
                         eff_force_fps, batch_info=binfo)

        cfg = ucfg

        # ── image-batch: whole selection/folder -> one ordered [N,H,W,3] batch ─
        if frames_on:
            folder = cfg["source"]
            if not folder:
                raise ValueError("[PLS] MediaLoader: batch mode is on but no source folder "
                                 "is set. Open the Batch panel and choose a source folder.")
            folder_rp = os.path.realpath(folder)
            if not os.path.isdir(folder_rp):
                raise FileNotFoundError(f"[PLS] MediaLoader: batch source folder not found: "
                                        f"{folder_rp}")
            image_t, mask_t, n, fps = _load_image_batch(
                folder_rp, cfg["sort_mode"], cfg["name_filter"], frame_skip,
                cfg["every_nth"], frame_load_cap, cfg["resize_method"], eff_force_fps,
                selection=selection)
            print(f"[PLS] MediaLoader: image-batch '{folder_rp}' -> {n} frame(s) "
                  f"{image_t.shape[2]}x{image_t.shape[1]} (sort={cfg['sort_mode']}, "
                  f"filter='{cfg['name_filter']}'"
                  f"{', ' + str(len(selection)) + ' checked' if selection else ''}, "
                  f"resize={cfg['resize_method']})")
            # batch (sequence) mode emits the whole folder as one clip — no single source
            # file (filename empty); the v478 multi-frame-batch fps fallback lives in _emit.
            return _emit(media_ref, image_t, mask_t, n, fps, "batch", None, "",
                         eff_force_fps, batch_info=f"Batch {n} frames (one pass)")

        # ── single: the clicked file (image or video) — unchanged ─────────────
        if not media_ref:
            raise _NothingToLoad("[PLS] MediaLoader: nothing selected (empty input folder, or no "
                                 "thumbnail clicked yet)")
        try:
            ref = json.loads(media_ref)
        except Exception:
            raise ValueError("[PLS] MediaLoader: malformed selection "
                             "(media_ref is not valid JSON).") from None
        folder = str(ref.get("folder", ""))
        file = str(ref.get("file", ""))
        if not folder or not file:
            raise _NothingToLoad("[PLS] MediaLoader: selection carries no folder/file (cleared "
                                 "or empty selection)")
        path = os.path.realpath(os.path.join(folder, file))
        if not os.path.isfile(path):
            raise _NothingToLoad(f"[PLS] MediaLoader: the selected file vanished from disk: {path}")

        kind = str(ref.get("kind", "")) or _kind_for(file)
        if kind == "video":
            image_t, mask_t, n, fps = _load_video(
                path, frame_load_cap, frame_skip, eff_force_fps,
                float(ref.get("vtrimStart") or 0.0), float(ref.get("vtrimEnd") or 0.0))
            print(f"[PLS] MediaLoader: video '{file}' -> {n} frame(s) "
                  f"{image_t.shape[2]}x{image_t.shape[1]} @ {fps:.3f} fps")
        elif kind == "audio":
            # v457 (Stufe A): audio is browser-previewable only — no graph output
            # yet (Stufe B adds an AUDIO socket). Emit a safe placeholder frame so
            # the graph never crashes when an audio file is the active selection.
            image_t, mask_t, n, fps = _audio_placeholder()
            print(f"[PLS] MediaLoader: audio '{file}' selected — browser preview only; "
                  f"AUDIO output arrives in a later version (placeholder frame emitted).")
        else:
            image_t, mask_t, n, fps = _load_image(path)
            print(f"[PLS] MediaLoader: image '{file}' -> {image_t.shape[2]}x{image_t.shape[1]}")

        # single mode emits the one clicked file; W/H, audio pairing, the audio-only
        # _fit_clip skip and the mux rules all live in _emit (the shared finalizer).
        return _emit(media_ref, image_t, mask_t, n, fps, kind, path, file,
                     eff_force_fps, batch_info=f"Single: {file}")
