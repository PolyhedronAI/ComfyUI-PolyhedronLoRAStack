# ph_save_util.py -- torch-free helpers for the Save node (ULSSave).
#
# Kept free of torch / av / PIL on purpose so the whole DECISION layer of the
# Save node -- which backend a preset uses, what container/codec/pix_fmt/audio
# codec applies, how a crf maps per codec, the ping-pong index expansion, the
# alpha decision, and the RGBA->(RGB,mask) split -- can be unit-tested in a bare
# sandbox (GATE-3). ph_save.py imports these; the heavy IO (Pillow stills, the
# native comfy_api VIDEO save, the PyAV encode) stays in the node and is
# live-tested. Mirrors the ph_media_util / ph_sprite_util split exactly.

import numpy as np


# ---------------------------------------------------------------------------
# Preset ladder. Two tiers on purpose: DELIVERY (small, universal) vs MASTER
# (edit / archive quality). `backend` decides the write path:
#   "native"  -> ComfyUI's own comfy_api VIDEO.save_to (MP4/H264 only, but its
#                code path is battle-tested and muxes the component audio for us)
#   "pillow"  -> Pillow writes an animated GIF / WebP from the frame list
#                (no external ffmpeg, no PyAV -- the most robust loop path)
#   "pyav"    -> our own PyAV encode (the pro codecs native can't do). For 10-bit
#                pix_fmts the node feeds a 16-bit intermediate (rgb48le) so the
#                diffusion float precision reaches the encoder -> less banding.
#                Intra presets set every frame a keyframe (particle/grain safe).
# `ten_bit`/`alpha`/`intra` are read by the node; `opts` are extra per-codec
# encoder options; `acodec` is the audio codec muxed alongside.
PRESETS = {
    "H.264 MP4 (delivery)": {
        "backend": "native", "ext": "mp4", "vcodec": "h264",
        "pix_fmt": "yuv420p", "acodec": "aac",
        "alpha": False, "intra": False, "ten_bit": False, "opts": {},
    },
    "H.265 MP4 (delivery, grain)": {
        "backend": "pyav", "ext": "mp4", "vcodec": "hevc",
        "pix_fmt": "yuv420p10le", "acodec": "aac",
        "alpha": False, "intra": False, "ten_bit": True,
        # grain-tuned so WAN particles/gradients aren't smeared by HEVC's
        # aggressive default in-loop filters (the SaveVideoHQ finding).
        "opts": {"tune": "grain", "x265-params": "deblock=-3,-3:no-sao"},
    },
    "ProRes 422 HQ (master)": {
        "backend": "pyav", "ext": "mov", "vcodec": "prores_ks",
        "pix_fmt": "yuv422p10le", "acodec": "pcm_s16le",
        "alpha": False, "intra": True, "ten_bit": True,
        "opts": {"profile": "3"},          # prores_ks: 3 = HQ
    },
    "ProRes 4444 (master + alpha)": {
        "backend": "pyav", "ext": "mov", "vcodec": "prores_ks",
        "pix_fmt": "yuva444p10le", "acodec": "pcm_s16le",
        "alpha": True, "intra": True, "ten_bit": True,
        "opts": {"profile": "4"},          # prores_ks: 4 = 4444
    },
    "FFV1 (lossless archive)": {
        "backend": "pyav", "ext": "mkv", "vcodec": "ffv1",
        "pix_fmt": "yuv420p", "acodec": "flac",
        "alpha": False, "intra": True, "ten_bit": False, "opts": {},
    },
    "WebM VP9": {
        "backend": "pyav", "ext": "webm", "vcodec": "vp9",
        "pix_fmt": "yuv420p", "acodec": "libopus",
        "alpha": False, "intra": False, "ten_bit": False, "opts": {},
    },
    "Animated WebP": {
        "backend": "pillow", "ext": "webp", "acodec": None,
        "alpha": True, "intra": False, "ten_bit": False, "opts": {},
    },
    "GIF": {
        "backend": "pillow", "ext": "gif", "acodec": None,
        "alpha": True, "intra": False, "ten_bit": False, "opts": {},
    },
}

PRESET_NAMES = list(PRESETS.keys())
DEFAULT_PRESET = "H.264 MP4 (delivery)"

IMAGE_FORMATS = ("png", "webp", "jpg")
MEDIA_KINDS = ("auto", "image", "video")


def preset(name):
    """Return the preset dict for `name`, defaulting to the H.264 delivery preset
    on an unknown name (so a stale workflow value can never hard-crash a save)."""
    return PRESETS.get(name, PRESETS[DEFAULT_PRESET])


def resolve_media_kind(kind, has_image, has_video, has_audio, n_frames=1,
                       has_mask=False):
    """Decide what to actually write. Principle of least surprise, v551 edition:
    MANY frames ARE a clip. The old rule ("an IMAGE batch saves stills unless
    you opt in") made a 65-frame Wan run land as 65 PNGs; now the frame count
    itself decides. A deliberate still batch stays one explicit click away:
    media_kind='image' always forces stills (the FBX sprite-sheet case).

    Explicit "image" / "video" win (validated against what is wired).
    "auto":
      * a wired VIDEO                              -> "video"
      * an IMAGE plus audio (a muxed clip)         -> "video"
      * an IMAGE batch with n_frames > 1 (v551)    -> "video"  (fps = frame_rate widget)
      * a single IMAGE frame                       -> "image"  (SaveImage-like)
      * only audio (Media Loader AUDIO, no visual) -> "audio"
      * ONLY a mask (v653)                         -> "mask"  (grayscale stills)
      * nothing                                    -> raise
    n_frames defaults to 1, so every pre-v551 call keeps its exact behaviour.
    Returns one of "image" | "video" | "audio".
    """
    if kind == "image":
        if not has_image:
            raise ValueError("Save: media_kind='image' but no IMAGE is wired.")
        return "image"
    if kind == "video":
        if not (has_video or has_image):
            raise ValueError("Save: media_kind='video' but neither VIDEO nor IMAGE is wired.")
        return "video"
    # auto
    if has_video:
        return "video"
    if has_image and has_audio:
        return "video"
    if has_image and int(n_frames) > 1:   # v551: many frames ARE a clip
        return "video"
    if has_image:
        return "image"
    if has_audio:
        return "audio"
    # v653 (declared): ONLY a mask wired used to be this raise -- now it saves
    # the mask itself as grayscale stills (look at a mask without a converter).
    if has_mask:
        return "mask"
    raise ValueError("Save: nothing wired -- connect image, video or audio.")


def image_ext(fmt):
    """Normalise an image-format choice to a file extension."""
    f = (fmt or "png").lower()
    return f if f in IMAGE_FORMATS else "png"


def image_supports_alpha(fmt):
    """png / webp carry alpha; jpg does not."""
    return image_ext(fmt) in ("png", "webp")


def still_uses_alpha(has_mask, channels, fmt):
    """v648: a still is written RGBA when the format can hold alpha AND
    either a MASK is wired (declared 1-mask transparency convention) or --
    new -- the IMAGE itself carries a 4th channel (RGBA sources like the
    Background Remove node). A wired MASK keeps priority over the image's
    own alpha; jpg always flattens."""
    if not image_supports_alpha(fmt):
        return False
    return bool(has_mask) or int(channels) >= 4


def compose_still(frame_float, mask_float_or_none, fmt):
    """One still frame -> (uint8 HWC array, 'RGBA'|'RGB'). frame_float is
    [H,W,3] or [H,W,4] in [0,1]; the three cases of still_uses_alpha are
    realised here: wired mask -> alpha = 1-mask (LoadImage convention),
    no mask but 4 channels -> alpha = the image's own 4th channel,
    otherwise (or jpg) -> RGB."""
    frame = np.clip(np.asarray(frame_float, dtype=np.float32), 0.0, 1.0)
    ch = frame.shape[-1]
    if not still_uses_alpha(mask_float_or_none is not None, ch, fmt):
        return ((frame[..., :3] * 255.0 + 0.5).astype(np.uint8), "RGB")
    if mask_float_or_none is not None:
        rgba = rgb_and_mask_to_rgba_uint8(frame[None, ..., :3],
                                          np.asarray(mask_float_or_none)[None])[0]
        return (rgba, "RGBA")
    return ((frame[..., :4] * 255.0 + 0.5).astype(np.uint8), "RGBA")


def wants_alpha(preset_name, has_mask, image_fmt=None, is_image=False):
    """Alpha is written only when a MASK is wired AND the target can hold it:
    for stills that means png/webp; for video an alpha-capable preset."""
    if not has_mask:
        return False
    if is_image:
        return image_supports_alpha(image_fmt)
    return bool(preset(preset_name).get("alpha"))


def is_ten_bit(pix_fmt):
    """True for a 10/12/16-bit ffmpeg pixel format name (needs the 16-bit
    intermediate on the encode side)."""
    p = (pix_fmt or "").lower()
    return ("10le" in p) or ("12le" in p) or ("16le" in p) or p.endswith("10") or p.endswith("12")


def crf_option(vcodec, quality):
    """Map the single `quality` widget to the right per-codec rate-control option
    dict. Codecs that are quality-fixed (prores via profile, ffv1 lossless) get
    an empty dict -- `quality` simply does not apply to them."""
    q = int(quality)
    q = 0 if q < 0 else (100 if q > 100 else q)
    vc = (vcodec or "").lower()
    if vc in ("h264", "libx264", "hevc", "libx265", "h265"):
        return {"crf": str(q)}                    # x264/x265 crf 0..51-ish; UI clamps 0..100
    if vc in ("vp9", "libvpx-vp9"):
        return {"crf": str(q), "b:v": "0"}        # vp9 constant-quality needs b:v 0
    if vc in ("av1", "libsvtav1", "svt-av1"):
        return {"crf": str(q)}
    return {}                                     # prores_ks, ffv1: not crf-driven


def pingpong_indices(n):
    """Frame indices for a ping-pong loop: forward then back WITHOUT repeating
    the two endpoints (0..n-1 then n-2..1). n<=1 -> unchanged. Matches VHS
    (np.concatenate((imgs, imgs[-2:0:-1]))) but as pure index math so it is
    testable without any array."""
    n = int(n)
    if n <= 1:
        return list(range(max(0, n)))
    return list(range(n)) + list(range(n - 2, 0, -1))


def apply_pingpong(frames, enabled):
    """Return `frames` (a numpy array [N,...]) with the ping-pong tail appended
    when enabled. Pure indexing -> no copy of the pixel logic in the node."""
    if not enabled or frames is None or int(frames.shape[0]) <= 1:
        return frames
    idx = pingpong_indices(int(frames.shape[0]))
    return frames[np.asarray(idx, dtype=np.int64)]


def rgb_and_mask_to_rgba_uint8(rgb_float, mask_float):
    """Compose an RGB float batch [N,H,W,3] in [0,1] and an optional MASK
    [N,H,W] (or [H,W]) into an RGBA uint8 batch [N,H,W,4], following the
    ComfyUI/LoadImage convention that a MASK stores transparency as opaque->0 /
    transparent->1 (so the alpha channel is 1-mask). mask=None -> fully opaque.
    Broadcasts a single mask over the batch; clamps to [0,1]."""
    rgb = np.clip(np.asarray(rgb_float, dtype=np.float32), 0.0, 1.0)
    if rgb.ndim != 4 or rgb.shape[-1] < 3:
        raise ValueError("rgb_and_mask_to_rgba_uint8 expects RGB [N,H,W,>=3].")
    rgb = rgb[..., :3]
    n, h, w = rgb.shape[0], rgb.shape[1], rgb.shape[2]
    if mask_float is None:
        alpha = np.ones((n, h, w), dtype=np.float32)
    else:
        m = np.clip(np.asarray(mask_float, dtype=np.float32), 0.0, 1.0)
        if m.ndim == 2:
            m = m[None, ...]
        if m.shape[0] == 1 and n > 1:
            m = np.repeat(m, n, axis=0)
        # opaque->0 in a MASK means alpha 1: alpha = 1 - mask
        alpha = 1.0 - m[:, :h, :w]
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    return (rgba * 255.0 + 0.5).astype(np.uint8)


import datetime as _dt
import re as _re2

# Windows-illegal filename characters. `folder_paths.get_save_image_path` does NOT
# expand %date:...% on every ComfyUI build, so a literal token (with its ':')
# reached os.makedirs and raised WinError 267. We therefore expand the tokens
# ourselves and strip any illegal character before the path is built.
_ILLEGAL_PATH_CHARS = set('<>:"|?*')

# %date:FORMAT% specifier -> strftime, longest-first so "yyyy" is consumed before
# "yy" and "MM"/"mm" (month vs minute) don't collide.
_DATE_MAP = (("yyyy", "%Y"), ("yy", "%y"), ("MM", "%m"), ("dd", "%d"),
             ("hh", "%H"), ("mm", "%M"), ("ss", "%S"))


def expand_filename_tokens(prefix, width=0, height=0, now=None):
    """Expand the ComfyUI filename tokens we can resolve without the prompt graph:
    %width% / %height% and %date:FORMAT% (yyyy MM dd hh mm ss). Non-date %...%
    tokens (e.g. %Node.widget%) are left untouched for the core to handle. Pure
    (a fixed `now` makes it deterministic for the guard)."""
    s = str(prefix or "")
    now = now or _dt.datetime.now()
    s = s.replace("%width%", str(int(width))).replace("%height%", str(int(height)))

    def _repl(m):
        fmt = m.group(1)
        for a, b in _DATE_MAP:
            fmt = fmt.replace(a, b)
        try:
            return now.strftime(fmt)
        except Exception:
            return ""

    # v1154 (audit): only %date:...% -- a generic %...% pair took "%_%" out of "100%_%date:yy%" and left the date
    # token literal (ComfyUI's own expander matches %date:(.*?)% the same way)
    return _re2.sub(r"%date:([^%]*)%", _repl, s)


def sanitize_prefix(prefix, width=0, height=0, now=None):
    """Expand what we can, then guarantee a filesystem-safe relative prefix:
    tokens resolved, backslashes normalised, absolute/traversal segments dropped,
    and every Windows-illegal character (notably the ':' from an unexpanded date
    token) stripped per segment. Never returns '' -> falls back to 'Polyhedron'."""
    s = expand_filename_tokens(prefix, width, height, now).replace("\\", "/")
    while s.startswith("/"):
        s = s[1:]
    parts = []
    for seg in s.split("/"):
        seg = "".join(ch for ch in seg if ch not in _ILLEGAL_PATH_CHARS and ord(ch) >= 32)
        seg = seg.strip(" .")                  # Windows: no trailing dot/space
        if seg in ("", ".", ".."):
            continue
        parts.append(seg)
    return "/".join(parts) if parts else "Polyhedron"


def audio_pass_supported(preset_name):
    """True when the chosen preset actually carries an audio track (the Pillow
    GIF/WebP paths do not)."""
    return preset(preset_name).get("acodec") is not None
