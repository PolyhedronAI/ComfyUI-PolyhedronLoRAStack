"""
uls_latent_math.py -- pure, torch-free latent-shape planning for
⬡ Polyhedron Empty Latent.

WHY A SEPARATE PURE MODULE (the uls_tile_math precedent): the shape recipe is
the one thing that MUST be exactly right per latent type -- a wrong temporal
axis or spatial divisor silently mis-shapes the whole generation. Keeping it
torch-free means it runs and is unit-tested OUTSIDE ComfyUI (Messen schlaegt
Glauben); the node's torch/comfy code is a thin caller.

v517 REDESIGN -- latent TYPE instead of a flat model list.
  A flat `model_family` list is structurally doomed: it is always incomplete
  (Flux2, Qwen-Edit, whatever ships next). The insight: for IMAGE models the
  distinction barely matters -- ComfyUI's fix_empty_latent_channels() re-fits the
  channel count to the loaded model, and practically every image VAE is /8. The
  ONLY thing that genuinely cannot be guessed is VIDEO temporal packing, which is
  architecture-specific. So the selector collapses to the minimum necessary:
      image                         one entry for ALL image models (future-proof)
      wan/hunyuan/mochi/ltxv/cosmos the video architectures (explicit temporal)

EXACTNESS FROM THE VAE (the future-proof escape hatch): when a VAE is connected
the node reads latent_channels + spatial ratio straight off the VAE object and
passes them in here as overrides -> the geometry is exact for ANY model, present
or future (SD=4ch, Flux=16ch, Mochi=12ch, even a /32 image VAE), and correct even
with non-zero init noise (where the channel re-fit does NOT fire). No VAE -> the
type defaults below (a documented /8 16ch image default, cited video recipes).

TYPE_SPEC: key -> (channels, spatial_div, is_video, temporal_div)
  channels     default latent channel count. NOTE fix_empty_latent_channels()
               re-fits this for ZERO latents, and a connected VAE overrides it
               exactly -- so it is a sane default, not a hard dependency.
  spatial_div  pixel->latent spatial divisor (VAE /8 for all but LTXV /32).
  is_video     True -> 5D latent with a temporal axis; drives length being live.
  temporal_div video frame packing: latent_frames = (length - 1)//temporal_div + 1.

Only recipes verified against the ComfyUI core are listed; more are added ONLY
after their exact recipe is confirmed -- never a guessed shape. Verified sources:
  image     nodes.py EmptyLatentImage / SD3 / Flux -> [B, C, H//8, W//8] (C re-fit)
  wan       comfy_extras/nodes_wan.py    WanImageToVideo  -> [B,16,(L-1)//4+1,H//8,W//8]
  hunyuan   comfy_extras/nodes_hunyuan.py EmptyHunyuanLatentVideo -> [B,16,(L-1)//4+1,H//8,W//8]
  mochi     comfy_extras/nodes_mochi.py  EmptyMochiLatentVideo   -> [B,12,(L-1)//6+1,H//8,W//8]
  ltxv      comfy_extras/nodes_lt.py     EmptyLTXVLatentVideo    -> [B,128,(L-1)//8+1,H//32,W//32]
  cosmos    comfy_extras/nodes_cosmos.py EmptyCosmosLatentVideo  -> [B,16,(L-1)//8+1,H//8,W//8]
"""

# ---- verified type table (extend only with a confirmed recipe) ------------------
TYPE_SPEC = {
    "image":   (16,  8, False, 0),   # 16ch image class (SD3 / Flux.1 / Qwen / Chroma)
    "wan":     (16,  8, True,  4),   # Wan 2.1 / 2.2 video (and stills at length=1)
    "hunyuan": (16,  8, True,  4),   # HunyuanVideo
    "mochi":   (12,  8, True,  6),   # Genmo Mochi
    "ltxv":    (128, 32, True, 8),   # Lightricks LTX-Video (highly compressed VAE)
    "cosmos":  (16,  8, True,  8),   # NVIDIA Cosmos video
    "sd":      (4,   8, False, 0),   # classic 4ch image class (SD 1.x / 2.x / SDXL)
    # v872: the video half's geometry, listed so is_video_family() and the
    # frontend greying behave. The REAL builder is the minimax_* arithmetic
    # below -- temporal_div 0 is a MARKER that this row cannot pack frames.
    "minimax_h3": (24, 16, True, 0),
    "flux2":   (128, 16, False, 0),  # MEASURED in the field (v679 run): a
                                     # 1440x1440 decode carried a
                                     # (1, 128, 90, 90) latent -- 128 channels,
                                     # /16 spatial. The wired VAE or the core
                                     # delegate still win when present; this row
                                     # now carries correctly on its own too.
}

# Families whose empty latent should be BORROWED from the host ComfyUI's own
# node when available (registry lookup, version-proof) instead of built from
# the fallback spec. Truth order in the node: wired VAE probe > core delegate
# > spec fallback.
CORE_DELEGATES = {
    "flux2": "EmptyFlux2LatentImage",
}

# UI order + human labels for the latent_type combo (single source; node + JS agree).
# The combo VALUE is the label; the node maps label -> key via LABEL_TO_KEY.
# APPEND-ONLY: existing labels are never renamed (the combo serializes the
# label VALUE into workflows); new families join at the end.
TYPE_ORDER = ["image", "wan", "hunyuan", "mochi", "ltxv", "cosmos",
              "sd", "flux2", "minimax_h3"]
TYPE_LABELS = {
    "image":   "Image",
    "wan":     "WAN video",
    "hunyuan": "Hunyuan video",
    "mochi":   "Mochi video",
    "ltxv":    "LTXV video",
    "cosmos":  "Cosmos video",
    "sd":      "SD/SDXL image",
    "flux2":   "Flux2 image",
    "minimax_h3": "MiniMax H3 AV (video+audio)",
}
LATENT_TYPE_LABELS = [TYPE_LABELS[k] for k in TYPE_ORDER]
LABEL_TO_KEY = {TYPE_LABELS[k]: k for k in TYPE_ORDER}

# Back-compat: old model_family values (and a few spellings) resolve gracefully.
# A brand-new node has no saved graphs in the wild, but presets/robustness are cheap.
ALIASES = {
    # 16ch class stays "image"; flux2 and the classic 4ch models now resolve
    # to their OWN families (spec-correct -- the old image mapping put SDXL
    # at 16 channels, which was wrong without a wired VAE).
    "flux": "image", "sd3": "image",
    "qwen": "image", "qwen_image": "image", "qwen_edit": "image", "chroma": "image",
    "sdxl": "sd", "sd15": "sd", "sd21": "sd",
    "hunyuan_video": "hunyuan", "ltx": "ltxv", "ltx_video": "ltxv",
}

# Kept for external callers / older guards that referenced the WAN spec by this name.
FAMILY_SPEC = TYPE_SPEC
FAMILY_ORDER = TYPE_ORDER


# ---- MiniMax H3: a JOINT audio/video latent -------------------------------
# MEASURED against Core comfy_extras/nodes_minimax_h3.py (align_frame_count,
# video_latent_t, temporal_shape, _empty_av_latent).
#
# This family deliberately gets its OWN arithmetic instead of a TYPE_SPEC row:
# the 4-tuple cannot express it. Its frame packing is NOT (L-1)//div+1 but a
# 17k+5 grid, and the latent is a PAIR of tensors (video + audio) inside a
# NestedTensor, not a single 5D block. Forcing it into the table would have
# meant a lying row -- and this file's rule is "never a guessed shape".
MINIMAX_FPS = 24
MINIMAX_AUDIO_LATENT_FPS = 40
MINIMAX_VIDEO_CH = 24
MINIMAX_AUDIO_CH = 32
MINIMAX_SPATIAL_DIV = 16
MINIMAX_CANVAS_MULTIPLE = 32


def minimax_align_frames(n):
    """Core's align_frame_count: the model's frame grid is 17k + 5."""
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def minimax_video_latent_t(frame_count):
    """Core's video_latent_t: latent frames for an ALIGNED pixel frame count."""
    fc = int(frame_count)
    return 2 if fc <= 5 else ((fc - 5) // 17) * 5 + 2


def minimax_frames_from_latent_t(latent_t):
    """EXACT inverse of minimax_video_latent_t.

    Exact because align_frame_count guarantees frame_count % 17 == 5, so
    (frame_count - 5) is always a multiple of 17 and the // 17 above loses
    nothing. This is what lets a downstream node recover the frame count from a
    latent it did not build."""
    lt = int(latent_t)
    return 5 if lt <= 2 else ((lt - 2) // 5) * 17 + 5


def minimax_temporal_shape(length):
    """Core's temporal_shape -> (frame_count, video_latent_t, audio_latent_t)."""
    fc = minimax_align_frames(length)
    audio_t = round(fc / float(MINIMAX_FPS) * MINIMAX_AUDIO_LATENT_FPS)
    return fc, minimax_video_latent_t(fc), int(audio_t)


def minimax_frames_from_seconds(seconds):
    """Seconds at 24 fps, snapped up to the 17k+5 grid. This is what replaces
    the Float + Math Expression pair in the stock template."""
    return minimax_align_frames(round(float(seconds) * MINIMAX_FPS))


def minimax_image_shape():
    """v899: the SINGLE-FRAME shape -> (frame_count, video_latent_t, audio_t).

    Deliberately NOT part of minimax_temporal_shape above. That function is a
    mirror of Core's temporal_shape, and a mirror that "improves" its source is
    a drift (the v749 rule). Core's temporal_shape belongs to
    EmptyMiniMaxH3LatentAV -- the AUDIO+VIDEO node -- and starts with
    max(5, length) because a clip below five frames is not a clip. That floor is
    the NODE's, not the model's, and v898 wrongly reported it as the model's.

    A single frame IS a supported state, read at two independent sources:

      comfy/ldm/minimax/vae.py, decode_output_shape:
          if t == 1: frames = 1
      ...and CausalConv3d.forward has its own branch for x.shape[2] == 1,
      truncating the temporal taps instead of convolving zero frames -- you do
      not write that for a shape that never occurs.

      ai-toolkit, extensions_built_in/.../minimax_h3/src/pipeline.py:
          is_video = num_frames > 1
          ...
          else:   # true single-frame generation (image mode)
              num_frames = 1
              t_lat = 1
      and the module header states image datasets train as single latent
      frames, with the 17n+5 snapping firing only for num_frames > 1.

    THE AUDIO HALF STAYS, even though the toolkit drops it for stills
    (with_audio = with_audio and is_video). Reason, measured: Core's model
    class reads `audio_src = x[1]` unconditionally (comfy/ldm/minimax/model.py)
    -- the same line three field crashes landed on in August. An empty audio
    part costs nothing; omitting it is a bet against the caller. Its length is
    the ordinary one for a single frame: round(1/24 * 40) = 2.
    """
    audio_t = int(round(1 / float(MINIMAX_FPS) * MINIMAX_AUDIO_LATENT_FPS))
    return 1, 1, audio_t


def minimax_size_from_latent(samples):
    """(width, height, frame_count) recovered from a joint latent, or None.

    The video half is [B, 24, latent_t, H//16, W//16]; every number the
    reference stage needs is in there, so a node fed a latent from OUTSIDE has
    exactly one source of truth instead of a second set of widgets that can
    drift out of step with it."""
    if not getattr(samples, "is_nested", False):
        return None
    parts = samples.unbind()
    if not parts or getattr(parts[0], "ndim", 0) != 5:
        return None
    v = parts[0]
    return (int(v.shape[4]) * MINIMAX_SPATIAL_DIV,
            int(v.shape[3]) * MINIMAX_SPATIAL_DIV,
            minimax_frames_from_latent_t(int(v.shape[2])))


def canonical_type(latent_type):
    """Map a combo label OR a raw key/alias to a canonical TYPE_SPEC key.
    Unknown -> 'image' (the safe, universal default)."""
    if latent_type in TYPE_SPEC:
        return latent_type
    if latent_type in LABEL_TO_KEY:
        return LABEL_TO_KEY[latent_type]
    if latent_type in ALIASES:
        return ALIASES[latent_type]
    low = str(latent_type).strip().lower()
    if low in TYPE_SPEC:
        return low
    if low in ALIASES:
        return ALIASES[low]
    return "image"


def is_video_family(latent_type):
    """True if the type produces a 5D (temporal) latent. Accepts label/key/alias."""
    spec = TYPE_SPEC.get(canonical_type(latent_type))
    return bool(spec and spec[2])


# alias kept for callers that used the v516 name
is_video_type = is_video_family


def resolve_geometry(latent_type, vae_channels=None, vae_spatial_div=None):
    """Return (channels, spatial_div, is_video, temporal_div) for the type, with
    optional exact overrides read off a connected VAE.

    vae_channels / vae_spatial_div are used verbatim WHEN VALID (positive ints);
    otherwise the verified type defaults apply. is_video and temporal_div always
    come from the type (temporal packing is architectural, never guessed)."""
    key = canonical_type(latent_type)
    ch, sdiv, is_video, tdiv = TYPE_SPEC[key]
    try:
        if vae_channels is not None and int(vae_channels) >= 1:
            ch = int(vae_channels)
    except (TypeError, ValueError):
        pass
    try:
        if vae_spatial_div is not None and int(vae_spatial_div) >= 1:
            sdiv = int(vae_spatial_div)
    except (TypeError, ValueError):
        pass
    return ch, sdiv, is_video, tdiv


def latent_frames(length, latent_type):
    """Latent temporal length for a video type (image types -> 1).
    length is clamped to >= 1 first (a 0/negative pixel length is meaningless)."""
    key = canonical_type(latent_type)
    spec = TYPE_SPEC.get(key)
    if not spec or not spec[2]:
        return 1
    tdiv = spec[3]
    L = int(length) if int(length) >= 1 else 1
    return (L - 1) // tdiv + 1


def plan_latent_shape(latent_type, width, height, length, batch_size,
                      channels=None, spatial_div=None):
    """Return the latent tensor shape tuple for the type.

    Image types -> (B, C, H//div, W//div)   [4D]
    Video types -> (B, C, T, H//div, W//div) [5D], T = (L-1)//tdiv + 1

    channels / spatial_div override the type defaults when provided (from a
    connected VAE) -> exact geometry for any model. Spatial dims are floor-divided
    (the canvas snaps to the VAE grid)."""
    ch, sdiv, is_video, tdiv = resolve_geometry(latent_type, channels, spatial_div)
    b = int(batch_size) if int(batch_size) >= 1 else 1
    lh = int(height) // sdiv
    lw = int(width) // sdiv
    if lh < 1:
        lh = 1
    if lw < 1:
        lw = 1
    if is_video:
        return (b, ch, latent_frames(length, latent_type), lh, lw)
    return (b, ch, lh, lw)


def spatial_divisor(latent_type):
    """VAE spatial divisor for the type (default 8 if unknown)."""
    spec = TYPE_SPEC.get(canonical_type(latent_type))
    return spec[1] if spec else 8
