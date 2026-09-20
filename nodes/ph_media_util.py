# ph_media_util.py — torch-free helpers for the Media Loader
#
# Kept free of torch (and av/cv2) on purpose so the pure logic — alpha pixel
# format detection and the RGBA -> (RGB, mask) split — can be unit-tested in a
# bare environment. ph_media_loader imports these; the heavy decoders stay in
# the loader.

import numpy as np

# ── The extension law (v377) ────────────────────────────────────────────────
# ONE list per media kind, for the loader AND the routes.
#
# Before v377 these tuples lived TWICE -- once in ph_media_loader, once in
# uls_routes -- with a comment promising they were "kept in lock-step". A
# comment is not a mechanism: the moment a format is added, it has to be
# remembered in two files, and whichever half is forgotten fails silently
# (the grid lists a file the loader refuses, or the loader accepts one the
# grid never shows). Public issue #4 is exactly that shape. The law lives
# here; both sides import it, and test_v377_media_formats forbids a second
# literal extension tuple anywhere else in the tree.
#
# .avif joins in v377. Listing a format and DECODING it are two different
# questions -- see avif_decoder_ready() below; the grid may show a file the
# installed Pillow cannot open, and the loader has to say so in words.
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp", ".gif",
              ".tif", ".tiff")
# Videos proper. The loader additionally treats ".gif" as a video source (a
# gif can go down either path: first frame as a still, or all frames as a
# clip), the routes deliberately do NOT -- a gif tile shows as an image.
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v")
AUDIO_EXTS = (".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus",
              ".aiff", ".aif", ".wma")
# Stills eligible for the image-batch path. No .mp4 &c; .gif decodes to its
# first frame here, exactly as a still gif does in _load_image.
BATCH_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".avif", ".bmp",
                    ".tif", ".tiff", ".gif")


def avif_decoder_ready() -> bool:
    """Can the INSTALLED Pillow actually decode AVIF? Measured, never assumed.

    Pillow gained native AVIF support in 11.3; older builds need the separate
    `pillow-avif-plugin` package. Both end up registering the same plugin, so
    the honest test is to ask Pillow's own feature registry and fall back to
    the extension registry when the registry predates the feature flag."""
    try:
        from PIL import features
        if features.check("avif"):
            return True
    except Exception:
        pass
    try:
        from PIL import Image
        return ".avif" in Image.registered_extensions()
    except Exception:
        return False


AVIF_HINT = ("AVIF support needs Pillow 11.3 or newer (or the "
             "'pillow-avif-plugin' package on older Pillow). Install one of "
             "those and restart ComfyUI, or convert the file to PNG/WebP.")

# ffmpeg/PyAV pixel-format name prefixes that carry an alpha channel. Prefix
# matching covers every bit-depth/endianness variant in one shot, e.g.
# "yuva420p" (VP9 alpha), "yuva444p10le" (ProRes 4444), "rgba"/"bgra64le", etc.
_ALPHA_PIX_PREFIXES = ("yuva", "rgba", "bgra", "argb", "abgr", "ya8", "ya16", "gbrap")


def pix_fmt_has_alpha(pix_fmt) -> bool:
    """True if an ffmpeg/PyAV pixel-format name denotes an alpha channel.

    `pal8` (palettised, e.g. GIF) is treated as alpha-capable because it can
    carry transparency that ffmpeg expands to a real alpha plane on decode.
    """
    p = (pix_fmt or "").lower()
    if not p:
        return False
    if p == "pal8":
        return True
    return any(p.startswith(x) for x in _ALPHA_PIX_PREFIXES)


def oriented_size(width, height, exif_orientation=None):
    """Return (w, h) after applying an EXIF orientation's axis swap.

    EXIF Orientation tag values 5/6/7/8 transpose the image (a 90deg rotation),
    so width and height swap; 1..4 (and None / unknown / garbage) leave them as
    they are. The Media Loader decodes images with ImageOps.exif_transpose, so
    the dimensions reported in the listing must reflect that transposed
    (displayed) size — this matches it. Pure integer logic, no decode, so it is
    unit-testable in a bare environment.
    """
    try:
        o = int(exif_orientation) if exif_orientation is not None else 1
    except (TypeError, ValueError):
        o = 1
    if o in (5, 6, 7, 8):
        return int(height), int(width)
    return int(width), int(height)


def rgb_and_mask_from_rgba(rgba_float: np.ndarray):
    """Split an RGBA float array in [0,1] into (rgb, mask).

    `rgba_float` is [..., 4] (a single frame [H,W,4] or a batch [N,H,W,4]).
    Returns rgb [..., 3] and mask [...] following the ComfyUI/LoadImage
    convention: opaque (alpha 1) -> 0, transparent (alpha 0) -> 1.
    """
    rgb = rgba_float[..., :3]
    mask = 1.0 - rgba_float[..., 3]
    return rgb, mask


def rgb_and_mask_from_still(rgba_float: np.ndarray):
    """Still-image split with the v654 grayscale rule.

    A GRAYSCALE still with NO alpha information (every pixel R==G==B exactly,
    alpha fully opaque) carries its mask in the pixels themselves -- saved
    masks come back that way (the Save writes mode-L, white = mask on). For
    those, the LUMINANCE is the mask DIRECTLY (white -> 1): a Background
    Remove subject mask round-trips Save -> Load unchanged. Anything with
    real alpha keeps the LoadImage convention untouched (opaque -> 0), and a
    color image without alpha keeps its all-zero mask as before.

    Returns (rgb, mask, used_luminance).
    """
    rgb, mask = rgb_and_mask_from_rgba(rgba_float)
    a = rgba_float[..., 3]
    if (a >= 1.0).all():
        r = rgba_float[..., 0]
        if (r == rgba_float[..., 1]).all() and (r == rgba_float[..., 2]).all():
            if (r > 0.0).any():   # an all-black frame stays an empty mask
                return rgb, np.ascontiguousarray(r), True
    return rgb, mask, False


# ── Image-batch sequencing (torch-free, bare-env unit-testable) ──────────────
# These power the Media Loader's `load_mode = "image batch"` path: turn a folder
# of stills into a deterministic, ordered file list before any pixels are read.
# Kept pure (stdlib only) so GATE-3 covers the ordering/selection logic without
# torch/Pillow; the actual decode + stack stays in ph_media_loader (live-tested).
import os as _os
import re as _re
import fnmatch as _fnmatch

_NUM_RE = _re.compile(r"(\d+)")


def natural_sort_key(name):
    """Split a name into text/number runs so 'img2' sorts before 'img10'.

    `re.split(r"(\\d+)", s)` yields a list whose parity is identical for every
    string (even indices = text, odd indices = digit runs), so mapping odd->int
    and even->str never collides int-vs-str across two keys. Lower-cased for
    case-insensitive ordering; stable sort preserves input order on case ties.
    """
    s = str(name).lower()
    return [int(tok) if (i % 2 == 1) else tok for i, tok in enumerate(_NUM_RE.split(s))]


def filter_names(names, pattern):
    """Case-insensitive glob filter on the bare name. '' or '*' -> pass-through."""
    pat = (pattern or "*").strip() or "*"
    if pat == "*":
        return list(names)
    pl = pat.lower()
    return [n for n in names if _fnmatch.fnmatch(str(n).lower(), pl)]


import re as _re


def match_names(names, expr):
    """The v528 smart matcher (shared spec with the JS mirror in
    ph_media_loader.js — the parity guard feeds both the same fixtures).

    Semantics, all case-insensitive on the bare name:
      * '' or '*'            -> everything (legacy pass-through).
      * 're:PATTERN'         -> PATTERN is one Python-style regex, matched with
                                search(); NO comma splitting (regexes contain commas).
      * otherwise, split on commas into tokens:
          - a leading '!' marks the token as an EXCLUSION;
          - a token containing * ? [ is a glob (fnmatch, like before);
          - any other token is a SUBSTRING test ('PH' hits 'PH_2020…' — the
            classic forgot-the-star trap is gone).
        Positive tokens OR together; no positive tokens (pure exclusions) start
        from everything. Exclusions then remove their matches.
    Original input order is preserved; ordering stays order_names' job."""
    e = str(expr or "*").strip()
    if e in ("", "*"):
        return list(names)
    low = [(n, str(n).lower()) for n in names]
    if e.lower().startswith("re:"):
        try:
            rx = _re.compile(e[3:], _re.IGNORECASE)
        except _re.error:
            return []                              # broken regex selects nothing (visibly)
        return [n for n, _ln in low if rx.search(str(n))]

    def _tok_match(tok, ln):
        if any(ch in tok for ch in "*?["):
            return _fnmatch.fnmatch(ln, tok)
        return tok in ln

    pos, neg = [], []
    for raw in e.split(","):
        t = raw.strip().lower()
        if not t:
            continue
        (neg if t.startswith("!") else pos).append(t.lstrip("!").strip())
    pos = [t for t in pos if t]
    neg = [t for t in neg if t]
    kept = []
    for n, ln in low:
        ok = True if not pos else any(_tok_match(t, ln) for t in pos)
        if ok and neg and any(_tok_match(t, ln) for t in neg):
            ok = False
        if ok:
            kept.append(n)
    return kept


def order_names(names, sort_mode="name (natural)", times=None):
    """Order names by the chosen mode. Time modes use the optional `times` map
    {name: float}; ties (and missing times) fall back to natural order."""
    names = list(names)
    if sort_mode == "name (literal)":
        return sorted(names)
    if sort_mode in ("mtime (oldest first)", "created"):
        t = times or {}
        return sorted(names, key=lambda n: (t.get(n, 0.0), natural_sort_key(n)))
    return sorted(names, key=natural_sort_key)  # "name (natural)" (default)


def select_slice(names, skip_first=0, every_nth=1, cap=0, hard_cap=2000):
    """skip_first -> select_every_nth -> cap, in that fixed order (VHS-compatible).
    cap<=0 means 'all' but is always clamped to hard_cap for safety."""
    s = max(0, int(skip_first))
    nth = max(1, int(every_nth))
    out = list(names)[s:][::nth]
    lim = int(cap) if int(cap) > 0 else int(hard_cap)
    lim = min(lim, int(hard_cap))
    return out[:lim]


def select_frames(names, sort_mode="name (natural)", name_filter="*",
                  skip_first=0, every_nth=1, cap=0, hard_cap=2000, times=None):
    """Full pipeline: filter -> order -> slice. Returns the final ordered names.
    v528: the filter step is the smart matcher (substring/glob/comma-OR/!excl/re:);
    every legacy value ('*', '*.png', 'frame*') behaves exactly as before."""
    return select_slice(
        order_names(match_names(names, name_filter), sort_mode, times=times),
        skip_first, every_nth, cap, hard_cap,
    )


def frames_target_and_offenders(dims):
    """dims: list of (w, h). Target = the FIRST frame's (w, h); offenders = the
    indices whose (w, h) differ from the target. Empty -> (None, [])."""
    if not dims:
        return None, []
    target = (int(dims[0][0]), int(dims[0][1]))
    offenders = [i for i, d in enumerate(dims) if (int(d[0]), int(d[1])) != target]
    return target, offenders


# ─── Sequence-library path logic (v443) ──────────────────────────
# Pure path/naming logic for the Media Loader's persistent sequence library,
# kept here (torch-free) so the safety-critical guard in front of a sequence's
# recursive delete is gate-testable without ComfyUI/folder_paths. uls_routes
# imports these and does the actual filesystem I/O (scan/copy/delete), supplying
# the one folder_paths touch the pure logic can't: the output root the library
# lives under (<output>/PLS_sequences). Replaces the v424 _pls_prep scratch model
# (one discardable subfolder per source) with a library of named, persistent
# project folders decoupled from the raw source -> no nesting, no cross-sequence
# clobbering, and each is directly selectable as the active batch.

SEQUENCES_DIRNAME = "PLS_sequences"

# Allowed characters in a sequence (project) folder name. Deliberately strict:
# the name becomes a real on-disk folder under the library, so anything that
# could escape that one segment (separators, traversal, drive colon, control
# chars) is stripped rather than escaped.
_SEQ_NAME_BAD = _re.compile(r"[^A-Za-z0-9 ._()\-]+")


def safe_project_name(name):
    """Reduce an arbitrary user string to ONE safe folder segment, or "" if
    nothing safe remains. Path separators, traversal, drive colons and control
    characters cannot survive: only [A-Za-z0-9 ._()-] is kept, leading/trailing
    dots, spaces and dashes are trimmed, and a lone "." / ".." collapses to "".
    The result, when non-empty, is always a single path component (basename ==
    itself) so it can only ever name a direct child of the library."""
    if not name:
        return ""
    n = _SEQ_NAME_BAD.sub("", str(name))
    n = n.strip(" .\t\r\n-")
    if n in ("", ".", ".."):
        return ""
    n = _os.path.basename(n)   # defensive: guarantee a single segment
    if n in ("", ".", ".."):
        return ""
    return n


def sequence_dir_for(library_rp, name):
    """<library>/<safe(name)> or "" if the name sanitises away. `library_rp` is
    expected already realpath-resolved; the folder is NOT created here."""
    safe = safe_project_name(name)
    if not safe:
        return ""
    return _os.path.join(library_rp, safe)


def is_within_library(library, project):
    """True iff `project` is a safe, DIRECT, named child of `library` and is safe
    to wipe. EVERY clause must hold, else refuse:
      * basename sanitises to itself and is non-empty (a real project name),
      * resolves inside library and is not the library itself,
      * is a DIRECT child of library (no nesting / traversal tricks),
      * is not a filesystem/drive root.
    Pure path logic -> the guard in front of Delete's recursive delete is unit-
    testable without a live ComfyUI. `library` is passed in (not fetched) so this
    stays free of folder_paths."""
    try:
        lib = _os.path.realpath(library)
        pr = _os.path.realpath(project)
        if not lib or not pr:
            return False
        base = _os.path.basename(pr)
        if not base or safe_project_name(base) != base:   # a real, safe name
            return False
        if pr == lib:                                      # not the library itself
            return False
        if not pr.startswith(lib + _os.sep):               # inside the library
            return False
        if _os.path.dirname(pr) != lib:                    # a DIRECT child
            return False
        if pr == _os.path.dirname(pr):                     # never a drive/fs root
            return False
        return True
    except Exception:
        return False


def renumbered_targets(selected_names, min_width=5):
    """Map an ordered name list to [(src_name, dst_name), ...] where dst is a
    zero-padded 1-based index plus the lower-cased original extension (>= min_width
    digits, widening for large sets). Ordering is preserved exactly."""
    n = len(selected_names)
    width = max(int(min_width), len(str(n)))
    out = []
    for i, nm in enumerate(selected_names, start=1):
        ext = _os.path.splitext(nm)[1].lower() or ".png"
        out.append((nm, f"{i:0{width}d}{ext}"))
    return out
