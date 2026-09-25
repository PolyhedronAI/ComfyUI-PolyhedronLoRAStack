"""
Polyhedron Filter (ULSFilter) -- one-node color grading with in-node preview.

The color pipeline is ACTIVE: _grade_np is the single ground truth, applied
chunked over the IMAGE batch on every run, and web/js/ph_filter.js mirrors
it op for op so the in-node preview reacts to the sliders live (the parity
guard drives both sides on the same input). The 3D LUT stage (.cube from
luts/, trilinear, strength blend) runs after the color controls, mirrored
in the preview through the /uls/filter/lut route, followed by the unsharp
mask (_sharpen_np, mirrored the same way). Presets live in the FRONTEND:
selecting one loads it over /uls/filter/preset and writes the values onto
the widgets (the serialized widget values stay the single truth), saving
POSTs the current look there; the preset widget itself is deliberately
ignored by apply() -- applying it here too would double-apply the look. The widget canon was frozen
at full size from day one (append-only law, HANDOVER 4); the preview
travels over one ui channel ({"ui": {"ph_filter": [...]}}) -- the exact
mechanic ph_save.py uses -- and always carries the UNGRADED source frame.

Canon rules that bind every future edit of this node:
  * FILTER_CANON is APPEND-ONLY. Never insert, never reorder, never remove --
    LiteGraph stores widgets_values positionally (HANDOVER 4).
  * The order doubles as the grading PIPELINE order: exposure -> white
    balance -> tone -> color -> LUT -> sharpen. New controls go at the end of
    the canon even if they run mid-pipeline.
  * Outputs are append-only too: IMAGE stays Slot 0 forever; a future edge
    MASK output gets appended behind it.
"""

import os
import uuid

import numpy as np

# v1010: the shared progress instrument (green bar + console + learned ETA).
# A harness that loads this file alone gets silent stand-ins -- the node's
# work never depends on its instruments.
try:
    from .ph_progress import NodeProgress as _NodeProgress, blocking as _blocking
except Exception:
    try:
        from ph_progress import NodeProgress as _NodeProgress, blocking as _blocking
    except Exception:
        class _blocking:
            est_total = est = None

            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def tick(self, n=1):
                pass

            def rate_left(self):
                return None, None

        _NodeProgress = _blocking


# ---------------------------------------------------------------------------
# canon (APPEND-ONLY -- see module docstring)
# ---------------------------------------------------------------------------
FILTER_CANON = (
    "exposure",
    "temperature",
    "tint",
    "contrast",
    "gamma",
    "shadows",
    "highlights",
    "saturation",
    "vibrance",
    "hue_shift",
    "lut_name",
    "lut_strength",
    "sharpen_amount",
    "sharpen_radius",
    "preset",
    "auto_mode",       # v998 F2 -- appended; runs FIRST in the pipeline
    "sharpen_threshold",  # v1000 F3 -- appended; part of the last stage
    "detail_amount",      # v1001 F4 -- appended; runs FIRST (before auto)
)

# Long edge of the in-node preview image (px). The preview is a downscaled
# proxy the frontend grades live; the run output is always full resolution.
PREVIEW_MAX_EDGE = 768

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LUT_DIR = os.path.join(_PACK_ROOT, "luts")
PRESET_DIR = os.path.join(_PACK_ROOT, "presets")


# ---------------------------------------------------------------------------
# the automatic (v998, F2) -- taken over from the Polyhedron Viewer
# (pv/grade.py, v022/v064), with two corrections measured on 24.09.2026
# ---------------------------------------------------------------------------
# One dial, four stops; every stop moves the same four corrections by its own
# share. OFF is in the table so nothing has to special-case it. The Viewer's
# contour-driven auto sharpening is NOT taken over: its thresholds (soft
# < 0.008, crisp > 0.020) read every H3 frame as crisp (measured 0.075-0.10)
# and film grain counts as contour -- no auto sharpening without a reading
# that tells grain from edges.
AUTO_MODES = ("off", "neutral", "high", "ultra")
AUTO_FACTORS = {
    "off":     {"wb": 0.0,  "luma": 0.0,  "spread": 0.0,  "sat": 0.0},
    "neutral": {"wb": 0.6,  "luma": 0.5,  "spread": 0.5,  "sat": 0.0},
    "high":    {"wb": 0.85, "luma": 0.75, "spread": 0.75, "sat": 0.25},
    "ultra":   {"wb": 1.0,  "luma": 1.0,  "spread": 1.0,  "sat": 0.5},
}
# Targets = medians of the 29 LIVE1 photographs, sampled every 8th pixel:
# luma 0.444 (the Viewer's 0.45 confirmed), luma spread 0.193 (Viewer 0.22),
# per-pixel chroma 0.046 (Viewer 0.085 was never measured).
AUTO_TARGET_LUMA = 0.45
AUTO_TARGET_SPREAD = 0.19
AUTO_TARGET_CHROMA = 0.046
AUTO_CHOICES = list(AUTO_MODES)   # the combo (a plain name: guards read the defaults)
AUTO_FRAMES = 8     # frames sampled per batch, evenly spread
AUTO_STEP = 8       # every 8th pixel in both directions


def _auto_stats(batch):
    """[mean_r, mean_g, mean_b, luma_mean, luma_std, chroma] of an IMAGE
    batch (float [N, H, W, 3] or [H, W, 3], 0..1), read from up to AUTO_FRAMES
    evenly spread frames, every AUTO_STEP-th pixel. ONE answer per batch: a
    per-frame automatic would flicker on video (the Viewer grades per clip for
    the same reason).

    chroma is the mean distance of each pixel's channels from that pixel's
    own mean -- CORRECTION 1: the Viewer took it from the frame's MEAN colour,
    which measures the cast, not the saturation; a balanced picture read as
    grey and HIGH doubled its saturation (measured x1.4-1.9 on Frank's H3
    frames). Pure -- guard-driven."""
    a = np.asarray(batch, dtype=np.float32)
    if a.ndim == 3:
        a = a[None]
    n = int(a.shape[0])
    if n <= 0:
        return None
    k = min(AUTO_FRAMES, n)
    idx = sorted(set(int(round(i * (n - 1) / max(k - 1, 1))) for i in range(k)))
    s = a[idx][:, ::AUTO_STEP, ::AUTO_STEP, :3].reshape(-1, 3).astype(np.float64)
    if s.shape[0] == 0:
        return None
    luma = 0.2126 * s[:, 0] + 0.7152 * s[:, 1] + 0.0722 * s[:, 2]
    chroma = np.abs(s - s.mean(axis=1, keepdims=True)).mean()
    m = s.mean(axis=0)
    return [float(m[0]), float(m[1]), float(m[2]), float(luma.mean()),
            float(luma.std()), float(chroma)]


def _auto_grade(stats, mode):
    """The automatic correction for these stats at this mode, or None when
    it would change nothing (off, unknown mode, no stats). Mirrored op for op
    by _autoGrade in ph_filter.js. Pure -- guard-driven.

    white balance  grey world: wb_r = G/R, wb_b = G/B, by the mode's share
    brightness     toward AUTO_TARGET_LUMA, both ways
    contrast       CORRECTION 2 (the Viewer's own words: "stretches a FLAT
                   luma spread"): only a spread UNDER the target is
                   stretched; a contrasty picture is never flattened
    saturation     only a dull picture is lifted (the Viewer's words:
                   "lifted when the frame is flat"), never lowered"""
    f = AUTO_FACTORS.get(mode)
    if not f or not stats or f["wb"] == 0.0:
        return None
    mr, mg, mb, luma, spread, chroma = [float(v) for v in stats[:6]]

    def lim(v, lo, hi):
        return lo if v < lo else hi if v > hi else v

    wb_r = wb_b = 1.0
    if mr > 0.01 and mg > 0.01 and mb > 0.01:
        wb_r = lim(1.0 + (mg / mr - 1.0) * f["wb"], 0.5, 2.0)
        wb_b = lim(1.0 + (mg / mb - 1.0) * f["wb"], 0.5, 2.0)
    brightness = lim((AUTO_TARGET_LUMA - luma) * f["luma"], -0.5, 0.5)
    contrast = 1.0
    if 0.01 < spread < AUTO_TARGET_SPREAD:
        contrast = lim(1.0 + (AUTO_TARGET_SPREAD / spread - 1.0) * f["spread"], 1.0, 2.0)
    saturation = 1.0
    if f["sat"] and 0.002 < chroma < AUTO_TARGET_CHROMA:
        saturation = lim(1.0 + (AUTO_TARGET_CHROMA / chroma - 1.0) * f["sat"], 1.0, 2.0)
    return {"wb_r": wb_r, "wb_b": wb_b, "brightness": brightness,
            "contrast": contrast, "saturation": saturation}


def _apply_auto_np(x, a):
    """Stage 0 of the pipeline: the automatic, in the Viewer's order (gains,
    brightness, contrast around 0.5, saturation against Rec.709 luma), then
    clamped so every later stage sees 0..1 as before. Mirrored op for op by
    _autoRGB in ph_filter.js."""
    x = np.asarray(x, dtype=np.float32).copy()
    if a is None:
        return x
    x[..., 0] *= np.float32(a["wb_r"])
    x[..., 2] *= np.float32(a["wb_b"])
    x = x + np.float32(a["brightness"])
    x = (x - np.float32(0.5)) * np.float32(a["contrast"]) + np.float32(0.5)
    luma = (np.float32(0.2126) * x[..., 0] + np.float32(0.7152) * x[..., 1]
            + np.float32(0.0722) * x[..., 2])[..., None]
    x = luma + (x - luma) * np.float32(a["saturation"])
    return np.clip(x, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# detail from the original (v1001, F4) -- the Polyhedron Viewer's steered
# frequency separation (pv/detail.py, v060), batched
# ---------------------------------------------------------------------------
# Both pictures are split into a tone layer (a small binomial blur) and a
# contour layer. The original's contours lose noise (CORE), ringing (CAP)
# and -- when the original carries compression blocks -- most of what sits
# on the 8 px grid; only the SURPLUS moves over (where both layers point the
# same way and the original's points further), weighted by the local 5x5
# agreement of the two layers. What the processing destroyed and the
# original still has comes back; nothing else does.
DETAIL_CORE = 0.002
DETAIL_CAP = 0.15
DETAIL_GRID = 0.6
DETAIL_BOX = 5
DETAIL_EPS = 1e-6
# THE GRID DECIDES ITSELF (measured 24.09.2026): the Viewer's grid was made
# for JPEG files; on a clean original it cost 0.9-1.5 dB. The block ratio of
# the source (8 px seam steps against all others) reads 0.95-1.045 on the 29
# clean LIVE1 pictures and 1.00-1.05 on Frank's H3 frames, at least 1.094 at
# JPEG quality 60 (median 1.28). The grid fades in from 1.05 to 1.20.
BLOCK_CLEAN = 1.05
BLOCK_FULL = 1.20


def _block_ratio(batch):
    """Mean 8 px seam step against every other step, both directions, on up
    to AUTO_FRAMES evenly spread frames. 1.0 = no block edges. Pure."""
    a = np.asarray(batch, dtype=np.float32)
    if a.ndim == 3:
        a = a[None]
    n = int(a.shape[0])
    k = min(AUTO_FRAMES, n)
    idx = sorted(set(int(round(i * (n - 1) / max(k - 1, 1))) for i in range(k)))
    out = []
    for f in a[idx]:
        lum = f[..., :3].astype(np.float64).mean(axis=2)
        for ax in (1, 0):
            if lum.shape[ax] < 16:
                continue
            d = np.abs(np.diff(lum, axis=ax))
            seam = (np.arange(d.shape[ax]) % 8) == 7
            on = d[:, seam] if ax == 1 else d[seam, :]
            off = d[:, ~seam] if ax == 1 else d[~seam, :]
            out.append(float(on.mean()) / max(float(off.mean()), 1e-6))
    return float(np.mean(out)) if out else 1.0


def _grid_strength(ratio):
    """How much of the seam suppression a source with this block ratio gets:
    0 below BLOCK_CLEAN, DETAIL_GRID from BLOCK_FULL on, linear between."""
    t = (float(ratio) - BLOCK_CLEAN) / (BLOCK_FULL - BLOCK_CLEAN)
    return DETAIL_GRID * (0.0 if t < 0.0 else 1.0 if t > 1.0 else t)


def _soft_np(x):
    """The tone layer: [1 2 1]/4 twice down, twice across, edges held.
    x: [..., H, W, C]. The Viewer's detail._soft, batched."""
    out = x
    for axis in (-3, -2):
        for _ in range(2):
            n = out.shape[axis]
            lo = np.take(out, np.r_[0, np.arange(n - 1)], axis=axis)
            hi = np.take(out, np.r_[np.arange(1, n), n - 1], axis=axis)
            out = (lo + 2.0 * out + hi) * 0.25
    return out


def _box_np(x, r=DETAIL_BOX // 2):
    """5x5 mean, edges held, over [..., H, W, C] (float64 summed tables)."""
    k = 2 * r + 1
    lead = [(0, 0)] * (x.ndim - 3)
    p = np.pad(x.astype(np.float64), lead + [(r, r), (r, r), (0, 0)], mode="edge")
    c = np.pad(p.cumsum(-3).cumsum(-2), lead + [(1, 0), (1, 0), (0, 0)])
    return ((c[..., k:, k:, :] - c[..., :-k, k:, :] - c[..., k:, :-k, :]
             + c[..., :-k, :-k, :]) / (k * k)).astype(np.float32)


def _grid_mask_np(height, width, grid):
    """1 inside the 8 px blocks, 1 - grid on the two pixels of each seam."""
    gx = np.ones(width, np.float32)
    gx[(np.arange(width) % 8 == 7) | (np.arange(width) % 8 == 0)] -= np.float32(grid)
    gy = np.ones(height, np.float32)
    gy[(np.arange(height) % 8 == 7) | (np.arange(height) % 8 == 0)] -= np.float32(grid)
    return gy[:, None] * gx[None, :]


def _detail_np(orig, ai, amount, grid=DETAIL_GRID):
    """Detail from the original onto the processed picture. orig, ai:
    float32 [..., H, W, 3] of the same size. With grid = DETAIL_GRID this is
    the Viewer's detail.reference exactly (the guard drives both). Mirrored
    op for op by _detailBuf in ph_filter.js."""
    orig = np.asarray(orig, np.float32)
    ai = np.asarray(ai, np.float32)
    if not (float(amount) > 0.0):
        return ai.copy()
    ho = orig - _soft_np(orig)
    ha = ai - _soft_np(ai)
    ho = np.sign(ho) * np.clip(np.abs(ho) - DETAIL_CORE, 0.0, DETAIL_CAP)
    if float(grid) > 0.0:
        ho = ho * _grid_mask_np(orig.shape[-3], orig.shape[-2], grid)[..., None]
    num = _box_np((ho * ha).sum(-1, keepdims=True))
    den = np.sqrt(np.maximum(_box_np((ho * ho).sum(-1, keepdims=True))
                             * _box_np((ha * ha).sum(-1, keepdims=True)), 0.0)) + DETAIL_EPS
    agree = np.clip(num / den, 0.0, 1.0)
    surplus = np.where(((ho * ha) > 0) & (np.abs(ho) > np.abs(ha)), ho - ha, 0.0)
    return np.clip(ai + float(amount) * agree * surplus, 0.0, 1.0).astype(np.float32)


def _detail_fit(image_shape, source_shape):
    """None when the source can carry detail onto the image, else the reason.
    Same height and width; one source frame for all, or one per frame."""
    if tuple(source_shape[1:3]) != tuple(image_shape[1:3]):
        return ("detail source is %dx%d, the image %dx%d -- sizes must match"
                % (source_shape[2], source_shape[1], image_shape[2], image_shape[1]))
    if int(source_shape[0]) not in (1, int(image_shape[0])):
        return ("detail source has %d frames, the image %d -- give 1 or %d"
                % (source_shape[0], image_shape[0], image_shape[0]))
    return None


def _preview_size(w, h, max_edge=PREVIEW_MAX_EDGE):
    """Downscale-only fit of (w, h) so the long edge is at most max_edge.
    Never upscales. Pure math -- guard-driven."""
    w = int(w)
    h = int(h)
    long_edge = w if w >= h else h
    if long_edge <= max_edge:
        return w, h
    scale = float(max_edge) / float(long_edge)
    pw = int(round(w * scale))
    ph = int(round(h * scale))
    return (pw if pw > 0 else 1), (ph if ph > 0 else 1)


def _grade_np(arr, exposure, temperature, tint, contrast, gamma,
              shadows, highlights, saturation, vibrance, hue_shift):
    """The color pipeline -- the single ground truth of this node.

    arr: float32 ndarray [..., 3] (RGB, nominally 0..1), returned as a new
    float32 array. web/js/ph_filter.js mirrors this function OP FOR OP in
    _gradeRGB for the live preview; the parity guard drives BOTH on the same
    input and pins the agreement, so any edit here must land there too.

    Op sequence (fixed; documented once, here):
      1  exposure     x *= 2^ev
      2  temperature  r *= 1 + 0.25t          b *= 1 - 0.25t
      3  tint         g *= 1 - 0.25t
      4  contrast     x  = 0.5 + (x - 0.5)(1 + c)
      5  gamma        x  = max(x, 0)^gamma
      6  shadows/highlights  additive lifts through soft Rec.709 luma masks
                      x += s * 0.25 * (1 - luma)^2
                      x += h * 0.25 * luma^2        (same luma snapshot)
      7  saturation   x  = luma + (x - luma)(1 + s)
      8  vibrance     saturation weighted by (1 - colorfulness):
                      x  = luma + (x - luma)(1 + v * (1 - clamp01(max-min)))
      9  hue rotate   SVG hue-rotate matrix (0.213 / 0.715 / 0.072 weights),
                      skipped exactly when hue_shift == 0 (mirrored in JS)
     10  clamp to 0..1
    """
    import math

    x = np.asarray(arr, dtype=np.float32).copy()

    x *= np.float32(2.0 ** float(exposure))

    x[..., 0] *= np.float32(1.0 + 0.25 * float(temperature))
    x[..., 2] *= np.float32(1.0 - 0.25 * float(temperature))
    x[..., 1] *= np.float32(1.0 - 0.25 * float(tint))

    x = np.float32(0.5) + (x - np.float32(0.5)) * np.float32(1.0 + float(contrast))

    x = np.maximum(x, np.float32(0.0)) ** np.float32(float(gamma))

    luma = (np.float32(0.2126) * x[..., 0] + np.float32(0.7152) * x[..., 1]
            + np.float32(0.0722) * x[..., 2])[..., None]
    x = x + np.float32(float(shadows) * 0.25) * (np.float32(1.0) - luma) ** 2
    x = x + np.float32(float(highlights) * 0.25) * luma ** 2

    luma = (np.float32(0.2126) * x[..., 0] + np.float32(0.7152) * x[..., 1]
            + np.float32(0.0722) * x[..., 2])[..., None]
    x = luma + (x - luma) * np.float32(1.0 + float(saturation))

    luma = (np.float32(0.2126) * x[..., 0] + np.float32(0.7152) * x[..., 1]
            + np.float32(0.0722) * x[..., 2])[..., None]
    rng = np.clip(x.max(axis=-1) - x.min(axis=-1), 0.0, 1.0)[..., None]
    x = luma + (x - luma) * (np.float32(1.0)
                             + np.float32(float(vibrance)) * (np.float32(1.0) - rng))

    if float(hue_shift) != 0.0:
        a = math.radians(float(hue_shift))
        c, s = math.cos(a), math.sin(a)
        m = np.array([
            [0.213 + 0.787 * c - 0.213 * s, 0.715 - 0.715 * c - 0.715 * s, 0.072 - 0.072 * c + 0.928 * s],
            [0.213 - 0.213 * c + 0.143 * s, 0.715 + 0.285 * c + 0.140 * s, 0.072 - 0.072 * c - 0.283 * s],
            [0.213 - 0.213 * c - 0.787 * s, 0.715 - 0.715 * c + 0.715 * s, 0.072 + 0.928 * c + 0.072 * s],
        ], dtype=np.float32)
        x = x @ m.T

    return np.clip(x, 0.0, 1.0).astype(np.float32)


def _is_neutral(exposure, temperature, tint, contrast, gamma,
                shadows, highlights, saturation, vibrance, hue_shift):
    """True when every color control sits at its no-op value -- the fast
    path skips the numpy roundtrip entirely and returns the input tensor."""
    return (float(exposure) == 0.0 and float(temperature) == 0.0
            and float(tint) == 0.0 and float(contrast) == 0.0
            and float(gamma) == 1.0 and float(shadows) == 0.0
            and float(highlights) == 0.0 and float(saturation) == 0.0
            and float(vibrance) == 0.0 and float(hue_shift) == 0.0)


def _parse_cube(text):
    """Parse an Iridas/Adobe .cube 3D LUT. Returns (size, data, dmin, dmax):
    data is float32 [size, size, size, 3] indexed [b, g, r] (the file's line
    order runs r fastest, then g, then b), dmin/dmax are float32 [3].
    Mirrored op for op by _parseCube in web/js/ph_filter.js -- the parity
    guard drives both on the same fixture. Raises ValueError on malformed
    input."""
    size = 0
    dmin = np.zeros(3, dtype=np.float32)
    dmax = np.ones(3, dtype=np.float32)
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        up = line.upper()
        if up.startswith("TITLE"):
            continue
        if up.startswith("LUT_3D_SIZE"):
            size = int(line.split()[1])
            continue
        if up.startswith("LUT_1D_SIZE"):
            raise ValueError("1D LUTs are not supported (need LUT_3D_SIZE)")
        if up.startswith("DOMAIN_MIN"):
            dmin = np.asarray([float(v) for v in line.split()[1:4]], dtype=np.float32)
            continue
        if up.startswith("DOMAIN_MAX"):
            dmax = np.asarray([float(v) for v in line.split()[1:4]], dtype=np.float32)
            continue
        parts = line.split()
        if len(parts) == 3:
            rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if size < 2:
        raise ValueError("missing or invalid LUT_3D_SIZE")
    if len(rows) != size ** 3:
        raise ValueError("expected %d entries, got %d" % (size ** 3, len(rows)))
    data = np.asarray(rows, dtype=np.float32).reshape(size, size, size, 3)
    return size, data, dmin, dmax


def _apply_lut_np(x, size, data, dmin, dmax, strength):
    """Trilinear 3D-LUT application, blended with the input by strength
    (0 = input untouched, 1 = full LUT). x: float32 [..., 3] nominally 0..1.
    Mirrored op for op by _lutRGB in web/js/ph_filter.js."""
    x = np.asarray(x, dtype=np.float32)
    span = np.maximum(dmax - dmin, np.float32(1e-6))
    c = np.clip((x - dmin) / span, 0.0, 1.0) * np.float32(size - 1)
    i0 = np.floor(c).astype(np.int64)
    i1 = np.minimum(i0 + 1, size - 1)
    f = (c - i0).astype(np.float32)
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    c000 = data[b0, g0, r0]
    c001 = data[b0, g0, r1]
    c010 = data[b0, g1, r0]
    c011 = data[b0, g1, r1]
    c100 = data[b1, g0, r0]
    c101 = data[b1, g0, r1]
    c110 = data[b1, g1, r0]
    c111 = data[b1, g1, r1]
    lo = (1 - fg) * ((1 - fr) * c000 + fr * c001) + fg * ((1 - fr) * c010 + fr * c011)
    hi = (1 - fg) * ((1 - fr) * c100 + fr * c101) + fg * ((1 - fr) * c110 + fr * c111)
    lut_out = (1 - fb) * lo + fb * hi
    s = np.float32(strength)
    return np.clip(x * (1 - s) + lut_out * s, 0.0, 1.0).astype(np.float32)


# Keys a preset may carry: the full look (color + LUT + sharpen), never the
# preset selector itself. The route sanitizes every load/save through
# _sanitize_preset before anything touches disk or widgets.
PRESET_KEYS = tuple(k for k in FILTER_CANON if k != "preset")


def _sanitize_preset(params):
    """Whitelist + coerce a preset parameter dict: only PRESET_KEYS survive,
    numeric controls are coerced to float (non-coercible values are DROPPED,
    not guessed), lut_name is basename-reduced to a plain string. Pure --
    guard-driven; both the save and the load path run through this, so a
    hand-edited or foreign preset file can neither smuggle keys nor paths."""
    out = {}
    if not isinstance(params, dict):
        return out
    for k, v in params.items():
        if k not in PRESET_KEYS:
            continue
        if k == "lut_name":
            out[k] = os.path.basename(str(v))
            continue
        if k == "auto_mode":
            # a string choice: only a known stop survives (mirrors AUTO_MODES)
            if str(v) in ("off", "neutral", "high", "ultra"):
                out[k] = str(v)
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


_LUT_CACHE = {}  # (path, mtime) -> (size, data, dmin, dmax)


def _gauss_kernel(radius):
    """Normalized 1D gaussian weights; sigma equals the radius, half-width
    ceil(3 sigma). Mirrored op for op by _gaussKernel in ph_filter.js."""
    import math
    sigma = max(float(radius), 0.1)
    half = max(1, int(math.ceil(3 * sigma)))
    idx = np.arange(-half, half + 1, dtype=np.float64)
    w = np.exp(-(idx * idx) / (2.0 * sigma * sigma))
    return half, (w / w.sum()).astype(np.float32)


def _sharpen_np(x, amount, radius, threshold=0.0):
    """Unsharp mask: out = x + amount * (x - gaussian_blur(x)), separable
    blur with replicate (edge-clamp) borders, clamped to 0..1.
    x: float32 [..., H, W, 3]. Mirrored op for op by _sharpenBuf in
    ph_filter.js -- the parity guard drives both on the same image."""
    if not (float(amount) > 0.0):
        return x
    x = np.asarray(x, dtype=np.float32)
    half, w = _gauss_kernel(radius)

    def _blur_axis(a, axis):
        pad = [(0, 0)] * a.ndim
        pad[axis] = (half, half)
        p = np.pad(a, pad, mode="edge")
        out = np.zeros(a.shape, dtype=np.float32)
        for i in range(2 * half + 1):
            sl = [slice(None)] * a.ndim
            sl[axis] = slice(i, i + a.shape[axis])
            out += w[i] * p[tuple(sl)]
        return out

    blur = _blur_axis(_blur_axis(x, x.ndim - 3), x.ndim - 2)
    d = x - blur
    t = float(threshold)
    if t > 0.0:
        # v1000 (F3) soft coring: |d| <= t adds nothing, a larger |d| is
        # shortened by t. Measured 24.09.2026 on 14 LIVE1 pictures (soft +
        # noise): at the SAME sharpness 1-3 levels give +0.3..0.4 dB and up
        # to 12 % less noise in flat areas than the plain mask.
        d = np.sign(d) * np.maximum(np.abs(d) - np.float32(t), np.float32(0.0))
    return np.clip(x + np.float32(float(amount)) * d, 0.0, 1.0).astype(np.float32)


def _load_lut(name):
    """Parsed LUT from luts/ with an mtime-keyed cache; None when the name is
    'none', missing or malformed (the caller degrades honestly)."""
    if not name or name == "none":
        return None
    path = os.path.join(LUT_DIR, os.path.basename(name))
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        print("[PLS] Polyhedron Filter: LUT not found: %s" % name)
        return None
    if key not in _LUT_CACHE:
        try:
            _LUT_CACHE.clear()  # one LUT at a time is the working set
            _LUT_CACHE[key] = _parse_cube(open(path, encoding="utf-8", errors="replace").read())
        except (OSError, ValueError) as e:
            print("[PLS] Polyhedron Filter: LUT unusable (%s): %s" % (name, e))
            return None
    return _LUT_CACHE[key]


def _list_files(folder, ext):
    """Sorted file names with the given extension inside folder; [] if the
    folder is missing or unreadable. Pure listing, no side effects."""
    try:
        names = [n for n in os.listdir(folder) if n.lower().endswith(ext)]
    except OSError:
        return []
    return sorted(names)


class ULSFilter:
    DESCRIPTION = (
        "Every numeric field doubles as a slider: click-drag it horizontally "
        "to scrub the value and watch the preview react live, or click once "
        "to type an exact number. "
        "One-node color grading with an in-node before/after preview. "
        "Pick white (above the preview) sets temperature and tint from a "
        "click on something that should be neutral. auto_mode corrects "
        "first, once per batch; the sliders work on top of it. Connect "
        "detail_source (the picture before an edit or re-render, same size) "
        "to carry its lost contours back -- before everything else. "
        "The color controls act on the IMAGE output and move the preview "
        "live; the divider compares the original (left) against the graded "
        "result (right). A .cube LUT from the pack's luts/ folder is applied "
        "after the color controls, the unsharp mask last. Presets load from and "
        "save to the pack's presets/ folder and set the sliders directly."
    )

    @classmethod
    def INPUT_TYPES(cls):
        luts = ["none"] + _list_files(LUT_DIR, ".cube")
        presets = ["none"] + _list_files(PRESET_DIR, ".json")
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.05,
                             "tooltip": "Exposure in EV stops. 0 leaves brightness unchanged; +1 doubles light, -1 halves it. First step of the grading pipeline. Click-drag to scrub live."}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                                "tooltip": "White balance temperature. Negative shifts toward blue (cooler), positive toward orange (warmer). Set by Pick white in the preview. Click-drag to scrub live."}),
                "tint": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01,
                         "tooltip": "White balance tint. Negative shifts toward green, positive toward magenta. Set by Pick white in the preview. Click-drag to scrub live."}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                             "tooltip": "Contrast around mid gray. Negative flattens, positive steepens. Click-drag to scrub live."}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.01,
                          "tooltip": "Midtone gamma. Values below 1 brighten midtones, above 1 darken them. 1 is neutral. Click-drag to scrub live."}),
                "shadows": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                            "tooltip": "Lifts (positive) or deepens (negative) the darkest tonal range through a soft luma mask. Click-drag to scrub live."}),
                "highlights": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                               "tooltip": "Recovers (negative) or pushes (positive) the brightest tonal range through a soft luma mask. Click-drag to scrub live."}),
                "saturation": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                               "tooltip": "Uniform colour saturation. -1 is grayscale, 0 unchanged, +1 strongly saturated. Click-drag to scrub live."}),
                "vibrance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                             "tooltip": "Saturation weighted toward muted colours: boosts dull areas more than already-vivid ones. Click-drag to scrub live."}),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0,
                              "tooltip": "Rotates all hues around the colour wheel by the given degrees. 0 is neutral. Click-drag to scrub live."}),
                "lut_name": (luts, {"default": "none",
                             "tooltip": "3D LUT (.cube) applied after the colour controls. Files are read from this pack's luts/ folder; 'none' skips the LUT."}),
                "lut_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                 "tooltip": "Blend between the ungraded (0) and fully LUT-graded (1) image. Only used when a LUT is selected. Click-drag to scrub live."}),
                "sharpen_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05,
                                   "tooltip": "Unsharp-mask strength applied as the last pipeline step. 0 disables sharpening. Click-drag to scrub live."}),
                "sharpen_radius": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1,
                                   "tooltip": "Unsharp-mask blur radius in pixels. Larger values sharpen coarser detail. Click-drag to scrub live."}),
                "preset": (presets, {"default": "none",
                           "tooltip": "Named parameter set from this pack's presets/ folder. Loading a preset sets the sliders; they stay freely adjustable afterwards."}),
                "auto_mode": (AUTO_CHOICES, {"default": "off",
                              "tooltip": "Automatic correction, measured once per run over the whole batch (no flicker on video) and applied FIRST, before every slider: grey-world white balance, brightness toward mid grey, a flat picture stretched, a dull one saturated. neutral / high / ultra take a growing share. The sliders shape the result by hand on top."}),
                "sharpen_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 32.0, "step": 1.0,
                                      "tooltip": "Unsharp-mask threshold in 8-bit levels, as in Photoshop. Differences smaller than this are not sharpened (grain and noise stay calm), larger ones are sharpened minus it. 1-3 levels with a higher amount (2-3) sharpen as much as the plain mask with less noise. 0 = off. Click-drag to scrub live."}),
                "detail_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                  "tooltip": "Detail from the original, used only when detail_source is connected: contours the processing lost and the original still has come back (noise, ringing and compression blocks stay out). 1 = as measured best, 2 = double. Click-drag to scrub live."}),
            },
            "optional": {
                "detail_source": ("IMAGE", {"tooltip": "The picture BEFORE the processing that lost detail (the original of an edit, restore or re-render at the SAME size). Its lost contours are carried back onto image, first in the pipeline. One frame for all, or one per frame."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Polyhedron/Image"
    OUTPUT_NODE = False

    def apply(self, image, exposure, temperature, tint, contrast, gamma,
              shadows, highlights, saturation, vibrance, hue_shift,
              lut_name, lut_strength, sharpen_amount, sharpen_radius, preset,
              auto_mode="off", sharpen_threshold=0.0, detail_amount=1.0,
              detail_source=None):
        # Color pipeline + LUT + sharpen stages are ACTIVE (ground truth:
        # _grade_np -> _apply_lut_np -> _sharpen_np). The preset widget is
        # deliberately NOT applied here: presets write widget values in the
        # frontend; honoring the selector here too would double-apply.
        # The preview always carries the UNGRADED source frame: the frontend
        # grades it live with the mirrored JS pipeline.
        stats = None
        try:
            stats = _auto_stats(image.detach().cpu().numpy() if hasattr(image, "detach") else image)
        except Exception as e:  # the automatic must never kill the run
            print("[PLS] Polyhedron Filter: auto analysis failed: %r" % (e,))
        auto = _auto_grade(stats, auto_mode)
        item = self._make_preview(image)
        item["auto"] = stats   # the preview mirrors every stop live from these

        # detail from the original (F4): checked, measured, previewed
        dsrc, grid = None, 0.0
        if detail_source is not None:
            try:
                dnp = (detail_source.detach().cpu().numpy() if hasattr(detail_source, "detach")
                       else np.asarray(detail_source, dtype=np.float32))
                ishape = tuple(image.shape)
                why = _detail_fit(ishape, dnp.shape)
                if why is None:
                    grid = _grid_strength(_block_ratio(dnp))
                    if float(detail_amount) > 0.0:
                        dsrc = dnp
                    item["detail"] = self._make_preview(detail_source)
                    item["detail_grid"] = grid
                    print("[PLS] Polyhedron Filter: detail from source %.2f, block grid %.2f"
                          % (float(detail_amount), grid))
                else:
                    item["detail_note"] = why
                    print("[PLS] Polyhedron Filter: detail skipped -- " + why)
            except Exception as e:  # the detail must never kill the run
                item["detail_note"] = "detail source unreadable"
                print("[PLS] Polyhedron Filter: detail skipped: %r" % (e,))
        ui = {"ph_filter": [item]}
        if auto is not None:
            print("[PLS] Polyhedron Filter: auto %s -> wb_r %.3f wb_b %.3f brightness %+.3f "
                  "contrast %.3f saturation %.3f" % (auto_mode, auto["wb_r"], auto["wb_b"],
                                                     auto["brightness"], auto["contrast"],
                                                     auto["saturation"]))

        lut = None
        if lut_name != "none" and float(lut_strength) > 0.0:
            lut = _load_lut(lut_name)  # None on failure -> honest console note

        if (dsrc is None and auto is None and lut is None and float(sharpen_amount) == 0.0
                and _is_neutral(exposure, temperature, tint, contrast, gamma,
                                shadows, highlights, saturation, vibrance, hue_shift)):
            return {"ui": ui, "result": (image,)}

        try:
            import torch
        except Exception:
            torch = None

        params = (exposure, temperature, tint, contrast, gamma,
                  shadows, highlights, saturation, vibrance, hue_shift)

        def _process(chunk, dchunk=None):
            if dchunk is not None:
                chunk = _detail_np(dchunk, chunk, float(detail_amount), grid)
            out = _grade_np(_apply_auto_np(chunk, auto) if auto is not None else chunk, *params)
            if lut is not None:
                size, data, dmin, dmax = lut
                out = _apply_lut_np(out, size, data, dmin, dmax, float(lut_strength))
            if float(sharpen_amount) > 0.0:
                out = _sharpen_np(out, float(sharpen_amount), float(sharpen_radius),
                                  float(sharpen_threshold) / 255.0)
            return out

        if torch is not None and hasattr(image, "cpu"):
            src = image.detach().cpu().numpy()
            out = np.empty_like(src, dtype=np.float32)
            step = 32  # frames per chunk: elementwise ops, bounded temporaries
            # v1010: the green bar only -- a grade is quick; the bar just shows
            # a long clip is moving (quiet: no console lines).
            with _NodeProgress("Filter", "filter", total=int(src.shape[0]), unit="frame",
                               quiet=True) as _prog:
                for i in range(0, src.shape[0], step):
                    dch = None
                    if dsrc is not None:
                        dch = dsrc[0:1] if dsrc.shape[0] == 1 else dsrc[i:i + step]
                    out[i:i + step] = _process(src[i:i + step], dch)
                    _prog.tick(min(step, int(src.shape[0]) - i))
            return {"ui": ui, "result": (torch.from_numpy(out).to(image.device, dtype=image.dtype),)}

        arr = np.asarray(image, dtype=np.float32)
        out = _process(arr, None if dsrc is None else np.asarray(dsrc, dtype=np.float32))
        return {"ui": ui, "result": (out,)}

    # ------------------------------------------------------------------
    # preview
    # ------------------------------------------------------------------
    def _make_preview(self, image):
        """Write a downscaled PNG of the first frame into ComfyUI's temp dir
        and return the ui entry the frontend loads via /view. Degrades
        honestly: on any failure the entry carries an 'error' note and no
        filename -- the run itself is never aborted by the preview."""
        try:
            import folder_paths
            from PIL import Image as PILImage

            frame = image[0]
            arr = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
            arr = np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8)
            h, w = int(arr.shape[0]), int(arr.shape[1])
            pw, ph = _preview_size(w, h)
            img = PILImage.fromarray(arr)
            if (pw, ph) != (w, h):
                img = img.resize((pw, ph), PILImage.LANCZOS)
            fname = "ph_filter_%s.png" % uuid.uuid4().hex[:12]
            img.save(os.path.join(folder_paths.get_temp_directory(), fname))
            return {"filename": fname, "subfolder": "", "type": "temp",
                    "width": pw, "height": ph, "src_width": w, "src_height": h}
        except Exception as e:  # preview must never kill the run
            print("[PLS] Polyhedron Filter: preview generation failed: %r" % (e,))
            return {"error": str(e)}


NODE_CLASS_MAPPINGS = {"ULSFilter": ULSFilter}
NODE_DISPLAY_NAME_MAPPINGS = {"ULSFilter": "\u2b21 Polyhedron Filter"}
