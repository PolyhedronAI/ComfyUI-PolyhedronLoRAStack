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
)

# Long edge of the in-node preview image (px). The preview is a downscaled
# proxy the frontend grades live; the run output is always full resolution.
PREVIEW_MAX_EDGE = 768

_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LUT_DIR = os.path.join(_PACK_ROOT, "luts")
PRESET_DIR = os.path.join(_PACK_ROOT, "presets")


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


def _sharpen_np(x, amount, radius):
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
    return np.clip(x + np.float32(float(amount)) * (x - blur), 0.0, 1.0).astype(np.float32)


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
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                                "tooltip": "White balance temperature. Negative shifts toward blue (cooler), positive toward orange (warmer). Click-drag to scrub live."}),
                "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01,
                         "tooltip": "White balance tint. Negative shifts toward green, positive toward magenta. Click-drag to scrub live."}),
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
                "sharpen_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.05,
                                   "tooltip": "Unsharp-mask strength applied as the last pipeline step. 0 disables sharpening. Click-drag to scrub live."}),
                "sharpen_radius": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.1,
                                   "tooltip": "Unsharp-mask blur radius in pixels. Larger values sharpen coarser detail. Click-drag to scrub live."}),
                "preset": (presets, {"default": "none",
                           "tooltip": "Named parameter set from this pack's presets/ folder. Loading a preset sets the sliders; they stay freely adjustable afterwards."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Polyhedron/Image"
    OUTPUT_NODE = False

    def apply(self, image, exposure, temperature, tint, contrast, gamma,
              shadows, highlights, saturation, vibrance, hue_shift,
              lut_name, lut_strength, sharpen_amount, sharpen_radius, preset):
        # Color pipeline + LUT + sharpen stages are ACTIVE (ground truth:
        # _grade_np -> _apply_lut_np -> _sharpen_np). The preset widget is
        # deliberately NOT applied here: presets write widget values in the
        # frontend; honoring the selector here too would double-apply.
        # The preview always carries the UNGRADED source frame: the frontend
        # grades it live with the mirrored JS pipeline.
        ui = {"ph_filter": [self._make_preview(image)]}

        lut = None
        if lut_name != "none" and float(lut_strength) > 0.0:
            lut = _load_lut(lut_name)  # None on failure -> honest console note

        if (lut is None and float(sharpen_amount) == 0.0
                and _is_neutral(exposure, temperature, tint, contrast, gamma,
                                shadows, highlights, saturation, vibrance, hue_shift)):
            return {"ui": ui, "result": (image,)}

        try:
            import torch
        except Exception:
            torch = None

        params = (exposure, temperature, tint, contrast, gamma,
                  shadows, highlights, saturation, vibrance, hue_shift)

        def _process(chunk):
            out = _grade_np(chunk, *params)
            if lut is not None:
                size, data, dmin, dmax = lut
                out = _apply_lut_np(out, size, data, dmin, dmax, float(lut_strength))
            if float(sharpen_amount) > 0.0:
                out = _sharpen_np(out, float(sharpen_amount), float(sharpen_radius))
            return out

        if torch is not None and hasattr(image, "cpu"):
            src = image.detach().cpu().numpy()
            out = np.empty_like(src, dtype=np.float32)
            step = 32  # frames per chunk: elementwise ops, bounded temporaries
            for i in range(0, src.shape[0], step):
                out[i:i + step] = _process(src[i:i + step])
            return {"ui": ui, "result": (torch.from_numpy(out).to(image.device, dtype=image.dtype),)}

        out = _process(np.asarray(image, dtype=np.float32))
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
