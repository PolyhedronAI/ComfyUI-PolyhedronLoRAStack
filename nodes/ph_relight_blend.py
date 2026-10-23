"""Polyhedron Relight Blend maths (public build: the blend maths only -- Polyhedron Blend stacks its layers with
them; the Relight Blend node itself is not part of this release) -- composite a layer (typically a Light Map) onto a
base image with the FULL classic Photoshop blend-mode set plus a physical
`relight_gain` mode built for the Light Map pairing.

Why an own node: the stock blend offers a handful of modes, and Multiply can
only ever DARKEN -- it lays shadow on, but can never ADD light (no lit key
side, no rim glow, no coloured spill). The classic answer is the 50%-grey
dodge&burn layer under Soft Light; the precise answer for a light map is a
gain: out = base * (layer / 0.5) ** strength -- neutral grey is identity,
bright lightens, dark darkens, and colour tints multiply through. Both live
here, next to every mode Photoshop ships.

Modes (Adobe PDF / W3C compositing spec -- the maths Photoshop implements;
the non-separable HSL family included):
  relight_gain | normal | dissolve
  darken | multiply | color_burn | linear_burn | darker_color
  lighten | screen | color_dodge | linear_dodge_add | lighter_color
  overlay | soft_light | hard_light | vivid_light | linear_light | pin_light |
  hard_mix
  difference | exclusion | subtract | divide
  hue | saturation | color | luminosity

Details that match Photoshop behaviour:
  * soft_light uses the PDF D(b) ramp (16b-12)b+4)b / sqrt(b) formula.
  * color_burn / color_dodge carry the s==0 / s==1 edge conventions.
  * hard_mix thresholds per channel at b + s >= 1.
  * hue / saturation / color / luminosity use Lum = .3r+.59g+.11b with
    ClipColor / SetLum / SetSat exactly as in the spec.
  * dissolve is the classic stippled coverage: opacity = coverage, driven by a
    DETERMINISTIC hash noise (same inputs -> same stipple; guard-testable).

`opacity` mixes the blended result over the base (dissolve consumes it as
coverage); an optional MASK limits the blend region (multiplied into the
opacity per pixel). `strength` only shapes relight_gain's exponent.

Batch friendly: a [1,H,W,3] layer broadcasts over an [N,H,W,3] base (one light
situation over a whole animation); a size-mismatched layer is bilinearly
resized to the base at the torch edge. Pure numpy core (`blend_images`) + lazy
torch, mirroring the pack's other headless nodes -- deterministic, numeric
guards (Messen schlaegt Glauben).
"""

import numpy as np

_EPS = 1e-6

BLEND_MODES = [
    "relight_gain", "normal", "dissolve",
    "darken", "multiply", "color_burn", "linear_burn", "darker_color",
    "lighten", "screen", "color_dodge", "linear_dodge_add", "lighter_color",
    "overlay", "soft_light", "hard_light", "vivid_light", "linear_light",
    "pin_light", "hard_mix",
    "difference", "exclusion", "subtract", "divide",
    "hue", "saturation", "color", "luminosity",
]

# ── separable helpers ─────────────────────────────────────────────────────────

def _color_burn(b, s):
    out = 1.0 - np.minimum(1.0, (1.0 - b) / np.maximum(s, _EPS))
    out = np.where(s <= _EPS, np.where(b >= 1.0 - _EPS, 1.0, 0.0), out)
    return out


def _color_dodge(b, s):
    out = np.minimum(1.0, b / np.maximum(1.0 - s, _EPS))
    out = np.where(s >= 1.0 - _EPS, np.where(b <= _EPS, 0.0, 1.0), out)
    return out


def _soft_light(b, s):
    # PDF spec: the formula behind Photoshop's Soft Light.
    d = np.where(b <= 0.25, ((16.0 * b - 12.0) * b + 4.0) * b, np.sqrt(np.maximum(b, 0.0)))
    return np.where(s <= 0.5,
                    b - (1.0 - 2.0 * s) * b * (1.0 - b),
                    b + (2.0 * s - 1.0) * (d - b))


def _hard_light(b, s):
    return np.where(s <= 0.5, 2.0 * b * s, 1.0 - 2.0 * (1.0 - b) * (1.0 - s))


def _vivid_light(b, s):
    return np.where(s <= 0.5, _color_burn(b, 2.0 * s), _color_dodge(b, 2.0 * (s - 0.5)))


# ── non-separable HSL family (PDF spec: Lum / ClipColor / SetLum / SetSat) ────

def _lum(c):
    return 0.3 * c[..., 0] + 0.59 * c[..., 1] + 0.11 * c[..., 2]


def _clip_color(c):
    l = _lum(c)[..., None]
    mn = c.min(axis=-1, keepdims=True)
    mx = c.max(axis=-1, keepdims=True)
    low = l + (c - l) * l / np.maximum(l - mn, _EPS)
    c = np.where(mn < 0.0, low, c)
    mx = c.max(axis=-1, keepdims=True)
    high = l + (c - l) * (1.0 - l) / np.maximum(mx - l, _EPS)
    c = np.where(mx > 1.0, high, c)
    return c


def _set_lum(c, l):
    return _clip_color(c + (l - _lum(c))[..., None])


def _sat(c):
    return c.max(axis=-1) - c.min(axis=-1)


def _set_sat(c, s):
    mn = c.min(axis=-1, keepdims=True)
    rng = c.max(axis=-1, keepdims=True) - mn
    scaled = (c - mn) / np.maximum(rng, _EPS) * s[..., None]
    return np.where(rng > _EPS, scaled, 0.0)


# ── deterministic stipple for dissolve ────────────────────────────────────────

def _hash_noise(h, w):
    """Per-pixel uniform noise in [0,1), fully determined by the pixel index
    (integer hash -- no RNG state, same size -> same stipple)."""
    yy, xx = np.meshgrid(np.arange(h, dtype=np.uint32),
                         np.arange(w, dtype=np.uint32), indexing="ij")
    n = xx * np.uint32(1973) + yy * np.uint32(9277) + np.uint32(26699)
    n = (n << np.uint32(13)) ^ n
    n = n * (n * n * np.uint32(15731) + np.uint32(789221)) + np.uint32(1376312589)
    return (n % np.uint32(65536)).astype(np.float32) / 65536.0


# ── the core ──────────────────────────────────────────────────────────────────

def blend_images(base, layer, mode, opacity=1.0, strength=1.0, mask=None):
    """Pure-numpy core. base/layer float32 [..., 3] in 0..1, same shape.
    Returns float32 in 0..1. `mask` (optional) is [..., 1]-broadcastable."""
    b = np.clip(np.asarray(base, dtype=np.float32), 0.0, 1.0)
    s = np.clip(np.asarray(layer, dtype=np.float32), 0.0, 1.0)
    m = str(mode)
    op = float(min(max(opacity, 0.0), 1.0))

    if m == "relight_gain":
        # Light as GAIN, not as a layer trick: neutral grey 0.5 = identity,
        # bright lightens, dark darkens, colours tint multiplicatively.
        st = float(min(max(strength, 0.0), 4.0))
        out = b * np.power(np.maximum(s, _EPS) / 0.5, st, dtype=np.float32)
    elif m == "normal":
        out = s
    elif m == "dissolve":
        cov = _hash_noise(b.shape[-3], b.shape[-2])[..., None] < op
        out = np.where(cov, s, b)
        op = 1.0                          # coverage consumed the opacity
    elif m == "darken":
        out = np.minimum(b, s)
    elif m == "multiply":
        out = b * s
    elif m == "color_burn":
        out = _color_burn(b, s)
    elif m == "linear_burn":
        out = b + s - 1.0
    elif m == "darker_color":
        out = np.where((_lum(s) < _lum(b))[..., None], s, b)
    elif m == "lighten":
        out = np.maximum(b, s)
    elif m == "screen":
        out = 1.0 - (1.0 - b) * (1.0 - s)
    elif m == "color_dodge":
        out = _color_dodge(b, s)
    elif m == "linear_dodge_add":
        out = b + s
    elif m == "lighter_color":
        out = np.where((_lum(s) > _lum(b))[..., None], s, b)
    elif m == "overlay":
        out = _hard_light(s, b)           # overlay(b, s) == hard_light(s, b)
    elif m == "soft_light":
        out = _soft_light(b, s)
    elif m == "hard_light":
        out = _hard_light(b, s)
    elif m == "vivid_light":
        out = _vivid_light(b, s)
    elif m == "linear_light":
        out = b + 2.0 * s - 1.0
    elif m == "pin_light":
        out = np.where(s <= 0.5, np.minimum(b, 2.0 * s), np.maximum(b, 2.0 * s - 1.0))
    elif m == "hard_mix":
        out = np.where(b + s >= 1.0, 1.0, 0.0)
    elif m == "difference":
        out = np.abs(b - s)
    elif m == "exclusion":
        out = b + s - 2.0 * b * s
    elif m == "subtract":
        out = b - s
    elif m == "divide":
        out = b / np.maximum(s, _EPS)
    elif m == "hue":
        out = _set_lum(_set_sat(s, _sat(b)), _lum(b))
    elif m == "saturation":
        out = _set_lum(_set_sat(b, _sat(s)), _lum(b))
    elif m == "color":
        out = _set_lum(s, _lum(b))
    elif m == "luminosity":
        out = _set_lum(b, _lum(s))
    else:
        out = s                           # unknown -> normal (defensive)

    out = np.clip(out, 0.0, 1.0)
    w = op
    if mask is not None:
        w = w * np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    return np.clip(b + (out - b) * w, 0.0, 1.0).astype(np.float32)
