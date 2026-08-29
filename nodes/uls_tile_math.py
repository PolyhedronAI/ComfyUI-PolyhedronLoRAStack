"""
uls_tile_math.py -- pure tile geometry for the Power Upscale node (v514).

The seam-free heart of ⬡ Polyhedron Power Upscale, kept free of torch /
ComfyUI so guard tests execute the real functions on frozen vectors
(Messen schlaegt Glauben). Design (the clean-room replacement for the
Ultimate-SD-Upscale seam_fix family):

  Tiles OVERLAP by a target amount and are stitched as a WEIGHTED SUM
  with separable smoothstep feather ramps. Two facing ramps live in the
  SAME overlap interval and are exact complements (s(t) + (1 - s(t)) = 1),
  and corner zones multiply out separably ((sum_x w)(sum_y w) = 1), so the
  accumulated weight is EXACTLY 1 everywhere -- no seam pass needed, ever.
  The guard proves it numerically instead of believing it.

Conventions:
  * axis positions are pixel offsets; a tile on an axis is [pos, pos+tile)
  * only DIRECT neighbours may overlap (invariant, guarded)
  * ramp widths equal the ACTUAL overlap with that neighbour, so both
    tiles sample the very same interval at the very same phase
"""

import math

import numpy as np

# Widths and heights the diffusion path touches snap to this grid (VAE /8).
GRID = 8

# splitmix64 constants -- the per-tile seed derivation. Deliberately a pure
# function of (seed, tile index) and NEVER of the frame number: every frame
# re-uses the same noise structure per tile, which keeps video batches from
# flickering tile-wise.
_SM_GAMMA = 0x9E3779B97F4A7C15
_SM_M1 = 0xBF58476D1CE4E5B9
_SM_M2 = 0x94D049BB133111EB
_MASK64 = 0xFFFFFFFFFFFFFFFF


def scaled_size(width, height, factor):
    """Target canvas for one stage: (round(w*f), round(h*f)) snapped to the
    VAE grid, never below one grid cell."""
    w = max(GRID, int(round(float(width) * float(factor) / GRID)) * GRID)
    h = max(GRID, int(round(float(height) * float(factor) / GRID)) * GRID)
    return w, h


def plan_axis(size, tile, overlap):
    """Tile positions along one axis.

    Returns (tile_len, [(pos, ramp_l, ramp_r), ...]) where tile_len is the
    (possibly clamped) tile length actually used on this axis and each ramp
    is the exact overlap shared with that neighbour (0 at the canvas edge).

    Construction: integer stride distribution. strides are span//(n-1) or
    span//(n-1)+1 (the remainder spread over the first gaps), with n capped
    at n_safe = span // ceil(tile/2) + 1 so every stride >= ceil(tile/2).
    That makes the two safety invariants STRUCTURAL, not statistical:
      * ramp_l + ramp_r <= tile_len for every tile  (ramps never collide)
      * pos[i+2] >= pos[i] + tile_len               (only neighbours overlap)
    When the requested overlap is geometrically unreachable (e.g. 760px at
    tile 384: three tiles would over-overlap), n drops and the feather is
    simply narrower than requested -- correct beats wide. The n == 2 case is
    always safe (a single pair; overlap 2*tile - size < tile). Remaining
    invariants (proved by test_v514, not assumed): coverage pos[0] == 0 and
    pos[-1] + tile_len == size; accumulated feather weight == 1 everywhere.
    """
    size = int(size)
    tile = int(tile)
    overlap = max(0, min(int(overlap), tile - 1))
    if tile >= size:
        return size, [(0, 0, 0)]
    span = size - tile
    stride_goal = max(1, tile - overlap)
    n1 = int(math.ceil(span / float(stride_goal))) + 1
    half = (tile + 1) // 2
    n_safe = span // half + 1
    n = max(2, min(n1, n_safe))
    base = span // (n - 1)
    extra = span - base * (n - 1)
    strides = [base + 1] * extra + [base] * (n - 1 - extra)
    pos = [0]
    for s in strides:
        pos.append(pos[-1] + s)
    out = []
    for i, p in enumerate(pos):
        ramp_l = (pos[i - 1] + tile - p) if i > 0 else 0
        ramp_r = (p + tile - pos[i + 1]) if i < n - 1 else 0
        out.append((p, max(0, ramp_l), max(0, ramp_r)))
    return tile, out


def _smooth(t):
    return t * t * (3.0 - 2.0 * t)


def feather_1d(tile_len, ramp_l, ramp_r):
    """Smoothstep feather weights for one axis of one tile (float64, len ==
    tile_len). Facing ramps of neighbouring tiles cover the SAME interval at
    the SAME phase, so they sum to exactly 1 by s(t) + (1 - s(t)) == 1."""
    w = np.ones(int(tile_len), dtype=np.float64)
    ramp_l = int(ramp_l)
    ramp_r = int(ramp_r)
    if ramp_l > 0:
        k = np.arange(ramp_l, dtype=np.float64)
        w[:ramp_l] = _smooth((k + 0.5) / ramp_l)
    if ramp_r > 0:
        k = np.arange(ramp_r, dtype=np.float64)
        w[tile_len - ramp_r:] = 1.0 - _smooth((k + 0.5) / ramp_r)
    return w


def plan_grid(width, height, tile, overlap):
    """Full 2D tiling plan. Returns a dict:
      tile_w, tile_h  -- axis tile lengths actually used (clamped to canvas)
      nx, ny          -- grid dimensions
      tiles           -- row-major list of dicts: ix, iy, x, y and the four
                         ramp widths (l, r, t, b); the 2D feather is the
                         outer product feather_1d(w-axis) x feather_1d(h-axis)
    """
    tile_w, xs = plan_axis(width, tile, overlap)
    tile_h, ys = plan_axis(height, tile, overlap)
    tiles = []
    for iy, (y, rt, rb) in enumerate(ys):
        for ix, (x, rl, rr) in enumerate(xs):
            tiles.append({"ix": ix, "iy": iy, "x": x, "y": y,
                          "l": rl, "r": rr, "t": rt, "b": rb})
    return {"tile_w": tile_w, "tile_h": tile_h,
            "nx": len(xs), "ny": len(ys), "tiles": tiles}


def feather_2d(plan, t):
    """The separable 2D weight mask for one tile dict from plan_grid."""
    wx = feather_1d(plan["tile_w"], t["l"], t["r"])
    wy = feather_1d(plan["tile_h"], t["t"], t["b"])
    return np.outer(wy, wx)


def tile_seed(seed, ix, iy, nx):
    """Deterministic per-tile noise seed: splitmix64 of (seed, tile index).
    A pure function of the tile position -- frame-independent by design, so
    a video batch keeps the same noise structure per tile over time."""
    index = int(iy) * int(nx) + int(ix)
    h = (int(seed) + (index + 1) * _SM_GAMMA) & _MASK64
    h = ((h ^ (h >> 30)) * _SM_M1) & _MASK64
    h = ((h ^ (h >> 27)) * _SM_M2) & _MASK64
    h = h ^ (h >> 31)
    return int(h & 0x7FFFFFFFFFFFFFFF)


def drop_refine_stages(stages, joint_model):
    """v883: with a joint audio-video model on the wire, the refine stages
    cannot run (the model's forward reads the audio half of a latent that a
    tile does not have) -- so they are DROPPED, whole, and the pixel path
    (final ESRGAN + fit) is what remains. Returns the list unchanged when the
    model is an ordinary image/video model. Pure data, so the guard can drive
    it without comfy."""
    return [] if joint_model else list(stages)


def plan_stages(dual_moe, upscale_by, denoise, steps, cfg,
                upscale_by_low, denoise_low, steps_low, cfg_low):
    """The MoE STAGE CHAIN as pure data (models stay in the node; the low
    inputs fall back there). Single: one stage. High + Low: the HIGH expert
    shapes structure, then the LOW expert polishes detail -- the correct MoE
    translation for refine-strength denoises, where a sigma-boundary split
    would give the HIGH expert zero steps."""
    if not dual_moe:
        return [{"tag": "single", "factor": float(upscale_by),
                 "denoise": float(denoise), "steps": int(steps),
                 "cfg": float(cfg)}]
    return [{"tag": "high", "factor": float(upscale_by),
             "denoise": float(denoise), "steps": int(steps),
             "cfg": float(cfg)},
            {"tag": "low", "factor": float(upscale_by_low),
             "denoise": float(denoise_low), "steps": int(steps_low),
             "cfg": float(cfg_low)}]
