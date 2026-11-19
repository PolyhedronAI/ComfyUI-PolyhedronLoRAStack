"""uls_grid_math.py -- schedules for a model that samples on its OWN sigma grid (v1337). Pure, no torch.

Some distilled checkpoints ship a bank of output heads, one per interval of a fixed sigma grid
(MiniMax H3 Image: six heads, seven grid points from 1 to 0). Core serves a step with the dt-weighted mean of
the heads it spans (comfy/ldm/minimax/model.py, FinalLayer.forward) -- so ANY step count up to the number of
heads is a trained run, as long as every step starts and ends ON a grid point. The stock schedulers only do
that by luck: 'simple' at 6, 3, 2 and 1 steps lands on the grid, at 4 and 5 it lands between two points
(4 steps: 0, 1.5, 3, 4.5, 6 in grid indices) and the first step runs to 1.5 on the head of 0-1 alone. A partial
denoise has the same problem: 1 step at 0.25 starts at index 4.5.

This module builds the schedule from the grid instead:

  partition(intervals, steps)   block sizes, as even as they come, the LONGER blocks first (at high sigma --
                                the same law the video PDD line follows: uls_pdd_math.partition_for_nfe)
  plan(grid, steps, denoise)    the run: its sigmas (all grid points), the blocks, the grid start a partial
                                denoise snaps to; None when the count cannot be served on the grid
  on_grid(grid, sigmas)         does a stock schedule already sit on grid points
  index_of(grid, sigma)         a sigma's position in grid indices (0 = the first, noisiest point)

The grid is DECREASING here (index 0 = sigma 1), the way a schedule runs.
"""

import math

ON_GRID_TOL = 0.02     # in grid indices: closer than this to a point IS the point (sgm_uniform sits at 0.99)
DENOISE_TOL = 0.011    # a dial of 0.16 / 0.33 means 1/6 / 2/6 -- no note for the rounding of the dial itself


def decreasing(grid):
    """The grid as a decreasing list of floats (Core stores it ascending)."""
    g = [float(v) for v in grid]
    return g if g[0] >= g[-1] else g[::-1]


def partition(intervals, steps):
    """Block sizes for `steps` steps over `intervals` grid intervals, or None. Sizes differ by at most one,
    the longer blocks come first."""
    intervals, steps = int(intervals), int(steps)
    if steps < 1 or intervals < 1 or steps > intervals:
        return None
    q, r = divmod(intervals, steps)
    return (q + 1,) * r + (q,) * (steps - r)


def index_of(grid, sigma):
    """Position of a sigma on the decreasing grid, in grid indices; clamped to the ends."""
    s = float(sigma)
    n = len(grid) - 1
    if s >= grid[0]:
        return 0.0
    if s <= grid[n]:
        return float(n)
    for k in range(n):
        hi, lo = grid[k], grid[k + 1]
        if lo <= s <= hi:
            return k + ((hi - s) / (hi - lo) if hi > lo else 0.0)
    return float(n)


def on_grid(grid, sigmas, tol=ON_GRID_TOL):
    """True when every sigma of the schedule is a grid point and the schedule walks DOWN the grid."""
    pos = [index_of(grid, s) for s in sigmas]
    if len(pos) < 2:
        return False
    if any(abs(p - round(p)) > tol for p in pos):
        return False
    idx = [int(round(p)) for p in pos]
    return all(b > a for a, b in zip(idx, idx[1:]))


def intervals_left(heads, denoise):
    """How many grid intervals a denoise covers: the nearest whole number, at least one. A tie goes DOWN --
    half way between two starts the gentler refine is taken (it keeps more of the picture it was given)."""
    d = float(denoise)
    if d >= 1.0:
        return int(heads)
    return max(1, min(int(heads), int(math.ceil(d * heads - 0.5 - 1e-9))))


def plan(grid, steps, denoise=1.0):
    """The on-grid run for `steps` steps at `denoise`, or None when it cannot be served (no step, denoise 0,
    or more steps than grid intervals are left).

    Returns {"sigmas", "blocks", "start", "heads", "denoise_grid", "snapped"}: start = the grid index the run
    begins on, denoise_grid = the denoise that index stands for, snapped = the dial was moved to it."""
    g = decreasing(grid)
    heads = len(g) - 1
    d = float(denoise)
    if heads < 1 or int(steps) < 1 or d <= 0.0:
        return None
    left = intervals_left(heads, d)
    blocks = partition(left, steps)
    if blocks is None:
        return None
    start = heads - left
    idx = [start]
    for b in blocks:
        idx.append(idx[-1] + b)
    d_grid = left / float(heads)
    return {"sigmas": [g[i] for i in idx], "blocks": blocks, "start": start, "heads": heads,
            "denoise_grid": d_grid, "snapped": d < 1.0 and abs(d - d_grid) > DENOISE_TOL}


def heads_text(p):
    """'1-2, 3-4, 5, 6' -- the heads each step of a plan takes, counted from 1."""
    out, k = [], p["start"] + 1
    for b in p["blocks"]:
        out.append(str(k) if b == 1 else "%d-%d" % (k, k + b - 1))
        k += b
    return ", ".join(out)
