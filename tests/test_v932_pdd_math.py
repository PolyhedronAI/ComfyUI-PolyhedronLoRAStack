# -*- coding: ascii -*-
"""v932 -- the PDD arithmetic, pinned without leaning on anyone's reference.

The published reference implementation is Apache-2.0 and was used to check this
module while it was written (grid and fusion matched exactly, plans to 1e-16),
but it is NOT in the pack -- so nothing here may depend on it. Every promise is
pinned against something self-standing:

  * the grid against its closed form, evaluated independently
  * fusion against an EXPLICIT LOOP -- an einsum is a compact way to write a
    weighted sum, so the loop is the honest witness that it sums what it should
  * fusion against the identity: at full resolution every block covers exactly
    one fine interval, so the fused heads must BE the bank
  * the envelope against the refusals it is supposed to make

The load-bearing detail is that fine intervals are NOT equal in length. A plain
average would pass a uniform-grid test and be wrong on the real shifted grid,
so the weights are pinned against the grid they come from, and a uniform
fixture is explicitly rejected as a witness.

torch for the fusion only. No comfy, no reference.
"""
import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uls_pdd_math as P  # noqa: E402

FAILED = []
CHECKS = []


def _fail(msg):
    FAILED.append(msg)
    CHECKS.append(msg)
    print("FAIL: {}".format(msg))


def _ok(msg):
    CHECKS.append(msg)
    print("ok  : {}".format(msg))


def _need(cond, msg):
    _ok(msg) if cond else _fail(msg)


N = P.DEFAULT_NUM_STEPS

# --- 0: the constants themselves ------------------------------------------------
# Pinned separately and FIRST, because everything below could be perfectly
# correct arithmetic driven with literal 12.0 and 3.0 while the constants the
# CALLERS read had drifted. A mutation setting AUDIO_SHIFT to 12.0 passed this
# guard green until these four lines existed.
_need(P.VIDEO_SHIFT == 12.0, "0  VIDEO_SHIFT is 12 (the trained video grid)")
_need(P.AUDIO_SHIFT == 3.0, "0  AUDIO_SHIFT is 3 (the trained audio grid)")
_need(P.VIDEO_SHIFT != P.AUDIO_SHIFT,
      "0  the two modalities are on DIFFERENT grids -- the whole reason plans "
      "are computed per modality")
_need(P.DEFAULT_NUM_STEPS == 32 and P.DEFAULT_BLOCK == 4,
      "0  the fine grid is 32 steps with a minimum block of 4")
_need(tuple(P.LEGAL_BLOCKS) == (4, 8), "0  only 4 and 8 are trained block sizes")
# and the arithmetic below must be driven through the constants, not past them
_need(P.fine_step_sizes(P.VIDEO_SHIFT, N) == P.fine_step_sizes(12.0, N),
      "0  VIDEO_SHIFT really is what the video grid is built from")
_need(P.fine_step_sizes(P.AUDIO_SHIFT, N) != P.fine_step_sizes(P.VIDEO_SHIFT, N),
      "0  and AUDIO_SHIFT yields a genuinely different grid")

# --- A: the shifted-sigma map ---------------------------------------------------
_need(abs(P.shifted_sigma(12.0, 0.0) - 0.0) < 1e-15, "A  shift maps 0 to 0")
_need(abs(P.shifted_sigma(12.0, 1.0) - 1.0) < 1e-15, "A  and 1 to 1")
_need(abs(P.shifted_sigma(1.0, 0.37) - 0.37) < 1e-15, "A  shift 1 is the identity")
# closed form, written out independently
for s in (0.1, 0.25, 0.5, 0.75, 0.9):
    want = 12.0 * s / (1.0 + 11.0 * s)
    _need(abs(P.shifted_sigma(12.0, s) - want) < 1e-15,
          "A  shift 12 at s=%.2f matches 12s/(1+11s)" % s)
_need(all(P.shifted_sigma(12.0, s) > s for s in (0.1, 0.5, 0.9)),
      "A  shift 12 pushes sigma UP (more steps spent at high noise)")

# --- B: the fine grid -----------------------------------------------------------
sig = P.fine_sigmas(12.0, N)
_need(len(sig) == N + 1, "B  the grid has N+1 = %d boundaries" % (N + 1))
_need(abs(sig[0] - 1.0) < 1e-12 and abs(sig[-1]) < 1e-12,
      "B  it runs from sigma 1 down to 0")
_need(all(sig[i] > sig[i + 1] for i in range(N)), "B  strictly descending")

steps = P.fine_step_sizes(12.0, N)
_need(len(steps) == N and all(x > 0 for x in steps), "B  N positive step sizes")
_need(abs(sum(steps) - 1.0) < 1e-12, "B  they sum to 1 (the whole schedule)")

ratio = max(steps) / min(steps)
_need(ratio > 5,
      "B  THE POINT: intervals are far from equal (longest/shortest = %.1f) -- "
      "a plain average would be wrong" % ratio)
uni = P.fine_step_sizes(1.0, N)
_need(abs(max(uni) - min(uni)) < 1e-12,
      "B  (at shift 1 they WOULD be equal -- which is why a shift-1 fixture "
      "cannot witness the weighting)")
_need(max(P.fine_step_sizes(P.AUDIO_SHIFT, N)) / min(P.fine_step_sizes(P.AUDIO_SHIFT, N)) != ratio,
      "B  video and audio grids really differ (shift 12 vs 3)")

# --- C: the envelope ------------------------------------------------------------
_need(P.resolve_partition(nfe=8) == (4,) * 8, "C  nfe 8 -> eight blocks of 4")
_need(P.resolve_partition(nfe=4) == (8,) * 4, "C  nfe 4 -> four blocks of 8")
_need(P.resolve_partition(nfe=6) == (8, 8, 4, 4, 4, 4), "C  nfe 6 -> the mixed partition")
_need(sum(P.resolve_partition(nfe=6)) == N, "C  and it still sums to N")
_need(P.resolve_partition(partition="8,4,4,4,4,4,4") == (8, 4, 4, 4, 4, 4, 4),
      "C  a custom partition of legal blocks is accepted")

for bad, why in ((dict(nfe=20), "an untrained step count"),
                 (dict(nfe=32), "the full fine grid as steps"),
                 (dict(nfe=16), "another untrained count"),
                 (dict(partition="4,4,4"), "a partition that is too short"),
                 (dict(partition="6,6,4,4,4,8"), "sizes outside {4,8}"),
                 (dict(partition="16,16"), "blocks that are too long"),
                 (dict(partition=""), "an empty partition")):
    try:
        P.resolve_partition(**bad)
    except ValueError:
        _ok("C  refused: %s" % why)
    except Exception as ex:
        _fail("C  %s raised %s, not ValueError" % (why, type(ex).__name__))
    else:
        _fail("C  %s was ACCEPTED" % why)
try:
    P.resolve_partition()
except ValueError:
    _ok("C  neither nfe nor partition is an error, not a default")
else:
    _fail("C  no argument silently defaulted")

# --- D: block plans -------------------------------------------------------------
sizes = P.resolve_partition(nfe=8)
_need(P.block_starts(sizes) == [0, 4, 8, 12, 16, 20, 24, 28],
      "D  block starts are cumulative")
plans = P.block_plans(12.0, sizes)
_need(len(plans) == len(sizes), "D  one plan per block")
for i, p in enumerate(plans):
    _need(len(p) == N, "D  plan %d spans the whole fine grid" % i)
    _need(abs(sum(p) - 1.0) < 1e-12, "D  plan %d sums to 1" % i)
    start = P.block_starts(sizes)[i]
    outside = [v for j, v in enumerate(p) if not (start <= j < start + sizes[i])]
    _need(not any(outside), "D  plan %d is zero outside its own block" % i)

# weights must follow the grid, not be uniform
p0 = plans[0]
inside = [p0[j] for j in range(0, 4)]
_need(abs(max(inside) - min(inside)) > 1e-6,
      "D  weights within a block are NOT uniform -- they follow the interval "
      "lengths (%.4f .. %.4f)" % (min(inside), max(inside)))
exp = [steps[j] / sum(steps[0:4]) for j in range(4)]
_need(max(abs(a - b) for a, b in zip(inside, exp)) < 1e-12,
      "D  and they are exactly length/span")

audio_plans = P.block_plans(P.AUDIO_SHIFT, sizes)
_need(max(abs(a - b) for a, b in zip(plans[0], audio_plans[0])) > 1e-6,
      "D  audio plans differ from video plans (own grid, same blocks)")

# --- E: fusion ------------------------------------------------------------------
torch.manual_seed(932)
OUT, IN = 12, 16
bank_w = torch.randn(N, OUT, IN)
bank_b = torch.randn(N, OUT)

fw, fb = P.fuse_heads(bank_w, bank_b, plans)
_need(tuple(fw.shape) == (len(sizes), OUT, IN),
      "E  fused weights are [P, out, in]: %s" % (list(fw.shape),))
_need(fb is not None and tuple(fb.shape) == (len(sizes), OUT),
      "E  fused biases are [P, out]")
_need(fw.dtype == torch.float32, "E  fusion is done in float32")

# the honest witness: an explicit loop
loop_w = torch.zeros_like(fw)
loop_b = torch.zeros_like(fb)
for pi, plan in enumerate(plans):
    for n in range(N):
        if plan[n]:
            loop_w[pi] += bank_w[n].to(torch.float32) * plan[n]
            loop_b[pi] += bank_b[n].to(torch.float32) * plan[n]
_need(torch.allclose(fw, loop_w, atol=1e-5) and torch.allclose(fb, loop_b, atol=1e-5),
      "E  THE SUM: the einsum equals an explicit weighted loop "
      "(max %.2e)" % max(float((fw - loop_w).abs().max()),
                         float((fb - loop_b).abs().max())))

full = P.block_plans(12.0, (1,) * N, N)
iw, ib = P.fuse_heads(bank_w, bank_b, full)
_need(torch.allclose(iw, bank_w, atol=1e-5) and torch.allclose(ib, bank_b, atol=1e-5),
      "E  THE IDENTITY: at full resolution the fusion returns the bank itself")

# a mean is bounded by its inputs
for pi in range(len(sizes)):
    st = P.block_starts(sizes)[pi]
    chunk = bank_w[st:st + sizes[pi]]
    if not (fw[pi].max() <= chunk.max() + 1e-5 and fw[pi].min() >= chunk.min() - 1e-5):
        _fail("E  fused head %d escapes the range of the heads it averages" % pi)
        break
else:
    _ok("E  every fused head stays within the range of its own inputs")

no_bias_w, no_bias_b = P.fuse_heads(bank_w, None, plans)
_need(no_bias_b is None and torch.allclose(no_bias_w, fw, atol=0),
      "E  a bank without biases fuses its weights and returns None for bias")

try:
    P.fuse_heads(bank_w, bank_b, [[0.5, 0.5]])
except ValueError:
    _ok("E  a plan of the wrong length is refused, not broadcast")
else:
    _fail("E  a short plan was accepted")

# --- F: sampler boundaries and block selection ----------------------------------
b = P.boundary_sigmas(sizes)
_need(len(b) == len(sizes) + 1, "F  one more boundary than blocks")
_need(abs(b[0] - 1.0) < 1e-12 and abs(b[-1]) < 1e-12, "F  from 1 down to 0")
_need(all(b[i] > b[i + 1] for i in range(len(b) - 1)), "F  strictly descending")
grid = P.fine_sigmas(12.0, N)
_need(all(any(abs(x - g) < 1e-12 for g in grid) for x in b),
      "F  every boundary IS a fine-grid point (never interpolated)")

_need([P.select_block(x, b) for x in b[:-1]] == list(range(len(sizes))),
      "F  each boundary selects its own block, in order")
_need(P.select_block(b[-1], b) is None,
      "F  the final sigma has no block after it")
_need(P.select_block((b[0] + b[1]) / 2.0, b) is None,
      "F  a sigma BETWEEN boundaries selects nothing -- no nearest-neighbour clamp")
_need(P.select_block(b[3] + 1e-6, b) == 3,
      "F  a hair of float drift still lands on its block")
_need(P.select_block(b[3] + 0.01, b) is None,
      "F  but a real gap does not")

b4 = P.boundary_sigmas(P.resolve_partition(nfe=4))
_need(len(b4) == 5 and all(any(abs(x - g) < 1e-12 for g in grid) for x in b4),
      "F  nfe 4 boundaries are fine-grid points too")
_need(abs(P.boundary_sigmas(sizes, shift=P.AUDIO_SHIFT)[1] - b[1]) > 1e-6,
      "F  the audio grid yields different boundaries")

print("\n%d checks, %d failed" % (len(CHECKS), len(FAILED)))
if FAILED:
    print("\n".join(FAILED))
    sys.exit(1)
print("v932 pdd math: PASS")
