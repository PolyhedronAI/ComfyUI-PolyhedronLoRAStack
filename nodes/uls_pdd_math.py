# -*- coding: utf-8 -*-
"""
uls_pdd_math.py
═══════════════
v932 -- the arithmetic of parallel-decoding distillation. Pure: no torch at
import time, no comfy, no model. Everything here is unit-testable in isolation.

WHAT PDD IS, IN ONE PARAGRAPH
-----------------------------
An ordinary sampler evaluates the model once per step and takes the velocity it
reports at that instant. A parallel-decoding-distilled model instead carries N
output heads, one per FINE interval of a trained grid, each predicting the MEAN
velocity over its own interval. A sampler step that spans several fine
intervals can therefore consume one fused head -- the length-weighted mean of
the heads it covers -- and cover the whole span in a single evaluation. Eight
steps instead of thirty-two, without asking the model to extrapolate.

THE GRID
--------
Both modalities live on a shifted-sigma schedule

    shifted_sigma(shift, s) = shift * s / (1 + (shift - 1) * s)

with the video stream at shift 12 and audio at shift 3. The fine grid is that
map applied to `linspace(1, 0, N+1)`; the ASCENDING time grid is 1 - that. Fine
intervals are NOT equal in length -- which is exactly why fusion is a weighted
mean and not a plain average, and why the weights differ between video and
audio even though the block structure is shared.

THE TRAINING ENVELOPE
---------------------
Blocks may only be 4 or 8 fine steps long, and must sum to N. This is not our
rule; it is where the heads were trained. Evaluating the trunk anywhere else
feeds the heads features they have never seen, and the output is noise -- so
`resolve_partition` refuses rather than degrades. More steps is NOT closer to
the teacher here: the heads are only ever decoded from envelope block starts.

Fusion, grid and plan follow alibaba-pai's published reference implementation
(Apache-2.0); the reference itself is not carried in this pack -- only the
arithmetic, rewritten, with the guard driving both against each other.
"""

DEFAULT_NUM_STEPS = 32
DEFAULT_BLOCK = 4
LEGAL_BLOCKS = (4, 8)

VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0

def partition_for_nfe(nfe, num_steps=DEFAULT_NUM_STEPS):
    """The block sizes for `nfe` steps, or None when no legal one exists.

    With only 4 and 8 admitted and a fixed total, the partition is FORCED, not
    chosen: a blocks of 8 and b of 4 give 8a + 4b = num_steps and a + b = nfe,
    hence a = num_steps/4 - nfe and b = 2*nfe - num_steps/4. Both must be
    non-negative, which is exactly why the legal step counts are the closed
    range below and nothing else -- fewer would need blocks longer than 8, more
    would need blocks shorter than 4.

    The size-8 blocks go FIRST, i.e. at high sigma. That is where the fine
    boundaries span the least sigma, so merging two of them costs the least.
    For 4, 6 and 8 this reproduces the release's own published partitions.
    """
    quarter = num_steps // 4
    a = quarter - int(nfe)
    b = 2 * int(nfe) - quarter
    if a < 0 or b < 0 or 8 * a + 4 * b != num_steps:
        return None
    return (8,) * a + (4,) * b


def legal_nfe(num_steps=DEFAULT_NUM_STEPS):
    """Every step count that has a legal partition, ascending."""
    return tuple(n for n in range(1, num_steps + 1)
                 if partition_for_nfe(n, num_steps) is not None)


LEGAL_NFE = legal_nfe()


def shifted_sigma(shift, s):
    """shift*s / (1 + (shift-1)*s). Works on floats and on sequences."""
    try:
        return [shifted_sigma(shift, x) for x in s]
    except TypeError:
        return shift * s / (1.0 + (shift - 1.0) * s)


def fine_sigmas(shift, num_steps=DEFAULT_NUM_STEPS):
    """The N+1 sigma boundaries of the fine grid, DESCENDING from 1 to 0."""
    lin = [1.0 - i / float(num_steps) for i in range(num_steps + 1)]
    return [shifted_sigma(shift, x) for x in lin]


def fine_step_sizes(shift, num_steps=DEFAULT_NUM_STEPS):
    """Length of each fine interval on the ASCENDING time grid (t = 1 - sigma).

    Positive, and NOT uniform: at shift 12 the first interval is many times the
    last. These are the fusion weights.
    """
    sig = fine_sigmas(shift, num_steps)
    return [sig[i] - sig[i + 1] for i in range(num_steps)]


def resolve_partition(nfe=None, partition=None, num_steps=DEFAULT_NUM_STEPS):
    """Block sizes, or raise ValueError saying exactly what is wrong.

    Refuses off-envelope requests instead of degrading them: a size that is
    neither 4 nor 8, or a partition that does not sum to num_steps.
    """
    if partition:
        if isinstance(partition, str):
            try:
                sizes = tuple(int(x) for x in partition.replace(" ", "").split(",") if x)
            except ValueError:
                raise ValueError("partition must be comma-separated integers, "
                                 "got %r" % (partition,))
        else:
            sizes = tuple(int(x) for x in partition)
        if not sizes:
            raise ValueError("partition is empty")
    elif nfe is not None:
        key = int(nfe)
        sizes = partition_for_nfe(key, num_steps)
        if sizes is None:
            raise ValueError(
                "nfe %d has no legal partition; the possible step counts are "
                "%s. Fewer would need blocks longer than 8, more would need "
                "blocks shorter than 4 -- and neither was trained, so the heads "
                "would be read at boundaries they have never seen. More steps "
                "is not closer to the teacher here."
                % (key, list(legal_nfe(num_steps))))
    else:
        raise ValueError("give either nfe or partition")

    bad = [s for s in sizes if s not in LEGAL_BLOCKS]
    if bad:
        raise ValueError(
            "block size(s) %s are outside the training envelope; only %s are "
            "trained. Anything else renders as noise." % (bad, list(LEGAL_BLOCKS)))
    total = sum(sizes)
    if total != num_steps:
        raise ValueError("block sizes sum to %d, need exactly %d"
                         % (total, num_steps))
    return sizes


def block_starts(sizes):
    """Index of the first fine step of each block."""
    out = []
    at = 0
    for s in sizes:
        out.append(at)
        at += s
    return out


def block_plan(step_sizes, start, size):
    """Weights over the fine grid for ONE block: length-weighted, summing to 1.

    Zero everywhere outside the block. This is the mean-velocity combination an
    Euler step across the block's boundaries consumes.
    """
    span = sum(step_sizes[start:start + size])
    if span <= 0:
        raise ValueError("block at %d spans no time" % start)
    plan = [0.0] * len(step_sizes)
    for i in range(start, start + size):
        plan[i] = step_sizes[i] / span
    return plan


def block_plans(shift, sizes, num_steps=DEFAULT_NUM_STEPS):
    """One plan per block, on this modality's own grid."""
    steps = fine_step_sizes(shift, num_steps)
    return [block_plan(steps, st, sz) for st, sz in zip(block_starts(sizes), sizes)]


def boundary_sigmas(sizes, shift=VIDEO_SHIFT, num_steps=DEFAULT_NUM_STEPS):
    """The sampler's sigma schedule: the block boundaries, descending to 0.

    len == len(sizes) + 1. These are the ONLY sigmas at which the trunk may be
    evaluated; anything between them is off-grid.
    """
    sig = fine_sigmas(shift, num_steps)
    idx = block_starts(sizes) + [num_steps]
    return [sig[i] for i in idx]


def select_block(sigma, boundaries, atol=1e-4):
    """Which block a sigma belongs to, or None when it is not a block start.

    Boundaries descend. The LAST boundary (sigma 0, the end of the schedule) is
    not a block start -- there is no block after it -- and returns None, which
    is why this answers None rather than clamping: an off-grid sigma means the
    schedule is wrong, and saying so beats quietly picking a neighbour.
    """
    for i in range(len(boundaries) - 1):
        if abs(float(sigma) - boundaries[i]) <= atol:
            return i
    return None


def fuse_heads(bank_w, bank_b, plans):
    """Fuse N per-interval heads into len(plans) heads. torch, lazily imported.

    bank_w  [N, out, in]      bank_b [N, out] or None
    plans   list of length-N weight vectors

    Returns (weights [P, out, in], biases [P, out] or None), in float32 --
    the heads are the model's output layer and a fusion in bf16 loses more than
    it saves.
    """
    import torch

    w = bank_w.to(torch.float32)
    n = w.shape[0]
    for p in plans:
        if len(p) != n:
            raise ValueError("plan has %d entries, bank has %d heads"
                             % (len(p), n))
    plan = torch.tensor(plans, dtype=torch.float32, device=w.device)
    out_w = torch.einsum("pn,noi->poi", plan, w)
    out_b = None
    if bank_b is not None:
        out_b = torch.einsum("pn,no->po", plan, bank_b.to(torch.float32))
    return out_w, out_b
