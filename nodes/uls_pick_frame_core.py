"""Polyhedron Pick Frame -- the selection logic, in ONE place.

WHY THIS FILE EXISTS. ULSImagePickFrame ships in two forms: the legacy class
in wan_frame_inflate.py and the V3 schema class in uls_pick_frame_v3.py. They
share a node id and are mutually exclusive at registration, so only one of
them ever runs -- which is precisely the shape of bug that cost us v898: two
copies of the same rule, one of them quietly out of date, and the symptom
only visible on machines where the other copy loads. Both classes now call
`select()` here. There is one rule.

THE -1 COLLISION (v901 finding, the reason for the mode widget). Core's
`ImageFromBatch` reads batch_index -1 as the LAST image. This node has always
read frame_index -1 as the MIDDLE one -- a deliberate choice for inflated Wan
T2I runs, where the first frame can carry anchor artifacts and the last can be
smeared by motion continuity. Both readings are defensible; having them share
a number is not. Swap the two nodes in a graph and you silently get a
different picture, with no error and no warning.

So the number stops deciding and a named mode does. `middle (legacy)` is the
default, so every saved workflow keeps the exact behaviour it had.
"""

MODES = [
    "middle (legacy)",
    "last",
    "first",
    "index",
    "range",
    "every Nth",
]

MODE_TOOLTIP = (
    "Which frame(s) to take. 'middle (legacy)' is this node's original "
    "behaviour and stays the default so saved workflows do not change: "
    "frame_index -1 means the MIDDLE frame, any other value is a plain "
    "0-based index. WATCH OUT -- Core's 'Get Image from Batch' reads -1 as "
    "the LAST frame instead, which is why the other modes are named rather "
    "than numbered. 'last' / 'first' take one frame from either end. "
    "'index' is 0-based and accepts negatives the way Core does (-1 last, "
    "-2 second to last). 'range' takes `count` frames starting at "
    "frame_index. 'every Nth' takes every count-th frame from the whole "
    "batch -- a contact sheet from a video."
)

COUNT_TOOLTIP = (
    "How many frames for 'range', and the step for 'every Nth'. Ignored by "
    "the other modes."
)


def plan(n, mode, frame_index, count):
    """Which frame indices to keep. Pure arithmetic -- no tensors, so it can
    be tested without torch.

    Returns (indices, note). `indices` is never empty for n > 0: a selection
    that lands nowhere is a bug, not a legitimate empty result, and silently
    handing back zero images would break every downstream node.
    """
    if n <= 0:
        return [], "empty batch"

    mode = str(mode)
    idx = int(frame_index)
    cnt = max(1, int(count))

    if mode == "last":
        return [n - 1], "last of %d" % n
    if mode == "first":
        return [0], "first of %d" % n

    if mode == "index":
        # Core's convention, deliberately: negative counts back from the end.
        i = idx + n if idx < 0 else idx
        i = max(0, min(n - 1, i))
        return [i], "index %d of %d" % (i, n)

    if mode == "range":
        i = idx + n if idx < 0 else idx
        i = max(0, min(n - 1, i))
        end = min(n, i + cnt)
        return list(range(i, end)), "range %d..%d of %d" % (i, end - 1, n)

    if mode == "every Nth":
        step = max(1, cnt)
        got = list(range(0, n, step))
        return got, "every %d of %d -> %d frame(s)" % (step, n, len(got))

    # "middle (legacy)" and anything unrecognised: the original rule, byte for
    # byte. An unknown mode falling back to the legacy path is deliberate --
    # a workflow saved by a newer build must not crash an older one.
    if idx == -1:
        return [n // 2], "middle (%d of %d)" % (n // 2, n)
    i = max(0, min(idx, n - 1))
    return [i], "explicit %d of %d" % (i, n)


def select(images, mode, frame_index, count):
    """Apply plan() to a real image batch. Returns (images, message)."""
    n = int(images.shape[0])
    if n == 0:
        return images, "empty image batch -- passing through"

    got, note = plan(n, mode, frame_index, count)

    # Contiguous runs stay a slice (no copy); anything else is gathered.
    if got == list(range(got[0], got[-1] + 1)):
        out = images[got[0]:got[-1] + 1]
    else:
        import torch
        out = torch.cat([images[i:i + 1] for i in got], dim=0)
    return out, "%s -> %d frame(s) [%s]" % (note, len(got), mode)
