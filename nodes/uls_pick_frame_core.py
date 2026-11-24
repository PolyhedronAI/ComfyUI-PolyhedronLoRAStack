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
    "name",
]

NAME_TOOLTIP = (
    "For mode 'name': which frame to take, by its NAME in the list wired "
    "to `names` (e.g. Turntable Pick's direction_names 'S, SE, E, ...': "
    "name E takes the E view). Upper and lower case do not matter. A name "
    "the list does not hold is an error that shows the list. Ignored by "
    "the other modes."
)

NAMES_TOOLTIP = (
    "For mode 'name': the names of the batch's frames, in order, separated "
    "by commas -- one per frame (e.g. from Turntable Pick, Camera Ring or "
    "Mesh Ring Render)."
)

MASK_TOOLTIP = (
    "Optional: a mask batch that goes the same way -- the same frame(s) "
    "are taken from it."
)

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
    "batch -- a contact sheet from a video. 'name' takes the frame called "
    "`name` in the list wired to `names` -- a view of a turnaround by its "
    "direction instead of by its number."
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


def names_list(text):
    """'S, SE, E' (commas or line breaks) -> ['S', 'SE', 'E']."""
    return [x.strip() for x in str(text or "").replace("\n", ",").split(",") if x.strip()]


def by_name(n, names, name):
    """The index of `name` in the list `names` (v1385). The list must name every frame of the batch -- a list
    of another length would pick a wrong view without a word."""
    lst = names_list(names)
    want = str(name or "").strip()
    if not lst:
        raise ValueError("[PLS] Pick Frame: mode 'name' needs the frames' names -- wire a names list (e.g. Turntable "
                         "Pick's direction_names) into `names`.")
    if not want:
        raise ValueError("[PLS] Pick Frame: mode 'name' -- `name` is empty; the list holds: %s" % ", ".join(lst))
    if len(lst) != int(n):
        raise ValueError("[PLS] Pick Frame: %d name(s) (%s) for %d frame(s) -- the list must name every frame."
                         % (len(lst), ", ".join(lst), int(n)))
    low = [x.lower() for x in lst]
    if want.lower() not in low:
        raise ValueError("[PLS] Pick Frame: no frame is called '%s' -- the list holds: %s" % (want, ", ".join(lst)))
    if low.count(want.lower()) > 1:
        raise ValueError("[PLS] Pick Frame: '%s' is in the list %d times (%s) -- names must be unique."
                         % (want, low.count(want.lower()), ", ".join(lst)))
    return low.index(want.lower())


def take(batch, got):
    """The frames `got` of a batch: a contiguous run stays a slice (no copy), anything else is gathered."""
    if got == list(range(got[0], got[-1] + 1)):
        return batch[got[0]:got[-1] + 1]
    import torch
    return torch.cat([batch[i:i + 1] for i in got], dim=0)


def select_mask(mask, got, like):
    """The mask that goes with the picked frames (v1385): the same frames of a mask batch of the same length, a
    single mask as it is; none wired -> an empty one of the picked frames' size."""
    if mask is None:
        if len(tuple(getattr(like, "shape", ()))) < 3:   # not an image batch (a stand-in): nothing to size it by
            return None
        import torch
        return torch.zeros((int(like.shape[0]), int(like.shape[1]), int(like.shape[2])), dtype=torch.float32)
    m = mask if mask.dim() == 3 else mask.unsqueeze(0)
    if m.shape[0] == 1:
        return m
    if max(got) >= m.shape[0]:
        raise ValueError("[PLS] Pick Frame: the mask has %d frame(s), frame %d was picked -- wire the mask that "
                         "belongs to the images." % (int(m.shape[0]), max(got)))
    return take(m, got)


def select(images, mode, frame_index, count, names=None, name="", with_indices=False):
    """Apply plan() to a real image batch. Returns (images, message) -- with_indices: (images, message, indices)."""
    n = int(images.shape[0])
    if n == 0:
        return (images, "empty image batch -- passing through", []) if with_indices else (images, "empty image batch -- passing through")

    if str(mode) == "name":                              # v1385: a view by its name
        i = by_name(n, names, name)
        got, note = [i], "'%s' = frame %d of %d" % (str(name).strip(), i, n)
        out = images[i:i + 1]
        msg = "%s -> 1 frame(s) [name]" % note
        return (out, msg, got) if with_indices else (out, msg)
    got, note = plan(n, mode, frame_index, count)

    # Contiguous runs stay a slice (no copy); anything else is gathered.
    if got == list(range(got[0], got[-1] + 1)):
        out = images[got[0]:got[-1] + 1]
    else:
        import torch
        out = torch.cat([images[i:i + 1] for i in got], dim=0)
    msg = "%s -> %d frame(s) [%s]" % (note, len(got), mode)
    return (out, msg, got) if with_indices else (out, msg)
