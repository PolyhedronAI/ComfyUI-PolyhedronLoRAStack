# -*- coding: ascii -*-
"""Polyhedron Transition Check (v1215) -- did the clip MOVE from A to B, or only fade?

Frank 23.10.: "First Frame to Last Frame ... den sauberen Uebergang hin vom ersten zum letzten Frame ...
A) Model steht B) Model hat sich auf einen Stuhl gesetzt". A first / last frame run pins both pictures; the
path between them is the model's. The official guide names the failure itself: a pair the model cannot
connect falls back to a DISSOLVE (A fades out while B fades in) -- the frames of a clip, not a motion.

This module measures that, on the pictures alone, with no model:

  pair_check(a, b)      BEFORE a run: how much of the picture differs between the two keyframes, and the
                        light. A standing -> sitting pair made from ONE picture (same camera, light and
                        background, B edited from A) changes a small part of the frame; two unrelated
                        photos change most of it -- the model has no common ground to move on.
  render_check(frames)  AFTER a run: each frame against its own first (A') and last (B') frame.
                          progress  how far along the way A' -> B' the frame is (distance based, works for a
                                    motion and for a fade alike)
                          blend     the best mix (1 - x) A' + x B' of the frame, and what is left over:
                                    residual = |frame - mix| / |B' - A'| over the pixels that change
                        A dissolve IS that mix -- its in-between frames have a residual near 0. A body that
                        moves through intermediate poses is no mix of the two end pictures -- the residual is
                        large. A clip that holds A and changes within a few frames at the end has (almost)
                        no in-between frames: a SNAP, reported with its time (the prompting notes: "H3
                        brakes into the keyframe").
  anchor_hit(...)       optional: how close the rendered first / last frame came to the keyframes given.

The thresholds below are RULES OF THUMB, set on synthetic clips (tests/test_v1215_transition.py: an exact
cross-fade, a cross-fade with codec-like noise, a bar that moves, a hold and snap). They are not measured on
H3 renders yet -- the info says so, and the field test sets them.

Torch only, CPU, the frames are shrunk to SIDE pixels first -- a 141-frame clip costs well under a second.
ASCII only.
"""

import math

import torch
import torch.nn.functional as F

SIDE = 96                 # the long side the frames are measured at (area downscale = a light blur, codec noise averages out)
PIXEL_LEVEL = 0.08        # a pixel "changes" between A and B when its mean channel difference is above this (0..1)
MIN_CHANGED = 0.01        # below 1 % changed pixels there is no way to judge (A and B are the same picture)
BETWEEN = (0.2, 0.8)      # progress range of the in-between frames
MIN_BETWEEN = 3           # fewer in-between frames than this = the change is a snap, not a passage
DISSOLVE_MAX = 0.2        # median residual of the in-between frames: below = the frames are mixes of A and B
MOTION_MIN = 0.35         # above = the frames are no mix -- something moved through
LATE = 0.8                # half of the way after 80 % (or before 20 %) of the clip = the passage is crowded into one end
PAIR_FAR = 0.6           # pair check: more than 60 % of the picture differs -> little common ground
PAIR_LIGHT = 0.08         # pair check: mean brightness shift above this -> the light differs
RULE_NOTE = "thresholds are rules of thumb from synthetic clips, not yet measured on renders"


# ------------------------------------------------------------------------------------------------ helpers
def _nchw(x):
    """IMAGE [N,H,W,C] (or [H,W,C]) -> float [N,3,H,W] on the CPU."""
    x = x.detach().to("cpu", torch.float32)
    if x.dim() == 3:
        x = x[None]
    x = x[..., :3]
    if x.shape[-1] == 1:
        x = x.expand(-1, -1, -1, 3)
    return x.movedim(-1, 1)


def shrink(x, side=SIDE):
    """[N,3,H,W] -> long side `side` (area). Never enlarges."""
    h, w = int(x.shape[-2]), int(x.shape[-1])
    s = float(side) / float(max(h, w))
    if s >= 1.0:
        return x
    return F.interpolate(x, size=(max(1, int(round(h * s))), max(1, int(round(w * s)))), mode="area")


def fit_to(x, h, w):
    """[N,3,H',W'] -> [N,3,h,w] by an aspect-keeping centre cover crop (Keyframes' 'cover')."""
    sh, sw = int(x.shape[-2]), int(x.shape[-1])
    if (sh, sw) == (h, w):
        return x
    s = max(float(h) / sh, float(w) / sw)
    nh, nw = max(h, int(math.ceil(sh * s))), max(w, int(math.ceil(sw * s)))
    y = F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)
    t, l = (nh - h) // 2, (nw - w) // 2
    return y[..., t:t + h, l:l + w]


def change_mask(a, b, level=PIXEL_LEVEL):
    """[3,H,W] x2 -> bool [H,W]: the pixels that differ between a and b."""
    return (b - a).abs().mean(0) > float(level)


def _norm(v):
    return float(torch.sqrt((v * v).sum()))


# ------------------------------------------------------------------------------------------------ the pair
def pair_check(a, b, level=PIXEL_LEVEL):
    """Two keyframes (IMAGE) -> dict(changed, mean_diff, light, verdict, far). b is fitted to a's frame."""
    A = shrink(_nchw(a)[:1])
    B = fit_to(shrink(_nchw(b)[:1]), int(A.shape[-2]), int(A.shape[-1]))
    A, B = A[0], B[0]
    m = change_mask(A, B, level)
    changed = float(m.float().mean())
    mean_diff = float((B - A).abs().mean())
    light = float(B.mean() - A.mean())
    far = changed > PAIR_FAR
    if changed < MIN_CHANGED:
        verdict = "A and B are (almost) the same picture -- the model holds still (right for a loop)"
    elif far:
        verdict = ("%.0f%% of the picture differs -- little common ground, a dissolve is likely: make B from A "
                   "(same camera, light and background, only the subject changed)" % (changed * 100.0))
    else:
        verdict = "%.0f%% of the picture differs -- a pair the model can move through" % (changed * 100.0)
    if abs(light) > PAIR_LIGHT:
        verdict += "; the light differs (%+.2f mean brightness) -- the model fades it" % light
    return {"changed": changed, "mean_diff": mean_diff, "light": light, "far": far, "verdict": verdict,
            "mask": m}


# ------------------------------------------------------------------------------------------------ the render
def render_check(frames, level=PIXEL_LEVEL, fps=24.0):
    """Rendered frames (IMAGE [N,H,W,3]) -> dict(kind, score, progress, residual, between, snap_at, changed,
    verdict). kind: 'motion' | 'dissolve' | 'mixed' | 'snap' | 'still' | 'short'."""
    X = shrink(_nchw(frames))
    n = int(X.shape[0])
    out = {"n": n, "progress": [], "residual": [], "between": 0, "score": -1.0, "snap_at": None, "changed": 0.0}
    if n < 5:
        out.update(kind="short", verdict="%d frame(s) -- too short to judge a passage" % n)
        return out
    A, B = X[0], X[-1]
    m = change_mask(A, B, level)
    out["changed"] = float(m.float().mean())
    if out["changed"] < MIN_CHANGED:
        out.update(kind="still", verdict="the first and the last frame are (almost) the same -- nothing to judge "
                                         "(a loop or a hold)")
        return out
    D = (B - A)[:, m]
    dd = float((D * D).sum())
    nd = math.sqrt(dd)
    prog, res = [], []
    for t in range(n):
        Ft = X[t][:, m]
        da, db = _norm(Ft - A[:, m]), _norm(Ft - B[:, m])
        prog.append(da / (da + db) if da + db > 1e-9 else 0.0)
        x = max(0.0, min(1.0, float(((Ft - A[:, m]) * D).sum()) / dd))
        res.append(_norm(Ft - A[:, m] - x * D) / nd)
    out["progress"], out["residual"] = prog, res
    mid = [res[t] for t in range(1, n - 1) if BETWEEN[0] <= prog[t] <= BETWEEN[1]]
    out["between"] = len(mid)
    # where the way is crossed: the first frame past half of it
    half = next((t for t in range(n) if prog[t] >= 0.5), n - 1)
    if len(mid) < MIN_BETWEEN:
        out["snap_at"] = half
        out.update(kind="snap", verdict="SNAP: the picture changes within %d frame(s) at %.2f s -- it holds and "
                                        "jumps instead of moving (describe the way, give it more time, or a pose "
                                        "or a guide in between)" % (max(1, len(mid)), half / float(fps)))
        return out
    s = sorted(mid)
    score = s[len(s) // 2]
    out["score"] = float(score)
    if score < DISSOLVE_MAX:
        out.update(kind="dissolve", verdict="DISSOLVE: the in-between frames are mixes of the first and the last "
                                            "(residual %.2f) -- a fade, not a motion" % score)
    elif score >= MOTION_MIN:
        out.update(kind="motion", verdict="MOTION: the in-between frames are no mix of the two ends (residual %.2f) "
                                          "-- something moves through" % score)
    else:
        out.update(kind="mixed", verdict="MIXED: partly a fade, partly a motion (residual %.2f)" % score)
    out["half_at"] = half
    # v1215: a passage crammed into one end of the clip (H3 "brakes into the keyframe")
    if half > LATE * (n - 1) or half < (1.0 - LATE) * (n - 1):
        out["verdict"] += ("; but half of the way falls at %.0f%% of the clip -- the passage is crowded into one end"
                           % (100.0 * half / float(n - 1)))
    return out


def anchor_hit(frames, first=None, last=None, level=PIXEL_LEVEL):
    """How far the rendered first / last frame is from the keyframe given, in % of the A -> B way.
    -> list of (which, percent) for the keyframes given."""
    X = shrink(_nchw(frames))
    h, w = int(X.shape[-2]), int(X.shape[-1])
    ends = {"first": X[0], "last": X[-1]}
    keys = {k: fit_to(shrink(_nchw(v)[:1]), h, w)[0] for k, v in (("first", first), ("last", last)) if v is not None}
    a = keys.get("first", X[0])
    b = keys.get("last", X[-1])
    m = change_mask(a, b, level)
    if float(m.float().mean()) < MIN_CHANGED:
        m = torch.ones_like(m)
    way = _norm((b - a)[:, m]) or 1.0
    return [(k, 100.0 * _norm((ends[k] - v)[:, m]) / way) for k, v in keys.items()]


# ------------------------------------------------------------------------------------------------ the report picture
def _plot(prog, res, w, h):
    """A small graph: progress (white) and residual (orange) over time, the two thresholds dotted."""
    img = torch.full((3, h, w), 0.08)
    n = len(prog)
    if n < 2:
        return img

    def y_of(v):
        return int(round((h - 1) * (1.0 - max(0.0, min(1.0, v)))))

    for thr, col in ((DISSOLVE_MAX, (0.25, 0.6, 1.0)), (MOTION_MIN, (0.3, 0.9, 0.4))):
        y = y_of(thr)
        for x in range(0, w, 4):
            img[:, y, x] = torch.tensor(col)
    for series, col in ((prog, (0.95, 0.95, 0.95)), (res, (1.0, 0.55, 0.1))):
        for x in range(w):
            t = x * (n - 1) / float(max(1, w - 1))
            i = min(n - 2, int(t))
            v = series[i] + (series[i + 1] - series[i]) * (t - i)
            y = y_of(v)
            img[:, max(0, y - 1):y + 1, x] = torch.tensor(col)[:, None]
    return img


def report_image(frames=None, first=None, last=None, r=None, tile=256):
    """IMAGE [1,H,W,3]: first | middle | last | change map, and the graph below when frames were judged."""
    pics = []
    if frames is not None:
        X = _nchw(frames)
        pics = [X[0], X[int(X.shape[0]) // 2], X[-1]]
    else:
        pics = [p for p in (first, last) if p is not None]
        pics = [_nchw(p)[0] for p in pics]
    if not pics:
        return torch.zeros(1, 64, 64, 3)
    h0, w0 = int(pics[0].shape[-2]), int(pics[0].shape[-1])
    th = max(16, int(round(tile * h0 / float(w0))))
    cells = [fit_to(p[None], th, tile)[0] for p in pics]
    m = change_mask(cells[0], cells[-1]).float()
    cells.append(torch.stack([m, m * 0.55, m * 0.1]))
    row = torch.cat(cells, dim=2)
    if r is not None and r.get("progress"):
        row = torch.cat([row, _plot(r["progress"], r["residual"], int(row.shape[-1]), 120)], dim=1)
    return row.clamp(0, 1).movedim(0, -1)[None]


# ------------------------------------------------------------------------------------------------ the node
class ULSTransitionCheck:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixel_level": ("FLOAT", {"default": PIXEL_LEVEL, "min": 0.01, "max": 0.5, "step": 0.01,
                    "tooltip": "How much a pixel must change between the first and the last picture to count as "
                               "changing (mean of the colour channels, 0..1). Higher = only the big changes are "
                               "judged."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "The rendered clip (after the VAE decode). Judged as a passage from "
                                                "its own first to its own last frame: motion, dissolve or snap."}),
                "first_frame": ("IMAGE", {"tooltip": "The first keyframe. Without images: the pair is checked before "
                                                     "a run. With images: how close the clip's first frame came."}),
                "last_frame": ("IMAGE", {"tooltip": "The last keyframe, the same way."}),
            },
        }

    RETURN_TYPES = ("STRING", "FLOAT", "IMAGE")
    RETURN_NAMES = ("info", "score", "report")
    OUTPUT_TOOLTIPS = ("What was measured and the verdict.",
                       "Median residual of the in-between frames: near 0 = a fade, high = a motion. -1 = not "
                       "judged (no clip, a snap, a still or too short).",
                       "First | middle | last frame | the pixels that change, and the graph: progress (white) and "
                       "residual (orange) over time, the fade and the motion threshold dotted.")
    FUNCTION = "check"
    OUTPUT_NODE = True
    CATEGORY = "Polyhedron/Video"
    DESCRIPTION = ("Checks a first-to-last-frame passage. Before a run: how much of the picture differs between the "
                   "two keyframes and whether the light differs (little common ground = a dissolve is likely). "
                   "After a run: whether the in-between frames are mixes of the two ends (a dissolve), something "
                   "moving through (a motion), or a hold that jumps at the end (a snap).")

    def check(self, pixel_level, images=None, first_frame=None, last_frame=None):
        lines, r, score = [], None, -1.0
        if images is None and (first_frame is None or last_frame is None):
            raise ValueError("[PLS] Transition Check: wire the rendered images, or both keyframes for the pair "
                             "check before a run.")
        if first_frame is not None and last_frame is not None:
            p = pair_check(first_frame, last_frame, pixel_level)
            lines.append("pair A -> B: %s" % p["verdict"])
        if images is not None:
            r = render_check(images, pixel_level)
            lines.append("clip: %d frames, %.0f%% of the picture changes, %d in-between frame(s) -- %s"
                         % (r["n"], r["changed"] * 100.0, r["between"], r["verdict"]))
            if r.get("half_at") is not None:
                lines.append("half of the way at frame %d (%.2f s)" % (r["half_at"], r["half_at"] / 24.0))
            score = float(r["score"])
            if first_frame is not None or last_frame is not None:
                for which, pc in anchor_hit(images, first_frame, last_frame, pixel_level):
                    lines.append("%s frame: %.0f%% of the way off its keyframe%s"
                                 % (which, pc, "" if pc < 15.0 else " -- the anchor was not reached"))
        lines.append("(%s)" % RULE_NOTE)
        info = "Transition Check: " + "\n".join(lines)
        print("[PLS] " + info.replace("\n", "\n[PLS]   "))
        rep = report_image(images, first_frame, last_frame, r)
        return {"ui": {"text": [info]}, "result": (info, score, rep)}


NODE_CLASS_MAPPINGS = {"ULSTransitionCheck": ULSTransitionCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"ULSTransitionCheck": "\u2b21 Polyhedron Transition Check"}
