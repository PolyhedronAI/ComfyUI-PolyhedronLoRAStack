# -*- coding: ascii -*-
"""Polyhedron Control -- pose / depth / edge control for video and image models (v1130, own runtime v1131).

Three nodes, one line from a template (a stock video, an image, or the Suite's own 3D chain) to a
model that follows it:

  ULSControlFit    a clip (or one image) onto the target: re-timed to the target frame rate, a start
                   offset, a frame-count grid (17n+5, 8n+1, 4n+1 or free), hold / loop / cut at the end,
                   and a size in steps of 32 by fit (pad), fill (crop) or stretch. A mask (inpainting)
                   goes the same way, frame for frame.
  ULSControlMap    frames -> control frames: pose (SDPose whole body), depth (Depth Anything 3), canny,
                   gray, scribble, soft edges -- or pose keypoints drawn straight (e.g. projected from a
                   rig by Mesh Ring Render: exact, no estimator).
  ULSApplyControl  a Fun ControlNet Union model patch on the MODEL line: the video family (frame-wise
                   control video, inpainting with mask + source) and the image family (one control
                   image per batch item). The family is read from the patch file; several Apply nodes
                   in a row add their controls (e.g. pose 0.7 + depth 0.3).

Everything model-side runs on the comfy.* model library only (nodes/ph_control_core.py, v1131): the
control-file loader, the video and image patches (where the branch enters the transformer, how its
skips are added), the pose drawer, SDPose and Depth Anything 3. No Core node module is imported
(guard v1063: the Core is GPL, the pack MIT).

Pose keypoints use the OpenPose frame format (canvas size + people with body 18 / feet 6 / face 70 /
hands 21+21), drawn in the OpenPose / DWPose convention the control models were trained on, so a
pose projected from a rig looks exactly like a pose detected in a video.

ASCII only. Torch / numpy / Pillow.
"""
import math
import os
import re

import numpy as np

GRIDS = ["17n+5 (video latent grid)", "8n+1", "4n+1", "any"]
END = ["hold last frame", "loop", "cut"]
SIZE_MODES = ["fit (pad)", "fill (crop)", "stretch"]
MAP_MODES = ["pose (SDPose)", "depth (Depth Anything 3)", "canny", "gray", "scribble", "soft edges",
             "keypoints (draw only)"]
POSE_TYPE = "POSE_KEYPOINT"


# ------------------------------------------------------------------------------ frame grids
def snap_count(n, grid):
    """Largest valid frame count <= n on the grid (at least the grid's smallest count)."""
    n = int(n)
    if grid.startswith("17n+5"):
        return max(5, ((n - 5) // 17) * 17 + 5) if n >= 5 else 5
    if grid.startswith("8n+1"):
        return max(1, ((n - 1) // 8) * 8 + 1)
    if grid.startswith("4n+1"):
        return max(1, ((n - 1) // 4) * 4 + 1)
    return max(1, n)


def frame_indices(n_src, src_fps, dst_fps, start_s, count, end):
    """Source frame index for each output frame (nearest in time); -> (indices, used, note)."""
    src_fps = float(src_fps) if src_fps and src_fps > 0 else float(dst_fps)
    out, note = [], ""
    k0 = start_s * src_fps
    for k in range(int(count)):
        i = int(math.floor(k0 + k * src_fps / float(dst_fps) + 1e-6))
        if i >= n_src:
            if end == "loop":
                span = max(1, n_src - int(math.floor(k0)))
                i = int(math.floor(k0)) + (i - int(math.floor(k0))) % span
                note = "looped"
            elif end == "cut":
                break
            else:
                i = n_src - 1
                note = "held the last frame"
        out.append(max(0, min(n_src - 1, i)))
    return out, note


def available_count(n_src, src_fps, dst_fps, start_s):
    src_fps = float(src_fps) if src_fps and src_fps > 0 else float(dst_fps)
    rest = max(0.0, n_src - start_s * src_fps)
    return int(math.floor(rest * float(dst_fps) / src_fps + 1e-6))


def target_size(w, h, want_w, want_h, multiple=32):
    """0 = keep the source side (or follow the other side's scale); always a multiple of `multiple`."""
    m = max(1, int(multiple))
    if want_w <= 0 and want_h <= 0:
        tw, th = w, h
    elif want_w <= 0:
        th = want_h
        tw = w * want_h / float(h)
    elif want_h <= 0:
        tw = want_w
        th = h * want_w / float(w)
    else:
        tw, th = want_w, want_h
    return max(m, int(round(tw / m)) * m), max(m, int(round(th / m)) * m)


def resize_frames(x, W, H, mode, pad=0.0):
    """x (N,H,W,C) float -> (N,H',W',C) by fit (pad) / fill (crop) / stretch."""
    import torch
    import torch.nn.functional as F
    n, h, w, c = x.shape
    t = x.movedim(-1, 1).float()
    if mode.startswith("stretch"):
        y = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
        return y.movedim(1, -1)
    s = min(W / w, H / h) if mode.startswith("fit") else max(W / w, H / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    y = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
    if mode.startswith("fit"):
        out = torch.full((n, c, H, W), float(pad), dtype=y.dtype)
        ox, oy = (W - nw) // 2, (H - nh) // 2
        out[:, :, oy:oy + nh, ox:ox + nw] = y[:, :, :H, :W]
    else:
        ox, oy = (nw - W) // 2, (nh - H) // 2
        out = y[:, :, oy:oy + H, ox:ox + W]
    return out.movedim(1, -1).contiguous()


def resize_picked(x, idx, W, H, mode, chunk=8):
    """x[idx] resized -- each source frame once, a few at a time, straight into the output (a loop or a held
    frame costs no extra resize, and the peak stays one chunk above the result)."""
    import torch
    out = torch.empty((len(idx), H, W, int(x.shape[-1])), dtype=torch.float32)
    where = {}
    for pos, i in enumerate(idx):
        where.setdefault(int(i), []).append(pos)
    uniq = sorted(where)
    for a in range(0, len(uniq), chunk):
        part = uniq[a:a + chunk]
        r = resize_frames(x[torch.as_tensor(part, dtype=torch.long)].float(), W, H, mode, 0.0)
        for j, i in enumerate(part):
            out[torch.as_tensor(where[i], dtype=torch.long)] = r[j]
    return out


def take_block(images, block, index, mask=None):
    """Block `index` of `block` frames (e.g. one direction of a ring render: directions x frames)."""
    block, index = int(block), int(index)
    if block <= 0:
        return images, mask, None
    n = int(images.shape[0])
    blocks = max(1, -(-n // block))
    if index >= blocks:
        raise ValueError("[PLS] Control Fit: block %d does not exist -- %d frames make %d block(s) of %d."
                         % (index, n, blocks, block))
    a, b = index * block, min(n, (index + 1) * block)
    if mask is not None and mask.ndim == 3 and mask.shape[0] == n:
        mask = mask[a:b]
    return images[a:b], mask, (index, blocks, a, b)


def fit(images, src_fps, dst_fps, start_s, frames, grid, end, width, height, size_mode, multiple=32, mask=None,
        block=0, block_index=0):
    """-> (images, mask or None, report dict)."""
    images, mask, blk = take_block(images, block, block_index, mask)
    n_src = int(images.shape[0])
    if n_src == 1 and frames <= 1 and grid == "any":
        idx, note = [0], ""
        count = 1
    else:
        avail = available_count(n_src, src_fps, dst_fps, start_s) if n_src > 1 else 1
        want = int(frames) if int(frames) > 0 else max(1, avail)
        if end == "cut":
            want = min(want, max(1, avail))
        count = snap_count(want, grid)
        if end == "cut" and count > max(1, avail):
            count = snap_count(max(1, avail), grid)
        idx, note = frame_indices(n_src, src_fps, dst_fps, start_s, count, end) if n_src > 1 else ([0] * count, "")
        if n_src == 1 and count > 1:
            note = "one image repeated"
    h, w = int(images.shape[1]), int(images.shape[2])
    W, H = target_size(w, h, int(width), int(height), multiple)
    out = resize_picked(images[..., :3], idx, W, H, size_mode)
    m_out = None
    if mask is not None:
        mk = mask if mask.ndim == 3 else mask[None]
        mi = [min(i, mk.shape[0] - 1) for i in idx]
        m_out = resize_picked(mk[..., None], mi, W, H, size_mode)[..., 0]
    rep = {"source_frames": n_src, "source_fps": float(src_fps) if src_fps else float(dst_fps), "fps": float(dst_fps),
           "frames": len(idx), "width": W, "height": H, "first": min(idx) if idx else 0, "last": max(idx) if idx else 0,
           "note": note, "seconds": len(idx) / float(dst_fps), "block": blk}
    return out, m_out, rep


# ------------------------------------------------------------------------------ simple maps
def compute_device():
    """The torch device of comfy.model_management (GPU when there is one); CPU outside ComfyUI."""
    try:
        import comfy.model_management as mm
        return mm.get_torch_device()
    except Exception:
        import torch
        return torch.device("cpu")


def by_chunks(fn, x, chunk=8, device=None):
    """fn over a clip a few frames at a time, straight into one output: the edge and depth maps hold several
    full-size buffers per frame, so a 141-frame clip in one go needs many GB -- in chunks only the result does.
    device: where each chunk is computed (the result is gathered on the CPU)."""
    import torch
    n = int(x.shape[0])
    out = None
    for a in range(0, n, chunk):
        c = x[a:a + chunk]
        r = fn(c.to(device) if device is not None else c).float().cpu()
        if r.shape[1:3] != x.shape[1:3]:
            r = resize_frames(r, int(x.shape[2]), int(x.shape[1]), "stretch")
        if out is None:
            out = torch.empty((n,) + tuple(r.shape[1:]), dtype=torch.float32)
        out[a:a + r.shape[0]] = r
    return out


def to_gray(x):
    g = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
    return g[..., None].expand(*g.shape, 3).contiguous()


def canny_map(x, low, high):
    import torch
    from kornia.filters import canny
    out = canny(x[..., :3].float().movedim(-1, 1), float(low), float(high))[1]
    return out.clamp(0, 1).repeat(1, 3, 1, 1).movedim(1, -1).to(torch.float32)


def soft_edges(x, sigma=1.2, gamma=0.6):
    """Sobel magnitude on a blurred luminance, normalised per frame -- a soft, HED-like edge map."""
    import torch
    import torch.nn.functional as F
    g = to_gray(x)[..., :1].movedim(-1, 1).float()
    r = max(1, int(round(3 * sigma)))
    k = torch.exp(-(torch.arange(-r, r + 1, dtype=torch.float32, device=g.device) ** 2) / (2 * sigma * sigma))
    k = k / k.sum()
    g = F.conv2d(F.pad(g, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    g = F.conv2d(F.pad(g, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1))
    sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=g.device).view(1, 1, 3, 3)
    gp = F.pad(g, (1, 1, 1, 1), mode="replicate")
    mag = torch.sqrt(F.conv2d(gp, sx) ** 2 + F.conv2d(gp, sx.transpose(2, 3)) ** 2)
    top = mag.flatten(1).quantile(0.995, dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
    e = (mag / top).clamp(0, 1) ** float(gamma)
    return e.repeat(1, 3, 1, 1).movedim(1, -1).contiguous()


def scribble(x, low=0.1, high=0.3, width=3):
    """Thick white strokes on black from a blurred canny -- the look of a hand scribble."""
    import torch.nn.functional as F
    import torch
    e = canny_map(soft_blur(x, 2.0), low, high)[..., :1].movedim(-1, 1)
    w = max(1, int(width))
    e = F.max_pool2d(e, kernel_size=2 * w + 1, stride=1, padding=w)
    return (e > 0.5).float().repeat(1, 3, 1, 1).movedim(1, -1).to(torch.float32)


def soft_blur(x, sigma):
    import torch
    import torch.nn.functional as F
    t = x[..., :3].movedim(-1, 1).float()
    r = max(1, int(round(3 * sigma)))
    k = torch.exp(-(torch.arange(-r, r + 1, dtype=torch.float32, device=t.device) ** 2) / (2 * sigma * sigma))
    k = (k / k.sum()).view(1, 1, 1, -1).repeat(3, 1, 1, 1)
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), k, groups=3)
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), k.transpose(2, 3), groups=3)
    return t.movedim(1, -1)


# ------------------------------------------------------------------------------ pose keypoints from a rig
def _side(name):
    s = re.sub(r"^(mixamorig\d*[:_])", "", str(name), flags=re.I)
    low = s.lower()
    if re.match(r"^(left|l[_.\- ])", low) or re.search(r"([_.\- ]l)$", low) or low.startswith("left"):
        return "L"
    if re.match(r"^(right|r[_.\- ])", low) or re.search(r"([_.\- ]r)$", low) or low.startswith("right"):
        return "R"
    if "left" in low:
        return "L"
    if "right" in low:
        return "R"
    return ""


def finger_key(name):
    """'mixamorig:LeftHandIndex2' -> ('L', 'index', 2); 'index_02_r' -> ('R', 'index', 2); else None."""
    s = re.sub(r"^(mixamorig\d*[:_])", "", str(name), flags=re.I)
    m = re.search(r"(thumb|index|middle|ring|pinky|little)[^0-9]*0?([1-4])", s, re.I)
    if not m:
        return None
    side = _side(s)
    if not side:
        return None
    f = m.group(1).lower()
    return side, ("pinky" if f == "little" else f), int(m.group(2))


def head_top_index(names):
    for i, n in enumerate(names):
        low = str(n).lower()
        if "headtop" in low or "head_end" in low or "head_top" in low or low.endswith("head.end"):
            return i
    return None


def map_joints(names, canonical):
    """names -> dict canonical -> index (first match wins), plus fingers and the head top."""
    out = {}
    for i, n in enumerate(names):
        c = canonical(n)
        if c is not None and c not in out:
            out[c] = i
        fk = finger_key(n)
        if fk is not None and fk not in out:
            out[fk] = i
    ht = head_top_index(names)
    if ht is not None:
        out["head top"] = ht
    for side in "LR":                                    # toe ends: '..._End' after the toe base
        for i, n in enumerate(names):
            low = str(n).lower()
            if ("toe" in low and ("end" in low or "tip" in low)) and _side(n) == side:
                out[side + " toe end"] = i
                break
    return out


def infer_humanoid(parent, P):
    """Joints of an UNNAMED humanoid skeleton (e.g. bone_000 ... from a body estimator) from the tree and the
    first frame's positions (Y up, the figure facing +Z: its right side on -X). -> dict like map_joints
    (canonical names -> index) for hips / limbs / neck / head / head top; fingers are not named."""
    P = np.asarray(P, np.float64)
    n = len(P)
    parent = [int(x) for x in parent]
    kids = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if 0 <= p < n:
            kids[p].append(i)

    def chain(i):
        out = []
        while i >= 0:
            out.append(i)
            i = parent[i]
        return out[::-1]

    def lca(a, b):
        ca, cb = chain(a), chain(b)
        k = 0
        while k < min(len(ca), len(cb)) and ca[k] == cb[k]:
            k += 1
        return ca[k - 1] if k else None

    # joints that belong to a skeleton: nodes with a parent or children, spread in space
    pts = [i for i in range(n) if kids[i] or parent[i] >= 0]
    if len(pts) < 10:
        return {}
    ys = P[pts, 1]
    y0, y1 = float(ys.min()), float(ys.max())
    h = (y1 - y0) or 1.0
    leaves = [i for i in pts if not kids[i]]
    cx = float(np.median(P[pts, 0]))
    out = {}
    feet = {}
    for side, sgn in (("R", -1.0), ("L", 1.0)):
        cand = [i for i in leaves if P[i, 1] < y0 + 0.15 * h and sgn * (P[i, 0] - cx) > 0.01 * h]
        if cand:
            feet[side] = min(cand, key=lambda i: P[i, 1])
    hands = {}
    for side, sgn in (("R", -1.0), ("L", 1.0)):
        cand = [i for i in leaves if y0 + 0.25 * h < P[i, 1] < y0 + 0.9 * h and sgn * (P[i, 0] - cx) > 0.05 * h]
        if cand:
            hands[side] = max(cand, key=lambda i: sgn * (P[i, 0] - cx))
    if len(feet) == 2:
        pelvis = lca(feet["R"], feet["L"])
        if pelvis is not None:
            out["hips"] = pelvis
        for side, leaf in feet.items():
            ch = chain(leaf)
            ch = ch[ch.index(pelvis) + 1:] if pelvis in ch else ch
            lat = [i for i in ch if abs(P[i, 0] - P[pelvis, 0]) > 0.03 * h] if pelvis is not None else ch
            if len(lat) < 3:
                continue
            hip = lat[0]
            above = [i for i in lat[1:-1] if P[i, 1] - P[leaf, 1] >= 0.03 * h]
            ankle = min(above, key=lambda i: P[i, 1]) if above else lat[-2]
            mid = 0.5 * (P[hip, 1] + P[ankle, 1])
            between = lat[lat.index(hip) + 1:lat.index(ankle)] if lat.index(ankle) > lat.index(hip) + 1 else []
            if between:
                knee = min(between, key=lambda i: abs(P[i, 1] - mid))
                out[side + " thigh"], out[side + " shin"], out[side + " foot"] = hip, knee, ankle
                after = lat[lat.index(ankle) + 1:]
                if after:
                    out[side + " toe"] = after[0]
                out[side + " toe end"] = leaf
    chest = lca(hands["R"], hands["L"]) if len(hands) == 2 else None
    if chest is not None:
        for side, leaf in hands.items():
            ch = chain(leaf)
            ch = ch[ch.index(chest) + 1:]
            wrist = None
            for i in ch:
                if len(kids[i]) >= 3:
                    wrist = i
                    break
            if wrist is None:
                if len(ch) < 3:
                    continue
                wrist = ch[-2] if len(ch) > 3 else ch[-1]
            arm = ch[:ch.index(wrist) + 1]
            lat = [i for i in arm if abs(P[i, 0] - P[chest, 0]) > 0.06 * h]
            if len(lat) < 2:
                continue
            shoulder = lat[0]
            seg = arm[arm.index(shoulder) + 1:arm.index(wrist)]
            if not seg:
                continue
            d_s = np.linalg.norm(P[seg] - P[shoulder], axis=1)
            d_w = np.linalg.norm(P[seg] - P[wrist], axis=1)
            elbow = seg[int(np.argmin(np.abs(d_s - d_w)))]
            out[side + " upperarm"], out[side + " forearm"], out[side + " hand"] = shoulder, elbow, wrist
        top = max(leaves, key=lambda i: P[i, 1])
        ch = chain(top)
        if chest in ch:
            up = ch[ch.index(chest) + 1:]
            if up:
                out["head top"] = top
                cand = [i for i in up[:-1] if P[top, 1] - P[i, 1] >= 0.08 * h] or up[:1]
                head = max(cand, key=lambda i: P[i, 1])
                out["head"] = head
                if up[0] != head:
                    out["neck"] = up[0]
    return out


_HAND_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def body_points(P, jm):
    """3D keypoints of one frame: P (N,3) joint positions, jm from map_joints -> dict name -> (xyz, score).
    Body follows OpenPose 18; eyes / ears / nose are placed from the head joint, the shoulder line and the
    spine (no face joints in a rig)."""
    pt = {}

    def g(k):
        return P[jm[k]] if k in jm else None

    for side, so in (("R", "R"), ("L", "L")):
        for key, can in (("shoulder", "upperarm"), ("elbow", "forearm"), ("wrist", "hand"),
                         ("hip", "thigh"), ("knee", "shin"), ("ankle", "foot")):
            p = g(so + " " + can)
            if p is not None:
                pt[side + " " + key] = (p, 1.0)
    if "R shoulder" in pt and "L shoulder" in pt:
        pt["neck"] = ((pt["R shoulder"][0] + pt["L shoulder"][0]) * 0.5, 1.0)
    elif g("neck") is not None:
        pt["neck"] = (g("neck"), 1.0)
    head = g("head")
    if head is not None and "neck" in pt and "R shoulder" in pt and "L shoulder" in pt:
        right = pt["R shoulder"][0] - pt["L shoulder"][0]
        sw = float(np.linalg.norm(right)) or 1.0
        right = right / sw
        top = g("head top")
        upv = (top - head) if top is not None else (head - pt["neck"][0])
        hh = float(np.linalg.norm(upv)) or 0.5 * sw
        up = upv / hh
        right = right - up * float(np.dot(right, up))
        right = right / (np.linalg.norm(right) or 1.0)
        fwd = np.cross(up, right)
        if top is None:
            hh = 0.9 * hh
        c = head + up * 0.45 * hh
        pt["nose"] = (c + fwd * 0.55 * hh - up * 0.05 * hh, 1.0)
        pt["R eye"] = (c + fwd * 0.45 * hh + up * 0.12 * hh + right * 0.18 * hh, 1.0)
        pt["L eye"] = (c + fwd * 0.45 * hh + up * 0.12 * hh - right * 0.18 * hh, 1.0)
        pt["R ear"] = (c + right * 0.42 * hh, 1.0)
        pt["L ear"] = (c - right * 0.42 * hh, 1.0)
        pt["_frame"] = (np.stack([right, up, fwd]), 1.0)
    for side in "LR":
        for t in ("toe", "toe end"):
            p = g(side + " " + t)
            if p is not None:
                pt[side + " " + t] = (p, 1.0)
        for f in _HAND_ORDER:
            for k in range(1, 5):
                p = g((side, f, k))
                if p is not None:
                    pt[(side, f, k)] = (p, 1.0)
    return pt


_BODY18 = ["nose", "neck", "R shoulder", "R elbow", "R wrist", "L shoulder", "L elbow", "L wrist",
           "R hip", "R knee", "R ankle", "L hip", "L knee", "L ankle", "R eye", "L eye", "R ear", "L ear"]


def keypoint_frame(pt, project2d, view_dir, W, H):
    """One person's 3D points -> an OpenPose frame dict at canvas W x H. project2d(p) -> (x, y);
    view_dir = the camera's looking direction (unit). Face points facing away get score 0."""
    toward = -np.asarray(view_dir, np.float64)
    fr = pt.get("_frame")
    vis = {}
    if fr is not None:
        right, up, fwd = fr[0]
        front = float(np.dot(fwd, toward))
        side_r = float(np.dot(right, toward))
        vis = {"nose": front > -0.2, "R eye": front > -0.2 and side_r > -0.75, "L eye": front > -0.2 and side_r < 0.75,
               "R ear": side_r > -0.55, "L ear": side_r < 0.55}

    def flat(names):
        out = []
        for n in names:
            if n in pt and vis.get(n, True):
                x, y = project2d(pt[n][0])
                out += [float(x), float(y), 1.0]
            else:
                out += [0.0, 0.0, 0.0]
        return out

    body = flat(_BODY18)
    feet = []
    for side in "LR":
        toe = side + " toe end" if (side + " toe end") in pt else side + " toe"
        heel = None
        if (side + " ankle") in pt and toe in pt:
            a, t = pt[side + " ankle"][0], pt[toe][0]
            heel = a - (t - a) * 0.25
            heel = np.array([heel[0], min(heel[1], a[1]), heel[2]])
        for p in (pt.get(toe, (None,))[0], pt.get(side + " toe", (None,))[0], heel):
            if p is None:
                feet += [0.0, 0.0, 0.0]
            else:
                x, y = project2d(p)
                feet += [float(x), float(y), 1.0]
    hands = {}
    for side in "RL":
        hv = []
        wrist = side + " wrist"
        names = [wrist] + [(side, f, k) for f in _HAND_ORDER for k in range(1, 5)]
        have = sum(1 for n in names[1:] if n in pt)
        for n in names:
            if have >= 10 and n in pt:
                x, y = project2d(pt[n][0])
                hv += [float(x), float(y), 1.0]
            else:
                hv += [0.0, 0.0, 0.0]
        hands[side] = hv
    face = [0.0] * (70 * 3)
    return {"pose_keypoints_2d": body, "foot_keypoints_2d": feet, "face_keypoints_2d": face,
            "hand_right_keypoints_2d": hands["R"], "hand_left_keypoints_2d": hands["L"]}


def _core():
    """The model-side runtime (nodes/ph_control_core.py), as a package module or a plain one."""
    try:
        from . import ph_control_core
    except ImportError:
        import ph_control_core
    return ph_control_core


def draw_keypoints(frames, hands=True, face=False, feet=False, stick_width=4, point_size=3, threshold=0.3, head=True):
    """POSE_KEYPOINT list -> IMAGE (N,H,W,3) in the OpenPose / DWPose drawing convention (own drawer)."""
    draw_pose = _core().draw_pose
    return draw_pose(frames, bool(hands), bool(face), bool(feet), int(stick_width), int(point_size),
                     float(threshold), bool(head))


# ------------------------------------------------------------------------------ model caches (comfy.* loaders)
_CACHE = {}


def _cached(key, make):
    if key not in _CACHE:
        if len(_CACHE) > 4:
            _CACHE.clear()
        _CACHE[key] = make()
    return _CACHE[key]


def load_sdpose(name):
    import folder_paths
    import comfy.sd
    path = resolve_file("checkpoints", name, "Control Map", "pose_model")      # v1216: door label + door files

    def make():
        out = comfy.sd.load_checkpoint_guess_config(path, output_vae=True, output_clip=False,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out[0], out[2]
    return _cached(("sdpose", path, os.path.getmtime(path)), make)


def load_da3(name):
    ph_da3_lib = _core().da3_lib()
    ph_da3_lib.ensure_folder()
    name = _ww().strip(name)                                                   # v1216: the door's label
    path = resolve_file("geometry_estimation", name, "Control Map", "depth_model")
    return _cached(("da3", path, os.path.getmtime(path)), lambda: ph_da3_lib.load(name))


def load_patch(name):
    load_control_file = _core().load_control_file
    path = resolve_file("model_patches", name, "Apply Control", "control_model")   # v1216: door label + door files
    return _cached(("patch", path, os.path.getmtime(path)), lambda: load_control_file(path))


def sdpose_keypoints(image, name, batch=16):
    sdpose_frames = _core().sdpose_frames
    model, vae = load_sdpose(name)
    return sdpose_frames(model, vae, image[..., :3], int(batch))


def da3_depth(image, name, resolution=504):
    depth_frames = _core().depth_frames
    return depth_frames(load_da3(name), image[..., :3], int(resolution))


# ------------------------------------------------------------------------------ apply (patch selection)
def patch_family(model_patch):
    m = getattr(model_patch, "model", None)
    mod = type(m).__module__ if m is not None else ""
    if "minimax" in mod:
        return "video"
    if "qwen_image21" in mod:
        return "image"
    return None


def model_family(model):
    try:
        dm = model.get_model_object("diffusion_model")
    except Exception:
        dm = getattr(getattr(model, "model", None), "diffusion_model", None)
    mod = type(dm).__module__ if dm is not None else ""
    if "minimax" in mod:
        return "video"
    if "qwen_image21" in mod:
        return "image"
    return None


def describe_patch(model_patch):
    m = model_patch.model
    fam = patch_family(model_patch)
    if fam == "video":
        n = len(m.injection_layers)
        return "video Fun ControlNet Union %s: %d control blocks at layers %s%s" % (
            "2.x" if n >= 10 else "1.x", n, list(m.injection_layers),
            ", inpaint post-norm" if getattr(m, "inpaint_post_norm", False) else "")
    if fam == "image":
        return "image Fun ControlNet Union: %d control blocks" % len(m.control_blocks)
    return "unknown model patch (%s)" % type(m).__name__


def _image_patch_class():
    """The image-family patch: one control per batch item (own runtime, ph_control_core)."""
    ImageControlPatch = _core().ImageControlPatch
    return ImageControlPatch


def pick_frame(pick, control=None, mask=None, source=None):
    """pick -1: unchanged. pick k: frame k of each given batch, as a batch of one (k past the end: error)."""
    if pick < 0:
        return control, mask, source
    out = []
    for name, t in (("control", control), ("mask", mask), ("source", source)):
        if t is not None:
            if t.dim() == 2:
                t = t.unsqueeze(0)
            if pick >= t.shape[0]:
                raise ValueError("[PLS] Apply Control: pick %d, but the %s has only %d frame(s) (0 ... %d)." % (
                    pick, name, t.shape[0], t.shape[0] - 1))
            t = t[pick:pick + 1]
        out.append(t)
    return tuple(out)


def apply(model, model_patch, vae, strength, start, end, control=None, mask=None, source=None):
    """-> (patched model, report). Video: frame by frame. Image: per batch item (ph_control_core)."""
    fam = patch_family(model_patch)
    mfam = model_family(model)
    if fam is None:
        raise ValueError("[PLS] Apply Control: this model patch is not a Fun ControlNet Union (%s)." %
                         type(getattr(model_patch, "model", None)).__name__)
    if mfam is not None and mfam != fam:
        raise ValueError("[PLS] Apply Control: the control model is for the %s family, the model is a %s model -- "
                         "pick the matching control file." % (fam, mfam))
    if float(strength) == 0 or (control is None and mask is None):
        return model, {"family": fam, "skipped": "strength 0" if float(strength) == 0 else "no control and no mask"}
    m = model.clone()
    ms = model.get_model_object("model_sampling")
    s0, s1 = float(ms.percent_to_sigma(float(start))), float(ms.percent_to_sigma(float(end)))
    if fam == "video":
        VideoControlPatch = _core().VideoControlPatch
        patch = VideoControlPatch(
            model_patch, vae,
            control[..., :3].movedim(-1, 1) if control is not None else None,
            mask,
            source[..., :3].movedim(-1, 1) if (mask is not None and source is not None) else None,
            float(strength), s0, s1)
        patch.register(m)
    else:
        cls = _image_patch_class()
        patch = cls(model_patch, vae, control[..., :3] if control is not None else None, float(strength),
                    source=source[..., :3] if (source is not None and mask is not None) else None,
                    mask=mask, sigma_range=(s0, s1))
        patch.register(m)
    rep = {"family": fam, "frames": int(control.shape[0]) if control is not None else 0,
           "size": (int(control.shape[2]), int(control.shape[1])) if control is not None else None,
           "inpaint": mask is not None, "sigma": (s0, s1)}
    return m, rep


# ------------------------------------------------------------------------------ nodes
def _files(kind):
    try:
        import folder_paths
        return folder_paths.get_filename_list(kind) or []
    except Exception:
        return []


def _pick(files, *words):
    for f in files:
        low = f.lower()
        if all(w in low for w in words):
            return f
    return files[0] if files else "(none)"


# ------------------------------------------------------------------------------ v1216 the model door
# Frank 23.10. (the first Control field run): "Auf jeden Fall braucht es wohl Modelle ... ausserdem muessen hier
# wohl auch offiziell Download-Optionen mit rein". The control files join the pack's ONE download door
# (ph_world_weights.REGISTRY, sets 'control' and 'sdpose'; DA3 was there already): the combos list the door's files
# in the house form ('file . size', a missing one '. not installed'), the node's line offers the Download of the
# PICKED file only while the mode needs it (web/js/ph_world_weights.js), a run never downloads.
def _ww():
    try:
        from . import ph_world_weights as WW
    except ImportError:              # flat import in the guards
        import ph_world_weights as WW
    return WW


def door_choices(folder, set_name, files=None):
    """The combo: the door's files of `set_name` first (labelled), then every other file ComfyUI lists in the
    folder (a file someone placed by hand), plain. Never empty."""
    WW = _ww()
    files = _files(folder) if files is None else list(files)
    reg = [n for n, s in WW.REGISTRY.items() if s.get("set") == set_name]
    out = [WW.label(n) for n in reg]
    out += [f for f in files if os.path.basename(f.replace("\\", "/")) not in reg]
    return out or ["(none)"]


def resolve_file(folder, value, node, what):
    """A combo value -> the file's path. Strips the door's label; a door file not on disk is refused with the
    way to the Download, '(none)' with what is missing."""
    WW = _ww()
    name = WW.strip(value)
    if not name or name.startswith("("):
        raise ValueError("[PLS] %s: %s -- no file picked (models/%s)." % (node, what, folder))
    if name in WW.REGISTRY:
        path = WW.local_path(name)
        if not path:
            spec = WW.REGISTRY[name]
            raise ValueError("[PLS] %s: %s (%s, %s) is not installed -- the Download on the node's line fetches it "
                             "into models/%s (or put the file from %s there)."
                             % (node, spec["label"], name, WW.fmt_bytes(spec["bytes"]), folder, spec["url"]))
        return path
    import folder_paths
    return folder_paths.get_full_path_or_raise(folder, name)


# ------------------------------------------------------------------------------ v1217 the preview in the node
# Frank 23.10.: "Ich will bei OpenPose bzw. generell bei Controlnet eine Preview haben ... entweder direkt in der Node
# oder auch gerne als Ausgang fuer einen Preview bzw. Download, denn diese Control-Net-Bilder bzw. Videos koennen auch
# wieder als Vorlage dienen". The output was there (`control`, IMAGE -> any Save node); the node now SHOWS what it
# made: a clip as an animated WebP in ComfyUI's temp folder (Core's own preview form, {"images", "animated"}), a
# single frame as a PNG. Small on purpose -- long side PREVIEW_SIDE, at most PREVIEW_MAX frames (evenly picked, the
# speed kept) -- it is a look, not the file; the file is the Save node's job.
PREVIEW_SIDE = 384
PREVIEW_MAX = 240
PREVIEW_FPS = 24.0


def preview_frames(frames, side=PREVIEW_SIDE, cap=PREVIEW_MAX):
    """IMAGE [N,H,W,3] -> (list of uint8 HxWx3 arrays, ms per frame). Pure: no ComfyUI."""
    import torch
    x = frames[..., :3].detach().float().cpu().clamp(0, 1)
    n = int(x.shape[0])
    idx = list(range(n)) if n <= cap else [int(round(i * (n - 1) / float(cap - 1))) for i in range(cap)]
    ms = int(round(1000.0 / PREVIEW_FPS * (n / float(len(idx)))))
    h, w = int(x.shape[1]), int(x.shape[2])
    s = min(1.0, float(side) / float(max(h, w)))
    nh, nw = max(2, int(round(h * s))), max(2, int(round(w * s)))
    sel = x[idx].movedim(-1, 1)
    if (nh, nw) != (h, w):
        sel = torch.nn.functional.interpolate(sel, size=(nh, nw), mode="area")
    arr = (sel.movedim(1, -1).numpy() * 255.0 + 0.5).astype(np.uint8)
    return [a for a in arr], ms


def preview_ui(frames, stem, folder=None):
    """Write the preview into ComfyUI's temp folder -> the node's ui dict, or None (never raises into the run)."""
    try:
        import uuid
        from PIL import Image
        if folder is None:
            import folder_paths
            folder = folder_paths.get_temp_directory()
        os.makedirs(folder, exist_ok=True)
        pics, ms = preview_frames(frames)
        ims = [Image.fromarray(a) for a in pics]
        if len(ims) == 1:
            name = "%s_%s.png" % (stem, uuid.uuid4().hex[:12])
            ims[0].save(os.path.join(folder, name))
            return {"images": [{"filename": name, "subfolder": "", "type": "temp"}]}
        name = "%s_%s.webp" % (stem, uuid.uuid4().hex[:12])
        ims[0].save(os.path.join(folder, name), save_all=True, append_images=ims[1:], duration=ms, loop=0,
                    quality=80, method=4)
        return {"images": [{"filename": name, "subfolder": "", "type": "temp"}], "animated": [True]}
    except Exception as e:
        print("[PLS] %s: preview not written (%r) -- the frames themselves are fine" % (stem, e))
        return None


class ULSControlFit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "A clip (frames) or one image -- e.g. a stock video from the Media Loader."}),
                "target_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.5,
                    "tooltip": "Frame rate of the model's video. The clip is re-timed to it (nearest frame)."}),
                "source_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.01,
                    "tooltip": "Frame rate of the clip. 0 = take it from the fps input, else the target rate."}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01,
                    "tooltip": "Seconds into the clip where the control starts."}),
                "frames": ("INT", {"default": 0, "min": 0, "max": 100000,
                    "tooltip": "Frames wanted (snapped down to the grid). 0 = as many as the clip gives from start."}),
                "grid": (GRIDS, {"default": GRIDS[0],
                    "tooltip": "Valid frame counts of the target model: 17n+5 (5, 22, 39 ... 124 ... 362), 8n+1, "
                               "4n+1 or any."}),
                "at_the_end": (END, {"default": END[0],
                    "tooltip": "When the clip is shorter than the frames wanted: hold its last frame, loop it, or "
                               "cut the count down."}),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 32,
                    "tooltip": "Target width. 0 = from the clip (or from height, keeping the aspect)."}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 32,
                    "tooltip": "Target height. 0 = from the clip (or from width, keeping the aspect)."}),
                "size_mode": (SIZE_MODES, {"default": SIZE_MODES[1],
                    "tooltip": "fit: whole clip inside, black bars. fill: fills the frame, crops the overhang. "
                               "stretch: distorts to the size."}),
                "multiple": ("INT", {"default": 32, "min": 1, "max": 128,
                    "tooltip": "Sides are rounded to a multiple of this (32 suits the video and image models)."}),
                "block": ("INT", {"default": 0, "min": 0, "max": 100000,
                    "tooltip": "Take one block of this many frames first (0 = all). A ring render is directions x "
                               "frames: block = the frames per direction picks one direction; 1 picks single images."}),
                "block_index": ("INT", {"default": 0, "min": 0, "max": 100000,
                    "tooltip": "Which block (0 = the first; for a ring: 0 = S, 1 = SE, 2 = E ... in the ring's order)."}),
            },
            "optional": {
                "fps": ("FLOAT", {"forceInput": True, "tooltip": "The clip's frame rate, e.g. from the Media Loader."}),
                "mask": ("MASK", {"tooltip": "Goes the same way, frame for frame (inpainting: white = regenerate)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("images", "mask", "frame_count", "width", "height", "seconds", "info")
    OUTPUT_TOOLTIPS = ("The fitted frames.", "The fitted mask (empty when none came in).",
                       "Frames -- e.g. for the latent length.", "Width.", "Height.",
                       "Length at the target rate.", "What was taken and how.")
    FUNCTION = "fit"
    CATEGORY = "Polyhedron/Control"
    DESCRIPTION = ("Fits a clip or an image to a model's video: re-timed to the target frame rate, a start offset, "
                   "a valid frame count (17n+5, 8n+1, 4n+1 or free), hold / loop / cut at the end, and a size in "
                   "steps of 32 (fit, fill or stretch). A mask follows frame for frame. A block picks one part "
                   "first (one direction of a ring render, one image of a batch). Use it for stock videos as "
                   "control templates and for inpainting sources.")

    def fit(self, images, target_fps, source_fps, start, frames, grid, at_the_end, width, height, size_mode,
            multiple, block=0, block_index=0, fps=None, mask=None):
        import torch
        try:          # v1216: no control from the loader's empty frame
            from .uls_placeholder import refuse
        except ImportError:  # flat import in the guards
            from uls_placeholder import refuse
        refuse("Control Fit", "images", images)
        src = float(source_fps) if float(source_fps) > 0 else (float(fps) if fps else float(target_fps))
        out, m_out, rep = fit(images, src, float(target_fps), float(start), int(frames), grid, at_the_end,
                              int(width), int(height), size_mode, int(multiple), mask, int(block), int(block_index))
        if m_out is None:
            m_out = torch.zeros((out.shape[0], out.shape[1], out.shape[2]))
        info = ("Control Fit: %d frame(s) %dx%d at %.2f fps (%.3f s) from %d source frame(s) at %.2f fps, frames "
                "%d-%d, %s, %s%s" % (rep["frames"], rep["width"], rep["height"], rep["fps"], rep["seconds"],
                                     rep["source_frames"], rep["source_fps"], rep["first"], rep["last"], grid,
                                     size_mode, (" -- " + rep["note"]) if rep["note"] else ""))
        if rep["block"]:
            info += "; block %d of %d (source frames %d-%d)" % (rep["block"][0], rep["block"][1], rep["block"][2],
                                                                 rep["block"][3] - 1)
        print("[PLS] " + info)
        return (out, m_out, rep["frames"], rep["width"], rep["height"], rep["seconds"], info)


class ULSControlMap:
    @classmethod
    def INPUT_TYPES(cls):
        ck = door_choices("checkpoints", "sdpose")                 # v1216: the door's SDPose first
        ge = _ww().da3_choices(_files("geometry_estimation"))      # v1216: the door's DA3 files (as World / Depth)
        return {
            "required": {
                "mode": (MAP_MODES, {"default": MAP_MODES[0],
                    "tooltip": "pose: whole-body skeleton found in the frames (SDPose). depth: near = bright (Depth "
                               "Anything 3). canny / scribble / soft edges: lines. gray: the frames in grey (recolour "
                               "control). keypoints: draw the keypoints input (e.g. from Mesh Ring Render) -- exact."}),
                "pose_model": (ck or ["(none)"], {"default": _pick(ck, "sdpose"),
                    "tooltip": "Pose: the SDPose checkpoint (models/checkpoints). A file marked 'not installed' is "
                               "fetched by the Download on the node's line (shown while mode is pose)."}),
                "depth_model": (ge or ["(none)"], {"default": _pick(ge, "depth_anything_3"),
                    "tooltip": "Depth: a Depth Anything 3 model (models/geometry_estimation). A file marked 'not "
                               "installed' is fetched by the Download on the node's line (shown while mode is "
                               "depth)."}),
                "draw_hands": ("BOOLEAN", {"default": True, "tooltip": "Pose: draw the hands."}),
                "draw_face": ("BOOLEAN", {"default": False, "tooltip": "Pose: draw the face landmarks (detected poses only)."}),
                "stick_width": ("INT", {"default": 4, "min": 1, "max": 16, "tooltip": "Pose: limb thickness in pixels."}),
                "low": ("FLOAT", {"default": 0.2, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "canny / scribble: lower threshold."}),
                "high": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01,
                    "tooltip": "canny / scribble: upper threshold."}),
                "depth_resolution": ("INT", {"default": 504, "min": 140, "max": 2520, "step": 14,
                    "tooltip": "Depth: working size of the model (longest side)."}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "The frames to read the control from (fitted clip or image)."}),
                "keypoints": (POSE_TYPE, {"tooltip": "Pose keypoints to draw (keypoints mode), e.g. from Mesh Ring "
                                                     "Render -- the skeleton of a rig, projected exactly."}),
                # v1217: APPENDED (#577 law) -- the control frames shown in the node
                "preview": ("BOOLEAN", {"default": True,
                    "tooltip": "Show the control frames in the node (a clip plays as a small animated WebP, an image "
                               "as a still). To keep them -- as a template for another run -- wire `control` into "
                               "a Save node."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, pose_model=None, depth_model=None):
        """v1216: the two model combos carry the door's label ('file . size' / '. not installed') -- a saved pick
        whose size changed on disk is a string the list no longer offers. ONLY these two inputs leave Core's
        membership check; resolve_file() strips the label and refuses a missing file by name."""
        return True

    RETURN_TYPES = ("IMAGE", POSE_TYPE, "STRING")
    RETURN_NAMES = ("control", "keypoints", "info")
    OUTPUT_TOOLTIPS = ("Control frames -- into Apply Control.", "The pose keypoints (pose and keypoints modes).",
                       "What was made.")
    FUNCTION = "make"
    CATEGORY = "Polyhedron/Control"
    DESCRIPTION = ("Turns frames into control frames: whole-body pose (SDPose), depth (Depth Anything 3), canny, "
                   "gray, scribble or soft edges -- or draws pose keypoints that came in exactly (from the 3D chain). "
                   "One node for every control a Fun ControlNet Union reads.")

    def make(self, mode, pose_model, depth_model, draw_hands, draw_face, stick_width, low, high, depth_resolution,
             images=None, keypoints=None, preview=True):
        import torch
        kp_out = keypoints
        note = ""
        if mode.startswith("keypoints"):
            if not keypoints:
                raise ValueError("[PLS] Control Map: keypoints mode needs the keypoints input (e.g. from Mesh Ring "
                                 "Render).")
            out = draw_keypoints(keypoints, draw_hands, draw_face, False, stick_width)
            note = "%d frame(s) drawn from keypoints" % len(keypoints)
        else:
            if images is None:
                raise ValueError("[PLS] Control Map: connect images (or use keypoints mode).")
            x = images[..., :3].float()
            dev = compute_device()
            if mode.startswith("pose"):
                if _ww().strip(pose_model) in ("(none)", "", None):
                    raise ValueError("[PLS] Control Map: pose needs the SDPose checkpoint in models/checkpoints "
                                     "(sdpose_wholebody_fp16.safetensors).")
                kp_out = sdpose_keypoints(x, pose_model)
                out = draw_keypoints(kp_out, draw_hands, draw_face, False, stick_width)
                found = sum(len(f.get("people", [])) for f in kp_out)
                note = "%d frame(s), %d person detection(s)" % (len(kp_out), found)
                if found == 0:
                    note += " -- no person found: the control frames are black"
            elif mode.startswith("depth"):
                if _ww().strip(depth_model) in ("(none)", "", None):
                    raise ValueError("[PLS] Control Map: depth needs a Depth Anything 3 model in "
                                     "models/geometry_estimation.")
                out = by_chunks(lambda c: da3_depth(c, depth_model, depth_resolution), x)
                note = "%d frame(s), near = bright" % x.shape[0]
            elif mode == "canny":
                out = by_chunks(lambda c: canny_map(c, low, high), x, device=dev)
                note = "thresholds %.2f / %.2f" % (low, high)
            elif mode == "gray":
                out = by_chunks(to_gray, x, device=dev)
            elif mode == "scribble":
                out = by_chunks(lambda c: scribble(c, low, high, max(1, stick_width // 2)), x, device=dev)
            else:
                out = by_chunks(soft_edges, x, device=dev)
            if out.shape[1:3] != x.shape[1:3]:
                out = resize_frames(out, int(x.shape[2]), int(x.shape[1]), "stretch")
        out = out.float().clamp_(0, 1)
        info = "Control Map: %s, %d frame(s) %dx%d%s" % (mode, out.shape[0], out.shape[2], out.shape[1],
                                                         (" -- " + note) if note else "")
        print("[PLS] " + info)
        result = (out.to(torch.float32), kp_out or [], info)
        ui = preview_ui(out, "ph_control_map") if preview else None
        if not ui:
            return result
        return {"ui": ui, "result": result}


class ULSApplyControl:
    @classmethod
    def INPUT_TYPES(cls):
        mp = door_choices("model_patches", "control")              # v1216: the door's Fun ControlNet Union files first
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model to steer (after LoRAs, before the sampler)."}),
                "vae": ("VAE", {"tooltip": "The model's (video) VAE -- it encodes the control frames."}),
                "control_model": (mp or ["(none)"], {"default": _pick(mp, "fun", "union"),
                    "tooltip": "A Fun ControlNet Union file in models/model_patches. The family (video or image) is "
                               "read from the file. A file marked 'not installed' is fetched by the Download on the "
                               "node's line."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01,
                    "tooltip": "How hard the control pulls. Several Apply nodes add up: keep their sum near 1 "
                               "(e.g. pose 0.7 + depth 0.3)."}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction of the sampling where the control starts (0 = from the first step)."}),
                "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction where it stops. Around 0.6 leaves the last steps free for texture."}),
                "pick": ("INT", {"default": -1, "min": -1, "max": 100000,
                    "tooltip": "-1 = the whole control (video: frame by frame; image batch: item by item). 0, 1, 2 "
                               "... = only that frame of the control (and of mask / source) -- e.g. one direction "
                               "of a ring render for one row of a turntable (0 = S, 1 = SE ...)."}),
            },
            "optional": {
                "control": ("IMAGE", {"tooltip": "Control frames (Control Map). Video: one per frame; image batch: one "
                                                 "per item, or one for all."}),
                "mask": ("MASK", {"tooltip": "Inpainting: white = regenerate under the control, black = keep."}),
                "source": ("IMAGE", {"tooltip": "Inpainting: the frames behind the mask (kept where it is black)."}),
                "control_patch": ("MODEL_PATCH", {"tooltip": "Overrides control_model: a patch loaded elsewhere."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, control_model=None):
        """v1216: control_model carries the door's label -- out of Core's membership check, stripped and resolved
        (a missing file refused by name) in load_patch()."""
        return True

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    OUTPUT_TOOLTIPS = ("The model with the control -- into the sampler (or the next Apply Control).",
                       "Control model, family, frames, range.")
    FUNCTION = "apply"
    CATEGORY = "Polyhedron/Control"
    DESCRIPTION = ("Puts a Fun ControlNet Union on the MODEL line: pose, depth, edges, scribble or gray frames steer "
                   "a video model frame by frame, or an image model per batch item; with mask + source it "
                   "regenerates only the masked part. The family is read from the control file; chain several to "
                   "combine controls.")

    def apply(self, model, vae, control_model, strength, start, end, pick=-1, control=None, mask=None, source=None,
              control_patch=None):
        if float(end) < float(start):
            raise ValueError("[PLS] Apply Control: end (%.2f) lies before start (%.2f)." % (end, start))
        control, mask, source = pick_frame(int(pick), control, mask, source)
        if control_patch is None:
            if _ww().strip(control_model) in ("(none)", "", None):
                raise ValueError("[PLS] Apply Control: no Fun ControlNet Union file in models/model_patches.")
            control_patch = load_patch(control_model)
        m, rep = apply(model, control_patch, vae, float(strength), float(start), float(end), control, mask, source)
        what = describe_patch(control_patch)
        if "skipped" in rep:
            info = "Apply Control: %s -- not applied (%s)" % (what, rep["skipped"])
        else:
            info = ("Apply Control: %s; strength %.2f, steps %.0f-%.0f %%; %s%s" % (
                what, float(strength), 100 * float(start), 100 * float(end),
                ("%d control frame(s) %dx%d" % (rep["frames"], rep["size"][0], rep["size"][1])) if rep["frames"]
                else "no control frames", "; inpainting with mask" if rep["inpaint"] else ""))
            if rep["family"] == "video" and rep["frames"]:
                info += " (fitted to the latent's length at run time: cut, or the last frame held)"
                if rep["frames"] != snap_count(rep["frames"], GRIDS[0]):
                    info += " -- %d frames are off the 17n+5 grid (Control Fit snaps them)" % rep["frames"]
        print("[PLS] " + info)
        return (m, info)
