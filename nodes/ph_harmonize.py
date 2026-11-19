# -*- coding: ascii -*-
"""Harmonize (v1322) -- carry the colour of one area onto another.

Frank 08.11.: "harmonize inside the mask" -- a painted area (inpaint, Qwen fill, a pasted door) takes the colour of
its surroundings; and the same move for whole pictures: reference stills take the colour of the photo they are
meant to fit (the Alien bridge: grey film stills against the amber set photo).

Two moves, both in Oklab (perceptual: L lightness, a green-red, b blue-yellow):

  to reference -- a mean/spread transfer measured ONCE over the whole batch (a clip does not flicker):
      lab' = (lab - mean_source) * k + mean_target,   k = spread_target / spread_source, held in SPREAD_LIMITS
      source = the whole image, or the inside of the mask (mask > 0.5); target = the reference image(s)

  mask to its ring -- the light of the surroundings, measured at the seam (Frank's Qwen back wall, 08.11.):
      colour: a white balance in linear RGB -- per-channel gains that move the MEDIAN chromaticity of the inside
              onto the median chromaticity of a ring of `ring` px around the mask (RING_GAP px off the edge, so a
              spilled seam is not "surroundings"); the median keeps the ring's own light sources (saturated lamps)
              from dominating;
      light:  one factor -- the median brightness ratio, outside to inside, of the local averages along the edge.
      A white door painted inside stays white; only the light falling on it changes. (A plain mean transfer here
      darkened the whole back wall by L -0.13 and a seam field smeared at content edges -- measured 08.11.; this
      lands on Frank's hand-matched version within L 0.004 / a 0.002 / b 0.005.)

      seam (v1323): what is left at the edge is measured locally along it and shared by both sides of the seam
      over `seam` px (seam_soften) -- pixels just outside the mask change too, the step becomes a gradient.

  Both are measured ONCE over the whole batch (a clip does not flicker) and applied to every frame:
    result = image + amount * weight * (moved - image)    weight = the (soft) mask, or 1 without a mask

The result is linear in `amount` -- the Filter's preview blends it live. Pure numpy; the dilation uses torch when it
is importable (ComfyUI always has it), else a numpy shift loop.
"""

import numpy as np

MODES = ("auto", "off", "to reference", "mask to its ring")
TONES = ("colour and light", "colour only")
SPREAD_LIMITS = (0.5, 2.0)
GAIN_LIMITS = (0.25, 4.0)   # white balance and light factor of the ring mode (a safety rail, not a style)
DARK_Y = 0.01               # linear luminance below which a pixel says nothing about the light
_Y = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
RING_GAP = 2
SEAM_BAND = 2           # px measured on each side of the seam by seam_soften
STAT_FRAMES = 16          # frames measured at most (evenly spread over the batch)
STAT_PIXELS = 400000      # pixels per frame measured at most (strided)


# ----------------------------------------------------------------------------- Oklab both ways

def srgb_to_oklab(rgb):
    """sRGB (0..1, gamma) -> Oklab, float32. Same matrices as ph_fx_core.srgb_to_oklab."""
    c = np.clip(np.asarray(rgb, dtype=np.float32), 0.0, 1.0)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    l_ = np.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    m_ = np.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    s_ = np.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return np.stack([0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
                     1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
                     0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_], axis=-1).astype(np.float32)


def oklab_to_srgb(lab):
    """Oklab -> sRGB (0..1, gamma, clipped), float32. Exact inverse of srgb_to_oklab inside the gamut."""
    lab = np.asarray(lab, dtype=np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    lin = np.stack([4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
                    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
                    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], axis=-1)
    lin = np.clip(lin, 0.0, 1.0)
    out = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ----------------------------------------------------------------------------- what runs

def resolve(mode, has_reference, has_mask):
    """(mode actually run, note). auto: a reference decides 'to reference', else a mask 'mask to its ring', else off.
    A mode whose input is missing does not run and says why. Pure."""
    mode = str(mode)
    if mode == "auto":
        if has_reference:
            return "to reference", None
        if has_mask:
            return "mask to its ring", None
        return "off", None
    if mode == "to reference" and not has_reference:
        return "off", "harmonize: 'to reference' needs harmonize_to connected -- skipped"
    if mode == "mask to its ring" and not has_mask:
        return "off", "harmonize: 'mask to its ring' needs harmonize_mask connected -- skipped"
    if mode not in MODES:
        return "off", "harmonize: unknown mode %r -- skipped" % (mode,)
    return mode, None


def _stat_frames(n):
    """Indices of the frames that are measured: all up to STAT_FRAMES, else evenly spread (first and last in)."""
    if n <= STAT_FRAMES:
        return list(range(n))
    return sorted(set(int(round(i * (n - 1) / (STAT_FRAMES - 1))) for i in range(STAT_FRAMES)))


def _stride(h, w):
    return max(1, int(np.ceil(np.sqrt(h * w / float(STAT_PIXELS)))))


def fit_mask(mask, n, h, w):
    """Mask (M,H',W') or (H',W') -> float32 (n,h,w) in 0..1: one mask for all frames or one per frame (frames past
    the last mask keep the last one); another size is resized bilinear. Returns (mask, note or None)."""
    m = np.asarray(mask, dtype=np.float32)
    if m.ndim == 2:
        m = m[None]
    if m.ndim == 4:          # (M,H,W,1) or (M,1,H,W)
        m = m[..., 0] if m.shape[-1] == 1 else m[:, 0]
    note = None
    if m.shape[1:] != (h, w):
        from PIL import Image as _I
        m = np.stack([np.asarray(_I.fromarray(np.clip(f, 0, 1).astype(np.float32), mode="F")
                                 .resize((w, h), _I.BILINEAR), dtype=np.float32) for f in m])
        note = "mask %dx%d resized to the image %dx%d" % (mask.shape[-1], mask.shape[-2], w, h)
    if m.shape[0] < n:
        if m.shape[0] != 1:
            note = (note + "; " if note else "") + "%d masks for %d frames -- the last one holds" % (m.shape[0], n)
        m = np.concatenate([m, np.repeat(m[-1:], n - m.shape[0], axis=0)], axis=0)
    return np.clip(m[:n], 0.0, 1.0), note


def dilate(hard, r):
    """Binary dilation of (n,h,w) bool by a square of radius r px."""
    r = int(r)
    if r <= 0:
        return hard.copy()
    try:
        import torch
        import torch.nn.functional as F
        t = torch.from_numpy(hard.astype(np.float32))[:, None]
        return (F.max_pool2d(t, 2 * r + 1, stride=1, padding=r)[:, 0] > 0.5).numpy()
    except Exception:  # separable numpy fallback: rows, then columns
        out = hard.copy()
        for d in range(1, r + 1):
            out[:, d:, :] |= hard[:, :-d, :]
            out[:, :-d, :] |= hard[:, d:, :]
        tmp = out.copy()
        for d in range(1, r + 1):
            out[:, :, d:] |= tmp[:, :, :-d]
            out[:, :, :-d] |= tmp[:, :, d:]
        return out


def ring_of(mask_hard, ring):
    """The surroundings of a hard mask: ring px wide, RING_GAP px off its edge. (n,h,w) bool."""
    return dilate(mask_hard, RING_GAP + int(ring)) & ~dilate(mask_hard, RING_GAP)


def _box(x, r):
    """Box sum of radius r over the two leading spatial axes of (h,w[,c]) -- integral image, O(N)."""
    r = int(r)
    if r <= 0:
        return x.astype(np.float64)
    pad = [(r + 1, r), (r + 1, r)] + [(0, 0)] * (x.ndim - 2)
    c = np.pad(x.astype(np.float64), pad, mode="constant").cumsum(0).cumsum(1)
    k = 2 * r + 1
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def _local_mean(lab, wgt, r):
    """Normalized box average of lab (h,w,3) over the pixels where wgt (h,w) is set. (mean, weight sum)."""
    ws = _box(wgt.astype(np.float64), r)
    vs = _box(lab * wgt[..., None], r)
    return vs / np.maximum(ws, 1e-9)[..., None], ws


def _moments(samples, weights=None):
    """Weighted mean and spread (std) per Oklab channel of (k,3) samples. None when nothing carries weight."""
    if samples.shape[0] == 0:
        return None
    s = samples.astype(np.float64)
    if weights is None:
        return s.mean(0), s.std(0)
    w = weights.astype(np.float64)
    tot = w.sum()
    if tot <= 1e-6:
        return None
    mu = (s * w[:, None]).sum(0) / tot
    var = ((s - mu) ** 2 * w[:, None]).sum(0) / tot
    return mu, np.sqrt(var)


def _lin(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gamma(lin):
    lin = np.clip(lin, 0.0, 1.0)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1.0 / 2.4) - 0.055)


def _chroma(px):
    """Linear RGB samples (k,3) -> luminance-normalised chromaticities of the lit ones (k',3)."""
    y = px @ _Y
    ok = y > DARK_Y
    return px[ok] / y[ok][:, None]


def ring_samples(frame, mask, ring):
    """One frame's evidence for the ring mode: (chroma inside, chroma ring, log-brightness steps along the edge)."""
    hard = (mask > 0.5)[None]
    if not hard.any() or hard.all():
        return None
    r = max(2, int(ring))
    lin = _lin(frame[..., :3])
    rg = ring_of(hard, r)[0]
    band = hard[0] & ~dilate(~hard, RING_GAP)[0] & dilate(~hard, RING_GAP + r)[0]
    if not rg.any() or not band.any():
        return None
    logy = np.log(np.maximum(lin @ _Y, 1e-4))[..., None]
    reach = r + 2 * RING_GAP + 1          # the averaging must bridge the gap on both sides of the edge
    mo, wo = _local_mean(logy, rg, reach)
    mi, wi = _local_mean(logy, band, reach)
    edge = band & (wo > 0.5) & (wi > 0.5)
    return _chroma(lin[hard[0]]), _chroma(lin[rg]), (mo - mi)[..., 0][edge]


def _signed_distance(hard, width):
    """Pixels (h,w) -> distance to the mask edge in px, capped at width: >0 inside, <0 outside (repeated 1-px
    dilations, square metric)."""
    w = int(width)
    din = np.full(hard.shape, float(w), np.float32)
    dout = np.full(hard.shape, float(w), np.float32)
    grow_out, grow_in = ~hard[None], hard[None]
    for d in range(1, w + 1):
        grow_out = dilate(grow_out, 1)
        grow_in = dilate(grow_in, 1)
        din[(grow_out[0] & hard) & (din == w)] = d - 0.5
        dout[(grow_in[0] & ~hard) & (dout == w)] = d - 0.5
    return np.where(hard, din, -dout)


def _spread(val, known, steps):
    """Fill (h,w,c) values outward from the `known` pixels, one 3x3 ring per step: every pixel ends up with the
    average of the nearest known ones (no long-range mixing). Pixels farther than `steps` stay 0."""
    v = val.astype(np.float64) * known[..., None]
    have = known.astype(bool).copy()
    for _ in range(int(steps)):
        bv = _box(v, 1)
        bw = _box(have.astype(np.float64), 1)
        new = (~have) & (bw > 0)
        if not new.any():
            break
        v[new] = bv[new] / bw[new][:, None]
        have |= new
    return v


def seam_soften(before, after, mask, ring=8, width=24, limit=0.4):
    """Two-sided seam (v1323, the back wall of 08.11.): after the ring move a step can remain right at the mask
    edge -- one light factor for the whole mask cannot fit every stretch of the seam. It is measured locally along
    the edge (the untouched outside of `before` against the inside of `after`, per channel in log linear RGB,
    smoothed along the edge over `width` px) and split: the inside takes half of it, the outside the other half in
    the opposite direction, each fading to nothing `width` px away from the edge. The two sides meet at the seam,
    the step becomes a gradient. Pixels OUTSIDE the mask change too, inside that band -- like a hand-made soft
    blend. Frames (h,w,3) 0..1, mask (h,w). Returns the frame."""
    hard = mask > 0.5
    W = int(width)
    if W <= 0 or not hard.any() or hard.all():
        return after
    r = SEAM_BAND                                            # thin across the seam: the room's own gradient
    lb, la = _lin(before[..., :3]), _lin(after[..., :3])     # must not be read as a step (08.11.: +-13 px overshot)
    lgb, lga = np.log(np.maximum(lb, 1e-4)), np.log(np.maximum(la, 1e-4))
    h = hard[None]
    outside = ring_of(h, r)[0]
    band = hard & ~dilate(~h, RING_GAP)[0] & dilate(~h, RING_GAP + r)[0]
    reach = r + 2 * RING_GAP + 1
    mo, wo = _local_mean(lgb, outside, reach)
    mi, wi = _local_mean(lga, band, reach)
    edge = band & (wo > 0.5) & (wi > 0.5)
    if not edge.any():
        return after
    step = np.clip(mo - mi, -limit, limit) * edge[..., None]
    sm, sw = _local_mean(step, edge, max(4, W // 2))         # smoothed along the edge (half the band)
    sm = sm * (sw > 0.5)[..., None]
    fld = _spread(sm, edge, W + RING_GAP + 1)                # each pixel takes its NEAREST stretch of the seam
    sd = _signed_distance(hard, W)
    fade = 0.5 * np.clip(1.0 - np.abs(sd) / float(W), 0.0, 1.0) ** 2
    sign = np.where(hard, 1.0, -1.0)
    moved = la * np.exp(fld * (fade * sign)[..., None])
    # the edge itself (the gap the measurement skips, partly painted pixels): bridged smoothly between the first
    # clean pixel inside and the first clean pixel outside -- 08.11.: left alone it kept a 1-px line along the seam
    G = RING_GAP
    sdg = _signed_distance(hard, G + 2)
    gap = np.abs(sdg) <= G
    if gap.any():
        lm = np.log(np.maximum(moved, 1e-6))
        vin = _spread(lm, hard & (sdg > G) & (sdg <= G + 1), 2 * G + 3)     # must reach across the whole gap
        vout = _spread(lm, ~hard & (sdg < -G) & (sdg >= -G - 1), 2 * G + 3)
        t = np.clip((sdg + G + 1.0) / (2.0 * G + 2.0), 0.0, 1.0)[..., None]
        bridge = np.exp((1.0 - t) * vout + t * vin)
        moved = np.where(gap[..., None], bridge, moved)
        fade = np.where(gap, np.maximum(fade, 1e-6), fade)
    touch = fade > 0                                         # only the band changes: the rest stays bit-equal
    res = np.array(after[..., :3], dtype=np.float32, copy=True)
    res[touch] = _gamma(moved[touch]).astype(np.float32)
    return res


def measure(image, mode, ring=8, tone=TONES[0], reference=None, mask=None, seam=0, window=0):
    """Measure once over the batch. image (n,h,w,3) 0..1; reference (r,h',w',3); mask (n,h,w) float 0..1 (fit_mask).
    Returns a plan dict {mode, mu_s, sd_s, mu_t, sd_t, k, light} or {mode: 'off', note} when there is nothing to
    measure. Pure."""
    n, h, w = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    st = _stride(h, w)
    frames = _stat_frames(n)
    src_s, tgt_s = [], []
    for i in frames:
        lab = srgb_to_oklab(image[i, ::st, ::st, :3]).reshape(-1, 3)
        if mode == "to reference":
            if mask is None:
                src_s.append(lab)
            else:
                mw = mask[i, ::st, ::st].reshape(-1)
                sel = mw > 0.5
                src_s.append(lab[sel])
        else:  # mask to its ring -- chromaticities inside / ring and the light steps along the edge
            ev = ring_samples(image[i], mask[i], ring)
            if ev is not None:
                src_s.append(ev)
    if mode == "to reference":
        ref = np.asarray(reference, dtype=np.float32)
        if ref.ndim == 3:
            ref = ref[None]
        rst = _stride(int(ref.shape[1]), int(ref.shape[2]))
        for i in _stat_frames(int(ref.shape[0])):
            tgt_s.append(srgb_to_oklab(ref[i, ::rst, ::rst, :3]).reshape(-1, 3))
    if mode == "mask to its ring":
        if not any((mask[i] > 0.5).any() for i in frames):
            return {"mode": "off", "note": "harmonize: the mask is empty -- nothing to change"}
        ci = np.concatenate([e[0] for e in src_s], 0) if src_s else np.zeros((0, 3))
        cr = np.concatenate([e[1] for e in src_s], 0) if src_s else np.zeros((0, 3))
        st_ = np.concatenate([e[2] for e in src_s], 0) if src_s else np.zeros((0,))
        if ci.shape[0] < 16 or cr.shape[0] < 16:
            return {"mode": "off", "note": "harmonize: the mask leaves no lit ring around it -- nothing to measure"}
        gains = np.clip(np.median(cr, 0) / np.maximum(np.median(ci, 0), 1e-6), GAIN_LIMITS[0], GAIN_LIMITS[1])
        light = str(tone) != "colour only"
        k = float(np.clip(np.exp(np.median(st_)), GAIN_LIMITS[0], GAIN_LIMITS[1])) if (light and st_.size) else 1.0
        return {"mode": mode, "ring": int(ring), "light": light, "gains": gains, "k": k, "seam": max(0, int(seam)),
                "measured": (len(frames), int(ci.shape[0]), int(cr.shape[0]))}
    S = np.concatenate(src_s, 0) if src_s else np.zeros((0, 3), np.float32)
    T = np.concatenate(tgt_s, 0) if tgt_s else np.zeros((0, 3), np.float32)
    if S.shape[0] < 16:
        return {"mode": "off", "note": "harmonize: the mask is empty -- nothing to change"}
    if T.shape[0] < 16:
        return {"mode": "off", "note": "harmonize: the reference is empty"}
    mu_s, sd_s = _moments(S)
    mu_t, sd_t = _moments(T)
    k = np.clip(sd_t / np.maximum(sd_s, 1e-4), SPREAD_LIMITS[0], SPREAD_LIMITS[1])
    light = str(tone) != "colour only"
    if not light:
        k[0], mu_t = 1.0, np.array([mu_s[0], mu_t[1], mu_t[2]])
    plan = {"mode": mode, "mu_s": mu_s, "sd_s": sd_s, "mu_t": mu_t, "sd_t": sd_t, "k": k, "light": light,
            "ring": int(ring), "measured": (len(frames), int(S.shape[0]), int(T.shape[0]))}
    W = int(window)
    if W > 1 and n > 1:
        # v1324: a clip that DRIFTS (warm at the start, grey later -- Frank's H3 bridge, 09.11.) needs a correction
        # that follows the drift: every frame's statistics, smoothed over a centred window of W frames (slow
        # enough not to flicker), each frame moved from ITS smoothed statistics onto the reference.
        plan.update(_window_stats(image, mask, W, mu_t, sd_t, light))
    return plan


def _window_stats(image, mask, W, mu_t, sd_t, light):
    """Per-frame Oklab mean / spread of the image (inside the mask if given), smoothed over a centred box of W
    frames (moments smoothed, weighted by the pixels each frame contributes); -> per-frame plan arrays."""
    n, h, w = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
    st = _stride(h, w)
    m1 = np.zeros((n, 3))
    m2 = np.zeros((n, 3))
    cnt = np.zeros(n)
    for i in range(n):
        lab = srgb_to_oklab(image[i, ::st, ::st, :3]).reshape(-1, 3).astype(np.float64)
        if mask is not None:
            lab = lab[mask[i, ::st, ::st].reshape(-1) > 0.5]
        if lab.shape[0]:
            cnt[i] = lab.shape[0]
            m1[i] = lab.sum(0)
            m2[i] = (lab ** 2).sum(0)
    half = W // 2
    c1, c2, cc = (np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, 0)]) for a in (m1, m2, cnt))
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    num = np.maximum(cc[hi] - cc[lo], 1.0)
    mu = (c1[hi] - c1[lo]) / num[:, None]
    sd = np.sqrt(np.maximum((c2[hi] - c2[lo]) / num[:, None] - mu ** 2, 0.0))
    k_t = np.clip(sd_t[None] / np.maximum(sd, 1e-4), SPREAD_LIMITS[0], SPREAD_LIMITS[1])
    mu_t_t = np.repeat(np.asarray(mu_t, np.float64)[None], n, 0)
    if not light:
        k_t[:, 0] = 1.0
        mu_t_t[:, 0] = mu[:, 0]
    return {"window": int(W), "mu_s_t": mu, "k_t": k_t, "mu_t_t": mu_t_t}


def transfer(chunk, plan, start=0):
    """The fully harmonized chunk (amount 1, no mask weight): (m,h,w,3) -> same. With a window plan (v1324) the
    chunk's frames start at batch frame `start` and take their own smoothed statistics."""
    lab = srgb_to_oklab(chunk[..., :3])
    if "mu_s_t" in plan:
        m = lab.shape[0]
        n = plan["mu_s_t"].shape[0]
        idx = np.clip(np.arange(start, start + m), 0, n - 1)
        mu_s = plan["mu_s_t"][idx].astype(np.float32)[:, None, None, :]
        mu_t = plan["mu_t_t"][idx].astype(np.float32)[:, None, None, :]
        k = plan["k_t"][idx].astype(np.float32)[:, None, None, :]
        return oklab_to_srgb((lab - mu_s) * k + mu_t)
    mu_s = plan["mu_s"].astype(np.float32)
    mu_t = plan["mu_t"].astype(np.float32)
    k = plan["k"].astype(np.float32)
    return oklab_to_srgb((lab - mu_s) * k + mu_t)


def apply(chunk, plan, amount=1.0, weight=None, start=0):
    """image + amount * weight * (transfer - image). weight (m,h,w) or None. Linear in amount. `start` = the batch
    index of the chunk's first frame (window plans, v1324)."""
    x = np.asarray(chunk, dtype=np.float32)
    if plan.get("mode", "off") == "off" or float(amount) <= 0.0:
        return x
    if plan["mode"] == "mask to its ring":
        d = (_gamma(_lin(x[..., :3]) * (plan["gains"] * plan["k"])) - x[..., :3]).astype(np.float32)
        if plan.get("seam", 0) > 0 and weight is not None:
            # v1323: the two-sided seam on the fully moved frame; the result stays linear in amount
            full = np.clip(x[..., :3] + weight[..., None] * d, 0.0, 1.0)
            out = x.copy()
            for i in range(x.shape[0]):
                soft = seam_soften(x[i, ..., :3], full[i], weight[i], plan["ring"], plan["seam"])
                out[i, ..., :3] = np.clip(x[i, ..., :3] + float(amount) * (soft - x[i, ..., :3]), 0.0, 1.0)
            return out
    else:
        d = transfer(x, plan, start) - x[..., :3]
    f = float(amount) if weight is None else float(amount) * weight[..., None]
    out = x.copy()
    out[..., :3] = np.clip(x[..., :3] + f * d, 0.0, 1.0)
    return out


def describe(plan):
    """One line for the console and the preview: what was moved, in Oklab units (L 0..1; a, b about +-0.3)."""
    if plan.get("mode", "off") == "off":
        return plan.get("note")
    if plan["mode"] == "mask to its ring":
        g, k = plan["gains"], plan["k"]
        warm = "warmer" if g[2] < g[0] * 0.98 else ("cooler" if g[2] > g[0] * 1.02 else "same warmth")
        return ("harmonize to the light of its %d px ring: white balance R x%.2f G x%.2f B x%.2f (%s), light x%.2f%s%s"
                % (plan["ring"], g[0], g[1], g[2], warm, k, "" if plan["light"] else " (colour only)",
                   (", seam softened %d px" % plan["seam"]) if plan.get("seam", 0) else ""))
    d = plan["mu_t"] - plan["mu_s"]
    warm = "warmer" if d[2] > 0.003 else ("cooler" if d[2] < -0.003 else "same warmth")
    if "mu_s_t" in plan:
        dt = plan["mu_t_t"] - plan["mu_s_t"]
        return ("harmonize to the reference over a %d-frame window: L %+.3f..%+.3f, b %+.3f..%+.3f along the clip%s"
                % (plan["window"], dt[:, 0].min(), dt[:, 0].max(), dt[:, 2].min(), dt[:, 2].max(),
                   "" if plan["light"] else ", light kept"))
    return ("harmonize to the reference: L %+.3f a %+.3f b %+.3f (%s), spread x%.2f/%.2f/%.2f%s"
            % (d[0], d[1], d[2], warm, plan["k"][0], plan["k"][1], plan["k"][2],
               "" if plan["light"] else ", light kept"))
