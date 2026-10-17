# -*- coding: ascii -*-
"""ph_music_style.py -- HOW is the template made? (v1074, Frank 10.10.2026: "es soll der Stil
kopiert werden, eben: die Machart des Liedes, nicht das Lied selber ... das muss alles
Stil-uebergreifend arbeiten")

Pure functions (numpy, scipy, librosa), no ComfyUI imports. Two doors for the template's
style -- never its melody, its words or its arrangement in time:

  1. WORDS for the composer (MiniMax-Music3 reads only text): brightness, stereo image,
     compression, tonal balance, reverb, percussion weight -- measured, put into the official
     caption fields. Plus, with the Demucs stems, which instrument groups play at the start,
     where they enter or drop out, and what the ending keeps -- in PERCENT of the song, so it
     carries over to any length.
  2. SOUND after the decode (the Style Match node): the template's tonal balance (a shape-only
     EQ from the long-term spectra), its stereo width, its dynamic range and its level.
     Time-free statistics only: what plays, and when, stays the new song's.

Measured 10.10. on Frank's goth-wave template (282 s) against his four Music3 results:
centroid 1400 Hz vs 1900-2400, width 0.07 vs 0.35-0.43, LRA 7.5 dB vs 11.5-16, low share
30 % vs 36-62 %, reverb tails 0.6 dB lost in 190 ms vs 3.5-6. After the match: the spectral
shape within 1 dB of the template, the width on the dot, the LRA on target after three passes.

A drum-pattern detector (backbeat / four-on-the-floor) was measured too and LEFT OUT: right
on synthetic beats, but on real mixes the beat-to-beat contrast was 0.85-1.0 -- a guess.
"""
import numpy as np

SR = 22050
N_FFT = 4096                  # the match's STFT (about 93 ms at 44.1 kHz)

# ---------------------------------------------------------------- the words' scales
# Anchors: Frank's template (dark, near mono, dense, mid-forward, long tails) and four Music3
# results (warm, moderately wide, dynamic, bass-heavy, drier). Named constants -- one place.
BRIGHT = ((1600, "a dark, muffled top end"), (2400, "a warm, rounded top end"),
          (3200, "a clear, present top end"), (1e9, "a bright, crisp top end"))
WIDTH = ((0.12, "an almost mono, centred image"), (0.30, "a narrow stereo image"),
         (0.50, "a moderately wide stereo image"), (1e9, "a very wide stereo image"))
# the loudness range of 0.4 s blocks: template 5.7 dB, Music3 results 6.6-9.3, synthetic beats 13-15
DYN = ((6.5, "dense compression with little dynamic movement"), (10.0, "moderate compression"),
       (1e9, "wide dynamics with strong loud-quiet contrasts"))
SPACE = ((1.5, "long reverb tails that hang after every hit"), (4.0, "a moderate sense of room"),
         (1e9, "a dry, close sound"))
LOW_HEAVY = 0.50
MID_FORWARD = 0.60
PERC_HEAVY = 0.50            # percussive power share (HPSS): template 0.25, results 0.39-0.46, beats 0.9
PERC_LIGHT = 0.30
STEM_ABSENT_DB = 18.0         # a stem this far under its own loudest section counts as not playing
STEM_NAMES = {"drums": "the drums", "bass": "the bass", "other": "the melodic instruments", "vocals": "the voice"}


def _scale(v, table):
    for lim, word in table:
        if v < lim:
            return word
    return table[-1][1]


def _stereo(x):
    """[C, S] / [S] / [B, C, S] (numpy or torch) -> float32 [C, S]."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x, dtype=np.float32)
    while x.ndim > 2:
        x = x[0]
    if x.ndim == 1:
        x = x[None]
    return x


def width_of(x):
    """Side / mid RMS of a [C, S] signal (0 for mono)."""
    x = _stereo(x)
    if x.shape[0] < 2:
        return 0.0
    m, s = (x[0] + x[1]) / 2.0, (x[0] - x[1]) / 2.0
    return float(np.sqrt(np.mean(s ** 2) / max(float(np.mean(m ** 2)), 1e-20)))


def block_db(mono, sr, win=0.4):
    """RMS in dB per 0.4 s block -> (dB array, block length in samples)."""
    n = max(1, int(sr * win))
    k = len(mono) // n
    if k == 0:
        return np.array([20.0 * np.log10(max(float(np.sqrt(np.mean(mono ** 2))), 1e-6))]), n
    r = np.sqrt(np.mean(mono[: k * n].reshape(k, n) ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(r, 1e-6)), n


def lra_of(db):
    """A loudness range: 95th - 10th percentile of the blocks within 40 dB of the loudest."""
    v = db[db > db.max() - 40.0]
    if len(v) < 2:
        return 0.0
    return float(np.percentile(v, 95) - np.percentile(v, 10))


PERC_WINDOW_S = 30.0          # the percussive share from three windows (20 / 50 / 80 %): HPSS over a
                              # whole 282 s track cost ~25 s, three windows cost ~3 s (measured 10.10.)


def percussive_share(y, sr=SR):
    import librosa
    n = int(PERC_WINDOW_S * sr)
    if len(y) > 3 * n:
        y = np.concatenate([y[int(c * len(y)) - n // 2: int(c * len(y)) + n // 2] for c in (0.2, 0.5, 0.8)])
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    H, P = librosa.decompose.hpss(S)
    return float(P.sum() / max(float((H + P).sum()), 1e-20))


def style_of(x, sr, percussive=True):
    """The time-free style numbers of a [C, S] signal (percussive=False skips the HPSS -- the
    Style Match's before / after report does not need it)."""
    import librosa
    x = _stereo(x)
    mono = x.mean(0)
    y = librosa.resample(mono, orig_sr=int(sr), target_sr=SR) if int(sr) != SR else mono
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    f = librosa.fft_frequencies(sr=SR, n_fft=2048)
    cent = float(np.median(librosa.feature.spectral_centroid(S=np.sqrt(S), sr=SR)))

    def band(a, b):
        return float(S[(f >= a) & (f < b)].sum())
    tot = max(band(20, 11000), 1e-20)
    perc = percussive_share(y) if percussive else None
    db = 10.0 * np.log10(np.maximum(S.sum(0), 1e-20))
    onsets = librosa.onset.onset_detect(y=y, sr=SR, units="frames")
    tails = [db[o + 1] - db[o + 9] for o in onsets if o + 9 < len(db)]
    bdb, _n = block_db(mono, sr)
    return {
        "centroid_hz": round(cent, 0),
        "width": round(width_of(x), 3),
        "lra_db": round(lra_of(bdb), 2),
        "low": round(band(20, 250) / tot, 3), "mid": round(band(250, 4000) / tot, 3),
        "high": round(band(4000, 11000) / tot, 3),
        "decay_db": round(float(np.median(tails)) if tails else 0.0, 2),     # dB lost in ~190 ms after a hit
        "percussive": None if perc is None else round(perc, 3),
        "level_db": round(float(20.0 * np.log10(max(float(np.sqrt(np.mean(mono ** 2))), 1e-9))), 2),
        "channels": int(x.shape[0]),
    }


def sound_sentence(st):
    """The measured sound as ONE sentence for 'Sonics & Production Profile'."""
    parts = [_scale(st["centroid_hz"], BRIGHT)]
    if st.get("channels", 2) >= 2:                  # a mono file says nothing about a stereo image
        parts.append(_scale(st["width"], WIDTH))
    parts.append(_scale(st["lra_db"], DYN))
    if st["low"] >= LOW_HEAVY:
        parts.append("a bass-heavy balance")
    elif st["mid"] >= MID_FORWARD:
        parts.append("a midrange-forward balance with a light low end")
    parts.append(_scale(st["decay_db"], SPACE))
    return "The sound has " + ", ".join(parts[:-1]) + " and " + parts[-1] + "."


def percussion_sentence(st):
    if st.get("percussive") is None:
        return ""
    if st["percussive"] >= PERC_HEAVY:
        return "The percussion is forward and drives the mix."
    if st["percussive"] <= PERC_LIGHT:
        return "The percussion stays light in the mix."
    return ""


# ---------------------------------------------------------------- the stems' life cycle (Demucs)
def stem_lifecycle(stems, sr, sections, seconds):
    """stems {name: mono}, the section starts in seconds -> sentences in PERCENT of the song.
    A stem plays in a section when its level there is within STEM_ABSENT_DB of its loudest
    section. Sentences only for what changes: the opening's line-up, entries, drop-outs,
    the ending."""
    secs = [float(s) for s in (sections or [0.0])] + [float(seconds)]
    if len(secs) < 3:
        return []
    names = [n for n in ("drums", "bass", "other", "vocals") if n in stems]
    play = {}
    for n in names:
        y = np.asarray(stems[n], dtype=np.float32)
        lv = []
        for a, b in zip(secs[:-1], secs[1:]):
            seg = y[int(a * sr): max(int(a * sr) + 1, int(b * sr))]
            lv.append(10.0 * np.log10(max(float(np.mean(seg ** 2)) if len(seg) else 0.0, 1e-20)))
        lv = np.array(lv)
        play[n] = lv > lv.max() - STEM_ABSENT_DB
    pct = [100.0 * s / float(seconds) for s in secs]
    out = []
    first = [STEM_NAMES[n] for n in names if play[n][0]]
    missing = [STEM_NAMES[n] for n in names if not play[n][0] and n != "vocals"]
    if missing and first:
        out.append("The opening has only %s, without %s." % (_and(first), _and(missing)))
    for n in names:
        if n == "vocals":
            continue
        p = play[n]
        for i in range(1, len(p)):
            if p[i] and not p[i - 1]:
                out.append("%s enter after about %d%% of the song." % (_cap(STEM_NAMES[n]), round(pct[i])))
            elif not p[i] and p[i - 1] and i < len(p) - 1:
                out.append("%s drop out at about %d%% and return later." % (_cap(STEM_NAMES[n]), round(pct[i]))
                           if p[i:].any() else "%s drop out at about %d%%." % (_cap(STEM_NAMES[n]), round(pct[i])))
    last = [STEM_NAMES[n] for n in names if play[n][-1] and n != "vocals"]
    gone = [STEM_NAMES[n] for n in names if not play[n][-1] and play[n][:-1].any() and n != "vocals"]
    if gone and last:
        out.append("The ending keeps only %s." % _and(last))
    return out


def _and(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _cap(t):
    return t[:1].upper() + t[1:]


# ---------------------------------------------------------------- the match (the Style Match node)
def _stft(x, sr):
    from scipy.signal import stft
    return stft(x, fs=sr, nperseg=N_FFT)


def ltas(mono, sr):
    """The long-term average power spectrum -> (freqs, power)."""
    f, _t, Z = _stft(mono, sr)
    return f, np.mean(np.abs(Z) ** 2, axis=1) + 1e-20


def smooth_third_octave(f, p):
    out = np.empty_like(p)
    for i, fc in enumerate(f):
        if fc <= 0:
            out[i] = p[i]
            continue
        m = (f >= fc * 2 ** (-1 / 6.0)) & (f <= fc * 2 ** (1 / 6.0))
        out[i] = p[m].mean()
    return out


def eq_curve(ref_mono, ref_sr, mono, sr, max_db=12.0):
    """The shape-only EQ (dB per STFT bin of the TARGET's grid) that gives the target the
    template's tonal balance. The overall level is removed (energy-weighted by the target):
    an EQ changes the shape, the level is a separate step. Clipped to +-max_db."""
    f, b = ltas(mono, sr)
    fr, a = ltas(ref_mono, ref_sr)
    a = np.interp(f, fr, a)
    a, b = smooth_third_octave(f, a), smooth_third_octave(f, b)
    g = 10.0 * np.log10(a / b)
    band = (f >= 40) & (f <= min(16000, sr / 2.0 - 1))
    lo, hi = np.argmax(band), len(band) - 1 - np.argmax(band[::-1])
    g[:lo] = g[lo]
    g[hi + 1:] = g[hi]
    # shape only: the offset that keeps the target's total POWER (the guard measured +2.8 dB when the
    # dB curve's energy-weighted mean was removed instead -- Jensen), applied before and after the clip
    for _ in range(2):
        g = g - 10.0 * np.log10(float((b * 10.0 ** (g / 10.0)).sum()) / float(b.sum()))
        g = np.clip(g, -max_db, max_db)
    return f, g


def apply_eq(x, sr, gdb):
    from scipy.signal import istft
    gain = (10.0 ** (gdb / 20.0))[:, None]
    out = np.zeros_like(x)
    for c in range(x.shape[0]):
        _f, _t, Z = _stft(x[c], sr)
        _t2, y = istft(Z * gain, fs=sr, nperseg=N_FFT)
        n = min(len(y), x.shape[1])
        out[c, :n] = y[:n]
    return out


def set_width(x, target):
    if x.shape[0] < 2:
        return x
    cur = width_of(x)
    if cur < 1e-9:
        return x
    m, s = (x[0] + x[1]) / 2.0, (x[0] - x[1]) / 2.0
    s = s * (target / cur)
    return np.stack([m + s, m - s] + [x[c] for c in range(2, x.shape[0])])


def match_dynamics(x, sr, target_lra, passes=3):
    """Pull the 0.4 s block levels toward their median (or push them apart) until the loudness
    range is the template's: k = target / current, clipped 0.3..1.5, three passes (measured:
    on target after three)."""
    for _ in range(passes):
        db, n = block_db(x.mean(0), sr)
        cur = lra_of(db)
        if cur < 1e-3:
            break
        k = float(np.clip(target_lra / cur, 0.3, 1.5))
        if abs(k - 1.0) < 0.01:
            break
        gate = db > db.max() - 40.0
        med = float(np.median(db[gate]))
        gdb = np.where(gate, (db - med) * (k - 1.0), 0.0)
        t = (np.arange(len(gdb)) + 0.5) * n
        g = np.interp(np.arange(x.shape[1]), t, gdb)
        x = x * (10.0 ** (g / 20.0))[None, :]
    return x


def limit_peaks(x, ceiling=0.98, sr=44100):
    """A soft peak limiter: the gain the peaks need, held over 5 ms and smoothed -- never clips."""
    a = np.abs(x).max(axis=0)
    if a.max() <= ceiling:
        return x
    need = np.minimum(1.0, ceiling / np.maximum(a, 1e-9))
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    w = max(1, int(0.005 * sr))
    # never over the ceiling: the mean runs over +-w/2, every value in it is a minimum over +-w around
    # its own sample -- a window that holds sample i itself -- so the mean is <= need[i] (proven, and
    # pinned by the guard on a 3x overdriven signal)
    g = uniform_filter1d(minimum_filter1d(need, size=2 * w + 1), size=w)
    return x * g[None, :]


def match(x, sr, ref, ref_sr, strength=1.0, eq=True, width=True, dynamics=True, loudness=True):
    """x, ref: [C, S] float -> (matched [C, S], report). strength 0..1 scales every step
    (dB-linear for the EQ, the dynamics and the level; log-linear for the width)."""
    x = _stereo(x).astype(np.float64)
    ref = _stereo(ref).astype(np.float64)
    s = float(np.clip(strength, 0.0, 1.0))
    rep = {"strength": s, "steps": []}
    if s <= 0:
        return x.astype(np.float32), rep
    rm = ref.mean(0)
    if eq:
        _f, g = eq_curve(rm, ref_sr, x.mean(0), sr)
        x = apply_eq(x, sr, g * s)
        rep["steps"].append("eq (shape only, max %+.1f / %+.1f dB)" % (float(g.min() * s), float(g.max() * s)))
    if width and x.shape[0] >= 2 and ref.shape[0] >= 2:
        cur, tw = width_of(x), width_of(ref)
        if cur > 1e-6 and tw > 1e-6:
            x = set_width(x, float(np.exp(np.log(cur) + s * (np.log(tw) - np.log(cur)))))
            rep["steps"].append("width %.3f -> %.3f" % (cur, width_of(x)))
    if dynamics:
        cur = lra_of(block_db(x.mean(0), sr)[0])
        tgt = lra_of(block_db(rm, ref_sr)[0])
        x = match_dynamics(x, sr, cur + s * (tgt - cur))
        rep["steps"].append("loudness range %.1f -> %.1f dB (template %.1f)" % (cur, lra_of(block_db(x.mean(0), sr)[0]), tgt))
    if loudness:
        cur = float(np.sqrt(np.mean(x.mean(0) ** 2)))
        tgt = float(np.sqrt(np.mean(rm ** 2)))
        if cur > 1e-9 and tgt > 1e-9:
            gdb = 20.0 * np.log10(tgt / cur) * s
            x = x * 10.0 ** (gdb / 20.0)
            rep["steps"].append("level %+.1f dB" % gdb)
    peak = float(np.abs(x).max())
    x = limit_peaks(x, 0.98, sr)
    if peak > 0.98:
        rep["steps"].append("peaks limited (%.2f -> 0.98)" % peak)
    return x.astype(np.float32), rep


def compare_lines(before, after, template):
    """The info: the three style measurements side by side."""
    rows = (("brightness (centroid Hz)", "centroid_hz", "%.0f"), ("stereo width", "width", "%.3f"),
            ("loudness range dB", "lra_db", "%.1f"), ("low / mid / high", None, None),
            ("reverb (dB lost in 190 ms)", "decay_db", "%.2f"))
    out = ["%-28s %12s %12s %12s" % ("", "before", "after", "template")]
    for label, k, fmt in rows:
        if k is None:
            cell = lambda d: "%.2f/%.2f/%.2f" % (d["low"], d["mid"], d["high"])
        else:
            cell = (lambda kk, ff: (lambda d: ff % d[kk]))(k, fmt)
        out.append("%-28s %12s %12s %12s" % (label, cell(before), cell(after), cell(template)))
    return out
