# -*- coding: ascii -*-
"""ph_music_analyse_lib.py -- what IS this audio? (Polyhedron Music Analyse, S1-A)

Pure functions, no ComfyUI imports: the node (ph_music_analyse.py) is a
handle on these, the guard drives them directly.

Stage A (v1065, this file): the deterministic part -- tempo (both octaves),
key with its confidence, the loudness arc, section boundaries. librosa on
the CPU; measured in S0 (08./09.10.) on Frank's goth-wave track and five CC
pieces: ~2 s per track, the key stable across thirds, the tempo's octave an
honest ambiguity (81 or 161 BPM on the goth track) -- so BOTH are reported.

Later stages (own cuts, own gates): B the vocals from a Demucs stem,
C genre / instruments from the music CLAP checkpoint. Until then the draft
names those lines as "not measured yet" -- never a guess.

Why a text draft at all: MiniMax-Music3's composer (its language model)
reads ONLY caption + lyrics (S0 reading of comfy/ldm/minimax_music). A
template can reach it only as a description in its three rubrics.
"""
import json

import numpy as np

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Schmuckler key profiles
MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

SR = 22050                 # the analysis rate (librosa's default)
ARC_STEP_DB = 1.5          # a third must move this much to count as a move
SECTION_MIN_S = 3.0        # no section shorter than this
MAX_SECTIONS = 6

MISSING_LIBROSA = ("[PLS] Music Analyse needs the Python package 'librosa' in ComfyUI's "
                   "environment (most builds ship it; add it to yours if not)")


def have_librosa():
    try:
        import librosa  # noqa: F401
        return True
    except Exception:
        return False


def to_mono(waveform):
    """ComfyUI AUDIO waveform [B, C, S] / [C, S] / [S] (torch or numpy) -> float32 mono [S]."""
    x = waveform
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x, dtype=np.float32)
    while x.ndim > 2:
        x = x[0]
    if x.ndim == 2:
        x = x.mean(axis=0)
    return np.ascontiguousarray(x)


def key_of(chroma_mean):
    """(key name, correlation) of the best-fitting major/minor profile."""
    v = np.asarray(chroma_mean, dtype=np.float64)
    if v.std() < 1e-9:
        return "unknown", 0.0
    best = None
    for i in range(12):
        for name, prof in (("major", MAJ), ("minor", MIN)):
            c = float(np.corrcoef(np.roll(prof, i), v)[0, 1])
            if best is None or c > best[0]:
                best = (c, KEYS[i] + " " + name)
    return best[1], round(best[0], 2)


# v1091 (Frank's CC run, 11.10.: 2 of 4 generator tracks asked in E minor came out "major"): the
# MODE needs its own margin. The best key of the other mode (parallel OR relative) must trail by
# MODE_MARGIN and the fit must reach KEY_MIN_CONF -- else the Sheet asks instead of guessing.
# Measured on 17 tracks: the wrong modes had margins 0.04 (and one fit at 0.57); the right ones 0.11+.
KEY_MIN_CONF = 0.60
MODE_MARGIN = 0.08


def key_detail(chroma_mean):
    """-> {"key", "conf", "alt" (the best key of the OTHER mode), "alt_conf", "margin", "mode_sure"}."""
    v = np.asarray(chroma_mean, dtype=np.float64)
    if v.std() < 1e-9:
        return {"key": "unknown", "conf": 0.0, "alt": "unknown", "alt_conf": 0.0, "margin": 0.0, "mode_sure": False}
    sc = []
    for i in range(12):
        for name, prof in (("major", MAJ), ("minor", MIN)):
            sc.append((float(np.corrcoef(np.roll(prof, i), v)[0, 1]), KEYS[i] + " " + name))
    sc.sort(key=lambda x: -x[0])
    best = sc[0]
    alt = next(x for x in sc if x[1].split()[1] != best[1].split()[1])
    margin = best[0] - alt[0]
    return {"key": best[1], "conf": round(best[0], 2), "alt": alt[1], "alt_conf": round(alt[0], 2),
            "margin": round(margin, 3), "mode_sure": bool(best[0] >= KEY_MIN_CONF and margin >= MODE_MARGIN)}


def arc_words(thirds):
    a, b, c = thirds
    d1, d2 = b - a, c - b
    first = "builds" if d1 > ARC_STEP_DB else "settles" if d1 < -ARC_STEP_DB else "holds steady"
    second = ("then rises to the end" if d2 > ARC_STEP_DB else
              "then falls away at the end" if d2 < -ARC_STEP_DB else "then stays level")
    return first + " " + second


def section_times(raw_times, duration, min_gap=SECTION_MIN_S):
    """Boundaries at least min_gap apart, none inside the last min_gap seconds."""
    out = []
    for t in raw_times:
        t = float(t)
        if (not out or t - out[-1] >= min_gap) and t <= duration - min_gap:
            out.append(round(t, 1))
    return out


def analyse(y, sr):
    """y: float mono samples, sr: its rate -> the stage-A dict."""
    import librosa
    y = np.asarray(y, dtype=np.float32)
    if sr != SR:
        y = librosa.resample(y, orig_sr=int(sr), target_sr=SR)
        sr = SR
    dur = len(y) / float(sr)
    if dur < 1.0:
        raise ValueError("the audio is shorter than 1 s (%.2f s) -- nothing to analyse" % dur)
    tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    pulse_bpm, pulse_ratio = pulse_of(y, sr, tempo)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key, conf = key_of(chroma.mean(axis=1))
    kd = key_detail(chroma.mean(axis=1))
    meter, meter_ev = meter_of(y, sr, pulse_bpm, chroma)
    # the key per third: a stable key names the same one three times
    n = chroma.shape[1]
    thirds_key = [key_of(chroma[:, i * n // 3:(i + 1) * n // 3].mean(axis=1))[0] for i in range(3)] if n >= 3 else [key] * 3
    rms = librosa.feature.rms(y=y)[0]
    db = 20.0 * np.log10(np.maximum(rms, 1e-6))
    m = len(db)
    thirds = [round(float(np.mean(db[i * m // 3:(i + 1) * m // 3])), 1) for i in range(3)] if m >= 3 else [round(float(db.mean()), 1)] * 3
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    k = int(min(MAX_SECTIONS, max(2, dur // 12)))
    if mfcc.shape[1] > k:
        bounds = librosa.segment.agglomerative(mfcc, k=k)
        raw = librosa.frames_to_time(bounds, sr=sr)
    else:
        raw = [0.0]
    sections = section_times(raw, dur)
    return {
        "seconds": round(dur, 1),
        "bpm": round(tempo, 1),
        "bpm_half": round(tempo / 2.0, 1),
        "bpm_double": round(tempo * 2.0, 1),
        "pulse_bpm": round(pulse_bpm, 1),          # v1072: the octave the onsets carry
        "pulse_ratio": round(pulse_ratio, 3),      # strong / weak beat onsets at the fast octave
        "tracker_bpm": round(tempo, 1),            # v1091: the beat tracker's own octave (the first vote)
        "key": key,
        "key_confidence": conf,
        "key_stable": all(t == key for t in thirds_key),
        "key_alt": kd["alt"],                      # v1091: the best key of the other mode
        "mode_margin": kd["margin"],
        "mode_sure": kd["mode_sure"],
        "meter": meter,                            # v1091: "3/4", "4/4" or None (not measurable)
        "meter_evidence": meter_ev,
        "loudness_db_thirds": thirds,
        "arc": arc_words(thirds),
        "sections_at": sections,
    }


PULSE_SLOW_BPM = 100.0      # under this the pulse is called slow / half-time


PULSE_ACCENT = 1.25         # strong / weak beat onset ratio from which the pulse is the slow octave


def pulse_of(y, sr, tempo, hop=512):
    """v1091: the slow octave only when the beat tracker chose it too (two votes).
    v1072 (Frank's field run, 10.10.: the song kept 81/162 BPM and A minor but felt
    twice as fast): WHICH octave does the music pulse in? The beats are tracked at the
    FAST octave; if every other beat carries clearly more onset energy (an accent
    pattern, strong / weak >= PULSE_ACCENT) the pulse is the slow octave, else the
    fast one. -> (pulse_bpm, the strong / weak ratio)."""
    import librosa
    if not tempo or tempo <= 0:            # no beat found (a drone, a pad, silence)
        return 0.0, 1.0
    fast = tempo * 2.0 if tempo < PULSE_SLOW_BPM else tempo
    slow = fast / 2.0
    tracker_slow = tempo < PULSE_SLOW_BPM
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    _t, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, start_bpm=fast, tightness=400)
    beats = np.asarray(beats, dtype=int)
    if len(beats) < 8:
        return tempo, 1.0
    w = max(1, int(0.05 * sr / hop))
    strength = np.array([float(oenv[max(0, b - w): b + w + 1].max()) for b in beats])
    even, odd = strength[0::2].mean(), strength[1::2].mean()
    hi, lo = max(even, odd), min(even, odd)
    ratio = (hi / lo) if lo > 1e-9 else 9.99
    # v1091: half-time needs BOTH votes -- the tracker chose the slow octave itself AND the accent
    # confirms it. The accent alone halved country boom-chick (117 -> 59), rock and drum & bass
    # (136 -> 68) in Frank's CC run (11.10.): every backbeat alternates strong / weak.
    return (slow if (tracker_slow and ratio >= PULSE_ACCENT) else fast), float(ratio)


# v1091 (P2): the meter. The beats at the felt pulse; per beat the onset strength, the low band
# (kick / bass) and the harmonic change (chroma distance to the previous beat); each sequence's
# autocorrelation at 2, 3 and 4 beats. TRIPLE: the harmony changes every 3 beats (lag 3 beats lags 2
# and 4 by METER_TRIPLE_GAP) and the onsets do not contradict it. DUPLE: some sequence repeats at 2
# or 4 beats clearly more than at 3 (METER_DUPLE_GAP). Neither -> None (the Sheet asks).
# Measured 11.10. on 17 tracks: the waltz alone triple (harmony 0.40 at 3 vs -0.24 / -0.23);
# 14 duple; 3 without evidence. 6/8 is not measured (a choice in the Sheet).
METER_MIN_BEATS = 16
METER_TRIPLE_GAP = 0.25
METER_DUPLE_GAP = 0.15


def _lag_ac(x, lags=(2, 3, 4)):
    x = np.asarray(x, dtype=np.float64)
    x = (x - x.mean()) / (x.std() + 1e-9)
    return {k: (float(np.mean(x[:-k] * x[k:])) if len(x) > k else 0.0) for k in lags}


def meter_word(ac):
    """ac: {"onsets"|"low"|"harmony": {2, 3, 4: autocorrelation}} -> "3/4", "4/4" or None. Pure."""
    h, o = ac.get("harmony") or {}, ac.get("onsets") or {}
    if h and o:
        h3 = h.get(3, 0.0)
        if (h3 >= METER_TRIPLE_GAP and h3 - max(h.get(2, 0.0), h.get(4, 0.0)) >= METER_TRIPLE_GAP
                and o.get(3, 0.0) >= max(o.get(2, 0.0), o.get(4, 0.0)) - 0.1):
            return "3/4"
    for seq in ac.values():
        if seq and max(seq.get(2, 0.0), seq.get(4, 0.0)) - seq.get(3, 0.0) >= METER_DUPLE_GAP:
            return "4/4"
    return None


def meter_of(y, sr, pulse_bpm, chroma=None, hop=512):
    """-> (meter word or None, the evidence {feature: {lag: ac}} or a reason string)."""
    import librosa
    if not pulse_bpm or pulse_bpm <= 0:
        return None, "no steady beat"
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    _t, beats = librosa.beat.beat_track(onset_envelope=oenv, sr=sr, hop_length=hop, start_bpm=pulse_bpm, tightness=400)
    beats = np.asarray(beats, dtype=int)
    if len(beats) < METER_MIN_BEATS:
        return None, "only %d beats" % len(beats)
    w = max(1, int(0.05 * sr / hop))
    ons = np.array([float(oenv[max(0, b - w): b + w + 1].max()) for b in beats])
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    f = librosa.fft_frequencies(sr=sr, n_fft=2048)
    lo = S[f < 200].sum(axis=0)
    low = np.array([float(lo[max(0, b - w): b + w + 1].mean()) for b in beats])
    ch = chroma if chroma is not None and chroma.shape[1] == S.shape[1] else librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    chb = librosa.util.sync(ch, beats)
    harm = np.r_[0.0, np.linalg.norm(np.diff(chb, axis=1), axis=0)][:len(beats)]
    ac = {"onsets": _lag_ac(ons), "low": _lag_ac(low), "harmony": _lag_ac(harm[1:])}
    return meter_word(ac), {k: {str(l): round(v, 2) for l, v in d.items()} for k, d in ac.items()}


def pulse_words(bpm):
    return "a slow, half-time pulse" if bpm < PULSE_SLOW_BPM else "a driving pulse"


def tempo_words(d):
    """The tempo line: the PULSE (the octave the onsets carry) as words and number,
    the other octave as a note to the human (stripped before the composer)."""
    bpm = d.get("pulse_bpm") or d["bpm"]
    if not bpm or bpm <= 0:
        return "no steady beat"
    twin = bpm * 2.0 if bpm < PULSE_SLOW_BPM else bpm / 2.0
    return ("about %.0f BPM, %s (or %.0f -- the beat tracker cannot tell the octave; use the felt tempo)"
            % (bpm, pulse_words(bpm), twin))


def _cap(t):
    """First letter up, the rest untouched (str.capitalize lowers 'BPM')."""
    return t[:1].upper() + t[1:]


# ----------------------------------------------------------------- v1073: the official caption schema
# Measured 10.10. on MiniMax's own caption rewriter (github MiniMax-AI/MiniMax-Music3, skills/
# music-caption-rewriter, 1000 reference captions, read only): ALL of them share one schema --
# the three heads alone on a line, the same labelled fields below, and the first field
# "Basic Attributes: bpm is N. key is K, and scale is S. Genre / Subgenre." (the core turns
# <|bpm 81|> into "bpm is 81" -- the training form). Frank's field runs with a free-text caption
# ("About 81 BPM ..., A minor") came out at 89 / 108 BPM in F / G# major: the composer did not
# read our form. The draft and the merge speak the schema now.
SCHEMA = (
    ("global", "Global Metadata", ("Basic Attributes", "Global Emotional Progression",
                                   "Application Scenarios & Imagery", "Sonics & Production Profile")),
    ("vocal", "Vocal Details", ("Vocal Gender & Timbre", "Vocal Style", "Harmony/Backing Vocals", "Vocal FX")),
    ("arrangement", "Arrangement", ("Primary", "Secondary", "Tertiary", "Groove & Foundation Progression",
                                    "Embellishments, Textures & Spatial FX")),
)
LIFECYCLE = "Instrument Lifecycle Description (Primary/Secondary Layering):"
LAYERS = ("Primary", "Secondary", "Tertiary")
# the official key spelling (1000 captions: Bb 222, Eb 162, Ab 42 -- never A#, D#, G#)
KEY_SPELL = {"A#": "Bb", "D#": "Eb", "G#": "Ab"}


def key_parts(key):
    """'G# major' -> ('Ab', 'major'); 'unknown' -> (None, None)."""
    parts = (key or "").split()
    if len(parts) != 2 or parts[1] not in ("major", "minor") or parts[0] not in KEYS:
        return None, None
    return KEY_SPELL.get(parts[0], parts[0]), parts[1]


def genre_words(labels):
    """CLAP labels -> the official 'Goth Rock / Darkwave' form."""
    return " / ".join(" ".join(w[:1].upper() + w[1:] for w in l.split()) for l in labels)


def caption_draft(d, vocals=None, clap=None, style=None, lifecycle=None):
    """The official MiniMax-Music3 schema, filled with what was measured. A field no stage
    measured says "(not measured yet -- ...)" -- the merge never sends it to the composer;
    the user fills it, the model never sees a guess dressed as a fact."""
    unsure = (clap or {}).get("uncertain") or {}
    flat = (clap or {}).get("flat") or {}

    def _todo(group, head, verb):
        if group in flat:    # v1079: a defect, said as one -- never a lean, never a fact
            return "%s (not measured -- stage C flat, cosine spread %.3f: tower/bank mismatch; %s)." % (
                head, flat[group], verb)
        lean = unsure.get(group)
        if lean:     # v1072: below the confidence gate -- shown as a lean, never as a fact
            return "%s (uncertain -- the music CLAP leans to %s; %s)." % (
                head, ", ".join("%s %d%%" % (l, round(100 * p)) for l, p in lean[:2]), verb)
        return "%s (not measured yet -- %s)." % (head, verb)

    bpm = d.get("pulse_bpm") or d.get("bpm") or 0
    basic = []
    if bpm and bpm > 0:
        basic.append("bpm is %d." % int(round(bpm)))
    k, sc = key_parts(d.get("key"))
    if k and d.get("mode_sure", True) is False:   # v1091: too close to call -- said, never guessed
        basic.append("KEY (uncertain -- %s or %s, too close to call; choose)." % (d.get("key"), d.get("key_alt")))
    elif k:
        basic.append("key is %s, and scale is %s.%s" % (k, sc, "" if d.get("key_stable", True) else
                                                        " (the key changes along the track)"))
    basic.append("%s." % genre_words(clap["genre"][:2]) if clap and clap.get("genre")
                 else _todo("genre", "GENRE", "name it, e.g. 'Genre / Subgenre'"))
    mood = (" The mood reads %s (the weakest measure -- trust your ear)." % ", ".join(clap["mood"][:2])
            if clap and clap.get("mood") else "")
    prod = ("The production is %s." % clap["production"][0] if clap and clap.get("production") else "")
    if style:                                  # v1074: the measured sound -- time-free, any genre
        from_style = _style_sentence(style)
        prod = (prod + " " + from_style).strip()
    if not prod:
        prod = _todo("production", "PRODUCTION", "the mix, the space, the era")
    if clap and clap.get("instruments"):
        ins = clap["instruments"][:4]
        primary = _cap(", ".join(ins[:-1]) + (" and " if len(ins) > 1 else "") + ins[-1]) + "."
    else:
        primary = _todo("instruments", "INSTRUMENTS", "what carries the song, and how it evolves")
    groove = []
    if bpm and bpm > 0:
        groove.append(_cap(pulse_words(bpm)) + ".")
    if d.get("meter"):
        groove.append("In %s time." % d["meter"])  # v1091
    if style:
        ps = _percussion_sentence(style)
        if ps:
            groove.append(ps)
    groove.extend(lifecycle or [])             # v1074: the Demucs stems' life cycle, in percent of the song
    secs = d.get("sections_at") or []
    groove.append("Sections change at about %s s of %.0f s." % (", ".join("%g" % s for s in secs[1:]), d["seconds"])
                  if len(secs) > 1 else "No clear section change in %.0f s." % d["seconds"])
    voc = vocal_fields(vocals)
    lines = [
        "// DRAFT from Polyhedron Music Analyse -- the official MiniMax-Music3 caption schema. Fill every",
        "// '(not measured yet' field you care about; the merge never sends a placeholder to the composer.",
        "// GLOBAL METADATA",
        "Global Metadata",
        "Basic Attributes: " + " ".join(basic),
        "Global Emotional Progression: The energy %s.%s" % (d["arc"], mood),
        "Application Scenarios & Imagery: " + _todo("-", "SCENE", "what is the song for? the imagery"),
        "Sonics & Production Profile: " + prod,
        "",
        "// VOCAL DETAILS",
        "Vocal Details",
    ] + ["%s: %s" % (lab, txt) for lab, txt in voc] + [
        "",
        "// ARRANGEMENT",
        "Arrangement",
        LIFECYCLE,
        "Primary: " + primary,
        "Secondary: " + _todo("-", "SECONDARY", "what joins later, what swells"),
        "Groove & Foundation Progression: " + " ".join(groove),
        "Embellishments, Textures & Spatial FX: " + _todo("-", "TEXTURES", "risers, reverbs, spatial effects"),
    ]
    return "\n".join(lines) + "\n"


def _style_mod():
    try:
        from . import ph_music_style as S
    except ImportError:
        import ph_music_style as S
    return S


def _style_sentence(st):
    return _style_mod().sound_sentence(st)


def _percussion_sentence(st):
    return _style_mod().percussion_sentence(st)


def vocal_fields(v):
    """The draft's Vocal Details fields: unmeasured, instrumental, or a voice with its numbers.
    The voice itself (Singer A (Male) / (Female), timbre) is never guessed."""
    if v is None:
        return [("Vocal Gender & Timbre", "VOICE (not measured yet -- 'Singer A (Male)' or 'Singer A (Female)', "
                                          "the timbre; or 'Instrumental, no vocals.')."),
                ("Vocal Style", "STYLE (not measured yet -- the delivery).")]
    if v.get("presence", 0.0) < PRESENT_MIN:
        return [("Vocal Gender & Timbre", "Instrumental, no vocals.")]
    out = "VOICE (name it: 'Singer A (Male)' or 'Singer A (Female)', the timbre)."
    if v.get("pitch_hz"):
        out += " The voice sits in %s (median pitch about %d Hz)." % (
            v.get("register") or register_words(v["pitch_hz"]), round(v["pitch_hz"]))
    pct = int(round(100 * v["presence"]))
    where = ("entering at about %g s" % v["enters_at"]) if v.get("enters_at") is not None else "from the start"
    return [("Vocal Gender & Timbre", out),
            ("Vocal Style", "Sung in about %d%% of the track, %s." % (pct, where)),
            ("Harmony/Backing Vocals", "HARMONIES (not measured yet -- name them, or say there are none)."),
            ("Vocal FX", "FX (not measured yet -- reverb, delay, doubling).")]


# ----------------------------------------------------------------- stage B words (pure)
PRESENT_MIN = 0.10          # under this share of the track: "no vocals"


def register_words(hz):
    """A ROUGH register from the median pitch -- never 'male / female' (S0, 09.10.:
    an octave error on Frank's goth track; the user names the voice)."""
    if hz is None:
        return "pitch not measurable"
    if hz < 140:
        return "a low register"
    if hz < 230:
        return "a mid register"
    return "a high register"


def vocals_words(v):
    """The draft's VOCAL DETAILS line: unmeasured, absent, or present with its numbers."""
    if v is None:
        return "VOCALS (not measured yet -- say 'No vocals, purely instrumental.' or describe the voice)"
    if v.get("presence", 0.0) < PRESENT_MIN:
        return "No vocals, purely instrumental"
    pct = int(round(100 * v["presence"]))
    where = ("from about %g s on" % v["enters_at"]) if v.get("enters_at") is not None else "from the start"
    if v.get("pitch_hz"):
        reg = " in %s (median pitch about %d Hz)" % (v.get("register") or register_words(v["pitch_hz"]),
                                                     round(v["pitch_hz"]))
    else:
        reg = ""                           # v1072: never "in pitch not measurable"
    return ("A VOICE (name it: male / female, tone, delivery)%s, singing in about %d%% of the track, %s"
            % (reg, pct, where))


def info_lines(d, source, vocals=None, vocals_note=None, clap=None, style=None, lifecycle=None):
    """The console / info text: every number with where it came from."""
    out = [
        "Music Analyse (stage A, librosa): %s" % source,
        "  %.1f s | tempo %.1f BPM (octave twin %.1f) | key %s (confidence %.2f, %s)" % (
            d["seconds"], d["bpm"], d["bpm_double"] if d["bpm"] < 100 else d["bpm_half"], d["key"], d["key_confidence"],
            "stable across thirds" if d.get("key_stable", True) else "CHANGES along the track"),
        "  pulse: %.1f BPM (strong / weak beats %.2f at the fast octave; half-time needs the tracker's vote too) -> %s" % (
            d.get("pulse_bpm") or d["bpm"], d.get("pulse_ratio") or 1.0, pulse_words(d.get("pulse_bpm") or d["bpm"])),
        "  key mode: %s (the other mode's best: %s, margin %.2f)" % (
            "sure" if d.get("mode_sure", True) else "TOO CLOSE TO CALL", d.get("key_alt", "-"), d.get("mode_margin") or 0.0),
        "  meter: %s %s" % (d.get("meter") or "not measurable", d.get("meter_evidence") or ""),
        "  loudness thirds %s dB -> %s" % (d["loudness_db_thirds"], d["arc"]),
        "  sections at %s s" % (", ".join("%g" % s for s in d["sections_at"]) or "0"),
    ]
    if style is not None:
        out.append("  style (v1074, time-free): centroid %.0f Hz | width %.3f | loudness range %.1f dB | low/mid/high "
                   "%.2f/%.2f/%.2f | reverb %.2f dB lost in 190 ms | percussive %s" % (
                       style["centroid_hz"], style["width"], style["lra_db"], style["low"], style["mid"], style["high"],
                       style["decay_db"], "-" if style.get("percussive") is None else "%.2f" % style["percussive"]))
    if lifecycle:
        out.append("  life cycle (Demucs stems): " + " ".join(lifecycle))
    if clap is not None:
        sc = clap.get("scored", {})
        for g in ("genre", "instruments", "production", "mood"):
            if g in sc:
                labels = clap.get("labels") or {}
                below = g in (labels.get("uncertain") or {})
                flat = (labels.get("flat") or {}).get(g)
                note = ""
                if flat is not None:
                    note = " -- FLAT (cosine spread %.4f): tower/bank mismatch, not used" % flat
                elif below:
                    note = " -- BELOW the confidence gate, not used as a fact"
                out.append("  %s (stage C, music CLAP, %d windows): %s%s" % (
                    g, clap.get("windows", 0), ", ".join("%s %.0f%% (cos %.3f)" % (l, 100 * p, c) for l, p, c in sc[g]),
                    note))
    if vocals is not None:
        out.append("  vocals (stage B, Demucs htdemucs_ft stem): present in %d of %d one-second windows (%d%%)%s%s" % (
            vocals.get("windows_on", 0), vocals.get("windows", 0), round(100 * vocals.get("presence", 0.0)),
            (", enter at %g s" % vocals["enters_at"]) if vocals.get("enters_at") is not None else "",
            (", median pitch %.0f Hz (%s)" % (vocals["pitch_hz"], vocals.get("register") or "")) if vocals.get("pitch_hz") else ""))
    else:
        out.append("  vocals (stage B): %s" % (vocals_note or "off"))
    missing = [m for m, have in (("vocals (stage B, Demucs stem)", vocals is not None),
                                 ("genre / instruments / production / mood (stage C, music CLAP)", clap is not None)) if not have]
    if missing:
        out.append("  not measured in this run: " + "; ".join(missing))
    return out


def tags_json(d, source, vocals=None, clap=None, style=None, lifecycle=None, fingerprint=None, voice=None):
    stage = "C" if clap is not None else ("B" if vocals is not None else "A")
    out = {"source": source, "stage": stage, "det": d, "vocals": vocals, "clap": clap,
           "style": style, "lifecycle": lifecycle}
    if fingerprint:
        out["fingerprint"] = fingerprint          # v1085: the track's identity for the Music Sheet
    if voice is not None:
        out["voice"] = voice                      # v1086: the voice's three paths and the verdict
    return json.dumps(out, indent=1)


def audio_fingerprint(y, sr):
    """v1085: the track's identity -- a hash over its samples (every 64th, 16 bit) and its rate.
    The same audio gives the same print whatever its file is called; another track another."""
    import hashlib
    a = np.asarray(y, dtype=np.float32).reshape(-1)[::64]
    q = np.clip(np.round(a * 32767.0), -32768, 32767).astype("<i2")
    h = hashlib.sha1(q.tobytes())
    h.update(str(int(sr)).encode("ascii"))
    return "a:" + h.hexdigest()[:20]


# ----------------------------------------------------------------- S2: the caption merge (pure)
RUBRICS = (("global", "Global Metadata:"), ("vocal", "Vocal Details:"), ("arrangement", "Arrangement:"))
UNMEASURED = "(not measured yet"
# v1072: every marker that makes a draft rubric INCOMPLETE -- checked BEFORE the notes to the
# human are stripped (Frank's field run: "(name it" was stripped first, the draft's nameless
# voice counted as complete and pushed his "deep male baritone" out)
INCOMPLETE_MARKS = (UNMEASURED, "(uncertain", "(name it", "(not measured -- stage C flat")   # v1079: flat = incomplete
# v1072: "your text wins" is the default -- the user's explicit words beat a measurement;
# "combine" keeps the user's rubric AND appends the draft's clean measured sentences
MIX_MODES = ("your text wins, draft fills", "combine: your text, then the measured facts",
             "draft wins, your text fills", "draft only", "your text only")
MIX_DEFAULT = MIX_MODES[0]
MIX_COMBINE = MIX_MODES[1]


def _norm(t):
    return " ".join((t or "").lower().split())


_HEADS = {_norm(h): k for k, h, _f in SCHEMA}
_FIELDS = {_norm(f): (k, f) for k, _h, fs in SCHEMA for f in fs}


def parse_caption(text):
    """v1073: a caption (comments removed) -> {"other": [lines], rubric: {"free": [lines],
    "fields": {label: text}}}. A head is recognised alone on its line ("Global Metadata",
    "### Global Metadata") or with a colon and text ("Global Metadata: a goth song ..." -- the
    old free form, its text goes to the rubric's free lines). A known field ("Vocal Style: ...")
    goes to ITS rubric wherever it stands. Anything else is free text of the current rubric."""
    import re
    out = {"other": []}
    for k, _h, _f in SCHEMA:
        out[k] = {"free": [], "fields": {}}
    cur = "other"
    for line in (text or "").splitlines():
        s = re.sub(r"^\s{0,3}#{1,6}\s+", "", line).strip()
        if not s:
            continue
        m = re.match(r"^([^:]{2,90}?)\s*:\s*(.*)$", s)
        label, rest = (m.group(1), m.group(2).strip()) if m else (s, None)
        if _norm(label) in _HEADS and (rest is None or rest == "" or m):
            cur = _HEADS[_norm(label)]
            if rest:
                out[cur]["free"].append(rest)
            continue
        if _norm(s) == _norm(LIFECYCLE) or _norm(s).startswith(_norm(LIFECYCLE[:-1])):
            cur = "arrangement"
            continue
        if m and _norm(label) in _FIELDS:
            k, lab = _FIELDS[_norm(label)]
            cur = k
            if rest:
                prev = out[k]["fields"].get(lab)
                out[k]["fields"][lab] = (prev + " " + rest) if prev else rest
            continue
        if cur == "other":
            out["other"].append(s)
        else:
            out[cur]["free"].append(s)
    return out


def split_rubrics(text):
    """{"global", "vocal", "arrangement", "other": [lines]} -- the rubric's free lines and its
    fields as "Label: text" lines (kept for callers of the v1069 shape)."""
    c = parse_caption(text)
    out = {"other": list(c["other"])}
    for k, _h, _f in SCHEMA:
        out[k] = list(c[k]["free"]) + ["%s: %s" % (l, t) for l, t in c[k]["fields"].items()]
    return out


def rubric_complete(lines):
    """A text counts only when it is there AND carries no placeholder, no uncertain
    lean and no "name it" (v1072)."""
    if isinstance(lines, str):
        lines = [lines]
    return bool(lines) and not any(m in l for l in lines for m in INCOMPLETE_MARKS)


def _sentences(text):
    import re
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+(?=[A-Z])", text or "") if x.strip()]


def measured_sentences(lines):
    """The clean measured sentences of a text: no placeholder, no lean, no "name it", no
    ALL-CAPS placeholder head; notes to the human stripped. A rubric head is removed."""
    import re
    text = " ".join([lines] if isinstance(lines, str) else lines)
    for _k, head, _f in SCHEMA:
        if text.lower().startswith(head.lower() + ":"):
            text = text[len(head) + 1:].strip()
            break
    out = []
    for sent in _sentences(text):
        if any(m in sent for m in INCOMPLETE_MARKS) or re.match(r"^[A-Z]{4,}\b", sent):
            continue
        sent = _cap(strip_notes(sent)).strip()
        if sent:
            out.append(sent)
    return out


# the draft's notes to the HUMAN -- never to the composer. Pinned phrases, nothing else is touched.
NOTE_PATTERNS = (
    r" \(or \d+ -- the beat tracker cannot tell the octave; use the felt tempo\)",
    r" \(the weakest measure -- trust your ear\)",
    r" \(name it: male / female, tone, delivery\)",
    r" \(median pitch about \d+ Hz\)",
    r" \(the key changes along the track\)",
)


def strip_notes(text):
    import re
    out = text or ""
    for pat in NOTE_PATTERNS:
        out = re.sub(pat, "", out)
    return out.replace("A VOICE", "A voice")


def basic_parts(text):
    """'bpm is 81. key is A, and scale is minor. Gothic Rock / Darkwave.' ->
    {"bpm": "81", "key": "A", "scale": "minor", "genre": "Gothic Rock / Darkwave"} (what is there).
    A part carrying a placeholder or a lean is left out."""
    import re
    t = text or ""
    out = {}
    m = re.search(r"\bbpm is (\d+(?:-\d+)?)", t, re.I)
    if m:
        out["bpm"] = m.group(1)
    m = re.search(r"\bkey is ([A-G][#b]?)", t, re.I)
    if m:
        out["key"] = m.group(1)[:1].upper() + m.group(1)[1:]
    m = re.search(r"\bscale is (major|minor)", t, re.I)
    if m:
        out["scale"] = m.group(1).lower()
    rest = re.sub(r"\bbpm is \d+(?:-\d+)?\.?", "", t, flags=re.I)
    rest = re.sub(r"\bkey is [A-G][#b]?,?\s*(and )?", "", rest, flags=re.I)
    rest = re.sub(r"\bscale is (major|minor)\.?", "", rest, flags=re.I)
    rest = strip_notes(rest).strip(" .,")
    if rest and rubric_complete(rest) and not re.match(r"^[A-Z]{4,}\b", rest):
        out["genre"] = rest
    return out


def basic_text(p):
    out = []
    if p.get("bpm"):
        out.append("bpm is %s." % p["bpm"])
    if p.get("key") and p.get("scale"):
        out.append("key is %s, and scale is %s." % (p["key"], p["scale"]))
    elif p.get("key"):
        out.append("key is %s." % p["key"])
    if p.get("genre"):
        out.append(p["genre"].rstrip(".") + ".")
    return " ".join(out)


SECTIONS_RE = r"(?:Sections change at about [\d., ]+ s of (\d+) s\.|No clear section change in (\d+) s\.)"
SECTIONS_TOLERANCE = 0.15     # the template's section times are kept only within +-15 % of the song's length


def drop_foreign_sections(text, seconds):
    """v1073 (Frank's run: "Sections change at about 12.5, 179.2 ... s of 282 s" in a 90 s song):
    the template's section times only when the song is about as long. -> (text, dropped or None)."""
    import re
    if not seconds or seconds <= 0:
        return text, None
    dropped = []

    def _sub(m):
        n = float(m.group(1) or m.group(2))
        if abs(seconds - n) > SECTIONS_TOLERANCE * n:
            dropped.append("the template's section times (template %.0f s, song %.0f s)" % (n, seconds))
            return ""
        return m.group(0)
    out = re.sub(r"\s*" + SECTIONS_RE, _sub, text)
    return out.strip(), (dropped[0] if dropped else None)


def merge_captions(own, external, mode=MIX_DEFAULT, seconds=None):
    """v1073: field by field in the official schema.
    -> (caption text in the official layout, where = {rubric: "yours N + draft M" | "-", "other",
    "fields": {label: source}, "dropped": [...]}).
    Per field the preferred source when complete, else the other when complete; an INCOMPLETE
    field gives only its clean measured sentences, a bare placeholder never reaches the composer.
    "combine": your field, then the draft's measured sentences you did not already write.
    Basic Attributes are merged per part (bpm / key / scale / genre). Free lines (the old
    'Global Metadata: <text>' form) of yours are kept in every mode but 'draft only'."""
    if mode not in MIX_MODES:
        raise ValueError("caption_mix %r is not one of %s" % (mode, ", ".join(MIX_MODES)))
    a = parse_caption(own)
    b = parse_caption(external)
    only = {"draft only": "draft", "your text only": "yours"}.get(mode)
    order = [only] if only else (["draft", "yours"] if mode.startswith("draft") else ["yours", "draft"])
    src = {"yours": a, "draft": b}
    where = {"fields": {}, "dropped": [], "conflicts": find_conflicts(a, b) if not only else []}
    lines = []
    if "yours" in order and a["other"]:
        lines.extend(a["other"])
    where["other"] = "yours" if ("yours" in order and a["other"]) else "-"
    # v1074 (Frank's run 00004: "The voice sits in a high register" next to his "deep male baritone"
    # -- the register is an octave-prone measure): once YOU wrote anything in Vocal Details, the
    # draft never speaks about the voice itself
    yours_voice = bool(a["vocal"]["fields"] or a["vocal"]["free"]) and only != "draft"
    for key, head, fields in SCHEMA:
        body, count = [], {"yours": 0, "draft": 0}
        for lab in fields:
            got = {s: src[s][key]["fields"].get(lab, "") for s in order}
            if lab == "Vocal Gender & Timbre" and yours_voice and "draft" in got:
                got["draft"] = ""
            if lab == "Basic Attributes":
                parts, owner = {}, {}
                for s in reversed(order):          # the preferred source is applied last
                    for x, v in basic_parts(got[s]).items():
                        parts[x], owner[x] = v, s
                txt = basic_text(parts)
                tag = "+".join(s for s in order if s in owner.values())
            else:
                full = [s for s in order if got[s] and rubric_complete(got[s])]
                if mode == MIX_COMBINE and "yours" in full and got["draft"]:
                    mine = strip_notes(got["yours"]).strip()
                    seen = {_norm(x) for x in _sentences(mine)}
                    extra = [x for x in measured_sentences(got["draft"]) if _norm(x) not in seen]
                    txt = (mine.rstrip() + ("" if mine.rstrip().endswith((".", "!", "?")) else ".")
                           + ((" " + " ".join(extra)) if extra else ""))
                    tag = "yours+draft" if extra else "yours"
                elif full:
                    txt, tag = strip_notes(got[full[0]]).strip(), full[0]
                else:
                    txt, tag = "", ""
                    for s in order:
                        ms = measured_sentences(got[s]) if got[s] else []
                        if ms:
                            txt, tag = " ".join(ms), s + " (measured part)"
                            break
            if lab == "Groove & Foundation Progression" and txt:
                txt, dr = drop_foreign_sections(txt, seconds)
                if dr:
                    where["dropped"].append(dr)
            if not txt:
                continue
            if lab in LAYERS and LIFECYCLE not in body:
                body.append(LIFECYCLE)
            body.append("%s: %s" % (lab, txt))
            where["fields"][lab] = tag
            for s in ("yours", "draft"):
                if s in tag:
                    count[s] += 1
        free = []
        for s in order:
            for x in src[s][key]["free"]:
                if s == "draft":
                    ms = measured_sentences(x)
                    x = " ".join(ms)
                else:
                    x = strip_notes(x)
                if x:
                    free.append(x)
                    count[s] += 1
        if body or free:
            lines.append(head)
            lines.extend(body)
            lines.extend(free)
        where[key] = (" + ".join("%s %d" % (s, count[s]) for s in ("yours", "draft") if count[s]) or "-")
    return "\n".join(lines), where


def merge_words(where):
    out = ", ".join("%s: %s" % (k, where.get(k, "-")) for k in ("global", "vocal", "arrangement")
                    if where.get(k, "-") != "-") or "nothing"
    for d in where.get("dropped") or []:
        out += "; dropped " + d
    for c in where.get("conflicts") or []:
        out += "; CONFLICT " + c
    return out


# v1079 (Frank's run 00005: a template caption from an earlier song -- "Gothic Rock", "Singer A
# (Male), deep baritone" -- met the measurement of a new, female alt-rock track; "your text wins"
# did exactly that, silently): where YOUR text and the MEASUREMENT claim the same thing and
# disagree, the merge says so. It never changes the outcome -- the mode decides -- it only
# refuses to be quiet about it. General: numbers, key, scale, the voice's presence and register.
REGISTER_HZ = {"a low register": (0, 140), "a mid register": (140, 230), "a high register": (230, 10 ** 6)}


def _voice_claims(text):
    """What a Vocal Gender & Timbre text claims: 'male' / 'female' / 'instrumental' / None."""
    import re
    t = (text or "").lower()
    if not t or UNMEASURED in t or "(name it" in t:
        return None
    if re.search(r"\binstrumental\b|\bno vocals\b", t):
        return "instrumental"
    if re.search(r"\bfemale\b|\bwoman\b|\bgirl\b|\bsoprano\b|\balto\b", t):
        return "female"
    if re.search(r"\bmale\b|\bman\b|\bboy\b|\bbaritone\b|\btenor\b|\bbass voice\b", t):
        return "male"
    return None


def _measured_voice(text):
    """What the draft measured about the voice: ('instrumental', None) or ('voice', register, hz)."""
    import re
    t = text or ""
    if not t:
        return None
    if re.search(r"\binstrumental, no vocals", t, re.I):
        return ("instrumental", None, None)
    m = re.search(r"in (a (?:low|mid|high) register) \(median pitch about (\d+) Hz\)", t, re.I)
    if m:
        return ("voice", m.group(1).lower(), int(m.group(2)))
    if "VOICE (name it" in t:
        return ("voice", None, None)
    return None


def find_conflicts(a, b):
    """a = parse_caption(yours), b = parse_caption(draft) -> list of plain sentences."""
    out = []
    pa = basic_parts(a["global"]["fields"].get("Basic Attributes", ""))
    pb = basic_parts(b["global"]["fields"].get("Basic Attributes", ""))
    for part, name in (("bpm", "tempo"), ("key", "key"), ("scale", "scale")):
        va, vb = pa.get(part), pb.get(part)
        if va and vb and str(va).lower() != str(vb).lower():
            if part == "bpm":
                try:
                    fa, fb = float(str(va).split("-")[0]), float(str(vb).split("-")[0])
                    if min(fa, fb) > 0 and abs(fa - fb) / min(fa, fb) < 0.04:
                        continue
                except ValueError:
                    pass
            out.append("%s: your text says %s, the measurement says %s" % (name, va, vb))
    if pa.get("genre") and pb.get("genre") and _norm(pa["genre"]) != _norm(pb["genre"]):
        out.append("genre: your text says '%s', the measurement says '%s'" % (pa["genre"], pb["genre"]))
    claim = _voice_claims(a["vocal"]["fields"].get("Vocal Gender & Timbre", "") or " ".join(a["vocal"]["free"]))
    meas = _measured_voice(b["vocal"]["fields"].get("Vocal Gender & Timbre", ""))
    if claim and meas:
        kind, reg, hz = meas
        if claim == "instrumental" and kind == "voice":
            out.append("voice: your text says instrumental, the measurement finds a voice")
        elif claim in ("male", "female") and kind == "instrumental":
            out.append("voice: your text names a singer, the measurement finds no vocals")
        elif claim == "male" and reg == "a high register":
            out.append("voice: your text says male, the measured voice sits in a high register (median about %d Hz)" % hz)
        elif claim == "female" and reg == "a low register":
            out.append("voice: your text says female, the measured voice sits in a low register (median about %d Hz)" % hz)
    return out
