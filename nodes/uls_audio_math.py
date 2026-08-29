"""
uls_audio_math.py -- pure, torch-free arithmetic for audio time-stretching.

WHY A SEPARATE PURE MODULE (the uls_tile_math precedent, third use): the tempo
factor is the one thing that MUST be exactly right -- a wrong factor does not
crash, it detunes the whole soundtrack against the picture, which is the silent
failure class this house fears most. Keeping the arithmetic import-free means
it runs and is unit-tested OUTSIDE ComfyUI (Messen schlaegt Glauben); the
stretch machinery in ph_audio_stretch is a thin caller.

TEMPO SEMANTICS, fixed here once: `tempo` follows ffmpeg's atempo -- it is the
SPEED factor. tempo 2.0 plays twice as fast (half the duration); tempo 0.5 is
2x slow motion (twice the duration). duration_out = duration_in / tempo.

THE COUPLING LAW (v890), corrected by its own guard before it ever shipped:
the first draft used tempo = d_in/d_out (durations). At the interpolate's
STANDARD case (out_fps = src*mult, n_out = mult*(n_in-1)+1) that lands at
1.008 -- and "stretching" by 0.8% would have shifted every syllable by up to
31 ms across the clip, INTRODUCING the very drift this module exists to
prevent. The v887 arithmetic puts source frame k on its ORIGINAL second; the
missing tail is an END problem, not a rate problem.

The right factor is the stretch of the TIME AXIS, taken over the frame SPAN
(first frame to last frame), which is what interpolation actually re-times:

    tempo = (span_in) / (span_out)
          = ((n_in - 1) / src_fps) / ((n_out - 1) / out_fps)
          = out_fps * (n_in - 1) / (src_fps * (n_out - 1))

Standard case: exactly 1.0 -> action "trim" (the tail is CUT, every sample
keeps its second). Slow motion (out_fps = src): exactly 0.5. The trim target
stays d_out = n_out / out_fps in every branch.
"""

# atempo accepts 0.5 .. 100.0 per stage (ffmpeg doc). We stay conservative and
# chain through [0.5, 2.0] so every stage sits in the filter's sweet spot.
ATEMPO_MIN = 0.5
ATEMPO_MAX = 2.0

# |tempo - 1| below this is "the timeline did not really change" -- stretching
# would resynthesise the whole track to fix microseconds. Trim instead.
TEMPO_EPS = 1e-3

# Guard rails: outside this range a stretch is almost certainly a wiring
# mistake (e.g. fps typed as frame count), and the result would be unusable
# noise. The caller refuses LOUDLY rather than delivering garbage.
TEMPO_SANE_MIN = 1.0 / 16.0
TEMPO_SANE_MAX = 16.0


def stretch_plan(n_in, src_fps, n_out, out_fps):
    """The whole decision, as data. Returns a dict:

        tempo      speed factor for the sound (atempo semantics)
        d_in       source video duration, seconds
        d_out      output video duration, seconds
        action     "stretch" | "trim" | "none" | "refuse"
        note       one printable sentence saying what and why

    Raises nothing; bad geometry comes back as action="refuse" with the reason
    in `note`, so a caller cannot forget to handle it.
    """
    try:
        n_in = int(n_in)
        n_out = int(n_out)
        src_fps = float(src_fps)
        out_fps = float(out_fps)
    except (TypeError, ValueError):
        return {"tempo": 1.0, "d_in": 0.0, "d_out": 0.0, "action": "refuse",
                "note": "audio stretch refused: non-numeric geometry"}
    if n_in <= 0 or n_out <= 0 or src_fps <= 0 or out_fps <= 0:
        return {"tempo": 1.0, "d_in": 0.0, "d_out": 0.0, "action": "refuse",
                "note": "audio stretch refused: frame counts and rates must "
                        "be positive (got %d @ %.3f -> %d @ %.3f)"
                        % (n_in, src_fps, n_out, out_fps)}
    d_in = n_in / src_fps
    d_out = n_out / out_fps
    if n_in < 2 or n_out < 2:
        # one frame has no span; there is nothing to re-time, only to fit
        return {"tempo": 1.0, "d_in": d_in, "d_out": d_out, "action": "trim",
                "note": "single-frame timeline; audio kept, trimmed to "
                        "%.3fs" % d_out}
    tempo = (out_fps * (n_in - 1)) / (src_fps * (n_out - 1))
    if tempo < TEMPO_SANE_MIN or tempo > TEMPO_SANE_MAX:
        return {"tempo": tempo, "d_in": d_in, "d_out": d_out,
                "action": "refuse",
                "note": "audio stretch refused: tempo %.4f is outside %s..%s "
                        "- this is almost certainly a wiring mistake, not a "
                        "slow-motion request" % (tempo, TEMPO_SANE_MIN,
                                                 TEMPO_SANE_MAX)}
    if abs(tempo - 1.0) < TEMPO_EPS:
        return {"tempo": 1.0, "d_in": d_in, "d_out": d_out, "action": "trim",
                "note": "timeline unchanged (tempo %.6f); audio kept, "
                        "trimmed to %.3fs" % (tempo, d_out)}
    return {"tempo": tempo, "d_in": d_in, "d_out": d_out, "action": "stretch",
            "note": "audio tempo %.4f: video %.3fs -> %.3fs (%d @ %.3f fps "
                    "-> %d @ %.3f fps)" % (tempo, d_in, d_out, n_in, src_fps,
                                           n_out, out_fps)}


def atempo_chain(tempo):
    """Split a tempo into atempo stages, each inside [ATEMPO_MIN, ATEMPO_MAX].

    ffmpeg's atempo degrades outside a narrow band, so extreme factors are
    CHAINED: 0.3 -> [0.5, 0.6], 5.0 -> [2.0, 2.0, 1.25]. The product of the
    returned list equals `tempo` (floating point exact up to rounding), and
    every element sits inside the band -- both properties are pinned by the
    guard, because a chain whose product drifts IS the detune bug.
    """
    tempo = float(tempo)
    if tempo <= 0:
        raise ValueError("tempo must be positive, got %r" % tempo)
    chain = []
    t = tempo
    while t < ATEMPO_MIN:
        chain.append(ATEMPO_MIN)
        t /= ATEMPO_MIN
    while t > ATEMPO_MAX:
        chain.append(ATEMPO_MAX)
        t /= ATEMPO_MAX
    chain.append(t)
    return chain


def target_samples(n_samples, sr, d_target):
    """How many samples a track of `d_target` seconds holds at `sr` --
    clamped to at most the input when trimming is the intent is NOT decided
    here; this is bare arithmetic, the caller owns the policy."""
    n_samples = int(n_samples)
    sr = int(sr)
    if sr <= 0 or d_target < 0:
        return n_samples
    return int(round(d_target * sr))
