"""
ph_audio_stretch.py -- the stretch machinery and ⬡ Polyhedron Audio Stretch.

THE JOB (Frank, 28.08.): when video interpolation changes a clip's LENGTH --
slow motion, retiming -- the soundtrack must change by the same factor, or
lips and voice drift apart. The maths lives in uls_audio_math (pure, guarded
against hand computation); this module owns the audio DSP and the free node.
ph_interpolate imports `stretch_audio`/`trim_audio` from HERE -- one source,
two doors, never a copy.

THE METHOD, measured before it was chosen (28.08., sandbox, 440 Hz probe +
syllable-like transient probe):

    ffmpeg atempo via PyAV     length exact, pitch exact, crest 3.89/3.80
    torchaudio phase_vocoder   length exact, pitch exact, crest 3.44/3.80

The phase vocoder smears transients -- the well-known "phasiness" that turns
stretched SPEECH mushy. atempo (WSOLA-family) keeps them. And `av` is a CORE
REQUIREMENT (requirements.txt line 24, checked at v0.33.4), already the muxer
ph_save runs on -- so the better method costs zero new dependencies. lum3on's
AudioTools and RyanOnTheInside both ship the phase-vocoder road; this is the
reason we did not simply take their code.

PITCH MODES:
    preserve      atempo. Voice stays intelligible at its own pitch. For
                  speech and lip-sync this is the mode.
    follow speed  sinc resample to the new length; pitch scales with tempo --
                  the classic slow-motion sound (an octave down at 2x slow).
                  torchaudio, also a core requirement.

STEREO IS STRETCHED AS STEREO: atempo picks WSOLA segment boundaries from the
signal, and two channels fed separately could pick DIFFERENT boundaries --
inter-channel phase wobble. Mono/stereo go through one filter graph as one
stream; exotic layouts (>2ch) fall back to per-channel with a printed notice,
never silently.
"""

from fractions import Fraction

import torch

try:
    from .uls_audio_math import (atempo_chain, stretch_plan,  # noqa: F401
                                 target_samples, TEMPO_EPS,
                                 TEMPO_SANE_MIN, TEMPO_SANE_MAX)
except ImportError:  # pragma: no cover - direct-run fallback, as elsewhere
    from uls_audio_math import (atempo_chain, stretch_plan,   # noqa: F401
                                target_samples, TEMPO_EPS,
                                TEMPO_SANE_MIN, TEMPO_SANE_MAX)


# ---------------------------------------------------------------------------
# waveform plumbing
# ---------------------------------------------------------------------------

def _norm_waveform(audio):
    """AUDIO dict -> (tensor [C, S] float32 cpu, sample_rate). None on junk.

    The tree's convention is {waveform: [1, C, S], sample_rate} (ph_save reads
    exactly this), but loaders in the wild also hand [C, S] and [S]."""
    if not isinstance(audio, dict):
        return None, 0
    wav = audio.get("waveform")
    sr = audio.get("sample_rate")
    if wav is None or not sr:
        return None, 0
    try:
        w = wav.detach().to("cpu", dtype=torch.float32)
    except Exception:
        return None, 0
    if w.dim() == 3:
        w = w[0]
    elif w.dim() == 1:
        w = w.unsqueeze(0)
    if w.dim() != 2 or w.shape[-1] == 0:
        return None, 0
    return w.contiguous(), int(sr)


def _pack(w, sr):
    """[C, S] back into the tree's {waveform: [1, C, S], sample_rate}."""
    return {"waveform": w.unsqueeze(0).contiguous(), "sample_rate": int(sr)}


# ---------------------------------------------------------------------------
# the two stretch roads
# ---------------------------------------------------------------------------

def _atempo_pass(w, sr, tempo):
    """One PyAV filter-graph run over a [C, S] tensor. C must be 1 or 2."""
    import av
    layout = "mono" if w.shape[0] == 1 else "stereo"
    graph = av.filter.Graph()
    src = graph.add_abuffer(sample_rate=sr, format="fltp", layout=layout,
                            time_base=Fraction(1, sr))
    node = src
    for stage in atempo_chain(tempo):
        f = graph.add("atempo", "%.10f" % stage)
        node.link_to(f)
        node = f
    # atempo emits PACKED flt: to_ndarray then returns (1, C*n) interleaved,
    # which read as mono of double length -- measured 28.08., one channel
    # silently lost. aformat before the sink pins the output back to planar,
    # so (C, n) is guaranteed instead of guessed.
    fmt = graph.add("aformat",
                    "sample_fmts=fltp:channel_layouts=" + layout +
                    ":sample_rates=%d" % sr)
    node.link_to(fmt)
    sink = graph.add("abuffersink")
    fmt.link_to(sink)
    graph.configure()

    frame = av.AudioFrame.from_ndarray(w.numpy(), format="fltp", layout=layout)
    frame.sample_rate = sr
    frame.time_base = Fraction(1, sr)
    frame.pts = 0
    out = []
    graph.push(frame)
    graph.push(None)                      # flush
    while True:
        try:
            of = graph.pull()
        except Exception:
            break
        out.append(torch.from_numpy(of.to_ndarray()))
    if not out:
        raise RuntimeError("atempo graph produced no frames")
    return torch.cat(out, dim=-1)


def _stretch_preserve(w, sr, tempo):
    """atempo, stereo-as-stereo; >2ch falls back per-channel, SAID out loud."""
    c = w.shape[0]
    if c <= 2:
        return _atempo_pass(w, sr, tempo)
    print("[PLS] Audio Stretch: %d channels - atempo runs per channel "
          "(inter-channel phase is not guaranteed beyond stereo)." % c)
    parts = [_atempo_pass(w[i:i + 1], sr, tempo) for i in range(c)]
    n = min(p.shape[-1] for p in parts)
    return torch.cat([p[..., :n] for p in parts], dim=0)


def _stretch_follow(w, sr, tempo):
    """Sinc resample to the new length; pitch follows speed by construction.

    resample(orig_freq=sr, new_freq=round(sr/tempo)) yields n/tempo samples;
    declared back at `sr`, the clip lasts (n/sr)/tempo seconds and every
    frequency is multiplied by tempo -- an octave down at tempo 0.5."""
    import torchaudio.functional as AF
    new_freq = max(1, int(round(sr / float(tempo))))
    out = AF.resample(w, orig_freq=sr, new_freq=new_freq)
    return out


def trim_audio(audio, d_target):
    """The tree's AUDIO dict, cut (or kept) to at most d_target seconds.
    Never pads: a soundtrack that is too short is information, not a fault."""
    w, sr = _norm_waveform(audio)
    if w is None or d_target is None or d_target <= 0:
        return audio
    keep = target_samples(w.shape[-1], sr, d_target)
    if keep >= w.shape[-1]:
        return audio
    return _pack(w[..., :keep], sr)


def stretch_audio(audio, tempo, pitch_mode="preserve", d_target=None):
    """AUDIO dict -> AUDIO dict at `tempo` (atempo semantics: 2.0 = twice as
    fast). Optionally trimmed to d_target seconds afterwards -- the caller
    that knows the video's exact duration passes it, and the 31 ms tail
    business dies here as a side effect.

    NEVER raises past its own wall: on any failure the ORIGINAL audio comes
    back and the reason is printed. Killing a finished interpolation over its
    soundtrack would invert the priorities."""
    w, sr = _norm_waveform(audio)
    if w is None:
        return audio
    try:
        t = float(tempo)
        if abs(t - 1.0) >= TEMPO_EPS:
            if pitch_mode == "follow speed":
                w = _stretch_follow(w, sr, t)
            else:
                w = _stretch_preserve(w, sr, t)
        out = _pack(w, sr)
        if d_target is not None and d_target > 0:
            out = trim_audio(out, d_target)
        return out
    except Exception as exc:
        print("[PLS] Audio Stretch failed (%s: %s) - the original audio is "
              "passed through unchanged." % (type(exc).__name__, exc))
        return audio


def audio_duration(audio):
    """Seconds of an AUDIO dict, or 0.0."""
    w, sr = _norm_waveform(audio)
    if w is None or sr <= 0:
        return 0.0
    return w.shape[-1] / float(sr)


# ---------------------------------------------------------------------------
# the free node
# ---------------------------------------------------------------------------

_MODES = ["match video", "frames x fps", "target seconds", "factor"]
_PITCH = ["preserve", "follow speed"]


class ULSAudioStretch:
    """Retime a soundtrack to a target the node measures itself."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "mode": (_MODES, {
                    "default": "match video",
                    "tooltip": "Where the target duration comes from. 'match "
                               "video' measures a wired VIDEO; 'frames x fps' "
                               "counts a wired IMAGE batch at the fps below; "
                               "'target seconds' and 'factor' are typed in.",
                }),
                "pitch": (_PITCH, {
                    "default": "preserve",
                    "tooltip": "preserve: ffmpeg atempo - speech stays "
                               "intelligible at its own pitch (the lip-sync "
                               "mode). follow speed: sinc resample - pitch "
                               "scales with tempo, the classic slow-motion "
                               "sound (an octave down at 2x slow).",
                }),
                "fps": ("FLOAT", {
                    "default": 16.0, "min": 0.1, "max": 480.0, "step": 0.1,
                    "tooltip": "Only read in 'frames x fps' mode: the rate "
                               "the wired frames will play at.",
                }),
                "target_seconds": ("FLOAT", {
                    "default": 5.0, "min": 0.05, "max": 36000.0, "step": 0.01,
                    "tooltip": "Only read in 'target seconds' mode.",
                }),
                "factor": ("FLOAT", {
                    "default": 1.0, "min": 0.0625, "max": 16.0, "step": 0.001,
                    "tooltip": "Only read in 'factor' mode. LENGTH factor: "
                               "2.0 makes the audio twice as long (2x slow "
                               "motion), 0.5 half as long.",
                }),
                "trim_to_target": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Cut the result to exactly the target "
                               "duration. Stretching lands within a filter "
                               "window of the target; this removes the "
                               "remainder. Never pads.",
                }),
            },
            "optional": {
                "video": ("VIDEO", {
                    "tooltip": "'match video': the target duration is this "
                               "video's frame count over its frame rate."}),
                "frames": ("IMAGE", {
                    "tooltip": "'frames x fps': the target duration is this "
                               "batch's frame count over the fps widget."}),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio", "info")
    OUTPUT_TOOLTIPS = (
        "The retimed soundtrack.",
        "What was measured and what was done - wire it into a Note.",
    )
    FUNCTION = "run"
    CATEGORY = "Polyhedron/Video"
    DESCRIPTION = (
        "Retimes a soundtrack to follow a retimed picture. Point it at the "
        "video (or frames) and it measures the target itself; or name the "
        "seconds or the factor directly. 'preserve' keeps speech at its own "
        "pitch (the lip-sync mode); 'follow speed' is the classic deep "
        "slow-motion sound. Refuses loudly outside 1/16x..16x - that is a "
        "wiring mistake, not a slow-motion request."
    )

    def _video_duration(self, video):
        try:
            comps = video.get_components()
            n = int(comps.images.shape[0])
            rate = float(comps.frame_rate)
            if n > 0 and rate > 0:
                return n / rate, "%d frames @ %.3f fps" % (n, rate)
        except Exception:
            pass
        return 0.0, "unreadable video"

    def run(self, audio, mode, pitch, fps, target_seconds, factor,
            trim_to_target, video=None, frames=None):
        d_in = audio_duration(audio)
        if d_in <= 0:
            return (audio, "no usable audio wired - passed through")

        if mode == "match video":
            if video is None:
                return (audio, "mode 'match video' but no video wired - "
                               "audio passed through unchanged")
            d_target, how = self._video_duration(video)
            if d_target <= 0:
                return (audio, "video duration could not be measured (%s) - "
                               "audio passed through unchanged" % how)
        elif mode == "frames x fps":
            if frames is None:
                return (audio, "mode 'frames x fps' but no frames wired - "
                               "audio passed through unchanged")
            n = int(frames.shape[0])
            if n <= 0 or fps <= 0:
                return (audio, "frames x fps needs frames and a positive "
                               "fps - audio passed through unchanged")
            d_target, how = n / float(fps), "%d frames @ %.3f fps" % (n, fps)
        elif mode == "target seconds":
            d_target, how = float(target_seconds), "typed target"
        else:  # factor: LENGTH factor, user-facing; tempo is its inverse
            d_target, how = d_in * float(factor), "factor %.4f" % factor

        if d_target <= 0:
            return (audio, "target duration is not positive - audio passed "
                           "through unchanged")
        tempo = d_in / d_target
        if tempo < TEMPO_SANE_MIN or tempo > TEMPO_SANE_MAX:
            note = ("audio stretch refused: tempo %.4f is outside %s..%s - "
                    "this is almost certainly a wiring mistake, not a "
                    "slow-motion request" % (tempo, TEMPO_SANE_MIN,
                                             TEMPO_SANE_MAX))
            print("[PLS] Audio Stretch: " + note)
            return (audio, note)

        out = stretch_audio(audio, tempo, pitch_mode=pitch,
                            d_target=(d_target if trim_to_target else None))
        info = ("audio %.3fs -> %.3fs (tempo %.4f, %s, %s)"
                % (d_in, audio_duration(out), tempo, pitch, how))
        print("[PLS] Audio Stretch: " + info)
        return (out, info)
