# -*- coding: ascii -*-
"""Polyhedron MiniMax Continue (Cine C5, v1021) -- clip n -> clip n+1 on Core master.

WHAT IT IS: the node the Keyframes docstring promised since v964 ("anchors at
other frames ... get their own node once the pack runs on that Core"). It
appends GUIDES to an existing `positive`, with the semantics of Core master's
MiniMaxH3AddGuide (comfy_extras/nodes_minimax_h3.py, 79be670, 25.09.2026):

    {"resolved_frame_index": i, "latent": <video latent>, "audio_latent": <a>}
    appended to cond["minimax_keyframes"]

Three jobs, one node:

  K2  motion context   the last N frames of clip n (Chain Source `tail`) as a
                       guide CLIP at frame 0, with their sound on the same
                       time axis -- AddGuide parity, frame for frame.
  K2L latent context   the same window cut out of clip n's OWN latent (Chain
                       Source `clip_latent`, written by Clip Store) -- no VAE
                       round trip at all. Exact because Core's video VAE
                       encodes independent 17-frame chunks (vae.py
                       encode_temporal, clip_length 17, token_drop 3): a clip
                       of 17k+5 frames ends on a chunk boundary, so its last
                       5m+2 tokens ARE the encoding of its last 17m+5 frames.
                       The audio tail is cut from the clip's audio latent by
                       TIME (FRAME_RESCALE = 5/3 audio latent frames per pixel
                       frame), not re-encoded.
  anchor               one image or clip anywhere in the new clip (frame_idx,
                       negative = from the end) -- keyframe-to-keyframe work,
                       the "mid anchor" of the Cine concept.

WHAT IT IS NOT: K1 (last frame -> first frame) stays with MiniMax Keyframes --
that node also shows the frame to the text encoder as <Picture n>, which a
guide does not. K4 (previous clip as <Video 1>) stays with MiniMax Reference.
Two nodes doing the same thing would be a second truth.

THE LIMIT, DECLARED: on Core 7a131a3a (the pin) the model places keyframes
only at frame 0 / frame_count-1, one frame each, no audio -- and a guide clip
there would not fail loudly, it would be laid out as one frame. So this node
REFUSES by name unless Core has MiniMaxH3AddGuide.

Outputs: positive (guides appended), latent (passed through), overlap (the
frames clip n+1 re-renders from clip n -> Seam Fix trim_head), info.

v1176 -- THE SOUND'S OWN WINDOW (Cine context continuation; the idea from the
H3 Motion Context notes, GPL -- read, not copied): `audio_context` frames of
clip n's sound, pinned so that they END where the pinned picture ends (at the
join) and reach backwards -- a separate audio-only guide at frame
overlap - audio_context. 0 = as before: the sound under the whole picture
window from frame 0 (Core's Add Guide, frame for frame). A multiple of 3 is
a whole number of audio latent frames (a pixel frame is 5/3 of one), so the
window is pinned at exactly the width asked; another value is widened to the
next multiple of 3. Their measured balance on music chains: 18 (longer holds
the sound closer but dulls it at every join, shorter lets it wander).
K2 cuts the last audio_context/24 s of tail_audio (exact to the sample); K2L
cuts the clip's audio latent BY TIME and names the sub-step offset of that cut
(a 17k+5 clip ends on a third of an audio step, so the cut can be up to half a
step, 12.5 ms, off -- said in the info, not hidden).

v1178 -- HOLD IN THE LATENT (`hold`; the "hard context" of the Motion Context
notes / Easy Media, GPL: read, not copied; the library is ph_av_mask): besides
the guide, the context is COPIED to the start of the run's own latent -- the
picture's tokens and the sound under them -- and held by a noise mask: locked,
then released over `release_frames` (picture, whole tokens nearest to it) and 4 audio
latent frames (sound, 100 ms) INSIDE the window, so the new clip grows out of
the old one instead of only being told about it. The `latent` output then
carries the copy and the mask: wire it into the sampler. "guide only" = as
before, the latent passes through untouched. Core master runs the held rows as
clean context; the pinned Core would only blend them back (said) -- Continue
needs master for its guides anyway.
v1182 (the audit): the HOLD keeps the sound under the WHOLE picture window (22
frames = 37 audio frames, cut by time like K2L's guide), not the guide's own
sound window -- `audio_context` shapes the guide only. The v1178 changelog said
"the same cut as the guide's"; that held only with audio_context 0.
"""


import node_helpers

from . import ph_av_mask as AM
from . import uls_latent_math as M

FRAME_RESCALE = AM.FRAME_RESCALE   # audio latent frames per pixel frame (Core model.py) -- v1183: one truth (ph_av_mask)
CHUNK_TOKENS = 5                # 17 frames -> 5 tokens (Core vae.py)
MODES = ["K2L latent context (no VAE)", "K2 motion context (frames)", "anchor only"]
HOLDS = ["guide only", "guide + hold in the latent"]   # v1178
AUDIO_RELEASE = 4               # v1178: audio latent frames (100 ms) the held sound is released over
AUDIO_GRID = 3                  # pixel frames per whole number of audio latent frames (3 * 5/3 = 5)


# --------------------------------------------------------------------------
# pure helpers (the guard drives these without Core)
# --------------------------------------------------------------------------

def snap_guide(n):
    """AddGuide's rule for a guide of n frames: < 5 -> 1, else down to 17k+5."""
    n = int(n)
    if n < 5:
        return 1
    while n % 17 != 5:
        n -= 1
    return n


def resolve_index(frame_idx, frame_count, guide_frames):
    """AddGuide's index rule; raises by name when the guide does not fit."""
    i = frame_idx if frame_idx >= 0 else frame_count + frame_idx
    if i < 0 or i + guide_frames > frame_count:
        if guide_frames == 1:
            raise ValueError("frame_idx %d is outside the video's %d frames"
                             % (frame_idx, frame_count))
        raise ValueError("a %d frame guide clip at frame_idx %d does not fit in "
                         "the video's %d frames" % (guide_frames, frame_idx, frame_count))
    return i


def latent_window(latent_t, frames):
    """(start_token, n_tokens) of the LAST `frames` (17m+5) of a clip latent
    with latent_t tokens. Raises when the window would not start on a chunk
    boundary -- then it would not be the encoding of those frames."""
    n_tok = M.minimax_video_latent_t(frames)
    if n_tok > latent_t:
        raise ValueError("the clip latent has %d tokens, a %d frame window needs %d"
                         % (latent_t, frames, n_tok))
    start = latent_t - n_tok
    if start % CHUNK_TOKENS != 0:
        raise ValueError("a window of %d tokens from a %d-token latent starts at "
                         "token %d, off the 17-frame chunk grid -- not a clip "
                         "latent of 17k+5 frames" % (n_tok, latent_t, start))
    return start, n_tok


def audio_window(clip_frames, frames, audio_t):
    """(start, end) audio latent frames of the last `frames` pixel frames of a
    clip of clip_frames, by time on Core's shared axis."""
    start = int(round((clip_frames - frames) * FRAME_RESCALE))
    return max(0, min(start, audio_t)), audio_t


def audio_frames(n, overlap):
    """v1176: the sound window in pixel frames -> (frames, audio latent frames, guide index).
    0 = None (the sound rides under the picture window, Add Guide parity). Else widened to the 3-frame
    grid, at most the picture window, pinned to END at the join: the guide sits at frame overlap - frames."""
    n, overlap = int(n), int(overlap)
    if n <= 0 or overlap < AUDIO_GRID:
        return None
    n = -(-n // AUDIO_GRID) * AUDIO_GRID                # up to the grid
    n = min(n, overlap // AUDIO_GRID * AUDIO_GRID)      # never past the picture window
    steps = int(round(n * FRAME_RESCALE))
    return n, steps, overlap - n


def audio_tail_cut(clip_frames, frames, clip_audio_t):
    """v1176: the last `frames` pixel frames of a clip's audio latent, by time -> (start, end, offset_ms).
    The window ends at the clip's end (clip_frames * 5/3 audio latent frames on Core's axis); its start is
    rounded to a whole latent frame, and the error of that rounding is named in ms (one step = 25 ms)."""
    steps = int(round(frames * FRAME_RESCALE))
    t_end = clip_frames * FRAME_RESCALE
    t0 = t_end - frames * FRAME_RESCALE
    start = int(round(t0))
    start = max(0, min(start, int(clip_audio_t) - steps))
    return start, start + steps, (start - t0) * 25.0


def _parts(latent):
    s = latent.get("samples") if isinstance(latent, dict) else latent
    if not getattr(s, "is_nested", False):
        return None
    parts = s.unbind()
    if not parts or getattr(parts[0], "ndim", 0) != 5 or parts[0].shape[1] != M.MINIMAX_VIDEO_CH:
        return None
    return parts


def model_has_guides():
    """Does ComfyUI's H3 MODEL code place keyframes anywhere? Read from the
    library (comfy/ldm/minimax/model.py), not from a node: the pinned Core's
    PackedLayout refuses everything but the first / last frame."""
    try:
        import inspect
        import comfy.ldm.minimax.model as mm
        src = inspect.getsource(mm.PackedLayout)
    except Exception:
        return None
    return guides_in_source(src)


def guides_in_source(src):
    """The predicate on the model code (kept apart so a guard can feed it both kinds of Core)."""
    return "resolved_frame_index" in src and "only first/last keyframe anchors" not in src


def core_has_guides():
    """True when guides can sit anywhere: the model code says so (v1057).
    v1061: an unreadable model library counts as "no" -- the guess from Core's
    node module is gone with the rest of the node-class calls;
    a "no" refuses guides by name and never places them wrongly."""
    return model_has_guides() is True


def _resize(image, width, height):
    """Core's _resize(..., "center"), mirrored: [B,H,W,C] -> [B,height,width,3]."""
    import comfy.utils
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")
    return samples.movedim(1, -1)


def _encode_audio(audio_vae, audio):
    """Core's _encode_ref_audio, mirrored (the Reference node's copy)."""
    from .ph_minimax_ref import encode_ref_audio
    return encode_ref_audio(audio_vae, audio)


def _last_seconds(audio, seconds):
    wav = audio["waveform"]
    sr = int(audio["sample_rate"])
    n = int(round(seconds * sr))
    if 0 < n < wav.shape[-1]:
        wav = wav[..., -n:]
    return {"waveform": wav, "sample_rate": sr}


# --------------------------------------------------------------------------
# the node
# --------------------------------------------------------------------------

class ULSMiniMaxContinue:
    """Guides from clip n (frames or latent) and mid anchors, on Core master."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING", {"tooltip":
                    "The run's positive conditioning (MiniMax Keyframes, "
                    "MiniMax Reference or CLIP Text Encode). The guides are "
                    "APPENDED -- references and keyframes already in it stay."}),
                "latent": ("LATENT", {"tooltip":
                    "The run's H3 AV latent (Polyhedron Empty Latent, 'MiniMax "
                    "H3 AV'). Size and length come from it; it is passed on "
                    "unchanged unless `hold` copies the context into it."}),
                "mode": (MODES, {"default": MODES[0], "tooltip":
                    "K2L: continue from clip n's OWN latent (Chain Source "
                    "clip_latent) -- no VAE round trip. K2: from its decoded "
                    "last frames (Chain Source tail + tail_audio). anchor only: "
                    "no context, just the anchor input."}),
                "context_frames": ("INT", {"default": 22, "min": 5, "max": 362,
                    "tooltip": "How much of clip n's end guides the start of "
                    "clip n+1, in frames. Snapped down to the H3 clip grid "
                    "5, 22, 39, ... (17k+5). 22 = 0.92 s."}),
                "with_audio": ("BOOLEAN", {"default": True, "tooltip":
                    "Carry clip n's sound under the context on the same time "
                    "axis, so the soundtrack continues too."}),
                "anchor_frame": ("INT", {"default": -1, "min": -9999, "max": 9999,
                    "tooltip": "Where the `anchor` input goes: its (first) "
                    "frame lands here. Negative counts from the end, -1 = the "
                    "last frame. Ignored without an anchor."}),
                "audio_context": ("INT", {"default": 0, "min": 0, "max": 362,
                    "tooltip": "How far back clip n's SOUND reaches, in frames, pinned to end at the "
                               "join (where the picture window ends). 0 = the sound under the whole picture "
                               "window from frame 0 (Core's Add Guide). Multiples of 3 land exactly on the "
                               "audio grid (others are widened). 18 holds a soundtrack well across a join; "
                               "longer holds it closer but dulls it a little at every join."}),
                "hold": (HOLDS, {"default": HOLDS[0], "tooltip":
                    "guide only: the end of clip n is a guide (as before), the latent passes through. "
                    "guide + hold in the latent: the same end is ALSO copied to the start of this run's latent "
                    "(picture and sound) and held by a noise mask, then released -- a harder join. Wire the "
                    "`latent` output into the sampler."}),
                "release_frames": ("INT", {"default": 16, "min": 0, "max": 64, "step": 4, "tooltip":
                    "With hold: over how many of the held frames the picture is released again, counted back from "
                    "the join (rounded to whole tokens -- a token is 1 or 4 frames; the info says how many frames it "
                    "came to); the frames before them stay locked. 0 = held hard up to the join. The sound is "
                    "released over its last 100 ms."}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Video VAE -- needed for K2 and for "
                    "an anchor image. K2L needs none."}),
                "audio_vae": ("VAE", {"tooltip": "Audio VAE -- needed for K2 "
                    "with audio. K2L needs none."}),
                "clip_latent": ("LATENT", {"tooltip":
                    "K2L: Clip Chain Source `clip_latent` -- clip n's AV "
                    "latent as Clip Store kept it."}),
                "tail": ("IMAGE", {"tooltip": "K2: Clip Chain Source `tail` -- "
                    "clip n's last frames (5 or more)."}),
                "tail_audio": ("AUDIO", {"tooltip": "K2: Clip Chain Source "
                    "`tail_audio` -- cut to the context length here."}),
                "anchor": ("IMAGE", {"tooltip": "An image (or a clip of 5+ "
                    "frames, snapped to 17k+5) to pin at `anchor_frame`: a "
                    "target pose, the end of a move, the next keyframe."}),
                "anchors": ("PLS_ANCHORS", {"tooltip": "Cine Timeline `anchors`: "
                    "every anchor between the first and the last frame, each at "
                    "its own frame."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "STRING")
    RETURN_NAMES = ("positive", "latent", "overlap", "info")
    FUNCTION = "build"
    CATEGORY = "Polyhedron/Cine"
    DESCRIPTION = ("Continues clip n into clip n+1 on Core master: the end of "
                   "clip n as a guide at frame 0 -- from its own latent (K2L, "
                   "no VAE round trip) or its decoded frames (K2) -- with its "
                   "sound on the same time axis, plus an optional anchor "
                   "anywhere in the new clip. Appends to the conditioning like "
                   "Core's Add Guide; `overlap` goes to Seam Fix trim_head. "
                   "Refuses on the pinned Core, which anchors only first/last "
                   "frames.")

    def build(self, positive, latent, mode, context_frames, with_audio, anchor_frame, audio_context=0,
              hold="guide only", release_frames=16, vae=None, audio_vae=None, clip_latent=None, tail=None,
              tail_audio=None, anchor=None, anchors=None):
        tag = "[PLS] MiniMax Continue"
        # v1051: nothing to add (anchor only, no anchor, no timeline guides) =
        # pass through, on ANY Core -- so one workflow serves a first/last-
        # frame run and a run with guides in between. Said, not silent.
        if mode.startswith("anchor") and anchor is None and not (anchors or []):
            info = ("MiniMax Continue: nothing to add -- no guides between the first and the "
                    "last frame, no anchor; positive and latent passed through unchanged")
            print("[PLS] " + info)
            return (positive, latent, 0, info)
        if not core_has_guides():
            raise ValueError(
                tag + ": this Core has no MiniMaxH3AddGuide -- it places H3 "
                "anchors only at the first and last frame, one frame each, no "
                "audio. Continue needs Core master (0.37+). On this Core: "
                "MiniMax Keyframes for the last frame (K1), MiniMax Reference "
                "with the previous clip as <Video 1> (K4).")
        parts = _parts(latent)
        if parts is None:
            raise ValueError(tag + ": `latent` is not a MiniMax H3 AV latent "
                             "(Polyhedron Empty Latent, latent_type 'MiniMax H3 AV').")
        video_t, audio_t = parts[0], parts[1]
        height, width = int(video_t.shape[3]) * 16, int(video_t.shape[4]) * 16
        frame_count = M.minimax_frames_from_latent_t(int(video_t.shape[2]))
        target_audio = int(audio_t.shape[-1])

        guides, lines, overlap = [], [], 0
        hold_v = hold_a = z_full = None
        lines.append("MiniMax Continue: target %dx%d, %d frames (%.2f s), mode %s"
                     % (width, height, frame_count, frame_count / 24.0, mode.split(" (")[0]))

        if mode.startswith("K2L"):
            if clip_latent is None:
                raise ValueError(tag + ": K2L needs `clip_latent` (Clip Chain Source) "
                                 "-- and that needs Clip Store after the previous clip's Save.")
            cp = _parts(clip_latent)
            if cp is None:
                raise ValueError(tag + ": `clip_latent` is not an H3 AV latent.")
            cv = cp[0]
            if (int(cv.shape[3]), int(cv.shape[4])) != (int(video_t.shape[3]), int(video_t.shape[4])):
                raise ValueError(tag + ": clip n is %dx%d, this run is %dx%d -- a latent "
                                 "context shares the target's grid; keep the size of the chain"
                                 % (int(cv.shape[4]) * 16, int(cv.shape[3]) * 16, width, height))
            clip_frames = M.minimax_frames_from_latent_t(int(cv.shape[2]))
            n = snap_guide(min(int(context_frames), clip_frames))
            if n < 5:
                raise ValueError(tag + ": a latent context needs 5 frames or more")
            start, n_tok = latent_window(int(cv.shape[2]), n)
            resolve_index(0, frame_count, n)
            kf = {"resolved_frame_index": 0, "latent": cv[:, :, start:].clone()}
            line = "K2L: last %d frame(s) of clip n (tokens %d..%d of %d) at frame 0, no VAE" % (
                n, start, start + n_tok - 1, int(cv.shape[2]))
            aw = audio_frames(audio_context, n)
            if with_audio and len(cp) > 1 and aw is not None:
                ca = cp[1]
                af, steps, at = aw
                a0, a1, off = audio_tail_cut(clip_frames, af, int(ca.shape[-1]))
                sound = {"resolved_frame_index": at, "audio_latent": ca[..., a0:a1].clone()}
                line += (" + sound: its last %d frame(s) (audio latent %d..%d, %d steps) at frame %d, ending at "
                         "the join; the cut is %+.1f ms off clip n's end (the audio grid)" % (af, a0, a1 - 1, steps, at, off))
            elif with_audio and len(cp) > 1:
                ca = cp[1]
                a0, a1 = audio_window(clip_frames, n, int(ca.shape[-1]))
                a = ca[..., a0:a1]
                if a.shape[-1] > target_audio:
                    a = a[..., :target_audio]
                if a.shape[-1] >= 1:
                    kf["audio_latent"] = a.clone()
                    line += " + sound (audio latent %d..%d)" % (a0, a0 + a.shape[-1] - 1)
            elif with_audio:
                line += " -- clip n has no audio latent, no sound carried"
            guides.append(kf)
            if with_audio and len(cp) > 1 and aw is not None:
                guides.append(sound)
            lines.append(line)
            overlap = n
            hold_v = kf["latent"]
            if hold.startswith("guide +") and with_audio and len(cp) > 1:   # v1183: only when a hold reads it
                h0, h1, _hoff = audio_tail_cut(clip_frames, n, int(cp[1].shape[-1]))
                hold_a = cp[1][..., h0:h1].clone()

        elif mode.startswith("K2 "):
            if tail is None and tail_audio is None:
                # v1176: the timeline hands out no tail for a scene that is not carried (scene 1, a cut) --
                # this run starts free; said, and the anchors below still go in
                lines.append("K2: no tail -- nothing to continue from (the first clip, or a scene the timeline does "
                             "not carry); this clip starts free, overlap 0")
            else:
                if tail is None or int(tail.shape[0]) < 5:
                    raise ValueError(tag + ": K2 needs `tail` with 5 frames or more "
                                     "(Clip Chain Source tail, tail_frames 22)")
                if vae is None:
                    raise ValueError(tag + ": K2 encodes the frames -- wire `vae`")
                n = snap_guide(min(int(context_frames), int(tail.shape[0])))
                resolve_index(0, frame_count, n)
                frames = _resize(tail[-n:], width, height)
                kf = {"resolved_frame_index": 0, "latent": vae.encode(frames)}
                line = "K2: last %d frame(s) of clip n at frame 0 (VAE encode)" % n
                aw = audio_frames(audio_context, n)
                sound = None
                if with_audio and tail_audio is not None and aw is not None:
                    if audio_vae is None:
                        raise ValueError(tag + ": the sound needs `audio_vae` "
                                         "(or switch with_audio off)")
                    af, steps, at = aw
                    z, rt = _encode_audio(audio_vae, _last_seconds(tail_audio, af / 24.0))
                    z = z[..., :steps].clone() if z.shape[-1] > steps else z
                    sound = {"resolved_frame_index": at, "audio_latent": z}
                    line += (" + sound: the last %d frame(s) of tail_audio (%d audio latent frames) at frame %d, "
                             "ending at the join" % (af, z.shape[-1], at))
                elif with_audio and tail_audio is not None:
                    if audio_vae is None:
                        raise ValueError(tag + ": the sound needs `audio_vae` "
                                         "(or switch with_audio off)")
                    z, rt = _encode_audio(audio_vae, _last_seconds(tail_audio, n / 24.0))
                    z_full = z                    # v1183: the hold below reuses this encoding (the same window)
                    if rt > target_audio:
                        z = z[..., :target_audio].clone()
                    kf["audio_latent"] = z
                    line += " + sound (%d audio latent frames)" % z.shape[-1]
                elif with_audio:
                    line += " -- no tail_audio wired, no sound carried"
                guides.append(kf)
                if sound is not None:
                    guides.append(sound)
                lines.append(line)
                overlap = n
                hold_v = kf["latent"]
                if hold.startswith("guide +") and with_audio and tail_audio is not None and audio_vae is not None:
                    # v1183: the guide already encoded the same last n/24 s when audio_context is 0 -- one encode
                    hz = z_full if z_full is not None else _encode_audio(audio_vae, _last_seconds(tail_audio, n / 24.0))[0]
                    steps_n = AM.audio_steps(n)
                    hold_a = hz[..., :steps_n].clone() if hz.shape[-1] > steps_n else hz

        # v1023: the timeline's anchors, then the single `anchor` -- the same
        # rules for both (AddGuide's snap, index and fit)
        todo = [(a["image"], int(a["frame"]), a.get("src", "timeline")) for a in (anchors or [])]
        if anchor is not None:
            todo.append((anchor, int(anchor_frame), "input"))
        for img, at, name in todo:
            if vae is None:
                raise ValueError(tag + ": an anchor image is encoded -- wire `vae`")
            g = snap_guide(int(img.shape[0]))
            i = resolve_index(at, frame_count, g)
            kf = {"resolved_frame_index": i,
                  "latent": vae.encode(_resize(img[:g], width, height))}
            guides.append(kf)
            line = "anchor %s: %s at frame %d (%.2f s)" % (
                name, "1 image" if g == 1 else "%d-frame clip" % g, i, i / 24.0)
            if overlap and i < overlap:
                line += "  WARN: inside the context (frames 0..%d) -- two guides on the same frames" % (overlap - 1)
            lines.append(line)

        # exactly AddGuide's append: the list of the FIRST entry, set on all (v1176: nothing to append -- the
        # conditioning passes unchanged)
        if guides:
            keyframes = list(positive[0][1].get("minimax_keyframes", [])) + guides
            out = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})
        else:
            out = positive

        lat_out = latent
        if hold.startswith("guide +"):
            if hold_v is None:
                lines.append("hold: nothing to hold -- no context in this run (the latent passes through)")
            else:
                rel, rel_f = AM.tokens_back(int(hold_v.shape[2]), release_frames)
                lat_out, rep = AM.hold_prefix(latent, hold_v, hold_a, video_release=rel,
                                              audio_release=AUDIO_RELEASE, tag=tag)
                hv = rep["video"]
                line = ("hold: the context is copied to the start of the latent -- picture %d token(s) (frames 0..%d): "
                        "%d locked, %d released over its last %d frame(s) (%s)"
                        % (hv["tokens"], overlap - 1, hv["locked"], hv["release"], rel_f,
                           " ".join("%.2f" % x for x in hv["values"][hv["locked"]:]) or "-"))
                if rep["audio"]:
                    ha = rep["audio"]
                    line += ("; sound %d audio frame(s) (%.2f s, under the whole picture window%s): %d locked, %d released"
                             % (ha["steps"], ha["seconds"], "" if not int(audio_context) else
                                "; the guide's sound window is audio_context %d" % int(audio_context),
                                ha["locked"], ha["release"]))
                elif with_audio:
                    line += "; no sound held (no audio context in this run)"
                lines.append(line)
                lines.append("  " + AM.core_line() + "; wire `latent` into the sampler")
        if overlap:
            lines.append("overlap: clip n+1 re-renders its first %d frame(s) from clip n "
                         "-> Seam Fix trim_head %d" % (overlap, overlap))
        info = "\n".join(lines)
        print("[PLS] " + info.replace("\n", "\n[PLS]   "))
        return (out, lat_out, int(overlap), info)


NODE_CLASS_MAPPINGS = {"ULSMiniMaxContinue": ULSMiniMaxContinue}
NODE_DISPLAY_NAME_MAPPINGS = {"ULSMiniMaxContinue": "\u2b21 Polyhedron MiniMax Continue"}
