"""Polyhedron MiniMax Reference -- <Picture i> conditioning on OUR rails.

WHAT THIS REPLACES: Core's MiniMaxH3ReferenceToVideo, plus the Resolution
Selector, the Float (Duration) and the Math Expression that the stock template
needs around it. Four boxes become one -- and the latent stops being built under
the hood.

THE DIFFERENCE THAT MATTERS (Frank's call, v872): Core's node BUILDS its own
joint latent internally and has no latent input, so the size, the clip length
and the init noise all live inside a box you cannot inspect. This node takes the
latent as an INPUT. ULSEmptyLatent (latent_type = "MiniMax H3 AV") makes it,
ULSSeed can shape its noise, and every number is on a wire. Core itself ships
EmptyMiniMaxH3LatentAV as a standalone node, so this split follows Core's own
design rather than working around it.

ONE SOURCE OF TRUTH: width, height and frame count are RECOVERED from the wired
latent, never re-entered as widgets. The video half is
[B, 24, latent_t, H//16, W//16] and align_frame_count guarantees
frame_count % 17 == 5, so the inverse is exact (uls_latent_math.
minimax_frames_from_latent_t). A second set of size widgets could drift out of
step with the latent; recovered numbers cannot.

WHERE WE BEAT CORE: Core scales every reference by ONE global rule -- the
generation's pixel area ("match") or a 2048 short edge ("max"). This node keeps
that as the default and adds a PER-IMAGE megapixel target, the same control
ULSReference has. A face reference and a background reference rarely deserve the
same budget, and reference tokens ride through every sampling step.

THE COUPLING, DECLARED. Two Core internals carry the whole <Picture i>
mechanism, and there is no way around them -- the tags are resolved INSIDE the
qwen3vl tokenizer, which needs the image tensors:

    tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
    cond   = node_helpers.conditioning_set_values(cond, {"minimax_refs": blocks})

Both names are pinned by test_v872_minimax_ref against Core's own source when it
can be found, so a rename upstream goes RED here instead of silently producing
a run with no references. Everything else in this file is ordinary public API
(comfy.utils.common_upscale, vae.encode) or arithmetic this pack owns
(uls_latent_math.minimax_*).
"""

import math

import comfy.utils
import node_helpers

from . import uls_latent_math as M
from . import h3_prompt as HP

_SIZE_MODES = ["match", "max"]
REF_IMAGE_SHORT_EDGE = 2048   # Core's constant, mirrored

# v1016 -- EVERY INPUT MINIMAX H3 TAKES. MiniMax's own README (Ref2VA):
#   <= 9 images; <= 3 videos, each 2-15 s, together <= 15 s;
#   <= 3 audios, each 2-15 s, together <= 15 s; <= 12 files in total.
# Core's MiniMaxH3ReferenceToVideo exposes exactly these counts (Autogrow max
# 9 / 3 / 3 / 3) and checks none of the durations. This node exposes the same
# counts and REPORTS every official limit it sees crossed -- loudly, without
# refusing: the limits are the API's, the local model has no hard stop there,
# and a refusal would take away an experiment. The one thing refused is what
# cannot be computed at all (an audio without an audio VAE, a video under
# five frames).
N_IMAGES = 9
N_VIDEOS = 3
N_AUDIOS = 3
LIMIT_FILES = 12
LIMIT_CLIP_MIN_S = 2.0
LIMIT_CLIP_MAX_S = 15.0
LIMIT_TOTAL_S = 15.0
BASE_SHORT_EDGE = 768             # Core's adapt_canvas constants, mirrored
MAX_PIXELS = 768 * 1344
AUDIO_SAMPLE_RATE = 32000         # Core's fallback when the VAE does not say


def adapt_canvas(src_w, src_h):
    """Core's adapt_canvas, mirrored: 768 short edge, 768*1344 area cap,
    each axis rounded to the 32 grid."""
    ratio = float(src_w) / float(src_h)
    if ratio >= 1.0:
        nom_w, nom_h = BASE_SHORT_EDGE * ratio, float(BASE_SHORT_EDGE)
    else:
        nom_w, nom_h = float(BASE_SHORT_EDGE), BASE_SHORT_EDGE / ratio
    if nom_w * nom_h > MAX_PIXELS:
        k = math.sqrt(MAX_PIXELS / (nom_w * nom_h))
        nom_w, nom_h = nom_w * k, nom_h * k
    return _snap(nom_w), _snap(nom_h)


def ref_video_canvas(src_w, src_h):
    """Core's rule for a reference VIDEO: the adapted canvas, but a source that
    is smaller than it keeps its own size (snapped) -- never upscaled."""
    cw, ch = adapt_canvas(src_w, src_h)
    if src_w * src_h < cw * ch:
        cw, ch = _snap(src_w), _snap(src_h)
    return cw, ch


def resample_indices(n_src, src_fps, dst_fps=24.0):
    """Frame indices that turn n_src frames at src_fps into dst_fps (nearest
    frame). Core ASSUMES its reference video is already 24 fps; a 30 fps clip
    then plays 25 % slow inside the model's sense of time. Pure."""
    if n_src <= 0:
        return []
    if src_fps <= 0 or abs(float(src_fps) - float(dst_fps)) < 1e-6:
        return list(range(n_src))
    dur = n_src / float(src_fps)
    n_out = max(1, int(math.floor(dur * dst_fps + 1e-6)))
    return [min(n_src - 1, int(round(i * float(src_fps) / dst_fps)))
            for i in range(n_out)]


def ref_video_frames(n, frame_count):
    """Core's trim: at most the generation's frame count, then DOWN to the
    17k+5 grid. None below five frames (Core raises there too)."""
    n = min(int(n), int(frame_count))
    if n < 5:
        return None
    while n % 17 != 5:
        n -= 1
    return n


def plan_tags(n_images, videos, n_audios):
    """The label order the tokenizer will produce, mirrored from Core:
    all images, then per video its soundtrack (<Audio j>) BEFORE the video,
    then the standalone audios. videos: [has_soundtrack, ...]. Returns
    [(kind, tag), ...]; kind in image / soundtrack / video / audio. Pure --
    this is what makes the info output's tags the tags the model sees."""
    out = []
    c = {"image": 0, "audio": 0, "video": 0}
    for _ in range(n_images):
        c["image"] += 1
        out.append(("image", "<Picture %d>" % c["image"]))
    for has_sound in videos:
        if has_sound:
            c["audio"] += 1
            out.append(("soundtrack", "<Audio %d>" % c["audio"]))
        c["video"] += 1
        out.append(("video", "<Video %d>" % c["video"]))
    for _ in range(n_audios):
        c["audio"] += 1
        out.append(("audio", "<Audio %d>" % c["audio"]))
    return out


def limit_notes(n_images, video_secs, audio_secs):
    """Every official Ref2VA limit that is crossed, as report lines. Pure.
    video_secs / audio_secs: [(tag, seconds), ...] -- audio includes the
    soundtracks, which occupy an <Audio j> label like any other audio."""
    notes = []
    for label, items in (("video", video_secs), ("audio", audio_secs)):
        for tag, sec in items:
            if sec < LIMIT_CLIP_MIN_S or sec > LIMIT_CLIP_MAX_S:
                notes.append("LIMIT: %s is %.2f s -- MiniMax specifies %g-%g s "
                             "per %s reference"
                             % (tag, sec, LIMIT_CLIP_MIN_S, LIMIT_CLIP_MAX_S,
                                label))
        total = sum(sec for _, sec in items)
        if total > LIMIT_TOTAL_S + 1e-6:
            notes.append("LIMIT: the %s references add up to %.2f s -- MiniMax "
                         "specifies at most %g s together"
                         % (label, total, LIMIT_TOTAL_S))
    files = n_images + len(video_secs) + len(audio_secs)
    if files > LIMIT_FILES:
        notes.append("LIMIT: %d reference files (soundtracks counted) -- "
                     "MiniMax specifies at most %d" % (files, LIMIT_FILES))
    if audio_secs and not n_images and not video_secs:
        notes.append("LIMIT: audio without any image or video reference -- "
                     "the MiniMax API refuses that combination; here it runs, "
                     "untested")
    return notes


def _video_parts(video):
    """(frames [N,H,W,C], fps) out of a comfy_api VIDEO. Public API only."""
    comp = video.get_components()
    frames = comp.images
    try:
        fps = float(comp.frame_rate)
    except Exception:
        fps = 0.0
    return frames, fps


def _audio_seconds(audio):
    try:
        return float(audio["waveform"].shape[-1]) / float(audio["sample_rate"])
    except Exception:
        return 0.0


def encode_ref_audio(audio_vae, audio):
    """Core's _encode_ref_audio, mirrored: resample to the VAE's rate, encode
    the first batch item. Returns (latent, latent_t)."""
    waveform = audio["waveform"]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", AUDIO_SAMPLE_RATE))
    if sr != vae_sr:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, int(z.shape[-1])


def _resize(image, width, height):
    """Core's _resize, mirrored: [B, H, W, C] -> [B, height, width, 3]."""
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height,
                                         "lanczos", "disabled")
    return samples.movedim(1, -1)


# v875: above this ratio the report calls the spread out by name. 2.0 is not a
# tuned number -- it is the point where one reference carries twice the tokens
# of another, which is already visible in a result and is never what someone
# meant to set up.
BALANCE_WARN = 2.0


def balance_note(cells):
    """cells: [(picture_number, latent_cells), ...] -> one report line or None.

    WHY THIS EXISTS: 'match' scales a reference DOWN to the generation's pixel
    area and never up, so two sources of very different size end up with very
    different token counts -- and reference tokens ride through EVERY sampling
    step. A 1:5 spread is not a subtle effect, but until v875 nothing said so;
    it only showed up in the picture, where it looks like a model problem
    instead of a budget problem.

    Pure, so the guard can drive it with numbers instead of reading it.
    """
    if len(cells) < 2:
        return None
    lo_n, lo = min(cells, key=lambda c: c[1])
    hi_n, hi = max(cells, key=lambda c: c[1])
    # ROUND ONCE, then judge the rounded number. Comparing the raw ratio while
    # printing a rounded one prints "1 : 2.0 (balanced)" next to
    # "1 : 2.0 -- UNBALANCED" on the very next run, which teaches the reader to
    # distrust the line.
    ratio = round(hi / float(max(1, lo)), 1)
    if lo == hi:
        return ("reference weight: all %d references carry %d latent cells "
                "-- 1 : 1.0 (balanced)" % (len(cells), lo))
    head = ("reference weight: <Picture %d> %d cells vs <Picture %d> %d cells "
            "-- 1 : %.1f" % (lo_n, lo, hi_n, hi, ratio))
    if ratio < BALANCE_WARN:
        return head + " (balanced)"
    return (head + " -- UNBALANCED. Reference tokens ride through EVERY "
            "sampling step, so the heavier one weighs more on the result. "
            "megapixels_n scales a reference DOWN; there is no way up, because "
            "a reference is never upscaled.")


def _snap(value):
    """Round to the model's 32 canvas grid, never below one cell."""
    m = M.MINIMAX_CANVAS_MULTIPLE
    return max(m, int(round(value / float(m))) * m)


def ref_scale(src_w, src_h, gen_w, gen_h, mode, megapixels):
    """The scale factor for ONE reference image. Pure, so the guard can drive it.

    megapixels > 0 -> our per-image target (ULSReference's control).
    otherwise      -> Core's rule: 'match' scales to the generation's pixel
                      area, 'max' to a 2048 short edge.

    DOWN ONLY in every branch, exactly like Core: a reference is never upscaled,
    because inventing pixels for something the model will treat as evidence is
    the wrong kind of help.
    """
    area = float(src_w) * float(src_h)
    if area <= 0:
        return 1.0
    if megapixels and megapixels > 0:
        return min(1.0, math.sqrt((float(megapixels) * 1000000.0) / area))
    if mode == "max":
        return min(1.0, REF_IMAGE_SHORT_EDGE / float(min(src_w, src_h)))
    return min(1.0, math.sqrt((float(gen_w) * float(gen_h)) / area))


class ULSMiniMaxReference:
    """<Picture i> reference conditioning for MiniMax H3, latent from a wire."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip":
                                  "The MiniMax H3 text encoder. The reference "
                                  "images are woven into the TOKENS here -- "
                                  "this is not an ordinary text encode and "
                                  "cannot be moved upstream."}),
                "vae": ("VAE", {"tooltip":
                                "The VIDEO vae. Each reference image is "
                                "encoded with it before it joins the "
                                "conditioning."}),
                "latent": ("LATENT", {"tooltip":
                                      "A joint MiniMax H3 AV latent -- from "
                                      "Polyhedron Empty Latent with "
                                      "latent_type 'MiniMax H3 AV'. Width, "
                                      "height and frame count are READ from "
                                      "it, so there is exactly one source of "
                                      "truth for the size."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "default": "",
                                      "tooltip":
                                      "Refer to the references as <Picture i>, "
                                      "<Video k> and <Audio j>. Each kind is "
                                      "counted in WIRED order; a video's "
                                      "soundtrack takes an <Audio j> number "
                                      "BEFORE the standalone audios. The info "
                                      "output lists the tags that are "
                                      "actually live."}),
                "ref_image_size": (_SIZE_MODES, {"default": "match",
                                   "tooltip":
                                   "The default rule for every reference "
                                   "without its own megapixel target. "
                                   "'match': scale to the generation's pixel "
                                   "area. 'max': a 2048 short edge, best "
                                   "identity fidelity. Reference tokens ride "
                                   "through EVERY sampling step, so 'max' can "
                                   "be several times slower."}),
                "megapixels_1": ("FLOAT", {"default": 0.0, "min": 0.0,
                                           "max": 16.0, "step": 0.1,
                                           "tooltip":
                                           "Per-image budget for image_1. 0 = "
                                           "follow ref_image_size. Core has "
                                           "only the global rule; a face and a "
                                           "backdrop rarely deserve the same "
                                           "token budget."}),
                "megapixels_2": ("FLOAT", {"default": 0.0, "min": 0.0,
                                           "max": 16.0, "step": 0.1,
                                           "tooltip":
                                           "Per-image budget for image_2. "
                                           "0 = follow ref_image_size."}),
                "megapixels_3": ("FLOAT", {"default": 0.0, "min": 0.0,
                                           "max": 16.0, "step": 0.1,
                                           "tooltip":
                                           "Per-image budget for image_3. "
                                           "0 = follow ref_image_size."}),
            },
            # v1016: EVERY INPUT MINIMAX H3 TAKES. ORDER IS LOAD-BEARING:
            # image_1..3 stay first (saved workflows); megapixels_4..9 are
            # the only new WIDGETS and follow megapixels_3 in the widget
            # order (the #577 append law); everything else is a socket.
            # Written out literally so the #577 gate can read it without
            # importing. ph_minimax_ref.js shows a socket only once the one
            # before it is wired, and a megapixels_n only with its image_n.
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Reference image 1. Numbered in WIRED order: the first wired image is <Picture 1>, whatever its slot. The info output lists the live tags."}),
                "image_2": ("IMAGE", {"tooltip": "Reference image 2. Numbered in WIRED order: the first wired image is <Picture 1>, whatever its slot. The info output lists the live tags."}),
                "image_3": ("IMAGE", {"tooltip": "Reference image 3. Numbered in WIRED order: the first wired image is <Picture 1>, whatever its slot. The info output lists the live tags."}),
                "image_4": ("IMAGE", {"tooltip": "Reference image 4. Numbered in WIRED order: the first wired image is <Picture 1>, whatever its slot. The info output lists the live tags."}),
                "image_5": ("IMAGE", {"tooltip": "Reference image 5. Numbered in WIRED order: the first wired image is <Picture 1>, whatever its slot. The info output lists the live tags."}),
                "image_6": ("IMAGE", {"tooltip": "Reference image 6. Numbered in WIRED order: the first wired image is <Picture 1>, whatever its slot. The info output lists the live tags."}),
                "image_7": ("IMAGE", {"tooltip": "Reference image 7. Numbered in WIRED order: the first wired image is <Picture 1>, whatever its slot. The info output lists the live tags."}),
                "image_8": ("IMAGE", {"tooltip": "Reference image 8. Numbered in WIRED order: the first wired image is <Picture 1>, whatever its slot. The info output lists the live tags."}),
                "image_9": ("IMAGE", {"tooltip": "Reference image 9. Numbered in WIRED order: the first wired image is <Picture 1>, whatever its slot. The info output lists the live tags."}),
                "megapixels_4": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.1, "tooltip": "Per-image budget for image_4. 0 = follow ref_image_size."}),
                "megapixels_5": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.1, "tooltip": "Per-image budget for image_5. 0 = follow ref_image_size."}),
                "megapixels_6": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.1, "tooltip": "Per-image budget for image_6. 0 = follow ref_image_size."}),
                "megapixels_7": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.1, "tooltip": "Per-image budget for image_7. 0 = follow ref_image_size."}),
                "megapixels_8": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.1, "tooltip": "Per-image budget for image_8. 0 = follow ref_image_size."}),
                "megapixels_9": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 16.0, "step": 0.1, "tooltip": "Per-image budget for image_9. 0 = follow ref_image_size."}),
                "audio_vae": ("VAE", {"tooltip": "The MiniMax H3 AUDIO vae. Needed only when an audio or a video soundtrack is wired -- the node refuses by name if one is and this is not."}),
                "video_1": ("VIDEO", {"tooltip": "Reference video 1 (a VIDEO, e.g. the Media Loader's `video`). Its real frame rate is read and resampled to 24 fps -- Core assumes 24. Trimmed to the clip length and to the 17k+5 grid. Official: 2-15 s, all videos together <= 15 s."}),
                "video_audio_1": ("AUDIO", {"tooltip": "Soundtrack OF video_1: its own <Audio j> label right before that <Video>. Used only when video_1 is wired. Wire it explicitly -- the node never lifts a track out of the VIDEO by itself."}),
                "video_2": ("VIDEO", {"tooltip": "Reference video 2 (a VIDEO, e.g. the Media Loader's `video`). Its real frame rate is read and resampled to 24 fps -- Core assumes 24. Trimmed to the clip length and to the 17k+5 grid. Official: 2-15 s, all videos together <= 15 s."}),
                "video_audio_2": ("AUDIO", {"tooltip": "Soundtrack OF video_2: its own <Audio j> label right before that <Video>. Used only when video_2 is wired. Wire it explicitly -- the node never lifts a track out of the VIDEO by itself."}),
                "video_3": ("VIDEO", {"tooltip": "Reference video 3 (a VIDEO, e.g. the Media Loader's `video`). Its real frame rate is read and resampled to 24 fps -- Core assumes 24. Trimmed to the clip length and to the 17k+5 grid. Official: 2-15 s, all videos together <= 15 s."}),
                "video_audio_3": ("AUDIO", {"tooltip": "Soundtrack OF video_3: its own <Audio j> label right before that <Video>. Used only when video_3 is wired. Wire it explicitly -- the node never lifts a track out of the VIDEO by itself."}),
                "audio_1": ("AUDIO", {"tooltip": "Standalone reference audio 1: an <Audio j> label after all video soundtracks. Official: 2-15 s, all audios together <= 15 s."}),
                "audio_2": ("AUDIO", {"tooltip": "Standalone reference audio 2: an <Audio j> label after all video soundtracks. Official: 2-15 s, all audios together <= 15 s."}),
                "audio_3": ("AUDIO", {"tooltip": "Standalone reference audio 3: an <Audio j> label after all video soundtracks. Official: 2-15 s, all audios together <= 15 s."}),
                # v1019 (Cine C2): a Reference Board's media + tags. A SOCKET,
                # so the widget order is untouched.
                "refs": ("PLS_REFS", {"tooltip": "From Polyhedron Reference Board: its images, videos (with their soundtrack) and audios fill this node's FREE slots in board order, and their @tags work in the prompt without a tags line. A slot that is wired stays the wire's."}),
                # v1017 (Cine C1): speaking names for the slots. APPENDED last
                # -- the newest widget, behind megapixels_9 (#577).
                "tags": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "Speaking names for the slots, one per line: 'fox = image_1', 'forest = image_3', 'prev = video_1'. In the prompt write @fox -- the node puts in the LIVE <Picture i> / <Video k> / <Audio j> right before the encode, so rewiring never breaks the prompt. Every slot name is a tag by itself (@image_2, @video_audio_1). An unknown tag stops the run by name."}),
                # v1049 (scene continuity): a SOCKET, appended -- the widget
                # order is untouched (#577).
                "first_frame": ("IMAGE", {"tooltip": "Pinned at pixel frame 0 of the clip, never denoised -- e.g. the Cine Timeline's first_frame when a scene CARRIES the last frame of the scene before. Next to <Picture i> references it is NOT a picture of the prompt (the references keep their numbers); alone it is presented like MiniMax Keyframes' <Picture 1>. Stretched to the canvas like Core's first frame."}),
            },
        }

    # v1017: prompt_text APPENDED -- the prompt as it reached the encoder,
    # tags resolved (for the CLIP Text Encode's pos_external, a Show Text, a
    # log). Output slots are positional in saved workflows, like widgets.
    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("positive", "latent", "info", "prompt_text")
    FUNCTION = "build"
    CATEGORY = "Polyhedron/Conditioning"
    DESCRIPTION = ("<Picture i> reference conditioning for MiniMax H3, with the "
                   "latent taken from a WIRE instead of built under the hood. "
                   "Reads width, height and frame count out of the wired joint "
                   "latent (one source of truth), scales each reference down "
                   "with an optional PER-IMAGE megapixel budget that Core does "
                   "not offer, and reports what it did. Replaces "
                   "MiniMaxH3ReferenceToVideo together with the Resolution "
                   "Selector, Float (Duration) and Math Expression the stock "
                   "template needs around it.")

    def build(self, clip, vae, latent, prompt, ref_image_size,
              megapixels_1, megapixels_2, megapixels_3, **opt):
        samples = latent.get("samples") if isinstance(latent, dict) else latent
        size = M.minimax_size_from_latent(samples)
        if size is None:
            raise ValueError(
                "[PLS] MiniMax Reference: the wired latent is not a joint "
                "MiniMax H3 AV latent. Use Polyhedron Empty Latent with "
                "latent_type 'MiniMax H3 AV' -- this node reads the width, "
                "height and frame count out of it and will not guess them.")
        gen_w, gen_h, frame_count = size

        mps = {1: megapixels_1, 2: megapixels_2, 3: megapixels_3}
        for n in range(4, N_IMAGES + 1):
            mps[n] = opt.get("megapixels_%d" % n, 0.0) or 0.0
        audio_vae = opt.get("audio_vae")

        # v1019 (Cine C2): a board fills the FREE slots, board order, and
        # brings its tags. Must run before the slot lists below are read.
        from .ph_reference_board import apply_refs
        opt = dict(opt)
        board_aliases, board_notes = apply_refs(opt.get("refs"), opt, mps)

        # --- what is wired, in slot order ---------------------------------
        images = [(n, opt.get("image_%d" % n)) for n in range(1, N_IMAGES + 1)]
        images = [(n, img) for n, img in images if img is not None]
        videos = []
        orphan_tracks = []
        for n in range(1, N_VIDEOS + 1):
            v = opt.get("video_%d" % n)
            track = opt.get("video_audio_%d" % n)
            if v is None:
                if track is not None:
                    orphan_tracks.append(n)
                continue
            videos.append((n, v, track))
        audios = [(n, opt.get("audio_%d" % n)) for n in range(1, N_AUDIOS + 1)]
        audios = [(n, a) for n, a in audios if a is not None]
        if (audios or any(t is not None for _, _, t in videos)) and audio_vae is None:
            raise ValueError(
                "[PLS] MiniMax Reference: an audio reference or a video "
                "soundtrack is wired, but no audio_vae. Wire the MiniMax H3 "
                "AUDIO vae into audio_vae -- an audio reference cannot be "
                "encoded without it.")

        tags = plan_tags(len(images), [t is not None for _, _, t in videos],
                         len(audios))
        tag_iter = iter(tags)

        # --- v1017 (Cine C1): @tags -> live labels, BEFORE anything is encoded
        slot_tags = {}
        seq = iter(tags)
        for idx, _img in images:
            slot_tags["image_%d" % idx] = next(seq)[1]
        for idx, _v, track in videos:
            if track is not None:
                slot_tags["video_audio_%d" % idx] = next(seq)[1]
            slot_tags["video_%d" % idx] = next(seq)[1]
        for idx, _a in audios:
            slot_tags["audio_%d" % idx] = next(seq)[1]
        all_slots = (["image_%d" % n for n in range(1, N_IMAGES + 1)]
                     + ["video_%d" % n for n in range(1, N_VIDEOS + 1)]
                     + ["video_audio_%d" % n for n in range(1, N_VIDEOS + 1)]
                     + ["audio_%d" % n for n in range(1, N_AUDIOS + 1)])
        aliases, alias_errors = HP.parse_aliases(opt.get("tags", "") or "")
        for tag, slot in board_aliases.items():
            if tag in aliases and aliases[tag] != slot:
                alias_errors.append("@%s is a board tile AND a tags line (%s) -- "
                                    "keep one" % (tag, aliases[tag]))
            aliases[tag] = slot
        prompt, used, tag_errors = HP.resolve(prompt, slot_tags, all_slots,
                                              aliases)
        if alias_errors or tag_errors:
            raise ValueError(
                "[PLS] MiniMax Reference: the prompt's tags do not resolve -- "
                "nothing was encoded.\n  " + "\n  ".join(alias_errors + tag_errors))
        live = {"Picture": len(images), "Video": len(videos),
                "Audio": sum(1 for k, _ in tags if k in ("audio", "soundtrack"))}
        check_lines = HP.report(used, HP.check(
            prompt, live, frame_count / float(M.MINIMAX_FPS)))

        ref_items = []
        ref_blocks = []
        lines = []
        cells = []
        for idx, img in images:
            mp = mps.get(idx, 0.0)
            src_h, src_w = int(img.shape[1]), int(img.shape[2])
            scale = ref_scale(src_w, src_h, gen_w, gen_h, ref_image_size, mp)
            tw, th = _snap(src_w * scale), _snap(src_h * scale)
            resized = _resize(img[:1], tw, th)
            z = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image",
                               "latent_h": th // M.MINIMAX_SPATIAL_DIV,
                               "latent_w": tw // M.MINIMAX_SPATIAL_DIV,
                               "latent": z})
            rule = ("megapixels_%d=%.2f" % (idx, mp)) if mp and mp > 0 \
                else ("ref_image_size=%s" % ref_image_size)
            n_cells = ((tw // M.MINIMAX_SPATIAL_DIV)
                       * (th // M.MINIMAX_SPATIAL_DIV))
            next(tag_iter)            # images lead the plan: <Picture k>
            cells.append((len(cells) + 1, n_cells))
            lines.append("<Picture %d>  image_%d  %dx%d -> %dx%d  (x%.3f via "
                         "%s)  %d latent cells"
                         % (len(cells), idx, src_w, src_h, tw, th, scale,
                            rule, n_cells))

        note = balance_note(cells)
        if note:
            lines.append(note)

        video_secs = []
        audio_secs = []
        fps_out = float(M.MINIMAX_FPS)
        for idx, video, track in videos:
            frames, src_fps = _video_parts(video)
            n_src = int(frames.shape[0])
            src_sec = n_src / src_fps if src_fps > 0 else n_src / fps_out
            pick = resample_indices(n_src, src_fps, fps_out)
            if len(pick) != n_src:
                frames = frames[pick]
            vh, vw = int(frames.shape[1]), int(frames.shape[2])
            cw, ch = ref_video_canvas(vw, vh)
            frames = _resize(frames, cw, ch)
            n = ref_video_frames(frames.shape[0], frame_count)
            if n is None:
                raise ValueError(
                    "[PLS] MiniMax Reference: video_%d has %d frame(s) at 24 fps "
                    "-- MiniMax H3 needs at least 5 (~0.2 s)"
                    % (idx, int(frames.shape[0])))
            frames = frames[:n]
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if track is not None:
                audio_latent, ref_audio_t = encode_ref_audio(audio_vae, track)
                ref_items.append({"type": "audio"})
                sound_tag = next(tag_iter)[1]
                t_sec = _audio_seconds(track)
                audio_secs.append((sound_tag, t_sec))
            sample_idx = list(range(0, n, M.MINIMAX_FPS // 2))
            ref_items.append({"type": "video", "data": frames[sample_idx],
                              "timestamps": [i / 2.0
                                             for i in range(len(sample_idx))]})
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": int(z.shape[2]),
                               "latent_h": ch // M.MINIMAX_SPATIAL_DIV,
                               "latent_w": cw // M.MINIMAX_SPATIAL_DIV,
                               "ref_audio_t": ref_audio_t, "latent": z,
                               "audio_latent": audio_latent})
            vtag = next(tag_iter)[1]
            video_secs.append((vtag, src_sec))
            fps_note = ("%.2f fps -> 24 (resampled)" % src_fps
                        if len(pick) != n_src else
                        ("24 fps" if src_fps > 0 else "fps unknown, taken as 24"))
            v_cells = int(z.shape[2]) * (ch // M.MINIMAX_SPATIAL_DIV) \
                * (cw // M.MINIMAX_SPATIAL_DIV)
            lines.append("%s  video_%d  %dx%d, %.2f s, %s -> %dx%d, %d frames "
                         "(%.2f s used)  %d latent cells"
                         % (vtag, idx, vw, vh, src_sec, fps_note, cw, ch, n,
                            n / fps_out, v_cells))
            if track is not None:
                lines.append("%s  video_audio_%d  soundtrack of %s, %.2f s, "
                             "%d audio latent steps"
                             % (sound_tag, idx, vtag, t_sec, ref_audio_t))

        for idx, audio in audios:
            audio_latent, ref_audio_t = encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                               "audio_latent": audio_latent})
            atag = next(tag_iter)[1]
            a_sec = _audio_seconds(audio)
            audio_secs.append((atag, a_sec))
            lines.append("%s  audio_%d  %.2f s, %d audio latent steps"
                         % (atag, idx, a_sec, ref_audio_t))

        for n in orphan_tracks:
            lines.append("IGNORED: video_audio_%d is wired but video_%d is "
                         "not -- a soundtrack belongs to its video" % (n, n))
        lines.extend(limit_notes(len(images), video_secs, audio_secs))
        lines.extend(board_notes)
        lines.extend(check_lines)

        # --- v1049: the first frame (scene carry) --------------------------
        first = opt.get("first_frame")
        kf_list, kf_images = [], []
        if first is not None:
            fitted = _resize(first[:1], gen_w, gen_h)
            kf = {"resolved_frame_index": 0, "latent": vae.encode(fitted)}
            if ref_blocks:
                from . import h3_kf_refs as KR
                who = KR.ensure()
                if who == "none":
                    raise ValueError(
                        "[PLS] MiniMax Reference: a first_frame next to references needs Core's "
                        "MiniMaxH3 model class -- not found. Unwire first_frame.")
                KR.mark(kf, who)
                lines.append("first_frame  %dx%d -> %dx%d (stretched), pinned at frame 0 next to the "
                             "references (%s merge)" % (int(first.shape[2]), int(first.shape[1]), gen_w, gen_h,
                                                        "our" if who == "ours" else "the Director's"))
            else:
                kf_images = [fitted]
                lines.append("first_frame  %dx%d -> %dx%d (stretched), pinned at frame 0 = <Picture 1> "
                             "(no references: the fl2va presentation)" % (int(first.shape[2]), int(first.shape[1]),
                                                                        gen_w, gen_h))
            kf_list = [kf]

        # --- THE COUPLING (see the module docstring) -----------------------
        if ref_items:
            tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        else:
            tokens = clip.tokenize(prompt, images=kf_images)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(
                cond, {"minimax_refs": ref_blocks})
        if kf_list:
            cond = node_helpers.conditioning_set_values(
                cond, {"minimax_keyframes": kf_list, "minimax_frame_count": frame_count})

        header = ("[PLS] MiniMax Reference: latent %dx%d, %d frames "
                  "(%.2f s @ %d fps) | %d image(s), %d video(s), %d audio(s)"
                  % (gen_w, gen_h, frame_count,
                     frame_count / float(M.MINIMAX_FPS), M.MINIMAX_FPS,
                     len(images), len(videos),
                     len(audios) + sum(1 for _, _, t in videos if t is not None)))
        print(header)
        for line in lines:
            print("[PLS]   " + line)
        if not ref_items and not kf_list:
            print("[PLS]   nothing wired -- this is a plain text run; any "
                  "<Picture n> / <Video n> / <Audio n> tag in the prompt "
                  "refers to nothing.")

        info = "\n".join([header[6:]] + lines)
        return (cond, latent, info, prompt)


NODE_CLASS_MAPPINGS = {"ULSMiniMaxReference": ULSMiniMaxReference}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ULSMiniMaxReference": "\u2b21 Polyhedron MiniMax Reference"}
