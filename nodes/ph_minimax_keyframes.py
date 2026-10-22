"""Polyhedron MiniMax Keyframes -- fl2va (first / last frame) on OUR rails.

WHAT THIS REPLACES: Core's MiniMaxH3ImageToVideo, which builds its own joint
AV latent from three size widgets and hides the frame grid inside the box.
This node takes the latent as an INPUT (ULSEmptyLatent, latent_type "MiniMax
H3 AV"), recovers width, height and frame count from it -- one source of truth
-- and reports what it anchored. Sister of ULSMiniMaxReference (v872): that one
is ref2va (<Picture i> references), this one is fl2va (keyframes).

WHY H3 NEEDS NO "I2V MODEL" FOR THIS: t2va and fl2va are ONE model. A keyframe
is an extra conditioning row in the packed token stream, pinned at pixel frame
0 (first) or frame_count - 1 (last) and never denoised. Without an image the
very same node is a plain text run. So "adding a picture" is not a switch of
worlds -- and the WAN-style start_image + concat path of ULSEmptyLatent is NOT
this mechanism (H3 does not read concat_latent_image at all).

THE COUPLING, DECLARED. Three Core internals carry fl2va, mirrored from
comfy_extras/nodes_minimax_h3.py (MiniMaxH3ImageToVideo.execute):

    tokens = clip.tokenize(prompt, images=images)      # keyframes enter the
                                                        # Qwen presentation
    cond = node_helpers.conditioning_set_values(cond, {
        "minimax_keyframes": [{"resolved_frame_index": i, "latent": z}, ...],
        "minimax_frame_count": frame_count})

test_v964_minimax_keyframes pins those names here AND in Core's source when
Core can be found, so an upstream rename goes RED instead of silently sampling
a clip with no anchor.

THE LIMIT, DECLARED. On Core 7a131a3a (Frank's pin, 24.08.2026) only index 0
and frame_count - 1 are legal anchors -- comfy/ldm/minimax/model.py:314 raises
"only first/last keyframe anchors are supported". Anchors at other frames
(motion context, guide clips) are a Core-master feature (MiniMaxH3AddGuide,
commit e01fb4c5) and have their own node since v1021: Polyhedron MiniMax
Continue (ph_minimax_continue.py), which refuses on this Core by name.
And: keyframes and <Picture i> references cannot share ONE conditioning on
7a131a3a -- MiniMaxH3.extra_conds writes payload["cond_video_latents"] from
the keyframes and then OVERWRITES it from the refs, while PackedLayout still
lays out both. That is why this node has no `positive` input to append to:
it builds the conditioning itself, like Core's node does.

Where we go past Core, measured on the same source: Core stretches the first
frame to the canvas ("geometry anchor: plain stretch") and cover-crops the
last one -- a 16:9 still on a 9:16 latent is silently squashed. `frame_fit`
keeps Core's rule as the default and offers stretch / cover for both; the
info output names every frame whose aspect is off by more than ASPECT_WARN.
"""

import comfy.utils
import node_helpers

from . import uls_latent_math as M
from . import h3_prompt as HP

FRAME_FIT = ["core (first: stretch, last: cover)", "stretch both", "cover both"]
ASPECT_WARN = 0.02   # 2 % -- the 1344x768 canvas (1.75) against a 1280x720 still
                     # (1.778) is 1.6 % and stays quiet; a 1024x768 still is 24 %
                     # and is named. The line separates the canvas grid from a
                     # genuinely different framing.


def fit_modes(frame_fit):
    """(first_crop, last_crop) for comfy.utils.common_upscale. Pure.

    'disabled' = stretch to the canvas, 'center' = aspect-preserving cover
    crop. The default reproduces Core's MiniMaxH3ImageToVideo exactly."""
    if frame_fit == FRAME_FIT[1]:
        return ("disabled", "disabled")
    if frame_fit == FRAME_FIT[2]:
        return ("center", "center")
    return ("disabled", "center")


def keyframe_index(which, frame_count):
    """Pixel frame an anchor is pinned at. Pure, and the ONLY place the rule lives.

    Core pins the first frame at 0 and the last at frame_count - 1; the layout
    (comfy/ldm/minimax/model.py PackedLayout) accepts exactly those two on
    Core 7a131a3a. A frame_count below 1 is a broken latent, not a corner case."""
    fc = int(frame_count)
    if fc < 1:
        raise ValueError("keyframe_index: frame_count must be >= 1, got %d" % fc)
    if which == "first":
        return 0
    if which == "last":
        return fc - 1
    raise ValueError("keyframe_index: unknown anchor %r" % (which,))


def aspect_off(src_w, src_h, gen_w, gen_h):
    """Relative aspect mismatch between a frame and the canvas. Pure.

    0.0 when the ratios agree; 0.23 for a 4:3 still on a 16:9 canvas. Used
    only for the report -- the pixels are fitted either way."""
    a = float(src_w) / float(src_h) if src_h else 0.0
    b = float(gen_w) / float(gen_h) if gen_h else 0.0
    if a <= 0 or b <= 0:
        return 0.0
    return abs(a - b) / b


def _resize(image, width, height, crop):
    """Core's _resize, mirrored: [B, H, W, C] -> [B, height, width, 3]."""
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


class ULSMiniMaxKeyframes:
    """First / last frame conditioning for MiniMax H3, latent from a wire."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP", {"tooltip":
                                  "The MiniMax H3 text encoder. The keyframes "
                                  "are woven into the TOKENS here (fl2va "
                                  "presentation) -- this is not an ordinary "
                                  "text encode and cannot be moved upstream."}),
                "vae": ("VAE", {"tooltip":
                                "The VIDEO vae. Each keyframe is encoded with "
                                "it and pinned into the conditioning, never "
                                "denoised."}),
                "latent": ("LATENT", {"tooltip":
                                      "A joint MiniMax H3 AV latent -- from "
                                      "Polyhedron Empty Latent with "
                                      "latent_type 'MiniMax H3 AV'. Width, "
                                      "height and frame count are READ from "
                                      "it; the last frame is pinned at "
                                      "frame_count - 1 of exactly this "
                                      "latent."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "default": "",
                                      "tooltip":
                                      "The H3 prompt. With a first frame the "
                                      "official guide expects the alignment "
                                      "line 'at 0.00 seconds ... <Picture 1> "
                                      "is fully referenced'; the info output "
                                      "tells you which <Picture n> is live. "
                                      "No image wired = a plain text run."}),
                "frame_fit": (FRAME_FIT, {"default": FRAME_FIT[0],
                              "tooltip":
                              "How a keyframe meets the canvas. 'core' is "
                              "Core's rule: first frame stretched (geometry "
                              "anchor), last frame cover-cropped. A still "
                              "whose aspect differs from the latent is "
                              "squashed by 'stretch' and trimmed by 'cover' "
                              "-- the info output names the offenders."}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip":
                                          "Pinned at pixel frame 0. Becomes "
                                          "<Picture 1> in the prompt."}),
                "last_frame": ("IMAGE", {"tooltip":
                                         "Pinned at the latent's last pixel "
                                         "frame. Becomes <Picture 2> when a "
                                         "first frame is wired, else "
                                         "<Picture 1>."}),
                # v1017 (Cine C1): the first OPTIONAL widget of this node --
                # it follows frame_fit in the widget order (#577 append law).
                "tags": ("STRING", {"multiline": True, "default": "",
                                    "tooltip": "Speaking names for the two frames, one per line: 'start = first_frame', 'goal = last_frame'. In the prompt write @start -- the node puts in the LIVE <Picture n> (the last frame is <Picture 2> only when a first frame is wired). @first_frame / @last_frame work without an alias. An unknown or unwired tag stops the run by name."}),
            },
        }

    # v965: prompt_text APPENDED (output slots are positional in saved
    # workflows, like widgets) -- the node can be THE prompt source and hand
    # its text on to ULSCLIPTextEncode.pos_external (mode "replace") for NAG
    # and the token counter, instead of the other way round.
    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("positive", "latent", "info", "prompt_text")
    FUNCTION = "build"
    CATEGORY = "Polyhedron/Conditioning"
    DESCRIPTION = ("First / last frame (fl2va) conditioning for MiniMax H3, "
                   "with the latent taken from a WIRE instead of built under "
                   "the hood. Reads width, height and frame count out of the "
                   "wired joint latent, pins the first frame at 0 and the last "
                   "at frame_count - 1 exactly like Core's "
                   "MiniMaxH3ImageToVideo, lets you choose stretch or cover "
                   "for both frames, and reports every frame whose aspect "
                   "does not match the canvas. Without an image it is a plain "
                   "text run -- t2va and fl2va are one model.")

    def build(self, clip, vae, latent, prompt, frame_fit,
              first_frame=None, last_frame=None, tags=""):
        samples = latent.get("samples") if isinstance(latent, dict) else latent
        size = M.minimax_size_from_latent(samples)
        if size is None:
            raise ValueError(
                "[PLS] MiniMax Keyframes: the wired latent is not a joint "
                "MiniMax H3 AV latent. Use Polyhedron Empty Latent with "
                "latent_type 'MiniMax H3 AV' -- this node reads the width, "
                "height and frame count out of it and will not guess them.")
        gen_w, gen_h, frame_count = size
        first_crop, last_crop = fit_modes(frame_fit)

        images = []      # the Qwen presentation, in <Picture n> order
        keyframes = []   # the DiT payload, same order
        lines = []
        for which, img, crop in (("first", first_frame, first_crop),
                                 ("last", last_frame, last_crop)):
            if img is None:
                continue
            src_h, src_w = int(img.shape[1]), int(img.shape[2])
            idx = keyframe_index(which, frame_count)
            fitted = _resize(img[:1], gen_w, gen_h, crop)
            images.append(fitted)
            keyframes.append({"resolved_frame_index": idx, "latent": None,
                              "image": fitted})
            off = aspect_off(src_w, src_h, gen_w, gen_h)
            mode = "stretch" if crop == "disabled" else "cover"
            line = ("<Picture %d>  %s frame  %dx%d -> %dx%d (%s)  pinned at "
                    "frame %d" % (len(images), which, src_w, src_h, gen_w,
                                  gen_h, mode, idx))
            if off > ASPECT_WARN:
                line += ("  ASPECT OFF by %.0f%% -- %s"
                         % (off * 100.0,
                            "the still is squashed to the canvas" if mode == "stretch"
                            else "the edges are trimmed to the canvas"))
            lines.append(line)

        # --- v1017 (Cine C1): @tags -> live labels, before the encode -------
        slot_tags, n = {}, 0
        for which, img in (("first_frame", first_frame), ("last_frame", last_frame)):
            if img is not None:
                n += 1
                slot_tags[which] = "<Picture %d>" % n
        aliases, alias_errors = HP.parse_aliases(tags or "")
        # v1084: the // rubrics (the timeline's head) never reach the model
        prompt, n_rub = HP.strip_rubrics(prompt)
        if n_rub:
            lines.append("%d // rubric line(s) stripped before the encode" % n_rub)
        prompt, used, tag_errors = HP.resolve(
            prompt, slot_tags, ["first_frame", "last_frame"], aliases)
        if alias_errors or tag_errors:
            raise ValueError(
                "[PLS] MiniMax Keyframes: the prompt's tags do not resolve -- "
                "nothing was encoded.\n  " + "\n  ".join(alias_errors + tag_errors))
        lines.extend(HP.report(used, HP.check(
            prompt, {"Picture": len(images), "Video": 0, "Audio": 0},
            frame_count / float(M.MINIMAX_FPS))))

        # --- THE COUPLING (see the module docstring) -----------------------
        tokens = clip.tokenize(prompt, images=images)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if keyframes:
            for kf in keyframes:
                kf["latent"] = vae.encode(kf.pop("image"))
            cond = node_helpers.conditioning_set_values(cond, {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": frame_count,
            })

        header = ("[PLS] MiniMax Keyframes: latent %dx%d, %d frames "
                  "(%.2f s @ %d fps) | %d keyframe(s)"
                  % (gen_w, gen_h, frame_count,
                     frame_count / float(M.MINIMAX_FPS), M.MINIMAX_FPS,
                     len(keyframes)))
        print(header)
        for line in lines:
            print("[PLS]   " + line)
        if not keyframes:
            print("[PLS]   no frame wired -- this is a plain text run (t2va); "
                  "any <Picture n> tag in the prompt refers to nothing.")

        info = "\n".join([header[6:]] + lines)
        # v965: the frontend shows the prompt that actually ran (it may have
        # arrived over a wire, in which case the prompt box is hidden) plus
        # the report, in a read-only field inside the node -- ph_minimax_keyframes.js.
        ui = [{"prompt": prompt, "info": info, "keyframes": len(keyframes)}]
        return {"ui": {"pls_kf": ui},
                "result": (cond, latent, info, prompt)}


NODE_CLASS_MAPPINGS = {"ULSMiniMaxKeyframes": ULSMiniMaxKeyframes}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ULSMiniMaxKeyframes": "⬡ Polyhedron MiniMax Keyframes"}
