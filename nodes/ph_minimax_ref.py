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

_SIZE_MODES = ["match", "max"]
REF_IMAGE_SHORT_EDGE = 2048   # Core's constant, mirrored


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
                                      "Refer to the wired images as <Picture "
                                      "1>, <Picture 2>, <Picture 3> -- the "
                                      "number follows the INPUT slot, not the "
                                      "order you wired them in. The info "
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
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "Becomes <Picture 1>."}),
                "image_2": ("IMAGE", {"tooltip": "Becomes <Picture 2>."}),
                "image_3": ("IMAGE", {"tooltip": "Becomes <Picture 3>."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "latent", "info")
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
              megapixels_1, megapixels_2, megapixels_3,
              image_1=None, image_2=None, image_3=None):
        samples = latent.get("samples") if isinstance(latent, dict) else latent
        size = M.minimax_size_from_latent(samples)
        if size is None:
            raise ValueError(
                "[PLS] MiniMax Reference: the wired latent is not a joint "
                "MiniMax H3 AV latent. Use Polyhedron Empty Latent with "
                "latent_type 'MiniMax H3 AV' -- this node reads the width, "
                "height and frame count out of it and will not guess them.")
        gen_w, gen_h, frame_count = size

        ref_items = []
        ref_blocks = []
        lines = []
        cells = []
        for idx, (img, mp) in enumerate(
                ((image_1, megapixels_1), (image_2, megapixels_2),
                 (image_3, megapixels_3)), start=1):
            if img is None:
                continue
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
            cells.append((len(ref_items), n_cells))
            lines.append("<Picture %d>  %dx%d -> %dx%d  (x%.3f via %s)  "
                         "%d latent cells"
                         % (len(ref_items), src_w, src_h, tw, th, scale, rule,
                            n_cells))
            if len(ref_items) != idx:
                lines.append("    NOTE: this is <Picture %d>, not <Picture %d> "
                             "-- an empty slot before it shifts the numbering"
                             % (len(ref_items), idx))

        note = balance_note(cells)
        if note:
            lines.append(note)

        # --- THE COUPLING (see the module docstring) -----------------------
        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = node_helpers.conditioning_set_values(
                cond, {"minimax_refs": ref_blocks})

        header = ("[PLS] MiniMax Reference: latent %dx%d, %d frames "
                  "(%.2f s @ %d fps) | %d reference image(s)"
                  % (gen_w, gen_h, frame_count,
                     frame_count / float(M.MINIMAX_FPS), M.MINIMAX_FPS,
                     len(ref_items)))
        print(header)
        for line in lines:
            print("[PLS]   " + line)
        if not ref_items:
            print("[PLS]   no images wired -- this is a plain text run; any "
                  "<Picture n> tag in the prompt refers to nothing.")

        info = "\n".join([header[6:]] + lines)
        return (cond, latent, info)


NODE_CLASS_MAPPINGS = {"ULSMiniMaxReference": ULSMiniMaxReference}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ULSMiniMaxReference": "\u2b21 Polyhedron MiniMax Reference"}
