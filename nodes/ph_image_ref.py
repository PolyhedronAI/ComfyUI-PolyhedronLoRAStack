# ph_image_ref.py -- Polyhedron Image Reference (ULSImageReference), v1078
#
# ONE reference-image conditioning node for every model family that takes
# images next to its prompt. The family is MEASURED on the loaded text
# encoder (its tokenizer class), never guessed from a filename, and each
# family gets the recipe its own Core node uses -- read from comfy_extras
# and comfy/text_encoders, not from memory:
#
#   qwen_image_2.1    comfy_extras/nodes_qwen.py TextEncodeQwenImage21 (core
#                     master 6bfaacc6+): every reference is resized to about
#                     ref_size x ref_size at multiples of 32 (aspect kept),
#                     the vision tower sees it with alpha over WHITE, the VAE
#                     keeps every channel it has (RGBA stays RGBA), the latent
#                     carries 64 channels at /16 on the first reference's size.
#                     Without a VAE the vision tokens are KEPT (keep_vision);
#                     with one they are spliced as latents by the DiT.
#   qwen_image_edit   comfy_extras/nodes_qwen.py TextEncodeQwenImageEditPlus
#                     (Qwen-Image-Edit 2509 / 2511, on the pinned core too):
#                     the vision tower sees ~384x384 (area), the VAE a ~1 MP
#                     copy at multiples of 8; "Picture i: <vision>" prefixes
#                     and the edit llama template. Latent 16 channels at /8.
#   minimax_h3        hands the images to Polyhedron MiniMax Reference (the
#                     <Picture i> ref2va recipe that already lives in
#                     ph_minimax_ref.py) -- same images, same node under the
#                     hood, so there is ONE H3 recipe in the tree.
#   reference_latents every other family: plain text tokens plus Core's
#                     ReferenceLatent semantics (reference_latents appended;
#                     the Flux2 / Kontext pattern of ph_reference.py). The
#                     VAE decides channels and grid.
#
# v1318 inpaint_mask: the latent output becomes image_1 itself (VAEEncode at the
# canvas size) with the mask as noise_mask (SetLatentNoiseMask) -- the sampler
# repaints only the masked part (Splat Key View's holes, any inpainting).
#
# The pure parts (sizes, family resolution, prompt prefix, channel rules) are
# torch-free and guarded; torch and comfy enter only in the run path.

import math

FAMILY_AUTO = "auto"
FAMILY_QWEN21 = "qwen_image_2.1"
FAMILY_QWEN_EDIT = "qwen_image_edit"
FAMILY_MINIMAX = "minimax_h3"
FAMILY_REFLAT = "reference_latents"
# v1325: MiniMax H3 IMAGE / p-h3-image. Same tokenizer class as the H3 video
# model (MiniMaxH3Tokenizer), so the TOKENIZER cannot tell them apart -- the
# wired VAE's class can (comfy/ldm/minimax/vae.py MiniMaxH3ImageVAE /
# PH3ImageVAE). Appended: combo values already saved keep their meaning.
FAMILY_MINIMAX_IMAGE = "minimax_h3_image"
FAMILIES = [FAMILY_AUTO, FAMILY_QWEN21, FAMILY_QWEN_EDIT, FAMILY_MINIMAX,
            FAMILY_REFLAT, FAMILY_MINIMAX_IMAGE]
MINIMAX_IMAGE_VAES = ("MiniMaxH3ImageVAE", "PH3ImageVAE")   # == h3_image.IMAGE_VAES (guarded)

N_IMAGES = 9
IMAGE_KEYS = ["image_%d" % i for i in range(1, N_IMAGES + 1)]

# tokenizer class name (anywhere in the MRO) -> family. Measured on core
# 7a131a3a (pin) and 5c460d81 (master); comfy/text_encoders/*.py.
_TOKENIZER_FAMILY = (
    ("QwenImage21Tokenizer", FAMILY_QWEN21),   # exact names; order is free (mutation-proved)
    ("QwenImageTokenizer", FAMILY_QWEN_EDIT),
    ("MiniMaxH3Tokenizer", FAMILY_MINIMAX),
)

# per-family geometry, from the Core nodes named above
QWEN21_GRID = 32          # vision + VAE share one size, multiples of 32
QWEN21_LATENT = (64, 16)  # channels, spatial ratio
EDIT_VL_SIDE = 384        # vision tower input area side
EDIT_VAE_SIDE = 1024      # reference latent area side
EDIT_VAE_GRID = 8
EDIT_LATENT = (16, 8)
EDIT_TEMPLATE = ("<|im_start|>system\nDescribe the key features of the input "
                 "image (color, shape, size, texture, objects, background), "
                 "then explain how the user's text instruction should alter "
                 "or modify the image. Generate a new image that meets the "
                 "user's requirements while maintaining consistency with the "
                 "original input where appropriate.<|im_end|>\n"
                 "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")


# --------------------------------------------------------------------------
# pure
# --------------------------------------------------------------------------

def fit_area(width, height, side, grid):
    """Resize (width, height) to about side x side pixels, aspect kept, both
    sides rounded to multiples of `grid` (grid <= 1: plain rounding). side
    <= 0 keeps the size, only snapped to the grid. Never below one grid."""
    w, h = int(width), int(height)
    if w <= 0 or h <= 0:
        raise ValueError("[PLS] Image Reference: empty image (%dx%d)" % (w, h))
    g = max(1, int(grid))
    if int(side) > 0:
        ratio = w / float(h)
        tw = math.sqrt(float(side) * float(side) * ratio)
        th = math.sqrt(float(side) * float(side) / ratio)
    else:
        tw, th = float(w), float(h)
    nw = int(round(tw / g)) * g
    nh = int(round(th / g)) * g
    return max(g, nw), max(g, nh)


def family_from_names(class_names):
    """Family for a tokenizer's MRO class names, or None when no recipe
    claims it (the caller then falls back to reference_latents and SAYS so)."""
    names = [str(n) for n in (class_names or [])]
    for needle, fam in _TOKENIZER_FAMILY:
        if any(needle == n for n in names):
            return fam
    return None


def resolve_family(choice, class_names, vae_class=""):
    """(family, how). An explicit choice wins; 'auto' measures the tokenizer
    and falls back to reference_latents when nothing claims it. v1325: an H3
    tokenizer with an H3 IMAGE VAE wired is the image model."""
    if choice and choice != FAMILY_AUTO:
        if choice not in FAMILIES:
            raise ValueError("[PLS] Image Reference: unknown family %r "
                             "(one of %s)" % (choice, ", ".join(FAMILIES[1:])))
        return choice, "set"
    fam = family_from_names(class_names)
    if fam == FAMILY_MINIMAX and str(vae_class or "") in MINIMAX_IMAGE_VAES:
        return FAMILY_MINIMAX_IMAGE, "auto from MiniMaxH3Tokenizer + %s" % vae_class
    if fam is not None:
        return fam, "auto from %s" % next(
            n for n in class_names if any(n == k for k, _ in _TOKENIZER_FAMILY))
    return FAMILY_REFLAT, "auto: no image-aware tokenizer recipe for %s -> reference latents only" % (
        class_names[0] if class_names else "unknown")


def edit_prefix(n_images):
    """The 'Picture i:' run EditPlus puts before the user text."""
    return "".join("Picture %d: <|vision_start|><|image_pad|><|vision_end|>" % (i + 1)
                   for i in range(int(n_images)))


def wired_images(opt):
    """(key, image) pairs in slot order, holes skipped -- the order the
    references are numbered in."""
    out = []
    for k in IMAGE_KEYS:
        v = opt.get(k)
        if v is not None:
            out.append((k, v))
    return out


def vae_channels(vae, default=3):
    """How many pixel channels the VAE takes. Core crops extra channels and
    pads missing ones itself (vae_encode_crop_pixels, pin and master); we
    hand it what it can use so a 4-channel reference keeps its alpha for a
    4-channel VAE and costs nothing for a 3-channel one."""
    n = getattr(vae, "output_channels", None)
    try:
        n = int(n)
    except Exception:
        return int(default)
    return n if n > 0 else int(default)


def plan(family, sizes, side, has_vae):
    """The geometry one run will use, as plain numbers (guardable, loggable).
    sizes: [(w, h), ...] of the wired references in order."""
    refs = []
    if family == FAMILY_QWEN21:
        for w, h in sizes:
            s = fit_area(w, h, side, QWEN21_GRID)
            refs.append({"vl": s, "vae": s})
        first = refs[0]["vae"] if refs else (max(QWEN21_GRID, int(side or 1024)),) * 2
        ch, div = QWEN21_LATENT
        # v1244: exactly Core's rule -- vision tokens are kept whenever no
        # reference latent is spliced (no VAE, or no image wired)
        return {"refs": refs, "latent": (ch, first[1] // div, first[0] // div),
                "keep_vision": (not has_vae) or not refs}
    if family == FAMILY_QWEN_EDIT:
        for w, h in sizes:
            refs.append({"vl": fit_area(w, h, EDIT_VL_SIDE, 1),
                         "vae": fit_area(w, h, EDIT_VAE_SIDE, EDIT_VAE_GRID)})
        first = refs[0]["vae"] if refs else (EDIT_VAE_SIDE, EDIT_VAE_SIDE)
        ch, div = EDIT_LATENT
        return {"refs": refs, "latent": (ch, first[1] // div, first[0] // div),
                "keep_vision": False}
    if family == FAMILY_REFLAT:
        for w, h in sizes:
            s = fit_area(w, h, side, 16)
            refs.append({"vl": None, "vae": s})
        return {"refs": refs, "latent": None, "keep_vision": False}
    raise ValueError("[PLS] Image Reference: no plan for family %r" % (family,))


def inpaint_noise_mask(mask, width, height, grow_px, feather_px):
    """v1318: the inpaint mask as the latent's noise_mask (SetLatentNoiseMask's form, (1, 1, H, W) at PIXEL
    size -- the sampler scales it to the latent): scaled to the canvas, binarised at 0.5, grown by grow_px, its
    edge softened over about feather_px (three box passes), and the original hole kept at exactly 1 -- the
    feather only reaches OUTWARD, never into what must be painted. -> (mask, share of the canvas >= 0.5)."""
    import torch
    import torch.nn.functional as F
    m = mask.float()
    if m.dim() == 3:
        m = m[:1]
    elif m.dim() == 2:
        m = m[None]
    m = m[None]                                                    # (1, 1, H, W)
    if tuple(m.shape[-2:]) != (int(height), int(width)):
        m = F.interpolate(m, size=(int(height), int(width)), mode="bilinear", align_corners=False)
    hole = (m >= 0.5).float()
    out = hole
    g = int(grow_px)
    if g > 0:
        k = 2 * g + 1
        out = F.max_pool2d(out, kernel_size=k, stride=1, padding=g)
    r = int(math.ceil(int(feather_px) / 3.0)) if int(feather_px) > 0 else 0
    if r > 0:
        k = 2 * r + 1
        for _ in range(3):
            out = F.avg_pool2d(F.pad(out, (r, r, r, r), mode="replicate"), kernel_size=k, stride=1)
    out = torch.maximum(out.clamp(0, 1), hole)
    return out, float((out >= 0.5).float().mean())


# --------------------------------------------------------------------------
# the node
# --------------------------------------------------------------------------

class ULSImageReference:
    """One reference-image conditioning node, family measured on the encoder."""

    @classmethod
    def INPUT_TYPES(cls):
        # WRITTEN OUT LITERALLY (no loop, no **merge): the #577 widget-order
        # gate reads this dict statically; a comprehension turns the node into
        # a bare DYNAMIC line and the order stops being guarded (v1020 lesson).
        return {
            "required": {
                "clip": ("CLIP", {"tooltip":
                                  "The loaded text encoder. Its tokenizer class "
                                  "tells the node which family's recipe to use -- "
                                  "measured, not guessed from a filename."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "default": "",
                                      "tooltip":
                                      "The positive text. Families that number "
                                      "their references get the numbering prefix "
                                      "from the node; write the instruction only."}),
                "negative_prompt": ("STRING", {"multiline": True,
                                               "dynamicPrompts": True,
                                               "default": "",
                                               "tooltip":
                                               "The negative text, encoded with the "
                                               "SAME references (the official "
                                               "templates do), so positive and "
                                               "negative see one picture."}),
                "family": (FAMILIES, {"default": FAMILY_AUTO,
                                      "tooltip":
                                      "auto: read from the encoder's tokenizer; a "
                                      "family with no image recipe falls back to "
                                      "reference latents and the info says so. "
                                      "Set it only to force a recipe."}),
                "ref_size": ("INT", {"default": 1024, "min": 0, "max": 4096,
                                     "step": 32,
                                     "tooltip":
                                     "Side of the pixel area each reference is "
                                     "fitted into (aspect kept, snapped to the "
                                     "family's grid). 0 = keep the image's own "
                                     "size, snapped. Families with a fixed rule "
                                     "ignore it and the info says which size was "
                                     "used."}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip":
                                "Encodes the references to latents where the "
                                "family splices them into the sequence. Without "
                                "it the encoder's vision tokens alone carry the "
                                "picture (where the family allows); the info "
                                "names the path taken."}),
                "latent": ("LATENT", {"tooltip":
                                      "Only for families whose canvas comes "
                                      "from a wire: the joint video+audio latent "
                                      "(minimax_h3), or for minimax_h3_image the "
                                      "Empty Latent of that type, whose size "
                                      "becomes the canvas (unwired: image_1 "
                                      "decides). Other image families build "
                                      "their latent from the first reference."}),
                "image_1": ("IMAGE", {"tooltip": "Reference image 1. Numbered in WIRED order; the info output lists what each one became (vision size, latent size). A fourth channel is kept for a VAE that takes four."}),
                "image_2": ("IMAGE", {"tooltip": "Reference image 2. Numbered in WIRED order; the info output lists what each one became (vision size, latent size). A fourth channel is kept for a VAE that takes four."}),
                "image_3": ("IMAGE", {"tooltip": "Reference image 3. Numbered in WIRED order; the info output lists what each one became (vision size, latent size). A fourth channel is kept for a VAE that takes four."}),
                "image_4": ("IMAGE", {"tooltip": "Reference image 4. Numbered in WIRED order; the info output lists what each one became (vision size, latent size). A fourth channel is kept for a VAE that takes four."}),
                "image_5": ("IMAGE", {"tooltip": "Reference image 5. Numbered in WIRED order; the info output lists what each one became (vision size, latent size). A fourth channel is kept for a VAE that takes four."}),
                "image_6": ("IMAGE", {"tooltip": "Reference image 6. Numbered in WIRED order; the info output lists what each one became (vision size, latent size). A fourth channel is kept for a VAE that takes four."}),
                "image_7": ("IMAGE", {"tooltip": "Reference image 7. Numbered in WIRED order; the info output lists what each one became (vision size, latent size). A fourth channel is kept for a VAE that takes four."}),
                "image_8": ("IMAGE", {"tooltip": "Reference image 8. Numbered in WIRED order; the info output lists what each one became (vision size, latent size). A fourth channel is kept for a VAE that takes four."}),
                "image_9": ("IMAGE", {"tooltip": "Reference image 9. Numbered in WIRED order; the info output lists what each one became (vision size, latent size). A fourth channel is kept for a VAE that takes four."}),
                "inpaint_mask": ("MASK", {"tooltip":
                                          "Inpainting (white = paint here): the latent output is image_1 "
                                          "itself, VAE-encoded at the canvas size, with this mask as its noise "
                                          "mask -- the sampler repaints only the white part, the rest stays "
                                          "image_1. Needs the VAE and image_1."}),
                "inpaint_grow": ("INT", {"default": 16, "min": 0, "max": 256, "step": 1,
                                         "tooltip":
                                         "inpaint_mask: grow the white part by this many px (on the "
                                         "canvas) so the new paint overlaps the edge."}),
                "inpaint_feather": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1,
                                            "tooltip":
                                            "inpaint_mask: soften the edge over about this many px, "
                                            "outward only -- the masked part itself is always fully "
                                            "repainted."}),
                # v1338 (Frank 17.11.): appended LAST -- saved graphs keep every slot.
                "blank_picture": (["off", "black", "mid grey", "white"], {"default": "off",
                                  "tooltip":
                                  "Text to image only (no reference wired): a blank picture of this tone "
                                  "stands in front of the prompt for the window below -- it softens the hard, "
                                  "over-rendered look of a distilled text-to-image run. For families that read "
                                  "pictures in their text encoder; the info says when a family does not take "
                                  "it. off = plain text, as before."}),
                "blank_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                                "tooltip":
                                "blank_picture: start of the window, as a share of the full run (0 = the "
                                "first step). The info lists the steps that see the picture."}),
                "blank_end": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.001,
                              "tooltip":
                              "blank_picture: end of the window. Step n of a run runs at (n-1)/steps: on 6 "
                              "steps 0.25 covers steps 1-2, 0.4 steps 1-3; on 8 steps 0.25 covers steps 1-3. "
                              "A step exactly on the edge sees both conditionings averaged; the info names the "
                              "steps and says so."}),
                # v1341 (Frank 17.11.): a wired text gets its OWN socket row. Wiring the prompt FIELD's socket
                # put that dot on top of another row (clip in Frank's frontend, the negative field in 1.59.1) --
                # a hidden field has no row of its own any more. Appended last; a socket, not a widget.
                "prompt_in": ("STRING", {"forceInput": True,
                              "tooltip": "Wired text (a CLIP Text Encode's positive_text, a prompt builder ...) "
                                         "replaces the prompt field. The field keeps its text for when the wire "
                                         "comes off."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "info", "prompt_text")
    OUTPUT_TOOLTIPS = ("Positive conditioning with the references woven in.",
                       "Negative conditioning, same references.",
                       "Empty latent sized by the family's rule (first reference), "
                       "or the wired one for wire-canvas families; with inpaint_mask "
                       "image_1 encoded, the mask as its noise mask.",
                       "Readout: family (how resolved) | references | sizes | latent.",
                       "The prompt as it reached the encoder, prefix included.")
    FUNCTION = "build"
    CATEGORY = "Polyhedron/Conditioning"
    DESCRIPTION = ("Reference-image conditioning for every model family that "
                   "takes pictures next to its prompt, in ONE node. The family "
                   "is read from the loaded encoder and each gets the recipe "
                   "its own Core node uses (vision tokens, reference latents, "
                   "numbering prefix, grid); families with a wired canvas are "
                   "handed to the pack's own node for them. What a family "
                   "cannot take is said in the info, never silently dropped.")

    # ---- family routes ---------------------------------------------------

    @staticmethod
    def _tokenizer_names(clip):
        tok = getattr(clip, "tokenizer", None)
        names = []
        for c in type(tok).__mro__ if tok is not None else ():
            names.append(c.__name__)
        return names

    def build(self, clip, prompt, negative_prompt, family=FAMILY_AUTO,
              ref_size=1024, vae=None, latent=None, inpaint_mask=None,
              inpaint_grow=16, inpaint_feather=8, blank_picture="off",
              blank_start=0.0, blank_end=0.25, prompt_in=None, **opt):
        if prompt_in is not None:            # v1341: the wired text wins over the field
            prompt = str(prompt_in)
        vae_class = type(getattr(vae, "first_stage_model", None)).__name__ if vae is not None else ""
        fam, how = resolve_family(family, self._tokenizer_names(clip), vae_class)
        refs = wired_images(opt)
        if inpaint_mask is not None:
            if fam == FAMILY_MINIMAX:
                raise ValueError(
                    "[PLS] Image Reference: inpaint_mask is for image families -- "
                    "minimax_h3 takes its canvas from the joint latent.")
            if vae is None or not refs or refs[0][0] != "image_1":
                raise ValueError(
                    "[PLS] Image Reference: inpaint_mask needs the VAE and image_1 "
                    "(the picture that is repainted inside the mask).")
        blank = str(blank_picture or "off")
        if fam == FAMILY_MINIMAX_IMAGE:
            return self._minimax_image(clip, vae, latent, prompt, refs, how, vae_class,
                                       blank, blank_start, blank_end, inpaint_mask,
                                       int(inpaint_grow), int(inpaint_feather))
        if fam == FAMILY_MINIMAX:
            out = self._minimax(clip, vae, latent, prompt, negative_prompt,
                                refs, how)
        else:
            out = self._image_family(fam, how, clip, vae, prompt, negative_prompt,
                                     int(ref_size), refs, inpaint_mask,
                                     int(inpaint_grow), int(inpaint_feather))
        if blank != "off":
            note = " | blank_picture %s NOT used: family %s does not take it" % (blank, fam)
            print("[PLS] Image Reference:" + note)
            out = out[:3] + (out[3] + note,) + out[4:]
        return out

    def _minimax(self, clip, vae, latent, prompt, negative_prompt, refs, how):
        from .ph_minimax_ref import ULSMiniMaxReference
        if vae is None or latent is None:
            raise ValueError(
                "[PLS] Image Reference: family minimax_h3 needs the video VAE "
                "and the joint AV latent wired (its canvas comes from the "
                "latent, not from a reference)."
            )
        kw = {k: v for k, v in refs}
        pos, lat, info, text = ULSMiniMaxReference().build(
            clip, vae, latent, prompt, "match", 0.0, 0.0, 0.0, **kw)
        # the H3 negative: plain text, no references (Core's own template)
        neg = clip.encode_from_tokens_scheduled(clip.tokenize(negative_prompt))
        info = "family minimax_h3 (%s) | via MiniMax Reference | %s" % (how, info)
        return (pos, neg, lat, info, text)

    def _minimax_image(self, clip, vae, latent, prompt, refs, how, vae_class,
                       blank="off", blank_start=0.0, blank_end=0.25, inpaint_mask=None,
                       inpaint_grow=16, inpaint_feather=8):
        """v1325: MiniMax H3 Image / p-h3-image -- Core's own recipe (h3_image.encode), one H3 Image recipe
        in the tree. The canvas: a wired latent (Empty Latent, type MiniMax H3 Image) sets the size;
        without one Core decides (image_1 after its reference scaling, without pictures the model's own
        side). ref_size does not apply. The negative is the zeroed positive: the model is cfg-distilled."""
        from . import h3_image as HI
        var = HI.variant_for(vae_class) or HI.VARIANT_H3
        w = h = 0
        if latent is not None:
            w, h = HI.latent_px(latent["samples"])
        # v1339: one frame per pin (Core takes [:1] as well), alpha laid over white -- Core would cut it
        alpha = [k for k, v in refs if int(v.shape[-1]) > 3]
        batch = [k for k, v in refs if int(v.shape[0]) > 1]
        refs = [(k, HI.flatten_alpha(v[:1])) for k, v in refs]
        pos, lat = HI.encode(clip, prompt, w, h, vae, {k: v for k, v in refs}, var)
        prime = self._blank_prime(HI, clip, pos, refs, blank, blank_start, blank_end, var, prompt)
        pos = prime[0]
        neg = HI.zero_out(pos)
        lw, lh = HI.latent_px(lat["samples"])
        info = ("family minimax_h3_image (%s) | %s | %d reference(s) | canvas %d x %d %s | negative = "
                "zeroed positive (cfg 1)" % (how, var, len(refs), lw, lh,
                                             "from the wired latent" if latent is not None
                                             else "from image_1 (Core's rule; ref_size not used)"))
        info += prime[1]
        if alpha:
            info += " | alpha laid over white: %s" % ", ".join(alpha)
        if batch:
            info += " | first frame only: %s" % ", ".join(batch)
        if inpaint_mask is not None:
            # v1339: the edit conditioning stays (image_1 is <Picture 1>); the latent becomes image_1 itself at the
            # canvas size with the mask as its noise mask -- outside the mask the sampler keeps image_1
            import comfy.utils
            px = refs[0][1]
            if (int(px.shape[2]), int(px.shape[1])) != (lw, lh):
                px = comfy.utils.common_upscale(px.movedim(-1, 1), lw, lh, "lanczos", "disabled").movedim(1, -1)
            enc = vae.encode(px[..., :3])
            nm, share = inpaint_noise_mask(inpaint_mask, lw, lh, inpaint_grow, inpaint_feather)
            lat = {"samples": enc, "noise_mask": nm}
            info += (" | inpaint: latent = image_1 encoded at %d x %d, noise mask grown %d px, feathered %d px "
                     "(outward), %.1f %% of the canvas repainted" % (lw, lh, inpaint_grow, inpaint_feather,
                                                                   100.0 * share))
        print("[PLS] Image Reference: " + info)
        return (pos, neg, lat, info, prompt)

    @staticmethod
    def _blank_prime(HI, clip, pos, refs, blank, start, end, variant=None, prompt=""):
        """v1338: (positive, info tail). The blank picture only stands in for MISSING pictures: with a
        reference wired it is not used, and the info says so. v1343: p-h3-image gets its own label and ONE
        joint encode (its text connector puts a sink on the first row of every encode); 8 steps."""
        if blank == "off":
            return pos, ""
        if refs:
            return pos, " | blank_picture %s NOT used: %d reference(s) wired" % (blank, len(refs))
        win = HI.blank_window(start, end)
        if win is None:
            return pos, " | blank_picture %s NOT used: empty window %.3f-%.3f" % (blank, start, end)
        n = HI.steps_for(variant)
        pic, shared = HI.steps_covered(win[0], win[1], n)
        if variant == HI.VARIANT_PH3:
            pos = HI.prime_joint(pos, HI.joint_block(clip, prompt, blank), *win)
            how = "one joint encode, label <Picture 1>, ModelSamplingMiniMaxH3 1.0/1.0"
        else:
            pos = HI.prime_blank_picture(pos, HI.picture_block(clip, blank), *win)
            how = ""
        tail = " | blank_picture %s %.3f-%.3f = step(s) %s of %d on the model's own grid" % (
            blank, win[0], win[1], ",".join(str(k) for k in pic) or "none", n)
        # v1346: a wired sigma curve moves this -- the window is a SIGMA range, not a step count; the
        # Sampler prints the steps it really covers (Frank 19.11.: 1-4 of 8 on the HyperFlow curve)
        tail += " (a curve on the Sampler's sigmas moves this: the Sampler names the real steps)"
        if shared:
            tail += " (step %s on the edge: both averaged)" % ",".join(str(k) for k in shared)
        if how:
            tail += " (%s)" % how
        return pos, tail

    def _image_family(self, fam, how, clip, vae, prompt, negative_prompt,
                      ref_size, refs, inpaint_mask=None, inpaint_grow=16,
                      inpaint_feather=8):
        import torch
        import comfy.utils
        import comfy.model_management
        from .ph_reference import append_reference

        sizes = [(int(img.shape[2]), int(img.shape[1])) for _, img in refs]
        p = plan(fam, sizes, ref_size, vae is not None)
        nch = vae_channels(vae) if vae is not None else 3

        images_vl, ref_latents, lines, canvas = [], [], [], None
        for (key, img), spec in zip(refs, p["refs"]):
            one = img[:1]
            vl_s, vae_s = spec["vl"], spec["vae"]
            if vl_s is not None:
                vl = _resize(one, vl_s, "lanczos" if fam == FAMILY_QWEN21 else "area",
                             comfy.utils)
                rgb = vl[:, :, :, :3]
                if fam == FAMILY_QWEN21 and vl.shape[-1] > 3:
                    a = vl[:, :, :, 3:4]
                    rgb = rgb * a + (1.0 - a)        # alpha over white, as Core
                images_vl.append(rgb)
            if vae is not None:
                px = _resize(one, vae_s, "lanczos" if fam == FAMILY_QWEN21 else "area",
                             comfy.utils)
                ref_latents.append(vae.encode(px[:, :, :, :max(3, nch)]))
                if canvas is None and key == "image_1":
                    canvas = (px, vae_s)
            lines.append("%s %dx%d -> vision %s, latent %s" % (
                key, sizes[len(lines)][0], sizes[len(lines)][1],
                "%dx%d" % vl_s if vl_s else "-",
                "%dx%d" % vae_s if vae is not None else "-"))

        if fam == FAMILY_QWEN21:
            kw = {"images": images_vl, "keep_vision": p["keep_vision"],
                  "prevent_empty_text": True}
            text = prompt
        elif fam == FAMILY_QWEN_EDIT:
            kw = {"images": images_vl, "llama_template": EDIT_TEMPLATE}
            text = edit_prefix(len(images_vl)) + prompt
        else:
            kw = {}
            text = prompt
        pos = clip.encode_from_tokens_scheduled(clip.tokenize(text, **kw))
        neg_text = negative_prompt if fam != FAMILY_QWEN_EDIT else (
            edit_prefix(len(images_vl)) + negative_prompt)
        neg = clip.encode_from_tokens_scheduled(clip.tokenize(neg_text, **kw))
        for rl in ref_latents:
            pos = append_reference(pos, rl)
            neg = append_reference(neg, rl)

        if p["latent"] is not None:
            ch, lh, lw = p["latent"]
            samples = torch.zeros([1, ch, lh, lw],
                                  device=comfy.model_management.intermediate_device())
        else:
            # reference_latents: the VAE knows its own geometry
            if vae is None:
                raise ValueError(
                    "[PLS] Image Reference: family reference_latents needs a "
                    "VAE -- this family carries pictures only as latents."
                )
            ch = int(getattr(vae, "latent_channels", 16) or 16)
            div = int(getattr(vae, "downscale_ratio", 8) or 8)
            fw, fh = (p["refs"][0]["vae"] if p["refs"]
                      else fit_area(ref_size or 1024, ref_size or 1024, ref_size, div))
            samples = torch.zeros([1, ch, fh // div, fw // div],
                                  device=comfy.model_management.intermediate_device())
        out = {"samples": samples}
        inote = ""
        if inpaint_mask is not None:
            # v1318: inpainting -- the canvas IS image_1 (VAEEncode at the canvas size), the mask its noise
            # mask (SetLatentNoiseMask), as the Core inpaint templates wire it
            px, (cw, ch_) = canvas
            enc = vae.encode(px[:, :, :, :3])
            nm, share = inpaint_noise_mask(inpaint_mask, cw, ch_, inpaint_grow, inpaint_feather)
            out = {"samples": enc, "noise_mask": nm}
            same = tuple(enc.shape[-2:]) == tuple(samples.shape[-2:])
            inote = (" | inpaint: latent = image_1 encoded %s, noise mask %dx%d grown %d px, feathered %d px "
                     "(outward), %.1f %% of the canvas repainted%s" % (
                         "x".join(str(int(x)) for x in enc.shape[1:]), cw, ch_, inpaint_grow,
                         inpaint_feather, 100.0 * share,
                         "" if same else " -- the VAE's grid differs from the family rule "
                         "(%s), the encoded size is used" % "x".join(str(int(x)) for x in samples.shape[-2:])))
            samples = enc
        info = "family %s (%s) | %d reference(s) | %s | vision %s | latent %s%s" % (
            fam, how, len(refs),
            "; ".join(lines) if lines else "no image wired -> text only",
            "kept" if p["keep_vision"] else ("tokens + latents" if vae is not None else "tokens"),
            "x".join(str(int(x)) for x in samples.shape[1:]), inote)
        print("[PLS] Image Reference: " + info)
        return (pos, neg, out, info, text)


def _resize(img, size, method, cu):
    w, h = size
    if int(img.shape[2]) == w and int(img.shape[1]) == h:
        return img
    s = cu.common_upscale(img.movedim(-1, 1), w, h, method, "disabled")
    return s.movedim(1, -1)
