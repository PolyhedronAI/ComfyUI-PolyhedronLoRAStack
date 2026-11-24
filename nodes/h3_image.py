"""h3_image.py -- MiniMax H3 IMAGE / p-h3-image support for the existing nodes (v1325). No node of its own.

MiniMax H3 Image (MiniMax, 10.11.2026) and p-h3-image (Pruna AI, 03.11.2026) are image models on the H3
backbone: one checkpoint for text to image and for edits of image_1 with the others as references
(<Picture 1>, <Picture 2>, ... in the prompt), six distilled steps (p-h3-image eight), cfg 1.

HOW THE PACK CARRIES THEM (Frank, 13.11.: our nodes support the model, no new node unless unavoidable)

  text to image   Polyhedron CLIP Text Encode + Polyhedron Empty Latent, type "MiniMax H3 Image (24ch, one
                  frame)". MEASURED in Core 0.45.0 (comfy/text_encoders/minimax.py tokenize_with_weights): with no
                  pictures the H3 tokenizer encodes the text exactly as a plain clip.tokenize -- the picture
                  label is only written in front of a picture. So the CTE's encode IS Core's text-to-image
                  conditioning, // rubrics included.
  edits           Polyhedron Image Reference, family minimax_h3_image (read from the H3 tokenizer + the IMAGE
                  VAE's class). The recipe is DELEGATED to Core's own class (MiniMaxH3ImageConditioning /
                  PH3ImageConditioning): it is young and will move, a transcription here would drift.
  sampler         Polyhedron Sampler, 6 steps, cfg 1, euler + simple, sigma_shift 0 (a shift is ignored on this
                  model's own sigma grid and the console says so).

This file holds the shared pieces: the variant from the VAE class, Core's class, the encode, the zeroed
negative, the latent's pixel size.
"""

VARIANT_H3 = "MiniMax H3 Image"
VARIANT_PH3 = "p-h3-image"

# Core's own node per variant (node_id == class name in comfy_extras/nodes_minimax_h3.py)
CORE_NODE = {VARIANT_H3: "MiniMaxH3ImageConditioning",
             VARIANT_PH3: "PH3ImageConditioning"}
# VAE class -> variant (comfy/sd.py picks the class from the decoder keys)
VAE_VARIANT = {"MiniMaxH3ImageVAE": VARIANT_H3, "PH3ImageVAE": VARIANT_PH3}
IMAGE_VAES = tuple(VAE_VARIANT)


def vae_class(vae):
    """Class name of the VAE's model, '' for no VAE."""
    if vae is None:
        return ""
    return type(getattr(vae, "first_stage_model", None)).__name__


def variant_for(cls_name):
    """The variant for an IMAGE VAE class; None for anything else."""
    return VAE_VARIANT.get(str(cls_name or ""))


def core_class(variant):
    """Core's own conditioning class for the variant, or None on a Core without it."""
    try:
        import comfy_extras.nodes_minimax_h3 as M
    except Exception:
        return None
    return getattr(M, CORE_NODE[variant], None)


def encode(clip, prompt, width, height, vae=None, images=None, variant=VARIANT_H3):
    """(positive, latent) from Core's own node. width/height 0 = Core decides (image_1 after its reference
    scaling, else the model's own side). Raises by name on a Core without the node."""
    cls = core_class(variant)
    if cls is None:
        raise RuntimeError(
            "[PLS] %s: this ComfyUI has no %s node -- MiniMax H3 Image needs Core 0.45.0 (10.11.2026) or "
            "newer, p-h3-image Core 0.44 (03.11.2026) or newer. Update ComfyUI." % (variant, CORE_NODE[variant]))
    res = cls.execute(clip=clip, prompt=prompt, width=int(width), height=int(height),
                      vae=vae, images=images or None)
    args = getattr(res, "args", res)
    return args[0], args[1]


def zero_out(conditioning):
    """ConditioningZeroOut's rule (comfy_extras/nodes_cond.py): the tensor and every pooled entry zeroed,
    the rest of the dict kept."""
    import torch
    out = []
    for t in conditioning:
        d = t[1].copy()
        for key in ("pooled_output", "conditioning_lyrics"):
            if d.get(key, None) is not None:
                d[key] = torch.zeros_like(d[key])
        out.append([torch.zeros_like(t[0]), d])
    return out


def latent_px(samples):
    """Pixel size (w, h) of an H3 image latent. p-h3-image hands back a joint latent (a NestedTensor, the
    video part first) -- read that part."""
    v = samples.unbind()[0] if type(samples).__name__ == "NestedTensor" else samples
    return int(v.shape[-1]) * 16, int(v.shape[-2]) * 16

# ---- v1338: blank picture prime (Frank 17.11.) ---------------------------------------------------------
# MEASURED in the field video (Aitrepreneur, MiniMax Image Ultra) and read in KJNodes' MiniMax H3 Prompt
# Weight, mode "text with empty image" (GPL-3; this is our own implementation of the idea, no code taken):
# a blank picture's vision rows put IN FRONT of the text rows for the first steps soften the hard
# text-to-image rendering of the distilled model. The H3 presentation has no chat template and the encoder
# is causal, so the rows "Picture 1: <vision>" encoded alone are exactly the rows that would stand in front
# of any prompt -- the text rows stay as they were encoded without a picture.
#
# WHERE THE WINDOW FALLS. MiniMaxH3ImageModelSampling.percent_to_sigma is LINEAR IN THE GRID INDEX, so on
# the 6-step grid step k (0-based) runs at percent k/6. Core's cond window is closed on both ends
# (comfy/samplers.py: a cond is skipped only above timestep_start or below timestep_end), so a step that
# sits exactly on `end` sees BOTH conds and they are averaged. steps_covered() says which steps are hit.

BLANK_TONES = ["off", "black", "mid grey", "white"]
TONE_VALUE = {"black": 0.0, "mid grey": 0.5, "white": 1.0}
BLANK_SIDE = 512            # 32 x 32 patches of 16 px, merged 2 x 2 -> 256 vision rows
PICTURE_LABEL = "Picture %d: "   # Core's MiniMaxH3ImageConditioning label (nodes_minimax_h3.py)
GRID_STEPS = 6

# ---- v1343: the blank picture on p-h3-image (Frank 18.11.) ------------------------------------------------
# MEASURED in Core 0.45.0, three differences to MiniMax H3 Image:
#   label   PH3ImageConditioning tokenizes WITHOUT picture_label, so the tokenizer's own default stands:
#           "<Picture %d>: " (comfy/text_encoders/minimax.py). "Picture 1: " is a label it never learned.
#   sink    p-h3-image's TextConnector replaces the FIRST row of every encode with a learned sink
#           embedding. Two blocks encoded apart and laid end to end would carry TWO sinks, one in the middle
#           of the prompt. So the picture and the prompt are encoded TOGETHER, in one pass, as Core encodes
#           pictures in front of a prompt -- one sink, the text rows see the picture.
#   steps   eight steps, sampled by ModelSamplingAV at shift 1.0/1.0 (ModelSamplingMiniMaxH3, as Core's
#           node description demands): percent_to_sigma = 1 - percent and the simple schedule is linear, so
#           step k (0-based) runs at percent k/8 -- the same rule as the 6-step grid, with 8.
PH3_PICTURE_LABEL = "<Picture %d>: "
PH3_STEPS = 8


def steps_for(variant):
    """Steps of a full run of the variant (the readout of the window)."""
    return PH3_STEPS if variant == VARIANT_PH3 else GRID_STEPS


def blank_image(tone, colour=None, side=BLANK_SIDE):
    """[1, side, side, 3] in the tone's value; tone "colour" takes the (r, g, b) in 0..1 of `colour`."""
    import torch
    if tone == TONE_COLOUR:
        rgb = torch.tensor([float(v) for v in colour], dtype=torch.float32)
        return rgb.view(1, 1, 1, 3).expand(1, int(side), int(side), 3).contiguous()
    return torch.full((1, int(side), int(side), 3), float(TONE_VALUE[tone]))


# ---- v1354: strength, any colour, the blank as a real picture (Frank 21.11., after WAS' reply) ------------
# READ in WAS Node Suite (MIT, modules/sampling/h3_blank.py, commit 44aec45) and in Core master
# (comfy_extras/nodes_minimax_h3.py, MiniMaxH3ImageConditioning + _image_refs); our own implementation.
#   strength  WAS mixes the two predictions on the guided steps: plain + s * (blank - plain); s = 2 on the first
#             two steps is his "subtraction". We need no guider of our own for it: Core's calc_cond_batch sums
#             every active cond as output * strength and divides by the summed strengths. Two conds in the
#             window, the blank one at strength s and the plain one at strength 1 - s, sum to 1 -> exactly
#             plain + s * (blank - plain), with ANY sampler. s = 1 leaves the plain one out (one pass a step,
#             the v1338 result bit for bit); every other s costs two passes a step inside the window.
#   colour    any colour, "#rrggbb" or "#rgb".
#   anchor    Core reads a real picture twice: its vision rows in front of the prompt (ONE encode, short edge
#             1024) AND the VAE latent as a reference block of the DiT (minimax_refs, short edge 1024..2048).
#             v1338 put only 512-px vision rows in front. With the anchor on, the blank IS a picture to Core:
#             Core's own class encodes prompt + a 1024 px blank with the VAE -- WAS builds the same by hand.
TONE_COLOUR = "colour"
BLANK_CHOICES = BLANK_TONES + [TONE_COLOUR]
ANCHOR_SIDE = 1024          # Core: IMAGE_REF_SHORT_EDGE (vision) = lower clamp of the VAE side


def parse_colour(text):
    """(r, g, b) in 0..1 from "#rrggbb" / "#rgb" (the "#" optional). Raises ValueError by name."""
    s = str(text or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(ch * 2 for ch in s)
    if len(s) != 6 or any(ch not in "0123456789abcdefABCDEF" for ch in s):
        raise ValueError("[PLS] Image Reference: blank_colour %r is not a colour -- write #rrggbb, e.g. "
                         "#808080 (mid grey) or #1a2440 (night blue)." % (text,))
    return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def colour_name(tone, colour=None):
    """The blank's name for the info: the tone, or the colour as #rrggbb."""
    if tone != TONE_COLOUR:
        return tone
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in colour)


def weigh_window(out, start, end, strength, plain):
    """v1354: entries of `out` inside [start, end] (the blank ones) get strength s; with s != 1 the plain
    conditioning joins them in the window at strength 1 - s. Core then mixes plain + s * (blank - plain)."""
    s = float(strength)
    if s == 1.0:
        return out
    res = []
    for c, d in out:
        if d.get("start_percent") == float(start) and d.get("end_percent") == float(end):
            res.append([c, dict(d, strength=s * float(d.get("strength", 1.0)))])
        else:
            res.append([c, d])
    for c, d in plain:
        res.append([c, dict(d, start_percent=float(start), end_percent=float(end),
                            strength=(1.0 - s) * float(d.get("strength", 1.0)))])
    return res


def prime_window(conditioning, blank_cond, start, end):
    """v1354 (anchor): the blank conditioning as a whole (rows, tags, minimax_refs -- Core's own encode of
    prompt + picture) during [start, end], the plain one outside."""
    out = []
    for (c, d), (bc, bd) in zip(conditioning, blank_cond):
        out.append([bc.to(c.device, c.dtype), dict(bd, start_percent=float(start), end_percent=float(end))])
        if start > 0.0:
            out.append([c, dict(d, start_percent=0.0, end_percent=float(start))])
        if end < 1.0:
            out.append([c, dict(d, start_percent=float(end), end_percent=1.0)])
    return out


def joint_block(clip, prompt, tone, colour=None):
    """[rows, dict] of "<Picture 1>: <vision> prompt" encoded in ONE pass (p-h3-image: one sink)."""
    return clip.encode_from_tokens_scheduled(
        clip.tokenize(prompt, images=[blank_image(tone, colour)], picture_label=PH3_PICTURE_LABEL))[0]


def prime_joint(conditioning, block, start, end):
    """The positive with the jointly encoded rows during [start, end], the plain text outside. The joint
    rows REPLACE the text rows inside the window (they already hold the prompt)."""
    jc, jd = block
    out = []
    for c, d in conditioning:
        nd = dict(d)
        if jd.get("minimax_token_tags") is not None:
            nd["minimax_token_tags"] = jd["minimax_token_tags"]
        elif "minimax_token_tags" in nd:
            del nd["minimax_token_tags"]
        nd["start_percent"], nd["end_percent"] = float(start), float(end)
        out.append([jc.to(c.device, c.dtype), nd])
        if start > 0.0:
            out.append([c, dict(d, start_percent=0.0, end_percent=float(start))])
        if end < 1.0:
            out.append([c, dict(d, start_percent=float(end), end_percent=1.0)])
    return out


def blank_window(start, end):
    """(start, end) clamped to 0..1; None when the window is empty."""
    s = min(1.0, max(0.0, float(start)))
    e = min(1.0, max(0.0, float(end)))
    return (s, e) if e > s else None


def steps_covered(start, end, steps=GRID_STEPS):
    """(picture_steps, shared_steps), 1-based, of a full run on the distilled grid. shared = a step that
    sits exactly on a window edge: both conditionings are active there and Core averages them."""
    eps = 1e-6
    pic, shared = [], []
    for k in range(int(steps)):
        p = k / float(steps)
        if start - eps <= p <= end + eps:
            pic.append(k + 1)
            if (abs(p - end) <= eps and end < 1.0) or (abs(p - start) <= eps and start > 0.0):
                shared.append(k + 1)
    return pic, shared


def picture_block(clip, tone, colour=None):
    """[rows, dict] of a blank picture alone: the label and its vision block, no text."""
    return clip.encode_from_tokens_scheduled(
        clip.tokenize("", images=[blank_image(tone, colour)], picture_label=PICTURE_LABEL))[0]


def prime_blank_picture(conditioning, block, start, end):
    """The positive with the blank picture's rows in front during [start, end], the plain text outside."""
    import torch
    pc, pd = block
    ptags = pd.get("minimax_token_tags")
    out = []
    for c, d in conditioning:
        nd = dict(d)
        tags = d.get("minimax_token_tags")
        if ptags is not None and tags is not None:
            nd["minimax_token_tags"] = torch.cat([ptags.to(tags.device, tags.dtype), tags])
        nd["start_percent"], nd["end_percent"] = float(start), float(end)
        out.append([torch.cat([pc.to(c.device, c.dtype), c], dim=1), nd])
        if start > 0.0:
            out.append([c, dict(d, start_percent=0.0, end_percent=float(start))])
        if end < 1.0:
            out.append([c, dict(d, start_percent=float(end), end_percent=1.0)])
    return out


# ---- v1339: what the pictures look like before Core sees them (Frank 17.11.) ----------------------------
# Core's _resize / _resize_to CUT a fourth channel (image[..., :3]): a cut-out PNG hands its hidden RGB to the
# encoder and the VAE -- under a transparent area that is often black. Our Qwen route lays alpha over WHITE
# (as Core's TextEncodeQwenImage21 does); the same rule here, applied before the pictures reach Core.


def flatten_alpha(image):
    """[B, H, W, C] -> [B, H, W, 3]; a fourth channel is laid over white, three channels pass untouched."""
    if int(image.shape[-1]) <= 3:
        return image
    rgb, a = image[..., :3], image[..., 3:4].clamp(0.0, 1.0)
    return rgb * a + (1.0 - a)
