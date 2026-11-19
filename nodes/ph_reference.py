# Polyhedron Reference -- reference-image packaging for Flux2-style
# edit models, on classic rails.
#
# Field truth behind this node: the "advanced" sampling tangle
# (RandomNoise + KSamplerSelect + CFGGuider + SamplerCustomAdvanced)
# is Core's sampler taken apart into pieces; the Polyhedron Sampler
# already runs comfy.sample.sample_custom and passes conditioning
# through untouched, so reference latents survive it as-is. What was
# actually missing is the PACKAGING side: scale both input images,
# VAE-encode them, append them to positive AND negative conditioning
# as reference latents, and provide the scene-sized empty latent plus
# the resolution-shifted sigma schedule. Eight stock nodes -> one.
#
# The conditioning operations are pure and guard-driven; the two
# Flux2-specific pieces (empty latent, scheduler) are the pack's own
# (ph_flux2, v1062) -- equal bit for bit to the core's
# EmptyFlux2LatentImage / Flux2Scheduler, which are no longer called.

import numpy as np  # noqa: F401  (torch-agnostic helpers accept both)

# v1010: the shared progress instrument (green bar + console + learned ETA).
# A harness that loads this file alone gets silent stand-ins -- the node's
# work never depends on its instruments.
try:
    from .ph_progress import NodeProgress as _NodeProgress, blocking as _blocking
except Exception:
    try:
        from ph_progress import NodeProgress as _NodeProgress, blocking as _blocking
    except Exception:
        class _blocking:
            est_total = est = None

            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def tick(self, n=1):
                pass

            def rate_left(self):
                return None, None

        _NodeProgress = _blocking


MEGAPIXEL = 1024 * 1024


def scale_to_megapixels(w, h, mp):
    """Target dimensions for a total-pixel scale, exactly Core's
    ImageScaleToTotalPixels arithmetic: uniform scale so that
    w*h ~= mp * 1024*1024, each side rounded."""
    if w <= 0 or h <= 0 or mp <= 0:
        raise RuntimeError("Reference: invalid scale request %dx%d @ %.2fMP"
                           % (w, h, mp))
    s = (mp * MEGAPIXEL / float(w * h)) ** 0.5
    return max(1, int(round(w * s))), max(1, int(round(h * s)))


def snap_to_grid(v, grid):
    """Snap a pixel dimension DOWN onto the latent grid (never up, so a
    megapixel budget is never exceeded), floor-based and deterministic.
    grid <= 1 is a no-op. Field truth (v679 run): the borrowed Flux2
    empty-latent node aligns to /16 internally -- 1448 came back as
    1440 -- so the reported width/height disagreed with the latent we
    actually produced. Snapping HERE, before anything is computed,
    makes width/height, the empty latent, the sigmas and both image
    encodes agree by construction, whatever the delegate's own rule."""
    g = int(grid)
    if g <= 1:
        return max(1, int(v))
    return max(g, (int(v) // g) * g)


def append_reference(conditioning, samples):
    """Append one reference latent to every conditioning entry --
    exactly Core's ReferenceLatent semantics (append, never replace;
    dicts are COPIED, the caller's conditioning is never mutated)."""
    out = []
    for t, d in conditioning:
        d = dict(d)
        d["reference_latents"] = list(d.get("reference_latents", [])) + [samples]
        out.append([t, d])
    return out


def zero_out(conditioning):
    """Core's ConditioningZeroOut: zeroed cond tensors (and zeroed
    pooled_output where present), dicts copied."""
    out = []
    for t, d in conditioning:
        d = dict(d)
        p = d.get("pooled_output")
        if p is not None:
            d["pooled_output"] = p * 0.0
        out.append([t * 0.0, d])
    return out


def _flux2():
    """The pack's own Flux2 empty latent + schedule (ph_flux2). v1062: they
    replace the core's EmptyFlux2LatentImage / Flux2Scheduler node classes
    (equal bit for bit, test_v1062)."""
    try:
        from . import ph_flux2 as F
    except ImportError:
        import ph_flux2 as F
    return F


def _scale_image(img, mp, grid=16, mode="fit"):
    """Scale a [B,H,W,C] image tensor to the target megapixels with
    lanczos (Core's method), then snap onto the latent grid. mp <= 0
    means NATIVE size (no megapixel scaling; the grid snap still
    applies). No-op when the size already matches.

    mode "fit" (the classic behaviour) always scales ONTO the budget --
    a small image is scaled UP to reach it. mode "shrink_only" treats
    the budget as a CEILING: an image already below it keeps its native
    size, because upscaling a reference invents no detail while paying
    the full token price for it."""
    h, w = int(img.shape[1]), int(img.shape[2])
    if mp is not None and float(mp) > 0:
        if mode == "shrink_only" and w * h <= float(mp) * MEGAPIXEL:
            nw, nh = w, h
        else:
            nw, nh = scale_to_megapixels(w, h, mp)
    else:
        nw, nh = w, h
    nw, nh = snap_to_grid(nw, grid), snap_to_grid(nh, grid)
    if (nw, nh) == (w, h):
        return img, w, h
    import comfy.utils
    s = img.movedim(-1, 1)
    s = comfy.utils.common_upscale(s, nw, nh, "lanczos", "disabled")
    return s.movedim(1, -1), nw, nh


class ULSReference:
    DESCRIPTION = (
        "Reference-image packaging for Flux2-style edit models (e.g. "
        "object or face swaps without masking). Wire the scene image "
        "into image_1 and the reference into image_2 (image_3 is there "
        "for models taking three, e.g. Qwen Edit): each is scaled to "
        "its megapixel target, VAE-encoded and appended to the "
        "positive AND negative conditioning as a reference latent, in "
        "input order -- so a prompt can say 'reference image 2'. The "
        "latent output is the empty scene-sized start latent, sigmas "
        "is the resolution-shifted schedule for the given steps, and "
        "width/height report the scene size (snapped onto size_grid, so "
        "an Empty Latent fed from them lands on the same geometry). Set "
        "megapixels_1 to 0 to keep the scene at its native size. Feed "
        "everything straight "
        "into the Polyhedron Sampler (cfg 1.0 for Flux2) -- no noise, "
        "guider or custom-sampler helper nodes needed. Without a wired "
        "negative, a zeroed copy of the positive is built for you."
    )

    @classmethod
    def INPUT_TYPES(cls):
        # SLOT ORDER NOTE (v682): LiteGraph draws every `required` pin
        # before every `optional` one, so `negative` could never sit
        # under `positive` while `vae`/`image_1` stood in `required`.
        # They moved to `optional` -- the pins are still MANDATORY in
        # substance and apply() fails loudly without them; what is lost
        # is ComfyUI's pre-run validation, what is won is the pin order
        # Frank asked for. The WIDGET canon is untouched by this: vae
        # and image_1 are pins, never widgets, so widgets_values keeps
        # its numbering.
        return {
            "required": {
                "positive": ("CONDITIONING", {
                    "tooltip": "Prompt conditioning; the reference "
                               "latents are appended to it."}),
                "megapixels_1": ("FLOAT", {
                    "default": 2.0, "min": 0.0, "max": 16.0, "step": 0.1,
                    "tooltip": "Scale target for image_1 in megapixels "
                               "before encoding. 0 = keep the image's "
                               "NATIVE size (the grid snap still "
                               "applies) -- this is your size control."}),
                "megapixels_2": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 16.0, "step": 0.1,
                    "tooltip": "Scale target for image_2 in megapixels "
                               "before encoding. 0 = native size."}),
                "zero_negative": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "ON: without a wired negative, a zeroed "
                               "copy of the positive is used (the "
                               "standard for cfg 1.0 edit models)."}),
                "steps": ("INT", {
                    "default": 4, "min": 1, "max": 100,
                    "tooltip": "Steps for the sigmas output "
                               "(resolution-shifted schedule)."}),
                "megapixels_3": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 16.0, "step": 0.1,
                    "tooltip": "Scale target for image_3 in megapixels "
                               "before encoding. 0 = native size."}),
                "size_grid": ("INT", {
                    "default": 16, "min": 1, "max": 64, "step": 1,
                    "tooltip": "Snap the scene size onto this pixel "
                               "grid (Flux2 = 16). Everything the node "
                               "reports and builds -- width/height, the "
                               "empty latent, the sigmas, both encodes "
                               "-- then agrees, so an Empty Latent fed "
                               "from width/height harmonizes exactly."}),
                "scale_mode": (["fit", "shrink_only"], {
                    "default": "fit",
                    "tooltip": "fit: always scale onto the megapixel "
                               "budget, small images included (the "
                               "classic behaviour). shrink_only: treat "
                               "the budget as a CEILING -- an image "
                               "already smaller keeps its native size "
                               "instead of being upscaled into detail "
                               "it does not have while paying the full "
                               "token price for it."}),
            },
            "optional": {
                "negative": ("CONDITIONING", {
                    "tooltip": "Optional wired negative; also receives "
                               "the reference latents."}),
                "vae": ("VAE", {
                    "tooltip": "The model's VAE -- encodes the "
                               "reference images. Mandatory in "
                               "substance; it sits here only so the "
                               "pin order can be what it is."}),
                "image_1": ("IMAGE", {
                    "tooltip": "The SCENE image (the one being "
                               "edited). Sets the output size. "
                               "Mandatory in substance."}),
                "image_2": ("IMAGE", {
                    "tooltip": "The REFERENCE image (the object or "
                               "face to bring in) -- 'reference image "
                               "2' in a prompt."}),
                "image_3": ("IMAGE", {
                    "tooltip": "A second reference image, for models "
                               "that take three (Qwen Edit; Flux2 "
                               "depending on build). Appended after "
                               "image_2 so a prompt can call it "
                               "'reference image 3'."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "SIGMAS",
                    "INT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent", "sigmas",
                    "width", "height")
    FUNCTION = "apply"
    CATEGORY = "Polyhedron/Sampling"

    def apply(self, positive, megapixels_1, megapixels_2,
              zero_negative, steps, megapixels_3=1.0, size_grid=16,
              scale_mode="fit", vae=None, image_1=None,
              negative=None, image_2=None, image_3=None,
              _encode=None, _empty=None, _sched=None):
        # vae/image_1 are optional only for pin ORDER (see INPUT_TYPES);
        # in substance they are mandatory, so say so loudly instead of
        # dying somewhere downstream with an unreadable message.
        if image_1 is None:
            raise RuntimeError(
                "Polyhedron Reference: image_1 (the scene image) is not "
                "wired -- it sets the output size, nothing can be built "
                "without it.")
        if vae is None and _encode is None:
            raise RuntimeError(
                "Polyhedron Reference: no VAE wired -- the reference "
                "images cannot be encoded.")
        if negative is None:
            if not zero_negative:
                raise RuntimeError(
                    "Polyhedron Reference: wire a negative or enable "
                    "zero_negative.")
            negative = zero_out(positive)
        pos, neg = positive, negative
        scene_w = scene_h = None
        # ORDER IS THE CONTRACT: image_1, image_2, image_3 are appended
        # in exactly this sequence, so a prompt may address them as
        # "reference image 2" / "reference image 3". Skipping image_2
        # does NOT renumber image_3 down -- wire them in order.
        g = max(1, int(size_grid))
        rows = []
        for name, img, mp in (("image_1", image_1, megapixels_1),
                              ("image_2", image_2, megapixels_2),
                              ("image_3", image_3, megapixels_3)):
            if img is None:
                continue
            src_h, src_w = int(img.shape[1]), int(img.shape[2])
            scaled, nw, nh = _scale_image(img, mp, size_grid, scale_mode)
            if scene_w is None:
                scene_w, scene_h = nw, nh
            with _blocking("Reference", "encode:%s" % type(getattr(vae, "first_stage_model", vae)).__name__,
                           size=float(scaled.shape[0] * scaled.shape[1] * scaled.shape[2]) / 1e6,
                           what="VAE encode %dx%d" % (nw, nh)):
                samples = (_encode(scaled) if _encode is not None
                           else vae.encode(scaled[:, :, :, :3]))
            pos = append_reference(pos, samples)
            neg = append_reference(neg, samples)
            # What the megapixel dial ACTUALLY produced, per input: the
            # source size, the built size, and the latent token count it
            # costs -- (w/grid)*(h/grid). The dial is an area budget, so
            # this is the only honest readout of it.
            rows.append({"name": name, "src_w": src_w, "src_h": src_h,
                         "w": nw, "h": nh, "mp": float(mp),
                         "tokens": (nw // g) * (nh // g)})
        empty = _empty if _empty is not None else _flux2().empty_latent
        latent = empty(width=scene_w, height=scene_h, batch_size=1)[0]
        sched = _sched if _sched is not None else _flux2().scheduler_sigmas
        sigmas = sched(steps=int(steps), width=scene_w, height=scene_h)[0]
        total = sum(r["tokens"] for r in rows)
        for r in rows:
            print("[PLS] Reference: %s %dx%d -> %dx%d (%.1f MP req) = "
                  "%d latent tokens"
                  % (r["name"], r["src_w"], r["src_h"], r["w"], r["h"],
                     r["mp"], r["tokens"]))
        print("[PLS] Reference: scene %dx%d (grid %d), %d reference "
              "latent(s), %d step(s), %d tokens total (mode %s)"
              % (scene_w, scene_h, int(size_grid),
                 len(pos[0][1].get("reference_latents", [])) if pos else 0,
                 int(steps), total, str(scale_mode)))
        return {"ui": {"ph_reference": [{"rows": rows, "total": total,
                                         "grid": g,
                                         "mode": str(scale_mode)}]},
                "result": (pos, neg, latent, sigmas,
                           int(scene_w), int(scene_h))}
