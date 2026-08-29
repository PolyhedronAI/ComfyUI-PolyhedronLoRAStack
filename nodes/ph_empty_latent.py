"""
ph_empty_latent.py -- ⬡ Polyhedron Empty Latent (ULSEmptyLatent).

A single empty-latent node that unifies three worlds:
  1. the FULL WanImageToVideo pin set (positive/negative/vae/clip_vision_output/
     start_image -> positive/negative/latent) -- a drop-in for that node,
  2. correct empty latents for ANY model, chosen by a categorized `latent_type`
     selector (Image for every image model; WAN/Hunyuan/Mochi/LTXV/Cosmos for the
     video architectures) with a torch-free shape recipe in uls_latent_math,
  3. reproducible, spectrally-typed init NOISE (uls_noise) with a live preview.

v517 -- latent TYPE, not a flat model list. A flat family list is always
incomplete (Flux2, Qwen-Edit, ...). For image models the family barely matters
(ComfyUI re-fits the channel count and every image VAE is /8), so they collapse
to ONE "Image" entry that absorbs every present and future image model. Only
VIDEO needs explicit entries (temporal packing is architectural, never guessed).

EXACTNESS FROM THE VAE: `vae` is now OPTIONAL and, when connected, is read for
latent_channels + spatial ratio -> the geometry is exact for any model (SD=4ch,
Flux=16ch, Mochi=12ch, even a /32 image VAE) and correct even with non-zero init
noise, where ComfyUI's zero-only channel re-fit does NOT fire. No VAE -> safe /8
16ch image defaults / the cited video recipes. VAE is also used on the WAN path
to encode start_image (I2V); WAN + start_image without a VAE is a clear error.

DROP-IN GUARANTEE: latent_type=WAN + noise=zeros delegates to the REAL registered
WanImageToVideo and returns its exact output. Noise and the other types are
strictly additive on top of that baseline.

REPRODUCIBILITY: every output-affecting value is a normal serialized widget, so
ComfyUI embeds them in the PNG workflow automatically; re-loading the PNG restores
them, and the seeded-CPU noise (uls_noise) regenerates bit-for-bit. Nothing that
affects the output lives in un-serialized frontend state.
"""

import torch

import comfy.model_management

try:  # package load (ComfyUI) vs direct module load (tools/tests)
    from . import uls_latent_math
    from . import uls_noise
except ImportError:  # pragma: no cover
    import os as _os
    import sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    import uls_latent_math
    import uls_noise

# MAX_RESOLUTION mirrors the core empty-latent nodes (kept local so import can't fail).
try:
    from nodes import MAX_RESOLUTION as _MAX_RES
except Exception:  # pragma: no cover
    _MAX_RES = 16384


def _vae_geometry(vae):
    """Read (channels, spatial_div) off a connected VAE object, defensively.

    Returns (None, None) on any uncertainty so the caller falls back to the type
    default -- ComfyUI VAE internals vary across versions, so every access is
    getattr/try-guarded. channels comes from latent_channels; spatial from the
    downscale/upscale ratio (same magnitude spatially) or spacial_compression."""
    if vae is None:
        return (None, None)
    ch = None
    try:
        c = getattr(vae, "latent_channels", None)
        if c is not None and int(c) >= 1:
            ch = int(c)
    except Exception:
        ch = None
    sdiv = None
    for attr in ("downscale_ratio", "upscale_ratio"):
        try:
            v = getattr(vae, attr, None)
            if callable(v):
                v = v()
            if v and float(v) >= 1:
                sdiv = int(round(float(v)))
                break
        except Exception:
            pass
    if sdiv is None:
        try:
            f = getattr(vae, "spacial_compression_encode", None)
            if callable(f):
                r = f()
                if r and float(r) >= 1:
                    sdiv = int(round(float(r)))
        except Exception:
            sdiv = None
    return (ch, sdiv)


def _core_empty(name, width, height, batch_size):
    """Borrow a core empty-latent node by registry name (same pattern as
    the WAN delegate below). Returns the latent dict or None when the
    host has no such node."""
    try:
        import nodes as comfy_nodes
        cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
        if cls is None:
            return None
        inst = cls()
        fn = getattr(inst, getattr(cls, "FUNCTION", "generate"))
        return fn(width=width, height=height, batch_size=batch_size)[0]
    except Exception as e:
        print(f"[PLS] Empty Latent: core delegate '{name}' failed: {e!r}")
        return None


def _wan_delegate(positive, negative, vae, width, height, length, batch_size,
                  start_image, clip_vision_output):
    """Call the registered core WanImageToVideo for the conditioning surgery +
    base (zeros) latent. Returns (positive, negative, latent_dict) or None if the
    core node is not registered."""
    try:
        import nodes as comfy_nodes
        cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("WanImageToVideo")
        if cls is None:
            return None
        inst = cls()
        fn = getattr(inst, getattr(cls, "FUNCTION", "encode"))
        return fn(positive, negative, vae, width, height, length, batch_size,
                  start_image=start_image, clip_vision_output=clip_vision_output)
    except Exception as e:
        print(f"[PLS] Empty Latent: WanImageToVideo delegation unavailable ({e}); "
              f"using built-in fallback.")
        return None


def _wan_fallback(positive, negative, vae, width, height, length, batch_size,
                  start_image, clip_vision_output):
    """Faithful transcription of core WanImageToVideo.encode -- fallback only.
    Live-verified against ComfyUI; the delegation path above is preferred."""
    import comfy.utils
    import node_helpers
    shape = uls_latent_math.plan_latent_shape("wan", width, height, length, batch_size)
    latent = torch.zeros(shape, device=comfy.model_management.intermediate_device())
    if start_image is not None:
        start_image = comfy.utils.common_upscale(
            start_image[:length].movedim(-1, 1), width, height, "bilinear", "center"
        ).movedim(1, -1)
        image = torch.ones((length, height, width, start_image.shape[-1]),
                           device=start_image.device, dtype=start_image.dtype) * 0.5
        image[:start_image.shape[0]] = start_image
        concat_latent_image = vae.encode(image[:, :, :, :3])
        mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2],
                           concat_latent_image.shape[-1]),
                          device=start_image.device, dtype=start_image.dtype)
        mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0
        positive = node_helpers.conditioning_set_values(
            positive, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
        negative = node_helpers.conditioning_set_values(
            negative, {"concat_latent_image": concat_latent_image, "concat_mask": mask})
    if clip_vision_output is not None:
        positive = node_helpers.conditioning_set_values(
            positive, {"clip_vision_output": clip_vision_output})
        negative = node_helpers.conditioning_set_values(
            negative, {"clip_vision_output": clip_vision_output})
    return (positive, negative, {"samples": latent})


def _legacy_noise_note(noise_type):
    """v685: the noise that reaches the model comes from the sampler, not from
    a latent's contents -- at a full schedule the latent is multiplied by
    (1 - sigma[0]) = 0. These widgets are hidden in the UI since v685 and kept
    only so saved graphs keep their widgets_values positions. A graph that
    still carries a non-zero value gets told, once per run, where noise lives
    now -- silent legacy behaviour is worse than a loud one."""
    if str(noise_type) != "zeros":
        print("[PLS] Empty Latent: noise_type=%s is LEGACY and only does "
              "anything below a full schedule (denoise < 1). The noise the "
              "model denoises comes from the sampler -- use the Polyhedron "
              "Seed node's `noise` output." % noise_type)


def _size_outputs(samples, width, height):
    """The four size outputs, taken from the TENSOR rather than from the
    widgets. The widgets are a request; the tensor is what was built after the
    VAE probe, a core delegate or the spec fallback has had its say -- and the
    difference between the two is exactly the bug this pack chased through
    v679/v680 (a delegate rounding 1448 down to the /16 grid). Report what
    exists, never what was asked for."""
    try:
        shp = tuple(samples.shape)
        lw, lh = int(shp[-1]), int(shp[-2])
    except Exception:
        lw = lh = 0
    return (lw, lh, int(width), int(height))


class ULSEmptyLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ── the categorized latent TYPE (drives shape + greying + WAN surgery) ──
                "latent_type": (uls_latent_math.LATENT_TYPE_LABELS,
                                {"default": "Image",
                                 "tooltip": "What kind of latent to emit.\n"
                                            "Image  -> 2D latent for ANY image model (SD, SDXL, SD3, "
                                            "Flux, Flux2, Qwen-Image/Edit, ...). Future image models "
                                            "need no new entry.\n"
                                            "WAN / Hunyuan / Mochi / LTXV / Cosmos -> 5D VIDEO latent "
                                            "with that architecture's frame packing; the 'length' "
                                            "control becomes active.\n"
                                            "Wire the VAE for exact channels/scale; otherwise safe /8 "
                                            "16ch defaults are used (fine for zeros noise)."}),
                "width": ("INT", {"default": 1024, "min": 16, "max": _MAX_RES, "step": 8,
                                  "tooltip": "Pixel width. Snaps to the VAE spatial grid."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": _MAX_RES, "step": 8,
                                   "tooltip": "Pixel height. Snaps to the VAE spatial grid."}),
                "length": ("INT", {"default": 81, "min": 1, "max": _MAX_RES, "step": 4,
                                   "tooltip": "VIDEO types only: pixel frame count (packed into latent "
                                              "frames per architecture; length=1 -> a single-frame "
                                              "still). Ignored / greyed for Image. MiniMax H3: "
                                              "length=1 builds a true single-frame IMAGE latent "
                                              "(one latent frame, the mode image datasets train in). "
                                              "Above 1 it is video and Core's grid applies: 2-4 come "
                                              "back as 5, then 22, 39, ... (17k+5). The pill under "
                                              "the fields shows what your value becomes, and "
                                              "duration_seconds above 0 overrides this field."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096,
                                       "tooltip": "Number of latents in the batch."}),
                # ── reproducible init noise ──
                "noise_type": (uls_noise.NOISE_TYPES,
                               {"default": "zeros",
                                "tooltip": "Init noise written INTO the latent. zeros = the standard "
                                           "empty latent (the sampler makes all the noise; correct at "
                                           "denoise 1.0). gaussian = flat white. pink/brown/blue = "
                                           "colored (brown biases composition, blue biases detail). "
                                           "fractal = coherent multi-octave structure. Effect is "
                                           "strongest on flow models (Wan/Flux) and at denoise<1.0. "
                                           "For non-zero noise on a non-16ch model, wire the VAE so "
                                           "the channel count is exact."}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                       "control_after_generate": True,
                                       "tooltip": "Seed for the init noise (CPU-deterministic -> "
                                                  "reproducible across machines and PNG re-loads). "
                                                  "Set control_after_generate to 'fixed' for a "
                                                  "reproducible image; drag the preview to scrub it."}),
                "noise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01,
                                             "round": 0.01,
                                             "tooltip": "Init-noise scale in units of standard latent "
                                                        "noise (1.0 == torch.randn scale). Greyed when "
                                                        "noise_type is zeros."}),
                # v872: APPENDED, never inserted. LiteGraph restores
                # widgets_values POSITIONALLY -- putting this next to 'length'
                # where it belongs semantically would shift every widget after
                # it in every saved workflow. The v585 law wins over tidiness.
                "duration_seconds": ("FLOAT",
                                     {"default": 0.0, "min": 0.0, "max": 3600.0,
                                      "step": 0.1,
                                      "tooltip":
                                      "MiniMax H3 AV only. 0 = off, 'length' "
                                      "decides. Above 0 this is the clip "
                                      "length in SECONDS at 24 fps, snapped up "
                                      "to the model's 17k+5 frame grid -- it "
                                      "replaces the Float + Math Expression "
                                      "pair from the stock template. The node "
                                      "prints which of the two won."}),
            },
            "optional": {
                # ── conditioning parity with WanImageToVideo (I2V returns these) ──
                "positive": ("CONDITIONING", {"tooltip": "Conditioning to include. On the WAN "
                                              "path this becomes the I2V positive; for every other "
                                              "type it passes through untouched to the output."}),
                "negative": ("CONDITIONING", {"tooltip": "Conditioning to exclude. Passes through "
                                              "untouched for non-WAN types."}),
                # v873: positive/negative moved required -> optional. In the
                # MiniMax H3 chain the conditioning is BORN downstream, in the
                # reference stage, which needs this latent as its input -- as
                # required sockets these two made the graph a RING that could
                # not be wired at all. They are link-typed sockets, so widget
                # order is untouched and both baselines still stand.
                # The one lane that genuinely needs them refuses by name below.
                # vae is OPTIONAL: used to read EXACT channels/scale when connected,
                # and on the WAN path to VAE-encode start_image into the I2V concat.
                "vae": ("VAE", {"tooltip": "Optional. When connected, its latent channel count and "
                                "spatial ratio make the emitted geometry exact for the loaded model "
                                "(and correct with non-zero noise). Also required on the WAN path if "
                                "a start_image is provided (to encode it into the I2V conditioning)."}),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",
                                       {"tooltip": "WAN only: CLIP-vision features injected into both "
                                        "conditionings (I2V semantic guidance). Ignored for other types."}),
                "start_image": ("IMAGE",
                                {"tooltip": "WAN only: reference frame(s), VAE-encoded into the "
                                 "concat conditioning with a frame mask (I2V). Ignored for other types."}),
            },
        }

    # v688: four APPENDED outputs. Outputs may grow at the END -- existing links
    # keep their slot indices -- but never be re-ordered, so this order is now
    # canon: the two LATENT sizes first (they sit directly under `latent`, which
    # is what they describe), the two PIXEL sizes after.
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",
                    "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("positive", "negative", "latent",
                    "latent_width", "latent_height", "width", "height")
    OUTPUT_TOOLTIPS = (
        "Conditioning to include (I2V positive on the WAN path).",
        "Conditioning to exclude.",
        "The empty latent.",
        "LATENT grid width -- the tensor's own last dimension, AFTER the VAE "
        "probe / core delegate / spec fallback has had its say. This is the "
        "number the Polyhedron Seed node's preview_width wants: it is already "
        "divided by the model's spatial factor (1024px at /8 = 128, 1440px "
        "Flux2 at /16 = 90).",
        "LATENT grid height -- see latent_width.",
        "PIXEL width actually used, after grid snapping. Not the same number "
        "as latent_width: this one is what the image comes out as.",
        "PIXEL height actually used, after grid snapping.",
    )
    FUNCTION = "generate"
    CATEGORY = "Polyhedron/Latent"   # v578: was "PolyhedronLoRAStack/latent" - a
    # SECOND top-level menu tree, so the node hid outside the Polyhedron branch.
    # Category is cosmetic: it moves the node in the menu, and touches no saved
    # workflow (those store the node_id, never the category).
    DESCRIPTION = ("Unified empty latent: one Image type for every image model, explicit video "
                   "architectures, exact geometry from a connected VAE, full WanImageToVideo pin "
                   "parity, and reproducible spectrally-typed init noise with a live preview.")

    def _minimax_av(self, positive, negative, width, height, length,
                    batch_size, noise_type, noise_seed, noise_strength,
                    duration_seconds):
        """A JOINT audio/video latent for MiniMax H3.

        MIRRORS Core's _empty_av_latent (comfy_extras/nodes_minimax_h3.py):
        video [B, 24, latent_t, H//16, W//16] and audio [B, 32, 2, audio_t]
        inside a NestedTensor. Core ships EmptyMiniMaxH3LatentAV as a standalone
        node, so building this OUTSIDE the reference stage is Core's own design,
        not a detour around it -- and it puts the size, the length and the noise
        back on wires you can see instead of under a hood.
        """
        M = uls_latent_math
        # v899: read the REQUEST first, snap second. The old form snapped inside
        # the seconds branch, so by the time anything could ask "is this one
        # frame?" the answer had already been rounded up to five.
        if duration_seconds and duration_seconds > 0:
            frames_in = int(round(float(duration_seconds) * M.MINIMAX_FPS))
            src = "duration_seconds=%.2f s" % float(duration_seconds)
        else:
            frames_in = int(length)
            src = "length=%d" % int(length)

        # v899: ONE frame is a still, and the model supports it -- see
        # uls_latent_math.minimax_image_shape for the two sources. Everything
        # else goes through Core's video arithmetic unchanged, floor included:
        # 2..4 really do become 5, because there Core's grid is what governs.
        image_mode = (frames_in == 1)
        if image_mode:
            frame_count, latent_t, audio_t = M.minimax_image_shape()
        else:
            frame_count, latent_t, audio_t = M.minimax_temporal_shape(frames_in)

        div = M.MINIMAX_CANVAS_MULTIPLE
        w = max(div, int(round(width / float(div))) * div)
        h = max(div, int(round(height / float(div))) * div)

        try:
            import comfy.model_management
            dev = comfy.model_management.intermediate_device()
        except Exception:
            dev = None
        video = torch.zeros([batch_size, M.MINIMAX_VIDEO_CH, latent_t,
                             h // M.MINIMAX_SPATIAL_DIV,
                             w // M.MINIMAX_SPATIAL_DIV], device=dev)
        audio = torch.zeros([batch_size, M.MINIMAX_AUDIO_CH, 2, audio_t],
                            device=dev)

        import comfy.nested_tensor
        samples = comfy.nested_tensor.NestedTensor((video, audio))
        if noise_type != "zeros":
            samples = uls_noise.shaped_noise_joint(
                samples, noise_type, noise_seed, noise_strength)
            samples = comfy.nested_tensor.NestedTensor(
                [t.to(video.device, dtype=video.dtype) for t in samples.unbind()])

        note = ""
        if (w, h) != (int(width), int(height)):
            note += " (size snapped from %dx%d to the model's 32 grid)" % (
                int(width), int(height))
        if image_mode:
            note += " (image mode: one latent frame, no video grid)"
        elif frame_count != frames_in:
            # v899 corrects v898's wording as well. Below five it is the FLOOR
            # of Core's AV node (align_frame_count(max(5, length))) -- not the
            # model's, which is what v898 claimed; above it, the 17k+5 grid.
            note += (" (frames %d -> %d: Core's video node has a floor of 5; "
                     "use length=1 for a still)"
                     if frames_in < 5 else
                     " (frames snapped %d -> %d onto the 17k+5 grid)") % (
                frames_in, frame_count)
        print("[PLS] Empty Latent: type=minimax_h3 %s -> %d frame%s = %.2f s "
              "@ %d fps | video %s + audio %s | noise=%s seed=%s%s"
              % (src, frame_count, "" if frame_count == 1 else "s",
                 frame_count / float(M.MINIMAX_FPS),
                 M.MINIMAX_FPS, tuple(video.shape), tuple(audio.shape),
                 noise_type, noise_seed, note))
        lat = {"samples": samples}
        return (positive, negative, lat,
                video.shape[4], video.shape[3], w, h)

    def generate(self, positive=None, negative=None, latent_type="Image",
                 width=1024, height=1024, length=1,
                 batch_size=1, noise_type="zeros", noise_seed=0,
                 noise_strength=1.0, duration_seconds=0.0,
                 vae=None, clip_vision_output=None, start_image=None,
                 _core=None):
        # v873: the PARAMETER ORDER is unchanged on purpose. positive/negative
        # became optional SOCKETS, but callers in this tree pass them
        # POSITIONALLY (test_v679 does). Moving them to the end would have been
        # tidier and would have broken every one of them, so everything from
        # here on simply carries a default instead.
        _legacy_noise_note(noise_type)
        key = uls_latent_math.canonical_type(latent_type)
        vae_ch, vae_sdiv = _vae_geometry(vae)

        if key == "minimax_h3":
            return self._minimax_av(positive, negative, width, height, length,
                                    batch_size, noise_type, noise_seed,
                                    noise_strength, duration_seconds)

        if key == "wan":
            # v873: WAN is the one lane that consumes the conditioning instead
            # of passing it through -- core's I2V surgery writes into it. Say so
            # by name; a None reaching _wan_delegate would break somewhere far
            # from here.
            if positive is None or negative is None:
                raise ValueError(
                    "Empty Latent: latent_type=WAN needs positive AND negative "
                    "conditioning -- the WAN lane performs core's I2V surgery "
                    "on them, it does not just pass them through. Wire both, "
                    "or pick a latent_type that only builds a latent.")
            # WAN path: reuse the real core I2V surgery, then overlay noise on the base latent.
            if start_image is not None and vae is None:
                raise ValueError(
                    "Empty Latent: latent_type=WAN with a start_image requires a VAE "
                    "(it encodes the reference frame into the I2V conditioning). "
                    "Connect the VAE, or remove the start_image for a plain WAN latent.")
            out = _wan_delegate(positive, negative, vae, width, height, length, batch_size,
                                start_image, clip_vision_output)
            if out is None:
                out = _wan_fallback(positive, negative, vae, width, height, length, batch_size,
                                    start_image, clip_vision_output)
            pos, neg, lat = out
            if noise_type != "zeros":
                base = lat["samples"]
                noise = uls_noise.make_noise(noise_type, tuple(base.shape),
                                             noise_seed, noise_strength)
                lat = dict(lat)
                lat["samples"] = noise.to(base.device, dtype=base.dtype)
            print(f"[PLS] Empty Latent: type=wan shape={tuple(lat['samples'].shape)} "
                  f"noise={noise_type} seed={noise_seed} "
                  f"i2v={'yes' if start_image is not None else 'no'} "
                  f"vae={'exact' if vae_ch else 'default'}")
            return (pos, neg, lat) + _size_outputs(lat["samples"], width, height)

        # All other types (Image + non-WAN video): pure latent factory; conditioning
        # passes through untouched. TRUTH ORDER: wired VAE probe > core delegate
        # (host ComfyUI's own empty node, registry lookup) > spec fallback.
        delegate = uls_latent_math.CORE_DELEGATES.get(key)
        if delegate and vae_ch is None and vae_sdiv is None:
            lat = (_core if _core is not None
                   else _core_empty)(delegate, width, height, batch_size)
            if lat is not None:
                base = lat["samples"]
                if noise_type != "zeros":
                    noise = uls_noise.make_noise(noise_type, tuple(base.shape),
                                                 noise_seed, noise_strength)
                    lat = dict(lat)
                    lat["samples"] = noise.to(base.device, dtype=base.dtype)
                print(f"[PLS] Empty Latent: type={key} "
                      f"shape={tuple(lat['samples'].shape)} "
                      f"noise={noise_type} seed={noise_seed} source=core")
                return ((positive, negative, lat)
                        + _size_outputs(lat["samples"], width, height))
            print(f"[PLS] Empty Latent: type={key} -- host has no "
                  f"'{delegate}', falling back to the spec row (wire the "
                  f"VAE for exact geometry)")
        shape = uls_latent_math.plan_latent_shape(key, width, height, length, batch_size,
                                                  channels=vae_ch, spatial_div=vae_sdiv)
        samples = uls_noise.make_noise(noise_type, shape, noise_seed, noise_strength)
        samples = samples.to(comfy.model_management.intermediate_device())
        print(f"[PLS] Empty Latent: type={key} shape={tuple(samples.shape)} "
              f"noise={noise_type} seed={noise_seed} "
              f"source={'vae' if (vae_ch or vae_sdiv) else 'spec'}")
        return ((positive, negative, {"samples": samples})
                + _size_outputs(samples, width, height))
