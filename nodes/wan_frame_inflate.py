"""
Polyhedron Wan Frame Inflate / Pick Frame

Workaround for kijai/ComfyUI-WanVideoWrapper Issue #1827:
    LoRAs have no effect when sampling with frames=1 + fp8_scaled model.

The bug is in kijai's CustomLinear / set_lora_params pipeline, which only
fires the LoRA forward path correctly when the latent has more than one
temporal frame (T > 1). At T == 1 the LoRAs load without errors but their
effect is invisible.

This pair of nodes works around the bug WITHOUT modifying kijai's code:

  • ULSWanFrameInflate  — sits between WanVideoEmptyEmbeds (or any embeds
    source) and WanVideoSampler. It bumps target_shape[1] from 1 to N
    latent frames (default 5, → 17 video frames at VAE_STRIDE=4) so the
    sampler runs in video mode and LoRAs become active.

  • ULSImagePickFrame  — sits after WanVideoDecode. Picks one frame from
    the decoded batch (default: middle frame, which the sampler tends to
    converge on best in short multi-frame runs).

Cost: sampling time roughly 2x a single-frame run (self-attention scales
sub-linearly with frame count for tiny T). VAE decode time is negligible
for 5 frames vs 1 frame at typical resolutions.

If kijai fixes Issue #1827 upstream, both nodes can be removed without
changing the rest of the workflow.
"""
import copy


VAE_STRIDE_T = 4  # WAN VAE temporal stride: 4 image frames per latent frame


class ULSWanFrameInflate:
    """
    Inflate a WANVIDIMAGE_EMBEDS dict from 1 frame to N frames in-place
    on a deep copy, so the sampler runs in video mode and LoRAs trigger.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_embeds": ("WANVIDIMAGE_EMBEDS",),
                "target_latent_frames": ("INT", {
                    "default": 5, "min": 2, "max": 41, "step": 1,
                    "tooltip": (
                        "Number of LATENT frames to inflate to. 5 latent "
                        "frames = 17 image frames (VAE 4:1 stride + anchor). "
                        "Higher = stronger LoRA activation but slower. "
                        "Common values: 5 (17 frames, fast), 9 (33 frames, "
                        "stronger), 21 (81 frames, full video quality)."
                    )
                }),
                "only_if_single_frame": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "When ON, only inflate if the input already has "
                        "exactly 1 latent frame. When OFF, always replace "
                        "target_latent_frames with the value above."
                    )
                }),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "inflate"
    CATEGORY = "Polyhedron/Wan"
    DESCRIPTION = (
        "Workaround for kijai Issue #1827: bumps latent frame count "
        "from 1 to N so LoRAs trigger in the sampler. Pair with "
        "ULSImagePickFrame after WanVideoDecode."
    )

    def inflate(self, image_embeds, target_latent_frames, only_if_single_frame):
        # Deep-copy so we don't mutate upstream node's cached embeds
        embeds = copy.deepcopy(image_embeds) if image_embeds is not None else {}

        target_shape = embeds.get("target_shape", None)
        if target_shape is None:
            # I2V path uses num_frames + lat_h/lat_w directly; we'd need
            # to construct target_shape ourselves, but that's risky because
            # there are I2V-specific fields (clip_context_emb, end_image,
            # add_cond_latents, etc.) we'd have to preserve. Bail out
            # cleanly and tell the user.
            cur_num = embeds.get("num_frames", "?")
            print(
                f"[ULSWanFrameInflate] ⚠ embeds has no 'target_shape' "
                f"(num_frames={cur_num}). This node currently supports "
                f"T2V-style empty embeds only. I2V flow detected — "
                f"passing through unchanged."
            )
            return (embeds,)

        cur_latent_frames = target_shape[1] if len(target_shape) > 1 else None
        if cur_latent_frames is None:
            print(
                f"[ULSWanFrameInflate] ⚠ target_shape malformed: "
                f"{target_shape}. Passing through unchanged."
            )
            return (embeds,)

        if only_if_single_frame and cur_latent_frames != 1:
            print(
                f"[ULSWanFrameInflate]   skip: latent_frames already "
                f"= {cur_latent_frames} (only_if_single_frame=True)."
            )
            return (embeds,)

        if cur_latent_frames == target_latent_frames:
            print(
                f"[ULSWanFrameInflate]   no-op: already at "
                f"{target_latent_frames} latent frames."
            )
            return (embeds,)

        # target_shape is typically a tuple — rebuild it
        new_shape = list(target_shape)
        new_shape[1] = target_latent_frames
        new_shape = tuple(new_shape)

        # num_frames is the IMAGE frame count: VAE_STRIDE_T * (lat - 1) + 1
        # For lat=5, image frames = 4*4 + 1 = 17. For lat=1, image frames = 1.
        new_num_frames = VAE_STRIDE_T * (target_latent_frames - 1) + 1

        embeds["target_shape"] = new_shape
        embeds["num_frames"] = new_num_frames

        # Mark with a tag so downstream nodes (and humans reading logs) can
        # tell that this is an inflated single-frame run.
        embeds["_uls_inflated_from"] = cur_latent_frames

        print(
            f"[ULSWanFrameInflate] ✓ inflated: latent_frames "
            f"{cur_latent_frames} → {target_latent_frames}  "
            f"(num_frames {new_num_frames}). "
            f"Pair with ULSImagePickFrame after the decoder to extract "
            f"a single image."
        )
        return (embeds,)


try:  # package import (normal ComfyUI load)
    from . import uls_pick_frame_core as _pf
except ImportError:  # script import (tests, tooling)
    import uls_pick_frame_core as _pf


class ULSImagePickFrame:
    """
    Pick a single frame from a decoded image batch. Companion to
    ULSWanFrameInflate.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_index": ("INT", {
                    "default": -1, "min": -4096, "max": 4096, "step": 1,
                    "tooltip": (
                        "Which frame to pick (0-based). In the default "
                        "'middle (legacy)' mode, -1 means the MIDDLE frame "
                        "(recommended for inflated T2I runs — the sampler "
                        "converges most cleanly on the central frame, since "
                        "the first frame can carry anchor artifacts and the "
                        "last can be slightly blurred by motion continuity). "
                        "In 'index' and 'range' it is Core's convention "
                        "instead: -1 is the LAST frame. The mode below says "
                        "which reading applies."
                    )
                }),
                # v902: APPENDED, never inserted -- widgets_values is
                # positional and an insert would shift every saved graph.
                "mode": (_pf.MODES, {"default": _pf.MODES[0],
                                     "tooltip": _pf.MODE_TOOLTIP}),
                "count": ("INT", {"default": 1, "min": 1, "max": 4096,
                                  "step": 1,
                                  "tooltip": _pf.COUNT_TOOLTIP}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "pick"
    CATEGORY = "Polyhedron/Wan"
    DESCRIPTION = (
        "Picks one frame from a video batch. Companion to "
        "ULSWanFrameInflate. -1 picks the middle frame."
    )

    def pick(self, images, frame_index, mode=None, count=1):
        # v902: the selection itself lives in nodes/uls_pick_frame_core.py
        # so this class and the V3 class cannot drift apart (v898 lesson).
        if mode is None:
            mode = _pf.MODES[0]
        picked, note = _pf.select(images, mode, frame_index, count)
        print(f"[ULSImagePickFrame] ✓ {note}")
        return (picked,)


# Node registration is centralised in the top-level __init__.py — these
# classes are referenced there. No module-level NODE_CLASS_MAPPINGS here
# to avoid the appearance of double-registration.
