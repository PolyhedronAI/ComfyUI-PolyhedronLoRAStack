"""
Polyhedron Wan Frame Inflate — V3 schema edition.

V3 (ComfyUI Nodes 2.0) form of ULSWanFrameInflate, the companion to
ULSImagePickFrame (see nodes/uls_pick_frame_v3.py). Stage 2 of the migration.

This node sits between kijai/WanVideoWrapper's WanVideoEmptyEmbeds and
WanVideoSampler and bumps the latent frame count from 1 to N so LoRAs trigger
(workaround for kijai Issue #1827). It consumes kijai's WANVIDIMAGE_EMBEDS type
but does NOT touch kijai's code — the WAN Bridge and the kijai sampler loop
remain no-touch zones; this is a separate module operating on the embeds dict.

The custom type WANVIDIMAGE_EMBEDS is declared via io.Custom("WANVIDIMAGE_EMBEDS").
Its io_type string matches kijai's V1 socket type exactly, so the V3 node's
sockets stay compatible with the surrounding legacy WanVideo nodes.

Registration: this class is collected by the pack's single V3 extension in
nodes/uls_v3_extension.py. node_id is kept IDENTICAL to the legacy key
("ULSWanFrameInflate") so existing workflows resolve it unchanged; legacy and V3
forms are mutually exclusive via the central _V3_OK flag in __init__.py.

The execute() body is a verbatim port of the legacy inflate() logic.
"""

import copy

try:  # v577: ONE door to the versioned API (nodes/ph_comfyapi.py).
    from .ph_comfyapi import io
except ImportError:  # pragma: no cover - direct module load (tools)
    from ph_comfyapi import io


VAE_STRIDE_T = 4  # WAN VAE temporal stride: 4 image frames per latent frame

# Custom type shared with kijai's WanVideoWrapper. The io_type string must match
# kijai's V1 "WANVIDIMAGE_EMBEDS" exactly for sockets to stay connectable.
_WAN_EMBEDS = io.Custom("WANVIDIMAGE_EMBEDS")

_TARGET_FRAMES_TOOLTIP = (
    "Number of LATENT frames to inflate to. 5 latent "
    "frames = 17 image frames (VAE 4:1 stride + anchor). "
    "Higher = stronger LoRA activation but slower. "
    "Common values: 5 (17 frames, fast), 9 (33 frames, "
    "stronger), 21 (81 frames, full video quality)."
)

_ONLY_IF_SINGLE_TOOLTIP = (
    "When ON, only inflate if the input already has "
    "exactly 1 latent frame. When OFF, always replace "
    "target_latent_frames with the value above."
)

_DESCRIPTION = (
    "Workaround for kijai Issue #1827: bumps latent frame count "
    "from 1 to N so LoRAs trigger in the sampler. Pair with "
    "ULSImagePickFrame after WanVideoDecode."
)


class ULSWanFrameInflateV3(io.ComfyNode):
    """
    Inflate a WANVIDIMAGE_EMBEDS dict from 1 frame to N frames in-place on a deep
    copy, so the sampler runs in video mode and LoRAs trigger. Stateless
    classmethods only — no __init__, no instance state.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ULSWanFrameInflate",         # identical to the legacy key
            display_name="\u2b21 Polyhedron Wan Frame Inflate (T2I LoRA fix)",
            category="Polyhedron/Wan",
            description=_DESCRIPTION,
            inputs=[
                _WAN_EMBEDS.Input("image_embeds"),
                io.Int.Input(
                    "target_latent_frames",
                    default=5, min=2, max=41, step=1,
                    tooltip=_TARGET_FRAMES_TOOLTIP,
                ),
                io.Boolean.Input(
                    "only_if_single_frame",
                    default=True,
                    tooltip=_ONLY_IF_SINGLE_TOOLTIP,
                ),
            ],
            outputs=[
                _WAN_EMBEDS.Output(display_name="image_embeds"),
            ],
        )

    @classmethod
    def execute(cls, image_embeds, target_latent_frames, only_if_single_frame) -> io.NodeOutput:
        # Verbatim port of the legacy ULSWanFrameInflate.inflate() logic.
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
                f"[ULSWanFrameInflate] \u26a0 embeds has no 'target_shape' "
                f"(num_frames={cur_num}). This node currently supports "
                f"T2V-style empty embeds only. I2V flow detected — "
                f"passing through unchanged."
            )
            return io.NodeOutput(embeds)

        cur_latent_frames = target_shape[1] if len(target_shape) > 1 else None
        if cur_latent_frames is None:
            print(
                f"[ULSWanFrameInflate] \u26a0 target_shape malformed: "
                f"{target_shape}. Passing through unchanged."
            )
            return io.NodeOutput(embeds)

        if only_if_single_frame and cur_latent_frames != 1:
            print(
                f"[ULSWanFrameInflate]   skip: latent_frames already "
                f"= {cur_latent_frames} (only_if_single_frame=True)."
            )
            return io.NodeOutput(embeds)

        if cur_latent_frames == target_latent_frames:
            print(
                f"[ULSWanFrameInflate]   no-op: already at "
                f"{target_latent_frames} latent frames."
            )
            return io.NodeOutput(embeds)

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
            f"[ULSWanFrameInflate] \u2713 inflated: latent_frames "
            f"{cur_latent_frames} → {target_latent_frames}  "
            f"(num_frames {new_num_frames}). "
            f"Pair with ULSImagePickFrame after the decoder to extract "
            f"a single image."
        )
        return io.NodeOutput(embeds)
