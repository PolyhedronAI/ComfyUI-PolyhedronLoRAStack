"""
Polyhedron Pick Frame — V3 schema edition.

This is the V3 (ComfyUI Nodes 2.0) form of ULSImagePickFrame, the companion to
ULSWanFrameInflate (see nodes/wan_frame_inflate.py). It is the FIRST node in the
pack migrated to the declarative comfy_api V3 schema, so it rides the Vue node
renderer instead of the legacy LiteGraph canvas path.

Registration: this class is collected by the pack's single V3 extension in
nodes/uls_v3_extension.py (which holds the one comfy_entrypoint). __init__.py
imports that entrypoint inside a try/except; if comfy_api.latest is unavailable,
the import fails, _V3_OK stays False, and every migrated node — including this
one — falls back to its legacy registration so nothing disappears.

The node id is kept IDENTICAL to the legacy key ("ULSImagePickFrame") so existing
saved workflows keep resolving the node unchanged. Because the id is shared, the
legacy and V3 forms are mutually exclusive (either/or), never both registered at
once — __init__.py enforces that via the central _V3_OK flag.

The execute() body is a verbatim port of the legacy pick() logic; behaviour is
identical (middle frame on -1, clamp otherwise, pass-through on an empty batch).
"""

try:  # v577: ONE door to the versioned API (nodes/ph_comfyapi.py).
    from .ph_comfyapi import io
except ImportError:  # pragma: no cover - direct module load (tools)
    from ph_comfyapi import io


# Kept byte-identical to the legacy node's tooltip / description text so the UI
# reads the same after migration.
_FRAME_INDEX_TOOLTIP = (
    "Which frame to pick (0-based). Use -1 for the "
    "middle frame (recommended for inflated T2I runs — "
    "the sampler converges most cleanly on the "
    "central frame, since the first frame can carry "
    "anchor artifacts and the last can be slightly "
    "blurred by motion continuity)."
)

_DESCRIPTION = (
    "Picks one frame from a video batch. Companion to "
    "ULSWanFrameInflate. -1 picks the middle frame."
)


class ULSImagePickFrameV3(io.ComfyNode):
    """
    V3 form of ULSImagePickFrame: pick a single frame from a decoded image
    batch. Companion to ULSWanFrameInflate. Stateless classmethods only — no
    __init__, no instance state (V3 sanitizes the class before execution).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ULSImagePickFrame",          # identical to the legacy key
            display_name="\u2b21 Polyhedron Pick Frame",
            category="Polyhedron/Wan",
            description=_DESCRIPTION,
            inputs=[
                io.Image.Input("images"),
                io.Int.Input(
                    "frame_index",
                    default=-1, min=-1, max=1000, step=1,
                    tooltip=_FRAME_INDEX_TOOLTIP,
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, images, frame_index) -> io.NodeOutput:
        # Verbatim port of the legacy ULSImagePickFrame.pick() logic.
        n = images.shape[0]
        if n == 0:
            print("[ULSImagePickFrame] \u26a0 empty image batch — passing through")
            return io.NodeOutput(images)

        if frame_index == -1:
            # Middle frame: integer middle for odd, lower-middle for even
            idx = n // 2
        else:
            idx = max(0, min(frame_index, n - 1))

        picked = images[idx:idx + 1]  # keep batch dimension
        print(
            f"[ULSImagePickFrame] \u2713 picked frame {idx} of {n}  "
            f"({'middle' if frame_index == -1 else 'explicit'})"
        )
        return io.NodeOutput(picked)
