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

try:  # v902: the SAME selection logic the legacy class uses.
    from . import uls_pick_frame_core as _pf
except ImportError:  # pragma: no cover - direct module load (tools)
    import uls_pick_frame_core as _pf


# Kept byte-identical to the legacy node's tooltip / description text so the UI
# reads the same after migration. v902 changed BOTH together, on purpose:
# the -1 reading now depends on the mode, and a tooltip that still promised
# "always the middle" would be a lie in three of the six modes.
_FRAME_INDEX_TOOLTIP = (
    "Which frame to pick (0-based). In the default 'middle (legacy)' "
    "mode, -1 means the MIDDLE frame (recommended for inflated T2I runs — "
    "the sampler converges most cleanly on the "
    "central frame, since the first frame can carry "
    "anchor artifacts and the last can be slightly "
    "blurred by motion continuity). In 'index' and 'range' it is Core's "
    "convention instead: -1 is the LAST frame. The mode says which "
    "reading applies."
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
                    default=-1, min=-4096, max=4096, step=1,
                    tooltip=_FRAME_INDEX_TOOLTIP,
                ),
                # v902: APPENDED, never inserted. Same order as the legacy
                # class, because a saved graph must resolve identically
                # whichever of the two registers.
                io.Combo.Input(
                    "mode",
                    options=list(_pf.MODES),
                    default=_pf.MODES[0],
                    tooltip=_pf.MODE_TOOLTIP,
                ),
                io.Int.Input(
                    "count",
                    default=1, min=1, max=4096, step=1,
                    tooltip=_pf.COUNT_TOOLTIP,
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, images, frame_index, mode=None, count=1) -> io.NodeOutput:
        # v902: delegates to nodes/uls_pick_frame_core.py, the one place the
        # rule lives. This class and the legacy one share a node id and are
        # mutually exclusive, so a rule kept in both would only ever be
        # wrong on machines running the other -- the v898 failure shape.
        if mode is None:
            mode = _pf.MODES[0]
        picked, note = _pf.select(images, mode, frame_index, count)
        print(f"[ULSImagePickFrame] ✓ {note}")
        return io.NodeOutput(picked)
