"""
Polyhedron LoRA Inspector — V3 schema edition.

V3 (Nodes 2.0) form of ULSInspector. Passive consistency-check node: reads the
active LoRAs + trigger words from the Stack's config output and checks whether
each trigger word appears in the prompt; emits a STRING report. No model patching.

Stage 3 of the migration. Standard types only (STRING in/out) → the Vue renderer
auto-generates the widgets. Collected by the central registry
(nodes/uls_v3_extension.py); node_id kept identical to the legacy key so saved
workflows resolve unchanged; legacy/V3 are mutually exclusive via _V3_OK.

execute() DELEGATES to the proven legacy ULSInspector.inspect() — behaviour is
identical (it runs the exact shipped code). The legacy class stays the single
source of truth and is kept byte-identical as the fallback. The legacy import is
lazy (inside execute) so importing this module only needs comfy_api.
"""

try:  # v577: ONE door to the versioned API (nodes/ph_comfyapi.py).
    from .ph_comfyapi import io
except ImportError:  # pragma: no cover - direct module load (tools)
    from ph_comfyapi import io


class ULSInspectorV3(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ULSInspector",
            display_name="\u2b21 Polyhedron LoRA Inspector",
            category="Polyhedron/Utils",
            description=(
                "Passive check: does each active LoRA's trigger word appear in "
                "the prompt? Emits a STRING report. Wire into a Show Text node."
            ),
            inputs=[
                io.String.Input("uls_config_out", default='{"rows":[]}',
                                multiline=False, force_input=True),
                io.String.Input("prompt", default="", multiline=True, force_input=True),
            ],
            outputs=[
                io.String.Output(display_name="inspector_report"),
            ],
        )

    @classmethod
    def execute(cls, uls_config_out, prompt) -> io.NodeOutput:
        from .uls_stack_node import ULSInspector
        out = ULSInspector().inspect(uls_config_out=uls_config_out, prompt=prompt)
        return io.NodeOutput(*out)
