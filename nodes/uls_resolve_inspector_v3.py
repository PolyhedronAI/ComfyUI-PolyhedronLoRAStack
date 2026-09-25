"""
Polyhedron Merge Analyzer (ULSResolveInspector) — V3 schema edition.

V3 (Nodes 2.0) form of the Merge Analyzer: reads the Stack's config output and
reports on the CONCAT/DARE/Resolve merge (Overview = instant; Deep analysis loads
the LoRAs + SVD per layer and measures Resolve fidelity). Emits a STRING report
plus FLOAT/FLOAT/BOOLEAN metrics. No re-implementation of merge math.

Stage 3 of the migration. Standard types only → Vue auto-renders the widgets.
Collected by the central registry; node_id identical to the legacy key; legacy/V3
mutually exclusive via _V3_OK. The in-UI report text stays English by design.

execute() DELEGATES to the proven legacy ULSResolveInspector.analyze() — identical
behaviour, single source of truth, kept byte-identical as the fallback. Lazy import.
"""

try:  # v577: ONE door to the versioned API (nodes/ph_comfyapi.py).
    from .ph_comfyapi import io
except ImportError:  # pragma: no cover - direct module load (tools)
    from ph_comfyapi import io
try:  # v980: pure module (no comfy) -- the depth list has one home
    from . import uls_overlap_math as _OV
except ImportError:  # pragma: no cover - direct module load (tools)
    import uls_overlap_math as _OV


class ULSResolveInspectorV3(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ULSResolveInspector",
            display_name="\u2b21 Polyhedron Merge Analyzer",
            category="Polyhedron/Utils",
            description=(
                "Analyzes the Stack's CONCAT/DARE/Resolve merge (no "
                "re-implementation). Overview = instant; Deep analysis measures "
                "Resolve fidelity via per-layer SVD; Overlap & energy measures "
                "what each LoRA adds and which pull together or apart."
            ),
            inputs=[
                io.String.Input("uls_config_out", default='{"rows":[]}',
                                multiline=False, force_input=True),
                io.Combo.Input("analysis_depth",
                               options=list(_OV.DEPTHS),
                               default="Overview",
                               tooltip=_OV.DEPTH_TIP),
                io.Int.Input("max_layers", default=24, min=1, max=200, step=1,
                             optional=True,
                             tooltip=("Deep analysis: how many of the largest conflict "
                                      "layers are fully measured (speed/memory).")),
                io.Combo.Input("device", options=["auto", "cpu"], default="auto",
                               optional=True,
                               tooltip=("auto = GPU if free (like the real Resolve path), "
                                        "else CPU. 'cpu' forces CPU.")),
                # v980: socket, appended last -- no widget slot moves.
                io.Model.Input("model", optional=True, tooltip=_OV.MODEL_TIP),
            ],
            outputs=[
                io.String.Output(display_name="report"),
                io.Float.Output(display_name="energy_1x_pct"),
                io.Float.Output(display_name="amplitude_ratio"),
                io.Boolean.Output(display_name="resolve_active"),
            ],
        )

    @classmethod
    def execute(cls, uls_config_out, analysis_depth, max_layers=24, device="auto",
                model=None) -> io.NodeOutput:
        from .uls_resolve_inspector import ULSResolveInspector
        out = ULSResolveInspector().analyze(
            uls_config_out=uls_config_out,
            analysis_depth=analysis_depth,
            max_layers=max_layers,
            device=device,
            model=model,
        )
        return io.NodeOutput(*out)
