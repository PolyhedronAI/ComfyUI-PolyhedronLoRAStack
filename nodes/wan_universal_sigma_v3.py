"""
Polyhedron Sigma Curve (ULSUniversalSigmaCurve) — V3 schema edition.

V3 (Nodes 2.0) form: a single named sigma schedule -> one SIGMAS output (steps
also passed through for downstream sync).

Stage 3 of the migration. Custom type SIGMAS (out) via io.Custom("SIGMAS"); the
schedule dropdown is io.Combo over SIGMA_SCHEDULE_NAMES, imported from the legacy
module. Standard INT/FLOAT for the rest.

Collected by the central registry; node_id identical to the legacy key; legacy/V3
mutually exclusive via _V3_OK. execute() DELEGATES to the proven legacy
ULSUniversalSigmaCurve.compute() — identical behaviour, single source of truth,
kept byte-identical as the fallback. Heavy compute import is lazy.
"""

try:  # v577: ONE door to the versioned API (nodes/ph_comfyapi.py).
    from .ph_comfyapi import io
except ImportError:  # pragma: no cover - direct module load (tools)
    from ph_comfyapi import io

from .wan_sigma_schedule import SIGMA_SCHEDULE_NAMES


class ULSUniversalSigmaCurveV3(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ULSUniversalSigmaCurve",
            display_name="\u2b21 Polyhedron Sigma Curve",
            category="Polyhedron/Sigma",
            description=(
                "One named sigma schedule, emitted as a SIGMAS curve the sampler "
                "can take directly. The schedule decides how the noise level "
                "falls across the steps - the same step count under a different "
                "curve spends its budget somewhere else entirely, which is why "
                "this is a wire and not a hidden default."
            ),
            inputs=[
                io.Combo.Input("sigma_schedule", options=SIGMA_SCHEDULE_NAMES,
                               default="karras",
                               tooltip="Sigma curve shape — affects how steps are distributed across the noise range"),
                io.Int.Input("steps", default=20, min=1, max=300,
                             tooltip="Number of steps. Also passed through as output for downstream sync."),
                io.Float.Input("sigma_max", default=1.0, min=0.0001, max=1000.0, step=0.001,
                               tooltip="Flow-matching (WAN/FLUX/SD3): 1.0 — k-diffusion (SDXL/SD1.5): 14.61"),
                io.Float.Input("sigma_min", default=0.002, min=0.00001, max=100.0, step=0.0001,
                               tooltip="Flow-matching (WAN/FLUX/SD3): 0.002 — k-diffusion (SDXL/SD1.5): 0.029"),
                io.Float.Input("rho", default=7.0, min=0.1, max=20.0, step=0.1,
                               tooltip="Shape param — only affects karras, exponential, laplace"),
            ],
            outputs=[
                io.Custom("SIGMAS").Output(display_name="sigmas"),
            ],
        )

    @classmethod
    def execute(cls, sigma_schedule, steps, sigma_max, sigma_min, rho) -> io.NodeOutput:
        from .wan_sigma_schedule import ULSUniversalSigmaCurve
        out = ULSUniversalSigmaCurve().compute(
            sigma_schedule=sigma_schedule, steps=steps,
            sigma_max=sigma_max, sigma_min=sigma_min, rho=rho,
        )
        return io.NodeOutput(*out)
