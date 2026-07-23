"""
Polyhedron Dual Sigma Curve (ULSWanSplitNoiseSchedule) — V3 schema edition.

V3 (Nodes 2.0) form: builds a split HIGH/LOW sigma schedule for the WAN dual-pass
pipeline -> two SIGMAS outputs.

Stage 3 of the migration. Custom type SIGMAS (out) via io.Custom("SIGMAS"); the
schedule dropdowns are io.Combo over SIGMA_SCHEDULE_NAMES, imported from the legacy
module (the single source of the schedule table). Standard INT/FLOAT for the rest.

Collected by the central registry; node_id identical to the legacy key; legacy/V3
mutually exclusive via _V3_OK. execute() DELEGATES to the proven legacy
ULSWanSplitNoiseSchedule.compute() — identical behaviour, single source of truth,
kept byte-identical as the fallback. Heavy compute import is lazy.
"""

try:  # v577: ONE door to the versioned API (nodes/ph_comfyapi.py).
    from .ph_comfyapi import io
except ImportError:  # pragma: no cover - direct module load (tools)
    from ph_comfyapi import io

from .wan_sigma_schedule import SIGMA_SCHEDULE_NAMES


class ULSWanSplitNoiseScheduleV3(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ULSWanSplitNoiseSchedule",
            display_name="\u2b21 Polyhedron Dual Sigma Curve",
            category="Polyhedron/Sigma",
            description=(
                "Splits one noise schedule into a HIGH and a LOW half at the "
                "expert boundary, giving each pass of a dual-expert (MoE) run "
                "its own curve. The two experts are not doing the same job - one "
                "sets structure, the other finishes it - so handing both the same "
                "schedule wastes steps at one end and starves the other."
            ),
            inputs=[
                io.Combo.Input("schedule_high", options=SIGMA_SCHEDULE_NAMES,
                               default="karras",
                               tooltip="Sigma curve for HIGH pass (structure phase)"),
                io.Combo.Input("schedule_low", options=SIGMA_SCHEDULE_NAMES,
                               default="bong_tangent",
                               tooltip="Sigma curve for LOW pass (detail phase)"),
                io.Int.Input("total_steps", default=20, min=2, max=300,
                             tooltip="Total steps across both passes"),
                io.Int.Input("split_step", default=8, min=1, max=299,
                             tooltip="Where HIGH ends and LOW begins"),
                io.Float.Input("sigma_max", default=1.0, min=0.0001, max=1000.0, step=0.001,
                               tooltip="Flow-matching (WAN/FLUX/SD3): 1.0 — k-diffusion (SDXL/SD1.5): 14.61"),
                io.Float.Input("sigma_min", default=0.002, min=0.00001, max=100.0, step=0.0001,
                               tooltip="Flow-matching (WAN/FLUX/SD3): 0.002 — k-diffusion (SDXL/SD1.5): 0.029"),
                io.Float.Input("rho_high", default=7.0, min=0.1, max=20.0, step=0.1,
                               tooltip="Shape param for HIGH schedule (karras/exponential/laplace only)"),
                io.Float.Input("rho_low", default=7.0, min=0.1, max=20.0, step=0.1,
                               tooltip="Shape param for LOW schedule (karras/exponential/laplace only)"),
            ],
            outputs=[
                io.Custom("SIGMAS").Output(display_name="sigmas_high"),
                io.Custom("SIGMAS").Output(display_name="sigmas_low"),
            ],
        )

    @classmethod
    def execute(cls, schedule_high, schedule_low, total_steps, split_step,
                sigma_max, sigma_min, rho_high, rho_low) -> io.NodeOutput:
        from .wan_sigma_schedule import ULSWanSplitNoiseSchedule
        out = ULSWanSplitNoiseSchedule().compute(
            schedule_high=schedule_high, schedule_low=schedule_low,
            total_steps=total_steps, split_step=split_step,
            sigma_max=sigma_max, sigma_min=sigma_min,
            rho_high=rho_high, rho_low=rho_low,
        )
        return io.NodeOutput(*out)
