"""
ULSSigmaList -- V3 schema wrapper (v971).

Same contract as the legacy class in wan_sigma_schedule.py; the execute path
delegates there so the two cannot drift. Registered only when comfy_api is
available; otherwise __init__.py falls back to the legacy class.
"""

from comfy_api.latest import io

from .wan_sigma_schedule import SIGMA_PRESET_NAMES as _PRESET_NAMES


class ULSSigmaListV3(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ULSSigmaList",
            display_name="⬡ Polyhedron Sigma List",
            category="Polyhedron/Sigma",
            description=(
                "An explicit sigma grid, typed in rather than computed from a "
                "curve shape. Distilled and few-step recipes ship a TRAINED "
                "grid - a fixed list of numbers - instead of a curve; paste it "
                "here and wire SIGMAS straight into the sampler, in place of a "
                "scheduler. Separators are free: commas, spaces or newlines."
            ),
            inputs=[
                io.String.Input(
                    "sigmas_text",
                    default="1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.2, 0.0",
                    multiline=True,
                    tooltip="The sigma grid, highest first, strictly decreasing. "
                            "Commas, spaces or newlines all separate."),
                io.Float.Input(
                    "shift", default=1.0, min=0.0001, max=1000.0, step=0.01,
                    tooltip="Flow-matching exponential shift. 1.0 = off (paste an "
                            "already-shifted grid). MiniMax-H3 video shift is 12.0."),
                io.Boolean.Input(
                    "enforce_terminal_zero", default=True,
                    tooltip="Append a final 0.0 when the list does not end at zero. "
                            "Samplers expect the grid to reach zero."),
                # v972: APPENDED, never inserted -- mirrors the legacy order.
                io.Combo.Input(
                    "preset", options=_PRESET_NAMES, default="custom",
                    tooltip="A published grid, typed once and kept here. Anything but "
                            "'custom' WINS over the text field -- the console says which "
                            "source was used. Each preset's note names the shift it "
                            "expects; shift itself stays yours to set."),
            ],
            outputs=[
                io.Custom("SIGMAS").Output(display_name="sigmas"),
                io.Int.Output(display_name="steps"),
            ],
        )

    @classmethod
    def execute(cls, sigmas_text, shift, enforce_terminal_zero, preset="custom") -> io.NodeOutput:
        from .wan_sigma_schedule import ULSSigmaList
        out = ULSSigmaList().compute(
            sigmas_text=sigmas_text, shift=shift,
            enforce_terminal_zero=enforce_terminal_zero, preset=preset,
        )
        return io.NodeOutput(*out)
