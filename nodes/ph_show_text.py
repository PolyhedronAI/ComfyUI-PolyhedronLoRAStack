# -*- coding: ascii -*-
"""
ph_show_text.py -- the Polyhedron Show Text.

WHY: the suite's reports (Merge Analyzer, LoRA Inspector, the Stack's
debug_info) are STRING outputs, and until now they had to be wired into a
third-party node (comfyui-custom-scripts' "Show Text") to be read. The suite
ships the node its own reports need.

WHAT IT DOES: shows the text it receives, in a read-only monospace field that
keeps tables aligned (no wrapping -- long lines scroll sideways), and passes
the text on unchanged. The field grows with the number of lines up to a cap,
then scrolls. The last text is kept in the node's properties so a saved
workflow still shows it after a reload. The colour row at the bottom is the
Polyhedron Note's own (one implementation, imported by the view half).

THE CLASSIC SHAPE, on purpose: text arrives as a list (INPUT_IS_LIST), so a
batch of strings shows as one field with a rule between the items, and goes
out as the same list (OUTPUT_IS_LIST). OUTPUT_NODE, so it runs even when
nothing is wired to its output -- a display that only runs when something
downstream wants it would show nothing.

No widgets in INPUT_TYPES: the text is a socket (forceInput), and the display
field is created by the frontend with serialize:false. widgets_values stays
empty, so nothing can ever shift in a saved workflow (guard #577).
"""

SEPARATOR = "\n" + "-" * 40 + "\n"


class ULSShowText:
    """Shows a STRING and passes it on."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Any text -- a report, a prompt, a caption. "
                               "It is shown as it is and passed on unchanged.",
                }),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_TOOLTIPS = ("The text it received, unchanged.",)
    OUTPUT_NODE = True
    FUNCTION = "show"
    CATEGORY = "Polyhedron/Logic"
    DESCRIPTION = (
        "Shows the text it receives in a read-only monospace field that keeps "
        "tables aligned, and passes it on unchanged. The field grows with the "
        "text and then scrolls; the last text stays visible after a reload. "
        "The row at the bottom sets the node colour, as on the Polyhedron Note."
    )

    def show(self, text):
        items = ["" if t is None else str(t) for t in (text or [])]
        return {"ui": {"text": [SEPARATOR.join(items)]}, "result": (items,)}
