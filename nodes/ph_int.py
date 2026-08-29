"""ph_int.py -- v878

⬡ Polyhedron Int  (ULSInt)

Replaces core's `PrimitiveInt` ("Int") wherever a graph needs a hand-held
number. Measured against the core source (comfy_extras/nodes_primitive.py,
2026-08-26): core declares ONE input, `value`, with
`control_after_generate=fixed`, and returns it unchanged. The second row a
user sees ("control after generate") is the frontend's doing, not the node's.

WHY A REPLACEMENT AT ALL. Three of these sat in the MiniMax graph:

    Int (Full)                          value 20      -> sampler steps
    Int (Lightning LoRA)                value 4       -> sampler steps
    Route  1 = Basis / 2 = Turbo-LoRA   value 1       -> switch select

They are one decision held in three places. The names live in the node TITLE,
which nothing reads, and the meaning of "1" is a note to self. Switching from
base to turbo means editing two numbers and remembering which is which.

WHAT THIS NODE ADDS

  * PRESETS: named values carried IN the node (`preset_config`, a hidden text
    widget in the manner of the engine/stack nodes). One click sets the value
    and the label. "Basis = 20", "Turbo = 4" stop being tribal knowledge.
  * A SECOND OUTPUT, `label`: the name of the active preset as a STRING. The
    number can now say what it means downstream -- into a filename prefix, a
    note, a text field. Core's Int cannot, because it has nothing to say.

The canon is `value` first: the INT output is always `value`, whether it came
from a click, the arrows, a drag or the keyboard. A preset is a way to WRITE
the value, never a second source of truth -- two places that decide the same
number would drift (the house lesson).

`label` resolves by VALUE, not by which chip was clicked last: the label is
the name of the preset whose value equals `value` right now, or "" if none
matches. That way a hand-typed 20 reports "Basis" just like the click does,
and a hand-typed 7 honestly reports nothing.
"""

import json
import logging

# The widest range core's own Int accepts is sys.maxsize; ComfyUI serialises
# widget values through JSON, and the frontend's number handling is IEEE-754.
# 2**31-1 is the largest value that survives every step of that chain intact
# and is far beyond any step count, index or seed offset this node feeds.
INT_MAX = 2147483647
INT_MIN = -2147483648


def parse_presets(raw):
    """Read the preset list out of the hidden config string.

    Returns a list of {"name": str, "value": int}. NEVER raises: a damaged
    config must not take a run down with it -- it degrades to "no presets",
    which is exactly core's Int and still emits `value` correctly.
    """
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        logging.warning("[PLS] Int: preset_config is not valid JSON -- "
                        "presets ignored, the value still stands.")
        return []
    rows = data.get("presets") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        return []
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            v = int(r.get("value"))
        except (TypeError, ValueError):
            continue
        name = r.get("name")
        name = "" if name is None else str(name)
        out.append({"name": name, "value": max(INT_MIN, min(INT_MAX, v))})
    return out


def label_for(value, presets):
    """The name of the preset that currently HOLDS this value, else "".

    First match wins, so a duplicated value reports the topmost chip -- the
    one the eye reads first.
    """
    for p in presets:
        if p["value"] == value:
            return p["name"]
    return ""


class ULSInt:
    """A hand-held integer that knows what its numbers mean."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {
                    "default": 0, "min": INT_MIN, "max": INT_MAX, "step": 1,
                    "tooltip": "The number this node emits. Presets write "
                               "here; they are never a second source.",
                }),
            },
            "optional": {
                # Hidden, frontend-owned. Appended LAST and it stays last:
                # LiteGraph restores widget values POSITIONALLY (guard #577).
                "preset_config": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Frontend-owned preset list (JSON). Not meant "
                               "for hand editing.",
                }),
            },
        }

    RETURN_TYPES = ("INT", "STRING")
    RETURN_NAMES = ("value", "label")
    OUTPUT_TOOLTIPS = (
        "The integer, exactly as shown.",
        "Name of the preset holding this value right now, or empty.",
    )
    FUNCTION = "emit"
    CATEGORY = "Polyhedron/Logic"
    DESCRIPTION = ("An integer with named presets. One click sets the number "
                   "AND says what it means; the label travels downstream as "
                   "a string.")

    def emit(self, value=0, preset_config=""):
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = 0
        value = max(INT_MIN, min(INT_MAX, value))
        label = label_for(value, parse_presets(preset_config))
        return (value, label)
