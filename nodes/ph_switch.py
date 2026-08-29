"""ph_switch.py -- v537

⬡ Polyhedron Switch          (N -> 1, any-type, truly lazy)
⬡ Polyhedron Switch Inverse  (1 -> N, any-type, cold non-selected branches)

Design goals (v537):
  * Fully dynamic slots, NO hard maximum. The frontend (web/js/ph_switch.js)
    grows one spare slot per connection; `select` is UI-capped at the number
    of connected slots.
  * Only the selected branch ever runs. Forward: lazy inputs +
    check_lazy_status. Inverse: non-selected outputs emit ExecutionBlocker,
    so everything downstream of them stays cold.
  * Bypass / mute NEVER errors the graph:
      - Forward `on_missing`: "use next active" (default) falls back to the
        next connected input; "block (skip)" silently blocks all outputs.
      - Inverse declares `input` as OPTIONAL: if the feeding branch was
        bypassed/muted away, every output blocks silently instead of failing
        prompt validation.
    In no case does a missing input forward a bare None downstream.

Mechanism credits (measured against the shipped sources, 2026-07-11):
  * dynamic lazy any-inputs via an INPUT_TYPES "AllContainer" behind an
    inspect-stack `get_input_info` gate -- pattern established by
    ComfyUI-Impact-Pack (modules/impact/util_nodes.GeneralSwitch).
  * unbounded outputs via a clamped RETURN_TYPES tuple plus a hidden-PROMPT
    consumer scan -- pattern established by Impact's GeneralInversedSwitch.
  * lazy inputs / check_lazy_status and comfy_execution.graph.ExecutionBlocker
    are core ComfyUI execution-model features (docs.comfy.org, Lazy Evaluation).
"""

import inspect
import logging

try:
    from comfy_execution.graph import ExecutionBlocker
except Exception:  # pragma: no cover -- ancient core; degrade loudly, once.
    ExecutionBlocker = None
    logging.warning(
        "[PLS] ph_switch: comfy_execution.graph.ExecutionBlocker unavailable "
        "-- ComfyUI is outdated; blocked branches will receive None instead."
    )


# --- AnyType: connects to any socket (house pattern, see uls_model_switch) ---

class _AnyType(str):
    def __ne__(self, other):
        return False


_any = _AnyType("*")


class _FlexOutTuple(tuple):
    """RETURN_TYPES helper: any output index resolves to the wildcard entry.

    The backend indexes RETURN_TYPES with the consumed output slot; clamping
    lets a statically declared 1-tuple serve an unbounded, frontend-grown
    output list (Impact's ByPassTypeTuple approach).
    """

    def __getitem__(self, index):
        if isinstance(index, int) and (index >= len(self) or index < -len(self)):
            index = len(self) - 1
        return super().__getitem__(index)


def _blocked():
    return ExecutionBlocker(None) if ExecutionBlocker is not None else None


# Slot-name prefixes. MUST stay identical to web/js/ph_switch.js (parity guard
# in tests/test_v537_switch.py).
_IN = "input_"
_OUT = "out_"

_UI_KEY = "pls_switch"  # ui channel, rendered by web/js/ph_switch.js


def _connected_inputs(kwargs):
    """Slot indices present in kwargs = connected in the prompt, sorted.

    A branch that was muted (or bypassed with no pass-through) simply has no
    key in the prompt, so absence here IS the 'missing input' signal.
    """
    found = []
    for key in kwargs:
        if key.startswith(_IN):
            try:
                found.append(int(key[len(_IN):]))
            except ValueError:
                pass
    return sorted(found)


def _pick(select, connected, use_fallback):
    """The one selection rule, shared by check_lazy_status and route()."""
    if select in connected:
        return select
    if not use_fallback or not connected:
        return None
    higher = [i for i in connected if i > select]
    return higher[0] if higher else connected[0]


# --- ⬡ Polyhedron Switch (N -> 1) -------------------------------------------

class ULSAnySwitch:
    """Route ONE of N any-type inputs to the output; the rest never run."""

    @classmethod
    def INPUT_TYPES(cls):
        dyn_inputs = {
            _IN + "1": (_any, {"lazy": True,
                               "tooltip": "Any input. Connect it and a fresh spare slot appears."}),
        }
        # Dynamic-input trick (Impact pattern): when the executor asks for
        # input info, answer 'yes, and it is a lazy any' for EVERY name, so
        # frontend-grown input_2..input_N validate and stay lazy.
        stack = inspect.stack()
        if len(stack) > 2 and stack[2].function == "get_input_info":
            class _AllLazyAny:
                def __contains__(self, item):
                    return True

                def __getitem__(self, key):
                    return _any, {"lazy": True}

            dyn_inputs = _AllLazyAny()

        return {
            "required": {
                "select": ("INT", {
                    "default": 1, "min": 1, "max": 999, "step": 1,
                    "tooltip": "Which input to route. The UI caps this at the "
                               "number of connected inputs.",
                }),
                "on_missing": ("BOOLEAN", {
                    "default": True,
                    "label_on": "use next active",
                    "label_off": "block (skip)",
                    "tooltip": "Selected input gone (branch bypassed/muted)? "
                               "ON: fall back to the next connected input. "
                               "OFF: emit nothing -- downstream stays cold. "
                               "Never errors either way.",
                }),
            },
            "optional": dyn_inputs,
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = (_any, "INT", "STRING")
    RETURN_NAMES = ("value", "index", "label")
    OUTPUT_TOOLTIPS = (
        "The selected input, passed through untouched.",
        "The input index that was actually used (after any fallback).",
        "Slot label of the used input (honors custom labels).",
    )
    FUNCTION = "route"
    CATEGORY = "Polyhedron/Logic"
    DESCRIPTION = ("Routes one of several any-type inputs to the output and leaves the rest "
                   "cold: the branches you did not select never execute, so an expensive "
                   "path can sit in the graph costing nothing until you pick it. Reports "
                   "the index and the label it chose.")

    # Called before evaluation: request ONLY the branch we will actually use.
    def check_lazy_status(self, select, on_missing=True, **kwargs):
        connected = _connected_inputs(kwargs)
        pick = _pick(int(select), connected, bool(on_missing))
        if pick is None:
            return []
        name = _IN + str(pick)
        if kwargs.get(name) is None:
            return [name]
        return []

    def route(self, select, on_missing=True, unique_id=None,
              extra_pnginfo=None, **kwargs):
        select = int(select)
        connected = _connected_inputs(kwargs)
        pick = _pick(select, connected, bool(on_missing))
        value = kwargs.get(_IN + str(pick)) if pick is not None else None

        if value is None:
            # Missing selection and either fallback is off, nothing is
            # connected, or the branch evaluated to nothing: block silently.
            blk = _blocked()
            if not connected:
                note = "no inputs connected"
            elif pick is None:
                note = f"{_IN}{select} missing (branch bypassed?) -- blocked"
            else:
                note = f"{_IN}{pick} yielded nothing -- blocked"
            text = "\u26d4 " + note
            logging.info(f"[PLS] Switch: {text}")
            return {"ui": {_UI_KEY: [text]}, "result": (blk, blk, blk)}

        label = _IN + str(pick)
        if extra_pnginfo:  # honor custom slot labels (Impact-measured pattern)
            try:
                for node in extra_pnginfo["workflow"]["nodes"]:
                    if str(node.get("id")) == str(unique_id):
                        for slot in node.get("inputs", []):
                            if slot.get("name") == label and slot.get("label"):
                                label = slot["label"]
                        break
            except Exception:
                pass

        fb = "" if pick == select else f" (fallback from {select})"
        text = f"\u25b6 {label}{fb} \u00b7 {len(connected)} connected"
        return {"ui": {_UI_KEY: [text]}, "result": (value, pick, label)}


# --- ⬡ Polyhedron Switch Inverse (1 -> N) -----------------------------------

class ULSAnySwitchInv:
    """Route the input to ONE of N outputs; every other branch stays cold."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": ("INT", {
                    "default": 1, "min": 1, "max": 999, "step": 1,
                    "tooltip": "Which output receives the input. The UI caps "
                               "this at the number of connected outputs.",
                }),
            },
            "optional": {
                # OPTIONAL on purpose: if the feeding branch is bypassed or
                # muted, the prompt simply omits this key and route() blocks
                # every output -- no validation error, no crash.
                "input": (_any, {"lazy": True,
                                 "tooltip": "Any value. If its branch is bypassed/"
                                            "muted, all outputs block silently."}),
            },
            "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = _FlexOutTuple((_any,))
    RETURN_NAMES = (_OUT + "1",)
    OUTPUT_TOOLTIPS = ("Only the selected output emits; the rest carry an "
                       "ExecutionBlocker so their branches never run.",)
    FUNCTION = "route"
    CATEGORY = "Polyhedron/Logic"
    DESCRIPTION = ("The mirror of the switch: one input, several outputs, and only the "
                   "selected branch runs. Everything hanging off the other outputs stays "
                   "cold.")

    @staticmethod
    def _consumed_outputs(prompt, unique_id):
        """Highest output index any prompt node consumes from us (-1 = none)."""
        cnt = -1
        uid = str(unique_id)
        if prompt:
            for node in prompt.values():
                for value in node.get("inputs", {}).values():
                    if (isinstance(value, list) and len(value) == 2
                            and str(value[0]) == uid):
                        cnt = max(cnt, int(value[1]))
        return cnt

    # If nothing consumes us, do not even evaluate the input branch.
    def check_lazy_status(self, select, prompt=None, unique_id=None, **kwargs):
        if self._consumed_outputs(prompt, unique_id) < 0:
            return []
        if "input" in kwargs and kwargs.get("input") is None:
            return ["input"]
        return []

    def route(self, select, prompt=None, unique_id=None, input=None):  # noqa: A002
        n_out = self._consumed_outputs(prompt, unique_id) + 1
        if n_out <= 0:
            return {"ui": {_UI_KEY: ["no outputs connected"]},
                    "result": (_blocked(),)}

        sel = int(select)
        result = []
        for i in range(n_out):
            if input is not None and sel == i + 1:
                result.append(input)
            else:
                result.append(_blocked())

        if input is None:
            text = "\u26d4 input missing (branch bypassed?) -- all outputs blocked"
        elif not (1 <= sel <= n_out):
            text = f"\u26d4 select={sel} but only {n_out} connected -- all outputs blocked"
        else:
            text = f"\u25b6 {_OUT}{sel} \u00b7 feeding 1 of {n_out}"
        logging.info(f"[PLS] SwitchInv: {text}")
        return {"ui": {_UI_KEY: [text]}, "result": tuple(result)}
