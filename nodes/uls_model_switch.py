"""
Polyhedron Model Switch
═══════════════════════════════════════
Select one model filename from up to 6 slots and pass it to any
COMBO-type Loader input (WanVideoModelLoader, UNETLoader, GGUFLoader, …).

Design notes:
  - Uses AnyType trick (Impact Pack / rgthree pattern) so the output
    connects to any COMBO input regardless of list mismatch.
  - All 6 slots are COMBO inputs drawn from folder_paths (unet,
    diffusion_models, checkpoints, unet_gguf, diffusion_models_gguf).
  - Placeholder "— select model —" is always first in the list →
    ComfyUI picks it as the default for freshly placed nodes → no
    model is pre-selected, no "Missing Models" validation error.
  - Empty slots (Placeholder selected) return None → skipped cleanly.
"""

import folder_paths


# ─── AnyType — connects to any COMBO input ────────────────────────────────
# Standard pattern from Impact Pack and rgthree.
# __ne__ returning False means ComfyUI's type-compatibility check always
# passes when comparing this type against any other type.

class _AnyType(str):
    def __ne__(self, other):
        return False

_any = _AnyType("*")

_PLACEHOLDER = "— select model —"


def _model_list():
    """Return merged list of all diffusion model filenames ComfyUI knows about.

    Covers:
      - Standard safetensors loaders  : unet, diffusion_models, checkpoints
      - GGUF plugin (comfyui-gguf)    : unet_gguf, diffusion_models_gguf
      - KJ / WanVideoWrapper extras   : any other folder that exists
    """
    names = set()
    for folder in ("unet", "diffusion_models", "checkpoints",
                   "unet_gguf", "diffusion_models_gguf"):
        try:
            names.update(folder_paths.get_filename_list(folder))
        except Exception:
            pass
    # Placeholder is first → ComfyUI picks it as the default for new nodes.
    return [_PLACEHOLDER] + sorted(names)


# ─── Node ─────────────────────────────────────────────────────────────────

class ULSModelSwitch:
    """
    ⬡ Polyhedron Select Model Switch

    Pick one of up to 6 model filenames. All slots show a dropdown of
    all known model files (unet, diffusion_models, checkpoints,
    unet_gguf, diffusion_models_gguf). The selected slot's filename is
    forwarded to any Loader node's COMBO input via the AnyType trick.
    """

    @classmethod
    def INPUT_TYPES(cls):
        models = _model_list()
        return {
            "required": {
                "select": ("INT", {
                    "default": 1, "min": 1, "max": 6, "step": 1,
                    "tooltip": "Which slot to use (1–6).",
                }),
            },
            "optional": {
                "model_1": (models, {"tooltip": "Slot 1 — choose from known model files."}),
                "model_2": (models, {"tooltip": "Slot 2 — choose from known model files."}),
                "model_3": (models, {"tooltip": "Slot 3 — choose from known model files."}),
                "model_4": (models, {"tooltip": "Slot 4 — choose from known model files."}),
                "model_5": (models, {"tooltip": "Slot 5 — choose from known model files."}),
                "model_6": (models, {"tooltip": "Slot 6 — choose from known model files."}),
            },
        }

    RETURN_TYPES = (_any,)
    RETURN_NAMES = ("model_name",)
    FUNCTION = "select_model"
    CATEGORY = "Polyhedron/Loaders"
    DESCRIPTION = ("Six model filenames and a selector. The pick is forwarded as a name, "
                   "not as a loaded model, so nothing is read from disk until a loader "
                   "downstream asks for it - swapping models costs a dropdown, not a "
                   "reload. Every slot lists all known model folders in one dropdown, GGUF "
                   "included.")

    def select_model(self, select, model_1=None, model_2=None,
                     model_3=None, model_4=None,
                     model_5=None, model_6=None):

        slots = {1: model_1, 2: model_2, 3: model_3, 4: model_4,
                 5: model_5, 6: model_6}
        value = slots.get(select)

        # Treat placeholder string and None as "empty"
        if value is None or str(value).strip() == _PLACEHOLDER:
            print(
                f"[PLS] ⚠ ModelSwitch: slot {select} is empty — "
                "connect a model or change the selection."
            )
            return (None,)

        print(f"[PLS] ModelSwitch: slot {select} → {value}")
        return (value,)
