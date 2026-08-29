"""
Polyhedron Load Upscale Model  (ULSLoadUpscaleModel)
════════════════════════════════════════════════════
v886 -- the last core loader in Frank's Power Upscale graph, replaced.

A DROP-IN for core's `UpscaleModelLoader`: same folder (models/upscale_models),
same UPSCALE_MODEL output, same spandrel descriptor, same quirks honoured
(the `module.` prefix of some SwinIR exports, safe_load, .eval()). What core
does, this does -- MEASURED against comfy_extras/nodes_upscale_model.py on the
release tags v0.30.0 / v0.31.0 / v0.32.0, which are identical in this file.

WHAT IT ADDS (and nothing else):

  1. THE SIZE IN THE LIST. Every Polyhedron loader shows it (v828); this one
     was the last that did not. `ESRGAN_4x.pth \u00b7 64 MB`.

  2. AN `info` READOUT, and one console line. v580 recorded Frank's own
     question after three clean runs -- "generell frage ich mich, ob die
     externen Upscale-Modelle ueberhaupt greifen". They did; nothing SAID so.
     Core hands on a bare spandrel descriptor with no filename in it, so the
     Power Upscale had to fingerprint the object to answer that question late,
     in the middle of a run. Here the answer is available at LOAD time, next
     to the filename that produced it: architecture, factor, parameters,
     precision.

  3. A FAILURE THAT NAMES THE FILE. Core raises "Upscale model must be a
     single-image model." -- true, and useless with two loaders on the canvas.

DELIBERATELY NOT ADDED, so the reasons survive the next reader:

  * No precision / dtype widget. `_esrgan_pass` in ph_power_upscale already
    decides device and half-precision from the descriptor's own
    `supports_half`. A second place deciding dtype is the drift this house
    keeps paying for; one owner or none.
  * No scale output. The consumers read `scale` off the descriptor themselves
    (that is what an UPSCALE_MODEL wire IS); a FLOAT socket carrying the same
    number invites a graph where the two disagree.
  * No download door. The pack's one door is ph_weights.ensure_weights, and
    upscale models have no pinned source here.
"""

import os

import folder_paths

try:  # package load (ComfyUI) vs direct module load (tools/tests)
    from .ph_basics import _sized_list, _strip_size
except ImportError:  # pragma: no cover
    from ph_basics import _sized_list, _strip_size


_FOLDER = "upscale_models"


def _describe(um, name):
    """The model card for the info output.

    ONE SOURCE: ph_power_upscale._model_card already turns a spandrel
    descriptor into 'ESRGAN x4 \u00b7 16.7M params \u00b7 fp16 ok'. It is imported, not
    copied -- two places computing the same sentence drift, and this house has
    the scars to prove it (pmHideWidget, v755). Imported LAZILY so a problem in
    the big node can never stop this small one from loading a file.
    """
    try:
        try:
            from .ph_power_upscale import _model_card
        except ImportError:  # pragma: no cover
            from ph_power_upscale import _model_card
        card = _model_card(um)
    except Exception as exc:
        # A SPOKEN fallback. The first draft of this returned the bare class
        # name silently -- and a silent degradation is the one failure mode
        # this house keeps paying for (v552, v885). The model still loads; the
        # readout is just poorer, and now it says why.
        card = type(um).__name__
        print("[PLS] Load Upscale Model: the detailed card is unavailable "
              "(%s: %s) - falling back to the bare type. The model itself "
              "loaded fine." % (type(exc).__name__, exc))
    size = ""
    try:
        path = folder_paths.get_full_path(_FOLDER, name)
        if path:
            size = " | %.0f MB" % (os.path.getsize(path) / float(1 << 20))
    except Exception:
        pass
    return "%s | %s%s" % (name, card, size)


class ULSLoadUpscaleModel:
    """\u2b21 Polyhedron Load Upscale Model -- core's loader, with the size in the
    list and a readout of what was actually loaded."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_sized_list(
                    folder_paths.get_filename_list(_FOLDER), (_FOLDER,)), {
                    "tooltip": "Pixel upscale model (models/upscale_models). "
                               "The size is shown so a 4x ESRGAN is not "
                               "confused with a small Compact model at a "
                               "glance; the factor itself is on the info "
                               "output once it is loaded.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **_kw):
        """v828 law, inherited: the combo DECORATES its values with a size, so
        any value that reaches this node from an older save (or from a file
        whose size changed on disk since the workflow was saved) is a string
        the list no longer offers. Without this, core's server-side membership
        check kills the run at the door. load() strips and resolves; a file
        that truly is not there still fails there, loudly and by name."""
        return True

    RETURN_TYPES = ("UPSCALE_MODEL", "STRING")
    RETURN_NAMES = ("upscale_model", "info")
    OUTPUT_TOOLTIPS = ("The loaded upscale model (spandrel descriptor -- the "
                       "exact type core's loader hands over).",
                       "Readout: file | architecture xN | parameters | "
                       "precision | size.")
    FUNCTION = "load"
    CATEGORY = "Polyhedron/Loaders"
    DESCRIPTION = ("Loads a pixel upscale model and says what it loaded. Core's "
                   "loader passes on a bare descriptor with no filename in it, "
                   "so with two upscalers on the canvas nothing tells you which "
                   "one is which until a run is already under way.")

    def load(self, model_name):
        model_name = _strip_size(model_name)   # v828: sized list entries

        # comfy is imported HERE, not at module scope: this file must be
        # importable by the guards without a ComfyUI around it.
        import comfy.model_management
        import comfy.model_patcher
        import comfy.utils

        model_path = folder_paths.get_full_path_or_raise(_FOLDER, model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        # Core's own quirk, carried over verbatim: some SwinIR exports carry a
        # 'module.' prefix from DataParallel training.
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})

        # spandrel is imported lazily as well. Core's nodes_upscale_model.py
        # imports it at module scope AND extends MAIN_REGISTRY with
        # spandrel_extra_arches there; that file is core, it is always loaded,
        # and it is loaded long before any node executes -- so the registry is
        # already complete by the time we get here. We deliberately do NOT add
        # to the registry a second time.
        from spandrel import ImageModelDescriptor, ModelLoader

        out = ModelLoader().load_from_state_dict(sd).eval()
        if not isinstance(out, ImageModelDescriptor):
            raise ValueError(
                "[PLS] Load Upscale Model: '%s' is not a single-image upscale "
                "model (spandrel read it as %s). Core's message for this names "
                "no file, which is no help with two loaders on the canvas -- "
                "this one does." % (model_name, type(out).__name__))

        patcher_cls = getattr(comfy.model_patcher, "CoreModelPatcher", None)
        if patcher_cls is None:   # older core generations
            patcher_cls = comfy.model_patcher.ModelPatcher
        out.patcher = patcher_cls(
            out.model,
            load_device=comfy.model_management.get_torch_device(),
            offload_device=comfy.model_management.unet_offload_device())

        info = _describe(out, model_name)
        print("[PLS] Load Upscale Model: " + info)
        return (out, info)
