"""ph_basics.py -- v542

The 'Basics' corner of the node set:
  ⬡ Polyhedron Seed        (v538)
  ⬡ Polyhedron Load CLIP   (v539)
  ⬡ Polyhedron Load VAE    (v539)
  ⬡ Polyhedron Load Model  (v541)  -- switch + loader fused, GGUF-aware

Load VAE design notes (research 2026-07-11, REVIEW_v537 §0):
  * Returns the NATIVE ComfyUI `VAE` type -- fits every sampler/decoder in
    every model family. Deliberately NO wrapper-only "WANVAE" clone.
  * `precision` override (auto/bf16/fp16/fp32) -- the genuinely useful part
    of the Kijai loader, applied to the native VAE object.
  * Family readout from `vae.latent_channels` PLUS an optional `model`
    cross-check that fails loudly BEFORE sampling. Background: the two Wan
    VAEs are near-identically named but incompatible -- wan_2.1_vae (16ch)
    serves Wan 2.1 AND all Wan 2.2 14B models; wan2.2_vae (48ch, 16x16x4
    high compression) serves ONLY the Wan 2.2 TI2V-5B model.

Load CLIP design notes:
  * The `type` list is pulled AT RUNTIME from core CLIPLoader, so new encoder
    types appear automatically after a ComfyUI update; a frozen snapshot is
    the fallback. Default is 'wan'.
  * `auto` resolves only UNAMBIGUOUS filename patterns (umt5 -> wan,
    qwen_2.5_vl -> qwen_image); anything else raises with a clear message
    instead of guessing.

v716 -- multiple text encoders + progressive slots
--------------------------------------------------
`comfy.sd.load_clip(ckpt_paths=[...])` has always taken a LIST. Core's four
separate loader nodes (CLIPLoader / Dual / Triple / Quadruple) exist only
because ComfyUI's static INPUT_TYPES cannot vary the widget count -- which is
precisely what web/js/ph_basics.js now solves by hiding widgets rather than
removing them. So one node covers all four.

MEASURED against ComfyUI master, 2026-07-23 (comfy/sd.py
load_text_encoder_state_dicts), because both points had been assumptions:

  * ORDER DOES NOT MATTER. Every multi-encoder branch identifies each file via
    detect_te_model / membership tests, never by position. clip_l + t5xxl and
    t5xxl + clip_l reach the same model.
  * THE COUNT OUTRANKS THE TYPE. The function branches on len(clip_data) first.
    At 3 files it always builds sd3, at 4 always hidream, and clip_type is not
    consulted at all -- which is why core's Triple and Quadruple loaders carry
    no `type` widget. This node keeps its widget (one node, all counts) and
    states in the readout when the count has overruled it, rather than leaving
    a control that silently does nothing.

Slots are APPEND-ONLY (clip_name stays at position 1, clip_name_2..4 go on the
end) because widgets_values is positional -- the v585 law.
"""

import re
import os

import folder_paths
import comfy.sd
import comfy.utils
import torch
from . import uls_noise
from . import ph_te_detect


_UI_KEY = "pls_basics"  # status line channel, rendered by web/js/ph_basics.js


# ---------------------------------------------------------------------------
# ⬡ Polyhedron Seed (v538, unchanged behavior)
# ---------------------------------------------------------------------------

class ULSSeed:
    """⬡ Polyhedron Seed -- seed source with used-seed readout."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                    "tooltip": "Seed value. 'control after generate' behaves "
                               "exactly like the core samplers. 🎲 Roll draws "
                               "a fresh random seed and pins it to 'fixed'; "
                               "↺ Reuse last restores the seed of the last "
                               "run.",
                }),
                # v685: APPENDED, never inserted -- widgets_values is positional.
                "noise_type": (uls_noise.NOISE_TYPES, {
                    "default": "gaussian",
                    "tooltip": "Character of the noise the `noise` output "
                               "produces. gaussian at strength 1.0 is bit "
                               "identical to what a sampler makes on its own, "
                               "so it changes nothing until you change it. "
                               "brown/pink push energy into low frequencies "
                               "(composition), blue into high (detail), "
                               "fractal is coherent multi-octave structure. "
                               "NOTE: models are trained on gaussian noise -- "
                               "the others are a deliberate excursion, "
                               "different is not automatically better.",
                }),
                "noise_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 20.0, "step": 0.01,
                    "round": 0.01,
                    "tooltip": "Scale in units of standard latent noise "
                               "(1.0 == torch.randn scale). Leave at 1.0 "
                               "unless you are deliberately experimenting; "
                               "the schedule assumes unit-scale noise.",
                }),
                # v687: PREVIEW GEOMETRY ONLY -- appended, and deliberately not
                # part of anything the run computes. The real field is built by
                # the sampler at the LATENT's own shape; these two only tell the
                # in-node preview which grid to draw, so you can look at the
                # actual noise instead of an artist's impression of it.
                "preview_width": ("INT", {
                    "default": 64, "min": 8, "max": 512, "step": 8,
                    "tooltip": "LATENT width for the in-node preview only -- it "
                               "changes nothing about the run. Latent grid, not "
                               "pixels: a 1024px image on a /8 model is 128, a "
                               "1440px Flux2 image on /16 is 90.",
                }),
                "preview_height": ("INT", {
                    "default": 64, "min": 8, "max": 512, "step": 8,
                    "tooltip": "LATENT height for the in-node preview only. See "
                               "preview_width.",
                }),
                # v833: APPENDED at the very end (house law #577 -- the canon
                # is positional; a thematically nicer seat next to noise_type
                # would renumber every saved workflow).
                "noise_character": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "round": 0.01,
                    "tooltip": "How much of the noise type's character "
                               "survives. 1.0 = the pure type (bit-identical "
                               "to earlier builds). Below 1.0 the field is "
                               "cross-faded with plain gaussian -- around "
                               "0.2-0.4 the spectrally heavy types (pink, "
                               "brown, fractal, pyramid) become a dosed "
                               "compositional bias instead of an excursion. "
                               "For offset it scales the offset amount; "
                               "gaussian and zeros ignore it.",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("INT", "STRING", "NOISE")
    RETURN_NAMES = ("seed", "seed_string", "noise")
    OUTPUT_TOOLTIPS = (
        "The seed as INT (wire into samplers / noise nodes).",
        "The same seed as STRING (filenames, notes, prompts).",
        "A NOISE source for the sampler's `noise` input -- THE noise the "
        "model actually denoises. This is where noise character belongs: "
        "an empty latent's contents are multiplied by zero at the start of "
        "a full schedule, the sampler's own noise is not.",
    )
    FUNCTION = "emit"
    CATEGORY = "Polyhedron/Loaders"
    DESCRIPTION = ("One seed for the whole graph, plus a readout of the value that was "
                   "actually used. The seed_string output carries the same number as text "
                   "for filename prefixes, so the file on disk and the graph agree on which "
                   "seed made it.")

    def emit(self, seed, noise_type="gaussian", noise_strength=1.0,
             preview_width=64, preview_height=64, unique_id=None,
             noise_character=1.0):
        used = int(seed)
        # v833, house law #577: the appended row heals here too. Older
        # graphs carry nothing; hand-edited ones can carry anything.
        try:
            noise_character = min(1.0, max(0.0, float(noise_character)))
        except Exception:
            noise_character = 1.0
        noise = uls_noise.ULSNoiseSource(used, noise_type, noise_strength,
                                         noise_character)
        if not noise.is_default():
            print("[PLS] Seed: noise=%s strength=%.2f character=%.2f -> "
                  "the sampler will denoise THIS instead of plain gaussian"
                  % (noise_type, float(noise_strength),
                     float(noise_character)))
        # v689: report the preview geometry back to the frontend. When
        # preview_width/height are CONVERTED TO INPUTS and wired, the widget
        # values are stale -- the real numbers arrive over a cable and only the
        # backend ever sees them. Same honest limit as the CTE fields and the
        # Reference band: what comes over a cable is knowable only after it has
        # flowed once.
        return {"ui": {"pls_seed": [{"used": used,
                                     "pw": int(preview_width),
                                     "ph": int(preview_height)}]},
                "result": (used, str(used), noise)}


# ---------------------------------------------------------------------------
# shared pure helpers (sandbox-testable, no comfy attribute access at import)
# ---------------------------------------------------------------------------

_VAE_FAMILIES = {
    4:  ("SD1 / SD2 / SDXL family",
         "classic 4ch image VAE"),
    16: ("Wan 2.1-family / Flux / SD3 (16ch)",
         "wan_2.1_vae serves Wan 2.1 AND all Wan 2.2 14B models"),
    48: ("Wan 2.2 high-compression (48ch)",
         "ONLY for Wan 2.2 TI2V-5B -- 14B models need wan_2.1_vae"),
    24: ("MiniMax H3 video (24ch)",
         "the VIDEO half of a joint H3 model; its audio VAE is 32ch"),
    32: ("MiniMax H3 audio (32ch)",
         "the AUDIO half; an H3 model reports 32 as the MAX over both "
         "streams, so 32 alone does not tell the two apart"),
}


def _is_joint_model(model):
    """True when the model builds MORE THAN ONE latent stream (MiniMax H3).

    Delegates to the three-witness probe in ph_joint_probe -- NOT re-implemented
    here. It fails OPEN (returns 1 when unanswerable), so an unknown model keeps
    the ordinary channel check it has always had.
    """
    try:
        try:
            from .ph_joint_probe import _joint_latent_parts
        except ImportError:  # pragma: no cover - direct-run fallback
            from ph_joint_probe import _joint_latent_parts
        return _joint_latent_parts(model) > 1
    except Exception:
        return False


def _vae_family(channels):
    if channels in _VAE_FAMILIES:
        return _VAE_FAMILIES[channels]
    return (f"{channels}ch VAE", "channel count not in the family table")


def _model_latent_channels(model):
    """Latent channel count the connected model expects, or None."""
    try:
        return int(model.model.latent_format.latent_channels)
    except Exception:
        return None


_CLIP_AUTO_PATTERNS = (
    ("umt5", "wan"),
    ("qwen_2.5_vl", "qwen_image"),
)


def _clip_auto_type(filename):
    """Conservative auto-detection: only unambiguous patterns, else None."""
    low = str(filename).lower()
    for pattern, clip_type in _CLIP_AUTO_PATTERNS:
        if pattern in low:
            return clip_type
    return None


# Frozen snapshot of core CLIPLoader's type list (measured 2026-07-11). Only a
# FALLBACK -- _clip_type_list() pulls the live list first, so new encoder types
# appear automatically after a ComfyUI update.
_CLIP_TYPE_FALLBACK = [
    "stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi",
    "ltxv", "pixart", "cosmos", "lumina2", "wan", "hidream", "chroma",
    "ace", "omnigen2", "qwen_image", "hunyuan_image", "flux2", "ovis",
    "longcat_image", "cogvideox", "lens", "pixeldit", "ideogram4", "boogu",
    "krea2",
]

# v716: the SAME fallback role for core DualCLIPLoader's list -- the families
# that only exist once two encoders are loaded. Note "flux" appears here and
# NOT above: it has never been a single-encoder type, which is why the old
# single-file node had to refuse it outright.
_CLIP_TYPE_DUAL_FALLBACK = [
    "sdxl", "flux", "hunyuan_video", "hunyuan_image", "hunyuan_video_15",
    "kandinsky5", "kandinsky5_image", "newbie",
]

# Placeholder for the appended slots. Slot 1 has no placeholder -- a Load CLIP
# node without a first encoder has nothing to do.
_CLIP_PLACEHOLDER = "\u2014 none \u2014"
_MAX_CLIP_SLOTS = 4          # core's ceiling too: Quadruple is the widest recipe


def _is_filled(value):
    """A slot counts as filled when it names a file, not a placeholder."""
    if value is None:
        return False
    text = str(value).strip()
    return bool(text) and text != _CLIP_PLACEHOLDER and text != _MODEL_PLACEHOLDER

# ---------------------------------------------------------------------------
# v542: resolve the CLIP type from the MODEL instead of the filename.
#
# The filename cannot carry the answer -- t5xxl serves sd3, ltxv, mochi, pixart,
# hidream; the SAME weights need a DIFFERENT `type` depending on the target
# family, because `type` selects the tokenizer wrapper + embedding contract the
# sampler downstream expects. The graph already knows the answer: the loaded
# MODEL knows its family. So we read it from there.
#
# Keys are comfy.model_base class names (measured against ComfyUI master,
# 2026-07-11); lookup walks the MRO, so every WAN2x subclass resolves through
# WAN21 and Chroma(Flux) resolves to chroma BEFORE its Flux base is reached.
# Every hit is validated against the LIVE type list before use -- a stale entry
# fails loudly instead of silently mis-conditioning.
_MODEL_CLASS_CLIP_TYPE = {
    "WAN21": "wan",                       # + WAN22, Vace, Animate, S2V, Camera...
    "SD3": "sd3",
    "LTXV": "ltxv", "LTXAV": "ltxv",
    "GenmoMochi": "mochi",
    "PixArt": "pixart",
    "CosmosVideo": "cosmos", "CosmosPredict2": "cosmos",
    "Lumina2": "lumina2",
    "HiDream": "hidream", "HiDreamO1": "hidream",
    "Chroma": "chroma",
    "ACEStep": "ace", "ACEStep15": "ace",
    "Omnigen2": "omnigen2",
    "QwenImage": "qwen_image",
    "StableCascade_C": "stable_cascade", "StableCascade_B": "stable_cascade",
    "StableAudio1": "stable_audio", "StableAudio3": "stable_audio",
    "HunyuanImage21": "hunyuan_image",
    "Flux2": "flux2",
    "LongCatImage": "longcat_image",
    "Lens": "lens",
    "PixelDiTT2I": "pixeldit",
    "Ideogram4": "ideogram4",
    "Boogu": "boogu",
    "Krea2": "krea2",
}

# Families that need TWO text encoders (core's DualCLIPLoader).
#
# Until v715 this node loaded exactly ONE encoder, so these families could only
# be named and refused. Since v716 it loads up to four, so each entry now also
# carries the type string to USE once a second slot is filled. The refusal only
# survives for the one-slot case, where it is still the honest answer.
_DUAL_ENCODER_CLASSES = {
    "Flux": "flux (clip_l + t5xxl)",
    "HunyuanVideo": "hunyuan_video (clip_l + llava)",
    "SDXL": "sdxl (clip_l + clip_g)",
    "SDXLRefiner": "sdxl (clip_l + clip_g)",
}

_MODEL_CLASS_DUAL_TYPE = {
    "Flux": "flux",
    "HunyuanVideo": "hunyuan_video",
    "SDXL": "sdxl",
    "SDXLRefiner": "sdxl",
}


class _DualEncoderNeeded(Exception):
    """Raised when the model's family needs a dual CLIP loader."""


def _clip_type_from_model(model, slots=1):
    """CLIP type for a loaded MODEL, or None. Walks the class hierarchy.

    v716: `slots` is how many encoder files are actually loaded. With two or
    more, a dual family resolves to its real type instead of raising -- the
    node can now serve it. With one it still raises, because one encoder for a
    two-encoder family is half a conditioning stack and silence there was
    always the wrong answer.
    """
    try:
        chain = type(model.model).__mro__
    except Exception:
        return None
    for cls in chain:
        name = cls.__name__
        if name in _MODEL_CLASS_CLIP_TYPE:
            return _MODEL_CLASS_CLIP_TYPE[name]
        if name in _DUAL_ENCODER_CLASSES:
            if int(slots) >= 2 and name in _MODEL_CLASS_DUAL_TYPE:
                return _MODEL_CLASS_DUAL_TYPE[name]
            raise _DualEncoderNeeded(f"{name} -- {_DUAL_ENCODER_CLASSES[name]}")
    return None


def _model_class_name(model):
    try:
        return type(model.model).__name__
    except Exception:
        return "unknown"


def _core_type_list(loader_name, fallback):
    """Live `type` list of a core loader class; frozen snapshot as fallback."""
    try:
        import nodes as _core_nodes  # ComfyUI core
        cls = getattr(_core_nodes, loader_name, None)
        if cls is None:
            cls = _core_nodes.NODE_CLASS_MAPPINGS.get(loader_name)
        types = cls.INPUT_TYPES()["required"]["type"][0]
        if isinstance(types, (list, tuple)) and types:
            return list(types)
    except Exception:
        pass
    return list(fallback)


def _clip_type_list():
    """Live single-encoder type list from core CLIPLoader; snapshot fallback."""
    return _core_type_list("CLIPLoader", _CLIP_TYPE_FALLBACK)


def _clip_type_list_dual():
    """Live two-encoder type list from core DualCLIPLoader; snapshot fallback."""
    return _core_type_list("DualCLIPLoader", _CLIP_TYPE_DUAL_FALLBACK)


def _clip_type_choices():
    """Everything this node can be told to be, single and multi, deduplicated.

    Order is deliberate: the single-encoder families first (the common case),
    then the multi-encoder families that only exist from two files upwards.
    """
    out = list(_clip_type_list())
    seen = set(out)
    for name in _clip_type_list_dual():
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# ⬡ Polyhedron Load CLIP (v539)
# ---------------------------------------------------------------------------

class ULSLoadCLIP:
    """⬡ Polyhedron Load CLIP -- multi-encoder loader with live type list."""

    @classmethod
    def INPUT_TYPES(cls):
        types = _clip_type_choices()
        files = _sized_list(folder_paths.get_filename_list("text_encoders"),
                            ("text_encoders",))  # v828: sizes in the list
        # v716: APPENDED, never inserted -- widgets_values is positional and
        # clip_name must keep position 1 for every workflow already saved.
        #
        # WRITTEN OUT ON PURPOSE, not built in a comprehension: a comprehension
        # makes INPUT_TYPES look dynamic, and test_v577_widget_order then stops
        # checking this node's widget ORDER -- which is exactly the protection a
        # node with appended positional slots needs most. Three spelled-out
        # entries cost a few lines and keep the gate watching.
        slot_2 = ([_CLIP_PLACEHOLDER] + list(files), {
            "default": _CLIP_PLACEHOLDER,
            "tooltip": "Text encoder 2. Encoders load TOGETHER -- this is not an "
                       "either/or switch; slot 1 plus this one is what core's "
                       "dual loader does. Order does not matter, each file is "
                       "identified by its own weights.",
        })
        slot_3 = ([_CLIP_PLACEHOLDER] + list(files), {
            "default": _CLIP_PLACEHOLDER,
            "tooltip": "Text encoder 3. With three encoders ComfyUI always "
                       "builds sd3 (clip_l + clip_g + t5xxl) and ignores the "
                       "'type' field -- the count decides.",
        })
        slot_4 = ([_CLIP_PLACEHOLDER] + list(files), {
            "default": _CLIP_PLACEHOLDER,
            "tooltip": "Text encoder 4. With four encoders ComfyUI always builds "
                       "hidream (long clip_l + long clip_g + t5xxl + llama) and "
                       "ignores the 'type' field -- the count decides.",
        })
        return {
            "required": {
                "clip_name": (files, {
                    "tooltip": "Text encoder file (models/text_encoders).",
                }),
                "type": (["auto"] + types, {
                    "default": "auto",
                    "tooltip": "Encoder architecture. The list is pulled live from "
                               "core CLIPLoader AND DualCLIPLoader, so new types "
                               "appear after a ComfyUI update. 'auto' (default): "
                               "connect the model and the family is read from it; "
                               "otherwise only unambiguous filenames resolve. "
                               "Never guesses -- it errors and asks you to pick. "
                               "With THREE or FOUR encoders this field is ignored: "
                               "ComfyUI derives the family from the count alone "
                               "(3 = sd3, 4 = hidream), and the readout says so.",
                }),
                "device": (["default", "cpu"], {
                    "default": "default",
                    "tooltip": "cpu keeps the encoder off the GPU (VRAM relief). "
                               "Applies to every loaded encoder.",
                }),
            },
            "optional": {
                "model": ("MODEL", {
                    "tooltip": "Optional: with 'type = auto' the encoder family "
                               "is resolved FROM THIS MODEL -- swap the model "
                               "and the CLIP type follows. The filename alone "
                               "cannot carry this (t5xxl serves sd3, flux, "
                               "ltxv, ...).",
                }),
                "clip_name_2": slot_2,
                "clip_name_3": slot_3,
                "clip_name_4": slot_4,
            },
        }


    @classmethod
    def VALIDATE_INPUTS(cls, **_kw):
        """v828: the combo list decorates values with sizes, so a workflow
        saved BEFORE this cut carries the bare filename -- a string the list
        no longer offers. Without this, server-side validation kills that
        run at the door (the v823 wound, different node). load() strips and
        resolves; unknown files still fail there, loudly and by name."""
        return True

    RETURN_TYPES = ("CLIP", "STRING")
    RETURN_NAMES = ("clip", "info")
    OUTPUT_TOOLTIPS = ("The loaded text encoder stack (native CLIP type).",
                       "Readout: files | resolved type | device | size | what "
                       "each slot was identified as.")
    FUNCTION = "load"
    CATEGORY = "Polyhedron/Loaders"
    DESCRIPTION = ("Text-encoder loader for one to four encoders in a single node, "
                   "replacing the separate single/dual/triple/quadruple loaders. A slot "
                   "appears once the one before it is filled, so the node is only as "
                   "large as the family needs. The type list is read from the running "
                   "ComfyUI rather than a hard-coded table, so a newly supported family "
                   "shows up the day core learns it. Before any weights are read, each "
                   "file is identified from its safetensors header and checked against "
                   "the recipe, so a wrong combination fails immediately and by name "
                   "instead of surfacing as bad conditioning much further downstream.")

    def load(self, clip_name, type="auto", device="default", model=None,  # noqa: A002
             **kwargs):
        # --- 1. which slots are actually filled, in order -------------------
        # Gaps are tolerated rather than rejected: the frontend keeps the slots
        # dense, but a hand-edited or older workflow must not fail over a hole.
        # v828: the list decorates values with sizes; strip HERE, at the one
        # collection point, so every consumer below sees the bare filename.
        clip_name = _strip_size(clip_name)
        names = [clip_name] if _is_filled(clip_name) else []
        for i in range(2, _MAX_CLIP_SLOTS + 1):
            value = _strip_size(kwargs.get(f"clip_name_{i}"))
            if _is_filled(value):
                names.append(str(value))
        if not names:
            raise ValueError(
                "[PLS] Load CLIP: no encoder selected -- pick a file in slot 1."
            )
        count = len(names)

        # --- 2. resolve the type -------------------------------------------
        # MEASURED (comfy/sd.py, 2026-07-23): at three and four files core does
        # not look at clip_type at all. Saying so beats honouring a widget that
        # has no effect.
        forced = ph_te_detect.forced_type_for_count(count)
        if forced is not None:
            resolved = forced
            how = f"{count} encoders -> {forced} (the count decides, 'type' ignored)"
        else:
            resolved, how = type, "set"
            if resolved == "auto":
                # (1) the MODEL knows its family -- the only source that always can
                if model is not None:
                    try:
                        resolved = _clip_type_from_model(model, slots=count)
                        how = "auto from " + _model_class_name(model)
                    except _DualEncoderNeeded as e:
                        raise ValueError(
                            f"[PLS] Load CLIP: this model needs TWO text encoders "
                            f"({e}) -- fill slot 2 and they load together, or set "
                            f"the type by hand. Only one slot is filled."
                        ) from e
                    if resolved is not None and resolved not in _clip_type_choices():
                        raise ValueError(
                            f"[PLS] Load CLIP: resolved type '{resolved}' is not "
                            f"offered by this ComfyUI -- update the pack or set the "
                            f"type by hand."
                        )
                # (2) no model (or unmapped): the conservative filename heuristic
                if resolved is None or resolved == "auto":
                    resolved = _clip_auto_type(clip_name)
                    how = "auto from filename"
                # (3) nothing worked -- name BOTH failed paths, never guess
                if resolved is None:
                    hint = ("connect the model to the 'model' input"
                            if model is None else
                            f"model family '{_model_class_name(model)}' is not in "
                            f"the table")
                    raise ValueError(
                        f"[PLS] Load CLIP: 'auto' could not resolve the encoder type "
                        f"({hint}; the filename '{clip_name}' is not conclusive "
                        f"either) -- set the type explicitly."
                    )

        clip_type = getattr(comfy.sd.CLIPType, str(resolved).upper(), None)
        if clip_type is None:
            raise ValueError(
                f"[PLS] Load CLIP: type '{resolved}' is unknown to this "
                f"ComfyUI (available: {', '.join(_clip_type_choices())})."
            )

        paths = [folder_paths.get_full_path_or_raise("text_encoders", n)
                 for n in names]

        # --- 3. check the slate BEFORE loading a single byte of weights -----
        # The whole point is to fail here rather than after a 10 GB read, or --
        # worse -- to load a wrong-but-valid stack that only shows up as bad
        # conditioning much further downstream.
        idents = [ph_te_detect.identify_file(p) for p in paths]
        ok, why = ph_te_detect.check_recipe(resolved, idents)
        if not ok:
            raise ValueError(
                f"[PLS] Load CLIP: incompatible encoder set -- {why}. "
                f"(Read from the safetensors headers; no weights were loaded.)"
            )
        slate = ", ".join(ph_te_detect.describe(i, os.path.basename(n))
                          for i, n in zip(idents, names))

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = torch.device("cpu")
            model_options["offload_device"] = torch.device("cpu")

        clip = comfy.sd.load_clip(
            ckpt_paths=paths,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options,
        )

        total = 0.0
        for path in paths:
            try:
                total += os.path.getsize(path) / (1024 * 1024)
            except Exception:
                pass
        size_txt = f"{total:,.0f} MB" if total else "size n/a"

        tag = "" if how == "set" else f" ({how})"
        head = names[0] if count == 1 else f"{count} encoders"
        info = f"{head} | type {resolved}{tag} | {device} | {size_txt}"
        if why:
            info += f" | {why}"
        if count > 1:
            info += f" | {slate}"
        print(f"[PLS] Load CLIP: {info}")
        return {"ui": {_UI_KEY: [info]}, "result": (clip, info)}


# ---------------------------------------------------------------------------
# ⬡ Polyhedron Load VAE (v539)
# ---------------------------------------------------------------------------

class ULSLoadVAE:
    """⬡ Polyhedron Load VAE -- native VAE with family readout + model check."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_name": (_sized_list(
                    folder_paths.get_filename_list("vae"), ("vae",)), {
                    "tooltip": "VAE file (models/vae). Wan note: wan_2.1_vae "
                               "= Wan 2.1 AND all Wan 2.2 14B models; "
                               "wan2.2_vae = TI2V-5B ONLY.",
                }),
                "precision": (["auto", "bf16", "fp16", "fp32"], {
                    "default": "auto",
                    "tooltip": "Override the VAE dtype (the useful part of the "
                               "wrapper loaders, applied to the NATIVE VAE). "
                               "auto = ComfyUI default.",
                }),
            },
            "optional": {
                "model": ("MODEL", {
                    "tooltip": "Optional cross-check: verifies the model's "
                               "latent format matches this VAE and fails "
                               "loudly BEFORE sampling instead of mid-run.",
                }),
            },
        }


    @classmethod
    def VALIDATE_INPUTS(cls, **_kw):
        """v828: the combo list decorates values with sizes, so a workflow
        saved BEFORE this cut carries the bare filename -- a string the list
        no longer offers. Without this, server-side validation kills that
        run at the door (the v823 wound, different node). load() strips and
        resolves; unknown files still fail there, loudly and by name."""
        return True

    RETURN_TYPES = ("VAE", "STRING")
    RETURN_NAMES = ("vae", "info")
    OUTPUT_TOOLTIPS = ("The loaded VAE (native type -- fits every decoder).",
                       "Readout: file | channels -> family | dtype | note.")
    FUNCTION = "load"
    CATEGORY = "Polyhedron/Loaders"
    DESCRIPTION = ("Native VAE loader that names the family it just loaded and, when a "
                   "MODEL is wired, checks that the two belong together. A VAE/model "
                   "mismatch otherwise appears as colour drift or noise at decode time, a "
                   "long way from its cause.")

    _DTYPES = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}

    def load(self, vae_name, precision="auto", model=None):
        vae_name = _strip_size(vae_name)  # v828: sized list entries
        dtype = getattr(torch, self._DTYPES[precision]) if precision in self._DTYPES else None

        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd, dtype=dtype)
        if hasattr(vae, "throw_exception_if_invalid"):
            vae.throw_exception_if_invalid()

        channels = getattr(vae, "latent_channels", None)
        family, note = (_vae_family(channels) if channels
                        else ("unknown family", "latent_channels not exposed"))

        model_ch = _model_latent_channels(model) if model is not None else None

        # v889: a JOINT model (MiniMax H3) carries TWO latent streams, and
        # latent_format.latent_channels reports only the MAXIMUM over both
        # (measured in v885: MiniMaxH3AV inherits 24-row rgb factors while
        # declaring 32). Comparing one number against one VAE is therefore
        # meaningless here: the correct 24ch VIDEO VAE would be refused against
        # a model that says 32, and the node would kill a legitimate run with a
        # message pointing at the wrong thing. So the check STEPS ASIDE for a
        # joint model -- the same move the None-conditioning gate makes on the
        # joint branch in v883. The family line above still names what was
        # loaded, so the user is informed rather than silently unguarded.
        if model_ch is not None and _is_joint_model(model):
            print(f"[PLS] Load VAE: '{vae_name}' ({channels}ch) -- connected "
                  f"model is a JOINT audio/video model reporting {model_ch}ch "
                  f"as the maximum over both streams; the channel check does "
                  f"not apply and is skipped.")
            model_ch = None

        if model_ch is not None and channels is not None and model_ch != channels:
            hint = ""
            if channels == 48 and model_ch == 16:
                hint = (" wan2.2_vae is for the Wan 2.2 TI2V-5B model ONLY; "
                        "Wan 2.2 14B (and Wan 2.1) need wan_2.1_vae.")
            elif channels == 16 and model_ch == 48:
                hint = (" This model is the Wan 2.2 TI2V-5B -- it needs "
                        "wan2.2_vae (48ch).")
            raise ValueError(
                f"[PLS] Load VAE: '{vae_name}' has {channels} latent channels "
                f"but the connected model expects {model_ch}.{hint}"
            )

        shown_dtype = precision
        if precision == "auto":
            vd = getattr(vae, "vae_dtype", None)
            shown_dtype = str(vd).replace("torch.", "") if vd is not None else "auto"

        info = f"{vae_name} | {channels}ch -> {family} | dtype {shown_dtype} | {note}"
        if model_ch is not None:
            info += f" | model check OK ({model_ch}ch)"
        return {"ui": {_UI_KEY: [info]}, "result": (vae, info)}


# ---------------------------------------------------------------------------
# ⬡ Polyhedron Load Model (v541) -- switch + loader fused, GGUF-aware
# ---------------------------------------------------------------------------
#
# Design (2026-07-11 session): the string-out ⬡ Select Model Switch stays for
# feeding FOREIGN loader combos; THIS node closes the main-path footgun by
# emitting a real MODEL directly. Six slots over the merged model list, the
# selected file is routed by type:
#   *.gguf            -> delegated to the REGISTERED UnetLoaderGGUF class of
#                        the installed ComfyUI-GGUF pack (runtime lookup via
#                        NODE_CLASS_MAPPINGS -- no import coupling)
#   diffusion_models/ -> core comfy.sd.load_diffusion_model (+ weight_dtype)
#   checkpoints/      -> core load_checkpoint_guess_config, MODEL ONLY
#                        (output_clip/output_vae=False -- CLIP/VAE stay with
#                        the dedicated ⬡ loaders)
# model_name (STRING) is still emitted, as before.

_MODEL_PLACEHOLDER = "\u2014 select model \u2014"


# ---------------------------------------------------------------------------
# v828 -- SIZES IN THE LIST ITSELF (Frank's ask, the cutout pattern lifted).
#
# The cutout (v752) established the house form: the size stands IN the
# dropdown entry, because that is the moment the number helps -- picking a
# 9.8 GB t5 over a 4.9 GB one is a decision made in the list, not in a
# readout after the fact. The PRICE is the same one the cutout named and
# paid: ComfyUI shows VALUES in the open list, so the decoration is part of
# the stored value, and every entry into the load path strips it again.
#
# TWO DELIBERATE DIFFERENCES from the cutout:
#   - NO diamond. The diamond marks a model the pack ships a DEFINITION
#     for; these three loaders list whatever lies in the folders --
#     nothing here is ours to prefer.
#   - The strip is an EXACT SUFFIX match, not the cutout's broad split on
#     " - " / double space: loader FILENAMES legitimately contain dashes
#     ("model - v2.safetensors"), and a strip that eats part of a filename
#     is the silent-wrong-model bug all over again.
_SIZE_SUFFIX = re.compile(r"\s\u00b7\s[\d.,]+\s?(KB|MB|GB|TB)$")


def _strip_size(value):
    """Inverse of _with_size: remove exactly one trailing ' · <n> <unit>'."""
    return _SIZE_SUFFIX.sub("", str(value)).rstrip()


def _fmt_size(n):
    """675 -> '675 B' is never shown; model files start at KB scale."""
    try:
        n = int(n)
    except Exception:
        return ""
    if n <= 0:
        return ""
    if n >= 1 << 30:
        return "%.1f GB" % (n / float(1 << 30))
    if n >= 1 << 20:
        return "%d MB" % round(n / float(1 << 20))
    return "%d KB" % max(1, round(n / float(1 << 10)))


def _with_size(name, folders):
    """One combo entry: '<name> \u00b7 <size>' when the file is on disk in any
    of the given folders; the bare name (or a placeholder) otherwise."""
    if not name or str(name).startswith("\u2014"):
        return name
    for folder in folders:
        try:
            path = folder_paths.get_full_path(folder, name)
            if path:
                s = _fmt_size(os.path.getsize(path))
                return (name + " \u00b7 " + s) if s else name
        except Exception:
            pass
    return name


def _sized_list(names, folders):
    return [_with_size(n, folders) for n in names]


def _model_source_list():
    """Merged model filename list (same folders the Select Model Switch scans)."""
    names = set()
    for folder in ("unet", "diffusion_models", "checkpoints",
                   "unet_gguf", "diffusion_models_gguf"):
        try:
            names.update(folder_paths.get_filename_list(folder))
        except Exception:
            pass
    return [_MODEL_PLACEHOLDER] + _sized_list(
        sorted(names), ("diffusion_models", "unet", "checkpoints",
                        "unet_gguf", "diffusion_models_gguf"))  # v828


def _detect_kind(name):
    """'gguf' | 'unet' | 'checkpoint' -- pure suffix/folder routing key."""
    if str(name).lower().endswith(".gguf"):
        return "gguf"
    for folder in ("diffusion_models", "unet"):
        try:
            if folder_paths.get_full_path(folder, name):
                return "unet"
        except Exception:
            pass
    try:
        if folder_paths.get_full_path("checkpoints", name):
            return "checkpoint"
    except Exception:
        pass
    return "unet"  # optimistic default; the load path raises with a clear text


def _size_mb(name, folders):
    for folder in folders:
        try:
            path = folder_paths.get_full_path(folder, name)
            if path:
                return f"{os.path.getsize(path) / (1024 * 1024):,.0f} MB"
        except Exception:
            pass
    return "size n/a"


class ULSLoadModel:
    """⬡ Polyhedron Load Model -- 6-slot switch that loads the pick itself."""

    _DTYPES = ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]

    @classmethod
    def INPUT_TYPES(cls):
        models = _model_source_list()
        slots = {f"model_{i}": (models, {"tooltip": f"Slot {i} -- any known "
                 "model file (safetensors, GGUF, checkpoint)."})
                 for i in range(1, 7)}
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 6, "step": 1,
                                   "tooltip": "Which slot to load. These slots "
                                              "are ALTERNATIVES -- exactly one "
                                              "model is loaded. The upper bound "
                                              "follows how many slots are filled, "
                                              "so it cannot point at an empty "
                                              "one; a further slot appears as "
                                              "soon as the last is in use."}),
                "weight_dtype": (cls._DTYPES, {"default": "default",
                                 "tooltip": "Safetensors weight dtype override "
                                            "(core UNETLoader options). Ignored "
                                            "for GGUF (already quantized)."}),
            },
            "optional": slots,
        }


    @classmethod
    def VALIDATE_INPUTS(cls, **_kw):
        """v828: the combo list decorates values with sizes, so a workflow
        saved BEFORE this cut carries the bare filename -- a string the list
        no longer offers. Without this, server-side validation kills that
        run at the door (the v823 wound, different node). load() strips and
        resolves; unknown files still fail there, loudly and by name."""
        return True

    RETURN_TYPES = ("MODEL", "STRING", "STRING")
    RETURN_NAMES = ("model", "model_name", "info")
    OUTPUT_TOOLTIPS = (
        "The loaded model -- a real MODEL, never a filename string.",
        "The selected filename (for Save paths, notes, foreign combos).",
        "Readout: file | source kind | dtype | size.",
    )
    FUNCTION = "load"
    CATEGORY = "Polyhedron/Loaders"
    DESCRIPTION = ("Six model slots and a selector, and this node loads the pick itself - "
                   "no separate loader downstream. One dropdown spans unet, "
                   "diffusion_models, checkpoints and both GGUF folders, so fp8 and GGUF "
                   "sit in the same list. The model_name output feeds filename prefixes, so "
                   "the output says which model made it.")

    @staticmethod
    def _load_gguf(name):
        import nodes as core_nodes
        cls = core_nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
        if cls is None:
            raise ValueError(
                "[PLS] Load Model: a .gguf file is selected but ComfyUI-GGUF "
                "is not installed -- install it or pick a safetensors model."
            )
        loader = cls()
        result = getattr(loader, cls.FUNCTION)(unet_name=name)
        return result[0]

    def load(self, select, weight_dtype="default", **kwargs):
        name = kwargs.get(f"model_{int(select)}")
        if name is not None:
            name = _strip_size(name)  # v828: sized list entries
        if name is None or str(name).strip() == _MODEL_PLACEHOLDER:
            raise ValueError(
                f"[PLS] Load Model: slot {int(select)} is empty -- pick a "
                f"model file there or change 'select'."
            )
        name = str(name)
        kind = _detect_kind(name)

        if kind == "gguf":
            model = self._load_gguf(name)
            dtype_txt = "GGUF (dtype n/a)"
            size = _size_mb(name, ("unet_gguf", "diffusion_models_gguf", "unet"))
        elif kind == "checkpoint":
            path = folder_paths.get_full_path_or_raise("checkpoints", name)
            out = comfy.sd.load_checkpoint_guess_config(
                path, output_vae=False, output_clip=False,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )
            model = out[0]
            dtype_txt = "checkpoint (MODEL only)"
            size = _size_mb(name, ("checkpoints",))
        else:
            model_options = {}
            if weight_dtype == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
            elif weight_dtype == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2
            path = None
            for folder in ("diffusion_models", "unet"):
                try:
                    path = folder_paths.get_full_path(folder, name)
                except Exception:
                    path = None
                if path:
                    break
            if not path:
                raise ValueError(
                    f"[PLS] Load Model: '{name}' not found in "
                    f"diffusion_models/unet/checkpoints."
                )
            model = comfy.sd.load_diffusion_model(path, model_options=model_options)
            dtype_txt = weight_dtype
            size = _size_mb(name, ("diffusion_models", "unet"))

        info = f"{name} | {kind} | {dtype_txt} | {size}"
        print(f"[PLS] Load Model: slot {int(select)} -> {info}")
        return {"ui": {_UI_KEY: [info]}, "result": (model, name, info)}
