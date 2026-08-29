"""ph_te_detect.py -- v716

Text-encoder identification from the SAFETENSORS HEADER, plus recipe checking
for the multi-encoder Load CLIP node.

WHY THE HEADER, AND WHY THIS IS NOT A HEURISTIC
-----------------------------------------------
The filename cannot carry the answer -- the same t5xxl weights serve sd3, flux,
ltxv, mochi and hidream, and anybody may rename a file. The previous approach
(_clip_auto_type) guessed from substrings and was honest about being thin.

Measured against ComfyUI master (2026-07-23): core's own `comfy.sd.detect_te_model`
touches NOTHING but tensor KEY NAMES and `.shape[0]`. It never reads a tensor
value. Both of those live in the safetensors header -- the leading 8-byte little
endian length followed by a JSON map of {tensor_name: {dtype, shape, offsets}}.

So we do not reimplement core's judgement, we FEED it: the header is turned into
a mapping of name -> shape-only proxy and handed to core's own function. The
answer is then by construction the same answer core will reach when it actually
loads the file, because it is literally the same code. A local table mirroring
core's key checks is kept as a fallback for installs where the import fails, and
it reports itself as the fallback rather than pretending to be the real thing.

Reading a header costs one open() and a few kilobytes -- no weights are touched,
so a 9.8 GB t5xxl is identified in milliseconds without any VRAM.

HONEST LIMITS (call them out, do not paper over them)
  * umt5-xxl (wan) and t5xxl share their encoder block structure. Core separates
    them by the presence of a `spiece_model` entry, so we report that flag
    instead of pretending to a distinction the tensors do not make.
  * GGUF encoders have no safetensors header. They report as UNKNOWN, and an
    unknown slot never blocks a load -- see check_recipe().
"""

import json
import os
import struct

# Maximum header size we will accept, guarding against a corrupt length field
# turning into a multi-gigabyte read.
_MAX_HEADER = 128 * 1024 * 1024


class _ShapeOnly:
    """Stand-in for a tensor that knows only its shape.

    core's detect_te_model asks for `sd[key].shape[0]` and nothing else, so this
    is a complete substitute for its purposes.
    """

    __slots__ = ("shape",)

    def __init__(self, shape):
        self.shape = tuple(shape)


def read_safetensors_header(path):
    """Return {tensor_name: [shape...]} for a safetensors file, or None.

    Returns None (never raises) when the file is missing, is not safetensors, or
    the header does not parse. A caller that cannot read a header must degrade
    to "unknown", not to a crash.
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                return None
            length = struct.unpack("<Q", raw)[0]
            if length <= 0 or length > _MAX_HEADER:
                return None
            blob = fh.read(length)
            if len(blob) != length:
                return None
        meta = json.loads(blob.decode("utf-8"))
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    out = {}
    for name, spec in meta.items():
        if name == "__metadata__" or not isinstance(spec, dict):
            continue
        shape = spec.get("shape")
        if isinstance(shape, list):
            out[name] = shape
    return out or None


# ---------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------

# Local mirror of the key checks in comfy.sd.detect_te_model, ORDER-SENSITIVE
# exactly as core has them (layer 30 before 22 before 0 -- a clip_g state dict
# contains layer 0 too, so an unordered check would call every clip_g a clip_l).
# Each entry: (key, shape0_or_None, name). shape0 None means "key is enough".
_LOCAL_RULES = (
    ("text_model.encoder.layers.30.mlp.fc1.weight", None, "clip_g"),
    ("text_model.encoder.layers.22.mlp.fc1.weight", None, "clip_h"),
    ("text_model.encoder.layers.0.mlp.fc1.weight", None, "clip_l"),
    ("model.encoder.layers.0.mixer.Wqkv.weight", None, "jina_clip_2"),
    ("encoder.block.23.layer.1.DenseReluDense.wi_1.weight", 10240, "t5xxl"),
    ("encoder.block.23.layer.1.DenseReluDense.wi_1.weight", 5120, "t5xl"),
    ("encoder.block.23.layer.1.DenseReluDense.wi.weight", None, "t5xxl_old"),
    ("encoder.block.0.layer.0.SelfAttention.k.weight", 384, "byt5_small"),
    ("encoder.block.0.layer.0.SelfAttention.k.weight", None, "t5_base"),
    ("model.layers.0.self_attn.k_proj.bias", 256, "qwen25_3b"),
    ("model.layers.0.self_attn.k_proj.bias", 512, "qwen25_7b"),
)

# core TEModel enum member -> the short name we speak in recipes and messages.
_TEMODEL_NAMES = {
    "CLIP_L": "clip_l", "CLIP_G": "clip_g", "CLIP_H": "clip_h",
    "T5_XXL": "t5xxl", "T5_XL": "t5xl", "T5_XXL_OLD": "t5xxl_old",
    "T5_BASE": "t5_base", "T5_GEMMA": "t5_gemma",
    "LLAMA3_8": "llama3_8",
    "GEMMA_2_2B": "gemma2_2b", "GEMMA_3_4B": "gemma3_4b",
    "GEMMA_3_4B_VISION": "gemma3_4b_vision", "GEMMA_3_12B": "gemma3_12b",
    "QWEN25_3B": "qwen25_3b", "QWEN25_7B": "qwen25_7b",
    "BYT5_SMALL_GLYPH": "byt5_small", "JINA_CLIP_2": "jina_clip_2",
}

# Long-CLIP: core reads the position embedding's row count. 77 is the classic
# context, anything larger is a long variant. Same key core looks at.
_POS_EMBED_KEYS = (
    "text_model.embeddings.position_embedding.weight",
    "clip_l.text_model.embeddings.position_embedding.weight",
    "clip_g.text_model.embeddings.position_embedding.weight",
)


def _core_detect(header):
    """Ask core's own detect_te_model, fed from the header. None if unavailable."""
    try:
        import comfy.sd  # noqa: F401  (runtime-only, never at import time)
        detect = getattr(comfy.sd, "detect_te_model", None)
        if detect is None:
            return None
        proxy = {k: _ShapeOnly(v) for k, v in header.items()}
        result = detect(proxy)
        if result is None:
            return None
        return _TEMODEL_NAMES.get(getattr(result, "name", ""), None)
    except Exception:
        return None


def _local_detect(header):
    for key, shape0, name in _LOCAL_RULES:
        shape = header.get(key)
        if shape is None:
            continue
        if shape0 is None:
            return name
        if shape and int(shape[0]) == shape0:
            return name
    return None


def identify_header(header):
    """Identify a text encoder from its header map.

    Returns a dict:
        kind      -- short name ("clip_l", "t5xxl", ...) or None if unknown
        source    -- "core" | "local" | "none": WHICH judgement answered
        long_ctx  -- position-embedding rows when known (77 classic, 248 long)
        is_long   -- True when long_ctx is present and above the classic 77
        spiece    -- True when a spiece_model entry is present (umt5/wan marker)
    """
    if not header:
        return {"kind": None, "source": "none", "long_ctx": None,
                "is_long": False, "spiece": False}

    kind = _core_detect(header)
    source = "core"
    if kind is None:
        kind = _local_detect(header)
        source = "local" if kind is not None else "none"

    long_ctx = None
    for key in _POS_EMBED_KEYS:
        shape = header.get(key)
        if shape:
            long_ctx = int(shape[0])
            break

    return {
        "kind": kind,
        "source": source,
        "long_ctx": long_ctx,
        "is_long": bool(long_ctx is not None and long_ctx > 77),
        "spiece": "spiece_model" in header,
    }


def identify_file(path):
    """identify_header() for a path. Unreadable/non-safetensors -> unknown."""
    if not path or not os.path.isfile(path):
        return {"kind": None, "source": "none", "long_ctx": None,
                "is_long": False, "spiece": False}
    return identify_header(read_safetensors_header(path))


def describe(ident, filename=""):
    """Short human-readable form for the info readout / error text."""
    kind = ident.get("kind")
    if kind is None:
        return "%s: unrecognised" % (filename or "slot")
    txt = kind
    if ident.get("is_long"):
        txt = "long %s (%d ctx)" % (kind, ident.get("long_ctx") or 0)
    if ident.get("spiece"):
        txt += " +spiece"
    return "%s: %s" % (filename, txt) if filename else txt


# ---------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------
#
# Taken from core's own loader DESCRIPTION strings (CLIPLoader / DualCLIPLoader /
# TripleCLIPLoader / QuadrupleCLIPLoader, read 2026-07-23), NOT from memory.
#
#   need    -- kinds that must ALL be present
#   any_of  -- at least one of these must be present
#   choose  -- the whole slate must be drawn from this set (exact count implied)
#
# A recipe absent from this table is simply not checked. The rule throughout:
# we only ever refuse something we can actually PROVE wrong.

_RECIPES = {
    ("sdxl", 2): {"need": ("clip_l", "clip_g")},
    ("sd3", 2): {"choose": ("clip_l", "clip_g", "t5xxl")},
    ("sd3", 3): {"need": ("clip_l", "clip_g", "t5xxl")},
    ("flux", 2): {"need": ("clip_l", "t5xxl")},
    ("hidream", 2): {"any_of": ("t5xxl", "llama3_8")},
    ("hidream", 4): {"need": ("clip_l", "clip_g", "t5xxl", "llama3_8")},
    ("hunyuan_video", 2): {"need": ("clip_l", "llama3_8")},
    ("hunyuan_image", 2): {"need": ("qwen25_7b", "byt5_small")},
    ("newbie", 2): {"need": ("gemma3_4b", "jina_clip_2")},
}


def recipe_for(clip_type, count):
    return _RECIPES.get((str(clip_type), int(count)))


def check_recipe(clip_type, idents):
    """Check a slate of identified encoders against the recipe.

    Returns (ok, message). ok=False means the load would be wrong and should be
    refused BEFORE anything is read from disk.

    THE RULE: an UNKNOWN slot never fails the check. We cannot read GGUF headers
    and we cannot promise core will never learn an encoder we have not heard of,
    so an unreadable slot buys the benefit of the doubt and says so. Refusing on
    ignorance would turn this from a safety net into an obstacle.
    """
    recipe = recipe_for(clip_type, len(idents))
    if recipe is None:
        return True, ""

    kinds = [i.get("kind") for i in idents]
    if any(k is None for k in kinds):
        return True, "compatibility check skipped (a slot could not be identified)"

    present = list(kinds)

    need = recipe.get("need")
    if need:
        missing = []
        pool = list(present)
        for want in need:
            if want in pool:
                pool.remove(want)
            else:
                missing.append(want)
        if missing:
            return False, ("type '%s' with %d encoders needs %s -- missing %s; "
                           "loaded slots are %s"
                           % (clip_type, len(idents), " + ".join(need),
                              " + ".join(missing), ", ".join(present)))
        return True, ""

    any_of = recipe.get("any_of")
    if any_of:
        if not any(k in any_of for k in present):
            return False, ("type '%s' with %d encoders needs at least one of %s "
                           "-- loaded slots are %s"
                           % (clip_type, len(idents), " or ".join(any_of),
                              ", ".join(present)))
        return True, ""

    choose = recipe.get("choose")
    if choose:
        bad = [k for k in present if k not in choose]
        if bad:
            return False, ("type '%s' with %d encoders draws from %s -- %s does "
                           "not belong; loaded slots are %s"
                           % (clip_type, len(idents), " / ".join(choose),
                              ", ".join(bad), ", ".join(present)))
        if len(set(present)) != len(present):
            return False, ("type '%s' needs %d DIFFERENT encoders -- got %s twice"
                           % (clip_type, len(idents),
                              [k for k in present if present.count(k) > 1][0]))
        return True, ""

    return True, ""


# ---------------------------------------------------------------------------
# how many files select which family -- MEASURED, not assumed
# ---------------------------------------------------------------------------
#
# comfy.sd.load_text_encoder_state_dicts branches on len(clip_data) FIRST:
#   1 file  -- clip_type selects among ~25 single-encoder families
#   2 files -- clip_type selects among the dual families
#   3 files -- clip_type is NOT CONSULTED AT ALL; always sd3
#   4 files -- clip_type is NOT CONSULTED AT ALL; always hidream
#
# That is exactly why core's TripleCLIPLoader and QuadrupleCLIPLoader have no
# `type` widget: at those counts the count IS the type. A node that offers a
# type field for 3 or 4 files would be offering a control that does nothing, so
# ours says plainly that the field is being ignored instead of implying it was
# honoured.

_FIXED_BY_COUNT = {3: "sd3", 4: "hidream"}


def forced_type_for_count(count):
    """The type core will use regardless of the widget, or None if the widget rules."""
    return _FIXED_BY_COUNT.get(int(count))
