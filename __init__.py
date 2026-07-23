"""
Polyhedron Suite
"""
import importlib.util

def _has(module):
    """True if an importable package is present (no import side effects)."""
    return importlib.util.find_spec(module) is not None

# Pillow / requests are NOT hard runtime requirements of the nodes — both are
# helper-script-only (uls_preview_gen.py / install.py). LoRA-preview decoding
# goes through ComfyUI's core LoadImage, not Pillow; the runtime Civitai fetch
# uses aiohttp, which ComfyUI already ships. Tracked only so a missing one is
# explained, never treated as a hard requirement.
print("[PLS] Checking optional dependencies...")
_HAS_PIL      = _has("PIL")
_HAS_REQUESTS = _has("requests")
if not (_HAS_PIL and _HAS_REQUESTS):
    _missing = ", ".join(n for n, ok in (("Pillow", _HAS_PIL), ("requests", _HAS_REQUESTS)) if not ok)
    print(f"[PLS]   note: {_missing} not installed — only the optional helper scripts need it, not the nodes")

# Node groups load INDEPENDENTLY (v254). A breaking ComfyUI/Core change in one
# group — e.g. comfy.lora moving, which only the Stack/Engine use — must not
# abort the whole pack. Each group is wrapped; on a failed import it logs a
# clear, actionable line and is simply skipped, so the remaining groups still
# register. When everything imports (the normal case) this is behaviour-
# identical to before: same classes, same display names.
_STACK_OK = _SWITCH_OK = _INFLATE_OK = _SIGMA_OK = _ANALYZER_OK = False
try:
    from .nodes.uls_stack_node import UltimateLoraStack, ULSAccelerator, ULSInspector, ULSTokenCounter
    _STACK_OK = True
except Exception as e:
    print(f"[PLS] ✗ Stack / Engine / Inspector / TokenCounter unavailable — import failed: {e!r}")
    print("[PLS]   (usually a changed ComfyUI Core API, e.g. comfy.lora). Other nodes still load.")
try:
    from .nodes.uls_resolve_inspector import ULSResolveInspector
    _ANALYZER_OK = True
except Exception as e:
    print(f"[PLS] ✗ Merge Analyzer unavailable — import failed: {e!r}")
try:
    from .nodes.uls_model_switch import ULSModelSwitch
    _SWITCH_OK = True
except Exception as e:
    print(f"[PLS] ✗ Model Switch unavailable — import failed: {e!r}")
try:
    from .nodes.wan_frame_inflate import ULSWanFrameInflate, ULSImagePickFrame
    _INFLATE_OK = True
except Exception as e:
    print(f"[PLS] ✗ Frame Inflate / Pick Frame unavailable — import failed: {e!r}")
try:
    from .nodes.wan_sigma_schedule import (ULSWanSigmaSchedule, ULSWanSplitNoiseSchedule,
                                            ULSUniversalSigmaCurve)
    _SIGMA_OK = True
except Exception as e:
    print(f"[PLS] ✗ Sigma Schedule nodes unavailable — import failed: {e!r}")

# ── Media I/O group (v362) ──────────────────────────────────
# Media Loader + Save. Registered as their OWN group so the blocks above stay
# the Stack's untouched registration: a further node arrives as one more
# guarded import plus one more mapping entry, nothing else moves. Both nodes
# are pure add-ons -- no Stack node imports them -- and their server routes
# live in their own module (nodes/ph_media_routes.py), so uls_routes.py stays
# upstream's file.
_MEDIA_OK = _SAVE_OK = False
try:
    from .nodes.ph_media_loader import ULSMediaLoader
    _MEDIA_OK = True
except Exception as e:
    print(f"[PLS] ✗ Media Loader unavailable — import failed: {e!r}")
try:
    from .nodes.ph_save import ULSSave
    _SAVE_OK = True
except Exception as e:
    print(f"[PLS] ✗ Save unavailable — import failed: {e!r}")


# Bridge — fragile, depends on ComfyUI/kijai internals. Already isolated.
_BRIDGE_OK = False
try:
    from .nodes.wan_model_bridge import ULSWanBridge, ULSWanBridgeReverse
    _BRIDGE_OK = True
except Exception as e:
    print(f"[PLS] ⚠ Bridge nodes failed to load: {e}")
    print("[PLS]   Bridge (MODEL ↔ WANVIDEOMODEL) will be unavailable this session.")


# --- V3 schema (Nodes 2.0) -------------------------------------------------
# ComfyUI's loader registers a pack as EITHER V1 (NODE_CLASS_MAPPINGS) OR V3
# (comfy_entrypoint): nodes.py processes NODE_CLASS_MAPPINGS and `return`s True
# BEFORE the `elif comfy_entrypoint` branch. Since this pack always exports a
# non-empty NODE_CLASS_MAPPINGS (the legacy nodes), a comfy_entrypoint would be
# silently ignored. So the V3 nodes are registered through NODE_CLASS_MAPPINGS
# too: a V3 io.ComfyNode exposes the full V1 interface (INPUT_TYPES/RETURN_TYPES/
# FUNCTION/CATEGORY via @classproperty) and ComfyUI unwraps its NodeOutput, so a
# V3 class is a drop-in for the V1 path. V3_NODE_CLASSES maps node_id -> V3 class.
# Guarded like every other import: if comfy_api.latest is missing (older ComfyUI)
# this import fails, _V3_OK stays False, and the proven legacy node is registered
# for every migrated node below so nothing disappears. Each V3 node uses the SAME
# node_id as its legacy key, so V3 and legacy are drop-in interchangeable.
# Migrated: Pick Frame, Wan Frame Inflate (v351/v352); LoRA Inspector, Merge
# Analyzer, Dual Sigma Curve, Sigma Curve (v353).
_V3_OK = False
V3_NODE_CLASSES = {}
try:
    from .nodes.uls_v3_extension import V3_NODE_CLASSES  # noqa: F811
    _V3_OK = True
except Exception as e:
    print(f"[PLS] ⚠ V3 nodes unavailable ({e!r}) — using legacy registration")


NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

if _STACK_OK:
    NODE_CLASS_MAPPINGS.update({
        "UltimateLoraStack": UltimateLoraStack,
        "ULSAccelerator":    ULSAccelerator,
        "ULSTokenCounter":   ULSTokenCounter,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "UltimateLoraStack": "⬡ Polyhedron LoRA Stack",
        "ULSAccelerator":    "⬡ Polyhedron LoRA Engine",
        "ULSTokenCounter":   "⬡ Polyhedron Token Counter",
    })
    # ULSInspector: V3 when comfy_api is available, else proven legacy. Both go
    # through NODE_CLASS_MAPPINGS (the path ComfyUI's loader actually processes).
    NODE_CLASS_MAPPINGS["ULSInspector"] = V3_NODE_CLASSES["ULSInspector"] if _V3_OK else ULSInspector
    NODE_DISPLAY_NAME_MAPPINGS["ULSInspector"] = "⬡ Polyhedron LoRA Inspector"

if _SWITCH_OK:
    NODE_CLASS_MAPPINGS["ULSModelSwitch"] = ULSModelSwitch
    NODE_DISPLAY_NAME_MAPPINGS["ULSModelSwitch"] = "⬡ Polyhedron Select Model Switch"

if _ANALYZER_OK:
    # ULSResolveInspector: V3 when available, else legacy.
    NODE_CLASS_MAPPINGS["ULSResolveInspector"] = V3_NODE_CLASSES["ULSResolveInspector"] if _V3_OK else ULSResolveInspector
    NODE_DISPLAY_NAME_MAPPINGS["ULSResolveInspector"] = "⬡ Polyhedron Merge Analyzer"

if _INFLATE_OK:
    # Frame Inflate and Pick Frame: V3 when comfy_api is available, else legacy.
    NODE_CLASS_MAPPINGS["ULSWanFrameInflate"] = V3_NODE_CLASSES["ULSWanFrameInflate"] if _V3_OK else ULSWanFrameInflate
    NODE_DISPLAY_NAME_MAPPINGS["ULSWanFrameInflate"] = "⬡ Polyhedron Wan Frame Inflate (T2I LoRA fix)"
    NODE_CLASS_MAPPINGS["ULSImagePickFrame"] = V3_NODE_CLASSES["ULSImagePickFrame"] if _V3_OK else ULSImagePickFrame
    NODE_DISPLAY_NAME_MAPPINGS["ULSImagePickFrame"] = "⬡ Polyhedron Pick Frame"

if _SIGMA_OK:
    NODE_CLASS_MAPPINGS["ULSWanSigmaSchedule"] = ULSWanSigmaSchedule
    NODE_DISPLAY_NAME_MAPPINGS["ULSWanSigmaSchedule"] = "⬡ Polyhedron Noise Schedule [deprecated]"
    # Dual Sigma Curve + Sigma Curve: V3 when available, else legacy.
    NODE_CLASS_MAPPINGS["ULSWanSplitNoiseSchedule"] = V3_NODE_CLASSES["ULSWanSplitNoiseSchedule"] if _V3_OK else ULSWanSplitNoiseSchedule
    NODE_DISPLAY_NAME_MAPPINGS["ULSWanSplitNoiseSchedule"] = "⬡ Polyhedron Dual Sigma Curve"
    NODE_CLASS_MAPPINGS["ULSUniversalSigmaCurve"] = V3_NODE_CLASSES["ULSUniversalSigmaCurve"] if _V3_OK else ULSUniversalSigmaCurve
    NODE_DISPLAY_NAME_MAPPINGS["ULSUniversalSigmaCurve"] = "⬡ Polyhedron Sigma Curve"

if _BRIDGE_OK:
    NODE_CLASS_MAPPINGS["ULSWanBridge"]        = ULSWanBridge
    NODE_CLASS_MAPPINGS["ULSWanBridgeReverse"] = ULSWanBridgeReverse
    NODE_DISPLAY_NAME_MAPPINGS["ULSWanBridge"]        = "⬡ Polyhedron Wan Bridge (MODEL → WANVIDEOMODEL)"
    NODE_DISPLAY_NAME_MAPPINGS["ULSWanBridgeReverse"] = "⬡ Polyhedron Wan Bridge (WANVIDEOMODEL → MODEL)"

if _MEDIA_OK:
    NODE_CLASS_MAPPINGS["ULSMediaLoader"] = ULSMediaLoader
    NODE_DISPLAY_NAME_MAPPINGS["ULSMediaLoader"] = "⬡ Polyhedron Media Loader"

if _SAVE_OK:
    NODE_CLASS_MAPPINGS["ULSSave"] = ULSSave
    NODE_DISPLAY_NAME_MAPPINGS["ULSSave"] = "⬡ Polyhedron Save"

WEB_DIRECTORY = "./web/js"

try:
    from .nodes.uls_routes import register_routes
    register_routes()
except Exception as e:
    print(f"[PLS] ⚠ Routes not registered: {e}")

# Media Loader routes: separate module, separate call. The Stack's route file
# is never touched, so an upstream refresh of uls_routes.py cannot break this.
if _MEDIA_OK:
    try:
        from .nodes.ph_media_routes import register_media_routes
        register_media_routes()
    except Exception as e:
        print(f"[PLS] ⚠ Media routes not registered: {e}")

_node_count = len(NODE_CLASS_MAPPINGS)
_bridge_str = "✅" if _BRIDGE_OK else "⚠ unavailable"
print(f"""
⚡ ============================================================
   Polyhedron Suite  v364
   {_node_count} Nodes  |  Bridge: {_bridge_str}
⚡ ============================================================
""")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
