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


# ── Sampling group (v365) ───────────────────────────
# Polyhedron Sampler + Polyhedron CLIP Text Encode. Their OWN group, following
# the Media I/O rule above: the blocks before this one stay the Stack's
# untouched registration. The Sampler delegates the sampling loop to ComfyUI
# core (comfy.sample / comfy.samplers / latent_preview, all always present) and
# keeps its server routes in its own module (nodes/ph_sampler_routes.py), so
# neither uls_routes.py nor ph_media_routes.py is re-opened. The encoder reuses
# the CORE encoder plus the Token Counter's own _count_tokens (one truth).
_SAMPLER_OK = _CTE_OK = False
try:
    from .nodes.uls_sampler import ULSSampler
    _SAMPLER_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron Sampler unavailable — import failed: {e!r}")
    print("[PLS]   (usually a changed ComfyUI Core sampling API). Other nodes still load.")
try:
    from .nodes.ph_clip_encode import ULSCLIPTextEncode
    _CTE_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron CLIP Text Encode unavailable — import failed: {e!r}")
    print("[PLS]   (usually a changed ComfyUI Core node API). Other nodes still load.")

# ── Upscale / Interpolate group (v368) ──────────────────────
# Three nodes, one group, its own files. Power Upscale owns the shared
# helpers (_resolve_input / _build_video / _MODEL_UPSCALER / _MuteInfoLogs)
# and the clean-room tile geometry in nodes/uls_tile_math.py; Fast Upscale
# imports them from there, so there is one source of truth. Interpolate
# carries the vendored MIT IFNet in nodes/vfi/ (see NOTICE in that package).
# None of the three registers a server route — Power Upscale reports tile
# progress through PromptServer.send_sync, which needs no endpoint, so
# uls_routes.py, ph_media_routes.py and ph_sampler_routes.py all stay shut.
# Each import is isolated: a changed Core API must not abort the pack.
_PUP_OK = _FUP_OK = _INTERP_OK = False
try:
    from .nodes.ph_power_upscale import ULSPowerUpscale
    _PUP_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron Power Upscale unavailable — import failed: {e!r}")
    print("[PLS]   (usually a changed ComfyUI Core sampling/upscale API). Other nodes still load.")
try:
    from .nodes.ph_fast_upscale import ULSFastUpscale
    _FUP_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron Fast Upscale unavailable — import failed: {e!r}")
    print("[PLS]   (it shares the Power Upscale helpers; usually the same Core cause). Other nodes still load.")
try:
    from .nodes.ph_interpolate import ULSInterpolate
    _INTERP_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron Interpolate unavailable — import failed: {e!r}")
    print("[PLS]   (usually torch or a changed Core device API). Other nodes still load.")

# -- Attention / grading group (v372) --------------------------
# Three nodes, one group, its own files. Attention patches the model's
# attention backend, NAG applies Normalized Attention Guidance to a
# conditioning, and Filter is the colour-grading node with a live preview.
#
# ROUTES, MEASURED BEFORE THE CUT: Attention and NAG register none and carry
# no frontend at all. Filter DOES need three -- its live preview reads the
# same .cube file the backend grades with, and presets load/save -- so they
# live in their OWN module nodes/ph_filter_routes.py, following the rule the
# Media Loader and Sampler set: one new module, one registration call, and
# uls_routes.py / ph_media_routes.py / ph_sampler_routes.py all stay shut.
# The console therefore still reports the same 28 media and 6 sampler paths,
# plus 6 filter paths (3 routes x bare + /api alias).
#
# Each import is isolated: a changed Core API must not abort the pack.
_ATTN_OK = _NAG_OK = _FILTER_OK = False
try:
    from .nodes.ph_attention import ULSAttention
    _ATTN_OK = True
except Exception as e:
    print(f"[PLS] \u2717 Polyhedron Attention unavailable \u2014 import failed: {e!r}")
    print("[PLS]   (usually torch or a changed Core attention API). Other nodes still load.")
try:
    from .nodes.ph_nag import ULSNag
    _NAG_OK = True
except Exception as e:
    print(f"[PLS] \u2717 Polyhedron NAG unavailable \u2014 import failed: {e!r}")
    print("[PLS]   (usually a changed comfy.ldm attention API). Other nodes still load.")
try:
    from .nodes.ph_filter import ULSFilter
    _FILTER_OK = True
except Exception as e:
    print(f"[PLS] \u2717 Polyhedron Filter unavailable \u2014 import failed: {e!r}")
    print("[PLS]   (usually numpy). Other nodes still load.")
# ULSAudioStretch came along because ph_interpolate's audio_mode="stretch"
# imports its machinery -- ONE source for the retime, shared between the
# Interpolate node and this free-standing one. Registering it is the honest
# option: the file is in the tree either way, it has its own guard, and it
# opens no route. Leaving the class unregistered would be dead code that the
# serialisation scanner reports on every run.
_ASTRETCH_OK = False
try:
    from .nodes.ph_audio_stretch import ULSAudioStretch
    _ASTRETCH_OK = True
except Exception as e:
    print(f"[PLS] \u2717 Polyhedron Audio Stretch unavailable \u2014 import failed: {e!r}")
    print("[PLS]   (usually torch or PyAV). Other nodes still load.")

# ── Workflow essentials group (v371) ───────────────────────
# Thirteen nodes, one group, its own files. These are the pieces a real
# Polyhedron graph is wired FROM: loaders (model / CLIP / VAE / upscale
# model), the codec, seed and int sources, the switch pair, empty latent,
# media info, the MiniMax reference builder and the note.
#
# NOT ONE OF THEM REGISTERS A SERVER ROUTE. Measured before the cut: no
# routes.get/post in any of the nine carrier modules, and no fetch() in any
# of their frontends. So uls_routes.py, ph_media_routes.py and
# ph_sampler_routes.py all stay shut and the console keeps reporting the
# same 28 media paths and 6 sampler paths as v370.
#
# ph_basics carries four of the thirteen (Load Model / Load CLIP / Load VAE /
# Seed) and pulls in ph_te_detect and uls_noise; ph_empty_latent pulls in
# uls_latent_math and uls_noise; ph_upscale_loader borrows the model card
# from the Power Upscale node that is already here. Each import is isolated:
# a changed Core API must not abort the pack.
_BASICS_OK = _SWITCH_OK = _INT_OK = _ELAT_OK = False
_MINFO_OK = _MMREF_OK = _NOTE_OK = _VAE_OK = _UPLOAD_OK = False
try:
    from .nodes.ph_basics import ULSLoadModel, ULSLoadCLIP, ULSLoadVAE, ULSSeed
    _BASICS_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron loaders/seed unavailable — import failed: {e!r}")
    print("[PLS]   (usually a changed ComfyUI Core sd/utils API). Other nodes still load.")
try:
    from .nodes.ph_switch import ULSAnySwitch, ULSAnySwitchInv
    _SWITCH_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron Switch unavailable — import failed: {e!r}")
try:
    from .nodes.ph_int import ULSInt
    _INT_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron Int unavailable — import failed: {e!r}")
try:
    from .nodes.ph_empty_latent import ULSEmptyLatent
    _ELAT_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron Empty Latent unavailable — import failed: {e!r}")
try:
    from .nodes.ph_media_info import ULSMediaInfo
    _MINFO_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron Media Info unavailable — import failed: {e!r}")
try:
    from .nodes.ph_minimax_ref import ULSMiniMaxReference
    _MMREF_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron MiniMax Reference unavailable — import failed: {e!r}")
try:
    from .nodes.ph_note import ULSNote
    _NOTE_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron Note unavailable — import failed: {e!r}")
try:
    from .nodes.ph_vae import ULSVAE
    _VAE_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron VAE Codec unavailable — import failed: {e!r}")
try:
    from .nodes.ph_upscale_loader import ULSLoadUpscaleModel
    _UPLOAD_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron Load Upscale Model unavailable — import failed: {e!r}")


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

if _SAMPLER_OK:
    NODE_CLASS_MAPPINGS["ULSSampler"] = ULSSampler
    NODE_DISPLAY_NAME_MAPPINGS["ULSSampler"] = "⬡ Polyhedron Sampler"

if _CTE_OK:
    NODE_CLASS_MAPPINGS["ULSCLIPTextEncode"] = ULSCLIPTextEncode
    NODE_DISPLAY_NAME_MAPPINGS["ULSCLIPTextEncode"] = "⬡ Polyhedron CLIP Text Encode"

if _PUP_OK:
    NODE_CLASS_MAPPINGS["ULSPowerUpscale"] = ULSPowerUpscale
    NODE_DISPLAY_NAME_MAPPINGS["ULSPowerUpscale"] = "⬡ Polyhedron Power Upscale"

if _FUP_OK:
    NODE_CLASS_MAPPINGS["ULSFastUpscale"] = ULSFastUpscale
    NODE_DISPLAY_NAME_MAPPINGS["ULSFastUpscale"] = "⬡ Polyhedron Fast Upscale"

if _INTERP_OK:
    NODE_CLASS_MAPPINGS["ULSInterpolate"] = ULSInterpolate
    NODE_DISPLAY_NAME_MAPPINGS["ULSInterpolate"] = "⬡ Polyhedron Interpolate"

# Attention / grading group (v372)
if _ATTN_OK:
    NODE_CLASS_MAPPINGS["ULSAttention"] = ULSAttention
    NODE_DISPLAY_NAME_MAPPINGS["ULSAttention"] = "⬡ Polyhedron Attention"
if _NAG_OK:
    NODE_CLASS_MAPPINGS["ULSNag"] = ULSNag
    NODE_DISPLAY_NAME_MAPPINGS["ULSNag"] = "⬡ Polyhedron NAG"
if _FILTER_OK:
    NODE_CLASS_MAPPINGS["ULSFilter"] = ULSFilter
    NODE_DISPLAY_NAME_MAPPINGS["ULSFilter"] = "⬡ Polyhedron Filter"
if _ASTRETCH_OK:
    NODE_CLASS_MAPPINGS["ULSAudioStretch"] = ULSAudioStretch
    NODE_DISPLAY_NAME_MAPPINGS["ULSAudioStretch"] = "⬡ Polyhedron Audio Stretch"

# ── Workflow essentials group (v371) ───────────────────────
# Each behind its own _OK flag, appended at the END of the mappings so no
# existing registration moves.
if _BASICS_OK:
    NODE_CLASS_MAPPINGS["ULSLoadModel"] = ULSLoadModel
    NODE_DISPLAY_NAME_MAPPINGS["ULSLoadModel"] = "⬡ Polyhedron Load Model"
    NODE_CLASS_MAPPINGS["ULSLoadCLIP"] = ULSLoadCLIP
    NODE_DISPLAY_NAME_MAPPINGS["ULSLoadCLIP"] = "⬡ Polyhedron Load CLIP"
    NODE_CLASS_MAPPINGS["ULSLoadVAE"] = ULSLoadVAE
    NODE_DISPLAY_NAME_MAPPINGS["ULSLoadVAE"] = "⬡ Polyhedron Load VAE"
    NODE_CLASS_MAPPINGS["ULSSeed"] = ULSSeed
    NODE_DISPLAY_NAME_MAPPINGS["ULSSeed"] = "⬡ Polyhedron Seed"
if _UPLOAD_OK:
    NODE_CLASS_MAPPINGS["ULSLoadUpscaleModel"] = ULSLoadUpscaleModel
    NODE_DISPLAY_NAME_MAPPINGS["ULSLoadUpscaleModel"] = "⬡ Polyhedron Load Upscale Model"
if _VAE_OK:
    NODE_CLASS_MAPPINGS["ULSVAE"] = ULSVAE
    NODE_DISPLAY_NAME_MAPPINGS["ULSVAE"] = "⬡ Polyhedron VAE Codec"
if _ELAT_OK:
    NODE_CLASS_MAPPINGS["ULSEmptyLatent"] = ULSEmptyLatent
    NODE_DISPLAY_NAME_MAPPINGS["ULSEmptyLatent"] = "⬡ Polyhedron Empty Latent"
if _SWITCH_OK:
    NODE_CLASS_MAPPINGS["ULSAnySwitch"] = ULSAnySwitch
    NODE_DISPLAY_NAME_MAPPINGS["ULSAnySwitch"] = "⬡ Polyhedron Switch"
    NODE_CLASS_MAPPINGS["ULSAnySwitchInv"] = ULSAnySwitchInv
    NODE_DISPLAY_NAME_MAPPINGS["ULSAnySwitchInv"] = "⬡ Polyhedron Switch Inverse"
if _INT_OK:
    NODE_CLASS_MAPPINGS["ULSInt"] = ULSInt
    NODE_DISPLAY_NAME_MAPPINGS["ULSInt"] = "⬡ Polyhedron Int"
if _MINFO_OK:
    NODE_CLASS_MAPPINGS["ULSMediaInfo"] = ULSMediaInfo
    NODE_DISPLAY_NAME_MAPPINGS["ULSMediaInfo"] = "⬡ Polyhedron Media Info"
if _MMREF_OK:
    NODE_CLASS_MAPPINGS["ULSMiniMaxReference"] = ULSMiniMaxReference
    NODE_DISPLAY_NAME_MAPPINGS["ULSMiniMaxReference"] = "⬡ Polyhedron MiniMax Reference"
if _NOTE_OK:
    NODE_CLASS_MAPPINGS["ULSNote"] = ULSNote
    NODE_DISPLAY_NAME_MAPPINGS["ULSNote"] = "⬡ Polyhedron Note"

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

# Polyhedron Sampler routes: again a separate module and a separate call, so
# neither the Stack's route file nor the Media Loader's is touched.
if _SAMPLER_OK:
    try:
        from .nodes.ph_sampler_routes import register_sampler_routes
        register_sampler_routes()
    except Exception as e:
        print(f"[PLS] ⚠ Sampler routes not registered: {e}")

# Polyhedron Filter routes: fourth module, fourth call. The Filter's live
# preview must read the SAME .cube the backend grades with, so it needs an
# endpoint -- but not at the price of re-opening a shared route file.
if _FILTER_OK:
    try:
        from .nodes.ph_filter_routes import register_filter_routes
        register_filter_routes()
    except Exception as e:
        print(f"[PLS] \u26a0 Filter routes not registered: {e}")

_node_count = len(NODE_CLASS_MAPPINGS)
_bridge_str = "✅" if _BRIDGE_OK else "⚠ unavailable"
print(f"""
⚡ ============================================================
   Polyhedron Suite  v372
   {_node_count} Nodes  |  Bridge: {_bridge_str}
⚡ ============================================================
""")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
