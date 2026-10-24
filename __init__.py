"""
Polyhedron Suite
"""
# Every node group imports on its own: a failed import (usually a changed ComfyUI API)
# logs one line and skips that group; the rest of the pack still registers.

# -- LoRA Stack group
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
                                            ULSUniversalSigmaCurve, ULSSigmaList)
    _SIGMA_OK = True
except Exception as e:
    print(f"[PLS] ✗ Sigma Schedule nodes unavailable — import failed: {e!r}")

# -- Media I/O: Media Loader, Save
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

# -- Sampling: Sampler, CLIP Text Encode
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

# -- Upscale / Interpolate (the RIFE network is vendored in nodes/vfi, see its NOTICE)
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

# -- Attention, NAG, Filter, Audio Stretch (Interpolate's stretch mode shares its retime)
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
_ASTRETCH_OK = False
try:
    from .nodes.ph_audio_stretch import ULSAudioStretch
    _ASTRETCH_OK = True
except Exception as e:
    print(f"[PLS] \u2717 Polyhedron Audio Stretch unavailable \u2014 import failed: {e!r}")
    print("[PLS]   (usually torch or PyAV). Other nodes still load.")

# -- Workflow essentials: loaders, VAE codec, seed, int, switches, empty latent, media info,
#    MiniMax Reference, Reference Board, Show Text, Note
_BASICS_OK = _SWITCH_OK = _INT_OK = _ELAT_OK = False
_MINFO_OK = _MMREF_OK = _NOTE_OK = _VAE_OK = _UPLOAD_OK = False
_RBOARD_OK = False
_SHOWTEXT_OK = False
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
    from .nodes.ph_show_text import ULSShowText
    _SHOWTEXT_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron Show Text unavailable — import failed: {e!r}")
try:
    from .nodes.ph_minimax_continue import ULSMiniMaxContinue   # v392: the fox templates' H3 chain
    _MMCONT_OK = True
except Exception as e:
    _MMCONT_OK = False
    print(f"[PLS] ✗ Polyhedron MiniMax Continue unavailable — import failed: {e!r}")
try:
    from .nodes.ph_minimax_keyframes import ULSMiniMaxKeyframes   # v392: the fox templates' H3 chain
    _MMKEYS_OK = True
except Exception as e:
    _MMKEYS_OK = False
    print(f"[PLS] ✗ Polyhedron MiniMax Keyframes unavailable — import failed: {e!r}")
try:
    from .nodes.ph_minimax_audio_lock import ULSMiniMaxAudioLock   # v392: the fox templates' H3 chain
    _MMALOCK_OK = True
except Exception as e:
    _MMALOCK_OK = False
    print(f"[PLS] ✗ Polyhedron MiniMax Audio Lock unavailable — import failed: {e!r}")
try:
    from .nodes.ph_minimax_drift import ULSMiniMaxDriftControl   # v392: the fox templates' H3 chain
    _MMDRIFT_OK = True
except Exception as e:
    _MMDRIFT_OK = False
    print(f"[PLS] ✗ Polyhedron MiniMax Drift Control unavailable — import failed: {e!r}")
try:
    from .nodes.ph_h3_quality import ULSH3LatentUpscale   # v392: the fox templates' H3 chain
    _H3LUP_OK = True
except Exception as e:
    _H3LUP_OK = False
    print(f"[PLS] ✗ Polyhedron H3 Latent Upscale unavailable — import failed: {e!r}")
try:
    from .nodes.ph_minimax_pass_hold import ULSMiniMaxSecondPassHold   # v392: the fox templates' H3 chain
    _MMHOLD_OK = True
except Exception as e:
    _MMHOLD_OK = False
    print(f"[PLS] ✗ Polyhedron MiniMax Second Pass Hold unavailable — import failed: {e!r}")
try:
    from .nodes.ph_reference_board import ULSReferenceBoard
    _RBOARD_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron Reference Board unavailable — import failed: {e!r}")
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
# -- Cine Timeline. The cine helper modules it imports ship with it; none of their own
#    nodes is registered in this pack.
_TIMELINE_OK = False
try:
    from .nodes.ph_cine_timeline import ULSCineTimeline
    _TIMELINE_OK = True
except Exception as e:
    print(f"[PLS] ✗ Polyhedron Cine Timeline unavailable — import failed: {e!r}")

# -- Wan Bridge (depends on WanVideoWrapper internals)
_BRIDGE_OK = False
try:
    from .nodes.wan_model_bridge import ULSWanBridge, ULSWanBridgeReverse
    _BRIDGE_OK = True
except Exception as e:
    print(f"[PLS] ⚠ Bridge nodes failed to load: {e}")
    print("[PLS]   Bridge (MODEL ↔ WANVIDEOMODEL) will be unavailable this session.")

# -- V3 schema. ComfyUI registers a pack as V1 (NODE_CLASS_MAPPINGS) OR V3 (comfy_entrypoint),
#    and this pack always has V1 mappings -- so the V3 classes go through NODE_CLASS_MAPPINGS too
#    (a V3 io.ComfyNode carries the full V1 interface). Same node_id either way; without
#    comfy_api.latest the legacy class is registered.
_V3_OK = False
V3_NODE_CLASSES = {}
try:
    from .nodes.uls_v3_extension import V3_NODE_CLASSES
    _V3_OK = True
except Exception as e:
    print(f"[PLS] ⚠ V3 nodes unavailable ({e!r}) — using legacy registration")

# -- Registration
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
    NODE_CLASS_MAPPINGS["ULSInspector"] = V3_NODE_CLASSES["ULSInspector"] if _V3_OK else ULSInspector
    NODE_DISPLAY_NAME_MAPPINGS["ULSInspector"] = "⬡ Polyhedron LoRA Inspector"

if _SWITCH_OK:
    NODE_CLASS_MAPPINGS["ULSModelSwitch"] = ULSModelSwitch
    NODE_DISPLAY_NAME_MAPPINGS["ULSModelSwitch"] = "⬡ Polyhedron Select Model Switch"

if _ANALYZER_OK:
    NODE_CLASS_MAPPINGS["ULSResolveInspector"] = V3_NODE_CLASSES["ULSResolveInspector"] if _V3_OK else ULSResolveInspector
    NODE_DISPLAY_NAME_MAPPINGS["ULSResolveInspector"] = "⬡ Polyhedron Merge Analyzer"

if _INFLATE_OK:
    NODE_CLASS_MAPPINGS["ULSWanFrameInflate"] = V3_NODE_CLASSES["ULSWanFrameInflate"] if _V3_OK else ULSWanFrameInflate
    NODE_DISPLAY_NAME_MAPPINGS["ULSWanFrameInflate"] = "⬡ Polyhedron Wan Frame Inflate (T2I LoRA fix)"
    NODE_CLASS_MAPPINGS["ULSImagePickFrame"] = V3_NODE_CLASSES["ULSImagePickFrame"] if _V3_OK else ULSImagePickFrame
    NODE_DISPLAY_NAME_MAPPINGS["ULSImagePickFrame"] = "⬡ Polyhedron Pick Frame"

if _SIGMA_OK:
    NODE_CLASS_MAPPINGS["ULSWanSigmaSchedule"] = ULSWanSigmaSchedule
    NODE_DISPLAY_NAME_MAPPINGS["ULSWanSigmaSchedule"] = "⬡ Polyhedron Noise Schedule [deprecated]"
    NODE_CLASS_MAPPINGS["ULSWanSplitNoiseSchedule"] = V3_NODE_CLASSES["ULSWanSplitNoiseSchedule"] if _V3_OK else ULSWanSplitNoiseSchedule
    NODE_DISPLAY_NAME_MAPPINGS["ULSWanSplitNoiseSchedule"] = "⬡ Polyhedron Dual Sigma Curve"
    NODE_CLASS_MAPPINGS["ULSUniversalSigmaCurve"] = V3_NODE_CLASSES["ULSUniversalSigmaCurve"] if _V3_OK else ULSUniversalSigmaCurve
    NODE_DISPLAY_NAME_MAPPINGS["ULSUniversalSigmaCurve"] = "⬡ Polyhedron Sigma Curve"
    NODE_CLASS_MAPPINGS["ULSSigmaList"] = V3_NODE_CLASSES["ULSSigmaList"] if (_V3_OK and "ULSSigmaList" in V3_NODE_CLASSES) else ULSSigmaList
    NODE_DISPLAY_NAME_MAPPINGS["ULSSigmaList"] = "⬡ Polyhedron Sigma List"

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
if _SHOWTEXT_OK:
    NODE_CLASS_MAPPINGS["ULSShowText"] = ULSShowText
    NODE_DISPLAY_NAME_MAPPINGS["ULSShowText"] = "⬡ Polyhedron Show Text"
if _RBOARD_OK:
    NODE_CLASS_MAPPINGS["ULSReferenceBoard"] = ULSReferenceBoard
    NODE_DISPLAY_NAME_MAPPINGS["ULSReferenceBoard"] = "⬡ Polyhedron Reference Board"
if _NOTE_OK:
    NODE_CLASS_MAPPINGS["ULSNote"] = ULSNote
    NODE_DISPLAY_NAME_MAPPINGS["ULSNote"] = "⬡ Polyhedron Note"
if _TIMELINE_OK:
    NODE_CLASS_MAPPINGS["ULSCineTimeline"] = ULSCineTimeline
    NODE_DISPLAY_NAME_MAPPINGS["ULSCineTimeline"] = "⬡ Polyhedron Cine Timeline"

if _MMCONT_OK:
    NODE_CLASS_MAPPINGS["ULSMiniMaxContinue"] = ULSMiniMaxContinue
    NODE_DISPLAY_NAME_MAPPINGS["ULSMiniMaxContinue"] = "⬡ Polyhedron MiniMax Continue"
if _MMKEYS_OK:
    NODE_CLASS_MAPPINGS["ULSMiniMaxKeyframes"] = ULSMiniMaxKeyframes
    NODE_DISPLAY_NAME_MAPPINGS["ULSMiniMaxKeyframes"] = "⬡ Polyhedron MiniMax Keyframes"
if _MMALOCK_OK:
    NODE_CLASS_MAPPINGS["ULSMiniMaxAudioLock"] = ULSMiniMaxAudioLock
    NODE_DISPLAY_NAME_MAPPINGS["ULSMiniMaxAudioLock"] = "⬡ Polyhedron MiniMax Audio Lock"
if _MMDRIFT_OK:
    NODE_CLASS_MAPPINGS["ULSMiniMaxDriftControl"] = ULSMiniMaxDriftControl
    NODE_DISPLAY_NAME_MAPPINGS["ULSMiniMaxDriftControl"] = "⬡ Polyhedron MiniMax Drift Control"
if _H3LUP_OK:
    NODE_CLASS_MAPPINGS["ULSH3LatentUpscale"] = ULSH3LatentUpscale
    NODE_DISPLAY_NAME_MAPPINGS["ULSH3LatentUpscale"] = "⬡ Polyhedron H3 Latent Upscale"
if _MMHOLD_OK:
    NODE_CLASS_MAPPINGS["ULSMiniMaxSecondPassHold"] = ULSMiniMaxSecondPassHold
    NODE_DISPLAY_NAME_MAPPINGS["ULSMiniMaxSecondPassHold"] = "⬡ Polyhedron MiniMax Second Pass Hold"

WEB_DIRECTORY = "./web/js"

# -- Routes: every group registers its own module, bare path + /api alias
try:
    from .nodes.uls_routes import register_routes
    register_routes()
except Exception as e:
    print(f"[PLS] ⚠ Routes not registered: {e}")

if _MEDIA_OK:
    try:
        from .nodes.ph_media_routes import register_media_routes
        register_media_routes()
    except Exception as e:
        print(f"[PLS] ⚠ Media routes not registered: {e}")

if _SAMPLER_OK:
    try:
        from .nodes.ph_sampler_routes import register_sampler_routes
        register_sampler_routes()
    except Exception as e:
        print(f"[PLS] ⚠ Sampler routes not registered: {e}")

if _FILTER_OK:
    try:
        from .nodes.ph_filter_routes import register_filter_routes
        register_filter_routes()
    except Exception as e:
        print(f"[PLS] \u26a0 Filter routes not registered: {e}")

if _SIGMA_OK:
    try:
        from .nodes.ph_sigma_routes import register_sigma_routes
        register_sigma_routes()
    except Exception as e:
        print(f"[PLS] \u26a0 Sigma routes not registered: {e}")

if _BASICS_OK or _ELAT_OK:
    try:
        from .nodes.ph_basics_routes import register_basics_routes
        register_basics_routes()
    except Exception as e:
        print(f"[PLS] ⚠ Basics routes not registered: {e}")

if _INTERP_OK:
    try:
        from .nodes.ph_world_weights import register_weights_routes
        register_weights_routes()
    except Exception as e:
        print(f"[PLS] ⚠ Weights routes not registered: {e}")

if _TIMELINE_OK:
    try:
        from .nodes.ph_cine_timeline import register_timeline_routes
        register_timeline_routes()
    except Exception as e:
        print(f"[PLS] ⚠ Cine Timeline routes not registered: {e}")

# v394: Control Fit / Control Map / Apply Control (pose, depth, edges as control), Blend (layers, Photoshop modes,
# clips), Transition Check (first -> last frame: motion or dissolve)
_CONTROL_OK = False
try:
    from .nodes.ph_control import ULSControlFit, ULSControlMap, ULSApplyControl
    _CONTROL_OK = True
except Exception as _e:
    print(f"[PolyhedronLoRAStack] Control nodes unavailable: {_e}")

_BLEND_OK = False
try:
    from .nodes.ph_blend import ULSBlend   # v1218: up to three layers over a base, Photoshop modes, clips too
    _BLEND_OK = True
except Exception as _e:
    print(f"[PolyhedronLoRAStack] ULSBlend unavailable: {_e}")

_TRANSITION_OK = False
try:
    from .nodes.ph_transition import ULSTransitionCheck   # v1215: first -> last frame, motion or dissolve
    _TRANSITION_OK = True
except Exception as _e:
    print(f"[PolyhedronLoRAStack] ULSTransitionCheck unavailable: {_e}")

if _BLEND_OK:
    NODE_CLASS_MAPPINGS["ULSBlend"] = ULSBlend
    NODE_DISPLAY_NAME_MAPPINGS["ULSBlend"] = "\u2b21 Polyhedron Blend"
if _TRANSITION_OK:
    NODE_CLASS_MAPPINGS["ULSTransitionCheck"] = ULSTransitionCheck
    NODE_DISPLAY_NAME_MAPPINGS["ULSTransitionCheck"] = "\u2b21 Polyhedron Transition Check"
if _CONTROL_OK:
    NODE_CLASS_MAPPINGS["ULSControlFit"] = ULSControlFit
    NODE_DISPLAY_NAME_MAPPINGS["ULSControlFit"] = "\u2b21 Polyhedron Control Fit"
    NODE_CLASS_MAPPINGS["ULSControlMap"] = ULSControlMap
    NODE_DISPLAY_NAME_MAPPINGS["ULSControlMap"] = "\u2b21 Polyhedron Control Map"
    NODE_CLASS_MAPPINGS["ULSApplyControl"] = ULSApplyControl
    NODE_DISPLAY_NAME_MAPPINGS["ULSApplyControl"] = "\u2b21 Polyhedron Apply Control"

_node_count = len(NODE_CLASS_MAPPINGS)
_bridge_str = "✅" if _BRIDGE_OK else "⚠ unavailable"
print(f"""
⚡ ============================================================
   Polyhedron Suite  v396
   {_node_count} Nodes  |  Bridge: {_bridge_str}
⚡ ============================================================
""")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
