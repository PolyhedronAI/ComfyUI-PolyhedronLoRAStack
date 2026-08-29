"""
Polyhedron Attention  (ph_attention)
====================================
Chooses which attention kernel the diffusion model runs on. Stage A1 of the
attention line: one node, one dropdown, honest detection. Replaces the two
"Patch Sage Attention KJ" nodes in Frank's graph, and is the rail that the
later per-step switching (stage A3) will ride on.

WE WRITE NO KERNELS. ComfyUI already carries the whole mechanism; almost
nobody uses it. comfy/ldm/modules/attention.py holds a registry
(REGISTERED_ATTENTION_FUNCTIONS + register/get_attention_function) and wraps
every backend in @wrap_attn, which -- before dispatching -- looks for
transformer_options["optimized_attention_override"] and hands the call to it.
This node only decides what goes in that slot.


WHY THIS EXISTS, MEASURED ON FRANK'S OWN CARD (RTX 5060 Ti, sm_120, 05.08.)
--------------------------------------------------------------------------
The probe ran every backend on Wan 2.2 geometry (40 heads, head_dim 128) at
seq 12288, plus an accuracy pass at seq 4096 against a head-by-head FP32
reference. bfloat16 throughout.

    backend                 ms      max deviation
    torch SDPA (default)  157.96          0.0005
    SDPA mem-efficient    158.10          0.0005
    SDPA cudnn             62.36          0.0006
    sage fp16 cuda         FAILS   "no kernel image is available"
    sage fp16 triton       33.34          0.0117
    sage fp8 cuda          26.17          0.0101
    sage fp8 cuda++        23.74          0.0091
    sage auto              23.08          0.0111

Four things came out of that, and all four are built into this file.

  1. THE DROPDOWN MUST NOT OFFER WHAT CANNOT RUN. `sage fp16 cuda` IMPORTS
     cleanly and then dies at kernel launch, because the wheel carries no
     fp16-CUDA kernels for sm_120. An import check would have listed it. So
     availability is settled by actually launching a tiny kernel once per
     process (see _probe), not by importing a name.

  2. THE RANKING IS NOT UNIVERSAL. On image geometry (24 heads, seq 4608)
     the order flips: fp16 triton is the MOST accurate sage mode there
     (0.0037) while fp8 cuda++ sits at 0.0083. One global setting cannot be
     right for both a video model and an image model -- which is the whole
     argument for a per-model node instead of a start-up flag.

  3. SPEED IS NOT THE HEADLINE FOR VIDEO. At Frank's real size -- 768x768,
     65 frames, 39168 tokens -- attention is about 59% of the arithmetic but
     only ~18% of the wall clock, because sage runs it in int8. Trading
     33.34 ms for 23.74 ms per call saves ~4 s of a ~55 s step. Real, small,
     and worth saying out loud rather than overselling.

  4. `auto` IS NOT A SETTING, IT IS A DELEGATION. It measured fastest by 3%
     and less accurate than fp8 cuda++, and its choice may change with the
     next wheel. In a tree that uses byte-identity as evidence, an explicit
     mode beats a self-adjusting one. It stays in the list; it is not the
     default.


THE THREE TRAPS, ALL PAID FOR ALREADY
-------------------------------------
  A. NEVER PUT None IN THE SLOT. Core calls
     transformer_options["optimized_attention_override"](func, ...) with no
     None-check, so a None there is "NoneType is not callable" mid-run. The
     key is either absent or a callable -- nothing else. Our own
     wan_model_bridge.py learned this and even strips non-callables back out.

  B. SWALLOW **kwargs. KJ's node once broke with "attention_sage() got an
     unexpected keyword argument 'transformer_options'" when core widened the
     signature. Every entry point here ends in **kwargs and forwards nothing
     it did not ask for.

  C. RE-ENTRY. wrap_attn passes the UNDECORATED function as `func` and marks
     kwargs with _inside_attn_wrapper, so calling another wrapped backend
     from inside the override does not loop. We rely on that mark rather than
     on our own flag -- but we still route the plain fallback through `func`,
     which is undecorated and therefore terminal.


WHAT THIS NODE DOES NOT DO
--------------------------
It patches the DIFFUSION MODEL only. The text encoder and the VAE do not read
transformer_options; they use the global choice that ComfyUI made at startup
(Frank's log: "Using pytorch attention", twice, plus "pytorch attention in
VAE"). Covering those would need a different mechanism -- a global swap with
process-wide blast radius -- and mixing two mechanisms in one node would be a
self-inflicted wound. Deliberately out of scope.

Per-step and per-phase switching is stage A3, not here. The ground for it is
already measured: comfy/samplers.py writes transformer_options["sigmas"] and
transformer_options["sample_sigmas"], so the exact step index is readable at
attention time. This file keeps the canon append-only so A3 can add widgets
without renumbering a single saved workflow.
"""

import inspect
import time

import torch

# --------------------------------------------------------------------------
# mode table
# --------------------------------------------------------------------------

DEFAULT_MODE = "default (leave the model alone)"

# v842: the window dropdowns carry this as their default, and it means exactly
# "do not switch". With both step counts at 0 as well, a v841 workflow keeps
# behaving like v841 -- the sigma_shift_low sentinel doctrine, again.
SAME_AS_MAIN = "same as main"

# label -> torch.nn.attention.SDPBackend member name (None = let torch pick)
SDPA_MODES = {
    "pytorch sdpa": None,
    "pytorch sdpa (cudnn)": "CUDNN_ATTENTION",
    "pytorch sdpa (flash)": "FLASH_ATTENTION",
    "pytorch sdpa (mem-efficient)": "EFFICIENT_ATTENTION",
}

# label -> (sageattention function name, extra keyword arguments)
# The accumulation dtypes are the same ones the KJ node uses, so a mode of the
# same name behaves the same way and Frank's numbers stay comparable.
SAGE_MODES = {
    "sage auto": ("sageattn", {}),
    "sage fp16 cuda": ("sageattn_qk_int8_pv_fp16_cuda", {"pv_accum_dtype": "fp32"}),
    "sage fp16 triton": ("sageattn_qk_int8_pv_fp16_triton", {}),
    "sage fp8 cuda": ("sageattn_qk_int8_pv_fp8_cuda", {"pv_accum_dtype": "fp32+fp32"}),
    "sage fp8 cuda++": ("sageattn_qk_int8_pv_fp8_cuda", {"pv_accum_dtype": "fp32+fp16"}),
}

SAGE3_MODE = "sage3 blackwell (fp4)"
XFORMERS_MODE = "xformers"

# v903: Core's OWN int8 attention, shipped inside the comfy_kitchen wheel.
# Registered by Core as "comfy_kitchen_int8" and surfaced by its
# ModelAttentionBackend node as "comfy kitchen attention".
#
# WHAT IT IS, read at the source rather than guessed from the name: the
# implementation lives in comfy_kitchen/sage_attention.py, carries an NVIDIA
# copyright header, and describes itself as pure INT8 scaled dot-product
# attention for tensor-core GPUs. So it is sage-shaped quantisation in
# NVIDIA's own version, bundled instead of installed -- NOT a MiniMax H3
# feature, though H3's stock template is where most people first meet it.
# Core registers it when the compiled kernel supports the card (compute
# capability 7.5 and up).
#
# WHY IT BELONGS IN THIS NODE. Both nodes write THE SAME KEY. Core's
# ModelPatcher.set_model_optimized_attention wraps the chosen backend and
# stores it as transformer_options["optimized_attention_override"] -- exactly
# where we put ours. (My first reading of this claimed our mechanism took
# precedence over Core's; it does not, and there is no precedence to have.
# Whichever node sits LATER in the chain simply overwrites the earlier one,
# with no error and no warning.) Offering the same backend here means the
# user picks once, in one dropdown, and can measure it against sage with
# live_check instead of stacking two nodes that quietly fight.
#
# Core also forwards the backend's container_function when it wraps
# (model_patcher.py, right after the wrap) -- so the pass-through below is
# not our invention, it is what Core does with the same object.
CK_INT8_MODE = "comfy kitchen int8"
CK_INT8_REGISTRY_NAME = "comfy_kitchen_int8"

# A4: ONE sparse mode, and a modest one on purpose. A token in latent frame i
# attends to frames [i-w, i+w], optionally plus frame 0 as a sink; spatially
# everything stays FULL. The published schemes (Radial, NABLA, SpargeAttn) are
# better and more intricate -- banded, exponentially decaying density -- and a
# subtly wrong mask does not crash, it degrades the video quietly. So this
# ships a mask simple enough to be proven exactly right against a dense
# reference on the user's own machine before it is used, and says plainly that
# it is not the paper's method.
#
# IT ONLY RUNS COMPILED (see _flex_pair): uncompiled, both the mask build and
# the attention itself materialise in full and no card holds them.
#
# THE ECONOMICS ARE GEOMETRY-DEPENDENT AND THE NODE MEASURES THEM. Sparse
# computes in bf16. Against an unquantised backend, keeping ~40% of the frame
# pairs is close to a 2x saving on the attention share. Against an int8 sage
# kernel -- which already costs a fraction of dense bf16 -- the same 40% is
# roughly a tie. Which of those a given run is in cannot be settled by a table
# written here, so live_check settles it on the real call.
SPARSE_LOCAL = "sparse local (video)"


class PassThrough(Exception):
    """Not an error: this call is not ours to serve (cross-attention, or a
    sequence that does not match the latent geometry). The override hands it
    to the model's own backend WITHOUT the failure warning -- a normal
    condition must not look like a broken kernel."""

# Positional order shared by every core attention backend. Used to read the
# call apart without forwarding anything we did not ask for (trap B).
ATTN_PARAMS = ("q", "k", "v", "heads", "mask", "attn_precision",
               "skip_reshape", "skip_output_reshape")


# --------------------------------------------------------------------------
# lazy resolvers -- nothing here may explode at import time
# --------------------------------------------------------------------------

def _core_attention(name):
    """Fetch a core backend by registry name, or None if unavailable.

    Kept lazy so this module imports without ComfyUI (the guards run it that
    way) and so a core rename degrades into a clean fallback instead of an
    import error at start-up.
    """
    try:
        from comfy.ldm.modules import attention as core
    except Exception:
        return None
    getter = getattr(core, "get_attention_function", None)
    if callable(getter):
        try:
            fn = getter(name, None)
            if callable(fn):
                return fn
        except Exception:
            pass
    return getattr(core, "attention_" + name, None)


def _xformers_wired():
    """True / False / None -- did ComfyUI actually WIRE xformers this session?

    Not "is the package installed". Frank has xformers 0.0.35 installed and
    still got, from us, "'xformers' did NOT run on this machine (NameError:
    name 'xformers' is not defined)". The NameError comes from inside CORE:
    comfy defines attention_xformers unconditionally but only IMPORTS the
    xformers module when its own availability check passes, and Frank's
    instance starts on "Using pytorch attention". So the function exists, the
    name behind it does not -- an import check on our side says yes to a mode
    that cannot run, which is precisely the dropdown law we enforce
    everywhere else.

    Returns None when core exposes no such flag: unknown is not the same as
    unavailable, so the mode is still offered and the launch probe decides.
    """
    for mod_name in ("comfy.ldm.modules.attention", "comfy.model_management"):
        try:
            mod = __import__(mod_name, fromlist=["_"])
        except Exception:
            continue
        for attr in ("XFORMERS_IS_AVAILABLE", "XFORMERS_ENABLED"):
            val = getattr(mod, attr, None)
            if isinstance(val, bool):
                return val
    return None


def _sage_function(fn_name):
    """Resolve one sageattention entry point, or None."""
    try:
        import sageattention
    except Exception:
        return None
    return getattr(sageattention, fn_name, None)


def _sage3_function():
    """SageAttention 3 lives under several names depending on how it was built."""
    for mod_name, attr in (("sageattn3", "sageattn3_blackwell"),
                           ("sageattention", "sageattn3"),
                           ("sageattn3", "sageattn3")):
        try:
            mod = __import__(mod_name)
        except Exception:
            continue
        fn = getattr(mod, attr, None)
        if callable(fn):
            return fn
    return None


def _accepts(fn, param):
    """Does fn take this keyword? Wheels differ on attn_mask support."""
    try:
        return param in inspect.signature(fn).parameters
    except Exception:
        return False


def _sdpa_backend(member):
    try:
        from torch.nn.attention import SDPBackend
    except Exception:
        return None
    return getattr(SDPBackend, member, None)


# --------------------------------------------------------------------------
# the sage call -- the only place where we touch tensor layout ourselves
# --------------------------------------------------------------------------

def run_sage(sage_fn, extra, q, k, v, heads, mask=None,
             skip_reshape=False, skip_output_reshape=False):
    """Drive a sageattention kernel with core's calling convention.

    Core hands us one of two layouts and wants one of two back:
      skip_reshape=False -> q is (b, seq, heads*dim_head)   -- "NHD" for sage
      skip_reshape=True  -> q is (b, heads, seq, dim_head)  -- "HND" for sage
    and mirrors that on the way out via skip_output_reshape. Getting this
    wrong does not crash, it silently scrambles the image, so it is the one
    piece of this file the guard drives with real tensors.
    """
    if skip_reshape:
        b, qh, _, dim_head = q.shape
        # Grouped/multi-query attention hands k and v fewer heads than q.
        # run_flex already refuses that case by name; this one used to walk
        # into a reshape and die with a size error that the fallback then
        # reported as a broken kernel. Refusing is both true and quiet.
        if k.shape[1] != qh or v.shape[1] != qh:
            raise PassThrough("grouped heads (q %d, k %d)"
                              % (qh, k.shape[1]))
        layout = "HND"
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        if k.shape[-1] != heads * dim_head or v.shape[-1] != heads * dim_head:
            raise PassThrough("grouped heads (inner q %d, k %d)"
                              % (q.shape[-1], k.shape[-1]))
        q = q.reshape(b, -1, heads, dim_head)
        k = k.reshape(b, -1, heads, dim_head)
        v = v.reshape(b, -1, heads, dim_head)
        layout = "NHD"

    kw = dict(extra)
    if _accepts(sage_fn, "tensor_layout"):
        kw["tensor_layout"] = layout
    elif layout == "NHD":
        # This kernel speaks HND only and takes no tensor_layout keyword
        # (SageAttention 3 wheels: sageattn3_blackwell(q, k, v, ...) with a
        # fixed (b, heads, seq, dim) layout). Passing the keyword anyway
        # would TypeError at the probe and falsely disqualify the mode;
        # feeding it NHD tensors without saying so would not crash -- it
        # would silently scramble the image, the exact trap this function
        # exists to prevent. So convert here, flip the label, and let the
        # HND output branch below answer in the shape the caller asked for.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        layout = "HND"
    if _accepts(sage_fn, "is_causal"):
        kw["is_causal"] = False
    if mask is not None:
        if not _accepts(sage_fn, "attn_mask"):
            # This wheel cannot mask. Say so by refusing, so the caller falls
            # back to the model's own backend for THIS call instead of
            # quietly dropping the mask -- a dropped mask is a wrong image,
            # not a slow one.
            raise NotImplementedError(
                "this sageattention build takes no attn_mask")
        kw["attn_mask"] = mask

    out = sage_fn(q, k, v, **kw)

    if layout == "HND":
        if not skip_output_reshape:
            out = out.transpose(1, 2).reshape(out.shape[0], -1, heads * dim_head)
    else:
        if skip_output_reshape:
            out = out.transpose(1, 2)
        else:
            out = out.reshape(out.shape[0], -1, heads * dim_head)
    return out


def latent_geometry(latent, patch=2):
    """(frames, tokens_per_frame, total) from a ComfyUI LATENT, or None.

    Wan's VAE is /8 spatially and /4 temporally, and the patch embedding is
    (1, 2, 2) -- so a (B, C, T, H, W) latent carries T frames of
    (H//2)*(W//2) tokens. Frank's 768x768 / 65 frames: T=17, 48x48 -> 39168,
    which is exactly the sequence the sampler shows. The total is recomputed
    here and CHECKED against the real q at run time; a mismatch means our
    idea of the geometry is wrong and the call is handed back rather than
    masked on a guess.
    """
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None:
        return None
    shape = tuple(getattr(samples, "shape", ()))
    if len(shape) != 5:
        return None
    _b, _c, t, h, w = shape
    hp, wp = int(h) // patch, int(w) // patch
    if hp < 1 or wp < 1 or int(t) < 1:
        return None
    per = hp * wp
    return (int(t), per, int(t) * per)


def local_keep(q_idx, kv_idx, per_frame, window, sink):
    """The mask, as one honest boolean. Integers only, so the guard can
    compare it against a naive nested loop and prove it exactly."""
    fq = q_idx // per_frame
    fk = kv_idx // per_frame
    if sink and fk == 0:
        return True
    return abs(fq - fk) <= window


def local_density(frames, window, sink):
    """Fraction of frame pairs kept -- the honest speed claim, computed, not
    promised. Printed on the console so a mask that collapsed to nothing (or
    to everything) is visible before the run, not after."""
    if frames < 1:
        return 1.0
    kept = 0
    for fq in range(frames):
        lo = max(0, fq - window)
        hi = min(frames - 1, fq + window)
        n = hi - lo + 1
        if sink and lo > 0:
            n += 1
        kept += n
    return kept / float(frames * frames)


_FLEX = {}


def _flex_pair():
    """(compiled flex_attention, compiled create_block_mask), cached per process.

    THE WHOLE SPARSE MODE HANGS ON THIS, and both halves were measured before
    they were relied on (05.08., torch 2.13 sandbox, CPU):

      * create_block_mask UNCOMPILED broadcasts the full Q x KV index grid
        before reducing it to blocks. At 39168 tokens the int64 intermediate
        alone asks for 12.3 GB and dies. COMPILED it evaluates block by block:
        the same mask built in 146 MB of Python peak, and reported exactly the
        sparsity our own local_density() computes -- two independent routes to
        41.5%, which is the cross-check that the mask means what we say.
      * flex_attention WITHOUT torch.compile falls back to an eager path that,
        in torch's own warning, "materializes the full scores matrix" -- over
        a hundred GB once the head count multiplies in. COMPILED it produced
        the band-masked answer to within 6.6e-07 of a dense SDPA reference
        carrying the same mask. That is the number that makes this mode
        trustworthy rather than merely fast.

    torch.compile is used rather than the create_block_mask(_compile=True)
    flag: torch deprecates the flag and points at exactly this form.
    """
    if "flex" not in _FLEX:
        from torch.nn.attention.flex_attention import (create_block_mask,
                                                       flex_attention)
        _FLEX["flex"] = torch.compile(flex_attention, dynamic=False)
        _FLEX["cbm"] = torch.compile(create_block_mask)
    return _FLEX["flex"], _FLEX["cbm"]


def build_block_mask(frames, per_frame, window, sink, device):
    """A flex_attention BlockMask for the local band, built compiled."""
    _flex, cbm = _flex_pair()
    total = frames * per_frame

    def mask_mod(_b, _h, q_idx, kv_idx):
        fq = q_idx // per_frame
        fk = kv_idx // per_frame
        near = (fq - fk).abs() <= window
        if sink:
            return near | (fk == 0)
        return near

    return cbm(mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total,
               device=device)


def run_flex(block_mask, q, k, v, heads, mask=None,
             skip_reshape=False, skip_output_reshape=False):
    """Drive flex_attention with core's calling convention.

    Refuses (PassThrough) rather than guesses whenever the call is not the
    self-attention this mask was built for: a text mask present, cross
    attention (kv shorter than q), a head count that does not line up, or a
    sequence length that is not the latent's.
    """
    flex, _cbm = _flex_pair()
    if mask is not None:
        raise PassThrough("masked call")

    if skip_reshape:
        b, h, n, dim_head = q.shape
        qq, kk, vv = q, k, v
    else:
        b, n, inner = q.shape
        dim_head = inner // heads
        h = heads
        qq = q.reshape(b, n, heads, dim_head).transpose(1, 2)
        kk = k.reshape(b, -1, heads, dim_head).transpose(1, 2)
        vv = v.reshape(b, -1, heads, dim_head).transpose(1, 2)

    if kk.shape[2] != n or vv.shape[2] != n:
        raise PassThrough("cross attention")
    if kk.shape[1] != h:
        raise PassThrough("grouped heads")

    out = flex(qq, kk, vv, block_mask=block_mask)

    if skip_output_reshape:
        return out
    return out.transpose(1, 2).reshape(b, n, h * dim_head)



def _unpack(args, kwargs):
    """Read a core attention call apart into named values."""
    vals = {"mask": None, "attn_precision": None,
            "skip_reshape": False, "skip_output_reshape": False}
    for i, name in enumerate(ATTN_PARAMS):
        if i < len(args):
            vals[name] = args[i]
        elif name in kwargs:
            vals[name] = kwargs[name]
    return vals


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------

def build_router(mode, sparse=None):
    """Return router(func, *args, **kwargs) for a mode, or None if unknown.

    The router may RAISE -- wrapping it in a fallback is the caller's job, so
    that the probe can tell "this kernel does not run here" apart from "this
    kernel ran and we quietly used something else".
    """
    if mode == SPARSE_LOCAL:
        if not isinstance(sparse, dict):
            return None
        frames = sparse["frames"]
        per_frame = sparse["per_frame"]
        total = sparse["total"]
        window = sparse["window"]
        sink = sparse["sink"]
        masks = {}

        def _sparse_router(func, *args, **kwargs):
            v = _unpack(args, kwargs)
            q = v["q"]
            n = q.shape[2] if v["skip_reshape"] else q.shape[1]
            if n != total:
                # Not the sequence this mask was built for. Never mask on a
                # guess -- hand it back.
                raise PassThrough("sequence %d, latent says %d" % (n, total))
            key = (q.device, q.dtype)
            if key not in masks:
                masks[key] = build_block_mask(frames, per_frame, window,
                                              sink, q.device)
            return run_flex(masks[key], q, v["k"], v["v"], v["heads"],
                            mask=v["mask"],
                            skip_reshape=v["skip_reshape"],
                            skip_output_reshape=v["skip_output_reshape"])

        return _sparse_router

    if mode in SDPA_MODES:
        member = SDPA_MODES[mode]

        def _sdpa_router(func, *args, **kwargs):
            base = _core_attention("pytorch") or func
            if member is None:
                return base(*args, **kwargs)
            backend = _sdpa_backend(member)
            if backend is None:
                raise NotImplementedError(
                    "this torch build has no SDPBackend.%s" % member)
            from torch.nn.attention import sdpa_kernel
            with sdpa_kernel([backend]):
                return base(*args, **kwargs)

        return _sdpa_router

    if mode in SAGE_MODES:
        fn_name, extra = SAGE_MODES[mode]

        def _sage_router(func, *args, **kwargs):
            sage_fn = _sage_function(fn_name)
            if sage_fn is None:
                raise NotImplementedError(
                    "sageattention has no %s" % fn_name)
            v = _unpack(args, kwargs)
            return run_sage(sage_fn, extra, v["q"], v["k"], v["v"],
                            v["heads"], mask=v["mask"],
                            skip_reshape=v["skip_reshape"],
                            skip_output_reshape=v["skip_output_reshape"])

        return _sage_router

    if mode == SAGE3_MODE:
        def _sage3_router(func, *args, **kwargs):
            sage_fn = _sage3_function()
            if sage_fn is None:
                raise NotImplementedError("SageAttention 3 is not installed")
            v = _unpack(args, kwargs)
            return run_sage(sage_fn, {}, v["q"], v["k"], v["v"],
                            v["heads"], mask=v["mask"],
                            skip_reshape=v["skip_reshape"],
                            skip_output_reshape=v["skip_output_reshape"])

        return _sage3_router

    if mode == XFORMERS_MODE:
        def _xf_router(func, *args, **kwargs):
            base = _core_attention("xformers")
            if base is None:
                raise NotImplementedError("core has no xformers backend")
            return base(*args, **kwargs)

        return _xf_router

    if mode == CK_INT8_MODE:
        def _ck_router(func, *args, **kwargs):
            base = _core_attention(CK_INT8_REGISTRY_NAME)
            if base is None:
                raise NotImplementedError(
                    "core has no %s backend" % CK_INT8_REGISTRY_NAME)
            return base(*args, **kwargs)

        # THE CONTAINER PASS-THROUGH, and the reason it is not optional.
        #
        # MiniMax H3 hands attention its tensors as AttentionTensorContainer
        # objects. Core's wrap_attn, when an override is installed, looks for
        # a `container_function` ATTRIBUTE ON THE OVERRIDE; finding none it
        # unpacks the containers with .take() and calls the override with
        # plain tensors. That works -- and throws away the whole point of the
        # container path, which for this backend is
        # prequantize_int8_attention(): it quantises once and drops the
        # floating-point q/k/v instead of holding them alongside.
        #
        # Without this attribute we would be offering a WORSE version of the
        # backend than Core's own node offers, on exactly the model where the
        # tensors are largest. So we carry the chosen backend's own container
        # function through, unmodified.
        _ck_base = _core_attention(CK_INT8_REGISTRY_NAME)
        _ck_container = getattr(_ck_base, "container_function", None)
        if callable(_ck_container):
            def _ck_containers(*args, **kwargs):
                return _ck_container(*args, **kwargs)
            _ck_router.container_function = _ck_containers

        return _ck_router

    return None


def locate_step(tops, cache):
    """(step, total) read out of transformer_options, or (None, None).

    WHY THIS IS POSSIBLE AT ALL, and how we know rather than hope: our
    override is looked up in transformer_options -- that is the only reason
    v841 fires. comfy/samplers.py writes transformer_options["sigmas"] on the
    way in and, on the very next line, reads transformer_options
    ["sample_sigmas"] out of the same dict. So the current noise level and the
    whole schedule sit in the dict that reaches us. Measured on Frank's own
    install (05.08.), lines 511/512 and 1221.

    WHY THE CACHE IS NOT AN OPTIMISATION BUT A REQUIREMENT: reading a value
    out of a CUDA tensor is a device sync. Wan 2.2 runs 40 blocks per step, so
    an uncached read would stall the pipeline 40 times per step for a number
    that cannot have changed. The sigma tensor is the SAME OBJECT for every
    block within one step, so identity is a sound cache key -- one sync per
    step, not per block.
    """
    if not isinstance(tops, dict):
        return (None, None)
    sig = tops.get("sigmas", None)
    sched = tops.get("sample_sigmas", None)
    if sig is None or sched is None:
        return (None, None)
    if cache.get("sig", None) is sig:
        return cache["step"], cache["total"]
    try:
        flat = sched.reshape(-1)
        total = int(flat.numel()) - 1          # trailing 0.0 is not a step
        if total < 1:
            return (None, None)
        cur = sig.reshape(-1)[0]
        step = int(torch.argmin((flat - cur).abs()).item())
        if step > total - 1:
            step = total - 1
    except Exception:
        return (None, None)
    cache["sig"] = sig
    cache["step"] = step
    cache["total"] = total
    return step, total


def pick_mode(step, total, main, first_mode, first_steps,
              last_mode, last_steps):
    """Which mode a given step belongs to. Pure, so the guard can drive it.

    The windows are counted from the ends of THIS sampler call. In a Wan MoE
    run that means each expert gets its own first and last steps, which is the
    useful reading: the high expert's opening step and the low expert's
    closing step are exactly the two places where quantisation shows.

    An unlocatable step (no schedule in transformer_options) falls to `main`.
    Overlapping windows on a short schedule resolve in favour of `first`.
    """
    if step is None:
        return main
    if first_steps > 0 and first_mode is not None and step < first_steps:
        return first_mode
    if (last_steps > 0 and last_mode is not None and total is not None
            and step >= total - last_steps):
        return last_mode
    return main


def deviation_stats(out, base, chunk=1 << 22):
    """How far apart two attention answers are -- WITH THE SCALE THEY LIVE ON.

    v845 printed a bare maximum absolute difference and Frank's first field
    run showed why that is not enough: 0.6172 against a reference whose own
    magnitude was never stated. The same number is alarming on activations
    that run to +/-2 and irrelevant on activations that run to +/-80, and the
    line gave no way to tell which. A quality figure nobody can interpret is
    not a quality figure.

    So two ratios are reported beside the raw maximum:

      rel_max  max|difference| against max|reference| -- the worst single
               element, expressed as a share of the largest value the layer
               actually produces.
      rel_rms  rms(difference) against rms(reference) -- the TYPICAL error.
               This is the one to read for "will I see it": a maximum can be
               driven by one outlier element out of forty million, while the
               rms says what the whole tensor is doing.

    Returns a dict, or None if the two cannot be compared at all.
    """
    try:
        if out is None or base is None or out.shape != base.shape:
            return None
        # v849: MEASURED IN SLICES, NOT IN THREE FULL COPIES. The old form
        # built out.float(), base.float() and their difference whole. On
        # Frank's real geometry the attention output is 1x40x39168x128 in
        # bf16 -- 401 MB -- so the diagnostic asked for ~2.4 GB of fp32 on a
        # card already at 82%. It survived, but the failure it invites is the
        # worst kind: the except below would swallow the OOM, return None,
        # and (before v849) the live check would then print nothing at all.
        # The arithmetic is unchanged -- sums of squares over the whole
        # tensor, accumulated in Python floats, which is if anything steadier
        # than an fp32 reduction. Only the working set shrinks.
        a = out.reshape(-1)
        b = base.reshape(-1)
        n = int(a.numel())
        if n == 0:
            return None
        step = max(1, int(chunk))
        max_abs = 0.0
        peak_base = 0.0
        sq_diff = 0.0
        sq_base = 0.0
        for i in range(0, n, step):
            ca = a[i:i + step].float()
            cb = b[i:i + step].float()
            cd = (ca - cb).abs()
            max_abs = max(max_abs, float(cd.max().item()))
            peak_base = max(peak_base, float(cb.abs().max().item()))
            sq_diff += float(cd.pow(2).sum().item())
            sq_base += float(cb.pow(2).sum().item())
        rms_diff = (sq_diff / n) ** 0.5
        rms_base = (sq_base / n) ** 0.5
        stats = {"max_abs": max_abs, "rms": rms_diff,
                 "peak_base": peak_base, "rms_base": rms_base,
                 "rel_max": None, "rel_rms": None}
        # A reference of all zeros has no scale to divide by -- report the
        # absolutes rather than inventing a ratio.
        if peak_base > 0:
            stats["rel_max"] = max_abs / peak_base
        if rms_base > 0:
            stats["rel_rms"] = rms_diff / rms_base
        return stats
    except Exception:
        return None


def live_compare(router, func, args, kwargs, repeats=1):
    """Time the chosen kernel against the model's OWN backend, on the real
    call, and measure how far apart their answers are. Returns
    (ms_chosen, ms_base, deviation_stats) or None if anything goes wrong --
    a diagnostic must never be able to break a run. The ONE exception it
    re-raises is PassThrough: that means the call was never this mode's to
    serve, so no measurement happened and none should be counted.

    WHY THIS EXISTS. This node has to serve models it has never seen. The
    published rankings do not survive a change of geometry: on Wan video
    shape one sage mode was both fastest and most accurate, and on image
    shape the ORDER REVERSED. No table shipped in a docstring can answer
    "is this setting sensible for what I am rendering right now" -- only a
    measurement on the actual tensors can, so the node takes one.

    WHY IT DOES NOT CHOOSE. The result is printed, never acted upon. A node
    that silently switched kernels on a timing would make two runs of the
    same seed differ for reasons invisible in the workflow. The user gets
    the number and keeps the decision.

    WHY EACH SIDE RUNS TWICE BY DEFAULT. The first call through a compiled
    or newly loaded kernel pays for compilation and allocator warm-up; timing
    that would slander it. The warm-up call is thrown away and the second one
    is measured.

    THE DEVIATION IS THE POINT, not a bonus. It is measured against the
    backend the model would otherwise have used, on real activations rather
    than the random tensors a synthetic benchmark feeds in -- so it answers
    the quality half of the question in the same breath as the speed half.
    """
    try:
        q = args[0] if args else kwargs.get("q", None)
        on_cuda = bool(getattr(q, "is_cuda", False))

        def _sync():
            if on_cuda:
                torch.cuda.synchronize()

        def _time(fn):
            fn()                      # warm-up, discarded
            _sync()
            t0 = time.perf_counter()
            for _ in range(max(1, int(repeats))):
                out = fn()
            _sync()
            ms = (time.perf_counter() - t0) * 1000.0 / max(1, int(repeats))
            return out, ms

        out_chosen, ms_chosen = _time(lambda: router(func, *args, **kwargs))
        out_base, ms_base = _time(lambda: func(*args, **kwargs))

        return (ms_chosen, ms_base, deviation_stats(out_chosen, out_base))
    except PassThrough:
        # NOT a failure and NOT a measurement: this call was never ours. Let
        # it escape so the caller serves it normally and keeps its one
        # measurement for a call the mode actually handles.
        raise
    except Exception:
        return None


# Rules of thumb for turning rel_rms into a word. These are JUDGEMENT, not
# measurement, and are labelled as such in the output -- the node states the
# numbers first and its opinion second, so a user who disagrees with the
# thresholds still has everything needed to decide. Anchored on what we have
# actually measured: pure bf16 rounding against an fp32 reference sits around
# 0.05% here, and the sage int8 kernels land roughly an order of magnitude
# above that without visibly harming an image.
_DEV_BANDS = ((0.005, "negligible"),
              (0.02, "small"),
              (0.05, "noticeable -- worth an A/B against 'default'"),
              (float("inf"), "LARGE -- check the output before trusting it"))


def describe_deviation(mode, stats):
    """The deviation half of the verdict line, scale included."""
    if not stats:
        return ""
    parts = ["max %.4f" % stats["max_abs"]]
    if stats.get("rel_max") is not None:
        parts[0] += " (%.2f%% of its peak %.3f)" % (
            stats["rel_max"] * 100.0, stats["peak_base"])
    if stats.get("rel_rms") is not None:
        parts.append("rms %.3f%% of its signal" % (stats["rel_rms"] * 100.0))
    else:
        parts.append("rms %.4f" % stats["rms"])
    # v849: NAME THE REFERENCE HERE TOO. The speed half says what the
    # comparison partner is ("the model's own backend"); the deviation half
    # used to say only "of the output's peak" and "of signal" and left the
    # reader to carry the reference across the semicolon. Frank read the line
    # and asked "deviation from what?" -- which is the whole answer: a figure
    # whose reference has to be inferred is not a stated figure. Every share
    # now says whose it is, in the same clause as the number.
    text = "; deviation from that same backend's answer: " + ", ".join(parts)

    if mode == SPARSE_LOCAL:
        # For every other mode the deviation is quantisation error and small
        # is good. For sparse it is the MASK -- information the kernel was
        # told to leave out. Reading it as a defect would be wrong, and
        # reading a tiny value as success would be worse: it would mean the
        # mask is barely doing anything.
        return text + (" (this is the information the mask omits, not kernel "
                       "error -- a SMALL value here would mean the mask is "
                       "barely doing anything)")
    if stats.get("rel_rms") is None:
        return text
    for limit, word in _DEV_BANDS:
        if stats["rel_rms"] < limit:
            return text + " -- %s" % word
    return text


def format_verdict(mode, ms_chosen, ms_base, stats):
    """The one line the user reads. Plain arithmetic, no salesmanship.

    `stats` is what deviation_stats() returns. A bare float is still accepted
    and read as an absolute maximum with no scale -- that is the degenerate
    case, and it prints without a share or a verdict rather than pretending
    to a precision it does not have.
    """
    if ms_chosen <= 0 or ms_base <= 0:
        return None
    if isinstance(stats, (int, float)):
        stats = {"max_abs": float(stats), "rms": float(stats),
                 "peak_base": 0.0, "rms_base": 0.0,
                 "rel_max": None, "rel_rms": None}
    ratio = ms_base / ms_chosen
    if ratio >= 1.05:
        speed = "%.2fx faster than" % ratio
    elif ratio <= 0.95:
        speed = "%.2fx the speed of (i.e. SLOWER than)" % ratio
    else:
        speed = "the same speed as"
    line = ("[PLS] Attention: live check on the real call -- '%s' %.1f ms vs "
            "the model's own backend %.1f ms, %s it"
            % (mode, ms_chosen, ms_base, speed))
    line += describe_deviation(mode, stats)
    if ratio <= 0.95:
        line += ("\n[PLS]   -> this setting costs time here. The model's own "
                 "backend is quicker on this geometry; consider 'default'.")
    return line


def build_override(mode, fallback=True, report=None,
                   first_mode=None, first_steps=0,
                   last_mode=None, last_steps=0, announce=None, sparse=None,
                   live_check=False, verdict=None, passthrough=None):
    """Wrap a router so a runtime failure degrades instead of killing the run.

    `report` is called once, with the exception, the first time a mode falls
    back -- a backend that silently stops being used is worse than one that
    fails, because the run still finishes and the numbers move for no visible
    reason.

    With no window active this returns exactly the v841 override: one router,
    no per-call bookkeeping, nothing read out of transformer_options. That is
    deliberate -- a workflow that does not ask for switching must not pay for
    it, and must not be able to break on it.

    `announce` is called once per patched model with (step, total), or with
    (None, None) if the schedule could not be found. A window that silently
    never fires would be indistinguishable from one that works.
    """
    router = build_router(mode, sparse=sparse)
    if router is None:
        return None

    windowed = ((first_steps > 0 and first_mode is not None)
                or (last_steps > 0 and last_mode is not None))
    routers = {mode: router}
    for extra in (first_mode, last_mode):
        if extra is not None and extra not in routers:
            r = build_router(extra, sparse=sparse)
            if r is None:
                return None
            routers[extra] = r

    state = {"warned": False, "announced": False, "measured": False,
             "passed": False}
    cache = {}

    def _override(func, *args, **kwargs):
        chosen = mode
        if windowed:
            step, total = locate_step(kwargs.get("transformer_options", None),
                                      cache)
            if not state["announced"]:
                state["announced"] = True
                if announce is not None:
                    try:
                        announce(step, total)
                    except Exception:
                        pass
            chosen = pick_mode(step, total, mode, first_mode, first_steps,
                               last_mode, last_steps)
        try:
            if live_check and not state["measured"]:
                # Once per patched model, and only for a call this mode
                # actually SERVES. The v845 code set the flag before it knew
                # that -- so a first call the mode passes through (cross
                # attention, a foreign sequence length) burned the one
                # measurement and the user got no line at all. It only
                # worked in the field because Wan happens to run
                # self-attention first. live_compare lets PassThrough
                # escape, so the flag below is reached only on a call this
                # mode really handled.
                got = live_compare(routers[chosen], func, args, kwargs)
                state["measured"] = True
                if verdict is not None:
                    # v849: THE FAILED MEASUREMENT SPEAKS TOO. live_compare
                    # returns None for any internal trouble -- most plausibly
                    # an OOM inside the deviation maths on a full card. The
                    # flag above is already spent, so staying quiet would
                    # leave a user who switched live_check ON with no line,
                    # no reason and no second attempt: indistinguishable from
                    # a mode that never served a call. Everything else in this
                    # file is loud about absence; this was the one place that
                    # was not. None in the timing slot IS the signal.
                    try:
                        if got is None:
                            verdict(chosen, None, None, None)
                        else:
                            verdict(chosen, got[0], got[1], got[2])
                    except Exception:
                        pass
            return routers[chosen](func, *args, **kwargs)
        except PassThrough as skip:
            # Normal condition, not a failure: this call was never ours.
            # v849: BUT SAY SO ONCE. Until now this branch was completely
            # mute, so a run could print "sage3 blackwell (fp4) (ran)" while
            # an unknown share of its attention calls never touched that
            # kernel -- cross attention, a masked call, grouped heads, a
            # foreign sequence length. The log looked complete and hid the
            # coverage. That is the same defect class as the window v847
            # dropped in silence. One line, with the reason, the first time
            # it happens: enough to know the mode is not carrying the whole
            # run, cheap enough to leave in the hot path (a bool test).
            if not state["passed"]:
                state["passed"] = True
                if passthrough is not None:
                    try:
                        passthrough(chosen, skip)
                    except Exception:
                        pass
            return func(*args, **kwargs)
        except Exception as exc:
            if not fallback:
                raise
            if not state["warned"]:
                state["warned"] = True
                if report is not None:
                    try:
                        report(exc)
                    except Exception:
                        pass
            return func(*args, **kwargs)

    return _override


# --------------------------------------------------------------------------
# capability probe -- launch it once, believe the launch, not the import
# --------------------------------------------------------------------------

_PROBE_CACHE = {}


def probe(mode, force=False):
    """(ok, note) for a mode, cached per process.

    Runs one tiny attention on the real device. seq is 2048 on purpose:
    SageAttention 3 refuses sequences at or below 1024, so a smaller probe
    would report a false negative for it.
    """
    if not force and mode in _PROBE_CACHE:
        return _PROBE_CACHE[mode]

    if mode == DEFAULT_MODE:
        result = (True, "nothing to probe")
    elif not torch.cuda.is_available():
        # Honest non-answer. The runtime fallback still protects the run.
        result = (True, "not probed (no CUDA device visible)")
    else:
        router = build_router(mode)
        if router is None:
            result = (False, "unknown mode")
        else:
            try:
                dev = torch.device("cuda")
                b, h, n, d = 1, 2, 2048, 128
                q = torch.randn(b, h, n, d, device=dev, dtype=torch.bfloat16)
                k = torch.randn(b, h, n, d, device=dev, dtype=torch.bfloat16)
                v = torch.randn(b, h, n, d, device=dev, dtype=torch.bfloat16)
                out = router(_probe_reference, q, k, v, h,
                             None, None, True, True)
                ok = bool(torch.isfinite(out).all().item())
                result = (ok, "ran" if ok else "ran but returned non-finite values")
                del q, k, v, out
            except Exception as exc:
                result = (False, "%s: %s" % (type(exc).__name__,
                                             str(exc).split("\n")[0][:80]))

    _PROBE_CACHE[mode] = result
    return result


def sparse_selfcheck(frames, window, sink, device, dtype=None):
    """Run the sparse path small AND CHECK ITS NUMBERS, on this machine.

    An attention kernel that is merely alive is not enough here. A block mask
    that quietly keeps the wrong blocks does not raise, it degrades the video
    -- so this builds the same mask the run will use (same frame count, same
    window, same sink logic, only 64 tokens per frame) and compares the
    compiled flex answer against a DENSE reference carrying the SAME mask as
    an ordinary boolean attn_mask. Two different machineries, one expected
    answer.

    Returns (ok, note, deviation). Measured on the reference implementation
    at build time: 6.6e-07 in fp32. The tolerance below is generous enough
    for bf16 accumulation and still an order of magnitude tighter than any
    mask error could hide in.
    """
    import torch.nn.functional as F

    per = 64
    frames = int(frames)
    total = frames * per
    dt = dtype if dtype is not None else torch.float32
    bm = build_block_mask(frames, per, int(window), bool(sink), device)

    q = torch.randn(1, 2, total, 64, device=device, dtype=dt)
    k = torch.randn(1, 2, total, 64, device=device, dtype=dt)
    v = torch.randn(1, 2, total, 64, device=device, dtype=dt)

    out = run_flex(bm, q, k, v, 2, skip_reshape=True, skip_output_reshape=True)
    if not bool(torch.isfinite(out).all().item()):
        return (False, "ran but returned non-finite values", None)

    idx = torch.arange(total, device=device)
    fq = (idx // per).view(-1, 1)
    fk = (idx // per).view(1, -1)
    keep = (fq - fk).abs() <= int(window)
    if sink:
        keep = keep | (fk == 0)
    ref = F.scaled_dot_product_attention(
        q.float(), k.float(), v.float(),
        attn_mask=keep.view(1, 1, total, total))
    dev = float((out.float() - ref).abs().max().item())
    if dev > 5e-2:
        return (False, "the compiled mask disagrees with a dense reference "
                       "by %.2e -- refusing to mask on it" % dev, dev)
    return (True, "ran, mask verified to %.1e against a dense reference" % dev,
            dev)


def probe_sparse(frames, window, sink, force=False):
    """(ok, note) for the sparse path, cached per process."""
    key = ("sparse", int(frames), int(window), bool(sink))
    if not force and key in _PROBE_CACHE:
        return _PROBE_CACHE[key]
    if not torch.cuda.is_available():
        result = (True, "not probed (no CUDA device visible)")
    else:
        try:
            ok, note, _dev = sparse_selfcheck(frames, window, sink,
                                              torch.device("cuda"),
                                              torch.bfloat16)
            result = (ok, note)
        except Exception as exc:
            result = (False, "%s: %s" % (type(exc).__name__,
                                         str(exc).split("\n")[0][:80]))
    _PROBE_CACHE[key] = result
    return result


def _probe_reference(q, k, v, heads, mask=None, attn_precision=None,
                     skip_reshape=False, skip_output_reshape=False, **kwargs):
    """Stand-in for core's backend during a probe: plain SDPA, HND in and out.

    Only reached if a router decides to fall back, which during a probe means
    the mode did not really run -- so it must not look like success. It
    returns a tensor of the right shape; the probe reads the router's own
    exception path, not this.
    """
    import torch.nn.functional as F
    return F.scaled_dot_product_attention(q, k, v)


def describe_absence(mode):
    """Why a known mode is not offered here -- a note string, or "".

    Only states what it can check. A guess dressed as a diagnosis would send
    the user down the wrong path, which is the exact mistake v847 fixed for
    the xformers refusal.
    """
    if mode == XFORMERS_MODE:
        if _xformers_wired() is False:
            return ("NameError: ComfyUI did not wire its xformers backend "
                    "this session")
        if not _core_attention("xformers"):
            return "core has no xformers backend"
    if mode in SAGE_MODES:
        fn_name, _extra = SAGE_MODES[mode]
        if _sage_function(fn_name) is None:
            return "sageattention is not installed (or has no %s)" % fn_name
    if mode == SAGE3_MODE and _sage3_function() is None:
        return "SageAttention 3 is not installed"
    if mode in SDPA_MODES and SDPA_MODES[mode] is not None:
        if _sdpa_backend(SDPA_MODES[mode]) is None:
            return "this torch build has no SDPBackend.%s" % SDPA_MODES[mode]
    return ""


def explain_failure(mode, note):
    """Extra console lines when we can name the real cause, else nothing.

    A refusal that misattributes the cause is worse than a terse one: it
    sends the user looking in the wrong place. The xformers case is the one
    we have actually seen in the field -- the package IS installed, so
    "did not run on this machine" reads as a lie unless we say what really
    happened.
    """
    lines = []
    if mode == XFORMERS_MODE and "NameError" in (note or ""):
        lines.append("this is NOT a missing package: ComfyUI defines its "
                     "xformers backend but only imports the module when it "
                     "selects xformers at start-up.")
        lines.append("your log line 'Using pytorch attention' says it chose "
                     "something else this session, so the name behind the "
                     "function is unbound. Installing xformers again will "
                     "not change it.")
    return lines


def available_modes(include_unprobed=True):
    """Modes worth putting in front of the user, cheapest checks first.

    Import-level filtering only -- this runs while ComfyUI builds its node
    list, so it must not touch the GPU. The launch test happens later, in
    patch(), where a failure can be explained on the console instead of
    slowing every start-up.
    """
    modes = [DEFAULT_MODE]

    for label, member in SDPA_MODES.items():
        if member is None or _sdpa_backend(member) is not None:
            modes.append(label)

    seen_sage = set()
    for label, (fn_name, _extra) in SAGE_MODES.items():
        if fn_name in seen_sage or _sage_function(fn_name) is not None:
            modes.append(label)
            seen_sage.add(fn_name)

    if _sage3_function() is not None:
        modes.append(SAGE3_MODE)

    if _xformers_wired() is not False and _core_attention("xformers"):
        modes.append(XFORMERS_MODE)

    # v903: offered only when CORE registered it -- which it does only when
    # the compiled kernel supports this card. Same rule as every other mode
    # here: availability is read, never assumed.
    if _core_attention(CK_INT8_REGISTRY_NAME) is not None:
        modes.append(CK_INT8_MODE)

    # SPARSE_LOCAL is offered only where its RUNWAY exists: flex_attention
    # plus a usable torch.compile. Without compilation both halves of this
    # mode degrade into full materialisation (12.3 GB for the mask grid,
    # over a hundred GB for the scores) -- so an importable flex alone is
    # NOT the condition. The kernel is launched and its mask checked against
    # a dense reference in patch(), not here: this runs while ComfyUI builds
    # its node list and must not touch the GPU.
    try:
        import importlib as _il
        fx = _il.import_module("torch.nn.attention.flex_attention")
        if (hasattr(fx, "flex_attention") and hasattr(fx, "create_block_mask")
                and hasattr(torch, "compile")):
            modes.append(SPARSE_LOCAL)
    except Exception:
        pass

    if include_unprobed and len(modes) == 1:
        modes.append("pytorch sdpa")
    return modes


# --------------------------------------------------------------------------
# the node
# --------------------------------------------------------------------------

class ULSAttention:
    """Pick the attention kernel the diffusion model runs on."""

    @classmethod
    def INPUT_TYPES(cls):
        modes = available_modes()
        return {
            "required": {
                "model": ("MODEL", {
                    "tooltip": "The diffusion model to patch. Text encoder and "
                               "VAE are NOT affected -- they do not read "
                               "transformer_options and keep whatever backend "
                               "ComfyUI chose at start-up."}),
                "attention": (modes, {
                    "default": DEFAULT_MODE,
                    "tooltip": "Only backends this installation can actually "
                               "load are listed. The chosen one is launched "
                               "once before patching; if the launch fails, the "
                               "console says so by name and the model is left "
                               "on its own backend. Quantised modes (sage) buy "
                               "speed with accuracy -- which of them is the "
                               "most accurate depends on the model geometry, "
                               "so measure per model rather than picking one "
                               "for the whole tree."}),
                "fallback": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "On a kernel error mid-run, quietly continue on "
                               "the model's own backend (the console names the "
                               "error once). Turn this off to let the run fail "
                               "instead -- useful when you are measuring and a "
                               "silent fallback would move the numbers without "
                               "telling you."}),
                "attention_first": ([SAME_AS_MAIN] + modes, {
                    "default": SAME_AS_MAIN,
                    "tooltip": "Backend for the opening steps. The first step "
                               "sets the composition out of pure noise, which "
                               "is where a quantised kernel costs the most and "
                               "saves the least. Leave on 'same as main' to "
                               "switch nothing."}),
                "first_steps": ("INT", {
                    "default": 0, "min": 0, "max": 1000,
                    "tooltip": "How many steps at the START of THIS sampler "
                               "call use attention_first. Counted per call, so "
                               "in a MoE run each expert gets its own opening "
                               "steps. 0 disables the window. Multi-stage "
                               "samplers evaluate the model between schedule "
                               "points, so the window edge can be fuzzy by one "
                               "stage there."}),
                "attention_last": ([SAME_AS_MAIN] + modes, {
                    "default": SAME_AS_MAIN,
                    "tooltip": "Backend for the closing steps, where fine "
                               "detail is settled. Leave on 'same as main' to "
                               "switch nothing."}),
                "last_steps": ("INT", {
                    "default": 0, "min": 0, "max": 1000,
                    "tooltip": "How many steps at the END of THIS sampler call "
                               "use attention_last. 0 disables the window. If "
                               "both windows overlap on a short schedule, the "
                               "first one wins."}),
                "sparse_time_window": ("INT", {
                    "default": 3, "min": 0, "max": 64,
                    "tooltip": "Only for the sparse mode: how many latent "
                               "frames each token may reach in each direction. "
                               "At 17 latent frames a window of 3 keeps about "
                               "41% of the frame pairs, and roughly that "
                               "fraction of the attention cost. Motion "
                               "spanning more frames than this can no longer "
                               "be attended directly -- that is the risk. "
                               "Sparse runs in bf16, so it only pays off "
                               "against an UNQUANTISED backend; against an "
                               "int8 sage kernel it is roughly a tie. Leave "
                               "live_check on and the node will tell you "
                               "which side you are on. 0 means dense."}),
                "sparse_sink": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Only for the sparse mode: let every token also "
                               "see all of latent frame 0. Costs one extra "
                               "frame of attention and gives the whole clip a "
                               "common anchor -- the cheapest guard against "
                               "drift over a long shot."}),
                "live_check": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Measure the chosen kernel ONCE against the "
                               "backend the model would otherwise use, on the "
                               "first real attention call, and print both "
                               "times plus how far their answers differ -- "
                               "the difference is given against the signal's "
                               "own size (peak and rms), because an absolute "
                               "number alone cannot tell you whether it "
                               "matters. "
                               "Nothing is switched -- the number is for you, "
                               "the decision stays in the workflow. Costs "
                               "about four extra attention calls per run "
                               "(two are warm-ups). Worth leaving on when you "
                               "meet a new model, since the right kernel "
                               "depends on the geometry: on video shape one "
                               "mode can be both fastest and most accurate "
                               "while the order reverses on image shape."}),
            },
            "optional": {
                "latent": ("LATENT", {
                    "tooltip": "REQUIRED by the sparse mode, ignored by every "
                               "other one: the mask has to know how the token "
                               "sequence splits into frames, and only the "
                               "latent knows that. Wire the same latent the "
                               "sampler gets. Without it the sparse mode "
                               "refuses out loud rather than guessing."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, attention, attention_first, attention_last):
        """Accept any mode STRING, including one this installation does not
        offer. Documented mechanism, not a trick: ComfyUI passes only the
        inputs a VALIDATE_INPUTS signature asks for, and exactly those skip
        the default checks -- core's own CustomCombo node disables its list
        check the same way.

        WHY THIS IS NEEDED, learned the hard way in v847. The dropdown law
        ("never offer what cannot run") is right, and it made saved workflows
        BRITTLE: v847 correctly removed xformers from the list, and ComfyUI
        then rejected Frank's whole prompt -- "Value not in list" on both
        attention nodes, output ignored -- because the saved workflow still
        carried the old value. The failure is not local to the setting; it
        kills the run.

        That generalises far past xformers: open this workflow on a machine
        without sageattention, or without a Blackwell card, and the same hard
        stop happens. A node meant to be a finished product has to hold both
        properties at once -- offer nothing that cannot run here, and break on
        nothing another machine offered.

        So validation is relaxed and the HONESTY MOVES TO patch(), which says
        plainly that the workflow asks for a mode this installation does not
        have and leaves the model on its own backend. Loud, and the rest of
        the graph still renders. Deliberately NOT a silent substitution:
        quietly swapping in another kernel would change results with nothing
        in the workflow to show for it.
        """
        return True

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Polyhedron/Sampling"
    DESCRIPTION = ("Chooses the attention kernel for the diffusion model. "
                   "Lists only backends that this installation can load, and "
                   "launches the chosen one once before patching.")

    def patch(self, model, attention, fallback=True,
              attention_first=SAME_AS_MAIN, first_steps=0,
              attention_last=SAME_AS_MAIN, last_steps=0,
              sparse_time_window=3, sparse_sink=True, live_check=True,
              latent=None):
        m = model.clone()

        # Sentinel: "same as main" and 0 steps mean no window at all, and the
        # override built below is then byte-for-byte the v841 one.
        first_mode = None if attention_first == SAME_AS_MAIN else attention_first
        last_mode = None if attention_last == SAME_AS_MAIN else attention_last
        if first_mode == DEFAULT_MODE or last_mode == DEFAULT_MODE:
            # "default" is the absence of an override, which cannot be
            # expressed per step -- the slot is set for the whole run or not
            # at all. Say so instead of pretending.
            print("[PLS] Attention: '%s' cannot be used inside a step window "
                  "(it means 'no override at all') -- window ignored"
                  % DEFAULT_MODE)
            if first_mode == DEFAULT_MODE:
                first_mode = None
            if last_mode == DEFAULT_MODE:
                last_mode = None

        # Copy-on-write: never mutate a transformer_options dict that another
        # branch of the graph may still be holding.
        tops = m.model_options.get("transformer_options", None)
        tops = dict(tops) if isinstance(tops, dict) else {}

        if attention == DEFAULT_MODE:
            # Trap A: remove the key, never park a None in it.
            tops.pop("optimized_attention_override", None)
            m.model_options["transformer_options"] = tops
            print("[PLS] Attention: default -- model left on its own backend")
            return (m,)

        # A workflow may carry a mode this installation does not offer -- it
        # was saved somewhere with a different card, wheel or start-up choice.
        # Validation no longer rejects it (see VALIDATE_INPUTS), so this is
        # where it gets an answer: name it, explain it where we can, and leave
        # the model alone. The run continues.
        offered = available_modes()
        if attention not in offered:
            tops.pop("optimized_attention_override", None)
            m.model_options["transformer_options"] = tops
            why = describe_absence(attention)
            print("[PLS] Attention: this workflow asks for '%s', which THIS "
                  "installation does not offer%s"
                  % (attention, (" (%s)" % why) if why else "."))
            for extra_line in explain_failure(attention, why):
                print("[PLS]   %s" % extra_line)
            print("[PLS]   available here: %s" % ", ".join(offered))
            print("[PLS]   -> model left on its own backend, nothing patched. "
                  "Pick one of the above to silence this.")
            return (m,)

        # --- sparse preparation ------------------------------------------
        sparse = None
        wants_sparse = SPARSE_LOCAL in (attention, first_mode, last_mode)

        def _drop_sparse(why, detail=None):
            """One exit for every reason sparse cannot run. Always loud:
            a mode that quietly turns itself off would leave the user
            believing a mask is in place that is not."""
            print("[PLS] Attention: %s" % why)
            if detail:
                print("[PLS]   %s" % detail)

        if wants_sparse:
            geom = latent_geometry(latent)
            if geom is None:
                _drop_sparse("the sparse mode needs a latent to know its frame "
                             "layout -- none wired (or not a 5D video latent).",
                             "-> sparse dropped, nothing masked.")
                wants_sparse = False
            elif int(sparse_time_window) <= 0:
                _drop_sparse("sparse_time_window is 0 -- that is dense "
                             "attention with extra steps.",
                             "-> sparse dropped, nothing masked.")
                wants_sparse = False
            else:
                frames, per_frame, total = geom
                dens = local_density(frames, int(sparse_time_window),
                                     bool(sparse_sink))
                sok, snote = probe_sparse(frames, int(sparse_time_window),
                                          bool(sparse_sink))
                if not sok:
                    _drop_sparse("sparse did NOT verify here (%s)" % snote,
                                 "-> sparse dropped, nothing masked.")
                    wants_sparse = False
                else:
                    sparse = {"frames": frames, "per_frame": per_frame,
                              "total": total,
                              "window": int(sparse_time_window),
                              "sink": bool(sparse_sink)}
                    print("[PLS] Attention: sparse geometry %d frame(s) x %d "
                          "token(s) = %d; window +/-%d%s -> %.0f%% of the "
                          "frame pairs kept (%s)"
                          % (frames, per_frame, total,
                             int(sparse_time_window),
                             ", frame 0 sink" if sparse_sink else "",
                             dens * 100.0, snote))
                    print("[PLS]   note: sparse runs in bf16. Against an "
                          "unquantised backend that is a real saving; against "
                          "an int8 kernel it is roughly a tie. The live check "
                          "below settles it on this geometry.")
            if not wants_sparse:
                if attention == SPARSE_LOCAL:
                    tops.pop("optimized_attention_override", None)
                    m.model_options["transformer_options"] = tops
                    print("[PLS]   -> model left on its own backend.")
                    return (m,)
                first_mode = None if first_mode == SPARSE_LOCAL else first_mode
                last_mode = None if last_mode == SPARSE_LOCAL else last_mode

        if attention == SPARSE_LOCAL:
            # Already launched and mask-verified above; the generic probe
            # would need the geometry we have only just resolved.
            ok, note = True, "sparse verified"
        else:
            ok, note = probe(attention)
        if not ok:
            tops.pop("optimized_attention_override", None)
            m.model_options["transformer_options"] = tops
            print("[PLS] Attention: '%s' did NOT run on this machine (%s)"
                  % (attention, note))
            for extra_line in explain_failure(attention, note):
                print("[PLS]   %s" % extra_line)
            print("[PLS]   -> model left on its own backend, nothing patched.")
            return (m,)

        def _report(exc):
            print("[PLS] Attention: '%s' failed mid-run (%s: %s) -- "
                  "continuing on the model's own backend"
                  % (attention, type(exc).__name__,
                     str(exc).split("\n")[0][:100]))

        # Every kernel that can be reached during the run is launched now, not
        # at the step where it would first be needed -- a window that dies on
        # step 7 of 8 has already cost the run.
        plan = ""
        for label, mode_, n in (("first", first_mode, first_steps),
                                ("last", last_mode, last_steps)):
            if mode_ is None or n <= 0:
                continue
            if mode_ not in offered:
                # Same reasoning as the main mode above: a saved workflow may
                # name a window backend this machine does not have. Say so
                # here rather than letting the launch probe report it as a
                # broken kernel -- and check it BEFORE probing, so the answer
                # does not depend on whether a GPU is present to probe with.
                why = describe_absence(mode_)
                print("[PLS] Attention: window '%s' asks for '%s', which THIS "
                      "installation does not offer%s -- window dropped, main "
                      "backend used throughout"
                      % (label, mode_, (" (%s)" % why) if why else "."))
                if label == "first":
                    first_mode = None
                else:
                    last_mode = None
                continue
            if mode_ == SPARSE_LOCAL:
                # Already launched AND mask-verified in the sparse block
                # above. The generic probe cannot judge it: build_router
                # needs the geometry, which only patch() has resolved, so
                # probe(SPARSE_LOCAL) answers "unknown mode" and would drop
                # a window that is perfectly prepared. That is exactly what
                # it did from v845 until this was found (v847) -- silently,
                # because dropping a window prints a plausible line.
                wok, wnote = (sparse is not None), "sparse verified"
                if not wok:
                    wnote = "sparse could not be prepared"
            else:
                wok, wnote = probe(mode_)
            if not wok:
                print("[PLS] Attention: window '%s' (%s) did NOT run here (%s)"
                      " -- window dropped, main backend used throughout"
                      % (label, mode_, wnote))
                if label == "first":
                    first_mode = None
                else:
                    last_mode = None
            else:
                plan += " | %s %d step(s): %s" % (label, n, mode_)

        def _announce(step, total):
            if step is None:
                print("[PLS] Attention: no step schedule in transformer_options"
                      " -- windows inactive, '%s' used throughout" % attention)
            else:
                print("[PLS] Attention: step schedule visible (%d steps)"
                      % total)

        def _passthrough(mode_, skip):
            print("[PLS] Attention: not every call is '%s' to serve (%s) -- "
                  "those run on the model's own backend, the rest on '%s'"
                  % (mode_, str(skip).split("\n")[0][:80], mode_))
            print("[PLS]   this is normal, not a failure -- but it means the "
                  "mode does not cover the whole run.")

        def _verdict(mode_, ms_chosen, ms_base, dev):
            if ms_chosen is None:
                print("[PLS] Attention: live check could NOT measure on this "
                      "call -- no numbers this run. The run itself is "
                      "unaffected; '%s' is serving normally." % mode_)
                return
            line = format_verdict(mode_, ms_chosen, ms_base, dev)
            if line:
                print(line)
            else:
                print("[PLS] Attention: live check ran but produced no usable "
                      "timing -- no numbers this run.")

        override = build_override(attention, fallback=fallback,
                                  report=_report,
                                  first_mode=first_mode,
                                  first_steps=first_steps,
                                  last_mode=last_mode,
                                  last_steps=last_steps,
                                  announce=_announce, sparse=sparse,
                                  live_check=bool(live_check),
                                  verdict=_verdict,
                                  passthrough=_passthrough)
        if override is None:
            tops.pop("optimized_attention_override", None)
            m.model_options["transformer_options"] = tops
            print("[PLS] Attention: unknown mode '%s' -- nothing patched"
                  % attention)
            return (m,)

        # v903: SAY IT when we are overwriting somebody else's choice.
        # Core's ModelAttentionBackend writes the same key, so a graph with
        # both nodes silently runs whichever sits later. That is a real trap
        # -- Frank's H3 template ships with ModelAttentionBackend in it -- and
        # a trap nobody can see is worse than the collision itself.
        prior = tops.get("optimized_attention_override", None)
        if prior is not None:
            if getattr(prior, "_pls_owner", None) == "ULSAttention":
                print("[PLS] Attention: another Polyhedron Attention node "
                      "already set a backend on this model -- this node "
                      "REPLACES it. Only the last one in the chain runs.")
            else:
                print("[PLS] Attention: something upstream already set an "
                      "attention backend on this model (Core's "
                      "ModelAttentionBackend writes the same key) -- this "
                      "node REPLACES it, so that setting will NOT run. Use "
                      "'%s' here if you meant to keep it." % DEFAULT_MODE)
        try:
            override._pls_owner = "ULSAttention"
        except Exception:
            pass                      # a builtin or slotted callable: harmless
        tops["optimized_attention_override"] = override
        m.model_options["transformer_options"] = tops
        print("[PLS] Attention: %s (%s)%s%s"
              % (attention, note, plan,
                 "" if fallback else " [no fallback]"))
        return (m,)
