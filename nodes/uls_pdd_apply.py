# -*- coding: utf-8 -*-
"""
uls_pdd_apply.py
════════════════
v933 -- putting the fused heads onto a model, and telling the sampler about it.

THE MECHANISM, AND WHY IT IS NOT A LORA
---------------------------------------
The trunk LoRA is an ordinary weight delta and travels the ordinary path. The
head bank is not: it REPLACES the model's output projections, and which copy is
in place depends on where in the schedule the sampler currently is. That is a
per-step swap of a module, driven by sigma -- nothing a LoRA loader can express,
which is why this module exists at all.

Two pieces do it:

  * two object patches, one on each output projection of `final_layer`
    (`video_out`, `audio_out`), each an nn.Module that serves the fused head
    of the CURRENT block;
  * a DIFFUSION_MODEL wrapper that records the sigma of each forward before
    delegating, so the heads know which block that is.

v935 -- WHY THE PATCH SITS ON THE PROJECTIONS (measured against core 7a131a3a)
-----------------------------------------------------------------------------
v933 replaced `final_layer` itself with a plain-object stand-in. Driven for
real, that failed twice over: torch refuses a non-Module as a child module
(`patch_model` raised TypeError), and even had it been accepted, core's
FinalLayer.forward runs with `self` bound to the NATIVE layer and calls
`self.video_out` -- the stand-in's heads would never have been reached, and
the run would have produced core's own output without a word. Patching the
projections leaves core's forward exactly as it is and makes it call ours.

The sigma is read, never counted. A counter cannot survive a looping or
chunked sampler, a resume, or a split schedule. Core writes the raw sigma into
transformer_options["sigmas"] before every model call (comfy/samplers.py);
that is what the wrapper reads. The `timestep` argument is NOT a sigma -- core
passes model_sampling.timestep(sigma) = sigma * multiplier (1000) -- and is
used only as a fallback, divided by the model's own multiplier. (v933 read
`timestep` as if it were sigma; every forward would have been off-grid.)

FAIL CLOSED
-----------
If a forward arrives at a sigma that is not a block start, the shim raises
rather than picking the nearest block. An off-grid sigma means the schedule is
wrong -- the wrong sampler, a foreign scheduler, a shift that is not 12/3 --
and every one of those produces noise that looks like a bad seed. Better a
message naming the cause.

WHAT THE SAMPLER SEES
---------------------
The state rides on the ModelPatcher as an attachment (the v913 pattern, which
survives clone). The sampler reads it to build the trained sigma schedule
itself; nothing needs wiring in the graph.
"""

PDD_KEY = "polyhedron_pdd"

VIDEO_ATTR = "video_out"
AUDIO_ATTR = "audio_out"
FINAL_PATH = "diffusion_model.final_layer"

# The head bank, exactly as the published files name it. Everything else in a
# PDD file is the trunk LoRA.
BANK_KEYS = ("proj_out.weight", "proj_out.bias",
             "audio_proj_out.weight", "audio_proj_out.bias")


class PDDState(object):
    """What a PDD-patched model carries. Immutable once built.

    fused_v/fused_a : (weights [P, out, in], biases [P, out] or None), fp32
    boundaries      : the sigma at which each block starts, plus the final 0
    sizes           : block sizes in fine steps
    bank            : the RAW per-interval bank, kept so the sampler can fuse
                      again for the step count it is actually asked for
    """

    __slots__ = ("fused_v", "fused_a", "boundaries", "sizes", "nfe",
                 "trunk", "source", "bank")

    def __init__(self, fused_v, fused_a, boundaries, sizes, nfe,
                 trunk=None, source=None, bank=None):
        self.fused_v = fused_v
        self.fused_a = fused_a
        self.boundaries = tuple(float(x) for x in boundaries)
        self.sizes = tuple(int(x) for x in sizes)
        self.nfe = int(nfe)
        self.trunk = trunk
        self.source = source
        self.bank = bank

    def on_model_patcher_clone(self):
        return self                     # immutable -> share across clones

    def block_of(self, sigma):
        """Which block this sigma starts, or None. Delegates to the maths."""
        try:
            from .uls_pdd_math import select_block
        except ImportError:
            from uls_pdd_math import select_block
        return select_block(sigma, self.boundaries)

    def describe(self):
        return ("PDD %s: %d steps, blocks %s, boundaries %s"
                % (self.source or "acc", self.nfe, list(self.sizes),
                   ["%.4f" % b for b in self.boundaries]))


class _SigmaHolder(object):
    """One mutable cell shared by the wrapper and the two heads."""

    __slots__ = ("sigma",)

    def __init__(self):
        self.sigma = None


_HEAD_CLS = None


def _head_cls():
    """The head shim class, built on first use so importing this module never
    needs torch (the maths and the guards run without it)."""
    global _HEAD_CLS
    if _HEAD_CLS is not None:
        return _HEAD_CLS
    import torch

    class PDDHead(torch.nn.Module):
        """Stands in for ONE output projection, serving the live block's head.

        An nn.Module because torch accepts nothing else as a child module. It
        owns NO parameters and no buffers on purpose: core's loader decides
        what to move, cast and count by named_parameters / comfy_cast_weights
        (model_patcher._load_list), so a parameter-free shim is never touched
        by it. The fused heads are plain attributes, moved to a device once
        and cached there. The native projection is kept as a plain attribute
        too -- registered, it would re-enter the module tree it was patched
        out of.
        """

        def __init__(self, native, fused, holder, state, which):
            super().__init__()
            object.__setattr__(self, "native", native)
            object.__setattr__(self, "_fused", fused)
            object.__setattr__(self, "_on", {})
            object.__setattr__(self, "holder", holder)
            object.__setattr__(self, "state", state)
            object.__setattr__(self, "which", which)

        @property
        def in_features(self):
            return int(self._fused[0].shape[2])

        @property
        def out_features(self):
            return int(self._fused[0].shape[1])

        # v965: Core master (commit 2504e68d, 28.08.2026) grew its own PDD head
        # bank and FinalLayer.forward now opens with
        #     n = self.video_out.weight.shape[0] // self.video_out.out_features
        # before it decides which path to take. A stand-in without `weight`
        # dies there with AttributeError. Handing back ONE block's weight
        # ([out, in]) makes n == 1, Core takes its ordinary path
        # `self.video_out(...)`, and that call lands in forward() below with
        # our block selection -- the documented 4..8-step schedule stays ours
        # on both Cores. Core's own bank (dt-weighted mean of the heads a step
        # spans, any step count) is a different doctrine and expects a LoRA
        # already in ComfyUI keys; it is NOT delegated to.
        @property
        def weight(self):
            return self._fused[0][0]

        @property
        def bias(self):
            b = self._fused[1]
            return None if b is None else b[0]

        def extra_repr(self):
            return "PDD %s head, %d blocks, %d -> %d" % (
                self.which, int(self._fused[0].shape[0]),
                self.in_features, self.out_features)

        def _block(self):
            sigma = self.holder.sigma
            if sigma is None:
                raise RuntimeError(
                    "[PLS] PDD: a forward reached the output head without a "
                    "sigma. The wrapper that records it is not installed -- the "
                    "model was probably cloned past it.")
            idx = self.state.block_of(sigma)
            if idx is None:
                raise RuntimeError(
                    "[PLS] PDD: sigma %.6f is not a trained block boundary. The "
                    "heads are only valid at %s. Run the Polyhedron Sampler with "
                    "sampler 'euler' and no external SIGMAS -- it builds exactly "
                    "this schedule from the head bank."
                    % (float(sigma), ["%.4f" % b for b in self.state.boundaries]))
            return idx

        def _at(self, device, dtype):
            key = (str(device), dtype)
            hit = self._on.get(key)
            if hit is None:
                w, b = self._fused
                hit = (w.to(device=device, dtype=dtype),
                       None if b is None else b.to(device=device, dtype=dtype))
                self._on[key] = hit
            return hit

        def forward(self, x):
            import torch.nn.functional as F
            idx = self._block()
            w, b = self._at(x.device, x.dtype)
            return F.linear(x, w[idx], None if b is None else b[idx])

    _HEAD_CLS = PDDHead
    return _HEAD_CLS


def _native_of(module):
    """The real projection behind `module`. If a previous run left our head in
    place (core keeps a model patched while it stays loaded), unwrap it --
    wrapping a head in a head would serve stale blocks."""
    cls = _HEAD_CLS
    while cls is not None and isinstance(module, cls):
        module = module.native
    return module


def split_head_bank(sd):
    """Pull the four head-bank tensors out of a PDD file. None when absent."""
    if not all(k in sd for k in BANK_KEYS):
        return None
    return {"video": (sd[BANK_KEYS[0]], sd[BANK_KEYS[1]]),
            "audio": (sd[BANK_KEYS[2]], sd[BANK_KEYS[3]])}


def build_state(bank, nfe=None, partition=None, source=None, trunk=None):
    """Fuse the bank for this step count and describe the schedule.

    Raises ValueError with a plain reason for an off-envelope request.
    """
    try:
        from . import uls_pdd_math as M
    except ImportError:
        import uls_pdd_math as M

    n_heads = int(bank["video"][0].shape[0])
    sizes = M.resolve_partition(nfe=nfe, partition=partition, num_steps=n_heads)
    fused_v = M.fuse_heads(bank["video"][0], bank["video"][1],
                           M.block_plans(M.VIDEO_SHIFT, sizes, n_heads))
    fused_a = M.fuse_heads(bank["audio"][0], bank["audio"][1],
                           M.block_plans(M.AUDIO_SHIFT, sizes, n_heads))
    return PDDState(fused_v, fused_a,
                    M.boundary_sigmas(sizes, M.VIDEO_SHIFT, n_heads),
                    sizes, len(sizes), trunk=trunk, source=source, bank=bank)


def final_layer_of(model):
    """core's final_layer with both projections, or raise RuntimeError."""
    inner = getattr(getattr(model, "model", model), "diffusion_model", None)
    final = getattr(inner, "final_layer", None) if inner is not None else None
    if (final is None or not hasattr(final, VIDEO_ATTR)
            or not hasattr(final, AUDIO_ATTR)):
        raise RuntimeError(
            "[PLS] PDD: this model has no final_layer with %s/%s to patch. The "
            "head bank belongs to MiniMax-H3; applying it elsewhere would be "
            "meaningless." % (VIDEO_ATTR, AUDIO_ATTR))
    return final


def _multiplier_of(model):
    try:
        return float(model.model.model_sampling.multiplier)
    except Exception:
        return 1000.0


def sigma_of_call(timestep, args, kwargs, multiplier):
    """The sigma of one diffusion-model call. transformer_options["sigmas"]
    first (core writes the raw sigma there), timestep / multiplier second."""
    to = None
    if len(args) >= 2 and isinstance(args[1], dict):
        to = args[1]
    elif isinstance(kwargs.get("transformer_options"), dict):
        to = kwargs["transformer_options"]
    try:
        s = to.get("sigmas") if to is not None else None
        if s is not None:
            return float(s.flatten()[0])
    except Exception:
        pass
    try:
        return float(timestep.flatten()[0]) / float(multiplier)
    except Exception:
        return None


def attach(model, state, log=None):
    """Return a clone carrying the heads, the sigma wrapper and the state.

    The input model is not mutated. Raises RuntimeError when the model has no
    final_layer to patch -- an honest failure beats a model that looks patched.
    """
    import comfy.patcher_extension

    final = final_layer_of(model)
    Head = _head_cls()
    holder = _SigmaHolder()
    v_head = Head(_native_of(getattr(final, VIDEO_ATTR)), state.fused_v,
                  holder, state, "video")
    a_head = Head(_native_of(getattr(final, AUDIO_ATTR)), state.fused_a,
                  holder, state, "audio")
    mult = _multiplier_of(model)

    m = model.clone()
    m.add_object_patch(FINAL_PATH + "." + VIDEO_ATTR, v_head)
    m.add_object_patch(FINAL_PATH + "." + AUDIO_ATTR, a_head)

    def _wrapper(executor, x, timestep, *args, **kwargs):
        holder.sigma = sigma_of_call(timestep, args, kwargs, mult)
        try:
            return executor(x, timestep, *args, **kwargs)
        finally:
            holder.sigma = None

    m.remove_wrappers_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, PDD_KEY)
    m.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, PDD_KEY, _wrapper)
    m.set_attachments(PDD_KEY, state)
    if log:
        log(state.describe())
    return m


def refit(model, nfe, log=None):
    """The model with its heads fused for `nfe` steps.

    Same model when it already carries that step count; otherwise a clone
    re-attached from the raw bank. Raises ValueError for an illegal count
    (the maths names the legal ones) and RuntimeError when the model carries
    no PDD state at all.
    """
    st = state_of(model)
    if st is None:
        raise RuntimeError("[PLS] PDD: refit on a model without a head bank")
    if int(nfe) == st.nfe:
        return model
    if st.bank is None:
        raise RuntimeError("[PLS] PDD: this state carries no raw bank to re-fuse")
    return attach(model, build_state(st.bank, nfe=int(nfe), source=st.source,
                                     trunk=st.trunk), log=log)


def state_of(model):
    """The PDD state on this model, or None. Never raises."""
    try:
        return model.get_attachment(PDD_KEY)
    except Exception:
        return None
