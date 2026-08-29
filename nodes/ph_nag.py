"""Polyhedron NAG -- normalized attention guidance for Wan, with the silent
parts made loud.

WHAT NAG IS FOR. At CFG 1 there is no second model pass, so the sampler's
negative prompt does nothing at all. NAG (arXiv 2505.21179, Chen et al.) puts
negative control back -- not at the model output where CFG works, but inside
the CROSS-ATTENTION. Each block runs its cross-attention twice, once against
the real prompt and once against a negative context, and then:

    1. extrapolate   Z~ = Z+ + scale * (Z+ - Z-)          -> direction
    2. L1 norm ratio Z~/Z+, pulled down to `tau` when above -> length
    3. blend         alpha * Z~ + (1 - alpha) * Z+         -> how much lands

The economics are the point: this doubles only the cross-attention over the
few hundred text tokens, not the whole model pass the way CFG does.

WHY THIS EXISTS ALONGSIDE KJNodes' WanVideoNAG. The maths in KJ's node is
correct and this file does not try to improve it -- the extrapolation, the L1
refinement and the blend are the paper's and are reproduced faithfully. What
this node changes is everything AROUND the maths, where KJ's version is
silent in ways that cost real runs:

  * `input_type` is a trap. On "default" it inspects context.shape[0] and,
    finding more than one, ASSUMES a CFG pair, halves the batch and guides
    only the first half. Under CFG 1 with a real batch of 2 that means half
    the batch silently renders without NAG (KJNodes issue #354). We read
    ComfyUI's own `cond_or_uncond` out of transformer_options instead, which
    says what the batch actually is, and fall back to the shape heuristic
    only when it is absent -- announcing that it did.
  * Two settings are SILENT no-ops. nag_scale = 1 cancels algebraically back
    to Z+, and nag_alpha = 0 blends the result away entirely. Both cost the
    full doubled cross-attention and change nothing. KJ's node only exits
    cleanly at scale 0. We say so, once, in words.
  * No start/end percent, so guidance runs on every step whether or not that
    is wanted. Late steps are where NAG's extrapolation is most likely to
    hurt fine detail, and there is no way to stop it.
  * The conditioning's mask, strength and timestep range are read off and
    discarded -- only conditioning[0][0], the bare tensor, survives. That is
    defensible, but a user who attached a mask deserves to hear that it was
    ignored.
  * A non-Wan model raises AttributeError somewhere deep in the patch loop.
    We check first and say which model this is and why it cannot work.

NOT SUPPORTED, and the reason is architectural, not laziness: MiniMax H3 has
no cross-attention at all. Text, video and audio are concatenated into ONE
sequence (see PackedLayout in comfy/ldm/minimax/model.py) and run through
self-attention together. NAG's cheap-negative trick relies on the negative
context passing through a SMALL attention against the text tokens only; on H3
the same trick would mean running the full self-attention over the whole
packed sequence twice, which costs what CFG 2 costs. The method would still
be correct there and the saving would be gone. So this node refuses H3 rather
than shipping an expensive surprise.
"""

import torch

import comfy.ldm.modules.attention

# Defaults are the paper's / KJ's, deliberately: a mode of the same name must
# behave the same way so numbers stay comparable across the two nodes.
NAG_SCALE_DEFAULT = 11.0
NAG_ALPHA_DEFAULT = 0.25
NAG_TAU_DEFAULT = 2.5


def _cross_attention(module, query, context, transformer_options):
    """One cross-attention pass. Verbatim in shape and order with Wan's own."""
    k = module.norm_k(module.k(context))
    v = module.v(context)
    return comfy.ldm.modules.attention.optimized_attention(
        query, k, v, heads=module.num_heads,
        transformer_options=transformer_options).flatten(2)


def normalized_attention_guidance(x_positive, x_negative, scale, alpha, tau):
    """The paper's three steps. Kept out of the class so a guard can run it
    against hand-computed numbers without building a model."""
    # 1. extrapolate: Z~ = Z+ * scale - Z- * (scale - 1)
    guidance = (x_positive * scale) - (x_negative * (scale - 1.0))

    # 2. L1 refinement: where the guided vector grew more than tau times the
    #    positive one, pull it back to exactly tau.
    norm_positive = torch.norm(x_positive, p=1, dim=-1, keepdim=True)
    norm_guidance = torch.norm(guidance, p=1, dim=-1, keepdim=True)
    ratio = norm_guidance / norm_positive
    torch.nan_to_num_(ratio, nan=10.0)
    over = ratio > tau
    adjustment = (norm_positive * tau) / (norm_guidance + 1e-7)
    guidance = guidance * torch.where(over, adjustment, torch.ones_like(adjustment))

    # 3. blend
    return guidance * alpha + x_positive * (1.0 - alpha)


def split_batch(x, context, transformer_options):
    """(positive slice, negative slice, how we know) for this call.

    THE WHOLE POINT OF THIS FUNCTION. ComfyUI puts `cond_or_uncond` into
    transformer_options: a list with one entry per batch item, 0 for a
    conditional row and 1 for an unconditional one. That is a STATEMENT about
    the batch, not a guess about it. KJ's node instead reads context.shape[0]
    and treats anything above 1 as a CFG pair -- which is right under CFG > 1
    and wrong for a real batch under CFG 1, where it quietly leaves half the
    batch unguided.

    Returns (n_positive, n_negative, source). n_negative counts the trailing
    rows that are genuine uncond rows and must NOT be guided.
    """
    n = x.shape[0]
    marks = None
    if isinstance(transformer_options, dict):
        marks = transformer_options.get("cond_or_uncond", None)

    if isinstance(marks, (list, tuple)) and len(marks) == n and n > 0:
        # Rows are grouped, conds first in every sampler Core ships.
        n_pos = sum(1 for m in marks if m == 0)
        return n_pos, n - n_pos, "cond_or_uncond"

    # No statement available: fall back to KJ's heuristic, and SAY so.
    if n % 2 == 0 and n > 1:
        return n // 2, n // 2, "shape guess (even batch assumed to be a CFG pair)"
    return n, 0, "shape guess (odd batch assumed all-conditional)"


class NagCrossAttention(object):
    """Replacement forward for one block's cross_attn.

    Bound per block. Holds no tensors of its own beyond the negative context,
    which is shared across blocks.
    """

    def __init__(self, context, scale, alpha, tau, i2v, window, report):
        self.nag_context = context
        self.scale = scale
        self.alpha = alpha
        self.tau = tau
        self.i2v = i2v
        self.window = window          # (start_pct, end_pct) or None
        self.report = report          # one-shot console callback

    def _in_window(self, transformer_options):
        """True when this step is inside start/end percent.

        Read from `sigmas` in transformer_options, the same place the sampler
        publishes its schedule. When it is absent we run -- a guidance node
        that silently stops guiding because it could not read a clock would
        be worse than one that ignores the window.
        """
        if self.window is None:
            return True, None
        lo, hi = self.window
        try:
            sigmas = transformer_options.get("sample_sigmas", None)
            step = transformer_options.get("sigmas", None)
            if sigmas is None or step is None:
                return True, "no schedule in transformer_options"
            s_max = float(sigmas[0])
            s_min = float(sigmas[-1])
            cur = float(step[0])
            if s_max <= s_min:
                return True, "degenerate schedule"
            pct = 1.0 - (cur - s_min) / (s_max - s_min)
        except Exception as exc:
            return True, "schedule unreadable (%s)" % exc
        return (lo <= pct <= hi), None

    def __call__(self, module, x, context, transformer_options={}, **kwargs):
        context_img = None
        if self.i2v:
            n_img = kwargs.get("context_img_len", None)
            if n_img is None:
                n_img = getattr(module, "context_img_len", 257)
            context_img = context[:, :n_img]
            context = context[:, n_img:]

        inside, why = self._in_window(transformer_options)
        n_pos, n_neg, how = split_batch(x, context, transformer_options)
        self.report(how, why, inside)

        img_x = None
        if self.i2v and context_img is not None:
            q_img = module.norm_q(module.q(x))
            k_img = module.norm_k_img(module.k_img(context_img))
            v_img = module.v_img(context_img)
            img_x = comfy.ldm.modules.attention.optimized_attention(
                q_img, k_img, v_img, heads=module.num_heads,
                transformer_options=transformer_options)
            del q_img, k_img, v_img

        if not inside:
            # Outside the window: the ordinary Wan cross-attention, untouched.
            q = module.norm_q(module.q(x))
            out = _cross_attention(module, q, context, transformer_options)
            return module.o(out if img_x is None else out + img_x)

        x_pos = x[:n_pos]
        ctx_pos = context[:n_pos]
        q_pos = module.norm_q(module.q(x_pos))

        neg_ctx = self.nag_context
        if neg_ctx.shape[0] != n_pos:
            neg_ctx = neg_ctx.repeat(n_pos, 1, 1)

        x_positive = _cross_attention(module, q_pos, ctx_pos,
                                      transformer_options)
        x_negative = _cross_attention(module, q_pos, neg_ctx,
                                      transformer_options)
        del q_pos, ctx_pos

        out = normalized_attention_guidance(x_positive, x_negative,
                                            self.scale, self.alpha, self.tau)
        del x_positive, x_negative

        if n_neg > 0:
            # Real uncond rows: plain cross-attention, NEVER guided. Guiding
            # them would push the negative branch toward the negative prompt,
            # which is the opposite of the intent.
            x_neg = x[n_pos:]
            q_neg = module.norm_q(module.q(x_neg))
            out_neg = _cross_attention(module, q_neg, context[n_pos:],
                                       transformer_options)
            out = torch.cat([out, out_neg], dim=0)

        return module.o(out if img_x is None else out + img_x)


def describe_model(model):
    """(diffusion_model, kind, why-not) -- the architecture check, up front.

    KJ's node walks straight into `diffusion_model.blocks` and lets a non-Wan
    model raise AttributeError from inside the patch loop, where the message
    names a missing attribute rather than the actual problem.
    """
    try:
        dm = model.get_model_object("diffusion_model")
    except Exception as exc:
        return None, "unknown", "the model exposes no diffusion_model (%s)" % exc

    name = type(dm).__name__
    if not hasattr(dm, "blocks"):
        return dm, name, ("%s has no `blocks` list -- this node patches Wan's "
                          "per-block cross-attention and has nothing to attach "
                          "to here" % name)
    if not hasattr(dm, "text_embedding"):
        return dm, name, ("%s has no `text_embedding` -- the negative prompt "
                          "cannot be embedded the way Wan embeds it" % name)
    blocks = list(dm.blocks)
    if not blocks or not hasattr(blocks[0], "cross_attn"):
        return dm, name, ("%s has no cross-attention in its blocks. On models "
                          "that concatenate text into ONE self-attention "
                          "sequence (MiniMax H3 is the one Frank runs), NAG "
                          "would have to double the FULL self-attention over "
                          "the whole packed sequence instead of a few hundred "
                          "text tokens -- costing what CFG 2 costs. The method "
                          "would still be correct; the saving that makes it "
                          "worth using would be gone" % name)
    return dm, name, None


class ULSNag:
    """Normalized attention guidance for Wan, with the quiet parts made loud."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING", {
                    "tooltip": "THE NEGATIVE PROMPT -- and not the one wired "
                               "into the sampler. At CFG 1 the sampler's "
                               "negative does nothing; this is the one that "
                               "acts. Mixing the two up is the most common "
                               "mistake with this method."}),
                "nag_scale": ("FLOAT", {
                    "default": NAG_SCALE_DEFAULT, "min": 0.0, "max": 100.0,
                    "step": 0.001,
                    "tooltip": "How far to extrapolate away from the negative "
                               "context. 0 disables the node entirely (no "
                               "cost). 1 is a NO-OP that still pays the full "
                               "doubled cross-attention -- the node says so. "
                               "Large values mostly hit the tau ceiling; if "
                               "raising this stops changing anything, tau is "
                               "the knob that still moves."}),
                "nag_alpha": ("FLOAT", {
                    "default": NAG_ALPHA_DEFAULT, "min": 0.0, "max": 1.0,
                    "step": 0.001,
                    "tooltip": "How much of the guided result is blended in. "
                               "0 blends none of it -- a no-op at full cost, "
                               "which the node reports."}),
                "nag_tau": ("FLOAT", {
                    "default": NAG_TAU_DEFAULT, "min": 0.0, "max": 10.0,
                    "step": 0.001,
                    "tooltip": "Ceiling on how far the guided attention may "
                               "grow, as a multiple of the positive "
                               "attention's L1 length. This is what stops "
                               "large nag_scale values from running away, and "
                               "the reason a very large scale plateaus."}),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Guide only from this fraction of the schedule "
                               "onward. Outside the window the block runs its "
                               "ordinary cross-attention at normal cost."}),
                "end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "...and up to this fraction. Late steps are "
                               "where extrapolation is most likely to cost "
                               "fine detail, so ending early is a real knob."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "Polyhedron/Wan"
    DESCRIPTION = (
        "Normalized attention guidance (arXiv 2505.21179) for Wan: brings the "
        "negative prompt back at CFG 1 by guiding inside the cross-attention "
        "instead of at the model output. Costs a second pass over the text "
        "tokens only, not a second model pass. Reads ComfyUI's own "
        "cond_or_uncond so a real batch is not mistaken for a CFG pair, has a "
        "start/end window, and says out loud when a setting makes it a no-op."
    )

    def patch(self, model, conditioning, nag_scale, nag_alpha, nag_tau,
              start_percent=0.0, end_percent=1.0):
        if nag_scale == 0:
            print("[PLS] NAG: nag_scale is 0 -> nothing patched, no cost.")
            return (model,)

        # The two silent no-ops, said out loud. Both cost the full doubled
        # cross-attention and change the result by nothing.
        if abs(nag_scale - 1.0) < 1e-9:
            print("[PLS] NAG: nag_scale is 1 -- the extrapolation cancels "
                  "algebraically back to the positive attention. This run "
                  "pays the DOUBLED cross-attention and changes nothing. Use "
                  "0 to switch the node off, or raise the scale.")
        if abs(nag_alpha) < 1e-9:
            print("[PLS] NAG: nag_alpha is 0 -- none of the guided result is "
                  "blended in. Same cost, no effect. Use nag_scale 0 to "
                  "switch the node off.")

        dm, kind, refusal = describe_model(model)
        if refusal is not None:
            print("[PLS] NAG: not patching. %s" % refusal)
            return (model,)

        # What we take from the conditioning, and what we drop.
        extras = set()
        try:
            for _tensor, meta in conditioning:
                if isinstance(meta, dict):
                    extras.update(k for k in meta
                                  if k in ("mask", "strength", "area",
                                           "start_percent", "end_percent",
                                           "set_area_to_bounds"))
        except Exception:
            pass
        if extras:
            print("[PLS] NAG: the negative conditioning carries %s -- IGNORED. "
                  "Only the raw embedding is used, the same as KJ's node. If "
                  "you meant those to apply, they will not."
                  % ", ".join(sorted(extras)))
        if len(conditioning) > 1:
            print("[PLS] NAG: the negative conditioning has %d parts; only the "
                  "first is used." % len(conditioning))

        import comfy.model_management as mm
        device = mm.get_torch_device()
        dtype = mm.unet_dtype()

        m = model.clone()
        dm.text_embedding.to(device)
        context = dm.text_embedding(conditioning[0][0].to(device, dtype))

        lo, hi = float(start_percent), float(end_percent)
        if lo > hi:
            print("[PLS] NAG: start_percent %.3f is above end_percent %.3f -- "
                  "that window is empty, so the node would never guide. "
                  "Swapping them." % (lo, hi))
            lo, hi = hi, lo
        window = None if (lo <= 0.0 and hi >= 1.0) else (lo, hi)

        said = {"how": None, "why": None, "outside": False}

        def report(how, why, inside):
            # One line per distinct fact, not one per block per step.
            if said["how"] != how:
                said["how"] = how
                if how != "cond_or_uncond":
                    print("[PLS] NAG: could not read cond_or_uncond -- falling "
                          "back to a %s. If this is a real batch under CFG 1, "
                          "check the result." % how)
            if why and said["why"] != why:
                said["why"] = why
                print("[PLS] NAG: start/end window requested but %s -- guiding "
                      "on every step." % why)
            if not inside and not said["outside"]:
                said["outside"] = True

        n = 0
        for idx, block in enumerate(dm.blocks):
            i2v = hasattr(block.cross_attn, "k_img")
            patch = NagCrossAttention(context, float(nag_scale),
                                      float(nag_alpha), float(nag_tau),
                                      i2v, window, report)
            m.add_object_patch(
                "diffusion_model.blocks.%d.cross_attn.forward" % idx,
                _bind(patch, block.cross_attn))
            n += 1

        print("[PLS] NAG: patched %d %s blocks (scale %.3f, alpha %.3f, "
              "tau %.3f%s)" % (n, kind, nag_scale, nag_alpha, nag_tau,
                               "" if window is None
                               else ", window %.0f%%-%.0f%%" % (lo * 100,
                                                                hi * 100)))
        return (m,)


def _bind(patch, module):
    """Bind the replacement forward to one module.

    add_object_patch replaces `...cross_attn.forward`, which is looked up as
    an ALREADY-BOUND method -- so the replacement is called WITHOUT self and
    has to close over its module.
    """
    def forward(x, context, transformer_options={}, **kwargs):
        return patch(module, x, context, transformer_options, **kwargs)
    return forward
