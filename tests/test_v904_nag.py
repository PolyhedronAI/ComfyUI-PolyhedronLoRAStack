#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v904 -- Polyhedron NAG: the maths, the batch, and the silences.

WHAT THIS NODE IS. Normalized attention guidance (arXiv 2505.21179) brings the
negative prompt back at CFG 1, where the sampler's negative does nothing at
all because there is no second model pass. It guides inside the
cross-attention instead of at the model output, so it doubles only the pass
over the few hundred text tokens rather than the whole model.

WHY OURS EXISTS NEXT TO KJNodes' WanVideoNAG. The maths there is the paper's
and is correct; this node reproduces it rather than improving it. What
changes is everything around it, where KJ's version is silent in ways that
cost real runs -- and the silences are what this guard is mostly about:

  * `input_type` on "default" reads context.shape[0], assumes anything above
    1 is a CFG pair, halves the batch and guides only the first half. Under
    CFG 1 with a real batch that silently leaves half the images unguided
    (KJNodes issue #354). We read ComfyUI's own cond_or_uncond instead.
  * nag_scale = 1 and nag_alpha = 0 are algebraic no-ops that still pay the
    full doubled cross-attention. KJ's node only exits at scale 0.
  * A non-Wan model raises AttributeError from inside the patch loop.

WHAT IS PINNED

  X1  the three paper steps, against numbers computed by hand here rather
      than by re-running the implementation.
  X2  nag_scale 1 really is the identity -- which is why the warning about it
      is true, and which no test of the warning STRING would establish.
  X3  the tau ceiling actually clips, and leaves short vectors alone.
  X4  batch splitting reads cond_or_uncond when present, in both orders and
      lengths, and falls back only when it is absent -- SAYING it fell back.
  X5  a real batch under CFG 1 is NOT mistaken for a CFG pair. This is the
      whole reason the function exists and is the case KJ's node gets wrong.
  X6  the architecture check refuses cleanly, and names H3's actual reason
      (one packed self-attention sequence, no cross-attention) rather than a
      missing attribute.
  X7  the no-op warnings are printed, driven through patch(), not grepped.
  X8  scale 0 returns the model untouched at no cost.

Script-style: exit 0 = pass.
"""

import contextlib
import io as _io
import os
import sys
import types

NAME = "v904"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "nodes"))

_fails = []
_checks = 0


def _need(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        _fails.append(msg)
        print("[%s] FAIL -- %s" % (NAME, msg))


def _install_stubs():
    """comfy.ldm.modules.attention and comfy.model_management, minimally."""
    attn = types.ModuleType("comfy.ldm.modules.attention")
    attn.optimized_attention = lambda q, k, v, heads=1, **kw: q
    comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    modules = types.ModuleType("comfy.ldm.modules")
    modules.attention = attn
    ldm.modules = modules
    comfy.ldm = ldm
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: "cpu"
    mm.unet_dtype = lambda: None
    comfy.model_management = mm
    for k, v in (("comfy", comfy), ("comfy.ldm", ldm),
                 ("comfy.ldm.modules", modules),
                 ("comfy.ldm.modules.attention", attn),
                 ("comfy.model_management", mm)):
        sys.modules[k] = v


def main():
    _install_stubs()
    try:
        import torch
    except ImportError:
        print("[%s] SKIP -- torch is not installed in this sandbox" % NAME)
        return 0

    import importlib
    N = importlib.import_module("ph_nag")
    importlib.reload(N)

    # ---------------- X1 / X2 / X3: the maths ---------------------------
    # Hand-computed. x+ = [3, 0], x- = [1, 0], scale 2:
    #   guidance = 3*2 - 1*1 = [5, 0]
    #   |x+|_1 = 3, |g|_1 = 5, ratio 5/3 = 1.667
    # With tau = 10 nothing clips, so the blend at alpha = 1 is [5, 0].
    xp = torch.tensor([[3.0, 0.0]])
    xn = torch.tensor([[1.0, 0.0]])
    got = N.normalized_attention_guidance(xp, xn, 2.0, 1.0, 10.0)
    _need(torch.allclose(got, torch.tensor([[5.0, 0.0]]), atol=1e-5),
          "X1: extrapolation must be Z+*s - Z-*(s-1); expected [5, 0], got %r"
          % (got.tolist(),))

    # alpha 0.25 blends: 0.25*5 + 0.75*3 = 3.5
    got = N.normalized_attention_guidance(xp, xn, 2.0, 0.25, 10.0)
    _need(torch.allclose(got, torch.tensor([[3.5, 0.0]]), atol=1e-5),
          "X1: the blend must be alpha*Z~ + (1-alpha)*Z+; expected [3.5, 0], "
          "got %r" % (got.tolist(),))

    # X2: scale 1 is the identity, whatever alpha and tau say. This is the
    # FACT behind the warning; a test of the warning text would not show it.
    for alpha in (0.0, 0.25, 1.0):
        got = N.normalized_attention_guidance(xp, xn, 1.0, alpha, 2.5)
        _need(torch.allclose(got, xp, atol=1e-5),
              "X2: nag_scale 1 must return the positive attention unchanged "
              "(alpha %.2f) -- that is what makes it a no-op worth warning "
              "about; got %r" % (alpha, got.tolist()))

    # X2b: alpha 0 is the identity too, whatever the scale.
    for scale in (2.0, 11.0, 35.0):
        got = N.normalized_attention_guidance(xp, xn, scale, 0.0, 2.5)
        _need(torch.allclose(got, xp, atol=1e-5),
              "X2: nag_alpha 0 must return the positive attention unchanged "
              "(scale %.1f); got %r" % (scale, got.tolist()))

    # X3: the tau ceiling. Same vectors, tau 1.2 -> |g| pulled to 1.2*3 = 3.6.
    got = N.normalized_attention_guidance(xp, xn, 2.0, 1.0, 1.2)
    _need(torch.allclose(got, torch.tensor([[3.6, 0.0]]), atol=1e-4),
          "X3: above tau the guided vector must be pulled to exactly "
          "tau * |Z+|; expected [3.6, 0], got %r" % (got.tolist(),))
    # ...and below tau it must not touch anything: ratio 1.667 < 2.5
    got_free = N.normalized_attention_guidance(xp, xn, 2.0, 1.0, 2.5)
    _need(torch.allclose(got_free, torch.tensor([[5.0, 0.0]]), atol=1e-5),
          "X3: below the ceiling the vector must pass through unclipped, "
          "got %r" % (got_free.tolist(),))

    # ---------------- X4 / X5: batch splitting --------------------------
    x4 = torch.zeros(4, 2, 2)
    ctx = torch.zeros(4, 2, 2)

    n_pos, n_neg, how = N.split_batch(x4, ctx, {"cond_or_uncond": [0, 0, 1, 1]})
    _need((n_pos, n_neg) == (2, 2) and how == "cond_or_uncond",
          "X4: a genuine CFG pair must be read from cond_or_uncond, got %r"
          % ((n_pos, n_neg, how),))

    n_pos, n_neg, how = N.split_batch(x4, ctx, {"cond_or_uncond": [0, 0, 0, 0]})
    _need((n_pos, n_neg) == (4, 0) and how == "cond_or_uncond",
          "X5: a REAL BATCH under CFG 1 must be guided in FULL. This is the "
          "case KJ's node halves and silently leaves half unguided -- the "
          "whole reason this function exists. Got %r" % ((n_pos, n_neg, how),))

    n_pos, n_neg, how = N.split_batch(x4, ctx, {"cond_or_uncond": [0, 1, 0, 1]})
    _need(n_pos + n_neg == 4 and n_pos == 2,
          "X4: mixed ordering must still count the conds, got %r"
          % ((n_pos, n_neg),))

    x1 = torch.zeros(1, 2, 2)
    n_pos, n_neg, how = N.split_batch(x1, x1, {"cond_or_uncond": [0]})
    _need((n_pos, n_neg) == (1, 0),
          "X4: a single conditional row is all positive, got %r"
          % ((n_pos, n_neg),))

    n_pos, n_neg, how = N.split_batch(x4, ctx, {})
    _need((n_pos, n_neg) == (2, 2) and how.startswith("shape guess"),
          "X4: with no cond_or_uncond the fallback may run, but it must "
          "IDENTIFY itself as a guess, got %r" % ((n_pos, n_neg, how),))

    n_pos, n_neg, how = N.split_batch(x4, ctx, {"cond_or_uncond": [0, 1]})
    _need(how.startswith("shape guess"),
          "X4: a cond_or_uncond that does not match the batch length must NOT "
          "be trusted, got %r" % (how,))

    # ---------------- X6: the architecture check ------------------------
    class _NoDM(object):
        def get_model_object(self, name):
            raise KeyError(name)

    _dm, _kind, why = N.describe_model(_NoDM())
    _need(why is not None and "diffusion_model" in why,
          "X6: a model with no diffusion_model must be refused by name, "
          "got %r" % (why,))

    class _Packed(object):
        """Stands in for MiniMax H3: blocks, but no cross-attention."""

        class _Block(object):
            pass

        def __init__(self):
            self.blocks = [self._Block()]
            self.text_embedding = lambda t: t

        def get_model_object(self, name):
            return self

    _dm, _kind, why = N.describe_model(_Packed())
    _need(why is not None and "cross-attention" in why,
          "X6: a model whose blocks have no cross-attention must be refused, "
          "got %r" % (why,))
    _need(why is not None and "MiniMax H3" in why,
          "X6: ...and the refusal must name the case Frank actually runs; "
          "got %r" % (why,))
    # The NAME alone is not the reason. What a reader needs is WHY it cannot
    # work: NAG's saving comes from doubling a small attention over the text
    # tokens, and on a packed model there is no small attention to double --
    # the same trick costs what CFG 2 costs. A refusal that omits this reads
    # as an arbitrary restriction. (My first mutation only trimmed the first
    # line of the message and the name survived, so this check earns its
    # keep only by asking for the argument.)
    _need(why is not None and "self-attention" in why and "CFG 2" in why,
          "X6: the refusal must give the ECONOMIC reason -- doubling the full "
          "self-attention costs what CFG 2 costs -- not merely state that "
          "cross-attention is missing; got %r" % (why,))

    class _Wan(object):
        class _Attn(object):
            pass

        class _Block(object):
            def __init__(self):
                self.cross_attn = _Wan._Attn()

        def __init__(self, n=2):
            self.blocks = [self._Block() for _ in range(n)]
            self.text_embedding = _Emb()

        def get_model_object(self, name):
            return self

    class _Emb(object):
        def to(self, device):
            return self

        def __call__(self, t):
            return t

    _dm, kind, why = N.describe_model(_Wan())
    _need(why is None,
          "X6: a Wan-shaped model must be ACCEPTED, got refusal %r" % (why,))

    # ---------------- X7 / X8: the node, driven -------------------------
    class _Model(_Wan):
        def __init__(self):
            _Wan.__init__(self)
            self.patched = {}

        def clone(self):
            return self

        def add_object_patch(self, path, fn):
            self.patched[path] = fn

    cond = [(torch.zeros(1, 4, 8), {})]

    def _run(**kw):
        args = dict(model=_Model(), conditioning=cond, nag_scale=11.0,
                    nag_alpha=0.25, nag_tau=2.5)
        args.update(kw)
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = N.ULSNag().patch(**args)[0]
        return out, buf.getvalue()

    m0 = _Model()
    out, said = _run(model=m0, nag_scale=0.0)
    _need(out is m0 and not m0.patched,
          "X8: nag_scale 0 must return the model UNTOUCHED -- no patch, no "
          "cost")
    _need("nothing patched" in said,
          "X8: ...and say so, said:\n%s" % said)

    out, said = _run(nag_scale=1.0)
    _need("cancels algebraically" in said or "no-op" in said.lower(),
          "X7: nag_scale 1 must be reported as the no-op it is (X2 proves it "
          "is one); said:\n%s" % said)
    _need("DOUBLED" in said or "doubled" in said,
          "X7: ...and the report must say it still COSTS, or the user has no "
          "reason to change it; said:\n%s" % said)

    out, said = _run(nag_alpha=0.0)
    _need("nag_alpha is 0" in said,
          "X7: nag_alpha 0 must be reported too; said:\n%s" % said)

    out, said = _run()
    _need("patched 2" in said,
          "X7: a normal run must report how many blocks it patched; said:\n%s"
          % said)
    _need(len(out.patched) == 2,
          "X7: ...and really patch them, got %r" % (list(out.patched),))

    masked = [(torch.zeros(1, 4, 8), {"mask": object(), "strength": 0.5})]
    out, said = _run(conditioning=masked)
    _need("IGNORED" in said and "mask" in said,
          "X7: a conditioning carrying a mask or strength must be told those "
          "are dropped -- KJ's node reads only the bare tensor and says "
          "nothing; said:\n%s" % said)

    packed = _Packed()
    packed.clone = lambda: packed
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = N.ULSNag().patch(model=packed, conditioning=cond,
                               nag_scale=11.0, nag_alpha=0.25,
                               nag_tau=2.5)[0]
    said = buf.getvalue()
    _need(out is packed and "not patching" in said
          and "cross-attention" in said,
          "X6: an unsupported model must come back untouched WITH a reason, "
          "not an AttributeError from inside the loop; said:\n%s" % said)

    out, said = _run(start_percent=0.8, end_percent=0.2)
    _need("Swapping" in said,
          "X7: an inverted window would never guide -- say so and swap, "
          "rather than silently doing nothing; said:\n%s" % said)

    if _fails:
        print("\n[%s] %d/%d FAILED" % (NAME, len(_fails), _checks))
        return 1
    print("[%s] OK -- the paper's maths, an honest batch split, and no silent "
          "no-ops (%d checks)" % (NAME, _checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
