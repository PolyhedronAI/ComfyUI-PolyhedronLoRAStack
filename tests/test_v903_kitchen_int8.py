#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v903 -- comfy kitchen int8: the mode, the containers, the collision.

WHAT THIS IS. Core's ModelAttentionBackend node offers "comfy kitchen
attention", registered internally as "comfy_kitchen_int8". Read at the source:
the implementation lives in comfy_kitchen/sage_attention.py, carries an NVIDIA
copyright header, and calls itself pure INT8 scaled dot-product attention for
tensor-core GPUs. It is sage-shaped quantisation in NVIDIA's own version,
bundled with ComfyUI rather than installed separately -- NOT a MiniMax H3
feature, although H3's stock template is where most people meet it. Core
registers it only when the compiled kernel supports the card.

THE COLLISION, and a correction to my own first reading. I reported that Core
patches through set_model_optimized_attention while we set
transformer_options["optimized_attention_override"], and concluded that
Core's wrap_attn consults ours first so ours wins. That was wrong.
ModelPatcher.set_model_optimized_attention wraps the backend and stores it
under THE SAME KEY. There is no precedence: whichever node sits later in the
chain overwrites the earlier one, silently. Frank's H3 template ships with
ModelAttentionBackend in it, so this is a live trap, not a hypothetical.

WHAT IS PINNED

  V1  the mode exists, is offered ONLY when Core registered the backend, and
      is never fabricated from an import guess.
  V2  the router calls Core's registered function, and raises a named error
      rather than falling silently to something else when it is absent.
  V3  CONTAINER PASS-THROUGH. H3 hands attention AttentionTensorContainer
      objects. Core's wrap_attn only takes the container path if the OVERRIDE
      carries a container_function attribute; without it the containers are
      unpacked and the backend's prequantize path -- which drops the
      floating-point q/k/v instead of holding them -- is lost. Pinned by
      identity: the attribute must be present AND must reach Core's own
      container function.
  V4  a backend WITHOUT a container function must not grow a fake one.
  V5  the collision is REPORTED: a foreign override upstream gets a message
      naming Core's node and the way to keep it; a Polyhedron one gets its
      own message. Both are pinned on content, not on merely printing.
  V6  our override is marked, so the two cases can be told apart at all.
  V7  DEFAULT_MODE still REMOVES the key rather than writing one -- that is
      what makes "leave the model alone" true and lets Core's node win on
      purpose.

Script-style: exit 0 = pass.
"""

import io as _io
import os
import sys
import types
import contextlib

NAME = "v903"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "nodes"))

PY = os.path.join(ROOT, "nodes", "ph_attention.py")

_fails = []
_checks = 0


def _need(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        _fails.append(msg)
        print("[%s] FAIL -- %s" % (NAME, msg))


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class _FakeCore(object):
    """Stands in for comfy.ldm.modules.attention's registry."""

    def __init__(self, funcs):
        self._funcs = funcs

    def get_attention_function(self, name, default=None):
        return self._funcs.get(name, default)


def _install_core(funcs):
    mod = types.ModuleType("comfy.ldm.modules.attention")
    fake = _FakeCore(funcs)
    mod.get_attention_function = fake.get_attention_function
    comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
    ldm = types.ModuleType("comfy.ldm")
    modules = types.ModuleType("comfy.ldm.modules")
    modules.attention = mod
    ldm.modules = modules
    comfy.ldm = ldm
    for k, v in (("comfy", comfy), ("comfy.ldm", ldm),
                 ("comfy.ldm.modules", modules),
                 ("comfy.ldm.modules.attention", mod)):
        sys.modules[k] = v


def main():
    py = _read(PY)
    import importlib
    A = importlib.import_module("ph_attention")
    importlib.reload(A)

    # ---------------- V1: name and availability -------------------------
    _need(getattr(A, "CK_INT8_MODE", None) == "comfy kitchen int8",
          "V1: the mode needs a stable label, got %r"
          % (getattr(A, "CK_INT8_MODE", None),))
    _need(getattr(A, "CK_INT8_REGISTRY_NAME", None) == "comfy_kitchen_int8",
          "V1: the registry name must match what Core registers, got %r"
          % (getattr(A, "CK_INT8_REGISTRY_NAME", None),))

    _install_core({})                      # Core registered nothing
    importlib.reload(A)
    _need(A.CK_INT8_MODE not in A.available_modes(),
          "V1: with no backend registered the mode must NOT be offered -- "
          "availability is read from Core, never assumed")

    def _plain(*a, **k):
        return ("plain", a, k)

    _install_core({"comfy_kitchen_int8": _plain})
    importlib.reload(A)
    _need(A.CK_INT8_MODE in A.available_modes(),
          "V1: once Core registers the backend the mode must appear")

    # ---------------- V2: the router calls Core's function --------------
    # The factory is build_router(mode, sparse=None) -- looked up in the
    # source, not guessed. A guard that hunts for a name it is not sure of
    # would pass just as happily against the wrong function.
    _need(callable(getattr(A, "build_router", None)),
          "V2: build_router is gone -- this guard must be pointed at its "
          "replacement rather than silently testing nothing")
    router = A.build_router(A.CK_INT8_MODE)
    _need(callable(router),
          "V2: no router was built for the mode")
    if not callable(router):
        print("\n[%s] %d/%d FAILED" % (NAME, len(_fails), _checks))
        return 1

    got = router(lambda *a, **k: ("base", a), 1, 2, 3, heads=4)
    _need(got[0] == "plain",
          "V2: the router must call CORE's registered function, got %r"
          % (got[0],))

    _install_core({})
    importlib.reload(A)
    router2 = A.build_router(A.CK_INT8_MODE)
    if callable(router2):
        raised = False
        try:
            router2(lambda *a, **k: None, 1, 2, 3)
        except NotImplementedError as exc:
            raised = "comfy_kitchen_int8" in str(exc)
        _need(raised,
              "V2: with the backend gone the router must raise a NAMED "
              "NotImplementedError, not fall through to something else")

    # ---------------- V3 / V4: container pass-through -------------------
    marker = object()

    def _with_container(*a, **k):
        return "flat"

    def _container_fn(*a, **k):
        return marker

    _with_container.container_function = _container_fn

    _install_core({"comfy_kitchen_int8": _with_container})
    importlib.reload(A)
    r3 = A.build_router(A.CK_INT8_MODE)
    cf = getattr(r3, "container_function", None)
    _need(callable(cf),
          "V3: the override MUST carry a container_function, or Core's "
          "wrap_attn unpacks the containers and the backend's prequantize "
          "path is lost -- on H3, where the tensors are largest")
    if callable(cf):
        _need(cf(1, 2, 3) is marker,
              "V3: the attribute must reach CORE's container function, not "
              "some lookalike of our own")

    _install_core({"comfy_kitchen_int8": _plain})   # no container_function
    importlib.reload(A)
    r4 = A.build_router(A.CK_INT8_MODE)
    _need(getattr(r4, "container_function", None) is None,
          "V4: a backend without a container path must NOT be given a fake "
          "one -- that would send containers down a route Core never built")

    # ---------------- V5 / V6 / V7: the collision report, DRIVEN --------
    #
    # My first draft asserted on the SOURCE TEXT here and three mutations
    # walked straight past it: moving the check after the overwrite, deleting
    # the owner mark, and disabling the DEFAULT_MODE branch all left the
    # strings in the file untouched. That is exactly the v900 fault -- reading
    # the file instead of running it -- so this section runs patch().
    _install_core({"comfy_kitchen_int8": _plain})
    importlib.reload(A)

    class _Model(object):
        def __init__(self, opts=None):
            self.model_options = opts if opts is not None else {}

        def clone(self):
            tops = dict(self.model_options.get("transformer_options", {}))
            return _Model({"transformer_options": tops})

    def _run(model, attention):
        buf = _io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = A.ULSAttention().patch(
                model=model, attention=attention, fallback=True,
                attention_first=A.SAME_AS_MAIN, first_steps=0,
                attention_last=A.SAME_AS_MAIN, last_steps=0,
                sparse_time_window=3, sparse_sink=True, live_check=False,
                latent=None)[0]
        return out, buf.getvalue()

    # (a) nothing upstream: no collision message, and the key is written
    clean, said = _run(_Model(), A.CK_INT8_MODE)
    key = clean.model_options["transformer_options"].get(
        "optimized_attention_override")
    _need(callable(key),
          "V5: a normal patch must install the override")
    _need("REPLACES" not in said,
          "V5: with nothing upstream there is no collision to report, said:\n%s"
          % said)

    # (b) FOREIGN override upstream -- Core's node writes the same key
    def _foreign(_, *a, **k):
        return None

    pre = _Model({"transformer_options":
                  {"optimized_attention_override": _foreign}})
    out_b, said_b = _run(pre, A.CK_INT8_MODE)
    _need("REPLACES it" in said_b and "ModelAttentionBackend" in said_b,
          "V5: a foreign upstream backend must be reported, naming where it "
          "came from; said:\n%s" % said_b)
    _need("if you meant to keep it" in said_b,
          "V5: the report must name the way out, said:\n%s" % said_b)
    _need(out_b.model_options["transformer_options"]
          ["optimized_attention_override"] is not _foreign,
          "V5: ...and we really do replace it -- the message must describe "
          "what happened, not something else")

    # (c) our OWN override upstream -> the other message
    marked, _ = _run(_Model(), A.CK_INT8_MODE)
    pre2 = _Model({"transformer_options": {
        "optimized_attention_override":
            marked.model_options["transformer_options"]
            ["optimized_attention_override"]}})
    _out_c, said_c = _run(pre2, A.CK_INT8_MODE)
    _need("another Polyhedron Attention node" in said_c,
          "V5: a second Polyhedron node must get its OWN message, said:\n%s"
          % said_c)
    _need("ModelAttentionBackend" not in said_c,
          "V6: ...which means the owner mark must actually distinguish the "
          "two cases; said:\n%s" % said_c)

    # (d) V6: the mark is on the object, not merely in the source
    _need(getattr(key, "_pls_owner", None) == "ULSAttention",
          "V6: the installed override must carry the owner mark, got %r"
          % (getattr(key, "_pls_owner", None),))

    # (e) V7: DEFAULT_MODE REMOVES the key instead of writing one
    pre3 = _Model({"transformer_options":
                   {"optimized_attention_override": _foreign}})
    out_d, said_d = _run(pre3, A.DEFAULT_MODE)
    _need("optimized_attention_override" not in
          out_d.model_options["transformer_options"],
          "V7: '%s' must REMOVE the key, or 'leave the model alone' is a lie "
          "and Core's node can never win on purpose" % A.DEFAULT_MODE)
    # ...and it must take the DEFAULT branch to get there, not fall through
    # the unknown-mode one. Both clear the key, so the key alone cannot tell
    # them apart -- the message can, and a user who deliberately picked
    # "default" must not be told his mode is unknown. (A mutation disabling
    # the default branch survived until this line existed.)
    _need("unknown mode" not in said_d,
          "V7: choosing '%s' must not be reported as an UNKNOWN mode -- that "
          "is the fall-through path, not the deliberate one; said:\n%s"
          % (A.DEFAULT_MODE, said_d))

    if _fails:
        print("\n[%s] %d/%d FAILED" % (NAME, len(_fails), _checks))
        return 1
    print("[%s] OK -- kitchen int8 offered on Core's word, containers pass "
          "through, collisions are announced (%d checks)" % (NAME, _checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
