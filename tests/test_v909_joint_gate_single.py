#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v909 -- the joint-latent refusal follows the WORK, not the dial.

THE WOUND (found while answering a question about Frank's H3 graph, not in the
field): the v870 refusal for a joint audio/video latent read

    if (sigma_shift and sigma_shift > 0) or (_low_shift and _low_shift > 0):

and _resolve_low_shift turns a DIALLED sigma_shift_low into a positive number
regardless of mode. In Single there is no LOW expert -- _apply_sigma_shift was
never called for it, guarded by `model_low is not None` at the patch site. So a
Single H3 run could be REFUSED over a value that could not have patched
anything. The dial is greyed in Single (DUAL_ONLY in uls_sampler.js) but its
serialised value travels: dial 5.0 in a Wan graph, point the same node at a
single model, and the run dies over a widget the user cannot even see.

Same shape as v894 on the twin node -- there Single did WORK for stage L, here
Single took a GATE for stage L. The cut makes both the refusal and the patches
read ONE decision (_shift_high / _shift_low) so they can never drift apart.

WHAT IS PINNED, and how:

  A  the two decisions exist and the refusal reads them -- checked on the AST
     of the lifted block, not as a substring (v908 lesson: a substring is not
     a proof; the words appear in the comment above the code as well).
  B  BEHAVIOUR, driven: eight cases over {Single, High+Low} x {joint, plain} x
     {dial off, dial set}. The block is lifted CLOSED with tests/_lift.py, so a
     future helper in the seam reports as a short lift instead of a NameError.
  C  the LOW patch still needs a model on the socket -- Single with a cable at
     model_low must build nothing (the v894 promise, restated for this node).
  D  the refusal still names ModelSamplingMiniMaxH3 and still stands BEFORE the
     patch (the v870 promises, so this cut cannot quietly undo them).

No torch, no comfy: the block is executed against injected values.
"""
import ast
import os
import pathlib
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = (ROOT / "nodes" / "uls_sampler.py").read_text(encoding="utf-8")
FAILED = []


def _fail(msg):
    FAILED.append(msg)
    print("FAIL: {}".format(msg))


def _ok(msg):
    print("ok  : {}".format(msg))


def _need(cond, msg):
    _ok(msg) if cond else _fail(msg)


# --- lift the apply block, closed -------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lift as _LIFT

start = SRC.index("_low_shift = _resolve_low_shift(")
end = SRC.index("_apply_sigma_shift(model_low, _low_shift)", start)
end = SRC.index("\n", end)
BLOCK = textwrap.dedent(SRC[SRC.rindex("\n", 0, start) + 1:end])

# What the harness supplies itself; everything else the seam needs is lifted
# out of the module, arithmetic included (a stubbed resolver would test the
# stub). _apply_sigma_shift is the ONE stub -- it is the spy.
PROVIDED = set(["_apply_sigma_shift", "sigma_shift", "sigma_shift_low",
                "dual_moe", "model", "model_low", "sigmas", "sigmas_high",
                "sigmas_low", "latent_image"])
DEPS_SRC, MISSING = _LIFT.close_over(SRC, ["_is_ragged_latent",
                                           "_resolve_low_shift"], PROVIDED)
_need(not MISSING,
      "the lift is closed" if not MISSING else
      "the lift is short of %s -- a GUARD fault, not a tree fault"
      % ", ".join(MISSING))


# --- A: the refusal reads the decisions, checked on the AST ------------------
def _refusal_test():
    """The `if` whose body raises ValueError, and the `if` that wraps it."""
    tree = ast.parse(BLOCK)
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Raise):
                return node, tree
    return None, tree


_gate, _tree = _refusal_test()
_outer = [n for n in _tree.body if isinstance(n, ast.If)]
if not _outer:
    _fail("A: the seam has no gate at all")
else:
    names = sorted(n.id for n in ast.walk(_outer[0].test)
                   if isinstance(n, ast.Name))
    _need(names == ["_shift_high", "_shift_low"],
          "A: the gate reads exactly the two decisions -- got %r" % names)
    # every _apply_sigma_shift call inside must be guarded by one of them
    calls = [n for n in ast.walk(_outer[0])
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == "_apply_sigma_shift"]
    _need(len(calls) == 2,
          "A: exactly two patch sites expected inside the gate -- got %d"
          % len(calls))
    guards = set()
    for node in ast.walk(_outer[0]):
        if isinstance(node, ast.If):
            for n in ast.walk(node.test):
                if isinstance(n, ast.Name) and n.id.startswith("_shift_"):
                    guards.add(n.id)
    _need(guards == set(["_shift_high", "_shift_low"]),
          "A: both patch sites must be guarded by their own decision -- got %r"
          % sorted(guards))

# --- D: the v870 promises still stand ---------------------------------------
i_gate = SRC.find('_is_ragged_latent((latent_image or {}).get("samples"))')
i_patch = SRC.find("model = _apply_sigma_shift(model, sigma_shift)")
_need(i_gate > 0 and i_patch > 0 and i_gate < i_patch,
      "D: the joint refusal still precedes the patch")
_need("ModelSamplingMiniMaxH3" in SRC,
      "D: the refusal still names the node to use instead")


# --- B/C: drive it ----------------------------------------------------------
class _Nested(object):
    """Mirrors the ONE attribute _is_ragged_latent reads (Core's NestedTensor
    answers is_nested=True). No torch needed."""
    is_nested = True


def _drive(dual, low, shift=0.0, joint=False, model_low="LO"):
    """Return (patches, refused)."""
    calls = []

    def spy(mdl, s):
        calls.append((mdl, float(s)))
        return mdl

    ns = {"_apply_sigma_shift": spy,
          "sigma_shift": shift, "sigma_shift_low": low, "dual_moe": dual,
          "model": "HI", "model_low": model_low,
          "sigmas": None, "sigmas_high": None, "sigmas_low": None,
          "latent_image": {"samples": _Nested() if joint else object()}}
    exec(DEPS_SRC, ns)
    try:
        exec(compile(BLOCK, "<applyblock>", "exec"), ns)
    except ValueError as exc:
        return calls, str(exc)
    return calls, None


# the wound itself
_p, _r = _drive(dual=False, low=5.0, shift=0.0, joint=True)
_need(_r is None and _p == [],
      "B1: Single + joint + a stale LOW dial must RUN and patch nothing -- "
      "got patches=%r refused=%r" % (_p, _r))

# the refusals that must survive
_p, _r = _drive(dual=False, low=-1.0, shift=8.0, joint=True)
_need(_r is not None,
      "B2: Single + joint + a real HIGH shift must still be refused")
_p, _r = _drive(dual=True, low=5.0, shift=0.0, joint=True)
_need(_r is not None,
      "B3: High+Low + joint + an own LOW shift must still be refused")
_p, _r = _drive(dual=True, low=-1.0, shift=8.0, joint=True)
_need(_r is not None,
      "B4: High+Low + joint + a HIGH shift must still be refused")

# nothing to patch -> nothing to refuse, joint or not
_p, _r = _drive(dual=True, low=-1.0, shift=0.0, joint=True)
_need(_r is None and _p == [],
      "B5: all dials off must never refuse a joint latent -- got %r/%r"
      % (_p, _r))

# the ordinary (non-joint) table is unchanged from v839
_p, _r = _drive(dual=True, low=-1.0, shift=8.0)
_need(_p == [("HI", 8.0), ("LO", 8.0)] and _r is None,
      "B6: dual defaults still shift both experts alike -- got %r" % (_p,))
_p, _r = _drive(dual=True, low=5.0, shift=0.0)
_need(_p == [("LO", 5.0)] and _r is None,
      "B7: dual, high OFF + own low still patches only LOW -- got %r" % (_p,))
_p, _r = _drive(dual=False, low=-1.0, shift=8.0)
_need(_p == [("HI", 8.0)] and _r is None,
      "B8: Single still gets its HIGH shift -- got %r" % (_p,))

# C: a cable at model_low does not make a LOW stage exist in Single
_p, _r = _drive(dual=False, low=5.0, shift=0.0, model_low="LO")
_need(_p == [] and _r is None,
      "C1: Single with a cable at model_low must build nothing for stage L "
      "(v894) -- got %r" % (_p,))
_p, _r = _drive(dual=True, low=5.0, shift=0.0, model_low=None)
_need(_p == [] and _r is None,
      "C2: High+Low without a model on the socket patches nothing -- got %r"
      % (_p,))

print("\n{}: {} failure(s)".format(pathlib.Path(__file__).name, len(FAILED)))
sys.exit(1 if FAILED else 0)
