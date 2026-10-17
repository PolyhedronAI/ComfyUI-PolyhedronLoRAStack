#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v902 -- Pick Frame: named modes, and ONE rule behind both classes.

THE FINDING THAT PROMPTED THIS. Core's `ImageFromBatch` reads batch_index -1
as the LAST image. ULSImagePickFrame has always read frame_index -1 as the
MIDDLE one -- a deliberate choice for inflated Wan T2I runs. Both readings are
defensible; sharing a number is not. Swap the two nodes in a graph and you get
a different picture with no error and no warning. Frank's own H3 workflow used
Core's node with -1 while ours sat in the same pack meaning something else.

So the number stopped deciding and a named mode does.

THE OTHER HALF. ULSImagePickFrame ships TWICE -- the legacy class in
wan_frame_inflate.py and the V3 schema class in uls_pick_frame_v3.py. They
share a node id and are mutually exclusive at registration, so only one runs
on any given machine. That is exactly the v898 failure shape: two copies of a
rule, one quietly stale, the symptom invisible wherever the other copy loads.
Both now call uls_pick_frame_core.select().

WHAT IS PINNED

  R1  the legacy default is byte-for-byte unchanged: mode "middle (legacy)"
      with -1 gives the middle frame, other values clamp as before. A saved
      workflow that predates this build must not move.
  R2  every mode does what its name says, including at the edges (n == 1,
      index past the end, negative index).
  R3  'index' and 'range' follow CORE's negative convention, so the two nodes
      finally agree once a user picks that mode.
  R4  a selection is never empty for a non-empty batch.
  R5  BOTH classes route through the shared core -- asserted by patching the
      core and watching both answers change. A copy of the rule would survive
      every other check here.
  R6  both schemas expose the same widget names in the same order, appended
      after frame_index (#577: widgets_values is positional).
  R7  an unrecognised mode falls back to legacy rather than raising, so a
      graph saved by a newer build cannot crash an older one.

Script-style: exit 0 = pass.
"""

import ast
import os
import sys
import types

NAME = "v902"
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


class _Batch(object):
    """A stand-in image batch that remembers WHICH frames it holds."""

    def __init__(self, n, ids=None):
        self.shape = (n,)
        self.ids = list(range(n)) if ids is None else list(ids)

    def __getitem__(self, k):
        if isinstance(k, slice):
            kept = self.ids[k]
            return _Batch(len(kept), kept)
        raise TypeError(k)


def _install_comfy_api():
    """Minimal stand-in for comfy_api.latest.io.

    The V3 class only needs it to IMPORT here -- the schema itself is read
    statically by AST in R6, and execute() is driven with io.NodeOutput
    swapped out. Stubbing the door rather than the room keeps this guard
    honest about what it actually tests."""
    class _In(object):
        def __init__(self, *a, **k):
            self.args, self.kwargs = a, k

    class _Field(object):
        Input = staticmethod(lambda *a, **k: _In(*a, **k))
        Output = staticmethod(lambda *a, **k: _In(*a, **k))

    class _ComfyNode(object):
        pass

    io_ns = types.SimpleNamespace(
        Image=_Field, Int=_Field, Combo=_Field, String=_Field,
        Boolean=_Field, Float=_Field, Mask=_Field, Latent=_Field,
        Schema=lambda **k: types.SimpleNamespace(**k),
        ComfyNode=_ComfyNode,
        NodeOutput=lambda *a, **k: (a[0] if a else None),
    )
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io_ns
    root = types.ModuleType("comfy_api")
    root.latest = latest
    sys.modules.setdefault("comfy_api", root)
    sys.modules.setdefault("comfy_api.latest", latest)


def _install_torch():
    torch = types.ModuleType("torch")

    def cat(parts, dim=0):
        ids = []
        for p in parts:
            ids.extend(p.ids)
        return _Batch(len(ids), ids)

    torch.cat = cat
    sys.modules["torch"] = torch


def _widget_names(path, cls_name):
    """Widget names in declaration order, statically, for either schema form."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for c in ast.walk(tree):
        if not isinstance(c, ast.ClassDef) or c.name != cls_name:
            continue
        names = []
        for n in ast.walk(c):
            # legacy: "name": (...) inside required
            if isinstance(n, ast.Dict):
                for k, v in zip(n.keys, n.values):
                    if (isinstance(k, ast.Constant)
                            and isinstance(k.value, str)
                            and k.value == "required"
                            and isinstance(v, ast.Dict)):
                        for kk in v.keys:
                            if isinstance(kk, ast.Constant):
                                names.append(str(kk.value))
            # V3: io.Xxx.Input("name", ...)
            if isinstance(n, ast.Call):
                f = n.func
                if (isinstance(f, ast.Attribute) and f.attr == "Input"
                        and n.args and isinstance(n.args[0], ast.Constant)):
                    names.append(str(n.args[0].value))
        return names
    return []


def main():
    _install_comfy_api()
    _install_torch()
    import uls_pick_frame_core as pf

    # ---------------- R1: the legacy default is untouched ---------------
    L = pf.MODES[0]
    _need(L == "middle (legacy)",
          "R1: the legacy behaviour must remain the DEFAULT mode, or every "
          "saved workflow changes meaning on upgrade; got %r" % (L,))
    for n, want in ((22, 11), (5, 2), (1, 0), (2, 1), (7, 3)):
        got, _ = pf.plan(n, L, -1, 1)
        _need(got == [want],
              "R1: legacy -1 on %d frames is the middle (%d), got %r"
              % (n, want, got))
    _need(pf.plan(22, L, 3, 1)[0] == [3],
          "R1: legacy with a plain index must still be that index")
    _need(pf.plan(22, L, 900, 1)[0] == [21],
          "R1: legacy must still clamp past the end")

    # ---------------- R2 / R3: the named modes --------------------------
    _need(pf.plan(22, "last", 0, 1)[0] == [21],
          "R2: 'last' is the last frame")
    _need(pf.plan(22, "first", 5, 1)[0] == [0],
          "R2: 'first' is frame 0 and ignores frame_index")
    _need(pf.plan(1, "last", 0, 1)[0] == [0]
          and pf.plan(1, "first", 0, 1)[0] == [0],
          "R2: a single-image batch answers 0 for both ends")

    _need(pf.plan(22, "index", -1, 1)[0] == [21],
          "R3: 'index' -1 must be the LAST frame -- Core's convention, which "
          "is the whole point of separating the modes")
    _need(pf.plan(22, "index", -2, 1)[0] == [20],
          "R3: 'index' -2 counts back from the end")
    _need(pf.plan(22, "index", 0, 1)[0] == [0],
          "R2: 'index' 0 is the first frame")
    _need(pf.plan(22, "index", 900, 1)[0] == [21],
          "R2: 'index' clamps past the end instead of failing")

    _need(pf.plan(22, "range", 4, 3)[0] == [4, 5, 6],
          "R2: 'range' takes count frames from frame_index, got %r"
          % (pf.plan(22, "range", 4, 3)[0],))
    _need(pf.plan(22, "range", 20, 9)[0] == [20, 21],
          "R2: 'range' must stop at the end of the batch, not run off it")
    _need(pf.plan(22, "range", -2, 5)[0] == [20, 21],
          "R3: 'range' honours a negative start the same way 'index' does")

    _need(pf.plan(22, "every Nth", 0, 5)[0] == [0, 5, 10, 15, 20],
          "R2: 'every Nth' walks the whole batch in steps, got %r"
          % (pf.plan(22, "every Nth", 0, 5)[0],))
    _need(pf.plan(22, "every Nth", 0, 1)[0] == list(range(22)),
          "R2: step 1 keeps everything")
    _need(pf.plan(3, "every Nth", 0, 99)[0] == [0],
          "R4: a step larger than the batch still returns one frame")

    # ---------------- R4: never empty -----------------------------------
    # count is swept too, INCLUDING 0 and negatives. The widget clamps at 1,
    # but an API call or an older saved graph need not, and both 'range' and
    # 'every Nth' divide by it -- count 0 would give an empty range or a
    # zero-step range() that raises. My first mutation round missed this
    # because every case here passed count=1.
    for mode in pf.MODES:
        for n in (1, 2, 5, 22):
            for idx in (-99, -2, -1, 0, 1, 3, 999):
                for cnt in (0, -5, 1, 3):
                    got, _ = pf.plan(n, mode, idx, cnt)
                    _need(len(got) >= 1 and all(0 <= i < n for i in got),
                          "R4: %s(n=%d, idx=%d, count=%d) returned %r -- a "
                          "selection must never be empty or out of range"
                          % (mode, n, idx, cnt, got))
    _need(pf.plan(0, "last", 0, 1)[0] == [],
          "R4: an EMPTY batch is the one case that yields nothing")

    # ---------------- R7: unknown mode falls back -----------------------
    got, _ = pf.plan(22, "some future mode", -1, 1)
    _need(got == [11],
          "R7: an unrecognised mode must fall back to legacy, so a graph "
          "saved by a newer build does not crash an older one; got %r"
          % (got,))

    # ---------------- R5: BOTH classes use the shared core --------------
    import importlib
    legacy_mod = importlib.import_module("wan_frame_inflate")
    importlib.reload(legacy_mod)
    v3_mod = importlib.import_module("uls_pick_frame_v3")
    importlib.reload(v3_mod)

    batch = _Batch(22)
    legacy = legacy_mod.ULSImagePickFrame()
    out_l = legacy.pick(batch, -1, "last", 1)[0]
    _need(out_l.ids == [21],
          "R5: the legacy class must honour the mode, got %r" % (out_l.ids,))

    class _Out(object):
        def __init__(self, v):
            self.v = v

    fake_io = types.SimpleNamespace(NodeOutput=_Out)
    real_io = v3_mod.io
    v3_mod.io = fake_io
    try:
        out_v = v3_mod.ULSImagePickFrameV3.execute(batch, -1, "last", 1).v
    finally:
        v3_mod.io = real_io
    _need(out_v.ids == [21],
          "R5: the V3 class must honour the mode, got %r" % (out_v.ids,))

    # Now prove it is the SAME code: patch the core, expect BOTH to change.
    real_plan = pf.plan
    pf.plan = lambda n, mode, idx, cnt: ([0], "patched")
    try:
        a = legacy.pick(batch, -1, "last", 1)[0]
        v3_mod.io = fake_io
        try:
            b = v3_mod.ULSImagePickFrameV3.execute(batch, -1, "last", 1).v
        finally:
            v3_mod.io = real_io
    finally:
        pf.plan = real_plan
    _need(a.ids == [0] and b.ids == [0],
          "R5: patching uls_pick_frame_core must change BOTH classes. One of "
          "them still carries its own copy of the rule -- that is the v898 "
          "shape; legacy gave %r, V3 gave %r" % (a.ids, b.ids))

    # ---------------- R6: same widgets, same order, appended ------------
    leg = _widget_names(os.path.join(ROOT, "nodes", "wan_frame_inflate.py"),
                        "ULSImagePickFrame")
    v3 = _widget_names(os.path.join(ROOT, "nodes", "uls_pick_frame_v3.py"),
                       "ULSImagePickFrameV3")
    _need(leg == v3,
          "R6: both forms must expose the SAME inputs in the SAME order, or "
          "a saved graph resolves differently depending on which one "
          "registers; legacy %r vs V3 %r" % (leg, v3))
    _need(leg[:2] == ["images", "frame_index"],
          "R6: the original two inputs must keep their slots (#577), got %r"
          % (leg,))
    _need(leg[2:] == ["mode", "count"],
          "R6: the new widgets must be APPENDED after frame_index, got %r"
          % (leg,))

    if _fails:
        print("\n[%s] %d/%d FAILED" % (NAME, len(_fails), _checks))
        return 1
    print("[%s] OK -- named modes, legacy default intact, one rule behind "
          "both classes (%d checks)" % (NAME, _checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
