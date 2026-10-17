#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v901 -- an INPUT_TYPES section ComfyUI does not know is a FAILURE.

WHAT HAPPENED. In v900 a new widget was appended to ULSVAE's INPUT_TYPES one
brace too late, so it landed as a THIRD top-level section beside "required"
and "optional":

    "required": {
        ...
        "mute_staging_logs": (...),
    },
        "h3_still_fix": (["auto", "off"], {...}),      # <- here
    "optional": {...}

Python accepted it. py_compile (GATE-1) accepted it. The full suite stayed at
256/256. And ComfyUI refused to register the node, so the entire VAE box came
up red with UNKNOWN on every widget in Frank's graph -- a delivered build in
which one node simply did not exist.

WHY NOTHING CAUGHT IT. test_v577 exists precisely to catch widgets moving in
the serialisation order, and it scans INPUT_TYPES statically. But its scanner
opened with

    if sec not in ("required", "optional"):
        continue

so a widget outside every section read as "no widgets changed" -- the most
severe form of the fault presented as the absence of one. A skipped key is not
a safe default; here it was the failure mode.

WHAT IS PINNED (all against the real scanner in test_v577)

  Q1  a stray literal section is REPORTED, not skipped.
  Q2  "hidden" is NOT stray -- it is a section ComfyUI reads -- and it
      contributes no widgets_values slot.
  Q3  required/optional still yield their widgets in declaration order, so
      the sharpening did not cost the original promise.
  Q4  a computed (non-literal) section key still marks the node dynamic
      rather than being reported as a stray name.
  Q5  the live tree is clean: no node ships a stray section.

Script-style: exit 0 = pass.
"""

import ast
import os
import sys

NAME = "v901"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

_fails = []
_checks = 0


def _need(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        _fails.append(msg)
        print("[%s] FAIL -- %s" % (NAME, msg))


def _order_of_src(src):
    """Run test_v577's real scanner over one INPUT_TYPES source snippet."""
    import importlib
    v577 = importlib.import_module("test_v577_widget_order")
    tree = ast.parse(src)
    fn = None
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "INPUT_TYPES":
            fn = n
    assert fn is not None
    return v577._order_of(fn)


CLEAN = '''
def INPUT_TYPES(cls):
    return {
        "required": {
            "alpha": ("INT", {"default": 1}),
            "beta": ("BOOLEAN", {"default": True}),
        },
        "optional": {
            "gamma": ("INT", {"default": 2}),
            "wired": ("IMAGE",),
        },
    }
'''

STRAY = '''
def INPUT_TYPES(cls):
    return {
        "required": {
            "alpha": ("INT", {"default": 1}),
        },
            "beta": ("BOOLEAN", {"default": True}),
        "optional": {
            "gamma": ("INT", {"default": 2}),
        },
    }
'''

HIDDEN = '''
def INPUT_TYPES(cls):
    return {
        "required": {
            "alpha": ("INT", {"default": 1}),
        },
        "hidden": {
            # Deliberately shaped LIKE A WIDGET. With a bare "UNIQUE_ID"
            # string the entry is not widget-shaped anyway, so a mutation that
            # dropped the hidden skip stayed green -- the check proved nothing.
            "node_id": ("INT", {"default": 0}),
        },
        "optional": {
            "gamma": ("INT", {"default": 2}),
        },
    }
'''

COMPUTED = '''
def INPUT_TYPES(cls):
    return {
        "required": {
            "alpha": ("INT", {"default": 1}),
        },
        _section_name(): {
            "beta": ("INT", {"default": 2}),
        },
    }
'''


def main():
    # ---------------- Q1: the stray is reported -------------------------
    order, dynamic, stray = _order_of_src(STRAY)
    _need(stray == ["beta"],
          "Q1: a widget written outside every section must be REPORTED as a "
          "stray section, got %r -- this is exactly the v900 fault and "
          "skipping it presents a dead node as 'no change'" % (stray,))
    _need("beta" not in order,
          "Q1: the stray must NOT be counted as a normal widget either, "
          "got order %r" % (order,))

    # ---------------- Q2: hidden is legitimate --------------------------
    order, dynamic, stray = _order_of_src(HIDDEN)
    _need(stray == [],
          "Q2: 'hidden' is a section ComfyUI reads and must not be reported, "
          "got %r" % (stray,))
    _need(order == ["alpha", "gamma"],
          "Q2: hidden entries fill no widgets_values slot, got %r" % (order,))

    # ---------------- Q3: the original promise survives -----------------
    order, dynamic, stray = _order_of_src(CLEAN)
    _need(order == ["alpha", "beta", "gamma"],
          "Q3: required widgets then optional widgets, in declaration order; "
          "'wired' is a link and holds no slot. Got %r" % (order,))
    _need(stray == [] and dynamic is False,
          "Q3: a clean node must be neither stray nor dynamic, got %r / %r"
          % (stray, dynamic))

    # ---------------- Q4: a computed key stays 'dynamic' ----------------
    order, dynamic, stray = _order_of_src(COMPUTED)
    _need(dynamic is True,
          "Q4: a computed section key cannot be read statically and must "
          "mark the node dynamic, got %r" % (dynamic,))
    _need(stray == [],
          "Q4: a computed key has no literal name to report -- reporting one "
          "would be inventing it, got %r" % (stray,))

    # ---------------- Q5: the reporting CHAIN, then the live tree -------
    # Driven through _scan() rather than around it. Calling _order_of
    # directly and collecting the strays myself tested my own loop, not the
    # gate's -- a mutation that found every stray and reported none stayed
    # green.
    import glob
    import importlib
    import shutil
    import tempfile
    v577 = importlib.import_module("test_v577_widget_order")

    said = []
    real_fail, real_root = v577._fail, v577.ROOT
    v577._fail = lambda m: said.append(m)
    try:
        tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp, "nodes"))
        with open(os.path.join(tmp, "nodes", "bad.py"), "w",
                  encoding="utf-8") as fh:
            fh.write("class Bad:\n    @classmethod\n" + "\n".join(
                "    " + ln for ln in STRAY.strip().split("\n")))
        v577.ROOT = tmp
        v577._scan()
        _need(len(said) == 1 and "stray" in said[0].lower()
              or (said and "section named" in said[0]),
              "Q5: _scan must REPORT the stray, not merely find it; said %r"
              % (said,))
        _need(said and "Bad" in said[0] and "bad.py" in said[0],
              "Q5: the report must name the class and the file, or nobody "
              "can find it; said %r" % (said,))

        said.clear()
        v577.ROOT = real_root
        v577._scan()
        _need(not said,
              "Q5: the live tree must ship no stray section; got %r" % (said,))
    finally:
        v577._fail, v577.ROOT = real_fail, real_root
        shutil.rmtree(tmp, ignore_errors=True)

    if _fails:
        print("\n[%s] %d/%d FAILED" % (NAME, len(_fails), _checks))
        return 1
    print("[%s] OK -- stray INPUT_TYPES sections are reported, hidden is not "
          "one, and the tree is clean (%d checks)" % (NAME, _checks))
    return 0


if __name__ == "__main__":
    sys.exit(main())
