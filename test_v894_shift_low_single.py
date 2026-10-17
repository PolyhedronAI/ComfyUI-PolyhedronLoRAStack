#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v894 -- sigma_shift_low is a DUAL_ONLY twin, front and back.

THE WOUND (found by Frank on his own screen, s_175458_power_upscale.png): mode
Single, sigma_shift 8.00, sigma_shift_low 5.00 -- and the row was ACTIVE. Three
separate things were wrong and only the first one was visible:

  1. web/js/ph_power_upscale.js had a DUAL_ONLY list without sigma_shift_low.
     The widget arrived in v851, went into the canon and into the twin/display
     arrays, and never reached that list. uls_sampler.js has carried it in its
     own DUAL_ONLY since v839, so the two nodes disagreed.
  2. The backend did REAL WORK for it. plan_stages(False, ...) yields ONE stage
     tagged 'single'; model_low is read only where the tag is 'low'. With
     Frank's values the node still built a second shifted clone through
     _apply_sigma_shift on every run -- memory and time for a model the stage
     never touches.
  3. The console announced 'sigma shift H=8 L=5', so the log agreed with the
     dial and not with the run.

WHAT IS PINNED HERE, and where it can break:

  D1  the frontend list itself, parsed as a LIST in node -- not a substring
      match (a substring is not proof, #v740 lesson).
  D2  the two nodes agree: whatever uls_sampler.js calls DUAL_ONLY contains
      sigma_shift_low too, so a future twin cannot be added to one and
      forgotten in the other.
  B1  SINGLE BUILDS NOTHING FOR STAGE L. The real block is lifted from the
      source and RUN with a counting _apply_sigma_shift. The count is the
      promise: exactly one call (the HIGH one), never two.
  B2  the HIGH shift is untouched in Single -- the gate must not swallow the
      one shift that does reach the single stage.
  B3  DUAL IS BIT-FOR-BIT v851. Four dual cases are run and compared against an
      independent expectation written from the v851 specification, not from the
      code.
  B4  a dialled value on a Single run is SAID, once (the v552/v885 rule against
      silent degradation, applied to the other direction).
  B5  the console line does not claim an L that never arrived.
  B6  the v851 module claim ("can never be inert") is gone from the source.

WHY _apply_sigma_shift IS A STAND-IN AND _resolve_low_shift IS NOT: the promise
under test is "this function is NOT CALLED", so a counter is exactly the right
instrument and a real model would prove nothing extra. The resolution of the -1
sentinel, by contrast, IS arithmetic under test -- so the real function is
lifted out of nodes/uls_sampler.py rather than restated here. A second copy of
a rule is a second truth (v749).
"""

import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _lift          # noqa: E402  -- the shared closed lift
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, "nodes", "ph_power_upscale.py")
SAMPLER = os.path.join(ROOT, "nodes", "uls_sampler.py")
JS = os.path.join(ROOT, "web", "js", "ph_power_upscale.js")
JS_SAMPLER = os.path.join(ROOT, "web", "js", "uls_sampler.js")

FAILED = []


def _fail(msg):
    FAILED.append(msg)
    print("FAIL: " + msg)


def _need(cond, msg):
    if not cond:
        _fail(msg)


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# lifting
# ---------------------------------------------------------------------------

def _lift_block(src):
    """The sigma-shift block of upscale(), from the _dual gate to the console
    line. Anchored on unique CODE lines, never on prose (a comment anchor pins
    the prose, v740 lesson)."""
    start = src.index("        _dual = bool(dual_moe)")
    end_anchor = "        stages = uls_tile_math.plan_stages("
    end = src.index(end_anchor, start)
    block = src[start:end]
    # dedent the 8 spaces of the method body
    return "\n".join(ln[8:] if ln.startswith("        ") else ln
                     for ln in block.split("\n"))


def _lift_resolver(src):
    start = src.index("def _resolve_low_shift(shift, shift_low):")
    end = src.index("\ndef ", start + 10)
    return src[start:end]


def _lift_js_list(src, name):
    """Return the named const array as real data, by evaluating it in node."""
    m = re.search(r"const\s+" + re.escape(name) + r"\s*=\s*(\[.*?\]);", src, re.S)
    if not m:
        return None
    body = m.group(1)
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                     encoding="utf-8") as fh:
        fh.write("const L = " + body + ";\n")
        fh.write("process.stdout.write(JSON.stringify(L));\n")
        tmp = fh.name
    try:
        out = subprocess.run([_node(), tmp], capture_output=True, text=True,
                             timeout=60)
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)
    finally:
        os.unlink(tmp)


def _node():
    for cand in ("node", "nodejs"):
        try:
            subprocess.run([cand, "--version"], capture_output=True, timeout=30)
            return cand
        except OSError:
            continue
    return "node"


# ---------------------------------------------------------------------------
# the driver: run the REAL block with a counting _apply_sigma_shift
# ---------------------------------------------------------------------------

class _Model(object):
    def __init__(self, tag):
        self.tag = tag

    def __repr__(self):
        return "<%s>" % self.tag


def _drive(block, resolver, dual, hi, low, low_wired):
    """Run the lifted block. Returns what the loop would then see."""
    calls = []
    lines = []

    def _apply(model, shift):
        calls.append((getattr(model, "tag", model), shift))
        return _Model("%s+shift%g" % (getattr(model, "tag", "?"), shift))

    ns = {}
    exec(compile(resolver, "<resolver>", "exec"), ns)
    ns["_apply_sigma_shift"] = _apply
    ns["print"] = lambda *a, **k: lines.append(" ".join(str(x) for x in a))
    ns["dual_moe"] = dual
    ns["sigma_shift"] = hi
    ns["sigma_shift_low"] = low
    ns["model"] = _Model("H")
    ns["model_low"] = _Model("L") if low_wired else None
    exec(compile(block, "<sigmablock>", "exec"), ns)
    return {
        "calls": calls,
        "lines": lines,
        "model": getattr(ns["model"], "tag", None),
        "model_low": (getattr(ns["model_low"], "tag", None)
                      if ns["model_low"] is not None else None),
    }


def main():
    py = _read(PY)
    js = _read(JS)
    block = _lift_block(py)
    resolver = _lift_resolver(_read(SAMPLER))

    # -- the lifted block must be closed: no free names beyond what we inject --
    # v896: this uses the SHARED reader in tests/_lift.py. It was written here
    # first, then v896 needed the same thing for the v685/v839 lifts -- and two
    # copies of a rule are two truths (v749). Read off the AST, not off a regex:
    # a regex counts 'if' and 'and' out of the prose and reports them as free
    # names (measured while writing this).
    injected = {"_apply_sigma_shift", "_resolve_low_shift", "print", "dual_moe",
                "sigma_shift", "sigma_shift_low", "model", "model_low"}
    unknown = _lift.free_names(block, injected)
    _need(not unknown,
          "lifted block calls names the harness does not provide: %s "
          "(this is the v685/v839 shape -- fix the lift, not the tree)" % unknown)

    # ---------------- D1 / D2: the frontend lists --------------------------
    dual_only = _lift_js_list(js, "DUAL_ONLY")
    _need(isinstance(dual_only, list),
          "D1: DUAL_ONLY could not be evaluated as a list in ph_power_upscale.js")
    if isinstance(dual_only, list):
        _need("sigma_shift_low" in dual_only,
              "D1: sigma_shift_low is NOT in ph_power_upscale.js DUAL_ONLY -- "
              "Single would show an active dial that reaches nothing")
        for twin in ("upscale_by_low", "denoise_low", "steps_low", "cfg_low",
                     "sampler_low", "scheduler_low"):
            _need(twin in dual_only,
                  "D1: %s fell out of DUAL_ONLY" % twin)

    s_dual = _lift_js_list(_read(JS_SAMPLER), "DUAL_ONLY")
    if isinstance(s_dual, list):
        _need("sigma_shift_low" in s_dual,
              "D2: uls_sampler.js DUAL_ONLY lost sigma_shift_low -- the two "
              "nodes must agree about which dials are dual-only")
    else:
        print("note: uls_sampler.js DUAL_ONLY not evaluable; D2 skipped")

    # ---------------- B1 / B2: Single builds nothing for stage L -----------
    for low_wired in (False, True):
        r = _drive(block, resolver, dual=False, hi=8.0, low=5.0,
                   low_wired=low_wired)
        _need(len(r["calls"]) == 1,
              "B1: Single with sigma_shift_low=5 made %d shift calls, expected "
              "exactly 1 (the HIGH one); low_wired=%s"
              % (len(r["calls"]), low_wired))
        _need(r["calls"][:1] == [("H", 8.0)],
              "B2: the HIGH shift must still run in Single (got %r)"
              % (r["calls"],))
        _need(r["model"] == "H+shift8",
              "B2: 'model' must carry the high shift in Single (got %r)"
              % r["model"])

    # a Single run with the sentinel untouched must be exactly as before
    r = _drive(block, resolver, dual=False, hi=8.0, low=-1.0, low_wired=False)
    _need(len(r["calls"]) == 1 and r["model_low"] is None,
          "B1: Single on the sentinel must build nothing and leave model_low "
          "alone (calls=%r, model_low=%r)" % (r["calls"], r["model_low"]))

    # and Single with no shift at all must call nothing
    r = _drive(block, resolver, dual=False, hi=0.0, low=5.0, low_wired=False)
    _need(not r["calls"],
          "B1: Single, high shift OFF -> no shift call at all (got %r)"
          % (r["calls"],))

    # ---------------- B3: dual is bit-for-bit v851 -------------------------
    # expectation written from the v851 SPEC, not from the code:
    #   low_shift = high when sigma_shift_low < 0 else sigma_shift_low
    #   high clone when high > 0
    #   low clone from the RAW low source when low_shift > 0 and
    #     (a low expert is wired OR the low shift differs from the high one)
    #   unwired + low OFF + high shifted -> the loop gets the RAW model
    cases = [
        # (hi, low, wired) -> expected calls
        (8.0, 5.0, False, [("H", 8.0), ("H", 5.0)]),
        (8.0, 5.0, True, [("H", 8.0), ("L", 5.0)]),
        (8.0, -1.0, True, [("H", 8.0), ("L", 8.0)]),
        (8.0, -1.0, False, [("H", 8.0)]),
        (8.0, 0.0, False, [("H", 8.0)]),
        (0.0, 5.0, False, [("H", 5.0)]),
    ]
    for hi, low, wired, want in cases:
        r = _drive(block, resolver, dual=True, hi=hi, low=low, low_wired=wired)
        _need(r["calls"] == want,
              "B3: dual hi=%g low=%g wired=%s -> %r, v851 spec says %r"
              % (hi, low, wired, r["calls"], want))

    # the v851 unwired-OFF fallback: the loop must get the RAW model, not the
    # shifted one
    r = _drive(block, resolver, dual=True, hi=8.0, low=0.0, low_wired=False)
    _need(r["model_low"] == "H",
          "B3: dual, low OFF, unwired -> stage L must get the RAW model "
          "(got %r)" % r["model_low"])

    # ---------------- B4 / B5: what the console says -----------------------
    r = _drive(block, resolver, dual=False, hi=8.0, low=5.0, low_wired=False)
    said = " ".join(r["lines"])
    _need("sigma_shift_low" in said and "ignored" in said,
          "B4: a dialled sigma_shift_low on a Single run must be reported "
          "once, got %r" % (r["lines"],))
    _need(said.count("ignored") == 1,
          "B4: the honesty line must speak exactly once (got %r)" % (r["lines"],))
    _need("L=5" not in said,
          "B5: the console must not claim an L shift that never reached a "
          "stage, got %r" % (r["lines"],))

    r = _drive(block, resolver, dual=False, hi=8.0, low=-1.0, low_wired=False)
    said = " ".join(r["lines"])
    _need("ignored" not in said,
          "B4: an untouched sentinel is not a mistake and must stay silent, "
          "got %r" % (r["lines"],))

    r = _drive(block, resolver, dual=True, hi=8.0, low=5.0, low_wired=False)
    said = " ".join(r["lines"])
    _need("L=5" in said,
          "B5: dual must still print the L shift, got %r" % (r["lines"],))
    _need("ignored" not in said,
          "B4: dual must not print the honesty line, got %r" % (r["lines"],))

    # ---------------- B6: the corrected claim ------------------------------
    _need("can never be inert" not in py,
          "B6: the v851 claim 'can never be inert' is still in the source -- a "
          "promise is re-grounded, not left standing next to its counter-example")
    _need("Always honoured" not in py,
          "B6: the sigma_shift_low tooltip still says 'Always honoured'")
    _need("HIGH + LOW ONLY" in py,
          "B6: the tooltip must name the mode in which the dial works")

    if FAILED:
        print("\n%d FAILED" % len(FAILED))
        return 1
    print("v894 OK -- sigma_shift_low is a dual-only twin in both halves "
          "(%d checks)" % (len(cases) + 16))
    return 0


if __name__ == "__main__":
    sys.exit(main())
