"""Guard v849 -- Polyhedron Attention: nothing important happens in silence.

The v848 audit found four places where the node knew something and did not
say it, or said it in a way that could not be read. Frank hit the first one
by simply asking "deviation from what?" -- and the honest answer was that the
line never said. The other three came out of reading the same file with that
question in mind.

  Z1  THE DEVIATION NAMES ITS REFERENCE. The speed half already said what the
      comparison partner is; the deviation half said "of the output's peak"
      and "of signal" and left the reader to carry the reference across a
      semicolon. Every share now states whose it is, in its own clause.

  Z2  THE DEVIATION IS MEASURED IN SLICES, AND THE ARITHMETIC IS UNCHANGED.
      Two promises, both driven: the numbers equal an independently computed
      reference AND no single conversion ever sees more than one chunk. The
      second half is instrumented rather than asserted in prose -- a comment
      claiming a property is not a guard (Lehre 4), so torch.Tensor.float is
      wrapped and the largest slice it is handed gets checked.

  Z3  A CALL THE MODE DOES NOT SERVE IS ANNOUNCED -- ONCE, WITH THE REASON.
      Until v849 this branch was mute, so a run could report a kernel it was
      only partly using. Once, because a per-call print in the hot path would
      be its own defect.

  Z4  A FAILED MEASUREMENT SPEAKS. live_compare returns None for any internal
      trouble; the one measurement is spent either way, so silence would
      leave the user unable to tell "could not measure" from "never ran".

  Z5  AND THE v847 PROMISE STILL HOLDS UNDERNEATH: the measurement is spent
      only on a call the mode really serves. Re-driven here because Z3 and Z4
      both touch the same branch, and a repair that quietly undid v847 would
      otherwise pass unnoticed.
"""

import importlib.util
import math
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def _fail(msg):
    FAILS.append(msg)
    print("  FAIL  " + msg)


def _ok(msg):
    print("  ok    " + msg)


def _load():
    path = os.path.join(ROOT, "nodes", "ph_attention.py")
    spec = importlib.util.spec_from_file_location("ph_attention_v849", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PH = _load()

print("[test_v849_audit_silence] nothing important happens in silence")

# --------------------------------------------------------------------------
# Z1 -- the deviation half names what it deviates FROM
# --------------------------------------------------------------------------

print("Z1 the deviation states its reference in its own clause")

torch.manual_seed(849)
base = torch.randn(2, 3, 64, 32) * 4.0
out = base + torch.randn_like(base) * 0.2
stats = PH.deviation_stats(out, base)
line = PH.format_verdict("sage fp8 cuda++", 10.0, 25.0, stats)

if line is None:
    _fail("Z1: no verdict line at all")
else:
    if "deviation from" not in line:
        _fail("Z1: the deviation is not tied to a reference: %r" % line)
    else:
        _ok("the deviation clause says what it is a deviation FROM")

    if "of its peak" not in line:
        _fail("Z1: the maximum's share does not name its owner: %r" % line)
    else:
        _ok("max share reads 'of its peak'")

    if "of its signal" not in line:
        _fail("Z1: the rms share does not name its owner: %r" % line)
    else:
        _ok("rms share reads 'of its signal'")

    # The exact wording that sent Frank looking. If it comes back, the
    # reference is unanchored again even if the numbers are right.
    if "of the output's peak" in line or "of signal," in line \
            or line.endswith("of signal"):
        _fail("Z1: the unanchored v846 wording is back: %r" % line)
    else:
        _ok("the unanchored wording is gone")

# --------------------------------------------------------------------------
# Z2 -- sliced maths: same numbers, bounded working set
# --------------------------------------------------------------------------

print("Z2 the deviation is computed in slices without changing the answer")

want_max = float((out - base).abs().max())
want_rms = math.sqrt(float(((out - base) ** 2).mean()))
want_peak = float(base.abs().max())
want_rms_base = math.sqrt(float((base ** 2).mean()))

whole = PH.deviation_stats(out, base, chunk=10 ** 9)
sliced = PH.deviation_stats(out, base, chunk=97)

for label, got in (("whole", whole), ("sliced", sliced)):
    if got is None:
        _fail("Z2: %s pass produced no statistics" % label)
        continue
    bad = []
    for key, want in (("max_abs", want_max), ("rms", want_rms),
                      ("peak_base", want_peak), ("rms_base", want_rms_base)):
        if abs(got[key] - want) > max(1e-5, abs(want) * 1e-5):
            bad.append("%s %.8f != %.8f" % (key, got[key], want))
    if bad:
        _fail("Z2: %s pass disagrees with the independent value: %s"
              % (label, "; ".join(bad)))
    else:
        _ok("%s pass matches an independently computed reference" % label)

if whole and sliced:
    drift = max(abs(whole[k] - sliced[k])
                for k in ("max_abs", "rms", "peak_base", "rms_base"))
    if drift > 1e-5:
        _fail("Z2: chunking changed the answer by %.8f" % drift)
    else:
        _ok("chunk size does not change the answer (drift %.2e)" % drift)

# The instrumented half: no single conversion may see more than one chunk.
# Without this, deleting the loop and converting the whole tensor would pass
# every numeric check above.
_orig_float = torch.Tensor.float
seen = []


def _recording_float(self, *a, **k):
    seen.append(int(self.numel()))
    return _orig_float(self, *a, **k)


torch.Tensor.float = _recording_float
try:
    PH.deviation_stats(out, base, chunk=128)
finally:
    torch.Tensor.float = _orig_float

total = int(out.numel())
if not seen:
    _fail("Z2: nothing was converted at all -- the probe did not run")
elif max(seen) > 128:
    _fail("Z2: a conversion saw %d elements at once, chunk was 128 "
          "(total %d) -- the working set is not bounded"
          % (max(seen), total))
elif len(seen) < 4:
    _fail("Z2: only %d conversion(s) for %d elements at chunk 128 -- "
          "the loop cannot have run" % (len(seen), total))
else:
    _ok("largest single conversion %d elements at chunk 128 (%d slices, "
        "total %d)" % (max(seen), len(seen), total))

# --------------------------------------------------------------------------
# harness for the override: a router we control, driven through the real
# build_override, so the branches under test are the shipped ones.
# --------------------------------------------------------------------------

MODE = "sage fp8 cuda++"


def _drive(router_factory, live_check=False):
    """Build the real override around a router we control. Returns the
    override plus the three call logs."""
    calls = {"passthrough": [], "verdict": [], "report": []}
    original = PH.build_router
    PH.build_router = lambda mode, sparse=None: router_factory()
    try:
        override = PH.build_override(
            MODE, fallback=True,
            report=lambda exc: calls["report"].append(exc),
            live_check=live_check,
            verdict=lambda m, a, b, d: calls["verdict"].append((m, a, b, d)),
            passthrough=lambda m, skip: calls["passthrough"].append(
                (m, str(skip))),
        )
    finally:
        PH.build_router = original
    return override, calls


def _served(*args, **kwargs):
    """Stands in for core's own attention: the thing the override wraps."""
    return torch.full((2, 2), 7.0)


# --------------------------------------------------------------------------
# Z3 -- a pass-through is announced once, with its reason
# --------------------------------------------------------------------------

print("Z3 a call the mode does not serve is announced once, with the reason")


def _always_passes():
    def _r(func, *a, **k):
        raise PH.PassThrough("cross attention")
    return _r


override, calls = _drive(_always_passes)
if override is None:
    _fail("Z3: no override was built")
else:
    outs = [override(_served) for _ in range(3)]
    if len(calls["passthrough"]) != 1:
        _fail("Z3: the pass-through was announced %d time(s), expected exactly 1"
              % len(calls["passthrough"]))
    else:
        _ok("announced exactly once across three pass-through calls")

    if calls["passthrough"] and "cross attention" not in calls["passthrough"][0][1]:
        _fail("Z3: the reason was not passed on: %r" % (calls["passthrough"][0],))
    elif calls["passthrough"]:
        _ok("the reason travels with the announcement (%r)"
            % calls["passthrough"][0][1])

    if not all(float(o[0][0]) == 7.0 for o in outs):
        _fail("Z3: a passed-through call was not served by the model's backend")
    else:
        _ok("every passed-through call was still served, all three times")

    if calls["report"]:
        _fail("Z3: a pass-through was reported as a failure: %r"
              % calls["report"])
    else:
        _ok("a pass-through is not counted as a fallback failure")

# --------------------------------------------------------------------------
# Z4 -- a measurement that fails says so
# --------------------------------------------------------------------------

print("Z4 a live check that cannot measure reports that, rather than nothing")


def _always_breaks():
    def _r(func, *a, **k):
        raise RuntimeError("out of memory (simulated)")
    return _r


override, calls = _drive(_always_breaks, live_check=True)
if override is None:
    _fail("Z4: no override was built")
else:
    result = override(_served)
    if len(calls["verdict"]) != 1:
        _fail("Z4: the verdict callback fired %d time(s), expected 1"
              % len(calls["verdict"]))
    else:
        mode_, ms_chosen, ms_base, dev = calls["verdict"][0]
        if ms_chosen is not None:
            _fail("Z4: a failed measurement reported a timing: %r"
                  % (calls["verdict"][0],))
        else:
            _ok("the failed measurement is signalled by a None timing")
        if mode_ != MODE:
            _fail("Z4: the failed measurement named the wrong mode: %r" % mode_)
        else:
            _ok("the failed measurement names the mode it belongs to")

    if float(result[0][0]) != 7.0:
        _fail("Z4: the run did not continue on the model's own backend")
    else:
        _ok("the run continued on the model's own backend")

# --------------------------------------------------------------------------
# Z5 -- v847 still holds: the measurement is spent on a SERVED call
# --------------------------------------------------------------------------

print("Z5 the one measurement is still spent only on a call the mode serves")


def _passes_then_serves():
    state = {"n": 0}

    def _r(func, *a, **k):
        state["n"] += 1
        if state["n"] <= 1:
            raise PH.PassThrough("grouped heads (q 40, k 8)")
        return torch.full((2, 2), 3.0)
    return _r


override, calls = _drive(_passes_then_serves, live_check=True)
if override is None:
    _fail("Z5: no override was built")
else:
    first = override(_served)
    if calls["verdict"]:
        _fail("Z5: the measurement was spent on a call the mode passed through")
    else:
        _ok("the passed-through call did not consume the measurement")
    if float(first[0][0]) != 7.0:
        _fail("Z5: the passed-through call was not served normally")
    else:
        _ok("the passed-through call was served by the model's backend")

    second = override(_served)
    if len(calls["verdict"]) != 1:
        _fail("Z5: the next served call produced %d verdict(s), expected 1"
              % len(calls["verdict"]))
    else:
        mode_, ms_chosen, ms_base, dev = calls["verdict"][0]
        if ms_chosen is None:
            _fail("Z5: the served call still could not be measured")
        else:
            _ok("the next served call was measured (%.4f ms vs %.4f ms)"
                % (ms_chosen, ms_base))

    third = override(_served)
    if len(calls["verdict"]) != 1:
        _fail("Z5: the measurement fired again on a later call (%d total)"
              % len(calls["verdict"]))
    else:
        _ok("it stays at exactly one measurement per patched model")

    if len(calls["passthrough"]) != 1:
        _fail("Z5: the pass-through notice fired %d time(s)"
              % len(calls["passthrough"]))
    else:
        _ok("the pass-through notice stayed at one as well")

# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------

if FAILS:
    print("[test_v849_audit_silence] FAIL -- %d problem(s)" % len(FAILS))
    sys.exit(1)
print("[test_v849_audit_silence] PASS -- the deviation names its reference, "
      "the maths is sliced, pass-throughs and failed measurements are heard")
