"""Guard v846 -- Polyhedron Attention: the deviation figure carries its scale.

v845 shipped a live check that printed a bare maximum absolute difference.
The first field run returned 0.6172 -- and there was no way to tell whether
that was a third of the signal or a hundredth of it, because the reference's
own magnitude was never stated. A quality figure nobody can interpret is not
a quality figure. This guard pins the repair.

  Y1  THE RATIOS ARE ARITHMETICALLY RIGHT. Driven on tensors whose deviation
      is known by construction, so the numbers are checked against a value
      computed independently of the code under test -- not against whatever
      the function happens to return.

  Y2  THE TWO FIGURES SAY DIFFERENT THINGS, AND BOTH ARE RIGHT. A moderate
      outlier -- one element a couple of times the signal -- makes the
      maximum look alarming while the rms stays calm, and the verdict must
      follow the rms. But a genuinely broken element (tens of times the
      signal) MUST reach the rms too, and be flagged.
      The first version of this guard promised "one outlier barely moves the
      rms" flatly, and driving it proved the promise wrong: a single element
      of size s over N elements moves the relative rms by about
      s / (rms_base * sqrt(N)), so a 50x spike over 65536 elements moves it
      by 19.5% -- correctly, because that is a real defect. The arithmetic
      was wrong, not the code. Both halves are now pinned.

  Y3  NO SCALE IS INVENTED WHERE NONE EXISTS. An all-zero reference has no
      magnitude to divide by; the node must then report absolutes and stay
      silent about shares rather than print a ratio it cannot support. Same
      for mismatched shapes: no comparison, no guess.

  Y4  THE WORDS MATCH THE NUMBERS. Each band of relative rms maps to its
      label, and the boundaries hold from both sides.

  Y5  SPARSE IS STILL READ THE OTHER WAY ROUND, now with scale: its deviation
      is the omitted information, and a SMALL value there is the bad sign --
      so it never receives a "negligible" verdict.

  Y6  THE OLD CALL SHAPE STILL WORKS. A bare float is accepted and printed as
      an absolute with no share and no verdict -- degrading honestly rather
      than crashing or faking a percentage.
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
    spec = importlib.util.spec_from_file_location("ph_attention_v846", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PH = _load()

print("[test_v846_deviation_scale] the deviation figure carries its scale")

# --------------------------------------------------------------------------
# Y1 -- the ratios are right, checked against independently computed values
# --------------------------------------------------------------------------

print("Y1 the reported ratios equal an independent computation")
torch.manual_seed(846)
base = torch.randn(4, 3, 128, 64) * 7.0
noise = torch.randn_like(base) * 0.05
out = base + noise

stats = PH.deviation_stats(out, base)

want_max = float((out - base).abs().max())
want_rms = math.sqrt(float(((out - base) ** 2).mean()))
want_peak = float(base.abs().max())
want_rms_base = math.sqrt(float((base ** 2).mean()))

checks = [
    ("max_abs", stats["max_abs"], want_max),
    ("rms", stats["rms"], want_rms),
    ("peak_base", stats["peak_base"], want_peak),
    ("rms_base", stats["rms_base"], want_rms_base),
    ("rel_max", stats["rel_max"], want_max / want_peak),
    ("rel_rms", stats["rel_rms"], want_rms / want_rms_base),
]
bad = [n for n, got, want in checks if abs(got - want) > 1e-5 * max(1.0, want)]
if bad:
    _fail("Y1: wrong value(s) for %s" % ", ".join(bad))
else:
    _ok("all six figures match an independent computation "
        "(rel_rms %.4f%%)" % (stats["rel_rms"] * 100.0))

# the point of the whole cut: identical absolute error, different scales,
# different meaning
small_scale = PH.deviation_stats(torch.full((64,), 2.6),
                                 torch.full((64,), 2.0))
large_scale = PH.deviation_stats(torch.full((64,), 80.6),
                                 torch.full((64,), 80.0))
if small_scale["max_abs"] - large_scale["max_abs"] > 1e-4:
    _fail("Y1: the two fixtures do not share the same absolute error")
elif not (small_scale["rel_rms"] > 10 * large_scale["rel_rms"]):
    _fail("Y1: the same absolute error reads the same on both scales -- the "
          "scale is not reaching the ratio")
else:
    _ok("the same absolute error reads %.1f%% on a small scale and %.2f%% on "
        "a large one" % (small_scale["rel_rms"] * 100.0,
                         large_scale["rel_rms"] * 100.0))

# --------------------------------------------------------------------------
# Y2 -- one outlier must not decide the verdict
# --------------------------------------------------------------------------

print("Y2 a moderate outlier alarms the maximum but not the verdict")
clean = torch.randn(1, 2, 512, 64)          # N = 65536, rms ~ 1
near = clean + torch.randn_like(clean) * 0.001
moderate = near.clone()
moderate[0, 0, 0, 0] += 2.0                 # one element, twice the signal

s_near = PH.deviation_stats(near, clean)
s_mod = PH.deviation_stats(moderate, clean)

if s_mod["max_abs"] < 1.9:
    _fail("Y2: the outlier did not reach the maximum at all")
elif s_mod["rel_max"] < 0.3:
    _fail("Y2: the outlier should look alarming in the peak-relative figure")
elif s_mod["rel_rms"] > 0.02:
    _fail("Y2: one moderate element out of 65536 pushed the typical error to "
          "%.3f%% -- the rms is not behaving as a typical error"
          % (s_mod["rel_rms"] * 100.0))
else:
    _ok("moderate outlier: max %.2f (%.0f%% of peak) but rms only %.3f%%"
        % (s_mod["max_abs"], s_mod["rel_max"] * 100.0,
           s_mod["rel_rms"] * 100.0))

line = PH.format_verdict("sage fp8 cuda++", 10.0, 20.0, s_mod)
if "LARGE" in line:
    _fail("Y2: the verdict condemned a kernel over one moderate element: %r"
          % line)
elif "small" not in line:
    _fail("Y2: the verdict did not follow the rms: %r" % line)
else:
    _ok("the verdict follows the typical error, not the worst element")

# the other half: a genuinely broken element MUST be flagged
broken = near.clone()
broken[0, 0, 0, 0] += 50.0
s_broken = PH.deviation_stats(broken, clean)
line = PH.format_verdict("sage fp8 cuda++", 10.0, 20.0, s_broken)
if "LARGE" not in line:
    _fail("Y2: an element 50x the signal was NOT flagged (rms %.2f%%): %r"
          % (s_broken["rel_rms"] * 100.0, line))
else:
    _ok("an element 50x the signal reaches the rms (%.1f%%) and is flagged"
        % (s_broken["rel_rms"] * 100.0))


# --------------------------------------------------------------------------
# Y3 -- no invented scale
# --------------------------------------------------------------------------

print("Y3 no share is printed where there is no scale")
zeros = torch.zeros(32, 8)
s_zero = PH.deviation_stats(torch.full((32, 8), 0.25), zeros)
if s_zero is None:
    _fail("Y3: an all-zero reference produced no statistics at all")
elif s_zero["rel_max"] is not None or s_zero["rel_rms"] is not None:
    _fail("Y3: a ratio was computed against an all-zero reference")
elif abs(s_zero["max_abs"] - 0.25) > 1e-6:
    _fail("Y3: the absolute figures are wrong when no scale exists")
else:
    _ok("all-zero reference: absolutes reported, no ratio invented")

line = PH.format_verdict("pytorch sdpa", 10.0, 20.0, s_zero)
if ("%" in line.split("deviation from")[-1].split("--")[0]
        and "of its signal" in line):
    _fail("Y3: the line still shows a percentage share: %r" % line)
elif "negligible" in line or "LARGE" in line:
    _fail("Y3: a verdict word was given without a relative figure: %r" % line)
else:
    _ok("the printed line states absolutes and offers no verdict")

if PH.deviation_stats(torch.randn(4, 4), torch.randn(4, 5)) is not None:
    _fail("Y3: mismatched shapes were compared anyway")
else:
    _ok("mismatched shapes: no comparison")

if PH.deviation_stats(None, base) is not None:
    _fail("Y3: a missing tensor did not stop the comparison")
else:
    _ok("missing tensor: no comparison")

# --------------------------------------------------------------------------
# Y4 -- the words match the numbers
# --------------------------------------------------------------------------

print("Y4 each band of relative rms gets its word")


def word_for(rel_rms):
    st = {"max_abs": 1.0, "rms": rel_rms, "peak_base": 10.0,
          "rms_base": 1.0, "rel_max": 0.1, "rel_rms": rel_rms}
    return PH.format_verdict("sage fp8 cuda++", 10.0, 20.0, st)


bands = [(0.0001, "negligible"), (0.0049, "negligible"),
         (0.0051, "small"), (0.019, "small"),
         (0.021, "noticeable"), (0.049, "noticeable"),
         (0.051, "LARGE"), (0.5, "LARGE")]
wrong = [(r, w) for r, w in bands if w not in word_for(r)]
if wrong:
    _fail("Y4: wrong word for rel_rms %s" % wrong)
else:
    _ok("all eight probes land in the intended band, boundaries included")

# --------------------------------------------------------------------------
# Y5 -- sparse is read the other way round
# --------------------------------------------------------------------------

print("Y5 the sparse deviation keeps its inverted reading")
st = {"max_abs": 0.6, "rms": 0.001, "peak_base": 10.0, "rms_base": 1.0,
      "rel_max": 0.06, "rel_rms": 0.001}
line = PH.format_verdict(PH.SPARSE_LOCAL, 30.0, 10.0, st)
if "negligible" in line:
    _fail("Y5: a tiny sparse deviation was praised as negligible -- there it "
          "means the mask is doing nothing: %r" % line)
elif "omits" not in line:
    _fail("Y5: the sparse deviation is not labelled as omitted information")
elif "%" not in line:
    _fail("Y5: sparse lost the scale that every other mode now gets")
else:
    _ok("sparse: labelled as omitted information, scale still shown, no "
        "praise for a small value")

line = PH.format_verdict("sage fp8 cuda++", 30.0, 10.0, st)
if "omits" in line:
    _fail("Y5: a non-sparse mode got the mask label")
else:
    _ok("only sparse gets the inverted reading")

# --------------------------------------------------------------------------
# Y6 -- the old call shape degrades honestly
# --------------------------------------------------------------------------

print("Y6 a bare float still works and claims nothing extra")
line = PH.format_verdict("pytorch sdpa (cudnn)", 10.0, 25.0, 0.0006)
if line is None:
    _fail("Y6: a float deviation broke the verdict entirely")
elif "0.0006" not in line:
    _fail("Y6: the float value is not reported: %r" % line)
elif "of its peak" in line or "of its signal" in line:
    _fail("Y6: a share was claimed although no scale was supplied: %r" % line)
elif "negligible" in line or "LARGE" in line:
    _fail("Y6: a verdict word was given without a relative figure: %r" % line)
else:
    _ok("float form: value printed, no share, no verdict")

if PH.format_verdict("pytorch sdpa", 10.0, 25.0, None) is None:
    _fail("Y6: a missing deviation removed the timing line as well")
else:
    _ok("no deviation at all: the timing half still prints")

# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------

if FAILS:
    print("[test_v846_deviation_scale] FAIL -- %d problem(s)" % len(FAILS))
    sys.exit(1)
print("[test_v846_deviation_scale] PASS -- deviation carries its scale, "
      "outliers do not decide, no ratio invented without one")
