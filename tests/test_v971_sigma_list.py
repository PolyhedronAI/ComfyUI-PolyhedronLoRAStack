#!/usr/bin/env python3
"""v971 -- an explicit sigma grid, GIVEN rather than computed.

Distilled few-step recipes (HyperFlow, AYS, turbo builds) publish a TRAINED
sigma grid: a fixed list of numbers, not a curve shape. The pack could build
curves from parameters and nothing else, so such a grid could not be entered
at all. ULSSigmaList closes that hole.

Why a separate node and not a ninth entry in ULSUniversalSigmaCurve's combo:
ComfyUI serialises widget values BY INDEX (guard #577). Appending a text
widget to the curve node -- which is wired into essentially every saved
workflow -- renumbers all of them. A new node leaves stored graphs untouched.

Every promise below is RUN, not read:

  S1  separators are tolerant: comma, semicolon, space, newline, mixed
  S2  a non-number is refused, and the message names the offending token
  S3  the contract is enforced: strictly decreasing, finite, non-negative
  S4  shift == 1.0 is the identity (an already-shifted grid is untouched)
  S5  shift applies s*x/(1+(s-1)*x) exactly, checked against the formula
      computed independently here, and maps 0 -> 0 and 1 -> 1
  S6  shift on a non-normalised (k-diffusion) grid is REFUSED, not mangled
  S7  the terminal zero is appended only when missing, and can be turned off
  S8  steps == points - 1, and the SIGMAS tensor carries every point
  S9  the V3 schema and the legacy INPUT_TYPES expose the SAME widgets in the
      SAME order -- two places describing one node WILL drift unless a guard
      runs both (the standing lesson from the mirror cases). RE-ARGUED in v972:
      the promise is that the BIRTH widgets keep their index and new ones are
      appended, not that the count never changes.
  S10 the node is registered under an id of its own and does not touch the
      curve node's widget list

MUTATION PROBE (run 20.09.2026, each mutation applied and reverted):
  1. _SIGMA_SPLIT narrowed to r","                     -> S1 caught
  2. the strictly-decreasing loop deleted              -> S3 caught
  3. math.isfinite check deleted                       -> S3 caught
  4. shift early-return for 1.0 changed to `return []` -> S4 caught
  5. shift formula denominator (shift-1) -> (shift+1)  -> S5 caught
  6. the >1.0 refusal in shift_sigma_list deleted      -> S6 caught
  7. terminal zero appended unconditionally            -> S7 caught
  8. steps computed as len(vals)                       -> S8 caught
  9. V3 'shift' input moved after 'enforce_terminal_zero' -> S9 caught
"""
import os, sys, math, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

failures = []
ran = []


def check(cond, msg):
    ran.append(msg)
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def raises(fn, needle=None):
    """True when fn() raises, and -- when given -- the message names `needle`."""
    try:
        fn()
    except Exception as e:
        return needle is None or needle in str(e)
    return False


from nodes.wan_sigma_schedule import (ULSSigmaList, parse_sigma_list,
                                      validate_sigma_list, shift_sigma_list)

N = ULSSigmaList()

# --- S1 separators ---------------------------------------------------------
REF = [1.0, 0.8, 0.6, 0.4, 0.0]
for label, text in [
    ("comma+space", "1.0, 0.8, 0.6, 0.4, 0.0"),
    ("comma only", "1.0,0.8,0.6,0.4,0.0"),
    ("spaces", "1.0 0.8 0.6 0.4 0.0"),
    ("newlines", "1.0\n0.8\n0.6\n0.4\n0.0"),
    ("semicolons", "1.0; 0.8; 0.6; 0.4; 0.0"),
    ("mixed + ragged whitespace", "  1.0,0.8   0.6\n , 0.4 ;0.0  "),
]:
    check(parse_sigma_list(text) == REF, "S1 separator form parses: %s" % label)

check(raises(lambda: parse_sigma_list("1.0")),
      "S1 a single value is refused (a grid needs at least two points)")

# --- S2 a non-number names itself -----------------------------------------
check(raises(lambda: parse_sigma_list("1.0, zero.eight, 0.0"), "zero.eight"),
      "S2 a non-number is refused and the message names the token")

# --- S3 the contract -------------------------------------------------------
check(raises(lambda: validate_sigma_list([1.0, 0.5, 0.5, 0.0])),
      "S3 equal neighbours are refused (strictly decreasing)")
check(raises(lambda: validate_sigma_list([1.0, 0.2, 0.6, 0.0])),
      "S3 an ascending step is refused")
check(raises(lambda: validate_sigma_list([1.0, float("nan"), 0.0])),
      "S3 NaN is refused")
check(raises(lambda: validate_sigma_list([float("inf"), 1.0, 0.0])),
      "S3 infinity is refused")
check(raises(lambda: validate_sigma_list([1.0, 0.5, -0.1])),
      "S3 a negative sigma is refused")
check(validate_sigma_list(list(REF)) == REF, "S3 a valid grid passes untouched")

# --- S4 shift 1.0 is the identity -----------------------------------------
check(shift_sigma_list(REF, 1.0) == REF, "S4 shift 1.0 leaves the grid alone")

# --- S5 the shift formula, computed independently --------------------------
RAW = [1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0]
for s in (2.0, 3.0, 12.0, 0.25):
    want = [s * v / (1.0 + (s - 1.0) * v) for v in RAW]
    got = shift_sigma_list(RAW, s)
    check(all(abs(a - b) < 1e-12 for a, b in zip(got, want)),
          "S5 shift %s matches s*x/(1+(s-1)*x) on every point" % s)
    check(abs(got[0] - 1.0) < 1e-12 and abs(got[-1]) < 1e-12,
          "S5 shift %s maps 1 -> 1 and 0 -> 0 (a valid grid stays valid)" % s)
    check(all(got[i] < got[i - 1] for i in range(1, len(got))),
          "S5 shift %s keeps the grid strictly decreasing" % s)
check(raises(lambda: shift_sigma_list(REF, 0.0)), "S5 a non-positive shift is refused")

# --- S6 a k-diffusion range is refused, not mangled ------------------------
KD = [14.61, 6.47, 3.86, 1.88, 0.029]
check(raises(lambda: shift_sigma_list(KD, 12.0), "normalised"),
      "S6 shifting a non-normalised grid is refused, and the message says why")
check(shift_sigma_list(KD, 1.0) == KD, "S6 the same grid passes with shift off")

# --- S7 terminal zero ------------------------------------------------------
sig, steps = N.compute("1.0, 0.8, 0.5", 1.0, True)
check([round(float(x), 6) for x in sig] == [1.0, 0.8, 0.5, 0.0],
      "S7 a missing terminal zero is appended")
sig, steps = N.compute("1.0, 0.8, 0.0", 1.0, True)
check([round(float(x), 6) for x in sig] == [1.0, 0.8, 0.0],
      "S7 an existing terminal zero is NOT doubled")
sig, steps = N.compute("1.0, 0.8, 0.5", 1.0, False)
check([round(float(x), 6) for x in sig] == [1.0, 0.8, 0.5],
      "S7 the append can be turned off")

# --- S8 steps and tensor ---------------------------------------------------
sig, steps = N.compute("1.0, 0.8, 0.6, 0.4, 0.0", 1.0, True)
check(steps == 4, "S8 steps == points - 1")
check(len(sig) == 5, "S8 the SIGMAS tensor carries every point")
check(str(sig.dtype).endswith("float32"), "S8 the tensor is float32, as samplers expect")

# --- S8b a published RAW grid, entered verbatim and shifted here -----------
sig, steps = N.compute(", ".join(repr(v) for v in RAW), 12.0, True)
want = [12.0 * v / (1.0 + 11.0 * v) for v in RAW]
check(steps == 8, "S8 a nine-point grid reports 8 steps")
check(all(abs(float(a) - b) < 1e-6 for a, b in zip(sig, want)),
      "S8 a raw grid pasted verbatim comes out shifted, no hand arithmetic")

# --- S9 the two descriptions of one node may not drift ---------------------
legacy_order = list(ULSSigmaList.INPUT_TYPES()["required"].keys())
# RE-ARGUED in v972. The promise was never "this node has exactly three
# widgets" -- it was "the widgets this node was born with keep their index,
# and anything new is APPENDED" (guard #577). v972 adds `preset`; pinning the
# literal list would have made the guard fight its own law. The check below is
# the promise in its stronger form: the birth widgets, in order, at the front.
BIRTH = ["sigmas_text", "shift", "enforce_terminal_zero"]
check(legacy_order[:len(BIRTH)] == BIRTH,
      "S9 the widgets this node was born with keep their original index")
try:
    from nodes.wan_sigma_list_v3 import ULSSigmaListV3
    schema = ULSSigmaListV3.define_schema()
    v3_order = [i.id if hasattr(i, "id") else i.name for i in schema.inputs]
    check(v3_order == legacy_order,
          "S9 the V3 schema exposes the same widgets in the same order")
    check(getattr(schema, "node_id", None) == "ULSSigmaList",
          "S9 the V3 schema claims the same node id")
except ImportError:
    # comfy_api absent in this sandbox: read the source instead, so the check
    # still RUNS rather than silently passing.
    src = open(os.path.join(ROOT, "nodes", "wan_sigma_list_v3.py"), encoding="utf-8").read()
    found = re.findall(r'io\.\w+\.Input\(\s*\n?\s*"(\w+)"', src)
    check(found == legacy_order,
          "S9 the V3 source declares the same widgets in the same order "
          "(comfy_api absent, read from source): %s" % found)
    check('node_id="ULSSigmaList"' in src, "S9 the V3 source claims the same node id")

# --- S10 registration, and the curve node left alone -----------------------
init = open(os.path.join(ROOT, "__init__.py"), encoding="utf-8").read()
init_code = re.sub(r"^\s*#.*$", "", init, flags=re.M)
check('NODE_CLASS_MAPPINGS["ULSSigmaList"]' in init_code,
      "S10 the node is registered under its own id")
from nodes.wan_sigma_schedule import ULSUniversalSigmaCurve
check(list(ULSUniversalSigmaCurve.INPUT_TYPES()["required"].keys()) ==
      ["sigma_schedule", "steps", "sigma_max", "sigma_min", "rho"],
      "S10 the Sigma Curve widget list is UNCHANGED (no saved workflow moves)")

print()
if failures:
    print("v971 sigma list: FAIL (%d)" % len(failures))
    for f in failures:
        print("   - " + f)
    sys.exit(1)
print("%d checks, 0 failed" % len(ran))
print("v971 sigma list: PASS")
