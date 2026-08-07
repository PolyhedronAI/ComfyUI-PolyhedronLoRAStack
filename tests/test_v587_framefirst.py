"""v587 guard: FRAME-FIRST tile ask + the fit stopwatch.

The 2026-07-13 field run measured the cost of the 1024 cap: a 1104 source ran
a 2x2 grid in the FINAL pass - four model calls per frame plus the feather,
for a frame whose single-call activation (est 10.0 GB) fit the free 15.7 GB
with room to spare. v587's law: on CUDA the FIRST ask is the frame edge, and
the v566 estimate + backoff talk it down when the card disagrees; without
CUDA (no estimate) the old cap stays the first ask. And the fit is no longer
folded silently into the pass total: it gets its own stopwatch and its own
line, with the kernel trade-offs said out loud when they cost real time.

Structure pins, not literals where avoidable (lesson 1): the claims are the
ORDER of decisions and the presence of the spoken measurements.

1st AMENDMENT (v590) - this guard was green over a NameError for three cuts.
The original section 2 asked whether the string `fit_s = {"s": 0.0}` appeared
ANYWHERE in the file (it did - inside _esrgan_resident) and whether there were
exactly TWO feed sites (there were - in two DIFFERENT functions). It never
asked which SCOPE any of them lived in. _esrgan_pass read a name it never
bound, and every path carrying an upscale model died at the line that was
supposed to speak the measurement - the one line this guard exists to protect.
A pin that counts occurrences of a string is a text pin wearing a structure
pin's coat (lesson 1). Section 2 now reads the AST: every function that USES
the stopwatch must BIND it (parameter or assignment), and the caller must hand
it across the call, or the resident path - the common one - would silently
measure zero. Positive and negative probes are built in: a guard that cannot
fail on the broken shape measures nothing (lesson: v581 §5).
"""
import ast, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PU = ROOT / "nodes" / "ph_power_upscale.py"

# v590: the shape that shipped in v587-v589 (declared in one scope, read in
# another) and the shape that fixes it. The checker below must reject the
# first and accept the second - otherwise it is not measuring.
_PROBE_BROKEN = '''
def resident(x):
    fit_s = {"s": 0.0}
    fit_s["s"] += 1.0
    return x

def outer(x):
    y = resident(x)
    fit_s["s"] += 2.0
    if fit_s["s"] > 0.5:
        print(fit_s["s"])
    return y
'''
_PROBE_SOUND = '''
def resident(x, fit_s=None):
    fit_s = {"s": 0.0} if fit_s is None else fit_s
    fit_s["s"] += 1.0
    return x

def outer(x):
    fit_s = {"s": 0.0}
    y = resident(x, fit_s=fit_s)
    fit_s["s"] += 2.0
    if fit_s["s"] > 0.5:
        print(fit_s["s"])
    return y
'''


def _fail(msg):
    print(f"[test_v587_framefirst] FAIL: {msg}")
    sys.exit(1)


def _functions(tree):
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _scope_offenders(src, name):
    """Every function that READS `name` without BINDING it in its own scope.

    A dict mutated through a subscript (`name["s"] += x`) reads the name -
    that is exactly why the v587 shape crashed: the augmented assignment
    targets the SUBSCRIPT, never the name. Binding = parameter or assignment.
    """
    out = []
    for fn in _functions(ast.parse(src)):
        params = {a.arg for a in (list(fn.args.posonlyargs) + list(fn.args.args)
                                  + list(fn.args.kwonlyargs))}
        if fn.args.vararg:
            params.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            params.add(fn.args.kwarg.arg)
        binds, reads = set(params), []
        for node in ast.walk(fn):
            if isinstance(node, ast.Name) and node.id == name:
                if isinstance(node.ctx, ast.Store):
                    binds.add(name)
                elif isinstance(node.ctx, ast.Load):
                    reads.append(node.lineno)
            elif isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
                binds.add(name)
        if reads and name not in binds:
            out.append((fn.name, min(reads)))
    return out


def _hands_across(src, caller, callee, kw):
    """The caller must pass `kw=` into `callee(...)` - without the channel the
    resident path measures zero and the line lies quietly. v590."""
    for fn in _functions(ast.parse(src)):
        if fn.name != caller:
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == callee
                    and any(k.arg == kw for k in node.keywords)):
                return True
    return False


def main():
    pu = PU.read_text(encoding="utf-8")

    # ---- 1: the frame-first ask, CUDA-gated ---------------------------------
    if "edge if is_cuda else min(int(_ESRGAN_TILE_CAP), edge)" not in pu:
        _fail("the first tile ask must be the frame edge on CUDA and the old "
              "cap without it - the estimate+backoff own the talking-down")
    # is_cuda must be DECIDED before the ask that depends on it.
    if pu.index('is_cuda = (getattr(device, "type"') > pu.index(
            "edge if is_cuda else"):
        _fail("is_cuda is decided AFTER the tile ask that reads it")
    # ...and the ask must sit BEFORE the estimate/backoff that tames it.
    if pu.index("edge if is_cuda else") > pu.index(
            "while tile > 256 and act + chunk * out_frame"):
        _fail("the frame-first ask must come BEFORE the v566 estimate/backoff "
              "- frame first, estimate talks it down")
    # The cap itself stays: it is the non-CUDA first ask and backoff floor
    # territory (v565 pins the backoff; this pin keeps the constant employed).
    if "_ESRGAN_TILE_CAP" not in pu:
        _fail("the cap constant is gone - non-CUDA still needs a first ask")

    # ---- 2: the fit stopwatch, PINNED BY SCOPE (1st amendment, v590) --------
    # 2a: the checker must prove it can fail. A guard that passes the broken
    # shape is decoration (v581 §5 - never lower the bar to get a green).
    if not _scope_offenders(_PROBE_BROKEN, "fit_s"):
        _fail("the scope checker does not catch the v587 shape (declared in "
              "one function, read in another) - it is measuring nothing")
    if _scope_offenders(_PROBE_SOUND, "fit_s"):
        _fail("the scope checker rejects the CORRECT shape - it would force "
              "the next author to break the code to get a green")

    # 2b: the real file. Every function that reads the stopwatch binds it.
    offenders = _scope_offenders(pu, "fit_s")
    if offenders:
        where = ", ".join(f"{fn}() line {ln}" for fn, ln in offenders)
        _fail(f"the fit stopwatch is READ where it is not BOUND: {where}. "
              f"This is the v587-v589 crash (NameError: fit_s) - it killed "
              f"every pass carrying an upscale model. py_compile cannot see "
              f"it: syntax is not scope.")

    # 2c: both paths still feed one clock, and the clock crosses the call.
    if pu.count('fit_s["s"] += time.monotonic() - _tf0') != 2:
        _fail("both fit sites (resident + core fallback) must feed the "
              "stopwatch - the closing line speaks for whichever path ran")
    if not _hands_across(pu, "_esrgan_pass", "_esrgan_resident", "fit_s"):
        _fail("_esrgan_pass must hand its stopwatch INTO _esrgan_resident - "
              "without the channel the resident path (the common one) times "
              "its fit into a dict nobody reads, and the line reports 0.0s")

    # 2d: the measurement is spoken, in the scope that owns the clock.
    if "of that, the fit (" not in pu:
        _fail("the pass must SAY the fit's share - a blended total hides "
              "where the seconds live")
    # The lanczos note fires only when the fit actually bites (>= 15%), and
    # it informs - the dial stays the user's.
    if '_share >= 15.0' not in pu or "The dial stays yours." not in pu:
        _fail("the lanczos note must be share-gated and advisory - never "
              "a silent kernel swap, never noise on a cheap fit")

    print("PASS: v587 -- frame-first tile ask (CUDA-gated, estimate-tamed), "
          "fit stopwatch SCOPE-pinned on both paths (1st amendment: probes "
          "in), handed across the call, share-gated kernel note")


if __name__ == "__main__":
    main()
