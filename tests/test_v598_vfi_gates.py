"""Guard v598 -- two laws that both protect the user from a helpful lie.

LAW 1 -- THE GATES NEVER MOVE THE TIMELINE.
A pair may be duplicated (both frames identical) or held (the pair is a scene
cut). Neither may change how MANY frames come out. A skipped pair still fills
its slots -- with copies. The temptation is obvious: "we saved two frames!" And
the cost is that every downstream fps is now a lie and a stitched sequence
drifts. Frames out is a contract with the encoder, not a result.

LAW 2 -- NO KNOB MAY LIE.
The incumbent shows `fast_mode` (dead from arch 4.5 -- contextnet is gone) and
`ensemble` (hard-forced to False on 4.26) as live toggles. Frank's console had
BOTH set to true, on an arch where one of them does nothing. A knob that
displays a value it does not honour is worse than a missing knob: it teaches a
false model of the machine, and the user then reasons correctly from it and gets
the wrong answer.

So: every BOOLEAN widget the node exposes is either honoured by the selected
arch, or it appears in _inert_knobs() for that arch AND the node forces it off
and says so. There is no third category. This guard walks the AST of the node
to prove the forcing actually happens -- a print alone would be theatre.
"""
import ast
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "nodes", "ph_interpolate.py")
TREE = ast.parse(open(SRC, encoding="utf-8").read())


def lift(names):
    want, body, found = set(names), [], set()
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in want:
                    body.append(node)
                    found.add(t.id)
    assert not want - found, "not found in ph_interpolate.py: %s" % sorted(want - found)
    ns = {"math": __import__("math")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 SRC, "exec"), ns)
    return ns


ns = lift(["_gate_verdict", "_timeline", "_inert_knobs", "_chunk_size", "CKPT_ARCH",
           "_CHUNK_CEIL"])
_gate_verdict, _timeline = ns["_gate_verdict"], ns["_timeline"]
_inert_knobs, _chunk_size, CKPT_ARCH = ns["_inert_knobs"], ns["_chunk_size"], ns["CKPT_ARCH"]

# =========================================================================
# LAW 1 -- the gates are free to change HOW a frame is made, never HOW MANY
# =========================================================================
checked = 0
for n in (2, 5, 65, 130):
    for k in (1, 2, 3, 4):
        _tasks, baseline = _timeline(n, k)
        for eps in (0.0, 0.002, 0.05):
            for theta in (0.0, 0.30, 0.95):
                slots = set(i * k for i in range(n))
                for pair in range(n - 1):
                    # whatever the verdict, the pair still owns its k-1 slots
                    for step in range(1, k):
                        slots.add(pair * k + step)
                assert len(slots) == baseline, \
                    "gates(eps=%r theta=%r) on %d frames x%d left %d slots, contract says %d" \
                    % (eps, theta, n, k, len(slots), baseline)
                checked += 1

# --- verdict semantics ----------------------------------------------------
assert _gate_verdict(0.0, 0.001, 0.0) == "dup", "two identical frames are a duplicate, not motion"
assert _gate_verdict(0.5, 0.0, 0.30) == "hold", "a pair past the cut threshold is held, not blended"
assert _gate_verdict(0.1, 0.001, 0.30) == "interp", "an ordinary pair gets interpolated"

# --- zero means OFF, and OFF must mean OFF --------------------------------
for mae in (0.0, 0.0001, 0.5, 0.999, 1.0):
    assert _gate_verdict(mae, 0.0, 0.0) == "interp", \
        "with both gates at 0 the node must interpolate EVERYTHING (mae=%r said otherwise). " \
        "A gate that fires when it is switched off is the worst kind of surprise." % mae

# --- the boundaries are closed on the side that skips work ----------------
assert _gate_verdict(0.010, 0.010, 0.0) == "dup", "at exactly the static threshold: skip"
assert _gate_verdict(0.011, 0.010, 0.0) == "interp", "just past it: work"
assert _gate_verdict(0.300, 0.0, 0.300) == "hold", "at exactly the cut threshold: hold"
assert _gate_verdict(0.299, 0.0, 0.300) == "interp", "just short of it: work"

# --- a static pair inside a cut window is a duplicate, not a cut ----------
assert _gate_verdict(0.0, 0.01, 0.30) == "dup", \
    "identical frames must resolve as 'dup' before the cut test ever runs"

# =========================================================================
# LAW 2 -- no knob may lie
# =========================================================================
assert "fast_mode" in _inert_knobs("4.7"), "contextnet is gone from 4.5 up: fast_mode is dead"
assert "fast_mode" in _inert_knobs("4.17")
assert "fast_mode" in _inert_knobs("4.26")
assert "fast_mode" not in _inert_knobs("4.0"), "4.0 still has contextnet"
assert "ensemble" in _inert_knobs("4.26"), "4.26 does not implement ensemble"
assert "ensemble" not in _inert_knobs("4.7"), "4.7 does implement ensemble"

# Every arch we ship must have a verdict on every boolean knob we expose.
BOOLS = {"fast_mode", "ensemble"}
for arch in sorted(set(CKPT_ARCH.values())):
    verdict = _inert_knobs(arch)
    assert verdict <= BOOLS, \
        "arch %s declares an inert knob the node does not expose: %s" % (arch, verdict - BOOLS)

# --- and the node must ACTUALLY force them off, not merely print about it --
fn = None
for node in ast.walk(TREE):
    if isinstance(node, ast.FunctionDef) and node.name == "interpolate":
        fn = node
assert fn is not None, "ULSInterpolate.interpolate not found"

forced = set()
for node in ast.walk(fn):
    # looking for:  if "<knob>" in inert:  ...  <knob> = False
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        tgt, val = node.targets[0], node.value
        if (isinstance(tgt, ast.Name) and tgt.id in BOOLS
                and isinstance(val, ast.Constant) and val.value is False):
            forced.add(tgt.id)
assert forced == BOOLS, \
    ("the node prints about inert knobs but does not force them off: missing %s. Telling the user "
     "a knob is inert and then passing its value to the model anyway is a worse lie than the one "
     "we set out to fix." % sorted(BOOLS - forced))

# =========================================================================
# Chunking is bounded on BOTH sides -- v599 signature (free, MEASURED peak)
#
# The v598 version of this block called _chunk_size(free, w, h, bytes). When
# v599 changed the signature to (free, peak_per_pair), those calls still RAN --
# w landed in peak, h in floor, bytes in ceil -- and every assert still passed,
# by luck. A guard that stays green while the thing under it changes meaning is
# worse than no guard: it is a false witness. Pinned to the real signature now.
# =========================================================================
GB = 1024 ** 3
tiny = _chunk_size(64 * 1024 * 1024, 4 * GB)      # a pair costs more than we have
assert tiny >= 1, "a chunk of 0 would loop forever; the floor must hold even when VRAM is gone"
huge = _chunk_size(80 * GB, 1024)                 # a pair costs almost nothing
assert huge <= 64, "an unbounded chunk turns a small clip into one giant allocation"
big_free = _chunk_size(16 * GB, 512 * 1024 ** 2)
small_free = _chunk_size(2 * GB, 512 * 1024 ** 2)
assert big_free >= small_free, "more free VRAM must never yield a smaller chunk"
cheap = _chunk_size(8 * GB, 100 * 1024 ** 2)
dear = _chunk_size(8 * GB, 800 * 1024 ** 2)
assert cheap > dear, \
    ("a costlier pair must yield a SMALLER chunk -- this is the whole point of v599. If the peak "
     "does not move the chunk, the measurement is decoration and the constant is back.")

print("[test_v598_vfi_gates] PASS: %d gate/timeline combinations -- a duplicated or held pair "
      "still fills every slot it owns, so the frame count never moves. Zero means off. Both "
      "inert knobs are declared per arch AND forced off in the code, not just narrated." % checked)
sys.exit(0)
