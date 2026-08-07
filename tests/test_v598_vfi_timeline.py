"""Guard v598 -- the timeline law.

THE LAW: k frames per pair over N inputs yields k*(N-1)+1 outputs. Not k*N.
Every input frame survives; each of the N-1 gaps receives k-1 new frames.

WHY IT IS PINNED: the field run of 2026-07-14 put 65 frames in and got
"129 frames generated" out at multiplier 2. 2*(65-1)+1 = 129. A node that
believed 130 would hand the encoder a frame count that does not match its own
fps, and the drift would be one frame per clip -- small enough to survive
review, large enough to desynchronise a stitched sequence.

WHAT THIS GUARD DOES NOT DO: it does not check that the file contains the
string "n_frames - 1". It EXECUTES the function and compares return values, on
1200 combinations. A guard that reads its expectation out of the file it is
checking measures nothing (ledger lesson: v595).
"""
import ast
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "nodes", "ph_interpolate.py")


def lift(names):
    """Exec the named module-level defs/assignments in a bare namespace.

    The module itself imports torch, which the sandbox does not have -- and
    importing it would test the import, not the law. So the pure layer is
    carved out with the AST and run on its own.
    """
    tree = ast.parse(open(SRC, encoding="utf-8").read())
    want, body, found = set(names), [], set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in want:
                    body.append(node)
                    found.add(t.id)
    missing = want - found
    assert not missing, "not found at module level in ph_interpolate.py: %s" % sorted(missing)
    ns = {"math": __import__("math")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 SRC, "exec"), ns)
    return ns


ns = lift(["_timeline", "_fps_plan"])
_timeline, _fps_plan = ns["_timeline"], ns["_fps_plan"]

checked = 0

# --- The law, swept -------------------------------------------------------
for n in range(2, 202):
    for k in range(1, 7):
        tasks, out = _timeline(n, k)
        assert out == k * (n - 1) + 1, \
            "timeline(%d, %d) promised %d frames, law says %d" % (n, k, out, k * (n - 1) + 1)
        assert len(tasks) == (k - 1) * (n - 1), \
            "timeline(%d, %d) built %d tasks, expected %d" % (n, k, len(tasks), (k - 1) * (n - 1))
        # every task lands strictly between its two source frames
        for pair, t in tasks:
            assert 0 <= pair < n - 1, "pair %d out of range for %d frames" % (pair, n)
            assert 0.0 < t < 1.0, "timestep %r is not strictly between the frames" % (t,)
        # every output slot is written exactly once: the source frames sit on
        # multiples of k, the new frames fill the gaps between them.
        slots = set(i * k for i in range(n))
        for pair, t in tasks:
            slot = pair * k + int(round(t * k))
            assert slot not in slots, \
                "slot %d written twice (pair %d, t %r) -- a frame would be lost" % (slot, pair, t)
            slots.add(slot)
        assert len(slots) == out, \
            "timeline(%d, %d) leaves holes: %d slots for %d frames" % (n, k, len(slots), out)
        checked += 1

# --- The field case, by name ----------------------------------------------
tasks, out = _timeline(65, 2)
assert out == 129, "THE FIELD RUN SAID 129. This says %d." % out
assert len(tasks) == 64, "65 frames have 64 gaps, not %d" % len(tasks)

# --- multiplier 1 is a pass-through, not an error -------------------------
tasks, out = _timeline(10, 1)
assert (tasks, out) == ([], 10), "multiplier 1 must be a clean no-op, got %r/%r" % (tasks, out)

# --- fps mode reports the truth, not the wish -----------------------------
mult, actual, out = _fps_plan(65, 16.0, 32.0)
assert (mult, actual, out) == (2, 32.0, 129), "16 -> 32 fps must be x2/129, got %r" % ((mult, actual, out),)
mult, actual, out = _fps_plan(65, 16.0, 24.0)
assert mult == 2 and abs(actual - 32.0) < 1e-9, \
    "24 fps is not an integer multiple of 16: the node must land on 32 and SAY so, not silently " \
    "fake 24. Got x%d -> %.2f" % (mult, actual)
mult, actual, _ = _fps_plan(65, 16.0, 16.0)
assert mult == 1, "same fps in, same fps out -- no work to do"

# --- the refusals ---------------------------------------------------------
for bad in [(1, 2), (0, 2)]:
    try:
        _timeline(*bad)
        raise AssertionError("timeline%r should refuse: fewer than 2 frames is not a pair" % (bad,))
    except ValueError:
        pass
try:
    _timeline(10, 0)
    raise AssertionError("multiplier 0 should refuse")
except ValueError:
    pass

print("[test_v598_vfi_timeline] PASS: %d combinations, out = k*(N-1)+1 holds, every slot written "
      "exactly once, 65@x2 = 129 as the field measured, fps mode reports the rate it will "
      "actually deliver." % checked)
sys.exit(0)
