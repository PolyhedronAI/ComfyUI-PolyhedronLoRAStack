# -*- coding: ascii -*-
"""Guard -- the toast must not promise a truncation that cannot happen (v908).

FRANK'S FIELD RUN. Screenshot, 29.08.: the report header reads

    Encoder : qwen3vl_32b - no cap (never truncates)

and half a second later a sticky toast pops up over the canvas:

    Token limit exceeded
    Prompt is 580/512 tokens (over by 68). It may be silently truncated or
    crash kijai's WanVideoSampler.

Both halves are false on this graph. qwen3vl carries max_length=99999999
(core v0.33.4), and there is no kijai sampler anywhere in it. The node's own
report and its own toast contradicted each other on screen, and the toast is
the louder one -- it appears unasked. Frank trimmed an entire prompt session
against that sentence.

WHAT THIS GUARD PINS. The toast may only promise truncation where truncation
is possible. The backend sends `can_truncate`, measured off the live encoder;
with no clip wired it is TRUE, so the old caveat survives exactly where it is
still warranted -- an unknown encoder is not a safe encoder.

The over-budget notice itself stays. Going over a self-set budget is worth
saying; claiming a consequence that cannot occur is not.
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
JS = os.path.join(ROOT, "web", "js", "uls_token_toast.js")
PY = os.path.join(ROOT, "nodes", "uls_stack_node.py")

_FAILED = []
_PASSED = []


def check(label, cond, detail=""):
    if cond:
        _PASSED.append(label)
    else:
        _FAILED.append(label + ((" -- " + detail) if detail else ""))


js = open(JS, encoding="utf-8").read()
py = open(PY, encoding="utf-8").read()


def _strip_comments(src):
    """Prose must not stand in for behaviour. The block below explains the old
    wording at length; searching the raw file would find the very sentences it
    exists to remove."""
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return "\n".join(l for l in src.split("\n")
                     if not l.lstrip().startswith("//"))


js_code = _strip_comments(js)

# --- P1: the claim is conditional now ---------------------------------------
check("P1 the truncation claim is gated on can_truncate",
      "info.can_truncate" in js_code,
      "without the gate the toast promises truncation on every encoder")
check("P1 the kijai sentence survives only inside the gated branch",
      js_code.count("WanVideoSampler") == 1,
      "it is true for kijai's fixed 512 buffer and nowhere else")

_gate = js_code.split("info.can_truncate")[1]
_true_branch, _false_branch = _gate.split(":", 1)[0], _gate.split(":", 1)[1]
check("P1 truncation is promised in the TRUE branch",
      "truncated" in _true_branch and "WanVideoSampler" in _true_branch)
# Not a bare word search: the false branch legitimately contains "truncated"
# inside "Nothing is truncated". What must be absent is the CLAIM. My first
# draft checked for the word and went red on the correct text -- the fourth
# time this session that a substring stood in for a meaning.
check("P1 the FALSE branch makes no truncation CLAIM",
      "may be silently truncated" not in _false_branch.split("toast(")[0],
      "the promise, not the word, is what must not appear")
check("P1 the FALSE branch states the opposite outright",
      "Nothing is truncated" in _false_branch.split("toast(")[0])
check("P1 the FALSE branch says it is the user's own budget",
      "not a cap" in _false_branch or "your own budget" in _false_branch)

# --- P2: the headline stops saying "limit" when there is none ---------------
check("P2 the headline is conditional too",
      "Over your token budget" in js_code and "Token limit exceeded" in js_code,
      "'limit exceeded' is the wrong word for a budget the user set himself")

# --- P3: the backend actually sends the two fields --------------------------
check("P3 the backend sends can_truncate", '"can_truncate"' in py)
check("P3 the backend sends the encoder name", '"encoder"' in py)

# --- P4: the safe default, driven -------------------------------------------
_fn = [n for n in ast.parse(py).body
       if isinstance(n, ast.FunctionDef) and n.name == "_any_encoder_truncates"]
if not _fn:
    check("P4 _any_encoder_truncates exists", False)
else:
    ns = {"_encoder_facts": lambda c: (c or {}).get("facts", [])}
    exec(compile(ast.Module(body=_fn, type_ignores=[]), "<t>", "exec"), ns)
    f = ns["_any_encoder_truncates"]
    check("P4 no clip -> assume it CAN truncate", f(None) is True,
          "an unknown encoder is not a safe encoder; the caveat must stay")
    check("P4 no readable facts -> assume it can", f({"facts": []}) is True)
    check("P4 a capless encoder -> False",
          f({"facts": [{"cap": None}]}) is False)
    check("P4 a capped encoder -> True",
          f({"facts": [{"cap": 512}]}) is True)
    check("P4 one capped among several is enough",
          f({"facts": [{"cap": None}, {"cap": 77}]}) is True,
          "the shortest cap decides what gets cut")

if _FAILED:
    print("[test_v908_toast_truth] FAIL (%d of %d):"
          % (len(_FAILED), len(_FAILED) + len(_PASSED)))
    for x in _FAILED:
        print("   - " + x)
    sys.exit(1)
print("[test_v908_toast_truth] PASS: %d promises -- the toast promises "
      "truncation only where an encoder can actually truncate." % len(_PASSED))
