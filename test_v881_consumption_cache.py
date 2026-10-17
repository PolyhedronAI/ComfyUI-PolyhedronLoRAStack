"""Guard -- a consumption gate must not hang on a stale cache (v881).

THE WOUND, from Frank's field log (26.08.), the second ten-minute run lost in a
row. The Power Upscale died in CORE:

    comfy/sampler_helpers.py:72   for c in cond:
    TypeError: 'NoneType' object is not iterable

A CONDITIONING input was None -- and the run's log carried NO CLIP Encode line
at all. The node had not executed.

THE MECHANISM, measured against Core's own source
(comfy_execution/caching.py::CacheKeySetInputSignature.get_node_signature):

    signature = [class_type, IS_CHANGED] + that node's OWN inputs
    (links recorded as ("ANCESTOR", index, socket) -- INCOMING edges only)

Nothing about who reads a node's OUTPUTS is in the key. `slot_consumed()`
reads the running graph through the hidden PROMPT input, so its ANSWER can
change while the signature does not: wire the conditioning output somewhere
new, and the node is served from cache -- handing out the None it produced
back when nobody read that slot.

PROMISES

  P1  Every node whose OUTPUT depends on who CONSUMES it declares IS_CHANGED,
      and that value tracks the consumption state. Checked for every seat of
      the probe, so a fourth seat added later is caught.
  P2  IS_CHANGED is DRIVEN: different graphs must give different answers, the
      same graph the same answer.
  P3  It never raises. Core turns an exception into NaN plus a warning, which
      would silently disable caching for that node forever.
  P4  The Power Upscale refuses a None CONDITIONING BEFORE any work, naming
      which input and where None comes from.
  P5  The refusal runs early -- after the joint-model check, before the pixel
      pass. A late refusal is what cost the ten minutes, twice.
"""

import ast
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

NAME = "v881"
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)


def _src(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


def _lift(rel, names):
    """Run selected top-level functions out of a node module without comfy."""
    tree = ast.parse(_src(rel))
    keep = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.Assign))
            and (getattr(n, "name", None) in names
                 or (isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id in names
                             for t in n.targets)))]
    ns = {"logging": __import__("logging"), "json": __import__("json")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<lift>", "exec"), ns)
    return ns


# ---------------------------------------------------------------------------
# P1 -- every seat of the probe declares IS_CHANGED
#
# Found by SEARCHING for the probe, not from a hand-kept list: a fourth seat
# added later must be caught by this guard, not by the next field crash.
# ---------------------------------------------------------------------------
SEATS = []
for fname in sorted(os.listdir(os.path.join(ROOT, "nodes"))):
    if not fname.endswith(".py"):
        continue
    src = _src(os.path.join("nodes", fname))
    if '"prompt": "PROMPT"' not in src:
        continue
    tree = ast.parse(src)
    # a probe is a module-level function that walks prompt.values()
    probes = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        body = ast.dump(node)
        if "prompt" in [a.arg for a in node.args.args] and \
                "values" in body and "unique_id" in body:
            probes.append(node.name)
    if probes:
        SEATS.append((fname, probes, tree))

_need(len(SEATS) >= 2,
      "%s: P1 expected at least two seats of the consumption probe, found %d "
      "(%s) -- if the probe moved, this guard must move with it"
      % (NAME, len(SEATS), [s[0] for s in SEATS]))

for fname, probes, tree in SEATS:
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    # which classes actually CALL a probe?
    users = []
    for cls in classes:
        calls = {sub.func.id for sub in ast.walk(cls)
                 if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)}
        if calls & set(probes):
            users.append(cls)
    for cls in users:
        methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        _need("IS_CHANGED" in methods,
              "%s: P1 %s.%s calls the consumption probe but declares no "
              "IS_CHANGED. Core's cache key covers only a node's OWN inputs, "
              "so a rewire of its OUTPUT leaves the key untouched and the "
              "node is served a stale answer" % (NAME, fname, cls.name))
        if "IS_CHANGED" in methods:
            ic = next(n for n in cls.body
                      if isinstance(n, ast.FunctionDef) and n.name == "IS_CHANGED")
            called = {sub.func.id for sub in ast.walk(ic)
                      if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)}
            _need(called & set(probes),
                  "%s: P1 %s.%s.IS_CHANGED does not consult the consumption "
                  "probe -- then it does not track the thing that changes the "
                  "answer" % (NAME, fname, cls.name))


# ---------------------------------------------------------------------------
# P2 / P3 -- IS_CHANGED is DRIVEN against real graph shapes
# ---------------------------------------------------------------------------
cte = _lift(os.path.join("nodes", "ph_clip_encode.py"),
            {"slot_consumed", "POS_SLOT", "NEG_SLOT"})
slot_consumed = cte["slot_consumed"]
POS, NEG = cte["POS_SLOT"], cte["NEG_SLOT"]


def _is_changed(prompt, uid):
    """The exact expression from the module, lifted rather than re-typed."""
    tree = ast.parse(_src(os.path.join("nodes", "ph_clip_encode.py")))
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef):
            for n in cls.body:
                if isinstance(n, ast.FunctionDef) and n.name == "IS_CHANGED":
                    ns = dict(cte)
                    # The @classmethod decorator has to come OFF: lifted out of
                    # its class, a classmethod object is not callable, and my
                    # first draft reported that as a fault in the NODE. A probe
                    # that cannot run is not a red test, it is a blind one.
                    lifted = ast.FunctionDef(
                        name=n.name, args=n.args, body=n.body,
                        decorator_list=[], returns=None, type_comment=None,
                        type_params=[])
                    fn = ast.Module(body=[lifted], type_ignores=[])
                    ast.fix_missing_locations(fn)
                    exec(compile(fn, "<ic>", "exec"), ns)
                    # cls is bound by the classmethod machinery in the real
                    # node; here it is passed explicitly.
                    return ns["IS_CHANGED"](None, prompt=prompt, unique_id=uid)
    return None


NOBODY = {"9": {"inputs": {"text": "hello"}}}
READS_POS = {"9": {"inputs": {"text": "hello"}},
             "12": {"inputs": {"positive": ["7", POS]}}}
READS_NEG = {"9": {"inputs": {"text": "hello"}},
             "12": {"inputs": {"negative": ["7", NEG]}}}
READS_BOTH = {"9": {"inputs": {"text": "hello"}},
              "12": {"inputs": {"positive": ["7", POS], "negative": ["7", NEG]}}}

vals = {}
for tag, graph in (("nobody", NOBODY), ("pos", READS_POS),
                   ("neg", READS_NEG), ("both", READS_BOTH)):
    try:
        vals[tag] = _is_changed(graph, "7")
    except Exception as exc:                       # noqa: BLE001
        _fails.append("%s: P3 IS_CHANGED raised on the %s graph: %r. Core "
                      "turns that into NaN plus a warning, which disables "
                      "caching for the node from then on" % (NAME, tag, exc))

if len(vals) == 4:
    _need(len(set(vals.values())) == 4,
          "%s: P2 the four wiring states must give four different answers, "
          "got %r. Any collision is a rewire the cache cannot see"
          % (NAME, vals))
    _need(vals["nobody"] != vals["both"],
          "%s: P2 'nothing reads it' and 'both read it' collide -- that IS "
          "the field crash" % NAME)
    # stable for the same graph
    _need(_is_changed(READS_BOTH, "7") == vals["both"],
          "%s: P2 the same graph gives different answers on two calls -- the "
          "node would re-run forever" % NAME)

# P3 -- the unanswerable cases must not raise either
for tag, args in (("no prompt", (None, "7")), ("no uid", ({}, None)),
                  ("junk prompt", ({"x": "not a dict"}, "7")),
                  ("junk inputs", ({"x": {"inputs": None}}, "7"))):
    try:
        _is_changed(*args)
    except Exception as exc:                       # noqa: BLE001
        _fails.append("%s: P3 IS_CHANGED raised on %s: %r" % (NAME, tag, exc))


# ---------------------------------------------------------------------------
# P4 / P5 -- the Power Upscale refuses None, early
# ---------------------------------------------------------------------------
PU = os.path.join("nodes", "ph_power_upscale.py")
pu_tree = ast.parse(_src(PU))
pu = _lift(PU, {"_reject_none_conditioning"})
reject_none = pu["_reject_none_conditioning"]

GOOD = [["cond", {}]]
for tag, args, want in (("both good", (GOOD, GOOD), None),
                        ("positive None", (None, GOOD), "positive"),
                        ("negative None", (GOOD, None), "negative"),
                        ("both None", (None, None), "positive")):
    try:
        reject_none(*args)
        got = None
    except ValueError as exc:
        got = str(exc)
    if want is None:
        _need(got is None,
              "%s: P4 %s must pass, got: %s" % (NAME, tag, got))
    else:
        _need(got is not None,
              "%s: P4 %s must be REFUSED -- it is the field crash" % (NAME, tag))
        if got:
            _need("'%s'" % want in got,
                  "%s: P4 %s -- the message must name WHICH input, got: %s"
                  % (NAME, tag, got[:100]))
            _need("CLIP Text Encode" in got,
                  "%s: P4 the message must name where None comes from -- our "
                  "own v876 gate. Without it the user hunts a core bug"
                  % NAME)

# a non-iterable that is not None must also be caught
try:
    reject_none(42, GOOD)
    _fails.append("%s: P4 a non-iterable conditioning slipped through -- the "
                  "sampler iterates it on its first line" % NAME)
except ValueError:
    pass

# P5 -- ordering inside upscale()
_up = next((n for n in ast.walk(pu_tree)
            if isinstance(n, ast.FunctionDef) and n.name == "upscale"), None)
_need(_up is not None, "%s: P5 upscale() is gone" % NAME)
if _up is not None:
    calls = []
    for i, stmt in enumerate(_up.body):
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                calls.append((i, sub.func.id))
    names = [c[1] for c in calls]
    _need("_reject_none_conditioning" in names,
          "%s: P5 upscale() never checks the conditioning" % NAME)
    if "_reject_none_conditioning" in names:
        pos = min(i for i, n in calls if n == "_reject_none_conditioning")
        for later in ("_resolve_input", "_refine_tiles", "_apply_sigma_shift"):
            if later in names:
                pos_later = min(i for i, n in calls if n == later)
                _need(pos < pos_later,
                      "%s: P5 the None check runs AFTER %s. Ten minutes of "
                      "wall clock is what a late refusal costs -- twice now"
                      % (NAME, later))


if _fails:
    print("FAIL %s" % NAME)
    for f in _fails:
        print("  - %s" % f)
    sys.exit(1)
print("ok %s -- consumption gates declare IS_CHANGED (%d seat(s) checked), "
      "None conditioning refused early and by name"
      % (NAME, len(SEATS)))
