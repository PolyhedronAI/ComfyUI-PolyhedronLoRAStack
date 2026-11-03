"""Guard v1268 -- Switch Inverse behind a blocked branch (field 03.11.).

Field: Upscale switch (Switch Inverse) on 1, its out_2 feeds Power Upscale,
whose result feeds the Interpolate switch (Switch Inverse) with consumers on
out_1 AND out_2. The Interpolate switch's input arrives blocked, so Core does
not run it: it hands the consumers len(obj.RETURN_TYPES) blockers. The class
declares a 1-tuple (its outputs grow in the frontend) -> the consumer on out_2
read past the end: "IndexError: list index out of range" in cache_update,
after everything had been saved.

  S1  static: the INSTANCE carries the long RETURN_TYPES (len = INV_MAX_OUT =
      the select cap), the CLASS keeps its 1-tuple (object_info unchanged);
      any index still resolves to the wildcard.
  S2  REAL CORE (core_master's own PromptExecutor, in process, CPU): the field
      chain -- A(select 1) out_1 -> sink, out_2 -> invert -> B; B out_1, out_2
      -> sinks -- runs to success; A's out_1 sink got the picture, B's sinks
      got nothing. A(select 2): B runs, B(select 1) feeds out_1 only.
  S3  the mirror: with the class 1-tuple put back on the instance the same
      chain fails with the field's IndexError (the guard sees the bug).

Script-style: exit 0 = pass.
"""
import importlib.util
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = os.path.join(ROOT, "nodes")
FAILS = []


def _fail(msg):
    FAILS.append(msg)
    print("  FAIL  " + msg)


def _ok(msg):
    print("  ok    " + msg)


def _core_root():
    for c in (os.environ.get("PLS_CORE_MASTER"), "/home/claude/core_master",
              os.environ.get("PLS_CORE_ROOT")):
        if c and os.path.isfile(os.path.join(c, "execution.py")) \
                and os.path.isfile(os.path.join(c, "comfy_execution", "graph.py")):
            return c
    return None


CORE = _core_root()
if CORE and CORE not in sys.path:
    sys.path.insert(0, CORE)

HAVE_CORE = False
if CORE:
    try:
        import comfy.cli_args
        comfy.cli_args.args.cpu = True
        import folder_paths
        import execution
        import nodes as core_nodes
        HAVE_CORE = True
    except Exception as e:
        print("  note  Core not importable (%s)" % e)

_VENV = "/home/claude/venv_master/bin/python3"
if not HAVE_CORE and os.environ.get("PLS_V1268_REEXEC") != "1" and os.path.isfile(_VENV) \
        and os.path.realpath(sys.prefix) != os.path.realpath(os.path.dirname(os.path.dirname(_VENV))):
    print("  note  re-running in %s, where core_master imports" % _VENV)
    sys.stdout.flush()
    os.execve(_VENV, [_VENV, os.path.abspath(__file__)], dict(os.environ, PLS_V1268_REEXEC="1"))

print("[test_v1268_switch_inv_blocked] Switch Inverse behind a blocked branch")
spec = importlib.util.spec_from_file_location("ph_switch_v1268", os.path.join(NODES, "ph_switch.py"))
SW = importlib.util.module_from_spec(spec)
spec.loader.exec_module(SW)
Inv = SW.ULSAnySwitchInv

# ------------------------------------------------------------------ S1
print("S1 instance vs class RETURN_TYPES")
inst = Inv()
if len(Inv.RETURN_TYPES) != 1 or list(Inv.RETURN_TYPES) != [SW._any]:
    _fail("S1 the CLASS must keep its 1-tuple (object_info): %r" % (Inv.RETURN_TYPES,))
elif len(inst.RETURN_TYPES) != SW.INV_MAX_OUT:
    _fail("S1 instance RETURN_TYPES len %d, want INV_MAX_OUT %d" % (len(inst.RETURN_TYPES), SW.INV_MAX_OUT))
elif inst.RETURN_TYPES[5] is not SW._any or inst.RETURN_TYPES[SW.INV_MAX_OUT + 3] is not SW._any:
    _fail("S1 an output index does not resolve to the wildcard")
elif Inv.INPUT_TYPES()["required"]["select"][1]["max"] != SW.INV_MAX_OUT:
    _fail("S1 the select cap and INV_MAX_OUT disagree")
else:
    _ok("class 1-tuple (object_info), instance len %d = select cap, any index -> '*'" % SW.INV_MAX_OUT)

# ------------------------------------------------------------------ S2/S3
if not HAVE_CORE:
    if os.environ.get("PLS_V1268_REEXEC") == "1":
        _fail("S2 no Core importable even in the venv -- the real-run sections cannot guard")
    else:
        print("  SKIP  S2/S3 (no Core reachable)")
else:
    import torch
    folder_paths.set_output_directory(tempfile.mkdtemp())
    folder_paths.set_temp_directory(tempfile.mkdtemp())
    SEEN = {}

    class _Src:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {}}
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "go"
        CATEGORY = "test"

        def go(self):
            return (torch.zeros(1, 8, 8, 3),)

    class _Inv:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",)}}
        RETURN_TYPES = ("IMAGE",)
        FUNCTION = "go"
        CATEGORY = "test"

        def go(self, image):
            return (1.0 - image,)

    class _Sink:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {"image": ("IMAGE",), "tag": ("STRING", {"default": ""})}}
        RETURN_TYPES = ()
        FUNCTION = "go"
        OUTPUT_NODE = True
        CATEGORY = "test"

        def go(self, image, tag):
            SEEN[tag] = float(image.mean())
            return ()

    core_nodes.NODE_CLASS_MAPPINGS.update({"ULSAnySwitchInv": Inv, "T1268Src": _Src,
                                           "T1268Inv": _Inv, "T1268Sink": _Sink})

    class _Server:
        client_id = None
        last_node_id = None
        last_prompt_id = None

        def send_sync(self, *a, **k):
            pass

        def queue_updated(self, *a, **k):
            pass

    def run(sel_a, sel_b, tag):
        SEEN.clear()
        p = {"1": {"class_type": "T1268Src", "inputs": {}},
             "2": {"class_type": "ULSAnySwitchInv", "inputs": {"select": sel_a, "input": ["1", 0]}},
             "3": {"class_type": "T1268Sink", "inputs": {"image": ["2", 0], "tag": "a1"}},
             "4": {"class_type": "T1268Inv", "inputs": {"image": ["2", 1]}},
             "5": {"class_type": "ULSAnySwitchInv", "inputs": {"select": sel_b, "input": ["4", 0]}},
             "6": {"class_type": "T1268Sink", "inputs": {"image": ["5", 0], "tag": "b1"}},
             "7": {"class_type": "T1268Sink", "inputs": {"image": ["5", 1], "tag": "b2"}}}
        try:
            ex = execution.PromptExecutor(_Server(), cache_type=False,
                                          cache_args={"lru": 0, "ram": 0.0, "ram_inactive": 0.0})
        except TypeError:   # an older Core without cache_args
            ex = execution.PromptExecutor(_Server())
        msgs = []
        ex.add_message = lambda event, data, broadcast: msgs.append((event, data))
        ex.execute(p, tag, extra_data={}, execute_outputs=["3", "6", "7"])
        err = [d for e, d in msgs if e == "execution_error"]
        return ex.success, err, dict(SEEN)

    print("S2 the field chain on Core's own executor")
    ok, err, seen = run(1, 1, "s2a")
    if not ok:
        _fail("S2 A=1: the chain failed: %s" % [(e.get("node_id"), e.get("exception_type"), e.get("exception_message")) for e in err])
    elif seen != {"a1": 0.0}:
        _fail("S2 A=1: wrong sinks fed %r (want only a1)" % seen)
    else:
        _ok("A=1 (B blocked, consumers on out_1 AND out_2): success, only A's out_1 fed")
    ok, err, seen = run(2, 1, "s2b")
    if not ok or seen != {"b1": 1.0}:
        _fail("S2 A=2 B=1: ok=%s sinks %r err %r (want only b1 = inverted)" % (ok, seen, err[:1]))
    else:
        _ok("A=2 B=1: B runs, only its out_1 fed (inverted picture)")
    ok, err, seen = run(2, 2, "s2c")
    if not ok or seen != {"b2": 1.0}:
        _fail("S2 A=2 B=2: ok=%s sinks %r" % (ok, seen))
    else:
        _ok("A=2 B=2: only B's out_2 fed")

    print("S3 mirror: the class 1-tuple on the instance brings the field error back")
    real_init = Inv.__init__
    Inv.__init__ = lambda self: None
    try:
        ok, err, seen = run(1, 1, "s3")
    finally:
        Inv.__init__ = real_init
    if ok or not any(e.get("exception_type", "").endswith("IndexError") for e in err):
        _fail("S3 without the instance length the chain should fail with IndexError -- ok=%s err=%r"
              % (ok, err[:1]))
    else:
        _ok("1-tuple -> IndexError at node %s (the field's error) -- the guard sees the bug"
            % err[0].get("node_id"))

if FAILS:
    print("[test_v1268_switch_inv_blocked] FAIL (%d)" % len(FAILS))
    sys.exit(1)
print("[test_v1268_switch_inv_blocked] PASS -- a blocked Switch Inverse blocks every output a consumer reads; "
      "object_info unchanged")
