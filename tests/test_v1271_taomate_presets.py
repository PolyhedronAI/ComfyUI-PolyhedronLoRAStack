"""Guard v402 (internal v1271, public cut) -- TaoMate-H3 3-step in Sigma List and the Sampler's Sched check.

  presets taomate_3step_h3 (1, 0.961165, 0.853333, 0) and taomate_4step_h3 (1, 0.970874, 0.907249, 0.64, 0) with
  shift 1.0 (final grids); a file named *taomate* is recognised; plan() accepts the 3- AND the 4-step grid without a
  note, notes another grid, and refuses a run without SIGMAS naming the preset.

Script-style: exit 0 = pass.
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = os.path.join(ROOT, "nodes")
FAILS = []
if NODES not in sys.path:
    sys.path.insert(0, NODES)


def _fail(msg):
    FAILS.append(msg)
    print("  FAIL  " + msg)


def _ok(msg):
    print("  ok    " + msg)


def _load(name, file):
    spec = importlib.util.spec_from_file_location(name, os.path.join(NODES, file))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


print("[test_v1271_taomate_presets] TaoMate")
# ------------------------------------------------------------------ T5
print("TaoMate")
WS = _load("wan_sigma_schedule_v1271", "wan_sigma_schedule.py")
SL = _load("uls_sched_loras_v1271", "uls_sched_loras.py")
p3 = WS.SIGMA_PRESETS.get("taomate_3step_h3")
p4 = WS.SIGMA_PRESETS.get("taomate_4step_h3")
if not p3 or tuple(p3[0]) != (1.0, 0.961165, 0.853333, 0.0) or p3[1] != 1.0:
    _fail("taomate_3step_h3 %r" % (p3,))
elif not p4 or tuple(p4[0]) != (1.0, 0.970874, 0.907249, 0.640000, 0.0) or p4[1] != 1.0:
    _fail("taomate_4step_h3 %r" % (p4,))
else:
    _ok("presets 3-step and 4-step, final grids, shift 1.0")
found = SL.recognise([("Minimax_H3/minimax_h3_taomate_fl2va_3step_avg_rank_19_bf16.safetensors", 1.0)])
rec = SL.record(found)
pg = lambda n: WS.SIGMA_PRESETS[n][0]  # noqa: E731
try:
    n3 = SL.plan(rec, list(p3[0]), "euler", 1.0, pg)
    n4 = SL.plan(rec, list(p4[0]), "euler", 1.0, pg)
    nx = SL.plan(rec, [1.0, 0.5, 0.0], "euler", 1.0, pg)
except Exception as e:
    n3 = n4 = nx = None
    _fail("plan raised with SIGMAS connected: %r" % e)
try:
    SL.plan(rec, None, "euler", 1.0, pg)
    refused = ""
except ValueError as e:
    refused = str(e)
if not found or found[0][0]["preset"] != "taomate_3step_h3":
    _fail("a *taomate* file is not recognised: %r" % (found,))
elif n3 is not None and (n3 or n4 or not nx):
    _fail("notes 3-step %r, 4-step %r, other %r" % (n3, n4, nx))
elif "taomate_3step_h3" not in refused:
    _fail("a run without SIGMAS is not refused by name: %r" % refused)
else:
    _ok("*taomate* recognised; 3- and 4-step grid accepted quietly, another grid noted, no SIGMAS refused by name")

if FAILS:
    print("[test_v1271_taomate_presets] FAIL (%d)" % len(FAILS))
    sys.exit(1)
print("[test_v1271_taomate_presets] PASS -- TaoMate on its grids")
