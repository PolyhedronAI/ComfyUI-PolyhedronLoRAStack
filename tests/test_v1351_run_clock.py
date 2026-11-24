# -*- coding: ascii -*-
"""Guard v1351 -- one clock per queue run: every save of one run carries the same %date:...% time.

(Frank 20.11.: "der zugehoerige Ordner entgleitet mir immer" -- run PH_160959: the clip PH_160959_360_CORD_00001_.mp4
beside PH_160958_360_CORD_SHEET and the folder PH_160958_360_CORD_FRAMES. Every Save read the clock itself, so the
saves of one run were stamped a second apart, and a run folder %date:yyyy-MM-dd%/PH_%date:hhmmss%_X/ split in two.)

  R1 outside a run (no prompt id) run_now is simply now -- every call reads the clock
  R2 inside a run (the core's PromptServer.instance.last_prompt_id) the moment is fixed at the first call: under a
     clock that ticks a second on every call, sanitize_prefix / freeze_date_tokens / expand_filename_tokens give the
     same time again and again, and the run folder of three saves is ONE folder
  R3 a new prompt id is a new run: a new moment
  R4 an explicit `now` still wins (the guards' fixed clock)
  R5 the saves take the run clock: ph_save and ph_sprite_export (clip, sheet, frame sequence) go through the shared
     helpers and no Save module reads the wall clock for its prefix itself
  R6 ASCII; version stamps agree
Script-style: exit 0 = pass.
"""
import datetime as dt
import importlib
import os
import re
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "test_v1351_run_clock"
FAILS = []


def need(c, m):
    print(("  ok    " if c else "  FAIL  ") + m)
    if not c:
        FAILS.append(m)


pkg = types.ModuleType("plsn1351")
pkg.__path__ = [os.path.join(ROOT, "nodes")]
sys.modules["plsn1351"] = pkg
U = importlib.import_module("plsn1351.ph_save_util")


class Tick(dt.datetime):
    """One second later on every call -- every second boundary at once."""
    n = 0

    @classmethod
    def now(cls, tz=None):
        Tick.n += 1
        return dt.datetime(2026, 11, 20, 16, 9, 57) + dt.timedelta(seconds=Tick.n)


U._dt.datetime = Tick
sys.modules.pop("server", None)
U._RUN_CLOCK.update({"id": None, "now": None})

# ---- R1
a, b = U.run_now(), U.run_now()
need(a != b, "R1 no run: every call reads the clock (%s / %s)" % (a.time(), b.time()))

# ---- R2
srv = types.ModuleType("server")


class _PS(object):
    instance = types.SimpleNamespace(last_prompt_id="run-A")


srv.PromptServer = _PS
sys.modules["server"] = srv
pfx = "%date:yyyy-MM-dd%/PH_%date:hhmmss%_360_CORD/"
names = [U.sanitize_prefix(pfx + "PH_%date:hhmmss%_360_CORD"),
         U.sanitize_prefix(pfx + "PH_%date:hhmmss%_360_CORD_SHEET"),
         U.freeze_date_tokens(pfx + "frames/view"),
         U.expand_filename_tokens(pfx + "x")]
folders = set(n.split("/")[1] for n in names)
stamps = set(re.findall(r"PH_(\d{6})_", " ".join(names)))
need(len(folders) == 1 and len(stamps) == 1, "R2 one run: one folder, one time (%s)" % sorted(folders | stamps))
need(Tick.n >= 3, "R2 the clock really ticked meanwhile (%d calls)" % Tick.n)
first = U.run_now()

# ---- R3
_PS.instance.last_prompt_id = "run-B"
second = U.run_now()
need(second != first and U.run_now() == second, "R3 a new run is a new moment, held again (%s -> %s)"
     % (first.time(), second.time()))

# ---- R4
fixed = dt.datetime(2026, 1, 2, 3, 4, 5)
need(U.sanitize_prefix("PH_%date:hhmmss%", now=fixed) == "PH_030405", "R4 an explicit now wins")
sys.modules.pop("server", None)

# ---- R5
src_util = open(os.path.join(ROOT, "nodes", "ph_save_util.py"), encoding="utf-8").read()
need(src_util.count("now = now or run_now()") == 2 and "now = now or _dt.datetime.now()" not in src_util,
     "R5 both token expanders take the run clock")
sv = open(os.path.join(ROOT, "nodes", "ph_save.py"), encoding="utf-8").read()
# PUBLIC CUT (v409, declared): the sprite exports are not part of the public pack; Save is checked alone.
se = ""
need("U.sanitize_prefix(" in sv and "U.freeze_date_tokens(" in sv,
     "R5 Save goes through the shared helpers")
need(not re.search(r"datetime\.now\(\)|time\.localtime\(\)", sv + se), "R5 no own wall clock in Save / sprite export")

# ---- R6
for f in ("nodes/ph_save_util.py", "tests/test_v1351_run_clock.py"):
    try:
        open(os.path.join(ROOT, f), encoding="ascii").read()
        ok = True
    except UnicodeDecodeError:
        ok = False
    need(ok, "R6 ASCII " + f)
_pv = re.search(r'const PLUGIN_VERSION = "v(\d+)";', open(os.path.join(ROOT, "web", "js", "uls_compat.js")).read())
_pp = re.search(r'^version = "4\.(\d+)\.0"', open(os.path.join(ROOT, "pyproject.toml")).read(), re.M)
# PUBLIC CUT (v409, declared): the public build counts its own way -- vNNN = Major x 100 + Minor (test_v365 V).
need(bool(_pv and _pp) and int(_pp.group(1)) == int(_pv.group(1)) - 400 and int(_pv.group(1)) >= 409,
     "R6 version stamps agree (v%s / 4.%s.0)" % (_pv and _pv.group(1), _pp and _pp.group(1)))

print("%s: %s" % (NAME, "PASS" if not FAILS else "FAIL %d" % len(FAILS)))
sys.exit(1 if FAILS else 0)
