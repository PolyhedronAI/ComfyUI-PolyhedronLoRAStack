#!/usr/bin/env python3
"""v972 -- named sigma grids in the Sigma List.

A published grid is a MEASUREMENT somebody else made. Typing it by hand is
exactly where a digit gets lost, so the grids live in a table and the console
says which source a run used.

The reason this is a table in the Sigma List and NOT a widget on the Sampler:
the Sampler hangs in every saved workflow, so an appended widget there moves a
baseline that much depends on -- and it would give the Sampler knowledge about
one particular third-party recipe. The Sigma List is a grid node; a table of
grids belongs to it, and it knows nothing about any node pack.

  P1  the preset widget is LAST -- the three v971 widgets keep their index
  P2  every preset obeys the contract the node enforces on typed input
  P3  'custom' is the default and leaves the text field in charge
  P4  a named preset WINS over the text field, and says so
  P5  each preset's RAW grid plus its shift reproduces the list its authors
      published, to 1e-6 -- HyperFlow's (MiniMax-H3, shift 12) and lightx2v's
      (Wan 2.2 Lightning, shift 5). That is the whole point of shipping RAW
      numbers, and it pins the shift in the table as tightly as the grid
  P6  an unknown preset name falls back to custom instead of raising: a
      workflow saved against a later build must still run
  P7  RE-ARGUED in v973. v972 held that the table's shift was documentation
      only and the widget always decided. That was wrong in the field: on the
      day it shipped, a preset ran with the widget's shift still at 1.0 and
      produced the RAW curve while looking applied. Half a preset is a trap.
      The promise now is the stronger one -- a preset carries a WHOLE recipe
      and overrides the widgets, and the console names every value replaced
  P8  the V3 schema mirrors the legacy widget order, preset included
  P9  no preset is a k-diffusion range pretending to be normalised: the ones
      above 1.0 are the ones whose note says to leave shift at 1.0

MUTATION PROBE (run 20.09.2026, each mutation applied and reverted):
  1. preset moved before sigmas_text in INPUT_TYPES      -> P1 caught
  2. a digit changed in the hyperflow grid               -> P5 caught
  3. preset_grid returns the grid for 'custom'           -> P3 caught
  4. compute ignores preset and always reads the text    -> P4 caught
  5. preset_grid raises on an unknown name               -> P6 caught
  6. compute ignores the preset shift and uses the widget -> P7 caught
  7. an ays grid made non-decreasing                     -> P2 caught
  9. the wan22 preset's shift changed to 3.0             -> P5 caught (v975)
  8. preset appended after 'preset' in the V3 schema only -> P8 caught
"""
import os, sys, re, math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

failures = []
ran = []


def check(cond, msg):
    ran.append(msg)
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


from nodes.wan_sigma_schedule import (ULSSigmaList, SIGMA_PRESETS,
                                      SIGMA_PRESET_NAMES, PRESET_CUSTOM,
                                      preset_grid, validate_sigma_list,
                                      shift_sigma_list)

N = ULSSigmaList()
W = list(ULSSigmaList.INPUT_TYPES()["required"].keys())

# --- P1 the append law -----------------------------------------------------
check(W[:3] == ["sigmas_text", "shift", "enforce_terminal_zero"],
      "P1 the three v971 widgets keep their original index")
check(W[-1] == "preset" and len(W) == 4,
      "P1 'preset' is appended LAST, so no saved workflow renumbers")

# --- P2 every preset obeys the contract ------------------------------------
for name in SIGMA_PRESET_NAMES:
    grid = preset_grid(name)
    if grid is None:
        continue
    try:
        validate_sigma_list(list(grid))
        ok = True
    except Exception:
        ok = False
    check(ok, "P2 preset '%s' is finite, non-negative and strictly decreasing" % name)
    check(len(grid) >= 2, "P2 preset '%s' has at least two points" % name)

# --- P3 custom is the default and yields to the text -----------------------
check(ULSSigmaList.INPUT_TYPES()["required"]["preset"][1]["default"] == PRESET_CUSTOM,
      "P3 'custom' is the default preset")
check(preset_grid(PRESET_CUSTOM) is None, "P3 'custom' carries no grid of its own")
sig, steps = N.compute("1.0, 0.4, 0.0", 1.0, True, PRESET_CUSTOM)
check([round(float(x), 6) for x in sig] == [1.0, 0.4, 0.0],
      "P3 under 'custom' the text field decides")

# --- P4 a named preset wins ------------------------------------------------
sig, steps = N.compute("1.0, 0.4, 0.0", 1.0, True, "ays_sdxl_10step")
# the AYS grids end at 0.029, not zero, so enforce_terminal_zero appends one:
# 11 published points -> 12, i.e. 11 steps. That is the node doing its job.
check(abs(float(sig[0]) - 14.615) < 1e-4 and len(sig) == 12,
      "P4 a named preset overrides the text field entirely")
check(steps == 11, "P4 the step count follows the preset, not the text")

# --- P5 the published grid, reproduced --------------------------------------
PUBLISHED = [1.0, 0.993910, 0.984287, 0.966064, 0.923077,
             0.834942, 0.696852, 0.468753, 0.0]
sig, steps = N.compute("", 12.0, True, "hyperflow_8step_h3_raw")
check(len(sig) == 9 and steps == 8, "P5 the HyperFlow preset is a nine-point, 8-step grid")
check(all(abs(float(a) - b) < 1e-6 for a, b in zip(sig, PUBLISHED)),
      "P5 raw grid + shift 12.0 reproduces the published shifted grid to 1e-6")

# --- P5b the OTHER published grid, reproduced (added v975) -----------------
# lightx2v ship this list hardcoded in their Wan 2.2 workflow. Our entry holds
# the RAW curve plus their shift; this check proves the two agree, so a wrong
# shift in the table cannot pass. (The v975 mutation probe slipped exactly
# there before this existed.)
LIGHTX2V = [1.0, 0.9375001, 0.8333333, 0.625, 0.0]
sig, steps = N.compute("", 1.0, True, "wan22_lightning_4step_raw")
check(steps == 4 and len(sig) == 5,
      "P5 the Wan 2.2 Lightning preset is a five-point, 4-step grid")
check(all(abs(float(a) - b) < 1e-6 for a, b in zip(sig, LIGHTX2V)),
      "P5 raw grid + shift 5.0 reproduces lightx2v's published list to 1e-6")

# --- P6 an unknown name degrades, never raises -----------------------------
check(preset_grid("a_preset_from_a_later_build") is None,
      "P6 an unknown preset name falls back to custom")
sig, steps = N.compute("1.0, 0.4, 0.0", 1.0, True, "a_preset_from_a_later_build")
check([round(float(x), 6) for x in sig] == [1.0, 0.4, 0.0],
      "P6 a workflow with an unknown preset still runs, on its text field")

# --- P7 a preset overrides the widgets (re-argued in v973) -----------------
PUB = [1.0, 0.993910, 0.984287, 0.966064, 0.923077, 0.834942, 0.696852, 0.468753, 0.0]
for widget_shift in (1.0, 3.0, 12.0, 0.5):
    sig, steps = N.compute("9, 8, 7", widget_shift, False, "hyperflow_8step_h3_raw")
    check(all(abs(float(a) - b) < 1e-6 for a, b in zip(sig, PUB)),
          "P7 preset wins whatever the shift widget says (%s) -- the field trap is shut"
          % widget_shift)
check(SIGMA_PRESETS["hyperflow_8step_h3_raw"][1] == 12.0,
      "P7 the preset's shift is recorded in the table")
check(SIGMA_PRESETS["hyperflow_8step_h3_raw"][2] is True,
      "P7 the preset's terminal-zero rule is recorded in the table")
check(isinstance(SIGMA_PRESETS["hyperflow_8step_h3_raw"][3], str)
      and SIGMA_PRESETS["hyperflow_8step_h3_raw"][3].strip() != "",
      "P7 every preset carries a note naming its recipe")
# the terminal-zero rule is part of the recipe too -- an AYS grid ends at
# 0.029, so the preset must append the zero even when the widget says not to.
# (Found by the v973 mutation probe: this was the one mutation that slipped.)
sig, steps = N.compute("1.0, 0.4, 0.0", 1.0, False, "ays_sdxl_10step")
check(len(sig) == 12 and abs(float(sig[-1])) < 1e-9,
      "P7 the preset's terminal-zero rule overrides the widget too")
sig_c, _ = N.compute("1.0, 0.8, 0.5", 1.0, False, PRESET_CUSTOM)
check(len(sig_c) == 3,
      "P7 under 'custom' the widget still rules the terminal zero")

# --- P8 the two descriptions of one node -----------------------------------
src = open(os.path.join(ROOT, "nodes", "wan_sigma_list_v3.py"), encoding="utf-8").read()
v3 = re.findall(r'io\.\w+\.Input\(\s*\n?\s*"(\w+)"', src)
check(v3 == W, "P8 the V3 schema mirrors the legacy widget order exactly: %s" % v3)

# --- P9 range and note agree -----------------------------------------------
for name in SIGMA_PRESET_NAMES:
    entry = SIGMA_PRESETS[name]
    grid, sug, zero, note = entry
    if grid is None:
        continue
    above_one = max(grid) > 1.0
    if above_one:
        check(abs(sug - 1.0) < 1e-9,
              "P9 '%s' is a k-diffusion range, so its suggested shift is 1.0" % name)
        try:
            shift_sigma_list(list(grid), 12.0)
            refused = False
        except ValueError:
            refused = True
        check(refused, "P9 '%s' would be REFUSED if shifted -- the guard rails hold" % name)
    else:
        check(sug >= 1.0, "P9 '%s' is normalised and its shift is usable" % name)

print()
if failures:
    print("v972 sigma presets: FAIL (%d)" % len(failures))
    for f in failures:
        print("   - " + f)
    sys.exit(1)
print("%d checks, 0 failed" % len(ran))
print("v972 sigma presets: PASS")
