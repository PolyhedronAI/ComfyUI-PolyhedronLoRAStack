#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1337 -- a model on its OWN sigma grid gets a schedule made of grid points.

Measured against Core 0.45.0 (16.11.2026): MiniMax H3 Image has six distilled heads on a seven-point grid and
Core serves a step with the dt-weighted mean of the heads it spans. 'simple' lands on the grid at 6, 3, 2 and 1
steps only; at 4 steps it lands on 0, 1.5, 3, 4.5, 6 and a refine of 1 step at denoise 0.25 starts at 4.5.

WHAT IS PINNED

  G1  uls_grid_math (pure): the partition (sizes differ by one at most, the longer first), the plan for every
      count 1..6, the denoise start (nearest grid point, a tie goes DOWN, at least one head), no plan for more
      steps than heads are left, on_grid, the heads text; the library imports no torch and is ASCII.
  G2  against the REAL Core 0.45.0 (when /home/claude/core_new or PLS_CORE_NEW imports), real model sampling
      class and real schedulers: 6 / 3 / 2 / 1 steps 'simple' stay with the stock scheduler; 4 and 5 steps and
      'beta' at 6 get the grid schedule; 12 steps is said and left alone; 1 @ 0.16, 2 @ 0.33, 3 @ 0.5 stay
      stock, 1 @ 0.25 snaps to 0.167; a sampler that drops the penultimate sigma never counts as on the grid;
      Core's head rule still reads the way the plan assumes.
  G3  the Sampler's seat without Core: no grid -> None and silent; a wired curve wins; High + Low is left
      alone; the manual slice cuts the grid schedule, an early end is clean unless leftover noise is on.
  G4  Core's head rule, replayed: every plan (all counts, all denoise starts) takes each head exactly once,
      in order -- and the stock 4-step schedule does NOT serve its own steps (the reason for this build).
  G5  wiring: _run calls _grid_prepare after the PDD step and before the High + Low branch; Power Upscale
      asks _grid_sigmas for a stage without a wired curve, right before the tile refine.

Script-style: exit 0 = pass.
"""

import json
import os
import re
import subprocess
import sys

NAME = "v1337"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODES = os.path.join(ROOT, "nodes")
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)
    print(("  ok    " if cond else "  FAIL  ") + msg)


sys.path.insert(0, NODES)
import uls_grid_math as G  # noqa: E402

GRID = [1.0, 0.97, 0.92, 0.84, 0.70, 0.45, 0.0]      # a stand-in: only the index positions matter
SRC_G = open(os.path.join(NODES, "uls_grid_math.py"), encoding="utf-8").read()
SRC_S = open(os.path.join(NODES, "uls_sampler.py"), encoding="utf-8").read()
SRC_P = open(os.path.join(NODES, "ph_power_upscale.py"), encoding="utf-8").read()

# ---- G1 the library -------------------------------------------------------------------
want = {1: (6,), 2: (3, 3), 3: (2, 2, 2), 4: (2, 2, 1, 1), 5: (2, 1, 1, 1, 1), 6: (1,) * 6}
_need(all(G.partition(6, n) == b for n, b in want.items()), "G1 partition of six heads for 1..6 steps, the longer first")
_need(G.partition(6, 7) is None and G.partition(6, 0) is None and G.partition(0, 1) is None,
      "G1 no partition for more steps than heads, none for zero")
_need(G.partition(8, 5) == (2, 2, 2, 1, 1) and G.partition(8, 3) == (3, 3, 2), "G1 the partition is general (eight heads)")
p4 = G.plan(GRID, 4)
_need(p4["sigmas"] == [1.0, 0.92, 0.70, 0.45, 0.0] and p4["start"] == 0 and not p4["snapped"],
      "G1 four steps run 0, 2, 4, 5, 6 -> %r" % (p4["sigmas"],))
_need(G.heads_text(p4) == "1-2, 3-4, 5, 6", "G1 the heads text of four steps")
_need(G.plan(list(reversed(GRID)), 4)["sigmas"] == p4["sigmas"], "G1 an ascending grid (Core's storage) gives the same plan")
_need(all(G.plan(GRID, n)["sigmas"][0] == 1.0 and G.plan(GRID, n)["sigmas"][-1] == 0.0 for n in range(1, 7)),
      "G1 every full plan runs from the first grid point to the last")
_need(G.plan(GRID, 7) is None and G.plan(GRID, 0) is None and G.plan(GRID, 3, 0.0) is None,
      "G1 no plan for 7 steps, for 0 steps, for denoise 0")
d = {(1, .16): ([0.45, 0.0], False), (2, .33): ([0.70, 0.45, 0.0], False), (3, .5): ([0.84, 0.70, 0.45, 0.0], False),
     (1, .25): ([0.45, 0.0], True), (2, .30): ([0.70, 0.45, 0.0], True), (1, .33): ([0.70, 0.0], False), (1, .05): ([0.45, 0.0], True),
     (1, .17): ([0.45, 0.0], False), (2, .34): ([0.70, 0.45, 0.0], False)}
for (n, dn), (sig, snap) in d.items():
    p = G.plan(GRID, n, dn)
    _need(p is not None and p["sigmas"] == sig and p["snapped"] is snap,
          "G1 %d @ %.2f starts on the grid: %r%s" % (n, dn, sig, " (snapped)" if snap else ""))
_need(G.plan(GRID, 2, .16) is None and G.plan(GRID, 3, .33) is None and G.plan(GRID, 2, .25) is None, "G1 more steps than heads are left -> no plan")
_need(abs(G.plan(GRID, 1, .25)["denoise_grid"] - 1 / 6.0) < 1e-9 and G.intervals_left(6, .75) == 4,
      "G1 0.25 snaps to 1/6 and 0.75 to 4/6 (a tie goes down: the gentler refine)")
_need(G.intervals_left(6, .41) == 2 and G.intervals_left(6, .42) == 3 and G.intervals_left(6, 1.0) == 6
      and G.intervals_left(6, .01) == 1, "G1 intervals_left: nearest, at least one, all at denoise 1")
_need(G.on_grid(GRID, [1.0, 0.92, 0.70, 0.0]) and not G.on_grid(GRID, [1.0, 0.945, 0.84, 0.575, 0.0]),
      "G1 on_grid: grid points yes, half way no")
_need(not G.on_grid(GRID, [1.0, 0.70, 0.70, 0.0]) and not G.on_grid(GRID, [1.0]), "G1 on_grid: a repeated point and a lone sigma are not a schedule")
_need(abs(G.index_of(GRID, 0.945) - 1.5) < 1e-6 and G.index_of(GRID, 2.0) == 0.0 and G.index_of(GRID, -1) == 6.0,
      "G1 index_of: half way between points, clamped at the ends")
_need(not re.search(r"^\s*(import|from)\s+(torch|comfy)", SRC_G, re.M), "G1 the library imports neither torch nor comfy")
_need(all(ord(c) < 128 for c in SRC_G), "G1 the library is ASCII")

# ---- the Sampler's functions, lifted from the source ----------------------------------
FNS = ("_sampling_name", "_owns_sigma_grid", "_grid_mod", "_model_grid", "_stock_sigmas", "_grid_sigmas", "_grid_prepare")
LIFT = "\n".join(re.search(r"^def %s\(.*?(?=^\S)" % fn, SRC_S, re.S | re.M).group(0) for fn in FNS)
LIFT += "\nGRID_SAMPLINGS = " + re.search(r"^GRID_SAMPLINGS = (.*)$", SRC_S, re.M).group(1) + "\n"

# ---- G2 the real Core -----------------------------------------------------------------
core_new = os.environ.get("PLS_CORE_NEW", "/home/claude/core_new")
probe = r"""
import sys, json, types, io, contextlib, torch
sys.path.insert(0, %r); sys.path.insert(0, %r)
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options; comfy.options.enable_args_parsing()
import comfy.samplers
import comfy.ldm.minimax.sampling as MS
GRID = %r
ms = MS.MiniMaxH3ImageModelSampling(types.SimpleNamespace(unet_config={"pdd_sigmas": GRID}, sampling_settings={}))
class Model:
    def get_model_object(self, k): return ms
ns = {"comfy": comfy, "torch": torch, "__name__": "lifted"}
exec(%r, ns)
out = {"cls": type(ms).__name__, "runs": {}}
for key, (steps, sampler, sched, dn) in {
        "6s": (6, "euler", "simple", 1.0), "3s": (3, "euler", "simple", 1.0), "2s": (2, "euler", "simple", 1.0),
        "1s": (1, "euler", "simple", 1.0), "4s": (4, "euler", "simple", 1.0), "5s": (5, "euler", "simple", 1.0),
        "4u": (4, "sa_solver", "sgm_uniform", 1.0), "6u": (6, "euler", "sgm_uniform", 1.0),
        "6b": (6, "euler", "beta", 1.0), "6k": (6, "euler", "karras", 1.0), "12s": (12, "euler", "simple", 1.0),
        "1@16": (1, "euler", "simple", .16), "2@33": (2, "euler", "simple", .33), "3@50": (3, "euler", "simple", .5),
        "1@25": (1, "euler", "simple", .25), "2@16": (2, "euler", "simple", .16),
        "6dpm2": (6, "dpm_2", "simple", 1.0)}.items():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = ns["_grid_sigmas"](Model(), steps, sampler, sched, dn)
    out["runs"][key] = [r, buf.getvalue()]
out["stock4"] = ns["_stock_sigmas"](Model(), "euler", "simple", 4, 1.0)
src = open(MS.__file__.replace("sampling.py", "model.py"), encoding="utf-8").read()
out["rule"] = ("i = int((sample_sigmas - sigma).abs().argmin())" in src
               and "start, stop = (min(range(n + 1), key=lambda k: abs(grids[0][k] - float(s))) for s in (sigma, sigma_next))" in src
               and "the step from grid[start] to grid[stop] takes the dt-weighted mean of the heads it spans" in src)
print("JSON" + json.dumps(out))
""" % (core_new, NODES, GRID, LIFT)
r = None
pys = [sys.executable] + [p for p in ("/home/claude/venv_new/bin/python",) if os.path.isfile(p)]
if os.path.isfile(os.path.join(core_new, "comfy", "ldm", "minimax", "sampling.py")):
    for py in pys:
        r = subprocess.run([py, "-c", probe], capture_output=True, text=True, timeout=600, cwd=core_new)
        if "JSON" in r.stdout:
            break
STOCK4 = None


def r4(v):
    """A float32 grid read back as a list: compare at four places."""
    return None if v is None else [round(float(x), 4) for x in v]


if r is not None and "JSON" in r.stdout:
    C = json.loads(r.stdout.split("JSON", 1)[1].strip().splitlines()[0])
    R = C["runs"]
    _need(C["cls"] == "MiniMaxH3ImageModelSampling", "G2 Core's own model sampling class drives the probe")
    for k in ("6s", "3s", "2s", "1s", "6u"):
        _need(R[k][0] is None and "already sits on the grid" in R[k][1], "G2 %s: the stock schedule is on the grid and stays" % k)
    _need(r4(R["4s"][0]) == [1.0, 0.92, 0.70, 0.45, 0.0] and "heads 1-2, 3-4, 5, 6" in R["4s"][1] and "'simple' would step between" in R["4s"][1],
          "G2 4 steps simple: the grid schedule, and the console says so")
    _need(r4(R["5s"][0]) == [1.0, 0.92, 0.84, 0.70, 0.45, 0.0], "G2 5 steps simple: the grid schedule")
    _need(R["4u"][0] == R["4s"][0], "G2 4 steps sa_solver + sgm_uniform: the same grid schedule")
    _need(r4(R["6b"][0]) == GRID and r4(R["6k"][0]) == GRID, "G2 beta and karras at 6 steps leave the grid -> the grid schedule")
    _need(R["12s"][0] is None and "12 steps" in R["12s"][1] and "1..6 steps run on the grid" in R["12s"][1],
          "G2 12 steps: said, not refused, left to the stock scheduler")
    for k in ("1@16", "2@33", "3@50"):
        _need(R[k][0] is None and "already sits" in R[k][1], "G2 refine %s is on the grid and stays" % k)
    _need(r4(R["1@25"][0]) == [0.45, 0.0] and "denoise 0.167 (1 of 6 heads)" in R["1@25"][1], "G2 refine 1 @ 0.25 snaps to 0.167 and says so")
    _need(R["2@16"][0] is None and "at denoise 0.16" in R["2@16"][1] and "only 1 head covers" in R["2@16"][1],
          "G2 refine 2 @ 0.16: one head is left for two steps -> said, left alone")
    _need(r4(R["6dpm2"][0]) == GRID, "G2 a sampler that drops the penultimate sigma never counts as on the grid")
    _need(C["rule"], "G2 Core's head rule reads the way the plan assumes (nearest grid point, dt-weighted mean)")
    STOCK4 = C["stock4"]
else:
    print("  SKIP  G2 no Core 0.45.0 importable here (%s)" % ((r.stderr or "")[-200:] if r else "not found"))

# ---- G3 the seat without Core ---------------------------------------------------------
import io  # noqa: E402
import contextlib  # noqa: E402
import torch  # noqa: E402

ns = {"torch": torch, "__name__": "lifted"}
exec(LIFT, ns)


class MS:
    grid = torch.tensor(list(reversed(GRID)))


class Model:
    def __init__(self, ms):
        self.ms = ms

    def get_model_object(self, k):
        return self.ms


GM = Model(type("MiniMaxH3ImageModelSampling", (MS,), {})())
OTHER = Model(type("ModelSamplingDiscreteFlow", (MS,), {})())
HALF = [1.0, 0.945, 0.84, 0.575, 0.0]
ns["_stock_sigmas"] = lambda *a, **k: HALF


def run(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = ns[fn](*a, **k)
    return out, buf.getvalue()


o, t = run("_grid_sigmas", OTHER, 4, "euler", "simple", 1.0)
_need(o is None and t == "", "G3 a model without a grid: None, and not a word")
o, t = run("_grid_prepare", OTHER, 4, "euler", "simple", 1.0, 0, 4, False, False, None)
_need(o is None and t == "", "G3 the Sampler's seat leaves a model without a grid alone")
o, t = run("_grid_prepare", GM, 4, "euler", "simple", 1.0, 0, 4, False, False, None)
_need(torch.is_tensor(o) and o.dtype == torch.float32 and [round(float(v), 4) for v in o] == [1.0, 0.92, 0.70, 0.45, 0.0],
      "G3 a grid model at 4 steps: the grid schedule as a float32 tensor")
wired = torch.tensor([1.0, 0.5, 0.0])
o, t = run("_grid_prepare", GM, 4, "euler", "simple", 1.0, 0, 4, False, False, wired)
_need(o is wired and t == "", "G3 a wired curve wins, untouched")
o, t = run("_grid_prepare", GM, 4, "euler", "simple", 1.0, 0, 4, False, True, None)
_need(o is None and t == "", "G3 High + Low is left alone")
o, t = run("_grid_prepare", GM, 4, "euler", "simple", 1.0, 0, 2, False, False, None)
_need([round(float(v), 4) for v in o] == [1.0, 0.92, 0.0] and "cut the grid schedule" in t, "G3 end_at_step 2: an early end is clean")
o, t = run("_grid_prepare", GM, 4, "euler", "simple", 1.0, 0, 2, True, False, None)
_need([round(float(v), 4) for v in o] == [1.0, 0.92, 0.70], "G3 end_at_step 2 with leftover noise keeps its sigma")
o, t = run("_grid_prepare", GM, 4, "euler", "simple", 1.0, 2, 10000, False, False, None)
_need([round(float(v), 4) for v in o] == [0.70, 0.45, 0.0], "G3 start_at_step 2 (end beyond the run) starts on the third point")
o, t = run("_grid_prepare", GM, 4, "euler", "simple", 1.0, 3, 3, False, False, None)
_need(o is None, "G3 an empty slice is left to the stock path")
ns["_stock_sigmas"] = lambda *a, **k: None
o, t = run("_grid_sigmas", GM, 6, "euler", "simple", 1.0)
_need(r4(o) == GRID, "G3 a stock schedule that cannot be told is never taken for the grid")
ns["_stock_sigmas"] = lambda *a, **k: list(GRID)
o, t = run("_grid_sigmas", GM, 6, "euler", "simple", 1.0)
_need(o is None and "already sits on the grid" in t, "G3 a stock schedule on the grid stays (bit for bit the old run)")
o, t = run("_grid_sigmas", GM, 0, "euler", "simple", 1.0)
o2, t2 = run("_grid_sigmas", GM, 4, "euler", "simple", 0.0)
_need(o is None and o2 is None and t == "" and t2 == "", "G3 no step or denoise 0: nothing built, nothing said")
bad = Model(type("MiniMaxH3ImageModelSampling", (), {})())
o, t = run("_grid_sigmas", bad, 4, "euler", "simple", 1.0)
_need(o is None and t == "", "G3 a grid class without a readable grid is left alone")
one = Model(type("MiniMaxH3ImageModelSampling", (), {"grid": torch.tensor([0.0])})())
o, t = run("_grid_sigmas", one, 1, "euler", "simple", 1.0)
_need(o is None and t == "", "G3 a grid of one point is no grid")

# ---- G4 Core's head rule, replayed ----------------------------------------------------


def core_heads(grid, sig):
    """FinalLayer.forward's selection (Core 0.45.0), per step: (start, stop)."""
    n = len(grid) - 1
    out = []
    for i in range(len(sig) - 1):
        j = min(range(len(sig)), key=lambda q: abs(sig[q] - sig[i]))
        nxt = sig[min(j + 1, len(sig) - 1)]
        st, sp = (min(range(n + 1), key=lambda k: abs(grid[k] - float(s))) for s in (sig[i], nxt))
        st = min(st, n - 1)
        sp = max(sp, st + 1)
        out.append((st, sp))
    return out


def served(grid, sig):
    """True when every step is served by exactly the heads of the interval it really walks."""
    hs = core_heads(grid, sig)
    return all(abs(G.index_of(grid, sig[i]) - a) < 1e-6 and abs(G.index_of(grid, sig[i + 1]) - b) < 1e-6
               for i, (a, b) in enumerate(hs))


ok = True
for n in range(1, 7):
    for left in range(n, 7):
        p = G.plan(GRID, n, left / 6.0)
        hs = core_heads(GRID, p["sigmas"])
        ok = ok and served(GRID, p["sigmas"]) and hs[0][0] == 6 - left and hs[-1][1] == 6 \
            and all(a[1] == b[0] for a, b in zip(hs, hs[1:]))
_need(ok, "G4 every plan (1..6 steps, every grid start): Core takes each head once, in order, for the step that walks it")
_need(not served(GRID, HALF), "G4 the stock 4-step schedule (0, 1.5, 3, 4.5, 6) is NOT served that way")
if STOCK4 is not None:
    _need(not served(GRID, STOCK4) and not G.on_grid(GRID, STOCK4),
          "G4 Core's real 'simple' at 4 steps steps between the points: %s" % " ".join("%.2f" % G.index_of(GRID, s) for s in STOCK4))

# ---- G5 wiring ------------------------------------------------------------------------
body = re.search(r"^    def _run\(self.*?(?=^    def |\Z)", SRC_S, re.S | re.M).group(0)
i_pdd, i_grid, i_dual = body.find("model, sigmas = _pdd_prepare("), body.find("sigmas = _grid_prepare("), body.find("        if dual_moe:\n            # High + Low")
_need(-1 < i_pdd < i_grid < i_dual, "G5 _run: PDD first, then the grid, then the High + Low branch")
call = re.search(r"sigmas = _grid_prepare\((.*?)\)\n", body, re.S).group(1)
_need(re.sub(r"\s+", " ", call) == "model, steps, sampler_name, scheduler, denoise, start_at_step, end_at_step, "
      "return_with_leftover_noise, dual_moe, sigmas", "G5 _run hands the grid step every dial it reads")
_need(SRC_S.count("_grid_prepare(") == 2, "G5 one call site for the grid step in the Sampler")
m = re.search(r'if st\.get\("sig_run"\) is None:\n(.*?)cur = _refine_tiles\(m, ', SRC_P, re.S)
_need(m is not None and '_gr = _grid_sigmas(m, st["steps"], samp, sched, st["denoise"],' in m.group(1)
      and 'st["sig_run"] = _gr' in m.group(1), "G5 Power Upscale: a stage without a wired curve asks the grid, right before the tile refine")
_need(SRC_P.count("from .uls_sampler import _grid_sigmas") == 1 and SRC_P.count("from uls_sampler import _grid_sigmas") == 1,
      "G5 Power Upscale imports the Sampler's seat (both load paths)")
_need(all(ord(c) < 128 for c in SRC_S[SRC_S.find("def _grid_mod"):SRC_S.find("def _apply_sigma_shift")]), "G5 the new Sampler code is ASCII")

print()
if _fails:
    print("%s: %d FAILED" % (NAME, len(_fails)))
    for f in _fails:
        print("   - " + f)
    sys.exit(1)
print("%s: all green" % NAME)
