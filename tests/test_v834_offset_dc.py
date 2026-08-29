"""Guard v834 -- offset noise: ONE DC per (batch, channel), audit find A1.

THE WOUND, measured on the v833 build: the offset DC was drawn as
shape[:-2]+(1,1). On a 4D still latent that is (B,C,1,1) -- correct. On a
5D video latent (B,C,T,H,W) it silently became (B,C,T,1,1): an INDEPENDENT
offset per FRAME. Measured frame-mean spread across T: 0.13-0.22 against
0.02-0.03 of pure white jitter -- 5-8x, i.e. visible brightness flicker on
exactly the model family (Wan) this suite is built around.

WHAT THIS PINS:

  O1  the recipe, rebuilt from the module's own parts: white first, then
      ONE dc anchored on the leading min(2, ndim-2) axes, broadcast over
      everything behind them. 4D is BIT-IDENTICAL to the v833 stream
      (same dc shape, same draw count); 5D draws (B,C,1,1,1).
  O2  the symptom is gone and STAYS gone: on 5D the frame-mean spread of
      offset at full character is within 2x of the gaussian white jitter
      -- while the per-channel DC (offset's whole point) is still an
      order of magnitude above gaussian's.
  O3  MUTATION: revert the dc shape to shape[:-2]+(1,1) and O2 must fail
      -- proof the pin drives the anchor, not the fixture.

Script-style: exit 0 = pass.
"""
import importlib.util
import os
import sys

import torch

NAME = "v834"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NOISE = os.path.join(ROOT, "nodes", "uls_noise.py")


def _fail(msg):
    print("[%s] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


def _load(src=None):
    if src is None:
        spec = importlib.util.spec_from_file_location("uls_noise_v834", NOISE)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    import types as _t
    m = _t.ModuleType("uls_noise_v834_mut")
    m.__file__ = NOISE
    exec(compile(src, NOISE, "exec"), m.__dict__)
    return m


mod = _load()

# ---- O1: the recipe ------------------------------------------------------
S4 = (2, 16, 64, 64)
g = mod._generator(7)
w = torch.randn(S4, generator=g)
dc = torch.randn((2, 16, 1, 1), generator=g)
want4 = (mod._normalize(w + mod._OFFSET_MAX * dc) * 1.0).to(torch.float32)
_need(torch.equal(want4, mod.make_noise("offset", S4, 7, 1.0, 1.0)),
      "O1: 4D must stay bit-identical to the v833 stream "
      "(white then (B,C,1,1) dc)")

S5 = (1, 4, 16, 32, 32)
g = mod._generator(7)
w = torch.randn(S5, generator=g)
dc = torch.randn((1, 4, 1, 1, 1), generator=g)
want5 = (mod._normalize(w + mod._OFFSET_MAX * dc) * 1.0).to(torch.float32)
_need(torch.equal(want5, mod.make_noise("offset", S5, 7, 1.0, 1.0)),
      "O1: 5D must draw ONE dc per (batch, channel) as (B,C,1,1,1)")


# ---- O2: the symptom -----------------------------------------------------
def _frame_spread(x):
    """Mean spread of per-frame spatial means across T (dim 2)."""
    return float(x.mean(dim=(-2, -1)).std(dim=-1).mean())


def _chan_dc(x):
    return float(x.mean(dim=tuple(range(2, x.ndim))).var())


def _o2(m):
    off = m.make_noise("offset", S5, 7, 1.0, 1.0)
    gau = m.make_noise("gaussian", S5, 7, 1.0)
    return _frame_spread(off), _frame_spread(gau), _chan_dc(off), _chan_dc(gau)


so, sg, do, dg = _o2(mod)
_need(so < 2.0 * sg,
      "O2: offset's frame-mean spread must sit at white-jitter level "
      "(offset %.4f vs gaussian %.4f) -- the flicker is back" % (so, sg))
_need(do > 10.0 * max(dg, 1e-9),
      "O2: the per-channel DC vanished (offset %.5f vs gaussian %.5f) -- "
      "the fix must not amputate the type's point" % (do, dg))

# ---- O3: mutation --------------------------------------------------------
src = open(NOISE, encoding="utf-8").read()
LINES = ("            lead = min(2, max(0, len(shape) - 2))\n"
         "            dc = torch.randn(tuple(shape[:lead])\n"
         "                             + (1,) * (len(shape) - lead), "
         "generator=g)\n")
_need(src.count(LINES) == 1,
      "O3: the dc-shape lines moved -- re-anchor the mutation")
mut = _load(src.replace(
    LINES,
    "            dc = torch.randn(tuple(shape[:-2]) + (1, 1), "
    "generator=g)\n"))
smo, smg, _, _ = _o2(mut)
_need(not (smo < 2.0 * smg),
      "O3: the reverted anchor must FAIL the spread pin (mutant offset "
      "%.4f vs gaussian %.4f) -- the pin does not drive the anchor"
      % (smo, smg))

print("[%s] PASS -- one DC per (batch, channel): 4D stream identical, "
      "5D flicker gone, DC alive, mutation caught" % NAME)
sys.exit(0)
