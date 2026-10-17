"""Guard v833 -- noise character dial + offset/pyramid types.

WHAT THIS PINS (driven where the mechanism lives):

  N1  the lists: NOISE_TYPES ends with the two APPENDED types (combo
      options serialise by string; #747 precedent); the seed node's
      canon ends with the APPENDED noise_character row (FLOAT, default
      1.0 = pure type, 0..1).
  N2  STREAM LAW: character 1.0 is bit-identical to the pre-dial
      recipe for every pre-v833 shaped type -- the shaped tensor is
      drawn FIRST and the blending white never at 1.0. The old recipe
      is rebuilt here from the module's own parts and compared
      tensor-equal; the bare call (no character arg) equals 1.0.
  N3  the spectral seats, measured by low-frequency energy fraction:
      blue < gaussian < pyramid < fractal. pyramid is the TAME cousin
      -- near-white with a mild lift, not fractal's clouds.
  N4  the dial works and works MONOTONICALLY: fractal's low-frequency
      fraction rises with character (0.1 < 0.5 < 1.0). MUTATION:
      neutralise the blend line and the monotonicity must fail --
      proof the pin drives the mix, not the fixture.
  N5  offset: character 0 is bit-identical to gaussian (the DC draw
      never happens), and the per-channel DC variance rises with the
      dial. The whole point of offset is that near-white spectrum.
  N6  the seed node passes the dial into the NOISE source and heals
      junk to the default per house law #577; the source's
      generate_noise hands it to make_noise.
  N7  the preview route reads &character= and hands it to make_noise
      -- the route's contract is "what the node draws is what the
      sampler will denoise", and a dial the preview ignores breaks it.
  N8  the JS half: ph_seed.js sends character in the fetch URL and
      keys the cache on it; ph_noise_field.js knows the two new
      painters and cross-fades the impression below 1.0.

Script-style: exit 0 = pass.
"""
import importlib.util
import os
import sys

import torch

NAME = "v833"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NOISE = os.path.join(ROOT, "nodes", "uls_noise.py")
ROUTES = os.path.join(ROOT, "nodes", "ph_basics_routes.py")   # public build (v388): the noise preview's own module
SEEDJS = os.path.join(ROOT, "web", "js", "ph_seed.js")
FIELDJS = os.path.join(ROOT, "web", "js", "ph_noise_field.js")
BASICS = os.path.join(ROOT, "nodes", "ph_basics.py")


def _fail(msg):
    print("[%s] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


def _load(path, name, src=None):
    if src is None:
        spec = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m
    import types as _t
    m = _t.ModuleType(name)
    m.__file__ = path
    exec(compile(src, path, "exec"), m.__dict__)
    return m


mod = _load(NOISE, "uls_noise_v833")

# ---- N1: the lists -------------------------------------------------------
_need(mod.NOISE_TYPES[-2:] == ["offset", "pyramid"],
      "N1: offset/pyramid must be APPENDED at the end of NOISE_TYPES "
      "(got %r)" % (mod.NOISE_TYPES,))
_need(mod.NOISE_TYPES[:6] == ["zeros", "gaussian", "pink", "brown", "blue",
                              "fractal"],
      "N1: the pre-v833 type order moved -- that renames nothing but "
      "reorders the combo everyone knows")

bsrc = open(BASICS, encoding="utf-8").read()
i = bsrc.index('"noise_character"')
_need(i > bsrc.index('"preview_height"'),
      "N1: noise_character must sit AFTER preview_height (appended, #577)")
_need('"default": 1.0, "min": 0.0, "max": 1.0' in bsrc[i:i + 300],
      "N1: noise_character must be FLOAT default 1.0 in 0..1")

# ---- N2: stream law ------------------------------------------------------
S = (1, 4, 32, 48)
for t in ("pink", "brown", "blue"):
    old = (mod._colored(S, mod.COLORED_BETA[t], mod._generator(7))
           * 1.0).to(torch.float32)
    _need(torch.equal(old, mod.make_noise(t, S, 7, 1.0, 1.0)),
          "N2: %s at character 1.0 must be bit-identical to the pre-dial "
          "recipe" % t)
old_f = (mod._fractal(S, mod._generator(7)) * 1.0).to(torch.float32)
_need(torch.equal(old_f, mod.make_noise("fractal", S, 7, 1.0, 1.0)),
      "N2: fractal at character 1.0 must be bit-identical")
_need(torch.equal(mod.make_noise("pink", S, 7, 1.0),
                  mod.make_noise("pink", S, 7, 1.0, 1.0)),
      "N2: the bare call (no character arg) must equal character 1.0")


# ---- N3 + N4: spectral seats + the dial ----------------------------------
def _lowfrac(x):
    x = x[0, 0]
    sp = torch.fft.rfft2(x).abs() ** 2
    H, W = x.shape
    fy = torch.fft.fftfreq(H).reshape(H, 1)
    fx = torch.fft.rfftfreq(W).reshape(1, -1)
    r = (fy * fy + fx * fx).sqrt()
    return float(sp[r < 0.0625].sum() / sp.sum())


S2 = (1, 1, 128, 128)
lf = {t: _lowfrac(mod.make_noise(t, S2, 11, 1.0))
      for t in ("gaussian", "blue", "pyramid", "fractal")}
_need(lf["blue"] < lf["gaussian"] < lf["pyramid"] < lf["fractal"],
      "N3: spectral seats broken (blue %.4f, gaussian %.4f, pyramid %.4f, "
      "fractal %.4f)" % (lf["blue"], lf["gaussian"], lf["pyramid"],
                         lf["fractal"]))
_need(lf["pyramid"] < 0.35,
      "N3: pyramid drifted into cloud territory (lowfreq %.3f) -- it must "
      "stay the TAME cousin" % lf["pyramid"])


def _dial(m):
    return [_lowfrac(m.make_noise("fractal", S2, 11, 1.0, c))
            for c in (0.1, 0.5, 1.0)]


d = _dial(mod)
_need(d[0] < d[1] < d[2],
      "N4: the dial must move fractal's low-frequency fraction "
      "monotonically (got %r)" % (d,))
_need(d[0] < 0.15 and d[2] > 0.5,
      "N4: the dial's range collapsed (%.3f .. %.3f)" % (d[0], d[2]))

# the mutation: neutralise the blend
src = open(NOISE, encoding="utf-8").read()
BLEND = ("            out = _normalize(math.sqrt(1.0 - c * c) * white "
         "+ c * out)")
_need(src.count(BLEND + "\n") == 1,
      "N4: the blend line moved -- re-anchor the mutation")
mut = _load(NOISE, "uls_noise_v833_mut",
            src.replace(BLEND + "\n",
                        "            out = _normalize(1.0 * out)\n"))
dm = _dial(mut)
_need(not (dm[0] < dm[1] < dm[2] and dm[0] < 0.15),
      "N4: a neutralised blend must FAIL the monotonicity pin -- the pin "
      "does not drive the mix (got %r)" % (dm,))

# ---- N5: offset ----------------------------------------------------------
_need(torch.equal(mod.make_noise("offset", S, 7, 1.0, 0.0),
                  mod.make_noise("gaussian", S, 7, 1.0)),
      "N5: offset at character 0 must be bit-identical to gaussian")


def _dcvar(x):
    return float(x.mean(dim=(-2, -1)).var())


SB = (1, 16, 64, 64)
g = _dcvar(mod.make_noise("gaussian", SB, 7, 1.0))
o3 = _dcvar(mod.make_noise("offset", SB, 7, 1.0, 0.3))
o10 = _dcvar(mod.make_noise("offset", SB, 7, 1.0, 1.0))
_need(o10 > o3 > g,
      "N5: the per-channel DC must rise with the dial "
      "(gaussian %.5f, 0.3 %.5f, 1.0 %.5f)" % (g, o3, o10))

# ---- N6: the seed node + the source --------------------------------------
ns = mod.ULSNoiseSource(5, "fractal", 1.0, 0.4)
_need(abs(ns.character - 0.4) < 1e-9, "N6: the source drops the dial")
_need(mod.ULSNoiseSource(5, "fractal", 1.0, "junk").character == 1.0,
      "N6: junk character must heal to 1.0 in the source")
lat = {"samples": torch.zeros(S)}
_need(torch.equal(ns.generate_noise(lat),
                  mod.make_noise("fractal", S, 5, 1.0, 0.4)),
      "N6: generate_noise must hand the dial to make_noise")
_need("noise_character=1.0" in bsrc.split("def emit", 1)[1][:400],
      "N6: emit() must take noise_character with default 1.0")
_need("ULSNoiseSource(used, noise_type, noise_strength,\n"
      "                                         noise_character)" in bsrc,
      "N6: emit() must pass the dial into the source")

# ---- N7: the route -------------------------------------------------------
rsrc = open(ROUTES, encoding="utf-8").read()
seg = rsrc.split("async def handle_noise_preview", 1)[1]
seg = seg.split("async def", 1)[0]
_need('q.get("character"' in seg,
      "N7: the preview route must read &character=")
_need("seed, strength,\n                                     character)"
      in seg,
      "N7: the route must hand the dial to make_noise")

# ---- N8: the JS half -----------------------------------------------------
sjs = open(SEEDJS, encoding="utf-8").read()
_need('"&character=" + st.character' in sjs,
      "N8: ph_seed.js must send the dial in the fetch URL")
_need('st.character.toFixed(2)' in sjs,
      "N8: the fetch cache key must carry the dial")
_need(sjs.count('character: Number(g("noise_character", 1.0)),') == 2,
      "N8: BOTH state branches must read the widget")
fjs = open(FIELDJS, encoding="utf-8").read()
for needle in ('type === "offset"', 'type === "pyramid"',
               "Math.sqrt(1 - ch * ch)", 'gridH, character)'):
    _need(needle in fjs, "N8: ph_noise_field.js needle missing %r" % needle)

print("[%s] PASS -- dial + offset/pyramid: stream law holds, seats "
      "measured, mutation caught, node/route/JS carry the dial" % NAME)
sys.exit(0)
