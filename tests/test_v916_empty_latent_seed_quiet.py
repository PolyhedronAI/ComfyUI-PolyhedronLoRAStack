"""v916 guard -- the Empty Latent's dead seed goes quiet.

Finding (05.09.): the Empty Latent's noise_seed is hidden since v685/v688
but its control_after_generate kept firing on "randomize" after every
queue. The console then printed `noise=zeros seed=<15 digits>` -- a seed
that does nothing (v685: at sigma 1.0 the latent is multiplied by zero,
the sampler's NOISE source makes all the noise). Two runs, two seeds,
and an A/B was declared invalid that was in fact valid.

Pins:
  P1  uls_noise.noise_log_tag: seedless -> no seed value, says who seeds
  P2  uls_noise.noise_log_tag: seeded type -> "noise=T seed=S" verbatim
  P3  ph_empty_latent.py has NO remaining "seed={noise_seed}" / "seed=%s"
      fragment -- every print goes through the helper (4 sites)
  P4  ph_empty_latent.js pins the hidden control_after_generate to "fixed"
      inside _hideLegacyNoise, and still HIDES rather than removes it
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "nodes"))

import uls_noise  # noqa: E402


def _fail(msg):
    print("[test_v916_empty_latent_seed_quiet] FAIL -- " + msg)
    sys.exit(1)


# P1
tag = uls_noise.noise_log_tag("zeros", 267009027629902)
if "267009027629902" in tag:
    _fail("P1 seedless tag still prints the seed value: " + tag)
if not tag.startswith("noise=zeros"):
    _fail("P1 seedless tag must start with noise=zeros: " + tag)
if "NOISE source" not in tag:
    _fail("P1 seedless tag must name the sampler's NOISE source as the seeder")
for t in uls_noise.SEEDLESS_TYPES:
    if "seed=" in uls_noise.noise_log_tag(t, 1):
        _fail("P1 seedless type %r prints a seed" % t)

# P2
if uls_noise.noise_log_tag("gaussian", 42) != "noise=gaussian seed=42":
    _fail("P2 seeded tag changed: " + uls_noise.noise_log_tag("gaussian", 42))

# P3
py = open(os.path.join(ROOT, "nodes", "ph_empty_latent.py"), encoding="utf-8").read()
if "seed={noise_seed}" in py or "noise=%s seed=%s" in py:
    _fail("P3 a raw seed fragment survives in ph_empty_latent.py")
n_helper = py.count("uls_noise.noise_log_tag(noise_type, noise_seed)")
if n_helper != 4:
    _fail("P3 expected the helper at 4 print sites, found %d" % n_helper)

# P4
js = open(os.path.join(ROOT, "web", "js", "ph_empty_latent.js"), encoding="utf-8").read()
m = re.search(r"function _hideLegacyNoise\(node\) \{(.*?)\n\}", js, re.S)
if not m:
    _fail("P4 _hideLegacyNoise not found")
body = m.group(1)
if 'w.name === "control_after_generate"' not in body or 'w.value = "fixed"' not in body:
    _fail("P4 _hideLegacyNoise does not pin control_after_generate to fixed")
if 'w.type = "hidden"' not in body:
    _fail("P4 the legacy rows must stay HIDDEN (v585: never removed)")
if "splice" in body or "widgets.filter" in body:
    _fail("P4 _hideLegacyNoise must not remove widgets from the array")

print("[test_v916_empty_latent_seed_quiet] OK - seedless tag names the real "
      "seeder and drops the value, 4 print sites on the helper, hidden control "
      "pinned to fixed and still hidden (4 pins)")
