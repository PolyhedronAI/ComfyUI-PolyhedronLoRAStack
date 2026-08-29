"""Guard v538 -- Seed: native control option, roll/reuse mechanics, parity."""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _fail(msg):
    print("FAIL: " + msg)
    sys.exit(1)


def _read(*parts):
    return open(os.path.join(ROOT, *parts), encoding="utf-8").read()


def main():
    py = _read("nodes", "ph_basics.py")
    js = _read("web", "js", "ph_seed.js")
    init = _read("__init__.py")

    # -- backend -------------------------------------------------------------
    if "0xffffffffffffffff" not in py:
        _fail("full 64-bit seed range missing")
    if '"control_after_generate": True' not in py:
        _fail("native control_after_generate option missing")
    # v685: the noise source was APPENDED as slot 2. Outputs may grow at the
    # end (slot indices of existing links are preserved); they may never be
    # re-ordered or shortened. Pin the PREFIX, not the whole tuple, so this
    # guard keeps protecting slots 0 and 1 without forbidding growth.
    if 'RETURN_TYPES = ("INT", "STRING"' not in py:
        _fail("seed + seed_string outputs changed")
    if 'RETURN_NAMES = ("seed", "seed_string"' not in py:
        _fail("seed + seed_string output NAMES changed")
    if '"pls_seed"' not in py or '"used"' not in py:
        _fail("used-seed ui channel missing in backend")

    # -- frontend ------------------------------------------------------------
    if "[PLS] ph_seed.js v538 loaded" not in js:
        _fail("self-proving banner missing/stale in ph_seed.js")
    if "crypto.getRandomValues" not in js or "0x1fffff" not in js:
        _fail("crypto 53-bit roll missing")
    if 'c.value = "fixed"' not in js:
        _fail("roll/reuse must pin control_after_generate to 'fixed'")
    if "pls_last_used" not in js:
        _fail("reload persistence (properties.pls_last_used) missing")
    if "pls_seed" not in js:
        _fail("ui channel key mismatch (js side)")
    if "Reuse last" not in js:
        _fail("reuse-last button missing")

    # -- registration ----------------------------------------------------------
    # RE-GROUNDED for the public build (v371), declared in the changelog:
    # the original pinned the exact SPELLING of the import line
    # ("from .nodes.ph_basics import ULSSeed"). The public build imports the
    # four ph_basics nodes in ONE grouped statement, which is the same fact
    # written differently -- and pinning spelling instead of the invariant is
    # the wound test_v352 was decoupled from back in v364. The INVARIANT is
    # what matters and is what is checked here: ULSSeed comes from ph_basics,
    # is registered, and has its display name.
    if not re.search(r"from \.nodes\.ph_basics import [^\n]*\bULSSeed\b", init):
        _fail("registration incomplete: ULSSeed is not imported from "
              ".nodes.ph_basics in __init__.py")
    for needle in ('"ULSSeed"', "⬡ Polyhedron Seed"):
        if needle not in init:
            _fail(f"registration incomplete: {needle!r} not in __init__.py")

    print("PASS: v538 seed -- native control, roll-and-keep, reuse-last, registration")
    sys.exit(0)


if __name__ == "__main__":
    main()
