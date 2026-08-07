"""Guard v576 -- the clock moves in with the Sampler.

THE STANDING WOUND (same family as v567's): the Polyhedron Sampler's bar was
a step-ticker. Ticks are not time -- on Frank's measured chain (HIGH 43.1 s/step,
LOW 93.4 s/step) the bar claimed 37.5% at the handoff while the wall clock said
~22%, and neither the bar nor the UI anywhere said how long was LEFT. Stock
ComfyUI never does: the console tqdm knows one sample() call at a time (it
resets at the MoE handoff and, per upstream issue #11643, mis-times Wan 2.2
fp8 -- Frank's exact stack), and the frontend shows percent, never remaining.

v576, three moves, all asserted here by RUNNING the real code:
  1. THE CLOCK HAS ITS OWN HOUSE. nodes/ph_runclock.py holds _fmt_clock +
     _RunClock verbatim (v567 code, new address); ph_power_upscale re-exports.
     The module IS the guard window now -- whole-file exec, no regex anchors.
  2. THE SAMPLER TELLS TIME. One shared clock per run: all three MoE chains
     post step:high/step:low via the pure _chain_posts (equal weight -- same
     latent -- so rung 2 hands the unmeasured expert the measured rate 1:1,
     the EMA corrects from its first own step), both single paths post
     step:main. One bar, ONE writer: the tick push survives only as the
     clock-less fallback.
  3. THE STEP SPEAKS OR FOLDS. Slow steps print duration + stage eta +
     tilde-marked run eta; steps under _CLOCK_FOLD_S fold into one summary
     line per stage (the v567 chunk courtesy). The known softness is SAID:
     the HIGH->LOW model swap lands lazily in the first LOW dt.

Script-style: exit 0 = pass.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v576_sampler_clock] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


class _FakeBar:
    def __init__(self):
        self.calls = []

    def update_absolute(self, value, total=None, preview=None):
        self.calls.append((value, total))


def main():
    # ---- 1: the clock's own house -------------------------------------------------
    path = os.path.join(ROOT, "nodes", "ph_runclock.py")
    if not os.path.isfile(path):
        _fail("nodes/ph_runclock.py is missing - the move did not land")
    rc = _read("nodes", "ph_runclock.py")
    if "import torch" in rc or "import comfy" in rc:
        _fail("ph_runclock must stay dependency-free (import time, nothing "
              "else) - the guards exec THIS WHOLE FILE in a torch-less sandbox")
    ns = {}
    exec(compile(rc, "<ph_runclock>", "exec"), ns)  # noqa: S102 - our own source
    Clock, fmt = ns["_RunClock"], ns["_fmt_clock"]
    if fmt(65) != "1:05" or fmt(3725) != "1:02:05":
        _fail("_fmt_clock lost its one true format on the way over")

    pu = _read("nodes", "ph_power_upscale.py")
    if "class _RunClock" in pu or "def _fmt_clock" in pu:
        _fail("the clock still lives in ph_power_upscale - the move must be "
              "real, not a copy (copies drift)")
    if pu.count("from .ph_runclock import _fmt_clock, _RunClock") != 1 or \
       pu.count("\n    from ph_runclock import _fmt_clock, _RunClock") != 1:
        _fail("PU must re-export the clock in BOTH import branches (house "
              "pattern: uls_stack_node -> uls_merge_math)")

    # ---- 2: the sampler tells time -------------------------------------------------
    sm = _read("nodes", "uls_sampler.py")
    if "from .ph_runclock import _fmt_clock, _RunClock" not in sm:
        _fail("the sampler does not import the clock")
    if sm.count("_RunClock(pbar)") != 5:
        _fail("expected exactly 5 clock owners (3 MoE chains + 2 single "
              "paths), got %d" % sm.count("_RunClock(pbar)"))
    if sm.count("_chain_posts(") != 4:
        _fail("the pure _chain_posts must be defined once and called by all "
              "3 chains")
    if sm.count('clock_key="step:high"') != 3 or \
       sm.count('clock_key="step:low"') != 3:
        _fail("every chain must key BOTH callbacks into the shared clock")
    if sm.count('"step:main"') != 6:
        _fail("both single paths must post, key AND close step:main "
              "(2 posts + 2 keys + 2 close tuples)")
    if "_CLOCK_FOLD_S = 5.0" not in sm:
        _fail("the fold threshold is gone - small-step runs would flood the "
              "console")

    # one bar, ONE writer: the tick push survives only as the clock-less fallback
    cb = sm[sm.index("def _make_preview_callback"):sm.index("def _fix_empty_latent")]
    if cb.count("pbar.update_absolute") != 1:
        _fail("the step-tick writer must survive exactly once (clock-less "
              "fallback) - two writers on one bar tell two stories")
    i_if = cb.index("if clock is not None and clock_key is not None:")
    i_push = cb.index("pbar.update_absolute")
    if not (i_if < cb.index("clock.measure(clock_key") < cb.index("else:", i_if) < i_push):
        _fail("the tick push must sit in the ELSE of the clock branch - with "
              "a clock the clock owns the bar")

    # the step speaks or folds, estimates wear their tilde
    if "run eta ~" not in cb:
        _fail("the run eta must ride the step line, tilde-marked (an "
              "estimate that does not say so is a lie)")
    if "stage eta " not in cb:
        _fail("the step line lost its stage eta")
    if "too fast to narrate" not in cb or "folded (" not in cb:
        _fail("the sub-threshold litany must fold into one line per stage "
              "(the v567 chunk courtesy)")
    if "swap happens lazily" not in cb:
        _fail("the known softness must be SAID where it lives: the HIGH->LOW "
              "model swap lands in the first LOW dt")
    # AMENDED IN v577 (3rd amendment): the cursor must NEVER move back into the
    # closure. A per-callback t_last is the v576 bug, and it is invisible from
    # the outside - the numbers just quietly lie.
    if "t_last" in sm:
        _fail("a per-callback time cursor is back in uls_sampler - that IS the "
              "v576 bug (cb_low is built before phase 1, so its cursor sits at "
              "t0 and the first LOW step swallows the whole HIGH phase)")
    if "clock.tick()" not in cb:
        _fail("the callback must take dt from the RUN's shared cursor "
              "(clock.tick()), not from a private clock")

    # the close line, once per owner
    if sm.count("_clock_close(") != 6:
        _fail("expected the close helper + 5 call sites (one per clock owner)")
    if "executor overhead outside sampling" not in sm:
        _fail("the done line must name why the node badge differs")

    # ---- 3: the pure plan, EXECUTED ------------------------------------------------
    # AMENDED IN v577 (1st amendment): the plan now weighs PHYSICS. A step at
    # cfg 1.0 is ONE model forward (Core skips the uncond pass); cfg > 1 is TWO.
    # On Frank's chain that IS the cross-expert cost ratio - lightning HIGH at
    # cfg 1.0 measured 43.1 s/step against a LOW at cfg 6 with 93.4 s/step,
    # factor 2.17 on the same latent. Without it, rung 2 handed LOW the HIGH
    # rate and the run eta at the handoff was HALF the truth (216 s against
    # 467 s). The window now starts at _cfg_forwards; with equal cfgs the
    # weights are equal and the plan is byte-identical to v576.
    m = re.search(r"def _cfg_forwards\(.*?\n(?=\n\ndef _clock_close)", sm, re.S)
    if not m:
        _fail("cannot carve the plan (start anchor def _cfg_forwards, end "
              "anchor def _clock_close)")
    ns2 = {}
    exec(compile(m.group(0), "<_chain_posts>", "exec"), ns2)  # noqa: S102
    W = float(16 * 21 * 60 * 104)
    fwd = ns2["_cfg_forwards"]
    if fwd(1.0) != 1.0 or fwd(6.0) != 2.0 or fwd(None) != 2.0:
        _fail("cfg 1.0 must cost ONE forward, cfg > 1 TWO, and an unreadable "
              "cfg must assume the expensive case")
    posts = ns2["_chain_posts"](3, 5, W, 3.5, 3.5)          # equal cfg
    if posts != [("step:high", 3, W * 2.0), ("step:low", 5, W * 2.0)]:
        _fail("equal cfgs must still plan equal weights - got %r" % (posts,))
    posts = ns2["_chain_posts"](3, 5, W, 1.0, 6.0)          # Frank's chain
    if posts != [("step:high", 3, W), ("step:low", 5, W * 2.0)]:
        _fail("a lightning HIGH (cfg 1.0) must weigh HALF a cfg-6 LOW - that "
              "factor is the measured 43.1 vs 93.4 - got %r" % (posts,))
    if ns2["_chain_posts"](1, 1, 0, 1.0, 1.0)[0][2] != 1.0:
        _fail("weight must floor at 1.0 (the clock divides by it)")

    # ---- 4: the ladder on Frank's chain, with the RUN's shared cursor ---------------
    # AMENDED IN v577 (2nd amendment): this now reproduces the exact mechanism
    # the sampler uses - TWO stage callbacks, BOTH built before phase 1 (that is
    # what the MoE chains do), both taking dt from clock.tick(). The v576 code
    # kept the cursor in each callback's closure, seeded at BUILD time, so
    # cb_low's cursor sat at t0 and the first LOW step measured the WHOLE HIGH
    # PHASE: 222.7 s where the truth was 93.4 s, and the run eta DOUBLED at the
    # handoff. Caught by audit before it ever ran on the card.
    t = {"now": 0.0}
    bar = _FakeBar()
    c = Clock(bar, now=lambda: t["now"])
    for k, u, w in posts:                       # Frank's chain: cfg 1.0 / cfg 6
        c.post(k, u, w)

    # Both "callbacks" exist from here on - exactly as _moe_sample builds them.
    def cb(key):
        return lambda: c.measure(key, c.tick())
    cb_high, cb_low = cb("step:high"), cb("step:low")

    for _ in range(3):
        t["now"] += 43.1
        cb_high()
    if abs(c.posts["step:high"]["rate"] - 43.1) > 1e-9:
        _fail("HIGH must measure its own 43.1 s")
    r_low = c._rate("step:low")
    if r_low is None or abs(r_low - 2 * 43.1) > 1e-9:
        _fail("rung 2 must scale by WEIGHT: a cfg-6 LOW is twice the forwards "
              "of a cfg-1.0 HIGH, so 43.1 -> 86.2 before LOW ever ran (the "
              "truth is 93.4 - within 8%%, where v576 was off by 54%%) - got "
              "%r" % r_low)
    v, tot = bar.calls[-1]
    if not (0.20 < v / tot < 0.28):
        _fail("at the handoff the TIME bar must sit near 23%% (129.3s elapsed "
              "of an estimated 560.3s; the TRUTH is 129.3/596.3 = 21.7%%, so "
              "the bar is honest to within 1.4 points before LOW has run a "
              "single step) - got %d/%d" % (v, tot))

    t["now"] += 93.4
    cb_low()                                     # THE trap: cb_low was built at t0
    dt1 = c.posts["step:low"]["spent"]
    if abs(dt1 - 93.4) > 1e-9:
        _fail("THE v576 BUG: the first LOW step measured %.1f s instead of "
              "93.4 s - the cursor must belong to the RUN (clock.tick()), not "
              "to a callback closure seeded at build time" % dt1)
    t["now"] += 92.5
    cb_low()
    if abs(c.posts["step:low"]["rate"] - (0.5 * 93.4 + 0.5 * 92.5)) > 1e-9:
        _fail("the rate must be an EMA (alpha 0.5)")
    e = c.eta()
    if e is None or not (3 * 90.0 * 0.9 < e < 3 * 96.0 * 1.1):
        _fail("run eta must cover the 3 open LOW steps at the EMA rate, got "
              "%r" % e)

    # THE INVARIANT that would have caught v576 on its own: measured seconds can
    # never exceed the wall clock. v576 summed 725.6 s against 596.3 s elapsed.
    spent = sum(p["spent"] for p in c.posts.values())
    if spent > c.elapsed() + 1e-6:
        _fail("sum(spent)=%.1f exceeds elapsed=%.1f - the clock is counting "
              "the same seconds twice" % (spent, c.elapsed()))
    for a, b in bar.calls:
        if b is not None and b <= a:
            _fail("total must stay ahead of value")

    print("[test_v576_sampler_clock] PASS: the clock has its own house, the "
          "sampler tells time, one bar has one writer, and the ladder runs "
          "on Frank's numbers.")
    sys.exit(0)


if __name__ == "__main__":
    main()
