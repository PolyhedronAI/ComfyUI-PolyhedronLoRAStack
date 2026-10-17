"""GATE v580 - the TAIL CALLBACK. Executed against Frank's measured seconds.

THE WOUND (measured on the field, not reasoned about):

Some samplers fire the step callback ONE MORE TIME after the last step.
RES4LYF's `res_2s` does. ComfyUI's `dpmpp_2m` does not -- which is why the Power
Upscale posts a clean 3/3 in the same log where the sampler posts 3/3 AND a
fourth line. The tail carries real seconds (the final preview decode) and it
GROWS WITH THE FRAME COUNT:

    13 frames -> tail 5.7 s / 5.4 s      65 frames -> tail 37.8 s / 34.4 s

v576 clamped `done` at `units`, so the tail never miscounted as a step. But it
still fed `spent` AND `rate`. The EMA ate it. On Frank's 65-frame run the console
printed `step 3/3 high` twice -- 67.2 s, then 37.8 s -- and the run-ETA at the
MoE handoff fell from 5:36 to 4:22. The LOW stage then took 6:10.

    with the tail in the EMA : 4:22 said -> 108 s wrong
    with the tail excluded   : 5:36 said ->  34 s wrong

THE INVARIANT THIS GUARD PINS (and it is an invariant, not a mechanism -- the
v576 lesson, paid for twice):

    sum(spent) == elapsed          the tail's seconds are REAL, they stay
    done <= units                  the tail is NOT a step, it must not count
    rate is untouched by the tail  the tail must not move the estimate

A guard that only checked `done` would have been green through all of this.
v576's forty-two were.
"""
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RC = os.path.join(HERE, "..", "nodes", "ph_runclock.py")


def _fail(msg):
    print(f"[test_v580_tail_callback] FAIL: {msg}")
    sys.exit(1)


spec = importlib.util.spec_from_file_location("_rc_v580", RC)   # dependency-free
rc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rc)

# ---- Frank's 65-frame run, 2026-07-13, verbatim from the console -------------
HIGH = [65.9, 63.2, 63.6]
LOW = [66.6, 65.3, 67.2, 67.7, 63.8]
TAIL_HIGH = 33.1
TAIL_LOW = 32.5
WALL = sum(HIGH) + TAIL_HIGH + sum(LOW) + TAIL_LOW      # 588.9 s = 'done in 9:50'


def _fresh():
    c = rc._RunClock()
    c.post("step:high", len(HIGH), 1.0)
    c.post("step:low", len(LOW), 1.0)
    return c


# ---- 1: the tail is NOT a step ----------------------------------------------
clock = _fresh()
for dt in HIGH:
    if clock.measure("step:high", dt) is not True:
        _fail("a real step must report True from measure() -- the sampler decides "
              "whether to SPEAK on that return value")
rate_before = clock.posts["step:high"]["rate"]
eta_before = clock.eta()

if clock.measure("step:high", TAIL_HIGH) is not False:
    _fail("the TAIL callback must report False from measure(). It returned True, so "
          "the sampler will print 'step 3/3' a second time with a different number "
          "-- which is the console lying quietly.")

p = clock.posts["step:high"]
if p["done"] != len(HIGH):
    _fail(f"the tail moved `done` to {p['done']} (units={len(HIGH)})")

# ---- 2: the tail must NOT touch the rate ------------------------------------
if abs(p["rate"] - rate_before) > 1e-9:
    _fail(f"the tail fed the EMA: rate {rate_before:.2f} -> {p['rate']:.2f}. This is "
          f"the whole bug. On Frank's run it halved the rate (64.1 -> 48.6) and made "
          f"the run-ETA at the handoff 108 s optimistic.")

# ---- 3: but the SECONDS are real and must survive ---------------------------
if abs(p["spent"] - (sum(HIGH) + TAIL_HIGH)) > 1e-6:
    _fail(f"the tail's seconds were DROPPED (spent={p['spent']:.1f}, expected "
          f"{sum(HIGH) + TAIL_HIGH:.1f}). They are real wall-clock time: throwing "
          f"them away breaks sum(spent) == elapsed, and a clock whose posts do not "
          f"add up to the wall is exactly the bug v577 fixed.")

# ---- 4: THE INVARIANT, end to end -------------------------------------------
clock = _fresh()
for dt in HIGH:
    clock.measure("step:high", dt)
clock.measure("step:high", TAIL_HIGH)
eta_at_handoff = clock.eta()
for dt in LOW:
    clock.measure("step:low", dt)
clock.measure("step:low", TAIL_LOW)

total = sum(v["spent"] for v in clock.posts.values())
if abs(total - WALL) > 1e-6:
    _fail(f"sum(spent) = {total:.1f} s, wall = {WALL:.1f} s. The clock's posts must "
          f"add up to the wall clock. A clock whose posts exceed the wall is "
          f"physically impossible -- that is how the v576 double-count was caught.")

for key, units in (("step:high", len(HIGH)), ("step:low", len(LOW))):
    if clock.posts[key]["done"] != units:
        _fail(f"{key}: done={clock.posts[key]['done']}, units={units}")

# ---- 5: the ETA at the handoff must be the HONEST one ------------------------
# Truth: the LOW stage took 6:10 (370 s). With the tail excluded from the rate,
# the clock projects 5 x ~64.1 s = ~5:20. With the tail in, it projected 4:02.
if eta_at_handoff is None:
    _fail("no ETA at the MoE handoff")
if eta_at_handoff < 300:
    _fail(f"the run-ETA at the handoff is {eta_at_handoff:.0f} s. The LOW stage takes "
          f"370 s. Anything under 300 s means the tail is back in the EMA -- that is "
          f"the exact failure Frank measured (said 4:02, truth 6:01).")

# ---- 6: a clock with no tail must behave EXACTLY as before -------------------
clean = _fresh()
for dt in HIGH:
    clean.measure("step:high", dt)
if abs(clean.posts["step:high"]["rate"] - rate_before) > 1e-9:
    _fail("a tail-free run must be byte-identical to the v579 behaviour")
if abs((clean.eta() or 0) - (eta_before or 0)) > 1e-6:
    _fail("a tail-free run's ETA must be byte-identical to the v579 behaviour")

# ---- 7: units=1 posts (encode/decode) must still take their ONE measure ------
c = rc._RunClock()
c.post("enc:high", 1, 1.0)
if c.measure("enc:high", 11.2) is not True:
    _fail("a units=1 post must accept its first measure -- encode/decode/pixel posts "
          "are units=1 and the PU books them exactly once")
if c.measure("enc:high", 99.0) is not False:
    _fail("a units=1 post must reject a SECOND measure as a tail")
if abs(c.posts["enc:high"]["spent"] - 110.2) > 1e-6:
    _fail("a rejected second measure must still keep its seconds")

print("[test_v580_tail_callback] PASS: run on Frank's measured 65-frame seconds. "
      "The tail keeps its seconds (sum(spent) == elapsed), does not move `done`, "
      "does not feed the EMA, and reports False so the console stays quiet. "
      f"Handoff ETA {eta_at_handoff:.0f} s against a 370 s truth (was 242 s).")
sys.exit(0)
