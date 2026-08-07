"""Guard v573 -- the clock decides, the memory narrates.

FRANK'S THIRD COMPLETED RUN (v572) confirmed the pool hypothesis - chunk 2
ran 22.5 s, exactly chunk-1 speed, the grind is dead - and in the same
breath exposed the watch's crude verdict: `peak 14.7 GB vs 0.0 GB free ->
SPILLED` on that perfectly healthy chunk. Under pool quiet, free~0 IS the
healthy steady state (the pool holds our blocks between chunks), and
peak>free promptly mistook our own property for exhaustion, paying a
needless backoff to tile 512.

THE LAW, PINNED HERE:

1. THE PURE _watch_verdict IS THE JUDGE. 'slow' (wall clock > 2x chunk 1)
   is the ONE signal the driver cannot fake in either direction, and the
   ONLY outer trigger allowed to touch the tile - precedence over 'tight',
   because a genuinely slow chunk with an empty free reading must still
   back off. 'tight' (peak>free) is narrative. Executed below with the
   measured numbers of ALL THREE runs.

2. THE LINE STILL TELLS EVERYTHING - peak, pool-held (memory_reserved),
   free, seconds - and explains the confusing zero ONCE ('narrative, not
   a verdict'), so the next reader of a console does not re-fight this
   battle.

3. THE SHARP NET STAYS: the in-chunk _TileGrind detector (v572) remains
   armed and untouched - it fires mid-chunk on synchronized per-call time,
   the one place a grind shows before the chunk ends.

Script-style: exit 0 = pass.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v573_clockverdict] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    pu = _read("nodes", "ph_power_upscale.py")
    res = pu[pu.index("def _esrgan_resident("):pu.index("def _esrgan_pass(")]
    loop = res[res.index("for k, (i, j) in enumerate(_chunks("):]

    # ---- 1: the judge -------------------------------------------------------------
    if "def _watch_verdict(" not in pu:
        _fail("the pure watch verdict is gone")
    if "_watch_verdict(dt_ck, t_c1, peak, free_i)" not in loop:
        _fail("the loop must consult the pure verdict - one spelling of "
              "the law")
    if 'if verdict == "slow" and tile > 256:' not in loop:
        _fail("only a SLOW verdict may touch the tile")
    if '(verdict == "tight" or "spilled") and' in loop or "spilled or slow" in loop:
        _fail("peak>free crept back into the backoff - its false verdict "
              "on the healthy 22.5 s chunk is MEASURED")

    # ---- 2: the narrative line ------------------------------------------------------
    if "torch.cuda.memory_reserved(device)" not in loop:
        _fail("the line must name what the POOL holds - that is why free "
              "reads zero")
    if "narrative, not a verdict" not in loop:
        _fail("the confusing zero must be explained, once, in the line")
    if "pool_said" not in res:
        _fail("the pool explanation must be one-time - eleven repeats of "
              "it would be litany, not telemetry")

    # ---- 3: the sharp net stays -------------------------------------------------------
    if "raise _TileGrind()" not in res or "except _TileGrind:" not in loop:
        _fail("the in-chunk detector (v572) must survive - it is the sharp "
              "half of the net")

    # ---- the verdict, EXECUTED with all three runs' numbers ----------------------------
    src = pu[pu.index("def _watch_verdict("):pu.index("def _esrgan_resident(")]
    ns = {}
    exec(compile(src, "<_watch_verdict>", "exec"), ns)
    v = ns["_watch_verdict"]
    # v570's grind chunk: 119.1 s vs 34.1 s baseline, memory looked fine.
    if v(119.1, 34.1, 14.7e9, 15.6e9) != "slow":
        _fail("v570's measured grind must verdict slow")
    # v571's grind chunk: 241.8 s vs 31.9 s baseline.
    if v(241.8, 31.9, 14.7e9, 15.6e9) != "slow":
        _fail("v571's measured grind must verdict slow")
    # v572's FALSE POSITIVE: 22.5 s (healthy) with free reading 0.0 - the
    # case that forced this cut. Tight, never slow, never a backoff.
    if v(22.5, 23.4, 14.7e9, 0.0) != "tight":
        _fail("the measured false positive must verdict tight (narrative), "
              "not slow")
    # A genuinely slow chunk with an empty free reading: slow WINS.
    if v(119.1, 23.4, 14.7e9, 0.0) != "slow":
        _fail("precedence broke - a slow chunk with free=0 must still back "
              "off")
    # Healthy chunk, healthy memory.
    if v(23.4, 23.4, 14.7e9, 15.6e9) != "ok":
        _fail("a healthy chunk must verdict ok")
    # The 2x boundary is strict.
    if v(46.8, 23.4, 1.0, 2.0) != "ok" or v(46.81, 23.4, 1.0, 2.0) != "slow":
        _fail("the 2x boundary moved")

    print("[test_v573_clockverdict] OK - the pure verdict reproduces all "
          "three runs (two real grinds slow, the false positive tight, "
          "precedence holds), only slow backs off, the pool is narrated "
          "once, the in-chunk net survives")
    sys.exit(0)


if __name__ == "__main__":
    main()
