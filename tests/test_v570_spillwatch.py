"""Guard v570 -- the spill watch, and kinds that mean operations.

FRANK'S v569 RUN MEASURED THE TWO WOUNDS THIS CUT CLOSES:

  * final pixel chunk 1: 32.1 s. Chunks 2-4: ~100 s each. And the telemetry
    was BLIND to it - the peak/free check ran on chunk 1 only, and the one
    chunk that fit said "fits, no spill" for a pass that then ground for
    minutes. WDDM never OOMs: it spills to system RAM over PCIe, and torch's
    allocated counter cannot see driver-side paging. The wall clock is the
    one spill detector the driver cannot hide from.

  * the run ETA at stage high already "included" the final pass - at the
    borrowed rate of a 63 ms FIT chunk, weight-scaled. A 4x model forward is
    not a resize. The early ETA was a fairy tale until chunk 1 measured, and
    the bar jumped exactly where Frank was told it must not.

TWO CUTS, PINNED HERE:

1. THE SPILL WATCH. Every chunk reads free VRAM before it runs, resets the
   peak counter, and is timed. A chunk that spilled (peak > free) OR ran
   > 2x chunk 1 (the paging torch cannot see) says so with numbers, and the
   REMAINING chunks drop the ESRGAN tile (floor 256). free_memory stays
   ONCE per pass (the v564/v566 law) - reads are not asks.

2. KINDS MEAN OPERATIONS. A stage posts its pixel budget to the clock as
   'fit:<tag>' when no model runs and 'pix:<tag>' when one does (the same
   null-set as the stage loop: 'fit only' and 'model final'). The final
   pass stays 'pix:final'. Rung-2 borrowing is thereby correct by
   construction: model rates predict model passes, fit rates predict fits,
   and never each other. Proven below by RUNNING the actual _RunClock.

Script-style: exit 0 = pass.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v570_spillwatch] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    pu = _read("nodes", "ph_power_upscale.py")

    # ---- 1: the spill watch, structurally ----------------------------------------
    res = pu[pu.index("def _esrgan_resident("):pu.index("def _esrgan_pass(")]
    loop = res[res.index("for k, (i, j) in enumerate(_chunks("):]
    if "torch.cuda.mem_get_info(device)[0]" not in loop:
        _fail("free VRAM must be read BEFORE every chunk - it is not static "
              "(browser/DWM/preview nibble between chunks)")
    if "torch.cuda.reset_peak_memory_stats(device)" not in loop:
        _fail("the peak counter must reset per chunk - a pass-wide peak "
              "cannot attribute the spill")
    # AMENDED IN v573 (1st amendment): two pins of this guard grew.
    # (a) The wall-clock comparison moved INTO the pure _watch_verdict -
    #     one spelling of the law, exec'd by test_v573 with the measured
    #     numbers of all three runs. The belief (the clock IS the detector)
    #     is stronger than ever.
    # (b) 'spilled or slow' as the backoff trigger is GONE on purpose:
    #     under v572's pool quiet, free reads ~0 in the healthy steady
    #     state, and peak>free false-fired on a 22.5 s chunk (measured).
    #     Only 'slow' - and the in-chunk _TileGrind - may touch the tile;
    #     peak/free/pool stay in the line as narrative.
    if "_watch_verdict(dt_ck, t_c1, peak, free_i)" not in loop:
        _fail("the wall-clock detector is gone - torch cannot see WDDM "
              "paging; the pure _watch_verdict (clock first) is the judge")
    if ('if verdict == "slow" and tile > 256:' not in loop
            or "tile = max(256, tile // 2)" not in loop):
        _fail("a SLOW chunk must drop the tile for the REMAINING chunks "
              "(floor 256) - and ONLY a slow one: peak>free is narrative "
              "under pool quiet, its false verdict is measured")
    if "for the remaining chunks" not in loop:
        _fail("the backoff must be ANNOUNCED with its reason")
    if "spill backoff(s)" not in res:
        _fail("the pass summary must count its backoffs - telemetry says "
              "what precision AND what geometry ran")
    # the v564/v566 law survives: ONE ask, many reads
    if "mm.free_memory(need, device)          # ONCE, not per chunk" not in res:
        _fail("free_memory must stay ONCE per pass - reads are not asks")
    if res.count("mm.free_memory") != 1:
        _fail("a second free_memory ask crept into the resident pass")

    # ---- 2: kinds mean operations, structurally ----------------------------------
    if ('st["pix_kind"] = "pix" if um_st is not None else "fit"' not in pu):
        _fail("each stage must declare its pixel KIND at plan time")
    plan = pu[pu.index("plans = []"):pu.index("pbar = comfy.utils.ProgressBar")]
    if 'in ("fit only", "model final")' not in plan:
        _fail("the plan-time null-set must mirror the stage loop exactly - "
              "two spellings of one law will drift")
    if "clock.post(f\"{st['pix_kind']}:{st['tag']}\"" not in pu:
        _fail("the stage budget must be posted under its declared kind")
    if "_key=f\"{st['pix_kind']}:{st['tag']}\"" not in pu:
        _fail("the stage closure must measure under the SAME key it posted")
    # v884: the final block's opener is `if _final_runs:` -- ONE name that ORs
    # the dial with the joint-model detection (see test_v880 P7). The promise
    # here is unchanged; only the anchor follows the one-source form.
    fin = pu[pu.index("if _final_runs:",
                      pu.index("_free()   # v561")):]
    if '"pix:final"' not in fin:
        _fail("the final pass is a model forward - it posts and measures "
              "under kind 'pix'")

    # ---- 3: the ladder, EXECUTED - the actual class, real numbers ----------------
    # AMENDED IN v576 (2nd amendment): the clock moved to nodes/ph_runclock.py
    # (shared with the Sampler; PU re-exports). The whole module is the window
    # now - the two regex windows and their anchors are retired (lesson #3).
    src = _read("nodes", "ph_runclock.py")
    ns = {}
    exec(compile(src, "<ph_runclock>", "exec"), ns)
    Clock = ns["_RunClock"]

    t = {"v": 0.0}
    c = Clock(pbar=None, now=lambda: t["v"])
    # Frank's v569 shape: two fit stages (fast) + one final model pass.
    c.post("fit:high", 9, 848.0 * 848.0)
    c.post("fit:low", 9, 1104.0 * 1104.0)
    c.post("pix:final", 11, 4416.0 * 4416.0)
    for _ in range(9):
        t["v"] += 0.063
        c.measure("fit:high", 0.063)
    # THE v569 BUG, now impossible: with only fits measured, the final pass
    # must NOT borrow their rate. rate=None -> rung 3/4, never rung 2.
    if c._rate("pix:final") is not None:
        _fail(f"the final pass borrowed a rate from a FIT "
              f"({c._rate('pix:final'):.4f}s) - the v569 fairy tale is back")
    # And within its kind, borrowing must still work: a measured model stage
    # (model+fit mode) predicts the final pass, weight-scaled.
    c.post("pix:low", 9, 1104.0 * 1104.0)
    t["v"] += 24.0
    c.measure("pix:low", 24.0)
    r = c._rate("pix:final")
    want = 24.0 * (4416.0 * 4416.0) / (1104.0 * 1104.0)
    if r is None or abs(r - want) > 1e-6:
        _fail(f"within-kind borrowing broke: expected {want:.1f}s "
              f"weight-scaled, got {r}")
    # The EMA self-corrects the moment the pass measures (chunk 1 was 3x
    # faster than steady state on Frank's run - two measures must converge
    # toward the truth, not the first impression).
    t["v"] += 32.1
    c.measure("pix:final", 32.1)
    t["v"] += 99.6
    c.measure("pix:final", 99.6)
    r2 = c._rate("pix:final")
    if not (60.0 < r2 < 100.0):
        _fail(f"the EMA must move toward the measured steady state after "
              f"two chunks, got {r2:.1f}s")

    print("[test_v570_spillwatch] OK - per-chunk spill watch (peak/free/"
          "clock, tile backoff, one free_memory ask), kinds mean operations "
          "(ladder executed: no cross-kind borrow, within-kind scaled, EMA "
          "converges)")
    sys.exit(0)


if __name__ == "__main__":
    main()
