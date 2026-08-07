"""Guard v572 -- pool quiet, and the grind caught in the act.

FRANK'S SECOND COMPLETED RUN (v571) sharpened the picture:

  * chunk 1 at tile 1024: 31.9 s. Chunk 2 at tile 1024: 241.8 s. Chunks
    3-10 at tile 512 (after the v570 watch's backoff): ~22 s STEADY -
    faster than the healthy 1024 chunk. The v566 up-front estimate stands
    measured-wrong on this card.
  * the v570 watch is correct and AFTER THE FACT: it fixed chunk 3 while
    chunk 2 donated four minutes to the driver.
  * the prime suspect for WHY chunk 2 grinds: ComfyUI runs cudaMallocAsync
    + DynamicVRAM hooks (the boot log says so), and v565-v571 called
    empty_cache after EVERY chunk - handing the pool back so the NEXT
    chunk's re-commit is the one WDDM slow-walks. Chunk 1 after the clean
    free_memory is fast; chunk 2 after our own housekeeping crawls. A
    hypothesis - and v572 makes the run itself test it.

THREE CUTS, PINNED HERE:

1. POOL QUIET. No empty_cache between chunks. The dels stay (the pool
   reuses same-size blocks); _free() stays in the backoffs, the fp16
   redo, and the finally - where releasing is the point.

2. THE IN-CHUNK DETECTOR. Every tile forward is timed SYNCHRONIZED (CUDA
   is async - an unsynced stopwatch times the launch, not the work). One
   call over the pure _grind_verdict threshold raises _TileGrind, caught
   BEFORE raise_non_oom; the tile halves and the CURRENT chunk is redone.
   Seconds lost, not 242.

3. EVERY BACKOFF RESETS THE DETECTOR (new size, new baseline), and the
   fp16 redo runs disarmed - it lives outside the retry net, and a grind
   escaping there would fall to the core fallback.

Script-style: exit 0 = pass.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v572_poolquiet] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    pu = _read("nodes", "ph_power_upscale.py")
    res = pu[pu.index("def _esrgan_resident("):pu.index("def _esrgan_pass(")]

    # ---- 1: pool quiet ------------------------------------------------------------
    seg = res[res.index("out.append(fit)"):res.index("dt_ck = time.monotonic()")]
    seg_code = [l for l in seg.splitlines() if not l.strip().startswith("#")]
    if any(l.strip() == "_free()" for l in seg_code):
        _fail("empty_cache is back between chunks - that hands the pool to "
              "the driver and chunk N+1 pays the re-commit (the measured "
              "chunk-2 grind)")
    if "del in_img, s, part, fit" not in seg:
        _fail("the dels must stay - dropping OUR references is what lets the "
              "pool reuse the blocks")
    if res.count("_free()") < 3:
        _fail("_free() must survive where releasing is the point: the "
              "backoffs, the fp16 redo, the finally")

    # ---- 2: the in-chunk detector ---------------------------------------------------
    if "class _TileGrind(Exception):" not in pu:
        _fail("_TileGrind is gone")
    if "def _grind_verdict(" not in pu:
        _fail("the pure grind predicate is gone")
    fn = res[res.index("def _model_fn(a):"):res.index("out, mid = [], None")]
    if "torch.cuda.synchronize(device)" not in fn:
        _fail("the stopwatch must synchronize - CUDA is async, an unsynced "
              "timer measures the launch, not the work")
    if "_grind_verdict(dc, grind[\"base\"], grind[\"armed\"])" not in fn:
        _fail("_model_fn must consult the pure predicate")
    if "raise _TileGrind()" not in fn:
        _fail("a grinding call must raise, mid-chunk - that is the whole cut")
    loop = res[res.index("while True:"):]
    ig = loop.index("except _TileGrind:")
    ie = loop.index("except Exception as exc:")
    if not (ig < ie):
        _fail("_TileGrind must be caught BEFORE raise_non_oom - paging is "
              "not an OOM")
    if "redoing THIS chunk" not in loop:
        _fail("the redo must be ANNOUNCED with its reason")

    # ---- 3: resets and the disarmed redo ---------------------------------------------
    if res.count('grind["base"] = None') < 3:
        _fail("every backoff (grind, OOM, outer watch) must reset the "
              "baseline - new size, new truth")
    if 'grind["armed"] = False' not in res or "_was_armed" not in res:
        _fail("the fp16 redo must run disarmed - it lives outside the retry "
              "net")

    # ---- the predicate, EXECUTED with the run's numbers -------------------------------
    src = pu[pu.index("def _grind_verdict("):pu.index("def _esrgan_resident(")]
    ns = {}
    exec(compile(src, "<_grind_verdict>", "exec"), ns)
    v = ns["_grind_verdict"]
    # chunk 2's grinding call vs chunk 1's healthy baseline (~31.9s/24 calls)
    if v(10.0, 1.33, True) is not True:
        _fail("the 242 s chunk's per-call grind must fire the verdict")
    if v(1.3, 1.33, True) is not False:
        _fail("a healthy call must never fire")
    if v(3.9, 1.3, True) is not False or v(3.91, 1.3, True) is not True:
        _fail("the 3x boundary moved")
    if v(0.2, 0.05, True) is not False:
        _fail("the 1.0 s absolute floor is gone - 3x of a 40 ms call is "
              "noise, not paging")
    if v(10.0, None, True) is not False:
        _fail("no baseline must mean no verdict")
    if v(10.0, 1.3, False) is not False:
        _fail("a disarmed detector must never fire")

    print("[test_v572_poolquiet] OK - pool quiet between chunks, the grind "
          "predicate fires on the measured numbers (and only on them), "
          "caught before raise_non_oom, every backoff resets the baseline, "
          "the fp16 redo runs disarmed")
    sys.exit(0)


if __name__ == "__main__":
    main()
