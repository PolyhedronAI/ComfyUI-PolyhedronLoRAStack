"""Guard v599 -- the chunk is MEASURED, and the clock outranks the allocator.

WHAT FRANK'S CARD MEASURED (768x768, rife47, float32, identical clip, only the
batch size changed):

    8 pairs  ->   39 ms/task
    16 pairs ->   43 ms/task
    64 pairs ->  296 ms/task

Flat, flat, cliff. Not the cost of a bigger batch -- the driver paging to system
RAM. v598 sized the chunk from `_chunk_size(free, w, h, bytes, in_flight=6)`,
where `in_flight=6` was a number I made up: "six full-size RGB tensors per task,
deliberately generous". The engine does not work in RGB. IFBlock.conv0 strides
the resolution down by four, block0 (c=192) only ever sees scale=8, ensemble
doubles the forwards -- the peak is not computable from outside, and every
constant I could have written would have been the same guess wearing a bigger
number. 18.9 s where 2.5 s was available. Eleventh case of the same disease.

THE CURE IS NOT A BETTER CONSTANT. This guard exists to make sure nobody --
including me -- ever puts one back.

THREE LAWS, all of them paid for in the upscaler between v566 and v573:

  1. NO GUESSED MEMORY CONSTANT. `_chunk_size` takes a peak it was GIVEN. It may
     not derive one from width, height, or a magic multiplier.

  2. THE PROBE IS REAL. The loop runs a small chunk first and reads
     max_memory_allocated. A probe that does not reset the peak counter, or does
     not synchronize, measures noise and calls it a fact.

  3. THE CLOCK OUTRANKS THE ALLOCATOR. WDDM never OOMs: it spills to system RAM
     over PCIe, and torch's counter cannot see driver-side paging (v570, on this
     same card). So a chunk that runs slower per task than the probe must be
     caught by the WALL CLOCK, and the node must back off and say so. And no
     empty_cache between chunks -- handing the pool back makes the NEXT chunk pay
     the re-commit, which cost four minutes in v572.
"""
import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "nodes", "ph_interpolate.py")
PY = open(SRC, encoding="utf-8").read()
TREE = ast.parse(PY)


def _fail(msg):
    print("[test_v599_chunk] FAIL: %s" % msg)
    sys.exit(1)


def lift(names):
    want, body, found = set(names), [], set()
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in want:
                    body.append(node)
                    found.add(t.id)
    assert not want - found, "not found in ph_interpolate.py: %s" % sorted(want - found)
    ns = {"math": __import__("math")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 SRC, "exec"), ns)
    return ns


ns = lift(["_chunk_size", "_PROBE_PAIRS", "_CHUNK_CEIL", "_GRIND_FACTOR"])
_chunk_size = ns["_chunk_size"]
_PROBE_PAIRS, _CHUNK_CEIL, _GRIND_FACTOR = (ns["_PROBE_PAIRS"], ns["_CHUNK_CEIL"],
                                            ns["_GRIND_FACTOR"])

# =========================================================================
# LAW 1 -- no guessed memory constant, anywhere in the signature
# =========================================================================
sig = None
for node in TREE.body:
    if isinstance(node, ast.FunctionDef) and node.name == "_chunk_size":
        sig = node
if sig is None:
    _fail("_chunk_size is gone")

args = [a.arg for a in sig.args.args]
if "in_flight" in args:
    _fail("`in_flight` is BACK. That constant is the entire v599 bug: it claimed a pair "
          "costs six RGB tensors and the field measured 7.6x. A chunk may not be sized "
          "from a number somebody believed.")
for banned in ("w", "h", "width", "height", "bytes_per_elem"):
    if banned in args:
        _fail("_chunk_size takes %r again -- it is deriving the peak from geometry instead "
              "of being handed a measured one. That derivation is exactly what was wrong." % banned)
if "peak_per_pair" not in args:
    _fail("_chunk_size must take a MEASURED peak_per_pair. Anything else is a guess with "
          "better manners.")

# The arithmetic must actually respond to the peak, or the measurement is decoration.
GB = 1024 ** 3
cheap = _chunk_size(8 * GB, 100 * 1024 ** 2)
dear = _chunk_size(8 * GB, 800 * 1024 ** 2)
if not cheap > dear:
    _fail("a costlier pair must produce a SMALLER chunk (%d vs %d). If the peak does not move "
          "the chunk, the probe is theatre." % (cheap, dear))
if _chunk_size(1024, 4 * GB) < 1:
    _fail("a chunk of 0 loops forever -- the floor must hold when the card is full")
if _chunk_size(80 * GB, 1) > _CHUNK_CEIL:
    _fail("the ceiling must hold -- an unbounded chunk is the 64-pair cliff all over again")

# Franks measured cliff, run through the real function: with a peak anywhere near
# what a 768 fp32 pair costs, 8 GB free must NOT yield 64.
for peak_mb in (200, 400, 800):
    got = _chunk_size(8 * GB, peak_mb * 1024 ** 2)
    if got >= 64:
        _fail("a %d MB/pair peak with 8 GB free still returns chunk %d. That is v598's answer, "
              "and Frank's card took 296 ms/task for it." % (peak_mb, got))

# =========================================================================
# LAW 2 -- the probe is real: reset, synchronize, read
# =========================================================================
body = PY[PY.index("chw = frames.permute"):PY.index("wall = _now() - t_start")]
# Strip comments before searching. The first cut of this guard failed on its OWN
# subject: the loop's comment says "empty_cache does NOT" and the guard read the
# word, not the code. A guard that cannot tell a promise from a call is exactly
# the thing it exists to prevent.
body = "\n".join(ln.split("#", 1)[0] for ln in body.splitlines())

for needed, why in (
    ("reset_peak_memory_stats",
     "the peak counter must be RESET before the probe, or it reports whatever the sampler "
     "left behind"),
    ("max_memory_allocated",
     "the probe must READ the real peak -- that reading is the only honest number in this file"),
    ("torch.cuda.synchronize",
     "without a synchronize the probe times kernel LAUNCHES, not kernel WORK, and every "
     "number after it is fiction"),
    ("mem_get_info",
     "the chunk must be sized against free VRAM as it is NOW, after the probe allocated"),
):
    if needed not in body:
        _fail("%s is missing from the loop -- %s" % (needed, why))

if "_chunk_size(" not in body:
    _fail("the loop never calls _chunk_size with the measured peak -- the probe measures and "
          "then throws the answer away")

# =========================================================================
# LAW 3 -- the clock outranks the allocator, and the pool stays quiet
# =========================================================================
if "empty_cache" in body:
    _fail("empty_cache is back between chunks. Handing the pool to the driver makes chunk N+1 "
          "pay the re-commit -- that is the v572 grind, measured on this exact card, and it "
          "cost four minutes.")

if "_GRIND_FACTOR" not in body:
    _fail("nothing watches the wall clock. The allocator cannot see WDDM paging (v570): a chunk "
          "that fits on paper and grinds in practice must be caught by TIME, or it is not "
          "caught at all.")
if "GRIND" not in body:
    _fail("the grind must be SAID. A node that silently halves its chunk teaches the user "
          "nothing, and the next person re-derives the same bug.")

# The peak is a CEILING, not a destination. The first cut of v599 sized the chunk
# from the measured peak and jumped straight to it -- and on a 64-task run that is
# ONE chunk, so when it ground, the watch fired into an empty room. 17.9 s instead
# of 18.9 s. That is v570's mistake, verbatim, and structure checks alone did not
# catch it: the code LOOKED right. Only running it showed the watch arriving after
# the battle.
if "chunk = _chunk_size(" in body:
    _fail("the measured peak is being assigned straight to `chunk`. It is a CEILING (`cap`), not "
          "a destination. Leap to it and a single bad chunk eats the whole run before the clock "
          "can rule -- the watch fires into an empty room. It must be climbed.")
if "chunk = min(cap, max(1, chunk * 2))" not in body:
    _fail("the ramp is gone. The chunk must DOUBLE from the probe toward the ceiling, one timed "
          "step at a time, so a grind costs one step and not the run.")
if "chunk = good" not in body or "cap = good" not in body:
    _fail("the grind verdict must fall back to the last size that ran CLEAN and stop climbing. "
          "A warning without a change is a spectator, and a ceiling that ground once has been "
          "measured wrong -- it does not get a second try.")

# =========================================================================
# LAW 3b -- RUN IT. The structure above was all present in the cut that
# still took 17.9 s. A guard that only reads is a guard that can be fooled
# by code that looks right. This one drives the REAL loop, lifted from the
# source, against Frank's measured cliff.
# =========================================================================
import textwrap
import types

CLIFF, FAST, SLOW = 16, 40.0, 296.0     # <=16 pairs fit; beyond that the card pages
N_LIVE = 64
loop_src = textwrap.dedent(PY[PY.index("        # THE CHUNK CLIMBS"):
                              PY.index("        wall = _now() - t_start")])


def _drive(peak_per_pair_mb):
    """Run the real loop. `peak_per_pair_mb` is what the allocator REPORTS --
    pass a lie to prove the wall clock still saves the run."""
    clk, peak_hi = [0.0], [0]

    class Cu:
        synchronize = staticmethod(lambda d=None: None)
        reset_peak_memory_stats = staticmethod(lambda d=None: peak_hi.__setitem__(0, 0))
        memory_allocated = staticmethod(lambda d=None: 0)
        max_memory_allocated = staticmethod(lambda d=None: peak_hi[0])
        mem_get_info = staticmethod(lambda d=None: (int(8.1 * 1024 ** 3), 16 * 1024 ** 3))

    class T(list):
        to = float = clamp = permute = view = lambda self, *a, **k: self
        __getitem__ = lambda self, i: "f"

    sizes = []

    def mdl(f0, f1, ts, scales, fm, en):
        k = len(f0)
        sizes.append(k)
        clk[0] += k * (FAST if k <= CLIFF else SLOW) / 1000.0
        peak_hi[0] = k * peak_per_pair_mb * 1024 ** 2
        return T(range(k))

    class O:
        device, dtype = "cpu", "f32"
        __setitem__ = lambda self, k, v: None

    env = dict(ns)
    env.update({
        "torch": types.SimpleNamespace(
            cuda=Cu, cat=lambda xs: T(xs), tensor=lambda *a, **k: T(),
            inference_mode=lambda: __import__("contextlib").nullcontext()),
        "model": mdl, "out": O(), "device": types.SimpleNamespace(type="cuda"),
        "live": [(i, 0.5) for i in range(N_LIVE)], "frames": T(), "chw": T(),
        "multiplier": 2, "dtype": types.SimpleNamespace(itemsize=4),
        "scales": [8, 4, 2, 1], "fast_mode": False, "ensemble": True,
        "w": 768, "h": 768, "pad_w": 0, "pad_h": 0, "_RunClock": None,
        "_now": lambda: clk[0], "t_start": 0.0, "t_said": 0.0, "done": 0,
        "print": lambda *a: None,
    })
    exec(loop_src, env)
    return env["done"], env["chunk"], clk[0], sizes


LEAP = N_LIVE * SLOW / 1000.0          # what v598 paid: 64 pairs in one go

# The honest case: the allocator reports the truth, the ramp finds the ceiling.
done, chunk, honest, sizes = _drive(420)
if done != N_LIVE:
    _fail("%d of %d tasks were computed -- the loop drops work when the chunk moves" % (done, N_LIVE))

# `pos += len(batch)`, not `pos += chunk`. Today these are equivalent, because pos
# is advanced BEFORE the chunk is resized -- I mutated it to `pos += chunk` and
# nothing broke, and saying so is worth more than a scary comment that turns out
# to be false. But the equivalence rests entirely on that ordering. Move the
# resize one line up and `pos += chunk` starts skipping tasks in silence. Pin the
# form that cannot break, not the one that happens to work.
if "pos += len(batch)" not in body:
    _fail("`pos` must advance by the batch actually computed, never by the chunk VARIABLE. The "
          "two agree only because the resize happens after this line -- that is an accident of "
          "ordering, and the next edit will not know it was load-bearing.")
if chunk > CLIFF:
    _fail("the run settled at chunk %d, past the cliff at %d. The probe measured and learned "
          "nothing." % (chunk, CLIFF))
if honest >= LEAP * 0.5:
    _fail("%.1f s against v598's %.1f s -- the measurement did not buy the run anything"
          % (honest, LEAP))

# The nasty case: the allocator LIES (reports 40 MB for a 420 MB pair -- exactly
# the scale of v598's error). torch cannot see WDDM paging, so only the clock is
# left. THIS is the case that catches a leap, and it must be judged on the CHUNK
# SEQUENCE, not on a stopwatch threshold: the first cut of this guard used
# `secs >= LEAP` and a leap that cost 17.9 s slid under 18.9 s and passed. A
# guard with a threshold the bug can duck is not a guard.
done, chunk, secs, sizes = _drive(40)
if done != N_LIVE:
    _fail("%d of %d tasks computed on the back-off path -- the fallback drops work" % (done, N_LIVE))
if len(sizes) < 2:
    _fail("the whole run went out in one chunk. Nothing was climbed, nothing could be watched, "
          "and a bad ceiling costs the entire run before the clock gets a vote.")
if sizes[1] > 2 * sizes[0]:
    _fail("the step after the probe jumped from %d to %d pair(s). The measured peak is a CEILING, "
          "not a destination -- it may only be approached by DOUBLING. Leap to it and one bad "
          "chunk eats the run: that leap measured 17.9 s against v598's 18.9 s, and a stopwatch "
          "check waved it through." % (sizes[0], sizes[1]))
past_cliff = sum(k for k in sizes if k > CLIFF)
if past_cliff > 2 * CLIFF:
    _fail("%d tasks were computed at a chunk size past the cliff. The clock is allowed ONE bad "
          "step -- that is the price of finding the edge. It is not allowed to donate the run "
          "to the driver." % past_cliff)
if chunk > CLIFF:
    _fail("the allocator lied and the clock did not catch it: settled at chunk %d, past the cliff "
          "at %d. WDDM pages instead of failing; if the wall clock does not rule here, NOTHING "
          "does." % (chunk, CLIFF))

# =========================================================================
# LAW 4 -- the DESCRIPTION may not claim a pad that was revoked
# =========================================================================
desc = PY[PY.index("DESCRIPTION = ("):PY.index("DESCRIPTION = (") + 1400]
if "Pads the picture" in desc:
    _fail("the DESCRIPTION still says the node pads the canvas. v598 REVOKED that pad -- our "
          "ring measured 4.18 dB worse than the engine's, because the network was trained with "
          "the zeros ring. The code was fixed and the prose was not. That is the eleventh case "
          "of the same disease, and it is the one the user actually reads.")

print("[test_v599_chunk] PASS: the chunk is measured, not believed -- and the loop was RUN, not "
      "just read. `in_flight` is gone; a costlier pair yields a smaller chunk (%d -> %d); the "
      "probe resets/synchronizes/reads a real peak; the peak is a ceiling the chunk CLIMBS, so a "
      "grind costs one step and not the run (%.1fs honest, %.1fs when the allocator lies, against "
      "v598's %.1fs); no task is lost when the chunk moves mid-run; the pool stays quiet between "
      "chunks; and the DESCRIPTION no longer claims a pad the node does not do."
      % (cheap, dear, honest, secs, LEAP))
sys.exit(0)
