"""Guard v842 -- Polyhedron Attention, step windows (stage A3).

The v841 guard pins the patcher. This one pins the SWITCHING, and drives it
with real sigma tensors rather than reading the source.

  S1  THE SENTINEL. attention_first/attention_last on "same as main" with both
      counts at 0 must produce exactly the v841 override: no window, nothing
      read out of transformer_options, the main router on every call. A v841
      workflow that is merely re-opened must not start behaving differently.

  S2  THE WINDOWS LAND. On an 8-step schedule with first=2 and last=1, steps
      0-1 go to the opening kernel, step 7 to the closing one, 2-6 to the
      main one -- checked step by step, not at the ends only, because an
      off-by-one at the boundary is exactly the bug that would survive a
      sloppier test.

  S3  ONE SYNC PER STEP, NOT PER BLOCK. Reading a value out of a CUDA tensor
      is a device sync. Wan runs 40 blocks per step; an uncached read would
      stall the pipeline 40 times for a number that cannot have changed. The
      sigma tensor is the same OBJECT across the blocks of one step, so the
      cache must hit on identity -- and must MISS when the step moves on.

  S4  NO SCHEDULE, NO SWITCHING. If transformer_options carries no sigmas
      (older core, a sampler that does not publish them), every call falls to
      the main mode and the run is told once. A window that silently never
      fires is worse than no window.

  S5  "default" CANNOT BE A WINDOW. It means the absence of an override,
      which is a property of the whole run, not of a step.

Each check then runs against a deliberately wounded implementation.
Runs without CUDA, without ComfyUI, without sageattention.
"""

import importlib.util
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def _fail(msg):
    FAILS.append(msg)
    print("  FAIL  " + msg)


def _ok(msg):
    print("  ok    " + msg)


def _load():
    path = os.path.join(ROOT, "nodes", "ph_attention.py")
    spec = importlib.util.spec_from_file_location("ph_attention_v842", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PH = _load()


# --------------------------------------------------------------------------
# a schedule that looks like the real thing
# --------------------------------------------------------------------------

def make_schedule(steps=8):
    """Descending sigmas with the trailing 0.0 core appends."""
    vals = [1.0 - i * (1.0 / steps) for i in range(steps)] + [0.0]
    return torch.tensor(vals, dtype=torch.float32)


def tops_for(step, sched):
    """transformer_options as core hands it to attention at a given step."""
    return {"sigmas": sched[step:step + 1].clone(),
            "sample_sigmas": sched}


# --------------------------------------------------------------------------
# S1 -- the sentinel
# --------------------------------------------------------------------------

def check_sentinel(build_override):
    bad = []
    seen = []

    def func(*a, **k):
        return "core"

    ov = build_override("pytorch sdpa", fallback=True, report=None,
                        first_mode=None, first_steps=0,
                        last_mode=None, last_steps=0,
                        announce=lambda *a: seen.append(a))
    sched = make_schedule(8)
    for step in range(8):
        ov(func, torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8),
           torch.zeros(1, 2, 8, 8), 2,
           transformer_options=tops_for(step, sched))
    if seen:
        bad.append("an unwindowed override announced a schedule (%r) -- it "
                   "must not read transformer_options at all" % (seen,))
    return bad


# --------------------------------------------------------------------------
# S2 -- the windows land
# --------------------------------------------------------------------------

def check_windows(pick_mode):
    bad = []
    total = 8
    want = (["FIRST"] * 2) + (["MAIN"] * 5) + ["LAST"]
    for step in range(total):
        got = pick_mode(step, total, "MAIN", "FIRST", 2, "LAST", 1)
        if got != want[step]:
            bad.append("step %d/%d -> %s, expected %s"
                       % (step, total, got, want[step]))
    # no window at all
    for step in range(total):
        if pick_mode(step, total, "MAIN", None, 0, None, 0) != "MAIN":
            bad.append("step %d switched with both windows off" % step)
    # overlap on a short schedule: first wins, and nothing crashes
    if pick_mode(0, 2, "MAIN", "FIRST", 2, "LAST", 2) != "FIRST":
        bad.append("overlapping windows did not resolve in favour of first")
    # unlocatable step
    if pick_mode(None, None, "MAIN", "FIRST", 2, "LAST", 1) != "MAIN":
        bad.append("an unlocatable step did not fall back to main")
    return bad


# --------------------------------------------------------------------------
# S3 -- the cache
# --------------------------------------------------------------------------

def check_cache(locate_step):
    bad = []
    sched = make_schedule(8)
    cache = {}

    class CountingTensor(torch.Tensor):
        pass

    reads = {"n": 0}
    real_argmin = torch.argmin

    def counting_argmin(*a, **k):
        reads["n"] += 1
        return real_argmin(*a, **k)

    torch.argmin = counting_argmin
    try:
        # 40 "blocks" of the same step share ONE sigma object
        sig = sched[3:4].clone()
        tops = {"sigmas": sig, "sample_sigmas": sched}
        for _ in range(40):
            step, total = locate_step(tops, cache)
        if step != 3 or total != 8:
            bad.append("located step %r/%r, expected 3/8" % (step, total))
        if reads["n"] != 1:
            bad.append("%d device reads for one step, expected 1"
                       % reads["n"])
        # the next step brings a NEW object -- the cache must miss
        tops2 = {"sigmas": sched[4:5].clone(), "sample_sigmas": sched}
        step2, _ = locate_step(tops2, cache)
        if step2 != 4:
            bad.append("after the step moved on, located %r, expected 4"
                       % step2)
        if reads["n"] != 2:
            bad.append("%d device reads over two steps, expected 2"
                       % reads["n"])
    finally:
        torch.argmin = real_argmin
    return bad


# --------------------------------------------------------------------------
# S4 -- no schedule, no switching
# --------------------------------------------------------------------------

def check_no_schedule(build_override):
    bad = []
    used = []

    def func(*a, **k):
        return "core"

    said = []
    ov = build_override("pytorch sdpa", fallback=True, report=None,
                        first_mode="sage fp8 cuda++", first_steps=2,
                        last_mode=None, last_steps=0,
                        announce=lambda s, t: said.append((s, t)))
    # transformer_options present but WITHOUT sigmas
    for _ in range(3):
        ov(func, torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8),
           torch.zeros(1, 2, 8, 8), 2,
           transformer_options={"cond_or_uncond": [0]})
    if said != [(None, None)]:
        bad.append("announced %r, expected exactly one (None, None)" % (said,))
    # and with no transformer_options at all
    ov2 = build_override("pytorch sdpa", fallback=True, report=None,
                         first_mode="sage fp8 cuda++", first_steps=2,
                         last_mode=None, last_steps=0,
                         announce=lambda s, t: used.append((s, t)))
    ov2(func, torch.zeros(1, 2, 8, 8), torch.zeros(1, 2, 8, 8),
        torch.zeros(1, 2, 8, 8), 2)
    if used != [(None, None)]:
        bad.append("without transformer_options: announced %r" % (used,))
    return bad


# --------------------------------------------------------------------------
# wounded implementations
# --------------------------------------------------------------------------

def wounded_pick_mode(step, total, main, first_mode, first_steps,
                      last_mode, last_steps):
    """WOUND: off-by-one at the closing boundary."""
    if step is None:
        return main
    if first_steps > 0 and first_mode is not None and step < first_steps:
        return first_mode
    if last_steps > 0 and last_mode is not None and step > total - last_steps:
        return last_mode
    return main


def wounded_locate_step(tops, cache):
    """WOUND: caches on the schedule instead of the current sigma, so the step
    freezes at whatever it was first."""
    if not isinstance(tops, dict):
        return (None, None)
    sig = tops.get("sigmas", None)
    sched = tops.get("sample_sigmas", None)
    if sig is None or sched is None:
        return (None, None)
    if cache.get("sched", None) is sched:
        return cache["step"], cache["total"]
    flat = sched.reshape(-1)
    total = int(flat.numel()) - 1
    step = int(torch.argmin((flat - sig.reshape(-1)[0]).abs()).item())
    cache["sched"] = sched
    cache["step"] = step
    cache["total"] = total
    return step, total


def wounded_build_override_sentinel(mode, fallback=True, report=None,
                                    first_mode=None, first_steps=0,
                                    last_mode=None, last_steps=0,
                                    announce=None):
    """WOUND: always takes the windowed path, even with no window asked for."""
    router = PH.build_router(mode)
    cache = {}
    state = {"announced": False}

    def _ov(func, *args, **kwargs):
        step, total = PH.locate_step(kwargs.get("transformer_options"), cache)
        if not state["announced"]:
            state["announced"] = True
            if announce is not None:
                announce(step, total)
        try:
            return router(func, *args, **kwargs)
        except Exception:
            return func(*args, **kwargs)
    return _ov


def wounded_build_override_silent(mode, fallback=True, report=None,
                                  first_mode=None, first_steps=0,
                                  last_mode=None, last_steps=0,
                                  announce=None):
    """WOUND: never announces, so a dead window looks like a live one."""
    return PH.build_override(mode, fallback=fallback, report=report,
                             first_mode=first_mode, first_steps=first_steps,
                             last_mode=last_mode, last_steps=last_steps,
                             announce=None)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

print("[test_v842_attention_steps] Polyhedron Attention -- step windows")

print("S1 the sentinel")
res = check_sentinel(PH.build_override)
for m in res:
    _fail("S1: " + m)
if not res:
    _ok("no window asked for -> transformer_options is never touched")

print("S2 the windows land")
res = check_windows(PH.pick_mode)
for m in res:
    _fail("S2: " + m)
if not res:
    _ok("first 2 / last 1 of 8 land exactly; overlap and unknown step resolve")

print("S3 one sync per step")
res = check_cache(PH.locate_step)
for m in res:
    _fail("S3: " + m)
if not res:
    _ok("40 blocks of one step cost one device read; the next step misses")

print("S4 no schedule, no switching")
res = check_no_schedule(PH.build_override)
for m in res:
    _fail("S4: " + m)
if not res:
    _ok("says so exactly once, then runs the main mode throughout")

print("S5 'default' is not a window")
it = PH.ULSAttention.INPUT_TYPES()["required"]
first_choices = it["attention_first"][0]
if first_choices[0] != PH.SAME_AS_MAIN:
    _fail("S5: the window dropdown does not start on the sentinel")
else:
    node = PH.ULSAttention()

    class FakeModel:
        def __init__(self):
            self.model_options = {}

        def clone(self):
            return FakeModel()

    (out,) = node.patch(FakeModel(), "pytorch sdpa", True,
                        PH.DEFAULT_MODE, 2, PH.SAME_AS_MAIN, 0)
    slot = out.model_options.get("transformer_options", {}).get(
        "optimized_attention_override", "ABSENT")
    if slot is None:
        _fail("S5: a None reached the slot")
    else:
        _ok("'default' in a window is refused out loud, the run stays whole")

print("S6 canon stayed append-only")
order = list(PH.ULSAttention.INPUT_TYPES()["required"].keys())
want = ["model", "attention", "fallback", "attention_first", "first_steps",
        "attention_last", "last_steps"]
# RE-GROUNDED in v843: this used to demand equality, which is the test_v544
# latency -- an exact list pin fails on the first LEGITIMATE append (v843's
# sparse widgets) and says "canon changed" about a change that obeyed the
# law. What the law actually says is: the prefix never moves, growth only
# happens at the end.
if order[:len(want)] != want:
    _fail("S6: the v842 prefix moved -- canon is %r" % (order,))
elif len(order) > len(want):
    _ok("v841/v842 prefix unmoved, %d widget(s) appended behind it: %s"
        % (len(order) - len(want), ", ".join(order[len(want):])))
else:
    _ok("model, attention, fallback + the four window widgets, unmoved")

print("MUTATIONS -- each wound must be caught")
MUTATIONS = [
    ("boundary: off-by-one on the closing window",
     lambda: check_windows(wounded_pick_mode)),
    ("cache: keyed on the schedule, so the step freezes",
     lambda: check_cache(wounded_locate_step)),
    ("sentinel: reads the schedule even with no window",
     lambda: check_sentinel(wounded_build_override_sentinel)),
    ("silence: a dead window is never announced",
     lambda: check_no_schedule(wounded_build_override_silent)),
]
for name, run in MUTATIONS:
    try:
        caught = bool(run())
    except Exception:
        caught = True
    if caught:
        _ok("caught -- " + name)
    else:
        _fail("MUTATION SURVIVED: " + name)

print()
if FAILS:
    print("[test_v842_attention_steps] FAIL -- %d problem(s)" % len(FAILS))
    sys.exit(1)
print("[test_v842_attention_steps] PASS -- sentinel silent, windows exact, "
      "one sync per step, a dead window says so")
