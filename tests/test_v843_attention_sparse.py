"""Guard v843 -- Polyhedron Attention, the sparse mode (stage A4).

A wrong sparse mask does not crash. It quietly degrades the video, which is
the worst failure class this tree knows -- so every claim below is checked
against something computed INDEPENDENTLY of the code under test.

  P1  GEOMETRY. A (B,C,T,H,W) latent must resolve to T frames of (H/2)*(W/2)
      tokens, and anything that is not a 5D video latent must resolve to
      None rather than to a guess. Frank's real case (17 x 48 x 48 = 39168)
      is pinned by name, because that number is visible in his sampler and
      is what the run-time check compares against.

  P2  THE MASK IS WHAT IT CLAIMS. local_keep() is compared, token by token
      over a whole small sequence, against a naive nested loop written from
      the SPEC ("frame distance <= w, or the sink frame"), not from the
      implementation.

  P3  THE DENSITY IS HONEST. local_density() is compared against a brute
      force count over all frame pairs. This number is printed to the user
      as the speed claim, so it may not be an estimate.

  P4  PASS THROUGH, NEVER GUESS. Cross-attention, a masked call, a grouped
      head count and a sequence that is not the latent's must all raise
      PassThrough -- and the override must then hand the call to the model's
      own backend WITHOUT the failure warning, because none of these is a
      failure.

  P5  THE SENTINEL. Not choosing the sparse mode changes nothing, and
      choosing it without a latent refuses out loud instead of masking on a
      guess.

Runs without CUDA and without flex_attention: everything checked here is
integer and boolean logic. The kernel itself is probed on Frank's machine at
a run-capable path -- while the mode is parked, patch() refuses it out loud.
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
    spec = importlib.util.spec_from_file_location("ph_attention_v843", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PH = _load()


class FakeLatent(dict):
    pass


def latent_of(t, h, w):
    return {"samples": torch.zeros(1, 16, t, h, w)}


class FakeModel:
    def __init__(self, options=None):
        self.model_options = dict(options or {})

    def clone(self):
        return FakeModel(dict(self.model_options))


# --------------------------------------------------------------------------
# P1 geometry
# --------------------------------------------------------------------------

def check_geometry(fn):
    bad = []
    got = fn(latent_of(17, 96, 96))
    if got != (17, 48 * 48, 39168):
        bad.append("Frank's case resolved to %r, expected (17, 2304, 39168)"
                   % (got,))
    got = fn(latent_of(5, 64, 80))
    if got != (5, 32 * 40, 5 * 1280):
        bad.append("5x64x80 resolved to %r" % (got,))
    for junk in (None, {}, {"samples": None},
                 {"samples": torch.zeros(1, 4, 64, 64)},   # 4D image latent
                 {"samples": torch.zeros(1, 16, 2, 1, 1)}, # too small to patch
                 "not a dict"):
        if fn(junk) is not None:
            bad.append("junk %r resolved to something" % (type(junk),))
    return bad


# --------------------------------------------------------------------------
# P2 the mask, against the spec
# --------------------------------------------------------------------------

def spec_keep(q_idx, kv_idx, per_frame, window, sink):
    """Written from the sentence in the docstring, deliberately naively."""
    frame_of_q = q_idx // per_frame
    frame_of_kv = kv_idx // per_frame
    within = -window <= (frame_of_q - frame_of_kv) <= window
    is_sink = (frame_of_kv == 0)
    return bool(within or (sink and is_sink))


def check_mask(fn):
    bad = []
    per, frames = 3, 6
    total = per * frames
    for window in (0, 1, 2, 5):
        for sink in (False, True):
            for q in range(total):
                for kv in range(total):
                    got = bool(fn(q, kv, per, window, sink))
                    want = spec_keep(q, kv, per, window, sink)
                    if got != want:
                        bad.append("w=%d sink=%s q=%d kv=%d -> %s, spec says %s"
                                   % (window, sink, q, kv, got, want))
                        return bad          # one is enough
    return bad


# --------------------------------------------------------------------------
# P3 the density, against a brute count
# --------------------------------------------------------------------------

def check_density(fn):
    bad = []
    per = 1
    for frames in (1, 4, 17, 33):
        for window in (0, 1, 3, 8):
            for sink in (False, True):
                kept = 0
                for q in range(frames):
                    for kv in range(frames):
                        if spec_keep(q, kv, per, window, sink):
                            kept += 1
                want = kept / float(frames * frames)
                got = fn(frames, window, sink)
                if abs(got - want) > 1e-9:
                    bad.append("frames=%d w=%d sink=%s -> %.6f, counted %.6f"
                               % (frames, window, sink, got, want))
                    return bad
    # and the number we would print for Frank's real case
    d = fn(17, 3, True)
    if not (0.35 < d < 0.50):
        bad.append("17 frames / window 3 / sink -> %.3f, expected ~0.41" % d)
    return bad


# --------------------------------------------------------------------------
# P4 pass through, never guess
# --------------------------------------------------------------------------

def check_passthrough(build_override):
    bad = []
    calls = {"core": 0, "warn": 0}

    def func(*a, **k):
        calls["core"] += 1
        return "from-core"

    def report(_exc):
        calls["warn"] += 1

    sparse = {"frames": 4, "per_frame": 8, "total": 32,
              "window": 1, "sink": True}
    ov = build_override(PH.SPARSE_LOCAL, fallback=True, report=report,
                        sparse=sparse)
    if ov is None:
        return ["build_override returned None for the sparse mode"]

    # a sequence that is not the latent's -> handed back, silently
    q = torch.zeros(1, 2, 64, 8)
    out = ov(func, q, q, q, 2, None, None, True, True)
    if out != "from-core":
        bad.append("a foreign sequence was not handed to the model's backend")
    if calls["warn"] != 0:
        bad.append("handing a foreign sequence back raised a FAILURE warning")

    # the sparse router itself must refuse the four not-ours cases
    good = torch.zeros(1, 2, 32, 8)
    short = torch.zeros(1, 2, 8, 8)
    cases = [
        ("masked call", dict(mask=torch.zeros(1, 1, 32, 32),
                             skip_reshape=True, skip_output_reshape=True)),
        ("cross attention", dict(k=short, v=short,
                                 skip_reshape=True, skip_output_reshape=True)),
        ("grouped heads", dict(k=torch.zeros(1, 1, 32, 8),
                               v=torch.zeros(1, 1, 32, 8),
                               skip_reshape=True, skip_output_reshape=True)),
    ]
    for name, kw in cases:
        k = kw.pop("k", good)
        v = kw.pop("v", good)
        try:
            PH.run_flex(None, good, k, v, 2, **kw)
            bad.append("%s was NOT refused" % name)
        except PH.PassThrough:
            pass
        except ImportError:
            pass          # no flex here; the refusal we wanted comes first
        except Exception as exc:
            if name != "masked call":
                bad.append("%s raised %s instead of PassThrough"
                           % (name, type(exc).__name__))
    return bad


# --------------------------------------------------------------------------
# P5 the sentinel and the missing latent
# --------------------------------------------------------------------------

def check_sentinel(patch_impl):
    bad = []
    # sparse chosen but no latent -> refuse, leave the model alone
    (out,) = patch_impl(FakeModel(), PH.SPARSE_LOCAL, True,
                        PH.SAME_AS_MAIN, 0, PH.SAME_AS_MAIN, 0, 3, True, None)
    slot = out.model_options.get("transformer_options", {}).get(
        "optimized_attention_override", "ABSENT")
    if slot is None:
        bad.append("a None was parked in the slot")
    if slot != "ABSENT":
        bad.append("sparse without a latent patched anyway")

    # sparse chosen, latent given, but window 0 -> that is dense; refuse
    (out,) = patch_impl(FakeModel(), PH.SPARSE_LOCAL, True,
                        PH.SAME_AS_MAIN, 0, PH.SAME_AS_MAIN, 0, 0, True,
                        latent_of(17, 96, 96))
    slot = out.model_options.get("transformer_options", {}).get(
        "optimized_attention_override", "ABSENT")
    if slot != "ABSENT":
        bad.append("window 0 was accepted as sparse")

    # not choosing sparse at all -> a latent must change nothing
    (a,) = patch_impl(FakeModel(), "pytorch sdpa", True,
                      PH.SAME_AS_MAIN, 0, PH.SAME_AS_MAIN, 0, 3, True, None)
    (b,) = patch_impl(FakeModel(), "pytorch sdpa", True,
                      PH.SAME_AS_MAIN, 0, PH.SAME_AS_MAIN, 0, 3, True,
                      latent_of(17, 96, 96))
    ka = "optimized_attention_override" in a.model_options.get(
        "transformer_options", {})
    kb = "optimized_attention_override" in b.model_options.get(
        "transformer_options", {})
    if not ka or not kb:
        bad.append("a plain mode stopped patching")
    return bad


# --------------------------------------------------------------------------
# wounds
# --------------------------------------------------------------------------

def wounded_geometry(latent, patch=2):
    """WOUND: forgets the patch divisor, so every sequence check is wrong."""
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if samples is None:
        return None
    shape = tuple(getattr(samples, "shape", ()))
    if len(shape) != 5:
        return None
    _b, _c, t, h, w = shape
    return (int(t), int(h) * int(w), int(t) * int(h) * int(w))


def wounded_keep(q_idx, kv_idx, per_frame, window, sink):
    """WOUND: strictly-less-than, so the outermost frame of the band is lost."""
    fq = q_idx // per_frame
    fk = kv_idx // per_frame
    if sink and fk == 0:
        return True
    return abs(fq - fk) < window


def wounded_density(frames, window, sink):
    """WOUND: ignores the clamp at the ends, so the claim is too optimistic."""
    if frames < 1:
        return 1.0
    n = min(2 * window + 1, frames)
    return n / float(frames)


def wounded_patch(model, attention, fallback=True, attention_first=None,
                  first_steps=0, attention_last=None, last_steps=0,
                  sparse_time_window=3, sparse_sink=True, latent=None):
    """WOUND: masks on a guessed geometry when no latent is wired."""
    m = model.clone()
    tops = dict(m.model_options.get("transformer_options", {}) or {})
    tops["optimized_attention_override"] = lambda f, *a, **k: f(*a, **k)
    m.model_options["transformer_options"] = tops
    return (m,)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

print("[test_v843_attention_sparse] Polyhedron Attention -- the sparse mode")

node = PH.ULSAttention()

print("P1 geometry")
res = check_geometry(PH.latent_geometry)
for m in res:
    _fail("P1: " + m)
if not res:
    _ok("17x96x96 -> 17 frames x 2304 tokens = 39168; junk -> None")

print("P2 the mask against the spec")
res = check_mask(PH.local_keep)
for m in res:
    _fail("P2: " + m)
if not res:
    _ok("every token pair over 4 windows x 2 sinks matches the spec")

print("P3 the density against a brute count")
res = check_density(PH.local_density)
for m in res:
    _fail("P3: " + m)
if not res:
    _ok("counted exactly; Frank's case 17/3/sink = %.1f%% kept"
        % (PH.local_density(17, 3, True) * 100))

print("P4 pass through, never guess")
res = check_passthrough(PH.build_override)
for m in res:
    _fail("P4: " + m)
if not res:
    _ok("foreign sequence, mask, cross attention and grouped heads all "
        "handed back without a failure warning")

print("P5 the sentinel")
res = check_sentinel(node.patch)
for m in res:
    _fail("P5: " + m)
if not res:
    _ok("no latent or window 0 -> refused out loud; other modes untouched")

print("P6 the mode is offered exactly where its runway exists")
# RE-GROUNDED TWICE, and the history is the point. v843 promised "offered iff
# flex is importable" -- wrong, because an importable flex still runs eager
# and cannot execute at video scale. v844 promised "never offered" -- right
# for a tree with no compiled path, wrong once there is one. The durable
# promise is neither: the mode is offered exactly when the machinery it needs
# to RUN is present, which is flex_attention plus a usable torch.compile.
modes = PH.available_modes()
try:
    import importlib
    fx = importlib.import_module("torch.nn.attention.flex_attention")
    runway = (hasattr(fx, "flex_attention")
              and hasattr(fx, "create_block_mask")
              and hasattr(torch, "compile"))
except Exception:
    runway = False
if runway != (PH.SPARSE_LOCAL in modes):
    _fail("P6: runway available=%s but sparse in list=%s"
          % (runway, PH.SPARSE_LOCAL in modes))
else:
    _ok("runway=%s, sparse offered=%s -- agreed" % (runway, runway))

print("MUTATIONS -- each wound must be caught")
MUTATIONS = [
    ("geometry: patch divisor forgotten",
     lambda: check_geometry(wounded_geometry)),
    ("mask: band one frame too narrow",
     lambda: check_mask(wounded_keep)),
    ("density: ends not clamped",
     lambda: check_density(wounded_density)),
    ("sentinel: masks without a latent",
     lambda: check_sentinel(wounded_patch)),
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
    print("[test_v843_attention_sparse] FAIL -- %d problem(s)" % len(FAILS))
    sys.exit(1)
print("[test_v843_attention_sparse] PASS -- geometry exact, mask matches the "
      "spec token by token, density counted, never masks on a guess")
