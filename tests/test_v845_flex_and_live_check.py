"""Guard v845 -- Polyhedron Attention: the compiled flex path and the live check.

This guard exists because v843 shipped a sparse mode whose mask was proven
exactly right and whose EXECUTOR could not run at scale -- the machinery was
verified, the runway was not. So every promise here is about the runway, and
every one of them is DRIVEN, including the numerics, which torch lets us do on
the CPU.

  X1  THE MASK IS HONOURED BY THE COMPILED KERNEL. Not "flex was called" --
      the band-masked answer is compared against a dense SDPA reference
      carrying the SAME mask as an ordinary boolean attn_mask. Two unrelated
      machineries, one expected answer. This is the promise that makes the
      mode trustworthy: a block mask that keeps the wrong blocks does not
      raise, it degrades the video.

  X2  THE SELF-CHECK REFUSES A DISAGREEING MASK. sparse_selfcheck must return
      not-ok when the compiled answer and the dense reference part company,
      because patch() hangs the whole mode on that verdict.

  X3  NOTHING RUNS UNCOMPILED. The uncompiled paths are the 12.3 GB mask grid
      and the full-scores fallback; both are unusable at real scale. The mode
      must go through torch.compile, and the compiled objects must be built
      ONCE per process, not per call.

  X4  THE LIVE CHECK DOES NOT CHANGE THE RUN. It measures and prints; the
      output of the first call is bit-identical to what the same override
      produces with the check off, it fires exactly once, and any failure
      inside it is swallowed. A diagnostic that can break a render is worse
      than no diagnostic.

  X5  THE VERDICT TELLS THE TRUTH IN BOTH DIRECTIONS. Faster reads faster,
      slower reads slower and says so plainly, and the deviation of the
      sparse mode is labelled as omitted information rather than as error.
"""

import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout

import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def _fail(msg):
    FAILS.append(msg)
    print("  FAIL  " + msg)


def _ok(msg):
    print("  ok    " + msg)


def _load():
    path = os.path.join(ROOT, "nodes", "ph_attention.py")
    spec = importlib.util.spec_from_file_location("ph_attention_v845", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PH = _load()
DEV = torch.device("cpu")


class FakeModel:
    def __init__(self, options=None):
        self.model_options = dict(options or {})

    def clone(self):
        return FakeModel(dict(self.model_options))


def latent_of(t, h, w):
    return {"samples": torch.zeros(1, 16, t, h, w)}


def dense_reference(q, k, v, per_frame, window, sink):
    """The same band, expressed the ordinary way: a boolean attn_mask on
    dense SDPA. Written from the SPECIFICATION, not from our mask_mod."""
    n = q.shape[2]
    idx = torch.arange(n)
    fq = (idx // per_frame).view(-1, 1)
    fk = (idx // per_frame).view(1, -1)
    keep = (fq - fk).abs() <= window
    if sink:
        keep = keep | (fk == 0)
    return F.scaled_dot_product_attention(q, k, v,
                                          attn_mask=keep.view(1, 1, n, n))


def plain_backend(q, k, v, heads, mask=None, attn_precision=None,
                  skip_reshape=False, skip_output_reshape=False, **kwargs):
    return F.scaled_dot_product_attention(q, k, v)


print("[test_v845_flex_and_live_check] compiled flex path + live check")

# --------------------------------------------------------------------------
# X1 -- the compiled kernel honours the mask
# --------------------------------------------------------------------------

print("X1 the compiled band mask equals a dense reference carrying that band")
torch.manual_seed(845)
FR, PER, W = 6, 64, 2
N = FR * PER
q = torch.randn(1, 2, N, 32)
k = torch.randn(1, 2, N, 32)
v = torch.randn(1, 2, N, 32)

worst = None
for sink in (True, False):
    bm = PH.build_block_mask(FR, PER, W, sink, DEV)
    got = PH.run_flex(bm, q, k, v, 2, skip_reshape=True,
                      skip_output_reshape=True)
    ref = dense_reference(q, k, v, PER, W, sink)
    dev = float((got - ref).abs().max().item())
    worst = dev if worst is None else max(worst, dev)
    if dev > 1e-4:
        _fail("X1: sink=%s deviates by %.2e from the dense reference" %
              (sink, dev))
if worst is not None and worst <= 1e-4:
    _ok("both sink settings match a dense reference (worst %.1e)" % worst)

# a window that keeps everything must equal UNMASKED attention
bm = PH.build_block_mask(FR, PER, FR, False, DEV)
got = PH.run_flex(bm, q, k, v, 2, skip_reshape=True, skip_output_reshape=True)
full = F.scaled_dot_product_attention(q, k, v)
dev = float((got - full).abs().max().item())
if dev > 1e-4:
    _fail("X1: an all-keeping window differs from unmasked attention by %.2e"
          % dev)
else:
    _ok("a window wide enough to keep everything equals unmasked attention")

# --------------------------------------------------------------------------
# X2 -- the self-check refuses a disagreeing mask
# --------------------------------------------------------------------------

print("X2 sparse_selfcheck verifies, and would refuse a wrong mask")
ok, note, dev = PH.sparse_selfcheck(FR, W, True, DEV)
if not ok:
    _fail("X2: the self-check failed on a correct build: %s" % note)
elif dev is None or dev > 1e-4:
    _ok("self-check reports ok but a loose deviation (%r) -- worth watching"
        % dev)
else:
    _ok("self-check passes and reports %.1e" % dev)

# drive the refusal: hand it a mask builder that keeps the wrong band
_real_builder = PH.build_block_mask
try:
    def wrong_builder(frames, per_frame, window, sink, device):
        return _real_builder(frames, per_frame, max(0, window - 1), False,
                             device)
    PH.build_block_mask = wrong_builder
    ok2, note2, dev2 = PH.sparse_selfcheck(FR, W, True, DEV)
finally:
    PH.build_block_mask = _real_builder
if ok2:
    _fail("X2: a mask keeping the WRONG band was accepted (dev %r)" % dev2)
else:
    _ok("a wrong band is caught and refused (%s)" % note2.split(" -- ")[0])

# --------------------------------------------------------------------------
# X3 -- compiled, and compiled once
# --------------------------------------------------------------------------

print("X3 the flex path is compiled, and built once per process")
src = open(os.path.join(ROOT, "nodes", "ph_attention.py"),
           encoding="utf-8").read()
if "torch.compile(flex_attention" not in src:
    _fail("X3: flex_attention is not routed through torch.compile -- the "
          "eager path materialises the full scores matrix")
elif "torch.compile(create_block_mask" not in src:
    _fail("X3: create_block_mask is not compiled -- the uncompiled build "
          "broadcasts the full index grid (12.3 GB at video scale)")
else:
    _ok("both halves go through torch.compile")

f1, c1 = PH._flex_pair()
f2, c2 = PH._flex_pair()
if f1 is not f2 or c1 is not c2:
    _fail("X3: the compiled objects are rebuilt on every call -- each rebuild "
          "pays compilation again")
else:
    _ok("compiled objects are cached (same objects on a second call)")

# --------------------------------------------------------------------------
# X4 -- the live check cannot change or break the run
# --------------------------------------------------------------------------

print("X4 the live check measures without changing or endangering the run")
node = PH.ULSAttention()
# DELIBERATELY the sparse mode, not "pytorch sdpa": that mode's router returns
# exactly what the plain backend returns, so a bit-identity check on it would
# pass even if the measurement handed back the WRONG output. The mask makes
# the two answers genuinely different, which is what gives this fixture teeth.
# (Found by mutation M3, which slipped past the first version of this check.)
lat = latent_of(FR, 16, 16)

buf = io.StringIO()
with redirect_stdout(buf):
    (m_on,) = node.patch(FakeModel(), PH.SPARSE_LOCAL, True,
                         sparse_time_window=W, sparse_sink=True,
                         live_check=True, latent=lat)
    (m_off,) = node.patch(FakeModel(), PH.SPARSE_LOCAL, True,
                          sparse_time_window=W, sparse_sink=True,
                          live_check=False, latent=lat)
ov_on = m_on.model_options["transformer_options"][
    "optimized_attention_override"]
ov_off = m_off.model_options["transformer_options"][
    "optimized_attention_override"]

args = (q, k.clone(), v.clone(), 2, None, None, True, True)
buf = io.StringIO()
with redirect_stdout(buf):
    out_on = ov_on(plain_backend, *args)
    out_on2 = ov_on(plain_backend, *args)
text = buf.getvalue()
out_off = ov_off(plain_backend, *args)

plain = plain_backend(*args)
if torch.equal(out_off, plain):
    _fail("X4: fixture is toothless -- the router returns exactly what the "
          "plain backend returns, so a swapped output could not be seen")
if not torch.equal(out_on, out_off):
    _fail("X4: the measured call returned something different from the "
          "unmeasured one -- the diagnostic changed the render")
elif not torch.equal(out_on, out_on2):
    _fail("X4: two identical calls disagree")
else:
    _ok("output with the check on is bit-identical to the check off")

if text.count("live check on the real call") != 1:
    _fail("X4: the verdict printed %d times, expected exactly once"
          % text.count("live check on the real call"))
else:
    _ok("the verdict fires exactly once per patched model")

buf = io.StringIO()
with redirect_stdout(buf):
    out_off_check = ov_off(plain_backend, *args)
if "live check" in buf.getvalue():
    _fail("X4: live_check=False still measured")
else:
    _ok("live_check=False stays silent")


def exploding_router(func, *a, **kw):
    raise RuntimeError("kernel exploded during measurement")


try:
    got = PH.live_compare(exploding_router, plain_backend, args, {})
except Exception as exc:
    got = "RAISED: %s" % type(exc).__name__
if got is not None:
    _fail("X4: a failing measurement was not swallowed (%r) -- a diagnostic "
          "must never be able to break a render" % (got,))
else:
    _ok("a failure inside the measurement is swallowed, not raised")

# --------------------------------------------------------------------------
# X5 -- the verdict reads honestly in both directions
# --------------------------------------------------------------------------

print("X5 the verdict is honest about slower, faster and about sparse")
line = PH.format_verdict("pytorch sdpa (cudnn)", 10.0, 25.0, 0.0006)
if "faster" not in line or "2.50x" not in line:
    _fail("X5: a genuine speed-up did not read as faster: %r" % line)
else:
    _ok("faster reads faster")

line = PH.format_verdict("sparse local (video)", 30.0, 10.0, 0.44)
if "SLOWER" not in line or "costs time here" not in line:
    _fail("X5: a slower kernel did not say so: %r" % line)
elif "not kernel error" not in line:
    # RE-GROUNDED (v846): this pinned the exact phrase "omitted information",
    # so rewording the same statement broke it. What matters is that the line
    # says the deviation is NOT the kernel being wrong -- pin the claim, not
    # the sentence that carries it.
    _fail("X5: the sparse deviation is not marked as something other than "
          "kernel error: %r" % line)
else:
    _ok("slower says so, and the sparse deviation is labelled by construction")

line = PH.format_verdict("sage fp8 cuda++", 10.0, 10.2, 0.009)
if "same speed" not in line:
    _fail("X5: a tie was not reported as a tie: %r" % line)
elif "not kernel error" in line:
    _fail("X5: a non-sparse mode got the sparse deviation label: %r" % line)
else:
    _ok("a tie reads as a tie, and only sparse gets the mask label")

# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------

if FAILS:
    print("[test_v845_flex_and_live_check] FAIL -- %d problem(s)" % len(FAILS))
    sys.exit(1)
print("[test_v845_flex_and_live_check] PASS -- compiled mask verified against "
      "a dense reference, live check measures without touching the run")
