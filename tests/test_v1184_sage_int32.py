# -*- coding: ascii -*-
"""Guard v1184 -- sageattention's 32-bit addressing, checked before the launch; a lost CUDA context said truly.

  A1 max_element_offset: shape x strides, storage offset left out, a view into a fused qkv buffer counted by ITS stride
  A2 sage_addressable: the H3 joint-refine geometry (56 heads x 128, qkv stride 21504): 99,865 tokens fit, 99,866 do not;
     the render pass (52k tokens) and the 19.10. refine (87k) fit; a contiguous k of the same length fits
  A3 run_sage: a q past the limit is a PassThrough that names the token count and the limit -- the kernel is NOT called;
     under the limit the kernel IS called (a fake kernel counts)
  A4 build_override: a PassThrough from the address check runs the model's own backend, says so once, no failure warning
  A5 build_override: an 'illegal memory access' from the kernel is NOT fallen back on -- ContextLost is raised, the
     report gets a ContextLost (not the kernel's error), and the model's own backend is never called
  A6 build_override: every other kernel error still falls back (v841 behaviour kept)
  A7 is_context_lost: the four device-death texts yes, an OOM and a TypeError no
  Z  ASCII
Script-style: exit 0 = pass.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


sys.path.insert(0, str(ROOT / "nodes"))
import ph_attention as PH  # noqa: E402

torch.set_num_threads(1)
LIM = 2 ** 31 - 1
H, D = 56, 128                 # MiniMax H3 (model_detection: heads x head_dim)
QKV = 3 * H * D                # the fused qkv buffer's row: 21504 elements per token


def h3_q(tokens):
    """A q as H3 hands it to optimized_attention: a VIEW into the fused (tokens, 3*H*D) buffer, (1, H, tokens, D)."""
    base = torch.empty((tokens, QKV), dtype=torch.int8)        # int8: 2.1 GB at 99k tokens, no copy below
    q = base[:, :H * D].view(tokens, H, D).transpose(0, 1).unsqueeze(0)
    return q, base


# ---- A1
t = torch.empty((4, 8), dtype=torch.int8)
_need(PH.max_element_offset(t) == 3 * 8 + 7, "A1 contiguous (4,8): offset 31")
v = t[:, :3]
_need(PH.max_element_offset(v) == 3 * 8 + 2, "A1 a column view keeps the parent's row stride (26)")
sl = t[1:, :]
_need(PH.max_element_offset(sl) == 2 * 8 + 7, "A1 the storage offset is left out (the kernel gets the data pointer)")
one = torch.empty((1, 1, 5, 1), dtype=torch.int8)
_need(PH.max_element_offset(one) == 4, "A1 size-1 dims do not count whatever their stride")

# ---- A2  (the arithmetic of the Neon refine, without allocating the real tensors)
def h3_offset(tokens):
    return (tokens - 1) * QKV + (H - 1) * D + (D - 1)

fit = max(n for n in range(99_000, 100_500) if h3_offset(n) <= LIM)
_need(fit == 99_865, "A2 H3 qkv-view geometry: the last token count that fits is 99,865 (got %d)" % fit)
_need(h3_offset(99_866) > LIM, "A2 99,866 tokens pass 2^31-1")
_need(h3_offset(52_416 + 6_838) <= LIM, "A2 the render pass (768x1344 x 175 f + text/refs/audio) fits")
_need(h3_offset(80_388 + 6_838) <= LIM, "A2 the 19.10. refine (141 f at 1056x1856 + text/refs/audio) fits")
_need(h3_offset(99_528 + 6_838) > LIM, "A2 the 20.10. refine (175 f at 1056x1856 + text/refs/audio) does not -- the crash")
q_small, _b = h3_q(1000)
ok, worst = PH.sage_addressable(q_small, q_small, q_small)
_need(ok and worst == h3_offset(1000), "A2 sage_addressable reads the view's stride (%d)" % worst)
k_contig = torch.empty((1, H, 110_000, D), dtype=torch.int8)
_need(PH.sage_addressable(k_contig)[0], "A2 a contiguous k of 110k tokens fits (stride H*D, not 3*H*D)")
del k_contig


# ---- A3  run_sage with a fake kernel
class _Called(Exception):
    pass


def fake_sage(q, k, v, **kw):
    raise _Called("kernel launched")


q_big, base_big = h3_q(99_870)
try:
    PH.run_sage(fake_sage, {}, q_big, q_big, q_big, H, skip_reshape=True, skip_output_reshape=True)
    _need(False, "A3 over the limit: the kernel must not be launched")
except PH.PassThrough as e:
    s = str(e)
    _need("99870" in s and str(LIM) in s and "32-bit" in s,
          "A3 over the limit: PassThrough names the tokens and the limit (%s)" % s[:90])
except _Called:
    _need(False, "A3 over the limit: the kernel was launched")
del q_big, base_big
q_ok, base_ok = h3_q(99_860)
try:
    PH.run_sage(fake_sage, {}, q_ok, q_ok, q_ok, H, skip_reshape=True, skip_output_reshape=True)
    _need(False, "A3 under the limit: the kernel must be launched")
except _Called:
    _need(True, "A3 under the limit (99,860 tokens): the kernel is launched")
except PH.PassThrough as e:
    _need(False, "A3 under the limit: wrongly passed through (%s)" % e)
del q_ok, base_ok


# ---- A4..A6  build_override with injected routers
def _override_with(router_exc):
    calls = {"own": 0, "report": [], "passed": []}

    def own_backend(*a, **k):
        calls["own"] += 1
        return "own"

    def router(func, *a, **k):
        raise router_exc

    saved = PH.build_router
    PH.build_router = lambda mode, sparse=None: router
    try:
        ov = PH.build_override("sage auto", fallback=True,
                               report=lambda e: calls["report"].append(e),
                               passthrough=lambda m, s: calls["passed"].append(str(s)))
    finally:
        PH.build_router = saved
    return ov, own_backend, calls


ov, own, c = _override_with(PH.PassThrough("sequence of 106366 tokens reaches element offset x, past 32-bit"))
r1 = ov(own, None, None, None, 8)
r2 = ov(own, None, None, None, 8)
_need(r1 == "own" and r2 == "own" and c["own"] == 2, "A4 address pass-through: the model's own backend serves the call")
_need(len(c["passed"]) == 1 and "32-bit" in c["passed"][0] and not c["report"],
      "A4 said ONCE as a pass-through, never as a failure")

ov, own, c = _override_with(RuntimeError("Triton Error [CUDA]: an illegal memory access was encountered"))
try:
    ov(own, None, None, None, 8)
    _need(False, "A5 illegal memory access: must raise")
except PH.ContextLost as e:
    _need("restart ComfyUI" in str(e) and "sage auto" in str(e), "A5 illegal memory access -> ContextLost names the mode and the restart")
except Exception as e:
    _need(False, "A5 wrong exception type %s" % type(e).__name__)
_need(c["own"] == 0, "A5 the model's own backend is NOT tried on a dead context")
_need(len(c["report"]) == 1 and isinstance(c["report"][0], PH.ContextLost), "A5 the report gets a ContextLost, once")

ov, own, c = _override_with(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"))
try:
    r6 = ov(own, None, None, None, 8)
except Exception as e:
    r6 = e
_need(r6 == "own" and c["own"] == 1 and len(c["report"]) == 1
      and not isinstance(c["report"][0], PH.ContextLost),
      "A6 an OOM still falls back with the failure warning (v841 kept) -- got %r" % (r6,))

# ---- A7
for s, want in (("an illegal memory access was encountered", True), ("CUDA error: an illegal address", True),
                ("unspecified launch failure", True), ("device-side assert triggered", True),
                ("CUDA out of memory", False), ("unexpected keyword argument", False)):
    _need(PH.is_context_lost(RuntimeError(s)) is want, "A7 is_context_lost(%r) is %s" % (s[:30], want))

# ---- Z
for p in ("nodes/ph_attention.py", "tests/test_v1184_sage_int32.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "Z  ASCII %s" % p)

print("%s: %d check(s) failed" % ("test_v1184_sage_int32", len(FAILS)) if FAILS
      else "test_v1184_sage_int32: all checks passed")
sys.exit(1 if FAILS else 0)
