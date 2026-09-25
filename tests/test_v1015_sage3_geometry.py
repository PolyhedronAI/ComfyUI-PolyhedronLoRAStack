# -*- coding: ascii -*-
"""Guard v1015 -- Polyhedron Attention: the Sage 3 router serves only what
the kernel can serve, and checks what it cannot know.

Public issue #5 (25.09.2026): "Polyhedron Attention set to Sage Attention 3
produces black images from KSamplers; DisTorch Sage Patch and Attention
Optimizer do not." Read at the source (thu-ml SageAttention, mengqin fork,
Core 7a131a3a attention3_sage): the kernel has two head-dim instantiations
(64, 128), no sequence-length check, accepts-and-ignores attn_mask, and Core
sends every call with N <= 1024 to pytorch. Our router called the kernel at
every length and trusted it to raise. The node was validated on one wheel
and one geometry (Wan, 39168 tokens, dim 128); the public field is every
wheel on every geometry.

The promises, each DRIVEN with a fake kernel through build_router:

  G1  A sequence at or below SAGE3_SEQ_FLOOR is passed through, by name,
      and the kernel is NOT called.
  G2  A head dim outside SAGE3_HEAD_DIMS is passed through; 64 and 128 are
      served.
  G3  A masked call is passed through -- the kernel's signature has
      attn_mask, the kernel drops it, _accepts() must not be believed here.
  G4  A kernel that answers with NaN: the first call on that geometry raises
      Sage3NonFinite (so the override reports once and falls back), the
      second call on the SAME geometry is a PassThrough carrying the reason,
      and a DIFFERENT geometry is still served. The finite check runs once
      per geometry, not per call (one device sync, not hundreds).
  G5  The in-place `k -= k.mean(dim=-2)` the kernel performs on the caller's
      k is softmax-invariant: measured against sdpa in fp32, so "not copied"
      is a decision with a number behind it, not a hope.
  G6  Static pins: floor 1024 (Core's number), head dims (64, 128) (api.cu's
      two kernels), the mask gate and the finite gate exist in the router,
      the probe launches the head-dim-64 geometry for Sage 3.

Script-style: exit 0 = pass. Torch-only, no ComfyUI, no CUDA.
"""
import importlib.util
import os
import re
import sys

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
    spec = importlib.util.spec_from_file_location("ph_attention_v1015", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PH = _load()
SRC = open(os.path.join(ROOT, "nodes", "ph_attention.py"), encoding="utf-8").read()


# --------------------------------------------------------------------------
# a fake Sage 3 kernel: HND only, no tensor_layout keyword, records calls
# --------------------------------------------------------------------------

class FakeKernel:
    def __init__(self, nan_on=None):
        self.calls = []
        self.nan_on = nan_on or set()     # {(seq, dim)} -> answer NaN

    def __call__(self, q, k, v, attn_mask=None, is_causal=False, per_block_mean=True):
        self.calls.append(tuple(q.shape))
        out = F.scaled_dot_product_attention(q, k, v)
        if (q.shape[2], q.shape[3]) in self.nan_on:
            out = out.clone()
            out[..., 0, 0] = float("nan")
        return out


def _router(kernel):
    PH._sage3_function = lambda: kernel
    return PH.build_router(PH.SAGE3_MODE)


def _reference(q, k, v, heads, *a, **kw):
    raise AssertionError("the reference must not be reached by the router")


def _call(router, n, d, heads=2, mask=None, dtype=torch.float32):
    q = torch.randn(1, heads, n, d, dtype=dtype)
    k = torch.randn(1, heads, n, d, dtype=dtype)
    v = torch.randn(1, heads, n, d, dtype=dtype)
    return router(_reference, q, k, v, heads, mask, None, True, True)


# G1 -- sequence floor
kern = FakeKernel()
r = _router(kern)
try:
    _call(r, PH.SAGE3_SEQ_FLOOR, 128)
    _fail("G1 seq == floor was served")
except PH.PassThrough as e:
    if "floor" in str(e) and str(PH.SAGE3_SEQ_FLOOR) in str(e):
        _ok("G1 seq <= floor is a PassThrough that names the floor")
    else:
        _fail("G1 PassThrough without the floor in its reason: %s" % e)
if kern.calls:
    _fail("G1 the kernel was called for a short sequence")
else:
    _ok("G1 the kernel was not called")
out = _call(r, PH.SAGE3_SEQ_FLOOR + 1, 128)
if kern.calls == [(1, 2, PH.SAGE3_SEQ_FLOOR + 1, 128)] and out.shape == (1, 2, PH.SAGE3_SEQ_FLOOR + 1, 128):
    _ok("G1 seq floor+1 is served")
else:
    _fail("G1 seq floor+1 not served: %r" % (kern.calls,))

# G2 -- head dims
kern = FakeKernel()
r = _router(kern)
for d in (32, 96, 256):
    try:
        _call(r, 2048, d)
        _fail("G2 head dim %d was served" % d)
    except PH.PassThrough as e:
        if "head dim %d" % d in str(e):
            _ok("G2 head dim %d passed through by name" % d)
        else:
            _fail("G2 head dim %d: reason does not name it: %s" % (d, e))
for d in PH.SAGE3_HEAD_DIMS:
    _call(r, 2048, d)
if [c[3] for c in kern.calls] == list(PH.SAGE3_HEAD_DIMS):
    _ok("G2 head dims %s are served" % (PH.SAGE3_HEAD_DIMS,))
else:
    _fail("G2 served dims %r" % (kern.calls,))

# G3 -- mask
kern = FakeKernel()
r = _router(kern)
try:
    _call(r, 2048, 128, mask=torch.zeros(1, 1, 2048, 2048, dtype=torch.bool))
    _fail("G3 a masked call reached the kernel")
except PH.PassThrough as e:
    if "mask" in str(e):
        _ok("G3 a masked call is a PassThrough (the kernel would drop the mask)")
    else:
        _fail("G3 PassThrough reason does not mention the mask: %s" % e)
if kern.calls:
    _fail("G3 the kernel was called with a mask")

# G4 -- non-finite answer, per geometry, once
kern = FakeKernel(nan_on={(2048, 128)})
r = _router(kern)
try:
    _call(r, 2048, 128)
    _fail("G4 a NaN answer was returned as if it were an image")
except PH.Sage3NonFinite as e:
    if "non-finite" in str(e) and "2048" in str(e) and "128" in str(e):
        _ok("G4 first NaN call raises Sage3NonFinite naming the geometry")
    else:
        _fail("G4 Sage3NonFinite without the geometry: %s" % e)
except PH.PassThrough:
    _fail("G4 the FIRST non-finite call must raise (report), not pass through silently")
try:
    _call(r, 2048, 128)
    _fail("G4 second call on the dead geometry reached the kernel")
except PH.PassThrough as e:
    if "non-finite" in str(e):
        _ok("G4 second call on the same geometry is a PassThrough carrying the reason")
    else:
        _fail("G4 second call: PassThrough without the reason: %s" % e)
n_before = len(kern.calls)
out = _call(r, 4096, 128)
if len(kern.calls) == n_before + 1 and torch.isfinite(out).all():
    _ok("G4 a different geometry is still served")
else:
    _fail("G4 a different geometry was not served")
# once per geometry: wrap isfinite and count
calls_isfinite = []
_orig = torch.isfinite
def _counting(t):
    calls_isfinite.append(tuple(t.shape))
    return _orig(t)
torch.isfinite = _counting
try:
    kern = FakeKernel()
    r = _router(kern)
    for _ in range(5):
        _call(r, 2048, 128)
    for _ in range(3):
        _call(r, 3072, 64)
finally:
    torch.isfinite = _orig
if len(calls_isfinite) == 2:
    _ok("G4 the finite check runs once per geometry (8 calls, 2 checks)")
else:
    _fail("G4 finite check ran %d times for 8 calls on 2 geometries" % len(calls_isfinite))

# G5 -- the kernel's in-place mean subtraction on k is softmax-invariant
torch.manual_seed(0)
q = torch.randn(1, 2, 256, 64)
k = torch.randn(1, 2, 256, 64)
v = torch.randn(1, 2, 256, 64)
ref = F.scaled_dot_product_attention(q, k, v)
k2 = k.clone()
k2 -= k2.mean(dim=-2, keepdim=True)          # exactly preprocess_qkv's line
got = F.scaled_dot_product_attention(q, k2, v)
dev = (ref - got).abs().max().item()
if dev < 1e-5:
    _ok("G5 k -= k.mean(dim=-2) leaves the attention answer unchanged (max %.2e)" % dev)
else:
    _fail("G5 mean subtraction on k changed the answer by %.2e" % dev)

# G6 -- static pins
if PH.SAGE3_SEQ_FLOOR == 1024:
    _ok("G6 floor is Core's 1024")
else:
    _fail("G6 floor is %r, Core's is 1024" % (PH.SAGE3_SEQ_FLOOR,))
if tuple(PH.SAGE3_HEAD_DIMS) == (64, 128):
    _ok("G6 head dims are api.cu's (64, 128)")
else:
    _fail("G6 head dims %r" % (PH.SAGE3_HEAD_DIMS,))
router_src = SRC[SRC.index("def _sage3_router"):SRC.index("return _sage3_router")]
for needle, what in (('if v["mask"] is not None:', "mask gate"),
                     ("torch.isfinite(out)", "finite gate"),
                     ("n <= SAGE3_SEQ_FLOOR", "sequence floor"),
                     ("d not in SAGE3_HEAD_DIMS", "head-dim gate"),
                     ("mask=None", "no mask handed to run_sage")):
    if needle in router_src:
        _ok("G6 router carries the %s" % what)
    else:
        _fail("G6 router lost the %s (%r)" % (what, needle))
probe_src = SRC[SRC.index("def probe("):SRC.index("def sparse_selfcheck")]
if re.search(r"mode == SAGE3_MODE", probe_src) and "n, 64," in probe_src:
    _ok("G6 probe launches the head-dim-64 geometry for Sage 3")
else:
    _fail("G6 probe does not launch head dim 64 for Sage 3")

print()
if FAILS:
    print("test_v1015_sage3_geometry: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1015_sage3_geometry: PASS (G1-G6)")
