"""Guard v841 -- Polyhedron Attention (nodes/ph_attention.py).

Pins the five invariants of the attention patcher, and DRIVES them with real
tensors rather than reading the source:

  A1  layout round-trip. run_sage() must return exactly what core's own
      backend would have returned, for all four combinations of
      skip_reshape x skip_output_reshape. Getting this wrong does not crash,
      it scrambles the picture -- so it is checked against an independently
      computed reference, not against itself.

  A2  the slot holds a callable or nothing. Core calls
      transformer_options["optimized_attention_override"](func, ...) with no
      None-check, so parking a None there is a mid-run crash. On the default
      mode the key must be ABSENT.

  A3  a failing kernel degrades, it does not kill the run -- and it says so
      exactly once. With fallback off it must raise instead, so a
      measurement cannot be silently corrupted by a fallback.

  A4  a mask is never dropped. If the installed sageattention takes no
      attn_mask, run_sage must refuse (the caller then falls back for that
      call). Quietly dropping a mask is a wrong image, not a slow one.

  A5  unknown keywords are swallowed. core widened the attention signature
      once already and broke the incumbent node with "unexpected keyword
      argument 'transformer_options'".

Every check is then run a second time against a deliberately wounded
implementation, to prove the check can fail at all.

Runs without CUDA, without ComfyUI, without sageattention.
"""

import importlib.util
import os
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
    spec = importlib.util.spec_from_file_location("ph_attention_guard", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PH = _load()


# --------------------------------------------------------------------------
# stand-ins
# --------------------------------------------------------------------------

def fake_sage(q, k, v, tensor_layout="HND", is_causal=False,
              attn_mask=None, **kw):
    """A sage kernel that is exact, so any deviation is OUR layout bug."""
    if tensor_layout == "NHD":
        qq, kk, vv = (t.transpose(1, 2) for t in (q, k, v))
    else:
        qq, kk, vv = q, k, v
    out = F.scaled_dot_product_attention(qq, kk, vv, attn_mask=attn_mask)
    if tensor_layout == "NHD":
        out = out.transpose(1, 2)
    return out


def fake_sage_no_mask(q, k, v, tensor_layout="HND", is_causal=False, **kw):
    """A wheel without attn_mask support."""
    return fake_sage(q, k, v, tensor_layout=tensor_layout)


class FakeModel:
    def __init__(self, options=None):
        self.model_options = dict(options or {})

    def clone(self):
        return FakeModel(dict(self.model_options))


# --------------------------------------------------------------------------
# A1 -- layout round-trip
# --------------------------------------------------------------------------

def check_layout(run_sage):
    b, h, n, d = 2, 4, 64, 32
    torch.manual_seed(0)
    hnd = [torch.randn(b, h, n, d) for _ in range(3)]
    ref_hnd = F.scaled_dot_product_attention(*hnd)              # (b,h,n,d)
    ref_flat = ref_hnd.transpose(1, 2).reshape(b, n, h * d)     # (b,n,h*d)

    bad = []
    for skip_in in (False, True):
        for skip_out in (False, True):
            if skip_in:
                q, k, v = hnd
            else:
                q, k, v = (t.transpose(1, 2).reshape(b, n, h * d) for t in hnd)
            got = run_sage(fake_sage, {}, q, k, v, h,
                           skip_reshape=skip_in, skip_output_reshape=skip_out)
            want = ref_hnd if skip_out else ref_flat
            if got.shape != want.shape:
                bad.append("in=%s out=%s shape %s != %s"
                           % (skip_in, skip_out, tuple(got.shape),
                              tuple(want.shape)))
            elif not torch.allclose(got, want, atol=1e-5):
                bad.append("in=%s out=%s values drift by %.2e"
                           % (skip_in, skip_out,
                              (got - want).abs().max().item()))
    return bad


# --------------------------------------------------------------------------
# A2 -- the slot never holds a None
# --------------------------------------------------------------------------

def check_default_key(patch_impl):
    bad = []
    model = FakeModel({"transformer_options": {"optimized_attention_override":
                                               lambda *a, **k: None}})
    (out,) = patch_impl(model, PH.DEFAULT_MODE, True)
    tops = out.model_options.get("transformer_options", {})
    if "optimized_attention_override" in tops:
        bad.append("default mode left the key in place (value %r)"
                   % (tops["optimized_attention_override"],))
    # and the caller's dict must be untouched (copy-on-write)
    if "optimized_attention_override" not in \
            model.model_options["transformer_options"]:
        bad.append("patch() mutated the caller's transformer_options")
    return bad


def check_missing_backend(patch_impl):
    """A mode whose kernel cannot run must leave the model alone."""
    bad = []
    model = FakeModel()
    (out,) = patch_impl(model, "sage fp8 cuda++", True)
    tops = out.model_options.get("transformer_options", {})
    slot = tops.get("optimized_attention_override", "ABSENT")
    if slot is None:
        bad.append("a None was parked in the override slot")
    if slot != "ABSENT" and not callable(slot):
        bad.append("the override slot holds a non-callable %r" % (slot,))
    return bad


# --------------------------------------------------------------------------
# A3 -- degrade, and say it once
# --------------------------------------------------------------------------

def check_fallback(build_override):
    bad = []
    calls = {"func": 0, "reports": 0}

    def func(*a, **k):
        calls["func"] += 1
        return "from-core"

    def report(_exc):
        calls["reports"] += 1

    ov = build_override("sage fp8 cuda++", fallback=True, report=report)
    if ov is None:
        return ["build_override returned None for a known mode"]
    for _ in range(3):
        got = ov(func, torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4),
                 torch.zeros(1, 1, 4, 4), 1)
        if got != "from-core":
            bad.append("fallback did not reach the model's own backend")
    if calls["func"] != 3:
        bad.append("core backend called %d times, expected 3" % calls["func"])
    if calls["reports"] != 1:
        bad.append("reported %d times, expected exactly 1" % calls["reports"])

    ov_strict = build_override("sage fp8 cuda++", fallback=False, report=None)
    try:
        ov_strict(func, torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4),
                  torch.zeros(1, 1, 4, 4), 1)
        bad.append("fallback=False swallowed the error instead of raising")
    except Exception:
        pass
    return bad


# --------------------------------------------------------------------------
# A4 -- a mask is never dropped
# --------------------------------------------------------------------------

def check_mask(run_sage):
    b, h, n, d = 1, 2, 16, 8
    torch.manual_seed(1)
    q, k, v = (torch.randn(b, h, n, d) for _ in range(3))
    mask = torch.zeros(b, h, n, n)
    mask[..., n // 2:] = float("-inf")

    bad = []
    try:
        run_sage(fake_sage_no_mask, {}, q, k, v, h, mask=mask,
                 skip_reshape=True, skip_output_reshape=True)
        bad.append("a mask was accepted by a kernel that cannot mask")
    except NotImplementedError:
        pass
    except Exception as exc:
        bad.append("expected NotImplementedError, got %s" % type(exc).__name__)

    # and a kernel that CAN mask must actually receive it
    got = run_sage(fake_sage, {}, q, k, v, h, mask=mask,
                   skip_reshape=True, skip_output_reshape=True)
    want = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    if not torch.allclose(got, want, atol=1e-5):
        bad.append("the mask did not reach a kernel that supports it")
    return bad


# --------------------------------------------------------------------------
# A5 -- unknown keywords are swallowed
# --------------------------------------------------------------------------

def check_kwargs(build_override):
    bad = []
    seen = {}

    def func(*a, **k):
        seen.update(k)
        return "from-core"

    ov = build_override("sage fp8 cuda++", fallback=True, report=None)
    try:
        ov(func, torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4),
           torch.zeros(1, 1, 4, 4), 1,
           transformer_options={"sigmas": torch.zeros(1)},
           _inside_attn_wrapper=True)
    except TypeError as exc:
        bad.append("unknown keyword blew up the override: %s" % exc)
        return bad
    if "transformer_options" not in seen:
        bad.append("the fallback did not forward core's own keywords")
    return bad


# --------------------------------------------------------------------------
# wounded implementations -- proof the checks can fail
# --------------------------------------------------------------------------

def wounded_run_sage_layout(sage_fn, extra, q, k, v, heads, mask=None,
                            skip_reshape=False, skip_output_reshape=False):
    """WOUND: forgets the transpose on the way out of the HND path."""
    if skip_reshape:
        b, _, _, dim_head = q.shape
        layout = "HND"
    else:
        b, _, dim_head = q.shape
        dim_head //= heads
        q = q.reshape(b, -1, heads, dim_head)
        k = k.reshape(b, -1, heads, dim_head)
        v = v.reshape(b, -1, heads, dim_head)
        layout = "NHD"
    out = sage_fn(q, k, v, tensor_layout=layout)
    if layout == "HND":
        if not skip_output_reshape:
            out = out.reshape(out.shape[0], -1, heads * dim_head)   # no transpose
    else:
        if skip_output_reshape:
            out = out.transpose(1, 2)
        else:
            out = out.reshape(out.shape[0], -1, heads * dim_head)
    return out


def wounded_run_sage_mask(sage_fn, extra, q, k, v, heads, mask=None,
                          skip_reshape=False, skip_output_reshape=False):
    """WOUND: drops the mask when the kernel cannot take it."""
    kw = {"tensor_layout": "HND" if skip_reshape else "NHD"}
    if mask is not None and PH._accepts(sage_fn, "attn_mask"):
        kw["attn_mask"] = mask
    return PH.run_sage(sage_fn if mask is None or "attn_mask" in kw
                       else fake_sage_no_mask, extra, q, k, v, heads,
                       mask=mask if "attn_mask" in kw else None,
                       skip_reshape=skip_reshape,
                       skip_output_reshape=skip_output_reshape)


def wounded_build_override(mode, fallback=True, report=None):
    """WOUND: swallows the error and returns None instead of the real result."""
    router = PH.build_router(mode)

    def _ov(func, *args, **kwargs):
        try:
            return router(func, *args, **kwargs)
        except Exception:
            if report is not None:
                report(Exception("x"))
            return None
    return _ov


def wounded_patch(model, attention, fallback=True):
    """WOUND: parks a None in the slot instead of removing the key."""
    m = model.clone()
    tops = dict(m.model_options.get("transformer_options", {}) or {})
    tops["optimized_attention_override"] = None
    m.model_options["transformer_options"] = tops
    return (m,)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

print("[test_v841_attention] Polyhedron Attention -- patcher invariants")

node = PH.ULSAttention()

print("A1 layout round-trip")
for msg in check_layout(PH.run_sage):
    _fail("A1: " + msg)
else:
    if not check_layout(PH.run_sage):
        _ok("all four skip_reshape/skip_output_reshape combinations match core")

print("A2 the override slot")
res = check_default_key(node.patch) + check_missing_backend(node.patch)
for msg in res:
    _fail("A2: " + msg)
if not res:
    _ok("default removes the key; no None is ever parked; caller untouched")

print("A3 degrade once, or raise")
res = check_fallback(PH.build_override)
for msg in res:
    _fail("A3: " + msg)
if not res:
    _ok("falls back to core, reports exactly once, raises when told to")

print("A4 the mask")
res = check_mask(PH.run_sage)
for msg in res:
    _fail("A4: " + msg)
if not res:
    _ok("refuses a mask no kernel can take; forwards one that can")

print("A5 unknown keywords")
res = check_kwargs(PH.build_override)
for msg in res:
    _fail("A5: " + msg)
if not res:
    _ok("transformer_options and _inside_attn_wrapper pass through unharmed")

print("A6 canon is append-only")
order = list(PH.ULSAttention.INPUT_TYPES()["required"].keys())
if order[:1] != ["model"] or "attention" not in order or "fallback" not in order:
    _fail("A6: canon changed -- widgets are %r" % (order,))
elif order.index("attention") > order.index("fallback"):
    _fail("A6: attention must precede fallback (widgets_values is positional)")
else:
    _ok("model, attention, fallback -- unchanged order")

print("A7 the list is never empty and default comes first")
modes = PH.available_modes()
if not modes or modes[0] != PH.DEFAULT_MODE:
    _fail("A7: available_modes() = %r" % (modes,))
else:
    _ok("%d mode(s), '%s' first" % (len(modes), modes[0]))

# --- mutations ------------------------------------------------------------
print("MUTATIONS -- each wound must be caught")
MUTATIONS = [
    ("layout: no transpose out of HND",
     lambda: check_layout(wounded_run_sage_layout)),
    ("slot: parks a None",
     lambda: check_default_key(wounded_patch)),
    ("fallback: swallows and returns None",
     lambda: check_fallback(wounded_build_override)),
    ("mask: silently dropped",
     lambda: check_mask(wounded_run_sage_mask)),
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
    print("[test_v841_attention] FAIL -- %d problem(s)" % len(FAILS))
    sys.exit(1)
print("[test_v841_attention] PASS -- layout exact, slot never None, "
      "degrades once, mask never dropped, canon stable")
