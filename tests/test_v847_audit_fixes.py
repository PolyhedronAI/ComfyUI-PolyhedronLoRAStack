"""Guard v847 -- Polyhedron Attention: six audit findings, pinned.

A full re-read of the node (06.08.) turned up two real defects that had been
shipping quietly, two honesty problems, and two hardening gaps. Quietly is the
operative word for both defects: neither crashes, both just make a feature
stop existing while the console keeps printing plausible lines. That is the
failure class this tree fears most, so each finding gets a promise here.

  Z1  A SPARSE STEP WINDOW IS NOT SILENTLY DROPPED. probe() builds its router
      WITHOUT the geometry, so probe(SPARSE_LOCAL) answers "unknown mode" --
      and patch()'s window loop believed it. From v845 until this was found,
      asking for sparse in a step window always printed "window dropped" and
      ran the main backend throughout. The window path must use the
      verification the sparse block already did.

  Z2  THE LIVE CHECK IS NOT CONSUMED BY A CALL IT NEVER SERVED. The v845 code
      set its once-only flag BEFORE knowing whether the call was passed
      through, so a first call that is cross-attention burned the single
      measurement and printed nothing. It only worked in the field because
      Wan runs self-attention first -- luck, not construction.

  Z3  XFORMERS IS NOT OFFERED WHEN COMFYUI HAS NOT WIRED IT. The package
      being installed is not the condition; core defines attention_xformers
      unconditionally but imports the module only when it selects xformers.
      Offering it anyway breaks the dropdown law we enforce for our own
      sparse mode.

  Z4  A REFUSAL NAMES THE REAL CAUSE. "did NOT run on this machine" is a lie
      when the package is installed and ComfyUI simply chose another backend.

  Z5  GROUPED-QUERY ATTENTION IS PASSED THROUGH, NOT CRASHED INTO. run_flex
      refuses it by name; run_sage walked into a reshape and let the fallback
      report a size error as a broken kernel.

  Z6  pick_mode SURVIVES A MISSING TOTAL. It is documented as pure and driven
      by guards, so it must not depend on locate_step's invariants holding.
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
    spec = importlib.util.spec_from_file_location("ph_attention_v847", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PH = _load()


class FakeModel:
    def __init__(self, options=None):
        self.model_options = dict(options or {})

    def clone(self):
        return FakeModel(dict(self.model_options))


def latent_of(t, h, w):
    return {"samples": torch.zeros(1, 16, t, h, w)}


def _slot(model):
    return model.model_options.get("transformer_options", {}).get(
        "optimized_attention_override", "ABSENT")


def plain_backend(q, k, v, heads, mask=None, attn_precision=None,
                  skip_reshape=False, skip_output_reshape=False, **kwargs):
    return F.scaled_dot_product_attention(q, k, v)


print("[test_v847_audit_fixes] six audit findings")

# --------------------------------------------------------------------------
# Z1 -- a sparse step window survives patch()
# --------------------------------------------------------------------------

print("Z1 sparse in a step window is not dropped")

# First: prove the trap is real, so this fixture cannot quietly stop testing
# anything if probe() ever changes.
_real_avail = torch.cuda.is_available
torch.cuda.is_available = lambda: True
try:
    trap_ok, trap_note = PH.probe(PH.SPARSE_LOCAL, force=True)
finally:
    torch.cuda.is_available = _real_avail
if trap_ok:
    _ok("probe() now judges sparse itself (%s) -- the trap below is moot but "
        "harmless" % trap_note)
else:
    _ok("confirmed: probe(SPARSE_LOCAL) still answers %r -- patch() must not "
        "ask it" % trap_note)

node = PH.ULSAttention()
buf = io.StringIO()
with redirect_stdout(buf):
    (out,) = node.patch(FakeModel(), "pytorch sdpa", True,
                        attention_first=PH.SPARSE_LOCAL, first_steps=1,
                        sparse_time_window=2, sparse_sink=True,
                        live_check=False, latent=latent_of(6, 16, 16))
text = buf.getvalue()
if _slot(out) == "ABSENT":
    _fail("Z1: nothing was patched at all")
elif "window dropped" in text:
    _fail("Z1: the sparse window was dropped although it verified: %r" % text)
elif "first 1 step(s): sparse" not in text:
    _fail("Z1: the plan line does not announce the sparse window: %r" % text)
else:
    _ok("sparse window verified and announced in the plan")

# and the inverse: a window whose sparse could NOT be prepared must still be
# dropped, loudly
buf = io.StringIO()
with redirect_stdout(buf):
    (out2,) = node.patch(FakeModel(), "pytorch sdpa", True,
                         attention_first=PH.SPARSE_LOCAL, first_steps=1,
                         live_check=False, latent=None)
text2 = buf.getvalue()
if _slot(out2) == "ABSENT":
    _fail("Z1: an unpreparable window took the main override down")
elif "first 1 step(s): sparse" in text2:
    _fail("Z1: a window was announced although sparse could not be prepared")
elif "latent" not in text2:
    _fail("Z1: the drop is silent about the reason: %r" % text2)
else:
    _ok("without a latent the window is dropped and says why")

# --------------------------------------------------------------------------
# Z2 -- the measurement is not burned on a pass-through call
# --------------------------------------------------------------------------

print("Z2 a passed-through first call does not consume the live check")

# The promise lives in live_compare: it must let PassThrough escape so the
# caller can tell "this call was never ours" from "the measurement failed".
q = torch.randn(1, 2, 64, 32)


def always_passes(func, *args, **kwargs):
    raise PH.PassThrough("not ours")


try:
    got = PH.live_compare(always_passes, plain_backend,
                          (q, q, q, 2), {})
    _fail("Z2: live_compare swallowed PassThrough (returned %r) -- the caller "
          "cannot tell 'not ours' from 'measurement failed'" % (got,))
except PH.PassThrough:
    _ok("live_compare re-raises PassThrough instead of reporting a result")

got2 = PH.live_compare(lambda f, *a, **k: (_ for _ in ()).throw(
    RuntimeError("boom")), plain_backend, (q, q, q, 2), {})
if got2 is not None:
    _fail("Z2: a real failure was not swallowed")
else:
    _ok("a real failure is still swallowed and returns None")

# end to end: first call passes through, second is measured
node2 = PH.ULSAttention()
buf = io.StringIO()
with redirect_stdout(buf):
    (m_on,) = node2.patch(FakeModel(), PH.SPARSE_LOCAL, True,
                          sparse_time_window=2, sparse_sink=True,
                          live_check=True, latent=latent_of(6, 16, 16))
ov = m_on.model_options["transformer_options"]["optimized_attention_override"]
short = torch.randn(1, 2, 99, 32)          # wrong sequence -> PassThrough
right = torch.randn(1, 2, 6 * 64, 32)      # the latent's own sequence
buf = io.StringIO()
with redirect_stdout(buf):
    ov(plain_backend, short, short.clone(), short.clone(), 2,
       None, None, True, True)
    first_text = buf.getvalue()
    ov(plain_backend, right, right.clone(), right.clone(), 2,
       None, None, True, True)
text = buf.getvalue()
if "live check" in first_text:
    _fail("Z2: the passed-through call was measured")
elif text.count("live check on the real call") != 1:
    _fail("Z2: after a pass-through, the real call was measured %d time(s), "
          "expected 1" % text.count("live check on the real call"))
else:
    _ok("pass-through not measured, the next real call is")

# --------------------------------------------------------------------------
# Z3 / Z4 -- xformers availability and an honest refusal
# --------------------------------------------------------------------------

print("Z3 xformers is offered only when core has it wired")

wired = PH._xformers_wired()
offered = PH.XFORMERS_MODE in PH.available_modes()
if wired is False and offered:
    _fail("Z3: core says xformers is NOT wired, yet the mode is offered")
elif not PH._core_attention("xformers") and offered:
    _fail("Z3: no core xformers backend at all, yet the mode is offered")
else:
    _ok("core wired=%r, backend=%r, offered=%s -- consistent"
        % (wired, bool(PH._core_attention("xformers")), offered))

if PH._xformers_wired() not in (True, False, None):
    _fail("Z3: the availability answer is not a tri-state: %r"
          % PH._xformers_wired())
else:
    _ok("the answer is True/False/None -- unknown stays offerable")

print("Z4 the refusal names the real cause")
lines = PH.explain_failure(PH.XFORMERS_MODE,
                           "NameError: name 'xformers' is not defined")
joined = " ".join(lines)
if not lines:
    _fail("Z4: the xformers NameError gets no explanation at all")
elif "NOT a missing package" not in joined:
    _fail("Z4: the explanation does not correct the wrong impression: %r"
          % joined)
elif "start-up" not in joined:
    _fail("Z4: the explanation does not say where the cause lies: %r" % joined)
else:
    _ok("the xformers NameError is explained as a start-up choice, not a "
        "missing install")

if PH.explain_failure("sage fp8 cuda++", "RuntimeError: no kernel image"):
    _fail("Z4: an unrelated failure was given the xformers explanation")
else:
    _ok("unrelated failures get no invented explanation")

# --------------------------------------------------------------------------
# Z5 -- grouped heads pass through instead of crashing
# --------------------------------------------------------------------------

print("Z5 grouped-query attention is passed through by run_sage")


def fake_sage(q, k, v, tensor_layout="HND", is_causal=False):
    return F.scaled_dot_product_attention(q, k, v)


qg = torch.randn(1, 8, 128, 64)
kg = torch.randn(1, 2, 128, 64)          # 4:1 grouped
try:
    PH.run_sage(fake_sage, {}, qg, kg, kg.clone(), 8, skip_reshape=True,
                skip_output_reshape=True)
    _fail("Z5: grouped heads were accepted (HND) -- the kernel got mismatched "
          "tensors")
except PH.PassThrough:
    _ok("HND grouped heads -> PassThrough")
except Exception as exc:
    _fail("Z5: HND grouped heads raised %s instead of PassThrough"
          % type(exc).__name__)

qf = torch.randn(1, 128, 8 * 64)
kf = torch.randn(1, 128, 2 * 64)
try:
    PH.run_sage(fake_sage, {}, qf, kf, kf.clone(), 8, skip_reshape=False,
                skip_output_reshape=False)
    _fail("Z5: grouped heads were accepted (NHD)")
except PH.PassThrough:
    _ok("NHD grouped heads -> PassThrough")
except Exception as exc:
    _fail("Z5: NHD grouped heads raised %s instead of PassThrough"
          % type(exc).__name__)

# the ordinary case must still work, bit-for-bit
qn = torch.randn(1, 4, 64, 32)
ref = F.scaled_dot_product_attention(qn, qn, qn)
got = PH.run_sage(fake_sage, {}, qn, qn.clone(), qn.clone(), 4,
                  skip_reshape=True, skip_output_reshape=True)
if not torch.equal(got, ref):
    _fail("Z5: the ordinary equal-head path changed")
else:
    _ok("equal heads still served, bit-identical")

# --------------------------------------------------------------------------
# Z6 -- pick_mode with a missing total
# --------------------------------------------------------------------------

print("Z6 pick_mode survives a missing total")
try:
    got = PH.pick_mode(3, None, "main", "first", 0, "last", 2)
    if got != "main":
        _fail("Z6: expected 'main' with no total, got %r" % got)
    else:
        _ok("no total -> main, no exception")
except Exception as exc:
    _fail("Z6: pick_mode raised %s on a missing total" % type(exc).__name__)

if PH.pick_mode(0, None, "main", "first", 1, "last", 0) != "first":
    _fail("Z6: a first window must still work without a total")
else:
    _ok("the first window does not depend on total")

# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------

if FAILS:
    print("[test_v847_audit_fixes] FAIL -- %d problem(s)" % len(FAILS))
    sys.exit(1)
print("[test_v847_audit_fixes] PASS -- sparse windows survive, the live check "
      "is not burned, xformers is honest, grouped heads pass through")
