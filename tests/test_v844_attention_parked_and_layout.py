"""Guard v844 -- Polyhedron Attention: the dropdown law, sage layout compat.

Two promises, both driven rather than read:

  F1  THE DROPDOWN LAW, applied to our own invention. RE-GROUNDED in v845:
      this guard first pinned "sparse is never offered", which was the right
      promise while the tree had no compiled flex path and the wrong one the
      moment it got one (v845). What was always true, and is what this now
      pins, is the LAW behind it: the mode appears only when the machinery
      it needs to run is present -- flex_attention AND torch.compile -- and
      never on the strength of an import alone.

  F2  A REFUSAL IS LOUD AND COMPLETE. Whenever sparse cannot run -- no
      latent, a zero window, a mask that fails verification -- patch() says
      so, leaves the model on its own backend when sparse was the main mode,
      and drops it from a step window without taking the main mode down.
      Silence would leave the user believing a mask is in place that is not.

  F3  LAYOUT COMPATIBILITY IN run_sage. SageAttention 3 wheels expose
      sageattn3_blackwell(q, k, v, ...) with a FIXED (b, heads, seq, dim)
      layout and NO tensor_layout keyword. The old code passed the keyword
      unconditionally: a TypeError at the probe would have falsely
      disqualified the mode -- and 'fixing' that by just dropping the
      keyword would hand NHD tensors to an HND kernel, which does not crash,
      it silently scrambles the image. The promise: a kernel without the
      keyword receives HND tensors regardless of what core handed us, the
      answer comes back in the shape core asked for, and every combination
      is bit-identical to an independent reference. A kernel WITH the
      keyword still receives it. A masked call on a kernel without attn_mask
      support still refuses instead of dropping the mask.
"""

import importlib.util
import inspect
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
    spec = importlib.util.spec_from_file_location("ph_attention_v844", path)
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


# --------------------------------------------------------------------------
# F1 -- not offered
# --------------------------------------------------------------------------

print("[test_v844_attention_parked_and_layout] dropdown law + sage layout")
print("F1 the sparse mode is offered only where its runway exists")
try:
    import importlib as _il
    fx = _il.import_module("torch.nn.attention.flex_attention")
    flex_here = hasattr(fx, "flex_attention") and hasattr(fx,
                                                          "create_block_mask")
except Exception:
    flex_here = False
runway = flex_here and hasattr(torch, "compile")
if not flex_here:
    _fail("F1: flex_attention is not importable in this sandbox -- the "
          "check below would be vacuous")
modes = PH.available_modes()
if runway != (PH.SPARSE_LOCAL in modes):
    _fail("F1: runway=%s but sparse offered=%s" %
          (runway, PH.SPARSE_LOCAL in modes))
else:
    _ok("runway=%s, sparse offered=%s" % (runway, PH.SPARSE_LOCAL in modes))

# the law, not the outcome: strip torch.compile and the mode must vanish
_real_compile = torch.compile
try:
    del torch.compile
    modes_nc = PH.available_modes()
finally:
    torch.compile = _real_compile
if PH.SPARSE_LOCAL in modes_nc:
    _fail("F1: sparse is still offered without torch.compile -- an import of "
          "flex alone is not a runway")
else:
    _ok("without torch.compile the mode disappears from the list")


# --------------------------------------------------------------------------
# F2 -- the parked refusal
# --------------------------------------------------------------------------

print("F2 an impossible sparse setting is refused out loud, main survives")
node = PH.ULSAttention()

# no latent -> the mask cannot know the frame layout -> refuse, do not guess
buf = io.StringIO()
with redirect_stdout(buf):
    (out,) = node.patch(FakeModel(), PH.SPARSE_LOCAL, True,
                        sparse_time_window=3, sparse_sink=True, latent=None)
text = buf.getvalue()
if _slot(out) != "ABSENT":
    _fail("F2: sparse without a latent still installed an override")
elif "latent" not in text:
    _fail("F2: the refusal does not name the missing latent: %r" % text)
elif "unknown mode" in text:
    _fail("F2: the refusal fell through into the unknown-mode branch: %r"
          % text)
else:
    _ok("no latent: no override, refusal names the reason and is terminal")

# window 0 is dense -- refuse rather than pretend
buf = io.StringIO()
with redirect_stdout(buf):
    (out,) = node.patch(FakeModel(), PH.SPARSE_LOCAL, True,
                        sparse_time_window=0, sparse_sink=True,
                        latent=latent_of(6, 16, 16))
if _slot(out) != "ABSENT":
    _fail("F2: a zero window was accepted as sparse")
elif "0" not in buf.getvalue():
    _fail("F2: the zero-window refusal is silent")
else:
    _ok("window 0: refused out loud")

# an impossible sparse WINDOW must not take the main mode down
buf = io.StringIO()
with redirect_stdout(buf):
    (out,) = node.patch(FakeModel(), "pytorch sdpa", True,
                        attention_first=PH.SPARSE_LOCAL, first_steps=1,
                        latent=None, live_check=False)
text = buf.getvalue()
if _slot(out) == "ABSENT":
    _fail("F2: a dropped sparse window took the MAIN override down with it")
elif "latent" not in text:
    _fail("F2: the window was dropped silently: %r" % text)
else:
    _ok("window sparse dropped loudly, main mode still patched")


# --------------------------------------------------------------------------
# F3 -- run_sage layout compatibility, driven with real tensors
# --------------------------------------------------------------------------

print("F3 run_sage vs kernels with and without tensor_layout")

B, H, N, D = 2, 3, 16, 8
torch.manual_seed(844)
q_hnd = torch.randn(B, H, N, D)
k_hnd = torch.randn(B, H, N, D)
v_hnd = torch.randn(B, H, N, D)
ref_hnd = F.scaled_dot_product_attention(q_hnd, k_hnd, v_hnd)
ref_nhd_flat = ref_hnd.transpose(1, 2).reshape(B, N, H * D)

q_nhd_flat = q_hnd.transpose(1, 2).reshape(B, N, H * D)
k_nhd_flat = k_hnd.transpose(1, 2).reshape(B, N, H * D)
v_nhd_flat = v_hnd.transpose(1, 2).reshape(B, N, H * D)


def hnd_only_kernel(q, k, v, is_causal=False, per_block_mean=False):
    """A SageAttention-3-shaped stand-in: HND fixed, no tensor_layout, no
    attn_mask. Exact SDPA math, so a layout mistake shows as wrong bits,
    not as an exception."""
    assert q.shape[-1] == D and q.shape[-3] == H, "kernel fed non-HND"
    return F.scaled_dot_product_attention(q, k, v)


SEEN_LAYOUTS = []


def layout_kernel(q, k, v, tensor_layout="HND", is_causal=False):
    """A SageAttention-2-shaped stand-in that records the layout keyword."""
    SEEN_LAYOUTS.append(tensor_layout)
    if tensor_layout == "NHD":
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v)
    if tensor_layout == "NHD":
        out = out.transpose(1, 2)
    return out


cases = []
# (skip_reshape, skip_output_reshape) x expected
for so in (False, True):
    got = PH.run_sage(hnd_only_kernel, {}, q_hnd, k_hnd, v_hnd, H,
                      skip_reshape=True, skip_output_reshape=so)
    want = ref_hnd if so else ref_nhd_flat
    cases.append(("HND-only kernel, skip_reshape=True, so=%s" % so,
                  torch.equal(got, want)))
for so in (False, True):
    got = PH.run_sage(hnd_only_kernel, {}, q_nhd_flat, k_nhd_flat,
                      v_nhd_flat, H, skip_reshape=False,
                      skip_output_reshape=so)
    want = ref_hnd if so else ref_nhd_flat
    cases.append(("HND-only kernel, skip_reshape=False, so=%s" % so,
                  torch.equal(got, want)))

for name, good in cases:
    if not good:
        _fail("F3: %s came back wrong -- layout scrambled or reshaped "
              "incorrectly" % name)
if all(g for _n, g in cases):
    _ok("HND-only kernel: all four reshape combinations bit-identical to "
        "the reference")

SEEN_LAYOUTS[:] = []
PH.run_sage(layout_kernel, {}, q_hnd, k_hnd, v_hnd, H,
            skip_reshape=True, skip_output_reshape=True)
PH.run_sage(layout_kernel, {}, q_nhd_flat, k_nhd_flat, v_nhd_flat, H,
            skip_reshape=False, skip_output_reshape=False)
if SEEN_LAYOUTS != ["HND", "NHD"]:
    _fail("F3: a tensor_layout-aware kernel saw %r, expected ['HND', 'NHD']"
          % (SEEN_LAYOUTS,))
else:
    _ok("layout-aware kernel still receives the keyword, both layouts")

try:
    PH.run_sage(hnd_only_kernel, {}, q_hnd, k_hnd, v_hnd, H,
                mask=torch.zeros(N, N), skip_reshape=True,
                skip_output_reshape=True)
    _fail("F3: a masked call on a maskless kernel did not refuse -- the "
          "mask would have been dropped silently")
except NotImplementedError:
    _ok("masked call on a maskless kernel refuses instead of dropping")
except PH.PassThrough:
    _fail("F3: masked call raised PassThrough instead of the honest refusal")


# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------

if FAILS:
    print("[test_v844_attention_parked_and_layout] FAIL -- %d problem(s)"
          % len(FAILS))
    sys.exit(1)
print("[test_v844_attention_parked_and_layout] PASS -- runway law held, "
      "refusals loud, layouts proven against the reference")
