# -*- coding: ascii -*-
"""v933 -- the head bank goes onto a model, and refuses to guess.

Two halves:

  THE PARTITION IS FORCED, NOT CHOSEN. With only 4 and 8 admitted and 32 fine
  steps fixed, a = 8 - nfe and b = 2*nfe - 8 follow, and both must be
  non-negative. So the legal step counts are exactly 4..8 -- not a taste, a
  consequence. Pinned by deriving it independently here and by checking the
  three counts the release published.

  THE SWAP IS DRIVEN BY SIGMA, NOT BY A COUNTER. A counter cannot survive a
  looping sampler, a resume or a split schedule. The guard therefore drives
  forwards OUT OF ORDER and repeatedly, and pins that each one still picks the
  head belonging to its sigma.

And it fails closed three ways: off-grid sigma, the final sigma (which starts
no block), and a forward with no sigma at all because the wrapper is missing.
Every one of those would otherwise produce noise that looks like a bad seed.

torch. comfy is stubbed to the one import attach() makes; the model is
built from real nn.Modules (v935, see section C).
"""
import os
import pathlib
import sys
import types

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_comfy = types.ModuleType("comfy")
_pe = types.ModuleType("comfy.patcher_extension")


class _W(object):
    DIFFUSION_MODEL = "dm"


_pe.WrappersMP = _W
_comfy.patcher_extension = _pe
sys.modules.setdefault("comfy", _comfy)
sys.modules.setdefault("comfy.patcher_extension", _pe)

import uls_pdd_apply as A  # noqa: E402
import uls_pdd_math as M  # noqa: E402

FAILED = []
CHECKS = []


def _fail(msg):
    FAILED.append(msg)
    CHECKS.append(msg)
    print("FAIL: {}".format(msg))


def _ok(msg):
    CHECKS.append(msg)
    print("ok  : {}".format(msg))


def _need(cond, msg):
    _ok(msg) if cond else _fail(msg)


# --- A: the partition is forced -------------------------------------------------
_need(M.LEGAL_NFE == (4, 5, 6, 7, 8),
      "A  the legal step counts are exactly 4..8: %s" % (M.LEGAL_NFE,))

# derived here, independently of the module
derived = []
for n in range(1, 33):
    a, b = 8 - n, 2 * n - 8
    if a >= 0 and b >= 0 and 8 * a + 4 * b == 32:
        derived.append(n)
_need(tuple(derived) == M.LEGAL_NFE,
      "A  and an independent derivation agrees")
_need(M.partition_for_nfe(3) is None and M.partition_for_nfe(9) is None
      and M.partition_for_nfe(12) is None,
      "A  3, 9 and 12 have no legal partition at all")

# the release's own three, reproduced
for n, want in ((4, (8,) * 4), (6, (8, 8, 4, 4, 4, 4)), (8, (4,) * 8)):
    _need(M.partition_for_nfe(n) == want,
          "A  nfe %d reproduces the published partition %s" % (n, want))
for n in M.LEGAL_NFE:
    p = M.partition_for_nfe(n)
    _need(sum(p) == 32 and len(p) == n and set(p) <= {4, 8},
          "A  nfe %d: %d blocks of legal size summing to 32" % (n, len(p)))
    _need(list(p) == sorted(p, reverse=True),
          "A  nfe %d puts the size-8 blocks first (high sigma)" % n)

# --- B: building the state ------------------------------------------------------
N, OUT_V, OUT_A, IN = 32, 96, 32, 128
torch.manual_seed(933)
sd = {"proj_out.weight": torch.randn(N, OUT_V, IN),
      "proj_out.bias": torch.randn(N, OUT_V),
      "audio_proj_out.weight": torch.randn(N, OUT_A, IN),
      "audio_proj_out.bias": torch.randn(N, OUT_A)}
bank = A.split_head_bank(sd)
_need(bank is not None, "B  a full head bank is recognised")
for drop in sorted(sd):
    partial = {k: v for k, v in sd.items() if k != drop}
    if A.split_head_bank(partial) is not None:
        _fail("B  a bank missing %s was accepted" % drop)
        break
else:
    _ok("B  a bank missing ANY of its four tensors is not")
_need(A.split_head_bank({"x.lora_A.weight": torch.zeros(1)}) is None,
      "B  an ordinary LoRA is not mistaken for a bank")

st = A.build_state(bank, nfe=8, source="test")
_need(st.nfe == 8 and len(st.boundaries) == 9,
      "B  nfe 8 -> 8 blocks and 9 boundaries")
_need(st.fused_v[0].shape == (8, OUT_V, IN)
      and st.fused_a[0].shape == (8, OUT_A, IN),
      "B  both modalities fused to the block count")
_need(all(A.build_state(bank, nfe=n).nfe == n for n in M.LEGAL_NFE),
      "B  every legal step count builds")
for bad in (3, 9, 12):
    try:
        A.build_state(bank, nfe=bad)
    except ValueError:
        _ok("B  nfe %d is refused at build time" % bad)
    else:
        _fail("B  nfe %d built a state" % bad)

# video and audio must NOT be fused with the same weights
same = torch.allclose(
    torch.tensor(M.block_plans(M.VIDEO_SHIFT, st.sizes)[1]),
    torch.tensor(M.block_plans(M.AUDIO_SHIFT, st.sizes)[1]))
_need(not same, "B  the two modalities are fused on their own grids")


# --- C: attaching ---------------------------------------------------------------
# v935, RE-JUSTIFIED. The first version of this section drove attach() against
# plain-object stand-ins, and three faults went through green because the
# stand-ins could not SHOW them: torch refuses a non-Module child (patch_model
# raised TypeError in core), core's FinalLayer.forward calls its OWN
# self.video_out (a wrapper around the layer is never reached), and core passes
# timestep = sigma * 1000 (the v933 wrapper read it as sigma). Lesson from the
# v933 sheet, applied to its own guard: a stand-in that cannot represent the
# fault does not witness it. So this stand-in is built from real nn.Modules,
# applies object patches by setattr the way comfy.utils.set_attr does, and is
# called the way core's H3 forward calls its wrappers.
class _FinalLike(torch.nn.Module):
    """core's FinalLayer, reduced to what matters here: it calls its own
    children."""

    def __init__(self):
        super().__init__()
        self.video_out = torch.nn.Linear(IN, OUT_V)
        self.audio_out = torch.nn.Linear(IN, OUT_A)
        self.eps = 1e-6

    def forward(self, x):
        return self.video_out(x), self.audio_out(x)


class _DMLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.final_layer = _FinalLike()


class _MS(object):
    multiplier = 1000.0


class _BaseLike(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = _DMLike()
        self.model_sampling = _MS()


def _set_attr(obj, path, value):
    """comfy.utils.set_attr, verbatim in effect: resolve, then setattr."""
    parts = path.split(".")
    for name in parts[:-1]:
        obj = getattr(obj, name)
    prev = getattr(obj, parts[-1])
    setattr(obj, parts[-1], value)
    return prev


class _Patcher(object):
    def __init__(self, base=None):
        self.model = base if base is not None else _BaseLike()
        self.object_patches = {}
        self.backup = {}
        self.wrappers = {}
        self.att = {}

    def clone(self):
        n = _Patcher(self.model)
        n.object_patches = dict(self.object_patches)
        n.wrappers = {k: list(v) for k, v in self.wrappers.items()}
        n.att = dict(self.att)
        return n

    def add_object_patch(self, k, v):
        self.object_patches[k] = v

    def patch_model(self):
        for k, v in self.object_patches.items():
            old = _set_attr(self.model, k, v)
            self.backup.setdefault(k, old)

    def unpatch_model(self):
        for k, v in self.backup.items():
            _set_attr(self.model, k, v)
        self.backup = {}

    # Core COLLECTS wrappers per key in a list -- that is the only reason
    # remove_wrappers_with_key exists. A stub that overwrites instead of
    # appending cannot show a missing removal.
    def remove_wrappers_with_key(self, t, k):
        self.wrappers.pop((t, k), None)

    def add_wrapper_with_key(self, t, k, w):
        self.wrappers.setdefault((t, k), []).append(w)

    def wrapper_count(self):
        return sum(len(v) for v in self.wrappers.values())

    def set_attachments(self, k, v):
        self.att[k] = v

    def get_attachment(self, k):
        return self.att[k]


# The stand-in must be able to show fault 2, or it witnesses nothing.
try:
    _set_attr(_BaseLike(), "diffusion_model.final_layer", object())
except TypeError:
    _ok("C  the stand-in refuses a non-Module child, as torch does in core")
else:
    _fail("C  the stand-in accepted a non-Module child -- it cannot show fault 2")

m = _Patcher()
m2 = A.attach(m, st)
VKEY = "diffusion_model.final_layer.video_out"
AKEY = "diffusion_model.final_layer.audio_out"
_need(not m.object_patches and not m.wrappers and not m.att,
      "C  the input model is NOT mutated")
_need(set(m2.object_patches) == {VKEY, AKEY},
      "C  the clone patches the two PROJECTIONS, not final_layer itself")
_need(m2.wrapper_count() == 1, "C  and carries exactly one wrapper")
_need(A.state_of(m2) is st, "C  and the state, readable by the sampler")
_need(A.state_of(m) is None, "C  which the original does not have")
_need(A.attach(m2, st).wrapper_count() == 1,
      "C  attaching to an already-attached model leaves ONE wrapper -- the old "
      "one is removed first, not stacked on top of itself")
_need(A.attach(A.attach(m2, st), st).wrapper_count() == 1,
      "C  and three times over, still one")

vhead = m2.object_patches[VKEY]
ahead = m2.object_patches[AKEY]
_need(isinstance(vhead, torch.nn.Module) and isinstance(ahead, torch.nn.Module),
      "C  the heads are nn.Modules (torch accepts nothing else as a child)")
_need(len(list(vhead.parameters())) == 0 and len(list(vhead.buffers())) == 0,
      "C  and own no parameters or buffers (core's loader never touches them)")
_need(vhead.in_features == IN and vhead.out_features == OUT_V,
      "C  the head reports the native shape")

native_v = m.model.diffusion_model.final_layer.video_out
try:
    m2.patch_model()
    _ok("C  patch_model applies both heads (the call that raised in core)")
    patched_ok = True
except TypeError as ex:
    _fail("C  patch_model raised %s" % ex)
    patched_ok = False
_need(patched_ok and m.model.diffusion_model.final_layer.video_out is vhead,
      "C  while patched, the layer's own child IS the head")

nofinal = _Patcher()
nofinal.model.diffusion_model = torch.nn.Linear(2, 2)
try:
    A.attach(nofinal, st)
except RuntimeError:
    _ok("C  a model without final_layer is refused, not silently unpatched")
else:
    _fail("C  a model without final_layer was 'patched'")


# --- D: the swap follows sigma, out of order and repeatedly ---------------------
# Called as core's H3 forward calls its DIFFUSION_MODEL wrappers:
# executor(x, timestep, context, transformer_options), timestep = sigma * 1000,
# and the RAW sigma in transformer_options["sigmas"] (comfy/samplers.py).
wrapper = list(m2.wrappers.values())[0][0]
x = torch.randn(2, IN)
final = m.model.diffusion_model.final_layer


def _exec(xx, t, context, transformer_options):
    return final(xx)


def _run(sigma, with_opts=True, which=0):
    opts = {"sigmas": torch.tensor([sigma])} if with_opts else {}
    return wrapper(_exec, x, torch.tensor([sigma * 1000.0]), None, opts)[which]


fw, fb = st.fused_v
order = [5, 0, 7, 2, 2, 7, 1]          # deliberately not sequential, with repeats
oks = []
for idx in order:
    got = _run(st.boundaries[idx])
    want = torch.nn.functional.linear(x, fw[idx].to(x.dtype), fb[idx].to(x.dtype))
    oks.append(torch.allclose(got, want, atol=1e-5))
_need(all(oks),
      "D  THE PROMISE: core's own forward reaches the head for ITS sigma, driven "
      "out of order and with repeats (a counter would fail this)")

outs = [_run(bb) for bb in st.boundaries[:-1]]
_need(all(not torch.allclose(outs[i], outs[j], atol=1e-6)
          for i in range(len(outs)) for j in range(i + 1, len(outs))),
      "D  the eight blocks really give eight different outputs")

got_nf = _run(st.boundaries[3], with_opts=False)
_need(torch.allclose(got_nf, torch.nn.functional.linear(
          x, fw[3].to(x.dtype), fb[3].to(x.dtype)), atol=1e-5),
      "D  without transformer_options the sigma comes from timestep / "
      "multiplier -- never from timestep as if it were sigma")

aw, ab = st.fused_a
got_a = _run(st.boundaries[4], which=1)
want_a = torch.nn.functional.linear(x, aw[4].to(x.dtype), ab[4].to(x.dtype))
_need(torch.allclose(got_a, want_a, atol=1e-5),
      "D  the audio head follows the same sigma into its own fused bank")

# --- E: fail closed -------------------------------------------------------------
mid = (st.boundaries[0] + st.boundaries[1]) / 2.0
for label, sigma in (("a sigma between boundaries", mid),
                     ("the final sigma, which starts no block", st.boundaries[-1]),
                     ("a sigma far outside", 5.0)):
    try:
        _run(sigma)
    except RuntimeError as ex:
        _need("not a trained block boundary" in str(ex),
              "E  %s is refused, with the cause named" % label)
    else:
        _fail("E  %s was served a head" % label)

try:
    final(x)
except RuntimeError as ex:
    _need("without a sigma" in str(ex),
          "E  a forward with no wrapper (hence no sigma) is refused")
else:
    _fail("E  a forward without a sigma was served")

_run(st.boundaries[2])
try:
    final(x)
except RuntimeError:
    _ok("E  the sigma is cleared after each forward -- it cannot leak into the next")
else:
    _fail("E  a stale sigma survived the forward")

# --- F: unpatch, re-attach while patched, refit ----------------------------------
m3 = A.attach(m, st)              # the layer still holds m2's head right now
_need(m3.object_patches[VKEY].native is native_v,
      "F  attaching while an old head is in place unwraps to the NATIVE "
      "projection -- never a head inside a head")
m2.unpatch_model()
_need(m.model.diffusion_model.final_layer.video_out is native_v,
      "F  unpatch restores the native projection")
_need(st.bank is bank, "F  the state keeps the raw bank for re-fusion")
m4 = A.refit(m2, 4)
st4 = A.state_of(m4)
_need(st4.nfe == 4 and list(st4.sizes) == [8, 8, 8, 8] and m4 is not m2,
      "F  refit re-fuses for another step count, on a clone")
_need(A.refit(m4, 4) is m4, "F  refit to the carried count is a no-op")
try:
    A.refit(m2, 12)
except ValueError:
    _ok("F  refit to an impossible count is refused by the maths")
else:
    _fail("F  refit accepted 12 steps")

print("\n%d checks, %d failed" % (len(CHECKS), len(FAILED)))
if FAILED:
    print("\n".join(FAILED))
    sys.exit(1)
print("v933 pdd apply: PASS")
