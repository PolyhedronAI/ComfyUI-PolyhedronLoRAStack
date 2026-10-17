"""v870 -- a joint (nested) latent must never end a run.

MiniMaxH3ReferenceToVideo hands ULSSampler a NestedTensor((video, audio)).
Core's comfy/nested_tensor.py answers .size()/.shape/.dtype/.layout/.device from
tensors[0] and .ndim as the MAX over parts -- but defines NEITHER numel() NOR
detach(). Those two gaps killed the run clock and the preview.

This guard runs WITHOUT torch and WITHOUT comfy: it lifts the four helpers out of
the source with ast and drives them against a fake that MIRRORS Core's real API.

v869 lesson 5, honoured: the fixture PINS ITS OWN PRECONDITION. If the fake ever
grows a numel() or a detach(), the probes below would sail through the fast path
and observe nothing -- so we assert the gaps first and fail loudly if they close.
"""
import ast
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = (ROOT / "nodes" / "uls_sampler.py").read_text(encoding="utf-8")
FAILED = []


def _fail(msg):
    FAILED.append(msg)
    print("FAIL: {}".format(msg))


def _ok(msg):
    print("ok  : {}".format(msg))


# --- lift the helpers out, no torch import ---------------------------------
WANT = ("_latent_parts", "_is_ragged_latent", "_latent_weight", "_preview_tensor")
ns = {}
tree = ast.parse(SRC)
found = {}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in WANT:
        found[node.name] = ast.get_source_segment(SRC, node)
for name in WANT:
    if name not in found:
        _fail("helper {} is missing from uls_sampler.py".format(name))
if len(found) == len(WANT):
    exec(compile("\n\n".join(found[n] for n in WANT), "<v870>", "exec"), ns)
    _ok("four helpers lifted and compiled without torch")


class FakeTensor(object):
    """An ordinary tensor: it HAS numel() and detach()."""

    def __init__(self, n, ndim=5):
        self._n = n
        self.ndim = ndim
        self.is_nested = False
        self.detached = False

    def numel(self):
        return self._n

    def detach(self):
        out = FakeTensor(self._n, self.ndim)
        out.detached = True
        return out


class FakeNested(object):
    """Mirror of Core comfy/nested_tensor.py: .tensors, .unbind(), .is_nested,
    .ndim as MAX over parts -- and deliberately NO numel(), NO detach()."""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors

    @property
    def ndim(self):
        return max(t.ndim for t in self.tensors)


class FakeUnbindOnly(object):
    """A nested latent that exposes only unbind() -- the fallback path."""

    def __init__(self, tensors):
        self._t = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self._t


class Opaque(object):
    """Answers nothing: neither numel() nor parts. The clock must still cope."""

    pass


class RaisingNumel(object):
    """HAS numel(), but it RAISES -- exactly how torch's own nested tensors
    behave. hasattr() is True here, so a hasattr-based guard would let the raise
    escape and end the run. Only a try/except closes this branch, and without
    this fixture no probe would ever drive it (v869 lesson 5)."""

    is_nested = False

    def numel(self):
        raise RuntimeError("nested tensor does not support numel()")


# --- PRECONDITION PINS: if these close, every probe below goes blind --------
video, audio = FakeTensor(1000, ndim=5), FakeTensor(24, ndim=4)
nested = FakeNested([video, audio])
if hasattr(nested, "numel"):
    _fail("fixture drifted: FakeNested grew numel() -- probes would take the "
          "fast path and observe nothing")
elif hasattr(nested, "detach"):
    _fail("fixture drifted: FakeNested grew detach() -- the preview probe would "
          "observe nothing")
else:
    _ok("fixture pins its own precondition (no numel, no detach)")
if nested.ndim != 5:
    _fail("fixture drifted: Core's ndim is the MAX over parts (5 here), got {}"
          .format(nested.ndim))
else:
    _ok("fixture mirrors Core's ndim = max over parts")

if not FAILED:
    lw = ns["_latent_weight"]
    lp = ns["_latent_parts"]
    ragged = ns["_is_ragged_latent"]
    prev = ns["_preview_tensor"]

    # --- _latent_weight ----------------------------------------------------
    if lw(FakeTensor(4096)) != 4096.0:
        _fail("_latent_weight: plain tensor must use numel() exactly")
    else:
        _ok("_latent_weight: plain tensor -> exact numel")

    got = lw(nested)
    if got != 1024.0:
        _fail("_latent_weight: joint latent must SUM its parts (1000+24=1024), "
              "got {}".format(got))
    else:
        _ok("_latent_weight: joint latent -> summed parts, exact not guessed")

    try:
        got = lw(Opaque())
    except Exception as exc:
        _fail("_latent_weight RAISED on an opaque object ({}) -- telemetry must "
              "never end a run".format(type(exc).__name__))
    else:
        if got != 1.0:
            _fail("_latent_weight: opaque object must fall back to 1.0, got {}"
                  .format(got))
        else:
            _ok("_latent_weight: opaque object -> neutral 1.0, no raise")

    probe = RaisingNumel()
    if not hasattr(probe, "numel"):
        _fail("fixture drifted: RaisingNumel must HAVE numel() -- that is the "
              "whole point of the branch it drives")
    else:
        try:
            got = lw(probe)
        except Exception as exc:
            _fail("_latent_weight let a raising numel() escape ({}) -- a hasattr "
                  "check is not enough, the call must be tried".format(
                      type(exc).__name__))
        else:
            if got != 1.0:
                _fail("_latent_weight: raising numel() must fall back to 1.0, "
                      "got {}".format(got))
            else:
                _ok("_latent_weight: numel() that RAISES -> neutral 1.0")

    if lw(FakeTensor(0)) < 1.0:
        _fail("_latent_weight: an empty latent must still weigh at least 1")
    else:
        _ok("_latent_weight: empty latent floors at 1")

    # --- _latent_parts / _is_ragged_latent ---------------------------------
    if lp(nested) != [video, audio]:
        _fail("_latent_parts: must return the parts of a joint latent")
    elif len(lp(FakeUnbindOnly([video, audio]))) != 2:
        _fail("_latent_parts: unbind()-only fallback did not resolve")
    elif len(lp(FakeTensor(8))) != 1:
        _fail("_latent_parts: a plain tensor is its own only part")
    else:
        _ok("_latent_parts: joint, unbind-only and plain all resolve")

    if not (ragged(nested) is True and ragged(FakeTensor(8)) is False
            and ragged(None) is False):
        _fail("_is_ragged_latent must answer True/False/False and never raise")
    else:
        _ok("_is_ragged_latent: joint True, plain False, None False")

    # --- _preview_tensor ---------------------------------------------------
    out = prev(nested)
    if out is None or getattr(out, "numel", lambda: -1)() != 1000:
        _fail("_preview_tensor: a joint latent must preview its VIDEO half")
    elif not getattr(out, "detached", False):
        _fail("_preview_tensor: the returned tensor must be detached")
    else:
        _ok("_preview_tensor: joint latent -> detached video half")

    try:
        out = prev(Opaque())
    except Exception as exc:
        _fail("_preview_tensor RAISED on an object without detach() ({})"
              .format(type(exc).__name__))
    else:
        _ok("_preview_tensor: no detach() -> returns as-is, no raise")

# --- STRUCTURE: the wound must stay closed in the file ---------------------
for pattern in ('latent_image.numel()', 'latent["samples"].numel()'):
    hits = SRC.count(pattern)
    inside = found.get("_latent_weight", "").count(pattern)
    if hits - inside:
        _fail("{} still appears {}x OUTSIDE _latent_weight -- the clock can "
              "still crash".format(pattern, hits - inside))
    else:
        _ok("{} appears only inside _latent_weight".format(pattern))

body = SRC[SRC.index("    def frames_b64(self, x0):"):]
body = body[:body.index("\n    def _encode")] if "\n    def _encode" in body else body
i_norm = body.find("_preview_tensor(x0)")
i_disp = min([p for p in (body.find("self._frames_core_single"),
                          body.find("self._frames_tae"),
                          body.find("self._frames_latent2rgb")) if p > 0] or [-1])
if i_norm < 0:
    _fail("frames_b64 does not normalise x0 -- the preview can still crash")
elif i_disp < 0 or i_norm > i_disp:
    _fail("frames_b64 normalises AFTER dispatching to a preview path")
else:
    _ok("frames_b64 normalises x0 before every preview path")

i_guard = SRC.find("_is_ragged_latent((latent_image or {}).get(\"samples\"))")
i_shift = SRC.find("model = _apply_sigma_shift(model, sigma_shift)")
if i_guard < 0:
    _fail("the sigma_shift refusal for joint latents is missing")
elif i_shift < 0 or i_guard > i_shift:
    _fail("the sigma_shift refusal does not precede _apply_sigma_shift")
elif "ModelSamplingMiniMaxH3" not in SRC:
    _fail("the refusal must NAME the node to use instead")
else:
    _ok("sigma_shift refusal precedes the patch and names the right node")

print("\n{}: {} failure(s)".format(pathlib.Path(__file__).name, len(FAILED)))
sys.exit(1 if FAILED else 0)
