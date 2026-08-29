"""Guard v848 -- Polyhedron Attention: a saved workflow survives a poorer machine.

v847 enforced the dropdown law correctly and made saved workflows brittle in
the same stroke: removing xformers from the list made ComfyUI reject Frank's
ENTIRE prompt ("Value not in list", output ignored) because the saved
workflow still carried that value. The failure is not local to the setting --
it kills the run, and it would hit anyone opening the workflow on a machine
without sageattention, without a Blackwell card, or with a different start-up
choice.

Both properties are needed at once, so both are pinned here:

  W1  THE RELAXATION IS SHAPED THE WAY COMFYUI READS IT. Core takes the names
      from inspect.getfullargspec(VALIDATE_INPUTS).args and skips default
      validation for exactly those. So the promise is not "a method exists"
      but "these three input names appear in its argument list, and it
      accepts a value that is not in the list".

  W2  THE DROPDOWN STAYS STRICT. Relaxing validation must not quietly relax
      what is offered -- the list is still only what can run here.

  W3  AN UNOFFERED MODE IS NAMED, NOT SUBSTITUTED. patch() says which mode
      the workflow wants, why it is missing where we can check that, and what
      IS available -- then leaves the model on its own backend. No override
      is installed. Silently swapping in another kernel would change results
      with nothing in the workflow to show for it.

  W4  THE RUN CONTINUES. Nothing raises; a MODEL still comes out, so the rest
      of the graph renders.

  W5  A MODE THAT IS OFFERED IS UNAFFECTED. The new gate must not stand in
      front of the normal path.
"""

import importlib.util
import inspect
import io
import os
import sys
from contextlib import redirect_stdout


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def _fail(msg):
    FAILS.append(msg)
    print("  FAIL  " + msg)


def _ok(msg):
    print("  ok    " + msg)


def _load():
    path = os.path.join(ROOT, "nodes", "ph_attention.py")
    spec = importlib.util.spec_from_file_location("ph_attention_v848", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PH = _load()


class FakeModel:
    def __init__(self, options=None):
        self.model_options = dict(options or {})

    def clone(self):
        return FakeModel(dict(self.model_options))


def _slot(model):
    return model.model_options.get("transformer_options", {}).get(
        "optimized_attention_override", "ABSENT")


print("[test_v848_unoffered_mode] a saved workflow survives a poorer machine")

# --------------------------------------------------------------------------
# W1 -- the relaxation is shaped the way ComfyUI reads it
# --------------------------------------------------------------------------

print("W1 VALIDATE_INPUTS is shaped so core skips the list check")

vi = getattr(PH.ULSAttention, "VALIDATE_INPUTS", None)
if vi is None:
    _fail("W1: no VALIDATE_INPUTS at all -- core will keep rejecting saved "
          "workflows that carry a mode this machine does not offer")
else:
    # This is the exact call core makes.
    args = inspect.getfullargspec(vi).args
    missing = [n for n in ("attention", "attention_first", "attention_last")
               if n not in args]
    if missing:
        _fail("W1: %s not in the VALIDATE_INPUTS signature -- core would "
              "still validate %s against the list" % (missing, missing))
    else:
        _ok("signature carries all three mode inputs %r" % args)

    verdict = vi("a mode this machine has never heard of",
                 PH.SAME_AS_MAIN, PH.SAME_AS_MAIN)
    if verdict is not True:
        _fail("W1: an unknown mode was not accepted (returned %r) -- core "
              "treats anything but True as a validation error" % (verdict,))
    else:
        _ok("an unknown mode value is accepted")

# --------------------------------------------------------------------------
# W2 -- the dropdown itself stays strict
# --------------------------------------------------------------------------

print("W2 relaxed validation did not relax the offer")

offered = PH.available_modes()
types = PH.ULSAttention.INPUT_TYPES()
listed = types["required"]["attention"][0]
if list(listed) != list(offered):
    _fail("W2: the widget list and available_modes() disagree")
elif PH.XFORMERS_MODE in offered and PH._xformers_wired() is False:
    _fail("W2: xformers is offered again although core has not wired it")
else:
    _ok("the list is still exactly what can run here (%d mode(s))"
        % len(offered))

# a mode that cannot be built must not be in the list
for mode in offered:
    if mode in (PH.DEFAULT_MODE, PH.SPARSE_LOCAL):
        continue          # default patches nothing; sparse needs geometry
    if PH.build_router(mode) is None:
        _fail("W2: '%s' is offered but has no router" % mode)
        break
else:
    _ok("every offered mode has a router")

# --------------------------------------------------------------------------
# W3 / W4 -- named, not substituted, and the run continues
# --------------------------------------------------------------------------

print("W3 an unoffered mode is named and nothing is patched")

node = PH.ULSAttention()
missing_mode = None
for candidate in ("sage fp8 cuda++", PH.SAGE3_MODE, PH.XFORMERS_MODE):
    if candidate not in offered:
        missing_mode = candidate
        break
if missing_mode is None:
    missing_mode = "a mode this machine has never heard of"
    _ok("this machine offers everything known -- using an invented mode")

buf = io.StringIO()
with redirect_stdout(buf):
    (out,) = node.patch(FakeModel(), missing_mode, True, live_check=False)
text = buf.getvalue()

if _slot(out) != "ABSENT":
    _fail("W3: an override was installed for a mode that is not offered")
elif missing_mode not in text:
    _fail("W3: the message does not name the requested mode: %r" % text)
elif "does not offer" not in text:
    _fail("W3: the message does not say the mode is unavailable HERE: %r"
          % text)
elif "available here" not in text:
    _fail("W3: the message does not list what IS available: %r" % text)
else:
    _ok("named, explained, alternatives listed, no override installed")

# the substitution test has teeth only if a substitute was available
if len(offered) > 1 and _slot(out) != "ABSENT":
    _fail("W3: a substitute kernel was installed silently")
else:
    _ok("no silent substitution")

print("W4 the run continues")
if not hasattr(out, "model_options"):
    _fail("W4: patch() did not return a usable model")
else:
    _ok("a MODEL comes back, so the rest of the graph still renders")

# a window mode that is not offered must not take the main mode down
buf = io.StringIO()
with redirect_stdout(buf):
    (out_w,) = node.patch(FakeModel(), "pytorch sdpa", True,
                          attention_first=missing_mode, first_steps=1,
                          live_check=False)
wtext = buf.getvalue()
if _slot(out_w) == "ABSENT":
    _fail("W4: an unoffered WINDOW mode took the main override down: %r"
          % wtext)
elif "first 1 step(s)" in wtext:
    _fail("W4: a window was announced for a mode that cannot run: %r" % wtext)
else:
    _ok("an unoffered window is dropped, the main mode still patches")

# --------------------------------------------------------------------------
# W5 -- the normal path is untouched
# --------------------------------------------------------------------------

print("W5 an offered mode still patches normally")

buf = io.StringIO()
with redirect_stdout(buf):
    (ok_out,) = node.patch(FakeModel(), "pytorch sdpa", True,
                           live_check=False)
otext = buf.getvalue()
if _slot(ok_out) == "ABSENT":
    _fail("W5: an offered mode was refused: %r" % otext)
elif "does not offer" in otext:
    _fail("W5: an offered mode got the unavailable message: %r" % otext)
else:
    _ok("an offered mode patches, no spurious warning")

buf = io.StringIO()
with redirect_stdout(buf):
    (def_out,) = node.patch(FakeModel(), PH.DEFAULT_MODE, True,
                            live_check=False)
if _slot(def_out) != "ABSENT":
    _fail("W5: 'default' installed an override")
elif "does not offer" in buf.getvalue():
    _fail("W5: 'default' was treated as unavailable")
else:
    _ok("'default' still means 'leave the model alone'")

# describe_absence must state only what it can check
print("W5b the stated reason is checkable, never invented")
if PH.describe_absence("a mode nobody has ever defined") != "":
    _fail("W5b: a reason was invented for a mode we know nothing about")
else:
    _ok("an unknown mode gets no invented reason")

if PH.SAGE3_MODE not in offered:
    why = PH.describe_absence(PH.SAGE3_MODE)
    if "not installed" not in why:
        _fail("W5b: sage3 absence is not explained: %r" % why)
    else:
        _ok("sage3 absence explained: %s" % why)

# --------------------------------------------------------------------------
# verdict
# --------------------------------------------------------------------------

if FAILS:
    print("[test_v848_unoffered_mode] FAIL -- %d problem(s)" % len(FAILS))
    sys.exit(1)
print("[test_v848_unoffered_mode] PASS -- strict dropdown, tolerant load, "
      "loud console, no silent substitution")
