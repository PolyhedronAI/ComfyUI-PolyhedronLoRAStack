"""Guard v827 -- the Load CLIP no-model amber warning.

DECLARED LIMITATION: this is a TEXT PIN, built inside a timed session.
The suite's culture prefers driven guards; a driven one needs a DOM/
LiteGraph harness for onDrawForeground and is noted as follow-up work
in CHANGELOG_v827. What this pin DOES hold:

  W1  the warning text exists in ph_basics.js and says the two things
      Frank asked it to say (since v829 in TWO voices: node line + bubble): that the model-side check CANNOT RUN, and
      what to do (connect 'model').
  W2  it is gated on the MODEL INPUT LINK (`inp.link == null`), not on
      a widget -- the condition is "nothing attached", exactly as asked.
  W3  it is painted in the suite's official amber (the AMBER constant,
      #ff8c00), not an ad-hoc colour.
  W4  the header-check half-truth is not overclaimed: the text says the
      headers are still checked.

Script-style: exit 0 = pass.
"""
import os
import re
import sys

NAME = "v827"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JS = os.path.join(ROOT, "web", "js", "ph_basics.js")


def _fail(msg):
    print("[%s] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


with open(JS, "r", encoding="utf-8") as fh:
    src = fh.read()

if "no model connected" not in src:
    _fail("W1: the warning text is gone from ph_basics.js")
if "cannot run" not in src or "Connect" not in src:
    _fail("W1: the warning no longer says what cannot run / what to do")

# RE-GROUNDED in v829: the words moved into ONE shared constant
# (NO_MODEL_TEXT) and the gate into ONE shared predicate (modelUnlinked),
# because the warning now speaks with TWO voices -- the amber status line
# on the node AND a bubble top right (Frank's ask). The promises are the
# ones v827 made; only their address changed.
mu = src[src.index("function modelUnlinked"):
         src.index("function modelUnlinked") + 400]
if "inp.link == null" not in mu:
    _fail("W2: modelUnlinked no longer gates on the model input LINK")
if 'name === "model"' not in mu:
    _fail("W2: modelUnlinked no longer reads the 'model' input")
if "modelUnlinked(this)" not in src:
    _fail("W2: nothing calls modelUnlinked -- the gate is orphaned")
blk = src[src.index("NO_MODEL_TEXT =") - 200:
          src.index("NO_MODEL_TEXT =") + 700]
# ("file headers" / "cannot run": the exact pair survives the JS string
# line-break that splits "headers only" in the source.)
if "file headers" not in blk or "cannot run" not in blk:
    _fail("W4: the text no longer says the header check still runs")
draw = src[src.index('"\\u26a0 " + NO_MODEL_TEXT') - 600:
           src.index('"\\u26a0 " + NO_MODEL_TEXT') + 200]
if "AMBER" not in draw:
    _fail("W3: the status line is not painted with the AMBER constant")
if not re.search(r'AMBER\s*=\s*"#ff8c00"', src):
    _fail("W3: AMBER is no longer the suite's #ff8c00")
# W5 (v829): the bubble. One source for the words, top-right form, and
# the flag resets on connect so an unplug warns again without redraw spam.
if "function toastNoModel" not in src or "ph-basics-toast" not in src:
    _fail("W5: the top-right bubble is gone")
tb = src[src.index("function toastNoModel"):
         src.index("function toastNoModel") + 900]
if "NO_MODEL_TEXT" not in tb:
    _fail("W5: the bubble no longer speaks from the shared constant")
if "_plsNoModelToasted = false" not in src:
    _fail("W5: connecting a model no longer re-arms the bubble")

print("[%s] PASS -- Load CLIP no-model warning pinned (text pin, "
      "driven harness is declared follow-up)" % NAME)
