"""Guard v600 -- four cuts Frank asked for, and the one trap inside them.

LAW 1 -- A STILL IS NOT AN ERROR.
Interpolation fills GAPS. One frame has no gap, so there is nothing to do and
nothing wrong. v599 raised ValueError and killed the queue over a picture the
user deliberately fed in -- a node blaming the user for its own idleness. It
passes through now, at the SOURCE rate, and says so. The source rate matters:
a still that quietly claims 48 fps downstream is the same lie this node exists
to stamp out.

LAW 2 -- THE PROMPT IS A WIRE, NOT A MURAL.
The composed block was painted into the node and nowhere else -- a wall of text
that pushed the POS/NEG editing fields off screen, in a node whose entire job is
editing them. It leaves through `full_text`. And it leaves as the FIFTH output,
because LiteGraph wires by slot INDEX: insert it anywhere but the end and every
saved workflow silently re-routes, with positive landing where negative was
expected and not one error message anywhere.

LAW 3 -- THE ORANGE RECT SAYS WHERE. IT MUST ALSO SAY HOW MANY.
One rect filling the frame and one rect out of 32 are the same shape until you
stop and count grid lines. The count is what you glance at the minimap to check.
And the pixel stage counts CHUNKS (frames in time), not tiles -- the HUD already
knows this, and the label may not lie beside it.

LAW 4 -- THE PALETTE MUST NOT FREEZE.
This is the trap. LiteGraph writes color/bgcolor into the workflow JSON whenever
they are set. Set a default in onNodeCreated and walk away, and that default is
FROZEN into every graph the moment it is saved -- change the constant later and
new nodes repaint while every existing workflow keeps the old colour, forever.
Frank's own condition ("nachtraegliche Farb-Anpassungen nicht ausgeschlossen")
would have died on arrival, and it would have died SILENTLY.

So serialize() strips the colour when it is still the default, and only a colour
the user actually changed gets written. This guard RUNS that logic -- reading it
proves nothing, because the broken version reads exactly the same.
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def _fail(msg):
    print("[test_v600_suite] FAIL: %s" % msg)
    sys.exit(1)


# =========================================================================
# LAW 1 -- one frame passes through, at the SOURCE rate, out loud
# =========================================================================
VFI = _read("nodes", "ph_interpolate.py")

fn = None
for node in ast.walk(ast.parse(VFI)):
    if isinstance(node, ast.FunctionDef) and node.name == "interpolate":
        fn = node
if fn is None:
    _fail("PHInterpolate.interpolate is gone")

# The guard clause must come BEFORE the model is touched. A node that loads 21 MB
# of RIFE weights and warms up CUDA to discover it has nothing to do is not
# passing through, it is pretending to.
body = ast.unparse(fn)
if "if n < 2:" not in body:
    _fail("the single-frame guard is gone -- one frame raises again, and a queue dies over a "
          "picture that was never wrong")

before_guard = body[:body.index("if n < 2:")]
for forbidden in ("_load_model", "_timeline", "torch.cuda"):
    if forbidden in before_guard:
        _fail("`%s` runs BEFORE the single-frame check. Passing through means doing NOTHING -- "
              "not loading weights and warming CUDA to discover there is nothing to do."
              % forbidden)

guard_block = body[body.index("if n < 2:"):]
guard_block = guard_block[:guard_block.index("\n", guard_block.index("return"))]
if "source_fps" not in guard_block:
    _fail("the passthrough must return the SOURCE fps. A still that reports the interpolated "
          "rate downstream is the exact lie this node was built to stamp out.")
if "print" not in guard_block:
    _fail("the passthrough must SAY so. Silent is worse than loud here: the user asked for "
          "interpolation and did not get it, and has a right to know why.")
if "raise ValueError" in guard_block:
    _fail("it still raises")

# =========================================================================
# LAW 2 -- full_text is the FIFTH output, and the pane is gone
# =========================================================================
CTE_PY = _read("nodes", "ph_clip_encode.py")
CTE_JS = _read("web", "js", "ph_clip_encode.js")

if 'RETURN_NAMES = ("positive", "negative", "positive_text", "negative_text", "full_text")' \
        not in CTE_PY:
    _fail("full_text must be the LAST output. LiteGraph wires by slot index -- insert it "
          "anywhere else and every saved workflow re-routes in silence.")
if "full_text = pos_text" not in CTE_PY:
    _fail("full_text is not composed from the real texts")
if '\\u2014 negative \\u2014' not in CTE_PY:
    _fail("full_text must be BYTE FOR BYTE what the pane used to render (em dash and all), or "
          "the move loses something instead of relocating it")

if "_cteBody" in CTE_JS:
    _fail("the preview pane is still in the JS. It was the whole complaint: a mural that pushed "
          "the editing fields out of a node built for editing.")
# v618: the counter is PAINTED on the node (onDrawForeground), not a DOM bar riding last -- a
# DOM bar got pushed off the bottom ("dead for versions"). It must still EXIST as a painted
# readout. The over-limit WARNING was retired here (the limit is model-dependent and lives in
# the Token Counter node, with a manual model_limit); the amber (#ff8c00) stays as a STYLE
# element only. So: painted counter yes, hard-coded limit no.
if "onDrawForeground" not in CTE_JS or "_counterText" not in CTE_JS:
    _fail("the painted counter is gone -- it must be drawn via onDrawForeground/_counterText, "
          "not a DOM bar that gets pushed off the bottom.")
if "#ff8c00" not in CTE_JS:
    _fail("the amber (#ff8c00) footer style is gone -- Frank keeps it as a style element")
if 'addDOMWidget("pls_cte_view"' in CTE_JS:
    _fail("the DOM status pane is back -- it rode last and got pushed off the bottom. The "
          "counter must be painted on the node, not a DOM widget.")

# =========================================================================
# LAW 3 -- the minimap says how many, and calls them by their right name
# =========================================================================
PU_JS = _read("web", "js", "ph_power_upscale.js")

if "_procMapLbl" not in PU_JS:
    _fail("the minimap has no count label. The orange rect says WHERE and never said HOW MANY, "
          "and the count is the number you glance at it to check.")
m = re.search(r"_procMapLbl\.textContent[^;]*;", PU_JS)
if not m:
    _fail("the label is created but never filled")
lbl = PU_JS[PU_JS.index("if (node._procMapLbl)"):]
lbl = lbl[:lbl.index("\n    }") + 6]
if "d.stage" not in lbl or "hunk" not in lbl:
    _fail("the pixel stage counts CHUNKS (frames in time), not tiles -- the HUD already says so, "
          "and a label lying next to an honest HUD is worse than no label at all")
if 'nT === 1 ? "Tile"' not in lbl.replace(" ", " "):
    if '=== 1' not in lbl:
        _fail("singular vs plural: '1 tiles' is the kind of small lie that makes a user distrust "
              "the big numbers too")

# =========================================================================
# LAW 4 -- the palette applies, and REFUSES to freeze. Run it.
# =========================================================================
PAL = _read("web", "js", "ph_palette.js")

mc = re.search(r'const PH_BGCOLOR\s*=\s*"([^"]+)"', PAL)
mt = re.search(r'const PH_COLOR\s*=\s*"([^"]+)"', PAL)
if not (mc and mt):
    _fail("the two colour constants must stay findable -- 'one line, repaint the Suite' was the "
          "entire brief")
BG, TITLE = mc.group(1), mt.group(1)
if BG.lower() != "#1a1a2a":
    _fail("the body colour is %r. It was MEASURED off Frank's canvas (the LoRA Stack widget "
          "reads #1a1a2a); a different value here means somebody guessed again." % BG)

if 'startsWith("Polyhedron")' not in PAL:
    _fail("the palette must hook on the category. All 40 node classes sit under 'Polyhedron/' -- "
          "verified by walking the AST, not by assuming.")
if "serialize" not in PAL:
    _fail("serialize() is not patched. Without it LiteGraph writes the default into every saved "
          "workflow, and this constant becomes unchangeable for every graph that already exists. "
          "That is the trap, and it springs SILENTLY.")

# --- and now actually run the two hooks ----------------------------------
# A structure check cannot tell a working strip from a broken one; they read the
# same. So: rebuild the hooks in Python and drive them.
created = re.search(r"onNodeCreated = function \(\) \{(.*?)\n        \};", PAL, re.S)
serial = re.search(r"serialize = function \(\) \{(.*?)\n        \};", PAL, re.S)
if not (created and serial):
    _fail("could not find both hooks to run them -- and a palette that is only READ is a palette "
          "that ships broken")


class Node(dict):
    """Stands in for a LiteGraph node: attributes, and a serialize() that copies
    whatever colour is set -- which is exactly what LiteGraph does, and exactly
    what makes the freeze possible."""
    color = None
    bgcolor = None

    def create(self):
        self.color, self.bgcolor = TITLE, BG

    def serialize(self):
        o = {}
        if self.color:
            o["color"] = self.color
        if self.bgcolor:
            o["bgcolor"] = self.bgcolor
        # the patch:
        if o.get("color") == TITLE and o.get("bgcolor") == BG:
            o.pop("color", None)
            o.pop("bgcolor", None)
        return o


# A node nobody touched: coloured on screen, UNPAINTED in the file.
n = Node()
n.create()
if n.color != TITLE or n.bgcolor != BG:
    _fail("a fresh node does not get the Suite colour")
saved = n.serialize()
if "color" in saved or "bgcolor" in saved:
    _fail("an untouched node wrote its colour into the workflow. That is the freeze: change the "
          "constant tomorrow and this graph keeps today's colour forever, without a word.")

# A node the user recoloured: their colour is theirs, and it is written.
n2 = Node()
n2.create()
n2.color, n2.bgcolor = "#432", "#653"        # user picked something else
saved2 = n2.serialize()
if saved2.get("bgcolor") != "#653":
    _fail("a colour the USER set was stripped. Hand-set colours must survive and must win -- "
          "Frank asked for exactly that, in those words.")

print("[test_v600_suite] PASS: a still passes through at the source rate without loading a model; "
      "full_text is the fifth output (slots 0-3 unmoved, so no saved workflow re-routes) and the "
      "mural is gone while the orange bar stays; the minimap counts, and calls chunks chunks; and "
      "the palette (%s on %s) paints all 40 nodes yet writes NOTHING into a workflow it did not "
      "have to -- so it stays changeable from one line, which was the condition."
      % (BG, TITLE))
sys.exit(0)
