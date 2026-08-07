"""Guard v604 -- CANON vs DISPLAY. The law the tree already knew, and I broke.

WHAT I SHIPPED YESTERDAY (v603): I re-ordered INPUT_TYPES so Frank's prompt filters
would sit above the prompt boxes. He asked for that, and he was right to.

WHAT CAME BACK: his prompt inside `separator`. The string "true" in a prompt box.
`comment_markers` in the negative prompt. Every saved graph he owns, shredded on
load, in silence.

AND THE TREE ALREADY KNEW. In ph_power_upscale.js, in capitals, dated the day
before:

    v585 LAW (measured 2026-07-13, the hard way): the live frontend serialises
    widgets_values in the WIDGET (display) order, not in ORDER_CANON. The v584 cut
    moved final_upscale_by into the display middle and every pre-v584 save loaded
    shifted by one slot from position 4 on - the seed landed in cfg_low. Therefore
    DISPLAY_ORDER is APPEND-ONLY... Re-sorting the display is only legal after the
    save path is normalised through the canon mapping - A MEASURED PROJECT OF ITS
    OWN, NOT A SIDE EFFECT.

I did it as a side effect. And I wrote a heal for it, and the heal was CORRECT, and
its guard ran GREEN, and none of that mattered -- because the heal was fighting the
symptom (values landing in the wrong slot) while the cause (the canon moving out
from under them) walked away untouched. A correct answer to the wrong question is
still the wrong answer, and it ships just as easily.

THE ARCHITECTURE, as ph_power_upscale learned it at v546/v588:

    PYTHON  = CANON.    Never re-ordered. What lives on disk, forever.
    JS      = DISPLAY.  Permuted freely. The filters sit above the prompts.
    load    : canon -> display   (configure)
    save    : display -> canon   (onSerialize)

The file on disk never learns that the display moved. That is the whole trick, and
it is the only version of this that is safe.

THIS GUARD RUNS THE ROUND TRIP. canon -> display -> canon must be the IDENTITY, or
every save eats the graph a little more.
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = open(os.path.join(ROOT, "nodes", "ph_clip_encode.py"), encoding="utf-8").read()
JS = open(os.path.join(ROOT, "web", "js", "ph_clip_encode.js"), encoding="utf-8").read()


def _fail(msg):
    print("[test_v604_canon] FAIL: %s" % msg)
    sys.exit(1)


# =========================================================================
# LAW 1 -- THE CANON IS APPEND-ONLY. This is the v585 law, now enforced in the
# file that violated it.
# =========================================================================
def _canon_from_ast():
    for node in ast.walk(ast.parse(PY)):
        if isinstance(node, ast.FunctionDef) and node.name == "INPUT_TYPES":
            ret = [n for n in ast.walk(node) if isinstance(n, ast.Return)][0]
            for k, v in zip(ret.value.keys, ret.value.values):
                if isinstance(k, ast.Constant) and k.value == "required":
                    order = []
                    for kk in v.keys:
                        if kk is None:
                            order += ["pos_%d" % i for i in range(1, 7)]
                        else:
                            order.append(kk.value)
                    return [o for o in order if o != "clip"]
    _fail("INPUT_TYPES is gone")


CANON_PY = _canon_from_ast()

# The order as it has stood since v557. It does not get to change. Ever.
CANON_FROZEN = ["segments", "pos_1", "pos_2", "pos_3", "pos_4", "pos_5", "pos_6",
                "use_negative", "neg_1", "separator", "strip_comments",
                "strip_newlines", "external_mode", "comment_markers"]

if CANON_PY != CANON_FROZEN:
    _fail("INPUT_TYPES was RE-ORDERED.\n"
          "         is: %s\n"
          "   must be: %s\n"
          "  The canon is APPEND-ONLY (v585, measured the hard way; v603, measured again on "
          "Frank's graph). The live frontend serialises widgets_values by WIDGET POSITION -- "
          "move a widget here and every graph ever saved loads shifted, in silence, with the "
          "prompts landing in the filter boxes. If you want a different ORDER ON SCREEN, add it "
          "to DISPLAY in the JS. That is what DISPLAY is for, and it costs nothing."
          % (CANON_PY, CANON_FROZEN))

# =========================================================================
# LAW 2 -- the JS's idea of canon must BE the Python canon. Not resemble it.
# =========================================================================
def _js_list(name):
    m = re.search(r"const %s = \[(.*?)\];" % name, JS, re.S)
    if not m:
        _fail("%s is gone from the JS" % name)
    return re.findall(r'"([a-z_0-9]+)"', m.group(1))


CANON_JS = _js_list("CANON")
DISPLAY_JS = _js_list("DISPLAY")

if CANON_JS != CANON_PY:
    _fail("the JS CANON does not match INPUT_TYPES.\n    js: %s\n    py: %s\n  These two drifting "
          "apart is not a cosmetic bug -- it is a silent shredder. Every load and every save maps "
          "through this list." % (CANON_JS, CANON_PY))
if sorted(DISPLAY_JS) != sorted(CANON_JS):
    _fail("DISPLAY and CANON hold different widgets: %s. A permutation may re-order; it may not "
          "invent or drop." % sorted(set(DISPLAY_JS) ^ set(CANON_JS)))

# Frank's request (v607): filters on top, in HIS order, segments at the very bottom.
FILTERS = ["separator", "strip_comments", "strip_newlines", "external_mode",
           "comment_markers"]
first_prompt = min(i for i, n in enumerate(DISPLAY_JS) if n.startswith("pos_"))
for f in FILTERS:
    if DISPLAY_JS.index(f) > first_prompt:
        _fail("`%s` is not above the prompt boxes in DISPLAY. A control you scroll past six "
              "textareas to reach is a control you stop using." % f)
# The exact order Frank dictated. If he re-orders again this line moves; it is here so
# a silent shuffle of DISPLAY cannot pass unnoticed.
WANT_TOP = ["external_mode", "strip_comments", "strip_newlines", "separator",
            "comment_markers", "segments"]
if DISPLAY_JS[:len(WANT_TOP)] != WANT_TOP:
    _fail("the top block is not in the dictated order.\n    is:   %s\n    want: %s"
          % (DISPLAY_JS[:len(WANT_TOP)], WANT_TOP))
# segments moved (v609) from the very bottom up to the 6th selector row, directly
# above pos_1 -- Frank's list put it with the other five. Every selector must still
# sit above every prompt box.
if DISPLAY_JS[len(WANT_TOP)] != "pos_1":
    _fail("segments must be the LAST selector before the prompt boxes; pos_1 does not follow it "
          "(found %r). All six selectors sit above the prompts." % DISPLAY_JS[len(WANT_TOP)])

# =========================================================================
# LAW 3 -- the three hooks. Any one missing and the reorder is v603 again.
# =========================================================================
if "_reorderWidgetsToDisplay(self)" not in JS:
    _fail("the rows are never permuted -- the filters stay below the prompts and the whole cut "
          "does nothing")
# v606: the save path no longer MAPS an array -- it swings the ROWS back to canon for
# the length of one serialize() call, so the array LiteGraph builds is canon by
# construction. No permutation table on the save path means nothing there to get
# backwards. But the swing must actually happen, and it must happen in serialize()
# itself, not in the onSerialize callback the base method may or may not invoke.
ser = JS[JS.index("nodeType.prototype.serialize = function"):]
ser = ser[:ser.index("\n        };")]
if "_canonOrder(this)" not in ser:
    _fail("serialize() does not swing the rows back to CANON. node.widgets is in DISPLAY order, "
          "and LiteGraph builds widgets_values straight out of it -- so the disk would get display "
          "order, and the next load would read it as canon and scramble it. THIS IS v603 with a "
          "different hook.")
if "_reorderWidgetsToDisplay(this)" not in ser:
    _fail("serialize() swings the rows to canon and LEAVES them there -- so the filters jump back "
          "below the prompts the moment the user hits save. The swing must swing back.")
# v606: configure() MUST NOT be load-bearing. It was measured NOT FIRING in Frank's
# frontend -- his rows moved (onNodeCreated) while his values did not (configure).
# I built on it three times before checking. So the rule is inverted now: nothing
# that matters may live in that hook.
cfg = JS[JS.index("nodeType.prototype.configure = function (info)"):]
cfg = cfg[:cfg.index("\n        };")]
if "_canonToDisplay" in cfg:
    _fail("configure() is mapping canon -> display again. THAT HOOK DOES NOT FIRE HERE. It was "
          "proven from Frank's own screen: the rows permuted and the values did not, and those "
          "two live in different hooks. Anything load-bearing must move to the onNodeCreated "
          "timeout, which is proven to run.")
if "_rescueScrambled(self)" not in JS:
    _fail("nothing rescues a graph poisoned by v603-v605. Those files may carry display order on "
          "disk, and the rescue must read the VALUES (a prompt is not a separator) because it "
          "cannot ask configure() -- configure() does not answer.")
if "nodeType.prototype.serialize = function" not in JS:
    _fail("the save path hooks onSerialize instead of serialize. onSerialize is a CALLBACK the "
          "base method chooses to invoke -- and this frontend has already been caught declining "
          "to invoke a base method. serialize() IS the method; there is no saving without it.")

# =========================================================================
# LAW 4 -- RUN THE ROUND TRIP. canon -> display -> canon must be the IDENTITY.
# =========================================================================
if shutil.which("node") is None:
    print("[test_v604_canon] SKIP: node not on PATH")
    sys.exit(0)


def _lift(sig):
    src = JS[JS.index(sig):]
    return src[:src.index("\n}") + 2]


harness = """
const CANON = %s;
const DISPLAY = %s;
const C2D = DISPLAY.map((n) => CANON.indexOf(n));
const D2C = CANON.map((n) => DISPLAY.indexOf(n));
%s
%s

// A real graph, in CANON order, exactly as it sits on Frank's disk.
const canon = [
    2,
    "SCENE: a cyborg", "SUBJECT: close-up", "", "", "", "",
    true,
    "bad anatomy, blurry",
    "comma", true, false, "append", "// #",
];

const display = _canonToDisplay(canon);
const back = _displayToCanon(display);

// v606 END-TO-END, without configure() lifting a finger.
//
//   1. LiteGraph pours the CANON array onto node.widgets, which is still in CANON
//      order. Values land correctly. Nothing was mapped, because nothing had to be.
//   2. The timeout permutes the ROWS. The values ride along inside their widgets.
//   3. serialize() swings the rows back to canon, reads them off, swings them back.
//
// The array that reaches the disk must be byte-for-byte the one that left it.
const mk = (n, v) => ({ name: n, value: v });
let widgets = CANON.map((n, i) => mk(n, canon[i]));      // step 1

const by = new Map(widgets.map((w) => [w.name, w]));      // step 2 (row permutation)
widgets = DISPLAY.map((n) => by.get(n));

const onScreen = {};                                       // what Frank sees
widgets.forEach((w) => { onScreen[w.name] = w.value; });

const by2 = new Map(widgets.map((w) => [w.name, w]));      // step 3 (serialize)
const canonRows = CANON.map((n) => by2.get(n));
const written = canonRows.map((w) => w.value);

console.log(JSON.stringify({ canon, display, back, onScreen, written }));
""" % (json.dumps(CANON_JS), json.dumps(DISPLAY_JS),
       _lift("function _canonToDisplay(wv)"),
       _lift("function _displayToCanon(wv)"))

with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as fh:
    fh.write(harness)
    path = fh.name
try:
    res = subprocess.run(["node", path], capture_output=True, text=True, timeout=30)
finally:
    os.unlink(path)
if res.returncode != 0:
    _fail("the mapping did not run: %s" % res.stderr.strip()[:300])
g = json.loads(res.stdout.strip().splitlines()[-1])

# THE ROUND TRIP. If this is not the identity, every save eats the graph a bit more.
if g["back"] != g["canon"]:
    _fail("canon -> display -> canon is NOT the identity.\n    in:  %s\n    out: %s\n  Every save "
          "rotates the values one more turn. The graph rots a little each time it is opened, and "
          "nothing warns anybody." % (g["canon"], g["back"]))

# THE ONE THAT MATTERS: what LiteGraph writes to disk, with configure() asleep.
if g["written"] != g["canon"]:
    _fail("the array that reaches the DISK is not the one that left it.\n    disk in:  %s\n"
          "    disk out: %s\n  This is the whole contract. Every save would rotate the graph one "
          "more turn." % (g["canon"], g["written"]))

# And what Frank actually sees on screen, values intact, filters on top.
osc = g["onScreen"]
if osc.get("separator") != "comma" or osc.get("pos_1") != "SCENE: a cyborg":
    _fail("on screen: separator=%r pos_1=%r. The values did not ride along with their widgets -- "
          "this is Frank's screenshot, again." % (osc.get("separator"), osc.get("pos_1")))

# The display really does put the filters first, with the right values in them.
d = dict(zip(DISPLAY_JS, g["display"]))
for name, want in (("separator", "comma"), ("strip_comments", True), ("strip_newlines", False),
                   ("external_mode", "append"), ("comment_markers", "// #"),
                   ("pos_1", "SCENE: a cyborg"), ("use_negative", True),
                   ("neg_1", "bad anatomy, blurry")):
    if d.get(name) != want:
        _fail("on screen, `%s` would show %r instead of %r -- the display map is wrong, and this "
              "is exactly what Frank photographed" % (name, d.get(name), want))

# The rescue (v606) reads VALUES, not the file -- it cannot ask configure(). A prompt
# is not a separator word, not a mode word, not a boolean. Pin that it demands TWO
# independent witnesses before touching anything: one stray value must not trigger a
# scramble of a healthy graph.
if "_looksScrambled" not in JS:
    _fail("the value-based rescue is gone. Poisoned graphs from v603-v605 carry shifted values, "
          "and configure() cannot be asked which kind a graph is -- so the widgets must be read.")
if "SEPARATORS.includes" not in JS or "MODES.includes" not in JS:
    _fail("the rescue no longer checks the filter values against their allowed sets. That check "
          "is how it tells a shifted array (prose in `separator`) from a healthy one.")


# =========================================================================
# LAW 5 (v605) -- THE PERMUTATION MUST SURVIVE THE DOM WIDGET.
#
# v604 shipped this and it NEVER RAN ONCE:
#
#     if (rows.length !== node.widgets.length) return;
#
# node.widgets is not the canon. _buildPane appends a DOM status bar first, so it
# holds 15 entries against DISPLAY's 14 -- the lengths differ, the guard bails, and
# Frank's filters stayed exactly where they were. The check was not wrong; it was
# asked the wrong question, and answered it perfectly.
#
# And the pane must land at the BACK. LiteGraph applies widgets_values against
# node.widgets BY POSITION, and the pane is serialize:false -- the 14 saved values
# have to meet the 14 canon widgets first. Put the pane anywhere but last and every
# value after it shifts by one, in silence.
#
# RUN IT, with the pane in place, because that is the only version that is real.
# =========================================================================
harness2 = """
const CANON = %s;
const DISPLAY = %s;
%s

const mk = (n) => ({ name: n });
// v613 paints the counter (no DOM pane), so onNodeCreated hands over the 14 canon
// widgets only. But the rule still matters: any NON-canon widget added later must ride
// last (widgets_values loads by position). Test that rule with a generic extra.
const node = { widgets: CANON.map(mk).concat([mk("extra_widget")]) };

_reorderWidgetsToDisplay(node);
console.log(JSON.stringify({
    names: node.widgets.map((w) => w.name),
    flagged: !!node._plsDisplayed,
}));
""" % (json.dumps(CANON_JS), json.dumps(DISPLAY_JS),
       _lift("function _reorderWidgetsToDisplay(node)"))

with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as fh:
    fh.write(harness2)
    path2 = fh.name
try:
    res2 = subprocess.run(["node", path2], capture_output=True, text=True, timeout=30)
finally:
    os.unlink(path2)
if res2.returncode != 0:
    _fail("the row permutation did not run: %s" % res2.stderr.strip()[:300])
r2 = json.loads(res2.stdout.strip().splitlines()[-1])

if not r2["flagged"]:
    _fail("_reorderWidgetsToDisplay BAILED OUT with a non-canon widget present. v604 verbatim "
          "was a length mismatch (DISPLAY vs node.widgets) that made it refuse to run at all -- "
          "the permutation never ran and the filters never moved.")

if r2["names"][:len(DISPLAY_JS)] != DISPLAY_JS:
    _fail("the rows are not in DISPLAY order: %s" % r2["names"][:len(DISPLAY_JS)])

if r2["names"][-1] != "extra_widget":
    _fail("a non-canon widget is not LAST (%s). widgets_values loads BY POSITION into the canon "
          "widgets, so anything non-canon must ride last or every value behind it shifts one slot, "
          "silently. Last is not a preference, it is the contract." % r2["names"][-1])

# =========================================================================
# LAW 6 (v613) -- AUTO-FIT with a STOP, loop-free; the counter is PAINTED
# =========================================================================
# The v613 height model: each field grows to show ALL its text and STOPS there (no
# whitespace past the text), and the node fits the sum. The two things that made v610
# march the node off-screen must NEVER return: an onResize hook (fought native resize /
# re-measured) and a ResizeObserver (re-fired setSize). _refit runs only from real
# events and is re-entry guarded, so there is no feedback loop.
for banned, why in (
    ('resize = "vertical"', "a per-box drag handle is back"),
    ("_onNodeResize", "a node-edge redistributor is back"),
    ("_plsSyncing", "the sync-fight flag is back"),
    ("plsBoxH", "per-box dragged heights are being stored again"),
    ("BOX_H", "the v608 fixed-box height model is back"),
    ("_fitBox", "the v608 fit is back"),
    ("_styleBox", "the v608/v611 per-field styler is back"),
    ("_makeAllResizable", "the v608 field-height installer is back"),
    ("_allocate", "the v610 content-weight distributor is back"),
    ("_spread", "the v611 delta-spread is back"),
    ("_sizeToFit", "the v611 programmatic refit is back"),
    ("_growToContent", "the v610 auto-grow driver is back"),
    ("_distribute", "the v610 distributor is back"),
    ("NODE_MAX", "the v610 node-height cap is back"),
    ("prototype.onResize", "an onResize hook is back -- it fought native resize and re-measured (v610/v611 loop)"),
    ("new ResizeObserver", "a ResizeObserver is back -- it re-fired setSize (half the v610 loop)"),
):
    for _ln in JS.splitlines():
        _s = _ln.lstrip()
        if banned in _ln and not _s.startswith("//") and not _s.startswith("*"):
            _fail("a known height dead-end is back in live code (%r): %s" % (banned, why))

# v613 auto-fit must be WIRED: the refit, the content measure, the visible-field set, the
# field wiring, and the re-entry guard that makes the no-loop guarantee explicit.
for _need in ("_refit", "_contentH", "_visibleFields", "_wireField", "_plsRefitting"):
    if _need not in JS:
        _fail("%s is missing -- the v613 auto-fit-with-stop model is not wired" % _need)
# the field computeSize must report the fitted height _plsH -- that IS the stop.
if "_plsH" not in JS:
    _fail("fields no longer report a fitted height (_plsH) -- nothing tells LiteGraph where to "
          "stop growing, so the endless-whitespace bug returns")

# The wordcount is PAINTED (onDrawForeground), not a DOM widget riding last (that got
# pushed off the bottom -- Frank: dead for versions). Orange past 512 tokens, nothing else.
if "onDrawForeground" not in JS or "_counterText" not in JS:
    _fail("the wordcount counter is not painted (onDrawForeground/_counterText) -- a DOM pane "
          "rides last and gets hidden, which is exactly the bug being fixed")
if "#ff8c00" not in JS:
    _fail("the amber (#ff8c00) footer style is gone -- Frank keeps it as a style element (the "
          "over-limit warning was retired; the limit is model-dependent and lives in the Token "
          "Counter node with a manual model_limit)")
if 'addDOMWidget("pls_cte_view"' in JS:
    _fail("the DOM status pane is back -- the counter must be painted on the node now")

# =========================================================================
# LAW 6b (v618) -- the counter is a painted FOOTER of HARD FACTS; use_negative has air.
#
# v613-617 built the painted footer; v618 RETIRES the over-limit warning from it -- the
# limit is model-dependent and lives in the Token Counter node (manual model_limit), which
# owns the warning. The footer shows hard facts only (pos/neg words, chars, pos/neg tokens,
# method) and the amber is a STYLE element (always on), not a data-driven warning. RUN
# _counterText and pin the hard facts; pin no limit/warning crept back; pin the band, the
# amber, and the use_negative air.
# =========================================================================
_dfg = JS[JS.index("nodeType.prototype.onDrawForeground = function"):]
_dfg = _dfg[:_dfg.index("\n        };")]
if "ctx.fill()" not in _dfg or "ctx.stroke()" not in _dfg:
    _fail("the counter is a bare watermark again -- no band fill + separator line (fill + "
          "stroke). A faint string with no bar reads as invisible on Frank's dark canvas.")
if "_barHeight(this)" not in _dfg or "size[1]" not in _dfg:
    _fail("the counter is not drawn in the FOOTER band at the node's foot "
          "(_barHeight + size[1]). Frank asked for it back in the footer, not a "
          "title chip. v715: the fixed BAR_H became the measured _barHeight, so "
          "the band can wrap -- but it is still the band at the foot.")
if "_barLines(this)" not in _dfg:
    _fail("the footer must paint the rows _barLines produced -- the SAME "
          "function that reserved the height, so the strip and its content "
          "cannot disagree (v715)")
if "BAR_FONT" not in _dfg:
    _fail("the footer no longer uses BAR_FONT")
if "#ff8c00" not in _dfg:
    _fail("the amber (#ff8c00) footer text is gone -- Frank keeps it as a style element")
if "orange" in _dfg:
    _fail("the footer draw re-introduced conditional 'orange' colouring -- the amber is a STYLE "
          "element now (always on), not a data-driven warning; the warning lives in the counter node")

# _refit must RESERVE the footer band and APPLY the use_negative air -- both live in _refit,
# checked against its code with comment lines stripped (BAR_H is also used in the draw, and a
# comment mention of either must not fool these into passing).
_rf = JS[JS.index("function _refit(node)"):]
_rf = _rf[:_rf.index("\n}") + 2]
_rf_code = "\n".join(_l for _l in _rf.splitlines() if not _l.lstrip().startswith("//"))
# v715: the reservation moved OUT of _refit and INTO computeSize -- that move is
# the fix. _refit reserved it alone, so LiteGraph never knew the band existed and
# dragging the node's bottom edge buried it under the neg_1 field. Pin the new
# home, and pin that _refit does NOT add it a second time (that would park the
# node one whole band too tall).
if "nodeType.prototype.computeSize" not in JS or "_barHeight(this)" not in JS:
    _fail("computeSize must RESERVE the footer band (size[1] += _barHeight) -- "
          "without it LiteGraph does not know the band exists and a manual "
          "resize puts it under the last field (v715, the ph_reference pattern)")
if "_barHeight(node)" in _rf_code:
    _fail("_refit must NOT add the band again -- computeSize already carries it "
          "(v715); counting it twice parks the node one band too tall")
if "NEG_GAP" not in JS or "NEG_GAP" not in _rf_code:
    _fail("the use_negative air (NEG_GAP) is gone or never applied in _refit -- Frank asked for "
          "a little air above the toggle; declaring the constant (or naming it in a comment) is "
          "not enough, it must pad the last positive field's reserved height inside _refit")

# RUN _counterText and pin the HARD FACTS. It returns a plain string now (no orange flag).
_MAX_SEG = int(re.search(r"const MAX_SEGMENTS = (\d+);", JS).group(1))
harness3 = """
const MAX_SEGMENTS = %d;
%s
%s
%s
%s

const mkw = (name, value) => ({ name, value });
function nodeWith(posText, posTok, negText, negTok) {
    const node = { widgets: [
        mkw("segments", 1),
        mkw("pos_1", posText), mkw("pos_2",""), mkw("pos_3",""),
        mkw("pos_4",""), mkw("pos_5",""), mkw("pos_6",""),
        mkw("use_negative", true), mkw("neg_1", negText || ""),
        mkw("comment_markers", "//"), mkw("strip_comments", true),
    ] };
    if (posTok != null) node._cteTokens = { pos: posTok, neg: negTok, method: "exact",
                                            posLen: posText.length, negLen: (negText||"").length };
    return node;
}

console.log(JSON.stringify({
    preRun: _counterText(nodeWith("a short prompt", null, "bad hands", null)),   // no run yet
    run:    _counterText(nodeWith("a short prompt", 554, "bad hands", 331)),     // pos + neg tokens
}));
""" % (_MAX_SEG,
       _lift("function _w(node, name)"),
       _lift("function _countText(rawTxt, node)"),
       _lift("function _liveCount(node)"),
       _lift("function _counterText(node)"))

with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as fh:
    fh.write(harness3)
    path3 = fh.name
try:
    res3 = subprocess.run(["node", path3], capture_output=True, text=True, timeout=30)
finally:
    os.unlink(path3)
if res3.returncode != 0:
    _fail("the counter logic did not run: %s" % res3.stderr.strip()[:300])
c = json.loads(res3.stdout.strip().splitlines()[-1])
_pre, _run = c["preRun"], c["run"]

# HARD FACTS: pos/neg words + chars, and after a run the pos AND neg token counts + method.
for _s in (_pre, _run):
    if "pos " not in _s or "neg " not in _s or "words" not in _s or "chars" not in _s:
        _fail("the footer lost its pos/neg word + char readout -- Frank wants all the hard facts "
              "('pos N words', 'neg N words', 'K chars')")
if "last run:" not in _run or "tokens" not in _run:
    _fail("the footer does not show the token counts after a run")
if "554" not in _run or "331" not in _run:
    _fail("the footer does not show BOTH the pos (554) and neg (331) token counts -- Frank asked "
          "for all the hard facts, and the negative token count is one of them")
if "(exact)" not in _run:
    _fail("the footer does not show the tokenizer method (exact/heuristic)")
if "run to count tokens" not in _pre:
    _fail("the pre-run footer does not tell the user to run to get tokens -- before a run the "
          "browser has no tokenizer, and silence reads as 'the counter is broken'")
# The WARNING was RETIRED: no hard-coded limit, no over-budget language in the readout. The
# limit is model-dependent and lives in the Token Counter node -- it must not creep back here.
for _bad in ("OVER", "/512", "budget", "limit"):
    if _bad in _run:
        _fail("the footer re-introduced a limit/warning (%r) -- that was retired in v618; the "
              "limit is model-dependent and lives in the Token Counter node, not hard-coded here"
              % _bad)

print("[test_v604_canon] PASS: the canon is frozen (append-only, v585's law, now enforced in the "
      "file that broke it); JS CANON matches INPUT_TYPES slot for slot; the filters sit above the "
      "prompts ON SCREEN ONLY; all three hooks are wired; and the round trip was RUN -- "
      "canon -> display -> canon is the identity, a v603-poisoned file is rescued, and a healthy "
      "one is not touched; a non-canon widget rides last; the fields AUTO-FIT to content and stop "
      "(no whitespace) with no onResize/ResizeObserver loop; and the painted footer shows the hard "
      "facts (pos/neg words, chars, pos/neg tokens, method) with the amber as a style element and "
      "no hard-coded limit, not a DOM pane that hides.")
sys.exit(0)
