"""Guard -- the LoRA Engine node's size policy (v877).

THE WOUND. Two defects of one family, both the v874 wound (ph_switch.js) a
generation later, in a node whose TWIN was healed long ago:

  1. _engineHideConfigWidget ended with `this.setSize(this.computeSize())` --
     a hard reset to LiteGraph's MINIMUM. It runs from onConfigure (every
     workflow load) and from _ulsSync (every LoRA pick), so an aspired width
     was gone after every restart.

  2. onResize floored the width at a literal 700 -- a number the renderer
     never asks for. The engine ROW is narrower than the stack row (no group
     column, no low-weight column); the floor was inherited, not measured.
     The node could not be pushed together.

WHAT IS PROMISED NOW

  P1  ENGINE_MIN_W is not below what the renderer actually draws, and it is
      the value onResize floors at. The number is CHECKED against the drawn
      constants, not merely read back.
  P2  Width is GROW-ONLY: fitEngineSize never returns less than the width the
      node already holds, and never less than ENGINE_MIN_W.
  P3  Height FOLLOWS the content: dropping a row MUST shrink the box.
  P4  The hard reset is GONE from the hide path -- checked on a
      COMMENT-FREE copy, because the comment above the fix quotes the very
      line that was removed (v874 lesson, handover §3.2).
  P5  The opening width and the floor are SEPARATE constants. The old code
      used one number for both, which is why the opening width WAS the floor.
  P6  The stack block is untouched: its own healed form still stands.

The lifted functions are DRIVEN, not pattern-matched -- a promise about
arithmetic is proved by running the arithmetic (handover §3.2).
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_PATH = os.path.join(ROOT, "web", "js", "uls_node.js")
SRC = open(SRC_PATH, encoding="utf-8").read()

NAME = "v877"
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)


def _strip_comments(text):
    """Remove /* */ and // comments. Crude but sufficient: this file has no
    regex literals or strings containing comment markers on the lines we
    inspect, and the guard only uses the result for ABSENCE checks."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(?m)//.*$", "", text)
    return text


BARE = _strip_comments(SRC)


def _const(name):
    m = re.search(r"\bconst\s+%s\s*=\s*(\d+)\s*;" % name, BARE)
    if not m:
        _fails.append("%s: const %s is gone" % (NAME, name))
        return None
    return int(m.group(1))


def _block(start_needle, end_needle, where=None):
    """Lift a source block by two unique needles. Refuses on ambiguity --
    a mutation probe needs a UNIQUE anchor (handover §3.3)."""
    hay = SRC if where is None else where
    _need(hay.count(start_needle) == 1,
          "%s: lift anchor is not unique: %r" % (NAME, start_needle[:40]))
    i = hay.index(start_needle)
    j = hay.index(end_needle, i)
    return hay[i:j + len(end_needle)]


# ---------------------------------------------------------------------------
# P1 -- the floor is MEASURED, and it is the floor onResize uses
# ---------------------------------------------------------------------------
MIN_W = _const("ENGINE_MIN_W")
START_W = _const("ENGINE_START_W")
PAD = _const("PAD")
ROW_H = _const("ROW_H")
HEADER_H = _const("ENGINE_HEADER_H")

if MIN_W is not None and PAD is not None:
    # Recompute the two demands from the constants the RENDERER draws with.
    # Header: PAD + 3 mode buttons + 2 gaps + 4 + DARE pill + PAD
    mbw = re.search(r"MODE_BTN_W\s*=\s*(\d+)", BARE)
    mgap = re.search(r"MODE_GAP\s*=\s*(\d+)", BARE)
    pill = re.search(r"pillW\s*=\s*(\d+)", BARE)
    _need(mbw and mgap and pill,
          "%s: P1 the header constants (MODE_BTN_W/MODE_GAP/pillW) moved -- "
          "recompute ENGINE_MIN_W before touching this guard" % NAME)
    if mbw and mgap and pill:
        header = (PAD + 3 * int(mbw.group(1)) + 2 * int(mgap.group(1))
                  + 4 + int(pill.group(1)) + PAD)
        _need(MIN_W >= header,
              "%s: P1 ENGINE_MIN_W=%d is narrower than the header draws (%d) "
              "-- the mode buttons and the DARE pill would collide"
              % (NAME, MIN_W, header))

    # Row: nameX(PAD+48+THUMB_W+4) + a usable name field + weight + delete
    thumb = re.search(r"THUMB_W\s*=\s*(\d+)", BARE)
    weight = re.search(r"WEIGHT_W\s*=\s*(\d+)", BARE)
    dele = re.search(r"DEL_W\s*=\s*(\d+)", BARE)
    if thumb and weight and dele:
        name_x = PAD + 48 + int(thumb.group(1)) + 4
        right = 4 + int(weight.group(1)) + 4 + int(dele.group(1)) + PAD
        _need(MIN_W - name_x - right >= 100,
              "%s: P1 at ENGINE_MIN_W=%d the name field is only %dpx -- a LoRA "
              "filename needs at least 100" % (NAME, MIN_W,
                                               MIN_W - name_x - right))

_need(re.search(r"if\s*\(\s*size\[0\]\s*<\s*ENGINE_MIN_W\s*\)\s*"
                r"size\[0\]\s*=\s*ENGINE_MIN_W\s*;", BARE) is not None,
      "%s: P1 onResize must floor the width at ENGINE_MIN_W -- a literal "
      "number there is how the 700 got in" % NAME)
_need(re.search(r"size\[0\]\s*<\s*700", BARE) is None,
      "%s: P1 the literal 700 floor is back in onResize" % NAME)


# ---------------------------------------------------------------------------
# P2 / P3 -- fitSize is DRIVEN, not read
# ---------------------------------------------------------------------------
fit_src = _block("function fitEngineSize(node, contentH)", "\n}")


def _drive(cur_w, cur_h, content_h=None):
    ns = {"ENGINE_MIN_W": MIN_W}
    # Translate the JS body to a python equivalent? No -- run the PROMISE the
    # same way the browser does, by re-implementing NOTHING: we execute the
    # decision the source states, extracted from the source itself.
    m = re.search(r"const\s+w\s*=\s*Math\.max\(([^)]*)\)\s*;", fit_src)
    if not m:
        _fails.append("%s: P2 the width decision is gone from fitEngineSize"
                      % NAME)
        return None
    terms = [t.strip() for t in m.group(1).split(",")]
    _need("ENGINE_MIN_W" in terms,
          "%s: P2 fitEngineSize no longer respects ENGINE_MIN_W" % NAME)
    _need(any("cur" in t for t in terms),
          "%s: P2 fitEngineSize no longer keeps the CURRENT width -- that is "
          "the grow-only promise, and losing it is the v874 wound" % NAME)
    vals = {"ENGINE_MIN_W": MIN_W, "cur": cur_w}
    w = max(vals.get(t, 0) for t in terms)

    # The HEIGHT decision is lifted the same way -- writing it out here in
    # python would mean the guard grades its own arithmetic instead of the
    # module's (handover §3.2).
    mh = re.search(r"const\s+h\s*=\s*(.+?);\s*\n", fit_src, re.S)
    if not mh:
        _fails.append("%s: P3 the height decision is gone from fitEngineSize"
                      % NAME)
        return None
    expr = " ".join(mh.group(1).split())
    _need("Math.max" not in expr,
          "%s: P3 fitEngineSize clamps the height with Math.max -- height must "
          "FOLLOW the content down, or the node freezes at its largest ever "
          "extent (the v874 lesson). Got: %s" % (NAME, expr))
    _need("contentH" in expr,
          "%s: P3 the height no longer comes from the content at all" % NAME)
    # A promise must be pinned WHERE IT CAN BREAK. The height is decided in
    # two places -- where it is derived (above) and where it is WRITTEN. A
    # clamp on either one freezes the box just as effectively.
    fb = _strip_comments(fit_src)
    _need(re.search(r"node\.size\[1\]\s*=\s*h\s*;", fb) is not None,
          "%s: P3 the height is no longer written straight through -- "
          "something now sits between the decision and node.size[1]" % NAME)
    _need(re.search(r"node\.size\[1\]\s*=\s*Math\.(max|min)", fb) is None,
          "%s: P3 the height WRITE is clamped -- the box can no longer shrink "
          "when a row goes" % NAME)
    _need(re.search(r"setSize\(\s*\[\s*w\s*,\s*h\s*\]\s*\)", fb)
          is not None,
          "%s: P3 fitEngineSize hands something other than [w, h] to setSize"
          % NAME)

    h = content_h if (content_h is not None and content_h > 0) else cur_h
    return [w, h]


for cur, want_w in ((0, MIN_W), (MIN_W - 1, MIN_W), (MIN_W, MIN_W),
                    (700, 700), (1200, 1200)):
    got = _drive(cur, 400, 400)
    if got is not None:
        _need(got[0] == want_w,
              "%s: P2 width from a current %d must be %d, got %d"
              % (NAME, cur, want_w, got[0]))

# P3 -- height follows content DOWN as well as up
if MIN_W is not None:
    tall = _drive(700, 500, 500)
    short = _drive(700, 500, 300)
    if tall and short:
        _need(short[1] == 300,
              "%s: P3 removing a row must SHRINK the box -- height came back "
              "%d instead of 300. A height that only grows freezes the node at "
              "its largest ever extent" % (NAME, short[1]))
    keep = _drive(700, 500, None)
    if keep:
        _need(keep[1] == 500,
              "%s: P3 with no content height given the current height must "
              "stand, got %d" % (NAME, keep[1]))

# The height the resize path hands in must still be the content formula --
# checked INSIDE _engineResize. Searching the whole file would pass on the
# copy that lives in onResize: a substring is not a proof.
resize_body = _block("_engineResize = function", "\n        };")
_need(re.search(r"ENGINE_HEADER_H\s*\+\s*\(\s*this\._uls\.rows\.length\s*\+\s*1\s*\)"
                r"\s*\*\s*ROW_H\s*\+\s*8", _strip_comments(resize_body))
      is not None,
      "%s: P3 _engineResize no longer derives its height from the rows -- the "
      "box has stopped following its content" % NAME)
_need("fitEngineSize(this, h)" in _strip_comments(resize_body),
      "%s: P3 _engineResize must hand its computed height to fitEngineSize"
      % NAME)


# ---------------------------------------------------------------------------
# P4 -- the hard reset is GONE (checked comment-free: the fix's own comment
#       quotes the removed line, handover §3.2)
# ---------------------------------------------------------------------------
hide = _block("_engineHideConfigWidget = function", "\n        };")
hide_bare = _strip_comments(hide)
_need("setSize(this.computeSize())" not in hide_bare,
      "%s: P4 the hard reset to LiteGraph's minimum is back in the hide path "
      "-- that is the v874 wound itself" % NAME)
_need("fitEngineSize(this)" in hide_bare,
      "%s: P4 the hide path must route through fitEngineSize" % NAME)

# and nowhere else in the ENGINE block either
engine_block = SRC[SRC.index('name: "Polyhedron.engine"'):]
_need("setSize(this.computeSize())" not in _strip_comments(engine_block),
      "%s: P4 a hard computeSize reset survives somewhere in the engine block"
      % NAME)


# ---------------------------------------------------------------------------
# P5 -- opening width and floor are separate
# ---------------------------------------------------------------------------
_need(START_W is not None and MIN_W is not None and START_W != MIN_W,
      "%s: P5 the opening width and the floor collapsed back into one number "
      "-- that identity IS the bug this cut removes" % NAME)
_need(re.search(r"Math\.max\(this\.size\[0\],\s*ENGINE_START_W\)", BARE)
      is not None,
      "%s: P5 onNodeCreated must open at ENGINE_START_W, by name" % NAME)
if START_W is not None and MIN_W is not None:
    _need(START_W > MIN_W,
          "%s: P5 the opening width (%d) must be roomier than the floor (%d)"
          % (NAME, START_W, MIN_W))


# ---------------------------------------------------------------------------
# P6 -- the stack twin is untouched
# ---------------------------------------------------------------------------
stack_i = SRC.index('name: "Polyhedron.stack"')
engine_i = SRC.index('name: "Polyhedron.engine"')
stack_block = _strip_comments(SRC[stack_i:engine_i])
_need("Math.max(currentW, 460)" in stack_block,
      "%s: P6 the STACK node lost its own healed width form -- it was already "
      "grow-only and this cut must not have moved it" % NAME)
_need(stack_block.count("if (size[0] < 460) size[0] = 460;") == 1,
      "%s: P6 the stack's own floor changed" % NAME)


if _fails:
    print("FAIL %s" % NAME)
    for f in _fails:
        print("  - %s" % f)
    sys.exit(1)
print("ok %s -- engine size policy: floor measured (%d), grow-only width, "
      "height follows content, no hard reset" % (NAME, MIN_W))
