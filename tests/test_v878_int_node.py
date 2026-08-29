"""Guard -- ⬡ Polyhedron Int (v878, RE-GROUNDED in v879).

v879 rebuilt the frontend: the canon widget is visible again and the drawn
number band is gone. P6 is re-grounded (the height floor is now REQUIRED, not
forbidden -- see the block itself), and P8/P9 are new. P1-P5 and P7 stand
unchanged; the backend was not touched.

The replacement for core's PrimitiveInt.

WHAT THIS NODE REPLACED. Three core `Int` nodes stood in the MiniMax graph:
"Int (Full)" = 20, "Int (Lightning LoRA)" = 4, and a route selector 1/2. The
names lived in node TITLES, which nothing reads. Core's node was measured
before this was written (comfy_extras/nodes_primitive.py, 2026-08-26): one
`value` input, `control_after_generate=fixed`, value returned unchanged.

PROMISES

  P1  `value` is the single source of truth. The INT output is the value,
      clamped, whatever the presets say. A preset is a way to WRITE it.
  P2  `label` resolves BY VALUE, not by what was clicked: a hand-typed 20
      reports "Basis" and a hand-typed 7 reports nothing. Proved by DRIVING
      the node, not by reading it.
  P3  A damaged preset config NEVER takes a run down. Telemetry may lose
      precision; it may not kill the run (handover §3.1).
  P4  Widget order is `value` then `preset_config`, and the hidden one is
      LAST -- LiteGraph restores widget values positionally (guard #577).
  P5  The frontend keeps NO private copy of the value: every path writes
      through the canon widget. Two places holding one number drift.
  P6  Size policy is the v877 form from the first line: a MEASURED floor,
      grow-only width, height following the content, and no
      setSize(computeSize()) anywhere.
  P7  The chip rectangles are computed ONCE and read by both painter and hit
      test -- one geometry, not two that can disagree.
"""

import ast
import importlib.util
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY_PATH = os.path.join(ROOT, "nodes", "ph_int.py")
JS_PATH = os.path.join(ROOT, "web", "js", "ph_int.js")

NAME = "v878"
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)


spec = importlib.util.spec_from_file_location("_ph_int_guard", PY_PATH)
MOD = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MOD)

JS = open(JS_PATH, encoding="utf-8").read()
PY_SRC = open(PY_PATH, encoding="utf-8").read()


def _strip_js_comments(text):
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"(?m)//.*$", "", text)
    return text


JS_BARE = _strip_js_comments(JS)

CFG = json.dumps({"presets": [{"name": "Basis", "value": 20},
                              {"name": "Turbo", "value": 4}]})


# ---------------------------------------------------------------------------
# P1 / P2 -- the node is DRIVEN
# ---------------------------------------------------------------------------
node = MOD.ULSInt()

for value, want_label in ((20, "Basis"), (4, "Turbo"), (7, ""), (0, "")):
    got = node.emit(value, CFG)
    _need(got[0] == value,
          "%s: P1 emit(%r) must pass the value through, got %r"
          % (NAME, value, got[0]))
    _need(got[1] == want_label,
          "%s: P2 emit(%r) label must be %r, got %r -- the label resolves by "
          "VALUE, not by what was clicked last"
          % (NAME, value, want_label, got[1]))

# a preset list cannot override the value
weird = json.dumps({"presets": [{"name": "Basis", "value": 999}]})
_need(node.emit(20, weird)[0] == 20,
      "%s: P1 a preset must never rewrite the emitted value -- that would be "
      "a second source of truth" % NAME)

# duplicate values: topmost wins, because that is the one the eye reads first
dup = json.dumps({"presets": [{"name": "First", "value": 5},
                              {"name": "Second", "value": 5}]})
_need(node.emit(5, dup)[1] == "First",
      "%s: P2 with a duplicated value the FIRST preset must name it" % NAME)

# clamping
_need(node.emit(10 ** 12, "")[0] == MOD.INT_MAX,
      "%s: P1 a value above INT_MAX must clamp, not overflow" % NAME)
_need(node.emit(-10 ** 12, "")[0] == MOD.INT_MIN,
      "%s: P1 a value below INT_MIN must clamp" % NAME)


# ---------------------------------------------------------------------------
# P3 -- damage degrades, never crashes
# ---------------------------------------------------------------------------
for junk in ("{kaputt", "[]", "null", "42", '{"presets": "nope"}',
             '{"presets": [{"name": "x"}]}',
             '{"presets": [{"value": "nope"}]}',
             '{"presets": [null, 7, {"name":"ok","value":3}]}', "", None):
    try:
        out = node.emit(11, junk)
    except Exception as exc:                       # noqa: BLE001
        _fails.append("%s: P3 preset_config %r killed the node: %r"
                      % (NAME, junk, exc))
        continue
    _need(out[0] == 11,
          "%s: P3 with preset_config %r the value must still be emitted, "
          "got %r" % (NAME, junk, out[0]))

# a non-integer value must not raise either
for bad in ("nope", None, [], {}):
    try:
        _need(node.emit(bad, CFG)[0] == 0,
              "%s: P3 a non-integer value must degrade to 0, got %r"
              % (NAME, node.emit(bad, CFG)[0]))
    except Exception as exc:                       # noqa: BLE001
        _fails.append("%s: P3 value %r killed the node: %r" % (NAME, bad, exc))

# and the last survivor of a damaged list still counts
_need(node.emit(3, '{"presets": [null, 7, {"name":"ok","value":3}]}')[1] == "ok",
      "%s: P3 a partly damaged list must still yield its readable rows" % NAME)


# ---------------------------------------------------------------------------
# P4 -- widget order, hidden one LAST
# ---------------------------------------------------------------------------
it = MOD.ULSInt.INPUT_TYPES()
order = list(it.get("required", {})) + list(it.get("optional", {}))
_need(order == ["value", "preset_config"],
      "%s: P4 widget order is %r -- must be ['value', 'preset_config']. "
      "LiteGraph restores widgets_values POSITIONALLY; the hidden one belongs "
      "at the END or every saved workflow shifts (guard #577)"
      % (NAME, order))
_need(MOD.ULSInt.RETURN_TYPES == ("INT", "STRING")
      and MOD.ULSInt.RETURN_NAMES == ("value", "label"),
      "%s: P4 the outputs changed -- INT first, the label second, appended"
      % NAME)


# ---------------------------------------------------------------------------
# P5 -- the frontend holds no private value
# ---------------------------------------------------------------------------
_need(re.search(r"function\s+setValue\s*\(\s*node\s*,\s*v\s*\)", JS_BARE),
      "%s: P5 setValue is gone -- every write must funnel through one place"
      % NAME)
set_fn = JS_BARE[JS_BARE.index("function setValue"):]
set_fn = set_fn[:set_fn.index("\n}")]
_need('widget(node, "value")' in set_fn,
      "%s: P5 setValue no longer writes the canon widget" % NAME)
_need("w.callback" in set_fn,
      "%s: P5 setValue must fire the widget callback -- a silent write leaves "
      "anything chained to it stale" % NAME)

# nothing may stash the number on the node itself
_need(re.search(r"_phi\.(value|val|num)\b", JS_BARE) is None,
      "%s: P5 the frontend stashes the value in its own state -- that is the "
      "second source of truth this node was built to avoid" % NAME)

# every mutation goes through setValue, never straight at the widget
writes = re.findall(r'widget\(\s*this\s*,\s*"value"\s*\)\s*\.\s*value\s*=', JS_BARE)
_need(not writes,
      "%s: P5 something writes the canon widget directly instead of through "
      "setValue (%d place(s))" % (NAME, len(writes)))


# ---------------------------------------------------------------------------
# P6 -- size policy, RE-GROUNDED in v879
#
# v878 promised "onResize must NOT floor the height". That was the right rule
# aimed at the wrong end: it guards against clamping UPWARD, where a floor
# freezes a box at its largest extent. DOWNWARD the content needs a floor, or
# the node is dragged smaller than what it holds -- which is exactly what
# Frank saw. The promise now reads: the height floor is the height the LAYOUT
# computed, and nothing clamps upward.
# ---------------------------------------------------------------------------
def _js_const(name):
    m = re.search(r"\bconst\s+%s\s*=\s*(\d+)\s*;" % name, JS_BARE)
    return int(m.group(1)) if m else None


MIN_W = _js_const("MIN_W")
START_W = _js_const("START_W")

_need(MIN_W is not None and START_W is not None,
      "%s: P6 MIN_W / START_W are gone" % NAME)
_need(START_W is not None and MIN_W is not None and START_W > MIN_W,
      "%s: P6 the opening width must be roomier than the floor -- one number "
      "for both IS the v877 bug" % NAME)
_need("setSize(this.computeSize())" not in JS_BARE
      and "setSize(node.computeSize())" not in JS_BARE,
      "%s: P6 a hard computeSize reset is in the file -- the v874 wound" % NAME)

lay = JS_BARE[JS_BARE.index("function relayout"):]
lay = lay[:lay.index("\n}")]
_need(re.search(r"Math\.max\(\s*MIN_W\s*,\s*node\.size\[0\]", lay),
      "%s: P6 width is no longer grow-only above MIN_W" % NAME)
_need(re.search(r"node\.size\[1\]\s*=\s*Math\.(max|min)", lay) is None,
      "%s: P6 the height write is clamped -- the box can no longer shrink "
      "when a chip row goes" % NAME)
_need("node.computeSize()[1]" in lay,
      "%s: P6 the chip block no longer sits on LiteGraph's own height. ONE "
      "source of height (the v809 lesson) -- re-deriving the widget stack is "
      "how the two drift" % NAME)

res = JS_BARE[JS_BARE.index("onResize = function"):]
res = res[:res.index("\n        };")]
_need(re.search(r"size\[1\]\s*<\s*minH", res),
      "%s: P6 onResize does not floor the height. v878 left this out ON "
      "PURPOSE and the box could be dragged smaller than its own contents -- "
      "this promise is re-grounded, not softened" % NAME)
_need("_phi?.minH" in res or "_phi.minH" in res,
      "%s: P6 the height floor is not the one the layout computed -- a second "
      "number here would drift from the drawn chips" % NAME)


# ---------------------------------------------------------------------------
# P8 (NEW in v879) -- the canon widget KEEPS LiteGraph's input
#
# v878 hid `value` and painted its own band. That removed the click, the
# keyboard and the built-in horizontal drag. A redrawn control must do
# everything the original did.
# ---------------------------------------------------------------------------
# Pinned by EFFECT, not by spelling: a first draft looked for
# widget(...,"value").type = "hidden" and a mutation slipped past through a
# local variable. Every hiding of a widget must live in hideConfig, which
# only ever touches preset_config.
hide_fn = JS_BARE[JS_BARE.index("function hideConfig"):]
hide_fn = hide_fn[:hide_fn.index("\n}")]
_need('widget(node, "preset_config")' in hide_fn,
      "%s: P8 hideConfig no longer targets preset_config" % NAME)

outside = JS_BARE.replace(hide_fn, "")
for pattern, what in ((r'type\s*=\s*"hidden"', 'type = "hidden"'),
                      (r'\.hidden\s*=\s*true', '.hidden = true'),
                      (r'computeSize\s*=\s*\(\)\s*=>', 'computeSize override')):
    hits = re.findall(pattern, outside)
    _need(not hits,
          "%s: P8 a widget is hidden OUTSIDE hideConfig (%s, %d place(s)). "
          "Only preset_config may be hidden -- hiding the canon `value` "
          "widget takes away the click, the keyboard and LiteGraph's own "
          "drag, which is what v878 did" % (NAME, what, len(hits)))
_need("hideConfig" in JS_BARE and 'widget(node, "preset_config")'
      in JS_BARE[JS_BARE.index("function hideConfig"):],
      "%s: P8 only preset_config may be the hidden one" % NAME)


# ---------------------------------------------------------------------------
# P9 (NEW in v879) -- no layout inside a paint
#
# v878 called layout() from onDrawForeground and wrote setSize there. A size
# written mid-paint lands one frame late; while dragging it never catches up.
# ---------------------------------------------------------------------------
draw_block = JS_BARE[JS_BARE.index("onDrawForeground = function"):]
draw_block = draw_block[:draw_block.index("\n        };")]
for forbidden in ("relayout(", "setSize(", "computeSize("):
    _need(forbidden not in draw_block,
          "%s: P9 the paint calls %s -- layout belongs on EVENTS. A size "
          "written during a draw lands a frame late, and while dragging that "
          "frame never arrives" % (NAME, forbidden.rstrip("(")))


# ---------------------------------------------------------------------------
# P7 -- one geometry, read twice
# ---------------------------------------------------------------------------
zone_writes = re.findall(r"\.zones\s*=\s*[^=]", JS_BARE)
_need(len(zone_writes) == 1,
      "%s: P7 the chip zones are written in %d places -- a second "
      "measurement is how a click lands one chip to the left"
      % (NAME, len(zone_writes)))
draw_i = JS_BARE.index("onDrawForeground")
down_i = JS_BARE.index("onMouseDown")
_need("st.zones" in JS_BARE[draw_i:down_i],
      "%s: P7 the painter no longer reads the shared zones" % NAME)
_need("st.zones" in JS_BARE[down_i:],
      "%s: P7 the hit test no longer reads the shared zones -- a second "
      "measurement is how a click lands one chip to the left" % NAME)
_need(re.search(r"(?s)onMouseDown.*?x\s*\+=\s*w\s*\+\s*CHIP_GAP", JS_BARE)
      is None,
      "%s: P7 the hit test re-derives chip positions instead of reading them"
      % NAME)


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------
init_src = open(os.path.join(ROOT, "__init__.py"), encoding="utf-8").read()
_need('NODE_CLASS_MAPPINGS["ULSInt"]' in init_src,
      "%s: ULSInt is not registered" % NAME)
_need("Polyhedron Int" in init_src,
      "%s: the display name is gone" % NAME)
ast.parse(PY_SRC)


if _fails:
    print("FAIL %s" % NAME)
    for f in _fails:
        print("  - %s" % f)
    sys.exit(1)
print("ok %s -- Polyhedron Int: value is canon, label resolves by value, "
      "damage degrades, size policy is v877" % NAME)
