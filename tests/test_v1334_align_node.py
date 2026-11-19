#!/usr/bin/env python3
"""
test_v1334_align_node -- the Polyhedron Align bar comes with a node
(web/js/ph_align.js, ph_group_switch.js).

Frank, 15.11.: "als einzelnes Float-Menue ... alle Funktionen mit rein, aber
mit sauberer Standard-Leiste ... das Float laed, wenn man einen kleinen
Custom-Node-Baustein reinlaedt, eben als wuerde man eine Note-Node setzen ...
dann kaeme es nicht immer, nur, wenn man es will".

RE-GROUNDED v1336 (DECLARED): alignWanted also asks the node's own switch
(v1336), so the mutation anchor follows the longer line; the drawing stub
gains beginPath/arc/fill/rect for the toggle.

RE-GROUNDED v1335 (DECLARED): the own badge is drawn only where the frontend
pushes badges into node.badges (1.49.6), read off the other nodes -- so the
driver's node sits in a graph with such a node; the bar position is kept
under the canvas-relative key ph_align_bar_pos_canvas.

  N1  NODE -- the real AlignNode class (LiteGraph stubbed): a virtual node
      titled "⬡ Polyhedron Align", Suite badge top right; draws "on", or
      "off" when muted/bypassed; its "reset bar position" button clears the
      stored position, a click elsewhere does nothing.
  N2  WANTED -- alignWanted: no Align node = no bar; a running one = bar; a
      muted (2) or bypassed (4) one alone = no bar.
  N3  ENABLED -- enabledOps: one node = only "group"; two = align + size +
      group, no distribute; three = distribute too; a selected group = the
      Polyhedron toggle; a node inside a selected group does not count twice.
  N4  WIRING -- the node registers in registerCustomNodes as PolyhedronAlign,
      its def gets the Suite module + hexagon title (patchFrontendDef); the
      bar is driven by alignWanted, the v1333 selection pop-up (barWanted) is
      gone; all twelve buttons in five groups, hexagon handle first.
  M   MUTATIONS -- four wounds, each must go red.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _groups_js as GJ  # noqa: E402

TAG = "[test_v1334_align_node]"
CORE = GJ.read("ph_groups_core.js")
TITLE = GJ.read("ph_group_title.js")
SWITCH = GJ.read("ph_group_switch.js")
ALIGN = GJ.read("ph_align.js")
FAILS = []

DRIVER = r"""
globalThis.document = { createElement: () => ({ getContext: () => ({ font: "", measureText: (s) => ({ width: String(s).length * 10 }) }) }), body: { appendChild() {} } };
const store = { ph_align_bar_pos_canvas: '{"dx":500,"dy":300}' };
globalThis.localStorage = { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); }, removeItem: (k) => { delete store[k]; } };
globalThis.window = { innerWidth: 1800, innerHeight: 1000 };
class LGraphGroup { constructor() { this._pos = [0, 0]; this._size = [10, 10]; this.flags = {}; }
    get _bounding() { return [this._pos[0], this._pos[1], this._size[0], this._size[1]]; } move() {} draw() {} }
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30, LGraphGroup };
const al = await import("./ph_align.js");
const out = {};
class LGraphNode { constructor(title) { this.title = title; this.properties = {}; this.size = [0, 0]; this.badges = []; this.mode = 0; } }
const Cls = al.makeAlignClass({ LGraphNode });
const n = new Cls();
class Real { } Real.nodeData = { name: "ULSFilter" };
const real = Object.assign(new Real(), { badges: [() => null] });
n.graph = { _nodes: [real, n] };     // an "array" frontend (1.49.6): the other node carries its badge
const b = n.badges[0];
const bv = typeof b === "function" ? b() : null;
out.n1 = [n.isVirtualNode === true, n.title, Cls.title, Cls.category, n.badges.length, n.badgePosition, bv && bv.text];
const texts = [];
const ctx = { save() {}, restore() {}, fillRect() {}, fillText(t) { texts.push(String(t)); },
    beginPath() {}, arc() {}, fill() {}, rect() {},
    set font(v) {}, set fillStyle(v) {}, set textAlign(v) {}, set textBaseline(v) {} };
n.onDrawForeground(ctx);
const on = texts.slice(); texts.length = 0;
n.mode = 4; n.onDrawForeground(ctx); const off = texts.slice();
out.n1draw = [on, off];
const h = n._phHit;
out.n1miss = [n.onMouseDown({}, [h.x + h.w + 20, h.y + 2]), "ph_align_bar_pos_canvas" in store];
out.n1hit = [n.onMouseDown({}, [h.x + 3, h.y + 3]), "ph_align_bar_pos_canvas" in store];
// N2
const A = (mode) => ({ type: "PolyhedronAlign", mode });
out.n2 = [al.alignWanted([]), al.alignWanted([{ type: "Note", mode: 0 }]), al.alignWanted([A(0)]),
          al.alignWanted([A(2)]), al.alignWanted([A(4)]), al.alignWanted([A(4), A(0)]), al.alignWanted(null)];
// N3
const N = (id, x, y) => ({ id, type: "T", pos: [x, y], size: [100, 50] });
const G = (x, y, w, h) => { const g = new LGraphGroup(); g._pos = [x, y]; g._size = [w, h]; return g; };
const ops = (sel) => Array.from(al.enabledOps(al.itemsOf(sel, 30))).sort();
const g1 = G(0, 0, 400, 300);
out.n3 = [ops([N(1, 10, 40)]), ops([N(1, 10, 40), N(2, 500, 40)]), ops([N(1, 10, 40), N(2, 500, 40), N(3, 900, 40)]),
          ops([g1]), ops([g1, N(4, 20, 60)]), ops([]), ops([g1, N(5, 900, 40)])];
console.log(JSON.stringify(out));
"""

ALIGN_ALL = ["bottom", "group", "hcenter", "height", "left", "right", "top", "vcenter", "width"]


def evaluate(core, title, switch, align):
    errs = []
    o, err = GJ.run({"ph_groups_core.js": core, "ph_group_title.js": title,
                     "ph_group_switch.js": switch, "ph_align.js": align}, DRIVER)
    if o is None:
        return ["node driver failed: " + str(err)]
    if o["n1"] != [True, "⬡ Polyhedron Align", "⬡ Polyhedron Align", "Polyhedron/Utility", 1,
                   "top-right", "PolyhedronLoRAStack"]:
        errs.append("N1 node %r" % o["n1"])
    on, off = o["n1draw"]
    if "Align bar: on" not in on or not any(t.startswith("Align bar: off") for t in off) \
            or "reset bar position" not in on:
        errs.append("N1 drawn %r / %r" % (on, off))
    if o["n1miss"] != [False, True] or o["n1hit"] != [True, False]:
        errs.append("N1 reset button: miss %r hit %r" % (o["n1miss"], o["n1hit"]))
    if o["n2"] != [False, False, True, False, False, True, False]:
        errs.append("N2 wanted %r" % o["n2"])
    want = [["group"], ALIGN_ALL, sorted(ALIGN_ALL + ["hdist", "vdist"]), ["group", "phtoggle"],
            ["group", "phtoggle"], [], sorted(ALIGN_ALL + ["phtoggle"])]
    if o["n3"] != want:
        errs.append("N3 enabled %r (want %r)" % (o["n3"], want))
    return errs


FAILS.extend(evaluate(CORE, TITLE, SWITCH, ALIGN))

# N4 wiring (static)
if "LG.registerNodeType(ALIGN_TYPE, makeAlignClass(LG));" not in ALIGN or "registerCustomNodes()" not in ALIGN \
        or 'export const ALIGN_TYPE = "PolyhedronAlign";' not in ALIGN:
    FAILS.append("N4 the node registers as PolyhedronAlign in registerCustomNodes")
if "patchFrontendDef(defs, ALIGN_TYPE, ALIGN_TITLE)" not in ALIGN or "beforeRegisterVueAppNodeDefs(defs)" not in ALIGN:
    FAILS.append("N4 the def gets the Suite module and the hexagon title")
if "want = alignWanted(graphNodes());" not in ALIGN or "barWanted" in ALIGN:
    FAILS.append("N4 the bar follows the node; the v1333 selection pop-up must be gone")
m = re.search(r"export const BAR_GROUPS = (\[.*?\]);", ALIGN, re.S)
groups = re.findall(r'"(\w+)"', m.group(1)) if m else []
if sorted(groups) != sorted(ALIGN_ALL + ["hdist", "vdist", "phtoggle"]) or (m and m.group(1).count("[") != 6):
    FAILS.append("N4 twelve buttons in five groups: %r" % groups)
if ALIGN.index("bar.appendChild(handle);") > ALIGN.index("bar.appendChild(b);"):
    FAILS.append("N4 the hexagon handle comes first")
if "return patchFrontendDef(defs, SWITCH_TYPE, TITLE);" not in SWITCH:
    FAILS.append("N4 the switch keeps its badge def through the shared patch")
for name, src in (("switch", SWITCH), ("align", ALIGN)):
    if re.search(r"[^\x00-\x7F]", src):
        FAILS.append("%s: the file must be plain ASCII" % name)

MUT = [
    ("bypassed node shows the bar", "&& !(n.mode === 2 || n.mode === 4) && barOn(n));",
     "&& barOn(n));"),
    ("distribute with two", 'if (top.length >= 3) { on.add("hdist"); on.add("vdist"); }',
     'if (top.length >= 2) { on.add("hdist"); on.add("vdist"); }'),
    ("reset keeps the position", "                resetBarPosition();\n                return true;",
     "                return true;"),
    ("no badge", "            addSuiteBadge(this);\n        }\n\n        computeSize()", "        }\n\n        computeSize()"),
]
caught = 0
for name, a, b in MUT:
    if ALIGN.count(a) != 1:
        FAILS.append("M mutation anchor not unique: " + name)
        continue
    if evaluate(CORE, TITLE, SWITCH, ALIGN.replace(a, b)):
        caught += 1
    else:
        FAILS.append("M mutation survived: " + name)

if FAILS:
    for f in FAILS:
        print(TAG + " FAIL: " + f)
    sys.exit(1)
print(TAG + " OK -- 4 promises, node + wanted + enabled + wiring, %d/4 mutations caught" % caught)
