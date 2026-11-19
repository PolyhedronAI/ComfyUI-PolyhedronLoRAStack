#!/usr/bin/env python3
"""
test_v1336_align_toggle -- the Polyhedron Align node switches its bar on and
off (web/js/ph_align.js).

Frank, 15.11.: "jetzt aber noch in der Node einen On/Off-Toggle".

  T1  DEFAULT -- a new node carries properties.ph_bar = true (saved with the
      workflow); a node loaded from an older workflow without the property
      counts as on (barOn).
  T2  WANTED -- alignWanted: a running node switched off = no bar; a second,
      switched-on node = bar; mute/bypass still hide.
  T3  CLICK -- a click on the toggle row flips ph_bar (and back), the drawn
      label follows ("Align bar: on" / "Align bar: off"); the reset button
      still resets and does not flip; a click elsewhere does nothing.
  M   MUTATIONS -- three wounds, each must go red.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _groups_js as GJ  # noqa: E402

TAG = "[test_v1336_align_toggle]"
CORE = GJ.read("ph_groups_core.js")
TITLE = GJ.read("ph_group_title.js")
SWITCH = GJ.read("ph_group_switch.js")
ALIGN = GJ.read("ph_align.js")
FAILS = []

DRIVER = r"""
globalThis.document = { createElement: () => ({ getContext: () => ({ font: "", measureText: (s) => ({ width: String(s).length * 10 }) }) }), body: { appendChild() {} } };
const store = { ph_align_bar_pos_canvas: '{"dx":5,"dy":5}' };
globalThis.localStorage = { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); }, removeItem: (k) => { delete store[k]; } };
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30 };
const al = await import("./ph_align.js");
class LGraphNode { constructor(title) { this.title = title; this.properties = {}; this.size = [0, 0]; this.badges = []; this.mode = 0; }
    setDirtyCanvas() {} }
const Cls = al.makeAlignClass({ LGraphNode });
const out = {};
const n = new Cls();
out.t1 = [n.properties.ph_bar, al.barOn({ type: "PolyhedronAlign", properties: {} }), al.barOn({ type: "PolyhedronAlign" }),
          al.barOn({ properties: { ph_bar: false } })];
const A = (mode, bar) => ({ type: "PolyhedronAlign", mode, properties: bar === undefined ? {} : { ph_bar: bar } });
out.t2 = [al.alignWanted([A(0, false)]), al.alignWanted([A(0, false), A(0, true)]), al.alignWanted([A(0, true)]),
          al.alignWanted([A(4, true)]), al.alignWanted([A(0)])];
const texts = [];
const ctx = { save() {}, restore() {}, fillRect() {}, fillText(t) { texts.push(String(t)); }, beginPath() {}, arc() {}, fill() {}, rect() {},
    set font(v) {}, set fillStyle(v) {}, set textAlign(v) {}, set textBaseline(v) {} };
const label = () => { texts.length = 0; n.onDrawForeground(ctx); return texts.find((t) => t.startsWith("Align bar")); };
const l0 = label();
const t = n._phToggle;
const c1 = n.onMouseDown({}, [t.x + 4, t.y + 4]);
const l1 = label(), v1 = n.properties.ph_bar;
const c2 = n.onMouseDown({}, [t.x + 4, t.y + 4]);
const v2 = n.properties.ph_bar;
const h = n._phHit;
const c3 = n.onMouseDown({}, [h.x + 2, h.y + 2]);
const v3 = n.properties.ph_bar, posGone = !("ph_align_bar_pos_canvas" in store);
const c4 = n.onMouseDown({}, [240, 45]);
out.t3 = [l0, c1, l1, v1, c2, v2, c3, v3, posGone, c4];
console.log(JSON.stringify(out));
"""


def evaluate(align):
    errs = []
    o, err = GJ.run({"ph_groups_core.js": CORE, "ph_group_title.js": TITLE,
                     "ph_group_switch.js": SWITCH, "ph_align.js": align}, DRIVER)
    if o is None:
        return ["node driver failed: " + str(err)]
    if o["t1"] != [True, True, True, False]:
        errs.append("T1 default on %r" % o["t1"])
    if o["t2"] != [False, True, True, False, True]:
        errs.append("T2 wanted %r" % o["t2"])
    if o["t3"] != ["Align bar: on", True, "Align bar: off", False, True, True, True, True, True, False]:
        errs.append("T3 toggle %r" % o["t3"])
    return errs


FAILS.extend(evaluate(ALIGN))
if re.search(r"[^\x00-\x7F]", ALIGN):
    FAILS.append("the file must be plain ASCII")
if "markChanged(this);" not in ALIGN:
    FAILS.append("T3 a flip marks the workflow changed (undo / unsaved dot)")

MUT = [
    ("switch ignored", "&& !(n.mode === 2 || n.mode === 4) && barOn(n));", "&& !(n.mode === 2 || n.mode === 4));"),
    ("toggle does not flip", "                this.properties.ph_bar = !barOn(this);\n", ""),
    ("old workflows off", "    return !(node && node.properties && node.properties.ph_bar === false);",
     "    return !!(node && node.properties && node.properties.ph_bar);"),
]
caught = 0
for name, a, b in MUT:
    if ALIGN.count(a) != 1:
        FAILS.append("M mutation anchor not unique: " + name)
        continue
    if evaluate(ALIGN.replace(a, b)):
        caught += 1
    else:
        FAILS.append("M mutation survived: " + name)

if FAILS:
    for f in FAILS:
        print(TAG + " FAIL: " + f)
    sys.exit(1)
print(TAG + " OK -- 3 promises, default + wanted + click, %d/3 mutations caught" % caught)
