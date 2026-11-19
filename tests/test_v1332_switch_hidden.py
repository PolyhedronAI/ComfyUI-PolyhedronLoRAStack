#!/usr/bin/env python3
"""
test_v1332_switch_hidden -- the Polyhedron Group Switch says what it hides
(web/js/ph_group_switch.js).

Frank, 15.11.: a switch preset to three stages showed three rows and "+ all
(10)" -- nothing said ten groups were hidden. v1332: the chip reads "N hidden -
show all", and the hidden groups stay listed in grey under the rows; a click on
one brings that group back alone.

RE-GROUNDED v1333 (DECLARED): the switch lists only Polyhedron groups (flag
ph_group), so the driver's groups carry the flag; the "stale ids" anchor
follows the v1333 line ("phOnly(groups)").

  H1  HIDDEN ROWS -- hiddenRowsOf lists only hidden groups that still exist,
      in reading order (the id of a deleted group is not counted, number or
      string ids alike).
  H2  LABEL + HEIGHT -- "N hidden \u00b7 show all" for N > 0, none for 0; the
      node height grows by the line and one row per hidden group; heightFor(n)
      without hidden groups is the v1331 height.
  H3  DRIVEN NODE -- the real GroupSwitch class (LiteGraph stubbed): drawn,
      the chip carries the label, every hidden group has a grey row; a click
      on one row takes exactly that id out of ph_hidden; "show all" empties it;
      x adds the id again.
  M   MUTATIONS -- three wounds, each must go red.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _groups_js as GJ  # noqa: E402

TAG = "[test_v1332_switch_hidden]"
CORE = GJ.read("ph_groups_core.js")
SWITCH = GJ.read("ph_group_switch.js")
FAILS = []

DRIVER = r"""
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30 };
const sw = await import("./ph_group_switch.js");
const { hiddenRowsOf, hiddenLabel, heightFor, makeClass } = sw;
const G = (id, x, y, w, h) => ({ id, _bounding: [x, y, w, h], title: "group " + id, color: "#335", flags: { ph_group: true } });
const groups = [G(3, 600, 0, 200, 200), G(1, 0, 0, 200, 200), G(2, 300, 0, 200, 200), G(4, 900, 0, 200, 200)];
const out = {};
out.hidden = hiddenRowsOf(groups, ["2", 4, "99"]).map((r) => r.group.id);
out.labels = [hiddenLabel(0), hiddenLabel(3)];
out.h = [heightFor(3), heightFor(3, 0), heightFor(2, 2)];
class LGraphNode { constructor(title) { this.title = title; this.properties = {}; this.size = [0, 0]; }
    setDirtyCanvas() {} }
const Cls = makeClass({ LGraphNode });
const node = new Cls();
node.graph = { _groups: groups, _nodes: [] };
node.properties.ph_hidden = ["2", "4", "99"];
node.size = [320, 10];
const texts = [];
const ctx = { save() {}, restore() {}, fillRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
    arc() {}, fill() {}, measureText: (s) => ({ width: String(s).length * 6 }),
    fillText(t) { texts.push(String(t)); }, set font(v) {}, set fillStyle(v) {}, set strokeStyle(v) {},
    set lineWidth(v) {}, set textAlign(v) {}, set textBaseline(v) {} };
node.onDrawForeground(ctx);
out.texts = texts;
out.size = node.size[1];
const unhide = node._phHits.filter((h) => h.kind === "unhide");
out.unhideIds = unhide.map((h) => h.group.id);
const u4 = unhide.find((h) => h.group.id === 4);
node.onMouseDown({}, [u4.x + 3, u4.y + 3]);
out.afterUnhide = node.properties.ph_hidden.slice();
node.onDrawForeground(ctx);
const all = node._phHits.find((h) => h.kind === "all");
node.onMouseDown({}, [all.x + 2, all.y + 2]);
out.afterAll = node.properties.ph_hidden.slice();
node.onDrawForeground(ctx);
const x1 = node._phHits.find((h) => h.kind === "hide" && h.group.id === 1);
node.onMouseDown({}, [x1.x + 2, x1.y + 2]);
out.afterHide = node.properties.ph_hidden.slice();
console.log(JSON.stringify(out));
"""


def evaluate(core, switch):
    errs = []
    o, err = GJ.run({"ph_groups_core.js": core, "ph_group_switch.js": switch}, DRIVER)
    if o is None:
        return ["node driver failed: " + str(err)]
    if o["hidden"] != [2, 4]:
        errs.append("H1 hidden rows %r (existing groups only, reading order)" % o["hidden"])
    if o["labels"] != [None, "3 hidden \u00b7 show all"]:
        errs.append("H2 labels %r" % o["labels"])
    if o["h"] != [98, 98, 24 + 44 + 8 + 36 + 8]:
        errs.append("H2 heights %r" % o["h"])
    if "2 hidden \u00b7 show all" not in o["texts"] or "+ group 2" not in o["texts"] \
            or "+ group 4" not in o["texts"]:
        errs.append("H3 drawn texts %r" % o["texts"])
    if o["size"] != 24 + 44 + 8 + 36 + 8:
        errs.append("H3 node height %r" % o["size"])
    if o["unhideIds"] != [2, 4]:
        errs.append("H3 grey rows %r" % o["unhideIds"])
    if o["afterUnhide"] != ["2", "99"]:
        errs.append("H3 a click on one grey row brings back that group alone: %r" % o["afterUnhide"])
    if o["afterAll"] != []:
        errs.append("H3 show all: %r" % o["afterAll"])
    if o["afterHide"] != ["1"]:
        errs.append("H3 x hides again: %r" % o["afterHide"])
    return errs


FAILS.extend(evaluate(CORE, SWITCH))

MUT = [
    ("stale ids counted", "return readingOrder(phOnly(groups)).filter((r) => want.has(String(r.group.id)));",
     "return (hidden || []).map((id) => ({ group: { id: Number(id), title: String(id) }, depth: 0 }));"),
    ("grey row does not restore",
     "p.ph_hidden = (p.ph_hidden || []).filter((id) => String(id) !== String(h.group.id));", "p.ph_hidden = p.ph_hidden;"),
    ("height ignores hidden rows", "(k > 0 ? HID_SEP + k * HID_H : 0)", "0"),
]
caught = 0
for name, a, b in MUT:
    if SWITCH.count(a) != 1:
        FAILS.append("M mutation anchor not unique: " + name)
        continue
    if evaluate(CORE, SWITCH.replace(a, b)):
        caught += 1
    else:
        FAILS.append("M mutation survived: " + name)

if FAILS:
    for f in FAILS:
        print(TAG + " FAIL: " + f)
    sys.exit(1)
print(TAG + " OK -- 3 promises, hidden rows + label + driven node, %d/3 mutations caught" % caught)
