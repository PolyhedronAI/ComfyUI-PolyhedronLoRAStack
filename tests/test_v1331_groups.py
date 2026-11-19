#!/usr/bin/env python3
"""
test_v1331_groups -- the group bar (colours, Fit, on/off) and the Polyhedron
Group Switch (web/js/ph_groups_core.js, ph_group_title.js, ph_group_switch.js).

RE-GROUNDED v1333 (DECLARED): only POLYHEDRON groups (flag ph_group) get the
bar, the Fit room and a switch row (Frank 15.11.: "nur unsere Gruppen").
The driver's groups carry the flag; B2 also pins that Fit of a CORE group
adds nothing, S1 that a core group gets no row; the two mutation anchors
follow the v1333 lines ("isPhGroup(this)" in Fit, "phOnly(groups)" in rows).

  C1  MEMBERS -- a node belongs to a group when the centre of its box (title
      bar included) lies inside; a Group Switch and virtual nodes never do.
  C2  STATE -- all running = on, all muted/bypassed = off, else mixed, nothing
      = empty; a click turns on -> off and off/mixed -> on; off = bypass (4)
      or mute (2).
  C3  ORDER -- reading order left to right, then top to bottom; a nested group
      has depth 1; one-at-a-time rivals never include the group around or
      inside the target.
  B1  BAR -- none + one swatch per colour of the frontend's table + Fit +
      on/off, all inside the group's bottom BAR_H; a narrow group drops
      colours, never the buttons; hitAt finds the button under a point.
  B2  FIT -- resizeTo (the frontend's Fit Group to Nodes, a stub with its
      semantics) gains BAR_H at the bottom when it had objects, and changes
      nothing for none.
  S1  SWITCH -- rows by id (a hidden id, number or string, drops the row);
      the height follows the rows; switching in free mode touches one group,
      in one-at-a-time mode switches the rivals off, mute mode uses 2.
  W1  WIRING -- the click hook listens in the CAPTURE phase on the canvas'
      parent, left button only, a node under the point wins; the switch is a
      virtual node registered in registerCustomNodes as PolyhedronGroupSwitch;
      v1330's grey Suite look and its "polyhedron" menu entry are gone.
  M   MUTATIONS -- six wounds, each must go red.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _groups_js as GJ  # noqa: E402

TAG = "[test_v1331_groups]"
CORE = GJ.read("ph_groups_core.js")
TITLE = GJ.read("ph_group_title.js")
SWITCH = GJ.read("ph_group_switch.js")
FAILS = []

DRIVER = r"""
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30, GROUP_TEXT_SIZE: 24, GROUP_FONT: "Arial" };
globalThis.document = { createElement: () => ({ getContext: () => ({ font: "", measureText: (s) => ({ width: String(s).length * 10 }) }) }) };
const core = await import("./ph_groups_core.js");
const title = await import("./ph_group_title.js");
const sw = await import("./ph_group_switch.js");
const { memberNodes, stateOf, nextOn, setNodes, readingOrder, rivalsOf, MODE_BYPASS, MODE_MUTE, SWITCH_TYPE } = core;
const { barLayout, hitAt, install, BAR_H } = title;
const { rowsOf, heightFor, switchGroup } = sw;
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30, GROUP_TEXT_SIZE: 24, GROUP_FONT: "Arial" };
globalThis.document = { createElement: () => ({ getContext: () => ({ font: "", measureText: (s) => ({ width: String(s).length * 10 }) }) }) };
const N = (id, x, y, w, h, extra) => Object.assign({ id, pos: [x, y], size: [w, h], mode: 0 }, extra || {});
const G = (id, x, y, w, h) => ({ id, _bounding: [x, y, w, h], title: "g" + id, flags: { ph_group: true } });
const out = {};
// C1
const nodes = [N(1, 10, 40, 100, 50), N(2, 300, 40, 100, 50), N(3, 10, 40, 100, 50, { type: SWITCH_TYPE }),
               N(4, 10, 40, 100, 50, { isVirtualNode: true }), N(5, 150, 120, 100, 50), N(6, 150, 220, 100, 50)];
out.members = memberNodes(nodes, [0, 0, 260, 200], 30).map((n) => n.id);
// C2
out.states = [stateOf([N(1,0,0,1,1)]), stateOf([N(1,0,0,1,1,{mode:4}), N(2,0,0,1,1,{mode:2})]),
              stateOf([N(1,0,0,1,1), N(2,0,0,1,1,{mode:4})]), stateOf([])];
out.next = [nextOn("on"), nextOn("off"), nextOn("mixed")];
const a = [N(1,0,0,1,1), N(2,0,0,1,1)]; setNodes(a, false, MODE_BYPASS); out.bypass = a.map((n) => n.mode);
setNodes(a, false, MODE_MUTE); out.mute = a.map((n) => n.mode); setNodes(a, true, MODE_MUTE); out.backOn = a.map((n) => n.mode);
// C3
const outer = G(1, 0, 0, 1000, 600), inner = G(2, 100, 100, 300, 200), right = G(3, 1200, 0, 300, 300), low = G(4, 1200, 400, 300, 100);
out.order = readingOrder([low, right, inner, outer]).map((r) => [r.group.id, r.depth]);
out.rivals = rivalsOf(inner, [outer, inner, right, low]).map((g) => g.id).sort();
// B1
const names = ["red", "brown", "green", "blue", "pale_blue", "cyan", "purple", "yellow", "black"];
const lay = barLayout([0, 0, 600, 400], names);
out.bar = lay.map((h) => h.kind + (h.name ? ":" + h.name : ""));
out.inside = lay.every((h) => h.x >= 0 && h.x + h.w <= 600 && h.y >= 400 - BAR_H && h.y + h.h <= 400);
const narrow = barLayout([0, 0, 150, 400], names);
out.narrow = narrow.map((h) => h.kind);
const tg = lay.find((h) => h.kind === "toggle");
out.hit = [hitAt(lay, tg.x + 2, tg.y + 2) && hitAt(lay, tg.x + 2, tg.y + 2).kind, hitAt(lay, 300, 100)];
// B2: a stub with the frontend's resizeTo semantics
class Grp { constructor() { this._pos = [0, 0]; this._size = [10, 10]; this.title = "t"; this.flags = { ph_group: true }; }
    get titleHeight() { return 30; } draw() {}
    resizeTo(objects, padding = 10) { const arr = Array.from(objects); if (!arr.length) return;
        this._size[0] = 500; this._size[1] = 200 + this.titleHeight; } }
install(Grp);
const gr = new Grp(); gr.resizeTo([N(1, 0, 0, 1, 1)], 10); out.fit = gr._size[1];
const ge = new Grp(); ge.resizeTo([], 10); out.fitEmpty = ge._size[1];
const gk = new Grp(); gk.flags = {}; gk.resizeTo([N(1, 0, 0, 1, 1)], 10); out.fitCore = gk._size[1];
// S1
const g1 = G(11, 0, 0, 260, 200), g2 = G(12, 300, 0, 260, 200), g3 = G(13, 600, 0, 260, 200);
const sn = [N(1, 10, 40, 100, 50), N(2, 310, 40, 100, 50), N(3, 610, 40, 100, 50)];
const graph = { _nodes: sn, _groups: [g3, g1, g2] };
out.rows = rowsOf(graph._groups, [12]).map((r) => r.group.id);
out.rowsStr = rowsOf(graph._groups, ["13"]).map((r) => r.group.id);
const gCore = { id: 14, _bounding: [900, 0, 100, 100], title: "core", flags: {} };
out.rowsCore = rowsOf([g3, g1, g2, gCore], []).map((r) => r.group.id);
out.h = [heightFor(0), heightFor(3)];
switchGroup(graph, g1, [g1, g2, g3], MODE_BYPASS, false); out.free = sn.map((n) => n.mode);
switchGroup(graph, g1, [g1, g2, g3], MODE_BYPASS, true); out.one = sn.map((n) => n.mode);
switchGroup(graph, g2, [g1, g2, g3], MODE_MUTE, true); out.oneMute = sn.map((n) => n.mode);
console.log(JSON.stringify(out));
"""


def evaluate(core, title, switch):
    errs = []
    o, err = GJ.run({"ph_groups_core.js": core, "ph_group_title.js": title, "ph_group_switch.js": switch}, DRIVER)
    if o is None:
        return ["node driver failed: " + str(err)]
    if o["members"] != [1, 5]:
        errs.append("C1 members %r (expected the two real nodes inside)" % o["members"])
    if o["states"] != ["on", "off", "mixed", "empty"]:
        errs.append("C2 states %r" % o["states"])
    if o["next"] != [False, True, True]:
        errs.append("C2 click %r" % o["next"])
    if o["bypass"] != [4, 4] or o["mute"] != [2, 2] or o["backOn"] != [0, 0]:
        errs.append("C2 modes %r %r %r" % (o["bypass"], o["mute"], o["backOn"]))
    if o["order"] != [[1, 0], [2, 1], [3, 0], [4, 0]]:
        errs.append("C3 order %r" % o["order"])
    if o["rivals"] != [3, 4]:
        errs.append("C3 rivals %r (the outer group is no rival)" % o["rivals"])
    want = ["colour"] + ["colour:" + n for n in ["red", "brown", "green", "blue", "pale_blue", "cyan",
                                                 "purple", "yellow", "black"]] + ["toggle", "fit"]
    if o["bar"] != want or not o["inside"]:
        errs.append("B1 bar %r inside=%r" % (o["bar"], o["inside"]))
    if "toggle" not in o["narrow"] or "fit" not in o["narrow"] or o["narrow"].count("colour") >= 10:
        errs.append("B1 a narrow group must keep the buttons and drop colours: %r" % o["narrow"])
    if o["hit"] != ["toggle", None]:
        errs.append("B1 hitAt %r" % o["hit"])
    if o["fitCore"] != 230:
        errs.append("B2 Fit of a core group must stay the frontend's: %r" % o["fitCore"])
    if o["rowsCore"] != [11, 12, 13]:
        errs.append("S1 a core group must get no row: %r" % o["rowsCore"])
    if o["fit"] != 230 + 22 or o["fitEmpty"] != 10:
        errs.append("B2 Fit leaves the bar free: %r, nothing for no objects: %r" % (o["fit"], o["fitEmpty"]))
    if o["rows"] != [11, 13] or o["rowsStr"] != [11, 12]:
        errs.append("S1 rows by id %r %r" % (o["rows"], o["rowsStr"]))
    if o["h"] != [24 + 22 + 8, 24 + 66 + 8]:
        errs.append("S1 height %r" % o["h"])
    if o["free"] != [4, 0, 0] or o["one"] != [0, 4, 4] or o["oneMute"] != [2, 0, 2]:
        errs.append("S1 switching free %r, one %r, one+mute %r" % (o["free"], o["one"], o["oneMute"]))
    return errs


FAILS.extend(evaluate(CORE, TITLE, SWITCH))

# W1 wiring (static, read in the source)
if not re.search(r'host\.addEventListener\("pointerdown", \(e\) => \{[\s\S]{0,2000}?\}, true\);', TITLE):
    FAILS.append("W1 the bar's click hook must listen in the capture phase on the canvas' parent")
if "if (graph.getNodeOnPos && graph.getNodeOnPos(p[0], p[1])) return;" not in TITLE:
    FAILS.append("W1 a node under the point must win over the bar")
if "e.button !== 0" not in TITLE:
    FAILS.append("W1 left button only")
if "this.isVirtualNode = true;" not in SWITCH or "registerCustomNodes()" not in SWITCH \
        or "LG.registerNodeType(SWITCH_TYPE, makeClass(LG));" not in SWITCH:
    FAILS.append("W1 the switch is a virtual node registered in registerCustomNodes")
if 'export const SWITCH_TYPE = "PolyhedronGroupSwitch";' not in CORE:
    FAILS.append("W1 node type name")
if "SUITE_BODY" in TITLE or "polyhedron:" in TITLE or "colours.polyhedron" in TITLE:
    FAILS.append("W1 v1330's Suite look / menu entry must be gone (Frank: like the normal groups)")
for name, src in (("core", CORE), ("title", TITLE), ("switch", SWITCH)):
    if re.search(r"[^\x00-\x7F]", src):
        FAILS.append("%s: the file must be plain ASCII" % name)

MUT = [
    ("core", "switch counted as member", "if (!n || n.type === SWITCH_TYPE || n.isVirtualNode) continue;",
     "if (!n || n.isVirtualNode) continue;"),
    ("core", "mixed reads as on", "if (on === nodes.length) return \"on\";", "if (on > 0) return \"on\";"),
    ("core", "nested group is a rival", "return !containsRect(b, tb) && !containsRect(tb, b);", "return true;"),
    ("title", "Fit adds the bar for nothing", "if (any && this._size && isPhGroup(this)) this._size[1] += BAR_H;",
     "if (this._size && isPhGroup(this)) this._size[1] += BAR_H;"),
    ("switch", "hidden ignored", "return readingOrder(phOnly(groups)).filter((r) => !skip.has(String(r.group.id)));",
     "return readingOrder(phOnly(groups));"),
    ("switch", "core groups listed", "return (groups || []).filter(isPhGroup);", "return groups || [];"),
    ("switch", "one at a time ignored", "if (on && oneAtATime) {", "if (false) {"),
]
caught = 0
for where, name, a, b in MUT:
    src = {"core": CORE, "title": TITLE, "switch": SWITCH}[where]
    if src.count(a) != 1:
        FAILS.append("M mutation anchor not unique: " + name)
        continue
    m = {"core": CORE, "title": TITLE, "switch": SWITCH}
    m[where] = src.replace(a, b)
    if evaluate(m["core"], m["title"], m["switch"]):
        caught += 1
    else:
        FAILS.append("M mutation survived: " + name)

if FAILS:
    for f in FAILS:
        print(TAG + " FAIL: " + f)
    sys.exit(1)
print(TAG + " OK -- 8 promises, members + state + order + bar + Fit + switch + wiring, %d/7 mutations caught" % caught)
