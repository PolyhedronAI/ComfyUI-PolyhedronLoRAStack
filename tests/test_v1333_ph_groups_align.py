#!/usr/bin/env python3
"""
test_v1333_ph_groups_align -- our groups split off from the frontend's, the
Group Switch as a Suite node, and the Polyhedron Align bar
(web/js/ph_groups_core.js, ph_group_title.js, ph_group_switch.js, ph_align.js).

Frank, 15.11.: "das ist eine LoRAStack-Node und nicht frontend_only ... auch
die Gruppen duerfen unser Hexagon kriegen", "Nur unsere Gruppen! ... komplett
vom Core abspalten", and: two selected groups + Align must move the whole
groups; "diese Align-Leiste floatet ... vorne links unser Sechseck".

  P1  FLAG -- setPhGroup/isPhGroup round trip in group.flags (which the
      frontend serializes); other flags are kept; off removes the key.
  P2  BADGE -- patchSwitchDef gives the switch's frontend-only def the module
      of a Suite node (badge) and the hexagon title; without a Suite def the
      pack's module name; without the switch def nothing is touched. The
      switch's own canvas badge reads PolyhedronLoRAStack, steps aside when
      the frontend gives the node a badge, hides for mode "None".
  P3  MAKE OURS -- togglePhGroup: on sets the flag and fits the group with
      the bottom bar's room; off clears it and takes the room back. The group
      menu (getMenuOptions, patched once) carries the toggle; its callback
      flips the flag; the canvas menu (getCanvasMenuItems) offers it on the
      group under the mouse, nothing elsewhere.
  A1  GEOMETRY -- planAlign for all ten ops (left/centre/right, top/middle/
      bottom to the selection's box; equal width/height = the largest;
      distribute = equal gaps between first and last, needs three, order by
      position).
  A2  WHAT ALIGNS -- topLevel drops a node whose centre lies in a selected
      group and a group wholly inside one; groupMembers holds nodes by centre
      (the switch node too) and nested groups.
  A3  DRIVEN -- runAlign("left") on two selected groups and a free node: the
      groups and the node line up at the leftmost x; each group's nodes and
      nested group move by the group's own dx, nothing moves twice, a node
      outside stays; "width" resizes frames; groupAroundSelection makes a
      FLAGGED group around the selection, with the bar's room.
  RE-GROUNDED v1334 (DECLARED): the bar no longer pops up on a selection --
  it shows while a running Polyhedron Align node is in the graph (Frank
  15.11.); A4 now pins alignWanted (no node / a muted-bypassed node: no bar),
  and the localStorage write follows the v1334 line (reset clears it).
  A4  BAR -- shown while a running Polyhedron Align node is in the graph; the hexagon
      handle is the first child; a licence note says clean-room / GPL-3.0
      idea only; position kept in localStorage inside try.
  M   MUTATIONS -- six wounds, each must go red.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _groups_js as GJ  # noqa: E402

TAG = "[test_v1333_ph_groups_align]"
CORE = GJ.read("ph_groups_core.js")
TITLE = GJ.read("ph_group_title.js")
SWITCH = GJ.read("ph_group_switch.js")
ALIGN = GJ.read("ph_align.js")
FAILS = []

DRIVER = r"""
globalThis.document = { createElement: () => ({ getContext: () => ({ font: "", measureText: (s) => ({ width: String(s).length * 10 }) }) }), body: { appendChild() {} } };
class LGraphGroup {
    constructor(title) { this.title = title || "Group"; this._pos = [0, 0]; this._size = [140, 80]; this.flags = {}; this.pinned = false; }
    get _bounding() { return [this._pos[0], this._pos[1], this._size[0], this._size[1]]; }
    get titleHeight() { return 30; }
    draw() {}
    move(dx, dy, skip) { this._pos = [this._pos[0] + dx, this._pos[1] + dy]; if (skip !== true) throw new Error("children moved by the frontend"); }
    setBounds(b) { this._pos = [b[0], b[1]]; this._size = [b[2], b[3]]; }
    resizeTo(objects, padding = 10) { const arr = Array.from(objects); if (!arr.length) return;
        let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
        for (const o of arr) { const b = o._bounding ? o._bounding : [o.pos[0], o.pos[1] - 30, o.size[0], o.size[1] + 30];
            x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]); x1 = Math.max(x1, b[0] + b[2]); y1 = Math.max(y1, b[1] + b[3]); }
        this._pos = [x0 - padding, y0 - padding - this.titleHeight];
        this._size = [x1 - x0 + 2 * padding, y1 - y0 + 2 * padding + this.titleHeight]; }
    getMenuOptions() { return [{ content: "Pin" }, null, { content: "Title" }]; }
}
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30, GROUP_TEXT_SIZE: 24, GROUP_FONT: "Arial", LGraphGroup };
const { app } = await import("../../scripts/app.js");
const core = await import("./ph_groups_core.js");
const title = await import("./ph_group_title.js");
const sw = await import("./ph_group_switch.js");
const al = await import("./ph_align.js");
title.install(LGraphGroup);
title.installMenu(LGraphGroup);
const out = {};
// P1
const g = { flags: { pinned: true } };
core.setPhGroup(g, true); out.p1on = [core.isPhGroup(g), JSON.stringify(g.flags)];
core.setPhGroup(g, false); out.p1off = [core.isPhGroup(g), JSON.stringify(g.flags)];
out.p1none = [core.isPhGroup(null), core.isPhGroup({}), core.isPhGroup({ flags: {} })];
// P2
const defs = [{ name: "ULSFilter", python_module: "custom_nodes.ComfyUI-PolyhedronLoRAStack" },
              { name: "PolyhedronGroupSwitch", display_name: "PolyhedronGroupSwitch", python_module: "custom_nodes.frontend_only" },
              { name: "Other", python_module: "nodes" }];
out.p2 = [sw.patchSwitchDef(defs), defs[1].python_module, defs[1].display_name];
const defs2 = [{ name: "PolyhedronGroupSwitch", python_module: "custom_nodes.frontend_only" }];
sw.patchSwitchDef(defs2); out.p2b = defs2[0].python_module;
const defs3 = [{ name: "ULSFilter", python_module: "x" }];
out.p2c = [sw.patchSwitchDef(defs3), defs3[0].python_module];
out.p2d = sw.patchSwitchDef({ a: { name: "PolyhedronGroupSwitch", python_module: "custom_nodes.frontend_only" } });
const mine = () => null;
out.p2badge = [sw.badgeFor([mine], mine, "Hide built-in").text, sw.badgeFor([mine, () => 1], mine, null).text,
               sw.badgeFor([mine], mine, "None").text];
const bctx = { font: "", measureText: (t) => ({ width: String(t).length * 7 }) };
out.p2w = new sw.SuiteBadge("abc").getWidth(bctx);
// graph + app stub
const N = (id, x, y, w, h, extra) => Object.assign({ id, type: "T", pos: [x, y], size: [w, h], mode: 0 }, extra || {});
const mk = (x, y, w, h, ph) => { const gg = new LGraphGroup("g"); gg._pos = [x, y]; gg._size = [w, h]; if (ph) core.setPhGroup(gg, true); return gg; };
// P3
const pg = mk(0, 0, 400, 300, false);
const pn = [N(1, 50, 80, 100, 50)];
app.graph = { _nodes: pn, _groups: [pg], setDirtyCanvas() {}, add(x) { this._groups.push(x); } };
app.canvas = { graph: app.graph, selectedItems: new Set() };
pg.graph = app.graph;
const hBefore = pg._size[1];
out.p3on = [title.togglePhGroup(pg), core.isPhGroup(pg)];
const hOn = pg._size[1];
// fit around node 1: box y 50..130 -> h = 80 + 20 + 30 = 130, + bar 22
out.p3fit = hOn;
out.p3off = [title.togglePhGroup(pg), core.isPhGroup(pg), pg._size[1]];
const menu = new LGraphGroup("m").getMenuOptions();
const entry = menu[menu.length - 1];
const mg = new LGraphGroup("m"); mg.graph = app.graph;
const m1 = mg.getMenuOptions(); const e1 = m1[m1.length - 1]; e1.callback();
const m2 = mg.getMenuOptions(); const e2 = m2[m2.length - 1];
out.p3menu = [menu.length, entry && entry.content, core.isPhGroup(mg), e2.content, title.installMenu(LGraphGroup) === undefined];
const cm = title.canvasMenuItems({ graph: { getGroupOnPos: () => mg }, graph_mouse: [0, 0] });
const cmNone = title.canvasMenuItems({ graph: { getGroupOnPos: () => null }, graph_mouse: [0, 0] });
out.p3canvas = [cm.length, cm[1] && cm[1].content, cmNone.length];
// A1
const R = (x, y, w, h) => ({ x, y, w, h });
const rs = [R(100, 10, 50, 20), R(10, 200, 100, 40), R(300, 50, 30, 60)];
const xs = (op) => al.planAlign(rs, op).map((r) => [r.x, r.y, r.w, r.h]);
for (const op of al.OPS) out["a1_" + op] = xs(op);
out.a1two = al.planAlign([R(0, 0, 10, 10), R(50, 0, 10, 10)], "hdist").map((r) => r.x);
out.a1orig = rs.map((r) => r.x);
// A2
const G1 = mk(0, 0, 400, 300, true), G2 = mk(600, 100, 300, 300, false), Gin = mk(20, 40, 200, 150, false);
const nA = N(11, 40, 90, 100, 50), nB = N(12, 250, 150, 80, 40), nC = N(13, 650, 200, 100, 50),
      nFree = N(14, 1000, 500, 120, 60), nOut = N(15, 2000, 0, 50, 50),
      nSw = N(16, 260, 60, 60, 40, { type: "PolyhedronGroupSwitch", isVirtualNode: true });
const graph = { _nodes: [nA, nB, nC, nFree, nOut, nSw], _groups: [G1, G2, Gin], setDirtyCanvas() {}, add(x) { this._groups.push(x); } };
const items = al.itemsOf([G1, G2, nA, Gin, nFree], 30);
out.a2top = al.topLevel(items).map((it) => it.kind === "group" ? "G" + [G1, G2, Gin].indexOf(it.ref) : "n" + it.ref.id);
const mem = al.groupMembers(graph, G1, 30);
out.a2mem = [mem.nodes.map((n) => n.id).sort(), mem.groups.map((x) => [G1, G2, Gin].indexOf(x))];
// A3
app.graph = graph; app.canvas = { graph, selectedItems: new Set([G1, G2, nA, nFree]) };
const before = { nA: nA.pos.slice(), nB: nB.pos.slice(), nC: nC.pos.slice(), nOut: nOut.pos.slice(), nSw: nSw.pos.slice(), Gin: Gin._pos.slice() };
out.a3n = al.runAlign("left");
out.a3 = { G1: G1._pos.slice(), G2: G2._pos.slice(), nFree: nFree.pos.slice(), nA: nA.pos, nB: nB.pos, nC: nC.pos,
           nOut: nOut.pos, nSw: nSw.pos, Gin: Gin._pos, before };
out.a3w = al.runAlign("width");
out.a3width = [G1._size[0], G2._size[0], nFree.size[0]];
app.canvas.selectedItems = new Set([nOut, nC]);
const ng = al.groupAroundSelection();
out.a3group = ng ? [core.isPhGroup(ng), graph._groups.includes(ng), ng._size[1], ng._pos[1]] : null;
// A4
out.a4 = [al.alignWanted([{ type: "T" }]), al.alignWanted([{ type: "PolyhedronAlign", mode: 0 }]),
          al.alignWanted([{ type: "PolyhedronAlign", mode: 4 }, { type: "T" }]), al.alignWanted([])];
console.log(JSON.stringify(out));
"""


def evaluate(core, title, switch, align):
    errs = []
    o, err = GJ.run({"ph_groups_core.js": core, "ph_group_title.js": title,
                     "ph_group_switch.js": switch, "ph_align.js": align}, DRIVER)
    if o is None:
        return ["node driver failed: " + str(err)]
    if o["p1on"] != [True, '{"pinned":true,"ph_group":true}'] or o["p1off"] != [False, '{"pinned":true}'] \
            or o["p1none"] != [False, False, False]:
        errs.append("P1 flag %r %r %r" % (o["p1on"], o["p1off"], o["p1none"]))
    if o["p2"] != [True, "custom_nodes.ComfyUI-PolyhedronLoRAStack", "⬡ Polyhedron Group Switch"] \
            or o["p2b"] != "custom_nodes.ComfyUI-PolyhedronLoRAStack" or o["p2c"] != [False, "x"] or o["p2d"] is not True:
        errs.append("P2 badge %r %r %r %r" % (o["p2"], o["p2b"], o["p2c"], o["p2d"]))
    if o["p2badge"] != ["PolyhedronLoRAStack", "", ""] or o["p2w"] != 21 + 12:
        errs.append("P2 own badge %r width %r" % (o["p2badge"], o["p2w"]))
    if o["p3on"] != [True, True] or o["p3fit"] != 130 + 22:
        errs.append("P3 make ours: %r, fitted height %r (want 152)" % (o["p3on"], o["p3fit"]))
    if o["p3off"] != [False, False, 130]:
        errs.append("P3 back to normal: %r" % o["p3off"])
    pm = o["p3menu"]
    if pm[0] != 5 or "Polyhedron group" not in (pm[1] or "") or pm[2] is not True or "off" not in pm[3]:
        errs.append("P3 menu %r" % pm)
    pc = o["p3canvas"]
    if pc[0] != 2 or "off" not in (pc[1] or "") or pc[2] != 0:
        errs.append("P3 canvas menu %r" % pc)
    a = o
    want = {
        "left": [[10, 10, 50, 20], [10, 200, 100, 40], [10, 50, 30, 60]],
        "right": [[280, 10, 50, 20], [230, 200, 100, 40], [300, 50, 30, 60]],
        "hcenter": [[145, 10, 50, 20], [120, 200, 100, 40], [155, 50, 30, 60]],
        "top": [[100, 10, 50, 20], [10, 10, 100, 40], [300, 10, 30, 60]],
        "bottom": [[100, 220, 50, 20], [10, 200, 100, 40], [300, 180, 30, 60]],
        "vcenter": [[100, 115, 50, 20], [10, 105, 100, 40], [300, 95, 30, 60]],
        "width": [[100, 10, 100, 20], [10, 200, 100, 40], [300, 50, 100, 60]],
        "height": [[100, 10, 50, 60], [10, 200, 100, 60], [300, 50, 30, 60]],
        # x order 10(w100) 100(w50) 300(w30): span 320, sum 180, gap 70 -> 10, 180, 300
        "hdist": [[180, 10, 50, 20], [10, 200, 100, 40], [300, 50, 30, 60]],
        # y order 10(h20) 50(h60) 200(h40): span 230, sum 120, gap 55 -> 10, 85, 200
        "vdist": [[100, 10, 50, 20], [10, 200, 100, 40], [300, 85, 30, 60]],
    }
    for op, w in want.items():
        if a["a1_" + op] != w:
            errs.append("A1 %s %r (want %r)" % (op, a["a1_" + op], w))
    if a["a1two"] != [0, 50] or a["a1orig"] != [100, 10, 300]:
        errs.append("A1 distribute needs three / input untouched: %r %r" % (a["a1two"], a["a1orig"]))
    if a["a2top"] != ["G0", "G1", "n14"]:
        errs.append("A2 top level %r (node and nested group inside G1 drop out)" % a["a2top"])
    if a["a2mem"] != [[11, 12, 16], [2]]:
        errs.append("A2 members %r" % a["a2mem"])
    s3 = a["a3"]
    b = s3["before"]
    if a["a3n"] != 2 or s3["G1"] != [0, 0] or s3["G2"] != [0, 100] or s3["nFree"] != [0, 500]:
        errs.append("A3 left: n=%r G1 %r G2 %r free node %r" % (a["a3n"], s3["G1"], s3["G2"], s3["nFree"]))
    if s3["nA"] != b["nA"] or s3["nB"] != b["nB"] or s3["nSw"] != b["nSw"] or s3["Gin"] != b["Gin"]:
        errs.append("A3 G1 did not move, its members must not either: %r" % s3)
    if s3["nC"] != [b["nC"][0] - 600, b["nC"][1]] or s3["nOut"] != b["nOut"]:
        errs.append("A3 G2's node moves with it (once), the outside node stays: nC %r nOut %r" % (s3["nC"], s3["nOut"]))
    if a["a3w"] != 2 or a["a3width"] != [400, 400, 400]:
        errs.append("A3 equal width %r %r" % (a["a3w"], a["a3width"]))
    ag = a["a3group"]
    # nOut box y -30..50, nC (moved to x 50) box y 170..250 -> h 280 + 20 + 30 + bar 22
    if not ag or ag[0] is not True or ag[1] is not True or ag[2] != 280 + 20 + 30 + 22:
        errs.append("A3 group around selection %r" % ag)
    if a["a4"] != [False, True, False, False]:
        errs.append("A4 bar wanted %r" % a["a4"])
    return errs


FAILS.extend(evaluate(CORE, TITLE, SWITCH, ALIGN))

# A4 static
if "GPL-3.0" not in ALIGN or "no code, icon or" not in ALIGN or "MIT" not in ALIGN:
    FAILS.append("A4 the licence note (own MIT implementation, NodeAligner GPL-3.0 idea only)")
if 'bar.appendChild(handle);' not in ALIGN or ALIGN.index("bar.appendChild(handle);") > ALIGN.index("bar.appendChild(b);"):
    FAILS.append("A4 the hexagon handle must be the bar's first child (front left)")
if not re.search(r"try \{ if \(p\) localStorage\.setItem", ALIGN) or not re.search(r"try \{ const p = JSON\.parse\(localStorage", ALIGN):
    FAILS.append("A4 localStorage only inside try")
if 'name: "polyhedron.align"' not in ALIGN or "position:fixed" not in ALIGN:
    FAILS.append("A4 floating bar extension")
if "beforeRegisterVueAppNodeDefs(defs)" not in SWITCH:
    FAILS.append("P2 the badge hook must be wired")
for name, src in (("core", CORE), ("title", TITLE), ("switch", SWITCH), ("align", ALIGN)):
    if re.search(r"[^\x00-\x7F]", src):
        FAILS.append("%s: the file must be plain ASCII" % name)

MUT = [
    ("switch", "badge stays frontend_only", "    mine.python_module = suite", "    mine.python_module = mine.python_module; void suite"),
    ("title", "toggle forgets the flag", "    setPhGroup(group, on);\n    group._phTitleKey", "    group._phTitleKey"),
    ("align", "right aligns left", 'case "right": out.forEach((r) => { r.x = maxX - r.w; }); break;',
     'case "right": out.forEach((r) => { r.x = minX; }); break;'),
    ("align", "distribute without gaps", "at += out[i][len] + gap; }", "at += out[i][len]; }"),
    ("align", "members aligned twice", "return items.filter((it) => !inside(it));", "return items;"),
    ("align", "group moves alone", "    for (const n of m.nodes) if (!moved.has(n))", "    for (const n of []) if (!moved.has(n))"),
]
caught = 0
for where, name, a, b in MUT:
    m = {"core": CORE, "title": TITLE, "switch": SWITCH, "align": ALIGN}
    if m[where].count(a) != 1:
        FAILS.append("M mutation anchor not unique: " + name)
        continue
    m[where] = m[where].replace(a, b)
    if evaluate(m["core"], m["title"], m["switch"], m["align"]):
        caught += 1
    else:
        FAILS.append("M mutation survived: " + name)

if FAILS:
    for f in FAILS:
        print(TAG + " FAIL: " + f)
    sys.exit(1)
print(TAG + " OK -- 8 promises, flag + badge + make ours + geometry + what aligns + driven + bar, %d/6 mutations caught" % caught)
