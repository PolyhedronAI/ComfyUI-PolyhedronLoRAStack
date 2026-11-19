#!/usr/bin/env python3
"""
test_v1335_badge_bar_place -- one badge on the frontend-only nodes in every
frontend, and the Align bar on the canvas (web/js/ph_group_switch.js,
ph_align.js).

Frank, 15.11., on a frontend newer than the sandbox's 1.49.6: "PolyhedronLoRAStack"
stood twice over Group Switch and Align, and the bar was nowhere. Measured with
frontend 1.53.10: it draws the source badge for every node from its def store
(a node's badges array stays empty), so our own badge came on top; and the bar,
pinned to the window at 240/50, sits right under the tab row, where a taller
top menu covers it.

  B1  MODE -- frontendBadgeMode reads the other nodes of the graph (with
      nodeData, not virtual): one carrying badges = "array", all empty =
      "provider", none = null (undecided).
  B2  BADGE -- badgeFor draws the text only for "array"; the real Group
      Switch class in a "provider" graph draws an empty badge, in an "array"
      graph the text.
  B3  PLACE -- barPlace: default = top middle of the canvas, EDGE below its
      top; a stored offset is relative to the canvas and pulled back inside
      it; a bar wider than the canvas sits at its left edge.
      freeTop walks down from the canvas' top until the bar's columns are the
      canvas itself (the canvas spans the window, menus lie over it): under a
      Manager row it lands below that row; a column that is never free (a
      side bar) gives 0.
  B4  WIRING -- the position key is the canvas one, place() reads the
      canvas' box, the drag keeps the bar inside the canvas and stores an
      offset, a window resize re-places it.
  M   MUTATIONS -- five wounds, each must go red.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _groups_js as GJ  # noqa: E402

TAG = "[test_v1335_badge_bar_place]"
CORE = GJ.read("ph_groups_core.js")
TITLE = GJ.read("ph_group_title.js")
SWITCH = GJ.read("ph_group_switch.js")
ALIGN = GJ.read("ph_align.js")
FAILS = []

DRIVER = r"""
globalThis.document = { createElement: () => ({ getContext: () => ({ font: "", measureText: (s) => ({ width: String(s).length * 10 }) }) }), body: { appendChild() {} } };
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30 };
const sw = await import("./ph_group_switch.js");
const al = await import("./ph_align.js");
const out = {};
class Real { } Real.nodeData = { name: "ULSFilter" };
const R = (badges) => Object.assign(new Real(), { badges });
const V = Object.assign(new Real(), { badges: [1], isVirtualNode: true });
out.b1 = [sw.frontendBadgeMode([R([() => 1]), R([])]), sw.frontendBadgeMode([R([]), R([])]),
          sw.frontendBadgeMode([]), sw.frontendBadgeMode([{ badges: [1] }]), sw.frontendBadgeMode([V, R([])])];
const mine = () => null;
out.b2pure = [sw.badgeFor([mine], mine, "Hide built-in", "array").text, sw.badgeFor([mine], mine, "Hide built-in", "provider").text,
              sw.badgeFor([mine], mine, "Hide built-in", null).text];
class LGraphNode { constructor(title) { this.title = title; this.properties = {}; this.size = [0, 0]; this.badges = []; } }
const Cls = sw.makeClass({ LGraphNode });
const node = new Cls();
node.graph = { _nodes: [R([]), R([]), node], _groups: [] };
out.b2provider = node.badges.map((b) => (typeof b === "function" ? b() : b).text);
// B3
const C = { left: 60, top: 80, width: 1000, height: 600 };
out.b3 = [al.barPlace(C, { w: 400, h: 38 }, null), al.barPlace(C, { w: 400, h: 38 }, { dx: 50, dy: 70 }),
          al.barPlace(C, { w: 400, h: 38 }, { dx: 900, dy: -40 }), al.barPlace(C, { w: 400, h: 38 }, { dx: -5, dy: 900 }),
          al.barPlace({ left: 10, top: 20, width: 300, height: 600 }, { w: 400, h: 38 }, null)];
// menus over the canvas: x 0..100 a side bar (full height), y < 38 tabs, 300..700 x y < 75 a Manager row
const covered = (x, y) => x < 100 || y < 38 || (x >= 300 && x <= 700 && y < 75);
out.b3free = [al.freeTop({ top: 0, height: 1000 }, [350, 500, 690], (x, y) => !covered(x, y)),
              al.freeTop({ top: 0, height: 1000 }, [800, 900], (x, y) => !covered(x, y)),
              al.freeTop({ top: 0, height: 1000 }, [50], (x, y) => !covered(x, y)),
              al.freeTop({ top: 0, height: 1000 }, [500], () => true)];
console.log(JSON.stringify(out));
"""

DRIVER_ARRAY = r"""
globalThis.document = { createElement: () => ({ getContext: () => ({ font: "", measureText: (s) => ({ width: String(s).length * 10 }) }) }), body: { appendChild() {} } };
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30 };
const sw = await import("./ph_group_switch.js");
class Real { } Real.nodeData = { name: "ULSFilter" };
class LGraphNode { constructor(title) { this.title = title; this.properties = {}; this.size = [0, 0]; this.badges = []; } }
const Cls = sw.makeClass({ LGraphNode });
const node = new Cls();
node.graph = { _nodes: [Object.assign(new Real(), { badges: [() => 1] }), node], _groups: [] };
console.log(JSON.stringify({ b2array: node.badges.map((b) => (typeof b === "function" ? b() : b).text) }));
"""


def evaluate(core, title, switch, align):
    errs = []
    files = {"ph_groups_core.js": core, "ph_group_title.js": title, "ph_group_switch.js": switch, "ph_align.js": align}
    o, err = GJ.run(files, DRIVER)
    if o is None:
        return ["node driver failed: " + str(err)]
    o2, err2 = GJ.run(files, DRIVER_ARRAY)
    if o2 is None:
        return ["node driver (array) failed: " + str(err2)]
    if o["b1"] != ["array", "provider", None, None, "provider"]:
        errs.append("B1 mode %r" % o["b1"])
    if o["b2pure"] != ["PolyhedronLoRAStack", "", ""]:
        errs.append("B2 badgeFor %r" % o["b2pure"])
    if o["b2provider"] != [""]:
        errs.append("B2 a provider frontend draws the badge itself, ours stays empty: %r" % o["b2provider"])
    if o2["b2array"] != ["PolyhedronLoRAStack"]:
        errs.append("B2 an array frontend gives our node none, ours draws: %r" % o2["b2array"])
    want = [{"x": 360, "y": 90}, {"x": 110, "y": 150}, {"x": 660, "y": 80}, {"x": 60, "y": 642}, {"x": 10, "y": 30}]
    if o["b3free"] != [76, 40, 0, 0]:
        errs.append("B3 free top under the menus %r (want [76, 40, 0, 0])" % o["b3free"])
    if o["b3"] != want:
        errs.append("B3 place %r (want %r)" % (o["b3"], want))
    return errs


FAILS.extend(evaluate(CORE, TITLE, SWITCH, ALIGN))

# B4 wiring (static)
if 'const POS_KEY = "ph_align_bar_pos_canvas";' not in ALIGN:
    FAILS.append("B4 the canvas position key")
if "const p = barPlace(canvasBox(bar), { w: r.width || 450, h: r.height || 38 }, readPos());" not in ALIGN:
    FAILS.append("B4 place() reads the canvas' box")
if "writePos({ dx: rr.left - box.left, dy: rr.top - box.top });" not in ALIGN or "innerWidth - 120" in ALIGN:
    FAILS.append("B4 the drag stores an offset on the canvas, no window pin left")
if 'window.addEventListener("resize"' not in ALIGN:
    FAILS.append("B4 a window resize re-places the bar")
if "frontendModeOf(node)" not in SWITCH:
    FAILS.append("B4 the badge asks which frontend runs")
for name, src in (("switch", SWITCH), ("align", ALIGN)):
    if re.search(r"[^\x00-\x7F]", src):
        FAILS.append("%s: the file must be plain ASCII" % name)

MUT = [
    ("switch", "always array", "    return seen ? \"provider\" : null;", "    return \"array\";"),
    ("switch", "badge ignores the frontend", "    return others.length || mode === \"None\" || frontend !== \"array\"",
     "    return others.length || mode === \"None\""),
    ("align", "no clamp", "    const cx = Math.max(0, Math.min(Math.max(0, c.width - w), dx));", "    const cx = dx;"),
    ("align", "free top ignored", "        if (xs.every((x) => hit(x, box.top + dy))) return dy;",
     "        return 0;"),
    ("align", "default left", "    const dx = stored ? stored.dx : (c.width - w) / 2;", "    const dx = stored ? stored.dx : 0;"),
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
print(TAG + " OK -- 4 promises, mode + badge + place + wiring, %d/5 mutations caught" % caught)
