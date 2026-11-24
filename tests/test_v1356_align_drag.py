#!/usr/bin/env python3
"""
test_v1356_align_drag -- the floating Align bar stays where it is put, and a
drag onto the tab row parks it (web/js/ph_align.js).

Frank, 21.11. (v1355 in 1.58.5): "jetzt ist die Leiste kaputt und springt auf
der Canvas herum ... Ich kann sie ... nicht an die Stelle der alten Align
schieben". Measured in the sandbox: v1355's tick placed a floating bar every
200 ms, so a drag was pulled back to the stored place (y 50, 140, 50, 215,
230, 50 ...) and the release stored the pulled-back place.

  D1  PLACE -- needsPlace: only a floating bar that really moved (unparked) is
      placed; floating and unmoved, parked, or dragging: not.
  D2  DROP -- dropParks: the pointer inside the tab row's rect parks; outside
      (below, left, right), no rect or a zero-width rect: not.
  D3  WIRING -- remount places only through needsPlace and leaves a dragged bar
      alone; the drag marks the bar (dataset.dragging) and clears it on
      release; released over the row it parks (writeParked(true)) and does
      not store a canvas place; over the row the bar shows a frame.
  M   MUTATIONS -- five wounds, each must go red.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _groups_js as GJ  # noqa: E402

TAG = "[test_v1356_align_drag]"
CORE = GJ.read("ph_groups_core.js")
TITLE = GJ.read("ph_group_title.js")
SWITCH = GJ.read("ph_group_switch.js")
ALIGN = GJ.read("ph_align.js")
FAILS = []

DRIVER = r"""
globalThis.document = { createElement: () => ({ getContext: () => ({ font: "", measureText: (s) => ({ width: String(s).length * 10 }) }) }), body: { appendChild() {} } };
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30 };
const al = await import("./ph_align.js");
const N = al.needsPlace;
const row = { left: 0, right: 1900, top: 0, bottom: 38, width: 1900, height: 38 };
const D = al.dropParks;
console.log(JSON.stringify({
  d1: [N({ mode: "floating", moved: true }), N({ mode: "floating", moved: false }), N({ mode: "parked", moved: true }),
       N({ mode: "dragging", moved: false }), N(null)],
  d2: [D(row, 1500, 20), D(row, 1500, 0), D(row, 1500, 38), D(row, 1500, 39), D(row, -1, 20), D(row, 1901, 20),
       D(null, 10, 10), D({ left: 0, right: 0, top: 0, bottom: 38, width: 0, height: 38 }, 0, 10)],
}));
"""


def evaluate(align):
    errs = []
    o, err = GJ.run({"ph_groups_core.js": CORE, "ph_group_title.js": TITLE,
                     "ph_group_switch.js": SWITCH, "ph_align.js": align}, DRIVER)
    if o is None:
        return ["node driver failed: " + str(err)]
    if o["d1"] != [True, False, False, False, False]:
        errs.append("D1 place %r" % o["d1"])
    if o["d2"] != [True, True, True, False, False, False, False, False]:
        errs.append("D2 drop %r" % o["d2"])
    r = re.search(r"function remount\(bar\) \{(.*?)\n\}", align, re.S)
    body = r.group(1) if r else ""
    if 'if (bar.dataset.dragging === "1") return' not in body:
        errs.append("D3 remount leaves a dragged bar alone")
    if "needsPlace(m)" not in body or re.search(r"m\.mode === \"floating\" && bar", body):
        errs.append("D3 remount places only through needsPlace")
    h = re.search(r'handle\.addEventListener\("pointerdown", \(e\) => \{(.*?)\n    \}\);', align, re.S)
    hb = h.group(1) if h else ""
    if 'bar.dataset.dragging = "1";' not in hb or "delete bar.dataset.dragging;" not in hb:
        errs.append("D3 the drag marks the bar and clears the mark")
    if not re.search(r"if \(overRow\(ev\)\) \{ writeParked\(true\); remount\(bar\); return; \}\s*const rr", hb):
        errs.append("D3 released over the row it parks, before any canvas place is stored")
    if 'bar.style.outline = overRow(ev) ? "2px solid' not in hb:
        errs.append("D3 a frame shows the drop will park")
    return errs


FAILS.extend(evaluate(ALIGN))
if re.search(r"[^\x00-\x7F]", ALIGN):
    FAILS.append("the file must be plain ASCII")

MUT = [
    ("places on every tick", 'return !!(m && m.mode === "floating" && m.moved);', 'return !!(m && m.mode === "floating");'),
    ("drag not respected", '    if (bar.dataset.dragging === "1") return { mode: "dragging", moved: false };\n', ""),
    ("drop does not park", "if (overRow(ev)) { writeParked(true); remount(bar); return; }", ""),
    ("drop ignores the height", "y >= rect.top && y <= rect.bottom", "y >= rect.top"),
    ("mark never cleared", "            delete bar.dataset.dragging;\n", ""),
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
print(TAG + " OK -- 3 promises, place + drop + wiring, %d/%d mutations caught" % (caught, len(MUT)))
