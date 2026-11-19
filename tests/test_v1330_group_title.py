#!/usr/bin/env python3
"""
test_v1330_group_title -- group titles wrap inside their group
(web/js/ph_group_title.js).

RE-GROUNDED v1331 (DECLARED): Frank wants the groups exactly like the normal
ones -- v1330's grey "Suite look" and its "polyhedron" colour-menu entry are
gone (G4 now pins that they stay gone); the module imports ph_groups_core.js,
so the driver loads the real modules from a temp tree (tests/_groups_js.py)
instead of one rewritten file; the title text is told from the v1331 bottom
bar by its font; the mutation anchors follow the v1331 draw.

RE-GROUNDED v1333 (DECLARED): Frank, 15.11.: "nur unsere Gruppen ... komplett
vom Core abspalten" -- everything here applies to a POLYHEDRON group (flag
ph_group) only, and such a group always has its title drawn by us with the
Suite's hexagon in front ("\u2B21 "). So the driver's groups carry the flag;
G3 now pins: a one-line Polyhedron title is drawn by us (frontend with an
empty title, "\u2B21 short" from us), and a CORE group (no flag) goes to the
frontend untouched with no text from us and the frontend's titleHeight. The
"one-line frontend path" mutation becomes "core groups not split off".
G4: the group menu entry ("\u2B21 Polyhedron group") is now wanted; the
Suite look stays gone.

  G1  WRAP -- wrapTitle (driven): a title that fits stays one line; a long one
      breaks at spaces into lines no wider than the group, the words in order;
      a word wider than the group is broken by characters; empty stays [""].
  G2  TITLE HEIGHT -- a stub group class (draw + titleHeight getter as in the
      frontend) after install(): one line = the frontend's constant, n lines =
      constant + (n-1) line heights; resizing the group re-measures; installed
      once.
  G3  DRAW -- a wrapped title: the frontend's draw runs with an EMPTY title
      (and the title is back afterwards), our lines are written in order; a
      one-line title goes to the frontend's draw untouched and we write no
      title text.
  G4  NORMAL LOOK -- a group in #1a1a2a is drawn like any other (frontend,
      full title); no Suite colours, no menu entry in the module.
  G5  SAFE -- a class without draw is not touched (install returns false); an
      error inside the measure falls back to the frontend's own draw.
  G6  MUTATIONS -- four wounds, each must go red.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _groups_js as GJ  # noqa: E402

TAG = "[test_v1330_group_title]"
CORE = GJ.read("ph_groups_core.js")
JS_SRC = GJ.read("ph_group_title.js")
FAILS = []

DRIVER = r"""
globalThis.document = { createElement: () => ({ getContext: () => ({
    font: "", measureText: (s) => ({ width: [...String(s)].length * 10 }) }) }) };
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30, GROUP_TEXT_SIZE: 24, GROUP_FONT: "Arial" };
const { wrapTitle, barHeight, install } = await import("./ph_group_title.js");
const calls = [];
class Group {
    constructor(title, w, colour) { this.title = title; this._pos = [0, 0]; this._size = [w, 400];
        this.color = colour; this.flags = { ph_group: true }; }
    get titleHeight() { return globalThis.LiteGraph.NODE_TITLE_HEIGHT; }
    get pinned() { return !!this.flags.pinned; }
    draw(canvas, ctx) { calls.push(["frontend", this.title]); }
}
function ctxRec() { const rec = { text: [] }; let font = "";
    return { rec, save() {}, restore() {}, fillText(t) { rec.text.push([font, t]); },
             fillRect() {}, strokeRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {},
             arc() {}, fill() {}, measureText: (s) => ({ width: String(s).length * 7 }),
             set font(v) { font = v; }, get font() { return font; } }; }
const titleText = (c) => c.rec.text.filter((t) => t[0] === "24px Arial").map((t) => t[1]);
const out = {};
const m10 = (s) => [...String(s)].length * 10;
out.w1 = wrapTitle("SHORT", 200, m10);
out.w2 = wrapTitle("one two three four five six seven", 120, m10);
out.w3 = wrapTitle("abcdefghijklmnopqrst", 80, m10);
out.w4 = wrapTitle("", 100, m10);
out.installed = install(Group);
out.again = install(Group);
out.noDraw = install(class X {});
const g1 = new Group("short", 400, "#3f789e");
const g3 = new Group("one two three four five six seven eight", 140, "#3f789e");
out.th1 = g1.titleHeight; out.th3 = g3.titleHeight; out.lines3 = g3._phTitleLines;
g3._size[0] = 1000; out.thWide = g3.titleHeight; g3._size[0] = 140;
let c = ctxRec(); calls.length = 0; g3.draw({ editor_alpha: 1 }, c);
out.drawWrapped = { calls: calls.slice(), text: titleText(c), title: g3.title };
c = ctxRec(); calls.length = 0; g1.draw({ editor_alpha: 1 }, c);
out.drawPlain = { calls: calls.slice(), text: titleText(c) };
const gs = new Group("suite", 400, "#1A1A2A");
c = ctxRec(); calls.length = 0; gs.draw({ editor_alpha: 1 }, c);
out.drawSuite = { calls: calls.slice(), text: titleText(c) };
const gc = new Group("one two three four five six seven eight", 140, "#3f789e"); gc.flags = {};
c = ctxRec(); calls.length = 0; gc.draw({ editor_alpha: 1 }, c);
out.drawCore = { calls: calls.slice(), text: titleText(c), th: gc.titleHeight, hits: gc._phBarHits || null };
const gb = new Group("x y z", 50, "#3f789e");
Object.defineProperty(gb, "font_size", { get() { throw new Error("measure broke"); } });
calls.length = 0; gb.draw({ editor_alpha: 1 }, ctxRec()); out.fallback = calls.slice();
out.bar = [barHeight(1, 30, 30), barHeight(3, 30, 30)];
console.log(JSON.stringify(out));
"""


def evaluate(src):
    errs = []
    o, err = GJ.run({"ph_groups_core.js": CORE, "ph_group_title.js": src}, DRIVER)
    if o is None:
        return ["node driver failed: " + str(err)]
    # G1
    if o["w1"] != ["SHORT"]:
        errs.append("G1 a fitting title must stay one line: %r" % o["w1"])
    if any(len(s) * 10 > 120 for s in o["w2"]) or " ".join(o["w2"]) != "one two three four five six seven" \
            or len(o["w2"]) < 3:
        errs.append("G1 wrap at spaces within the width: %r" % o["w2"])
    if o["w3"] != ["abcdefgh", "ijklmnop", "qrst"]:
        errs.append("G1 a long word breaks by characters: %r" % o["w3"])
    if o["w4"] != [""]:
        errs.append("G1 empty title: %r" % o["w4"])
    # G2
    if o["installed"] is not True or o["again"] is not False:
        errs.append("G2 install once (then refuse a second patch): %r %r" % (o["installed"], o["again"]))
    n3 = len(o["lines3"] or [])
    if o["th1"] != 30 or n3 < 2 or o["th3"] != 30 + (n3 - 1) * 30:
        errs.append("G2 titleHeight one line %r, %d lines %r" % (o["th1"], n3, o["th3"]))
    if o["thWide"] != 30:
        errs.append("G2 a wider group must re-measure to one line: %r" % o["thWide"])
    if o["bar"] != [30, 90]:
        errs.append("G2 barHeight: %r" % o["bar"])
    # G3
    dw = o["drawWrapped"]
    if dw["calls"] != [["frontend", ""]] or dw["title"] != "one two three four five six seven eight" \
            or dw["text"] != o["lines3"]:
        errs.append("G3 wrapped draw: %r" % dw)
    dp = o["drawPlain"]
    if dp["calls"] != [["frontend", ""]] or dp["text"] != ["\u2B21 short"]:
        errs.append("G3 a one-line Polyhedron title is ours, with the hexagon: %r" % dp)
    if not (o["lines3"] or [""])[0].startswith("\u2B21 "):
        errs.append("G3 the wrapped title starts with the hexagon: %r" % o["lines3"])
    dc = o["drawCore"]
    if dc["calls"] != [["frontend", "one two three four five six seven eight"]] or dc["text"] \
            or dc["th"] != 30 or dc["hits"]:
        errs.append("G3 a core group must stay the frontend's, untouched: %r" % dc)
    # G4
    ds = o["drawSuite"]
    if ds["calls"] != [["frontend", ""]] or ds["text"] != ["\u2B21 suite"]:
        errs.append("G4 a #1a1a2a group must be drawn like any other: %r" % ds)
    # G5
    if o["noDraw"] is not False:
        errs.append("G5 a class without draw must not be patched")
    if o["fallback"] != [["frontend", "x y z"]]:
        errs.append("G5 an error must fall back to the frontend's draw: %r" % o["fallback"])
    return errs


FAILS.extend(evaluate(JS_SRC))
if "SUITE_BODY" in JS_SRC or "colours.polyhedron" in JS_SRC or "isSuiteGroup" in JS_SRC:
    FAILS.append("G4 the Suite look must stay gone (v1331)")
if 'name: "polyhedron.group_title"' not in JS_SRC:
    FAILS.append("G4 extension name")
if re.search(r"[^\x00-\x7F]", JS_SRC):
    FAILS.append("the file must be plain ASCII")

MUT = [
    ("titleHeight does not grow", "return n > 1 ? barHeight(n, base, lineHeightOf(this)) : base;", "return base;"),
    ("core groups not split off", "if (!isPhGroup(this)) { this._phBarHits = null;", "if (false) { this._phBarHits = null;"),
    ("wrap ignores the width", "if (!(maxW > 0) || measure(t) <= maxW) return [t];", "return [t];"),
    ("title not restored", "                this.title = title;\n                if (pinned)", "                if (pinned)"),
]
caught = 0
for name, a, b in MUT:
    if JS_SRC.count(a) != 1:
        FAILS.append("G6 mutation anchor not unique: " + name)
        continue
    if evaluate(JS_SRC.replace(a, b)):
        caught += 1
    else:
        FAILS.append("G6 mutation survived: " + name)

if FAILS:
    for f in FAILS:
        print(TAG + " FAIL: " + f)
    sys.exit(1)
print(TAG + " OK -- 6 promises, wrap + title height + frontend path + normal look, %d/4 mutations caught" % caught)
