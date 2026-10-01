# -*- coding: ascii -*-
"""
test_v992_cte_search.py -- the CLIP Text Encode's search row.

Frank, 22.09.2026: a search row under `segments`; a word, a half-sentence or
a paragraph is marked live in the prompt; two triangle arrows like the LoRA
Stack step through the hits; it replaces the browser's own find.

Browser probe (sandbox ComfyUI, classic + Nodes 2.0, real keyboard):
'robot' -> 1/7, 7 boxes on the text; Enter x3 -> 4/7; Shift+Enter -> 3/7;
'thin limbs.   //   MOTION' found across a blank line; a hit below the view
pans the canvas vertically only (x offset unchanged); 'zzzq' -> 0/0 in red;
Esc clears; Ctrl+F in pos_2 with 'armored core' selected focuses the search,
prefilled, 1/1; typing ' Another robot.' in pos_1 -> 1/8 live; segments 3->2
-> 1/6; widgets_values stays 14 long in canon order; after a reload the row
sits under `segments` again; no page errors.

Parts:
  P  the pure core: literal, case-insensitive, whitespace-flexible matching;
     regex characters are literal; the cap; stepping wraps; the counter
  R  the row: a serialize:false DOM widget of fixed height; the arrows and
     the counter; Enter / Shift+Enter / Esc
  O  the CTE's row order: on screen right under `segments`, for the disk
     among the extras at the back (widgets_values untouched)
  W  the wiring: its own extension, no import between the two files (the
     guards that lift ph_clip_encode.js alone keep working), the CTE only
     publishes _plsCteApi and calls _plsSearchSchedule; repaint never sizes
"""
import json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "js"
FAILED = []


def _need(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILED.append(msg)


NODE = shutil.which("node")
SRC = (WEB / "ph_cte_search.js").read_text(encoding="utf-8")
CTE = (WEB / "ph_clip_encode.js").read_text(encoding="utf-8")

HARNESS = r"""
import { app } from "./scripts/app.js";
globalThis.requestAnimationFrame = (f) => f();
function el(tag) {
    const e = { tagName: String(tag).toUpperCase(), style: {}, children: [], _ev: {}, dataset: {},
        attrs: {}, setAttribute(k, v) { this.attrs[k] = v; },
        append(...c) { this.children.push(...c); }, appendChild(c) { this.children.push(c); return c; },
        addEventListener(t, f) { (this._ev[t] = this._ev[t] || []).push(f); },
        focus() {}, select() {} };
    return e;
}
globalThis.document = { createElement: el, addEventListener() {} };
const M = await import("./web/js/ph_cte_search.js");
const out = {};
// P
const T = "The Robot walks.\nA robot, ROBOT and robots.\n\nthin limbs.\n\n// MOTION then (a+b)*c robot";
out.m1 = M.findMatches(T, "robot").length;
out.m2 = M.findMatches(T, "thin limbs. // MOTION");
out.m3 = M.findMatches(T, "(a+b)");
out.m4 = M.findMatches(T, "   ");
out.m5 = M.findMatches("x".repeat(3000), "x").length;
out.max = M.MAX_HITS;
out.s = [M.stepIndex(-1, 1, 5), M.stepIndex(4, 1, 5), M.stepIndex(0, -1, 5), M.stepIndex(2, 1, 0)];
out.c = [M.counterText(2, 12, "x"), M.counterText(-1, 0, "x"), M.counterText(-1, 0, "  ")];
const f1 = { value: "robot one robot" }, f2 = { value: "no" }, f3 = { value: "Robot" };
out.hits = M.collectHits([f1, f2, f3], "robot").map((h) => [h.w === f1 ? 1 : h.w === f3 ? 3 : 2, h.start]);
// R -- the row on a stub node
const node = { widgets: [], addDOMWidget(name, type, e, opts) {
    const w = { name, type, element: e, options: opts || {} }; this.widgets.push(w); return w; } };
const w = M.installSearch(node, { fields: () => [f1, f3], textareaFor: () => null });
out.ser = [w.serialize, w.options.serialize];
out.h = [w.computeSize(300)[1], w.options.getMinHeight(), w.options.getMaxHeight()];
const row = w.element;
out.kids = row.children.map((c) => c.tagName + ":" + (c.children && c.children.length ? c.children.map((k) => k.textContent).join("") : (c.textContent || c.placeholder || "").slice(0, 3)));
out.idle = node._plsSearchCounter.textContent;
out.jumps = [];
const _rf = M.refresh;
out.again = M.installSearch(node, {}) === w && node.widgets.length === 1;
// typing + Enter / Shift+Enter / Esc (timers run for real)
const inp = node._plsSearchInput;
inp.value = "robot"; inp._ev.input[0]();
await new Promise((r) => setTimeout(r, 150));
out.afterType = [node._plsSearchCounter.textContent, node._plsSearch.cur];
const key = (k, shift) => { const e = { key: k, shiftKey: !!shift, preventDefault() {}, stopPropagation() {} }; inp._ev.keydown[0](e); };
key("Enter"); out.enter = node._plsSearchCounter.textContent;
key("Enter", true); key("Enter", true); out.back = node._plsSearchCounter.textContent;
key("Escape"); out.esc = [node._plsSearchCounter.textContent, inp.value, node._plsSearch.hits.length];
out.sched = typeof M.schedule === "function" && typeof M.refresh === "function";
out.ext = app._ext.map((e) => e.name);
console.log(JSON.stringify(out));
"""


def run_harness():
    if NODE is None:
        return None, "node missing"
    tmp = tempfile.mkdtemp(prefix="v992_")
    try:
        os.makedirs(os.path.join(tmp, "scripts"))
        with open(os.path.join(tmp, "scripts", "app.js"), "w", encoding="utf-8") as fh:
            fh.write("export const app = { _ext: [], canvas: null,"
                     " registerExtension(e) { this._ext.push(e); } };\n")
        os.makedirs(os.path.join(tmp, "web", "js"))
        shutil.copyfile(str(WEB / "ph_cte_search.js"), os.path.join(tmp, "web", "js", "ph_cte_search.js"))
        with open(os.path.join(tmp, "h.mjs"), "w", encoding="utf-8") as fh:
            fh.write(HARNESS)
        rr = subprocess.run([NODE, "h.mjs"], capture_output=True, text=True, cwd=tmp, timeout=60)
        if rr.returncode != 0:
            return None, rr.stderr[-1500:]
        return json.loads(rr.stdout.strip().splitlines()[-1]), ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


got, err = run_harness()
_need(got is not None, "harness runs the real ph_cte_search.js " + err)
if got:
    _need(got["m1"] == 5, "P  case-insensitive, 'robot' in 'robots' too (%s hits)" % got["m1"])
    _need(len(got["m2"]) == 1, "P  a query spans blank lines -- whitespace runs match (%s)" % got["m2"])
    _need(len(got["m3"]) == 1, "P  regex characters are literal: '(a+b)' (%s)" % got["m3"])
    _need(got["m4"] == [], "P  a blank query finds nothing")
    _need(got["m5"] == got["max"] == 999, "P  hits are capped at %s" % got["max"])
    _need(got["s"] == [0, 0, 4, -1], "P  stepping wraps both ways (%s)" % got["s"])
    _need(got["c"] == ["3/12", "0/0", ""], "P  the counter reads 3/12, 0/0, or nothing (%s)" % got["c"])
    _need(got["hits"] == [[1, 0], [1, 10], [3, 0]], "P  hits run field by field, in order (%s)" % got["hits"])
    _need(got["ser"] == [False, False], "R  the row never reaches widgets_values")
    _need(got["h"] == [34, 34, 34], "R  a fixed 34 px row in both renderers (v993) (%s)" % got["h"])
    # v993: the counter sits between the arrows and shows a dim dash while idle
    _need(got["kids"][0].startswith("INPUT") and [k.split(":")[1] for k in got["kids"][1:]]
          == ["\u25c0", "\u2013", "\u25b6", "\u2715"],
          "R  field, \u25c0, counter, \u25b6, \u2715 (%s)" % got["kids"])
    _need(got["idle"] == "\u2013", "R  the idle counter shows a dash, the middle never gapes (v993)")
    _need(got["again"] is True, "R  installing twice keeps one row")
    _need(got["afterType"] == ["1/3", 0], "R  typing marks live, current = the first hit (%s)" % got["afterType"])
    _need(got["enter"] == "2/3" and got["back"] == "3/3",
          "R  Enter steps on, Shift+Enter back and wraps (%s, %s)" % (got["enter"], got["back"]))
    _need(got["esc"] == ["\u2013", "", 0], "R  Esc clears the field and the marks (%s)" % got["esc"])
    _need("polyhedron.cte_search" in got["ext"], "W  the search is its own extension")

# O -- lift the CTE's two ordering functions and run them on a fake widget list
def _lift(sig):
    i = CTE.index(sig)
    j = CTE.index("\n}\n", i)
    return CTE[i:j + 3]

consts = "\n".join(m.group(0) for m in re.finditer(r"const (CANON|DISPLAY) = \[.*?\];", CTE, re.S))
ORDER = consts + "\n" + _lift("function _reorderWidgetsToDisplay(node)") + "\n" + _lift("function _canonOrder(node)") + r"""
const CAN = CANON.slice();
const node = { widgets: CAN.map((n) => ({ name: n })).concat([{ name: "pls_ext_pos" }, { name: "pls_ext_neg" }, { name: "pls_search" }]) };
_reorderWidgetsToDisplay(node);
const shown = node.widgets.map((w) => w.name);
_canonOrder(node);
const disk = node.widgets.map((w) => w.name);
console.log(JSON.stringify({ shown, disk, canon: CAN }));
"""
if NODE:
    tmp = tempfile.mkdtemp(prefix="v992o_")
    try:
        p = os.path.join(tmp, "o.mjs")
        open(p, "w", encoding="utf-8").write(ORDER)
        rr = subprocess.run([NODE, p], capture_output=True, text=True, timeout=60)
        o = json.loads(rr.stdout.strip().splitlines()[-1]) if rr.returncode == 0 else None
        if o is None:
            print(rr.stderr[-800:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    _need(o is not None, "O  the CTE's ordering functions run on their own")
    if o:
        si = o["shown"].index("pls_search")
        _need(o["shown"][si - 1] == "segments" and o["shown"][si + 1] == "pos_1",
              "O  on screen the row sits right under `segments` (%s)" % o["shown"][si - 1:si + 2])
        _need(o["disk"][:len(o["canon"])] == o["canon"]
              and sorted(o["disk"][len(o["canon"]):]) == ["pls_ext_neg", "pls_ext_pos", "pls_search"],
              "O  for the disk the 14 canon rows lead, the row rides among the extras behind them (%s)"
              % o["disk"][len(o["canon"]):])

# W -- the wiring by source
_need("ph_cte_search" not in CTE.split("function _reorderWidgetsToDisplay")[0].split("*/", 1)[-1]
      or "import" not in [ln for ln in CTE.splitlines() if "ph_cte_search" in ln][0],
      "W  ph_clip_encode.js does not import the search module")
_need(not any(ln.startswith("import") and "ph_cte_search" in ln for ln in CTE.splitlines()),
      "W  no import line between the two files")
_need("self._plsCteApi = { fields: () => _visibleFields(self)" in CTE,
      "W  the CTE publishes the visible fields for the search")
_need('if (fw && fw.element) fw.element.addEventListener("input", () => {' in CTE
      and "for (const nm of FIELD_NAMES) {" in CTE,
      "W  typing in any prompt field re-marks the hits")
_need("if (self._plsSearchSchedule) self._plsSearchSchedule();   // v992: marks follow" in CTE,
      "W  changing segments / use_negative re-marks the hits")
_need('addEventListener("input", () => _refit(self));' in CTE,
      "W  the CTE's own input refit stays word for word (guard v620)")
_need(".setSize(" not in SRC and "ResizeObserver" in SRC,
      "W  the marks repaint on resize but never size the node (#604)")
# v993: typing never pans the view -- only Enter / the arrows (step) do
_need('input.addEventListener("input", () => { _state(node).q = input.value; schedule(node, "mark"); });' in SRC
      and '_repaint(node, jump === "first");' in SRC and SRC.count('schedule(node, "first")') == 0,
      "W  typing marks live but never jumps (v993)")
_need("_repaint(node, true);" in SRC.split("export function step(")[1].split("export function clear(")[0],
      "W  Enter / the arrows do jump")
_need('document.addEventListener("keydown", _onKey, true)' in SRC and '"f"' in SRC,
      "W  Ctrl+F in a prompt field opens this search")
_need(all(ord(c) < 128 for c in SRC), "W  ph_cte_search.js is pure ASCII")

print()
if FAILED:
    print("test_v992_cte_search: %d failure(s)" % len(FAILED))
    sys.exit(1)
print("test_v992_cte_search: 0 failure(s)")
