# -*- coding: ascii -*-
"""
test_v1223_cte_search_slot.py -- the CTE's search row stays inside its own slot.

Frank, 24.10.2026 (screenshot, classic canvas): the search row of the CLIP Text Encode lay over `use_negative`.
Measured in the sandbox (ComfyUI frontend 1.49, classic renderer): the frontend places a DOM widget `margin`
(BaseDOMWidgetImpl.DEFAULT_MARGIN = 10) inside its slot and gives it the slot's height minus twice that; the row is
a fixed 34 px, so it started 10 px low and ran 6 px (zoom 1.0) to 8 px (1.3) into the next widget. With margin 0
the row starts on the slot's top and ends 4 px above the next widget at 0.75 / 1.0 / 1.3, in replace AND fill mode;
under Nodes 2.0 (where the margin is not used) the row is unchanged (34 px, no overlap).

Parts:
  M  the real ph_cte_search.js (node harness): the DOM widget asks for margin 0, the slot is ROW_H (min = max =
     computeSize), so frontend placement = slot top .. slot top + ROW_H -- inside the slot
  S  the sides: 15 px padding on the row (the classic widgets' inset), dropped under Nodes 2.0 by a style rule
     scoped to [data-node-id]; the style helper never throws on a minimal document (v992's harness)
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
DEFAULT_MARGIN = 10          # the frontend's BaseDOMWidgetImpl.DEFAULT_MARGIN (1.49)

HARNESS = r"""
import { app } from "./scripts/app.js";
globalThis.requestAnimationFrame = (f) => f();
const styles = [];
function el(tag) {
    return { tagName: String(tag).toUpperCase(), style: {}, children: [], _ev: {}, dataset: {}, attrs: {},
        setAttribute(k, v) { this.attrs[k] = v; }, append(...c) { this.children.push(...c); },
        appendChild(c) { this.children.push(c); if (c.tagName === "STYLE") styles.push(c); return c; },
        addEventListener(t, f) { (this._ev[t] = this._ev[t] || []).push(f); }, focus() {}, select() {} };
}
const head = el("head");
globalThis.document = { createElement: el, addEventListener() {}, head,
    getElementById: (id) => styles.find((s) => s.id === id) || null };
const M = await import("./web/js/ph_cte_search.js");
const node = { widgets: [], addDOMWidget(name, type, e, opts) {
    const w = { name, type, element: e, options: opts || {} }; this.widgets.push(w); return w; } };
const w = M.installSearch(node, { fields: () => [], textareaFor: () => null });
const node2 = { widgets: [], addDOMWidget: node.addDOMWidget };
M.installSearch(node2, { fields: () => [], textareaFor: () => null });
console.log(JSON.stringify({
    margin: w.options.margin, slot: [w.computeSize(300)[1], w.options.getMinHeight(), w.options.getMaxHeight()],
    css: w.element.style.cssText, cls: w.element.className,
    styles: styles.map((s) => [s.id, s.textContent]) }));
"""


def run(harness):
    if NODE is None:
        return None, "node missing"
    tmp = tempfile.mkdtemp(prefix="v1223_")
    try:
        os.makedirs(os.path.join(tmp, "scripts"))
        with open(os.path.join(tmp, "scripts", "app.js"), "w", encoding="utf-8") as fh:
            fh.write("export const app = { _ext: [], canvas: null, registerExtension(e) { this._ext.push(e); } };\n")
        os.makedirs(os.path.join(tmp, "web", "js"))
        shutil.copyfile(str(WEB / "ph_cte_search.js"), os.path.join(tmp, "web", "js", "ph_cte_search.js"))
        with open(os.path.join(tmp, "h.mjs"), "w", encoding="utf-8") as fh:
            fh.write(harness)
        rr = subprocess.run([NODE, "h.mjs"], capture_output=True, text=True, cwd=tmp, timeout=60)
        if rr.returncode != 0:
            return None, rr.stderr[-1200:]
        return json.loads(rr.stdout.strip().splitlines()[-1]), ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


got, err = run(HARNESS)
_need(got is not None, "M  harness runs the real ph_cte_search.js " + err)
if got:
    margin = got.get("margin") if got.get("margin") is not None else DEFAULT_MARGIN
    row_h = int(re.search(r"height:(\d+)px", got["css"]).group(1))
    slot = got["slot"][0]
    # the frontend: top = slot top + margin, the row's own height row_h -> its bottom inside the slot?
    bottom = margin + row_h
    _need(got.get("margin") == 0, "M  the DOM widget asks for margin 0 (got %r; the frontend's default is %d)"
          % (got.get("margin"), DEFAULT_MARGIN))
    _need(got["slot"] == [row_h, row_h, row_h], "M  slot = min = max = the row's %d px (%s)" % (row_h, got["slot"]))
    _need(bottom <= slot, "M  the row ends inside its slot: %d + %d <= %d (with the default margin: %d -> %d px over)"
          % (margin, row_h, slot, DEFAULT_MARGIN + row_h, DEFAULT_MARGIN + row_h - slot))
    _need("padding:3px 15px" in got["css"].replace(" ", "").replace("padding:3px15px", "padding:3px 15px")
          or "padding:3px 15px" in got["css"], "S  15 px sides on the classic canvas (%s)" % got["css"][-80:])
    _need(got["cls"] == "pls-cte-search", "S  the row carries the class the style rule targets")
    _need(len(got["styles"]) == 1 and got["styles"][0][0] == "pls-cte-search-css"
          and "[data-node-id] .pls-cte-search" in got["styles"][0][1] and "padding-left:0" in got["styles"][0][1],
          "S  ONE style rule (two nodes), scoped to Nodes 2.0, drops the sides there (%s)" % got["styles"])
_need(re.search(r"function _sideCss\(\)\s*\{\s*try\s*\{", SRC) is not None,
      "S  the style helper is wrapped: a minimal document never breaks the row (v992's harness)")
try:
    SRC.encode("ascii")
    _need(True, "S  ph_cte_search.js is pure ASCII")
except UnicodeEncodeError:
    _need(False, "S  ph_cte_search.js is pure ASCII")

print("\ntest_v1223_cte_search_slot: %d failure(s)" % len(FAILED))
sys.exit(1 if FAILED else 0)
