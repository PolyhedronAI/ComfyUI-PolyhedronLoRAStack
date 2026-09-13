#!/usr/bin/env python3
"""v954 -- the "control after generate" row under Nodes 2.0.

LiteGraph paints the control widget as its own row under the seed; Nodes 2.0
folds it into a dice button and lists no row. The parity tick puts the row back
for every pack node that carries such a widget: a select on the grid's own
columns (subgrid, spanning both), straight after the owner row, wired to the
same widget (value both ways), added once and never while the user holds it.
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []
def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m)
    if not c: FAILS.append(m)
def main():
    print("v954: control-after-generate row under Nodes 2.0")
    s = open(os.path.join(ROOT, "web", "js", "uls_vue_parity.js"), encoding="utf-8").read()
    check("export function applyControlRows(node, host)" in s, "S  applyControlRows exported")
    check(re.search(r"try \{ applyControlRows\(node, host\); \}", s) is not None, "S  the tick applies it for every pack node")
    fn = s.split("export function applyControlRows(node, host) {", 1)[1].split("\nexport function", 1)[0]
    check('/control_after_generate$/.test(String(w.name || ""))' in s and "!w.hidden" in s,
          "S  every unhidden control_after_generate widget gets a row (generic, not Seed-only)")
    check("if (!row) {" in fn and 'anchor.insertAdjacentElement("afterend", row);' in fn,
          "S  the row is added once, straight after the owner row")
    check("grid-column:1 / -1" in fn and "grid-template-columns:subgrid" in fn,
          "S  the row spans both grid columns on the grid's own columns")
    check("c.value = sel.value;" in fn and "if (c.callback) c.callback(c.value, app.canvas, node);" in fn,
          "S  the select writes the widget value and fires its callback")
    check("if (sel && sel.value !== String(c.value) && document.activeElement !== sel) sel.value = String(c.value);" in fn,
          "S  the widget value flows back into the select, never while it is focused")
    check("w.linkedWidgets && w.linkedWidgets.includes(ctrlW)" in s, "S  the owner row is found through linkedWidgets")
    check('row.setAttribute("data-testid", "uls-ctrl-row");' in fn,
          "S  the row is NOT a node-widget (the frontend's row list stays intact)")
    gt = s.split("export function growthTemplate(node, host) {", 1)[1].split("\n}\n", 1)[0]
    check('if (c.classList && c.classList.contains(CTRL_CLASS)) out.push("24px");' in gt and 'if (ri !== t.length) return "";' in gt,
          "S  the template gives every control row its own 24 px track, or leaves the grid alone")
    print("v954: %s" % ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
    return 0 if not FAILS else 1
if __name__ == "__main__": sys.exit(main())
