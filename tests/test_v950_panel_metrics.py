#!/usr/bin/env python3
"""v950 -- Nodes 2.0 parity round 3: the Stack/Engine panels keep the painted
row metrics, the Apply pill reads APPLY_INFO, disabled FIELDS are dimmed.

1. `.uls-dom` has no gap between rows; `.uls-dom-row` is 28 px = painted ROW_H.
2. The Stack panel has no Nodes-2.0 hint line (the painted node has none).
3. The Stack Apply pill takes label + colour from APPLY_INFO (imported).
4. The dim rule fires on a disabled field (input/select/textarea/switch/combobox),
   NEVER on a bare ':disabled' (a stepper button at its range limit is disabled
   too -- measured 12.09.: denoise 1.00, start_at_step 0). Pack nodes only.

Run: python tests/test_v950_panel_metrics.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, "web", "js")
FAILS = []


def read(name):
    with open(os.path.join(JS, name), encoding="utf-8") as f:
        return f.read()


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def main():
    print("v950: panel metrics, Apply pill, dimmed fields")
    dom = read("uls_stack_dom.js")
    node = read("uls_node.js")
    par = read("uls_vue_parity.js")

    row_h = re.search(r"^const ROW_H\s*=\s*(\d+);", node, re.M)
    check(row_h is not None, "S  painted ROW_H found in uls_node.js")
    painted = int(row_h.group(1)) if row_h else 0

    css = dom.split("const CSS = `", 1)[1].split("`;", 1)[0]
    m = re.search(r"\.uls-dom\s*\{[^}]*gap:\s*0;", css)
    check(m is not None, "S  .uls-dom has gap:0 (no air between rows)")
    m = re.search(r"\.uls-dom-row\s*\{[^}]*height:\s*(\d+)px;[^}]*box-sizing:border-box", css)
    check(m is not None and int(m.group(1)) == painted,
          "S  .uls-dom-row is %d px = painted ROW_H" % painted)

    check('className = "uls-dom-note"' not in dom and "Drag-to-reorder is classic-only \"" not in dom,
          "S  no hint line in the Stack panel")
    check("drag-to-reorder is classic-only" in dom, "S  the hint moved to the arrows' tooltip")

    imp = dom.split('} from "./uls_node.js";', 1)[0]
    check("APPLY_INFO" in imp, "S  uls_stack_dom.js imports APPLY_INFO")
    check(re.search(r"const apInfo = APPLY_INFO\[applyNorm\(uls\?\.apply\)\]", dom) is not None
          and re.search(r"ap\.textContent = String\(apInfo\.label\)\.toUpperCase\(\);", dom) is not None
          and re.search(r"ap\.style\.color = apInfo\.color;", dom) is not None,
          "S  Apply pill label + colour from APPLY_INFO")
    check(re.search(r"ap\.textContent = String\(applyNorm\(uls\?\.apply\)\)", dom) is None,
          "S  the raw apply value is no longer the pill label")

    m = re.search(r'\[data-testid="node-widget"\]:has\(([^)]*)\)\{opacity:\.45\}', par)
    check(m is not None, "S  dim rule present")
    sel = m.group(1) if m else ""
    parts = [p.strip() for p in sel.split(",")]
    check(all(p.endswith(":disabled") and p != ":disabled" and p[:-9] for p in parts),
          "S  every dim selector names a FIELD (no bare :disabled)")
    check("input:disabled" in parts and "select:disabled" in parts,
          "S  input and select fields dim")
    check("button:disabled" not in parts, "S  a disabled stepper button never dims the row")
    check(re.search(r"const dim = \[\.\.\.\(packIds \|\| \[\]\)\]\.map", par) is not None
          and re.search(r"writeGrowthRules\(rules, packIds\);", par) is not None,
          "S  dim rules are written per pack node id")

    print("v950: %s" % ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
