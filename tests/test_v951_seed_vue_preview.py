#!/usr/bin/env python3
"""v951 -- Seed noise preview under Nodes 2.0 + Empty Latent's Custom label.

The frontend paints a legacy canvas widget once and again only through
widget.triggerDraw. So: the fetched field, the failure branch and setUsed call
_vueRepaint; a grown box defers a repaint (never from inside draw); the "last
used" line is painted in the widget under Nodes 2.0 and reserved in
computeSize; classic keeps onDrawForeground untouched. Empty Latent lists
"— Custom —" in the combo values so Nodes 2.0 shows no "not in list" frame.
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, "web", "js")
FAILS = []
def read(n):
    with open(os.path.join(JS, n), encoding="utf-8") as f: return f.read()
def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m)
    if not c: FAILS.append(m)
def main():
    print("v951: seed preview under Nodes 2.0 + Custom label")
    s = read("ph_seed.js"); l = read("ph_empty_latent.js")
    check('import { vueMode } from "./uls_vue_views.js";' in s, "S  seed asks the switchboard")
    check(re.search(r"function _vueRepaint\(widget\) \{\s*if \(typeof widget\.triggerDraw === \"function\"\)", s) is not None,
          "S  _vueRepaint goes through widget.triggerDraw (the frontend's door)")
    onload = s.split("img.onload = () => {", 1)[1].split("};", 1)[0]
    onerr = s.split("img.onerror = () => {", 1)[1].split("};", 1)[0]
    check("_vueRepaint(widget)" in onload, "S  the fetched field repaints the Vue canvas")
    check("_vueRepaint(widget)" in onerr, "S  the failure branch repaints too")
    used = s.split("function setUsed(node, used) {", 1)[1].split("\n}", 1)[0]
    check("_vueRepaint(pw)" in used, "S  a new 'last used' repaints")
    check(re.search(r"if \(grew && vueMode\(\)\) setTimeout\(\(\) => _vueRepaint\(widget\), 0\);", s) is not None,
          "S  a grown box defers its repaint (never from inside draw)")
    check(re.search(r"bh \+ TOP_PAD \+ PREV_BOTTOM \+ \(vueMode\(\) \? USED_LINE_H : 0\)", s) is not None,
          "S  computeSize reserves the 'last used' line only under Nodes 2.0")
    check(re.search(r"if \(vueMode\(\) && n\._plsUsed != null\) \{[^}]*\"last used: \"", s) is not None,
          "S  'last used' painted in the widget under Nodes 2.0 only")
    check("nodeType.prototype.onDrawForeground = function (ctx)" in s and "fillText(\"last used: \" + this._plsUsed, 8" in s,
          "S  classic onDrawForeground untouched")
    check(re.search(r"comboW\.options\.values = ordered\.map\(_presetLabel\)\.concat\(\[CUSTOM_LABEL\]\);", l) is not None,
          "S  Empty Latent lists the Custom label in the combo")
    check("if (v === CUSTOM_LABEL) return;" in l, "S  picking Custom stays a no-op")
    print("v951: %s" % ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
    return 0 if not FAILS else 1
if __name__ == "__main__": sys.exit(main())
