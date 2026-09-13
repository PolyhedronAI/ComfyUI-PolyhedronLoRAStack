#!/usr/bin/env python3
"""v953 -- Seed scrub is visible under Nodes 2.0.

Measured 12.09.: the drag DID change the seed (mouse() receives pointerdown /
pointermove / pointerup from the frontend's legacy-widget canvas) but the box
never repainted -- Frank saw nothing move. Now every scrub step and the
pointerup schedule one repaint per animation frame through widget.triggerDraw.
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []
def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m)
    if not c: FAILS.append(m)
def main():
    print("v953: seed scrub repaints under Nodes 2.0")
    s = open(os.path.join(ROOT, "web", "js", "ph_seed.js"), encoding="utf-8").read()
    fn = re.search(r"function _vueRepaintSoon\(widget\) \{[\s\S]*?\n\}", s)
    check(fn is not None, "S  _vueRepaintSoon exists")
    body = fn.group(0) if fn else ""
    check("widget._ulsRafPending" in body and "requestAnimationFrame" in body and "_vueRepaint(widget)" in body,
          "S  one repaint per animation frame, through _vueRepaint (triggerDraw)")
    mouse = s.split("        mouse(event, pos, n) {", 1)[1].split("\n        },\n", 1)[0]
    mv = mouse.split('if (et === "pointermove"', 1)[1].split('if (et === "pointerup"', 1)[0]
    up = mouse.split('if (et === "pointerup"', 1)[1]
    check("_vueRepaintSoon(widget)" in mv and "sdW.value = v;" in mv.split("_vueRepaintSoon(widget)")[0],
          "S  a scrub step repaints after the seed changed")
    check("_vueRepaintSoon(widget)" in up, "S  pointerup repaints (drag end and click re-roll)")
    print("v953: %s" % ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
    return 0 if not FAILS else 1
if __name__ == "__main__": sys.exit(main())
