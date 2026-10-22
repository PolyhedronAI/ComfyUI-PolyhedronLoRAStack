# -*- coding: ascii -*-
"""Guard v1170 -- the NEW mark of a replaced clip sits bottom right.

(Frank 18.10., screenshot of the neon timeline: "bitte das NEW unten rechts hinsetzen, damit es nicht von der CUT
verdeckt wird". The cut chip sits on the seam at the clip's top, the key diamond on the seam lower down.)

  N1 the NEW mark is anchored bottom right, inset from the seam (right >= 10 px), never at the top
  N2 the v1165 parts stay: data-pls="newmark", the green, the tooltip line
  N3 ASCII
Script-style: exit 0 = pass.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1170_new_corner"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


JS = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
m = re.search(r'const nw = _el\("div", "([^"]*)"', JS)
css = m.group(1) if m else ""
r = re.search(r"right:(\d+)px", css)
_need(m is not None and "bottom:2px" in css and "top:" not in css and r is not None and int(r.group(1)) >= 10,
      "N1 NEW sits bottom right, clear of the seam (%s)" % css)
_need('nw.dataset.pls = "newmark"' in JS and "background:#4caf50" in JS and "NEW: came in with the latest render" in JS,
      "N2 the mark, its green and its tooltip line stay")
for p in ("web/js/ph_cine_timeline.js", "tests/" + NAME + ".py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "N3 ASCII %s" % p)
print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
