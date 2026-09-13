#!/usr/bin/env python3
"""v956 -- scrub, click re-roll and Roll leave control_after_generate alone;
only "Reuse last" pins it to fixed (Frank, 12.09.: a drag is the randomize
gesture, it must not flip the node to fixed)."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s = open(os.path.join(ROOT, "web", "js", "ph_seed.js"), encoding="utf-8").read()
FAILS = []
def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m)
    if not c: FAILS.append(m)
print("v956: the seed control is not pinned by scrub / click / Roll")
calls = [m.start() for m in re.finditer(r"(?<!function )\bpinFixed\((?:n|this|node)\)", s)]
check(len(calls) == 1, "S  exactly ONE pinFixed call remains (found %d)" % len(calls))
reuse = s.split('"\\u21ba Reuse last"', 1)[1].split("});", 1)[0]
check("pinFixed(this);" in reuse, "S  ... and it is the one in Reuse last")
mouse = s.split("        mouse(event, pos, n) {", 1)[1].split("\n        },\n", 1)[0]
check("pinFixed" not in mouse, "S  scrub and click re-roll do not pin")
roll = s.split('"\\ud83c\\udfb2 Roll"', 1)[1].split("});", 1)[0]
check("pinFixed" not in roll, "S  Roll does not pin")
print("v956: %s" % ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
sys.exit(0 if not FAILS else 1)
