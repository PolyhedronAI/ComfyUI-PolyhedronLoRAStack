"""
test_v310_engine_header.py — source-wiring checks for the Engine
"Weight / CLIP Strength" header (mirror of the Stack v305/v308 composite).

Sandbox-safe (no torch/comfy, no canvas): verifies at the source level that
the Engine block contains the full composite (measured, right-anchored),
the hover zone, and the drawn-last tooltip — and that the Stack header is
untouched.

Run directly:  python3 tests/test_v310_engine_header.py
"""

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JS_PATH = os.path.join(ROOT, "web", "js", "uls_node.js")

passed = 0
failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}")


print("════════════════════════════════════════════════════════")
print(" test_v310_engine_header — Engine Weight/CLIP header")
print("════════════════════════════════════════════════════════")

js = open(JS_PATH, encoding="utf-8").read()

# The Engine section starts at its banner comment; everything before it is
# the Stack (plus shared helpers).
eng_start = js.find("⬡ Polyhedron Engine")
check("Engine section banner found", eng_start > 0)
stack_js = js[:eng_start]
eng_js = js[eng_start:]

# ── 1. Composite present in the ENGINE block ───────────────────────────
print("\n[1] Engine header composite")
check("Engine draws 'Weight' caption", 'fillText("Weight"' in eng_js)
check("Engine draws blue ' / CLIP Strength'",
      'fillText(" / CLIP Strength"' in eng_js)
check("amber Weight color (#b07820)", '"#b07820"' in eng_js)
check("blue CLIP color (#6aa0d0)", '"#6aa0d0"' in eng_js)
check("info icon: stroked circle r=3.2", "_ICO_R = 3.2" in eng_js)
check("info icon: drawn 'i' (not platform emoji)",
      'fillText("i", _ix' in eng_js)

# ── 2. Measured + right-anchored (v304/v307/v308 lessons) ──────────────
print("\n[2] Geometry discipline")
check("widths via measureText (Weight)",
      'measureText("Weight")' in eng_js)
check("widths via measureText (CLIP)",
      'measureText(" / CLIP Strength")' in eng_js)
check("right anchor = node content edge (W - PAD)",
      re.search(r"_right = W - PAD", eng_js) is not None)
check("composite x derived from right - total",
      "_right - _total" in eng_js)
check("baseline shared with mode label",
      "modeY + MODE_BTN_H + 14" in eng_js)

# ── 3. Hover zone + handler order ──────────────────────────────────────
print("\n[3] Hover wiring")
check("Engine stores _weightHdrRect", "_weightHdrRect" in eng_js)
check("Engine hover zone 'weightHdr' set",
      '"weightHdr"' in eng_js)
# The header check must run before the DARE pill check in onMouseMove.
m = re.search(r"onMouseMove[\s\S]{0,1500}?_weightHdrRect[\s\S]{0,1500}?_dareVariantRect",
              eng_js)
check("header hover checked FIRST in onMouseMove (before DARE pill)",
      m is not None)

# ── 4. Tooltip overlay (drawn last) ────────────────────────────────────
print("\n[4] Tooltip")
check("tooltip overlay guarded by weightHdr zone",
      'uls.hoverZone === "weightHdr" && uls._weightHdrRect' in eng_js)
# v374: the tooltip lines are shared (WEIGHT_HDR_TIP_LINES, defined above the
# Engine section since v915) -- the Engine reads them, the text lives once.
check("tooltip explains Shift+Click",
      "Shift+Click" in js and "WEIGHT_HDR_TIP_LINES" in eng_js)
check("tooltip anchored LEFT of composite",
      "hr.x - TW - 8" in eng_js)

# ── 5. Stack header untouched ──────────────────────────────────────────
print("\n[5] Stack header intact (v308)")
check("Stack still draws the composite",
      'fillText("Weight"' in stack_js and 'fillText(" / CLIP Strength"' in stack_js)
check("Stack hover zone intact", '"weightHdr"' in stack_js)
check("Stack right-anchor comment (v308) intact",
      "RIGHT-anchored" in stack_js)

# ── Result ─────────────────────────────────────────────────────────────
print("\n════════════════════════════════════════════════════════")
total = passed + failed
if failed == 0:
    print(f"RESULT: ALL CHECKS PASS ({passed}/{total})")
    sys.exit(0)
else:
    print(f"RESULT: {failed} FAILED ({passed}/{total} passed)")
    sys.exit(1)
