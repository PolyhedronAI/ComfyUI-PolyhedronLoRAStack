#!/usr/bin/env python3
"""v952 -- CLIP Text Encode auto-fit under Nodes 2.0 shrinks as well as grows.

Frank, 12.09.: "the field does not shrink when I take text out". Causes, measured:
(a) the v949 bridge took max(classic height, Vue content) -- the classic height is
stale under Nodes 2.0, so the field could only grow; (b) typing in the Vue field
never reached the classic element's 'input' listener that drives _refit; (c) a
measurement under the bridge's own min-height could never come out smaller.
Now: _refit measures the Vue field standing in (vueFieldFor) when the classic
element is not laid out, _contentH lifts min-height while measuring, the bridge
COPIES the classic inline height and relays value changes as 'input'.
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
    print("v952: CTE auto-fit shrinks under Nodes 2.0")
    c = read("ph_clip_encode.js"); p = read("uls_vue_parity.js")
    check('import { vueFieldFor } from "./uls_vue_parity.js";' in c, "S  CTE imports vueFieldFor")
    check("export function vueFieldFor(node, w)" in p and "for (const [cw, t] of tintPairs(node, host)) if (cw === w) return t;" in p,
          "S  vueFieldFor resolves through the same pairing as the tint bridge (one truth)")
    refit = c.split("function _refit(node) {", 1)[1].split("\n}\n", 1)[0]
    check(re.search(r"if \(!laidOut\) \{[^}]*const vf = vueFieldFor\(node, w\);[^}]*if \(vf && vf\.clientWidth > 0\) \{ measureEl = vf; laidOut = true; \}", refit) is not None,
          "S  _refit measures the Vue field when the classic one is not laid out")
    check("h = _contentH(w, measureEl);" in refit, "S  the measurement uses the stand-in element")
    ch = c.split("function _contentH(w, el) {", 1)[1].split("\n}\n", 1)[0]
    check('el.style.minHeight = "0";' in ch and "el.style.minHeight = prevMin;" in ch,
          "S  _contentH lifts min-height while measuring and restores it")
    check(re.search(r"if \(t\.style\[HEIGHT_KEY\] !== h\) \{ t\.style\[HEIGHT_KEY\] = h; t\.style\.minHeight = h; \}", p) is not None,
          "S  the bridge copies the classic inline height (no max)")
    check("Math.max(c, need)" not in p.split("export function applyTints", 1)[1],
          "S  applyTints no longer maxes against the stale classic height")
    check(re.search(r"if \(t\._ulsLastVal !== t\.value\) \{\s*t\._ulsLastVal = t\.value;\s*try \{ w\.element\.dispatchEvent\(new Event\(\"input\"\)\); \}", p) is not None,
          "S  a changed Vue value is relayed as 'input' to the classic element (drives _refit)")
    print("v952: %s" % ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
    return 0 if not FAILS else 1
if __name__ == "__main__": sys.exit(main())
