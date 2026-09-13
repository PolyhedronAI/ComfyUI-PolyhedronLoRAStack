#!/usr/bin/env python3
"""v947 -- a row the classic layout gives no height is hidden under Nodes 2.0 too.

v940 derived options.hidden from the classic state and named four forms of hiding
in its own header: widget.hidden, type "hidden", a "pls-hidden-" type, and a zero
computeSize. It read three of them. The fourth was never asked for, so the Empty
Latent's frame pill -- a real widget that reports height 0 while it has nothing to
say -- kept getting a row under Nodes 2.0 (P1 class F2).

Measured over all 58 nodes on 12.09.: 65 widgets report a computeSize height <= 0,
64 already carried one of the other three flags. This rule changes exactly the
one that did not, and it is the general form, not a name in a list.

  Z1 height <= 0 counts as hidden (0 and the pack's -4 row-spacing cancel alike)
  Z2 a positive height does not, and a conditional computeSize is read at its
     CURRENT answer (the pill speaks again for an H3 model)
  Z3 a widget with a mounted ELEMENT is never judged this way: an unmounted DOM
     panel cannot grow back, so those keep their row and hide via widget.hidden
  Z4 a computeSize that throws, returns nothing, or is absent leaves the widget
     visible -- a layout rule never removes a row on an error
  Z5 the three older forms still decide on their own

Mutations prove each promise can fail.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "web", "js", "uls_vue_hidden.js")

STUBS = r"""
const app = { registerExtension() {}, get graph() { return { _nodes: [] }; } };
const vueMode = () => false;
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30 };
"""

TEST = r"""
const out = {};
const W = (o) => Object.assign({ name: "w", type: "number" }, o);

// Z1 zero and negative heights count
out.z1 = classicHidden(W({ computeSize: () => [0, 0] })) === true
      && classicHidden(W({ computeSize: () => [0, -4] })) === true;

// Z2 a positive height does not; the answer is read fresh each time
let speaks = false;
const pill = W({ computeSize: () => [0, speaks ? 20 : 0] });
const quiet = classicHidden(pill);
speaks = true;
const loud = classicHidden(pill);
out.z2 = classicHidden(W({ computeSize: () => [0, 20] })) === false
      && quiet === true && loud === false;

// Z3 a mounted DOM widget is never hidden by its height
out.z3 = classicHidden(W({ computeSize: () => [0, 0], element: { tagName: "DIV" } })) === false;

// Z4 errors leave the row alone
out.z4 = classicHidden(W({ computeSize: () => { throw new Error("boom"); } })) === false
      && classicHidden(W({ computeSize: () => null })) === false
      && classicHidden(W({})) === false;

// Z5 the older forms still stand on their own
out.z5 = classicHidden(W({ hidden: true, computeSize: () => [0, 99] })) === true
      && classicHidden(W({ type: "hidden", computeSize: () => [0, 99] })) === true
      && classicHidden(W({ type: "pls-hidden-number", computeSize: () => [0, 99] })) === true
      && classicHidden(W({ type: "number" })) === false;
console.log(JSON.stringify(out));
"""

MUTATIONS = [
    ("M1 zero height is not read",
     "    return zeroHeight(w);", "    return false;", "z1"),
    ("M2 the height is remembered instead of re-read",
     "    let h;\n    try { h = (w.computeSize(0) || [])[1]; } catch (e) { return false; }",
     "    let h;\n    if (w._z === undefined) { try { w._z = (w.computeSize(0) || [])[1]; } catch (e) { return false; } }\n    h = w._z;",
     "z2"),
    ("M3 mounted DOM widgets are judged too",
     "if (!w || w.element || typeof w.computeSize !== \"function\") return false;",
     "if (!w || typeof w.computeSize !== \"function\") return false;", "z3"),
    ("M4 a throwing computeSize hides the row",
     "    try { h = (w.computeSize(0) || [])[1]; } catch (e) { return false; }",
     "    try { h = (w.computeSize(0) || [])[1]; } catch (e) { return true; }", "z4"),
    ("M5 a positive height hides the row as well",
     "return typeof h === \"number\" && isFinite(h) && h <= 0;",
     "return typeof h === \"number\" && isFinite(h);", "z2"),
]


def build(src):
    body = re.sub(r"^import .*?;\s*$", "", src, flags=re.M)
    body = re.sub(r"^export ", "", body, flags=re.M)
    body = body.replace("app.registerExtension({", "void ({", 1)
    return STUBS + body + TEST


def run(src):
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
        fh.write(build(src))
        path = fh.name
    try:
        p = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(path)
    if p.returncode != 0:
        return None
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    src = open(SRC, encoding="utf-8").read()
    fails = []
    res = run(src)
    if res is None:
        print("[v947] FAIL -- harness did not run")
        return 1
    for k in ("z1", "z2", "z3", "z4", "z5"):
        print("  %s  %s" % ("ok  " if res.get(k) else "FAIL", k))
        if not res.get(k):
            fails.append(k)
    caught = 0
    for name, a, b, key in MUTATIONS:
        if src.count(a) != 1:
            print("  MUTATION ANCHOR BROKEN:", name)
            fails.append("anchor " + name)
            continue
        r = run(src.replace(a, b, 1))
        hit = r is None or not r.get(key)
        caught += hit
        print("  %s %s" % ("caught:" if hit else "MISSED:", name))
        if not hit:
            fails.append("mutation " + name)
    print("  mutations caught %d/%d" % (caught, len(MUTATIONS)))
    if fails:
        print("[test_v947_zero_height_hidden] FAIL --", fails)
        return 1
    print("[test_v947_zero_height_hidden] PASS -- Z1-Z5, %d/%d mutations" % (caught, len(MUTATIONS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
