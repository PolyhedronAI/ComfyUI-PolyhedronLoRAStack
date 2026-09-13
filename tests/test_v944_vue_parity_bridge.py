#!/usr/bin/env python3
"""v944 -- Nodes 2.0 parity bridge (web/js/uls_vue_parity.js), driven in node.

Promises:
  P1 classic mode passes straight through: the node's own clamp runs, no
     computeSize floor is added, the element is never written
  P2 Nodes 2.0: computeSize floor first, then the node's clamp, the clamped size
     goes back to the element (--node-width/--node-height) and the revealed
     floor is pinned as min-width / min-height for the frontend's own resize
  P3 a revealed cap becomes max-height ONLY if it cannot clip the Vue content
  P4 wrapping is idempotent and re-wraps a later onResize of the node
  P5 tints: the classic element's inline style reaches the Vue textarea of the
     same widget; hidden and mounted DOM widgets are skipped; if the counts do
     not match nothing is touched (never a guess)
Mutations prove each promise can fail.
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "web", "js", "uls_vue_parity.js")

STUBS = r"""
const app = { registerExtension() {}, graph: null };
let VUE = true;
const vueMode = () => VUE;
const isPackNode = () => true;
const classicHidden = (w) => !!(w && w.hidden);
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30 };
function mkStyle() { const v = {}; return { setProperty(k, x) { v[k] = x; }, getPropertyValue(k) { return v[k] || ""; }, _v: v }; }
let CONTENT = 700;
function mkEl() { const style = mkStyle(); return { style, getBoundingClientRect() {
    const nh = parseFloat(style.getPropertyValue("--node-height") || "0"); return { height: Math.max(nh, CONTENT) }; } }; }
let EL = mkEl();
globalThis.document = { querySelector: () => EL };
"""

TEST = r"""
const out = {};
const clampNode = (cap) => ({ id: 1, computeSize: () => [300, 100],
    onResize(size) { if (size[0] < 460) size[0] = 460; if (cap && size[1] > cap) size[1] = cap; } });
// P1 classic
VUE = false; EL = mkEl(); let n = clampNode(0); wrapResize(n); let s = [200, 50]; n.onResize(s);
out.p1 = s[0] === 460 && s[1] === 50 && !EL.style.getPropertyValue("--node-width") && !EL.style.minWidth;
// P2 vue floor
VUE = true; EL = mkEl(); n = clampNode(0); wrapResize(n); s = [200, 50]; n.onResize(s);
out.p2 = s[0] === 460 && s[1] === 100 && EL.style.getPropertyValue("--node-width") === "460px"
    && EL.style.getPropertyValue("--node-height") === "130px" && EL.style.minWidth === "460px"
    && EL.style.minHeight === "max(var(--node-height), 130px)";
// P3 cap, content fits / content taller
EL = mkEl(); CONTENT = 700; n = clampNode(900); wrapResize(n); s = [500, 1200]; n.onResize(s);
const capSet = EL.style.maxHeight === "930px";
EL = mkEl(); CONTENT = 1000; n = clampNode(900); wrapResize(n); s = [500, 1200]; n.onResize(s);
out.p3 = capSet && s[1] === 900 && !EL.style.maxHeight;
// P4 idempotent + re-wrap
n = clampNode(0); const a = wrapResize(n), b = wrapResize(n); n.onResize = function (size) {}; const c = wrapResize(n);
out.p4 = a === true && b === false && c === true && n.onResize._ulsParity === true;
// P5 tints
const ta = () => ({ style: {}, closest: () => null });
const mkW = (name, bg, hidden) => ({ name, hidden, element: { tagName: "TEXTAREA", style: { backgroundColor: bg, borderLeft: bg ? "4px solid x" : "", borderRadius: bg ? "4px" : "" } } });
const t1 = ta(), t2 = ta(), dom = { tagName: "TEXTAREA", style: { backgroundColor: "red" } };
const host = { querySelectorAll: () => [t1, t2], contains: (e) => e === dom };
n = { widgets: [mkW("pos_1", "rgb(47, 79, 47)"), mkW("pos_2", "", true), { name: "ext", element: dom }, mkW("neg_1", "rgb(74, 47, 47)")] };
const cnt = applyTints(n, host);
const okMap = cnt === 2 && t1.style.backgroundColor === "rgb(47, 79, 47)" && t2.style.backgroundColor === "rgb(74, 47, 47)"
    && t2.style.borderLeft === "4px solid x";
const t3 = ta(), t4 = ta(), t5 = ta();
const host2 = { querySelectorAll: () => [t3, t4, t5], contains: (e) => e === dom };   // same host, one textarea more
applyTints(n, host2);
out.p5 = okMap && !t3.style.backgroundColor && !t4.style.backgroundColor;
console.log(JSON.stringify(out));
"""

MUTATIONS = [
    ("M1 acts in classic mode", "        if (!vueMode() || !size || size.length < 2) {", "        if (!size || size.length < 2) {", "p1"),
    ("M2 no computeSize floor", "        computeFloor(this, size);\n", "", "p2"),
    ("M3 clamp never reaches the element", "            pushSize(this, size);\n", "", "p2"),
    ("M4 cap clips the Vue content", " && b.cap[1] + T >= contentMin(el) - EPS", "", "p3"),
    ("M5 guesses on a count mismatch", "    if (vts.length !== ws.length) return [];\n", "", "p5"),
]


def run(src):
    lines = [l for l in src.splitlines() if not l.startswith("import ")]
    body = STUBS + "\n".join(lines) + "\n" + TEST
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as f:
        f.write(body)
        p = f.name
    try:
        r = subprocess.run(["node", p], capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(p)
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout)[-300:]}
    import json
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    src = open(SRC, encoding="utf-8").read()
    fails = []
    res = run(src)
    if "error" in res:
        print("[v944] FAIL -- harness: " + res["error"])
        sys.exit(1)
    for k in ("p1", "p2", "p3", "p4", "p5"):
        if not res.get(k):
            fails.append("promise %s broken" % k)
    caught = 0
    for name, old, new, key in MUTATIONS:
        if src.count(old) != 1:
            fails.append("mutation anchor not unique/absent: %s" % name)
            continue
        r = run(src.replace(old, new))
        if "error" in r or not r.get(key):
            caught += 1
            print("  caught: " + name)
        else:
            fails.append("mutation survived: " + name)
    if fails:
        for f in fails:
            print("[v944] FAIL -- " + f)
        sys.exit(1)
    print("PASS: v944 -- 5 promises pinned, %d/%d mutations caught" % (caught, len(MUTATIONS)))


if __name__ == "__main__":
    main()
