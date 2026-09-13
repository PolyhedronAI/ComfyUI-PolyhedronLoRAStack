#!/usr/bin/env python3
"""v945 -- Nodes 2.0 growth rule (M3 in web/js/uls_vue_parity.js), driven in node.

LiteGraph grows a widget only when it has no computeSize; Nodes 2.0 gives every
DOM widget and every textarea the growing track 'auto'. The rule maps the
frontend's rows to the node's widgets and turns 'auto' into 'min-content' for
every widget with a computeSize, written as one stylesheet rule per node.

Promises:
  G1 the rule: 'auto' + computeSize -> 'min-content'; 'auto' without computeSize
     stays 'auto'; 'min-content' rows are never touched
  G2 never a guess: a count mismatch or a mounted DOM element in a foreign row
     leaves the node alone (null)
  G3 the row list mirrors the frontend: hidden, advanced, canvasOnly and untyped
     widgets get no row
  G4 growthTemplate reads the grid's own inline template and returns "" when the
     rule changes nothing (then no stylesheet rule exists for the node)
  G5 all rules live in ONE style element, keyed by data-node-id, !important;
     classic mode empties it
  G6 the grid's inline style is never written (Vue re-applies its binding)
Mutations prove each promise can fail.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "web", "js", "uls_vue_parity.js")

STUBS = r"""
let GRAPH_NODES = [];
const app = { registerExtension() {}, get graph() { return { _nodes: GRAPH_NODES }; } };
let VUE = true;
const vueMode = () => VUE;
const isPackNode = () => true;
const classicHidden = (w) => false;
globalThis.LiteGraph = { NODE_TITLE_HEIGHT: 30 };
const STYLES = {};
globalThis.document = {
    querySelector: () => null,
    getElementById: (id) => STYLES[id] || null,
    createElement: () => ({ id: "", textContent: "" }),
    head: { appendChild(el) { STYLES[el.id] = el; } },
};
"""

TEST = r"""
const out = {};
const el = (n) => ({ n });
const row = (mounted) => ({ getAttribute: (k) => (k === "data-testid" ? "node-widget" : null),
                            contains: (e) => !!mounted && e === mounted });
const mk = () => {
    const eFix = el("fix"), eGrow = el("grow");
    const widgets = [
        { name: "seed", type: "number" },
        { name: "fixed_panel", type: "fixed", element: eFix, computeSize: () => [0, 120] },
        { name: "grow_panel", type: "grow", element: eGrow },
        { name: "text", type: "customtext", element: el("ta"), computeSize: () => [0, 60] },
    ];
    const rows = [row(null), row(eFix), row(eGrow), row(null)];
    const host = { contains: (e) => e === eFix || e === eGrow };
    return { widgets, rows, host, eFix, eGrow };
};
// G1 the rule
let m = mk();
const t1 = growthTracks(["min-content", "auto", "auto", "auto"], m.widgets, m.rows, m.host);
out.g1 = !!t1 && t1.join(" ") === "min-content min-content auto min-content";
// G2 never a guess
m = mk();
const cnt = growthTracks(["min-content", "auto", "auto"], m.widgets, m.rows, m.host);
m = mk(); const swapped = [m.rows[0], m.rows[2], m.rows[1], m.rows[3]];
const foreign = growthTracks(["min-content", "auto", "auto", "auto"], m.widgets, swapped, m.host);
out.g2 = cnt === null && foreign === null;
// G3 row list
const vis = vueRowWidgets({ widgets: [
    { name: "a", type: "number" }, { name: "b", type: "number", options: { hidden: true } },
    { name: "c", type: "number", options: { advanced: true } }, { name: "d", type: "x", options: { canvasOnly: true } },
    { name: "e" }, { name: "f", type: "combo", options: {} } ] }).map(w => w.name).join(",");
out.g3 = vis === "a,f";
// G4 template from the grid's inline value
const mkHost = (tpl, m) => {
    const grid = { style: { gridTemplateRows: tpl }, children: m.rows };
    return { contains: m.host.contains, querySelector: (q) => (q === '[data-testid="node-widgets"]' ? grid : null) };
};
m = mk(); const node = { id: 7, widgets: m.widgets };
const tplA = growthTemplate(node, mkHost("min-content auto auto auto", m));
m = mk(); const node2 = { id: 8, widgets: m.widgets };
const tplB = growthTemplate(node2, mkHost("min-content min-content auto min-content", m));
out.g4 = tplA === "min-content min-content auto min-content" && tplB === "";
// G5 one style element, keyed, !important; classic empties it
writeGrowthRules(new Map([[7, tplA], [9, "auto min-content"]]));
const st = STYLES["uls-vue-growth"];
const css = st ? st.textContent : "";
const keyed = css.includes('[data-node-id="7"] [data-testid="node-widgets"]{grid-template-rows:min-content min-content auto min-content !important}')
    && css.includes('[data-node-id="9"]') && css.split("\n").length === 2;
VUE = false; GRAPH_NODES = []; tick();
out.g5 = keyed && STYLES["uls-vue-growth"].textContent === "";
console.log(JSON.stringify(out));
"""

MUTATIONS = [
    ("M1 rule ignores computeSize",
     '(t === "auto" && widgets[i].computeSize ? "min-content" : t)', "(t)", "g1"),
    ("M2 no count check",
     "if (!tracks.length || tracks.length !== widgets.length || widgets.length !== rows.length) return null;",
     "if (!tracks.length) return null;", "g2"),
    ("M3 no mounted-row check",
     "        if (el && host.contains(el) && !rows[i].contains(el)) return null;\n", "", "g2"),
    ("M4 hidden widgets get a row",
     "return !!(w && w.type) && !o.canvasOnly && !o.hidden && !o.advanced;",
     "return !!(w && w.type) && !o.canvasOnly && !o.advanced;", "g3"),
    ("M5 unchanged template still ruled",
     # v954: the template also carries the control-row tracks; the anchor is
     # the final "only when it differs" return
     'return out.join(" ") !== tracks.join(" ") ? out.join(" ") : "";',
     'return out.join(" ");', "g4"),
    ("M6 classic mode keeps the rules",
     "if (!vueMode()) { writeGrowthRules(new Map()); return; }", "if (!vueMode()) return;", "g5"),
    ("M7 rule without !important",
     "{grid-template-rows:${t} !important}", "{grid-template-rows:${t}}", "g5"),
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
        return None, p.stderr[-800:]
    import json
    return json.loads(p.stdout.strip().splitlines()[-1]), ""


def main():
    src = open(SRC, encoding="utf-8").read()
    fails = []
    res, err = run(src)
    if res is None:
        print("[v945] FAIL -- harness did not run:", err)
        return 1
    for k in ("g1", "g2", "g3", "g4", "g5"):
        print("  %s  %s" % ("ok  " if res.get(k) else "FAIL", k))
        if not res.get(k):
            fails.append(k)
    # G6 the inline binding is never written
    g6 = not re.search(r"gridTemplateRows\s*=[^=]", src) and "setProperty(\"grid-template-rows\"" not in src \
        and "setProperty('grid-template-rows'" not in src
    print("  %s  g6 no inline write of grid-template-rows" % ("ok  " if g6 else "FAIL"))
    if not g6:
        fails.append("g6")
    caught = 0
    for name, a, b, key in MUTATIONS:
        if src.count(a) != 1:
            print("  MUTATION ANCHOR BROKEN:", name)
            fails.append("anchor " + name)
            continue
        r, err = run(src.replace(a, b, 1))
        hit = r is None or not r.get(key)
        caught += hit
        print("  %s %s" % ("caught:" if hit else "MISSED:", name))
        if not hit:
            fails.append("mutation " + name)
    print("  mutations caught %d/%d" % (caught, len(MUTATIONS)))
    if fails:
        print("[test_v945_vue_growth_rule] FAIL --", fails)
        return 1
    print("[test_v945_vue_growth_rule] PASS -- growth rule G1-G6, %d/%d mutations" % (caught, len(MUTATIONS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
