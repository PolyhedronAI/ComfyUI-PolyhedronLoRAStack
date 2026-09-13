#!/usr/bin/env python3
"""v940 -- S5: a widget hidden in the classic view is hidden in Nodes 2.0 too.

Measured 10.09.2026 (ComfyUI 0.33.4 / frontend 1.49.6): the classic renderer
reads widget.hidden, the Vue renderer reads widget.options.hidden. The suite
hides the classic way at ~50 places in 13 files, so under Nodes 2.0 internal
widgets came back as visible, editable fields (the Mask Editor's five stores,
blank rows in Cutout / Empty Latent). web/js/uls_vue_hidden.js makes
options.hidden of every pack widget DERIVED from the classic state.

  R  the REAL module, driven in Node.js: the classic test in all its forms;
     the derived value follows later changes; an explicit write wins and
     undefined clears it; a pre-existing true is kept; idempotent; options
     created if missing; foreign nodes untouched; one interval
  S  static: the hooks; the scanner no longer calls C2 "EMPTY", marks nodes
     with a view, and does not mistake this file for a node UI
The browser half: tools/browser_vue_hidden_audit.py (all 58 classes).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(ROOT, "web", "js", "uls_vue_hidden.js")
SCAN = os.path.join(ROOT, "tools", "renderer_scan.py")

HEAD = r"""
const _iv = []; globalThis.setInterval = (f) => { _iv.push(f); return _iv.length; };
const _to = []; globalThis.setTimeout = (f) => { _to.push(f); return 0; };
let _ext = null; const _nodes = [];
globalThis.app = { registerExtension(e) { _ext = e; }, graph: { _nodes: _nodes }, canvas: {} };
"""
TAIL = r"""
const out = {};
out.classic = [classicHidden({hidden: true}), classicHidden({type: "hidden"}),
  classicHidden({type: "pls-hidden-customtext"}), classicHidden({type: "number"}), classicHidden(null)];
const w = { name: "mask_store", type: "customtext", hidden: false, options: { serialize: true } };
out.installed = mirrorWidget(w); out.again = mirrorWidget(w);
out.v0 = w.options.hidden; w.hidden = true; out.v1 = w.options.hidden;
w.hidden = false; w.type = "pls-hidden-customtext"; out.v2 = w.options.hidden;
w.type = "customtext"; out.v3 = w.options.hidden;
w.options.hidden = true; out.ov = w.options.hidden; w.options.hidden = undefined; out.cleared = w.options.hidden;
out.keep_serialize = w.options.serialize;
const pre = { type: "number", options: { hidden: true } }; mirrorWidget(pre); out.pre = pre.options.hidden;
const bare = { type: "number", hidden: true }; mirrorWidget(bare); out.bare = bare.options && bare.options.hidden;
out.enum = Object.keys(w.options).includes("hidden") && !Object.keys(w.options).includes("__ulsHiddenMirror");
const mine = { comfyClass: "ULSMaskEditor", widgets: [{ type: "hidden" }] };
const stack = { comfyClass: "UltimateLoraStack", widgets: [{ type: "text", hidden: true }] };
const foreign = { comfyClass: "KSampler", widgets: [{ type: "number", hidden: true }] };
_nodes.push(mine, stack, foreign);
_ext.setup?.(); _ext.nodeCreated(mine); _ext.nodeCreated(stack); _ext.nodeCreated(foreign);
for (const f of _to.splice(0)) f(); for (const f of _iv) f();
out.mine = mine.widgets[0].options.hidden; out.stack = stack.widgets[0].options.hidden;
out.foreign = foreign.widgets[0].options === undefined;
out.iv = _iv.length;
const late = { comfyClass: "ULSCutout", widgets: [] }; _nodes.push(late); _ext.nodeCreated(late);
late.widgets.push({ type: "pls-hidden-number" }); for (const f of _iv) f();
out.late = late.widgets[0].options.hidden;
// the nudge: under Nodes 2.0 ONE trigger per changed node, none otherwise
const trig = [];
const g = { trigger(a, p) { trig.push([a, p.property, p.oldValue === p.newValue, p.nodeId]); } };
const nA = { id: 7, comfyClass: "ULSSeed", graph: g, showAdvanced: false, widgets: [{ type: "number" }, { type: "text" }] };
const nB = { id: 8, comfyClass: "ULSInt", graph: g, showAdvanced: false, widgets: [{ type: "number" }] };
_nodes.push(nA, nB);
globalThis.LiteGraph = { vueNodesMode: true };
for (const f of _iv) f();                       // first sight: remember, no nudge
out.first = trig.length;
nA.widgets[0].hidden = true; nA.widgets[1].hidden = true;
for (const f of _iv) f();
out.after_change = trig.slice();
trig.length = 0; for (const f of _iv) f(); out.unchanged = trig.length;
globalThis.LiteGraph.vueNodesMode = false; nB.widgets[0].hidden = true;
for (const f of _iv) f(); out.classic_nudge = trig.length;
console.error(JSON.stringify(out));
"""

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main():
    print("v940: options.hidden follows the classic hidden state")
    if shutil.which("node") is None:
        print("  FAIL node (Node.js) not found -- this guard needs it")
        return 1
    src = open(MOD, encoding="utf-8").read()
    body = re.sub(r'import\s*\{\s*app\s*\}\s*from\s*"\.\./\.\./scripts/app\.js";', "", src)
    # v942: vueMode now comes from the switch point (one home). The harness
    # gives it the switch point's own rule; test_v942 pins the import.
    body = re.sub(r'import\s*\{\s*vueMode\s*\}\s*from\s*"\./uls_vue_views\.js";',
                  "const vueMode = () => globalThis.LiteGraph?.vueNodesMode === true;", body)
    body = re.sub(r"^export\s+", "", body, flags=re.M)
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "h.mjs")
        open(path, "w", encoding="utf-8").write(HEAD + body + TAIL)
        p = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    lines = [ln for ln in p.stderr.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        check(False, "harness ran: " + (p.stderr or p.stdout)[-500:])
        return 1
    r = json.loads(lines[-1])
    check(r["classic"] == [True, True, True, False, False],
          "R  classic hidden = widget.hidden / type 'hidden' / 'pls-hidden-' prefix")
    check(r["installed"] is True and r["again"] is False, "R  installing is idempotent")
    check([r["v0"], r["v1"], r["v2"], r["v3"]] == [False, True, True, False],
          "R  options.hidden FOLLOWS every later classic change")
    check(r["ov"] is True and r["cleared"] is False,
          "R  an explicit write wins; undefined hands back to the classic state")
    check(r["keep_serialize"] is True, "R  the rest of options is untouched")
    check(r["pre"] is True, "R  a pre-existing options.hidden === true is kept")
    check(r["bare"] is True, "R  a widget without options gets them")
    check(r["enum"], "R  hidden is enumerable (spread/merge sees it); the marker is not")
    check(r["mine"] is True and r["stack"] is True,
          "R  pack nodes (ULS*, UltimateLoraStack) are covered at creation")
    check(r["foreign"] is True, "R  a foreign node is never touched")
    check(r["late"] is True, "R  a widget added later is picked up by the watch")
    check(r["iv"] == 1, "R  one interval")
    check(r["first"] == 0, "R  first sight of a widget remembers it, no nudge")
    check(r["after_change"] == [["node:property:changed", "showAdvanced", True, 7]],
          "R  a changed node under Nodes 2.0 gets ONE nudge, value unchanged, only that node (%s)"
          % r["after_change"])
    check(r["unchanged"] == 0 and r["classic_nudge"] == 0,
          "R  no nudge without a change, none under the classic renderer")
    check(re.search(r"nodeCreated\s*\(", src) and re.search(r"loadedGraphNode\s*\(", src),
          "S  hooks nodeCreated and loadedGraphNode")
    scan = open(SCAN, encoding="utf-8").read()
    check("EMPTY under Nodes 2.0\"" not in scan and "hand-drawn parts on the canvas" in scan,
          "S  the scanner no longer calls C2 'EMPTY'")
    check("def vue_views(" in scan and "[view]" in scan, "S  the scanner marks nodes with a view")
    check('"uls_vue_hidden.js"' in scan, "S  the scanner does not take this file for a node UI")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import renderer_scan as RS  # noqa: E402
    check(RS.vue_views(ROOT) >= {"UltimateLoraStack", "ULSAccelerator"},
          "S  vue_views finds the Stack and the Engine (%s)" % sorted(RS.vue_views(ROOT)))
    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    print("v940: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
