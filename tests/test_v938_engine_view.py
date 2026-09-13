#!/usr/bin/env python3
"""v938 -- S4: the Polyhedron Engine gets its Nodes 2.0 view, through shared code.

The Engine (ULSAccelerator) paints its whole body; under the Vue renderer it
was empty. Its view (web/js/uls_engine_dom.js) registers with the shared
switch point (v936) and CALLS the painted Engine's pieces, moved VERBATIM out
of its draw/onMouseDown into uls_node.js: ENGINE_MODES, ENGINE_MODE_LABELS,
stepEngineWeight, editEngineWeight. It also passes the new optional onPicked
of openLoraSelect, which fixes the Stack panel's picker (v928-v937 redrew
60 ms after OPENING the picker, before any choice -- measured 10.09.: state
test_b, panel "Select LoRA...").

  R  the REAL renderEngine(), driven in Node.js against recording stubs
  M  stepEngineWeight against the ORIGINAL inline expressions (kept here as
     the oracle) over a grid of weights up to the +-10 limits, both
     directions, weight and CLIP, with and without a decoupled CLIP value
  S  static: the painted Engine CALLS the pieces, every moved text exists
     ONCE, openLoraSelect reports the pick in both branches, the views
     rebuild nothing, both register correctly
Both renderers are driven in a real browser by tools/browser_probe_stack.py
(sections J and K).
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENG = os.path.join(ROOT, "web", "js", "uls_engine_dom.js")
STACK = os.path.join(ROOT, "web", "js", "uls_stack_dom.js")
NODE = os.path.join(ROOT, "web", "js", "uls_node.js")

HEAD = r"""
function mkEl(tag) {
  const el = { tagName: String(tag).toUpperCase(), className: "", title: "", children: [],
    _text: "", style: {}, type: "", checked: false,
    appendChild(c) { this.children.push(c); return c; } };
  Object.defineProperty(el, "textContent", { get() { return el._text; },
    set(v) { el._text = v; if (v === "") el.children = []; } });
  return el;
}
globalThis.document = { createElement: mkEl, head: mkEl("head") };
const calls = [];
let SPEC = null;
const registerVueView = (cls, spec) => { SPEC = Object.assign({ cls }, spec); };
const ensureCss = () => calls.push(["css"]);
const openLoraSelect = (row, list, e, node, cb) => calls.push(["pick", row.name, typeof cb, cb]);
const loadLoraList = () => Promise.resolve(), getLoraList = () => ["x"];
const applyNorm = (x) => x || "auto", applyNext = (x) => (x === "bypass" ? "baked" : "bypass");
const WEIGHT_HDR_TIP_LINES = ["Weight / CLIP Strength", "TIP2"];
const newEngineRow = () => ({ enabled: true, name: "None", weight: 1.0, group: "\u2014" });
const APPLY_INFO = { auto: { label: "Auto", color: "#aaa" }, bypass: { label: "Bypass", color: "#bbb" }, baked: { label: "Baked", color: "#ccc" } };
const openGroupPreviewOverlay = (row, e, n) => calls.push(["preview", row.name, n]);
const ENGINE_MODES = [{ key: "SEQ", letter: "S", color: "#1" }, { key: "CONCAT", letter: "C", color: "#2" }, { key: "DARE", letter: "D", color: "#3" }];
const ENGINE_MODE_LABELS = { SEQ: "Sequential (SEQ)", CONCAT: "Combined (CONCAT)", DARE: "Smooth Mix (DARE)" };
const ENGINE_MODE_TIPS = { SEQ: { label: "Sequential (SEQ)", hint: "h1" }, CONCAT: { label: "Combined (CONCAT)", hint: "h2" },
                           DARE: { label: "Smooth Mix (DARE)", hint: "h3" } };
const stepEngineWeight = (row, dir, clip) => calls.push(["step", row.name, dir, clip]);
const editEngineWeight = (row, e, clip, cb) => calls.push(["edit", row.name, clip, typeof cb, cb]);
"""

TAIL = r"""
function walk(el, f) { f(el); for (const c of el.children || []) walk(c, f); }
function find(root, cls) { const out = []; walk(root, e => { if ((" " + e.className + " ").includes(" " + cls + " ")) out.push(e); }); return out; }
let syncs = 0;
const node = { _uls: { isEngine: true, mode: "SEQ", apply: "auto", dareVariant: "channel",
  rows: [ { enabled: true, name: "a.safetensors", weight: 1.0 } ] },
  _ulsSync() { syncs++; }, setDirtyCanvas() {}, _engineResize() { calls.push(["resize"]); } };
const root = mkEl("div");
const out = { cls: SPEC.cls, widget: SPEC.widgetName, ready: SPEC.ready(node),
              ready_no: SPEC.ready({ _uls: { rows: [] } }) };
SPEC.prepare(); out.css = calls.some(c => c[0] === "css");
SPEC.render(node, root);
out.modes = find(root, "uls-eng-mode").map(b => [b.textContent, b.className.includes("on"), b.title]);
out.dv0 = find(root, "uls-eng-dv").length;
out.line = find(root, "uls-eng-line")[0].textContent;
out.tip = find(root, "uls-dom-whdr")[0].title;
find(root, "uls-eng-mode")[2].onclick({});
out.mode = node._uls.mode; out.sync_after_mode = syncs;
out.dv1 = find(root, "uls-eng-dv").map(b => b.textContent);
find(root, "uls-eng-dv")[0].onclick({});
out.variant = node._uls.dareVariant;
find(root, "uls-eng-apply")[0].onclick({});
out.apply = node._uls.apply; out.apply_label = find(root, "uls-eng-apply")[0].textContent;
calls.length = 0;
const st = find(root, "uls-dom-step");
st[0].onclick({ shiftKey: false }); st[1].onclick({ shiftKey: true });
find(root, "uls-dom-w")[0].onclick({ shiftKey: true });
find(root, "uls-dom-name")[0].onclick({});
find(root, "uls-dom-badge")[0].onclick({});
out.calls = calls.map(c => c.slice(0, 4).map(x => typeof x === "function" ? "fn" : x));
const pick = calls.find(c => c[0] === "pick");
node._uls.rows[0].name = "zeta.safetensors";
if (pick && typeof pick[3] === "function") pick[3]();
out.after_pick = find(root, "uls-dom-name")[0].textContent;
node._uls.rows[0].wClip = 0.8; SPEC.render(node, root);
out.wbox = find(root, "uls-dom-w")[0].textContent;
find(root, "uls-dom-x")[0].onclick({});
out.after_del = node._uls.rows.map(r => r.name);
find(root, "uls-dom-add")[0].onclick({});
out.after_add = node._uls.rows.length;
calls.length = 0; SPEC.leave(node);
out.leave = calls.map(c => c[0]);
console.error(JSON.stringify(out));
"""

ORACLE = r"""
// the ORIGINAL inline expressions of the painted Engine (v937), as the oracle
function oldStep(row, dir, shiftKey) {
  if (dir < 0) {
    if (shiftKey) { const base = (typeof row.wClip === "number") ? row.wClip : (row.weight || 0);
      row.wClip = Math.round(Math.max(-10, base - 0.05) * 100) / 100; }
    else { row.weight = Math.round(Math.max(-10, (row.weight || 0) - 0.05) * 100) / 100; }
  } else {
    if (shiftKey) { const base = (typeof row.wClip === "number") ? row.wClip : (row.weight || 0);
      row.wClip = Math.round(Math.min(10, base + 0.05) * 100) / 100; }
    else { row.weight = Math.round(Math.min(10, (row.weight || 0) + 0.05) * 100) / 100; }
  }
}
let bad = 0, n = 0;
for (let w = -10.2; w <= 10.2; w += 0.07) for (const wc of [undefined, 0.33, -9.98, 9.99]) {
  for (const dir of [-1, 1]) for (const sh of [false, true]) {
    const a = { weight: +w.toFixed(4) }, b = { weight: +w.toFixed(4) };
    if (wc !== undefined) { a.wClip = wc; b.wClip = wc; }
    oldStep(a, dir, sh); stepEngineWeight(b, dir, sh); n++;
    if (JSON.stringify(a) !== JSON.stringify(b)) bad++;
  }
}
console.error(JSON.stringify({ n, bad }));
"""

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def code_only(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"^\s*//.*$", "", src, flags=re.M)
    return re.sub(r"\s//.*$", "", src, flags=re.M)


def body_of(src, head, start=0):
    i = src.index(head, start)
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
    return src[i:]


def run_node(code):
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "h.mjs")
        open(path, "w", encoding="utf-8").write(code)
        p = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    lines = [ln for ln in p.stderr.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        return None, (p.stderr or p.stdout)[-500:]
    return json.loads(lines[-1]), None


def main():
    print("v938: the Engine's Nodes 2.0 view, through shared code")
    if shutil.which("node") is None:
        print("  FAIL node (Node.js) not found -- this guard needs it")
        return 1
    eng = open(ENG, encoding="utf-8").read()
    node = open(NODE, encoding="utf-8").read()
    stack = open(STACK, encoding="utf-8").read()

    body = eng.split('} from "./uls_node.js";', 1)[1]
    r, err = run_node(HEAD + body + TAIL)
    if r is None:
        check(False, "harness ran: " + err)
        return 1
    check(r["cls"] == "ULSAccelerator" and r["widget"] == "uls_engine_dom",
          "R  registers the Engine class with its own widget name")
    check(r["ready"] is True and r["ready_no"] is False,
          "R  ready only for a configured Engine (_uls.isEngine)")
    check(r["css"], "R  wears the Stack panel's styles (ensureCss), not a copy")
    check([m[0] for m in r["modes"]] == ["S", "C", "D"] and r["modes"][0][1]
          and r["modes"][2][2] == "Smooth Mix (DARE) \u2014 h3",
          "R  S | C | D from the SHARED modes, active one marked, the painted tooltip as title")
    check(r["dv0"] == 0 and r["dv1"] == ["CHAN"],
          "R  the CHAN/ELEM pill exists in DARE only")
    check(r["line"] == "\u25b8 Sequential (SEQ)", "R  the mode line from the SHARED labels")
    check(r["tip"] == "Weight / CLIP Strength\nTIP2", "R  the weight-header explainer, shared")
    check(r["mode"] == "DARE" and r["sync_after_mode"] >= 1,
          "R  a mode click sets the node's state and persists through ITS _ulsSync")
    check(r["variant"] == "element", "R  the variant pill toggles channel -> element")
    check(r["apply"] == "bypass" and r["apply_label"] == "BYPASS",
          "R  the Apply pill cycles through the SHARED applyNext, label from APPLY_INFO")
    check(r["calls"] == [["step", "a.safetensors", -1, False],
                         ["step", "a.safetensors", 1, True],
                         ["edit", "a.safetensors", True, "function"],
                         ["pick", "a.safetensors", "function", "fn"],
                         ["preview", "a.safetensors", None]],
          "R  \u25c0 / Shift+\u25b6 / Shift+box / name / thumb call the SHARED pieces (%s)"
          % r["calls"])
    check(r["after_pick"] == "zeta", "R  the picker's onPicked redraws the view (%r)" % r["after_pick"])
    check(r["wbox"] == "1.00 / 0.80", "R  a decoupled CLIP strength shows beside the weight")
    check(r["after_del"] == ["None"], "R  removing the last row leaves a fresh Engine row")
    check(r["after_add"] == 2, "R  '+' adds an Engine row")
    check(r["leave"] == ["resize"], "R  leaving hands the height back to _engineResize")

    # M: the moved weight rule equals the original inline rule
    step_src = body_of(node, "function stepEngineWeight(")
    m, err = run_node(step_src + "\n" + ORACLE)
    if m is None:
        check(False, "oracle ran: " + err)
    else:
        check(m["n"] > 4000 and m["bad"] == 0,
              "M  stepEngineWeight == the original inline expressions on %d cases (%d differ)"
              % (m["n"], m["bad"]))

    nc = code_only(node)
    em = body_of(nc, "nodeType.prototype.onMouseDown = function",
                 nc.index("name: \"Polyhedron.engine\""))
    check(re.search(r"\bstepEngineWeight\(row, -1, e\.shiftKey\)", em)
          and re.search(r"\bstepEngineWeight\(row, \+1, e\.shiftKey\)", em)
          and re.search(r"\beditEngineWeight\(row, e, e\.shiftKey,", em),
          "S  the painted Engine CALLS the shared weight pieces")
    check("Math.round(Math.max(-10" not in em and "showWeightInput(" not in em,
          "S  the inline copies are gone from the painted Engine's onMouseDown")
    check(node.count('"Smooth Mix (DARE)"') == 1 and node.count('letter: "D"') == 1
          and node.count("bundles and spreads") == 1
          and "modesArr = ENGINE_MODES" in nc and "modeLabels = ENGINE_MODE_LABELS" in nc
          and "TOOLTIP_HINTS = ENGINE_MODE_TIPS" in nc,
          "S  modes, labels and hints exist ONCE; the painted header and tooltip read them")
    ls = body_of(nc, "function openLoraSelect(")
    check(ls.startswith("function openLoraSelect(row, loraList, e, node, onPicked)")
          and ls.count("node._ulsSync(); onPicked?.();") == 2,
          "S  openLoraSelect reports the pick (onPicked) after the node's own sync, both branches")
    mx = re.search(r"export\s*\{([^}]*)\}", node)
    exported = {x.strip() for x in re.sub(r"//[^\n]*", "", mx.group(1)).replace("\n", " ").split(",")}
    for name in ("newEngineRow", "APPLY_INFO", "openGroupPreviewOverlay", "ENGINE_MODES",
                 "ENGINE_MODE_LABELS", "stepEngineWeight", "editEngineWeight"):
        check(name in exported, "S  uls_node.js exports %s" % name)
    ec = code_only(eng)
    for bad in ("Math.round", "showWeightInput", "Smooth Mix", "Sequential (SEQ)", "bundles",
                "setInterval", "vueNodesMode", "addDOMWidget"):
        check(bad not in ec, "S  the Engine view rebuilds nothing: no %r" % bad)
    sc = code_only(stack)
    check(re.search(r"openLoraSelect\(row, getLoraList\(\), e, node, \(\) => commit\(node, root\)\)", sc)
          and "setTimeout(() => commit(node, root), 60)" not in sc,
          "S  the Stack panel redraws AFTER the pick, not 60 ms after opening")
    check(re.search(r"export\s+function\s+ensureCss\b", stack) is not None,
          "S  the Stack panel exports its styles for the Engine view")
    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    print("v938: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
