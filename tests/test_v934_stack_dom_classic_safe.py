#!/usr/bin/env python3
"""v934 -- the classic LoRA Stack stays usable AT ANY TIME.

FIELD FINDING 10.09.2026
------------------------
With Nodes 2.0 switched OFF the painted Stack looked normal and took no click.
Measured in a real browser (ComfyUI 0.33.4, frontend 1.49.6, Chromium): the
frontend wraps every DOM widget in a container that stays clickable
(pointer-events:auto) for as long as the WIDGET is not `hidden`. v922-v928
hid only the inner element and zeroed computeSize; the container kept a stale
height (68 px, 1000 px after a trip through Nodes 2.0) over the painted rows.
onMouseDown of the node: 0 calls. Control runs: v920, and v933 without
uls_stack_dom.js, took every click. A second fault sat beside it: the v928
watch treated "no canvas frame for 1500 ms" as "Nodes 2.0" -- LiteGraph does
not paint while idle, so the classic node was rebuilt every two seconds.

WHAT MUST STAY TRUE (driven here on the REAL module, not grepped)
----------------------------------------------------------------
  P1  classic renderer: nodeCreated + many watch ticks create NO widget
  P2  flag true: the panel is attached, visible, and stores nothing
      (widget.serialize === false AND options.serialize === false --
      the frontend does not copy one onto the other)
  P3  flag false again: widget.hidden = true (the one switch both LiteGraph's
      layout and the DOM overlay honour) and the height goes back through the
      painted view's own _ulsResize
  P4  flag true again: visible again, re-rendered from node._uls
  P5  ONE global watch, never a per-node timer; an unknown/missing flag means
      classic
  P6  an unconfigured node (no _uls yet) is left alone, then picked up

The browser half of this promise is tools/browser_probe_stack.py (12 checks,
needs a running ComfyUI + Chromium). This file is the half every cold start
can run.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, "web", "js", "uls_stack_dom.js")

HARNESS_HEAD = r"""
function mkEl(tag) {
  const el = {
    tagName: String(tag || "").toUpperCase(), className: "", title: "",
    style: {}, dataset: {}, children: [], _text: "", type: "", checked: false,
    appendChild(c) { this.children.push(c); c.parentNode = this; return c; },
    addEventListener() {}, removeEventListener() {}, setAttribute(k, v) { this[k] = v; },
  };
  Object.defineProperty(el, "textContent", {
    get() { return el._text; },
    set(v) { el._text = v; if (v === "") el.children = []; },
  });
  return el;
}
globalThis.document = { createElement: mkEl, head: mkEl("head"), body: mkEl("body") };
globalThis.LiteGraph = {};                       // vueNodesMode unset = unknown
const _intervals = [];
globalThis.setInterval = (fn, ms) => { _intervals.push({ fn, ms }); return _intervals.length; };
globalThis.clearInterval = () => {};
const _timeouts = [];
globalThis.setTimeout = (fn) => { _timeouts.push(fn); return 0; };
let _ext = null;
const _graphNodes = [];
globalThis.app = { registerExtension(e) { _ext = e; }, graph: { _nodes: _graphNodes },
                   canvas: { graph: null } };
globalThis.__stubs = {
  GROUP_COLORS: { "\u2014": "#404050" }, APPLY_INFO: { auto: { label: "Auto", color: "#9a9aaa" } },
  app: globalThis.app,
  openLoraSelect() {}, showWeightInput() {}, newRow() { return { enabled: true, name: "", wLow: 1 }; },
  loadLoraList() { return Promise.resolve(); }, getLoraList() { return []; },
  applyNorm(x) { return x || "auto"; }, applyNext(x) { return x; },
  showGroupModePopup() {}, openPreviewOverlay() {},
  // v937: the parity pieces the panel now imports (stubs; driven in test_v937)
  insertRowTrigger() {}, openStackOrderInput() {}, checkConflicts() { return []; },
  WEIGHT_HDR_TIP_LINES: ["Weight / CLIP Strength"],
};
"""

HARNESS_TAIL = r"""
function flush() { while (_timeouts.length) _timeouts.shift()(); }
function tick(n) { for (let i = 0; i < n; i++) for (const t of _intervals) t.fn(); flush(); }
function mkNode(configured) {
  const n = {
    comfyClass: "UltimateLoraStack", type: "UltimateLoraStack", widgets: [],
    size: [460, 300], graph: app.graph, _resizes: 0, _dirty: 0,
    addDOMWidget(name, type, element, options) {
      const w = { name, type, element, options, hidden: false, computeSize: undefined };
      this.widgets.push(w); this._added = (this._added || 0) + 1; return w;
    },
    _ulsResize() { this._resizes++; },
    setDirtyCanvas() { this._dirty++; },
  };
  if (configured) n._uls = { rows: [{ enabled: true, name: "a.safetensors", wLow: 1 }], apply: "auto" };
  _graphNodes.push(n);
  return n;
}
const W = n => n.widgets.find(w => w.name === "uls_rows_dom");
function walk(el, f) { f(el); for (const c of el.children || []) walk(c, f); }
function rowsIn(el) { let k = 0; walk(el, e => { if (/(^| )uls-dom-row( |$)/.test(e.className || "")) k++; }); return k; }
function textIn(el) { let t = ""; walk(el, e => { t += " " + (e._text || "") + " " + (e.title || ""); }); return t; }
const out = {};
_ext.setup?.();

// P1 -- classic, flag unset AND flag false
const a = mkNode(true);
_ext.nodeCreated(a); flush(); tick(20);
out.p1_unset_widgets = a.widgets.length;
LiteGraph.vueNodesMode = false; tick(20);
out.p1_false_widgets = a.widgets.length;
out.p5_intervals = _intervals.length;

// P6 -- unconfigured node, flag true
const b = mkNode(false);
_ext.nodeCreated(b); flush();
LiteGraph.vueNodesMode = true; tick(3);
out.p6_unconfigured_widgets = b.widgets.length;
b._uls = { rows: [], apply: "auto" }; tick(2);
out.p6_picked_up = !!W(b) && W(b).hidden === false;

// P2 -- attached under the Vue renderer
const w = W(a);
out.p2_attached = !!w;
out.p2_hidden = w ? w.hidden : null;
const _v = x => (x === undefined ? "undefined" : x);   // JSON drops undefined
out.p2_ser_widget = w ? _v(w.serialize) : "missing";
out.p2_ser_option = w ? _v(w.options.serialize) : "missing";
out.p2_rendered = w ? w.element.children.length > 0 : false;
out.p2_added_once = a._added;
tick(10);
out.p2_still_once = a._added;

// P3 -- back to classic
const resBefore = a._resizes;
LiteGraph.vueNodesMode = false; tick(3);
out.p3_hidden = W(a).hidden;
out.p3_resized = a._resizes - resBefore;
out.p3_resize_once = (tick(10), a._resizes - resBefore);

// P4 -- Vue again, state moved meanwhile
out.p4_rows_before = rowsIn(W(a).element);
a._uls.rows.push({ enabled: true, name: "zeta_probe.safetensors", wLow: 1 });
LiteGraph.vueNodesMode = true; tick(2);
out.p4_hidden = W(a).hidden;
out.p4_still_one_widget = a.widgets.length;
out.p4_rows_rendered = rowsIn(W(a).element);
out.p4_text_has_new = textIn(W(a).element).includes("zeta_probe");
out.p5_intervals_end = _intervals.length;
console.log(JSON.stringify(out));
"""

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


VIEWS = os.path.join(ROOT, "web", "js", "uls_vue_views.js")


def _inline_views():
    """v936: the switching logic moved into the shared switch point
    (uls_vue_views.js). ES imports are evaluated BEFORE the harness stubs, so
    the switch point is inlined here -- its own source, imports and `export`
    keywords removed -- instead of being imported. The promises below (P1-P6)
    are unchanged; they now drive the Stack THROUGH the switch point."""
    if not os.path.exists(VIEWS):
        return ""
    v = open(VIEWS, encoding="utf-8").read()
    v = re.sub(r'import\s*\{\s*app\s*\}\s*from\s*"\.\./\.\./scripts/app\.js";', "", v)
    v = re.sub(r"^export\s+", "", v, flags=re.M)
    return v + "\n"


def build(src):
    head = src.split('} from "./uls_node.js";', 1)
    if len(head) != 2:
        return None
    body = head[1]
    pre = re.sub(r'import\s*\{\s*app\s*\}\s*from\s*"\.\./\.\./scripts/app\.js";', "", head[0])
    pre = re.sub(r'import\s*\{\s*registerVueView\s*\}\s*from\s*"\./uls_vue_views\.js";', "", pre)
    pre = re.sub(r"import\s*\{[^}]*$", "", pre, flags=re.S)
    return (HARNESS_HEAD + _inline_views()
            + "const { app: _app, openLoraSelect, showWeightInput, newRow, loadLoraList,"
              " getLoraList, applyNorm, applyNext, showGroupModePopup,"
              " openPreviewOverlay, insertRowTrigger, openStackOrderInput,"
              " checkConflicts, WEIGHT_HDR_TIP_LINES,"
              " GROUP_COLORS, APPLY_INFO } = globalThis.__stubs;\n"   # v949/v950 imports
            + pre + body + HARNESS_TAIL)


def drive(src):
    code = build(src)
    if code is None:
        return None, "import block not found"
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "h.mjs")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        p = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
        if p.returncode != 0:
            return None, (p.stderr or p.stdout)[-600:]
        return json.loads(p.stdout.strip().splitlines()[-1]), None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def judge(r):
    check(r["p1_unset_widgets"] == 0,
          "P1 flag unknown: no widget after nodeCreated + 20 ticks")
    check(r["p1_false_widgets"] == 0,
          "P1 classic renderer: still no widget after 20 more ticks")
    check(r["p2_attached"] and r["p2_hidden"] is False,
          "P2 Nodes 2.0: panel attached and visible")
    check(r["p2_ser_widget"] is False,
          "P2 widget.serialize === false (keeps it out of widgets_values; got %r)"
          % (r["p2_ser_widget"],))
    check(r["p2_ser_option"] is False,
          "P2 options.serialize === false (keeps it out of the prompt)")
    check(r["p2_rendered"], "P2 the panel is rendered from node._uls")
    check(r["p2_added_once"] == 1 and r["p2_still_once"] == 1,
          "P2 attached exactly once, not per tick")
    check(r["p3_hidden"] is True,
          "P3 back to classic: widget.hidden = true")
    check(r["p3_resized"] == 1 and r["p3_resize_once"] == 1,
          "P3 height handed back once through the painted _ulsResize")
    check(r["p4_hidden"] is False and r["p4_still_one_widget"] == 1,
          "P4 Nodes 2.0 again: same widget, visible again")
    check(r["p4_rows_before"] == 1 and r["p4_rows_rendered"] == 2
          and r["p4_text_has_new"],
          "P4 re-rendered from node._uls: rows %s -> %s, new LoRA shown"
          % (r["p4_rows_before"], r["p4_rows_rendered"]))
    check(r["p5_intervals"] == 1 and r["p5_intervals_end"] == 1,
          "P5 one global watch for all Stacks, never a timer per node")
    check(r["p6_unconfigured_widgets"] == 0 and r["p6_picked_up"],
          "P6 an unconfigured node is left alone, then picked up")


def main():
    print("v934: classic Stack stays usable -- the panel exists only under Nodes 2.0")
    if shutil.which("node") is None:
        print("  FAIL node (Node.js) not found -- this guard needs it")
        return 1
    src = open(JS, encoding="utf-8").read()
    r, err = drive(src)
    if r is None:
        check(False, "harness ran: " + str(err))
        return 1
    judge(r)

    # Static half: the removed mechanism must not creep back (code only).
    code = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
    code = re.sub(r"\s//.*$", "", code, flags=re.M)
    check("onDrawForeground" not in code,
          "no draw hook on the Stack from this file")
    check(re.search(r"style\.display\s*=", code) is None,
          "visibility is never faked by display:none on the inner element")
    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    print("v934: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
