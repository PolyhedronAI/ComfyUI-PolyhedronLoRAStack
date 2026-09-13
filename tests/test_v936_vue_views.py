#!/usr/bin/env python3
"""v936 -- ONE switch point decides when a node's Nodes 2.0 view exists.

WHY (S1 of the Nodes 2.0 concept, 10.09.2026)
---------------------------------------------
v934 made the Stack's DOM panel safe for the classic renderer: no widget
under LiteGraph, attach on LiteGraph.vueNodesMode, widget.hidden on the way
back, both serialize flags false, one global watch. Twenty painted nodes need
the same logic. Twenty copies would drift -- the tree has paid for "two places
compute the same thing" more than once. So the logic moved, unchanged, into
web/js/uls_vue_views.js, and a node now only REGISTERS what is its own.

WHAT IS PINNED HERE (the REAL module, driven in Node.js with TWO invented node
classes, so nothing Stack-specific can hide in it)
  G1  classic or unknown renderer: no widget on any registered node
  G2  Nodes 2.0: every registered class gets its view, built by ITS render,
      storing nothing (widget.serialize AND options.serialize false); the
      nodeCreated mark (_ulsDomPanel) is set for registered classes only;
      an unregistered class is never touched
  G3  back to classic: widget.hidden, each class's own leave() once
  G4  one class whose render throws does not stop the other, nor the watch
  G5  refreshVueView redraws a shown view and reports false otherwise
  G6  ready() false -> left alone, picked up once ready
  G7  ONE interval, however many classes register and nodes appear
  G8  registerVueView refuses a spec without widgetName or render
  S   static: the module names no node class, never fakes visibility with
      display:none, carries no draw hook and no computeSize edit
The Stack's own behaviour through this switch point stays pinned by
test_v934_stack_dom_classic_safe.py (P1-P6, unchanged) and, in a real
browser, by tools/browser_probe_stack.py.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWS = os.path.join(ROOT, "web", "js", "uls_vue_views.js")
STACK = os.path.join(ROOT, "web", "js", "uls_stack_dom.js")

HEAD = r"""
function mkEl(tag) {
  const el = { tagName: String(tag).toUpperCase(), className: "", children: [], _text: "",
    style: {}, appendChild(c) { this.children.push(c); return c; } };
  Object.defineProperty(el, "textContent", { get() { return el._text; },
    set(v) { el._text = v; if (v === "") el.children = []; } });
  return el;
}
globalThis.document = { createElement: mkEl };
globalThis.LiteGraph = {};
const _intervals = [];
globalThis.setInterval = (fn, ms) => { _intervals.push(fn); return _intervals.length; };
const _timeouts = [];
globalThis.setTimeout = (fn) => { _timeouts.push(fn); return 0; };
const _warn = []; console.warn = (...a) => _warn.push(a.join(" "));
console.log = () => {};
let _ext = null;
const _nodes = [];
globalThis.app = { registerExtension(e) { _ext = e; }, graph: { _nodes: _nodes }, canvas: {} };
"""

TAIL = r"""
function flush() { while (_timeouts.length) _timeouts.shift()(); }
function tick(n) { for (let i = 0; i < n; i++) for (const f of _intervals) f(); flush(); }
function mk(cls) {
  const n = { comfyClass: cls, type: cls, widgets: [], dirty: 0,
    addDOMWidget(name, type, element, options) {
      const w = { name, type, element, options, hidden: false };
      this.widgets.push(w); return w; },
    setDirtyCanvas() { this.dirty++; } };
  _nodes.push(n); _ext.nodeCreated(n); flush(); return n;
}
const W = (n, name) => n.widgets.find(w => w.name === name);
const log = { renderA: 0, renderB: 0, leaveA: 0, prepA: 0, throwB: false };
registerVueView("ClassA", { widgetName: "a_view", label: "A",
  prepare() { log.prepA++; },
  ready: n => !!n.stateA,
  render(n, root) { log.renderA++; root.textContent = ""; root.appendChild(document.createElement("span")); },
  leave(n) { log.leaveA++; } });
registerVueView("ClassB", { widgetName: "b_view",
  render(n, root) { if (log.throwB) throw new Error("boom"); log.renderB++; } });
_ext.setup?.();
const out = {};
let bad = 0;
try { registerVueView("X", { render() {} }); } catch (e) { bad++; }
try { registerVueView("X", { widgetName: "x" }); } catch (e) { bad++; }
out.g8 = bad;

const a = mk("ClassA"); a.stateA = { v: 1 };
const b = mk("ClassB");
const u = mk("Unregistered");
tick(10);
out.g1_unknown = a.widgets.length + b.widgets.length;
LiteGraph.vueNodesMode = false; tick(10);
out.g1_classic = a.widgets.length + b.widgets.length;
out.marks = [!!a._ulsDomPanel, !!b._ulsDomPanel, !!u._ulsDomPanel];

const ghost = mk("ClassA");                     // not ready yet
LiteGraph.vueNodesMode = true; tick(3);
out.g2_a = !!W(a, "a_view") && W(a, "a_view").hidden === false;
out.g2_b = !!W(b, "b_view") && W(b, "b_view").hidden === false;
out.g2_ser = [W(a, "a_view").serialize, W(a, "a_view").options.serialize,
              W(b, "b_view").serialize, W(b, "b_view").options.serialize].map(x => x === undefined ? "undefined" : x);
out.g2_renders = [log.renderA, log.renderB];
out.g2_prep = log.prepA;
out.g2_root_class = W(a, "a_view").element.className;
out.g2_unreg = u.widgets.length;
out.g6_ghost_before = ghost.widgets.length;
ghost.stateA = {}; tick(1);
out.g6_ghost_after = !!W(ghost, "a_view");
out.g5_refresh_shown = refreshVueView(a);
out.g5_render_after = log.renderA;

LiteGraph.vueNodesMode = false; tick(3);
out.g3_hidden = [W(a, "a_view").hidden, W(b, "b_view").hidden];
out.g3_leave = log.leaveA;          // two ClassA views were on screen: a, ghost
out.g5_refresh_hidden = refreshVueView(a);
out.g5_unreg = refreshVueView(u);

log.throwB = true;
const b2 = mk("ClassB"); const a2 = mk("ClassA"); a2.stateA = {};
LiteGraph.vueNodesMode = true;
try { tick(2); out.g4_threw = false; } catch (e) { out.g4_threw = true; }
out.g4_a2 = !!W(a2, "a_view") && W(a2, "a_view").hidden === false;
out.g4_warned = _warn.length > 0;
out.g7 = _intervals.length;
console.error(JSON.stringify(out));
"""

failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def inline(src):
    src = re.sub(r'import\s*\{\s*app\s*\}\s*from\s*"\.\./\.\./scripts/app\.js";', "", src)
    return re.sub(r"^export\s+", "", src, flags=re.M)


def code_only(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"^\s*//.*$", "", src, flags=re.M)
    return re.sub(r"\s//.*$", "", src, flags=re.M)


def main():
    print("v936: one switch point for every Nodes 2.0 view")
    if shutil.which("node") is None:
        print("  FAIL node (Node.js) not found -- this guard needs it")
        return 1
    if not os.path.exists(VIEWS):
        print("  FAIL web/js/uls_vue_views.js is missing")
        return 1
    src = open(VIEWS, encoding="utf-8").read()
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "h.mjs")
        open(path, "w", encoding="utf-8").write(HEAD + inline(src) + TAIL)
        p = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    lines = [ln for ln in p.stderr.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        check(False, "harness ran: " + (p.stderr or p.stdout)[-500:])
        return 1
    r = json.loads(lines[-1])

    check(r["g8"] == 2, "G8 a spec without widgetName or render is refused")
    check(r["g1_unknown"] == 0, "G1 flag unknown: no widget on any registered node")
    check(r["g1_classic"] == 0, "G1 classic renderer: still no widget")
    check(r["marks"] == [True, True, False],
          "G2 nodeCreated marks registered classes only (_ulsDomPanel)")
    check(r["g2_a"] and r["g2_b"], "G2 Nodes 2.0: each registered class gets its view")
    check(r["g2_ser"] == [False, False, False, False],
          "G2 every view stores nothing: widget.serialize AND options.serialize false")
    check(r["g2_renders"] == [1, 1], "G2 each view is built by ITS OWN render, once")
    check(r["g2_prep"] == 1, "G2 prepare() runs once per class, before its first view")
    check(r["g2_root_class"] == "", "G2 no class name is invented for a view that set none")
    check(r["g2_unreg"] == 0, "G2 an unregistered class is never touched")
    check(r["g6_ghost_before"] == 0 and r["g6_ghost_after"],
          "G6 ready() false -> left alone; picked up once ready")
    check(r["g5_refresh_shown"] is True and r["g5_render_after"] == 3,
          "G5 refreshVueView redraws a view that is on screen")
    check(r["g3_hidden"] == [True, True], "G3 back to classic: widget.hidden on every view")
    check(r["g3_leave"] == 2,
          "G3 ... and the class's own leave() runs once PER shown node (a, ghost: 2)")
    check(r["g5_refresh_hidden"] is False and r["g5_unreg"] is False,
          "G5 refreshVueView reports false for a hidden view and an unregistered node")
    check(r["g4_threw"] is False,
          "G4 a render that throws never escapes the watch")
    check(r["g4_a2"] and r["g4_warned"],
          "G4 a render that throws is contained: the other class still gets its view")
    check(r["g7"] == 1, "G7 ONE interval for everything (got %s)" % r["g7"])

    code = code_only(src)
    check(re.search(r"\bULS\w+|UltimateLoraStack", code) is None,
          "S  the switch point names no node class -- classes appear where they register")
    check(re.search(r"style\.display\s*=", code) is None,
          "S  visibility is never faked with display:none on an inner element")
    check("onDrawForeground" not in code and "computeSize" not in code,
          "S  no draw hook, no computeSize edit")
    for name in ("registerVueView", "vueMode", "refreshVueView"):
        check(re.search(r"export\s+function\s+" + name + r"\b", src) is not None,
              "S  exports %s" % name)
    stack = code_only(open(STACK, encoding="utf-8").read())
    check(re.search(r"registerVueView\(\s*STACK_CLASS\s*,", stack) is not None
          and "_ulsResize" in stack and "_uls?.rows" in stack,
          "S  the Stack registers with its leave (_ulsResize) and ready (_uls.rows)")
    check("setInterval" not in stack and "vueNodesMode" not in stack,
          "S  no switching logic is left behind in the Stack module")
    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    print("v936: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
