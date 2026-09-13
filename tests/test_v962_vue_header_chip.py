#!/usr/bin/env python3
"""v962 -- a control in the Vue title bar: spec.header at the switch point.

The painted Filter node carries its Reset chip in the title bar; under Nodes
2.0 (v941) it sat in a row at the foot -- the frontend offers no hook for the
title bar. Measured 13.09.2026 (frontend 1.49.6): the Vue title bar is
`.lg-node-header > div`, a flex row with the title at the left; an element
appended there sits at the right and survives title change, resize and redraw.
v962 gives the switch point (uls_vue_views.js) an optional spec.header(node,
chip): built once, appended to that row while the view is shown, put back by
the watch if a Vue re-render dropped it, removed on leave. A header-only spec
adds no DOM widget row. The Filter's Reset moved up; classic is untouched.

  W  the switch point, driven on a fake DOM: chip appended on show, put back
     when dropped, removed on leave; header-only spec adds no widget;
     pointerdown on the chip does not bubble (the title bar is the drag handle)
  R  registerVueView accepts header-only, still rejects a spec with neither
  S  Filter registers header-only; its chip calls _pfReset; no classic hook
"""
import json, os, re, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, "web", "js")
failures = []

def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)

def src(name):
    return open(os.path.join(JS, name), encoding="utf-8").read()

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
        return None, (p.stderr or p.stdout)[-600:]
    return json.loads(lines[-1]), None

HARNESS = r"""
function mkEl(tag) { const el = { tagName: String(tag).toUpperCase(), className: "", children: [], parentElement: null, _l: {},
  appendChild(c) { if (c.parentElement) c.parentElement.children = c.parentElement.children.filter(x => x !== c); c.parentElement = this; this.children.push(c); return c; },
  remove() { if (this.parentElement) { this.parentElement.children = this.parentElement.children.filter(x => x !== this); this.parentElement = null; } },
  addEventListener(n, f) { this._l[n] = f; }, querySelector() { return null; } };
  Object.defineProperty(el, "textContent", { get() { return ""; }, set(v) { if (v === "") el.children = []; } }); return el; }
let ROW = mkEl("div"); let NODE_EL = { querySelector: (q) => (q === ".lg-node-header > div" ? ROW : null) }; let HAVE_EL = true;
globalThis.document = { createElement: mkEl, head: mkEl("head"), querySelector: (q) => (HAVE_EL && q === '[data-node-id="7"]') ? NODE_EL : null };
globalThis.LiteGraph = { vueNodesMode: true };
const _iv = []; globalThis.setInterval = (f) => { _iv.push(f); return 1; };
const _to = []; globalThis.setTimeout = (f) => { _to.push(f); return 0; };
console.log = () => {}; console.warn = () => {};
let _ext = null; const _nodes = [];
globalThis.app = { registerExtension(e) { _ext = e; }, graph: { _nodes }, canvas: {} };
"""
TAIL = r"""
let built = 0; let widgetsAdded = 0;
registerVueView("HDR", { header: (n, chip) => { built++; chip.className += " built"; } });
let rejected = false; try { registerVueView("BAD", { label: "x" }); } catch (e) { rejected = true; }
let both = 0;
registerVueView("BOTH", { widgetName: "w", render: () => {}, header: () => { both++; } });
const mk = (c, id) => { const n = { id, comfyClass: c, type: c, widgets: [], addDOMWidget(name, t, el, o) { widgetsAdded++; const w = { name, element: el, options: o, hidden: false }; this.widgets.push(w); return w; }, setDirtyCanvas() {} }; _nodes.push(n); _ext.nodeCreated(n); return n; };
const n = mk("HDR", 7);
for (const f of _to.splice(0)) f();
const tick = () => { for (const f of _iv) f(); };
tick();
const out = {};
out.rejected = rejected;
out.chipInRow = ROW.children.length === 1 && ROW.children[0].className.includes("uls-vue-hdr") && ROW.children[0].className.includes("built");
out.widgetsAfterHeaderOnly = widgetsAdded; out.builtOnce = built;
const chip = ROW.children[0];
let bubbled = true; chip._l.pointerdown({ stopPropagation() { bubbled = false; } }); out.stopped = !bubbled;
// Vue re-renders: the row is rebuilt without our chip
ROW = mkEl("div"); NODE_EL = { querySelector: (q) => (q === ".lg-node-header > div" ? ROW : null) };
tick(); out.putBack = ROW.children.length === 1 && ROW.children[0] === chip; out.builtStill = built;
// no Vue element on screen: nothing to do, no throw
HAVE_EL = false; let threw = false; try { tick(); } catch (e) { threw = true; } out.noElOk = !threw; HAVE_EL = true;
// leave
LiteGraph.vueNodesMode = false; tick(); out.removedOnLeave = ROW.children.length === 0 && chip.parentElement === null;
LiteGraph.vueNodesMode = true; tick(); out.backAfterReturn = ROW.children.length === 1;
// a spec with row AND header adds its widget and builds its chip
const b = mk("BOTH", 7); for (const f of _to.splice(0)) f(); tick();
out.bothWidget = b.widgets.length === 1; out.bothChip = both === 1;
console.error(JSON.stringify(out));
"""

def main():
    print("v962: a control in the Vue title bar (spec.header)")
    if shutil.which("node") is None:
        print("  FAIL node (Node.js) not found -- this guard needs it")
        return 1
    views = src("uls_vue_views.js")
    vb = re.sub(r'import\s*\{\s*app\s*\}\s*from\s*"\.\./\.\./scripts/app\.js";', "", views)
    vb = re.sub(r"^export\s+", "", vb, flags=re.M)
    r, err = run_node(HARNESS + vb + TAIL)
    if r is None:
        check(False, "switch-point harness ran: " + err)
        return 1
    check(r["chipInRow"] and r["builtOnce"] == 1, "W  the chip is built once and appended to the title-bar row on show")
    check(r["widgetsAfterHeaderOnly"] == 0, "W  a header-only spec adds NO DOM widget row")
    check(r["stopped"], "W  pointerdown on the chip does not bubble into the drag handle")
    check(r["putBack"] and r["builtStill"] == 1, "W  a re-rendered row gets the SAME chip back; header() not called again")
    check(r["noElOk"], "W  no Vue element on screen: the tick does nothing and never throws")
    check(r["removedOnLeave"] and r["backAfterReturn"], "W  leave removes the chip from the title bar; return puts it back")
    check(r["bothWidget"] and r["bothChip"], "W  a spec with a row AND a header gets both")
    check(r["rejected"], "R  a spec with neither row nor header is rejected")
    ex = src("uls_extras_dom.js")
    m = re.search(r'registerVueView\("ULSFilter", \{([^}]*)\}\);', ex, re.S)
    check(m is not None and "header: renderFilter" in m.group(1) and "widgetName" not in m.group(1),
          "S  Filter registers header-only (Reset in the title bar, no row)")
    check(re.search(r"function renderFilter\(node, chip\) \{[^}]*_pfReset\(node\)", ex, re.S) is not None,
          "S  the chip's Reset calls the SHARED _pfReset")
    code = re.sub(r"/\*.*?\*/", "", views, flags=re.S); code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
    check(code.count("if (v.chip) { v.chip.remove(); v.chip = null; }") == 1
          and re.search(r"function headerRow\(node\) \{[^}]*\.lg-node-header > div", code, re.S) is not None,
          "S  the switch point removes the chip on leave and finds the row by the measured selector")
    return 1 if failures else 0

if __name__ == "__main__":
    rc = main()
    print("v962: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
