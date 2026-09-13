#!/usr/bin/env python3
"""v941 -- S5b: the painted extras of five nodes, under Nodes 2.0.

S5 measured that eighteen C2 nodes are not empty under Nodes 2.0; what is
missing is what their canvas code paints. Three of those parts are functions
(Int's preset chips and "+", Filter's Reset), the rest information (CLIP
Encode's word band, the Load nodes' status line, Everywhere's fed line).
web/js/uls_extras_dom.js gives each a small view through the switch point,
and every view CALLS the node's own pieces, moved verbatim in v941.

  W  the switch point's new signature: redraw on a change only; a throwing
     signature never escapes; views without one unchanged
  V  the REAL views, driven against recording stubs
  S  static: the painted code calls the shared pieces; every moved text
     exists once; the views rebuild nothing; seven classes register
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

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
        return None, (p.stderr or p.stdout)[-500:]
    return json.loads(lines[-1]), None


DOM = r"""
function mkEl(tag) { const el = { tagName: String(tag).toUpperCase(), className: "", title: "", children: [], _text: "", style: {},
  appendChild(c) { this.children.push(c); return c; } };
  Object.defineProperty(el, "textContent", { get() { return el._text; }, set(v) { el._text = v; if (v === "") el.children = []; } });
  return el; }
globalThis.document = { createElement: mkEl, head: mkEl("head") };
function walk(el, f) { f(el); for (const c of el.children || []) walk(c, f); }
function find(root, cls) { const o = []; walk(root, e => { if ((" " + e.className + " ").includes(" " + cls + " ")) o.push(e); }); return o; }
"""

W_HARNESS = r"""
globalThis.LiteGraph = { vueNodesMode: true };
const _iv = []; globalThis.setInterval = (f) => { _iv.push(f); return 1; };
const _to = []; globalThis.setTimeout = (f) => { _to.push(f); return 0; };
console.log = () => {}; console.warn = () => {};
let _ext = null; const _nodes = [];
globalThis.app = { registerExtension(e) { _ext = e; }, graph: { _nodes }, canvas: {} };
"""
W_TAIL = r"""
let renders = 0, state = 1;
registerVueView("CLS", { widgetName: "v", signature: n => state, render: () => { renders++; } });
let bad = 0;
registerVueView("BAD", { widgetName: "b", signature: n => { throw new Error("x"); }, render: () => { bad++; } });
let plain = 0;
registerVueView("PLAIN", { widgetName: "p", render: () => { plain++; } });
const mk = (c) => { const n = { comfyClass: c, type: c, widgets: [], addDOMWidget(name, t, el, o) { const w = { name, element: el, options: o, hidden: false }; this.widgets.push(w); return w; }, setDirtyCanvas() {} }; _nodes.push(n); _ext.nodeCreated(n); return n; };
mk("CLS"); mk("BAD"); mk("PLAIN");
for (const f of _to.splice(0)) f();
const tick = () => { for (const f of _iv) f(); };
tick(); tick();
const out = { first: renders };
tick(); out.same = renders;
state = 2; tick(); out.changed = renders;
tick(); out.again = renders;
let threw = false; try { tick(); tick(); } catch (e) { threw = true; }
out.bad = bad; out.threw = threw; out.plain = plain;
console.error(JSON.stringify(out));
"""

V_STUBS = r"""
const calls = [];
let SPECS = {};
const registerVueView = (cls, spec) => { SPECS[cls] = spec; };
let VALUE = 4;
const getValue = () => VALUE, setValue = (n, v) => { calls.push(["setValue", v]); VALUE = v; };
const addIntPreset = (n) => { calls.push(["add"]); return true; };
const removeIntPreset = (n, i) => { calls.push(["remove", i]); return true; };
const INT_EMPTY_HINT = "HINT";
const _pfReset = (n) => calls.push(["reset"]);
const _barLines = (n) => ["L1", "L2"];
const _counterText = (n) => "CT";
const loadStatusLine = (n) => ({ text: "STATUS", colour: "#ff8c00" });
const fedLine = (n) => "FED";
// v959/v962 views live in the same file; their piece is stubbed so the module loads
const switchStatusLine = (n) => "SW";
"""
V_TAIL = r"""
const out = {};
out.classes = Object.keys(SPECS).sort();
const r = mkEl("div");
const nInt = { _phi: { rows: [{ name: "low", value: 4 }, { name: "", value: 9 }] } };
SPECS.ULSInt.render(nInt, r);
out.chips = find(r, "uls-x-chip").map(c => [c.textContent, c.className.includes("on")]);
find(r, "uls-x-chip")[1].onclick();
let prevented = false;
find(r, "uls-x-chip")[0].oncontextmenu({ preventDefault() { prevented = true; } });
find(r, "uls-x-add")[0].onclick();
out.int_calls = calls.splice(0);
out.prevented = prevented;
SPECS.ULSInt.render({ _phi: { rows: [] } }, r);
out.hint = (find(r, "uls-x-hint")[0] || {}).textContent;
out.int_ready = [SPECS.ULSInt.ready({ _phi: {} }), SPECS.ULSInt.ready({})];
SPECS.ULSFilter.header({}, r); find(r, "uls-x-btn")[0].onclick(); out.filter = calls.splice(0);   // v962: header chip
out.filter_no_row = !SPECS.ULSFilter.widgetName && !SPECS.ULSFilter.render;
SPECS.ULSCLIPTextEncode.render({}, r); out.band = find(r, "uls-x-line").map(e => e.textContent);
out.cte_sig = SPECS.ULSCLIPTextEncode.signature({ size: [300, 100] });
SPECS.ULSLoadCLIP.render({}, r); const ln = find(r, "uls-x-line")[0]; out.load = [ln.textContent, ln.style.color];
out.load_same = SPECS.ULSLoadModel.widgetName === SPECS.ULSLoadVAE.widgetName && SPECS.ULSLoadCLIP.render === SPECS.ULSLoadVAE.render;
// public build (v374): ULSEverywhere is internal-only, its view is not carried
out.stores = Object.values(SPECS).every(s => !("serialize" in s));
console.error(JSON.stringify(out));
"""


def main():
    print("v941: the painted extras of five nodes, under Nodes 2.0")
    if shutil.which("node") is None:
        print("  FAIL node (Node.js) not found -- this guard needs it")
        return 1
    views = src("uls_vue_views.js")
    vb = re.sub(r'import\s*\{\s*app\s*\}\s*from\s*"\.\./\.\./scripts/app\.js";', "", views)
    vb = re.sub(r"^export\s+", "", vb, flags=re.M)
    r, err = run_node(DOM + W_HARNESS + vb + W_TAIL)
    if r is None:
        check(False, "switch-point harness ran: " + err)
        return 1
    check(r["first"] == 1 and r["same"] == 1, "W  a signature view renders once, not again while unchanged")
    check(r["changed"] == 2 and r["again"] == 2, "W  a changed signature redraws exactly once")
    check(r["bad"] == 1 and not r["threw"], "W  a throwing signature never escapes the watch")
    check(r["plain"] == 1, "W  a view without a signature is drawn once, as before")

    ex = src("uls_extras_dom.js")
    body = re.sub(r'^import[^;]*;\s*$', "", ex, flags=re.M)
    v, err = run_node(DOM + V_STUBS + body + V_TAIL)
    if v is None:
        check(False, "views harness ran: " + err)
        return 1
    six = ["ULSInt", "ULSFilter", "ULSCLIPTextEncode", "ULSLoadCLIP", "ULSLoadModel", "ULSLoadVAE"]
    # v959 registers the two switches in this file too (guarded by test_v959);
    # this guard pins that the six of v941 are still all there.
    check(all(c in v["classes"] for c in six),
          "V  the six v941 classes register a view (%s)" % v["classes"])
    check(v["chips"] == [["low", True], ["9", False]],
          "V  Int: one chip per preset, the current value marked, nameless shows its value")
    check(v["int_calls"] == [["setValue", 9], ["remove", 0], ["add"]] and v["prevented"],
          "V  Int: click selects, right-click removes (no context menu), '+' adds -- the SHARED pieces")
    check(v["hint"] == "HINT" and v["int_ready"] == [True, False],
          "V  Int: the SHARED empty hint; ready only once the node has its chip state")
    check(v["filter"] == [["reset"]] and v["filter_no_row"],
          "V  Filter: the title-bar Reset chip calls the SHARED _pfReset; no widget row (v962)")
    check(v["band"] == ["L1", "L2"] and v["cte_sig"] == "CT|300",
          "V  CLIP Encode: the band shows the SHARED _barLines; signature = counter text + width")
    check(v["load"] == ["STATUS", "#ff8c00"] and v["load_same"],
          "V  Load CLIP/Model/VAE: the SHARED status line with its colour, one spec for all three")
    # public build (v374): no Everywhere view here

    it = src("ph_int.js"); ba = src("ph_basics.js")   # public build (v374): no ph_everywhere.js
    check("addIntPreset(this);" in it and "removeIntPreset(this, z.i);" in it
          and it.count('prompt("Preset name"') == 1, "S  Int's painted chips call the shared pieces; the dialog exists once")
    check(it.count('"name a value"') == 1 and "ctx.fillText(INT_EMPTY_HINT," in it,
          "S  Int's hint text exists once; the painted hint reads it")
    check("let { text, colour } = loadStatusLine(this, nodeData.name);" in ba
          and ba.count("text = \"\\u26a0 \" + NO_MODEL_TEXT;") == 1,
          "S  the Load nodes paint the shared status line; the warning exists once")
    for f, names in (("ph_int.js", ["addIntPreset", "removeIntPreset", "INT_EMPTY_HINT"]),
                     ("ph_filter.js", ["_pfReset"]), ("ph_clip_encode.js", ["_barLines", "_counterText"]),
                     ("ph_basics.js", ["loadStatusLine"])):
        s = src(f)
        check(all(re.search(r"export\s*\{[^}]*\b%s\b" % n, s) for n in names), "S  %s exports %s" % (f, names))
    code = re.sub(r"/\*.*?\*/", "", ex, flags=re.S)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)      # comments are not code
    for bad in ("prompt(", "fillText", "NO_MODEL_TEXT", "feeds on next queue", "setInterval", "vueNodesMode"):
        check(bad not in code, "S  the views rebuild nothing: no %r" % bad)
    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    print("v941: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
