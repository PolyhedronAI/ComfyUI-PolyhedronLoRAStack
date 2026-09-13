#!/usr/bin/env python3
"""v959 (public v376) -- the status line that only shows AFTER a run, under Nodes 2.0.

Public build: of the three after-run parts of the internal v959, only
AnySwitch/-Inv ship here (Reference and Vectorize are internal nodes; the
Cutout heal likewise). The promise is the same: the canvas and the view read
ONE function, switchStatusLine.

Three nodes paint something in onDrawForeground that no parity sheet could see,
because it appears only after a run or a wiring: AnySwitch/-Inv's status line at
the foot, Reference's token band, Vectorize's swatch beside fill_color. Nodes 2.0
never calls onDrawForeground, so those parts were simply absent there. v959 gives
each a view in web/js/uls_extras_dom.js; the canvas and the view read the SAME
function (switchStatusLine / bandLinesOf / validateHex), so they cannot drift.
Fourth item: the Cutout model rows heal a value that carries a size the list
does not offer (" \\u00b7 309 MB" saved on a machine that has the file) onto the
entry the list offers -- the JS twin of ph_cutout._undecorate(); Nodes 2.0 framed
such a value red.

  V  the three views, driven with recording stubs
  S  the canvas paints through the same reader the view uses
"""
import json, os, re, shutil, subprocess, sys, tempfile, importlib.util

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

DOM = r"""
function mkEl(tag) { const el = { tagName: String(tag).toUpperCase(), className: "", title: "", children: [], _text: "", style: {},
  appendChild(c) { this.children.push(c); return c; } };
  Object.defineProperty(el, "textContent", { get() { return el._text; }, set(v) { el._text = v; if (v === "") el.children = []; } });
  return el; }
globalThis.document = { createElement: mkEl, head: mkEl("head") };
function walk(el, f) { f(el); for (const c of el.children || []) walk(c, f); }
function find(root, cls) { const o = []; walk(root, e => { if ((" " + e.className + " ").includes(" " + cls + " ")) o.push(e); }); return o; }
"""
STUBS = r"""
let SPECS = {};
const registerVueView = (cls, spec) => { SPECS[cls] = spec; };
const getValue = () => 0, setValue = () => {}, addIntPreset = () => {}, removeIntPreset = () => {}, INT_EMPTY_HINT = "";
const _pfReset = () => {}, _barLines = () => [], _counterText = () => "", loadStatusLine = () => ({text:"",colour:""}), fedLine = () => "";
let STATUS = "\u2192 A (image)";
const switchStatusLine = (n) => STATUS;
let LINES = ["image_1  -- run to measure", "total -- run to measure"];
const bandLinesOf = (n) => LINES;
const ROW_COLOR = "#cfe7ff", SUM_COLOR = "#ff8c00";
const validateHex = (v) => (/^#[0-9a-f]{6}$/i.test(String(v)) ? String(v).toLowerCase() : null);
let FILL = { name: "fill_color", value: "#00ff00" };
const fillWidget = (n) => FILL;
"""
TAIL = r"""
const out = {};
out.classes = Object.keys(SPECS).sort();
const r = mkEl("div");
SPECS.ULSAnySwitch.render({}, r); out.sw = find(r, "uls-x-line")[0].textContent;
out.sw_sig = [SPECS.ULSAnySwitch.signature({}), (STATUS = "x", SPECS.ULSAnySwitchInv.signature({}))];
out.sw_same = SPECS.ULSAnySwitch.widgetName === SPECS.ULSAnySwitchInv.widgetName && SPECS.ULSAnySwitch.render === SPECS.ULSAnySwitchInv.render;
out.stores = Object.values(SPECS).every(s => !("serialize" in s));
console.error(JSON.stringify(out));
"""

CUT_TAIL = r"""
_pcDir = { selector: [ { entry: "\u25c8 sam2.1 base_plus", short: "sam2.1 base_plus", name: "sam2.1 base_plus", file: "sam2.1_base_plus.pt", is_default: true },
                       { entry: "\u25c8 sam2.1 tiny", short: "sam2.1 tiny", name: "sam2.1 tiny", file: "t.pt" } ] };
const out = { heal: (_pcEntry("selector", "\u25c8 sam2.1 base_plus \u00b7 309 MB") || {}).entry,
              exact: (_pcEntry("selector", "\u25c8 sam2.1 tiny") || {}).entry,
              none: _pcEntry("selector", "nothing like it"),
              und: INPUTS.map(_pcUndecorate) };
console.error(JSON.stringify(out));
"""
INPUTS = ["\u25c8 sam2.1 base_plus \u00b7 309 MB", "\u2713 BiRefNet general - 844 MB", "  plain  ",
          "\u2717 x  two spaces", "\u25c8 sam2.1 tiny", "", "a \u00b7 b \u00b7 c"]


def main():
    print("v959 (public): the after-run status line under Nodes 2.0")
    if shutil.which("node") is None:
        print("  FAIL node (Node.js) not found -- this guard needs it")
        return 1
    ex = src("uls_extras_dom.js")
    body = re.sub(r'^import[^;]*;\s*$', "", ex, flags=re.M)
    v, err = run_node(DOM + STUBS + body + TAIL)
    if v is None:
        check(False, "views harness ran: " + err)
        return 1
    check(all(c in v["classes"] for c in ("ULSAnySwitch", "ULSAnySwitchInv")),
          "V  both switches register a view (%s)" % v["classes"])
    check(v["sw"] == "\u2192 A (image)" and v["sw_sig"] == ["\u2192 A (image)", "x"] and v["sw_same"],
          "V  Switch: the view shows the SHARED status line, signature follows it, one spec for both switches")
    check(v["stores"], "V  no view stores anything")

    sw = src("ph_switch.js")
    check("const status = switchStatusLine(this);" in sw and "ctx.fillText(status, 8," in sw
          and sw.count("export function switchStatusLine") == 1,
          "S  the painted switch reads the same status line the view reads")
    code = re.sub(r"/\*.*?\*/", "", ex, flags=re.S); code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
    for bad in ("fillText", "_plsStatus", "vueNodesMode"):
        check(bad not in code, "S  the views rebuild nothing: no %r" % bad)

    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    print("v959: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
