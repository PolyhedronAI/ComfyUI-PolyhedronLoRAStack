# -*- coding: ascii -*-
"""
test_v990_show_text_fit.py -- Polyhedron Show Text fits its text like the CTE.

v990 (22.09.2026): no scrollbars inside the text -- the node widened itself.
v991 RE-GROUNDED (22.09.2026, Frank: "die kann man immer noch nicht sauber
zusammenschieben und die spreizt sich dann in die Breite statt in der Laenge
... schau, wie das bei der Clip Text Encode geloest ist"). The CTE rule
(ph_clip_encode.js v715): the reserved height is computed from node.size[0]
on every call, LiteGraph clamps a manual resize against computeSize, and
there is NO onResize hook (guard #604). So now:

  W  the width is the user's: a run never widens the node, a reload keeps it
  R  the text wraps; the reserved height follows the CURRENT width -- a
     narrower node reserves more height, a wider one less
  O  no scrollbar while the text fits (never sideways), scroll past the cap
  H  the pixels a renderer withholds after layout are added once; an
     unlaid-out field is left alone
  N  no onResize hook, no auto-widening code left

Browser probe (sandbox ComfyUI, classic + Nodes 2.0, real mouse drag of the
corner 450 px left / 400 px up): classic 420x585 -> 210x963, Vue 420x595 ->
225x963, field scrollHeight == clientHeight both times, no bars, reload keeps
the size, no page errors.
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "js"
FAILED = []


def _need(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILED.append(msg)


HARNESS = r"""
import { app } from "./scripts/app.js";
globalThis.requestAnimationFrame = (f) => f();
globalThis.document = { createElement: () => ({ style: {}, classList: { add() {} } }) };
const M = await import("./web/js/ph_show_text.js");
const ext = app._ext.find((e) => e.name === "polyhedron.showtext");
const T = function () {};
T.prototype = {};
await ext.beforeRegisterNodeDef(T, { name: "ULSShowText" });
function make(w) {
    const n = Object.create(T.prototype);
    n.widgets = []; n.properties = {}; n.size = [w || 200, 100];
    n.addDOMWidget = function (name, type, el, opts) {
        const x = { name, type, element: el, options: opts || {} }; this.widgets.push(x); return x; };
    n.addCustomWidget = function (x) { this.widgets.push(x); return x; };
    n.computeSize = function () {
        let h = 30;
        for (const x of this.widgets) h += x.options && x.options.getMinHeight
            ? x.options.getMinHeight() : (x.computeSize ? x.computeSize(this.size[0])[1] : 20);
        return [this.size[0], h];
    };
    n.setSize = function (s) { this.size = [s[0], s[1]]; };
    n.setDirtyCanvas = () => {};
    T.prototype.onNodeCreated.call(n);
    return n;
}
const fld = (n) => n.widgets.find((x) => x.name === "ph_show_text_field");
const run = (n, t) => T.prototype.onExecuted.call(n, { text: [t] });
const table = Array.from({ length: 25 }, (_, i) => "row " + i + " " + "x".repeat(i === 20 ? 130 : 60)).join("\n");
const out = {};

// W
const a = make(320); out.aW0 = a.size[0]; run(a, table); out.aW = a.size[0];
const b = make(1500); run(b, table); out.bW = b.size[0];
const r = make(320); r.properties = { ph_show_text: table }; r.size = [410, 300];
T.prototype.onConfigure.call(r, {}); out.rW = r.size[0];

// R -- the reservation follows the width, on every call, with no hook
const f = fld(a).options.getMinHeight;
a.size[0] = 300; out.h300 = f();
a.size[0] = 900; out.h900 = f();
a.size[0] = 1500; out.h1500 = f();
out.lines = table.split("\n").length;
out.lineH = M.LINE_H;
out.wrap = fld(a).element.wrap; out.ws = fld(a).element.style.whiteSpace;

// O
out.ox = fld(a).element.style.overflowX; out.oy = fld(a).element.style.overflowY;
const huge = make(420); run(huge, Array.from({ length: 400 }, (_, i) => "h" + i).join("\n"));
out.hOy = fld(huge).element.style.overflowY; out.hH = fld(huge).options.getMinHeight();
out.maxH = M.SHOW_MAX_H;

// H
const g = make(420); const ge = fld(g).element; ge.clientHeight = 336; ge.scrollHeight = 360;
run(g, table); out.gCH = g.computeSize()[1]; out.gH = g.size[1];
const z = make(420); const ze = fld(z).element; ze.clientHeight = 0; ze.scrollHeight = 360;
run(z, table); out.zCH = z.computeSize()[1]; out.zH = z.size[1];

// N
out.hasResize = typeof T.prototype.onResize === "function";
console.log(JSON.stringify(out));
"""


def run_harness():
    node = shutil.which("node")
    if node is None:
        return None, "node missing"
    tmp = tempfile.mkdtemp(prefix="v990_")
    try:
        os.makedirs(os.path.join(tmp, "scripts"))
        with open(os.path.join(tmp, "scripts", "app.js"), "w", encoding="utf-8") as fh:
            fh.write("export const app = { _ext: [], canvas: null,"
                     " registerExtension(e) { this._ext.push(e); } };\n")
        work = os.path.join(tmp, "web", "js")
        os.makedirs(work)
        for fn_ in ("ph_show_text.js", "ph_note.js", "ph_widget_vis.js"):
            shutil.copyfile(str(WEB / fn_), os.path.join(work, fn_))
        with open(os.path.join(tmp, "h.mjs"), "w", encoding="utf-8") as fh:
            fh.write(HARNESS)
        rr = subprocess.run([node, "h.mjs"], capture_output=True, text=True, cwd=tmp, timeout=60)
        if rr.returncode != 0:
            return None, rr.stderr[-1500:]
        return json.loads(rr.stdout.strip().splitlines()[-1]), ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


got, err = run_harness()
_need(got is not None, "harness runs the real ph_show_text.js with its real siblings " + err)
if got:
    _need(got["aW"] == got["aW0"] == 320, "W  a run never widens the node (%s)" % got["aW"])
    _need(got["bW"] == 1500, "W  a wide node keeps its width (%s)" % got["bW"])
    _need(got["rW"] == 410, "W  a reload keeps the saved width (%s)" % got["rW"])
    _need(got["wrap"] == "soft" and got["ws"] == "pre-wrap", "R  the text wraps (%s/%s)" % (got["wrap"], got["ws"]))
    _need(got["h300"] > got["h900"] > got["h1500"],
          "R  narrower reserves more height, wider less (%s > %s > %s)"
          % (got["h300"], got["h900"], got["h1500"]))
    _need(got["h1500"] < got["lines"] * got["lineH"] + 40,
          "R  at full width no line wraps -- one row per line (%s)" % got["h1500"])
    _need(got["ox"] == "hidden" and got["oy"] == "hidden",
          "O  a text that fits shows no scrollbar (%s/%s)" % (got["ox"], got["oy"]))
    _need(got["hOy"] == "auto" and got["hH"] == got["maxH"] >= 2000,
          "O  past the cap (%s) the field scrolls" % got["maxH"])
    _need(got["gH"] == got["gCH"] + 24,
          "H  the 24 px the renderer kept are added once (%s = %s + 24)" % (got["gH"], got["gCH"]))
    _need(got["zH"] == got["zCH"], "H  an unlaid-out field is left alone (%s = %s)" % (got["zH"], got["zCH"]))
    _need(got["hasResize"] is False, "N  no onResize hook (guard #604)")

src = (WEB / "ph_show_text.js").read_text(encoding="utf-8")
_need("wantedWidth" not in src and "MAX_AUTO_W" not in src and "ph_show_text_widened" not in src,
      "N  no auto-widening code left")
_need("node.size ? node.size[0]" in src, "R  getMinHeight reads the current width")

print()
if FAILED:
    print("test_v990_show_text_fit: %d failure(s)" % len(FAILED))
    sys.exit(1)
print("test_v990_show_text_fit: 0 failure(s)")
