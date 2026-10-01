#!/usr/bin/env python3
"""v976 -- the Sigma List draws its own curve.

The node showed numbers and a greyed-out field; what the curve actually looked
like was only visible after a run. Now a strip under the widgets plots the grid
the RUN would use, with the step count, the sigma range and the active shift as
a header line.

The curve is computed by the BACKEND, through /pls/sigma_preview, using the
same parse/validate/shift functions the node itself calls. Mirroring the shift
arithmetic into JavaScript would put one formula in two places, and two places
drift -- the standing lesson of this tree. So the frontend asks and never
calculates.

That also buys the real prize: an invalid list comes back as the REFUSAL TEXT
the run would raise, and the strip shows it. A typo is visible in the node
instead of at run time.

  C1  the route is registered and its handler calls resolve_sigma_request --
      the SAME chain compute() runs, not a re-assembly of its parts; and that
      chain CALLS its parts rather than inlining them. (C3/C4 compare the
      resolver with compute(), which since v976 is A against A -- so what
      happens inside the chain is pinned here structurally and, semantically,
      by the published-value checks in test_v972.)
  C2  the JS contains no shift arithmetic of its own
  C3  RUN: a valid custom list resolves to the same grid the node computes
  C4  RUN: a preset resolves to the preset's recipe, whatever the widgets say
  C5  RUN: an invalid list answers ok:false with the refusal text, not a 500 --
      and the text names the offending token
  C6  RE-ARGUED v977: the plot is a custom WIDGET, measured and placed by
      LiteGraph, carrying serialize:false. v976 painted it at the node's lower
      edge, where the widgets actually sit -- painted space is not reserved
      space, and the curve covered the fields in the field
  C7  RUN: the JS asks the preview route on a widget change and stores what it
      gets; a repeated identical body is NOT re-sent
  C8  RUN: an unreachable preview route leaves the node drawing nothing rather
      than throwing
  C9  RUN: the plot's height FOLLOWS the point count -- 20 knots get more room
      than 8 -- and stays within its bounds

MUTATION PROBE (run 20.09.2026, each mutation applied and reverted):
  1. handler recomputes the shift inline instead of the shared chain -> C1 caught
  2. handler raises instead of answering ok:false                    -> C5 caught
  3. preset ignored in the preview handler                           -> C4 caught
  4. the plot widget's serialize:false removed                       -> C6 caught
  7. the plot height made constant (ignores the point count)         -> C9 caught
  5. the body cache dropped (every draw re-fetches)                  -> C7 caught
  6. a shift formula pasted into the JS, named `s` not `shift`       -> C2 caught
     (slipped on the first pass: the check matched a NAME, not the formula's
      shape. Re-argued to match the arithmetic itself.)
  3. preset ignored in the preview handler                           -> C4 caught
     (slipped on the first pass: C3/C4 ran a reimplementation of the handler
      instead of the shipped function. Now they run the real one.)
"""
import os, re, sys, json, shutil, subprocess, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, "web", "js", "ph_sigma_list.js")
ROUTES = os.path.join(ROOT, "nodes", "ph_sigma_routes.py")   # public build: the sigma routes live in their own module (v384)
sys.path.insert(0, ROOT)

failures = []
ran = []


def check(cond, msg):
    ran.append(msg)
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


js_src = open(JS, encoding="utf-8").read()
# Structure checks must read CODE, not prose: the file DESCRIBES the v976
# construction it replaced, and a check that greps the raw text fires on the
# explanation. (A substring is not a proof -- and prose is not a substring of
# the program.)
js_code = re.sub(r"^\s*//.*$", "", js_src, flags=re.M)
js_code = re.sub(r"/\*(?:.|\n)*?\*/", "", js_code)
routes_src = open(ROUTES, encoding="utf-8").read()
NODE = shutil.which("node")

# --- C1 one implementation, reused -----------------------------------------
check('"/pls/sigma_preview"' in routes_src, "C1 the route /pls/sigma_preview is registered")
m = re.search(r"async def handle_sigma_preview_post.*?(?=\nasync def |\ndef )",
              routes_src, re.S)
check(m is not None, "C1 the preview handler exists")
if m:
    body = m.group(0)
    check("resolve_sigma_request" in body,
          "C1 the handler calls resolve_sigma_request -- the node's own chain")
    for fn in ("parse_sigma_list", "validate_sigma_list", "shift_sigma_list"):
        check(fn not in body,
              "C1 the handler does NOT re-assemble the chain itself (%s)" % fn)
    # the shift formula must NOT appear a second time here
    check(not re.search(r"1\.0\s*\+\s*\(\s*\w*shift\w*\s*-\s*1", body),
          "C1 the handler carries no shift formula of its own")

# --- C1b the shared chain is a CHAIN, not a second implementation ----------
# C3/C4 below compare the resolver against compute(). Since v976 both call the
# same function, that comparison can no longer catch a change INSIDE it -- it
# would compare A with A. Two things cover that instead: the structural promise
# here, and the published-value checks in test_v972 (P5), which are independent
# of our code entirely.
sched_src = open(os.path.join(ROOT, "nodes", "wan_sigma_schedule.py"),
                 encoding="utf-8").read()
chain = re.search(r"def resolve_sigma_request.*?(?=\ndef |\nclass )", sched_src, re.S)
check(chain is not None, "C1 resolve_sigma_request exists")
if chain:
    cb = chain.group(0)
    for fn in ("preset_entry", "parse_sigma_list", "validate_sigma_list",
               "shift_sigma_list"):
        check(fn + "(" in cb, "C1 the chain CALLS %s rather than inlining it" % fn)
    check(not re.search(r"/\s*\(\s*1(\.0)?\s*\+", cb),
          "C1 the chain holds no shift formula of its own")

# --- C2 the JS does no sigma arithmetic ------------------------------------
# The formula's SIGNATURE, not a variable name: v976's probe slipped a copy
# through by calling the shift `s` instead of `shift`.
check(not re.search(r"/\s*\(\s*1(\.0)?\s*\+", js_code),
      "C2 the JS holds no shift formula (x / (1 + ...) in any spelling)")
check(not re.search(r"-\s*1(\.0)?\s*\)\s*\*\s*\w+", js_code),
      "C2 the JS holds no (s - 1) * v term either")
check(PREVIEW := ("/pls/sigma_preview" in js_src), "C2 the JS asks the preview route")

# --- C3..C5 the resolver, RUN against the node itself ----------------------
# The REAL function the handler calls -- not a copy of it. A guard that runs
# its own reimplementation proves nothing about the thing that ships: the v976
# probe ignored the preset in the handler and this check stayed green until it
# was pointed at the real chain.
from nodes.wan_sigma_schedule import ULSSigmaList, resolve_sigma_request


def resolve(sigmas_text, shift, zero, preset):
    return resolve_sigma_request(sigmas_text, shift, zero, preset)[0]


N = ULSSigmaList()
sig, steps = N.compute("1.0, 0.8, 0.5, 0.0", 3.0, True, "custom")
got = resolve("1.0, 0.8, 0.5, 0.0", 3.0, True, "custom")
check(len(got) == len(sig) and all(abs(float(a) - b) < 1e-6 for a, b in zip(sig, got)),
      "C3 the preview resolves a custom list to the grid the node computes")

sig, steps = N.compute("9, 8, 7", 1.0, False, "hyperflow_8step_h3_raw")
got = resolve("9, 8, 7", 1.0, False, "hyperflow_8step_h3_raw")
check(len(got) == len(sig) and all(abs(float(a) - b) < 1e-6 for a, b in zip(sig, got)),
      "C4 the preview follows the preset, not the widgets")

err = None
try:
    resolve("1.0, zero.eight, 0.0", 1.0, True, "custom")
except Exception as e:
    err = str(e)
check(err is not None and "zero.eight" in err,
      "C5 an invalid list yields the refusal text, naming the token: %r" % (err or "")[:60])
check("web.json_response({\"ok\": False, \"error\": str(e)})" in (m.group(0) if m else ""),
      "C5 the handler ANSWERS the refusal instead of returning a 500")

# --- C6 RE-ARGUED in v977: the plot is a WIDGET ----------------------------
# v976 reserved height in computeSize and painted at `node.size[1] - PLOT_H`.
# In the field the curve landed ON the widgets: the multiline field stretches
# into whatever height the node gains, so that lower edge is not empty space.
# Painted space is not reserved space. The promise is now the stronger one --
# the plot is a custom widget, measured and placed by LiteGraph, so it cannot
# cover a field and cannot be covered by one.
check("addCustomWidget" in js_src or "_slPlot" in js_src,
      "C6 the plot exists as a widget, not as paint on the node's edge")
check("onDrawForeground" not in js_code,
      "C6 nothing is painted onto the node outside the widget's own rect")
check(not re.search(r"node\.size\[1\]\s*-\s*PLOT", js_code),
      "C6 the plot never positions itself from node.size")
check(re.search(r"serialize:\s*false", js_code) is not None,
      "C6 the plot widget declares serialize:false -- it must never reach "
      "widgets_values, or every saved workflow renumbers (guard #577)")
check("computeSize(width)" in js_code,
      "C6 the widget reports its own height, so LiteGraph reserves the space")
# v984: the bounds moved with the drawing into the shared ph_sigma_plot.js
# (one plot for all sigma nodes). The list must take its height from there.
_plot_src = open(os.path.join(ROOT, "web", "js", "ph_sigma_plot.js"), encoding="utf-8").read()
check(re.search(r"PLOT_MIN_H|PLOT_MAX_H", js_code) is not None
      or ("plotHeightFor" in js_code
          and re.search(r"Math\.min\(PLOT_MAX_H", _plot_src) is not None),
      "C6 the height is bounded -- a 60-point grid cannot grow the node forever")

# --- C7, C8 the JS, RUN ----------------------------------------------------
if NODE is None:
    check(False, "C7 node is available to run the JS harness")
else:
    tmp = tempfile.mkdtemp(prefix="v976_")
    os.makedirs(os.path.join(tmp, "scripts"), exist_ok=True)
    with open(os.path.join(tmp, "scripts", "app.js"), "w", encoding="utf-8") as f:
        f.write("export const app = { _ext: null,"
                " registerExtension(e) { this._ext = e; } };\n")
    work = os.path.join(tmp, "web", "js")
    os.makedirs(work, exist_ok=True)
    shutil.copyfile(JS, os.path.join(work, "ph_sigma_list.js"))
    # ph_sigma_list.js imports refit from its sibling. A harness that strips or
    # stubs an import measures a file the pack does not ship -- the standing
    # merkposten since v737. Put the REAL sibling beside it.
    shutil.copyfile(os.path.join(ROOT, "web", "js", "ph_widget_vis.js"),
                    os.path.join(work, "ph_widget_vis.js"))
    # v984: and the shared plot module it draws with -- the real one.
    shutil.copyfile(os.path.join(ROOT, "web", "js", "ph_sigma_plot.js"),
                    os.path.join(work, "ph_sigma_plot.js"))

    harness = r"""
import { app } from "./scripts/app.js";
const MODE = process.argv[2] || "ok";
const calls = [];
const grid = (n) => Array.from({ length: n }, (_, i) => 1 - i / (n - 1));
globalThis.fetch = async (url, opts) => {
    if (String(url).includes("sigma_preview")) {
        calls.push(opts ? opts.body : null);
        if (MODE === "down") throw new Error("unreachable");
        const g = MODE === "wide" ? grid(20) : [1, 0.5, 0];
        return { ok: true, json: async () => ({
            ok: true, sigmas: g, steps: g.length - 1, shift: 1, source: "text", note: "" }) };
    }
    return { ok: true, json: async () => ({ ok: true, presets: {
        custom: { grid: null, shift: null, enforce_terminal_zero: null, note: "" } } }) };
};

await import("./web/js/ph_sigma_list.js");
const ext = app._ext;
const nodeType = function () {};
nodeType.prototype = { computeSize: () => [300, 200] };
await ext.beforeRegisterNodeDef(nodeType, { name: "ULSSigmaList" });

const node = Object.create(nodeType.prototype);
node.widgets = [
    { name: "sigmas_text", value: "1.0, 0.5, 0.0",
      inputEl: { value: "1.0, 0.5, 0.0", readOnly: false, style: {} } },
    { name: "shift", value: 1.0, callback: null },
    { name: "enforce_terminal_zero", value: true },
    { name: "preset", value: "custom", callback: null },
];
node.properties = {};
node.size = [300, 200];
node.flags = {};
node.setDirtyCanvas = () => {};
nodeType.prototype.onNodeCreated.call(node);
const settle = () => new Promise((r) => setTimeout(r, 200));
await settle();

const before = calls.length;
// same values again -> the body is identical -> must NOT be re-sent
const sw = node.widgets.find((w) => w.name === "shift");
await sw.callback(1.0); await settle();
const afterSame = calls.length;
sw.value = 7.0; await sw.callback(7.0); await settle();
const afterChange = calls.length;

// draw into a stub ctx: must not throw, with or without a preview
const ops = [];
const ctx = new Proxy({}, {
    get: (t, k) => {
        if (k === "measureText") return () => ({ width: 10 });
        if (k === "save" || k === "restore") return () => ops.push(k);
        if (typeof k === "string") return (...a) => ops.push(k);
        return undefined;
    },
    set: () => true,
});
let threw = null;
const plotW = node.widgets.find((w) => w.type === "poly_sigma_curve");
try { plotW.draw(ctx, node, 300, 24); }
catch (e) { threw = String(e); }

// height as a function of the point count, measured through the real widget
const plot = node.widgets.find((w) => w.type === "poly_sigma_curve");
const at = (n) => {
    node._slPreview = { ok: true, sigmas: grid(n), steps: n - 1, shift: 1 };
    return plot.computeSize(300)[1];
};
// Measure, then put the node back exactly as it was: the checks after this
// read _slPreview, and a probe that leaves its own fixture behind measures
// itself. (Copy state, never borrow it -- the v750 harness lesson.)
const savedPreview = node._slPreview;
const plotH9 = at(9), plotH20 = at(20), plotH200 = at(200);
node._slPreview = savedPreview;

console.log(JSON.stringify({
    before, afterSame, afterChange,
    plotH9, plotH20, plotH200, serialize: plot.serialize,
    preview: node._slPreview ? node._slPreview.ok : null,
    drew: ops.length, threw,
}));
"""
    hp = os.path.join(tmp, "harness.mjs")
    with open(hp, "w", encoding="utf-8") as f:
        f.write(harness)

    def run(mode):
        rr = subprocess.run([NODE, hp, mode], capture_output=True, text=True, cwd=tmp)
        if rr.returncode != 0:
            print(rr.stderr[-900:])
            return None
        return json.loads(rr.stdout.strip().splitlines()[-1])

    got = run("ok")
    check(got is not None, "C7 the harness runs the real file")
    if got:
        check(got["before"] >= 1, "C7 the node asks the preview route on creation")
        check(got["afterSame"] == got["before"],
              "C7 an identical body is NOT re-sent (%d -> %d)"
              % (got["before"], got["afterSame"]))
        check(got["afterChange"] > got["afterSame"],
              "C7 a changed widget DOES trigger a new request")
        check(got["preview"] is True, "C7 the answer is stored on the node")
        check(got["threw"] is None and got["drew"] > 0,
              "C7 drawing runs without throwing (%s)" % got["threw"])

    got = run("wide")
    check(got is not None, "C9 the harness runs with a 20-point grid")
    if got:
        check(got["plotH20"] > got["plotH9"],
              "C9 a 20-point grid gets a taller plot than a 9-point one (%s vs %s)"
              % (got["plotH20"], got["plotH9"]))
        check(got["plotH200"] <= 260,
              "C9 the plot height is capped (%s)" % got["plotH200"])
        check(got["serialize"] is False,
              "C9 the plot widget carries serialize:false at runtime")

    got = run("down")
    check(got is not None, "C8 the harness runs with the preview route unreachable")
    if got:
        check(got["threw"] is None,
              "C8 an unreachable preview route does not throw while drawing")
        check(got["preview"] is None,
              "C8 no stale curve is kept when the route cannot be reached")

    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print("v976 sigma curve: FAIL (%d)" % len(failures))
    for f in failures:
        print("   - " + f)
    sys.exit(1)
print("%d checks, 0 failed" % len(ran))
print("v976 sigma curve: PASS")
