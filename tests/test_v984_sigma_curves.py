#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v984 -- the Sigma Curve and the Dual Sigma Curve draw the curve they output.

Frank, 20.09.2026: the Sigma List has had its curve since v976; the two curve
nodes had none. Before the plot, the nodes were cleaned up: their math moved
out of compute() into pure functions (universal_curve, split_curves) that
compute() and the new preview route BOTH call, and the texts that promised
outputs these nodes never had (a `steps` pass-through, a sliced LOW list)
were corrected. The deprecated ULSWanSigmaSchedule is left alone.

  S1  RUN: the move changed nothing -- 496 parameter combinations through
      both nodes' compute() hash to the value the v983 module produced
      (a8ccd14b..., measured against the shipped v983 file on 21.09.2026)
  S2  RUN: the preview is the run -- curve_preview() returns exactly what
      compute() outputs; compute() calls the pure functions, it does not
      recompute
  S3  RUN: the preview is safe to call with anything -- an unknown schedule
      answers ok:false with the name, a split beyond the range is clamped the
      way the node clamps it, garbage numbers fall back, steps are bounded by
      the widget's own maximum
  S4  the interface did not move: RETURN_TYPES and INPUT_TYPES names/order
      unchanged (saved workflows serialise widgets BY INDEX, guard #577), and
      the texts no longer promise outputs that do not exist
  S5  the route is registered and its handler calls curve_preview
  S6  the JS computes no sigma, asks the route, and draws through the ONE
      plot module (ph_sigma_plot.js) -- the Sigma List too; the deprecated
      node gets no plot
  S7  RUN (node): the real ph_sigma_curves.js with its real siblings. Asks on
      creation with every widget, does not re-send an identical body, grows
      with the knots, never serialises the plot, draws curve/dual/refusal/
      nothing without throwing, marks HIGH 0..split and LOW split..end, warns
      when the split was clamped, and under Nodes 2.0 hands the new height to
      the frontend before asking it to redraw (triggerDraw + computedHeight --
      found in the v984 browser probe: without it the plot said "no preview"
      forever, then stayed clipped at its first height)

MUTATION PROBE (21.09.2026, each applied and reverted; see CHANGELOG_v984):
  listed in CHANGELOG_v984.md with the check that caught each.
"""

import hashlib, io, contextlib, json, os, re, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
WEB = os.path.join(ROOT, "web", "js")

failures = []
ran = []


def check(cond, msg):
    ran.append(msg)
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def code_of(path):
    s = open(path, encoding="utf-8").read()
    c = re.sub(r"^\s*//.*$", "", s, flags=re.M)
    c = re.sub(r"/\*(?:.|\n)*?\*/", "", c)
    c = re.sub(r"\s//[^\n\"'`]*$", "", c, flags=re.M)
    return s, c


import nodes.wan_sigma_schedule as W

# --- S1 bit-identical to v983 ---------------------------------------------
REF_N, REF_MD5 = 496, "a8ccd14bebb1e9f8b01305c8c19b3198"


def digest():
    h = hashlib.md5()
    n = 0
    S = W.SIGMA_SCHEDULE_NAMES
    with contextlib.redirect_stdout(io.StringIO()):
        for s in S:
            for steps in (1, 2, 7, 20, 41):
                for (mx, mn) in ((1.0, 0.002), (14.61, 0.029), (0.5, 0.9)):
                    for rho in (1.0, 7.0):
                        (sig,) = W.ULSUniversalSigmaCurve().compute(s, steps, mx, mn, rho)
                        h.update(sig.numpy().tobytes())
                        n += 1
        for i, sh in enumerate(S):
            for sl in S:
                for (tot, sp) in ((2, 1), (20, 8), (30, 29), (12, 40)):
                    mx, mn = ((1.0, 0.002), (14.61, 0.029), (0.3, 0.8))[(i + tot) % 3]
                    a, b = W.ULSWanSplitNoiseSchedule().compute(sh, sl, tot, sp, mx, mn, 7.0, 3.0)
                    h.update(a.numpy().tobytes())
                    h.update(b.numpy().tobytes())
                    n += 1
    return n, h.hexdigest()


n, md5 = digest()
check(n == REF_N and md5 == REF_MD5,
      "S1 %d combos through both nodes are bit-identical to v983 (%s)" % (n, md5[:8]))

# --- S2 the preview IS the run --------------------------------------------
same = True
with contextlib.redirect_stdout(io.StringIO()):
    for s in W.SIGMA_SCHEDULE_NAMES:
        (sig,) = W.ULSUniversalSigmaCurve().compute(s, 17, 1.0, 0.002, 5.0)
        p = W.curve_preview({"node": "curve", "sigma_schedule": s, "steps": 17,
                             "sigma_max": 1.0, "sigma_min": 0.002, "rho": 5.0})
        same &= p["ok"] and p["sigmas"] == [float(x) for x in sig]
        a, b = W.ULSWanSplitNoiseSchedule().compute(s, "karras", 24, 9, 14.61, 0.029, 7.0, 2.0)
        p = W.curve_preview({"node": "dual", "schedule_high": s, "schedule_low": "karras",
                             "total_steps": 24, "split_step": 9, "sigma_max": 14.61,
                             "sigma_min": 0.029, "rho_high": 7.0, "rho_low": 2.0})
        same &= (p["ok"] and p["sigmas_high"] == [float(x) for x in a]
                 and p["sigmas_low"] == [float(x) for x in b] and p["split_step"] == 9)
check(same, "S2 curve_preview returns exactly what compute() outputs (both nodes, every schedule)")

src_py = open(os.path.join(ROOT, "nodes", "wan_sigma_schedule.py"), encoding="utf-8").read()


def method_body(cls, meth):
    m = re.search(r"class %s\b.*?\n    def %s\(.*?(?=\n    def |\n    @|\nclass |\ndef |\Z)"
                  % (cls, meth), src_py, re.S)
    return m.group(0) if m else ""


cu = method_body("ULSUniversalSigmaCurve", "compute")
du = method_body("ULSWanSplitNoiseSchedule", "compute")
check("universal_curve(" in cu and "_compute_raw" not in cu,
      "S2 the Sigma Curve's compute() calls universal_curve, it does not recompute")
check("split_curves(" in du and "_compute_raw" not in du and "handoff =" not in du,
      "S2 the Dual Sigma Curve's compute() calls split_curves, it does not recompute")
pv = re.search(r"def curve_preview.*?(?=\nclass |\ndef |\Z)", src_py, re.S)
pv = pv.group(0) if pv else ""
check("split_curves(" in pv and "universal_curve(" in pv and "_compute_raw" not in pv,
      "S2 curve_preview calls the same two functions")

# --- S3 safe with anything -------------------------------------------------
p = W.curve_preview({"node": "curve", "sigma_schedule": "nope"})
check(p["ok"] is False and "nope" in p["error"], "S3 an unknown schedule is refused by name")
p = W.curve_preview({"node": "sideways"})
check(p["ok"] is False, "S3 an unknown node kind is refused")
p = W.curve_preview({"node": "dual", "total_steps": 20, "split_step": 500})
check(p["ok"] and p["split_step"] == 19 and p["sigmas_high"][19] == p["sigmas_low"][19],
      "S3 a split beyond the range is clamped as the node clamps it, handoff exact")
p = W.curve_preview({"node": "curve", "steps": "abc", "sigma_max": None})
check(p["ok"] and len(p["sigmas"]) == 21, "S3 garbage numbers fall back to the widget defaults")
p = W.curve_preview({"node": "curve", "steps": 5000})
check(p["ok"] and len(p["sigmas"]) == 301, "S3 steps are bounded by the widget's maximum (300)")
p = W.curve_preview({"node": "curve", "sigma_max": 0.1, "sigma_min": 0.9})
check(p["ok"] and p["swapped"] is True, "S3 a swapped range is reported, as the node reports it")
p = W.curve_preview(None)
check(p["ok"] is True and len(p["sigmas"]) == 21, "S3 no body at all draws the default curve")

# --- S4 interface unchanged, texts honest -----------------------------------
U, D = W.ULSUniversalSigmaCurve, W.ULSWanSplitNoiseSchedule
check(U.RETURN_TYPES == ("SIGMAS",) and D.RETURN_TYPES == ("SIGMAS", "SIGMAS"),
      "S4 outputs unchanged")
check(list(U.INPUT_TYPES()["required"]) == ["sigma_schedule", "steps", "sigma_max", "sigma_min", "rho"]
      and "optional" not in U.INPUT_TYPES(),
      "S4 Sigma Curve widgets unchanged in name and order")
check(list(D.INPUT_TYPES()["required"]) == ["schedule_high", "schedule_low", "total_steps",
                                             "split_step", "sigma_max", "sigma_min",
                                             "rho_high", "rho_low"]
      and "optional" not in D.INPUT_TYPES(),
      "S4 Dual Sigma Curve widgets unchanged in name and order")
txt_u = (U.__doc__ or "") + U.DESCRIPTION + U.INPUT_TYPES()["required"]["steps"][1]["tooltip"]
txt_d = (D.__doc__ or "") + D.DESCRIPTION
check("steps + 1" in txt_u and "never had" in (U.__doc__ or ""),
      "S4 the Sigma Curve says what it outputs (steps + 1 values) and no pass-through")
check("FULL-length" in (D.__doc__ or "") and "full-length" in D.DESCRIPTION,
      "S4 the Dual Sigma Curve says both outputs are full-length lists")

# --- S5 route ------------------------------------------------------------------
routes = open(os.path.join(ROOT, "nodes", "ph_sigma_routes.py"), encoding="utf-8").read()   # public build: the sigma routes live in their own module (v384)
check('("POST", "/pls/sigma_curve_preview", handle_sigma_curve_preview_post)' in routes,
      "S5 the route is registered")
hm = re.search(r"async def handle_sigma_curve_preview_post.*?(?=\nasync def |\ndef )", routes, re.S)
hb = hm.group(0) if hm else ""
hb = re.sub(r'"""(?:.|\n)*?"""', "", hb)   # code, not the docstring that names the chain
check("curve_preview(data)" in hb and "_compute_raw" not in hb and "split_curves" not in hb,
      "S5 the handler calls curve_preview, nothing of its own")

# --- S6 JS structure -----------------------------------------------------------
cur_src, cur = code_of(os.path.join(WEB, "ph_sigma_curves.js"))
plot_src, plot = code_of(os.path.join(WEB, "ph_sigma_plot.js"))
lst_src, lst = code_of(os.path.join(WEB, "ph_sigma_list.js"))
check(not re.search(r"Math\.(pow|exp|log|tan|atan|cos|sin|sqrt)\b", cur)
      and not re.search(r"\*\*", cur),
      "S6 ph_sigma_curves.js computes no sigma of its own")
check("/pls/sigma_curve_preview" in cur, "S6 the curve nodes ask the preview route")
check("ULSWanSigmaSchedule" not in cur, "S6 the deprecated node gets no plot")
check(".arc(" not in cur and ".arc(" not in lst and ".arc(" in plot,
      "S6 drawing lives in ph_sigma_plot.js only -- the list draws through it too")
check('from "./ph_sigma_plot.js"' in cur and 'from "./ph_sigma_plot.js"' in lst,
      "S6 both nodes import the shared plot module")
check(re.search(r"serialize:\s*false", cur) is not None and "onDrawForeground" not in cur,
      "S6 the plot is a serialize:false widget, not paint on the node")
check("setSize(" not in cur and "refit(node)" in cur,
      "S6 the height is re-measured with refit(), never setSize(computeSize())")
check("triggerDraw" in lst and "computedHeight" in lst,
      "S6 the Sigma List redraws under Nodes 2.0 too (same probe finding)")

# --- S7 the JS, RUN ---------------------------------------------------------
NODE = shutil.which("node")
if NODE is None:
    check(False, "S7 node is available to run the JS harness")
else:
    tmp = tempfile.mkdtemp(prefix="v984_")
    os.makedirs(os.path.join(tmp, "scripts"), exist_ok=True)
    with open(os.path.join(tmp, "scripts", "app.js"), "w", encoding="utf-8") as f:
        f.write("export const app = { _ext: null,"
                " registerExtension(e) { this._ext = e; } };\n")
    work = os.path.join(tmp, "web", "js")
    os.makedirs(work, exist_ok=True)
    # The REAL files and their REAL siblings -- never a stubbed import.
    for fn in ("ph_sigma_curves.js", "ph_sigma_plot.js", "ph_widget_vis.js"):
        shutil.copyfile(os.path.join(WEB, fn), os.path.join(work, fn))
    # The answers the backend really gives, for the harness's fetch stub.
    answers = {
        "curve": W.curve_preview({"node": "curve", "steps": 8}),
        "curve40": W.curve_preview({"node": "curve", "steps": 40}),
        "dual": W.curve_preview({"node": "dual", "total_steps": 30, "split_step": 12}),
        "dualclamp": W.curve_preview({"node": "dual", "total_steps": 20, "split_step": 50}),
    }
    with open(os.path.join(tmp, "answers.json"), "w", encoding="utf-8") as f:
        json.dump(answers, f)

    harness = r"""
import { app } from "./scripts/app.js";
import fs from "node:fs";
const A = JSON.parse(fs.readFileSync("./answers.json", "utf8"));
const MODE = process.argv[2] || "ok";
const calls = [];
globalThis.fetch = async (url, opts) => {
    if (!String(url).includes("sigma_curve_preview")) throw new Error("wrong route " + url);
    const req = JSON.parse(opts.body);
    calls.push(req);
    if (MODE === "down") throw new Error("unreachable");
    let a;
    if (req.node === "dual") a = Number(req.split_step) > 29 ? A.dualclamp : A.dual;
    else a = Number(req.steps) >= 40 ? A.curve40 : A.curve;
    return { ok: true, json: async () => JSON.parse(JSON.stringify(a)) };
};
const mod = await import("./web/js/ph_sigma_curves.js");
const ext = app._ext;
const settle = () => new Promise((r) => setTimeout(r, 200));

async function make(cls, widgets) {
    const nodeType = function () {};
    nodeType.prototype = { computeSize: () => [300, 200] };
    await ext.beforeRegisterNodeDef(nodeType, { name: cls });
    const node = Object.create(nodeType.prototype);
    node.widgets = widgets.map(([name, value]) => ({ name, value, callback: null }));
    node.properties = {}; node.size = [300, 200]; node.flags = {};
    node.setDirtyCanvas = () => {};
    if (nodeType.prototype.onNodeCreated) nodeType.prototype.onNodeCreated.call(node);
    return node;
}
const ops = [];
const ctx = new Proxy({}, {
    get: (t, k) => {
        if (k === "measureText") return () => ({ width: 10 });
        if (typeof k === "string") return (...a) => ops.push(k);
        return undefined;
    },
    set: () => true,
});
function tryDraw(node) {
    const w = node.widgets.find((x) => x.type === "poly_sigma_curve");
    try { w.draw(ctx, node, 300, 24); return null; } catch (e) { return String(e); }
}

const cn = await make("ULSUniversalSigmaCurve", [["sigma_schedule", "karras"], ["steps", 8],
    ["sigma_max", 1.0], ["sigma_min", 0.002], ["rho", 7.0]]);
// a Nodes 2.0 binding: the frontend installs triggerDraw on the widget
const plotC = cn.widgets.find((x) => x.type === "poly_sigma_curve");
let vueDraws = 0, vueH = null;
plotC.triggerDraw = () => { vueDraws++; vueH = plotC.computedHeight; };
await settle();
const first = calls.length;
const hSmall = plotC.computeSize(300)[1];
const st = cn.widgets.find((w) => w.name === "steps");
st.callback(8); await settle();
const same = calls.length;
st.value = 40; st.callback(40); await settle();
const changed = calls.length;
const hBig = plotC.computeSize(300)[1];
const drawC = tryDraw(cn);

const dn = await make("ULSWanSplitNoiseSchedule", [["schedule_high", "karras"],
    ["schedule_low", "bong_tangent"], ["total_steps", 30], ["split_step", 12],
    ["sigma_max", 1.0], ["sigma_min", 0.002], ["rho_high", 7.0], ["rho_low", 7.0]]);
await settle();
const dreq = calls[calls.length - 1];
const spec = mod.specFor(dn._scPreview, dn._scAsked);
const drawD = tryDraw(dn);
const sp = dn.widgets.find((w) => w.name === "split_step");
sp.value = 50; sp.callback(50); await settle();
const specClamp = mod.specFor(dn._scPreview, dn._scAsked);

const errSpec = mod.specFor({ ok: false, error: "unknown schedule 'x'" }, {});
cn._scPreview = { ok: false, error: "unknown schedule 'x'" };
const drawErr = tryDraw(cn);
cn._scPreview = null;
const drawNone = tryDraw(cn);

const dep = await make("ULSWanSigmaSchedule", [["steps", 20]]);

console.log(JSON.stringify({
    mode: MODE, first, same, changed, hSmall, hBig,
    req0: calls[0] || null, dreq: dreq || null,
    serialize: plotC.serialize, optSerialize: plotC.options.serialize,
    preview: cn._scPreview === null ? null : true,
    drawC, drawD, drawErr, drawNone,
    vueDraws, vueH,
    series: spec.series ? spec.series.map((s) => [s.label, s.from, s.to, s.sigmas.length]) : null,
    marker: spec.marker ? spec.marker.index : null,
    warnClamp: specClamp.warn || "", markerClamp: specClamp.marker ? specClamp.marker.index : null,
    errState: errSpec.state,
    depPlot: dep.widgets.some((x) => x.type === "poly_sigma_curve"),
    drew: ops.length,
}));
"""
    hp = os.path.join(tmp, "harness.mjs")
    with open(hp, "w", encoding="utf-8") as f:
        f.write(harness)

    def run(mode):
        rr = subprocess.run([NODE, hp, mode], capture_output=True, text=True, cwd=tmp)
        if rr.returncode != 0:
            print(rr.stderr[-1200:])
            return None
        return json.loads(rr.stdout.strip().splitlines()[-1])

    got = run("ok")
    check(got is not None, "S7 the harness runs the real files")
    if got:
        r0 = got["req0"] or {}
        check(got["first"] >= 1 and r0.get("node") == "curve"
              and all(k in r0 for k in ("sigma_schedule", "steps", "sigma_max", "sigma_min", "rho")),
              "S7 the Sigma Curve asks on creation, sending every widget")
        dq = got["dreq"] or {}
        check(dq.get("node") == "dual" and all(k in dq for k in (
            "schedule_high", "schedule_low", "total_steps", "split_step",
            "sigma_max", "sigma_min", "rho_high", "rho_low")),
              "S7 the Dual Sigma Curve asks with all eight widgets")
        check(got["same"] == got["first"], "S7 an identical body is not re-sent")
        check(got["changed"] > got["same"], "S7 a changed widget asks again")
        check(got["hBig"] > got["hSmall"] and got["hBig"] <= 260,
              "S7 the plot grows with its knots and stays bounded (%s -> %s)"
              % (got["hSmall"], got["hBig"]))
        check(got["serialize"] is False and got["optSerialize"] is False,
              "S7 the plot never reaches widgets_values")
        check(got["drawC"] is None and got["drawD"] is None and got["drawErr"] is None
              and got["drawNone"] is None and got["drew"] > 0,
              "S7 curve, dual, refusal and no-answer all draw without throwing")
        check(got["series"] == [["LOW", 12, 30, 31], ["HIGH", 0, 12, 31]] and got["marker"] == 12,
              "S7 dual: HIGH bold 0..split, LOW bold split..end, marker at the split (%s)"
              % (got["series"],))
        check(got["markerClamp"] == 19 and "50 -> 19" in got["warnClamp"],
              "S7 a clamped split is drawn where the node puts it, and said (%r)" % got["warnClamp"])
        check(got["errState"] == {"error": "unknown schedule 'x'"},
              "S7 a refusal is drawn as its text")
        check(got["vueDraws"] >= 2 and got.get("vueH") == got["hBig"],
              "S7 Nodes 2.0: each answer triggers a redraw with the NEW height (%s draws, h=%s)"
              % (got["vueDraws"], got.get("vueH")))
        check(got["depPlot"] is False, "S7 the deprecated ULSWanSigmaSchedule gets no plot")

    got = run("down")
    check(got is not None, "S7 the harness runs with the route unreachable")
    if got:
        check(got["drawC"] is None and got["drawNone"] is None,
              "S7 an unreachable route does not throw while drawing")
        check(got["same"] > got["first"],
              "S7 after an unreachable answer even an identical body asks again "
              "(nothing was drawn, so nothing is cached)")

    shutil.rmtree(tmp, ignore_errors=True)

print()
if failures:
    print("v984 sigma curves: FAIL (%d)" % len(failures))
    for f in failures:
        print("   - " + f)
    sys.exit(1)
print("%d checks, 0 failed" % len(ran))
print("v984 sigma curves: PASS")
