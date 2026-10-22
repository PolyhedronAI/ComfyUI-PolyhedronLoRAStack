# -*- coding: ascii -*-
"""Guard v1212 -- with a curve wired, the Power Upscale shows the steps it RUNS.

(Frank 22.10.: "Unser Upscaler laeuft btw. seit jeher mit 8 Steps" -- it ran 2: a wired curve replaces steps and
scheduler, the refine starts at sigma = denoise and runs every curve point below it. "Ja, das bauen wir direkt ...
go.")

  C1 the arithmetic the label shows is the run's own: Frank's curve (HyperFlow 8-step, shift 12) at denoise 0.5 ->
     2 steps (0.5, 0.469, 0), 0.25 -> 1, 0.7 -> 3 (_sigma_run, lifted from ph_power_upscale.py)
  C2 the route /pls/pu_sigma_run exists, is registered, and answers from _sigma_run (never a copy)
  C3 the frontend, driven in node: _curveWired / _curveSource / _curveValues / _sourceKey / _setSuffix on a mock
     graph (a Sigma List's drawn preview is read; no link -> not wired; the suffix never stacks)
  C4 statics: steps / scheduler / steps_low / scheduler_low hide while wired (the v888 hide, slots kept), the
     denoise label carries the count, wiring re-applies, the Vue timer is cleared on remove, a not-yet-drawn Sigma
     List is ASKED on its own route
Script-style: exit 0 = pass.
"""
import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


# ---- C1
SRC = ROOT / "nodes" / "ph_power_upscale.py"
T = ast.parse(SRC.read_text(encoding="utf-8"))
body = [n for n in T.body if isinstance(n, ast.FunctionDef) and n.name in ("_sigma_tail", "_sigma_run", "_is_flow_curve",
                                                                           "_sigma_on_grid")]
ns = {}
exec(compile(ast.Module(body=body, type_ignores=[]), str(SRC), "exec"), ns)
raw = [1, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0]
curve = [12.0 * v / (1.0 + 11.0 * v) for v in raw]
got = {d: len(ns["_sigma_run"](curve, d)[0]) - 1 for d in (0.25, 0.5, 0.7)}
r05 = [round(v, 3) for v in ns["_sigma_run"](curve, 0.5)[0]]
_need(got == {0.25: 1, 0.5: 2, 0.7: 3} and r05 == [0.5, 0.469, 0.0],
      "C1 Frank's curve: denoise 0.25/0.5/0.7 -> 1/2/3 steps, 0.5 runs %s" % r05)

# ---- C2
# public build: the route lives in the sigma route module (v392), not in uls_routes.py
R = (ROOT / "nodes" / "ph_sigma_routes.py").read_text(encoding="utf-8")
h = R[R.index("async def handle_pu_sigma_run_post"):R.index("_SIGMA_ROUTES = (")]
_need('("POST", "/pls/pu_sigma_run",        handle_pu_sigma_run_post)' in R, "C2 route registered")
_need("from .ph_power_upscale import _sigma_run, _sigma_on_grid" in h and "_sigma_run(curve, float(d))" in h
      and '"steps": max(0, len(run) - 1)' in h, "C2 the route answers from _sigma_run itself")

# ---- C3 driven in node
JS = (ROOT / "web" / "js" / "ph_power_upscale.js").read_text(encoding="utf-8")


def fn(name):
    m = re.search(r"\nfunction %s\(.*?\n}\n" % re.escape(name), JS, re.S)
    assert m, name
    return m.group(0)


prog = "\n".join(fn(n) for n in ("_curveSource", "_curveWired", "_curveValues", "_sourceKey", "_setSuffix")) + r"""
const sl = { id: 380, type: "ULSSigmaList", widgets: [{ name: "preset", value: "hyperflow_8step_h3_raw" }],
             _slPreview: { ok: true, sigmas: [1, 0.5, 0] } };
const links = { 7: { origin_id: 380 } };
const graph = { links, getNodeById: (i) => (i === 380 ? sl : null) };
const pu = { graph, inputs: [{ name: "latent", link: null }, { name: "sigmas", link: 7 }] };
const bare = { graph, inputs: [{ name: "sigmas", link: null }] };
const w = { name: "denoise", label: undefined };
_setSuffix(w, " \u00b7 2 steps on the curve"); _setSuffix(w, " \u00b7 3 steps on the curve");
const w2 = { name: "denoise" }; _setSuffix(w2, " x"); _setSuffix(w2, "");
console.log(JSON.stringify({
  wired: _curveWired(pu), bareWired: _curveWired(bare), src: (_curveSource(pu) || {}).id,
  vals: _curveValues(_curveSource(pu)), noSrc: _curveSource(bare), key: _sourceKey(sl),
  label: w.label, back: w2.label,
  undrawn: _curveValues({ type: "ULSSigmaList", widgets: [] }) }));
"""
with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
    f.write(prog)
out = subprocess.run(["node", f.name], capture_output=True, text=True)
try:
    r = json.loads(out.stdout.strip().splitlines()[-1])
except Exception:
    r = {}
    print(out.stderr[:400])
_need(r.get("wired") is True and r.get("bareWired") is False and r.get("src") == 380 and r.get("noSrc") is None,
      "C3 wired follows the sigmas link and finds its source")
_need(r.get("vals") == [1, 0.5, 0] and r.get("undrawn") is None and r.get("key") == [380, "ULSSigmaList",
                                                                                        ["hyperflow_8step_h3_raw"], [1, 0.5, 0]],
      "C3 the source's drawn curve is read; not drawn -> none; the key carries widgets + curve")
_need(r.get("label") == "denoise \u00b7 3 steps on the curve" and r.get("back") == "denoise",
      "C3 the suffix replaces, never stacks, and clears back to the base label")

# ---- C4 statics
_need('const CURVE_UNREAD = ["steps", "scheduler", "steps_low", "scheduler_low"];' in JS, "C4 the four unread dials")
ams = JS[JS.index("function _applyModeState(node) {"):JS.index("// \u2500\u2500 v1212: a wired curve IS the schedule")]
_need("const off = !dual || (wired && CURVE_UNREAD.includes(name));" in ams
      and "_setDisabled(_findWidget(node, name), wired)" in ams and "_curveLabels(node, wired, dual);" in ams,
      "C4 _applyModeState hides them while wired and labels denoise")
_need('" step" + (x.steps === 1 ? "" : "s") + " on the curve"' in JS and "/pls/pu_sigma_run" in JS,
      "C4 the label says the asked count")
_need("nodeType.prototype.onConnectionsChange = function" in JS and "_applyModeState(this);" in
      JS[JS.index("nodeType.prototype.onConnectionsChange = function"):][:300], "C4 wiring re-applies at once")
rm = JS[JS.index("nodeType.prototype.onRemoved = function"):][:300]
_need("self._puCurveTimer = setInterval(" in JS and "clearInterval(this._puCurveTimer)" in rm,
      "C4 the Vue timer is cleared on remove (leak law)")
_need('"/api/pls/sigma_preview"' in JS[JS.index("async function _resolveCurve"):][:1400],
      "C4 a Sigma List that has not drawn is asked on its own route")

print("test_v1212_pu_curve_steps: %s" % ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
sys.exit(1 if FAILS else 0)
