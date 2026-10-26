# -*- coding: ascii -*-
"""Guard v1225 -- the attention coverage gets a toast where the run lost real time, and an honest refine plan.

Frank, 25.10.2026 (v1224 field test, 1.40 on purpose): the WARN came as designed (100 % fell back, x3.0, 'final_
upscale_by 1.37'), but no toast -- "haengt von der Schwere ab, ob wir eine machen". And the refine planned step 1 at
~3:19 and took 6:29: the learned rate knew only runs on the fast kernel.

  T1 coverage_verdict.toast: WARN with a measured cost >= x1.5 -> toast; WARN at x1.2 -> none; a note -> none;
     unmeasured (no live check): >= 50 % of the work on the fallback -> toast, 30 % -> none
  T2 coverage_report sends toast / share / factor with the event (a fake PromptServer catches it)
  T3 Power Upscale: _slow_plan_rates scales only the step; _clean_step learns the step without the factor;
     the wiring -- the plan reads _ATTN_SLOW by (section, w, h, frames), a WARN with a factor stores it, a clean
     run forgets it, _rates_learn gets the clean step
  J  ph_attention_cover.js: toastOf builds ONE warn toast (summary with the stage, the tip, 15 s) only when the
     server said toast; the listener shows it through app.extensionManager.toast and still paints the nodes
  Z  ASCII
Script-style: exit 0 = pass.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


sys.path.insert(0, str(ROOT / "nodes"))
import ph_attention as PH  # noqa: E402


def snap(share, ms=None, mode="sage fp8 cuda++"):
    tot = 1000.0
    return {"cost_s": tot * (1 - share), "cost_p": tot * share, "served": 5, "passed": 5, "mode": mode,
            "session": "S", "ms": ms, "addr": None, "other": {}, "nodes": ["352"]}


# ---- T1
v = PH.coverage_verdict(snap(1.0, ("sage fp8 cuda++", 620.0, 1873.0)))
_need(v["level"] == "warn" and v["toast"] is True, "T1 100 %% at x%.1f -> toast" % v["factor"])
v = PH.coverage_verdict(snap(0.2, ("sage fp8 cuda++", 600.0, 1800.0)))      # factor 1.4
_need(v["level"] == "warn" and v["toast"] is False, "T1 WARN at x%.2f (< 1.5) -> no toast" % v["factor"])
v = PH.coverage_verdict(snap(0.05, ("sage fp8 cuda++", 600.0, 1800.0)))
_need(v["level"] == "note" and v["toast"] is False, "T1 a note -> no toast")
v = PH.coverage_verdict(snap(0.6))
_need(v["factor"] is None and v["toast"] is True, "T1 unmeasured, 60 % on the fallback -> toast")
v = PH.coverage_verdict(snap(0.3))
_need(v["factor"] is None and v["toast"] is False, "T1 unmeasured, 30 % -> no toast")
v = PH.coverage_verdict(snap(0.0))
_need(v["level"] == "ok" and v["toast"] is False, "T1 ok -> no toast")

# ---- T2
sent = []
srv = types.ModuleType("server")
srv.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(send_sync=lambda e, d: sent.append((e, d))))
sys.modules["server"] = srv
PH.coverage_report(snap(1.0, ("sage fp8 cuda++", 620.0, 1873.0)), consumer="322", tip="final_upscale_by <= 1.37")
_need(len(sent) == 1 and sent[0][0] == "pls.attention_cover", "T2 one event")
if sent:
    d = sent[0][1]
    _need(d["toast"] is True and d["share"] == 1.0 and d["factor"] == round(1873.0 / 620.0, 2)
          and d["nodes"] == ["352", "322"] and d["tip"].startswith("final_upscale_by"),
          "T2 the event carries toast, share, factor, nodes, tip (%s)" % {k: d[k] for k in ("toast", "share", "factor")})
del sys.modules["server"]

# ---- T3
PU = (ROOT / "nodes" / "ph_power_upscale.py").read_text(encoding="utf-8")
ns = {}
for fn in ("_slow_plan_rates", "_clean_step"):
    m = re.search(r"\ndef %s\(.*?(?=\n(?:def |class |_[A-Z]|[A-Z_]+ =))" % fn, PU, re.S)
    _need(m is not None, "T3 %s found" % fn)
    exec(m.group(0), ns)
r = {"enc": 0.46, "step": 0.37, "dec": 0.55, "runs": 145}
s2 = ns["_slow_plan_rates"](r, 3.0)
_need(abs(s2["step"] - 1.11) < 1e-9 and s2["enc"] == 0.46 and s2["dec"] == 0.55 and r["step"] == 0.37,
      "T3 the plan scales the step only, the stored rates stay untouched")
_need(ns["_slow_plan_rates"](r, None) == r, "T3 no remembered factor -> the plan is unchanged")
_need(abs(ns["_clean_step"](389.7, 3.0) - 129.9) < 1e-6, "T3 389.7 s measured / x3.0 -> 129.9 s learned")
_need(ns["_clean_step"](150.0, None) == 150.0 and ns["_clean_step"](150.0, 1.02) == 150.0,
      "T3 no fallback (or within noise) -> the measured step is learned")
_need("slow_key = (str(rsec), int(sw), int(sh), int(n_in))" in PU and "slow_f = _ATTN_SLOW.get(slow_key)" in PU
      and "_phase_plan(mpf, steps_run, vid_lat_c is not None, _slow_plan_rates(rates, slow_f))" in PU,
      "T3 the plan reads the remembered factor for this size")
_need(re.search(r"if slow_now and slow_now > 1\.05:\s*_ATTN_SLOW\[slow_key\] = slow_now.*?else:\s*_ATTN_SLOW\.pop\(slow_key, None\)",
                PU, re.S) is not None, "T3 a WARN with a factor stores it, a clean run forgets it")
_need("_rates_learn(rsec, mpf, enc_s, _clean_step(step_meas, slow_now)," in PU, "T3 the rates learn the clean step")
_need(PU.index("slow_now = None") > PU.index('cov_snap = _cov.coverage_end()'),
      "T3 the factor is read after the books are closed")

# ---- J
NODE = shutil.which("node")
HARNESS = r"""
globalThis.LiteGraph = { vueNodesMode: false };
globalThis.document = { createElement: () => ({ style: {}, children: [], appendChild() {}, insertBefore() {} }), querySelector: () => null };
const { app } = await import("./scripts/app.js");
const { api } = await import("./scripts/api.js");
const M = await import("./web/js/ph_attention_cover.js");
const toasts = [];
app.extensionManager = { toast: { add: (t) => toasts.push(t) } };
const attn = { id: 5, comfyClass: "ULSAttention", widgets: [], size: [300, 300], addCustomWidget(w) { this.widgets.push(w); return w; },
  computeSize() { return [300, 120]; }, setSize(s) { this.size = s; }, setDirtyCanvas() {} };
app.graph = { getNodeById: (i) => (i === 5 ? attn : null) };
app._ext.find((e) => e.name === "polyhedron.attention.cover").setup();
const fire = (d) => api._l["pls.attention_cover"]({ detail: d });
fire({ nodes: ["5"], level: "warn", text: "attention: 100% fell back (~3.0x time)", tip: "final_upscale_by <= 1.37 keeps the refine on the fast kernel",
       stage: "Power Upscale joint refine", toast: true, share: 1, factor: 3.02 });
fire({ nodes: ["5"], level: "warn", text: "attention: 20% fell back (~1.4x time)", tip: "", stage: "Sampler", toast: false });
fire({ nodes: ["5"], level: "ok", text: "attention: 100% on x", stage: "Sampler" });
console.log(JSON.stringify({ n: toasts.length, t: toasts[0] || null, painted: attn._plsCover ? attn._plsCover.text : null,
  none: M.toastOf({ toast: false }), nul: M.toastOf(null) }));
process.exit(0);
"""


def run_js():
    if NODE is None:
        return None, "node missing"
    tmp = tempfile.mkdtemp(prefix="v1225_")
    try:
        os.makedirs(os.path.join(tmp, "scripts"))
        with open(os.path.join(tmp, "scripts", "app.js"), "w", encoding="utf-8") as fh:
            fh.write("export const app = { _ext: [], graph: null, registerExtension(e) { this._ext.push(e); } };\n")
        with open(os.path.join(tmp, "scripts", "api.js"), "w", encoding="utf-8") as fh:
            fh.write("export const api = { _l: {}, addEventListener(n, f) { this._l[n] = f; } };\n")
        os.makedirs(os.path.join(tmp, "web", "js"))
        for f in ("ph_attention_cover.js", "uls_vue_views.js"):
            shutil.copyfile(str(ROOT / "web" / "js" / f), os.path.join(tmp, "web", "js", f))
        with open(os.path.join(tmp, "h.mjs"), "w", encoding="utf-8") as fh:
            fh.write(HARNESS)
        rr = subprocess.run([NODE, "h.mjs"], capture_output=True, text=True, cwd=tmp, timeout=60)
        if rr.returncode != 0:
            return None, rr.stderr[-1500:]
        return json.loads(rr.stdout.strip().splitlines()[-1]), ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


got, err = run_js()
_need(got is not None, "J  harness runs the real ph_attention_cover.js " + err)
if got:
    t = got["t"] or {}
    _need(got["n"] == 1, "J  three reports, ONE toast (only the one the server marked) -- got %d" % got["n"])
    _need(t.get("severity") == "warn" and t.get("life") == 15000
          and t.get("summary") == "Attention fell back -- Power Upscale joint refine",
          "J  warn toast, 15 s, the stage in the summary (%s)" % t.get("summary"))
    _need("100% fell back" in t.get("detail", "") and "1.37" in t.get("detail", "") and "console" in t.get("detail", ""),
          "J  the detail: the line, the tip, where to read more")
    _need(got["painted"] == "attention: 100% on x", "J  the nodes are still painted by every report")
    _need(got["none"] is None and got["nul"] is None, "J  no toast flag -> no toast object")

for p in ("nodes/ph_attention.py", "web/js/ph_attention_cover.js", "tests/test_v1225_attention_toast.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "Z  ASCII %s" % p)
part = PU[PU.index("_ATTN_SLOW = {}"):PU.index("def _cover_mod():")]
_need(all(ord(c) < 128 for c in part), "Z  ASCII v1225 part of ph_power_upscale.py")

print("%s: %d check(s) failed" % ("test_v1225_attention_toast", len(FAILS)) if FAILS
      else "test_v1225_attention_toast: all checks passed")
sys.exit(1 if FAILS else 0)
