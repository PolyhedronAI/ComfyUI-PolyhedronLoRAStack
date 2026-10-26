# -*- coding: ascii -*-
"""Guard v1227 -- the address case warns on its FIRST call; the toast leaves a trace in the browser console.

Frank, 25.10.2026: the v1226 early WARN came 2:30 into the refine step -- it waited for 8 calls of ~7 s each on the
fallback -- and the toast did not show: the pane line was painted by the v1224 frontend file still in the browser
cache, which has no toast. Now:
  N1 an AddressPassThrough that carries most of the booked work (>= COVER_ADDRESS_SHARE) reports at once -- call 1
  N2 a single address pass-through among heavy served work does NOT report at once (the share decides)
  N3 any other fallback reason still waits for COVER_EARLY_CALLS calls
  N4 the Attention node prints no separate 'not every call' line for the address case (the WARN is that line);
     other reasons keep it
  J  ph_attention_cover.js names its version on load, logs every toast, falls back to addAlert, and warns in the
     console when no toast API exists (never silent)
  Z  ASCII
Script-style: exit 0 = pass.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


sys.path.insert(0, str(ROOT / "nodes"))
import ph_attention as PH  # noqa: E402

LIM = 2 ** 31 - 1


def make(ledger, early, other=False):
    def router(func, q, k, v, heads, *a, **kw):
        n = q.shape[1]
        if n >= 1000:
            if other:
                raise PH.PassThrough("masked call")
            raise PH.AddressPassThrough("sequence of %d tokens ... 32-bit" % n, seq=n, worst=LIM + 5, limit=LIM, fit=n - 20)
        return "sage"
    saved = PH.build_router
    PH.build_router = lambda mode, sparse=None: router
    try:
        return PH.build_override("sage fp8 cuda++", fallback=True, passthrough=lambda m, s: None,
                                 ledger=ledger, node="352", early=early)
    finally:
        PH.build_router = saved


def own(*a, **k):
    return "own"


def t(n):
    return torch.empty((1, n, 8), dtype=torch.int8)


# N1
said = []
L = PH.CoverageLedger(); ov = make(L, lambda s: said.append(s["passed"] + s["served"]))
L.begin("Power Upscale joint refine")
ov(own, t(2000), t(2000), t(2000), 1)
_need(said == [1], "N1 address case: the WARN on the FIRST call (%s)" % said)
for _ in range(10):
    ov(own, t(2000), t(2000), t(2000), 1)
_need(said == [1], "N1 ... and once per stage")
L.end()

# N2
said.clear(); L.begin("mixed")
for _ in range(6):
    ov(own, t(900), t(900), t(900), 1)                 # served, heavy
ov(own, t(1000), t(1000), t(1000), 1)                  # one address pass, ~16 % of the work
_need(said == [], "N2 one address pass among heavy served work: not at once (share %.0f %%)"
      % (100 * L.cost_p / (L.cost_p + L.cost_s)))
ov(own, t(1000), t(1000), t(1000), 1)                  # 8th call, a second address pass (~29 %): the 8-call rule
_need(said == [8], "N2 ... the 8-call rule takes it from there (%s)" % said)
L.end()

# N3
said.clear()
L3 = PH.CoverageLedger(); ov3 = make(L3, lambda s: said.append(s["passed"] + s["served"]), other=True)
L3.begin("S")
for _ in range(10):
    ov3(own, t(2000), t(2000), t(2000), 1)
_need(said == [PH.COVER_EARLY_CALLS], "N3 another reason waits for %d calls (%s)" % (PH.COVER_EARLY_CALLS, said))
L3.end()

# N4
A = (ROOT / "nodes" / "ph_attention.py").read_text(encoding="utf-8")
i = A.index("        def _passthrough(mode_, skip):")
blk = A[i:i + 900]
_need("if isinstance(skip, AddressPassThrough):" in blk and blk.index("return") < blk.index("not every call"),
      "N4 the address case prints no separate 'not every call' line; other reasons keep it")
_need("COVER_ADDRESS_SHARE = 0.5" in A, "N4 the address share is 0.5")

# J
NODE = shutil.which("node")
HARNESS = r"""
const logs = [];
console.info = (m) => logs.push(["info", String(m)]);
console.warn = (m) => logs.push(["warn", String(m)]);
globalThis.LiteGraph = { vueNodesMode: false };
globalThis.document = { createElement: () => ({ style: {}, children: [], appendChild() {}, insertBefore() {} }), querySelector: () => null };
const { app } = await import("./scripts/app.js");
const { api } = await import("./scripts/api.js");
await import("./web/js/ph_attention_cover.js");
app.graph = { getNodeById: () => null };
app._ext.find((e) => e.name === "polyhedron.attention.cover").setup();
const fire = (d) => api._l["pls.attention_cover"]({ detail: d });
const ev = { nodes: [], level: "warn", text: "attention: 100% fell back (~3.0x time)", tip: "t", stage: "S (early)", toast: true };
const shown = [];
app.extensionManager = { toast: { add: (x) => shown.push(x) } }; fire(ev);
const alerts = [];
app.extensionManager = { toast: { addAlert: (x) => alerts.push(x) } }; fire(ev);
app.extensionManager = {}; fire(ev);
console.log(JSON.stringify({ logs, shown: shown.length, alerts }));
process.exit(0);
"""


def run_js():
    if NODE is None:
        return None, "node missing"
    tmp = tempfile.mkdtemp(prefix="v1227_")
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
        out = [ln for ln in rr.stdout.strip().splitlines() if ln.startswith("{")]
        return json.loads(out[-1]), ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


got, err = run_js()
_need(got is not None, "J  harness runs the real ph_attention_cover.js " + err)
if got:
    msgs = [m for _k, m in got["logs"]]
    _need(any("ph_attention_cover.js v1227 loaded" in m for m in msgs), "J  the file names its version on load")
    _need(got["shown"] == 1 and any(m.startswith("[PLS] attention toast shown: Attention fell back") for m in msgs),
          "J  a toast is shown AND logged")
    _need(len(got["alerts"]) == 1 and any("(alert)" in m for m in msgs), "J  addAlert as the house fallback, logged")
    _need(any(k == "warn" and "could NOT be shown" in m for k, m in got["logs"]), "J  no toast API -> a console warning, never silence")

for p in ("nodes/ph_attention.py", "web/js/ph_attention_cover.js", "tests/test_v1227_attention_now.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "Z  ASCII %s" % p)

print("%s: %d check(s) failed" % ("test_v1227_attention_now", len(FAILS)) if FAILS
      else "test_v1227_attention_now: all checks passed")
sys.exit(1 if FAILS else 0)
