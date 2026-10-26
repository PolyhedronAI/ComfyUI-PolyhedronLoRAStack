# -*- coding: ascii -*-
"""Guard v1224 -- attention coverage: how much of the WORK a kernel carried, said with a number and a cure.

Frank, 25.10.2026: the Power Upscale's joint refine at 1056x1856 x 175 frames ran 6:25 per step. Every self-attention
call held 102,003 tokens -- 2.1 % past sageattention's 32-bit addressing -- and ran on the model's own backend (3x
slower on his card). The log said so in one line ending "this is normal, not a failure". final_upscale_by 1.34
instead of 1.40 (1024x1792) made the same step 2:30. Frank: "eine model-unabhaengige Warnung ... wenn man zu stark auf
eine Standard-Loesung ausweichen muss und was man tun kann".

  C1 sage_fit_tokens: the H3 qkv-view geometry fits 99,865 tokens -- solved from the strides (v1184's number)
  C2 run_sage over the limit raises AddressPassThrough (a PassThrough) with seq / worst / limit / fit; kernel not run
  C3 the ledger books per call weighted by work (q x k tokens): a passed full self-attention call outweighs served
     short cross-attention; a passed SHORT call stays a note, not a warning
  C4 coverage_verdict: share, level, factor from the live check, the cause with the fitting length, the advice
  C5 the early self-report: no stage open (a foreign sampler) -> said once when the share crosses the line
  C6 _suggest_final_by: the 25.10. case -> 1.37 (1024x1824), and that canvas fits with Frank's text + audio tokens
  C7 wiring: Attention node (hidden unique_id, ledger, early, ms), Sampler (begin/end/report around the run),
     Power Upscale (books around the joint sample, report at both refine orders, the dial in the advice)
  C8 the old "this is normal, not a failure" line is gone
  J  ph_attention_cover.js in a node harness: the line, the colours, the Attention row grows only with a report,
     the Power Upscale pane gets its span, unknown ids are ignored
  Z  ASCII
Script-style: exit 0 = pass.
"""
import json
import math
import os
import re
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

torch.set_num_threads(1)
LIM = 2 ** 31 - 1
H, D = 56, 128
QKV = 3 * H * D


def h3_q(tokens):
    base = torch.empty((tokens, QKV), dtype=torch.int8)
    q = base[:, :H * D].view(tokens, H, D).transpose(0, 1).unsqueeze(0)
    return q, base


# ---- C1
q_small, _b = h3_q(1000)
_need(PH.sage_fit_tokens(2, q_small, q_small, q_small) == 99_865,
      "C1 H3 qkv view: 99,865 tokens fit (got %d)" % PH.sage_fit_tokens(2, q_small))
k_c = torch.empty((1, H, 10, D), dtype=torch.int8)
fit_c = PH.sage_fit_tokens(2, k_c)
_need(fit_c == (LIM - ((H - 1) * 10 * D + (D - 1))) // D + 1,
      "C1 a contiguous (1,H,N,D) k: solved from ITS strides (%d)" % fit_c)
_need(PH.sage_fit_tokens(2, q_small, k_c) == 99_865, "C1 several tensors: the tightest one decides")
del _b


# ---- C2
class _Called(Exception):
    pass


def fake_sage(q, k, v, **kw):
    raise _Called("kernel launched")


q_big, base_big = h3_q(99_870)
try:
    PH.run_sage(fake_sage, {}, q_big, q_big, q_big, H, skip_reshape=True, skip_output_reshape=True)
    _need(False, "C2 over the limit: the kernel must not be launched")
except PH.AddressPassThrough as e:
    _need(isinstance(e, PH.PassThrough), "C2 AddressPassThrough IS a PassThrough (every old handler still catches it)")
    _need(e.seq == 99_870 and e.fit == 99_865 and e.limit == LIM and e.worst > LIM,
          "C2 it carries seq %d, fit %d, limit %d, worst %d" % (e.seq, e.fit, e.limit, e.worst))
    _need("32-bit" in str(e) and "99870" in str(e), "C2 the message still names tokens and limit (v1184 kept)")
except _Called:
    _need(False, "C2 the kernel was launched")
del q_big, base_big


# ---- C3  build_override with an injected router, real tensors for the cost (shapes only)
def make_override(ledger, early=None):
    def router(func, q, k, v, heads, *a, **kw):
        n = q.shape[1]
        if n >= 1000:
            raise PH.AddressPassThrough("sequence of %d tokens ... 32-bit" % n, seq=n, worst=LIM + 5, limit=LIM,
                                        fit=n - 20)
        if k.shape[1] < 100:
            raise PH.PassThrough("cross attention")
        return "sage"
    saved = PH.build_router
    PH.build_router = lambda mode, sparse=None: router
    try:
        return PH.build_override("sage fp8 cuda++", fallback=True, passthrough=lambda m, s: None,
                                 ledger=ledger, node="7", early=early)
    finally:
        PH.build_router = saved


def own(*a, **k):
    return "own"


def t(n, c=8):
    return torch.empty((1, n, c), dtype=torch.int8)


L = PH.CoverageLedger()
ov = make_override(L)
L.begin("T stage")
for _ in range(4):
    _need(ov(own, t(500), t(500), t(500), 1) == "sage", "C3 a served call returns the kernel's answer") if _ == 0 else ov(own, t(500), t(500), t(500), 1)
    ov(own, t(500), t(50), t(50), 1)                       # cross attention: passed, cheap
snap = L.end()
_need(snap["served"] == 4 and snap["passed"] == 4, "C3 4 served + 4 passed booked (%d / %d)" % (snap["served"], snap["passed"]))
_need(abs(snap["cost_s"] - 4 * 500 * 500) < 1 and abs(snap["cost_p"] - 4 * 500 * 50) < 1,
      "C3 the work is q x k tokens (served %.0f, passed %.0f)" % (snap["cost_s"], snap["cost_p"]))
_need("7" in snap["nodes"] and snap["session"] == "T stage", "C3 the node id and the stage are on the books")
v = PH.coverage_verdict(snap)
_need(v["level"] == "note" and abs(v["share"] - 1 / 11) < 1e-6,
      "C3 cheap cross-attention passes: a note (%.1f %%), not a warning" % (100 * v["share"]))
L.begin("T big")
ov(own, t(500), t(500), t(500), 1)
ov(own, t(2000), t(2000), t(2000), 1)                      # one big call past the limit
snap2 = L.end()
v2 = PH.coverage_verdict(snap2)
_need(v2["level"] == "warn" and v2["share"] > 0.9, "C3 one passed self-attention call outweighs a served one: WARN (%.0f %%)"
      % (100 * v2["share"]))
_need(snap2["addr"] == (2000, LIM + 5, LIM, 1980), "C3 the worst address case is kept (%s)" % (snap2["addr"],))
_need(L.session is None and L.served == 0, "C3 end() closes and clears the books")

# ---- C4 verdict text
L.note_ms("sage fp8 cuda++", 600.0, 1800.0)
L.begin("Power Upscale joint refine")
for _ in range(3):
    ov(own, t(2000), t(2000), t(2000), 1)
s4 = L.end()
v4 = PH.coverage_verdict(s4)
_need(v4["ratio"] == 3.0 and abs(v4["factor"] - 3.0) < 1e-9, "C4 the live check's ratio makes the cost: ~3.0x (%s)" % v4["factor"])
txt = "\n".join(v4["lines"])
_need(txt.startswith("[PLS] WARN Attention coverage (Power Upscale joint refine): 100%"), "C4 headline: WARN, stage, share")
_need("1980" in txt and "too long" in txt and "32 bit" in txt, "C4 the cause names the fitting length")
_need("pixel area x0.980" in txt and "fewer frames" in txt and "prompt shorter" in txt,
      "C4 model-independent advice: area, frames, prompt (%s)" % [l for l in v4["lines"] if "helps" in l])
_need(v4["short"].startswith("attention: 100% fell back") and "3.0x" in v4["short"], "C4 the node line: %s" % v4["short"])
_need(PH.coverage_verdict({"cost_s": 0, "cost_p": 0, "served": 0, "passed": 0}) is None,
      "C4 nothing booked (default backend) -> no verdict, no line")
ok_v = PH.coverage_verdict({"cost_s": 10.0, "cost_p": 0.0, "served": 3, "passed": 0, "mode": "m", "session": "S"})
_need(ok_v["level"] == "ok" and "all of the attention work ran on 'm'" in ok_v["lines"][0], "C4 full coverage: one ok line")

# ---- C5 early self-report
said = []
L2 = PH.CoverageLedger()
ov2 = make_override(L2, early=lambda s: said.append(s))
for _ in range(12):
    ov2(own, t(2000), t(2000), t(2000), 1)
_need(len(said) == 1 and said[0]["passed"] == 1,
      "C5 no stage open: said ONCE -- v1227: the address case on its first call (%s)" % [s["passed"] for s in said])
L2.begin("S")
for _ in range(12):
    ov2(own, t(2000), t(2000), t(2000), 1)
# v1226: an open stage reports early too (once per stage) -- Frank 25.10. stopped a refine before its end
_need(len(said) == 2 and said[1]["session"] == "S", "C5 v1226: inside an open stage the early report comes once too")
L2.end()

# ---- C6 the Power Upscale dial
PU_SRC = (ROOT / "nodes" / "ph_power_upscale.py").read_text(encoding="utf-8")
ns = {}
for fn in ("_final_canvas", "_snap_to", "_suggest_final_by"):
    m = re.search(r"\ndef %s\(.*?(?=\n(?:def |class |_[A-Z]|[A-Z_]+ =))" % fn, PU_SRC, re.S)
    _need(m is not None, "C6 %s found" % fn)
    exec(m.group(0), ns)
fr = 99_865 / 102_003                                       # the 25.10. refine: fit / sequence
got = ns["_suggest_final_by"](768, 1344, 4.0, "lanczos (gpu)", 1.40, 32, fr - 0.01)
_need(got == (1.37, (1024, 1824)), "C6 25.10.: 1.40 -> 1.37, canvas 1024x1824 (got %s)" % (got,))
if got:
    nw, nh = got[1]
    seq_new = 52 * (nw // 32) * (nh // 32) + 1891 + 584     # video tokens + Frank's prompt + audio
    _need(seq_new <= 99_865, "C6 ... and that canvas fits with the text and audio tokens (%d <= 99,865)" % seq_new)
_need(ns["_suggest_final_by"](768, 1344, 4.0, "lanczos (gpu)", 1.40, 32, 0.20) is None,
      "C6 nothing down to half the value fits -> None (the advice stays general)")
seq134 = 52 * (1024 // 32) * (1792 // 32) + 1891 + 584
_need(seq134 <= 99_865, "C6 the measured 1.34 (1024x1792) is inside the range too (%d)" % seq134)

# ---- C7 wiring
A_SRC = (ROOT / "nodes" / "ph_attention.py").read_text(encoding="utf-8")
it = PH.ULSAttention.INPUT_TYPES()
_need(it.get("hidden", {}).get("unique_id") == "UNIQUE_ID", "C7 Attention node: hidden unique_id")
_need("latent=None, unique_id=None):" in A_SRC and "ledger=LEDGER, node=unique_id," in A_SRC and "early=_early)" in A_SRC,
      "C7 Attention node: patch() books on LEDGER with its id and the early report")
_need("LEDGER.note_ms(mode_, ms_chosen, ms_base)" in A_SRC, "C7 the live check's times go on the books")
S_SRC = (ROOT / "nodes" / "uls_sampler.py").read_text(encoding="utf-8")
_need(re.search(r'coverage_begin\("Sampler"(?:, consumer=node_id)?\).*?self\._sample_impl\(\*\*kwargs\).*?coverage_end\(\).*?'
                r'coverage_report\(snap, consumer=node_id\)', S_SRC, re.S) is not None,
      "C7 Sampler: books opened before the run, closed and reported after")
_need(re.search(r'coverage_begin\("Power Upscale joint refine"[^)]*\)?.*?sample_custom\(.*?coverage_end\(\)', PU_SRC, re.S)
      is not None and '"coverage": cov_snap}' in PU_SRC, "C7 Power Upscale: books around the joint sample, snapshot in info")
_need(PU_SRC.count('_cover_report_refine(_jr_info, _cov_args, "after pixel")') == 1
      and PU_SRC.count('_cover_report_refine(_jr_info, None, "before pixel")') == 1,
      "C7 Power Upscale: the report at both refine orders")
_need("_cov_args = (fw, fh, fsc, str(resize_method), fby, _jr_snap)" in PU_SRC,
      "C7 Power Upscale: the refine canvas's size law is kept for the dial")

# ---- C8
_need("this is normal, not a failure" not in A_SRC, "C8 the quiet 'normal, not a failure' line is gone")

# ---- J
NODE = shutil.which("node")
JS = ROOT / "web" / "js" / "ph_attention_cover.js"
HARNESS = r"""
globalThis.LiteGraph = { vueNodesMode: false };
const made = [];
function el(tag) { const e = { tagName: tag, style: {}, children: [], parentElement: null, textContent: "", title: "",
  insertBefore(c, ref) { c.parentElement = this; this.children.unshift(c); return c; },
  appendChild(c) { c.parentElement = this; this.children.push(c); return c; } }; made.push(e); return e; }
globalThis.document = { createElement: el, querySelector: () => null };
const { app } = await import("./scripts/app.js");
const M = await import("./web/js/ph_attention_cover.js");
function mk(id, cls) { return { id, comfyClass: cls, type: cls, widgets: [], size: [300, 100], refits: 0,
  addCustomWidget(w) { this.widgets.push(w); return w; }, computeSize() { return [300, 100 + this.widgets.reduce((a, w) => a + (w.computeSize ? w.computeSize(300)[1] : 0), 0)]; },
  setSize(s) { this.size = s; this.refits++; }, setDirtyCanvas() {} }; }
const attn = mk(5, "ULSAttention"), pu = mk(9, "ULSPowerUpscale");
const head = el("div"); const hud = el("span"); head.appendChild(hud); pu._procHud = hud;
const nodes = { 5: attn, 9: pu };
app.graph = { getNodeById: (i) => nodes[i] || null };
const ext = app._ext.find((e) => e.name === "polyhedron.attention.cover");
await ext.beforeRegisterNodeDef({ prototype: {} }, { name: "Other" });
const proto = {}; await ext.beforeRegisterNodeDef({ prototype: proto }, { name: "ULSAttention" });
proto.onNodeCreated.call(attn);
const before = attn.computeSize()[1];
const n = M.takeCover({ nodes: ["5", "9", "77", "5"], level: "warn", text: "attention: 100% fell back (~3.0x time)",
  tip: "final_upscale_by <= 1.37 keeps the refine on the fast kernel", stage: "Power Upscale joint refine" });
const after = attn.computeSize()[1];
const span = head.children.find((c) => c !== hud);
console.log(JSON.stringify({ n, before, after, refits: attn.refits, rows: attn.widgets.length,
  line: M.coverLine(attn._plsCover), warn: M.colourOf("warn"), ok: M.colourOf("ok"),
  span: span ? [span.textContent, span.style.color, head.children.indexOf(span) < head.children.indexOf(hud)] : null,
  serialize: attn.widgets[0] && attn.widgets[0].serialize, size: attn.size[1],
  opt: attn.widgets[0] && attn.widgets[0].options && attn.widgets[0].options.serialize,
  tall: (() => { const t2 = mk(6, "ULSAttention"); t2.size = [300, 400]; nodes[6] = t2; proto.onNodeCreated.call(t2);
          M.takeCover({ nodes: ["6"], level: "ok", text: "attention: 100% on x" }); return t2.size[1]; })(),
  none: M.takeCover(null), bad: M.takeCover({ nodes: "x" }) }));
process.exit(0);   // uls_vue_views keeps a renderer watch running
"""


def run_js():
    if NODE is None:
        return None, "node missing"
    tmp = tempfile.mkdtemp(prefix="v1224_")
    try:
        os.makedirs(os.path.join(tmp, "scripts"))
        with open(os.path.join(tmp, "scripts", "app.js"), "w", encoding="utf-8") as fh:
            fh.write("export const app = { _ext: [], graph: null, registerExtension(e) { this._ext.push(e); } };\n")
        with open(os.path.join(tmp, "scripts", "api.js"), "w", encoding="utf-8") as fh:
            fh.write("export const api = { addEventListener() {} };\n")
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
    _need(got["n"] == 2, "J  two known nodes taken, the unknown id and the duplicate ignored (%d)" % got["n"])
    _need(got["before"] == 100 and got["after"] == 120 and got["refits"] == 1 and got["size"] == 120,
          "J  the Attention row is 0 px without a report, 20 px with one; a tight node grows (%d -> %d, size %d)"
          % (got["before"], got["after"], got["size"]))
    _need(got["tall"] == 400, "J  a node made taller does not shrink (%d)" % got["tall"])
    _need(got["opt"] is False, "J  the row is kept out of the prompt (options.serialize false)")
    _need(got["rows"] == 1 and got["serialize"] is False, "J  ONE row, never serialised")
    _need(got["line"].startswith("\u26a0 attention: 100% fell back") and "1.37" in got["line"]
          and "(Power Upscale joint refine)" in got["line"], "J  the line: %s" % got["line"])
    _need(got["warn"] != got["ok"], "J  WARN and ok have their own colours")
    _need(got["span"] is not None and got["span"][0].startswith("\u26a0 attention: 100%") and got["span"][1] == got["warn"]
          and got["span"][2], "J  the Power Upscale pane: a span before the HUD, in the warn colour (%s)" % (got["span"],))
    _need(got["none"] == 0 and got["bad"] == 0, "J  a broken event is ignored")

# ---- Z
for p in ("nodes/ph_attention.py", "web/js/ph_attention_cover.js", "tests/test_v1224_attention_cover.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "Z  ASCII %s" % p)
# uls_sampler.py and ph_power_upscale.py carry older non-ASCII comments; the v1224 additions are checked alone
for p, a, b in (("nodes/uls_sampler.py", "def _attention_cover():", "def _current_node_id():"),
                ("nodes/ph_power_upscale.py", "def _suggest_final_by(", "def _chunks(")):
    src = (ROOT / p).read_text(encoding="utf-8")
    part = src[src.index(a):src.index(b)]
    _need(all(ord(c) < 128 for c in part), "Z  ASCII v1224 part of %s" % p)

print("%s: %d check(s) failed" % ("test_v1224_attention_cover", len(FAILS)) if FAILS
      else "test_v1224_attention_cover: all checks passed")
sys.exit(1 if FAILS else 0)
