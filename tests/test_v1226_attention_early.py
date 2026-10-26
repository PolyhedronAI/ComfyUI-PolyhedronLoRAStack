# -*- coding: ascii -*-
"""Guard v1226 -- the attention WARN comes early, INSIDE the running stage, with the stage's own advice.

Frank, 25.10.2026: he stopped a 1.40 joint refine 45 s into step 1 -- and had seen no toast. The v1224/v1225 report
came only when the stage ended (~13 min later at 1.40), while after a handful of calls it is certain that every
self-attention call falls back. Now the books report EARLY once per stage (COVER_EARLY_CALLS calls, the share over
the warning line): worded as a forecast, with the toast by the v1225 severity rule and the stage's own tip; the
stage's final report follows with the real numbers and no second toast.

  E1 inside an open stage: the early report fires once, after COVER_EARLY_CALLS calls, through coverage_early
  E2 the early event: stage "(early)", the stage's node, the stage's tip (advise), toast; the books marked toasted
  E3 the early wording: "early -- the stage is still running", "will run about 3.0x", "stop it now"
  E4 the final report of that stage: real numbers, NO second toast; a stage that never toasted early still toasts
  E5 no stage open (a foreign sampler): the v1224 self-report, once, toast by severity
  E6 Power Upscale: _refine_advice gives the 1.37 tip for the 25.10. verdict; the refine opens its books with its
     node and that advice (cov_ctx from both call sites); the final report uses the same advice
  E7 Sampler: opens its books with its node id
  Z  ASCII
Script-style: exit 0 = pass.
"""
import re
import sys
import types
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
sent = []
srv = types.ModuleType("server")
srv.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(send_sync=lambda e, d: sent.append(d)))
sys.modules["server"] = srv


def make_override(early):
    def router(func, q, k, v, heads, *a, **kw):
        n = q.shape[1]
        if n >= 1000:
            raise PH.AddressPassThrough("sequence of %d tokens ... 32-bit" % n, seq=n, worst=LIM + 5, limit=LIM,
                                        fit=n - 20)
        return "sage"
    saved = PH.build_router
    PH.build_router = lambda mode, sparse=None: router
    try:
        return PH.build_override("sage fp8 cuda++", fallback=True, passthrough=lambda m, s: None,
                                 ledger=PH.LEDGER, node="352", early=early)
    finally:
        PH.build_router = saved


def own(*a, **k):
    return "own"


def t(n):
    return torch.empty((1, n, 8), dtype=torch.int8)


calls = []


def early(snap):
    calls.append(snap["passed"])
    return PH.coverage_early(snap)


ov = make_override(early)
PH.LEDGER.note_ms("sage fp8 cuda++", 620.0, 1873.0)
advice_seen = []


def advise(v):
    advice_seen.append(v["level"])
    return (["[PLS] Power Upscale: -> final_upscale_by 1.37 (now 1.40) ..."],
            "final_upscale_by <= 1.37 keeps the refine on the fast kernel")


# ---- E1..E3
PH.coverage_begin("Power Upscale joint refine", consumer="322", advise=advise)
for _ in range(12):
    ov(own, t(2000), t(2000), t(2000), 1)
_need(calls == [1], "E1 inside the stage: early report ONCE -- v1227: the address case on its first call (%s)" % calls)
_need(len(sent) == 1, "E2 one event so far")
if sent:
    d = sent[0]
    _need(d["stage"] == "Power Upscale joint refine (early)" and "322" in d["nodes"] and "352" in d["nodes"],
          "E2 the stage marked (early), the stage's node and the Attention node (%s, %s)" % (d["stage"], d["nodes"]))
    _need(d["tip"].startswith("final_upscale_by <= 1.37") and advice_seen == ["warn"],
          "E2 the stage's own tip, from its advise() (%s)" % d["tip"])
    _need(d["toast"] is True and d["factor"] == round(1873.0 / 620.0, 2), "E2 toast by severity (x%s)" % d["factor"])
_need(PH.LEDGER.toasted is True, "E2 the books are marked toasted")
v_early = PH.coverage_verdict(PH.LEDGER.snapshot(), early=True)
txt = "\n".join(v_early["lines"])
_need("early -- the stage is still running" in txt and "will run about 3.0x" in txt and "stop it now" in txt,
      "E3 worded as a forecast: %s" % v_early["lines"][1][:110])

# ---- E4
snap = PH.coverage_end()
PH.coverage_report(snap, consumer="322", tip="final_upscale_by <= 1.37")
_need(len(sent) == 2 and sent[1]["toast"] is False and sent[1]["stage"] == "Power Upscale joint refine",
      "E4 the final report: real numbers, no second toast (%s)" % ({k: sent[-1][k] for k in ("stage", "toast")},))
v_final = PH.coverage_verdict(snap)
_need("ran about 3.0x as long as it had to" in "\n".join(v_final["lines"]), "E4 the final wording is past tense")
sent.clear()
PH.coverage_report(dict(snap, toasted=False), consumer="322")
_need(sent and sent[0]["toast"] is True, "E4 a stage that never toasted early still toasts at its end")

# ---- E5
sent.clear(); calls.clear()
PH.LEDGER.end()                                    # no stage open
for _ in range(12):
    ov(own, t(2000), t(2000), t(2000), 1)
_need(calls == [1] and len(sent) == 1 and sent[0]["stage"] == "running stage, reported early"
      and sent[0]["toast"] is True, "E5 no stage: the self-report once, toast by severity (%s)" % (sent[0]["stage"] if sent else None,))
PH.LEDGER.end()
del sys.modules["server"]

# ---- E6
PU = (ROOT / "nodes" / "ph_power_upscale.py").read_text(encoding="utf-8")
ns = {}
for fn in ("_final_canvas", "_snap_to", "_suggest_final_by", "_refine_advice"):
    m = re.search(r"\ndef %s\(.*?(?=\n(?:def |class |_[A-Z]|[A-Z_]+ =))" % fn, PU, re.S)
    _need(m is not None, "E6 %s found" % fn)
    exec(m.group(0), ns)
ex, tip = ns["_refine_advice"]({"level": "warn", "fit_ratio": 99_865 / 102_321}, (768, 1344, 4.0, "lanczos (gpu)", 1.40, 32),
                               "after pixel")
_need(tip == "final_upscale_by <= 1.37 keeps the refine on the fast kernel" and "1024x1824" in ex[0],
      "E6 the 25.10. verdict -> tip 1.37, 1024x1824 (%s)" % tip)
_need(ns["_refine_advice"]({"level": "note", "fit_ratio": 0.9}, None, "after pixel") == ([], ""),
      "E6 no advice below the warning line")
_need('vae_tiling="Off", cov_ctx=None):' in PU and
      re.search(r'coverage_begin\("Power Upscale joint refine", consumer=node_id,\s*advise=lambda _v: _refine_advice\(_v, _ca, _co\)\)', PU)
      is not None, "E6 the refine opens its books with its node and its advice")
_need('cov_ctx=(_cov_args, "after pixel"))' in PU and 'cov_ctx=(None, "before pixel"))' in PU,
      "E6 both refine call sites hand over their context")
_need("extra, tip = _refine_advice(v, cov_args, order)" in PU, "E6 the final report uses the same advice")

# ---- E7
S = (ROOT / "nodes" / "uls_sampler.py").read_text(encoding="utf-8")
_need('_cov.coverage_begin("Sampler", consumer=node_id)' in S, "E7 the Sampler opens its books with its node id")

for p in ("nodes/ph_attention.py", "tests/test_v1226_attention_early.py"):
    _need(all(c < 128 for c in (ROOT / p).read_bytes()), "Z  ASCII %s" % p)
part = PU[PU.index("def _refine_advice("):PU.index("def _chunks(")]
_need(all(ord(c) < 128 for c in part), "Z  ASCII v1226 part of ph_power_upscale.py")

print("%s: %d check(s) failed" % ("test_v1226_attention_early", len(FAILS)) if FAILS
      else "test_v1226_attention_early: all checks passed")
sys.exit(1 if FAILS else 0)
