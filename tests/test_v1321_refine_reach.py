# -*- coding: ascii -*-
"""Guard v1321 -- the Power Upscale's joint refine stays inside the fast attention kernel's MEASURED reach.

Frank 08.11., alien bridge 360 scene 1 (243 frames, refine after pixel at 1792x1024): one attention call held
135969 tokens, sageattention's 32-bit offsets reach 99865 -> the whole refine fell back to the model's own
backend, step 1 ran ~3x slower and the card ran OUT OF MEMORY (12.9 GiB held, 1.36 GiB asked). The node had
already printed the cure (final_upscale_by 1.34 -> 1.16) -- and kept going into the OOM.

  R1  _fit_capped_final_by (pure): None when the dialled value fits or no reach is known; the largest value
      (0.01 steps) whose snapped canvas x frames stays inside the reach -- Frank's numbers: 1.34 at 1344x768 x
      243 inside ~323 megapixel-frames lands below 1.34 and inside; (None, None) when even half does not fit
  R2  _rates_fit_learn: fit_mpf = megapixel-frames seen x (fit_ratio - 1 %), persisted in rates.json with the
      section; nonsense (ratio >= 1, <= 0, no mpf) learns nothing; _rates_clean keeps fit_mpf through a reload
  R3  _is_oom: torch.OutOfMemoryError and an 'out of memory' RuntimeError are OOM; anything else is not
  R4  wiring (source): the pre-check sits BEFORE the cache key (one canvas for key, pixel pass and refine) and
      rebinds final_upscale_by; the refine learns the reach from the address verdict (fit_box); an OOM with a
      learned reach on the after-pixel path raises _RefineRefit; the caller retries ONCE (cached=None, no cache
      the retry keeps the cache key honest (the new canvas)
  S   ASCII of this guard
"""
import ast
import importlib.util
import json
import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1321_refine_reach"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


SRC = (ROOT / "nodes" / "ph_power_upscale.py").read_text(encoding="utf-8")
tree = ast.parse(SRC)


def _seg(name):
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
            return ast.get_source_segment(SRC, n)
    raise KeyError(name)


# -- R1 -----------------------------------------------------------------------
ns = {}
exec(_seg("_snap_to") + "\n\n" + _seg("_fit_capped_final_by"), ns)
fcap, snap_to = ns["_fit_capped_final_by"], ns["_snap_to"]
W, H, N, SNAP = 1344, 768, 243, 32
seen = 1792 * 1024 * 243 / 1e6                     # 446 megapixel-frames, the OOM run
reach = seen * (99865.0 / 135969.0 - 0.01)          # what _rates_fit_learn keeps from that verdict
got = fcap(W, H, 1.34, N, SNAP, reach)
ok = isinstance(got, tuple) and got[0] is not None and got[0] < 1.34
if ok:
    sw, sh = got[1]
    ok = sw * sh * N / 1e6 <= reach and (sw, sh) == snap_to(int(round(W * got[0])), int(round(H * got[0])), SNAP)
    nxt = got[0] + 0.01
    nsw, nsh = snap_to(int(round(W * nxt)), int(round(H * nxt)), SNAP)
    ok = ok and nsw * nsh * N / 1e6 > reach      # the LARGEST that fits
print("  R1 1.34 ->", got)
_need(ok, "R1 Frank's 360 refine: 1.34 lands on the largest value inside the measured reach")
_need(fcap(W, H, 1.34, 175, SNAP, reach) is None, "R1 175 frames at 1.34 already fit (the 02 runs): no change")
_need(fcap(W, H, 1.34, N, SNAP, None) is None and fcap(W, H, 1.34, N, SNAP, 0) is None,
      "R1 no reach known: no change")
_need(fcap(W, H, 1.34, N, SNAP, 5.0) == (None, None), "R1 even half does not fit: (None, None)")

# -- R2 -----------------------------------------------------------------------
tmp = tempfile.mkdtemp()
fp = types.ModuleType("folder_paths")
fp.get_user_directory = lambda: tmp
sys.modules["folder_paths"] = fp
sys.path.insert(0, str(ROOT / "nodes"))
spec = importlib.util.spec_from_file_location("ph_progress_t1321", ROOT / "nodes" / "ph_progress.py")
pp = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(pp)
    v = pp._rates_fit_learn("joint:TestModel", seen, 99865.0 / 135969.0)
    _need(v is not None and abs(v - reach) < 1e-6, "R2 fit_mpf = mpf seen x (fit_ratio - 1 %)")
    f = os.path.join(tmp, "polyhedron", "rates.json")
    d = json.load(open(f, encoding="utf-8")) if os.path.isfile(f) else {}
    _need(abs(((d.get("sections") or {}).get("joint:TestModel") or {}).get("fit_mpf", 0) - reach) < 1e-6,
          "R2 persisted in rates.json")
    _need(abs((pp._rates_clean(d["sections"]["joint:TestModel"]).get("fit_mpf") or 0) - reach) < 1e-6,
          "R2 _rates_clean keeps fit_mpf through a reload")
    _need(pp._rates_fit_learn("joint:X", seen, 1.2) is None and pp._rates_fit_learn("joint:X", seen, 0) is None
          and pp._rates_fit_learn("joint:X", 0, 0.5) is None and pp._rates_fit_learn("joint:X", "a", 0.5) is None,
          "R2 nonsense learns nothing")
except Exception as e:  # pragma: no cover
    _need(False, "R2 ph_progress loads (%s)" % e)

# -- R3 -----------------------------------------------------------------------
try:
    import torch
    ns3 = {"torch": torch}
    exec(_seg("_is_oom"), ns3)
    is_oom = ns3["_is_oom"]
    oom_cls = getattr(torch, "OutOfMemoryError", None)
    _need((oom_cls is None or is_oom(oom_cls("x"))) and is_oom(RuntimeError("CUDA out of memory. Tried"))
          and not is_oom(ValueError("shape mismatch")), "R3 OOM recognised, other errors not")
except ImportError:
    ns3 = {"torch": types.SimpleNamespace()}
    exec(_seg("_is_oom"), ns3)
    _need(ns3["_is_oom"](RuntimeError("out of memory")) and not ns3["_is_oom"](ValueError("x")),
          "R3 OOM recognised (no torch here)")

# -- R4 -----------------------------------------------------------------------
pre = SRC.find("v1321 (Frank 08.11., bridge 360 scene 1)")
key = SRC.find("_jr_key = _cache_key(frames, vae, _um_key, (_hw, _hh), resize_method)")
_need(0 < pre < key, "R4 the pre-check sits before the cache key")
blk = SRC[pre:key]
_need("final_upscale_by = _cf" in blk and "_fit_capped_final_by(w, h, final_upscale_by, n_frames" in blk
      and '.get("fit_mpf")' in blk, "R4 the pre-check rebinds final_upscale_by from the learned reach")
jr = _seg("_joint_refine")
_need("_oom = _exc" in jr and "_fit = _learn_reach(cov_snap, rsec, mpf)" in jr
      and jr.count("_learn_reach(cov_snap, rsec, mpf)") == 2,
      "R4 the refine learns the reach from its coverage books (after an OOM AND after a slow run)")
_need("raise _RefineRefit(_fit) from _oom" in jr and 'cov_ctx[1] == "after pixel"' in jr and "raise _oom" in jr,
      "R4 OOM with a learned reach -> _RefineRefit (after pixel only), otherwise the OOM as it was")
lr = _seg("_learn_reach")
_need('_rates_fit_learn(rsec, mpf, v["fit_ratio"])' in lr and 'v.get("level") == "warn"' in lr,
      "R4 _learn_reach: only an address warning (fit_ratio) is learned")
cal = SRC[SRC.find("for _try in (1, 2):"):]
cal = cal[:cal.find("_cover_report_refine(_jr_info, _cov_args")]
_need("_fit_capped_final_by(_cw, _ch, 1.0" in cal and "_lanczos_to(cur, _nw, _nh)" in cal
      and "if _try == 2 or _c is not None:" in cal and cal.count("_joint_refine(") == 1
      and "_jr_key = _cache_key(frames, vae, _um_key, (_nw, _nh), resize_method)" in cal,
      "R4 one call site, ONE retry at the fitting canvas (not after a cache hit), cache keyed to that canvas")
_need(SRC.count("_jr_info = _joint_refine(model, positive") == 2, "R4 still exactly two refine call sites (before / after)")

_need(all(b < 128 for b in Path(__file__).read_bytes()), "S  ASCII")
print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
