# -*- coding: ascii -*-
"""Guard v1210 -- the restore runs like the joint refine: a plan with learned estimates, heartbeats, no chatter.

(Frank 21.10., U5 log: a "Model SeedVR2 prepared ..." line, a tqdm bar and a chunk line per chunk -- "dieses
Rumgehopse mag ich nicht ... das sollte sauber zusammengefasst werden ... generell mal gucken, ob wir hier kein ETA
einbinden koennen". "Dann bau mal das Update ... go.")

  P1 count_chunks (the plan, before the encode) == what split_latent makes, over many clips / chunks / overlaps
  P2 plan_banner: with learned rates -> three phases with estimates, the total and the learned rungs named;
     without -> "no rate learned yet" (never an invented number)
  P3 learned_estimates: the learned seconds per megapixel-frame x the size; {} without rates
  P4 restore(): the sampler's bar is off, the staging chatter is muted with the node's switch, the sampling is ONE
     heartbeat phase with ONE summary line, the three phases are measured on the run clock, the rates are learned
     from the rung that RAN (not the failed ones), and a closing line names the plan
  P5 the node posts the three restore phases on its run clock up front with the learned priors, and hands the
     restore its clock, its node id, its mute switch and the picked model as the rates key
  Z  ASCII
Script-style: exit 0 = pass.
"""
import importlib
import os
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


CORE = os.environ.get("PLS_CORE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, CORE)
sys.path.insert(0, str(ROOT / "nodes"))
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
fp = types.ModuleType("folder_paths")
for _n in ("get_temp_directory", "get_output_directory", "get_input_directory", "get_user_directory"):
    setattr(fp, _n, lambda: "/tmp/pls1210_none")
fp.get_filename_list = lambda *a, **k: []
fp.get_full_path = lambda *a, **k: None
fp.models_dir = "/tmp/models"
fp.add_model_folder_path = lambda *a, **k: None
fp.get_folder_paths = lambda f: ["/tmp/models/" + f]
sys.modules.setdefault("folder_paths", fp)
R = importlib.import_module("ph_restore")

# ---- P1
bad = []
for t_lat in (1, 2, 3, 5, 9, 18, 36, 37):
    for fpc in (1, 5, 9, 13, 21, 33):
        for ov in (0, 1, 2, 3):
            lat = torch.zeros(1, 1, t_lat, 1, 1)
            got = len(R.split_latent(lat, fpc, ov)[0])
            if R.count_chunks(t_lat, fpc, ov) != got:
                bad.append((t_lat, fpc, ov, got))
_need(not bad, "P1 count_chunks == split_latent over %d combinations (%s)" % (8 * 6 * 4, bad[:3]))
_need(R.count_chunks(36, 9, 1) == 18, "P1 U4's clip: 36 latent frames, 9 per chunk, overlap 1 -> 18 chunks")

# ---- P2
rates = {"enc": 0.675, "step": 0.125, "dec": 1.09, "runs": 2, "when": "2026-10-21 18:41"}
lines, est = R.plan_banner("seedvr2", 141, 1056, 1856, 9, 18, rates, (0, 3))
txt = "\n".join(lines)
mpf = 141 * 1056 * 1856 / 1e6
_need(abs(est["enc"] - 0.675 * mpf) < 1e-6 and abs(est["dec"] - 1.09 * mpf) < 1e-6, "P2 estimates = rate x megapixel-frames")
_need("=== RESTORE (seedvr2) === 141 frames 1056x1856" in txt and "18 chunk(s) of 9 frames" in txt,
      "P2 the head names frames, size and chunks")
_need("phase 3/3 decode   ~" in txt and "tiles 128 px / 8 frames, learned" in txt and "total ~" in txt
      and "2 run(s)" in txt, "P2 three phases, the learned decode rung, the total and the runs")
_need(("total %s" % R._clock_fmt(est["enc"] + est["step"] + est["dec"])) in txt, "P2 the total is the sum of the phases")
lines, est = R.plan_banner("seedvr2", 141, 1056, 1856, 9, 18, {}, (0, 0))
txt = "\n".join(lines)
_need("no rate learned yet" in txt and "~" not in txt.split("phase 1/3")[1].split("\n")[0], "P2 without rates: no number")

# ---- P3
_real = R._prog


class _P:
    @staticmethod
    def _rates_load(section):
        return {"enc": 2.0, "step": None, "dec": 4.0} if section == "restore:m.safetensors" else {}


R._prog = lambda: _P
e = R.learned_estimates("m.safetensors", 10, 1000, 1000)
_need(e == {"enc": 20.0, "dec": 40.0}, "P3 learned seconds x size, unlearned phases left out (%s)" % e)
_need(R.learned_estimates("other", 10, 1000, 1000) == {}, "P3 another model: {}")
R._prog = _real

# ---- P4 (statics on the module)
src = (ROOT / "nodes" / "ph_restore.py").read_text()
sc = src[src.index("def sample_chunk("):src.index("def color_match(")]
_need("disable_pbar=True" in sc, "P4 the sampler's own bar is off")
body = src[src.index("def restore(model"):src.index("def verdict(")]
_need("MuteStagingLogs(bool(mute)" in body and "with mute_cm:" in body, "P4 the staging chatter is muted with the switch")
loop = body[body.index("for i, ch in enumerate(chunks):"):body.index("t_sample = time.monotonic() - ts0")]
_need("print(" not in loop and 'with _phase("sample"' in body, "P4 the sampling is one heartbeat phase, no line per chunk")
_need(all(k in body for k in ('clock.measure("renc:restore"', 'clock.measure("rstep:restore"',
                              'clock.measure("rdec:restore"')), "P4 the three phases are measured on the run clock")
_need('LAST_STEP_S.get("encode"), t_sample, LAST_STEP_S.get("decode")' in body,
      "P4 the rates are learned from the rung that ran")
vs = src[src.index("def vae_step("):src.index("def count_chunks(")]
_need("LAST_STEP_S[op] = time.monotonic() - ts" in vs and vs.index("LAST_STEP_S[op]") > vs.index("out = fn("),
      "P4 LAST_STEP_S holds the seconds of the successful call only")
_need("restore finished in" in body and "(plan %s)" in body, "P4 a closing line names the plan")

# ---- P5
pu = (ROOT / "nodes" / "ph_power_upscale.py").read_text()
_need(all('"%s:restore"' % k in pu for k in ("renc", "rstep", "rdec")) and "_RS.learned_estimates(" in pu
      and "clock.prior(_k, _rs_est[_e])" in pu, "P5 the node posts the restore phases with learned priors up front")
call = pu[pu.index("out, info = _RS.restore("):][:600]
_need(all(k in call for k in ("clock=clock", "node_id=_current_node_id()", "mute=bool(mute_staging_logs)",
                              "rates_key=str(restore_model)")), "P5 the node hands clock, node id, mute and the rates key")

# ---- Z
for f in ("nodes/ph_restore.py", "tests/test_v1210_restore_console.py"):
    try:
        (ROOT / f).read_bytes().decode("ascii")
        _need(True, "Z ASCII " + f)
    except UnicodeDecodeError:
        _need(False, "Z ASCII " + f)

print("%s: %d check(s) failed" % ("test_v1210_restore_console", len(FAILS)) if FAILS else "test_v1210_restore_console: all checks passed")
sys.exit(1 if FAILS else 0)
