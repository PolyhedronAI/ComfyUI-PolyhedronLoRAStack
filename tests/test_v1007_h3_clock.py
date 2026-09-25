#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v1007 -- the H3 clock: no silent phase, a plan up front, rates that learn.

THE WOUND (Frank, 24.09.2026, v1006 field run, four screenshots): "final pass
done" -> 126 s of nothing -> "h3 refine begin" -> one 60 s step with no tick
-> 80 s of nothing -> "done". Console, node HUD and progress bar were dark
for minutes; "Etwa passiert, keiner weiss, was".

WHAT v1007 PROMISES, and where it can break:

  P1  _phase_plan (pure): three phase lines + a total; with rates the estimates
      are rate * megapixel-frames; without rates every line says so ("no
      rate yet") and the total names the heartbeat; a cache hit says the
      encode is skipped and costs 0 in the total.
  R1  _rates_learn: first run stores the raw rates; the second run folds
      in by EMA 0.5; a phase that did not run (None) leaves its rate alone;
      the file in the user directory is written and read back by
      _rates_load (survives a restart) -- driven in a temp directory.
  H1  _Phase: DRIVEN with a real thread and a short period -- it prints at
      entry and then every period; the line carries elapsed, and "left of"
      when an estimate exists / "no estimate yet" when none; it sends the HUD
      event with stage 'h3', the phase name and phase_elapsed/phase_left and
      NO jpeg; it pushes the clock; it stops on exit (no tick after).
  H2  _h3_refine: the encode and the decode run INSIDE an _Phase; every
      sampling step opens its own phase and the callback closes it; the plan
      banner is printed BEFORE the encode; the rates are learned at the end
      with enc None on a cache hit (source pins at the effect).
  J1  the HUD: stage 'h3' is known (STAGE_MARK), a heartbeat without jpeg
      never touches the tile picture, the text carries the phase and its
      clock, the map label reads "1 latent".

v1008 RE-GROUNDING (declared): the clock is general now -- _Phase (stage and
console label are parameters, default 'joint'), rates live in SECTIONS of
pu_rates.json ('joint:<model class>' / 'tile:<model class>'), and v1007's flat
h3_rates.json seeds a joint section that has not learned yet (R2 below). The
HUD mark of the whole-clip stage is 'AV' (joint); 'h3' stays known for old
events. Every promise stands; only names, the file and the section moved.
"""

import io
import os
import re
import sys
import tempfile
import time
import types
import contextlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _lift          # noqa: E402
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, "nodes", "ph_power_upscale.py")
PROG = os.path.join(ROOT, "nodes", "ph_progress.py")   # v1010: the clock moved there (declared)
JS = os.path.join(ROOT, "web", "js", "ph_power_upscale.js")
NAME = "test_v1007_h3_clock"
_fails = []


def _fail(msg):
    _fails.append(msg)
    print("  - " + msg)


def _need(cond, msg):
    if not cond:
        _fail(msg)


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


class _Clock:
    def __init__(self):
        self.pushes = 0

    def push(self):
        self.pushes += 1

    def elapsed(self):
        return 123.0

    def eta(self, *a):
        return 45.0


def main():
    print("[%s]" % NAME)
    src = _read(PY)
    js = _read(JS)
    tmp = tempfile.mkdtemp(prefix="pls_v1007_")
    sent = []
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(
        send_sync=lambda ev, d, sid=None: sent.append((ev, d))))
    sys.modules["server"] = server
    provided = {"os": os, "json": __import__("json"), "time": time, "threading": __import__("threading"),
                "folder_paths": types.SimpleNamespace(get_user_directory=lambda: tmp),
                "_fmt_clock": lambda s: "%d:%02d" % (int(s) // 60, int(s) % 60),
                "print": print,
                "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))}
    # v1010 RE-GROUNDING (declared): the clock lives in nodes/ph_progress.py
    # now (shared by every long-running node); the file is rates.json.
    code, missing = _lift.close_over(
        _read(PROG), ["_phase_plan", "_rates_load", "_rates_learn", "_rates_path", "_Phase",
              "_RATES", "_RATES_LOADED", "_HEARTBEAT_S", "_RATES_EMA", "_HUD_EVENT",
              "_rates_legacy_path", "_rates_v1008_path", "_rates_blank", "_rates_clean", "_RATES_LEGACY"],
        provided=set(provided))
    _need(not missing, "%s: lift not closed: %s" % (NAME, sorted(missing)))
    ns = dict(provided)
    exec(compile(code, "<v1007-lift>", "exec"), ns)

    # ---- P1 ---------------------------------------------------------------
    plan = ns["_phase_plan"]
    lines, est = plan(188.0, 1, False, {"enc": None, "step": None, "dec": None, "runs": 0, "when": ""})
    _need(len(lines) == 4 and all("no rate yet" in ln for ln in lines[:3]) and "measures them" in lines[3],
          "%s: P1 without rates every phase says 'no rate yet' and the total names the measuring run: %s" % (NAME, lines))
    _need(all(v is None for v in est.values()), "%s: P1 no rates -> no estimates" % NAME)
    rates = {"enc": 0.67, "step": 0.32, "dec": 0.43, "runs": 2, "when": "2026-09-24 17:26"}
    lines, est = plan(100.0, 3, False, rates)
    _need(abs(est["enc"] - 67.0) < 1e-9 and abs(est["step"] - 32.0) < 1e-9 and abs(est["dec"] - 43.0) < 1e-9,
          "%s: P1 estimates are rate * mpf: %s" % (NAME, est))
    _need("~1:07" in lines[0] and "3 steps" in lines[1] and "~0:32/step" in lines[1] and "~0:43" in lines[2],
          "%s: P1 the lines carry the phase estimates: %s" % (NAME, lines))
    _need("total ~3:26" in lines[3] and "2 run(s)" in lines[3], "%s: P1 total = 67 + 3*32 + 43 = 206 s: %s" % (NAME, lines[3]))
    lines, est = plan(100.0, 1, True, rates)
    _need("skipped (cache)" in lines[0] and est["enc"] == 0.0 and "total ~1:15" in lines[3],
          "%s: P1 a cache hit skips the encode in the plan and the total: %s" % (NAME, lines))

    # ---- R1 (v1008: a SECTION of pu_rates.json) ---------------------------
    learn, load = ns["_rates_learn"], ns["_rates_load"]
    sec = "joint:TestModel"
    r = learn(sec, 100.0, 60.0, 30.0, 40.0)
    _need(abs(r["enc"] - 0.6) < 1e-9 and abs(r["step"] - 0.3) < 1e-9 and abs(r["dec"] - 0.4) < 1e-9 and r["runs"] == 1,
          "%s: R1 the first run stores the raw rates: %s" % (NAME, r))
    r = learn(sec, 100.0, None, 50.0, 40.0, peak_gb=11.5)
    _need(abs(r["enc"] - 0.6) < 1e-9, "%s: R1 a phase that did not run (None) leaves its rate alone" % NAME)
    _need(abs(r["step"] - 0.4) < 1e-9, "%s: R1 EMA 0.5: 0.5*0.5 + 0.5*0.3 = 0.4, got %s" % (NAME, r["step"]))
    _need(r["runs"] == 2, "%s: R1 runs count up" % NAME)
    _need(r["peak_gb"] == 11.5 and r["peak_mpf"] == 100.0, "%s: R1 the peak is kept WITH the size it was measured at" % NAME)
    r_t = learn("tile:TestModel", 10.0, 1.0, 1.0, 1.0)
    _need(abs(r_t["step"] - 0.1) < 1e-9 and abs(load(sec)["step"] - 0.4) < 1e-9,
          "%s: R1 sections are independent (tile rates never bleed into joint)" % NAME)
    path = os.path.join(tmp, "polyhedron", "rates.json")
    _need(os.path.isfile(path), "%s: R1 the rates file is written in the user directory" % NAME)
    # a fresh load must read the file back
    ns["_RATES"].clear()
    exec("_RATES_LOADED = False", ns)
    r2 = load(sec)
    _need(r2["runs"] == 2 and abs(r2["step"] - 0.4) < 1e-9 and r2["peak_gb"] == 11.5,
          "%s: R1 the file is read back on load (survives a restart): %s" % (NAME, r2))
    # ---- R2: v1007's flat h3_rates.json seeds an unlearned joint section --
    tmp2 = tempfile.mkdtemp(prefix="pls_v1008_legacy_")
    os.makedirs(os.path.join(tmp2, "polyhedron"))
    with open(os.path.join(tmp2, "polyhedron", "h3_rates.json"), "w") as fh:
        fh.write('{"enc": 0.67, "step": 0.32, "dec": 0.43, "runs": 3, "when": "2026-09-24 17:26"}')
    ns["folder_paths"] = types.SimpleNamespace(get_user_directory=lambda: tmp2)
    ns["_RATES"].clear()
    exec("_RATES_LOADED = False", ns)
    rl = load("joint:MiniMaxH3")
    _need(abs(rl["step"] - 0.32) < 1e-9 and rl["runs"] == 3,
          "%s: R2 the v1007 rates seed the joint section -- no measurement is lost: %s" % (NAME, rl))
    _need(load("tile:WAN22")["step"] is None, "%s: R2 the legacy rates never seed a TILE section" % NAME)
    ns["folder_paths"] = types.SimpleNamespace(get_user_directory=lambda: tmp)

    # ---- H1: the heartbeat, driven ----------------------------------------
    Phase = ns["_Phase"]
    clk = _Clock()
    buf = io.StringIO()
    del sent[:]
    with contextlib.redirect_stdout(buf):
        with Phase("encode", 10.0, clk, node_id=42, canvas=(1872, 1072), period=0.15):
            time.sleep(0.5)
        time.sleep(0.4)   # after exit: nothing more may tick
    out = buf.getvalue().strip().splitlines()
    _need(len(out) >= 3, "%s: H1 entry tick + at least two period ticks in 0.5 s at period 0.15, got %d" % (NAME, len(out)))
    _need(all("joint encode ..." in ln and "elapsed" in ln and "left of ~0:10" in ln for ln in out),
          "%s: H1 every line carries elapsed and 'left of' the estimate: %s" % (NAME, out[:2]))
    n_after = len(out)
    _need(clk.pushes == n_after, "%s: H1 the clock is pushed once per tick (%d vs %d)" % (NAME, clk.pushes, n_after))
    _need(len(sent) == n_after, "%s: H1 one HUD event per tick" % NAME)
    if sent:
        ev, d = sent[0]
        _need(ev == "polyhedron.pu_tile" and d["stage"] == "joint" and d["phase"] == "encode" and "jpeg" not in d,
              "%s: H1 the HUD event carries stage joint + phase and NO jpeg: %s" % (NAME, {k: d.get(k) for k in ('stage', 'phase')}))
        _need(d["node"] == "42" and d["canvas"] == [1872, 1072] and isinstance(d["phase_elapsed"], int) and isinstance(d["phase_left"], int),
              "%s: H1 the event carries node, canvas, phase_elapsed and phase_left" % NAME)
        _need(d["elapsed"] == 123 and d["eta"] == 45, "%s: H1 the run clock rides along (elapsed/eta from the clock)" % NAME)
    # after exit: no more ticks
    n_before = len(sent)
    time.sleep(0.4)
    _need(len(sent) == n_before, "%s: H1 the thread stops on exit" % NAME)
    # no estimate -> says so; a step phase names its step
    buf = io.StringIO(); del sent[:]
    with contextlib.redirect_stdout(buf):
        with Phase("sample", None, clk, node_id=42, canvas=(8, 8), step=2, steps=3, period=5.0):
            pass
    ln = buf.getvalue().strip()
    _need("no estimate yet" in ln and "step 2/3" in ln, "%s: H1 without an estimate the line says so and a step names itself: %s" % (NAME, ln))
    _need(sent and sent[0][1]["step"] == 2 and sent[0][1]["steps"] == 3 and sent[0][1]["phase_left"] is None,
          "%s: H1 the step event carries step/steps and no phase_left" % NAME)

    # ---- H1b (v1008): a TILE phase -- its own stage/label, the tile's rect,
    #      and announce=False keeps the entry tick off the console ----------
    buf = io.StringIO(); del sent[:]
    with contextlib.redirect_stdout(buf):
        with Phase("sample", 5.0, clk, node_id=7, canvas=(1024, 768), step=1, steps=3,
                   period=5.0, stage="high", label="high tile 2", tile=2, tiles=4,
                   rect=(512, 0, 512, 768), announce=False):
            pass
    _need(buf.getvalue().strip() == "", "%s: H1b announce=False prints nothing at entry" % NAME)
    _need(sent and sent[0][1]["stage"] == "high" and sent[0][1]["tile"] == 2 and sent[0][1]["tiles"] == 4
          and sent[0][1]["rect"] == [512, 0, 512, 768],
          "%s: H1b the tile phase's HUD event carries its stage, tile count and rect" % NAME)

    # ---- H2: source pins on _joint_refine -------------------------------------
    ref = src[src.index("def _joint_refine("):src.index("class ULSPowerUpscale")]
    i_plan = ref.index("plan_lines, est = _phase_plan(")
    i_banner = ref.index("=== JOINT REFINE (video+audio model) ===")
    i_enc = ref.index("vid_lat = v_enc(src[:, :, :, :3])")
    i_encph = ref.index('with _Phase("encode", est["enc"], clock, node_id, (sw, sh)) as _pe:')
    i_decph = ref.index('with _Phase("decode", est["dec"], clock, node_id, (sw, sh)) as _pd:')
    i_dec = ref.index("px = v_dec(vid_out)")
    i_learn = ref.index("learned = _rates_learn(rsec, mpf, enc_s,")
    _need(i_plan < i_banner < i_encph < i_enc, "%s: H2 plan, banner, then the encode inside its phase" % NAME)
    _need(i_decph < i_dec, "%s: H2 the decode runs inside its phase" % NAME)
    _need(i_dec < i_learn, "%s: H2 the rates are learned after the decode" % NAME)
    _need('ph = _Phase("sample", est["step"], clock, node_id, (sw, sh),' in ref
          and "_open_step(step + 2)" in ref and "_open_step(1)" in ref
          and "_open_step(steps_run + 99)" in ref,
          "%s: H2 every sampling step has its own phase: opened before the sampler, rolled over in the callback, closed in finally" % NAME)
    _need("enc_s = None" in ref and "enc_s = tt[1] - tt[0]" in ref,
          "%s: H2 a cache hit learns no encode rate (enc_s stays None)" % NAME)
    _need("joint rates learned (s per megapixel-frame" in ref, "%s: H2 the learned rates are said" % NAME)

    # ---- J1: the HUD --------------------------------------------------------
    _need(re.search(r'const STAGE_MARK = \{[^}]*joint: "AV"', js) is not None
          and re.search(r'const STAGE_MARK = \{[^}]*h3: "H3"', js) is not None,
          "%s: J1 STAGE_MARK knows joint (and h3 for old events)" % NAME)
    _need('if (d.jpeg) node._procTile.src = "data:image/jpeg;base64," + d.jpeg;' in js,
          "%s: J1 a heartbeat without jpeg never touches the tile picture" % NAME)
    _need('const phaseTxt = d.phase' in js and 'const wholeClip = (d.stage === "joint" || d.stage === "h3");' in js
          and "_fmtClock(d.phase_elapsed)" in js
          and "_fmtClock(d.phase_left)" in js and '(d.phase === "sample" ? " " + d.step + "/" + d.steps : "")' in js,
          "%s: J1 the HUD text carries phase, its elapsed/left and the step" % NAME)
    _need('node._procMapLbl.textContent = "1 latent";' in js, "%s: J1 the map label reads '1 latent' for h3" % NAME)
    _need("if (!d.node || (!d.jpeg && !d.phase)) return;" in js,
          "%s: J1 the event listener must let a picture-less heartbeat (phase) through -- the sandbox browser caught the old gate dropping every one" % NAME)

    if _fails:
        print("%s: %d failure(s)" % (NAME, len(_fails)))
        return 1
    print("%s: PASS -- P1 plan, R1 rates learn + persist, H1 heartbeat driven, H2 refine phases, J1 HUD" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
