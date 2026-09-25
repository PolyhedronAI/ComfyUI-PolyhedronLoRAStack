#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v1010 -- ONE progress instrument for the long-running nodes (S1).

THE WOUND (Frank, 25.09.2026): "dann fehlen mir fuer unsere Nodes, die ein
wenig laenger brauchen: Interpolate oder VAE decode ... und andere dieser
gruene Progress-Balken oder auch in der Konsole, dass man sieht, was gerade
laeuft ... wo man noch etwas on the fly oder eben im zweiten Laufen messen
und anzeigen kann".

WHAT v1010 PROMISES, and where it can break:

  N1  NodeProgress (counting work) DRIVEN: the plan line says "no rate
      learned yet" on a first run and names the learned estimate on the
      second; the bar follows done/total (monotone, never above 1, full on a
      clean exit); the console speaks at most every say_every seconds with
      count, percent, rate and ETA; quiet=True prints nothing; a clean run
      learns seconds per (unit x size); an EXCEPTION learns nothing.
  B1  blocking (one call) DRIVEN: the heartbeat prints every period with
      elapsed and -- once a rate is learned -- what is left of the estimate;
      the bar follows elapsed / estimate but NEVER passes 95 % before the call
      returns, and is full after it; bar=False leaves the bar alone (Core
      drives it); the rate and the peak are learned on a clean exit only.
  R1  the shared file is <user>/polyhedron/rates.json; v1008/v1009's
      pu_rates.json seeds it when it does not exist yet.
  W1  the VAE Codec runs encode / decode (full and tiled) / audio decode
      inside `blocking`, per VAE class and lane; a 2-D tiled pass leaves the
      bar to Core. Interpolate loads its model inside `blocking` and counts
      its tasks with NodeProgress (bar + ETA on the console line).
  W2  Power Upscale takes its clock from ph_progress (re-export), not from
      a private copy.
"""

import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODES = os.path.join(ROOT, "nodes")
NAME = "test_v1010_progress"
_fails = []


def _fail(msg):
    _fails.append(msg)
    print("  - " + msg)


def _need(cond, msg):
    if not cond:
        _fail(msg)


def _read(*p):
    with open(os.path.join(ROOT, *p), "r", encoding="utf-8") as fh:
        return fh.read()


class _Bar:
    log = []

    def __init__(self, total, node_id=None):
        self.total = total
        _Bar.log.append(("new", total))

    def update_absolute(self, value, total=None, preview=None):
        _Bar.log.append(("set", value, total))


def _load(tmp):
    comfy = types.ModuleType("comfy")
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = _Bar
    comfy.utils = utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = utils
    fp = types.ModuleType("folder_paths")
    fp.get_user_directory = lambda: tmp
    sys.modules["folder_paths"] = fp
    pkg = types.ModuleType("plsv1010")
    pkg.__path__ = [NODES]
    sys.modules["plsv1010"] = pkg
    mods = {}
    for name in ("ph_runclock", "ph_progress"):
        spec = importlib.util.spec_from_file_location("plsv1010." + name, os.path.join(NODES, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "plsv1010"
        sys.modules["plsv1010." + name] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods["ph_progress"]


def _sets():
    return [e[1] / float(e[2]) for e in _Bar.log if e[0] == "set"]


def main():
    print("[%s]" % NAME)
    tmp = tempfile.mkdtemp(prefix="pls_v1010_")
    pp = _load(tmp)

    # ---- N1 ----------------------------------------------------------------
    del _Bar.log[:]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pp.NodeProgress("Save", "save:test", total=10, unit="frame", size=2.0, say_every=0.05) as p:
            for _ in range(10):
                time.sleep(0.02)
                p.tick()
    out = buf.getvalue()
    _need("10 frames -- estimate no rate learned yet" in out, "%s: N1 first run says no rate yet: %r" % (NAME, out[:120]))
    _need("frames (" in out and " %) " in out and "left" in out and ("frame/s" in out or "s/frame" in out),
          "%s: N1 the progress line carries count, percent, rate and ETA: %r" % (NAME, out))
    n_lines = out.count("[PLS] Save:   ")
    _need(1 <= n_lines <= 6, "%s: N1 the console is throttled (got %d progress lines for 10 ticks)" % (NAME, n_lines))
    vals = _sets()
    _need(vals == sorted(vals) and max(vals) <= 1.0 and abs(vals[-1] - 1.0) < 1e-9,
          "%s: N1 the bar is monotone, never above 1, full at the end: %s" % (NAME, vals[-3:]))
    _need("10 frames in" in out and "rate learned (1 run(s))" in out, "%s: N1 a clean run learns" % NAME)
    r = pp._rates_load("save:test")
    _need(r["step"] is not None and 0.005 < r["step"] < 0.1,
          "%s: N1 the rate is seconds per (frame x size): %s" % (NAME, r["step"]))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pp.NodeProgress("Save", "save:test", total=20, unit="frame", size=2.0) as p:
            _need(p.est_total is not None and abs(p.est_total - r["step"] * 40.0) < 1e-6,
                  "%s: N1 the second run opens with the learned estimate" % NAME)
            p.tick(20)
    _need("estimate ~" in buf.getvalue() and "(learned)" in buf.getvalue(), "%s: N1 the second plan line names it" % NAME)
    runs_before = pp._rates_load("save:test")["runs"]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            with pp.NodeProgress("Save", "save:test", total=5) as p:
                p.tick(2)
                raise RuntimeError("interrupted")
    except RuntimeError:
        pass
    _need(pp._rates_load("save:test")["runs"] == runs_before, "%s: N1 an interrupted run learns nothing" % NAME)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pp.NodeProgress("Q", "q:test", total=3, quiet=True) as p:
            p.tick(3)
    _need(buf.getvalue() == "", "%s: N1 quiet prints nothing" % NAME)

    # ---- B1 ----------------------------------------------------------------
    del _Bar.log[:]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pp.blocking("VAE", "vae:test", size=4.0, what="decode", period=0.1) as b:
            time.sleep(0.35)
    out = buf.getvalue()
    _need("decode begin -- no rate learned yet" in out and out.count("decode ... ") >= 2
          and "no estimate yet" in out and "decode done in" in out,
          "%s: B1 first run: begin, heartbeats without an estimate, done: %r" % (NAME, out))
    _need(b.seconds is not None and b.seconds >= 0.3, "%s: B1 the call's seconds are measured" % NAME)
    del _Bar.log[:]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with pp.blocking("VAE", "vae:test", size=4.0, what="decode", period=0.05) as b:
            _need(b.est is not None and b.est > 0.25, "%s: B1 the second run has an estimate (%s)" % (NAME, b.est))
            time.sleep(0.5)          # longer than the estimate: the bar must still hold at 95 %
            held = _sets()
    out = buf.getvalue()
    _need("left of ~" in out and "(learned)" in out, "%s: B1 the heartbeat says what is left of the estimate" % NAME)
    _need(held and max(held) <= 0.95 + 1e-9, "%s: B1 the bar never passes 95 %% before the call returns: %s" % (NAME, held[-3:]))
    _need(abs(_sets()[-1] - 1.0) < 1e-9, "%s: B1 ...and is full after it" % NAME)
    del _Bar.log[:]
    with contextlib.redirect_stdout(io.StringIO()):
        with pp.blocking("VAE", "vae:test2", size=1.0, what="x", period=0.05, bar=False):
            time.sleep(0.12)
    _need(_Bar.log == [], "%s: B1 bar=False leaves the bar to Core: %s" % (NAME, _Bar.log[:3]))
    runs = pp._rates_load("vae:test")["runs"]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            with pp.blocking("VAE", "vae:test", size=4.0, what="decode", period=5.0):
                raise MemoryError("oom")
    except MemoryError:
        pass
    _need(pp._rates_load("vae:test")["runs"] == runs, "%s: B1 a failed call learns nothing" % NAME)

    # ---- R1 ----------------------------------------------------------------
    rf = os.path.join(tmp, "polyhedron", "rates.json")
    _need(os.path.isfile(rf) and "vae:test" in _read(rf) and "save:test" in _read(rf),
          "%s: R1 one shared rates.json holds every section" % NAME)
    tmp2 = tempfile.mkdtemp(prefix="pls_v1010_seed_")
    os.makedirs(os.path.join(tmp2, "polyhedron"))
    with open(os.path.join(tmp2, "polyhedron", "pu_rates.json"), "w") as fh:
        json.dump({"sections": {"tile:WAN22": {"step": 0.25, "runs": 4}}}, fh)
    pp2 = _load(tmp2)
    _need(abs((pp2._rates_load("tile:WAN22")["step"] or 0) - 0.25) < 1e-9,
          "%s: R1 the v1008/v1009 pu_rates.json seeds the shared file" % NAME)

    # ---- W1 / W2: the nodes are wired --------------------------------------
    vae = _read("nodes", "ph_vae.py")
    _need('with _blocking("VAE", "vae:encode:%s:%s" % (_model_key(vae), verdict),' in vae
          and 'with _blocking("VAE", "vae:decode:%s:full" % _model_key(vae),' in vae
          and 'with _blocking("VAE", "vae:decode:%s:tiled" % _model_key(vae),' in vae
          and 'with _blocking("VAE", "vae:audio:%s" % _model_key(audio_vae),' in vae,
          "%s: W1 every VAE lane runs inside blocking" % NAME)
    # the CALL must sit in the with-body (AST), not merely near it in the text
    import ast as _ast
    want = {"vae:encode:%s:%s": ("encode", "encode_tiled"), "vae:decode:%s:full": ("decode",),
            "vae:decode:%s:tiled": ("decode_tiled",), "vae:audio:%s": ("decode",)}
    seen = {}
    for node in _ast.walk(_ast.parse(vae)):
        if isinstance(node, _ast.With):
            head = _ast.dump(node.items[0].context_expr)
            for sec, calls in want.items():
                if repr(sec)[1:-1] in head or sec in head:
                    body = "".join(_ast.dump(b) for b in node.body)
                    seen[sec] = any("attr='%s'" % c in body for c in calls)
    _need(len(seen) == 4 and all(seen.values()),
          "%s: W1 every VAE call runs INSIDE its blocking body: %s" % (NAME, seen))
    _need('bar=(getattr(latent, "ndim", 4) != 4)' in vae, "%s: W1 a 2-D tiled decode leaves the bar to Core" % NAME)
    ip = _read("nodes", "ph_interpolate.py")
    _need('with _blocking("Interpolate", "vfi:load:%s" % ckpt_name' in ip
          and "model = _load_model(ckpt_name, arch_ver, precision, device)" in ip,
          "%s: W1 Interpolate loads its model inside blocking" % NAME)
    _need("prog = _NodeProgress(\"Interpolate\"" in ip and "prog.tick(len(batch))" in ip
          and "prog.__exit__(None, None, None)" in ip and "-- ~%s left" in ip,
          "%s: W1 Interpolate counts its tasks (bar + ETA)" % NAME)
    i_t = ip.index("prog.tick(len(batch))")
    i_d = ip.index("done += len(batch)")
    _need(i_d < i_t < ip.index("wall = _now() - t_start"), "%s: W1 the tick follows each finished chunk" % NAME)
    pu = _read("nodes", "ph_power_upscale.py")
    _need("from .ph_progress import (_HEARTBEAT_S" in pu and "\nclass _Phase" not in pu
          and "\ndef _rates_learn" not in pu,
          "%s: W2 Power Upscale takes its clock from ph_progress (no private copy)" % NAME)

    if _fails:
        print("%s: %d failure(s)" % (NAME, len(_fails)))
        return 1
    print("%s: PASS -- N1 counting progress, B1 blocking heartbeat, R1 shared rates, W1/W2 wiring" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
