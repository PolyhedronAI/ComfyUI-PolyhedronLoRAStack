#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v1008 -- ONE refine vocabulary for EVERY model (Power Upscale).

THE WOUND (Frank, 25.09.2026): "Warum heisst das eigentlich H3 Refine? Das
sollte fuer alle Modelle brauchbar bleiben, diese Node ... auch WAN 2.2 sollte
einen Refiner bedienen koennen, ebenso SD1 ... Wir bauen allgemein-gueltige
Nodes fuer alle Modelle." And: "Ich habe nichts gegen Kacheln, sofern das den
Speicher entlastet ... Wir muessen eben auch bei schwachen Karten
einsatzbereit bleiben ueber viele Frames beim Video."

v1004-v1007 built the exact start, the cache, the heartbeat and the order
choice for the JOINT branch only, under an H3 name. v1008 promises:

  G1  _sigma_run reads ANY wired curve: a flow curve (0..1) starts EXACTLY at
      denoise (_sigma_tail); an eps curve (SD1 karras, 14.6..0) is SLICED to
      the last round(n * denoise) steps -- 'denoise 0.30' is not a sigma
      there; denoise >= 1 runs it whole, <= 0 runs nothing.
  G2  the VAE is READ, not assumed: _vae_spatial (8 for an SD VAE, 16 for the
      H3 VAE, the default when a VAE states nothing) and _vae_frame_law (H3
      17k+5, Wan 4k+1, a still VAE (n, n)).
  G3  the verdict exists for EVERY model and names where the pixel model sat
      (_tile_verdict / _joint_verdict).
  G4  the PIXEL cache stores a CPU clone at full precision only within its
      RAM budget; over budget (or RAM unreadable) it stores nothing AND drops
      the old entry; the key reads content + wire + canvas + fit + kind.
  G5  _memory_note speaks only from a MEASURED pair (peak, size) and only when
      this run is larger or the last one sat at the card's limit.
  G6  _refine_tiles DRIVEN: with sig_run the sampler is sample_custom on
      EXACTLY that run (Core's sample() is not touched), the step count is the
      run's; an empty run samples nothing (a VAE round trip); without sig_run
      the old sample() path runs unchanged; with rsec every phase opens a
      heartbeat (HUD events carry the stage and the tile) and the rates are
      learned per tile.
  G7  upscale(): refine_order is read by EVERY model -- 'before pixel' turns
      'model + fit' / 'model only' into 'model final' for the tile path,
      'off' drops the stages and routes delivery through the final pass
      (plain fit without a wire); the curve reaches the stages (sig_run);
      the pixel cache serves ONLY the first stage's model pass; the joint
      refine reads vae_tiling; the ui carries the path.
  G8  INPUT_TYPES / frontend: refine_order + audio_stream in slots 27/28, the
      tooltips speak for every model (no 'H3 refine:' prefix left on sigmas),
      the HUD shows phases for tile stages too.
"""

import ast
import io
import contextlib
import os
import re
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _lift          # noqa: E402
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, "nodes", "ph_power_upscale.py")
PROG = os.path.join(ROOT, "nodes", "ph_progress.py")   # v1010: the clock moved there (declared)
JS = os.path.join(ROOT, "web", "js", "ph_power_upscale.js")
TM = os.path.join(ROOT, "nodes", "uls_tile_math.py")
NAME = "test_v1008_refine_general"
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
        self.measured = []

    def measure(self, key, dt):
        self.measured.append(key)

    def eta(self, *a):
        return 0.0

    def elapsed(self):
        return 0.0

    def push(self):
        pass


def main():
    print("[%s]" % NAME)
    src = _read(PY)
    js = _read(JS)
    import torch
    tmp = tempfile.mkdtemp(prefix="pls_v1008_")
    sent = []
    server = types.ModuleType("server")
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(
        send_sync=lambda ev, d, sid=None: sent.append((ev, d))))
    sys.modules["server"] = server

    calls = []

    def _sample(*a, **k):
        calls.append(("sample", a[2]))          # steps
        return a[8]                              # the latent passes through

    def _sample_custom(model, noise, cfg, sampler, sigmas, positive, negative,
                       latent, noise_mask=None, callback=None, disable_pbar=False, seed=None):
        calls.append(("sample_custom", [round(float(v), 6) for v in sigmas.tolist()], sampler))
        n = int(sigmas.shape[0]) - 1
        for i in range(n):
            if callback is not None:
                callback(i, latent, latent, n)
        return latent

    comfy = types.SimpleNamespace(
        sample=types.SimpleNamespace(prepare_noise=lambda lat, seed, inds=None: torch.zeros_like(lat),
                                     sample=_sample, sample_custom=_sample_custom),
        samplers=types.SimpleNamespace(sampler_object=lambda name: ("sampler", name)))
    tm_ns = {}
    exec(compile(_read(TM), TM, "exec"), tm_ns)
    uls_tile_math = types.SimpleNamespace(**{k: v for k, v in tm_ns.items() if not k.startswith("__")})
    provided = {"torch": torch, "time": __import__("time"), "os": os, "json": __import__("json"),
                "threading": __import__("threading"),
                "folder_paths": types.SimpleNamespace(get_user_directory=lambda: tmp),
                "_fmt_clock": lambda s: "%d:%02d" % (int(s) // 60, int(s) % 60),
                "comfy": comfy, "uls_tile_math": uls_tile_math,
                "_current_node_id": lambda: 42, "_free": lambda: None,
                "_lanczos_to": lambda f, w, h: f, "print": print}
    code, missing = _lift.close_over(
        src + "\n" + _read(PROG), ["_sigma_run", "_vae_spatial", "_vae_frame_law", "_tile_verdict", "_joint_verdict",
              "_pixel_cache_key", "_pixel_cache_get", "_pixel_cache_put", "_PIXEL_CACHE",
              "_memory_note", "_peak_line", "_refine_tiles", "_rates_section", "_RATES_LOADED"],
        provided=set(provided))
    _need(not missing, "%s: lift not closed: %s" % (NAME, sorted(missing)))
    ns = dict(provided)
    exec(compile(code, "<v1008-lift>", "exec"), ns)

    # ---- G1 ---------------------------------------------------------------
    run = ns["_sigma_run"]
    flow = [1.0, 0.994, 0.984, 0.966, 0.923, 0.835, 0.697, 0.469, 0.0]
    r, how = run(flow, 0.30)
    _need(how == "exact" and r == [0.30, 0.0], "%s: G1 flow curve starts EXACTLY at 0.30: %s %s" % (NAME, r, how))
    karras = [14.61, 7.49, 3.66, 1.66, 0.69, 0.25, 0.08, 0.03, 0.0]
    r, how = run(karras, 0.30)
    _need(how == "slice" and len(r) - 1 == round(8 * 0.30) and r == karras[-(round(8 * 0.30) + 1):],
          "%s: G1 an eps curve is sliced to the last round(8 * 0.30) steps, got %s %s" % (NAME, r, how))
    _need(run(karras, 0.01)[0] == karras[-2:], "%s: G1 a tiny denoise on an eps curve still runs ONE step" % NAME)
    _need(run(flow, 1.0) == (flow, "whole") and run(karras, 1.0)[1] == "whole", "%s: G1 denoise 1 runs the whole curve" % NAME)
    _need(run(flow, 0.0) == ([], "none"), "%s: G1 denoise 0 runs nothing" % NAME)

    # ---- G2 ---------------------------------------------------------------
    sp, law = ns["_vae_spatial"], ns["_vae_frame_law"]

    class _SD:
        downscale_ratio = 8

    class _H3:
        downscale_ratio = (lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16)
        upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)

    class _WAN:
        downscale_ratio = (lambda a: max(0, (a + 3) // 4), 8, 8)
        upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)

    _need(sp(_SD()) == 8 and sp(_H3()) == 16 and sp(object(), 16) == 16 and sp(object()) == 8,
          "%s: G2 spatial factor READ from the VAE (default only when it states none)" % NAME)
    _need(law(_H3(), 25) == (7, 22) and law(_H3(), 22) == (7, 22), "%s: G2 H3 frame law 17k+5 <-> 5k+2" % NAME)
    _need(law(_WAN(), 81) == (21, 81) and law(_WAN(), 80) == (20, 77), "%s: G2 Wan frame law 4k+1" % NAME)
    _need(law(_SD(), 33) == (33, 33), "%s: G2 a still VAE has no time law" % NAME)

    # ---- G3 ---------------------------------------------------------------
    tv, jv = ns["_tile_verdict"], ns["_joint_verdict"]
    t = tv("before pixel", 2, 5, 6, 0.30, 8, "behind")
    _need(t.startswith("Tile refine before pixel: 2 stages") and "5 tiles" in t and "6 steps" in t
          and "0.30" in t and "8-step curve" in t and "behind the last decode" in t,
          "%s: G3 the tile verdict names stages, tiles, steps, curve and the pixel place: %r" % (NAME, t))
    _need("(scheduler)" in tv("after pixel", 1, 1, 3, 0.0, 0, "front") and
          "in front of the VAE" in tv("after pixel", 1, 1, 3, 0.0, 0, "front"),
          "%s: G3 without a curve the verdict says 'scheduler'" % NAME)
    _need(tv("off", 0, 0, 0, 0.0, 0, "behind", why="refine_order=off").startswith("Refine OFF (refine_order=off)"),
          "%s: G3 an OFF tile run says so" % NAME)
    _need("behind the last decode" in jv("before pixel", True, 8, 1, 0.3, "keep", 3, 22, 22, where="behind"),
          "%s: G3 the joint verdict names the pixel place too" % NAME)

    # ---- G4 ---------------------------------------------------------------
    key_of, get, put, store = ns["_pixel_cache_key"], ns["_pixel_cache_get"], ns["_pixel_cache_put"], ns["_PIXEL_CACHE"]
    g = torch.Generator().manual_seed(3)
    a = torch.rand((5, 40, 48, 3), generator=g)
    um = types.SimpleNamespace(scale=4.0, model=types.SimpleNamespace())
    k0 = key_of(a, um, (96, 80), "lanczos (gpu)", "model + fit")
    _need(k0 == key_of(a.clone(), um, (96, 80), "lanczos (gpu)", "model + fit"), "%s: G4 same content -> same key" % NAME)
    _need(k0 != key_of(a, um, (96, 80), "bicubic", "model + fit") and k0 != key_of(a, um, (88, 80), "lanczos (gpu)", "model + fit")
          and k0 != key_of(a, um, (96, 80), "lanczos (gpu)", "model only") and k0 != key_of(a, None, (96, 80), "lanczos (gpu)", "model + fit"),
          "%s: G4 fit, canvas, kind and wire each change the key" % NAME)
    ok, why = put(k0, a, "high", avail=10 ** 10)
    _need(ok and get(k0) is not None and torch.equal(get(k0), a) and get(k0).dtype == a.dtype,
          "%s: G4 within budget: stored at full precision (%s)" % (NAME, why))
    a[0, 0, 0, 0] += 1.0
    _need(not torch.equal(get(k0), a), "%s: G4 the store is a CLONE" % NAME)
    ok2, why2 = put("other", a, "high", avail=int(a.numel() * a.element_size() * 2))
    _need(not ok2 and "not stored" in why2 and store["frames"] is None and get(k0) is None,
          "%s: G4 over budget: NOTHING stored, and the old entry is dropped (%s)" % (NAME, why2))
    store["key"], store["frames"] = None, None
    ns["_ram_available"] = lambda: None
    ok3, why3 = put(k0, a, "high")
    _need(not ok3 and "unreadable" in why3, "%s: G4 RAM unreadable -> not stored, said (%s)" % (NAME, why3))

    # ---- G5 ---------------------------------------------------------------
    mn = ns["_memory_note"]
    blank = {"peak_gb": None, "peak_mpf": None}
    _need(mn(blank, 100.0, 16.0, "x") is None, "%s: G5 nothing measured -> nothing said" % NAME)
    _need(mn({"peak_gb": 9.0, "peak_mpf": 100.0}, 100.0, 16.0, "x") is None, "%s: G5 same size, room left -> silent" % NAME)
    n1 = mn({"peak_gb": 9.0, "peak_mpf": 100.0}, 180.0, 16.0, "LEVERS")
    _need(n1 is not None and "9.0 GB" in n1 and "16.0 GB" in n1 and "x1.80" in n1 and "LEVERS" in n1,
          "%s: G5 a larger run names the measured pair, the ratio and the levers: %r" % (NAME, n1))
    _need(mn({"peak_gb": 15.2, "peak_mpf": 100.0}, 100.0, 16.0, "x") is not None, "%s: G5 a peak at the limit is said even at equal size" % NAME)
    pl = ns["_peak_line"]
    _need(pl("stage=high", [("encode", None), ("sample", None)], 16.0) is None, "%s: G5 no peaks -> no line" % NAME)
    line = pl("stage=high", [("encode", 2.0), ("sample", 15.0), ("decode", 3.0)], 16.0)
    _need(line is not None and "sample 15.0 GB" in line and "of 16.0 GB" in line and "limit" in line,
          "%s: G5 the peak line names phases, the card and the limit: %r" % (NAME, line))

    # ---- G6: _refine_tiles DRIVEN -------------------------------------------
    rt = ns["_refine_tiles"]

    class _VAE:
        def encode(self, px):
            return torch.zeros((1, 4, int(px.shape[1]) // 8, int(px.shape[2]) // 8))

        def decode(self, lat):
            n = 3
            return torch.full((n, int(lat.shape[2]) * 8, int(lat.shape[3]) * 8, 3), 0.5)
    img = torch.rand((3, 64, 64, 3))
    grid = uls_tile_math.plan_grid(64, 64, 64, 8)
    del calls[:]
    del sent[:]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = rt(None, [], [], _VAE(), img, grid, 1, 99, 1.0, "euler", "simple", 0.3,
                 _Clock(), "high", sig_run=[0.30, 0.15, 0.0], rsec="tile:TestModel")
    _need([c[0] for c in calls] == ["sample_custom"] and calls[0][1] == [0.3, 0.15, 0.0],
          "%s: G6 with sig_run: sample_custom on EXACTLY the run, sample() untouched: %s" % (NAME, calls))
    _need(calls and calls[0][2] == ("sampler", "euler"), "%s: G6 the sampler dial still picks the sampler" % NAME)
    _need("step 2/2" in buf.getvalue() and "step 3/" not in buf.getvalue(),
          "%s: G6 the step count is the run's (2), not the dial's (99)" % NAME)
    _need(tuple(out.shape) == (3, 64, 64, 3), "%s: G6 frames come back [N,H,W,3]" % NAME)
    phases = sorted(set(d.get("phase") for _e, d in sent))
    _need(phases == ["decode", "encode", "sample"] and all(d["stage"] == "high" and d["tiles"] == 1 for _e, d in sent),
          "%s: G6 every phase opens a heartbeat with the stage and the tile: %s" % (NAME, phases))
    rfile = os.path.join(tmp, "polyhedron", "rates.json")   # v1010: the shared file
    _need(os.path.isfile(rfile) and "tile:TestModel" in _read(rfile), "%s: G6 the tile rates are learned and remembered" % NAME)
    del calls[:]
    with contextlib.redirect_stdout(io.StringIO()) as b2:
        rt(None, [], [], _VAE(), img, grid, 1, 99, 1.0, "euler", "simple", 0.0,
           _Clock(), "high", sig_run=[], rsec=None)
    _need(calls == [] and "VAE round trip" in b2.getvalue(), "%s: G6 an empty run samples nothing and says so" % NAME)
    del calls[:]
    with contextlib.redirect_stdout(io.StringIO()):
        rt(None, [], [], _VAE(), img, grid, 1, 4, 1.0, "euler", "simple", 0.3, _Clock(), "single")
    _need(calls == [("sample", 4)], "%s: G6 without sig_run the old sample() path runs with the dialled steps: %s" % (NAME, calls))

    # ---- G7: upscale() pins ----------------------------------------------------
    up = src[src.index("    def upscale(self"):]
    _need('if _joint is None and _order == "before pixel":' in up and 'pixel_stage = "model final"' in up,
          "%s: G7 tile path: 'before pixel' becomes 'model final'" % NAME)
    i_bp = up.index('if _joint is None and _order == "before pixel":')
    i_mf = up.index('pixel_stage = "model final"')
    i_fr = up.index("_final_runs = ")
    _need(i_bp < i_mf < i_fr, "%s: G7 the rebind happens BEFORE _final_runs reads the dial" % NAME)
    _need("_tile_off = (_joint is None and _order == \"off\")" in up
          and "drop_refine_stages(stages, (_joint is not None) or _tile_off)" in up,
          "%s: G7 'off' drops the tile stages" % NAME)
    _need("_final_runs = (str(pixel_stage) == \"model final\") or (_joint is not None) or _tile_off" in up,
          "%s: G7 'off' routes delivery through the final pass" % NAME)
    _need("if um_fin is None and _final_delivers:" in up and "_esrgan_pass(cur, None, tw, th, resize_method, per_batch," in up,
          "%s: G7 a delivering final pass without a wire is a plain fit to the dialled canvas" % NAME)
    _need('sig_run=st.get("sig_run"), rsec=_tile_rsec,' in up and 'st["sig_run"], st["sig_how"] = run, how' in up,
          "%s: G7 the wired curve reaches every stage" % NAME)
    _need("if um is not None and st is plans[0][0]:" in up, "%s: G7 the pixel cache serves ONLY the first stage's model pass" % NAME)
    _need(up.count("vae_tiling=vae_tiling)") == 2, "%s: G7 BOTH joint call sites hand vae_tiling to the joint refine" % NAME)
    _need('"pls_pu_path": ["joint" if _joint is not None else "tile"]' in up, "%s: G7 the ui carries the path" % NAME)
    ref = src[src.index("def _joint_refine("):src.index("class ULSPowerUpscale")]
    _need("v_enc, v_dec, v_label = _vae_ops(vae, vae_tiling)" in ref and "snap = _vae_spatial(vae, _JOINT_SNAP_DEFAULT)" in ref
          and "_vae_frame_law(vae, n_in)" in ref,
          "%s: G7 the joint refine READS the VAE (tiling, factor, frame law)" % NAME)

    # ---- G8: widgets, tooltips, frontend ---------------------------------------
    it = src[src.index("def INPUT_TYPES"):src.index("RETURN_TYPES")]
    names = re.findall(r'^\s{16}"([a-z_0-9]+)": \(', it[:it.index('"optional": {')], re.M)
    _need(names[-2:] == ["refine_order", "audio_stream"], "%s: G8 slots 27/28 carry the new names: %s" % (NAME, names[-2:]))
    _need("h3_refine" not in re.sub(r"#[^\n]*", "", it) and "H3 refine:" not in it,
          "%s: G8 no H3-only name or tooltip prefix left in INPUT_TYPES" % NAME)
    _need("for EVERY" in it[it.index('"refine_order"'):it.index('"audio_stream"')]
          and "for EVERY model" in it[it.index('"sigmas"'):],
          "%s: G8 the tooltips speak for every model" % NAME)
    _need('const phaseTxt = d.phase' in js and '(wholeClip ? "" : " \u00b7 Tile " + d.tile + "/" + d.tiles)' in js,
          "%s: G8 the HUD shows phases for tile stages (with the tile) and whole-clip stages" % NAME)
    _need(re.search(r'const STAGE_MARK = \{[^}]*single: "S"', js) is not None,
          "%s: G8 the Single stage has its HUD mark (it rendered '?' before)" % NAME)

    if _fails:
        print("%s: %d failure(s)" % (NAME, len(_fails)))
        return 1
    print("%s: PASS -- G1 curves, G2 VAE read, G3 verdicts, G4 pixel cache, G5 memory, G6 tile refine driven, G7 upscale, G8 ui" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
