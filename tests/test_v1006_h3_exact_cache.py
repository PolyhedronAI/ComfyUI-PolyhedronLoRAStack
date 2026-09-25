#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v1006 -- Power Upscale H3 refine: the exact start, the encode cache, cfg.

THE WOUNDS (Frank's field log, 24.09.2026, v1004):
  1. denoise 0.30 on the shift-12 Hyperflow curve ran ONE step from sigma
     0.469 -- the curve has no point at or below 0.30 except its terminal, and
     the v1004 rule "first sigma <= denoise" fell back to the last step. The
     dial said 30 %, the run did 47 %.
  2. 682 s per run, of which ESRGAN 351 s + encode 127 s happen BEFORE the
     first refine step -- and every denoise/cfg/seed experiment on the same
     clip paid them again.
  3. cfg 0.97 arrived from a scrambled load, displayed as "1.0" (step 0.1 ->
     one decimal), and doubled every refine step (121 s instead of ~60 s).

WHAT v1006 PROMISES, and where it can break:

  E1  _sigma_tail starts EXACTLY at denoise (test_v1004 S1 carries the
      values); _sigma_on_grid says whether that start is a curve point, and the
      console line says so when it is not.
  C1  _content_fingerprint reads CONTENT: equal tensors -> equal key, one changed
      pixel -> different key, a different shape -> different key. Never id().
  C2  _cache_key folds in the VAE, the pixel wire, the canvas and the
      resize method -- change any one and the key changes.
  C3  put/get: a stored latent comes back on the same key, not on another;
      the store keeps a CPU CLONE (a later in-place change of the source
      never reaches the cache).
  C4  _h3_refine with `cached` runs NO encode and reports cached=True; the
      returned info carries vid_lat / n_in / canvas for the store.
  C5  upscale(): the key is built BEFORE the clock posts; on a hit the final
      pass is skipped, enc:h3 is not posted, both call sites pass `cached`
      and store on a miss; the verdict says "latent from cache".
  F1  cfg and cfg_low declare step 0.01 (a 0.97 is shown as 0.97, not "1.0");
      a cfg != 1 on the H3 refine is said out loud; denoise <= 0 turns the
      refine off by name.

v1008 RE-GROUNDING (declared): the joint refine carries general names
(_joint_refine, _latent_cache_*, _content_fingerprint, _cache_key; locals
_jr_key / _jr_hit / _jr_info / _order / _joint_on / _joint_why; clock stage
'joint'), and the encode runs through _vae_ops (vae_tiling reaches the joint
refine). Every promise stands as written; the pins follow the names.
"""

import ast
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _lift          # noqa: E402
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, "nodes", "ph_power_upscale.py")
NAME = "test_v1006_h3_exact_cache"
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


def main():
    print("[%s]" % NAME)
    src = _read(PY)

    # ---- C1-C3 with REAL torch (the fingerprint reads tensors) --------------
    try:
        import torch
    except Exception:
        torch = None
    if torch is None:
        _fail("%s: torch missing -- C1-C4 need it" % NAME)
    else:
        code, missing = _lift.close_over(
            src, ["_content_fingerprint", "_cache_key", "_latent_cache_get", "_latent_cache_put",
                  "_LATENT_CACHE", "_sigma_on_grid", "_sigma_tail"],
            provided={"torch", "time", "hashlib", "_fmt_clock"})
        _need(not missing, "%s: lift not closed: %s" % (NAME, sorted(missing)))
        ns = {"torch": torch, "time": __import__("time")}
        exec(compile(code, "<v1006-lift>", "exec"), ns)
        fp, key, get, put = ns["_content_fingerprint"], ns["_cache_key"], ns["_latent_cache_get"], ns["_latent_cache_put"]
        g = torch.Generator().manual_seed(7)
        a = torch.rand((9, 64, 96, 3), generator=g)
        b = a.clone()
        c = a.clone()
        c[4, 10, 20, 1] += 0.5
        _need(fp(a) == fp(b), "%s: C1 equal content -> equal fingerprint" % NAME)
        _need(fp(a) != fp(c), "%s: C1 one changed pixel -> different fingerprint" % NAME)
        _need(fp(a) != fp(a[:8]), "%s: C1 a different shape -> different fingerprint" % NAME)
        _need("id(" not in src[src.index("def _content_fingerprint"):src.index("def _wire_id")],
              "%s: C1 the fingerprint must never read id()" % NAME)

        class _VAE:
            latent_channels = 24
            first_stage_model = types.SimpleNamespace()
        um1 = types.SimpleNamespace(scale=4.0, model=types.SimpleNamespace())
        um2 = types.SimpleNamespace(scale=2.0, model=types.SimpleNamespace())
        k0 = key(a, _VAE(), um1, (1872, 1072), "lanczos (cpu)")
        _need(k0 == key(b, _VAE(), um1, (1872, 1072), "lanczos (cpu)"), "%s: C2 same everything -> same key" % NAME)
        _need(k0 != key(a, _VAE(), um2, (1872, 1072), "lanczos (cpu)"), "%s: C2 another pixel wire -> other key" % NAME)
        _need(k0 != key(a, _VAE(), um1, (1344, 768), "lanczos (cpu)"), "%s: C2 another canvas -> other key" % NAME)
        _need(k0 != key(a, _VAE(), um1, (1872, 1072), "bicubic"), "%s: C2 another resize method -> other key" % NAME)
        _need(k0 != key(a, _VAE(), None, (1872, 1072), "lanczos (cpu)"), "%s: C2 no pixel wire -> other key" % NAME)

        lat = torch.rand((1, 24, 7, 67, 117))
        put(k0, lat, "t")
        got = get(k0)
        _need(got is not None and torch.equal(got, lat), "%s: C3 the latent comes back on its key" % NAME)
        _need(get(k0 + "x") is None, "%s: C3 another key returns nothing" % NAME)
        lat[0, 0, 0, 0, 0] += 1.0
        _need(not torch.equal(get(k0), lat), "%s: C3 the store is a CLONE -- a later change never reaches it" % NAME)
        _need(get(k0).device.type == "cpu", "%s: C3 the cache lives on the CPU" % NAME)

        # E1
        on_grid = ns["_sigma_on_grid"]
        shifted = [1, 0.994, 0.984, 0.966, 0.923, 0.835, 0.697, 0.469, 0]
        _need(on_grid(shifted, 0.469) and not on_grid(shifted, 0.30), "%s: E1 on-grid reading" % NAME)

    # ---- C4: the refine with a cached latent runs NO encode -----------------
    # (driven through test_v1004's stub torch would duplicate 200 lines; the
    #  promise is pinned on the SOURCE of _h3_refine instead, at the effect.)
    ref = src[src.index("def _joint_refine("):src.index("class ULSPowerUpscale")]
    _need("cached=None" in ref, "%s: C4 _h3_refine takes `cached`" % NAME)
    i_if = ref.index("if vid_lat_c is not None:")
    i_enc = ref.index("vid_lat = v_enc(src[:, :, :, :3])")
    # v1007 RE-GROUNDING (declared): the encode now sits inside its heartbeat
    # phase, one nesting deeper -- the promise (encode only in the ELSE) stands.
    i_else = ref.index("    else:\n        with _Phase(\"encode\"")
    _need(i_if < i_else and i_if < i_enc, "%s: C4 the encode sits in the ELSE of the cached branch" % NAME)
    _need('"cached": vid_lat_c is not None' in ref and '"vid_lat": vid_lat_keep' in ref
          and '"n_in": n_in' in ref and '"canvas": (sw, sh)' in ref,
          "%s: C4 info carries cached/vid_lat/n_in/canvas" % NAME)
    _need("vid_lat_keep = vid_lat.detach().to(\"cpu\")" in ref, "%s: C4 the kept latent is a CPU copy" % NAME)
    _need("REUSED from the cache" in ref, "%s: C4 a reuse is said in the console" % NAME)
    _need("the start is NOT a point of the curve" in ref, "%s: E1 an off-grid start is said in the console" % NAME)

    # ---- C5: upscale() call sites ------------------------------------------
    up = src[src.index("    def upscale(self"):]
    i_key = up.index("_jr_key = _cache_key(frames, vae, _um_key")
    i_hit = up.index("_jr_hit = _latent_cache_get(_jr_key)")
    i_post = up.index('clock.post("step:joint", _jr_steps, _jr_px)')
    i_push = up.index("clock.push()", i_post)
    _need(i_key < i_hit < i_post < i_push, "%s: C5 key and hit are decided BEFORE the h3 clock posts" % NAME)
    _need('if _jr_hit is None:\n                clock.post("enc:joint", 1, _jr_px)' in up,
          "%s: C5 enc:h3 is posted only on a miss" % NAME)
    _need('clock.resize("pix:final" if _um_key is not None else "fit:final", 0)' in up,
          "%s: C5 a hit takes the final pass (model or plain fit) off the clock" % NAME)
    _need('if _final_runs and _jr_hit is not None and _order == "after pixel":' in up
          and "final pass SKIPPED (joint cache hit" in up,
          "%s: C5 the final pass is skipped on a hit, and said" % NAME)
    _need(up.count("mute=mute_staging_logs, cached=_c,") == 2, "%s: C5 BOTH call sites pass `cached`" % NAME)
    _need(up.count("_latent_cache_put(_jr_key, _jr_info[\"vid_lat\"]") == 2, "%s: C5 BOTH call sites store on a miss" % NAME)
    _need(up.count("if _jr_hit is None:\n                _latent_cache_put(") == 2,
          "%s: C5 the store happens only on a miss (never re-stores a hit)" % NAME)
    _need('_verdict += " \\u00b7 latent from cache"' in up, "%s: C5 the verdict says when the latent came from the cache" % NAME)
    # the before-pixel key must not fold in the pixel wire (it runs BEFORE it)
    _need('_jr_key = _cache_key(frames, vae, None, (_hw, _hh), "input")' in up,
          "%s: C5 the before-pixel key carries no pixel wire" % NAME)

    # ---- F1 ---------------------------------------------------------------
    it = src[src.index("def INPUT_TYPES"):src.index("RETURN_TYPES")]
    for w in ("cfg", "cfg_low"):
        m = re.search(r'"%s": \("FLOAT", \{[^}]*"step": ([0-9.]+)' % w, it)
        _need(m is not None and m.group(1) == "0.01", "%s: F1 %s must declare step 0.01 (a 0.97 must show as 0.97)" % (NAME, w))
    _need("any value but 1.0 runs a SECOND forward per step" in up, "%s: F1 cfg != 1 is said on the joint refine" % NAME)
    _need('_joint_why = "denoise=0 (nothing to refine)"' in up and "_joint_on = False" in up[up.index("denoise=0 (nothing"):][:0] + up,
          "%s: F1 denoise <= 0 turns the refine off by name" % NAME)
    i_d0 = up.index('_joint_why = "denoise=0 (nothing to refine)"')
    _need("_joint_on = False" in up[i_d0 - 120:i_d0], "%s: F1 ...and really turns it off" % NAME)

    if _fails:
        print("%s: %d failure(s)" % (NAME, len(_fails)))
        return 1
    print("%s: PASS -- E1 exact start, C1-C5 content cache, F1 cfg honesty" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
