#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v1004 -- the H3 refine: a joint (video+audio) model refines the WHOLE clip.

THE WOUND (Frank, 24.09., screenshots of his H3 workflow): Power Upscale wired
to MiniMax H3, denoise 0.97, 8 steps -- and nothing happened to the picture.
Correct since v883: the tile refine cannot serve a joint model, the stages were
dropped and the pixel path ran alone. The node SAID so -- in the console, once,
where nobody looked. Two failures in one: no refine a joint model CAN run, and
a dial row that pretended to work.

WHAT v1004 PROMISES, and where each promise can break:

  S1  _sigma_tail reads denoise as the NOISE FRACTION: the refine starts at
      the first sigma <= denoise; denoise >= 1 runs the whole curve; a curve
      with nothing at or below denoise keeps its last step (never zero steps).
  S2  _vae_frame_law is Core's 17k+5 <-> 5k+2 law (sd.py:985-988), run for the
      values that matter (1, 5, 22, 25, 39) and checked against the formula
      written independently here, not read back from the module.
  S3  _snap_to lands on /16 (the H3 VAE's factor) and never below 16.
  B1  _h3_refine packs ONE video latent with the WIRED audio latent, hands
      comfy.sample.sample_custom that pack, and with h3_audio='keep' a mask
      whose audio part is ZERO and whose video part is ONE. With 'denoise' the
      mask is None. Driven with stub torch objects against a counting sampler.
  B2  with a wired curve the sampler receives EXACTLY the tail (S1); without
      one it receives the KSampler schedule.
  B3  the decision: upscale() turns the refine ON only for a joint model with a
      packed 'latent'; every 'no' is printed by name; the sigma shift dials are
      ZEROED before the shift block for a joint model (the block itself is the
      v851 closed unit and must not read _joint -- v851/v894 exec it closed).
  B4  BOTH modes call _h3_refine: 'before pixel' before the stage loop, 'after
      pixel' after the final pass; the final canvas is snapped /16 first.
  W1  INPUT_TYPES ends on (..., 'sigma_shift_low', 'h3_refine', 'h3_audio') and
      the JS ORDER_CANON ends the same (the serialisation law, #577); 'latent'
      and 'sigmas' are OPTIONAL inputs (no widgets_values slot).
  F1  the frontend reads pls_pu_verdict, shows it on the viewer, remembers it
      in node.properties.pu_verdict, and hides tile_size / tile_overlap /
      sigma_shift for a joint run through the v888 mechanics (_setDisabled).
  F2  the backend returns the verdict in 'ui' under that exact key.

Mutations run by the harness (tests/_mutate contract): see MUTATIONS below --
each is applied to a COPY and must turn this guard red.

v1008 RE-GROUNDING (declared, Frank 25.09.: "Warum heisst das eigentlich H3
Refine? ... allgemein-gueltige Nodes fuer alle Modelle"): the widgets are
renamed in their slots (h3_refine -> refine_order, h3_audio -> audio_stream),
the helpers carry general names (_joint_refine, _sigma_tail, _snap_to,
_joint_verdict ...), the frame law and the snap are READ from the VAE
(_vae_frame_law / _vae_spatial) instead of hard-wired to H3, and the frontend
learns 'joint' from the backend's path key instead of the verdict text. Every
promise below stands; only the names and the signatures moved.
"""

import ast
import json
import os
import re
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _lift          # noqa: E402
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, "nodes", "ph_power_upscale.py")
PROG = os.path.join(ROOT, "nodes", "ph_progress.py")   # v1010: the clock moved there (declared)
JS = os.path.join(ROOT, "web", "js", "ph_power_upscale.js")
NAME = "test_v1004_h3_refine"

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


# ---------------------------------------------------------------------------
# a tiny torch stand-in: enough for the pure helpers and the refine driver
# ---------------------------------------------------------------------------
class _T:
    """A shape-carrying stand-in for a tensor. dim()/shape/reshape/to/cpu."""
    def __init__(self, shape, tag=""):
        self.shape = tuple(shape)
        self.tag = tag
        self.dtype = "f32"
        self.is_nested = False

    def dim(self):
        return len(self.shape)

    def to(self, *a, **k):
        return self

    def cpu(self):
        return self

    def detach(self):
        return self

    def clone(self):
        return _T(self.shape, self.tag)

    def movedim(self, a, b):
        s = list(self.shape)
        v = s.pop(a)
        s.insert(b, v)
        return _T(s, self.tag)

    def unsqueeze(self, i):
        s = list(self.shape)
        s.insert(i, 1)
        return _T(s, self.tag)

    def reshape(self, *s):
        if len(s) == 1 and isinstance(s[0], (tuple, list)):
            s = tuple(s[0])
        n = 1
        for v in self.shape:
            n *= v
        known = 1
        for v in s:
            if v != -1:
                known *= v
        s = tuple(n // known if v == -1 else v for v in s)
        return _T(s, self.tag)

    def __getitem__(self, idx):
        # frames[:, :, :, :3] / px[:, :, :, :3] -- keep the shape
        return self

    def flatten(self):
        return self

    def tolist(self):
        return list(self.vals)


class _Sig(_T):
    def __init__(self, vals):
        _T.__init__(self, (len(vals),))
        self.vals = list(vals)

    def __getitem__(self, i):
        return self.vals[i]

    def __len__(self):
        return len(self.vals)


class _Nested:
    is_nested = True

    def __init__(self, parts):
        self.tensors = list(parts)

    def unbind(self):
        return list(self.tensors)


def _stub_modules(calls):
    """Install stub comfy.* / torch modules and return a restore function."""
    saved = {k: sys.modules.get(k) for k in
             ("torch", "comfy", "comfy.sample", "comfy.samplers", "comfy.utils",
              "comfy.model_management", "comfy.nested_tensor", "folder_paths",
              "comfy_api", "comfy_api.input_impl", "comfy_api.util",
              "comfy_extras", "comfy_extras.nodes_upscale_model")}
    torch = types.ModuleType("torch")
    torch.float32 = "f32"

    def _tensor(vals, dtype=None):
        return _Sig([float(v) for v in vals])
    torch.tensor = _tensor
    torch.zeros = lambda shape, dtype=None, **k: _T(shape, "zeros")
    torch.ones = lambda shape, dtype=None, **k: _T(shape, "ones")
    torch.Tensor = _T
    sys.modules["torch"] = torch
    comfy = types.ModuleType("comfy")
    sample = types.ModuleType("comfy.sample")

    def prepare_noise(latent, seed, inds=None):
        calls.append(("prepare_noise", latent, seed))
        return _Nested([_T(p.shape, "noise") for p in latent.tensors])

    def sample_custom(model, noise, cfg, sampler, sigmas, positive, negative,
                      latent_image, noise_mask=None, callback=None,
                      disable_pbar=False, seed=None):
        calls.append(("sample_custom", dict(noise=noise, cfg=cfg, sampler=sampler,
                                            sigmas=sigmas, latent=latent_image,
                                            mask=noise_mask, seed=seed)))
        if callback is not None:
            callback(0, latent_image, latent_image, 1)
        return latent_image
    sample.prepare_noise = prepare_noise
    sample.sample_custom = sample_custom
    sample.sample = lambda *a, **k: (_ for _ in ()).throw(AssertionError("tile path used"))
    samplers = types.ModuleType("comfy.samplers")

    class KSampler:
        def __init__(self, model, steps, device, sampler, scheduler, denoise,
                     model_options=None):
            calls.append(("KSampler", steps, scheduler, denoise))
            n = max(1, int(steps))
            self.sigmas = _Sig([1.0 - i / n for i in range(n)] + [0.0])
    samplers.KSampler = KSampler
    samplers.sampler_object = lambda name: ("sampler", name)
    utils = types.ModuleType("comfy.utils")
    utils.ProgressBar = lambda n: types.SimpleNamespace(update_absolute=lambda *a: None)
    mm = types.ModuleType("comfy.model_management")
    mm.get_torch_device = lambda: "cpu"
    mm.soft_empty_cache = lambda: None
    nt = types.ModuleType("comfy.nested_tensor")
    nt.NestedTensor = _Nested
    comfy.sample, comfy.samplers, comfy.utils = sample, samplers, utils
    comfy.model_management, comfy.nested_tensor = mm, nt
    for k, v in (("comfy", comfy), ("comfy.sample", sample),
                 ("comfy.samplers", samplers), ("comfy.utils", utils),
                 ("comfy.model_management", mm), ("comfy.nested_tensor", nt)):
        sys.modules[k] = v
    sys.modules["folder_paths"] = types.ModuleType("folder_paths")

    def restore():
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return restore


def _lift_helpers(src, names, extra):
    code, missing = _lift.close_over(src, names, provided=set(extra))
    _need(not missing, "%s: lift of %s is not closed, missing %s" % (NAME, names, sorted(missing)))
    ns = dict(extra)
    exec(compile(code, "<v1004-lift>", "exec"), ns)
    return ns


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
    calls = []
    restore = _stub_modules(calls)
    try:
        import torch  # the stub
        extra = {"torch": torch, "time": __import__("time"), "print": print,
                 "_fmt_clock": lambda s: "0s", "_free": lambda: None,
                 "_lanczos_to": lambda f, w, h: _T((f.shape[0], h, w, 3), "fit"),
                 "_MuteInfoLogs": None,
                 "_latent_parts": lambda x: list(x.tensors) if getattr(x, "is_nested", False) else [x],
                 "_joint_video_half": lambda x: x.tensors[0] if getattr(x, "is_nested", False) else x,
                 "comfy": sys.modules["comfy"],
                 # v1007: the clock helpers -- node id unknown in the harness,
                 # rates in a throwaway dir, heartbeat threads real but short
                 "_current_node_id": lambda: None,
                 "folder_paths": types.SimpleNamespace(get_user_directory=lambda: "/tmp/pls_v1004_user"),
                 "threading": __import__("threading"), "json": __import__("json"), "os": __import__("os")}

        class _Mute:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        extra["_MuteInfoLogs"] = _Mute
        extra["inspect"] = __import__("inspect")
        ns = _lift_helpers(src + "\n" + _read(PROG), ["_sigma_tail", "_vae_frame_law", "_snap_to",
                                 "_joint_verdict", "_patch_count", "_joint_refine",
                                 "_sigma_on_grid", "_LATENT_CACHE", "_latent_cache_get", "_latent_cache_put",
                                 "_Phase", "_phase_plan", "_rates_load", "_rates_learn",
                                 "_RATES", "_RATES_LOADED", "_HEARTBEAT_S", "_RATES_EMA",
                                 "_rates_path", "_PROBE_EVENT",
                                 "_JOINT_SNAP_DEFAULT", "_REFINE_ORDERS", "_AUDIO_STREAM_MODES"],
                           extra)
        tail = ns["_sigma_tail"]

        class _H3VAE:   # Core's own H3 video VAE ratios (sd.py:985-988), copied as data
            downscale_ratio = (lambda a: max(1, (a - 5) // 17 * 5 + 2) if a > 1 else 1, 16, 16)
            upscale_ratio = (lambda a: max(1, (a - 2) // 5 * 17 + 5), 16, 16)
        law = lambda n: ns["_vae_frame_law"](_H3VAE, n)
        snap = lambda w, h: ns["_snap_to"](w, h, 16)

        # ---- S1 -- v1006 RE-GROUNDING (declared): the refine starts EXACTLY
        #      at sigma = denoise (inserted), then the curve points below it.
        #      v1004's "first sigma <= denoise" ran 47 % on a shift-12 curve
        #      when 30 % was dialled (field log 24.09.). ------------------
        curve = [1, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0]
        t = tail(curve, 0.30)
        _need(t == [0.30, 0.296538, 0.160764, 0.068494, 0],
              "%s: S1 denoise 0.30 starts AT 0.30 and runs the points below, got %s" % (NAME, t))
        _need(tail(curve, 1.0) == [float(v) for v in curve],
              "%s: S1 denoise 1.0 must run the whole curve" % NAME)
        _need(tail(curve, 0.01) == [0.01, 0],
              "%s: S1 a denoise below every step is ONE step from that value, got %s" % (NAME, tail(curve, 0.01)))
        _need(tail(curve, 0.5) == [0.5, 0.296538, 0.160764, 0.068494, 0],
              "%s: S1 a value EQUAL to a sigma starts there once (not twice)" % NAME)
        _need(tail(curve, 0.0) == [], "%s: S1 denoise 0 is no refine" % NAME)
        shifted = [1, 0.994, 0.984, 0.966, 0.923, 0.835, 0.697, 0.469, 0]
        _need(tail(shifted, 0.30) == [0.30, 0],
              "%s: S1 THE FIELD CASE: 0.30 on the shift-12 curve is [0.30, 0], got %s" % (NAME, tail(shifted, 0.30)))
        _need(tail([0.3], 0.3) == [0.3], "%s: S1 a one-value list passes through" % NAME)

        # ---- S2 -------------------------------------------------------
        def _law(n):
            t_ = max(1, (n - 5) // 17 * 5 + 2) if n > 1 else 1
            b_ = max(1, (t_ - 2) // 5 * 17 + 5) if t_ > 1 else 1
            return t_, b_
        for n in (1, 5, 22, 25, 39, 100):
            _need(law(n) == _law(n), "%s: S2 frame law at n=%d: %s vs %s" % (NAME, n, law(n), _law(n)))
        _need(law(25) == (7, 22), "%s: S2 25 frames -> 7 latents -> 22 frames (the field case)" % NAME)

        # ---- S3 -------------------------------------------------------
        _need(snap(1882, 1075) == (1872, 1072), "%s: S3 1882x1075 snaps to 1872x1072" % NAME)
        _need(snap(1344, 768) == (1344, 768), "%s: S3 an aligned canvas is untouched" % NAME)
        _need(snap(7, 3) == (16, 16), "%s: S3 never below 16" % NAME)

        # ---- B1 / B2: drive _h3_refine with the counting sampler -----
        refine = ns["_joint_refine"]

        class _VAE:
            def encode(self, px):
                calls.append(("encode", px.shape))
                n, h, w = px.shape[0], px.shape[1], px.shape[2]
                t_, _ = _law(n)
                return _T((1, 24, t_, h // 16, w // 16), "vid")

            def decode(self, lat):
                calls.append(("decode", lat.shape))
                t_ = lat.shape[2]
                return _T((1, (t_ - 2) // 5 * 17 + 5, lat.shape[3] * 16, lat.shape[4] * 16, 3), "px")
        model = types.SimpleNamespace(patches={"a": 1, "b": 2}, model_options={})
        frames = _T((22, 1075, 1882, 3), "frames")
        audio = _T((1, 32, 300), "audio")
        latent = {"samples": _Nested([_T((1, 24, 7, 48, 84), "vid0"), audio])}
        sig = _Sig(curve)
        clock = _Clock()
        del calls[:]
        out, info = refine(model, ["pos"], ["neg"], _VAE(), frames, latent, sig,
                           seed=7, steps=8, cfg=1.0, sampler_name="dpmpp_2m",
                           scheduler="karras", denoise=0.30, audio_mode="keep",
                           clock=clock, probe=None, mute=True)
        sc = [c for c in calls if c[0] == "sample_custom"]
        _need(len(sc) == 1, "%s: B1 exactly one sample_custom call, got %d" % (NAME, len(sc)))
        k = sc[0][1]
        _need(getattr(k["latent"], "is_nested", False) and len(k["latent"].tensors) == 2,
              "%s: B1 the sampler must receive a TWO-stream pack" % NAME)
        _need(k["latent"].tensors[1] is audio,
              "%s: B1 the audio stream must be the WIRED one, untouched" % NAME)
        vid = k["latent"].tensors[0]
        _need(vid.shape == (1, 24, 7, 67, 117),
              "%s: B1 the video stream is the re-encoded, /16-snapped clip: got %s" % (NAME, vid.shape))
        enc = [c for c in calls if c[0] == "encode"]
        _need(enc and enc[0][1][1:3] == (1072, 1872),
              "%s: B1 the encode must see the snapped canvas, got %s" % (NAME, enc))
        m = k["mask"]
        _need(m is not None and getattr(m, "is_nested", False), "%s: B1 'keep' needs a nested mask" % NAME)
        if m is not None and getattr(m, "is_nested", False):
            _need(m.tensors[0].tag == "ones" and m.tensors[1].tag == "zeros",
                  "%s: B1 mask = ones on video, ZEROS on audio (got %s/%s)" % (NAME, m.tensors[0].tag, m.tensors[1].tag))
            _need(m.tensors[1].shape == (1, 1, 300),
                  "%s: B1 the audio mask carries the audio geometry, got %s" % (NAME, m.tensors[1].shape))
        _need(k["sigmas"].vals == [0.30, 0.296538, 0.160764, 0.068494, 0.0],
              "%s: B2 the sampler must get the TAIL (S1), got %s" % (NAME, k["sigmas"].vals))
        _need(k["seed"] == 7 and k["cfg"] == 1.0, "%s: B1 seed/cfg pass through" % NAME)
        _need(not any(c[0] == "KSampler" for c in calls), "%s: B2 with a curve no KSampler schedule is built" % NAME)
        _need(info["steps_run"] == 4 and abs(info["start_sigma"] - 0.30) < 1e-9,
              "%s: B2 info reports 4 steps from 0.30, got %s" % (NAME, info))
        _need(out.shape == (22, 1072, 1872, 3), "%s: B1 output frames [N,H,W,3], got %s" % (NAME, out.shape))
        _need("enc:joint" in clock.measured and "step:joint" in clock.measured and "dec:joint" in clock.measured,
              "%s: B1 the clock hears encode, step and decode of the joint stage" % NAME)

        # 'denoise' audio mode -> no mask; no curve -> KSampler schedule
        del calls[:]
        out2, info2 = refine(model, ["pos"], ["neg"], _VAE(), frames, latent, None,
                             seed=1, steps=8, cfg=1.0, sampler_name="euler",
                             scheduler="simple", denoise=0.25, audio_mode="denoise",
                             clock=_Clock(), probe=None, mute=True)
        sc = [c for c in calls if c[0] == "sample_custom"]
        _need(sc and sc[0][1]["mask"] is None, "%s: B1 'denoise' audio -> no mask" % NAME)
        ks = [c for c in calls if c[0] == "KSampler"]
        _need(ks and ks[0][1:] == (8, "simple", 0.25),
              "%s: B2 without a curve the KSampler schedule runs (steps, scheduler, denoise), got %s" % (NAME, ks))
        _need(info2["sigmas_n"] == 0, "%s: B2 no curve -> sigmas_n 0 (verdict says 'scheduler')" % NAME)

        # v1006: a CACHED latent runs no encode, and the info says so
        del calls[:]
        out3, info3 = refine(model, ["pos"], ["neg"], _VAE(), None, latent, sig,
                             seed=7, steps=8, cfg=1.0, sampler_name="euler",
                             scheduler="simple", denoise=0.30, audio_mode="keep",
                             clock=_Clock(), probe=None, mute=True,
                             cached=(_T((1, 24, 7, 67, 117), "vid"), 22, 1872, 1072))
        _need(not any(c[0] == "encode" for c in calls), "%s: C4 a cached latent must run NO encode" % NAME)
        _need(info3["cached"] is True and info3["n_in"] == 22 and info3["canvas"] == (1872, 1072),
              "%s: C4 info reports the cache hit and the geometry, got %s" % (NAME, {k: info3[k] for k in ('cached', 'n_in', 'canvas')}))
        sc3 = [c for c in calls if c[0] == "sample_custom"]
        _need(sc3 and sc3[0][1]["latent"].tensors[0].shape == (1, 24, 7, 67, 117),
              "%s: C4 the cached latent is the one sampled" % NAME)

        # a single-stream latent is refused BY NAME
        try:
            refine(model, ["pos"], ["neg"], _VAE(), frames, {"samples": _T((1, 24, 7, 48, 84))},
                   sig, 1, 8, 1.0, "euler", "simple", 0.3, "keep", _Clock())
            _fail("%s: B1 a single-stream 'latent' must be refused" % NAME)
        except ValueError as e:
            _need("audio" in str(e) and "latent" in str(e), "%s: B1 the refusal names the wire and the reason" % NAME)

        # ---- verdict text (v1008: path-free signature, pixel-model place) --
        v = ns["_joint_verdict"]
        line = v("after pixel", True, 8, 3, 0.296538, "keep", 2, 22, 22, where="front")
        _need(line.startswith("Joint refine after pixel: 3 steps from \u03c3 0.30 (8-step curve)"),
              "%s: verdict line shape: %r" % (NAME, line))
        _need("audio kept" in line and "2 patches" in line, "%s: verdict names audio and patches" % NAME)
        _need("pixel model in front of the VAE" in line, "%s: verdict names where the pixel model sat" % NAME)
        _need("NO patches" in v("after pixel", True, 8, 3, 0.3, "keep", 0, 22, 22),
              "%s: verdict must shout a raw model" % NAME)
        _need("frame law" in v("after pixel", True, 8, 3, 0.3, "keep", 2, 25, 22),
              "%s: verdict says when frames were lost" % NAME)
        off = v("off", False, 0, 0, 0.0, "keep", 0, 22, 22, why="refine_order=off", where="behind")
        _need(off.startswith("Joint refine OFF (refine_order=off)") and "behind the last decode" in off,
              "%s: OFF verdict: %r" % (NAME, off))
    finally:
        restore()

    # ---- B3 / B4 / F2: source pins on upscale() ----------------------------
    tree = ast.parse(src)
    up = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ULSPowerUpscale":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "upscale":
                    up = sub
    _need(up is not None, "%s: upscale() is gone" % NAME)
    if up is not None:
        args = [a.arg for a in up.args.args] + [a.arg for a in up.args.kwonlyargs]
        for a in ("refine_order", "audio_stream", "latent", "sigmas"):
            _need(a in args, "%s: upscale() lacks the %r argument" % (NAME, a))
        body = ast.dump(up)
        n_ref = body.count("id='_joint_refine'")
        _need(n_ref == 2, "%s: B4 _joint_refine must be called in BOTH modes (before / after), found %d call sites" % (NAME, n_ref))
        # the CONDITION that guards each call: _joint_on AND the mode -- a
        # mutated `if False and ...` keeps the text and loses the call
        gated = set()
        for sub in ast.walk(up):
            if isinstance(sub, ast.If) and "id='_joint_refine'" in "".join(ast.dump(x) for x in sub.body):
                td = ast.dump(sub.test)
                if isinstance(sub.test, ast.BoolOp) and isinstance(sub.test.op, ast.And) \
                        and "id='_joint_on'" in td and "Constant(value=False)" not in td:
                    for mode in ("before pixel", "after pixel"):
                        if mode in td:
                            gated.add(mode)
        _need(gated == {"before pixel", "after pixel"},
              "%s: B4 each _joint_refine call must be gated on `_joint_on and _order == <mode>` -- gated: %s" % (NAME, sorted(gated)))
        seg = src[src.index("def upscale(self"):]
        i_before = seg.find('_order == "before pixel":\n            # v1004')
        i_loop = seg.find("for st, sw, sh, grid in plans:")
        i_after = seg.find('_order == "after pixel":\n            # v1004: the hires-fix')
        i_video = seg.find("video_out = _build_video(")
        i_final = seg.find("if _final_runs:\n            um_fin =")
        _need(0 < i_before < i_loop, "%s: B4 the 'before pixel' refine must sit BEFORE the stage loop" % NAME)
        _need(0 < i_final < i_after < i_video, "%s: B4 the 'after pixel' refine must sit AFTER the final pass and before delivery" % NAME)
        _need("_snap_to(tw, th, _jr_snap)" in seg, "%s: B4 the final canvas is snapped to the joint VAE's grid for the refine" % NAME)
        # B3: the shift dials are zeroed BEFORE the block; the block does not read _joint
        i_zero = seg.find("sigma_shift, sigma_shift_low = 0.0, -1.0")
        i_block = seg.find("_dual = bool(dual_moe)")
        i_plan = seg.find("stages = uls_tile_math.plan_stages")
        _need(0 < i_zero < i_block, "%s: B3 the joint model must zero the shift dials BEFORE the shift block" % NAME)
        _need("_joint" not in seg[i_block:i_plan], "%s: B3 the v851 shift block must not read _joint (v851/v894 exec it closed)" % NAME)
        # B3: every 'no' is named
        for why in ("refine_order=off", "'latent' not wired", "not a packed AV latent"):
            _need(why in seg, "%s: B3 the refusal %r must be said by name" % (NAME, why))
        _need("joint refine OFF --" in seg, "%s: B3 an OFF decision is printed" % NAME)
        _need("NO weight patches" in seg, "%s: B3 a raw model is called out (Frank's 24.09. wiring)" % NAME)
        # the decision gates on a PACKED latent, through _latent_parts
        _need("len(_latent_parts(latent.get(\"samples\"))) < 2" in seg,
              "%s: B3 the decision must measure the latent's streams, not trust the wire" % NAME)
        # F2: the ui payload
        _need('"pls_pu_verdict": [_verdict]' in seg, "%s: F2 the verdict rides in ui.pls_pu_verdict" % NAME)
        _need('"pls_pu_path": ["joint" if _joint is not None else "tile"]' in seg,
              "%s: F2 (v1008) the path rides along -- the frontend must not parse the text" % NAME)
        _need("print(f\"[PLS] Power Upscale: {_verdict}\")" in seg, "%s: F2 the verdict is also printed" % NAME)

    # ---- W1: INPUT_TYPES tail == JS canon tail ------------------------------
    _it = src[src.index("def INPUT_TYPES"):src.index("RETURN_TYPES")]
    it_tail = re.findall(r'^\s{16}"([a-z_0-9]+)": \(', _it[:_it.index('"optional": {')], re.M)
    _need(it_tail[-3:] == ["sigma_shift_low", "refine_order", "audio_stream"],
          "%s: W1 required must END on sigma_shift_low, refine_order, audio_stream -- got %s" % (NAME, it_tail[-3:]))
    opt = src[src.index('"optional": {', src.index("def INPUT_TYPES")):src.index("RETURN_TYPES")]
    _need('"latent": ("LATENT"' in opt and '"sigmas": ("SIGMAS"' in opt,
          "%s: W1 latent and sigmas are OPTIONAL inputs (no widgets_values slot)" % NAME)
    canon = re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1)
    canon = re.sub(r"//[^\n]*", "", canon)   # a commented-out entry is no entry
    names = re.findall(r'"([a-z_0-9]+)"', canon)
    _need(names[-3:] == ["sigma_shift_low", "refine_order", "audio_stream"],
          "%s: W1 JS ORDER_CANON must end the same way, got %s" % (NAME, names[-3:]))
    _need("refine_order" not in re.search(r"const DUAL_ONLY = \[(.*?)\];", js, re.S).group(1),
          "%s: W1 refine_order is not a DUAL_ONLY twin" % NAME)

    # ---- F1: the frontend shows and remembers the verdict -------------------
    _need("message.pls_pu_verdict" in js, "%s: F1 onExecuted must read pls_pu_verdict" % NAME)
    _need("node.properties.pu_verdict = t" in js, "%s: F1 the verdict is remembered in node.properties" % NAME)
    _need("p.pu_verdict" in js and "_applyVerdict(this, String(p.pu_verdict" in js,
          "%s: F1 onConfigure re-applies the remembered verdict" % NAME)
    unread = re.search(r"const JOINT_UNREAD = \[(.*?)\];", js, re.S)
    _need(unread is not None, "%s: F1 JOINT_UNREAD list missing" % NAME)
    if unread is not None:
        lst = re.findall(r'"([a-z_0-9]+)"', unread.group(1))
        _need(sorted(lst) == ["sigma_shift", "tile_overlap", "tile_size"],
              "%s: F1 JOINT_UNREAD must be exactly tile_size/tile_overlap/sigma_shift, got %s" % (NAME, lst))
    fn = js[js.index("function _applyVerdict("):]
    fn = fn[:fn.index("\n}\n") + 3]
    _need("_setDisabled(_findWidget(node, name), joint)" in fn,
          "%s: F1 the dials hide through _setDisabled (v888 mechanics), gated on the joint verdict" % NAME)
    _need('(p === "joint")' in fn and '(p === "" && t.indexOf("H3") === 0)' in fn,
          "%s: F1 'joint' is read off the PATH (v1008), the H3 text only for a legacy save" % NAME)
    _need("message.pls_pu_path" in js and "node.properties.pu_path = String(path" in js,
          "%s: F1 the path is read from the run and remembered" % NAME)
    _need("node._pvVerdict.textContent = t" in fn and 'style.display = t ? "block" : "none"' in fn,
          "%s: F1 the line is shown on the viewer and hidden when empty" % NAME)
    _need("node._pvVerdict = verdict" in js and "box.append(img, bar, verdict)" in js,
          "%s: F1 the verdict element lives inside the result viewer box" % NAME)

    if _fails:
        print("%s: %d failure(s)" % (NAME, len(_fails)))
        return 1
    print("%s: PASS -- S1-S3 helpers, B1-B4 refine + decision, W1 canon, F1/F2 frontend" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
