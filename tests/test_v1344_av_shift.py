#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1344 -- the Sampler's sigma_shift on an audio-video model (Frank 18.11.: "Bau alles fertig").

Until v1343 a joint audio/video latent refused sigma_shift (v870) and the graph needed Core's
ModelSamplingMiniMaxH3. Now the Sampler shifts such a model the way that Core node does.

  A1  _apply_sigma_shift on a ModelSamplingAV model == Core's MiniMaxH3SigmaShift.execute, compared
      on the same stand-in model: class family, shift, audio_shift, noise_scale, both
      transformer_options keys. 1.0/1.0 (p-h3-image) and 12/0 (= keep the model's own audio 3.0).
  A2  a plain flow model still gets the v491 DiscreteFlow patch (unchanged), shift 0 stays a no-op,
      the input model is never mutated.
  A3  the v870 refusal stands for a joint latent on a model that is NOT audio-video (lifted block).
  A4  canon: sigma_shift_audio is the LAST required widget, FLOAT default 0.0; threaded through
      _sample_impl / _sample_gated / _run.
  A5  JS machine (node): ORDER ends on sigma_shift_audio (21), DISPLAY puts it under sigma_shift_low,
      a v839..v1343 save (20) heals to 21 with 0.0, a v544 save (19) reaches 21, SIGMA_INERT greys it.
  A6  ASCII-clean Python.

Script-style: exit 0 = pass.
"""

import ast
import json
import os
import re
import subprocess
import sys
import tempfile

NAME = "v1344"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = open(os.path.join(ROOT, "nodes", "uls_sampler.py"), encoding="utf-8").read()
JS = open(os.path.join(ROOT, "web", "js", "uls_sampler.js"), encoding="utf-8").read()
CORE = os.environ.get("PLS_CORE_ROOT") or "/home/claude/core_now"
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)
        print("  FAIL " + msg)
    else:
        print("  ok   " + msg)


def _lift(names):
    ns = {}
    for fn in names:
        exec(re.search(r"^def %s\(.*?(?=^\S)" % fn, SRC, re.S | re.M).group(0), ns)
    return ns


# ---- A1/A2: against the real Core ----------------------------------------------------------------------------
print("A1/A2 Core parity")
try:
    sys.path.insert(0, CORE)
    _argv = sys.argv
    sys.argv = [_argv[0], "--cpu"]
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy.model_sampling as cms
        import comfy_extras.nodes_minimax_h3 as nmh
    finally:
        sys.argv = _argv

    class _Cfg(object):
        sampling_settings = {"shift": 12.0, "audio_shift": 3.0, "noise_scale": 1.0}

    class _Inner(object):
        model_config = _Cfg()

    class _Model(object):
        def __init__(self, ms):
            self.model = _Inner()
            self.model_options = {"transformer_options": {"keep": 1}}
            self.patches = {"model_sampling": ms}
            self.cloned = False

        def clone(self):
            m = _Model(self.patches["model_sampling"])
            m.model_options = {"transformer_options": dict(self.model_options["transformer_options"])}
            m.patches = dict(self.patches)
            self.cloned = True
            return m

        def get_model_object(self, name):
            return self.patches[name]

        def add_object_patch(self, name, obj):
            self.patches[name] = obj

    class _AV(cms.ModelSamplingAV, cms.CONST):
        pass

    def _av_model():
        ms = _AV(_Cfg())
        ms.set_parameters(shift=12.0, audio_shift=3.0)
        ms.set_noise_scale(0.7)
        return _Model(ms)

    ns = _lift(["_sampling_name", "_owns_sigma_grid", "_is_av_sampling", "_apply_av_shift", "_apply_sigma_shift"])
    ns["GRID_SAMPLINGS"] = set()
    for v, a, want_a in ((1.0, 1.0, 1.0), (12.0, 0.0, 3.0), (6.0, 2.0, 2.0)):
        src_m = _av_model()
        ours = ns["_apply_sigma_shift"](src_m, v, a)
        core = nmh.MiniMaxH3SigmaShift.execute(_av_model(), v, want_a).args[0]
        mo, mc = ours.get_model_object("model_sampling"), core.get_model_object("model_sampling")
        _need(isinstance(mo, cms.ModelSamplingAV) and isinstance(mo, cms.CONST), "%g/%g: AV + CONST" % (v, a))
        _need(abs(mo.shift - mc.shift) < 1e-9 and mo.audio_shift == mc.audio_shift == want_a,
              "%g/%g: shift %g audio %s == Core" % (v, a, mo.shift, mo.audio_shift))
        _need(abs(float(mo.noise_scale) - float(mc.noise_scale)) < 1e-9, "%g/%g: noise_scale carried" % (v, a))
        to, tc = ours.model_options["transformer_options"], core.model_options["transformer_options"]
        _need(to.get("minimax_h3_sigma_shift_video") == tc["minimax_h3_sigma_shift_video"]
              and to.get("minimax_h3_sigma_shift_audio") == tc["minimax_h3_sigma_shift_audio"]
              and to.get("keep") == 1, "%g/%g: transformer_options == Core, rest kept" % (v, a))
        _need(src_m.get_model_object("model_sampling").shift == 12.0
              and "minimax_h3_sigma_shift_video" not in src_m.model_options["transformer_options"],
              "%g/%g: input model untouched" % (v, a))
        _need(abs(float(mo.percent_to_sigma(0.25)) - float(mc.percent_to_sigma(0.25))) < 1e-9,
              "%g/%g: same schedule as Core" % (v, a))
    plain = cms.ModelSamplingDiscreteFlow(_Cfg())
    pm = _Model(plain)
    out = ns["_apply_sigma_shift"](pm, 8.0, 1.0)
    po = out.get_model_object("model_sampling")
    _need(not isinstance(po, cms.ModelSamplingAV) and abs(po.shift - 8.0) < 1e-9
          and "minimax_h3_sigma_shift_video" not in out.model_options["transformer_options"],
          "A2 plain flow model: v491 DiscreteFlow patch, no AV keys")
    _need(ns["_apply_sigma_shift"](pm, 0.0, 1.0) is pm, "A2 shift 0 stays a no-op")
except Exception as e:   # noqa: BLE001
    _need(False, "Core importable for the parity check: %r" % e)

# ---- A3: the refusal for a non-AV joint latent ------------------------------------------------------------------
print("A3 refusal")
i = SRC.index("if (_is_ragged_latent((latent_image or {}).get(\"samples\"))")
body = SRC[i:SRC.index("raise ValueError(", i)]
_need("_is_av_sampling(model)" in body and "_is_av_sampling(model_low)" in body,
      "the refusal is skipped only when the shifted models are audio-video")
_need(SRC.index("raise ValueError(", i) < SRC.index("model = _apply_sigma_shift(model, sigma_shift", i),
      "refusal precedes the patch")

# ---- A4: canon --------------------------------------------------------------------------------------------------
print("A4 canon")
tree = ast.parse(SRC)
order, spec = [], None
for c in ast.walk(tree):
    if isinstance(c, ast.ClassDef) and c.name == "ULSSampler":
        for fn in c.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "INPUT_TYPES":
                for n in ast.walk(fn):
                    if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict):
                        for k, v in zip(n.value.keys, n.value.values):
                            if getattr(k, "value", None) == "required":
                                for kk, vv in zip(v.keys, v.values):
                                    order.append(kk.value)
                                    if kk.value == "sigma_shift_audio":
                                        spec = vv
_need(order[-2:] == ["sigma_shift_low", "sigma_shift_audio"], "sigma_shift_audio is the LAST required widget")
_need(spec is not None and spec.elts[0].value == "FLOAT"
      and [kv.value for kk, kv in zip(spec.elts[1].keys, spec.elts[1].values) if kk.value == "default"] == [0.0],
      "FLOAT, default 0.0")
for f in ("_sample_impl", "_sample_gated", "_run"):
    m = re.search(r"def %s\(.*?\):" % f, SRC, re.S)
    _need(m and "sigma_shift_audio" in m.group(0), "%s takes sigma_shift_audio" % f)
_need(SRC.count("sigma_shift_audio=sigma_shift_audio") >= 1
      and "_apply_sigma_shift(model, sigma_shift, sigma_shift_audio)" in SRC
      and "_apply_sigma_shift(model_low, _low_shift, sigma_shift_audio)" in SRC, "threaded to both patch sites")

# ---- A5: JS machine ---------------------------------------------------------------------------------------------
print("A5 JS")


def _grab(sig):
    i = JS.index(sig)
    if sig.startswith("function"):
        depth, j = 0, JS.index("{", i)
        while True:
            depth += {"{": 1, "}": -1}.get(JS[j], 0)
            j += 1
            if depth == 0:
                return JS[i:j]
    line_end = JS.index("\n", i)
    if "[" in JS[i:line_end] and "]" not in JS[i:line_end]:
        return JS[i:JS.index("];", i) + 2]          # a multi-line array
    return JS[i:line_end]


parts = [_grab(s) for s in (
    "const ORDER_V404", "const DEFAULT_PREVIEW_MODE", "const DEFAULT_SIGMA_SHIFT ", "const DEFAULT_HANDOFF_MODE",
    "const SAME_AS_HIGH", "const DEFAULT_SAMPLER_LOW", "const DEFAULT_SCHEDULER_LOW", "const DEFAULT_SIGMA_SHIFT_LOW",
    "const DEFAULT_SIGMA_SHIFT_AUDIO", "const DISPLAY_ORDER", "const SIGMA_INERT",
    "const LEN_V544_CURRENT", "const LEN_V839_CURRENT",
    "function _looksV544Current", "function _healV544Current",
    "function _looksV839Current", "function _healV839Current")]
harness = "\n".join(parts) + r"""
const out = {};
out.len = ORDER_V404.length; out.last = ORDER_V404[ORDER_V404.length - 1];
out.disp = DISPLAY_ORDER[DISPLAY_ORDER.indexOf("sigma_shift_low") + 1];
out.dispLen = DISPLAY_ORDER.length;
out.inert = SIGMA_INERT.includes("sigma_shift_audio");
const v544 = [true, 0.875, 1.0, 1, "fixed", 8, 1.0, "res_2s", "beta57", 1.0, true, 0, 10000, false,
              "Still \u00b7 ComfyUI", 0.0, "Continuous", "same as high", "same as high"];
const v839 = v544.concat([-1.0]);
out.h839 = _looksV839Current(v839) ? _healV839Current(v839) : null;
out.h544 = _looksV544Current(v544) ? _healV544Current(v544) : null;
out.cur = _looksV839Current(v839.concat([0.0])) || _looksV544Current(v839.concat([0.0]));
console.log(JSON.stringify(out));
"""
mjs = os.path.join(tempfile.mkdtemp(), "drive.mjs")
open(mjs, "w", encoding="utf-8").write(harness)
r = subprocess.run(["node", mjs], capture_output=True, text=True, timeout=60)
if r.returncode:
    _need(False, "node harness: " + r.stderr[:600])
else:
    o = json.loads(r.stdout.strip().splitlines()[-1])
    _need(o["len"] == 21 and o["last"] == "sigma_shift_audio", "ORDER ends on sigma_shift_audio (21)")
    _need(o["disp"] == "sigma_shift_audio" and o["dispLen"] == 21, "DISPLAY: under sigma_shift_low, 21 names")
    _need(o["inert"], "SIGMA_INERT greys it with an external curve")
    _need(o["h839"] and len(o["h839"]) == 21 and o["h839"][-1] == 0.0 and o["h839"][-2] == -1.0,
          "v839..v1343 save (20) heals to 21 with 0.0")
    _need(o["h544"] and len(o["h544"]) == 21 and o["h544"][-1] == 0.0, "v544 save (19) reaches 21")
    _need(not o["cur"], "a current save (21) is left alone")

# the load path really calls the heal (a unit-driven heal that the loader never reaches heals nothing)
_m = re.search(r"else if \(_looksV544Current\(info\.widgets_values\)\) \{.*?\}\s*else if \(_looksV839Current\(info\.widgets_values\)\) \{\s*(?://[^\n]*\n\s*)*info\.widgets_values = _healV839Current\(info\.widgets_values\);", JS, re.S)
_need(_m is not None, "the loader chain heals a v839..v1343 save right after the v544 heal")

# ---- A6 ---------------------------------------------------------------------------------------------------------
_new = SRC[SRC.index("def _is_av_sampling("):SRC.index("def _apply_sigma_shift(")]
_need(all(ord(ch) < 128 for ch in _new), "the v1344 blocks are ASCII-clean (the file itself predates the rule)")
print("%s: %s" % (NAME, "PASS" if not _fails else "FAIL %d" % len(_fails)))
sys.exit(1 if _fails else 0)
