"""Guard -- Power Upscale refuses a joint audio/video model (v880).

THE WOUND, from Frank's field log (26.08.). MiniMax H3 was wired into the
Power Upscale's `model` input. The node ran everything: 4:46 of sampling
upstream, a 30s VAE decode, 275s of ESRGAN across 73 frames -- and then died
on the FIRST refine tile, inside CORE:

    comfy/ldm/minimax/model.py:493   audio_src = x[1]
    IndexError: list index out of range

That line is the first statement of the H3 forward pass: it reads the audio
half of a joint latent unconditionally. A tile refine hands it a plain image
latent, which has no [1]. Ten and a half minutes of wall clock, spent to reach
a one-line failure in someone else's file.

It is not a setting: the model would also need the `minimax_payload` token
geometry the Reference node builds, and it would denoise an audio stream that
does not exist inside a tile. So the answer is a hard NO -- said EARLY and BY
NAME, which is the house rule for refusals.

PROMISES (P5 RE-FOUNDED in v883 -- the v879 precedent: right rule, wrong
end. v880 pinned "the refusal raises"; the field answer to that raise was a
user with NO upscale at all. The promise that mattered was never the raise --
it was "the refine NEVER runs with a joint model, and the user learns why in
second one". v883 keeps exactly that and lets the PIXEL path deliver a file.)

  P1  The stream count is MEASURED, never name-matched. A guard that greps for
      "MiniMax" passes the next joint model straight through to the same
      crash.
  P2  Two streams -> detected, and the answer says WHICH input. One stream ->
      None. Both `model` and `model_low` are checked.
  P3  The detection is the FIRST thing the node does, and the downgrade is
      ANNOUNCED before any work: which input, that the refine stages are OFF,
      that the pixel path still delivers, and what to wire for a true refine.
  P4  The probe FAILS OPEN (None). A false detection would silence a
      legitimate refine, which is worse than the crash it guards against.
  P5  With a joint model the REFINE STAGES ARE DROPPED WHOLE -- the stage list
      the loop consumes is empty -- and the None-conditioning check steps
      aside (a pixel-only run needs no conditioning). With an ordinary model
      nothing changes: stages untouched, None still refused.

The probe is DRIVEN against a faithful rebuild of Core's own MiniMaxH3AV
(taken from comfy/latent_formats.py), not only against a convenient stub.
"""

import ast
import os
import re
import sys
import types

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC_PATH = os.path.join(ROOT, "nodes", "ph_power_upscale.py")
SRC = open(SRC_PATH, encoding="utf-8").read()

NAME = "v880"
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)


# ---------------------------------------------------------------------------
# lift the two helpers -- the node module needs comfy, the helpers do not
# ---------------------------------------------------------------------------
# v889 RE-GROUNDING, declared: this block used to demand that BOTH helpers be
# defined IN ph_power_upscale.py. That pinned the LOCATION, which was never the
# promise -- v880's promise is that a joint model is DETECTED by three
# independent witnesses before any work starts. v889 moved the probe to
# nodes/ph_joint_probe.py because ph_basics needs the same answer and a second
# copy would drift; ph_power_upscale re-exports both names. So the guard now
# follows the definition wherever it lives and keeps driving it exactly as
# before. What it must NOT lose: the helpers have to be reachable from the
# module under test, and there must be only ONE definition of each in the tree.
_tree = ast.parse(SRC)
_want = ("_joint_latent_parts", "_joint_streams")
_keep = [n for n in _tree.body
         if isinstance(n, ast.FunctionDef) and n.name in _want]
if len(_keep) != len(_want):
    # not defined here -> they must be imported here AND defined in exactly one
    # sibling module (the re-export shape). Anything else is a real break.
    _imported = re.findall(r"from \.?(\w+) import \(?_joint_latent_parts", SRC)
    _need(len(_imported) >= 1,
          "%s: helpers neither defined nor imported in ph_power_upscale" % NAME)
    _homes = []
    for _mod in set(_imported):
        _pth = os.path.join(ROOT, "nodes", _mod + ".py")
        _need(os.path.exists(_pth),
              "%s: re-export names a module that does not exist (%s)"
              % (NAME, _mod))
        if os.path.exists(_pth):
            _src2 = open(_pth, encoding="utf-8").read()
            _t2 = ast.parse(_src2)
            _k2 = [n for n in _t2.body
                   if isinstance(n, ast.FunctionDef) and n.name in _want]
            if len(_k2) == len(_want):
                _homes.append(_mod)
                SRC = _src2
                _keep = _k2
    _need(len(_homes) == 1,
          "%s: the probe must have exactly ONE home, found %s"
          % (NAME, _homes))
_need(len(_keep) == len(_want),
      "%s: the helpers are gone (found %s)"
      % (NAME, [n.name for n in _keep]))
if len(_keep) != len(_want):
    print("FAIL %s" % NAME)
    for f in _fails:
        print("  - %s" % f)
    sys.exit(1)

NS = {}
exec(compile(ast.Module(body=_keep, type_ignores=[]), "<lift>", "exec"), NS)
joint_parts = NS["_joint_latent_parts"]
detect = NS["_joint_streams"]


# ---------------------------------------------------------------------------
# Core's own shapes, rebuilt from comfy/latent_formats.py + nested_tensor.py
# ---------------------------------------------------------------------------
class _NestedTensor:
    """Core's class delegates .shape/.dtype/.device to tensors[0] and has NO
    numel() -- that absence is what killed the sampler's run clock in v870."""

    def __init__(self, tensors):
        self.tensors = tuple(tensors)

    @property
    def shape(self):
        return self.tensors[0].shape

    @property
    def dtype(self):
        return self.tensors[0].dtype


class _PlainFormat:
    latent_channels = 16

    def fix_empty_latent(self, latent):
        return latent            # LatentFormat's base behaviour


class _MiniMaxH3AV:
    """Faithful to comfy/latent_formats.py::MiniMaxH3AV."""

    latent_channels = 32          # max across the two streams

    def fix_empty_latent(self, latent):
        video_latent_channels = 24
        audio_latent_channels = 32
        audio_channels = 2
        frames_per_token = (1, 4, 4, 4, 4)
        audio_frame_rescale = 5.0 / 3.0

        video = latent[:, :video_latent_channels].clone()
        frame_count = sum(frames_per_token[i % len(frames_per_token)]
                          for i in range(video.shape[2]))
        audio_t = round(frame_count * audio_frame_rescale)
        audio = latent.new_zeros((latent.shape[0], audio_latent_channels,
                                  audio_channels, audio_t))
        return _NestedTensor((video, audio))


def _model(fmt):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(latent_format=fmt))


# ---------------------------------------------------------------------------
# P1 / P2 -- measured, and both inputs are checked
# ---------------------------------------------------------------------------
_need(joint_parts(_model(_PlainFormat())) == 1,
      "%s: P1 an ordinary image model must count as ONE stream" % NAME)
_need(joint_parts(_model(_MiniMaxH3AV())) == 2,
      "%s: P1 Core's own MiniMaxH3AV must be seen as TWO streams -- this is "
      "the model that crashed the field run" % NAME)

# The measurement must not lean on a NAME. Checked as an EFFECT, at the AST:
# a first draft searched the source text for "MiniMax" and fired on the
# docstring and on the error message -- which are exactly the places where
# naming the model is RIGHT. What must not exist is a name COMPARISON.
_probe_fn = _keep[0] if _keep[0].name == "_joint_latent_parts" else _keep[1]
for sub in ast.walk(_probe_fn):
    if isinstance(sub, ast.Attribute) and sub.attr in ("__name__", "__qualname__"):
        _fails.append(
            "%s: P1 the probe reads %s -- that is a name match. It passes the "
            "next joint model straight through to the same crash; measure the "
            "format instead" % (NAME, sub.attr))
    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
            and sub.func.id in ("isinstance", "type"):
        for a in sub.args[1:]:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                _fails.append("%s: P1 the probe type-checks against a string"
                              % NAME)
    if isinstance(sub, ast.Compare):
        for c in sub.comparators:
            if isinstance(c, ast.Constant) and isinstance(c.value, str):
                _fails.append(
                    "%s: P1 the probe compares against the string %r -- "
                    "measure the format, do not recognise it"
                    % (NAME, c.value[:30]))

# And it must actually RUN the format rather than inspect it.
_calls_fix = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "fix"
                 for n in ast.walk(_probe_fn))
_need(_calls_fix,
      "%s: P1 the probe no longer EXECUTES fix_empty_latent -- reading an "
      "attribute is not a measurement" % NAME)

for tag, args, want in (
        ("both plain", (_model(_PlainFormat()), _model(_PlainFormat())), None),
        ("no low", (_model(_PlainFormat()), None), None),
        ("model is AV", (_model(_MiniMaxH3AV()), _model(_PlainFormat())), "model"),
        ("model_low is AV", (_model(_PlainFormat()), _model(_MiniMaxH3AV())), "model_low"),
        ("both AV", (_model(_MiniMaxH3AV()), _model(_MiniMaxH3AV())), "model")):
    got = detect(*args)
    if want is None:
        _need(got is None,
              "%s: P2 %s must be None, got %r" % (NAME, tag, got))
    else:
        _need(isinstance(got, tuple) and got and got[0] == want,
              "%s: P2 %s -- detection must name WHICH input carries the "
              "joint model (want %r), got %r" % (NAME, tag, want, got))


# ---------------------------------------------------------------------------
# P6 (v882) -- the OLD-CORE shapes must be refused too
#
# THE FIELD HOLE: v880 measured only fix_empty_latent -- true on Core MASTER,
# absent on the ~v0.31 Core in the field. MiniMaxH3AV there inherits the base
# method, the probe counted one stream, failed open, and the same ten-minute
# crash came back. A measurement is only valid for the build it was taken on,
# so this guard now drives BOTH Core generations.
# ---------------------------------------------------------------------------
class _Exploding:
    latent_channels = 16

    def fix_empty_latent(self, latent):
        raise RuntimeError("nope")


class _OldCoreH3Format:
    """v0.30-v0.32 MiniMaxH3AV: NO fix_empty_latent override of its own."""
    latent_channels = 32

    def fix_empty_latent(self, latent):     # the BASE behaviour, inherited
        return latent


class _FakeModelType:
    FLOW_AV = object()          # identity is what matters, like an Enum member
    FLOW = object()


class _AVSampling:
    audio_shift = 3.0           # only ModelSamplingAV carries this attribute


class _PlainSampling:
    pass


def _old_core_model(model_type, sampling):
    return types.SimpleNamespace(model=types.SimpleNamespace(
        latent_format=_OldCoreH3Format(), model_type=model_type,
        model_sampling=sampling))


# witness B alone (Core importable, model_type says FLOW_AV)
_fake_mb = types.ModuleType("comfy.model_base")
_fake_mb.ModelType = _FakeModelType
_fake_comfy = types.ModuleType("comfy")
_fake_comfy.model_base = _fake_mb
sys.modules.setdefault("comfy", _fake_comfy)
sys.modules["comfy.model_base"] = _fake_mb

_need(joint_parts(_old_core_model(_FakeModelType.FLOW_AV, _PlainSampling())) > 1,
      "%s: P6 an old-Core H3 (no fix_empty_latent override, model_type "
      "FLOW_AV) must be seen as joint -- this exact shape crashed the field "
      "run TWICE" % NAME)

# an ordinary flow model on the same old Core must still pass
_need(joint_parts(_old_core_model(_FakeModelType.FLOW, _PlainSampling())) == 1,
      "%s: P6 an ordinary FLOW model on the old Core must stay ONE stream -- "
      "witness B must compare by identity, not fire on everything" % NAME)

# witness C alone (comfy not importable -> only the audio clock speaks)
del sys.modules["comfy.model_base"]
_had = sys.modules.pop("comfy", None)
try:
    _need(joint_parts(_old_core_model(_FakeModelType.FLOW_AV, _AVSampling())) > 1,
          "%s: P6 with comfy unimportable, the separate audio clock "
          "(audio_shift on model_sampling) must still refuse" % NAME)
    _need(joint_parts(_old_core_model(_FakeModelType.FLOW, _PlainSampling())) == 1,
          "%s: P6 with comfy unimportable, a plain sampling object must "
          "stay ONE stream" % NAME)
finally:
    if _had is not None:
        sys.modules["comfy"] = _had
        sys.modules["comfy.model_base"] = _fake_mb

# a format whose probe RAISES on an old Core must still reach witness B:
# an exception in witness A is "unanswered", never "answered with one".
_need(joint_parts(types.SimpleNamespace(model=types.SimpleNamespace(
        latent_format=_Exploding(), model_type=_FakeModelType.FLOW_AV,
        model_sampling=_PlainSampling()))) > 1,
      "%s: P6 witness A raising must fall THROUGH to witness B, not return "
      "one stream -- an exception is an unanswered question" % NAME)

# and the master-Core witness must still work with the two new ones in place
_need(joint_parts(_model(_MiniMaxH3AV())) == 2,
      "%s: P6 witness A (fix_empty_latent) must still answer on master "
      "shapes after adding B and C" % NAME)


# ---------------------------------------------------------------------------
# P4 -- fail open
# ---------------------------------------------------------------------------
for tag, obj in (("no .model", types.SimpleNamespace()),
                 ("no latent_format",
                  types.SimpleNamespace(model=types.SimpleNamespace())),
                 ("probe raises", _model(_Exploding())),
                 ("format returns junk",
                  _model(types.SimpleNamespace(
                      latent_channels=16,
                      fix_empty_latent=lambda lat: "not a tensor")))):
    try:
        n = joint_parts(obj)
    except Exception as exc:                       # noqa: BLE001
        _fails.append("%s: P4 %s raised instead of failing open: %r"
                      % (NAME, tag, exc))
        continue
    _need(n == 1,
          "%s: P4 %s must count as ONE stream. A false refusal blocks a "
          "legitimate upscale, which is worse than the crash" % (NAME, tag))


# ---------------------------------------------------------------------------
# P3 / P5 -- detection first, downgrade announced, stages dropped WHOLE
# ---------------------------------------------------------------------------
_up = None
for node in ast.walk(_tree):
    if isinstance(node, ast.FunctionDef) and node.name == "upscale":
        _up = node
        break
_need(_up is not None, "%s: P3 upscale() is gone" % NAME)

if _up is not None:
    calls = []
    for i, stmt in enumerate(_up.body):
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call):
                if isinstance(sub.func, ast.Name):
                    calls.append((i, sub.func.id))
                elif isinstance(sub.func, ast.Attribute):
                    calls.append((i, sub.func.attr))
    names = [c[1] for c in calls]
    _need("_joint_streams" in names,
          "%s: P3 upscale() never runs the detection" % NAME)
    if "_joint_streams" in names:
        pos_det = min(i for i, n in calls if n == "_joint_streams")
        for later in ("_resolve_input", "_apply_sigma_shift", "_refine_tiles",
                      "_build_video"):
            if later in names:
                pos_later = min(i for i, n in calls if n == later)
                _need(pos_det < pos_later,
                      "%s: P3 detection runs AFTER %s. A late answer is what "
                      "cost ten minutes of wall clock in the field"
                      % (NAME, later))
    # the stage list the loop consumes must pass through the dropper,
    # gated on the detection result
    _need("drop_refine_stages" in names,
          "%s: P5 upscale() never drops the refine stages on a joint model "
          "-- the loop would run them into the same crash" % NAME)
    if "drop_refine_stages" in names and "plan_stages" in names:
        _need(min(i for i, n in calls if n == "plan_stages")
              < min(i for i, n in calls if n == "drop_refine_stages"),
              "%s: P5 the dropper must run ON the planned stages" % NAME)
    drop_src = ""
    for sub in ast.walk(_up):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) \
                and sub.func.attr == "drop_refine_stages":
            drop_src = ast.dump(sub)
    _need("_joint" in drop_src,
          "%s: P5 drop_refine_stages is not gated on the detection result -- "
          "then it drops always or never" % NAME)

    # the announcement: the downgrade must be SAID, with the essentials
    ann = ""
    for sub in ast.walk(_up):
        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) \
                and sub.func.id == "print":
            txt = ast.dump(sub)
            if "REFINE" in txt and "PIXEL" in txt:
                ann = txt
    _need(bool(ann),
          "%s: P3 no announcement of the downgrade found -- a silent "
          "degradation is a forbidden quiet path" % NAME)
    for token in ("OFF", "Wan", "%s"):
        _need(token in ann,
              "%s: P3 the announcement must carry %r -- which input, that "
              "the stages are off, and what to wire instead" % (NAME, token))

    # the None-conditioning check must step aside on the joint branch: it
    # lives in the if/else that consumes the detection result
    # A first pin here checked CO-PRESENCE of "_joint" and the call inside
    # one If -- and a mutated `if True:` slipped past because the ELSE branch
    # still mentioned _joint in its announcement. The CONDITION itself must
    # read the detection, and the call and the announcement must sit in
    # OPPOSITE branches: pin the effect, not the neighbourhood.
    guard_ok = False
    for sub in ast.walk(_up):
        if isinstance(sub, ast.If):
            test_d = ast.dump(sub.test)
            body_d = "".join(ast.dump(x) for x in sub.body)
            else_d = "".join(ast.dump(x) for x in sub.orelse)
            if "_joint" not in test_d:
                continue
            in_body = "_reject_none_conditioning" in body_d
            in_else = "_reject_none_conditioning" in else_d
            ann_body = "REFINE" in body_d
            ann_else = "REFINE" in else_d
            if (in_body != in_else) and (ann_body != ann_else) \
                    and (in_body != ann_body):
                guard_ok = True
    _need(guard_ok,
          "%s: P5 the None-conditioning check must sit in the branch the "
          "DETECTION selects (condition reads _joint), opposite the "
          "downgrade announcement -- a pixel-only run needs no conditioning "
          "and must not die on a None one" % NAME)

    # v884: the DELIVERY. Emptying the stages is only half the downgrade --
    # the final pass must run for a joint model REGARDLESS of the pixel_stage
    # dial, or the node ships the input size (the field result: three control
    # frames, all 1344x768, and dark chunk tiles). Pinned as EFFECTS:
    #   (a) one assignment `_final_runs = ... "model final" ... or ... _joint`
    #   (b) upscale() holds NO other comparison against "model final" -- the
    #       v571 "one condition, two places" law collapses to one NAME; a
    #       half-converted build (one place on the name, one on the string)
    #       re-opens the field bug and must go red.
    fr_assign = None
    for sub in ast.walk(_up):
        if isinstance(sub, ast.Assign) and len(sub.targets) == 1 \
                and isinstance(sub.targets[0], ast.Name) \
                and sub.targets[0].id == "_final_runs":
            fr_assign = sub
    _need(fr_assign is not None,
          "%s: P7 upscale() no longer decides the final pass in ONE place "
          "(_final_runs is gone)" % NAME)
    if fr_assign is not None:
        d = ast.dump(fr_assign.value)
        _need("model final" in d and "_joint" in d and "BoolOp" in d
              and "Or" in d,
              "%s: P7 _final_runs must OR the dial position with the joint "
              "detection -- got %s" % (NAME, d[:120]))
    n_cmp = 0
    for sub in ast.walk(_up):
        if isinstance(sub, ast.Compare):
            for c in sub.comparators:
                if isinstance(c, ast.Constant) and c.value == "model final":
                    n_cmp += 1
    _need(n_cmp == 1,
          "%s: P7 found %d comparisons against 'model final' in upscale() -- "
          "exactly ONE may exist (inside _final_runs). A second one is a "
          "half-converted gate and re-opens the field bug" % (NAME, n_cmp))
    n_reads = sum(1 for sub in ast.walk(_up)
                  if isinstance(sub, ast.Name) and sub.id == "_final_runs"
                  and isinstance(sub.ctx, ast.Load))
    _need(n_reads >= 2,
          "%s: P7 _final_runs must gate BOTH places (clock plan + pass "
          "block); found %d read(s)" % (NAME, n_reads))

# drop_refine_stages itself, DRIVEN (pure, lifted from uls_tile_math)
_tm_ns = {}
_tm_tree = ast.parse(open(os.path.join(ROOT, "nodes", "uls_tile_math.py"),
                          encoding="utf-8").read())
_tm_keep = [n for n in _tm_tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("drop_refine_stages", "plan_stages")]
exec(compile(ast.Module(body=_tm_keep, type_ignores=[]), "<tm>", "exec"), _tm_ns)
_drop = _tm_ns.get("drop_refine_stages")
_plan = _tm_ns.get("plan_stages")
_need(_drop is not None and _plan is not None,
      "%s: P5 drop_refine_stages/plan_stages missing from uls_tile_math" % NAME)
if _drop and _plan:
    st1 = _plan(False, 1.0, 0.19, 3, 1.0, 1.0, 0.1, 2, 1.0)
    st2 = _plan(True, 1.0, 0.19, 3, 1.0, 1.0, 0.1, 2, 1.0)
    _need(_drop(st1, True) == [] and _drop(st2, True) == [],
          "%s: P5 a joint model must drop ALL stages (single and dual)" % NAME)
    _need(_drop(st1, False) == st1 and _drop(st2, False) == st2,
          "%s: P5 an ordinary model must keep the stages untouched" % NAME)


if _fails:
    print("FAIL %s" % NAME)
    for f in _fails:
        print("  - %s" % f)
    sys.exit(1)
print("ok %s -- joint AV models detected first by three measured witnesses; "
      "refine stages dropped whole, downgrade announced, pixel path kept; "
      "unanswerable cases fail open" % NAME)
