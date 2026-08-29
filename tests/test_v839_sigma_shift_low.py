#!/usr/bin/env python3
"""Guard v839 -- sigma_shift_low: the LOW expert's own flow-matching shift.

WHY THIS EXISTS
The sampler's sigma_shift patched BOTH experts with one value. A real graph
(8.0 on the HIGH expert, 5.0 on the LOW, via two upstream ModelSamplingSD3
nodes) could not be expressed by the widget. v839 appends `sigma_shift_low`
at the canon end: -1 = "same as high" sentinel (default; a run is
byte-identical to v838), 0 = OFF for the LOW expert, positive = its own shift.

WHAT THIS PINS (driven, not read):
  L1 canon law   -- sigma_shift_low is the LAST required widget, directly
                    after scheduler_low, FLOAT with default -1.0 / min -1.0.
  L2 resolver    -- _resolve_low_shift table: sentinel -> HIGH value, 0 -> 0,
                    own value -> itself, junk/None -> sentinel behaviour.
  L3 apply split -- the lifted apply block, exec-driven with a spy: defaults
                    shift both alike (v838 parity), 8/5 lands 8 on HIGH and
                    5 on LOW, 0/5 patches only LOW, 8/0 only HIGH, an
                    engaged external SIGMAS path patches nothing.
  L4 JS machine  -- driven in node: ORDER_V404 ends on sigma_shift_low (20
                    names), DISPLAY puts it directly under sigma_shift, the
                    v544-length save (19) heals to 20 with the -1 sentinel,
                    the v492 save (17) reaches 20, the v406/v407 phantom
                    reaches 20 (the v544 straggler this cut fixed), and a
                    pre-v404 migration contains NO undefined (the second
                    straggler). _applyModeState greys the widget in Single
                    and in Continuous, and frees it in 'Wan MoE parity'.
  L5 honesty     -- the lifted console-notice block prints exactly when an
                    OWN low value meets 'Continuous' in High + Low.

MUTATIONS LANDED DURING THE BUILD (each turned this guard red):
  M1 the LOW apply line reverted to sigma_shift   -> L3
  M2 _healV544Current loses its push              -> L4
  M3 sigma_shift_low moved before scheduler_low   -> L1
  M4 sigma_shift_low dropped from DUAL_ONLY       -> L4
"""
import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOD = os.path.join(ROOT, "nodes", "uls_sampler.py")
JS = os.path.join(ROOT, "web", "js", "uls_sampler.js")
NAME = "v839"


def _fail(msg):
    print("[test_%s_sigma_shift_low] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


SRC = open(MOD, encoding="utf-8").read()
JSSRC = open(JS, encoding="utf-8").read()

# ---- L1: canon law (AST, no import needed) ---------------------------------
tree = ast.parse(SRC)
order = []
spec = None
for c in ast.walk(tree):
    if isinstance(c, ast.ClassDef) and c.name == "ULSSampler":
        for fn in c.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "INPUT_TYPES":
                for n in ast.walk(fn):
                    if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict):
                        for k, v in zip(n.value.keys, n.value.values):
                            if getattr(k, "value", None) == "required":
                                for kk, vv in zip(v.keys, v.values):
                                    order.append(str(kk.value))
                                    if kk.value == "sigma_shift_low":
                                        spec = vv
_need(order, "L1: ULSSampler INPUT_TYPES not found")
_need(order[-1] == "sigma_shift_low",
      "L1: sigma_shift_low is not the LAST required widget (canon law: growth "
      "goes at the END) -- required ends on %r" % order[-1])
_need(order[-2] == "scheduler_low",
      "L1: sigma_shift_low does not directly follow scheduler_low (%r sits "
      "between)" % order[-2])
_need(spec is not None and isinstance(spec, ast.Tuple), "L1: spec shape drifted")
_need(getattr(spec.elts[0], "value", None) == "FLOAT", "L1: not a FLOAT widget")
def _const(v):
    """A literal, including the negative literal (-1.0 parses as UnaryOp)."""
    if isinstance(v, ast.UnaryOp) and isinstance(v.op, ast.USub):
        inner = getattr(v.operand, "value", None)
        return -inner if isinstance(inner, (int, float)) else None
    return getattr(v, "value", None)


opts = {k.value: _const(v)
        for k, v in zip(spec.elts[1].keys, spec.elts[1].values)
        if isinstance(k, ast.Constant)}
_need(opts.get("default") == -1.0,
      "L1: default is %r, must be the -1.0 'same as high' sentinel (anything "
      "else changes every existing run)" % (opts.get("default"),))
_need(opts.get("min") == -1.0, "L1: min is %r, must admit the sentinel" % (opts.get("min"),))

# ---- load the module behind stubs (the v830 pattern) -----------------------
tmp = tempfile.mkdtemp()
fp = types.ModuleType("folder_paths")
fp.models_dir = tmp
fp.get_filename_list = lambda f: []
fp.get_full_path = lambda f, n: None
fp.get_folder_paths = lambda f: [tmp]
sys.modules["folder_paths"] = fp
for name in ("comfy", "comfy.samplers", "comfy.sample", "comfy.utils",
             "comfy.model_management", "comfy.cli_args"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["comfy.cli_args"].args = types.SimpleNamespace(preview_method=None)
sys.modules["comfy.cli_args"].LatentPreviewMethod = object
sys.modules["comfy"].samplers = sys.modules["comfy.samplers"]
sys.modules["comfy"].sample = sys.modules["comfy.sample"]
sys.modules["comfy"].utils = sys.modules["comfy.utils"]
sys.modules["comfy.samplers"].KSampler = types.SimpleNamespace(
    SAMPLERS=["euler"], SCHEDULERS=["simple"])

pkg = types.ModuleType("plspack")
pkg.__path__ = [os.path.join(ROOT, "nodes")]
sys.modules["plspack"] = pkg
spec_ = importlib.util.spec_from_file_location("plspack.uls_sampler", MOD)
m = importlib.util.module_from_spec(spec_)
sys.modules["plspack.uls_sampler"] = m
spec_.loader.exec_module(m)

# ---- L2: resolver table ----------------------------------------------------
R = m._resolve_low_shift
for shift, low, want in ((8.0, -1.0, 8.0), (8.0, 5.0, 5.0), (8.0, 0.0, 0.0),
                         (0.0, 5.0, 5.0), (0.0, -1.0, 0.0), (8.0, None, 8.0),
                         (8.0, "junk", 8.0), (8.0, -0.5, 8.0)):
    got = R(shift, low)
    _need(got == want,
          "L2: _resolve_low_shift(%r, %r) -> %r, want %r" % (shift, low, got, want))

# ---- L3: the apply block, lifted and DRIVEN with a spy ---------------------
_need("_apply_sigma_shift(model_low, _low_shift)" in SRC,
      "L3: the LOW apply must use the RESOLVED value (_low_shift) -- the "
      "one-value-for-both v838 form is the wound this cut heals")
start = SRC.index("_low_shift = _resolve_low_shift(")
end = SRC.index("_apply_sigma_shift(model_low, _low_shift)", start)
end = SRC.index("\n", end)
block = textwrap.dedent(SRC[SRC.rindex("\n", 0, start) + 1:end])
block = textwrap.dedent(block)

# v896: the seam's own dependencies, lifted CLOSED out of the module instead of
# hand-listed. tests/_lift.py reports a short lift BY NAME, so the next time a
# cut gives this block a new helper the failure says so, instead of surfacing as
# a bare NameError that reads like a broken tree.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lift as _LIFT
_RAGGED_SRC, _RAGGED_MISSING = _LIFT.close_over(SRC, ["_is_ragged_latent"], set())
_need(not _RAGGED_MISSING,
      "L3: lift of _is_ragged_latent is SHORT of %s -- a GUARD fault, not a "
      "tree fault" % ", ".join(_RAGGED_MISSING))


def _drive(shift, low, dual=True, sig=None, sig_h=None, sig_l=None,
           latent=None):
    calls = []

    def spy(mdl, s):
        calls.append((mdl, float(s)))
        return mdl
    ns = {"_resolve_low_shift": R, "_apply_sigma_shift": spy,
          "sigma_shift": shift, "sigma_shift_low": low, "dual_moe": dual,
          "model": "HI", "model_low": "LO",
          "sigmas": sig, "sigmas_high": sig_h, "sigmas_low": sig_l,
          # v896: the block gained `_is_ragged_latent` and `latent_image` in
          # v870 and this harness was not updated -- it raised NameError and
          # this guard stood red for many versions while the tree was fine.
          # The helper is LIFTED from the module (real arithmetic, not a stub);
          # the latent is injected because it is an input, not a dependency.
          "latent_image": latent if latent is not None else {"samples": None}}
    exec(_RAGGED_SRC, ns)
    exec(compile(textwrap.dedent(block), "<applyblock>", "exec"), ns)
    return calls


_need(_drive(8.0, -1.0) == [("HI", 8.0), ("LO", 8.0)],
      "L3: defaults must reproduce v838 (one shift, both experts) -- got %r"
      % _drive(8.0, -1.0))
_need(_drive(8.0, 5.0) == [("HI", 8.0), ("LO", 5.0)],
      "L3: 8/5 must land 8 on HIGH and 5 on LOW -- got %r" % _drive(8.0, 5.0))
_need(_drive(0.0, 5.0) == [("LO", 5.0)],
      "L3: high OFF + own low must patch ONLY the LOW expert -- got %r"
      % _drive(0.0, 5.0))
_need(_drive(8.0, 0.0) == [("HI", 8.0)],
      "L3: low OFF must leave the LOW expert untouched -- got %r"
      % _drive(8.0, 0.0))
_need(_drive(0.0, -1.0) == [], "L3: all-off must patch nothing")
_need(_drive(8.0, 5.0, dual=True, sig="CURVE") == [],
      "L3: an engaged external SIGMAS path owns the schedule -- nothing may "
      "be patched (v495 law)")

# ---- L5: the honesty notice, lifted and driven -----------------------------
h0 = SRC.index("_sl_own = (float(sigma_shift_low)")
h0 = SRC.rindex("try:", 0, h0)
h1 = SRC.index("'Wan MoE parity'.\")", h0)
h1 = SRC.index("\n", h1)
notice = textwrap.dedent(SRC[SRC.rindex("\n", 0, h0) + 1:h1])


def _says(shift, low, dual, mode):
    said = []
    ns = {"sigma_shift": shift, "sigma_shift_low": low, "dual_moe": dual,
          "handoff_mode": mode, "print": lambda *a, **k: said.append(a)}
    exec(compile(textwrap.dedent(notice), "<notice>", "exec"), ns)
    return bool(said)


_need(_says(8.0, 5.0, True, "Continuous"),
      "L5: an OWN low value in Continuous must say it is inert")
_need(not _says(8.0, 5.0, True, "Wan MoE parity"),
      "L5: parity must stay silent (the value bites there)")
_need(not _says(8.0, -1.0, True, "Continuous"),
      "L5: the sentinel is not an OWN value -- no notice")
_need(not _says(8.0, 5.0, False, "Continuous"),
      "L5: Single mode is the widget's grey, not this notice")

# ---- L4: the JS machine, driven in node ------------------------------------
def _lift(sig):
    i = JSSRC.index(sig)
    j = JSSRC.index("\n}", i)
    return JSSRC[i:j + 2]



def _grab_line(sig):
    """One whole statement line, lifted verbatim."""
    i = JSSRC.index(sig)
    j = JSSRC.index("\n", i)
    return JSSRC[i:j]

js_parts = []
# constants + arrays (slice from ORDER_V404 through the maps)
for sig in ("const STATE_WIDGETS", "const ORDER_V404", "const DISPLAY_ORDER"):
    i = JSSRC.index(sig)
    j = JSSRC.index("];", i)
    js_parts.append(JSSRC[i:j + 2])
for line_sig in ("const DUAL_ONLY", "const SINGLE_ONLY",
                 "const SIGMA_INERT", "const INACTIVE_MARK",
                 "const SAME_AS_HIGH", "const DEFAULT_SAMPLER_LOW",
                 "const DEFAULT_SCHEDULER_LOW", "const DEFAULT_SIGMA_SHIFT_LOW",
                 "const DEFAULT_PREVIEW_MODE", "const DEFAULT_SIGMA_SHIFT",
                 "const DEFAULT_HANDOFF_MODE", "const LEN_PHANTOM",
                 "const LEN_PRE_V404", "const LEN_V492_CURRENT",
                 "const LEN_V544_CURRENT", "const LEGACY_PRESET_VALUES",
                 "const ORDER_PRE_V404", "const CONTROL_VALUES"):
    i = JSSRC.index(line_sig)
    j = JSSRC.index(";", JSSRC.index("=", i))
    # arrays span to ]; -- take the longer end
    k = JSSRC.find("];", i)
    if k != -1 and k < JSSRC.index("\nconst", i + 1 if JSSRC.find("\nconst", i + 1) != -1 else i):
        j = max(j, k + 1)
    js_parts.append(JSSRC[i:j + 1])
# v896: the v888 rewrite also introduced module state (`let _visMoved`) and a
# reader (`_visChanged`) that _applyModeState calls. Lift BOTH -- a bench that
# invents a replacement would be testing its own invention.
js_parts.append(_grab_line("let _visMoved"))
js_parts.append(_lift("function _visChanged"))

for fn in ("function _looksPhantomPreset", "function _healPhantomPreset",
           "function _looksPreV404", "function _migratePreV404",
           "function _looksV492Current", "function _healV492Current",
           "function _looksV544Current", "function _healV544Current",
           "function _findWidget", "function _inputLinked",
           "function _setDisabled", "function _applyModeState"):
    js_parts.append(_lift(fn))

# v896: v888 moved _setDisabled onto the SHARED module web/js/ph_widget_vis.js
# (setHidden / refit). This harness lifts functions out of uls_sampler.js and so
# inherited a reference it does not define -- ReferenceError: setHidden. It went
# unseen because the python half of this guard was already red on the v870
# NameError; one wound hid the other. The REAL module is placed alongside, per
# the standing rule that a JS bench which strips imports must lay the genuine
# files next to it -- never a stub of the thing under test.
_VIS = os.path.join(ROOT, "web", "js", "ph_widget_vis.js")
_vis_src = open(_VIS, encoding="utf-8").read()
_vis_src = re.sub(r"^\s*import\s+.*?;\s*$", "", _vis_src, flags=re.M)
_vis_src = _vis_src.replace("export function", "function").replace("export const", "const")
js_parts.insert(0, _vis_src)

harness = "\n".join(js_parts) + r"""

const out = {};
out.orderLen = ORDER_V404.length;
out.orderLast = ORDER_V404[ORDER_V404.length - 1];
out.displayAdj = DISPLAY_ORDER[DISPLAY_ORDER.indexOf("sigma_shift") + 1];
out.dualOnly = DUAL_ONLY.includes("sigma_shift_low");
// v839: NOT in SIGMA_INERT on purpose -- that list re-enables its members when
// no curve is wired, which would overwrite the DUAL_ONLY greying in Single.
out.sigmaInertClean = !SIGMA_INERT.includes("sigma_shift_low");
out.defLow = DEFAULT_SIGMA_SHIFT_LOW;

// v544-length save heals to 20 with the sentinel at the end
const v544 = [true, 0.875, 1.0, 1, "fixed", 8, 1.0, "res_2s", "beta57", 1.0,
              true, 0, 10000, false, "Still · ComfyUI", 0.0, "Continuous",
              "same as high", "same as high"];
const healed = _looksV544Current(v544) ? _healV544Current(v544) : null;
out.heal544 = healed ? [healed.length, healed[healed.length - 1]] : null;

// v492 save (17) must reach 20 through its own heal
const v492 = v544.slice(0, 17);
out.heal492 = _looksV492Current(v492)
    ? (() => { const h = _healV492Current(v492); return [h.length, h[h.length - 1]]; })()
    : null;

// v406/v407 phantom (15, preset string at wv[1]) must reach 20 (straggler fix)
const phantom = [true, "Custom", 1.0, 1, "fixed", 8, 1.0, "res_2s", "beta57",
                 1.0, true, 0, 10000, false, "x"];
out.healPhantom = _looksPhantomPreset(phantom)
    ? (() => { const h = _healPhantomPreset(phantom); return [h.length, h[h.length - 1]]; })()
    : null;

// pre-v404 migration must carry NO undefined (straggler fix)
const pre = [1, "fixed", 8, 1.0, "res_2s", "beta57", 1.0, true, 0, 10000,
             false, true, 0.875, 1.0];
out.migrate = _looksPreV404(pre)
    ? (() => { const g = _migratePreV404(pre);
               return [g.length, g.some((x) => x === undefined)]; })()
    : null;

// grey truth, driven: Single greys it, Continuous greys it, parity frees it
function fakeNode(dual, mode, wired) {
    const names = ORDER_V404.filter((n) => n !== "control_after_generate");
    const node = {
        inputs: (wired || []).map((n) => ({ name: n, link: 7 })),
        widgets: names.map((n) => ({ name: n, value: 0 })),
    };
    node.widgets.find((w) => w.name === "dual_moe").value = dual;
    node.widgets.find((w) => w.name === "handoff_mode").value = mode;
    return node;
}
function lowDisabled(dual, mode, wired) {
    const n = fakeNode(dual, mode, wired);
    _applyModeState(n);
    return !!n.widgets.find((w) => w.name === "sigma_shift_low").disabled;
}
out.greySingle = lowDisabled(false, "Continuous");
out.greyCont = lowDisabled(true, "Continuous");
out.freeParity = !lowDisabled(true, "Wan MoE parity");
// an engaged external curve greys it even in parity (the curve owns the schedule)
out.greyExt = lowDisabled(true, "Wan MoE parity", ["sigmas_high", "sigmas_low"]);

console.log(JSON.stringify(out));
"""
mjs = os.path.join(tempfile.mkdtemp(), "drive.mjs")
open(mjs, "w", encoding="utf-8").write(harness)
r = subprocess.run(["node", mjs], capture_output=True, text=True, timeout=60)
_need(r.returncode == 0, "L4: node harness failed:\n%s" % (r.stderr[-800:],))
out = json.loads(r.stdout.strip().splitlines()[-1])
_need(out["orderLen"] == 20 and out["orderLast"] == "sigma_shift_low",
      "L4: ORDER_V404 must end on sigma_shift_low at length 20 -- got %s/%s"
      % (out["orderLen"], out["orderLast"]))
_need(out["displayAdj"] == "sigma_shift_low",
      "L4: DISPLAY must put sigma_shift_low directly under sigma_shift")
_need(out["dualOnly"], "L4: sigma_shift_low missing from DUAL_ONLY (Single "
      "would show a live widget the run ignores)")
_need(out["sigmaInertClean"],
      "L4: sigma_shift_low crept into SIGMA_INERT -- that list RE-ENABLES its "
      "members when no curve is wired and would overwrite the Single greying")
_need(out["defLow"] == -1.0, "L4: JS default drifted from the -1 sentinel")
_need(out["heal544"] == [20, -1.0],
      "L4: a v544..v838 save (19) must heal to 20 with the sentinel -- got %r"
      % (out["heal544"],))
_need(out["heal492"] == [20, -1.0],
      "L4: a v492 save (17) must reach 20 -- got %r" % (out["heal492"],))
_need(out["healPhantom"] == [20, -1.0],
      "L4: the phantom heal must reach 20 (it had stopped at 17 since v544) "
      "-- got %r" % (out["healPhantom"],))
_need(out["migrate"] == [20, False],
      "L4: a pre-v404 migration must be 20 long with NO undefined -- got %r"
      % (out["migrate"],))
_need(out["greySingle"], "L4: Single must grey sigma_shift_low (DUAL_ONLY)")
_need(out["greyCont"], "L4: Continuous must grey sigma_shift_low (ONE "
      "schedule, built from the HIGH expert)")
_need(out["freeParity"], "L4: 'Wan MoE parity' must FREE sigma_shift_low "
      "(the LOW segment's schedule is built from the LOW expert)")
_need(out["greyExt"], "L4: an engaged external SIGMAS pair must grey "
      "sigma_shift_low even in parity (the curve owns the schedule)")

print("[test_%s_sigma_shift_low] OK -- canon appends at the end (-1 sentinel), "
      "the resolver and the apply split drive true (8/5 lands 8 HIGH / 5 LOW), "
      "every save format heals to 20, and the grey tells the truth in all "
      "three states" % NAME)
