# -*- coding: ascii -*-
"""v935 -- a PDD Acc file is applied WHOLE by the Stack/Engine, and the
Sampler runs the schedule its heads were trained for.

WHAT WAS WRONG BEFORE (measured 10.09. against core 7a131a3a)
-------------------------------------------------------------
v933's building block had never run. Driven against core's real classes it
failed three ways: the wrapper read `timestep` (= sigma * 1000) as sigma, so
every forward would have been off-grid; patch_model raised TypeError because
the stand-in for final_layer was not an nn.Module; and core's FinalLayer.forward
calls its OWN projections, so the heads would never have been reached -- the
run would have produced core's output, silently. Nothing called the blocks
either: SEQ hands core's loader a NAME, the v930 conversion lived only on the
merge path and the v931 rebase was called by nobody.

WHAT IS PINNED HERE
-------------------
  A  pure: the Sampler's decision (pdd_plan) over the whole matrix, the step
     mirror against the maths, and the audit that every trunk module lands on
     the model and fits it
  B  structure (AST, code only): the PDD switch sits in the Sampler before
     EVERY path; the gate routes before the apply decision; _apply_pdd audits
     before it patches and never rebinds the input model, so every refusal
     hands back exactly what came in
  C  core-backed (subprocess, tests/_v935_core_half.py): dense and curve-form
     models, five refusals, mixed rows, the heads on core's real FinalLayer
     through core's real wrapper path, core's Euler on the block starts, and
     the Sampler's preparation. Runs when a ComfyUI core is reachable
     (PLS_CORE_ROOT, or the ComfyUI root around the pack); otherwise it says
     SKIPPED -- out loud, never silently.
"""
import ast
import json
import os
import pathlib
import re
import subprocess
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lift as _LIFT  # noqa: E402
import uls_pdd_math as M  # noqa: E402

FAILED = []
CHECKS = []


def _fail(msg):
    FAILED.append(msg)
    CHECKS.append(msg)
    print("FAIL: {}".format(msg))


def _ok(msg):
    CHECKS.append(msg)
    print("ok  : {}".format(msg))


def _need(cond, msg):
    _ok(msg) if cond else _fail(msg)


def _code(src):
    """Source with comments and docstrings gone -- guards read CODE."""
    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(body, list) and body and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)):
            body[0].value.value = ""
    return ast.unparse(tree)


SAMPLER = (ROOT / "nodes" / "uls_sampler.py").read_text(encoding="utf-8")
STACK = (ROOT / "nodes" / "uls_stack_node.py").read_text(encoding="utf-8")


def _lift(src, names):
    code, missing = _LIFT.close_over(src, names, provided={"torch", "os"})
    if missing:
        _fail("lift of %s is short of %s (GUARD fault, not a tree fault)"
              % (names, sorted(missing)))
        return None
    ns = {"torch": torch, "os": os}
    exec(compile(code, "<lift:%s>" % ",".join(names), "exec"), ns)  # noqa: S102
    return ns


# --- A: the Sampler's decision -------------------------------------------------
NS = _lift(SAMPLER, ["pdd_plan", "PDD_LEGAL_STEPS", "PDD_SAMPLER"])
if NS:
    plan = NS["pdd_plan"]
    _need(tuple(NS["PDD_LEGAL_STEPS"]) == tuple(M.LEGAL_NFE),
          "A  the Sampler's step list mirrors uls_pdd_math.LEGAL_NFE exactly")
    for n in M.LEGAL_NFE:
        got = plan(n, "euler", 1.0, 0, 10000, False)
        _need(got[0] == n and got[1] == [],
              "A  %d steps, euler, full schedule -> fused for %d, no notes" % (n, n))

    def _refused(label, *a, **k):
        try:
            plan(*a, **k)
        except ValueError as ex:
            return str(ex)
        _fail("A  %s was accepted" % label)
        return None

    for bad in (3, 9, 12, 20):
        msg = _refused("%d steps" % bad, bad, "euler", 1.0, 0, 10000, False)
        if msg is not None:
            _need("4/5/6/7/8" in msg,
                  "A  %d steps refused, naming the legal counts" % bad)
    for sname in ("dpmpp_2m", "euler_ancestral", "heun", "res_multistep"):
        msg = _refused(sname, 8, sname, 1.0, 0, 10000, False)
        if msg is not None:
            _need("'euler'" in msg, "A  sampler %s refused, naming euler" % sname)
    msg = _refused("denoise 0.6", 8, "euler", 0.6, 0, 10000, False)
    if msg is not None:
        _need("denoise" in msg, "A  denoise < 1 refused")
    msg = _refused("High + Low", 8, "euler", 1.0, 0, 10000, True)
    if msg is not None:
        _need("Single" in msg, "A  High + Low refused, naming Single")
    for st_, en in ((2, 10000), (0, 5)):
        msg = _refused("start %d end %d" % (st_, en), 8, "euler", 1.0, st_, en, False)
        if msg is not None:
            _need("cut" in msg, "A  start %d / end %d refused (would cut)" % (st_, en))
    got = plan(20, "euler", 1.0, 0, 10000, False, ext_steps=6)
    _need(got[0] == 6 and "external SIGMAS" in got[1][0],
          "A  external SIGMAS own the count (steps widget 20 ignored -> 6)")
    got = plan(20, "euler", 1.0, 0, 10000, False, ext_steps=13)
    _need(got[0] is None and got[1],
          "A  an external count with no partition keeps the fusion, with a note")
    try:
        plan(8, "dpmpp_2m", 1.0, 0, 10000, False, ext_steps=8)
        _fail("A  external SIGMAS let a non-euler sampler through")
    except ValueError:
        _ok("A  external SIGMAS do not lift the euler rule")

# --- A: every trunk module lands, and fits ---------------------------------------
NS2 = _lift(STACK, ["_pdd_audit", "_pdd_name_trunk"])
if NS2:
    audit = NS2["_pdd_audit"]
    t = torch.zeros
    km = {"diffusion_model.m": "diffusion_model.m.weight"}
    shapes = {"diffusion_model.m.weight": (6, 4), "diffusion_model.m.bias": (6,)}
    good = {"diffusion_model.m.lora_A.weight": t(2, 4),
            "diffusion_model.m.lora_B.weight": t(6, 2),
            "diffusion_model.m.alpha": t(()),
            "diffusion_model.m.diff_b": t(6)}
    _need(audit(good, km, shapes) == (1, []), "A  a fitting module passes clean")
    cases = [
        ("a stray tensor", dict(good, **{"transformer_blocks.0.attn.to_q.lora_down.weight": t(2, 4)}), km, "no known factor form"),
        ("a module the model lacks", dict(good, **{"diffusion_model.x.lora_A.weight": t(2, 4), "diffusion_model.x.lora_B.weight": t(6, 2)}), km, "no such module"),
        ("a missing factor", {"diffusion_model.m.lora_A.weight": t(2, 4)}, km, "factor is missing"),
        ("factors of the wrong width", dict(good, **{"diffusion_model.m.lora_A.weight": t(2, 14)}), km, "do not fit the weight"),
        ("a bias delta of the wrong size", dict(good, **{"diffusion_model.m.diff_b": t(5)}), km, "bias delta"),
    ]
    for label, td, kmap, needle in cases:
        n, probs = audit(td, kmap, shapes)
        _need(any(needle in p for p in probs), "A  audit flags %s" % label)
    name_trunk = NS2["_pdd_name_trunk"]
    _need(name_trunk("MiniMax-H3-FL2VA-Acc-8Step.safetensors") == "fl2va"
          and name_trunk("x/MiniMax-H3-Ref2VA-Acc-8Step.safetensors") == "ref2va"
          and name_trunk("turbo.safetensors") is None,
          "A  the file-name trunk hint reads fl2va / ref2va (a warning, never identity)")

# --- B: structure ------------------------------------------------------------------
tree = ast.parse(SAMPLER)
run = next((n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "_run"), None)
_need(run is not None, "B  Sampler._run found")
if run is not None:
    prep = [n.lineno for n in ast.walk(run) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == "_pdd_prepare"]
    ifs = [n.lineno for n in ast.walk(run) if isinstance(n, ast.If)
           and isinstance(n.test, ast.Name) and n.test.id == "dual_moe"]
    rets = [n.lineno for n in ast.walk(run) if isinstance(n, ast.Return)]
    _need(len(prep) == 1, "B  _run CALLS _pdd_prepare exactly once")
    _need(prep and ifs and prep[0] < min(ifs),
          "B  ... before the High + Low branch")
    _need(prep and rets and prep[0] < min(rets),
          "B  ... and before every sampling return")
    assigned = any(isinstance(n, ast.Assign) and n.lineno == prep[0]
                   and {getattr(t_, "id", None) for t_ in
                        getattr(n.targets[0], "elts", [n.targets[0]])} >= {"model", "sigmas"}
                   for n in ast.walk(run)) if prep else False
    _need(assigned, "B  ... and its model AND sigmas are the ones used after it")
prep_fn = next((n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_pdd_prepare"), None)
if prep_fn is not None:
    body = _code(ast.unparse(prep_fn))
    _need("pdd_plan(" in body and ".refit(" in body and ".boundaries" in body,
          "B  _pdd_prepare plans, re-fuses and builds the schedule from the bank")
    first = prep_fn.body[1] if len(prep_fn.body) > 1 else None
    _need("state_of(model) is None" in body.split(".refit(")[0],
          "B  _pdd_prepare leaves a model without a bank alone before anything else")

stree = ast.parse(STACK)
als = next(n for n in ast.walk(stree)
           if isinstance(n, ast.FunctionDef) and n.name == "apply_lora_set")
als_code = _code(ast.unparse(als))
_need("_PAYLOAD_HANDLERS[" in als_code
      and als_code.index("_PAYLOAD_HANDLERS[") < als_code.index("_apply_decision("),
      "B  the gate hands payload files to their whole-file path before the apply decision")
ap = next(n for n in ast.walk(stree)
          if isinstance(n, ast.FunctionDef) and n.name == "_apply_pdd")
rebinds = [n.lineno for n in ast.walk(ap)
           if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))
           for tg in ([n.target] if not isinstance(n, ast.Assign) else n.targets)
           for nm in ast.walk(tg) if isinstance(nm, ast.Name) and nm.id == "model"]
_need(not rebinds,
      "B  _apply_pdd never rebinds `model` -- every refusal returns the input untouched")
ap_code = _code(ast.unparse(ap))
_need("_pdd_audit(" in ap_code and ".add_patches(" in ap_code
      and ap_code.index("_pdd_audit(") < ap_code.index(".add_patches("),
      "B  _apply_pdd audits every module BEFORE it patches anything")
_need(".add_patches(" in ap_code and ".attach(" in ap_code
      and ap_code.index(".add_patches(") < ap_code.index(".attach("),
      "B  ... and attaches the bank only after the whole trunk landed")
_need("_apply_seq" not in ap_code and "load_lora(m" not in ap_code,
      "B  _apply_pdd never hands the file to a name-based loader")
_need(re.search(r"_PAYLOAD_HANDLERS\s*=\s*\{[^}]*_apply_pdd", STACK) is not None,
      "B  _apply_pdd is registered as the PDD family's handler")


def _if_returns_refuse(fn, test_src):
    """An `if <test_src>:` inside fn whose body RETURNS _refuse(...)."""
    for n in ast.walk(fn):
        if isinstance(n, ast.If) and ast.unparse(n.test) == test_src:
            for r in ast.walk(n):
                if (isinstance(r, ast.Return) and isinstance(r.value, ast.Call)
                        and getattr(r.value.func, "id", None) == "_refuse"):
                    return True
    return False


# The three promises the core half drives, pinned in CODE as well, so a cold
# start without core is not blind to them (mutation round 10.09.: without these
# three, "audit ignored", "no rebase" and "no conversion" passed A and B).
_need(_if_returns_refuse(ap, "problems"),
      "B  a non-empty audit REFUSES the file (never patches around a gap)")
rebase_guarded = any(
    isinstance(n, ast.If) and ast.unparse(n.test) == "table is not None"
    and "rebase_lora_adaln(" in ast.unparse(n) for n in ast.walk(ap))
_need(rebase_guarded,
      "B  on a curve-form model (table is not None) the adaln rebase runs")
_need(_if_returns_refuse(ap, "prev is not None"),
      "B  a model that already carries a bank refuses a second one")
_need(_if_returns_refuse(ap, "hit is None"),
      "B  a curve table without a matching basis refuses the file")
_need("_convert_lora_like_core(" in ap_code
      and ap_code.index("_convert_lora_like_core(") < ap_code.index("_pdd_audit("),
      "B  the trunk is converted (core + v930) before it is audited")

# --- C: core-backed ------------------------------------------------------------------
def _core_root():
    env = os.environ.get("PLS_CORE_ROOT")
    cands = [env] if env else []
    cands.append(str(ROOT.parent.parent))       # custom_nodes/<pack> -> ComfyUI
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "comfy", "ldm", "minimax", "model.py")):
            return c
    return None


CORE = _core_root()
if CORE is None:
    print("note: C  SKIPPED -- no ComfyUI core with comfy/ldm/minimax found "
          "(set PLS_CORE_ROOT). A and B above still ran.")
else:
    helper = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_v935_core_half.py")
    p = subprocess.run([sys.executable, helper, CORE, str(ROOT)],
                       capture_output=True, text=True, timeout=240)
    lines = [ln for ln in p.stdout.splitlines() if ln.startswith("{")]
    if p.returncode != 0 or not lines:
        _fail("C  core half did not run: %s" % (p.stderr or p.stdout)[-400:])
    else:
        R = json.loads(lines[-1])
        d = R["dense"]
        _need(d["errs"] == [] and d["loader"] == [] and d["state"] and d["nfe"] == 8,
              "C  dense H3: applied whole, never through the loader, bank attached")
        _need(d["patches"] == 5 and d["delta_exact"],
              "C  ... all five trunk modules baked; out_proj delta == up@down*alpha/rank")
        c = R["curve"]
        _need(c["errs"] == [] and c["state"] and c["bias_patch"]
              and c["width"] == 8 and c["bias_moved"],
              "C  curve-form H3: adaln rebased to curve width, constant term as bias")
        for key, label in (("no_basis", "a curve table with no basis"),
                           ("missing_module", "a trunk module the model lacks"),
                           ("second_bank", "a second bank on the same model"),
                           ("not_h3", "a non-H3 target")):
            r_ = R[key]
            _need(len(r_["errs"]) == 1 and "Refused (PDD)" in r_["errs"][0]
                  and r_["untouched"],
                  "C  %s: refused whole, model untouched" % label)
        mx = R["mixed"]
        _need(mx["errs"] == [] and mx["loader"] == ["char.safetensors"] and mx["state"],
              "C  PDD beside a plain LoRA: the loader sees only the plain one")
        h = R["heads"]
        _need(h["forward"] and h["fallback"] and h["offgrid"] and h["native_back"],
              "C  core's real FinalLayer serves the fused head of the live block, "
              "falls back to timestep/multiplier, refuses off-grid, unpatches clean")
        _need(all(v == list(range(int(k))) for k, v in R["euler"].items()),
              "C  core's Euler evaluates exactly the block starts for 4..8 steps")
        s = R["sampler"]
        _need(s["nfe"] == 6 and s["sig_match"],
              "C  Sampler: steps 6 -> heads re-fused for 6, schedule = the bank's starts")
        _need(s["refuse12"] and s["ext_nfe"] == 4,
              "C  Sampler: 12 steps refused; external SIGMAS (4) own the count")
        _need(s["no_bank_untouched"],
              "C  Sampler: a model without a bank passes through untouched")

print("\n%d checks, %d failed" % (len(CHECKS), len(FAILED)))
if FAILED:
    print("\n".join(FAILED))
    sys.exit(1)
print("v935 pdd wiring: PASS")
