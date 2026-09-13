#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v911 -- the "almost full" notice, second half of the v908 wound.

WHERE THIS CAME FROM. Frank's screen, 30.08.: with the clip wired the Token
Counter dropped under 512 and the sticky toast changed to "Token budget almost
full -- Quality may start to degrade". v908 had gated the OVER-budget claim on
can_truncate and left this one ungated. On qwen3vl nothing is cut (the encoder
carries max_length=99999999), and the degradation it points at -- motion
slowing, grid patterns, kijai issue #1781 -- comes from a fixed 512-wide
buffer in kijai's WanVideoWrapper, which such a graph does not run. The notice
said something true-sounding for a reason that does not apply.

THE CUT: both wordings are now PURE FUNCTIONS, one per side of the fence, and
both are gated on the same can_truncate the over-budget branch uses. The
report and the toast say the same thing again -- the v908 lesson was that a
node contradicting itself on one screen is worse than either half alone.

WHAT IS PINNED:

  R1  _near_limit_hints exists, is pure, and is DRIVEN on both sides: the
      capped wording keeps the v318 warning verbatim; the capless one makes no
      degradation claim and names the mark as the user's own.
  R2  the report's near-limit branch actually CALLS it, with the measured
      can_truncate -- not with a constant (checked on the AST).
  F1  nearNotice() lifted out of the JS and RUN in node: same two shapes, and
      the capless one carries neither "degrade" nor a kijai reference.
  F2  the toast and the report AGREE per side -- both say "nothing is cut"
      where nothing is cut, and neither does where something can be.
  F3  can_truncate keeps its v908 safe default: this guard fails if the
      capless wording could ever appear for an unwired clip.
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = os.path.join(ROOT, "nodes", "uls_stack_node.py")
JS = os.path.join(ROOT, "web", "js", "uls_token_toast.js")

FAILED = []


def _need(cond, msg):
    print("ok  : " + msg) if cond else (FAILED.append(msg),
                                        print("FAIL: " + msg))


def _read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def _node():
    for cand in ("node", "nodejs"):
        try:
            subprocess.run([cand, "--version"], capture_output=True, timeout=30)
            return cand
        except OSError:
            continue
    return "node"


SRC = _read(PY)
SRC_JS = _read(JS)
TREE = ast.parse(SRC)

# --- R1: lift and drive the report wording ---------------------------------
_fn = [n for n in TREE.body
       if isinstance(n, ast.FunctionDef) and n.name == "_near_limit_hints"]
_need(bool(_fn), "R1: _near_limit_hints is a top-level function")

capped = capless = None
if _fn:
    ns = {}
    exec(compile(ast.Module(body=_fn, type_ignores=[]), "<v911>", "exec"), ns)
    f = ns["_near_limit_hints"]
    capped = "\n".join(f(460, 512, 0.90, True))
    capless = "\n".join(f(460, 512, 0.90, False, "qwen3vl_32b"))

    _need("Approaching limit" in capped and "kijai issue #1781" in capped,
          "R1: the capped wording keeps the v318 warning")
    _need("degrade" not in capless,
          "R1: the capless wording makes NO degradation claim")
    _need("not the encoder's" in capless and "Nothing is cut" in capless,
          "R1: the capless wording names the mark as the user's own")
    _need("qwen3vl_32b" in capless,
          "R1: the capless wording names the encoder it measured")
    _need("Nothing is cut here:" in "\n".join(f(460, 512, 0.90, False)),
          "R1: with no encoder name it still reads as a sentence")
    _need(capped != capless, "R1: the two sides really differ")

# --- R2: the report calls it, with the MEASURED flag ------------------------
_calls = [n for n in ast.walk(TREE)
          if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
          and n.func.id == "_near_limit_hints"]
_need(len(_calls) == 1, "R2: exactly one call site -- got %d" % len(_calls))
if _calls:
    args = [ast.unparse(a) for a in _calls[0].args]
    _need(any("_any_encoder_truncates" in a for a in args),
          "R2: the call passes the MEASURED can_truncate, not a constant -- "
          "got %r" % (args,))
    _need(not any(isinstance(a, ast.Constant) and isinstance(a.value, bool)
                  for a in _calls[0].args),
          "R2: no literal True/False is handed in")

# --- F1: lift and RUN the toast wording ------------------------------------
_m = re.search(r"export function nearNotice\(info\) \{.*?\n\}\n", SRC_JS, re.S)
_need(_m is not None, "F1: nearNotice() is liftable from the frontend file")
js_capped = js_capless = None
if _m:
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(_m.group(0).replace("export ", ""))
        fh.write("const base = {pos: 478, neg: 356, limit: 512, warn_at: 460};\n")
        fh.write("process.stdout.write(JSON.stringify([\n")
        fh.write("  nearNotice({...base, can_truncate: true}),\n")
        fh.write("  nearNotice({...base, can_truncate: false, "
                 "encoder: 'qwen3vl_32b'}),\n")
        fh.write("  nearNotice({...base, can_truncate: false})]));\n")
        tmp = fh.name
    try:
        run = subprocess.run([_node(), tmp], capture_output=True, text=True,
                             timeout=60)
        if run.returncode != 0:
            _need(False, "F1: nearNotice() does not execute: %s"
                  % run.stderr.strip()[:120])
        else:
            got = json.loads(run.stdout)
            js_capped, js_capless, js_noname = got
            _need("almost full" in js_capped["summary"]
                  and "degrade" in js_capped["detail"],
                  "F1: the capped notice keeps the v318 wording")
            _need("degrade" not in js_capless["detail"],
                  "F1: the capless notice claims no degradation")
            _need("kijai" not in js_capless["detail"].lower()
                  and "WanVideo" not in js_capless["detail"],
                  "F1: the capless notice names no WAN-only effect")
            _need("not the encoder's" in js_capless["detail"],
                  "F1: the capless notice says whose mark it is")
            _need("478/512" in js_capped["detail"]
                  and "478/512" in js_capless["detail"],
                  "F1: both still carry the real numbers")
            _need("(qwen3vl_32b)" in js_capless["detail"]
                  and "(" not in js_noname["detail"].split("Nothing is cut")[1][:3],
                  "F1: the encoder is named when known and skipped when not")
    finally:
        os.unlink(tmp)

# --- F2: toast and report agree, per side -----------------------------------
if capped and capless and js_capped and js_capless:
    _need(("degrade" in capped) == ("degrade" in js_capped["detail"]),
          "F2: capped side -- report and toast make the same claim")
    _need(("Nothing is cut" in capless)
          and ("Nothing is cut" in js_capless["detail"]),
          "F2: capless side -- both say nothing is cut")
    _need(("kijai" in capped.lower()) and ("kijai" not in capless.lower()),
          "F2: the kijai reference lives only on the capped side")

# --- F3: the safe default is untouched --------------------------------------
_fn2 = [n for n in TREE.body if isinstance(n, ast.FunctionDef)
        and n.name == "_any_encoder_truncates"]
if not _fn2:
    _need(False, "F3: _any_encoder_truncates is gone")
else:
    ns2 = {"_encoder_facts": lambda c: (c or {}).get("facts", [])}
    exec(compile(ast.Module(body=_fn2, type_ignores=[]), "<v911b>", "exec"), ns2)
    _need(ns2["_any_encoder_truncates"](None) is True,
          "F3: no clip wired -> the CAPPED wording, as since v908 (an unknown "
          "encoder is not a safe encoder)")

print("\ntest_v911_near_limit_honesty.py: %d failure(s)" % len(FAILED))
sys.exit(1 if FAILED else 0)
