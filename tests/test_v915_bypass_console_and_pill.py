#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v915 -- two ergonomics cuts from the 04.09. review, and one measurement.

  1  console: the merge path printed "BYPASS: structural fallbacks below apply
     the group BAKED (SEQ)" for EVERY bypass group, clean ones included. Gone;
     instead each of the ten "falling back to SEQ" messages inside
     _apply_concat_or_dare carries " (BAKED, not bypass)" itself under bypass
     and nothing under patch -- one local `_fb`, defined before the first use.
  2  Stack header: an Apply pill right of the flat pill shows Auto / Bypass /
     Baked without opening a group popup and cycles on click, like the Engine
     pill (v913). Config plumbing is unchanged (counted in test_v913).
  3  the auto-bypass default is now backed by a measurement in the policy
     docstring (delta fidelity: bypass 0.4 %, baked bf16 6.7 %, baked int8
     25 %) -- pinned here only as "the docstring names the numbers", so a
     future edit that drops the rationale is visible.

WHAT IS PINNED, and how:

  A  AST of _apply_concat_or_dare: no upfront BYPASS print; exactly one `_fb`
     assignment; every f-string containing "falling back to SEQ" also contains
     "{_fb}"; the assignment precedes the first use.
  B  the `_fb` value, driven: bypass -> " (BAKED, not bypass)", patch -> "".
  C  JS: the Stack pill draws, hovers, clicks and tooltips (counted zones), and
     the helpers it uses are the v913 ones (driven in test_v913).
  D  the policy docstring carries the three delta-fidelity numbers.

No torch, no comfy.
"""
import ast
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uls_merge_policy as MP  # noqa: E402

FAILED = []


def _fail(msg):
    FAILED.append(msg)
    print("FAIL: {}".format(msg))


def _ok(msg):
    print("ok  : {}".format(msg))


def _need(cond, msg):
    _ok(msg) if cond else _fail(msg)


SRC = (ROOT / "nodes" / "uls_stack_node.py").read_text(encoding="utf-8")
tree = ast.parse(SRC)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_apply_concat_or_dare")

# --- A: the console shape, on the AST ------------------------------------------
fb_assign = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Assign)
             and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "_fb"]
_need(len(fb_assign) == 1, "A  exactly one `_fb` assignment in _apply_concat_or_dare")


def _fstring_texts(node):
    """(lineno, literal text, has_fb) for every JoinedStr under node."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.JoinedStr):
            lit = "".join(v.value for v in n.values if isinstance(v, ast.Constant))
            fb = any(isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name) and v.value.id == "_fb"
                     for v in n.values)
            out.append((n.lineno, lit, fb))
    return out


fallbacks = [(ln, fb) for ln, lit, fb in _fstring_texts(fn) if "falling back to SEQ" in lit]
# v986: nine -- the mixed-naming fallback is gone, such a group now merges per
# layer (test_v986). v987: ten -- the target-clash guard (two spellings of one
# weight) is a new fallback. Every one carries {_fb}.
_need(len(fallbacks) == 10 and all(fb for _, fb in fallbacks),
      "A  all %d 'falling back to SEQ' f-strings carry {_fb}" % len(fallbacks))
_need(fb_assign and all(ln > fb_assign[0] for ln, _ in fallbacks), "A  `_fb` is assigned before its first use")
upfront = [lit for _, lit, _ in _fstring_texts(fn) if "structural fallbacks below" in lit]
plain = [n for n in ast.walk(fn) if isinstance(n, ast.Constant) and isinstance(n.value, str)
         and "structural fallbacks below" in n.value]
_need(not upfront and not plain, "A  the upfront BYPASS line is gone")

# --- B: the value, driven ----------------------------------------------------------
assign_node = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and n.lineno == fb_assign[0])
expr = ast.unparse(assign_node.value)
_need(eval(expr, {"handoff": "bypass"}) == " (BAKED, not bypass)" and eval(expr, {"handoff": "patch"}) == "",
      "B  `_fb` is the suffix under bypass and empty under patch")

# --- C: the Stack pill (JS, counted) ------------------------------------------------
JS = (ROOT / "web" / "js" / "uls_node.js").read_text(encoding="utf-8")
_need(JS.count("uls._applyPillRect = {") == 1, "C  the Stack pill is drawn once")
_need(JS.count('uls.hoverZone === "applyPill"') >= 2, "C  the Stack pill has hover + tooltip")
_need(JS.count('uls.hoverZone = "applyPill"') == 1, "C  the Stack pill sets its hover zone")
click = JS.count("const apR = uls._applyPillRect;")
_need(click == 2, "C  the Stack pill is hit-tested in hover and in click (got %d)" % click)
_need(JS.count("uls.apply = applyNext(uls.apply);") == 2, "C  Stack pill and Engine pill both cycle the value")

# --- D: the rationale stays in the policy ----------------------------------------
doc = MP._apply_decision.__doc__ or ""
_need(all(t in doc for t in ("0.4 %", "6.7 %", "25 %")), "D  the policy docstring names the three delta-fidelity numbers")

if FAILED:
    print("\n{} check(s) FAILED".format(len(FAILED)))
    sys.exit(1)
print("\nall checks passed")
