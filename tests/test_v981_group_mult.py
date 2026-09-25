#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v981 -- per-group strength, and the Nodes 2.0 group popup that wrote nothing.

THE FINDINGS (21.09.):
  1  Nodes 2.0 (uls_stack_dom.js) opened the shared group popup with
     `() => commit(node, root)` for BOTH callbacks. The popup reports every
     choice through them, so mode, DARE variant, Trim, Resolve, Trim strength
     and Apply could not be set in that renderer at all. It also passed the
     node-wide legacy `dare_variant` instead of the group's variant.
  2  The global multiplier lost its slider long ago (FOOTER_H = 0) and the
     backend never applied `mult`. A saved value != 1 ran silently at x1.0.

WHAT IS PINNED, and how:

  A  _group_mult / _group_scaled (uls_stack_node), lifted closed and driven:
     absent / unreadable -> 1.0, clamp 0..2, the explicit CLIP weight scales
     with its group, rounding as the weights are rounded.
  B  UltimateLoraStack.apply, lifted and DRIVEN with a recording
     apply_lora_set: a group with strength 0.5 reaches the merge at half
     weight (model AND CLIP), the other group untouched; no group_mult ->
     weights bit-identical; lora_info carries the applied weights; the
     console names 'group x0.5'; a saved mult != 1 is announced as NOT
     applied, mult == 1 says nothing.
  C  groupPopupHandlers (uls_node.js), lifted and run in node: every popup
     choice lands in node._uls (mode, DARE variant, back to SEQ, trim, resolve,
     trim_amount, group_mult incl. 1.0 -> key removed and clamp, apply), and
     after() fires each time.
  D  both renderers use it: the painted view and the Nodes 2.0 pill pass
     h.onChange / h.onToggle; the DOM pill passes the GROUP's DARE variant
     and the group strength; the popup sends onToggle("group_mult", ...);
     group_mult rides in BOTH config builders and is read back on configure.
  E  the Merge Analyzer folds the strength in the same way (AST: it calls
     _group_scaled in the overview loop and in the overlap block).

No torch, no comfy. Needs node for C.
"""
import ast
import json
import math
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lift as _LIFT  # noqa: E402

FAILED = []


def _need(cond, msg):
    if cond:
        print("ok  : " + msg)
    else:
        FAILED.append(msg)
        print("FAIL: " + msg)


SRC = (ROOT / "nodes" / "uls_stack_node.py").read_text(encoding="utf-8")

# --- A: the two pure helpers --------------------------------------------------
lift, missing = _LIFT.close_over(SRC, ["_group_mult", "_group_scaled"], provided={"math"})
_need(not missing, "A  helpers lift closed (%s)" % (sorted(missing) or "none missing"))
ns = {"math": math}
exec(compile(lift, "<v981 helpers>", "exec"), ns)
gmf, gsc = ns["_group_mult"], ns["_group_scaled"]
_need(gmf({}, "subject") == 1.0 and gmf(None, "subject") == 1.0 and gmf({"scene": 0.5}, "subject") == 1.0,
      "A  absent group / no map -> 1.0")
_need(gmf({"subject": "x"}, "subject") == 1.0 and gmf({"subject": None}, "subject") == 1.0,
      "A  unreadable value -> 1.0")
_need(gmf({"subject": 0.8}, "subject") == 0.8 and gmf({"subject": "0.8"}, "subject") == 0.8,
      "A  a number (or numeric string) is taken as is")
_need(gmf({"subject": -1}, "subject") == 0.0 and gmf({"subject": 7}, "subject") == 2.0,
      "A  clamped to 0..2 (no sign flip, no blow-up)")
rows = [{"name": "a", "wClip": 0.4}, {"name": "b"}]
ws, cs = gsc(rows, [0.3, 0.7], 0.5)
_need(ws == [0.15, 0.35] and cs == [0.2, 0.35],
      "A  model weights x0.5, explicit CLIP 0.4 -> 0.2, implicit CLIP follows the model (%s %s)" % (ws, cs))
ws1, cs1 = gsc(rows, [0.3, 0.7], 1.0)
_need(ws1 == [0.3, 0.7] and cs1 == [0.4, 0.7], "A  factor 1.0 changes nothing")

# --- B: the Stack's apply(), driven ------------------------------------------------
tree = ast.parse(SRC)
apply_src = None
for n in tree.body:
    if isinstance(n, ast.ClassDef) and n.name == "UltimateLoraStack":
        for f in n.body:
            if isinstance(f, ast.FunctionDef) and f.name == "apply":
                apply_src = textwrap.dedent(ast.get_source_segment(SRC, f))
_need(apply_src is not None, "B  UltimateLoraStack.apply found")
helpers, missing = _LIFT.close_over(
    SRC, ["_group_mult", "_group_scaled", "_safe_weight", "_short_name", "_sort_active_rows",
          "_group_effective", "_cap_text"],
    provided={"math", "os", "folder_paths", "OrderedDict", "_cached_load_torch_file", "_convert_lora_like_core", "_detect_convention", "_resolve_pick_device", "_check_interrupt", "INTERRUPT_EXC", "_ov_measure", "_ov_cap"})
_need(not missing, "B  apply's helpers lift closed (%s)" % (sorted(missing) or "none missing"))
CALLS = []


def _rec(loader, model, clip, names, weights, **kw):
    CALLS.append((list(names), list(weights), list(kw.get("clip_weights") or [])))
    return model, clip, []


bns = {"math": math, "os": os, "json": json,
       "apply_lora_set": _rec, "_get_trigger": lambda name: ("", "none"),
       # v983: the cap's imports -- the cap is off in these fixtures, so the
       # measurement is never reached; test_v983 drives it on real files
       "OrderedDict": __import__("collections").OrderedDict,
       # v988: the console's name shortener (the real one)
       "_attach_schedule_loras": lambda m, nw: m,   # v989: identity in this harness
       "_ov_names": (sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nodes")) or __import__("uls_overlap_math").name_shortener)}
exec(compile(helpers, "<v981 apply helpers>", "exec"), bns)
exec(compile(apply_src, "<v981 apply>", "exec"), bns)
APPLY = bns["apply"]


class _Self:
    _loader = None


def run(cfg):
    CALLS.clear()
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = APPLY(_Self(), object(), object(), json.dumps(cfg))
    return out, buf.getvalue()


BASE = {"rows": [
    {"enabled": True, "name": "s1.safetensors", "group": "subject", "weight": 0.3},
    {"enabled": True, "name": "s2.safetensors", "group": "subject", "weight": 0.2, "wClip": 0.6},
    {"enabled": True, "name": "c1.safetensors", "group": "scene", "weight": 0.4}]}
(o0, dbg0) = run(BASE)
ref = [c for c in CALLS]
cfg = dict(BASE, group_mult={"subject": 0.5})
(o1, dbg1) = run(cfg)
by = {tuple(n): (w, c) for n, w, c in CALLS}
by0 = {tuple(n): (w, c) for n, w, c in ref}
subj = ("s1.safetensors", "s2.safetensors")
_need(by.get(subj) == ([0.15, 0.1], [0.15, 0.3]),
      "B  subject x0.5 reaches the merge at half weight, model AND CLIP (%s)" % (by.get(subj),))
_need(by.get(("c1.safetensors",)) == by0.get(("c1.safetensors",)),
      "B  the other group is untouched")
_need(by0.get(subj) == ([0.3, 0.2], [0.3, 0.6]), "B  without group_mult: weights exactly as set")
info = json.loads(o1[3])["lora_info"]
_need([i["weight"] for i in info if i["group"] == "subject"] == [0.15, 0.1],
      "B  lora_info (Inspector) carries the applied weights")
_need("group \u00d70.5" in dbg1 and "group \u00d7" not in dbg0, "B  the console names 'group x0.5', only when set")
(_o, dbg2) = run(dict(BASE, mult=0.8))
_need("NOT applied" in dbg2 and "NOT applied" not in dbg0,
      "B  a saved global multiplier != 1 is announced as NOT applied")
(_o, dbg3) = run(dict(BASE, mult=1.0))
_need("NOT applied" not in dbg3, "B  mult == 1 says nothing")

# --- C: groupPopupHandlers, run in node ----------------------------------------------
JS = (ROOT / "web" / "js" / "uls_node.js").read_text(encoding="utf-8")
i0 = JS.index("export const GROUP_MULT_MIN")
i1 = JS.index("\nfunction showGroupModePopup(")
chunk = JS[i0:i1]
node_bin = shutil.which("node")
if not node_bin:
    _need(False, "C  node is installed")
else:
    harness = (
        "const posts = [];\n"
        "const api = { fetchApi: (u, o) => { posts.push([u, JSON.parse(o.body)]); return Promise.resolve(); } };\n"
        "const applyNorm = (v) => (v === 'bypass' || v === 'patch') ? v : 'auto';\n"
        + chunk.replace("export const", "const").replace("export function", "function") +
        "\nconst node = { _uls: {} };\nlet n = 0;\n"
        "const h = groupPopupHandlers(node, 'subject', () => { n++; });\n"
        "const out = {};\n"
        "h.onChange('DARE:element'); out.dare = [node._uls.groupModes.subject, node._uls.groupDare.subject];\n"
        "h.onChange('CONCAT'); out.concat = [node._uls.groupModes.subject, node._uls.groupDare.subject === undefined];\n"
        "h.onChange('SEQ'); out.seq = node._uls.groupModes.subject === undefined;\n"
        "h.onToggle('trim', true); h.onToggle('resolve', true); out.tr = [node._uls.groupTrim.subject, node._uls.groupResolve.subject];\n"
        "h.onToggle('trim', false); out.troff = node._uls.groupTrim.subject === undefined;\n"
        "h.onToggle('trim_amount', 0.8); out.ta = node._uls.groupTrimAmount.subject;\n"
        "h.onToggle('trim_amount', null); out.taauto = node._uls.groupTrimAmount.subject === undefined;\n"
        "h.onToggle('group_mult', 0.55); out.gm = node._uls.groupMult.subject;\n"
        "h.onToggle('group_mult', 9); out.gmclamp = node._uls.groupMult.subject;\n"
        "h.onToggle('group_mult', 1.0); out.gmone = node._uls.groupMult.subject === undefined;\n"
        "h.onToggle('apply', 'bypass'); out.apply = node._uls.apply;\n"
        "out.after = n; out.posts = posts.map(p => p[1].mode);\n"
        "console.log(JSON.stringify(out));\n")
    d = tempfile.mkdtemp()
    p = os.path.join(d, "h.mjs")
    open(p, "w", encoding="utf-8").write(harness)
    r = subprocess.run([node_bin, p], capture_output=True, text=True, timeout=60)
    shutil.rmtree(d, ignore_errors=True)
    try:
        o = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        o = {}
        print(r.stdout, r.stderr)
    _need(o.get("dare") == ["DARE", "element"], "C  DARE:element -> mode DARE + the group's variant")
    _need(o.get("concat") == ["CONCAT", True], "C  CONCAT clears the DARE variant")
    _need(o.get("seq") is True, "C  SEQ removes the key (default)")
    _need(o.get("tr") == [True, True] and o.get("troff") is True, "C  trim / resolve on and off")
    _need(o.get("ta") == 0.8 and o.get("taauto") is True, "C  trim strength, null = Auto removes the key")
    _need(o.get("gm") == 0.55 and o.get("gmclamp") == 2 and o.get("gmone") is True,
          "C  group strength stored, clamped to 2, 1.0 removes the key")
    _need(o.get("apply") == "bypass", "C  apply is node-wide")
    _need(o.get("after") == 12, "C  after() fires on every choice (%s)" % o.get("after"))
    _need(o.get("posts") == ["DARE", "CONCAT", "SEQ"], "C  the mode still reaches /uls/group_modes")

# --- D: both renderers use it --------------------------------------------------------
DOM = (ROOT / "web" / "js" / "uls_stack_dom.js").read_text(encoding="utf-8")
j0 = DOM.index("grp.onclick = (e) => {")
j1 = DOM.index("el.appendChild(grp);", j0)
click = DOM[j0:j1]
_need("groupPopupHandlers(node, row.group" in click and "h.onChange, h.onToggle" in click,
      "D  Nodes 2.0 pill writes through the shared handlers")
_need("() => commit(node, root), () => commit(node, root)" not in DOM,
      "D  the choice-discarding callbacks are gone")
_need("uls.groupDare?.[row.group]" in click and "uls.dare_variant" not in click,
      "D  Nodes 2.0 passes the GROUP's DARE variant")
_need("uls.groupMult?.[row.group]" in click, "D  Nodes 2.0 passes the group strength")
_need(re.search(r"import \{[^}]*groupPopupHandlers[^}]*\} from \"\./uls_node\.js\"", DOM) is not None,
      "D  uls_stack_dom.js imports groupPopupHandlers from uls_node.js")
k0 = JS.index("const h = groupPopupHandlers(node, row.group, () => {")
_need("h.onChange, h.onToggle" in JS[k0:k0 + 600], "D  the painted view writes through the same handlers")
_need('onToggle?.("group_mult", gMult)' in JS, "D  the popup reports the strength via onToggle('group_mult')")
_need(JS.count("group_mult: this._uls.groupMult || {},") == 2,
      "D  group_mult rides in BOTH config builders (sync + serialize)")
_need("this._uls.groupMult = (d.group_mult && typeof d.group_mult" in JS,
      "D  group_mult is read back on configure")

# --- E: the Analyzer folds it in the same way -----------------------------------------
RI = (ROOT / "nodes" / "uls_resolve_inspector.py").read_text(encoding="utf-8")
# RE-GROUNDED v983: the Analyzer now calls _group_effective (strength AND
# energy cap), the one function the Stack and the bake call too. The promise
# is unchanged -- the Analyzer reports the weights the Stack runs with.
calls = [n for n in ast.walk(ast.parse(RI))
         if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "_group_effective"]
_need(len(calls) >= 2, "E  the Analyzer calls _group_effective in overview and overlap (%d)" % len(calls))

print()
print("test_v981_group_mult: %d failure(s)" % len(FAILED))
sys.exit(1 if FAILED else 0)
