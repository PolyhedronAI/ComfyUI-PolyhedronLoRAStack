#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1327 -- the Sampler's negative is optional; not wired, it is the zeroed positive.

Frank, 13.11.: a graph of the pack carried a foreign node (Core's Conditioning Zero Out) only to feed the
Sampler's negative. The rule lives in the Sampler now.

WHAT IS PINNED

  Z1  _zeroed: the main tensor and the four named extras are zero, every other entry of the dict is kept (same
      object), the input is not touched, shapes / dtypes stay. Against Core's own ConditioningZeroOut (lifted from
      PLS_CORE_ROOT/nodes.py when that tree is there): the same result, key for key.
  Z2  INPUT_TYPES: `negative` is no longer required, it is the FIRST optional input, and the required block is the
      old one minus that single key, in the old order (widgets_values do not move: `negative` never was a widget).
  Z3  one seat: sample() hands an unwired negative on as None, _sample_impl turns None into _zeroed(positive)
      BEFORE the run, says so on the console, and a WIRED negative is passed on as the object it is.
  Z4  no core node class is imported for it, the file's new lines are ASCII, the tooltip names no model.
  Z5  the encoder side of the saving: with the negative output unread, Polyhedron CLIP Text Encode encodes once.

Script-style: exit 0 = pass.
"""

import ast
import os
import re
import sys

NAME = "v1327"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SMP = os.path.join(ROOT, "nodes", "uls_sampler.py")
CTE = os.path.join(ROOT, "nodes", "ph_clip_encode.py")

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL " + msg)


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


SRC = read(SMP)
TREE = ast.parse(SRC)


def top(name):
    for n in TREE.body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
            return n
        if isinstance(n, ast.Assign) and any(getattr(t, "id", None) == name for t in n.targets):
            return n
    return None


def method(cls, name):
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def seg(node):
    return ast.get_source_segment(SRC, node)


# ---------------------------------------------------------------- Z1 the rule
import torch  # noqa: E402

keys_node, fn_node = top("ZERO_KEYS"), top("_zeroed")
check(keys_node is not None and fn_node is not None, "Z1 ZERO_KEYS and _zeroed exist at module level")
ns = {"torch": torch}
if keys_node is not None and fn_node is not None:
    exec(seg(keys_node) + "\n\n" + seg(fn_node), ns)
    zeroed = ns["_zeroed"]
    check(tuple(ns["ZERO_KEYS"]) == ("pooled_output", "conditioning_lyrics", "conditioning_scale", "direct_context"),
          "Z1 the four keys of Core's zero-out")

    def sample_cond():
        marker = object()
        ref = [torch.ones(1, 4, 2, 2)]
        d = {"pooled_output": torch.ones(1, 3), "conditioning_lyrics": torch.ones(1, 5, 2),
             "conditioning_scale": torch.ones(1, 1, 1), "direct_context": torch.ones(1, 2, 2),
             "reference_latents": ref, "marker": marker, "none_key": None}
        return [[torch.ones(1, 4, 8, dtype=torch.float16), d],
                [torch.full((1, 2, 8), 3.0), {"pooled_output": None, "x": 1}]], marker, ref

    cond, marker, ref = sample_cond()
    out = zeroed(cond)
    check(len(out) == 2, "Z1 one entry out per entry in")
    check(all(float(e[0].abs().sum()) == 0.0 for e in out), "Z1 the main tensor is zero")
    check(out[0][0].shape == cond[0][0].shape and out[0][0].dtype == torch.float16, "Z1 shape and dtype stay")
    check(all(float(out[0][1][k].abs().sum()) == 0.0 for k in ns["ZERO_KEYS"]), "Z1 the four extras are zero")
    check(out[0][1]["marker"] is marker and out[0][1]["reference_latents"] is ref, "Z1 other entries kept as they are")
    check(out[1][1]["pooled_output"] is None and out[1][1]["x"] == 1, "Z1 a None extra stays None")
    check(float(cond[0][0].sum()) == 32.0 and float(cond[0][1]["pooled_output"].sum()) == 3.0
          and out[0][1] is not cond[0][1], "Z1 the input is not touched")
    check(zeroed([]) == [], "Z1 an empty conditioning stays empty")

    core_root = os.environ.get("PLS_CORE_ROOT", "")
    core_nodes = os.path.join(core_root, "nodes.py") if core_root else ""
    if core_nodes and os.path.isfile(core_nodes):
        csrc = read(core_nodes)
        ccls = [n for n in ast.parse(csrc).body if isinstance(n, ast.ClassDef) and n.name == "ConditioningZeroOut"]
        check(len(ccls) == 1, "Z1 Core's ConditioningZeroOut found in PLS_CORE_ROOT")
        if ccls:
            cns = {"torch": torch}
            exec(ast.get_source_segment(csrc, ccls[0]), cns)
            cond2, _m, _r = sample_cond()
            core_out = cns["ConditioningZeroOut"]().zero_out(cond2)[0]
            core_keys = [k for k in ns["ZERO_KEYS"] if ('"%s"' % k) in ast.get_source_segment(csrc, ccls[0])]
            check(len(core_keys) >= 2, "Z1 Core's class names the extras it zeroes")
            same = len(core_out) == len(out)
            for a, b in zip(core_out, out):
                same = same and torch.equal(a[0], b[0]) and a[0].dtype == b[0].dtype
                for k in core_keys:
                    va, vb = a[1].get(k), b[1].get(k)
                    same = same and ((va is None and vb is None) or (va is not None and vb is not None
                                                                     and torch.equal(va, vb)))
                same = same and set(a[1].keys()) == set(b[1].keys())
            check(same, "Z1 equal to Core's ConditioningZeroOut on every key Core handles")
    else:
        print("  note: PLS_CORE_ROOT not set -- the comparison with Core's class was skipped")

# ---------------------------------------------------------------- Z2 the inputs
cls = top("ULSSampler")
it = method(cls, "INPUT_TYPES") if cls is not None else None
ret = None
if it is not None:
    for n in ast.walk(it):
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict):
            ret = n.value
check(ret is not None, "Z2 INPUT_TYPES returns a dict literal")
if ret is not None:
    blocks = {}
    for k, v in zip(ret.keys, ret.values):
        if isinstance(k, ast.Constant) and isinstance(v, ast.Dict):
            blocks[k.value] = [kk.value for kk in v.keys if isinstance(kk, ast.Constant)]
    req, opt = blocks.get("required", []), blocks.get("optional", [])
    check("negative" not in req, "Z2 negative is not required")
    check(opt[:1] == ["negative"], "Z2 negative is the first optional input")
    check(req[:3] == ["model", "positive", "latent_image"], "Z2 the required sockets keep their order")
    check(opt[1:2] == ["model_low"] and "noise" in opt, "Z2 the other optional inputs follow unchanged")
    # RE-GROUNDED v1334 (DECLARED): the baseline name was hard-coded (v1324) and went
    # red when v1329 wrote a new baseline; the guard now takes the one baseline in the
    # tree, exactly as test_v577 resolves it -- the structure, not the file name.
    bases = sorted(f for f in os.listdir(ROOT) if re.fullmatch(r"WIDGET_ORDER_baseline_v\d+\.txt", f))
    check(len(bases) == 1, "Z2 exactly one widget baseline in the tree: %r" % bases)
    base = os.path.join(ROOT, bases[-1] if bases else "WIDGET_ORDER_baseline_missing.txt")
    line = [ln for ln in read(base).splitlines() if ln.startswith("ULSSampler\t")]
    check(len(line) == 1 and "negative" not in line[0].split("\t")[1].split(","),
          "Z2 negative never was a widget (the widget baseline stands)")
    m = re.search(r'"negative": \("CONDITIONING", \{"tooltip": (.*?)\}\),', SRC, re.S)
    tip = " ".join(re.findall(r'"([^"]*)"', m.group(1))) if m else ""
    check("Not wired" in tip and "zero" in tip, "Z2 the tooltip says what an unwired negative means")
    check(not re.search(r"minimax|h3|flux|qwen|wan\b|hunyuan|sdxl", tip, re.I), "Z4 the tooltip names no model")

# ---------------------------------------------------------------- Z3 one seat
smp_fn, impl = (method(cls, "sample"), method(cls, "_sample_impl")) if cls is not None else (None, None)
check(smp_fn is not None and impl is not None, "Z3 sample and _sample_impl exist")
if smp_fn is not None and impl is not None:
    s_src, i_src = seg(smp_fn), seg(impl)
    a = s_src.find('kwargs.setdefault("negative", None)')
    b = s_src.find("self._sample_impl(**kwargs)")
    check(0 <= a < b, "Z3 sample() defaults an unwired negative to None before the run")
    args = [x.arg for x in impl.args.args]
    check(args[:4] == ["self", "model", "positive", "negative"], "Z3 _sample_impl keeps its signature")
    g = i_src.find("if negative is None:")
    z = i_src.find("negative = _zeroed(positive)")
    r = i_src.find("self._sample_gated(")
    check(0 <= g < z < r, "Z3 None becomes the zeroed positive before _sample_gated")
    check("no negative wired" in i_src and "flush=True" in i_src, "Z3 the console names it")
    check(SRC.count("_zeroed(") == 2, "Z3 one seat: one definition, one call")

    # behaviour of the seat, lifted: only the head of _sample_impl up to the gate lookup
    head = i_src[:i_src.find("gate_id = _current_node_id()")]
    head = head[head.find("if negative is None:"):]
    body = "def seat(positive, negative):\n" + "\n".join("    " + ln[8:] if ln.startswith("        ") else "    " + ln
                                                         for ln in head.rstrip().splitlines()
                                                         if not ln.strip().startswith("#")) + "\n    return negative\n"
    sns = {"torch": torch, "_zeroed": ns.get("_zeroed")}
    try:
        import contextlib
        import io
        exec(body, sns)
        pos = [[torch.ones(1, 2, 4), {"pooled_output": torch.ones(1, 2)}]]
        wired = [[torch.full((1, 2, 4), 7.0), {}]]
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got_wired = sns["seat"](pos, wired)
            got_free = sns["seat"](pos, None)
        check(got_wired is wired, "Z3 a wired negative is passed on untouched")
        check(float(got_free[0][0].abs().sum()) == 0.0 and float(got_free[0][1]["pooled_output"].abs().sum()) == 0.0,
              "Z3 an unwired negative is the zeroed positive")
        check(buf.getvalue().count("no negative wired") == 1, "Z3 the console line is printed for the unwired case only")
    except Exception as e:  # noqa: BLE001
        check(False, "Z3 the seat runs lifted (%s: %s)" % (type(e).__name__, e))

# ---------------------------------------------------------------- Z4 statics
new_lines = [ln for ln in SRC.splitlines() if "v1327" in ln or "_zeroed" in ln or "ZERO_KEYS" in ln]
check(len(new_lines) >= 6 and all(all(ord(c) < 128 for c in ln) for ln in new_lines), "Z4 the new lines are ASCII")
check("ConditioningZeroOut(" not in SRC and "nodes_cond" not in SRC, "Z4 no core node class is called for it")
with open(os.path.abspath(__file__), "rb") as f:
    check(all(c < 128 for c in f.read()), "Z4 this guard is ASCII")

# ---------------------------------------------------------------- Z5 the encoder saving
csrc = read(CTE)
ctree = ast.parse(csrc)
wanted = [n for n in ctree.body if (isinstance(n, ast.FunctionDef) and n.name == "slot_consumed")
          or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("POS_SLOT", "NEG_SLOT") for t in n.targets))]
cns = {}
exec("\n\n".join(ast.get_source_segment(csrc, n) for n in wanted), cns)
graph = {"1": {"class_type": "ULSCLIPTextEncode", "inputs": {}},
         "2": {"class_type": "ULSSampler", "inputs": {"positive": ["1", 0], "model": ["9", 0]}}}
check(cns["slot_consumed"](graph, "1", cns["POS_SLOT"]) is True, "Z5 the positive output is read")
check(cns["slot_consumed"](graph, "1", cns["NEG_SLOT"]) is False, "Z5 nobody reads the negative output")
check("neg_cond = _encode(clip, neg_text) if want_neg else None" in csrc, "Z5 an unread negative is not encoded")

if FAILS:
    print("%s FAIL (%d)" % (NAME, len(FAILS)))
    sys.exit(1)
print("%s PASS" % NAME)
