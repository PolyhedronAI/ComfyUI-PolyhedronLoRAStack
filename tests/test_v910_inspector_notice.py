#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v910 -- a missing feed on the LoRA Inspector is ADVICE, not an error.

THE WOUND (Frank's screen, 30.08.): the Inspector was framed in ComfyUI's red
NODE_ERROR_COLOUR and the queue failed with the core's red `failedToQueue`
toast, because a source node was muted. Measured cause: BOTH inputs stood in
`required`, so `execution.py` answered `required_input_missing` before the node
ever ran. Red is the core's ERROR channel; a passive read-only node whose whole
job is to report what it can see must not spend it on "your Stack is muted".

THE CUT: both inputs optional in BOTH registrations, inspect() tolerates the
absence and NAMES it (in the report and as a structured ui state), and a warn
("orange") toast carries it the way uls_token_toast.js already does for the
Token Counter.

WHAT IS PINNED, and where it can break:

  N1  legacy INPUT_TYPES: `required` is EMPTY and both sockets sit in
      `optional` -- read off the literal dict via ast (law v577), not as text.
  N2  the V3 schema passes optional=True for both -- read off the AST of the
      Input(...) calls, so a reordered keyword cannot fool it.
  N3  the V3 execute() goes through NodeOutput.from_dict. With the old
      `NodeOutput(*out)` the dict would unpack to its KEYS and the notice
      would never reach the frontend in a Nodes-2.0 install.
  B1  inspect() is DRIVEN over the three feed states, with real None inputs.
      It must never raise, must always answer the {"ui", "result"} shape, and
      must report the right state.
  B2  a missing prompt is SAID in the report -- otherwise every trigger reads
      as a real miss (the v552/v885 rule against silent degradation).
  F1  runNotice() lifted out of the JS and RUN in node over every state: the
      severity is "warn" for both notices and the healthy state is silent.
      The string "error" may not appear as a severity anywhere in the file.
  F2  the mirror rule: this file's toast() helper is byte-identical to the one
      in uls_token_toast.js (same API, same fallback chain).
  F3  no prose parsing: the hook reads message.pls_inspector and nothing else.
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
V3 = os.path.join(ROOT, "nodes", "uls_inspector_v3.py")
JS = os.path.join(ROOT, "web", "js", "ph_inspector_toast.js")
JS_TWIN = os.path.join(ROOT, "web", "js", "uls_token_toast.js")

FAILED = []


def _fail(msg):
    FAILED.append(msg)
    print("FAIL: " + msg)


def _need(cond, msg):
    print("ok  : " + msg) if cond else _fail(msg)


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
SRC_V3 = _read(V3)
SRC_JS = _read(JS)
SRC_TWIN = _read(JS_TWIN)


# --- the class, lifted alone (the module needs the whole comfy stack) -------
def _lift_class(src, name):
    for node in ast.parse(src).body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return ast.get_source_segment(src, node)
    return None


SEG = _lift_class(SRC, "ULSInspector")
_need(SEG is not None, "ULSInspector is present in uls_stack_node.py")

# --- N1: the literal dict ---------------------------------------------------
_types = None
if SEG:
    for node in ast.walk(ast.parse(SEG)):
        if isinstance(node, ast.FunctionDef) and node.name == "INPUT_TYPES":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Return) and isinstance(sub.value, ast.Dict):
                    _types = ast.literal_eval(sub.value)
if _types is None:
    _fail("N1: INPUT_TYPES does not return a LITERAL dict (law v577)")
else:
    _need(_types.get("required") == {},
          "N1: `required` is empty -- got %r" % (list(_types.get("required", {})),))
    _need(sorted(_types.get("optional", {})) == ["prompt", "uls_config_out"],
          "N1: both sockets are optional -- got %r"
          % (sorted(_types.get("optional", {})),))
    _need(all(spec[1].get("forceInput") is True
              for spec in _types.get("optional", {}).values()),
          "N1: both stay forceInput sockets (the widget form would be a "
          "different node)")

# --- N2/N3: the V3 twin -----------------------------------------------------
_v3_inputs = {}
_from_dict = False
for node in ast.walk(ast.parse(SRC_V3)):
    if isinstance(node, ast.Call):
        fname = ast.unparse(node.func)
        if fname.endswith("String.Input") and node.args:
            try:
                name = ast.literal_eval(node.args[0])
            except Exception:
                continue
            kw = dict((k.arg, k.value) for k in node.keywords)
            opt = kw.get("optional")
            _v3_inputs[name] = isinstance(opt, ast.Constant) and opt.value is True
        if fname.endswith("NodeOutput.from_dict"):
            _from_dict = True

_need(_v3_inputs.get("uls_config_out") is True
      and _v3_inputs.get("prompt") is True,
      "N2: the V3 schema marks both inputs optional -- got %r" % (_v3_inputs,))
_need(_from_dict,
      "N3: V3 execute() must hand the legacy dict to NodeOutput.from_dict -- "
      "NodeOutput(*dict) would unpack the KEYS and lose the notice")

# --- B1/B2: drive the real code --------------------------------------------
ns = {"json": json, "re": re}
# v987: since v986 the Inspector shortens LoRA names with the shared
# name_shortener (the Stack module imports it at top as _ov_names). The lifted
# class needs the REAL function beside it, not a stub.
sys.path.insert(0, os.path.join(ROOT, "nodes"))
import uls_overlap_math as _OVM  # noqa: E402
ns["_ov_names"] = _OVM.name_shortener
try:
    exec(compile(SEG, "<inspector>", "exec"), ns)
    node_obj = ns["ULSInspector"]()
except Exception as exc:                              # pragma: no cover
    node_obj = None
    _fail("B1: the class does not lift cleanly: %r" % (exc,))

CFG = json.dumps({"lora_info": [
    {"name": "a", "weight": 1.0, "trigger_words": "trg", "trigger_src": "meta"},
]})


def _state(out):
    return out["ui"]["pls_inspector"][0]["state"]


if node_obj is not None:
    cases = (
        ("nothing wired at all", {}, "no_config"),
        ("config only", {"uls_config_out": CFG}, "no_prompt"),
        ("both wired", {"uls_config_out": CFG, "prompt": "a trg here"}, "ok"),
        ("empty strings, not None", {"uls_config_out": "", "prompt": ""},
         "no_config"),
    )
    for label, kwargs, want in cases:
        try:
            out = node_obj.inspect(**kwargs)
        except Exception as exc:
            _fail("B1: %s raised %r -- a missing feed must never end a run"
                  % (label, exc))
            continue
        shape_ok = (isinstance(out, dict) and "ui" in out and "result" in out
                    and isinstance(out["result"], tuple)
                    and isinstance(out["result"][0], str))
        _need(shape_ok, "B1: %s answers the {ui, result} shape" % label)
        if shape_ok:
            _need(_state(out) == want,
                  "B1: %s -> state %r (want %r)" % (label, _state(out), want))

    out = node_obj.inspect(uls_config_out=CFG)
    _need("prompt input" in out["result"][0],
          "B2: a missing prompt is SAID in the report, not implied by a table "
          "full of misses")
    out_ok = node_obj.inspect(uls_config_out=CFG, prompt="a trg here")
    _need("prompt input" not in out_ok["result"][0],
          "B2: a healthy run does not carry the note")

# --- F1: run the notice logic in node ---------------------------------------
_m = re.search(r"export function runNotice\(info\) \{.*?\n\}\n", SRC_JS, re.S)
if not _m:
    _fail("F1: runNotice() is not liftable from the frontend file")
else:
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(_m.group(0).replace("export ", ""))
        fh.write("const out = [null, {state:'no_config',loras:0,missing:0},"
                 "{state:'no_prompt',loras:3,missing:3},"
                 "{state:'ok',loras:3,missing:0}].map(runNotice);\n")
        fh.write("process.stdout.write(JSON.stringify(out));\n")
        tmp = fh.name
    try:
        run = subprocess.run([_node(), tmp], capture_output=True, text=True,
                             timeout=60)
        if run.returncode != 0:
            _fail("F1: runNotice() does not execute: %s"
                  % run.stderr.strip()[:120])
        else:
            got = json.loads(run.stdout)
            _need(got[0] is None and got[3] is None,
                  "F1: no payload and a healthy run stay SILENT")
            sev = [n["severity"] for n in got if n]
            _need(sev == ["warn", "warn"],
                  "F1: every notice is warn (orange) -- got %r" % (sev,))
            _prompt_notice = [n for n in got
                              if n and n["sig"].startswith("no_prompt")]
            _need(len(_prompt_notice) == 1
                  and "3" in _prompt_notice[0]["detail"],
                  "F1: the no-prompt notice names the LoRA count it is about "
                  "(driven with 3) -- got %r" % (_prompt_notice,))
    finally:
        os.unlink(tmp)

# The words "error" and "severity" both appear in this file's own prose, so
# the check is aimed at CODE: a severity assignment and a first argument to
# toast(). Written the other way round it went red on its own explanation --
# the v908 lesson, walked into once more while writing this guard.
_sev_error = re.findall(r"severity\s*:\s*[\"']error", SRC_JS)
_toast_error = re.findall(r"toast\(\s*[\"']error", SRC_JS)
_need(not _sev_error and not _toast_error,
      "F1: no error severity is ever assigned or passed -- red is the core's "
      "channel, not ours (found %r / %r)" % (_sev_error, _toast_error))

# --- F2: the mirror rule ----------------------------------------------------
def _toast_helper(src):
    m = re.search(r"function toast\(severity, summary, detail, life\) \{"
                  r".*?\n\}\n", src, re.S)
    if not m:
        return None
    body = m.group(0)
    # the last-resort console line names its own node; compare the mechanism
    return re.sub(r"\[PLS[^\]]*\]", "[PLS]", body)


_a, _b = _toast_helper(SRC_JS), _toast_helper(SRC_TWIN)
_need(_a is not None and _b is not None and _a == _b,
      "F2: toast() is a byte-identical mirror of the uls_token_toast.js helper")

# --- F3: no prose parsing ---------------------------------------------------
_need("pls_inspector" in SRC_JS,
      "F3: the hook reads the structured ui channel")
_need("inspector_report" not in SRC_JS and "Polyhedron LoRA Inspector\\n" not in SRC_JS,
      "F3: the hook never reads the report TEXT (the uls_token_toast rule)")

print("\ntest_v910_inspector_notice.py: %d failure(s)" % len(FAILED))
sys.exit(1 if FAILED else 0)
