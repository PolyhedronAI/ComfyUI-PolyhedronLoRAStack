"""Guard v1077 -- weight_dtype on Load CLIP (per-node fp8 cast of the encoder).

WHY (Frank, 10.10.): the Music3 composer in bf16 is 15.6 GB, the card has
14.8 GiB usable. Core loads it partially and streams the remainder from RAM on
every one of the ~2000 autoregressive forwards -- 2 min became 25 min. Core's
only lever is the process-wide --fp8_e4m3fn-text-enc flag; the Suite gets the
lever ON THE NODE that pays for it.

WHAT THIS PINS (driven, not read, wherever a mechanism exists):

  C1  the widget is APPENDED: last key of INPUT_TYPES()["optional"], choices
      are ULSLoadCLIP._DTYPES, default "default". Never in "required" -- an
      insert there shifts every saved workflow's slots by one (#577 law).
  C2  dtype_options is pure and exact: default -> {}, fp8_e4m3fn ->
      torch.float8_e4m3fn, fp8_e5m2 -> torch.float8_e5m2, anything else
      raises (never a silent fall-through to a full-size encoder).
  C3  DRIVEN through ULSLoadCLIP.load against a recording comfy.sd.load_clip:
      fp8 reaches model_options["dtype"]; device=cpu and the dtype coexist
      (the dtype can never shadow the device keys); default sends no dtype
      key at all (core then applies its own text_encoder_dtype / launch flag);
      the info readout names the dtype word.
  C4  the frontend half: CLIP_CANON ends with weight_dtype, CLIP_DISPLAY shows
      it with type/device at the top and ahead of the slot block; and a graph
      saved BEFORE this cut (six values) loads with every old slot in place
      and weight_dtype untouched -- driven through the lifted v718 functions.
  C5  one vocabulary: every Load CLIP dtype word is a Load Model dtype word.
  C6  (found driving C3) a graph WITHOUT the slot keys 2-4 counts ONE
      encoder: _strip_size(None) used to yield the string "None", which
      _is_filled took for a file -- three phantoms, forced hidream recipe.

Script-style: exit 0 = pass.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types

NAME = "v1077"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MOD = os.path.join(ROOT, "nodes", "ph_basics.py")
JS = os.path.join(ROOT, "web", "js", "ph_basics.js")


def _fail(msg):
    print("[%s] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(cond, msg):
    if not cond:
        _fail(msg)


# ---- stand-ins: folder_paths + a recording comfy.sd ----------------------
tmp = tempfile.mkdtemp()
os.makedirs(os.path.join(tmp, "text_encoders"), exist_ok=True)
ENC = "minimax_music3_text_encoder_pruned_bf16.safetensors"
with open(os.path.join(tmp, "text_encoders", ENC), "wb") as fh:
    fh.truncate(4096)

fp = types.ModuleType("folder_paths")
fp.get_filename_list = lambda f: [ENC] if f == "text_encoders" else []
fp.get_full_path = lambda f, n: os.path.join(tmp, f, str(n))
fp.get_full_path_or_raise = fp.get_full_path
fp.get_folder_paths = lambda f: [os.path.join(tmp, f)]
sys.modules["folder_paths"] = fp

import torch  # noqa: E402  (the pack imports it at module level too)

CALLS = []


class _CLIPType:
    MINIMAX = "minimax"
    STABLE_DIFFUSION = "stable_diffusion"


def _load_clip(ckpt_paths, embedding_directory=None, clip_type=None,
               model_options=None, **kw):
    CALLS.append({"paths": list(ckpt_paths), "clip_type": clip_type,
                  "model_options": dict(model_options or {})})
    return "CLIP"


comfy = types.ModuleType("comfy")
comfy_sd = types.ModuleType("comfy.sd")
comfy_sd.CLIPType = _CLIPType
comfy_sd.load_clip = _load_clip
comfy_utils = types.ModuleType("comfy.utils")
comfy.sd = comfy_sd
comfy.utils = comfy_utils
sys.modules["comfy"] = comfy
sys.modules["comfy.sd"] = comfy_sd
sys.modules["comfy.utils"] = comfy_utils

pkg = types.ModuleType("plspack")
pkg.__path__ = [os.path.join(ROOT, "nodes")]
sys.modules["plspack"] = pkg
spec = importlib.util.spec_from_file_location("plspack.ph_basics", MOD)
mod = importlib.util.module_from_spec(spec)
sys.modules["plspack.ph_basics"] = mod
spec.loader.exec_module(mod)

# the header reader needs a real safetensors file; the recipe check is not
# under test here -- both are seams, patched at the module the node uses.
mod.ph_te_detect.identify_file = lambda path: {"name": "stub", "model": "stub"}
mod.ph_te_detect.check_recipe = lambda clip_type, idents: (True, "")
mod.ph_te_detect.describe = lambda ident, filename="": filename
mod._clip_type_choices = lambda: ["minimax", "stable_diffusion"]

Node = mod.ULSLoadCLIP

# ---- C1: appended, optional, last ----------------------------------------
it = Node.INPUT_TYPES()
_need("weight_dtype" not in it["required"],
      "C1: weight_dtype sits in REQUIRED -- that inserts a widget before the "
      "slots and shifts every saved workflow (#577)")
opt_keys = list(it["optional"].keys())
_need(opt_keys[-1] == "weight_dtype",
      "C1: weight_dtype must be the LAST optional key, got %r" % (opt_keys,))
choices, meta = it["optional"]["weight_dtype"]
_need(list(choices) == list(Node._DTYPES),
      "C1: choices %r != ULSLoadCLIP._DTYPES %r" % (choices, Node._DTYPES))
_need(meta.get("default") == "default", "C1: default must be 'default'")
_need(Node._DTYPES[0] == "default" and "fp8_e4m3fn" in Node._DTYPES
      and "fp8_e5m2" in Node._DTYPES,
      "C1: the table must offer default, fp8_e4m3fn and fp8_e5m2")

# ---- C2: the pure mapping -------------------------------------------------
_need(Node.dtype_options("default") == {}, "C2: default must send NO dtype key")
_need(Node.dtype_options(None) == {}, "C2: None (older graph) must act as default")
_need(Node.dtype_options("fp8_e4m3fn") == {"dtype": torch.float8_e4m3fn},
      "C2: fp8_e4m3fn -> torch.float8_e4m3fn")
_need(Node.dtype_options("fp8_e5m2") == {"dtype": torch.float8_e5m2},
      "C2: fp8_e5m2 -> torch.float8_e5m2")
try:
    Node.dtype_options("fp8_e4m3fn_fast")
    _fail("C2: an unknown word must raise, not fall through to full size")
except ValueError as e:
    _need("weight_dtype" in str(e), "C2: the error must name the field")


# ---- C3: driven through load() -------------------------------------------
def _run(**kw):
    CALLS.clear()
    out = Node().load(clip_name=ENC, type="minimax", **kw)
    _need(len(CALLS) == 1, "C3: load_clip must be called exactly once")
    return CALLS[0], out["result"][1]


call, info = _run(weight_dtype="fp8_e4m3fn")
_need(call["model_options"].get("dtype") is torch.float8_e4m3fn,
      "C3: fp8_e4m3fn did not reach model_options['dtype'], got %r"
      % (call["model_options"],))
_need("| fp8_e4m3fn |" in info,
      "C3: the readout must name the dtype word, got %r" % (info,))

call, info = _run(weight_dtype="fp8_e5m2", device="cpu")
mo = call["model_options"]
_need(mo.get("dtype") is torch.float8_e5m2 and "load_device" in mo
      and "offload_device" in mo,
      "C3: dtype and the cpu device keys must coexist, got %r" % (mo,))
_need(str(mo["load_device"]) == "cpu", "C3: cpu device lost its value")

call, info = _run()
_need("dtype" not in call["model_options"],
      "C3: default must leave the dtype to core (no key), got %r"
      % (call["model_options"],))
_need("| default |" in info, "C3: the readout must say 'default' when nothing was cast")

# ---- C4: the frontend half ------------------------------------------------
js = open(JS, encoding="utf-8").read()


def _lift(src, signature):
    i = src.find(signature)
    _need(i >= 0, "C4: could not lift %r from ph_basics.js" % signature)
    end = src.find("\n}", i)
    return src[i:end + 2]


def _lift_const(src, name):
    i = src.find("const %s = " % name)
    _need(i >= 0, "C4: constant %s is gone from ph_basics.js" % name)
    end = src.find("];", i)
    return src[i:end + 2]


script = "\n".join([
    _lift_const(js, "CLIP_CANON"),
    _lift_const(js, "CLIP_DISPLAY"),
    _lift(js, "function _reorder(node, order)"),
    _lift(js, "function _toDisplay(node, spec)"),
    """
const spec = { canon: CLIP_CANON, display: CLIP_DISPLAY };
const out = { canon: CLIP_CANON, display: CLIP_DISPLAY };
// a graph saved BEFORE v1077: six values, canon order
const onDisk = ["umt5.safetensors", "flux", "cpu", "clip_l.safetensors", "- none -", "- none -"];
const widgets = CLIP_CANON.map((n) => ({ name: n, type: "combo", value: "default" }));
widgets.push({ name: "pls_remove_slot", type: "button", value: null, serialize: false });
const node = { widgets, _plsDisplayed: false };
let i = 0;
for (const w of node.widgets) { if (w.serialize !== false) { if (i >= onDisk.length) break; w.value = onDisk[i++]; } }
_toDisplay(node, spec);
out.screen = node.widgets.map((w) => w.name);
out.values = Object.fromEntries(node.widgets.map((w) => [w.name, w.value]));
console.log(JSON.stringify(out));
""",
])
with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                 encoding="utf-8") as fh:
    fh.write(script)
    path = fh.name
try:
    proc = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
finally:
    os.unlink(path)
_need(proc.returncode == 0, "C4: node run failed: " + (proc.stderr or "")[-600:])
res = json.loads(proc.stdout.strip())
_need(res["canon"][-1] == "weight_dtype",
      "C4: CLIP_CANON must END with weight_dtype (append-only), got %r" % (res["canon"],))
_need(res["canon"][:6] == ["clip_name", "type", "device", "clip_name_2",
                           "clip_name_3", "clip_name_4"],
      "C4: the six older canon entries moved: %r" % (res["canon"],))
disp = res["display"]
_need("weight_dtype" in disp and disp.index("weight_dtype") < disp.index("clip_name")
      and disp.index("weight_dtype") > disp.index("device"),
      "C4: on screen weight_dtype belongs after device and ahead of the slot "
      "block, got %r" % (disp,))
_need(res["screen"][-1] == "pls_remove_slot",
      "C4: the remove button must stay LAST on screen, got %r" % (res["screen"],))
v = res["values"]
_need(v["clip_name"] == "umt5.safetensors" and v["type"] == "flux"
      and v["device"] == "cpu" and v["clip_name_2"] == "clip_l.safetensors",
      "C4: an old six-value graph loaded shifted: %r" % (v,))
_need(v["weight_dtype"] == "default",
      "C4: an old graph must leave weight_dtype at its default, got %r"
      % (v["weight_dtype"],))

# ---- C6: a graph WITHOUT the slot keys counts one encoder ----------------
# Found driving C3: _strip_size(None) is the string "None", and "None" is
# a filename to _is_filled -- three phantom encoders, forced hidream recipe.
call, info = _run(weight_dtype="fp8_e4m3fn")
_need(call["paths"] == [os.path.join(tmp, "text_encoders", ENC)],
      "C6: missing slot keys must mean EMPTY slots, got paths %r" % (call["paths"],))
_need(call["clip_type"] == "minimax",
      "C6: the chosen type must survive (no phantom-count override), got %r"
      % (call["clip_type"],))
call, info = _run(weight_dtype="default", clip_name_2=mod._CLIP_PLACEHOLDER,
                  clip_name_3=None, clip_name_4=mod._CLIP_PLACEHOLDER + " \u00b7 1 KB")
_need(len(call["paths"]) == 1,
      "C6: placeholder / None / decorated placeholder must all count as empty, got %r"
      % (call["paths"],))

# ---- C5: one vocabulary ---------------------------------------------------
for word in Node._DTYPES:
    _need(word in mod.ULSLoadModel._DTYPES,
          "C5: Load CLIP word %r is not a Load Model word -- one vocabulary" % word)

print("[%s] PASS -- weight_dtype appended last, pure mapping exact, fp8 reaches "
      "core's text-encoder loader with the device keys intact, default sends "
      "no key, old graphs load unshifted, one dtype vocabulary" % NAME)
