"""Guard v828 -- sizes in the loader combos (the cutout pattern, lifted).

WHAT THIS PINS (driven against a real temp models tree, not read):

  L1  the three loaders' combo lists carry " \u00b7 <size>" for files that
      exist on disk; placeholders stay undecorated; a GB-scale file is
      shown in GB, a KB-scale one in KB.
  L2  the strip is an EXACT SUFFIX match: a filename containing " - v2"
      or a bare " \u00b7 " without a size unit survives untouched. The
      cutout's broad split would eat loader filenames -- this one must
      never.
  L3  ROUND TRIP: a decorated entry taken from the list reaches the load
      path as the bare filename and resolves to the real file
      (ULSLoadModel driven end to end against a recording comfy stand-in).
  L4  a workflow saved BEFORE this cut (bare filename) passes validation:
      VALIDATE_INPUTS exists on all three loaders and is permissive.
  L5  MUTATION: with the strip removed from ULSLoadModel.load, the
      decorated value must NOT resolve -- proof the strip is load-bearing
      and L3 is driven by the mechanism, not by the fixture.
  L6  the frontend half exists and mirrors the backend: normalizeSized in
      ph_basics.js, a SIZE_SUFFIX regex with the same shape, wired after
      the widgets_values pour (text pin for the JS half).

Script-style: exit 0 = pass.
"""
import importlib.util
import os
import sys
import tempfile
import types

NAME = "v828"
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


# ---- the temp models tree ------------------------------------------------
tmp = tempfile.mkdtemp()


def _mkfile(rel, size):
    path = os.path.join(tmp, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.truncate(size)          # sparse -- honest getsize, no real bytes
    return path


_mkfile("text_encoders/clip_l.safetensors", 4 * 1024)
_mkfile("text_encoders/t5xxl_fp16.safetensors", 3 * 1024 * 1024)
_mkfile("vae/wan_vae.safetensors", int(1.5 * (1 << 30)))
_mkfile("diffusion_models/model - v2.safetensors", 7 * 1024 * 1024)

FOLDERS = {
    "text_encoders": ["clip_l.safetensors", "t5xxl_fp16.safetensors"],
    "vae": ["wan_vae.safetensors"],
    "diffusion_models": ["model - v2.safetensors"],
}


def _full(folder, name):
    p = os.path.join(tmp, folder, str(name))
    return p if os.path.isfile(p) else None


fp = types.ModuleType("folder_paths")
fp.get_filename_list = lambda f: list(FOLDERS.get(f, []))
fp.get_full_path = _full
fp.get_folder_paths = lambda f: [os.path.join(tmp, f)]


def _full_or_raise(folder, name):
    p = _full(folder, name)
    if not p:
        raise FileNotFoundError("%s/%s" % (folder, name))
    return p


fp.get_full_path_or_raise = _full_or_raise
sys.modules["folder_paths"] = fp

LOADED = {}
comfy = types.ModuleType("comfy")
comfy_sd = types.ModuleType("comfy.sd")
comfy_utils = types.ModuleType("comfy.utils")
comfy_sd.load_diffusion_model = (
    lambda path, model_options=None: LOADED.setdefault("path", path) or "M")
comfy.sd = comfy_sd
comfy.utils = comfy_utils
sys.modules["comfy"] = comfy
sys.modules["comfy.sd"] = comfy_sd
sys.modules["comfy.utils"] = comfy_utils


def _load(src=None):
    """ph_basics has top-level relative imports -- load it as a member of a
    throwaway package so `from . import uls_noise` resolves."""
    pkg = types.ModuleType("plspack")
    pkg.__path__ = [os.path.join(ROOT, "nodes")]
    sys.modules["plspack"] = pkg
    if src is None:
        spec = importlib.util.spec_from_file_location("plspack.ph_basics", MOD)
        m = importlib.util.module_from_spec(spec)
        sys.modules["plspack.ph_basics"] = m
        spec.loader.exec_module(m)
        return m
    m = types.ModuleType("plspack.ph_basics_mut")
    m.__package__ = "plspack"
    m.__file__ = MOD
    exec(compile(src, MOD, "exec"), m.__dict__)
    return m


mod = _load()

# ---- L1: the lists carry sizes ------------------------------------------
clip_inputs = mod.ULSLoadCLIP.INPUT_TYPES()
clip_list = clip_inputs["required"]["clip_name"][0]
_need("clip_l.safetensors \u00b7 4 KB" in clip_list,
      "L1: clip list lacks the KB entry, got %r" % (clip_list,))
_need("t5xxl_fp16.safetensors \u00b7 3 MB" in clip_list,
      "L1: clip list lacks the MB entry, got %r" % (clip_list,))
slot2 = clip_inputs["optional"]["clip_name_2"][0]
_need(slot2[0] == mod._CLIP_PLACEHOLDER,
      "L1: slot placeholder must stay first and undecorated")
_need("\u00b7" not in slot2[0], "L1: the placeholder got decorated")

vae_list = mod.ULSLoadVAE.INPUT_TYPES()["required"]["vae_name"][0]
_need("wan_vae.safetensors \u00b7 1.5 GB" in vae_list,
      "L1: vae list lacks the GB entry, got %r" % (vae_list,))

model_list = mod._model_source_list()
_need(model_list[0] == mod._MODEL_PLACEHOLDER,
      "L1: model placeholder must stay first")
_need("model - v2.safetensors \u00b7 7 MB" in model_list,
      "L1: model list lacks the sized entry, got %r" % (model_list,))

# ---- L2: the strip is exact ---------------------------------------------
_need(mod._strip_size("model - v2.safetensors \u00b7 7 MB")
      == "model - v2.safetensors",
      "L2: strip failed on the dashed filename")
_need(mod._strip_size("model - v2.safetensors") == "model - v2.safetensors",
      "L2: strip ate part of an undecorated dashed filename")
_need(mod._strip_size("a \u00b7 b.safetensors") == "a \u00b7 b.safetensors",
      "L2: strip ate a middot that carries no size unit")
_need(mod._strip_size("x \u00b7 12.5 GB") == "x",
      "L2: strip missed a plain decorated value")

# ---- L3: round trip through ULSLoadModel.load ----------------------------
node = mod.ULSLoadModel()
decorated = "model - v2.safetensors \u00b7 7 MB"
out = node.load(select=1, weight_dtype="default", model_1=decorated)
want = os.path.join(tmp, "diffusion_models", "model - v2.safetensors")
_need(LOADED.get("path") == want,
      "L3: the decorated pick did not resolve to the real file "
      "(got %r)" % LOADED.get("path"))

# ---- L4: pre-cut workflows pass validation -------------------------------
for cls_name in ("ULSLoadCLIP", "ULSLoadVAE", "ULSLoadModel"):
    cls = getattr(mod, cls_name)
    _need(hasattr(cls, "VALIDATE_INPUTS"),
          "L4: %s has no VALIDATE_INPUTS -- a pre-v828 graph dies at "
          "server validation" % cls_name)
    _need(cls.VALIDATE_INPUTS(vae_name="wan_vae.safetensors") is True,
          "L4: %s.VALIDATE_INPUTS is not permissive" % cls_name)

# ---- L5: mutation -- remove the strip, the round trip must break ---------
with open(MOD, "r", encoding="utf-8") as fh:
    SRC = fh.read()
MUT = SRC.replace("            name = _strip_size(name)  # v828",
                  "            pass  # v828 strip removed", 1)
_need(MUT != SRC, "L5: mutation anchor not found")
LOADED.clear()
mmod = _load(MUT)
mnode = mmod.ULSLoadModel()
try:
    mnode.load(select=1, weight_dtype="default", model_1=decorated)
    broke = False
except Exception:
    broke = True
_need(broke and not LOADED.get("path"),
      "L5: with the strip removed the decorated value STILL resolved -- "
      "L3 is not driven by the strip")

# ---- L6: the frontend half ----------------------------------------------
with open(JS, "r", encoding="utf-8") as fh:
    js = fh.read()
_need("function normalizeSized" in js,
      "L6: normalizeSized is gone from ph_basics.js")
_need("\\s\\u00b7\\s[\\d.,]+\\s?(KB|MB|GB|TB)$" in js,
      "L6: the JS SIZE_SUFFIX no longer mirrors the backend regex")
_need("normalizeSized(this)" in js,
      "L6: normalizeSized is not wired on node creation")

print("[%s] PASS -- sized loader combos: lists decorated, strip exact, "
      "round trip driven, mutation caught, JS half present" % NAME)
