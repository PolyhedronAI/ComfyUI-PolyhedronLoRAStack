#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1326 -- a model file that moved into a sub folder is found again by its file name.

Frank, 13.11.: the models folder is sorted by family (Family/model.safetensors); saved workflows keep the old flat
name and must keep running.

WHAT IS PINNED

  R1  ph_model_path.relocate (library, no node): a name that still resolves is returned untouched; a name that is
      gone and has exactly ONE file of the same file name in the same kind of folder returns that entry (case and
      both slash kinds ignored, folders merged); nothing found returns the stored name; two or more are REFUSED by
      name (never a guess); the same path listed by two folder types counts once; the move is logged.
  R2  the callers: Load CLIP (both collection points), Load VAE, Load CLIP Vision, Load Model (over exactly the
      folder types its list merges) and the control file resolver ask relocate BEFORE they resolve the path.
  R3  the frontend mirror ph_model_relocate.js: same size-suffix pattern as ph_basics.js, model files only, one
      match or nothing, runs on loadedGraphNode for Polyhedron nodes only, says what moved; behaviour checked by
      running the pure functions under node when node is installed.
  R4  no node of its own; both files are ASCII-clean; version stamps agree with each other.

Script-style: exit 0 = pass.
"""

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile

NAME = "v1326"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODES = os.path.join(ROOT, "nodes")
JS = os.path.join(ROOT, "web", "js")

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)
        print("  FAIL " + msg)


def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(NODES, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakePaths(object):
    def __init__(self, lists):
        self.lists = lists

    def get_filename_list(self, folder):
        if folder not in self.lists:
            raise KeyError(folder)
        return list(self.lists[folder])

    def get_full_path(self, folder, name):
        return ("/m/%s/%s" % (folder, name)) if name in self.lists.get(folder, ()) else None


# ---------------------------------------------------------------- R1
P = load("ph_model_path")
fp = FakePaths({
    "diffusion_models": ["MiniMax_H3\\a.safetensors", "Qwen_Image/b.safetensors", "flat.safetensors",
                         "X\\dup.safetensors", "Y\\dup.safetensors"],
    "unet": ["MiniMax_H3\\a.safetensors", "MiniMax_H3\\a.safetensors", "Other\\u.safetensors"],
    "vae": ["MiniMax_H3\\v.safetensors"],
})
said = []
rel = lambda folders, name: P.relocate(folders, name, "T", folder_paths=fp, log=said.append)  # noqa: E731
DM = ("diffusion_models", "unet")
check(rel(DM, "flat.safetensors") == "flat.safetensors" and not said, "R1 a name that resolves is untouched and silent")
check(rel(DM, "a.safetensors") == "MiniMax_H3\\a.safetensors", "R1 flat -> family (one match, listed by two folder types)")
check(len(said) == 1 and "a.safetensors" in said[0] and "MiniMax_H3" in said[0], "R1 the move is logged with both names")
check(rel(DM, "old/sub\\B.SAFETENSORS") == "Qwen_Image/b.safetensors", "R1 case and both slash kinds ignored")
check(rel(DM, "u.safetensors") == "Other\\u.safetensors", "R1 the second folder type counts")
check(rel(DM, "nowhere.safetensors") == "nowhere.safetensors", "R1 nothing found keeps the stored name")
check(rel(("vae",), "a.safetensors") == "a.safetensors", "R1 another kind of folder is not searched")
check(rel("vae", "v.safetensors") == "MiniMax_H3\\v.safetensors", "R1 a single folder type may be passed as a string")
check(rel(("nope", "vae"), "v.safetensors") == "MiniMax_H3\\v.safetensors", "R1 an unknown folder type is skipped")
try:
    rel(DM, "dup.safetensors")
    check(False, "R1 two files of one file name are refused")
except ValueError as e:
    check("X\\dup.safetensors" in str(e) and "Y\\dup.safetensors" in str(e) and "T:" in str(e),
          "R1 the refusal names both candidates and the node")
check(P.file_name("A\\B/C.Safetensors ") == "c.safetensors", "R1 file_name")
check(P.candidates(["a/x.pt", "A\\x.pt", "b/x.pt", "y.pt"], "x.pt") == ["a/x.pt", "b/x.pt"], "R1 candidates: one per distinct path")

# ---------------------------------------------------------------- R2
basics = read(os.path.join(NODES, "ph_basics.py"))
control = read(os.path.join(NODES, "ph_control.py"))
check(re.search(r"^from \. import ph_model_path\b", basics, re.M) is not None, "R2 ph_basics imports ph_model_path")
check(len(re.findall(r'ph_model_path\.relocate\(\("text_encoders",\), _strip_size\((clip_name|value)\), "Load CLIP"\)', basics)) == 2,
      "R2 Load CLIP relocates at both collection points")
m = re.search(r'vae_name = ph_model_path\.relocate\(\("vae",\), vae_name, "Load VAE"\)', basics)
g = basics.find('folder_paths.get_full_path_or_raise("vae", vae_name)')
check(m is not None and 0 < m.start() < g, "R2 Load VAE relocates before it resolves the path")
m = re.search(r'name = ph_model_path\.relocate\(\("clip_vision",\), _strip_size\(clip_name\), "Load CLIP Vision"\)', basics)
g = basics.find('folder_paths.get_full_path_or_raise("clip_vision", name)')
check(m is not None and 0 < m.start() < g, "R2 Load CLIP Vision relocates before it resolves the path")
m = re.search(r'name = ph_model_path\.relocate\(_MODEL_FOLDERS, str\(name\), "Load Model"\)\s*(#[^\n]*)?\n\s*kind = _detect_kind\(name\)', basics)
check(m is not None, "R2 Load Model relocates before it detects the kind")
mf = re.search(r"^_MODEL_FOLDERS = \(([^)]*)\)", basics, re.M)
want = ["diffusion_models", "unet", "checkpoints", "unet_gguf", "diffusion_models_gguf"]
check(mf is not None and re.findall(r'"([a-z_]+)"', mf.group(1)) == want, "R2 _MODEL_FOLDERS names the five folder types")
src = re.search(r"def _model_source_list\(\):.*?return ", basics, re.S)
check(src is not None and all(('"%s"' % w) in src.group(0) for w in want), "R2 the Load Model list merges those same folder types")
m = re.search(r"name = ph_model_path\.relocate\(\(folder,\), name, node\)[^\n]*\n\s*return folder_paths\.get_full_path_or_raise\(folder, name\)", control)
check(m is not None, "R2 the control file resolver relocates before it resolves the path")

# ---------------------------------------------------------------- R3
jsp = os.path.join(JS, "ph_model_relocate.js")
js = read(jsp)
pat = r"const SIZE_SUFFIX = (/[^\n]+/);"
a = re.search(pat, js)
b = re.search(pat, read(os.path.join(JS, "ph_basics.js")))
check(a is not None and b is not None and a.group(1) == b.group(1), "R3 the size-suffix pattern equals ph_basics.js")
check('name: "polyhedron.modelRelocate"' in js and "loadedGraphNode(node)" in js, "R3 registered, runs on loadedGraphNode")
check('cls.startsWith("ULS")' in js and "if (!isOurs(node)) return;" in js, "R3 Polyhedron nodes only")
check("return hits.length === 1 ? hits[0] : undefined;" in js, "R3 one match or nothing")
check("if (!MODEL_FILE.test(bare)) return undefined;" in js, "R3 model files only")
check("model found at a new place" in js and "console.info" in js, "R3 says what moved")
node_bin = shutil.which("node")
if node_bin:
    tmp = tempfile.mkdtemp(prefix="pls1326_")
    try:
        with open(os.path.join(tmp, "m.mjs"), "w", encoding="ascii") as f:
            f.write(re.sub(r"^import \{ app \} from [^\n]*", "const app = { registerExtension(){} };", js, flags=re.M))
        with open(os.path.join(tmp, "t.mjs"), "w", encoding="ascii") as f:
            f.write(r'''
import { relocated, relocateNode } from "./m.mjs";
const D = "\u00b7";
const O = ["MiniMax_H3\\a.safetensors " + D + " 19.5 GB", "Qwen_Image\\b.safetensors " + D + " 6.8 GB",
           "c.safetensors " + D + " 1.0 GB", "X\\d.safetensors", "Y\\d.safetensors", "MiniMax_H3\\sub\\e.safetensors"];
const bad = [];
const eq = (a, b, m) => { if (a !== b) bad.push(m); };
eq(relocated(O, "a.safetensors"), O[0], "flat->family");
eq(relocated(O, "A.safetensors " + D + " 1 GB"), O[0], "case+size");
eq(relocated(O, "c.safetensors"), O[2], "size only");
eq(relocated(O, "d.safetensors"), undefined, "ambiguous");
eq(relocated(O, "zzz.safetensors"), undefined, "none");
eq(relocated(O, "old/e.safetensors"), O[5], "other folder");
eq(relocated(O, "euler"), undefined, "not a model file");
const n = { comfyClass: "ULSLoadModel", widgets: [{ value: "a.safetensors", options: { values: O } },
    { value: "d.safetensors", options: { values: O } }, { value: "c.safetensors", options: { values: O } }, { value: 3 }] };
const mv = relocateNode(n);
eq(mv.length, 1, "only a real move is reported"); eq(n.widgets[0].value, O[0], "w0 moved");
eq(n.widgets[1].value, "d.safetensors", "w1 ambiguous kept"); eq(n.widgets[2].value, O[2], "w2 sized");
console.log(bad.length ? "BAD " + bad.join("|") : "JSOK");
''')
        out = subprocess.run([node_bin, os.path.join(tmp, "t.mjs")], capture_output=True, text=True, timeout=60)
        check(out.stdout.strip() == "JSOK", "R3 behaviour under node: " + (out.stdout.strip() or out.stderr.strip()[:200]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
else:
    print("  note: node not installed, R3 behaviour check skipped (source pins still apply)")

# ---------------------------------------------------------------- R4
lib = read(os.path.join(NODES, "ph_model_path.py"))
check("NODE_CLASS_MAPPINGS" not in lib and "INPUT_TYPES" not in lib, "R4 the library registers no node")
for path in (os.path.join(NODES, "ph_model_path.py"), jsp):
    with open(path, "rb") as f:
        raw = f.read()
    check(all(c < 128 for c in raw), "R4 ASCII-clean: " + os.path.basename(path))
# v1327: the two stamps are checked against EACH OTHER (minor = build - 400), not against this build's number --
# pinned to 4.926.0 / v1326 this guard went red on the very next build.
_pv = re.search(r'const PLUGIN_VERSION = "v(\d+)";', read(os.path.join(JS, "uls_compat.js")))
_pp = re.search(r'^version = "4\.(\d+)\.0"', read(os.path.join(ROOT, "pyproject.toml")), re.M)
# PUBLIC CUT (v407, declared): the public build counts its own way -- vNNN = Major x 100 + Minor (test_v365 V) --
# so the stamps are checked against each other in that scheme here.
check(_pv is not None and _pp is not None and int(_pv.group(1)) >= 407
      and int(_pp.group(1)) == int(_pv.group(1)) - 400, "R4 version stamps agree (pyproject 4.<build-400>.0, public vNNN)")

if FAILS:
    print("%s FAIL (%d)" % (NAME, len(FAILS)))
    sys.exit(1)
print("%s PASS" % NAME)
