# -*- coding: ascii -*-
"""Guard v1347 -- the model relocation puts nothing on screen any more.

  Q1  a working move (flat name -> name in a sub folder) changes the widget and is said in the console only
  Q2  a stored model file that stays unresolved is said in the console only: missing and ambiguous (with the
      count of files of that name); a non-model value ("euler", the select placeholder) never; the page (document)
      is never touched -- a bubble on load also fired for slots that never run, and Core names a real failure
  Q3  no bubble code, the old line ("found at a new place") gone; relocateNode(node) without the second argument
      still works (v1326 callers)
  Q4  ASCII-clean; version stamps agree (pyproject 4.<build-400>.0)

Script-style: exit 0 = pass.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, "web", "js")
NAME = "test_v1347_relocate_quiet"
FAILS = []


def check(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


jsp = os.path.join(JS, "ph_model_relocate.js")
js = read(jsp)
check("found at a new place" not in js, "Q3 the old success line is gone")
check("ph-relocate-toast" not in js and "document." not in js and "_pending" not in js, "Q3 no bubble code")
check(re.search(r"const moves = relocateNode\(node, missing\);", js) is not None, "Q2 the hook collects the unresolved")

JS_TEST = r'''const touched = [];
globalThis.document = new Proxy({}, { get(t, k) { touched.push(String(k)); return () => ({}); } });
const logs = [];
console.info = (m) => logs.push("I " + m); console.warn = (m) => logs.push("W " + m);
const { relocateNode, sameNameCount, _ext } = await import("./m.mjs");
const D = "MIDDOT";
const O = ["MiniMax_H3\\a.safetensors " + D + " 19.5 GB", "X\\d.safetensors", "Y\\d.safetensors", "c.safetensors"];
const bad = [];
const eq = (a, b, m) => { if (JSON.stringify(a) !== JSON.stringify(b)) bad.push(m + "=" + JSON.stringify(a)); };
const ext = _ext[0];
const n1 = { comfyClass: "ULSLoadModel", widgets: [{ value: "a.safetensors", options: { values: O } }] };
ext.loadedGraphNode(n1);
eq(n1.widgets[0].value, O[0], "Q1 moved");
eq(logs.filter(l => l.startsWith("I [PLS] model moved")).length, 1, "Q1 console says it");
const n2 = { comfyClass: "ULSLoadModel", widgets: [{ value: "zzz.safetensors", options: { values: O } },
   { value: "d.safetensors " + D + " 1 GB", options: { values: O } }, { value: "euler", options: { values: O } },
   { value: "EMDASH select model EMDASH", options: { values: O } }, { value: 3 }] };
ext.loadedGraphNode(n2);
const nf = logs.filter(l => l.includes("model not found"));
eq(nf.length, 2, "Q2 two unresolved, non-model values never");
eq(/zzz\.safetensors -- only matters/.test(nf[0] || ""), true, "Q2 missing named");
eq(/d\.safetensors \(2 files of this name\)/.test(nf[1] || ""), true, "Q2 ambiguous with count");
eq(sameNameCount(O, "d.safetensors"), 2, "Q2 count 2"); eq(sameNameCount(O.concat(["x\\D.safetensors"]), "d.safetensors"), 2, "Q2 one file written twice counts once"); eq(sameNameCount(O, "q.safetensors"), 0, "Q2 count 0");
eq(touched.length, 0, "Q2 the page is never touched");
const n3 = { widgets: [{ value: "zzz.safetensors", options: { values: O } }, { value: "c.safetensors", options: { values: O } }] };
eq(relocateNode(n3).length, 0, "Q3 old call works");
const before = logs.length;
ext.loadedGraphNode({ comfyClass: "CheckpointLoaderSimple", widgets: [{ value: "zzz.safetensors", options: { values: O } }] });
eq(logs.length, before, "foreign node untouched");
console.log(bad.length ? "BAD " + bad.join("|") : "JSOK");
'''.replace("MIDDOT", "\\" + "u00b7").replace("EMDASH", "\\" + "u2014")

node_bin = shutil.which("node")
if node_bin:
    tmp = tempfile.mkdtemp(prefix="pls1347_")
    try:
        with open(os.path.join(tmp, "m.mjs"), "w", encoding="ascii") as f:
            f.write(re.sub(r"^import \{ app \} from [^\n]*",
                           "export const _ext = []; const app = { registerExtension(e){ _ext.push(e); } };",
                           js, flags=re.M))
        with open(os.path.join(tmp, "t.mjs"), "w", encoding="ascii") as f:
            f.write(JS_TEST)
        out = subprocess.run([node_bin, os.path.join(tmp, "t.mjs")], capture_output=True, text=True, timeout=60)
        res = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr.strip()[:300]
        check(res == "JSOK", "Q1-Q3 behaviour under node: " + res)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
else:
    print("  note: node not installed, behaviour check skipped (source pins still apply)")

with open(jsp, "rb") as f:
    check(all(c < 128 for c in f.read()), "Q4 ASCII-clean: ph_model_relocate.js")
_pv = re.search(r'const PLUGIN_VERSION = "v(\d+)";', read(os.path.join(JS, "uls_compat.js")))
_pp = re.search(r'^version = "4\.(\d+)\.0"', read(os.path.join(ROOT, "pyproject.toml")), re.M)
# PUBLIC CUT (v409, declared): the public build counts its own way -- vNNN = Major x 100 + Minor (test_v365 V).
check(_pv is not None and _pp is not None and int(_pv.group(1)) >= 409
      and int(_pp.group(1)) == int(_pv.group(1)) - 400, "Q4 version stamps agree (pyproject 4.<build-400>.0)")

if FAILS:
    print("%s FAIL (%d)" % (NAME, len(FAILS)))
    sys.exit(1)
print("%s PASS" % NAME)
