# -*- coding: ascii -*-
"""Guard v1361 -- Media Loader: no old picture stays in the Selection, whatever the folder's spelling.

Frank, 22.11. (v1360 in the field): one loader still showed a 1872 x 3408 still that was no longer in the input
folder. v1360 took the cache stamp from the listing only when the browsed folder and the selection's folder were
the same STRING; two spellings of one folder (backslash / slash, case, a trailing separator) missed each other and
the stored stamp came back. And a page keeps every picture it loaded under its URL without asking the server.

  F1  _folderKey / _sameFolder: backslash = slash, case ignored, trailing separators ignored; an empty folder is
      never the same as anything
  F2  _stampOf uses _sameFolder: the listing stamp wins for every spelling of the same folder, not for another one
  F3  the counter-check: the Selection's <img> compares its natural size with the size the listing reports for
      the file now, and on a difference loads once more under a fresh URL (&r=...) instead of taking the size;
      only once per picture, only from the full file (not the thumbnail fallback), only for the listed folder
  Z   ASCII; version stamps agree

Script-style: exit 0 = pass.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(c, m):
    if not c:
        FAILS.append(m)


JS = open(os.path.join(ROOT, "web", "js", "ph_media_loader.js"), encoding="utf-8").read()
helpers = re.findall(r"^function _(?:folderKey|sameFolder)\(.*$", JS, re.M)
check(len(helpers) == 2, "F1 the two folder helpers exist at module level")
m = re.search(r"\n    _stampOf\(([^)]*)\) \{\n(.*?)\n    \}\n", JS, re.S)
check(m is not None and "_sameFolder(folder, this._filesFolder)" in m.group(2), "F2 _stampOf compares through _sameFolder")

if len(helpers) == 2 and m:
    prog = "\n".join(helpers) + """
class L { _stampOf(%s) {\n%s\n} }
const o = [];
o.push(_sameFolder("D:\\\\COMFYUI_2026\\\\ComfyUI\\\\input", "D:/COMFYUI_2026/ComfyUI/input"));
o.push(_sameFolder("D:\\\\COMFYUI_2026\\\\ComfyUI\\\\input\\\\", "d:\\\\comfyui_2026\\\\comfyui\\\\input"));
o.push(_sameFolder("D:/a/input/", "D:/a/input"));
o.push(_sameFolder("D:/a/input", "D:/a/inputs"));
o.push(_sameFolder("", ""));
o.push(_sameFolder("D:/a", ""));
const l = new L(); l._filesFolder = "D:/COMFYUI_2026/ComfyUI/input/"; l._files = [{name: "v.png", mtime: 50.5, size: 9}];
o.push(l._stampOf("D:\\\\COMFYUI_2026\\\\ComfyUI\\\\input", {name: "v.png", mtime: 0}));
o.push(l._stampOf("D:\\\\COMFYUI_2026\\\\ComfyUI\\\\output", {name: "v.png", mtime: 3}));
console.log(o.join("|"));
""" % (m.group(1), m.group(2))
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
        fh.write(prog)
        path = fh.name
    try:
        r = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(path)
    got = r.stdout.strip().split("|")
    check(r.returncode == 0, "node ran: " + r.stderr[-300:])
    want = ["true", "true", "true", "false", "false", "false", "50_9", "3"]
    names = ["F1 backslash = slash", "F1 case + trailing separator", "F1 trailing slash", "F1 another folder",
             "F1 empty is never the same", "F1 empty against a folder", "F2 listing stamp for another spelling",
             "F2 another folder keeps the stored stamp"]
    if len(got) == len(want):
        for g, w, n in zip(got, want, names):
            check(g == w, "%s: %s" % (n, g))
    else:
        check(False, "node output %r" % r.stdout)

rp = JS[JS.index("\n    _renderPreview() {"):]
rp = rp[:rp.index("\n    _thumbURL(f)")]
check("let refetched = false;" in rp, "F3 one refetch flag per picture")
check(re.search(r"const now = \(this\._files \|\| \[\]\)\.find\(\(x\) => x\.name === s\.file\);", rp) is not None,
      "F3 the size the listing reports NOW")
check("(now.w !== img.naturalWidth || now.h !== img.naturalHeight)" in rp, "F3 compares width and height")
check("!refetched && _sameFolder(s.folder, this._filesFolder)" in rp, "F3 only once, only for the listed folder")
check('img.src = this._fileURLFor(s.folder, f) + "&r=" + Date.now();' in rp, "F3 reloads under a fresh URL")
check(rp.index("refetched = true;") < rp.index('"&r=" + Date.now()'), "F3 the flag is set before the reload")
check("if (usingThumb || !img.naturalWidth) return;" in rp, "F3 not for the thumbnail fallback")
check(rp.index("return;\n                }\n                this._setPrevDim(img.naturalWidth") > 0, "F3 a wrong picture never sets the badge")

for rel in ("tests/test_v1361_media_counter_check.py",):
    with open(os.path.join(ROOT, rel), "rb") as fh:
        check(all(c < 128 for c in fh.read()), "Z  ASCII-clean: " + rel)
_pv = re.search(r'const PLUGIN_VERSION = "v(\d+)";', open(os.path.join(ROOT, "web", "js", "uls_compat.js")).read())
_pp = re.search(r'^version = "4\.(\d+)\.0"', open(os.path.join(ROOT, "pyproject.toml")).read(), re.M)
check(_pv and _pp and int(_pp.group(1)) == int(_pv.group(1)) - 400, "Z  version stamps agree")

if FAILS:
    print("test_v1361_media_counter_check: %d FAIL" % len(FAILS))
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("test_v1361_media_counter_check: all pass")
