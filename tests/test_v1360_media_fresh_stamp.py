# -*- coding: ascii -*-
"""Guard v1360 -- Media Loader: the Selection shows the file as it is on disk now, not a cached one of the same name.

Frank, 22.11.: a built walk workflow (stored mtime 0) named view_NE_0000.png; the Selection showed 1872 x 3408 with a
wide transparent border, the tiles 704 x 1280. The preview URL carried the stamp stored in the workflow (&t=0), the
same URL an earlier workflow had asked for, and the browser handed back the HUGE still it had cached under it. The
backend always read the current file -- only the picture in the Selection lied.

  S1  _stampOf: a file the listing of the same folder holds -> "<mtime>_<size>" from the listing (wins over the
      stored mtime); a file not in the listing, or a listing of another folder -> the stored mtime
  S2  a copied file that keeps its mtime but changes its size gets a new stamp
  S3  _fileURLFor and _thumbURLFor both use _stampOf (no bare f.mtime left in a preview URL)
  S4  _renderPreview notes the stamp it drew from; _refreshStaleSelection re-draws only when the stamp moved
      (and never without a selection or before a first draw)
  S5  both listing paths (refreshGrid, the silent focus re-read) call _refreshStaleSelection when nothing vanished
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


def method(name):
    m = re.search(r"\n    " + name + r"\(([^)]*)\) \{\n(.*?)\n    \}\n", JS, re.S)
    return (m.group(1), m.group(2)) if m else None


stamp, refresh = method("_stampOf"), method("_refreshStaleSelection")
check(stamp is not None, "S1 _stampOf exists")
check(refresh is not None, "S4 _refreshStaleSelection exists")

if stamp and refresh:
    # AMENDED v1361, declared: _stampOf compares folders through the module helper _sameFolder (spelling-proof);
    # the driver carries the two helpers verbatim, the promise of S1-S4 is unchanged.
    helpers = "\n".join(re.findall(r"^function _(?:folderKey|sameFolder)\(.*$", JS, re.M))
    prog = helpers + """
class L {
  _stampOf(%s) {\n%s\n  }
  _refreshStaleSelection(%s) {\n%s\n  }
  _renderPreview() { this.draws = (this.draws || 0) + 1; this._shownStamp = this._stampOf(this.state.folder, {name: this.state.file, mtime: this.state.mtime}); }
}
const out = [];
const l = new L(); l._filesFolder = "D:\\\\in"; l._files = [{name: "a.png", mtime: 1763744439.7, size: 29846}];
out.push(l._stampOf("D:\\\\in", {name: "a.png", mtime: 0}));
out.push(l._stampOf("D:\\\\in", {name: "b.png", mtime: 12.9}));
out.push(l._stampOf("D:\\\\other", {name: "a.png", mtime: 5}));
const before = l._stampOf("D:\\\\in", {name: "a.png", mtime: 0});
l._files = [{name: "a.png", mtime: 1763744439.7, size: 811002}];
out.push(String(before !== l._stampOf("D:\\\\in", {name: "a.png", mtime: 0})));
// S4
const r = new L(); r._filesFolder = "D:\\\\in"; r._files = [];
r.state = {folder: "D:\\\\in", file: "a.png", mtime: 0};
r._refreshStaleSelection(); out.push(String(r.draws || 0));          // before a first draw: nothing
r._renderPreview(); r._refreshStaleSelection(); out.push(String(r.draws));   // same stamp: no re-draw
r._files = [{name: "a.png", mtime: 99, size: 7}];
r._refreshStaleSelection(); out.push(String(r.draws));                // listing moved the stamp: re-draw
r._refreshStaleSelection(); out.push(String(r.draws));                // and only once
const e = new L(); e.state = {folder: "", file: "", mtime: 0}; e._shownStamp = "0";
e._refreshStaleSelection(); out.push(String(e.draws || 0));          // no selection: nothing
console.log(out.join("|"));
""" % (stamp[0], stamp[1], refresh[0], refresh[1])
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
        fh.write(prog)
        path = fh.name
    try:
        res = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(path)
    got = res.stdout.strip().split("|")
    check(res.returncode == 0, "node ran: " + res.stderr[-300:])
    if len(got) == 9:
        check(got[0] == "1763744439_29846", "S1 listing wins over the stored mtime: " + got[0])
        check(got[1] == "12", "S1 not in the listing -> stored mtime: " + got[1])
        check(got[2] == "5", "S1 another folder's listing is not used: " + got[2])
        check(got[3] == "true", "S2 same mtime, new size -> new stamp")
        check(got[4] == "0", "S4 nothing before a first draw")
        check(got[5] == "1", "S4 an unchanged stamp does not re-draw")
        check(got[6] == "2", "S4 a moved stamp re-draws")
        check(got[7] == "2", "S4 re-draws once")
        check(got[8] == "0", "S4 no selection, no re-draw")
    else:
        check(False, "node output: %r" % res.stdout)

# S3
for name in ("_fileURLFor", "_thumbURLFor"):
    m = method(name)
    check(m is not None and "this._stampOf(folder, f)" in m[1] and "f.mtime" not in m[1],
          "S3 %s uses _stampOf" % name)
check(len(re.findall(r'"&t=" \+ Math\.floor\(f\.mtime', JS)) == 0, "S3 no preview URL with the bare stored mtime")

# S4 render notes the stamp
rp = JS[JS.index("\n    _renderPreview() {"):]
rp = rp[:rp.index("\n    }\n")]
check("this._shownStamp = this._stampOf(s.folder, f);" in rp, "S4 _renderPreview notes its stamp")

# S5
check(JS.count("if (gone.length) { this._renderPreview(); this._announceVanished(gone); }\n"
               "            else this._refreshStaleSelection();") == 1, "S5 refreshGrid re-checks the selection")
check(JS.count("if (gone.length) { this._renderPreview(); this._announceVanished(gone); }\n"
               "        else this._refreshStaleSelection();") == 1, "S5 the focus re-read re-checks the selection")

# Z
for rel in ("web/js/ph_media_loader.js", "tests/test_v1360_media_fresh_stamp.py"):
    with open(os.path.join(ROOT, rel), "rb") as fh:
        data = fh.read()
    if rel.endswith(".py"):
        check(all(c < 128 for c in data), "Z  ASCII-clean: " + rel)
_pv = re.search(r'const PLUGIN_VERSION = "v(\d+)";', open(os.path.join(ROOT, "web", "js", "uls_compat.js")).read())
_pp = re.search(r'^version = "4\.(\d+)\.0"', open(os.path.join(ROOT, "pyproject.toml")).read(), re.M)
check(_pv and _pp and int(_pp.group(1)) == int(_pv.group(1)) - 400, "Z  version stamps agree")

if FAILS:
    print("test_v1360_media_fresh_stamp: %d FAIL" % len(FAILS))
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("test_v1360_media_fresh_stamp: all pass")
