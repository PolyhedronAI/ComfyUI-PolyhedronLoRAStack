"""test_v1249_drop_lookup -- "load from where it is" finds a dropped file in ComfyUI's own folders.

v1249 (Frank 30.10.): a file dragged from ComfyUI's output tree onto "load from where it is" ended in the folder
picker -- the browser hid the path and the lookup only knew the recent folders and input. And a path the browser DID
hand over but that did not resolve went straight to the picker. Now:

  L1 the lookup walks ComfyUI's output and input trees (subfolders) when the known folders miss (driven)
  L2 the same file in two folders stays an honest refusal: "ambiguous" (driven)
  L3 a file nowhere in ComfyUI's trees: "not_found" -> the picker, never a copy (driven)
  L4 the walk is bounded (dir cap and depth) and skips "_polyhedron" (source pins)
  L5 /uls/media/dirs_exist names which folders exist on this machine (driven); the loader prunes its recents
     with it once per page load (source pin)
  L6 the drop: a path that does not resolve falls through to the lookup; one console line per drop says what the
     browser handed over and where the file was found; B never copies (source pins)
  L7 this file is pure ASCII
"""
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = "test_v1249_drop_lookup"
FAILS = []


def need(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def load_pkg(mod):
    pkg = "pls_v1249_nodes"
    if pkg not in sys.modules:
        p = types.ModuleType(pkg)
        p.__path__ = [os.path.join(ROOT, "nodes")]
        sys.modules[pkg] = p
    spec = importlib.util.spec_from_file_location(pkg + "." + mod, os.path.join(ROOT, "nodes", mod + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[pkg + "." + mod] = m
    spec.loader.exec_module(m)
    return m


class Req:
    def __init__(self, body):
        self._b = body
        self.query = {}
        self.remote = "127.0.0.1"
        self.headers = {}

    async def json(self):
        return self._b


def main():
    print(NAME)
    if "--cpu" not in sys.argv:
        sys.argv = [sys.argv[0], "--cpu"]
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import folder_paths
    except Exception as e:
        print("[%s] SKIP -- the Core is not importable (%s)" % (NAME, e))
        return 0
    base = tempfile.mkdtemp()
    inp, out = os.path.join(base, "input"), os.path.join(base, "output")
    deep = os.path.join(out, "2026-10-30", "rittertest", "ritter_ref")
    os.makedirs(deep)
    os.makedirs(inp)
    folder_paths.set_input_directory(inp)
    folder_paths.set_output_directory(out)
    R = load_pkg("ph_media_routes")
    R._lan_browse_locked = lambda: False

    def write(path, n):
        with open(path, "wb") as fh:
            fh.write(b"\x89PNG" + b"x" * n)
        return int(os.path.getmtime(path))

    mt = write(os.path.join(deep, "PH_100227_ritter_px64_00001_.png"), 100)

    def locate(name, size, mtime, folders=()):
        resp = asyncio.run(R.handle_media_locate(Req({"name": name, "size": size, "mtime": mtime, "folders": list(folders)})))
        return json.loads(resp.body.decode("utf-8"))

    d = locate("PH_100227_ritter_px64_00001_.png", 104, mt, [])
    need(d.get("ok") is True and os.path.realpath(d.get("folder", "")) == os.path.realpath(deep) and d.get("where") == "ComfyUI folders",
         "L1 found deep in output without any known folder (%r)" % (d,))
    twin = os.path.join(out, "2026-10-29")
    os.makedirs(twin)
    write(os.path.join(twin, "twin.png"), 50)
    os.utime(os.path.join(twin, "twin.png"), (time.time(), time.time()))
    t2 = os.path.join(inp, "sub")
    os.makedirs(t2)
    write(os.path.join(t2, "twin.png"), 50)
    mtt = int(os.path.getmtime(os.path.join(t2, "twin.png")))
    os.utime(os.path.join(twin, "twin.png"), (mtt, mtt))
    d2 = locate("twin.png", 54, mtt, [])
    need(d2.get("ok") is False and d2.get("reason") == "ambiguous", "L2 two matches -> ambiguous (%r)" % (d2,))
    d3 = locate("nowhere.png", 10, mt, [])
    need(d3.get("ok") is False and d3.get("reason") == "not_found", "L3 nowhere -> not_found")

    src = open(os.path.join(ROOT, "nodes", "ph_media_routes.py"), encoding="utf-8").read()
    need("_LOCATE_MAX_DIRS = 20000" in src and "_LOCATE_MAX_DEPTH = 8" in src and 'd != "_polyhedron"' in src
         and "if n > _LOCATE_MAX_DIRS:" in src, "L4 the walk is bounded and skips _polyhedron")

    resp = asyncio.run(R.handle_media_dirs_exist(Req({"paths": [deep, "/no/such/dir", inp]})))
    de = json.loads(resp.body.decode("utf-8"))
    need(de.get("ok") is True and de.get("exist") == [deep, inp], "L5 dirs_exist names the real folders (%r)" % (de,))
    js = open(os.path.join(ROOT, "web", "js", "ph_media_loader.js"), encoding="utf-8").read()
    need("function pruneRecentFolders()" in js and "try { pruneRecentFolders(); }" in js
         and '"/uls/media/dirs_exist"' in js, "L5 the loader prunes its recents once per page load")
    need("if (wantPin && !resolved && dt.files && dt.files.length) {" in js
         and "did not resolve here" in js and "MediaLoader drop (load from where it is): " in js
         and "the folder picker opens (nothing is copied)" in js, "L6 a dead path falls through to the lookup, said in the console")
    i = js.index("if (wantPin) {\n            // v1211")
    need("_pinPick();" in js[i:i + 400] and "return;" in js[i:i + 400], "L6 B ends in the picker, never in the copy")

    raw = open(os.path.abspath(__file__), "rb").read()
    need(all(c < 128 for c in raw), "L7 %s is pure ASCII" % os.path.basename(__file__))
    if FAILS:
        print("[%s] FAIL -- %d" % (NAME, len(FAILS)))
        return 1
    print("[%s] PASS -- B finds the file in ComfyUI's own folders; a miss is honest; nothing is copied" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
