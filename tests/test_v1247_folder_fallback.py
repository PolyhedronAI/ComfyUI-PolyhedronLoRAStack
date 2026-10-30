"""test_v1247_folder_fallback (public cut: no F4, the Palette node is not in this build) -- a workflow from another machine never meets a dead loader.

v1247 (Frank 30.10., Ritter-Test RT-2a): the Media Loader opened on the build machine's folder
(/home/claude/srv/ComfyUI/input) -- "Cannot read this folder", and the Palette said only "file not
found" when the workflow asked for an older palette. Now:

  F1 the listing route answers a missing folder with ComfyUI's input folder as `fallback` (driven)
  F2 the same for no folder at all; a real folder still lists ok (driven)
  F3 the loader adopts the fallback ONCE (loop guard), says it in the caption and the console,
     and no longer stops at "Pick a folder" before asking the server (source pins)
  F4 Palette: a missing file names the palettes that ARE in the folder (driven)
  F5 tools/wf_hygiene.py finds a bench path anywhere in a workflow (driven) and the pack's
     example_workflows carry none
  F6 this file is pure ASCII
"""
import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = "test_v1247_folder_fallback"
FAILS = []


def need(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def load_pkg(mod):
    pkg = "pls_v1247_nodes"
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
    def __init__(self, q):
        self.query = q
        self.remote = "127.0.0.1"
        self.headers = {}
        self.host = "127.0.0.1:8188"


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
    inp = tempfile.mkdtemp()
    folder_paths.set_input_directory(inp)
    try:
        R = load_pkg("ph_media_routes")
    except Exception as e:
        print("[%s] SKIP -- ph_media_routes not importable here (%s)" % (NAME, e))
        return 0
    R._lan_browse_locked = lambda: False

    def call(q):
        resp = asyncio.run(R.handle_media_list(Req(q)))
        return resp.status, json.loads(resp.body.decode("utf-8"))

    st, d = call({"folder": "/no/such/folder/on/this/machine", "dims": "0"})
    need(st == 404 and d.get("ok") is False and d.get("fallback") == os.path.realpath(inp),
         "F1 a missing folder -> fallback = ComfyUI input (%r)" % (d.get("fallback"),))
    st2, d2 = call({"folder": "", "dims": "0"})
    need(d2.get("ok") is False and d2.get("fallback") == os.path.realpath(inp), "F2 no folder -> the same fallback")
    st3, d3 = call({"folder": inp, "dims": "0"})
    need(st3 == 200 and d3.get("ok") is True, "F2 a real folder still lists ok")

    js = open(os.path.join(ROOT, "web", "js", "ph_media_loader.js"), encoding="utf-8").read()
    i = js.index("async refreshGrid(keepPage = false) {")
    body = js[i:i + 4000]
    need("if (!this.folder) {" not in body[:300], "F3 no early stop before asking the server")
    need("if (r) { try { d = await r.json(); }" in body and "if (r && r.ok) d = await r.json();" not in body,
         "F3 the 404 body (with its fallback) is read, not dropped")
    need('typeof d.fallback === "string"' in body and "this._fallbackFrom !== target" in body
         and "this.node.properties.ph_media_browse = fb;" in body and "return this.refreshGrid(false);" in body,
         "F3 the fallback is adopted once (loop guard) and the grid re-reads")
    need("not found here, showing input" in body and "is not on this machine" in body
         and "this.renderPath();" in body, "F3 the switch is said (console + note) and the path label follows")
    need("_showFallbackBanner() {" in js and "} else if (this._fallbackActive()) {" in js
         and "Workflow folder missing here" in js and "_renderGridCore" not in js,
         "F3 the note sits in the node's status line (node width, ellipsis); renderGrid stays the one grid builder")
    need("if (this._fallbackBanner) { this._fallbackBanner = null;" in js,
         "F3 picking a file takes the note away")

    # F4 (the Palette node's message) is not part of the public build.

    spec = importlib.util.spec_from_file_location("wf_hygiene_v1247", os.path.join(ROOT, "tools", "wf_hygiene.py"))
    H = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(H)
    wf = {"nodes": [{"properties": {"ph_media_browse": "/home/claude/srv/ComfyUI/input"}},
                    {"widgets_values": ["D:\\COMFYUI_2026\\ComfyUI\\input", "a calm sea"]}]}
    f = H.findings(wf)
    need(len(f) == 1 and f[0][0].endswith("ph_media_browse"), "F5 a bench path is found, a user path is not (%r)" % (f,))
    ex = os.path.join(ROOT, "example_workflows")
    hits = []
    for fn in sorted(os.listdir(ex)):
        if fn.endswith(".json"):
            hits += H.check_file(os.path.join(ex, fn))
    need(not hits, "F5 example_workflows carry no bench path (%d)" % len(hits))

    raw = open(os.path.abspath(__file__), "rb").read()
    need(all(c < 128 for c in raw), "F6 %s is pure ASCII" % os.path.basename(__file__))
    if FAILS:
        print("[%s] FAIL -- %d" % (NAME, len(FAILS)))
        return 1
    print("[%s] PASS -- a foreign folder falls back to input; Palette names what is there; workflows leave clean" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
