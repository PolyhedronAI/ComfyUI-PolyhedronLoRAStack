# -*- coding: ascii -*-
"""Guard v388 -- the public cut of 17.10.2026: what came over from the development tree, and what stayed out.

  M1  the media route fixes of the audit (dev v1154 / v1155) are in ph_media_routes.py:
      a drive root is "within" itself; /uls/media/proc_count stands behind the LAN lock;
      /uls/media/file answers with FileResponse (no whole-file read on the event loop);
      thumbnails, sequence build and sequence delete run in worker threads;
      the opencv hint names no package-install command (registry hygiene)
      and the Stack's LoRA preview video / LoRA list (FileResponse, worker thread)
  M2  the TAE install refuses while downloads are locked (ph_weights.download_locked)
  B1  the Stack's group Bake stays out of this build: uls_node.js and uls_stack_dom.js hand no
      Bake callback in, and nothing calls /uls/bake_group
  T1  the Cine Timeline is registered, and every module its import hull needs ships with it
  T2  the timeline's carrier modules register none of their own nodes (v392: Continue is a node now)
  S   the new modules are ASCII
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NAME = "test_v388_public_cut"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def _fn(src, name):
    t = ast.parse(src)
    for n in t.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return ast.get_source_segment(src, n)
    return ""


MR = _read("nodes", "ph_media_routes.py")
w = _fn(MR, "_within")
_need("base = rp if rp.endswith(os.sep) else rp + os.sep" in w and "rc.startswith(base)" in w,
      "M1 _within: a drive root is within itself (v1154)")
pc = _fn(MR, "handle_media_proc_count")
_need("_deny = _lan_deny(request)" in pc and "return _deny" in pc, "M1 /uls/media/proc_count is behind the LAN lock (v1154)")
mf = _fn(MR, "handle_media_file")
_need("web.FileResponse(path" in mf and "f.read(" not in mf, "M1 /uls/media/file streams with FileResponse (v1155)")
_need("await asyncio.to_thread(_thumb_bytes, path, kind)" in _fn(MR, "handle_media_thumb") and _fn(MR, "_thumb_bytes"),
      "M1 thumbnails decode in a worker thread (v1155)")
_need("await asyncio.to_thread(_build)" in _fn(MR, "handle_media_seq_build"), "M1 a sequence is built in a worker thread (v1155)")
_need("await asyncio.to_thread(shutil.rmtree, target)" in _fn(MR, "handle_media_seq_delete"), "M1 a sequence is deleted in a worker thread (v1155)")
_need("pip install" not in _fn(MR, "_hint_cv2_once"), "M1 the opencv hint names no install command")

UR = _read("nodes", "uls_routes.py")
_need("web.FileResponse(vid_path" in _fn(UR, "handle_preview_video") and "f.read(" not in _fn(UR, "handle_preview_video"),
      "M1 the LoRA preview video streams with FileResponse (v1155)")
_need("await asyncio.to_thread(_scan)" in _fn(UR, "handle_list"), "M1 the LoRA list is built in a worker thread (v1155)")

SR = _read("nodes", "ph_sampler_routes.py")
ti = _fn(SR, "handle_sampler_tae_install")
_need("_PW.download_locked()" in ti and "status=403" in ti, "M2 the TAE install refuses while downloads are locked")

UN, SD = _read("web", "js", "uls_node.js"), _read("web", "js", "uls_stack_dom.js")
# the popup keeps its optional onBake parameter (it renders the Bake section only when a callback is handed in);
# what must not exist is the callback and its route
_need("/uls/bake_group" not in UN + SD and "const onBake" not in UN and "h.onBake" not in UN + SD,
      "B1 no group Bake in the public Stack frontends (no callback, no route)")
_need("uls_bake" not in "".join(_read("nodes", f) for f in os.listdir(os.path.join(ROOT, "nodes")) if f.endswith(".py")),
      "B1 no module imports the Bake machinery")

INIT = _read("__init__.py")
_need('NODE_CLASS_MAPPINGS["ULSCineTimeline"] = ULSCineTimeline' in INIT, "T1 the Cine Timeline is registered")
miss = []
for d, _, fs in os.walk(os.path.join(ROOT, "nodes")):
    for f in fs:
        if not f.endswith(".py"):
            continue
        p = os.path.join(d, f)
        for n in ast.walk(ast.parse(open(p, encoding="utf-8").read())):
            if isinstance(n, ast.ImportFrom) and n.level >= 1:
                base = d
                for _ in range(n.level - 1):
                    base = os.path.dirname(base)
                for m in ([n.module] if n.module else [a.name for a in n.names]):
                    q = os.path.join(base, *m.split("."))
                    if not (os.path.exists(q + ".py") or os.path.isdir(q)):
                        miss.append("%s -> %s" % (os.path.relpath(p, ROOT), m))
for d, _, fs in os.walk(os.path.join(ROOT, "web")):
    for f in fs:
        if f.endswith(".js"):
            p = os.path.join(d, f)
            for m in re.findall(r'from\s+"(\./[^"]+)"', open(p, encoding="utf-8").read()):
                if not os.path.exists(os.path.normpath(os.path.join(d, m))):
                    miss.append("%s -> %s" % (os.path.relpath(p, ROOT), m))
_need(not miss, "T1 the import hull is closed (every relative import ships) %r" % miss[:5])

registered = set(re.findall(r'NODE_CLASS_MAPPINGS\["(\w+)"\]', INIT))
for mod, classes in (("ph_cine_chain", ("ULSClipStore", "ULSClipChainSource", "ULSSeamCheck", "ULSSeamFix", "ULSClipAssemble")),
                     ("ph_rough_cut_gate", ("ULSRoughCutGate",))):
    # v392 AMENDED (declared, Frank 22.10.: the fox templates' H3 chain goes public): ph_minimax_continue
    # registers ULSMiniMaxContinue now -- it left this carrier list; the other carriers stay carriers
    _need(not (set(classes) & registered) and "from .nodes.%s import" % mod not in INIT,
          "T2 %s ships as a carrier only (none of its nodes registered)" % mod)

for f in ("ph_basics_routes.py", "ph_world_weights.py", "ph_cine_timeline.py", "uls_routes.py", "ph_media_routes.py"):
    _need(open(os.path.join(ROOT, "nodes", f), "rb").read().isascii(), "S  nodes/%s is ASCII" % f)
_need(open(__file__, "rb").read().isascii(), "S  this guard is ASCII")

print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
