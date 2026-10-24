# -*- coding: ascii -*-
"""Guard v1220 -- the UI freeze (Frank 23.10.: the frontend hung after loading a workflow with many model lines; gone
after a reload) and the carry without the browser (audit 24.10. #13).

  F1  GET /pls/weights/status computes in a thread: a slow status() leaves the event loop free (measured: a ticker
      keeps ticking while it runs).
  F2  find_file: outside the status route every call walks (a file written right after a miss is found at once --
      runs, conversions, installs); inside cached_finds() a hit is kept while the file is there, a miss for
      FIND_MISS_TTL; forget_found() and every write of the pack (ensure_weights, an install) drop them. Measured on a
      tree of 6000 files: the kept answer against the walk.
  F3  the frontend: one status request per set for everything that asks together (STATUS_KEEP), forgotten after an
      install, a failure not kept; refreshModelLists folds a burst into ONE refreshComboInNodes; the model lines
      check once per burst (schedule, LINE_WAIT) -- created + configured + callbacks + mode changes.
  F4  the carry without the browser: day_clip finds scene k's newest clip in the day folder by the clip_prefix name
      (<project>_scene_<k>), never another project's or scene's; build() lists it before it starts free.
  S   ASCII.

Script-style: exit 0 = pass.
"""
import asyncio
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


TD = tempfile.mkdtemp(prefix="v1220_")
MODELS, OUT_D, IN_D = os.path.join(TD, "models"), os.path.join(TD, "output"), os.path.join(TD, "input")
for d in (MODELS, OUT_D, IN_D):
    os.makedirs(d)
fp = types.ModuleType("folder_paths")
fp.models_dir = MODELS
fp.folder_names_and_paths = {}
fp.get_folder_paths = lambda name: [os.path.join(MODELS, name)]
fp.get_output_directory = lambda: OUT_D
fp.get_input_directory = lambda: IN_D
fp.get_temp_directory = lambda: os.path.join(TD, "temp")
sys.modules["folder_paths"] = fp
PKG = "pls1220"
pkg = types.ModuleType(PKG)
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules[PKG] = pkg
PW = importlib.import_module(PKG + ".ph_weights")
WW = importlib.import_module(PKG + ".ph_world_weights")

# ------------------------------------------------------------------ F1 the status route off the event loop
class _Req(object):
    query = {"set": ""}


orig_status = WW.status
WW.status = lambda s=None: (time.sleep(0.6), [])[1]
ticks = []


async def _main():
    async def ticker():
        for _ in range(12):
            ticks.append(time.monotonic())
            await asyncio.sleep(0.05)
    t = asyncio.ensure_future(ticker())
    t0 = time.monotonic()
    r = await WW.handle_status(_Req())
    await t
    return time.monotonic() - t0, r


try:
    dt, resp = asyncio.run(_main())
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    _need(dt >= 0.55 and max(gaps) < 0.3 and json.loads(resp.text)["ok"],
          "F1 a status taking 0.6 s leaves the event loop free (longest tick gap %.2f s)" % max(gaps))
except Exception as e:
    _need(False, "F1 %s: %s" % (type(e).__name__, e))
WW.status = orig_status
src_ww = (ROOT / "nodes" / "ph_world_weights.py").read_text(encoding="utf-8")
_need("await asyncio.to_thread(_status_kept, want or None)" in src_ww and "with PW.cached_finds():" in src_ww,
      "F1 the route runs status() in a thread, inside cached_finds()")

# ------------------------------------------------------------------ F2 find_file
root = os.path.join(MODELS, "f2")
for i in range(60):
    sub = os.path.join(root, "d%02d" % i)
    os.makedirs(sub)
    for j in range(100):
        open(os.path.join(sub, "x%03d.bin" % j), "wb").close()
PW.model_dirs = lambda folder: [os.path.join(MODELS, folder)]
walks = [0]
real_walk = os.walk


def counting_walk(*a, **k):
    walks[0] += 1
    return real_walk(*a, **k)


os.walk = counting_walk
try:
    PW.forget_found()
    _need(PW.find_file("f2", "late.safetensors") is None, "F2 (a miss)")
    open(os.path.join(root, "d59", "late.safetensors"), "wb").close()
    _need(PW.find_file("f2", "late.safetensors") is not None,
          "F2 outside the status route: a file written right after a miss is found at once (runs, conversions)")
    os.remove(os.path.join(root, "d59", "late.safetensors"))
    t0 = time.perf_counter()
    PW.find_file("f2", "deep.bin")
    t_walk = time.perf_counter() - t0
    with PW.cached_finds():
        w0 = walks[0]
        PW.find_file("f2", "nothing.bin")
        t1 = time.perf_counter()
        for _ in range(50):
            PW.find_file("f2", "nothing.bin")
        t_kept = (time.perf_counter() - t1) / 50
        _need(walks[0] == w0 + 1, "F2 inside cached_finds: a miss walks once, then is kept (%d walks for 51 asks)" % (walks[0] - w0))
        _need(t_kept < t_walk / 20, "F2 measured on 6000 files: walk %.1f ms, kept answer %.3f ms" % (t_walk * 1e3, t_kept * 1e3))
        open(os.path.join(root, "d10", "nothing.bin"), "wb").close()
        _need(PW.find_file("f2", "nothing.bin") is None, "F2 a kept miss holds for FIND_MISS_TTL (a file copied in by hand)")
        PW.forget_found()
        hit = PW.find_file("f2", "nothing.bin")
        _need(hit is not None and hit.endswith(os.path.join("d10", "nothing.bin")), "F2 forget_found drops the kept answers")
        w1 = walks[0]
        PW.find_file("f2", "nothing.bin")
        _need(walks[0] == w1, "F2 a kept hit is not walked again")
        os.remove(hit)
        _need(PW.find_file("f2", "nothing.bin") is None, "F2 a kept hit whose file went is walked again (gone)")
    _need(PW.FIND_MISS_TTL <= 5.0, "F2 a miss is kept at most 5 s")
finally:
    os.walk = real_walk
src_pw = (ROOT / "nodes" / "ph_weights.py").read_text(encoding="utf-8")
_need("        os.replace(part, dest)\n        forget_found()" in src_pw, "F2 ensure_weights forgets the kept answers when it writes")
_need(src_ww.count("PW.forget_found()") >= 2, "F2 an install forgets them before and after")

# ------------------------------------------------------------------ F3 the frontend
JS_D = (ROOT / "web" / "js" / "ph_model_door.js").read_text(encoding="utf-8")
JS_W = (ROOT / "web" / "js" / "ph_world_weights.js").read_text(encoding="utf-8")
if shutil.which("node") is None:
    _need(False, "F3 needs node.js")
else:
    tmp = Path(tempfile.mkdtemp(prefix="v1220js_"))
    body = (JS_D.replace('import { app } from "../../scripts/app.js";', "export const app = { refreshComboInNodes() { globalThis.REFRESH = (globalThis.REFRESH || 0) + 1; }, registerExtension() {} };")
                .replace('import { api } from "../../scripts/api.js";', "export const api = { fetchApi: (u) => globalThis.FETCH(u), addEventListener() {} };"))
    (tmp / "m.mjs").write_text(body, encoding="utf-8")
    (tmp / "h.mjs").write_text(r"""
globalThis.document = { createElement: () => ({ style: {}, appendChild() {}, append() {}, addEventListener() {} }), head: { appendChild() {} }, body: { appendChild() {} } };
let n = 0, fail = false;
globalThis.FETCH = async (u) => { n++; if (fail) throw new Error("down"); return { json: async () => ({ ok: true, models: [{ name: u }] }) }; };
const T = await import("./m.mjs");
const out = {};
const rs = await Promise.all([1, 2, 3, 4, 5].map(() => T.modelStatus("rife")));
out.together = n; out.same = rs.every((r) => r[0].name === rs[0][0].name);
await T.modelStatus("depth"); out.otherSet = n;
T.forgetStatus(); await T.modelStatus("rife"); out.afterForget = n;
fail = true; T.forgetStatus(); try { await T.modelStatus("x"); } catch (e) { out.failed = true; }
fail = false; await new Promise((r) => setTimeout(r, 0)); await T.modelStatus("x"); out.afterFail = n;
for (let i = 0; i < 5; i++) T.refreshModelLists();
out.refreshNow = globalThis.REFRESH || 0;
await new Promise((r) => setTimeout(r, T.REFRESH_WAIT + 150));
out.refreshAfter = globalThis.REFRESH || 0;
out.keep = T.STATUS_KEEP;
console.log(JSON.stringify(out));
""", encoding="utf-8")
    r = subprocess.run(["node", str(tmp / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(tmp, ignore_errors=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if r.returncode != 0 or not line:
        _need(False, "F3 node.js run failed: %s" % r.stderr[-500:])
    else:
        o = json.loads(line[-1])
        _need(o["together"] == 1 and o["same"], "F3 five lines asking for one set together: ONE request")
        _need(o["otherSet"] == 2 and o["afterForget"] == 3, "F3 another set asks on its own; an install (forgetStatus) asks again")
        _need(o.get("failed") and o["afterFail"] == 5, "F3 a failed status is not kept -- the next ask goes to the server")
        _need(o["refreshNow"] == 0 and o["refreshAfter"] == 1, "F3 five list refreshes in a burst: ONE refreshComboInNodes, after REFRESH_WAIT")
        _need(0 < o["keep"] <= 2000, "F3 STATUS_KEEP %d ms" % o["keep"])
_need("forgetStatus();                  // v1220" in JS_D, "F3 a finished install forgets the kept status")
_need(JS_W.count("schedule(node, \"pick\");") == 2 and JS_W.count("schedule(node, \"set\");") == 2
      and "watchMode(node, (n) => schedule(n, kind));" in JS_W and "schedule(this, kind);" in JS_W
      and "export const LINE_WAIT = 120;" in JS_W,
      "F3 a model line is checked once per burst: created, configured, callbacks and mode changes go through schedule")

# ------------------------------------------------------------------ F4 the carry without the browser
TL = importlib.import_module(PKG + ".ph_cine_timeline")
T = importlib.import_module(PKG + ".cine_timeline")
day = os.path.join(OUT_D, time.strftime("%Y-%m-%d"))
os.makedirs(day)
files = [("PH_100000_flf_sit2_scene_1_00001_.mp4", 100), ("PH_110000_flf_sit2_scene_1_00001_.mp4", 200),
         ("PH_120000_other_scene_1_00001_.mp4", 300), ("PH_130000_flf_sit2_scene_2_00001_.mp4", 400),
         ("PH_140000_flf_sit2_scene_1_00001_.png", 500), ("PH_090000_flf_sit2_whole_00001_.mp4", 50)]
now = time.time()
for nm, age in files:
    p = os.path.join(day, nm)
    open(p, "wb").close()
    os.utime(p, (now - 1000 + age, now - 1000 + age))
m = T.load(json.dumps({"project": "flf_sit2", "seconds": 8.0, "shots": [{"at": 0, "text": "a"}, {"at": 4, "text": "b", "scene": True, "carry": True}]}))
got = TL.day_clip(m, 1)
_need(got and os.path.basename(got) == "PH_110000_flf_sit2_scene_1_00001_.mp4",
      "F4 day_clip: scene 1's newest clip of THIS project (not another project's, scene's, or a picture) %s" % (got and os.path.basename(got)))
_need(os.path.basename(TL.day_clip(m, 0) or "") == "PH_090000_flf_sit2_whole_00001_.mp4" and TL.day_clip(m, 3) is None,
      "F4 day_clip: _whole for place 0; nothing for a scene with no clip")
src_tl = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
i_fb, i_wait = src_tl.find("found = day_clip(model, k0 - 1)"), src_tl.find("for _i in range(REGISTER_WAIT_TRIES):")
_need(0 < i_fb < i_wait and "A.register(project_folder(model.get(\"project\", \"\")), k0 - 1, found)" in src_tl,
      "F4 build(): a scene that carries looks in the day folder (and lists the clip) before it waits and starts free")

for f in ("nodes/ph_weights.py", "nodes/ph_world_weights.py", "nodes/ph_cine_timeline.py", "web/js/ph_model_door.js",
          "web/js/ph_world_weights.js", "tests/test_v1220_ui_freeze.py"):
    _need(all(b < 128 for b in (ROOT / f).read_bytes()), "S  ASCII %s" % f)

shutil.rmtree(TD, ignore_errors=True)
if FAILS:
    print("test_v1220_ui_freeze: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1220_ui_freeze: all checks passed")
