# -*- coding: ascii -*-
"""Guard v1089 -- the small finds of the Timeline audit (C8 / C9 / C10).

  K1  C8: build() decodes only the @clip anchors the run uses -- active scene 2 decodes
      the anchor of scene 2, not the one of scene 1 (which still counts for the check).
  K2  C9: _savePieces files the cut under the MODEL's render kind, not the last plan's
      (driven in node: model on 'scene', a stale plan saying whole -> 'scenes').
  K3  C10: V1's reset asks first -- declined, the edit stays (driven).
  K4  C10: the pick is the Save that ran LAST, even when its register answer comes back
      first (driven with out-of-order answers).
  K5  C10: the unused state fields are gone; concat_videos' docstring says what it returns.
  S   ASCII.

Script-style: exit 0 = pass.
"""
import importlib
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1089_timeline_small"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


# ------------------------------------------------------------------ K1
TMP = Path(tempfile.mkdtemp(prefix="v1089_"))
for d in ("output/fx", "input", "temp"):
    (TMP / d).mkdir(parents=True, exist_ok=True)


class _NT:
    is_nested = True


comfy = types.ModuleType("comfy"); nt = types.ModuleType("comfy.nested_tensor"); nt.NestedTensor = _NT; comfy.nested_tensor = nt
capi = types.ModuleType("comfy_api"); ii = types.ModuleType("comfy_api.input_impl"); ii.VideoFromComponents = object
ut = types.ModuleType("comfy_api.util"); ut.VideoComponents = object
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: str(TMP / "output"); fp.get_input_directory = lambda: str(TMP / "input"); fp.get_temp_directory = lambda: str(TMP / "temp")
sys.modules.update({"comfy": comfy, "comfy.nested_tensor": nt, "comfy_api": capi, "comfy_api.input_impl": ii, "comfy_api.util": ut, "folder_paths": fp})
pkg = types.ModuleType("plsn1089")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1089"] = pkg
TL = importlib.import_module("plsn1089.ph_cine_timeline")
import torch  # noqa: E402
decoded, saved = [], []
TL.clip_anchor_paths = lambda m: {"@clip:1:last": "/x/fx/scene_1.mp4", "@clip:5:first": "/x/fx/scene_5.mp4"}
TL.frame_index = lambda path, idx: (decoded.append((path, idx)), torch.zeros((1, 8, 8, 3)))[1]
TL.save_frame = lambda model, src, img: (saved.append(src), None)[1]
TL.carry_source = lambda model, k: None
model = {"seconds": 6.0, "project": "fx", "render": "scene", "active": 2, "carry_mode": "keyframe",
         "shots": [{"at": 0, "text": "a"}, {"at": 2.5, "scene": True, "cut": "the camera cuts to", "text": "b"}],
         "anchors": [{"src": "@clip:5:first", "at": 1.0}, {"src": "@clip:1:last", "at": 2.5}]}
try:
    out = TL.ULSCineTimeline().build(json.dumps(model))
    err = None
except Exception as e:      # pragma: no cover
    out, err = None, e
_need(err is None and decoded == [("/x/fx/scene_1.mp4", 59)] and saved == ["@clip:1:last"],
      "K1 only the active scene's @clip anchor is decoded and saved (frame 60 of scene 1): %r %r %r" % (decoded, saved, err))
shutil.rmtree(TMP, ignore_errors=True)

# ------------------------------------------------------------------ K2-K4 (node)
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
tmpd = Path(tempfile.mkdtemp(prefix="v1089js_"))
h = js.replace('import { app } from "../../scripts/app.js";', "const app = globalThis.__app;")
h = h.replace('import { api } from "../../scripts/api.js";', "const api = globalThis.__api;")
h = h.replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}; const refit = () => {};")
h = h.replace("function _render(node) {", "function _render(node) { if (globalThis.__noRender) return;", 1)
h = h.replace("async function _stack(node) {", "async function _stack(node) { if (globalThis.__noStack) return;", 1)
h += "\nexport { NODE, _savePieces, _resetLayer };\n"
(tmpd / "tl.mjs").write_text(h, encoding="utf-8")
(tmpd / "drive.mjs").write_text(r'''
const listeners = {}; let ext = null; const reg = []; let confirmAnswer = false;
globalThis.window = { addEventListener() {}, confirm: () => confirmAnswer };
globalThis.requestAnimationFrame = () => 0; globalThis.__noRender = true; globalThis.__noStack = true;
const later = (ms, v) => new Promise((res) => setTimeout(() => res(v), ms));
globalThis.__app = { graph: { _nodes: [], links: new Map() }, registerExtension(e) { ext = e; }, queuePrompt: async () => true };
globalThis.__api = { addEventListener: (n, f) => { (listeners[n] = listeners[n] || []).push(f); }, apiURL: (u) => u,
  fetchApi: async (url, opt) => {
    if (url.startsWith("/pls/cine/register")) { const b = JSON.parse(opt.body); reg.push(b);
      const rel = b.file.subfolder + "/" + b.file.filename;
      const wait = b.file.filename.startsWith("first") ? 60 : 0;      // the FIRST Save's answer comes back LAST
      return { json: () => later(wait, { rel }) }; }
    if (url.startsWith("/pls/cine/probe")) return { json: async () => ({ name: "x.mp4", frames: 48, fps: 24, audio: true }) };
    return { json: async () => ({ error: "stub" }) }; } };
const M = await import("./tl.mjs");
await ext.setup();
const fire = async (n, detail) => { for (const f of listeners[n] || []) await f({ detail }); };
const mk = (model, edit) => ({ id: 7, type: M.NODE, properties: { [M.EDIT_KEY]: edit || {} }, outputs: [{ links: [] }],
  widgets: [{ name: "timeline", value: JSON.stringify(Object.assign({}, M.DEFAULT, model)) }],
  _plsTl: { state: { textContent: "" }, player: { t: 0 }, active: "base", recent: [], plan: { whole: true } } });
const out = {};
// K2: model on scene, a stale plan saying whole
let n = mk({ render: "scene", project: "fx" });
M._savePieces(n, [{ k: 1, a: 0, b: 40, speed: 1 }]);
out.k2 = Object.keys(n.properties[M.EDIT_KEY].pieces || {});
// K3: V1 reset declined, then accepted
n = mk({ project: "fx" }, { pieces: { whole: [{ k: 0, a: 0, b: 50, speed: 1 }] }, keep_tail: true });
M._resetLayer(n, { id: "base", lock: false }, null);
out.k3a = Object.keys(n.properties[M.EDIT_KEY]).sort();
confirmAnswer = true;
M._resetLayer(n, { id: "base", lock: false }, null);
out.k3b = Object.keys(n.properties[M.EDIT_KEY]).sort();
// K4: two Saves, the first one's answer comes back last
n = mk({ project: "fx" }); globalThis.__app.graph._nodes.push(n);
const save = (name) => ({ images: [{ filename: name, subfolder: "d", type: "output", format: "video/h264-mp4" }] });
await fire("executed", { prompt_id: "q1", node: "7", output: { pls_timeline: [{ clip_prefix: "fx/whole", scene: 0 }] } });
await fire("executed", { prompt_id: "q1", node: "11", output: save("first_00001_.mp4") });
await fire("executed", { prompt_id: "q1", node: "12", output: save("second_00001_.mp4") });
await fire("execution_success", { prompt_id: "q1" });
out.k4 = { primary: reg.filter((b) => b.primary).map((b) => b.file.filename), takes: n.properties[M.EDIT_KEY].takes };
console.log(JSON.stringify(out));
process.exit(0);
''', encoding="utf-8")
r = subprocess.run(["node", str(tmpd / "drive.mjs")], capture_output=True, text=True, timeout=120, cwd=str(tmpd))
shutil.rmtree(tmpd, ignore_errors=True)
ok = r.returncode == 0
if not ok:
    print(r.stderr[-1500:])
o = json.loads(r.stdout.strip().splitlines()[-1]) if ok else {}
_need(ok and o.get("k2") == ["scenes"], "K2 the cut is filed under the model's kind (scenes), not the stale plan's: %r" % (o.get("k2"),))
_need(ok and o.get("k3a") == ["keep_tail", "pieces"] and o.get("k3b") == [], "K3 V1's reset asks: declined keeps, accepted clears: %r / %r" % (o.get("k3a"), o.get("k3b")))
_need(ok and (o.get("k4") or {}).get("primary") == ["second_00001_.mp4"] and (o.get("k4") or {}).get("takes") == {"0": "d/second_00001_.mp4"},
      "K4 the pick is the Save that ran last, whatever answered first: %r" % (o.get("k4"),))

# ------------------------------------------------------------------ K5
_need("zoomInfo: null" not in js and "film: null, stackData: null" not in js, "K5 the unused state fields are gone")
cc = (ROOT / "nodes" / "cine_clip.py").read_text(encoding="utf-8")
_need("(frames_total_estimate, seconds)" not in cc and "the total seconds" in cc, "K5 concat_videos says it returns the seconds")
bad = [f for f in ("web/js/ph_cine_timeline.js", "nodes/ph_cine_timeline.py", "nodes/cine_clip.py", "tests/test_v1089_timeline_small.py")
       if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not bad, "S ASCII only %r" % bad)
if FAILS:
    print("%s: %d FAILURE(S)" % (NAME, len(FAILS)))
    sys.exit(1)
print("%s: PASS (K1-K5, S)" % NAME)
