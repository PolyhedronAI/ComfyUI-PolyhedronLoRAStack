# -*- coding: ascii -*-
"""Guard v1088 -- the rest of the Timeline audit (B3 / B6 / B7 / B9 / B12).

The extension's own listeners are driven in node with stubbed routes and a real graph
shape (nodes with outputs, links), in the event orders a Core sends.

  T1  B3: two timelines in one graph (projects fa / fb), each with its own chain to a Save:
      each Save's clip is registered under ITS timeline's project and pins ITS V1 --
      in both orders (A, B, SaveA, SaveB) and (A, SaveA, B, SaveB).
  T2  B3: a Save that hangs below neither timeline registers nothing, the status says so on
      both nodes (no guess).
  T3  B9: a cached timeline whose last run was another scene: the Save is HELD until the
      replayed "executed" names the real scene, then registered under it; the pin lands on
      that scene.
  T4  B9: an older server that replays nothing: the held Save is flushed at the success with
      the provisional identity (as before v1088).
  T5  B6: seenKeyOf is per graph and node.
  T6  B7: the locate request carries the Media Loader's shared recent folders; a found file's
      folder is remembered in both lists; not found says how to make the next drop read
      where it lies.
  T7  the pure helpers: downstream over Map and object links, attributeSave with one run.
  S   ASCII.

Script-style: exit 0 = pass.
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1088_timeline_rest"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
tmpd = Path(tempfile.mkdtemp(prefix="v1088js_"))
h = js.replace('import { app } from "../../scripts/app.js";', "const app = globalThis.__app;")
h = h.replace('import { api } from "../../scripts/api.js";', "const api = globalThis.__api;")
h = h.replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}; const refit = () => {};")
h = h.replace("function _render(node) {", "function _render(node) { if (globalThis.__noRender) return;", 1)
h = h.replace("async function _stack(node) {", "async function _stack(node) { if (globalThis.__noStack) return;", 1)
h += "\nexport { NODE, _dropInto };\n"
(tmpd / "tl.mjs").write_text(h, encoding="utf-8")
(tmpd / "drive.mjs").write_text(r'''
const listeners = {}; let ext = null; let reg = []; let locateBodies = []; let locateAnswer = { ok: false, reason: "not_found" };
const store = { ph_media_recent_folders: JSON.stringify(["D:/clips/shared"]) };
globalThis.localStorage = { getItem: (k) => (k in store ? store[k] : null), setItem: (k, v) => { store[k] = String(v); } };
globalThis.window = { addEventListener() {} };
globalThis.requestAnimationFrame = () => 0; globalThis.__noRender = true; globalThis.__noStack = true;
globalThis.__app = { graph: { _nodes: [], links: new Map() }, registerExtension(e) { ext = e; }, queuePrompt: async () => true };
globalThis.__api = { addEventListener: (n, f) => { (listeners[n] = listeners[n] || []).push(f); }, apiURL: (u) => u,
  fetchApi: async (url, opt) => {
    if (url.startsWith("/pls/cine/register")) { const b = JSON.parse(opt.body); reg.push(b); return { json: async () => ({ rel: b.file.subfolder + "/" + b.file.filename }) }; }
    if (url.startsWith("/pls/cine/probe")) return { json: async () => ({ name: "x.mp4", frames: 48, fps: 24, audio: true }) };
    if (url.startsWith("/uls/media/locate")) { locateBodies.push(JSON.parse(opt.body)); return { json: async () => locateAnswer }; }
    if (url.startsWith("/pls/cine/media_dir")) return { json: async () => ({ dir: "/in", rel: "cine_media/fx" }) };
    if (url.startsWith("/uls/media/upload")) return { json: async () => ({ ok: true, names: ["x.mp4"] }) };
    return { json: async () => ({ error: "stub" }) }; } };
const M = await import("./tl.mjs");
await ext.setup();
const fire = async (n, detail) => { for (const f of listeners[n] || []) await f({ detail }); };
const G = globalThis.__app.graph;
let lid = 1;
const plain = (id) => ({ id, type: "Other", outputs: [{ links: [] }] });
const tl = (id, project, extra) => ({ id, type: M.NODE, properties: { [M.EDIT_KEY]: {} }, outputs: [{ links: [] }],
  widgets: [{ name: "timeline", value: JSON.stringify(Object.assign({}, M.DEFAULT, { project }, extra || {})) }],
  _plsTl: { state: { textContent: "" }, player: { t: 0 }, active: "base", recent: [] } });
const wire = (a, b) => { const id = lid++; a.outputs[0].links.push(id); G.links.set(id, { id, origin_id: a.id, target_id: b.id }); };
const reset = () => { G._nodes.length = 0; G.links.clear(); reg = []; };
const save = (name) => ({ images: [{ filename: name, subfolder: "2026-10-10", type: "output", format: "video/h264-mp4" }] });
const out = {};
// T1 order 1
reset();
let A = tl(7, "fa"), B = tl(8, "fb"), X = plain(10), SA = plain(11), Y = plain(12), SB = plain(13);
G._nodes.push(A, B, X, SA, Y, SB); wire(A, X); wire(X, SA); wire(B, Y); wire(Y, SB);
await fire("executed", { prompt_id: "p1", node: "7", output: { pls_timeline: [{ clip_prefix: "fa/whole", scene: 0 }] } });
await fire("executed", { prompt_id: "p1", node: "8", output: { pls_timeline: [{ clip_prefix: "fb/whole", scene: 0 }] } });
await fire("executed", { prompt_id: "p1", node: "11", output: save("a_1.mp4") });
await fire("executed", { prompt_id: "p1", node: "13", output: save("b_1.mp4") });
await fire("execution_success", { prompt_id: "p1" });
out.t1a = { reg: reg.map((b) => (b.primary ? "P:" : "") + b.project + ":" + b.file.filename), A: A.properties[M.EDIT_KEY].takes, B: B.properties[M.EDIT_KEY].takes };
// T1 order 2
reset();
A = tl(7, "fa"); B = tl(8, "fb"); X = plain(10); SA = plain(11); Y = plain(12); SB = plain(13);
G._nodes.push(A, B, X, SA, Y, SB); wire(A, X); wire(X, SA); wire(B, Y); wire(Y, SB);
await fire("executed", { prompt_id: "p2", node: "7", output: { pls_timeline: [{ clip_prefix: "fa/whole", scene: 0 }] } });
await fire("executed", { prompt_id: "p2", node: "11", output: save("a_2.mp4") });
await fire("executed", { prompt_id: "p2", node: "8", output: { pls_timeline: [{ clip_prefix: "fb/whole", scene: 0 }] } });
await fire("executed", { prompt_id: "p2", node: "13", output: save("b_2.mp4") });
await fire("execution_success", { prompt_id: "p2" });
out.t1b = { reg: reg.map((b) => (b.primary ? "P:" : "") + b.project + ":" + b.file.filename), A: A.properties[M.EDIT_KEY].takes, B: B.properties[M.EDIT_KEY].takes };
// T2 a Save below neither
reset();
A = tl(7, "fa"); B = tl(8, "fb"); const Z = plain(20);
G._nodes.push(A, B, Z);
await fire("executed", { prompt_id: "p3", node: "7", output: { pls_timeline: [{ clip_prefix: "fa/whole", scene: 0 }] } });
await fire("executed", { prompt_id: "p3", node: "8", output: { pls_timeline: [{ clip_prefix: "fb/whole", scene: 0 }] } });
await fire("executed", { prompt_id: "p3", node: "20", output: save("z.mp4") });
await fire("execution_success", { prompt_id: "p3" });
out.t2 = { reg: reg.length, sa: A._plsTl.state.textContent, sb: B._plsTl.state.textContent };
// T3 cached with a stale last run, replay after the Save
reset();
const sc = { render: "scene", active: 1, shots: [{ at: 0, text: "a" }, { at: 2, scene: true, cut: "the camera cuts to", text: "b" }, { at: 4, scene: true, cut: "the camera cuts to", text: "c" }], seconds: 6 };
A = tl(7, "fs", sc); X = plain(10); SA = plain(11);
G._nodes.push(A, X, SA); wire(A, X); wire(X, SA);
A._plsLastRun = { project: "fs", scene: 3, clip_prefix: "fs/scene_3" };
await fire("execution_cached", { prompt_id: "p4", nodes: ["7"] });
await fire("executed", { prompt_id: "p4", node: "11", output: save("s_1.mp4") });
const heldBeforeReplay = reg.length;
await fire("executed", { prompt_id: "p4", node: "7", output: { pls_timeline: [{ clip_prefix: "fs/scene_1", scene: 1 }] } });
await fire("execution_success", { prompt_id: "p4" });
out.t3 = { heldBeforeReplay, reg: reg.map((b) => (b.primary ? "P:" : "") + b.scene + ":" + b.file.filename), takes: A.properties[M.EDIT_KEY].takes };
// T4 no replay (older server)
reset();
A = tl(7, "fs", sc); X = plain(10); SA = plain(11);
G._nodes.push(A, X, SA); wire(A, X); wire(X, SA);
A._plsLastRun = { project: "fs", scene: 2, clip_prefix: "fs/scene_2" };
await fire("execution_cached", { prompt_id: "p5", nodes: ["7"] });
await fire("executed", { prompt_id: "p5", node: "11", output: save("s_2.mp4") });
await fire("execution_success", { prompt_id: "p5" });
out.t4 = reg.map((b) => (b.primary ? "P:" : "") + b.scene + ":" + b.file.filename);
// T5
out.t5 = [M.seenKeyOf({ id: 7, graph: { id: "g1" } }), M.seenKeyOf({ id: 7, graph: { id: "g2" } }), M.seenKeyOf({ id: 7, graph: { id: "g1" } })];
// T6
reset();
A = tl(7, "fx"); G._nodes.push(A);
const F = (name) => new File(["x"], name);
await M._dropInto(A, "+new", { files: [F("clip.mp4")], getData: () => "" }, "pin", 0);
const t6a = { folders: locateBodies[0] && locateBodies[0].folders, note: A._plsTl.state.textContent };
locateAnswer = { ok: true, folder: "E:/renders/day1", file: "clip.mp4" };
await M._dropInto(A, "+new", { files: [F("clip.mp4")], getData: () => "" }, "pin", 0);
out.t6 = { a: t6a, recentNode: A._plsTl.recent, shared: JSON.parse(store.ph_media_recent_folders), merged: M.recentFolders(["E:/x", "D:/clips/shared"]) };
// T7
const g = { _nodes: [{ id: 1, outputs: [{ links: [5] }] }, { id: 2, outputs: [{ links: [6] }] }, { id: 3, outputs: [] }], links: { 5: [5, 1, 0, 2, 0, "X"], 6: { target_id: 3 } } };
out.t7 = { d: [...M.downstream(g, 1)].sort(), one: M.attributeSave({ tls: new Map([["9", { node: "9" }]]) }, "77", g).node };
console.log(JSON.stringify(out));
process.exit(0);
''', encoding="utf-8")
r = subprocess.run(["node", str(tmpd / "drive.mjs")], capture_output=True, text=True, timeout=120, cwd=str(tmpd))
shutil.rmtree(tmpd, ignore_errors=True)
ok = r.returncode == 0
if not ok:
    print(r.stderr[-1500:])
o = json.loads(r.stdout.strip().splitlines()[-1]) if ok else {}
for key, order, n in (("t1a", "A, B, SaveA, SaveB", 1), ("t1b", "A, SaveA, B, SaveB", 2)):
    v = o.get(key, {})
    want = sorted(["fa:a_%d.mp4" % n, "P:fa:a_%d.mp4" % n, "fb:b_%d.mp4" % n, "P:fb:b_%d.mp4" % n])
    _need(ok and sorted(v.get("reg") or []) == want and v.get("A") == {"0": "2026-10-10/a_%d.mp4" % n} and v.get("B") == {"0": "2026-10-10/b_%d.mp4" % n},
          "T1 two timelines (%s): each Save under its own timeline, each V1 pinned: %r" % (order, v))
_need(ok and o["t2"]["reg"] == 0 and "hangs below neither" in o["t2"]["sa"] and "hangs below neither" in o["t2"]["sb"],
      "T2 a Save below neither: nothing registered, said on both: %r" % (o.get("t2"),))
_need(ok and o["t3"]["heldBeforeReplay"] == 0 and o["t3"]["reg"] == ["1:s_1.mp4", "P:1:s_1.mp4"] and o["t3"]["takes"] == {"1": "2026-10-10/s_1.mp4"},
      "T3 cached with a stale last run: held until the replay, registered and pinned under scene 1: %r" % (o.get("t3"),))
_need(ok and o["t4"] == ["2:s_2.mp4", "P:2:s_2.mp4"], "T4 no replay: flushed at the success with the provisional identity: %r" % (o.get("t4"),))
_need(ok and o["t5"][0] != o["t5"][1] and o["t5"][0] == o["t5"][2], "T5 seenKeyOf per graph and node: %r" % (o.get("t5"),))
t6 = o.get("t6", {})
first = lambda lst: (lst or [None])[0]
_need(ok and t6["a"]["folders"] == ["D:/clips/shared"] and "open its folder once in a Media Loader" in t6["a"]["note"],
      "T6 the locate asks the shared recent folders; not found says how to fix it: %r" % (t6.get("a"),))
_need(ok and first(t6.get("recentNode")) == "E:/renders/day1" and (t6.get("shared") or [])[:2] == ["E:/renders/day1", "D:/clips/shared"]
      and t6["merged"] == ["E:/x", "D:/clips/shared", "E:/renders/day1"],
      "T6 a found folder is remembered in the node and the shared list; recentFolders merges without doubles: %r" % (t6,))
_need(ok and o["t7"]["d"] == ["2", "3"] and o["t7"]["one"] == "9", "T7 downstream over array and object links; one run takes every Save")
bad = [f for f in ("web/js/ph_cine_timeline.js", "tests/test_v1088_timeline_rest.py") if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not bad, "S ASCII only %r" % bad)
if FAILS:
    print("%s: %d FAILURE(S)" % (NAME, len(FAILS)))
    sys.exit(1)
print("%s: PASS (T1-T7, S)" % NAME)
