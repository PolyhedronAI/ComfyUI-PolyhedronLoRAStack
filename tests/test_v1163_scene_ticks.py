# -*- coding: ascii -*-
"""Guard v1163 -- ticked scenes render in one go; a new render can replace instead of opening a track.

Frank 18.10., the neon lane: scenes 2..5 had to be rendered again (with the Power Upscale of v1162) -- "Szene 2
bis 5 zusammen machen ... in einem Rutsch rendern", and "nicht new Track, sondern Replace".

  K1  skipOf / pickedOrder (pure): the node stores the UNticked scenes (node.properties.pls_cine_skip); the ticked
      ones come in order (and reversed for a front-of-queue run, as splitOrder); junk in the property is ignored
  K2  _renderAll queues exactly the ticked scenes, in order, each with render "scene" + active k, and gives the
      timeline its render choice back; none ticked -> nothing queued, said
  K3  the all-scenes Run (the split around app.queuePrompt) skips the unticked scenes; none ticked -> refused
  K4  the Scenes row: a tick per scene (2+ scenes) that never selects the bar; an unticked scene is dimmed;
      the button reads "render ticked scenes (n)" and the render choice "ticked scenes, one run each (n of m)"
  R1  "new render -> new track | replace": an explicit choice, mirroring autoOn; replace = _setAuto(false)
      (V1 shows the newest run of every scene, the files stay)
  P1  Space plays / pauses the ONE selected timeline (spaceNode, pure): never while typing (input, textarea,
      select, a button keeps its own Space), never with a modifier, never with two timelines selected or a
      bypassed one; one capture listener for the page; a held key does not re-toggle
  M1  the monitor: monitorBox fits the clip's aspect into the area (contain) -- portrait 9:16 in a wide box is
      as tall as the box, landscape in a narrow column as wide as the column; the plan sets the clip's aspect;
      every width check refits; the live mix and the exact preview draw contained, never stretched
  M2  resize: the work row's columns shrink down to their floors; the node has a floor width through
      computeSize (LiteGraph clamps a drag against it) -- still no onResize hook, no ResizeObserver
  S   ASCII of this guard
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1163_scene_ticks"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)
    return c


js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")
tmpd = Path(tempfile.mkdtemp(prefix="v1163js_"))
h = js.replace('import { app } from "../../scripts/app.js";', "const app = globalThis.__app;")
h = h.replace('import { api } from "../../scripts/api.js";', "const api = globalThis.__api;")
h = h.replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}; const refit = () => {};")
h = h.replace("function _render(node) {", "function _render(node) { if (globalThis.__noRender) return;", 1)
h += "\nexport { _renderAll };\n"
(tmpd / "tl.mjs").write_text(h, encoding="utf-8")
(tmpd / "drive.mjs").write_text(r'''
let node = null; const queued = []; let answers = [];
globalThis.__app = { graph: { _nodes: [] }, registerExtension() {}, queuePrompt: async () => {
  const m = JSON.parse(node.widgets[0].value); queued.push([m.render, m.active]); return answers.length ? answers.shift() : true; } };
globalThis.__api = { fetchApi: async () => ({ json: async () => ({}) }), apiURL: (u) => u };
globalThis.console.info = () => {};
globalThis.requestAnimationFrame = () => 0; globalThis.__noRender = true;
const M = await import("./tl.mjs");
const shots = [{ at: 0, text: "a" }, { at: 2, scene: true, cut: "the camera cuts to", text: "b" },
               { at: 4, scene: true, cut: "the camera cuts to", text: "c" }, { at: 6, scene: true, cut: "the camera cuts to", text: "d" },
               { at: 8, scene: true, cut: "the camera cuts to", text: "e" }];
const mk = (skip) => ({ properties: { [M.SKIP_KEY]: skip }, widgets: [{ name: "timeline", value: JSON.stringify(Object.assign({}, M.DEFAULT, { shots, seconds: 10, render: "all", active: 3 })) }],
  _plsTl: { state: { textContent: "" }, player: { t: 0 }, active: "base" } });
const out = {};
out.k1 = { skip: [...M.skipOf({ properties: { [M.SKIP_KEY]: [1, "3", 0, -2, "x", 9.7] } })].sort(),
           none: [...M.skipOf({})], order: M.pickedOrder(5, 0, new Set([1])), rev: M.pickedOrder(5, -1, new Set([1, 4])),
           all: M.pickedOrder(3, 0, null) };
node = mk([1]); await M._renderAll(node);
out.k2 = { queued: queued.splice(0), state: node._plsTl.state.textContent, back: JSON.parse(node.widgets[0].value) };
node = mk([1, 2, 3, 4, 5]); await M._renderAll(node);
out.k2_none = { queued: queued.splice(0), state: node._plsTl.state.textContent };
node = mk([1]); answers = [true, false]; await M._renderAll(node);
out.k2_refused = { queued: queued.splice(0), state: node._plsTl.state.textContent };
const tl = (id, mode) => ({ id, type: "ULSCineTimeline", _plsTl: {}, mode });
const sp = (o) => Object.assign({ code: "Space", key: " ", target: { tagName: "CANVAS" } }, o || {});
const A = tl(1, 0);
out.p1 = [M.spaceNode(sp(), [A]) === A, M.spaceNode(sp({ target: { tagName: "TEXTAREA" } }), [A]), M.spaceNode(sp({ target: { tagName: "INPUT" } }), [A]),
          M.spaceNode(sp({ target: { tagName: "SELECT" } }), [A]), M.spaceNode(sp({ target: { tagName: "BUTTON" } }), [A]),
          M.spaceNode(sp({ target: { tagName: "DIV", isContentEditable: true } }), [A]), M.spaceNode(sp({ ctrlKey: true }), [A]),
          M.spaceNode(sp(), [A, tl(2, 0)]), M.spaceNode(sp(), [tl(3, 4)]), M.spaceNode(sp(), [{ id: 9, type: "Other" }]),
          M.spaceNode(sp({ code: "KeyK", key: "k" }), [A]), M.spaceNode(sp(), [A, { id: 9, type: "Other" }]) === A];
out.m1 = [M.monitorBox(460, 276, 9 / 16), M.monitorBox(460, 276, 16 / 9), M.monitorBox(200, 276, 16 / 9), M.monitorBox(460, 276, 0), M.monitorBox(0, 276, 1)];
console.log(JSON.stringify(out));
process.exit(0);
''', encoding="utf-8")
r = subprocess.run(["node", str(tmpd / "drive.mjs")], capture_output=True, text=True, timeout=120)
ok = r.returncode == 0
if not ok:
    print(r.stderr[-1500:])
o = json.loads(r.stdout.strip().splitlines()[-1]) if ok else {}
shutil.rmtree(tmpd, ignore_errors=True)

# K1
_need(ok and o["k1"]["skip"] == [1, 3, 9] and o["k1"]["none"] == [], "K1 skipOf: the unticked scenes, junk ignored: %r" % (o.get("k1"),))
_need(ok and o["k1"]["order"] == [2, 3, 4, 5] and o["k1"]["rev"] == [5, 3, 2] and o["k1"]["all"] == [1, 2, 3],
      "K1 pickedOrder: the ticked scenes in order, reversed for a front run, all without a skip")
# K2
_need(ok and o["k2"]["queued"] == [["scene", 2], ["scene", 3], ["scene", 4], ["scene", 5]],
      "K2 render all with scene 1 unticked queues scenes 2..5, in order: %r" % (o.get("k2", {}).get("queued"),))
_need(ok and o["k2"]["back"]["render"] == "all" and o["k2"]["back"]["active"] == 3 and "4 scene run(s) queued (scenes 2, 3, 4, 5)" in o["k2"]["state"],
      "K2 the render choice comes back, the status names the scenes: %r" % (o.get("k2", {}).get("state"),))
_need(ok and o["k2_none"]["queued"] == [] and "no scene is ticked" in o["k2_none"]["state"], "K2 none ticked: nothing queued, said")
_need(ok and o["k2_refused"]["queued"] == [["scene", 2], ["scene", 3]] and "scene 3 was refused" in o["k2_refused"]["state"]
      and "scenes 3..5 not queued (1 queued)" in o["k2_refused"]["state"], "K2 a refused run stops and names the rest: %r" % (o.get("k2_refused"),))
# K3
i = js.find("const split = async function (number, batchCount)")
sp = js[i:i + 3000]
_need("const skip = skipOf(node);" in sp and "if (skip.has(k)) continue;" in sp
      and 'if (pickedOrder(n, number, skip).length === 0) { node._plsTl.state.textContent = "no scene is ticked -- nothing queued"; return false; }' in sp,
      "K3 the all-scenes Run skips unticked scenes, refuses when none is ticked")
# K4
_need('tick.dataset.pls = "scenetick";' in js and 'tick.addEventListener("pointerdown", (ev) => ev.stopPropagation());' in js
      and "_setSkip(node, k + 1, tick.checked); _render(node);" in js and 'if (!on) bar.style.opacity = "0.45";' in js
      and "if (sc.length > 1) {" in js, "K4 a tick per scene on the Scenes row, never selecting the bar, unticked dimmed")
_need('st.all.textContent = nTick < sc.length ? "render ticked scenes (" + nTick + ")" : "render all scenes";' in js
      and '"ticked scenes, one run each (" + nTick + " of " + sc.length + ")"' in js, "K4 the button and the render choice count the ticked scenes")
# R1
_need('export const AUTO_CHOICES = [["track", "new track"], ["replace", "replace"]];' in js
      and 'autoChk.addEventListener("change", () => _setAuto(node, autoChk.value === "track"));' in js
      and 'if (st.autoChk) st.autoChk.value = autoOn(editOf(node)) ? "track" : "replace";' in js
      and "replace: a new run REPLACES the clip of its scene in V1" in js, "R1 new track | replace: an explicit choice on _setAuto")
j = js.find("function _setAuto(node, on)")
_need(j > 0 and "} else { e[AUTO_KEY] = false; delete e.takes; }" in js[j:j + 600], "R1 replace drops the pins: V1 follows the newest run")

# P1
_need(ok and o["p1"] == [True, None, None, None, None, None, None, None, None, None, None, True],
      "P1 spaceNode: the one selected timeline, never while typing / with a modifier / two timelines / bypassed: %r" % (o.get("p1"),))
_need('if (!globalThis.__plsTlSpace && typeof document !== "undefined" && document.addEventListener) { globalThis.__plsTlSpace = true; document.addEventListener("keydown", _onSpace, true); }' in js
      and "if (!ev.repeat) { try { _playerToggle(n); }" in js and "ev.preventDefault(); ev.stopPropagation();" in js,
      "P1 one capture listener; a held key does not re-toggle; the page does not scroll")
# M1
_need(ok and o["m1"] == [[155, 276], [460, 259], [200, 113], [460, 259], [1, 1]],
      "M1 monitorBox: portrait as tall as the box, landscape as wide as the column, a missing aspect = 16:9: %r" % (o.get("m1"),))
_need("if (c0) { pl.ar = c0.w / c0.h; pl.boxW = null; }" in js and "_monitorFit(node);                                   // v1163" in js
      and "const [w, h] = monitorBox(aw, MON_AREA_H, pl.ar);" in js and 'pl.canvas.style.width = w + "px"; pl.canvas.style.height = h + "px";' in js,
      "M1 the plan sets the aspect, every width check refits, the canvas takes the box (CSS and pixels)")
_need("if (!own) _drawMix(ctx, W, H, pl.videos, D.plan, t, true);" in js and "else if (own.v && own.v.base) _drawMix(ctx, W, H, pl.videos, D.plan, t, true);" in js
      and "if (sc) ctx.drawImage(v, (W - v.videoWidth * sc) / 2" in js and "aspect-ratio:16/9" not in js,
      "M1 the live mix and the exact preview draw contained; no 16:9 frame left")
# M2
_need('flex:0 1 " + SND_W + "px;width:" + SND_W + "px;min-width:" + SND_MIN_W + "px;' in js and 'flex:0 1 " + MON_W + "px;width:" + MON_W + "px;align-items:stretch;min-width:" + MON_MIN_W + "px;' in js
      and 'flex:1 1 auto;min-width:" + LEFT_MIN_W + "px;' in js, "M2 the work row's columns shrink to their floors")
_need("nodeType.prototype.computeSize = function () {" in js and "size[0] = Math.max(size[0], TL_MIN_W);" in js
      and "new ResizeObserver" not in js and re.search(r"\.onResize\s*=", js) is None, "M2 the node's floor width through computeSize; no onResize, no observer")

_need(all(b < 128 for b in Path(__file__).read_bytes()), "S  ASCII")
print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
