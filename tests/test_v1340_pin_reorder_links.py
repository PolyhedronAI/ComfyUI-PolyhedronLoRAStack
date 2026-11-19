#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1340 -- the pin reorder keeps every wire on its own pin, in all five nodes that reorder pins.

MEASURED 17.11. in the live frontend: a pin's `link` is read THROUGH the links' target_slot (the link whose
target_slot is the pin's index). Two consequences, both met in the field build of v1339:
  - swapping node.inputs before the links learn their new slots hands a wire to whatever pin now sits on the
    old slot (Image Reference: prompt -> inpaint_mask; MiniMax Reference: prompt -> ref_noise);
  - reading next[i].link INSIDE the loop that rewrites target_slot returns another pin's wire once an earlier
    slot was rewritten -- so all ids are read FIRST.

WHAT IS PINNED

  P1  static, in ph_image_ref.js, ph_minimax_ref.js, ph_cine_timeline_pins.js, ph_power_upscale.js and
      ph_music_mixer.js: the ids are snapshotted from the NEW list before any target_slot is written, the
      list is assigned AFTER that, and no loop reads next[i].link.
  P2  ph_image_ref.js and ph_minimax_ref.js driven against a node that behaves like the live frontend (a pin's
      link is derived from the links' target_slot): a wired widget socket (prompt) keeps its wire while
      image pins are wired and the spare pins appear; every link's slot equals its pin's index. The same
      harness on the v1339 MiniMax order (swap first) loses the wire -- the probe can see the bug.
  P3  ASCII (ph_power_upscale.js: the v1340 lines -- the file carries UI glyphs since long before).

Script-style: exit 0 = pass.
"""

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

NAME = "v1340"
ROOT = Path(__file__).resolve().parent.parent
JSDIR = ROOT / "web" / "js"
# PUBLIC CUT (v407, declared): ph_music_mixer.js is not public -- the four reorders this build ships
FILES = [f for f in ["ph_image_ref.js", "ph_minimax_ref.js", "ph_cine_timeline_pins.js", "ph_power_upscale.js",
                     "ph_music_mixer.js"] if (JSDIR / f).exists()]
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)
        print("  FAIL " + msg)
    else:
        print("  ok   " + msg)


# ---- P1 ---------------------------------------------------------------------------------------------------
print("P1 static order in the five reorders")
for f in FILES:
    s = (JSDIR / f).read_text(encoding="utf-8")
    snap = re.search(r"const ids = next\.map\(", s)
    assign = s.find("node.inputs = next;")
    write = s.find("target_slot = ", snap.start()) if snap else -1
    _need(snap is not None and assign > 0 and snap.start() < write < assign,
          "%s: ids snapshotted, slots written, THEN the list swapped" % f)
    _need(not re.search(r"next\[i\]\s*\?\s*next\[i\]\.link", s) and "inp.link); if (l) l.target_slot" not in s,
          "%s: no link read inside the slot loop" % f)

# ---- P2 ---------------------------------------------------------------------------------------------------
print("P2 live-like frontend: a pin's link follows target_slot")
tmp = Path(tempfile.mkdtemp(prefix="v1340js_"))
for f in ("ph_image_ref.js", "ph_minimax_ref.js"):
    src = (JSDIR / f).read_text(encoding="utf-8").replace(
        'import { app } from "../../scripts/app.js";', 'const app = { registerExtension() {} };')
    (tmp / f.replace(".js", ".mjs")).write_text(src, encoding="utf-8")
shutil.copy(JSDIR / "ph_widget_vis.js", tmp / "ph_widget_vis.js")
mr = (tmp / "ph_minimax_ref.mjs").read_text(encoding="utf-8")
old = mr.replace("    const links = node.graph && node.graph.links;\n    if (links) {\n        // ALL ids first",
                 "    node.inputs = next;\n    const links = node.graph && node.graph.links;\n    if (links) {\n"
                 "        // ALL ids first", 1)
_need(old != mr, "the v1339 order can be rebuilt for the probe")
(tmp / "old_mr.mjs").write_text(old, encoding="utf-8")
harness = r"""
import * as IR from "./ph_image_ref.mjs";
import * as MR from "./ph_minimax_ref.mjs";
import * as OLD from "./old_mr.mjs";
globalThis.LiteGraph = { INPUT: 1 };
function live(names) {
  const links = new Map();
  let arr = [];
  const node = { graph: { links }, size: [300, 200], widgets: [], id: 7,
    get inputs() { return arr; }, set inputs(v) { arr = v; },
    addInput(name, type) { arr.push(slot(name, type)); },
    removeInput(i) { arr.splice(i, 1); for (const l of links.values()) if (l.target_slot > i) l.target_slot--; },
    setSize(s) { this.size = s; }, computeSize() { return [300, 20 * arr.length]; }, setDirtyCanvas() {} };
  // the live frontend: a pin's link is the link whose target_slot is the pin's index
  function slot(name, type) {
    const p = { name, type };
    Object.defineProperty(p, "link", { enumerable: true,
      get() { const i = arr.indexOf(p); for (const [id, l] of links) if (l.target_slot === i) return id; return null; },
      set(v) { if (v == null) return; const l = links.get(v); if (l) l.target_slot = arr.indexOf(p); } });
    return p;
  }
  for (const n of names) arr.push(slot(n, "X"));
  const wire = (name, id) => links.set(id, { id, target_slot: arr.findIndex((p) => p.name === name) });
  const by = () => Object.fromEntries(arr.map((p) => [p.name, p.link]));
  const slotsOk = () => [...links.values()].every((l) => arr[l.target_slot] && arr[l.target_slot].link === l.id);
  return { node, wire, by, slotsOk };
}
function runIR(IRm) {
  const t = live(["clip", "vae", "latent", "image_1", "inpaint_mask", "prompt", "negative_prompt"]);
  t.wire("prompt", 1); IRm.tidy(t.node); t.wire("image_1", 2); IRm.tidy(t.node); t.wire("image_2", 3); IRm.tidy(t.node);
  const b = t.by(); return { prompt: b.prompt, mask: b.inpaint_mask, i1: b.image_1, i2: b.image_2, i3: b.image_3, ok: t.slotsOk() };
}
function runMR(MRm) {
  const t = live(["clip", "vae", "audio_vae", "latent", "refs", "first_frame", "image_1", "video_1", "audio_1",
                  "prompt", "ref_image_size", "tags", "ref_noise"]);
  t.wire("prompt", 1); MRm.tidy(t.node); t.wire("image_1", 2); MRm.tidy(t.node); t.wire("image_2", 3); MRm.tidy(t.node);
  const b = t.by(); return { prompt: b.prompt, noise: b.ref_noise, i1: b.image_1, i2: b.image_2, ok: t.slotsOk() };
}
console.log(JSON.stringify({ ir: runIR(IR), mr: runMR(MR), old: runMR(OLD) }));
"""
(tmp / "h.mjs").write_text(harness, encoding="utf-8")
r = subprocess.run(["node", str(tmp / "h.mjs")], capture_output=True, text=True, timeout=60)
_need(r.returncode == 0, "harness ran " + (r.stderr or "")[-400:])
if r.returncode == 0:
    js = json.loads(r.stdout.strip().splitlines()[-1])
    ir, mr_, old_ = js["ir"], js["mr"], js["old"]
    _need(ir["prompt"] == 1 and ir["mask"] is None and ir["i1"] == 2 and ir["i2"] == 3 and ir["ok"],
          "Image Reference: prompt keeps its wire, images on their pins, slots consistent: %r" % ir)
    _need(mr_["prompt"] == 1 and mr_["noise"] is None and mr_["i1"] == 2 and mr_["i2"] == 3 and mr_["ok"],
          "MiniMax Reference: prompt keeps its wire, ref_noise stays free: %r" % mr_)
    _need(old_["prompt"] != 1 or old_["noise"] is not None or not old_["ok"],
          "the probe sees the v1339 MiniMax order lose the wire: %r" % old_)

# ---- P3 ---------------------------------------------------------------------------------------------------
print("P3 ASCII")
# ph_power_upscale.js carries UI glyphs (hexagon, info mark) since long before v1340 -- there only the
# v1340 lines are checked
for f in FILES:
    s = (JSDIR / f).read_text(encoding="utf-8")
    if f == "ph_power_upscale.js":
        s = "\n".join(l for l in s.splitlines() if "v1340" in l or "const ids = next" in l)
    _need(all(ord(c) < 128 for c in s), f + " ASCII-clean (v1340 lines)" if f == "ph_power_upscale.js" else f + " ASCII-clean")

print("%s: %s" % (NAME, "PASS" if not _fails else "FAIL %d" % len(_fails)))
sys.exit(1 if _fails else 0)
