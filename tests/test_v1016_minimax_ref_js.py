# -*- coding: ascii -*-
"""Guard v1016 -- ph_minimax_ref.js: the reference node grows with what is
wired, and never loses a wire doing it.

DRIVEN in node.js against a fake LiteGraph node (real ph_widget_vis.js next
to it, only the `app` import stubbed):

  J1  PARITY: every pin name the backend declares as a socket is in the
      frontend's DISPLAY_ORDER with the right type; the group counts match
      N_IMAGES / N_VIDEOS / N_AUDIOS of nodes/ph_minimax_ref.py.
  J2  A fresh node (all declared pins present, nothing wired) collapses to
      clip, vae, audio_vae, latent, image_1, video_1, audio_1 -- and every
      megapixels_n is hidden.
  J3  Wiring image_1 adds image_2 as the spare and shows megapixels_1;
      wiring video_1 adds video_audio_1 and video_2.
  J4  A HOLE stays: image_1 and image_3 wired -> image_1..4 present.
  J5  An OLD SAVE (clip, vae, latent, image_1..3, prompt as a converted
      widget, image_2 wired) heals: audio_vae and the spares are added, the
      groups come back into DISPLAY_ORDER, the converted `prompt` pin keeps a
      place AFTER them, and every link's target_slot equals its new index.
  J6  A wired pin is never removed, even where the rule would drop it.
  J7  Idempotent: a second tidy changes nothing.

Script-style: exit 0 = pass.
"""
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "web" / "js" / "ph_minimax_ref.js"
VIS = ROOT / "web" / "js" / "ph_widget_vis.js"
PY = ROOT / "nodes" / "ph_minimax_ref.py"
FAILS = []


def _fail(msg):
    FAILS.append(msg)
    print("  FAIL  " + msg)


def _ok(msg):
    print("  ok    " + msg)


if shutil.which("node") is None:
    print("  note  node.js not found -- test_v1016_minimax_ref_js SKIPPED")
    sys.exit(0)

# ---------------------------------------------------------------- backend facts
py = PY.read_text(encoding="utf-8")
consts = {}
for n in ast.parse(py).body:
    if isinstance(n, ast.Assign) and len(n.targets) == 1 \
            and getattr(n.targets[0], "id", "") in ("N_IMAGES", "N_VIDEOS", "N_AUDIOS"):
        consts[n.targets[0].id] = n.value.value
opt = {}
for n in ast.walk(ast.parse(py)):
    if isinstance(n, ast.Dict):
        keys = [k.value for k in n.keys if isinstance(k, ast.Constant)]
        if "image_1" in keys and "audio_vae" in keys:
            for k, v in zip(n.keys, n.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Tuple):
                    opt[k.value] = v.elts[0].value
sockets = {k: t for k, t in opt.items() if t in ("IMAGE", "VIDEO", "AUDIO", "VAE")}

tmp = Path(tempfile.mkdtemp(prefix="v1016js_"))
src = JS.read_text(encoding="utf-8").replace(
    'import { app } from "../../scripts/app.js";',
    'const app = { registerExtension() {} };')
(tmp / "ph_minimax_ref.mjs").write_text(src, encoding="utf-8")
shutil.copy(VIS, tmp / "ph_widget_vis.js")
# the module imports "./ph_widget_vis.js" -- keep that exact name
(tmp / "ph_minimax_ref.mjs").write_text(src, encoding="utf-8")

harness = r"""
import * as MR from "./ph_minimax_ref.mjs";
globalThis.LiteGraph = { INPUT: 1 };
const out = {};
out.order = MR.DISPLAY_ORDER; out.types = MR.PIN_TYPE;
out.counts = [MR.N_IMAGES, MR.N_VIDEOS, MR.N_AUDIOS];

let LID = 100;
function mkNode(names, links) {
  const g = { links: new Map() };
  const node = {
    graph: g, size: [300, 200], inputs: [], widgets: [],
    addInput(name, type) { this.inputs.push({ name, type, link: null }); },
    removeInput(i) { this.inputs.splice(i, 1); },
    setSize(s) { this.size = s; }, computeSize() { return [300, 20 * this.inputs.length]; },
    setDirtyCanvas() {},
  };
  for (const n of names) node.inputs.push({ name: n, type: MR.PIN_TYPE[n] || "STRING", link: null });
  for (const n of (links || [])) wire(node, n);
  for (let k = 1; k <= 9; k++) node.widgets.push({ name: "megapixels_" + k, type: "number" });
  return node;
}
function wire(node, name) {
  const p = node.inputs.find((i) => i.name === name);
  const id = LID++; p.link = id;
  node.graph.links.set(id, { id, target_slot: node.inputs.indexOf(p) });
}
const names = (n) => n.inputs.map((i) => i.name);
const hidden = (n) => n.widgets.filter((w) => String(w.type).startsWith("pls-hidden-")).map((w) => w.name);
const slotsOk = (n) => n.inputs.every((p, i) => p.link == null || n.graph.links.get(p.link).target_slot === i);

// J2 fresh
const fresh = mkNode(MR.DISPLAY_ORDER);
MR.tidy(fresh);
out.j2 = { names: names(fresh), hidden: hidden(fresh) };

// J3 grow
wire(fresh, "image_1"); MR.tidy(fresh);
out.j3a = { names: names(fresh), hidden: hidden(fresh) };
wire(fresh, "video_1"); MR.tidy(fresh);
out.j3b = names(fresh);

// J4 hole
const h = mkNode(MR.DISPLAY_ORDER);
MR.tidy(h);
wire(h, "image_1"); MR.tidy(h);
wire(h, "image_2"); MR.tidy(h);
wire(h, "image_3"); MR.tidy(h);
h.inputs.find((i) => i.name === "image_2").link = null; MR.tidy(h);
out.j4 = names(h);

// J5 old save
const old = mkNode(["clip", "vae", "latent", "image_1", "image_2", "image_3", "prompt"], ["image_2", "prompt", "clip"]);
MR.tidy(old);
out.j5 = { names: names(old), slots: slotsOk(old),
           wired: old.inputs.filter((i) => i.link != null).map((i) => i.name) };

// J6 wired pin survives the rule: audio_3 wired alone, then its predecessors unwired
const w6 = mkNode(MR.DISPLAY_ORDER, ["audio_3", "video_audio_2"]);
MR.tidy(w6);
out.j6 = names(w6);

// J7 idempotent
const before = JSON.stringify(names(old));
MR.tidy(old);
out.j7 = before === JSON.stringify(names(old)) && slotsOk(old);

console.log(JSON.stringify(out));
"""
(tmp / "h.mjs").write_text(harness, encoding="utf-8")
r = subprocess.run(["node", str(tmp / "h.mjs")], capture_output=True, text=True, timeout=60)
if r.returncode != 0:
    print(r.stderr[-2000:])
    _fail("harness crashed")
    sys.exit(1)
d = json.loads(r.stdout.strip().splitlines()[-1])

# J1
if d["counts"] == [consts.get("N_IMAGES"), consts.get("N_VIDEOS"), consts.get("N_AUDIOS")]:
    _ok("J1 group counts match the backend (%s)" % d["counts"])
else:
    _fail("J1 counts js %s py %s" % (d["counts"], consts))
bad = [k for k, t in sockets.items() if d["types"].get(k) != t or k not in d["order"]]
if not bad and len(sockets) == 9 + 3 + 3 + 3 + 1 + 1:   # PLS_REFS is not counted here; v1049: + first_frame
    _ok("J1 all %d backend sockets are in DISPLAY_ORDER with the same type" % len(sockets))
else:
    _fail("J1 socket parity broken: %r (backend sockets %d)" % (bad, len(sockets)))

# J2
# v1019: + refs (Reference Board) -- a fixed pin like audio_vae; v1049: + first_frame (scene carry), fixed too
base = ["clip", "vae", "audio_vae", "latent", "refs", "first_frame", "image_1", "video_1", "audio_1"]
if d["j2"]["names"] == base:
    _ok("J2 a fresh node collapses to %s" % base)
else:
    _fail("J2 fresh node pins %s" % d["j2"]["names"])
if sorted(d["j2"]["hidden"]) == sorted("megapixels_%d" % n for n in range(1, 10)):
    _ok("J2 every megapixels_n is hidden while nothing is wired")
else:
    _fail("J2 hidden widgets %s" % d["j2"]["hidden"])

# J3
if d["j3a"]["names"][:8] == ["clip", "vae", "audio_vae", "latent", "refs", "first_frame", "image_1", "image_2"] \
        and "megapixels_1" not in d["j3a"]["hidden"] and "megapixels_2" in d["j3a"]["hidden"]:
    _ok("J3 wiring image_1 adds image_2 and shows megapixels_1 only")
else:
    _fail("J3 after image_1: %s / hidden %s" % (d["j3a"]["names"], d["j3a"]["hidden"]))
want3 = ["clip", "vae", "audio_vae", "latent", "refs", "first_frame", "image_1", "image_2",
         "video_1", "video_audio_1", "video_2", "audio_1"]
if d["j3b"] == want3:
    _ok("J3 wiring video_1 adds its soundtrack pin and video_2, in display order")
else:
    _fail("J3 after video_1: %s" % d["j3b"])

# J4
if [n for n in d["j4"] if n.startswith("image_")] == ["image_1", "image_2", "image_3", "image_4"]:
    _ok("J4 a hole stays: image_1 + image_3 wired -> image_1..4")
else:
    _fail("J4 %s" % d["j4"])

# J5
j5 = d["j5"]
want5 = ["clip", "vae", "audio_vae", "latent", "refs", "first_frame", "image_1", "image_2", "image_3",
         "video_1", "audio_1", "prompt"]
if j5["names"] == want5:
    _ok("J5 an old save heals: audio_vae + spares added, groups in order, `prompt` after them")
else:
    _fail("J5 healed pins %s" % j5["names"])
if j5["slots"] and sorted(j5["wired"]) == ["clip", "image_2", "prompt"]:
    _ok("J5 every link's target_slot equals its new index; no wire lost")
else:
    _fail("J5 links: slots ok=%s wired=%s" % (j5["slots"], j5["wired"]))

# J6
if "audio_3" in d["j6"] and "video_audio_2" in d["j6"]:
    _ok("J6 wired pins survive the rule (audio_3 alone, video_audio_2 without video_2)")
else:
    _fail("J6 %s" % d["j6"])

# J6b -- the belt, pinned where it acts. The rule above already keeps every
# wired pin, so no drive can reach this line today (measured: removing it let
# the whole suite stay green). It is kept as a second lock for the day the rule
# changes, and pinned here as CODE: the only removeInput call must sit behind
# `p.link == null`.
import re as _re
_js = JS.read_text(encoding="utf-8")
_rm = [l.strip() for l in _js.splitlines() if "removeInput(" in l and not l.strip().startswith("*")]
if len(_rm) == 1 and _re.search(r"p\.link\s*==\s*null\s*&&", _rm[0]):
    _ok("J6b the only removeInput sits behind `p.link == null` (second lock)")
else:
    _fail("J6b removeInput not guarded by the link test: %r" % _rm)

# J7
if d["j7"]:
    _ok("J7 a second tidy changes nothing")
else:
    _fail("J7 tidy is not idempotent")

shutil.rmtree(tmp, ignore_errors=True)
print()
if FAILS:
    print("test_v1016_minimax_ref_js: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1016_minimax_ref_js: PASS (J1-J7)")
