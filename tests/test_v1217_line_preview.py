# -*- coding: ascii -*-
"""Guard v1217 -- the model line says what it sees, the notice ends with the download, Control Map shows its frames.

Frank 23.10.:
  - "Dieses None on Disk und gruen ist missverstaendlich ... 'Aktuell sind keine Modelle vorhanden. Waehle eins aus
    oder lade sie ...' So was in Kurzform" -- "Ja, bitte korrigieren und zwar ueberall."
  - the amber notice stayed on "1 KB of 1 KB ... downloading ..." after the line's download had finished.
  - "Ich will bei OpenPose bzw. generell bei Controlnet eine Preview haben ... entweder direkt in der Node oder auch
    gerne als Ausgang ... diese Control-Net-Bilder bzw. Videos koennen auch wieder als Vorlage dienen".

  L1 pickState (node.js): nothing picked ('', '(none)', '(put ...)') -> 'none' with the set's recommended file
     (else its first required one); a door file -> 'door' / 'missing'; a listed file the door does not know ->
     'own'; a name neither in the door nor in the list -> 'lost'
  L2 updatePick draws: green ONLY in the 'door' and 'own' branches; 'none' says "No model picked -- ... needs one.
     Pick a file, or:" with the Download of the recommended file, which is then PICKED; the v1216 text
     "-- on disk" for an unknown value is gone; 'lost' says "not found ... pick another"
  L3 the set line (neededModels): a node that cannot run without its pick (required) and has nothing picked needs
     the set's own file and says "No model picked"; 'off' is still a choice (nothing needed)
  L4 the notice (node.js with a minimal DOM): the server's "done" event ends the box (installed, Close) instead
     of drawing "1 KB of 1 KB"; an install finished elsewhere (the node line) ends it too (onInstalled); "error"
     turns it into Retry; after the end no progress event redraws it
  L5 the Control Map preview: preview_frames keeps every frame of a short clip, at most PREVIEW_MAX of a long
     one (the speed kept), long side PREVIEW_SIDE; preview_ui writes an animated WebP (all frames readable back)
     for a clip, a PNG for one frame, into the given folder; make() returns {"ui", "result"} with preview on and
     the plain tuple with it off; a preview that cannot be written never stops the run
  L6 the widget: `preview` is APPENDED (the last optional, default True); WIDGET_ORDER baseline carries it
  S  ASCII
Script-style: exit 0 = pass.
"""
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1217_line_preview"
FAILS = []
TMP = Path(tempfile.mkdtemp(prefix="pls1217_"))


def _need(c, m):
    print(("  ok    " if c else "  FAIL  ") + m)
    if not c:
        FAILS.append(m)


WJ = (ROOT / "web" / "js" / "ph_world_weights.js").read_text(encoding="utf-8")
DJ = (ROOT / "web" / "js" / "ph_model_door.js").read_text(encoding="utf-8")
node_js = shutil.which("node")


def run_js(code, name):
    (TMP / name).write_text(code, encoding="utf-8")
    out = subprocess.run([node_js, str(TMP / name)], capture_output=True, text=True, timeout=60)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr
    except Exception:
        return None, out.stderr + out.stdout


def cut(src, start):
    i = src.find(start)
    return src[i:src.find("\n}", i) + 2]


# ---------------------------------------------------------------- L1
print("L1 pickState")
if node_js:
    code = (WJ[WJ.find("const TAIL = "):WJ.find("\n", WJ.find("const TAIL = ")) + 1] + cut(WJ, "export function stripLabel(v)")
            + cut(WJ, "export function pickState(")).replace("export function", "function") + r"""
const M = [{ name: "a.safetensors", installed: false, optional: false }, { name: "b.safetensors", installed: true, optional: false, recommended: false },
           { name: "c.safetensors", installed: false, optional: false, recommended: true }];
const L = ["a.safetensors \u00b7 1 GB \u00b7 not installed", "b.safetensors \u00b7 2 GB", "mine.safetensors"];
const k = (v, m, l) => { const s = pickState(v, m, l); return [s.kind, s.m ? s.m.name : null]; };
console.log(JSON.stringify([k("", M, L), k("(none)", M, L), k("(put an adapter into models/x)", M, L),
  k("(none)", [{ name: "x", optional: true }, { name: "y", optional: false }], []), k("(none)", [], []),
  k("a.safetensors \u00b7 1 GB \u00b7 not installed", M, L), k("b.safetensors \u00b7 2 GB", M, L), k("sub/b.safetensors", M, L),
  k("mine.safetensors", M, L), k("gone.safetensors", M, L)]));
"""
    r, err = run_js(code, "l1.mjs")
    _need(r == [["none", "c.safetensors"], ["none", "c.safetensors"], ["none", "c.safetensors"], ["none", "y"], ["none", None],
                ["missing", "a.safetensors"], ["door", "b.safetensors"], ["door", "b.safetensors"], ["own", None], ["lost", None]],
          "L1 none (recommended, else first required) / missing / door (also in a subfolder) / own / lost (%s %s)" % (r, err[-200:]))
else:
    _need(False, "L1 node.js needed")

# ---------------------------------------------------------------- L2
print("L2 updatePick")
up = WJ[WJ.find("async function updatePick(node)"):WJ.find("const SET_TEXT")]
greens = [ln for ln in up.split("\n") if '"#9c9"' in ln]
_need(len(greens) == 2 and 'st.m.title + " on disk ("' in greens[0] and '" on disk (your own file)"' in greens[1],
      "L2 green only twice: a door file on disk, your own file on disk")
_need('if (st.kind === "door")' in up and 'if (st.kind === "own")' in up and up.find('if (st.kind === "door")') < up.find('"#9c9"'),
      "L2 each green behind its own state")
_need('"No model picked -- " + who + " needs one. Pick a file, or:"' in up and 'm.title + " not on disk --"' in up,
      "L2 nothing picked / a door file missing: the short texts")
_need('" -- on disk", "#9c9"' not in WJ and '" not found in " + folder + " -- pick another"' in up, "L2 the old '-- on disk' is gone; lost says so")
_need("w.value = fresh.label;" in up and "vals.unshift(fresh.label)" in up, "L2 the fetched file becomes the pick")
_need('need: "pose"' in WJ and 'need: "depth"' in WJ, "L2 Control Map names what needs the model (pose / depth)")

# ---------------------------------------------------------------- L3
print("L3 the set line")
if node_js:
    code = (WJ[WJ.find("const TAIL = "):WJ.find("\n", WJ.find("const TAIL = ")) + 1] + cut(WJ, "export function stripLabel(v)")
            + cut(WJ, "export function neededModels(")).replace("export function", "function") + r"""
const B = [{ name: "v1.safetensors", optional: false }, { name: "bunny.safetensors", optional: true }];
const R = [{ name: "r3b.safetensors", optional: true }, { name: "vae.safetensors", optional: true }];
const n = (L) => L.map((m) => m.name);
const req = { kind: "pick", widget: "adapter", required: true }, rs = { kind: "pick", widget: "restore_model", companion: /vae/i };
console.log(JSON.stringify([n(neededModels(B, req, "(put a Semantic Bridge adapter into models/semantic_bridge)")), n(neededModels(B, req, "")),
  n(neededModels(B, req, "bunny.safetensors \u00b7 21 MB \u00b7 not installed")), n(neededModels(R, rs, "off")), n(neededModels(R, rs, "(none)"))]));
"""
    r, err = run_js(code, "l3.mjs")
    _need(r == [["v1.safetensors"], ["v1.safetensors"], ["bunny.safetensors"], [], []],
          "L3 required + nothing picked -> the set's own file; a pick -> that pick; 'off' / not required -> nothing (%s %s)" % (r, err[-200:]))
else:
    _need(False, "L3 node.js needed")
# public build: the Semantic Bridge is not part of it -- the required path and its text are checked without it
_need('"No model picked -- this node needs one. Pick a file, or:"' in WJ, "L3 a required node with nothing picked says so")

# ---------------------------------------------------------------- L4
print("L4 the notice ends with the download")
if node_js:
    harness = r"""
const listeners = {};
const mk = (tag) => ({ tag, style: {}, children: [], textContent: "", className: "", disabled: false, onclick: null,
  appendChild(c) { this.children.push(c); c.parent = this; return c; }, remove() { this.removed = true; }, setAttribute() {} });
globalThis.document = { head: mk("head"), body: Object.assign(mk("body"), { contains: () => true }), createElement: mk };
globalThis.setTimeout = (fn) => 0;
const app = { registerExtension() {}, refreshComboInNodes() {} };
let answer = { ok: true, path: "D:/m/x.safetensors" };
const api = { addEventListener(n, fn) { listeners[n] = fn; }, fetchApi: async () => ({ json: async () => answer }) };
"""
    src = DJ.replace('import { app } from "../../scripts/app.js";', "").replace('import { api } from "../../scripts/api.js";', "")
    code = harness + src + r"""
const ev = (name, done, total, phase) => listeners["pls.weights.progress"]({ detail: { name, done, total, phase } });
const box = () => document.body.children[0].children[document.body.children[0].children.length - 1];
const st = (b) => [b.children[0].textContent, b.children[1].textContent, b.children[3].children[1].textContent];
const M = (n) => ({ name: n, title: "T " + n, size: "4.2 GB", bytes: 4531220608, dir: "D:/m" });
const out = [];
noticeMissing(M("a"), "Apply Control"); const A = box();
ev("a", 1000, 4531220608, "run"); out.push(A.children[1].textContent.indexOf(" of ") > 0);
ev("a", 1, 1, "done"); out.push(st(A));
ev("a", 5, 10, "run"); out.push(st(A)[0] + " | " + st(A)[1] + " | " + A.children[2].style.display);
noticeMissing(M("b"), "Control Map"); const B = box();
await installModel("b"); out.push(st(B)[0]);
noticeMissing(M("c"), "Control Map"); const C = box();
ev("c", 0, 0, "error"); out.push([st(C)[0], C.children[3].children[0].textContent]);
console.log(JSON.stringify(out));
"""
    r, err = run_js(code, "l4.mjs")
    ok = (r is not None and r[0] is True and r[1][0].endswith("T a installed") and r[1][2] == "Close"
          and r[2].endswith("T a installed | \u2713 Installed: D:/m | none") and r[3].endswith("T b installed")
          and r[4][0].endswith("download failed") and r[4][1] == "Retry")
    _need(ok, "L4 'done' ends the box, a line's install ends it, no redraw after, 'error' -> Retry (%s %s)" % (r, err[-300:]))
else:
    _need(False, "L4 node.js needed")
_need('if (phase === "done") { finish(""); return; }' in DJ and "offDone = onInstalled(" in DJ, "L4 the two endings are wired")

# ---------------------------------------------------------------- L5
print("L5 the Control Map preview")
fp = types.ModuleType("folder_paths")
fp.get_temp_directory = lambda: str(TMP / "temp")
fp.get_filename_list = lambda f: []
fp.get_folder_paths = lambda f: [str(TMP / f)]
sys.modules["folder_paths"] = fp
pkg = types.ModuleType("plsn1217")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1217"] = pkg
CT = importlib.import_module("plsn1217.ph_control")
pics, ms = CT.preview_frames(torch.rand(107, 1344, 768, 3))
_need(len(pics) == 107 and pics[0].shape == (384, 219, 3) and ms == 42, "L5 a 107-frame clip: every frame, long side 384, 42 ms (%d, %s, %d)"
      % (len(pics), pics[0].shape, ms))
pics, ms = CT.preview_frames(torch.rand(500, 64, 96, 3))
_need(len(pics) == CT.PREVIEW_MAX and pics[0].shape == (64, 96, 3) and ms == 87,
      "L5 a 500-frame clip: 240 frames, the speed kept (87 ms), a small frame not enlarged (%d, %d)" % (len(pics), ms))
from PIL import Image  # noqa: E402
u = CT.preview_ui(torch.rand(30, 64, 48, 3), "t", folder=str(TMP / "pv"))
im = Image.open(TMP / "pv" / u["images"][0]["filename"]) if u else None
_need(u and u.get("animated") == [True] and u["images"][0]["type"] == "temp" and u["images"][0]["filename"].endswith(".webp")
      and im is not None and getattr(im, "n_frames", 1) == 30, "L5 a clip -> an animated WebP in temp, all 30 frames readable")
u1 = CT.preview_ui(torch.rand(1, 64, 48, 3), "t", folder=str(TMP / "pv"))
_need(u1 and "animated" not in u1 and u1["images"][0]["filename"].endswith(".png"), "L5 one frame -> a PNG, not animated")
node = CT.ULSControlMap()
args = ("canny", "(none)", "(none)", True, False, 4, 0.2, 0.5, 504)
on = node.make(*args, images=torch.rand(6, 32, 32, 3))
off = node.make(*args, images=torch.rand(6, 32, 32, 3), preview=False)
_need(isinstance(on, dict) and on["ui"].get("animated") == [True] and len(on["result"]) == 3 and tuple(on["result"][0].shape) == (6, 32, 32, 3),
      "L5 make(): preview on -> ui + result (the frames unchanged)")
_need(isinstance(off, tuple) and len(off) == 3, "L5 make(): preview off -> the plain result")
fp.get_temp_directory = lambda: str(TMP / "temp" / "\0bad")
bad = node.make(*args, images=torch.rand(2, 16, 16, 3))
_need(isinstance(bad, tuple) and len(bad) == 3, "L5 a preview that cannot be written never stops the run")

# ---------------------------------------------------------------- L6
print("L6 the widget")
opt = list(CT.ULSControlMap.INPUT_TYPES()["optional"].items())
_need(opt[-1][0] == "preview" and opt[-1][1][0] == "BOOLEAN" and opt[-1][1][1]["default"] is True, "L6 preview: appended, default True")
wo = list(ROOT.glob("WIDGET_ORDER_baseline_v*.txt"))
_need(len(wo) == 1 and "\nULSControlMap\tmode,draw_hands,draw_face,stick_width,low,high,depth_resolution,keypoints,preview\n" in wo[0].read_text(),
      "L6 WIDGET_ORDER: preview at the end of Control Map")

# ---------------------------------------------------------------- S
print("S ASCII")
for f in ("web/js/ph_world_weights.js", "web/js/ph_model_door.js", "nodes/ph_control.py", "tests/test_v1217_line_preview.py"):
    try:
        (ROOT / f).read_bytes().decode("ascii")
        _need(True, "S %s" % f)
    except UnicodeDecodeError:
        _need(False, "S %s is not ASCII" % f)

shutil.rmtree(TMP, ignore_errors=True)
print()
print("%s: %s (%d fail)" % (NAME, "PASS" if not FAILS else "FAIL", len(FAILS)))
sys.exit(1 if FAILS else 0)
