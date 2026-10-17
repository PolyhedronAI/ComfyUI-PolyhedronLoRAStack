# -*- coding: ascii -*-
"""Guard v1044 -- the LAYERS on screen (E2 rows) and their sources (E3), and
the height grip Frank could not find (30.09.: "I cannot make the height
smaller again; I find no handle for it").

  R1  the stack helpers (node.js): addLayer (an EMPTY layer on top),
      moveLayer (one step, not past the ends), removeLayer (never the render
      layer), setLayer, layerOf, newLayerId (unique), resetPieces (the clip
      whole again where its material began), ownerAt, sizeOf, stackExtra
      (nothing without a stack; the head + every row with the lanes' gap),
      layerHeights (clamped), pathFromUri (file:/// -> a Windows / a posix
      path), dropPathOf (only a video path).
  R2  the server: clean_layers keeps an EMPTY layer and a file layer whose
      pieces were all deleted; stack_of gives an empty layer an empty plan;
      media_dir makes input/cine_media/<project>/; probe_layer reads a clip
      (frames at 24) and refuses a non-video; the routes are registered.
  R3  statics: the height grip -- DIV_H tall, above the cut layer (z 8 > 7),
      centred on the gap, a visible pill that lights up; the size menus
      (S/M/L/XL) for the render lanes and each layer; the head buttons one
      fixed size; the drop zones stacked (upper = copy, lower = in place, the
      time is x); Delete stopped at the field (never reaches the canvas); the
      layer rows around the render lane in stack order; the player picks the
      stack's owner; the base lane draws a gap as empty; ASCII.

Script-style: exit 0 = pass.
"""
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _fail(m):
    FAILS.append(m)
    print("  FAIL  " + m)


def _ok(m):
    print("  ok    " + m)


def _need(c, m):
    (_ok if c else _fail)(m)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _load("ce1044", "nodes/cine_edit.py")
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")

# ------------------------------------------------------------------ R1
if shutil.which("node") is None:
    print("  note  node.js not found -- R1 SKIPPED")
else:
    t2 = Path(tempfile.mkdtemp(prefix="v1044js_"))
    body = (js.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
              .replace('import { api } from "../../scripts/api.js";', "const api = {};")
              .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (t2 / "m.mjs").write_text(body, encoding="utf-8")
    (t2 / "h.mjs").write_text(r"""
import * as T from "./m.mjs";
const o = {};
let e = {};
e = T.addLayer(e, "L1", "one"); e = T.addLayer(e, "L2");
o.order = T.stackLayers(e).map((x) => x.id);
o.empty = T.stackLayers(e)[0].src.kind + "|" + T.stackLayers(e)[0].pieces.length;
o.up = T.stackLayers(T.moveLayer(e, "L1", -1)).map((x) => x.id);
o.topUp = T.stackLayers(T.moveLayer(e, "L2", -1)).map((x) => x.id);
o.baseDown = T.stackLayers(T.moveLayer(e, "base", 1)).map((x) => x.id);
o.rmBase = T.stackLayers(T.removeLayer(e, "base")).map((x) => x.id);
o.rm = T.stackLayers(T.removeLayer(e, "L1")).map((x) => x.id);
o.rmBaseMid = T.stackLayers(T.removeLayer(T.moveLayer(e, "base", -1), "base")).map((x) => x.id);
o.overBase = T.stackLayers(T.moveLayer(e, "L1", 1)).map((x) => x.id);
o.baseLast = T.stackLayers({ layers: [{ id: "base", src: { kind: "base" }, audio: false }, { id: "X", src: { kind: "empty" } }] }).map((x) => x.id + (x.audio === false ? "-" : ""));
o.solo = T.flattenRuns([{ video: true, audio: true, has_audio: true, solo: true, track: { segments: [], asegments: [{ film_in: 0, film_out: 4 }] } },
                        { video: true, audio: true, has_audio: true, track: { segments: [], asegments: [{ film_in: 0, film_out: 8 }] } }], 8).aruns.map((r) => [r.film_in, r.film_out, r.layer]);
o.tc = [T.timecode(0), T.timecode(1.75), T.timecode(3725.5)];
o.set = T.layerOf(T.setLayer(e, "L1", { video: false }), "L1").video;
o.orig = T.layerOf(e, "L1").video;
o.ids = [T.newLayerId(e, 1), T.newLayerId(T.addLayer({}, T.newLayerId({}, 5)), 5)];
o.has = [T.hasStack({}), T.hasStack({ layers: [{ id: "base" }] }), T.hasStack(e)];
const P = T.normLayer([T.gapPiece(30), { k: 0, a: 10, b: 40, speed: 1 }, T.gapPiece(5), { k: 0, a: 50, b: 60, speed: 2 }]);
o.reset = T.resetPieces(P, 90).map((p) => [p.k, p.b - p.a]);
o.resetEmpty = T.resetPieces([], 48).map((p) => [p.k, p.b - p.a]);
o.owner = [T.ownerAt([{ film_in: 0, film_out: 10, layer: 1 }, { film_in: 10, film_out: 20, layer: null }], 5),
           T.ownerAt([{ film_in: 0, film_out: 10, layer: 1 }, { film_in: 10, film_out: 20, layer: null }], 15), T.ownerAt([], 3)];
o.size = [T.sizeOf(44), T.sizeOf(100), T.sizeOf(160), T.sizeOf(239)];
const node = { properties: {} };
o.extra0 = T.stackExtra(node);
node.properties[T.EDIT_KEY] = e;
node.properties[T.LH_KEY] = { L1: { clip: 150, audio: 60 } };
o.extra = T.stackExtra(node);
o.rowH = [T.layerRowH(node, "L1"), T.layerRowH(node, "L2")];
o.lh = T.layerHeights({ properties: { [T.LH_KEY]: { x: { clip: 5000, audio: 1 } } } }, "x");
o.uri = [T.pathFromUri("file:///C:/My%20Clips/fox.mp4"), T.pathFromUri("file:///home/me/a%20b.webm"), T.pathFromUri("C:\\x\\y.mov"), T.pathFromUri("http://x/y.mp4")];
const dt = (m) => ({ getData: (t) => m[t] || "", files: [] });
o.drop = [T.dropPathOf(dt({ "text/uri-list": "file:///D:/v/clip.mkv" })), T.dropPathOf(dt({ "text/plain": "/tmp/notes.txt" })), T.dropPathOf(dt({}))];
console.log(JSON.stringify(o));
""", encoding="utf-8")
    r = subprocess.run(["node", str(t2 / "h.mjs")], capture_output=True, text=True, timeout=60)
    shutil.rmtree(t2, ignore_errors=True)
    line = [x for x in r.stdout.splitlines() if x.startswith("{")]
    if r.returncode != 0 or not line:
        _fail("R1 harness failed: %s" % r.stderr.strip()[-400:])
    else:
        o = json.loads(line[-1])
        _need(o["order"] == ["L2", "L1", "base"] and o["empty"] == "empty|0", "R1 addLayer: an EMPTY layer on top, the render layer at the bottom %r" % o["order"])
        _need(o["up"] == ["L1", "L2", "base"] and o["topUp"] == ["L2", "L1", "base"] and o["baseDown"] == ["L2", "L1", "base"],
              "R1 moveLayer: one step; not past the top, not past the bottom")
        _need(o["rmBase"] == ["L2", "L1", "base"] and o["rm"] == ["L2", "base"] and o["rmBaseMid"] == ["L2", "L1", "base"],
              "R1 removeLayer: never the render layer %r" % o["rmBaseMid"])
        _need(o["overBase"] == ["L2", "L1", "base"] and o["baseLast"] == ["X", "base-"],
              "R1 v1045: the render stays V1 (the bottom) -- no move crosses it, an old stack puts it back there, its switches kept %r" % o["baseLast"])
        _need(o["solo"] == [[0, 4, 0], [4, 8, None]], "R1 v1045: solo -- only the solo track sounds, the others are silent %r" % o["solo"])
        _need(o["tc"] == ["00:00:00:00", "00:00:01:18", "01:02:05:12"], "R1 v1045: the timecode HH:MM:SS:FF %r" % o["tc"])
        _need(o["set"] is False and o["orig"] is True, "R1 setLayer hands back a NEW edit (the old one untouched)")
        _need(len(set(o["ids"])) == 2 and all(i.startswith("L") for i in o["ids"]), "R1 newLayerId: unique %r" % o["ids"])
        _need(o["has"] == [False, False, True], "R1 hasStack: only with a layer besides the render one")
        _need(o["reset"] == [[-1, 30], [0, 90]] and o["resetEmpty"] == [[0, 48]], "R1 resetPieces: the whole clip where its material began %r" % o["reset"])
        _need(o["owner"] == [1, None, None], "R1 ownerAt")
        _need(o["size"] == ["S", "M", "L", "XL"], "R1 sizeOf: the nearest size")
        # AMENDED v1153: a track's sound row is 28 by default
        _need(o["extra0"] == 22 + 4 and o["rowH"] == [150 + 4 + 60, 44 + 4 + 28]
              and o["extra"] == 22 + 4 + o["rowH"][0] + 4 + o["rowH"][1] + 4, "R1 stackExtra (v1045): the new-track row always; each track's picture + sound row %r" % [o["extra"], o["rowH"]])
        _need(o["lh"] == {"clip": 240, "audio": 14}, "R1 layerHeights clamped %r" % o["lh"])
        _need(o["uri"] == ["C:\\My Clips\\fox.mp4", "/home/me/a b.webm", "C:\\x\\y.mov", ""], "R1 pathFromUri %r" % o["uri"])
        _need(o["drop"] == ["D:\\v\\clip.mkv", "", ""], "R1 dropPathOf: only a video path %r" % o["drop"])

# ------------------------------------------------------------------ R2
cl = E.clean_layers([{"id": "e1", "src": {"kind": "empty"}}, {"id": "f1", "src": {"kind": "file", "path": "/x/a.mp4"}, "pieces": []},
                     {"id": "base", "src": {"kind": "base"}}])
_need([(x["id"], x["src"]["kind"], x.get("pieces")) for x in cl] == [("e1", "empty", []), ("f1", "file", []), ("base", "base", None)],
      "R2 clean_layers keeps an empty layer and a file layer with every piece deleted")
TMP = Path(tempfile.mkdtemp(prefix="v1044_"))
for d in ("output/cine", "input", "temp"):
    (TMP / d).mkdir(parents=True, exist_ok=True)


class _NT:
    is_nested = True


comfy = types.ModuleType("comfy")
nt = types.ModuleType("comfy.nested_tensor")
nt.NestedTensor = _NT
comfy.nested_tensor = nt
capi = types.ModuleType("comfy_api")
ii = types.ModuleType("comfy_api.input_impl")
ii.VideoFromComponents = object
ut = types.ModuleType("comfy_api.util")
ut.VideoComponents = object
fp = types.ModuleType("folder_paths")
fp.get_output_directory = lambda: str(TMP / "output")
fp.get_input_directory = lambda: str(TMP / "input")
fp.get_temp_directory = lambda: str(TMP / "temp")
sys.modules.update({"comfy": comfy, "comfy.nested_tensor": nt, "comfy_api": capi,
                    "comfy_api.input_impl": ii, "comfy_api.util": ut, "folder_paths": fp})
pkg = types.ModuleType("plsn1044")
pkg.__path__ = [str(ROOT / "nodes")]
sys.modules["plsn1044"] = pkg
TL = importlib.import_module("plsn1044.ph_cine_timeline")
T = sys.modules["plsn1044.cine_timeline"]
md = TL.media_dir("my fox!")
_need(Path(md).is_dir() and Path(md) == TMP / "input" / "cine_media" / "my_fox", "R2 media_dir: input/cine_media/<project>/ made (the name made safe)")
ff = shutil.which("ffmpeg")
if ff:
    clip = TMP / "input" / "cine_media" / "my_fox" / "c.mp4"
    subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=red:s=64x36:r=25", "-frames:v", "50", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)],
                   capture_output=True)
    pr = TL.probe_layer({"kind": "file", "path": "cine_media/my_fox/c.mp4"})
    _need(pr["frames"] == 48 and abs(pr["fps"] - 25) < 0.01 and pr["name"] == "c.mp4" and not pr["audio"], "R2 probe_layer: 50 frames at 25 fps = 48 film frames %r" % pr)
    try:
        TL.probe_layer({"kind": "render", "path": "../input/cine_media/my_fox/c.mp4"})
        _fail("R2 probe_layer refuses a render source outside output/")
    except ValueError:
        _ok("R2 probe_layer refuses a render source outside output/")
    (TMP / "input" / "x.txt").write_text("x")
    try:
        TL.probe_layer({"kind": "file", "path": "x.txt"})
        _fail("R2 probe_layer refuses a non-video")
    except ValueError:
        _ok("R2 probe_layer refuses a non-video")
    subprocess.run([ff, "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=s=64x36:r=24", "-frames:v", "144", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(TMP / "output" / "cine" / "whole_00001_.mp4")], capture_output=True)
    model = T.load(json.dumps({"seconds": 6.0, "project": "cine", "render": "whole", "shots": [{"at": 0, "cut": "the camera cuts to", "text": "x"}]}))
    d = TL.edit_plan(model, {"layers": [{"id": "e1", "src": {"kind": "empty"}}, {"id": "base", "src": {"kind": "base"}}]}, strips=False)
    r0 = d["film"]["layers"][0]
    _need(r0.get("empty") and r0["plan"]["length"] == 0 and d["film"]["complete"] and [x["layer"] for x in d["film"]["flat"]["vruns"]] == [1],
          "R2 stack_of: an empty layer has an empty plan; the render shows everywhere")
else:
    print("  note  ffmpeg not found -- R2 probe SKIPPED")
shutil.rmtree(TMP, ignore_errors=True)
node_py = (ROOT / "nodes" / "ph_cine_timeline.py").read_text(encoding="utf-8")
_need('("GET", "/pls/cine/media_dir", handle_media_dir)' in node_py and '("POST", "/pls/cine/probe", handle_probe)' in node_py
      and '"/pls/cine/media_dir?project="' in js and '"/pls/cine/probe"' in js, "R2 the routes are registered and called")

# ------------------------------------------------------------------ R3
divs = js[js.index("function _dividers("):js.index("function _setLaneH(")]
_need("export const DIV_H = 10;" in js and '"px;cursor:ns-resize;z-index:8;background:#ffffff0d;pointer-events:auto;"' in divs and "z-index:7;" in js
      and 'pill.style.background = "#ffb300";' in divs and "const c = Math.floor(LANE_GAP / 2 - DIV_H / 2);" in divs,
      "R3 the height grip: 10 px, above the cut layer, centred on the gap, a pill that lights up")
_need("export const SIZES = [[\"S\", 44, 26], [\"M\", 90, 40], [\"L\", 150, 60], [\"XL\", 240, 90]];" in js
      and "el.appendChild(_sizeSel(sizeOf(hs), " in js and "el.appendChild(_sizeSel(sizeOf(hs, 2), " in js,
      "R3 the size menus: in every picture and sound head (v1045)")
_need('const HB = "background:transparent;color:#c4c4c4;border:1px solid transparent;border-radius:3px;font-size:12px;line-height:18px;width:22px;height:20px;' in js,   # AMENDED v1150: bigger
      "R3 the head buttons: one fixed size (the arrows as big as the rest)")
_need('return ev.clientY - r.top < r.height / 2 ? "copy" : "pin";' in js and "flex-direction:column;z-index:12;pointer-events:none;" in js
      and "const zone = zoneOf(ev), at = Math.round(_secFromX(pic, ev.clientX" in js, "R3 the drop zones stacked -- upper copy, lower in place, the time is x")
_need('(ev.key === "Delete" || ev.key === "Backspace") && t !== "INPUT" && t !== "TEXTAREA" && t !== "SELECT" && _deleteSel(node)) { ev.preventDefault(); ev.stopPropagation(); }' in js,
      "R3 Delete on a selected piece is stopped at the field (the canvas never deletes the node)")
# AMENDED v1153: pairs by default (Vn over An, together); "V/A by type" = the v1045 order
_need("v1Row, _row(a1Head, audio), below," in js and "st.above.appendChild(r.v);" in js
      and "if (byType(node)) st.below.insertBefore(r.a, st.below.firstChild);   // A2 right under A1" in js
      and "else st.above.appendChild(r.a);" in js,
      "R3 (v1045) the picture rows above V1 top first, the sound rows below A1 (A2 first) -- by type; pairs by default")
_need("const own = _owners(D, t);" in js and "_syncChain(vids, r.plan, tt, playing, own.v === r, own.a === r);" in js,
      "R3 the player shows the stack's picture owner and sounds its sound owner")
_need('g.dataset.pls = "gap";' in js and "if (sg.gap) return;                       // v1044: silence" in js, "R3 the render lane draws a gap as empty, silent")
# v1157: the re-layout lives in _relayStrips (called by the preview fit AND the lanes' own _stripCheck)
_need("_stackRows(node, pct);" in js[js.find("function _relayStrips("):js.find("function _stripCheck(")]
      and "_stripsStale(st)) _relayStrips(node);" in js, "R3 a new width redraws the layer rows too (the strips are in px)")
bad = [f for f in ("web/js/ph_cine_timeline.js", "nodes/ph_cine_timeline.py", "nodes/cine_edit.py", "tests/test_v1044_layer_rows.py")
       if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not bad, "R3 ASCII only %r" % bad)

if FAILS:
    print("test_v1044_layer_rows: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1044_layer_rows: PASS (R1-R3)")
