# -*- coding: ascii -*-
"""Guard v1162 -- a pinned frame survives the Power Upscale's refine; the timeline sees the pictures downstream.

Frank 18.10., the neon lane (Reference, five scenes, carry): scene 2 died in the Power Upscale's joint refine --
'shape mismatch: value tensor of shape [2801, 96] cannot be broadcast to indexing result of shape [3799, 96]'.
Scene 2 carries the last frame of scene 1: MiniMax Reference puts it in the conditioning as minimax_keyframes,
a latent ENCODED AT THE RENDER CANVAS (768x1344). The Power Upscale reused that conditioning for its refine at
1072x1872. Scene 1 (no pin) ran; every carried scene died; no upscale, no interpolation, no final clip.

  P1  _pins_of: the pinned frame indices of a conditioning (both sides), [] when none
  P2  _repin_conds: every pin re-encoded by encode_at(i) on the refine's canvas; every other key kept
      (the merge marker); the input never changed in place
  P3  a pin encode_at cannot give is dropped; the last pin takes the frame count with it; a conditioning
      without pins passes through as it is
  P4  the shapes: the re-encoded pin has the refine canvas's latent size, not the render's
  P5  _joint_refine re-encodes the pins from the frames it refines (v_enc on src[i:i+1]) BEFORE it samples,
      and says so; _refine_tiles leaves pins out (a tile is not the frame) and says so
  T1  the timeline (Reference mode) lists the pictures wired at the MiniMax Reference its prompt flows into
      -- @image_k in wired order, through the CLIP Text Encode -- and only falls back to them when it has none
      of its own; the "no reference pictures" hint is gone when they exist
  S   ASCII of this guard
"""
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1162_upscale_pins"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


SRC = (ROOT / "nodes" / "ph_power_upscale.py").read_text(encoding="utf-8")
tree = ast.parse(SRC)


def _seg(name):
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(SRC, n)
    return ""


ns = {"PIN_KEY": "minimax_keyframes", "PIN_COUNT": "minimax_frame_count"}
_need('PIN_KEY = "minimax_keyframes"' in SRC and 'PIN_COUNT = "minimax_frame_count"' in SRC, "P0 the keys are Core's / MiniMax Reference's")
exec(_seg("_pins_of") + "\n\n" + _seg("_repin_conds"), ns)
pins_of, repin = ns["_pins_of"], ns["_repin_conds"]


class Lat(object):
    """A latent stand-in: its shape is what matters."""
    def __init__(self, w, h):
        self.shape = (1, 24, 1, h // 16, w // 16)


T = object()
pos = [[T, {"minimax_refs": ["r"], "minimax_keyframes": [{"resolved_frame_index": 0, "latent": Lat(768, 1344),
                                                          "pls_with_refs": True}],
            "minimax_frame_count": 141}]]
neg = [[T, {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": Lat(768, 1344)},
                                  {"resolved_frame_index": 140, "latent": Lat(768, 1344)}],
            "minimax_frame_count": 141}]]
plain = [[T, {"pooled_output": None}]]

# P1
_need(pins_of(pos) == [0] and pins_of(neg) == [0, 140] and pins_of(plain) == [] and pins_of(None) == [],
      "P1 _pins_of reads the pinned frame indices, [] without pins")

# P2 + P4
asked = []


def enc_refine(i):
    asked.append(i)
    return Lat(1056, 1856)   # v1164: the joint grid is /32 (H3: VAE 16 x patch 2)


before = json.dumps(pos, default=lambda o: "obj")
p2, notes = repin(pos, enc_refine)
kf = p2[0][1]["minimax_keyframes"][0]
_need(asked == [0] and not notes, "P2 the pin is re-encoded by encode_at(its frame index)")
_need(kf["pls_with_refs"] is True and p2[0][1]["minimax_refs"] == ["r"] and p2[0][1]["minimax_frame_count"] == 141
      and kf["resolved_frame_index"] == 0 and p2[0][0] is T, "P2 every other key is kept (merge marker, refs, frame count)")
_need(json.dumps(pos, default=lambda o: "obj") == before and pos[0][1]["minimax_keyframes"][0]["latent"].shape[3] == 84,
      "P2 the input conditioning is not changed in place")
_need(kf["latent"].shape[3:] == (116, 66) and pos[0][1]["minimax_keyframes"][0]["latent"].shape[3:] == (84, 48),
      "P4 the pin now has the refine canvas's latent size (116x66, even), not the render's (84x48)")

# P3
p3, notes3 = repin(pos, None)
_need(p3 is not None and "minimax_keyframes" not in p3[0][1] and "minimax_frame_count" not in p3[0][1]
      and p3[0][1]["minimax_refs"] == ["r"] and notes3 == ["pinned frame 0 dropped (not on this canvas)"],
      "P3 no encoder: the pin is dropped and said, the frame count goes with the last pin, refs stay")
p3b, notes3b = repin(neg, lambda i: Lat(1072, 1872) if i < 100 else None)
_need([k["resolved_frame_index"] for k in p3b[0][1]["minimax_keyframes"]] == [0]
      and p3b[0][1]["minimax_frame_count"] == 141 and len(notes3b) == 1,
      "P3 a pin beyond the frames is dropped, the one that fits stays with the frame count")
p3c, n3c = repin(plain, enc_refine)
_need(p3c[0] is plain[0] and not n3c and repin(None, enc_refine) == (None, []),
      "P3 a conditioning without pins passes through untouched")

# P5
jr = _seg("_joint_refine")
i_rep, i_smp = jr.find("_repin_conds(positive, _enc_at)"), jr.find("comfy.sample.sample_custom(")
_need(0 < i_rep < i_smp and "_repin_conds(negative, _enc_at)" in jr and "v_enc(src[i:i + 1, :, :, :3])" in jr
      and "re-encoded for the refine canvas" in jr,
      "P5 _joint_refine re-encodes the pins from the frames it refines, before it samples, and says so")
rt = _seg("_refine_tiles")
_need("_repin_conds(positive, None)" in rt and "_repin_conds(negative, None)" in rt and "left out of the tile refine" in rt,
      "P5 _refine_tiles leaves pins out and says so")

# T1 -- the timeline's downstream lookup, run in node against a stub graph
JS = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")


def body(name):
    i = JS.find("function " + name + "(")
    j = JS.find("\n}", i)
    return JS[i:j + 2] if i >= 0 else ""


down, link = body("refItemsDown"), body("_link")
row = body("_refsRow")
_need('export const REF_NODE = "ULSMiniMaxReference", DOWN_HOPS = 4;' in JS and bool(down),
      "T1 refItemsDown exists (MiniMax Reference, a few hops)")
_need(re.search(r"let items = refItems\(node\);\s*if \(!items\.length && md === \"ref\"\) items = refItemsDown\(node\);", row)
      is not None, "T1 _refsRow falls back to the downstream pictures in Reference mode only")
node = shutil.which("node")
if node and down and link:
    harness = r"""
const app = { graph: null };
const REF_NODE = "ULSMiniMaxReference", DOWN_HOPS = 4;
function _thumbOf(s) { return "thumb:" + s.id; }
%s
%s
const links = {};
let lid = 1;
const nodes = {};
function N(id, type, title) { nodes[id] = { id, type, title, inputs: [], outputs: [] }; return nodes[id]; }
function wire(a, ao, b, bname) {
  const A = nodes[a], B = nodes[b];
  while (A.outputs.length <= ao) A.outputs.push({ links: [] });
  let inp = B.inputs.find((x) => x.name === bname); if (!inp) { inp = { name: bname, link: null }; B.inputs.push(inp); }
  links[lid] = { id: lid, origin_id: a, target_id: b }; A.outputs[ao].links.push(lid); inp.link = lid; lid++;
}
const g = { links, getNodeById: (i) => nodes[i] };
const tl = N(1, "ULSCineTimeline"); tl.graph = g;
N(2, "ULSCLIPTextEncode"); N(3, "ULSMiniMaxReference"); N(11, "ULSMediaLoader", "ML a"); N(12, "ULSMediaLoader", "ML b"); N(13, "ULSMediaLoader", "ML c");
for (const n of Object.values(nodes)) n.graph = g;
nodes[3].inputs.push({ name: "image_1", link: null });   // a free slot first: numbering follows the WIRED order
wire(1, 0, 2, "pos_external"); wire(2, 2, 3, "prompt");
wire(12, 0, 3, "image_2"); wire(11, 0, 3, "image_3"); wire(13, 0, 3, "image_5");
const got = refItemsDown(tl);
const lone = N(20, "ULSCineTimeline"); lone.graph = g;
console.log(JSON.stringify({ got, lone: refItemsDown(lone) }));
""" % (link.replace("export ", ""), down.replace("export ", ""))
    d = tempfile.mkdtemp(prefix="v1162_")
    f = os.path.join(d, "h.js")
    open(f, "w").write(harness)
    r = subprocess.run([node, f], capture_output=True, text=True, timeout=60)
    shutil.rmtree(d, ignore_errors=True)
    try:
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception:
        res = None
        print(r.stdout, r.stderr)
    got = res and res["got"]
    _need(got is not None and [x["name"] for x in got] == ["@image_2", "@image_3", "@image_5"]
          and [x["title"].split(" ")[0] for x in got] == ["<Picture", "<Picture", "<Picture"]
          and got[0]["title"].startswith("<Picture 1>") and got[2]["title"].startswith("<Picture 3>")
          and got[1]["url"] == "thumb:11",
          "T1 through the CLIP Text Encode: @image_2/3/5 as <Picture 1..3> in wired order, with their thumbnails")
    _need(res is not None and res["lone"] == [], "T1 a timeline whose prompt reaches no MiniMax Reference lists nothing")
else:
    _need(False, "T1 node.js is needed for the timeline harness")

_need(all(b < 128 for b in Path(__file__).read_bytes()), "S  ASCII")
print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
