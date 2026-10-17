# -*- coding: ascii -*-
"""Guard v1039 -- the cutting desk cleaned up (Frank, 29.09., three
screenshots): "clean everything up first; that I can drag the cut
afterwards without it being a real cut through the tracks does not seem
right either; spreading the clip works, but at the back it is one big
clutter to grab things cleanly."

  H  the heights: ONE place sets the lanes' heights (_applyLanes), first
     thing in _clipLane, before a box is drawn -- the boxes are drawn for
     exactly the lane they sit in (screenshot 1: 190 px pictures over a
     44 px lane painted over the sound lane and the rows below); the lanes
     clip what reaches past them; _setLaneH goes through the same place.
  C  the cut: one layer over the picture AND the sound lane; every cut is a
     picture part + a sound part on the same x, one under the other (one
     line through both tracks); linked, a drag of either part moves both
     LIVE; unlinked / Alt, the part alone, the dashed bracket ties them; the
     film's end is a line through both lanes; the diamond carries its line;
     the sound lane has no grips of its own any more.
  E  the end: room after the film (viewAxis: +0.25 s or 4 %, never less);
     the height grips left the corner -- dividers in the gap under each lane,
     full width; the transition chip lies over the cut's line; the legend is
     one line (a narrow node wrapped it to seven and pushed the rows out).
  J  viewAxis (node.js).

Script-style: exit 0 = pass.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    print(("  ok    " if c else "  FAIL  ") + m)
    if not c:
        FAILS.append(m)


js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")


def body(name, nxt):
    return js[js.index("function %s(" % name):js.index("function %s(" % nxt)]


clip = body("_clipLane", "_applyLanes")
apply_ = body("_applyLanes", "_cutLayer")
layer = body("_cutLayer", "_diamond")
dia = body("_diamond", "_dividers")
divs = body("_dividers", "_placeDividers")
setl = body("_setLaneH", "_audioLane")
aud = body("_audioLane", "_cutButtons")
field = body("_field", "_field") if False else js[js.index("function _field("):]

# ------------------------------------------------------------------ H
_need(clip.index("const LH = _applyLanes(node);") < clip.index("lane.replaceChildren();") < clip.index('box.dataset.pls = "clip";'),
      "H the lanes take their heights first, then the boxes are drawn")
_need('st.clips.style.height !== LH.clip + "px"' in apply_ and 'st.audio.style.height !== LH.audio + "px"' in apply_
      and "_layout(node);" in apply_ and "st.root.style.height = fh;" in js and "refit(node)" in js[js.index("function _layout("):js.index("function _fitHead(")],
      "H _applyLanes sets both lanes and the field, and refits the node")   # AMENDED v1150: through _layout
setters = [m.start() for m in re.finditer(r"st\.clips\.style\.height\s*=", js)]
_need(len(setters) == 1 and js.index("function _applyLanes(") < setters[0] < js.index("function _cutLayer("),
      "H exactly one place writes the picture lane's height (%d)" % len(setters))
_need("_applyLanes(node);" in setl and "_placeDividers(node);" in setl, "H a divider drag goes through the same place")
_need('lane + "height:" + LANE_CLIP_H + "px;background:#121212;overflow:hidden;"' in field
      and 'lane + "height:" + LANE_AUDIO_H + "px;background:#101418;overflow:hidden;"' in field, "H the lanes clip what reaches past them")

# ------------------------------------------------------------------ C
_need("tl.append(head, headGrip, cuts);" in field and "v1Row, _row(a1Head, audio), below," in field   # v1045: the overlay right of the heads; AMENDED v1153: v1Row
      and 'cuts.dataset.pls = "cutlayer";' in field and "pointer-events:none;z-index:7;" in field, "C one cut layer over both lanes")
_need('const pic = part(0, picH, at, col, "cutgrip");' in layer and 'const snd = part(sndTop, sndH, aat, ao ? "#ffcc80" : col, "soundcut");' in layer
      and "const picH = LH.clip, sndTop = LH.clip, sndH = LANE_GAP + LH.audio;" in layer,
      "C a cut = its picture part (the picture lane) + its sound part right under it (the gap + the sound lane)")
_need('if (who === "pic") { pic.style.left = pct(f / FPS); if (lk) snd.style.left = pct((f + ao) / FPS); }' in layer
      and 'else { snd.style.left = pct(f / FPS); if (lk) pic.style.left = pct((f - ao) / FPS); }' in layer
      and 'follow("pic", f, alt);' in layer and 'follow("snd", f, alt);' in layer,
      "C linked: either part moves both, live; unlinked / Alt: the part alone")
_need('br.dataset.pls = "cutbracket";' in layer and "border-top:2px dashed #ffcc80" in layer, "C a J-/L-cut: the dashed bracket ties the two parts")
_need('part(0, picH, end, "#b0bec5", "endgrip")' in layer and 'part(sndTop, sndH, end, "#b0bec5", "endline")' in layer
      and "ep.style.left = es.style.left = pct(sec);" in layer, "C the film's end is one line through both lanes")
_need("kn.style.left = pic.style.left = pct((st0[j] + len) / FPS);" in dia and "snd.style.left = pct((st0[j] + len + ao) / FPS);" in dia,
      "C the diamond carries its line (the piece's end is the cut)")
_need('"soundcut"' not in aud and "_drag(" not in aud and "cutgrip" not in clip, "C the lanes have no cut grips of their own any more")

# ------------------------------------------------------------------ E
# AMENDED v1151: the monitor's transport row (v1150) -- the CSS on its own line, so v755's spelling scan (it reads
# lines that mention a tooltip) does not take the CSS property for prose
CTL_CSS = 'gap:6px;align-items:center;font-size:12px;color:#aaa;min-width:0;flex:0 0 auto;overflow:hidden;"'
_need("const ax = viewAxis(axisOf(m, st.plan));" in js and "export function viewAxis(ax)" in js, "E the axis has room after the film's end")
_need('"position:absolute;left:0;right:0;height:" + DIV_H + "px;cursor:ns-resize;z-index:8;' in divs   # v1044: a grip you see, above the cuts
      and 'd.dataset.pls = which === "clip" ? "clipheight" : "audioheight";' in divs and "right:0;bottom:0;width:16px" not in js
      and "function _laneGrip(" not in js, "E the height grips are dividers in the gap (full width); the corner grips are gone")
_need("translateX(-50%);z-index:8;cursor:pointer;" in clip, "E the transition chip lies over the cut's line (8 > the layer's 7)")
_need("white-space:nowrap;overflow:hidden;text-overflow:ellipsis;min-width:0;flex:0 1 auto;" in js and "e.title = t;" in js
      and CTL_CSS in js, "E the legend is one line (its text in the tooltip); the player's column clips")   # AMENDED v1150
_need("the divider under a lane = its height" in js and "the grip bottom right" not in js, "E the help line names the divider")

# ------------------------------------------------------------------ J
if shutil.which("node") is None:
    print("  note  node.js not found -- J SKIPPED")
else:
    t2 = Path(tempfile.mkdtemp(prefix="pls1039_"))
    b = (js.replace('import { app } from "../../scripts/app.js";', "const app = { registerExtension() {} };")
           .replace('import { api } from "../../scripts/api.js";', "const api = {};")
           .replace('import { setHidden, refit } from "./ph_widget_vis.js";', "const setHidden = () => {}, refit = () => {};"))
    (t2 / "m.mjs").write_text(b, encoding="utf-8")
    (t2 / "h.mjs").write_text('import * as T from "./m.mjs"; console.log(JSON.stringify([0, 2, 6.58, 30].map(T.viewAxis)));', encoding="utf-8")
    r = subprocess.run(["node", str(t2 / "h.mjs")], capture_output=True, text=True, timeout=60)
    v = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else None
    _need(v and abs(v[0] - 0.25) < 1e-9 and abs(v[1] - 2.25) < 1e-9 and abs(v[2] - 6.58 * 1.04) < 1e-9 and abs(v[3] - 31.2) < 1e-9,
          "J viewAxis: +0.25 s, or 4 %% when longer %r" % (v,))
    shutil.rmtree(t2, ignore_errors=True)

if FAILS:
    print("test_v1039_cut_layer: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1039_cut_layer: PASS (H, C, E, J)")
