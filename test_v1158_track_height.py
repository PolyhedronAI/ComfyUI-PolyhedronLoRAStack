# -*- coding: ascii -*-
"""Guard v1158 -- a new track starts at V1's height, not at the smallest size.

Frank 17.10.: the clips that came in "not in the minimal view" -- a new track came at S (44 / 28 px) while V1 stood
at M; its pictures and waveform were the smallest on screen.

  H1  _seedTrackH: stores laneHeights(node) (V1 / A1 now) for a track WITHOUT a height, clamped by layerHeights;
      a track that has one keeps it
  H2  every place a track is born calls it: a run's clip / takes -> tracks (_renderTrack), a drop or a file into
      a new track (_loadInto, NEW_TRACK), "+ new track"
  H3  the seed is written before the edit is saved (the save renders the new rows at their height)
  H4  the defaults stay (an old track without a stored height is still S): LAYER_CLIP_H 44, LAYER_AUDIO_H 28
  S   ASCII
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1158_track_height"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


P = ROOT / "web" / "js" / "ph_cine_timeline.js"
JS = P.read_text(encoding="utf-8")


def body(name):
    for pre in ("function ", "async function "):
        i = JS.find(pre + name + "(")
        if i >= 0 and (i == 0 or JS[i - 1] in "\n "):
            return JS[i:JS.find("\n}", i) + 2]
    return ""


seed = body("_seedTrackH")
_need(bool(seed) and "if (all[id]) return;" in seed and "const v1 = laneHeights(node);" in seed
      and "all[id] = { clip: v1.clip, audio: v1.audio };" in seed and "node.properties[LH_KEY][id] = layerHeights(node, id);" in seed,
      "H1 _seedTrackH stores V1's heights for a track without one (clamped), keeps an existing one")

rt = body("_renderTrack")
_need(re.search(r"const e2 = addLayer\(e, id, r\.name\);\s*_seedTrackH\(node, id\);", rt) is not None,
      "H2 a run's clip / takes -> tracks: the new track is seeded (_renderTrack)")
li = body("_loadInto")
_need("if (id === NEW_TRACK) { id = newLayerId(e); e = addLayer(e, id); _seedTrackH(node, id); }" in li,
      "H2 a drop / a file into a new track is seeded (_loadInto)")
_need("node._plsTl.active = id; _seedTrackH(node, id); _saveEdit(node, addLayer(ed, id));" in JS,
      "H2 '+ new track' is seeded")
_need(JS.count("_seedTrackH(node, id)") == 4, "H2 exactly the three birth places + the definition's caller count (4)")

_need(li.find("_seedTrackH(node, id)") < li.find("_saveEdit(") and rt.find("_seedTrackH") < rt.find("return setLayer"),
      "H3 the seed comes before the save / the returned edit")

_need("export const HEAD_H = 18, LAYER_CLIP_H = 44, LAYER_AUDIO_H = 28," in JS, "H4 the stored-height defaults unchanged (old tracks stay as they were)")

_need(all(b < 128 for b in P.read_bytes()) and all(b < 128 for b in Path(__file__).read_bytes()), "S  ASCII")

print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
