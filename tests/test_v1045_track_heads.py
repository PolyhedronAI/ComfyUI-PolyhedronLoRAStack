# -*- coding: ascii -*-
"""Guard v1045 -- the cutting room as the editors lay it out (Frank 30.09.,
a Premiere screenshot: "roughly like this, plus the up / down triangles";
"a new track may be more prominent ... everything tidier").

  H1  the model: the render is V1 / A1 -- clean_layers puts it at the bottom
      of the stack (its switches kept); solo -- when a track is solo, only
      the solo tracks sound (flatten), the pictures are untouched.
  H2  the layout: every row = a head column (GUT_W) + its lane; the time axis
      (playhead, cuts, grips) in one overlay right of the heads; the ruler's
      head shows the timecode; Scenes / Shots / Anchors named in theirs.
  H3  the heads: a picture track = lock, Vn, eye, up, down, reset, x, size;
      a sound track = lock, An, M, S, size; the name chip blue on the active
      track; the render cannot move or go.
  H4  the new track: its own row -- a button in the head, a lane that takes a
      dropped video (it opens the track AND loads the clip); the old "+ layer"
      by the player is gone; a new node is GUT_W wider.
  S   ASCII.

Script-style: exit 0 = pass.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


spec = importlib.util.spec_from_file_location("ce1045", str(ROOT / "nodes" / "cine_edit.py"))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
js = (ROOT / "web" / "js" / "ph_cine_timeline.js").read_text(encoding="utf-8")

# ------------------------------------------------------------------ H1
cl = E.clean_layers([{"id": "base", "src": {"kind": "base"}, "audio": False}, {"id": "e1", "src": {"kind": "empty"}, "solo": True}])
_need([x["id"] for x in cl] == ["e1", "base"] and cl[1]["audio"] is False and cl[0].get("solo") is True,
      "H1 the render goes to the bottom (V1 / A1), its switches kept; solo kept")
_need(E.clean_layers([{"id": "base", "src": {"kind": "base"}, "solo": True}]) != [], "H1 a render alone on solo is a stack")
tr = lambda a, b: {"segments": [{"film_in": a, "film_out": b}], "asegments": [{"film_in": a, "film_out": b}]}
L = [dict(video=True, audio=True, has_audio=True, track=tr(0, 4)), dict(video=True, audio=True, has_audio=True, solo=True, track=tr(0, 8))]
fl = E.flatten(L, 8)
_need([(r["film_in"], r["film_out"], r["layer"]) for r in fl["aruns"]] == [(0, 8, 1)]
      and [(r["film_in"], r["film_out"], r["layer"]) for r in fl["vruns"]] == [(0, 4, 0), (4, 8, 1)],
      "H1 solo on the lower track: only it sounds (the upper is silent), the pictures unchanged")

# ------------------------------------------------------------------ H2
_need("export const GUT_W = 216, NEWROW_H = 22;" in js and 'const tl = _el("div", "position:absolute;top:0;bottom:0;right:0;left:" + GUT_W + "px;pointer-events:none;z-index:6;");' in js
      and "tl.append(head, headGrip, cuts);" in js, "H2 a head column; the time axis overlay starts right of it")
_need("lanes.append(_row(tc, ruler), _row(gut(\"Scenes\"), scenes), _row(gut(\"Shots\"), shots), _row(newHead, newLane), above," in js
      and "v1Row, _row(a1Head, audio), below, anchorsRow, tl);" in js and 'const anchorsRow = _row(gut("Anchors"), anchors);' in js,   # AMENDED v1150, v1153 (v1Row)
      "H2 every row = head + lane: ruler (timecode), Scenes, Shots, new track, V tracks, V1, A1, A tracks, Anchors")
_need("if (st.tc) st.tc.textContent = timecode(pl.t || 0);" in js and 'tc.dataset.pls = "timecode";' in js, "H2 the ruler's head shows the playhead's timecode")
_need("const st = node._plsTl, box = st.tl || st.lanes;" in js, "H2 the render lanes' height grips live in the overlay too")

# ------------------------------------------------------------------ H3
vh = js[js.index("function _vHead("):js.index("function _aHead(")]
ah = js[js.index("function _aHead("):js.index("function _resetLayer(")]
order = ['"lock")', '_chip("V" + num, act)', '"eye")', '"up")', '"down")', '"reset")', '"remove")', "_sizeSel(sizeOf(hs)"]
pos = [vh.find(x) for x in order]
_need(all(p >= 0 for p in pos) and pos == sorted(pos), "H3 the picture head: lock, Vn, eye, up, down, reset, x, size %r" % pos)
order = ['"lock")', '_chip("A" + num, act)', '"mute")', '"solo")', "_sizeSel(sizeOf(hs, 2)"]
pos = [ah.find(x) for x in order]
_need(all(p >= 0 for p in pos) and pos == sorted(pos), "H3 the sound head: lock, An, M, S, size %r" % pos)
_need('(act ? "background:#2d6ec9;color:#fff;" : "background:#2a2a2a;color:#9a9a9a;")' in js, "H3 the active track's chip is blue")
_need('_hb("\\u25b2", base ? "the render stays V1"' in vh and '!base && !lay.lock, () => {' in vh
      and "L[i].id === BASE_ID || L[j].id === BASE_ID) return _withLayers(e, L);" in js, "H3 the render cannot move or be deleted; no move crosses it")

# ------------------------------------------------------------------ H4
_need('const addBtn = _el("button", "margin-left:6px;background:#1f3550;color:#9fc8ff;' in js and '"+ new track");' in js
      and 'mk("+ layer"' not in js, "H4 '+ new track' in its own row's head, prominent; the old button is gone")
_need("_dropZone(node, newLane, (dt, zone, at) => _dropNewTrack(node, dt, zone, at));" in js
      # v1082 (DECLARED): the track is opened by the load, after the probe -- a refused drop leaves no layer
      and "return _dropInto(node, NEW_TRACK, dt, zone, at);" in js
      # v1159: the new track also takes V1's height there (v1158 _seedTrackH) -- the needle ends before it
      and "if (id === NEW_TRACK) { id = newLayerId(e); e = addLayer(e, id);" in js,
      "H4 a video dropped on the new-track row opens a track and loads it")
_need("if (this.size && this.size[0] < 680 + GUT_W) this.setSize([680 + GUT_W, this.size[1]]);" in js, "H4 a new node is wider by the heads")

bad = [f for f in ("web/js/ph_cine_timeline.js", "nodes/cine_edit.py", "tests/test_v1045_track_heads.py")
       if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not bad, "S ASCII only %r" % bad)
if FAILS:
    print("test_v1045_track_heads: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1045_track_heads: PASS (H1-H4, S)")
