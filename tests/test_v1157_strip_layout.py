# -*- coding: ascii -*-
"""Guard v1157 -- the timeline's film strips are drawn again once the lanes have their width.

Frank 17.10.: the clips that came in showed no pictures (V2..: an empty box with the name; V1: the strip squeezed
over the material) -- also after a reload. Measured: the plan came while the lanes were 0 px wide (the node off
screen, a reload, the first frames after a load); the strips were laid out for width 0, and the only re-layout hung
on the PREVIEW's width check, whose width had not changed (lanes 1356 px, laneW still 0).

  S1  _stripsStale: the lane's width against BOTH widths the strips were laid out for (V1: laneW, V2..: layW),
      only when the lane has a width
  S2  _stackRows records layW; _clipLane records laneW
  S3  after every plan (_stack): _stripCheck(node, true) after _render
  S4  the lanes coming on screen: the IntersectionObserver watches the scroller too and calls _stripCheck
  S5  every pointer release: _stripCheck next to _widthCheck
  S6  _stripCheck: one-shot frames, at most 10 retries while not laid out, skips while a zoom's render is pending
  S7  the preview fit's re-layout goes through the same two helpers
  S8  a layer row not laid out yet shows the strip stretched (as V1) -- never an empty track
  S9  no ResizeObserver, no onResize (v610 / test_v1030)
  S   ASCII
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1157_strip_layout"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


P = ROOT / "web" / "js" / "ph_cine_timeline.js"
JS = P.read_text(encoding="utf-8")


def body(name):
    i = JS.find("function " + name + "(")
    if i < 0:
        i = JS.find("async function " + name + "(")
    if i < 0:
        return ""
    j = JS.find("\n}", i)
    return JS[i:j + 2]


stale = body("_stripsStale")
_need(bool(stale) and "st.clips.clientWidth" in stale and "w > 0 &&" in stale
      and "(st.laneW || 0)" in stale and "(st.layW || 0)" in stale and "> 2" in stale,
      "S1 _stripsStale compares the lane width with laneW AND layW, only when laid out")

rows, lane = body("_stackRows"), body("_clipLane")
_need("st.layW = st.clips.clientWidth || 0;" in rows, "S2 _stackRows records the width its strips are laid out for")
_need("st.laneW = laneW;" in lane, "S2 _clipLane records laneW")

stk = body("_stack")
_need(re.search(r"_render\(node\);\s*_playerDraw\(node\);\s*_stripCheck\(node, true\);", stk) is not None,
      "S3 after every plan: _stripCheck(node, true) after the render")

_need(re.search(r"new IntersectionObserver\(\(es\) => \{ if \(es\.some\(\(e\) => e\.isIntersecting\)\) \{ _widthCheck\(node, true\); _stripCheck\(node, true\); \} \}\);", JS) is not None
      and "io.observe(scroller);" in JS and "io.observe(prev);" in JS,
      "S4 the IntersectionObserver watches the lanes (scroller) too and checks the strips")

_need("_widthCheck(n, true); _stripCheck(n, true);" in JS, "S5 every pointer release checks the strips too")

chk = body("_stripCheck")
_need(bool(chk) and "st.stripQueued" in chk and "st.zoomTimer" in chk and "(st.stripTries || 0) < 10" in chk
      and "if (fresh) st.stripTries = 0;" in chk and "_relayStrips(node)" in chk and "_stripsStale(st)" in chk,
      "S6 _stripCheck: queued once, bounded retries, skips a pending zoom, re-lays only when stale")

fit = body("_fitPreview")
_need("_stripsStale(st)) _relayStrips(node);" in fit and "_stackRows(node, (x) =>" not in fit,
      "S7 the preview fit re-lays the strips through the same helpers")
rel = body("_relayStrips")
_need("_stackRows(node, pct);" in rel and "_clipLane(node, _model(node), pct);" in rel, "S7 _relayStrips draws V1 and the layer rows")

lr = body("_layerRows")
_need(re.search(r"\} else if \(url && sg\.frames\) \{[^}]*background-size:100% 100%", lr, re.S) is not None,
      "S8 a layer row not laid out yet shows the stretched strip (never an empty track)")

_need("new ResizeObserver" not in JS and re.search(r"\.onResize\s*=", JS) is None, "S9 no ResizeObserver, no onResize")

raw = P.read_bytes()
_need(all(b < 128 for b in raw), "S  ph_cine_timeline.js is ASCII")
_need(all(b < 128 for b in Path(__file__).read_bytes()), "S  this guard is ASCII")

print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
