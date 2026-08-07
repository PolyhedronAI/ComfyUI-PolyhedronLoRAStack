"""v596 guard: THE PANE NEVER OUTGROWS ITS JPEG, AND THE PAIR SITS TOGETHER.

Frank, 2026-07-14 11:25, on v594: "Vorschau immer noch zu gross; das sollte max
512 fuer jede Kante sein ... und auch wieder unscharf."

The tile carried `flex:1`. That means "take everything", and on a node pulled
wider than ~620px it took more than 512 - at which point it was UPSCALING the
probe jpeg, which the backend caps at _PROBE_MAX_EDGE = 512. An upscaled preview
is just a blurrier preview, and the pane was manufacturing its own softness.

The v592 guard pinned `PV_TARGET_W <= _PROBE_MAX_EDGE` and felt safe. That pin
compares two CONSTANTS. It says nothing about the number that actually reaches
the <img>, which is computed at runtime from the node's width. Fifth time this
sprint that a structure was pinned where a VALUE needed measuring.

So this guard does not read the code. It RUNS it.

_procFit is lifted out of the JS, handed a fake node, and swept across node
widths from 300 to 1600 and across portrait, square and landscape sources. For
every single result the invariant must hold:

    tileW <= 512  AND  tileH <= 512

That is the whole promise, and it is checked as a return value, not as a
sentence in a comment. The map's share is checked the same way: it must stay
inside its bounds and it must never be squeezed to nothing - it is an
instrument (v594), and an instrument you cannot read is decoration.
"""
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PU_JS = ROOT / "web" / "js" / "ph_power_upscale.js"
PU_PY = ROOT / "nodes" / "ph_power_upscale.py"

# v597: the cap is NOT a number this guard owns. It is whatever the BACKEND
# actually ships in its jpeg (_PROBE_MAX_EDGE), because that is the physical
# line: above it the pane is upscaling, and an upscaled preview is a blurrier
# preview. PROC_TILE_MAX is taste - Frank moved it 512 -> 384 because a 1:1 pane
# shows every artefact of a T=1 video-vae decode, and downscaling is a low-pass.
# The guard holds the LAW (never above the jpeg) and lets the taste move.
# The guard's OWN floor, not one it reads from the source. The first draft
# checked the map width against PROC_MAP_MIN - a constant it had just parsed
# out of the file it was auditing - so setting PROC_MAP_MIN = 4 produced a
# 4px "minimap" and a green guard: it dutifully confirmed that 4 >= 4. A
# guard that takes its yardstick from the thing being measured is a mirror.
# Below this, an instrument cannot be read, and unreadable is the one thing
# a control instrument may not be.
MAP_READABLE = 64


def _fail(msg):
    print(f"[test_v597_panelayout] FAIL: {msg}")
    sys.exit(1)


def _lift(src, name):
    m = re.search(r"function\s+" + name + r"\s*\([^)]*\)\s*\{.*?\n\}", src, re.S)
    if not m:
        _fail(f"{name}() is gone from the viewer")
    return m.group(0)


def main():
    js = PU_JS.read_text(encoding="utf-8")

    # the constants must exist and be named - a magic number in the arithmetic
    # is a number nobody can find again
    consts = {}
    for k in ("PROC_TILE_MAX", "PROC_MAP_MIN", "PROC_MAP_MAX", "PROC_GAP",
              "PROC_PAD", "PROC_BODY_H", "PROC_HEAD_H", "PROC_MAP_RATIO"):
        m = re.search(r"\b" + k + r"\s*=\s*([\d.]+)", js)
        if not m:
            _fail(f"{k} is gone - the pane's geometry has no name for its own "
                  f"limits")
        consts[k] = float(m.group(1))
    m = re.search(r"\bPROC_NAT_W\s*=", js)
    if not m:
        _fail("PROC_NAT_W is gone - the pane no longer knows the width it needs "
              "to show BOTH panels, so the widen cannot aim at it and the tile "
              "lands at a compromise size on a default node")

    if consts["PROC_MAP_MIN"] < MAP_READABLE:
        _fail(f"PROC_MAP_MIN = {consts['PROC_MAP_MIN']:.0f}, under the "
              f"{MAP_READABLE}px readability floor - the instrument would be a "
              f"smudge at narrow node widths")
    edge = re.search(r"^_PROBE_MAX_EDGE\s*=\s*(\d+)", PU_PY.read_text(encoding="utf-8"), re.M)
    if not edge:
        _fail("_PROBE_MAX_EDGE is gone from the backend - the pane cannot know "
              "the size of the jpeg it is asked to display")
    CAP = int(edge.group(1))
    if consts["PROC_TILE_MAX"] > CAP:
        _fail(f"PROC_TILE_MAX ({consts['PROC_TILE_MAX']:.0f}) exceeds the probe's "
              f"jpeg ({CAP}px) - the pane would upscale every frame it is given. "
              f"An upscaled preview is a blurrier preview, and the pane would be "
              f"manufacturing its own softness (v594's bug).")
    if consts["PROC_TILE_MAX"] < 200:
        _fail(f"PROC_TILE_MAX = {consts['PROC_TILE_MAX']:.0f} - back to a postage "
              f"stamp. The whole sprint started because 160px was too small.")

    # the tile must NOT be allowed to grow by flex - that is what ate the node
    if re.search(r"tile\.style\.cssText[^;]*flex:\s*1", js):
        _fail("the tile still carries flex:1 - it will take whatever width the "
              "node has and upscale the jpeg past the cap (v594's bug). Its box "
              "is computed, not negotiated.")

    # v596: the pair is CENTRED. v595 pinned both boxes to the left of a wide
    # node and dumped the surplus on the right as a black slab - 188px at node
    # 900, 388px at 1100, with the minimap stranded at the far edge of it. The
    # cap was right and the layout was still wrong.
    body = re.search(r"body\.style\.cssText\s*=(.*?);\n", js, re.S)
    if not body or "justify-content:center" not in body.group(1).replace(" ", ""):
        _fail("the process body does not centre its pair. On a node wider than "
              "the two panels need, the surplus must become equal air on BOTH "
              "sides - not a slab on one, with the instrument stranded past it.")
    # ...and each view is its own framed area (Frank: "in einen eigenen Bereich")
    if "_procTileBox" not in js:
        _fail("the tile has no panel of its own - the two views must read as two "
              "areas, not as one image with a smudge next to it")

    # ---- RUN the real thing ------------------------------------------------
    # The constants come from the SOURCE (parsed above), not retyped here. A
    # harness that carries its own copy of the numbers is testing the harness.
    prelude = "\n".join(
        f"const {k} = {int(v) if float(v).is_integer() else v};"
        for k, v in consts.items())
    harness = prelude + "\n" + _lift(js, "_procFit") + """
const CAP = %d;
const out = [];
for (let w = 300; w <= 1600; w += 20) {
  for (const [iw, ih] of [[768,768],[480,832],[1280,720],[1104,832],[512,512],[96,96]]) {
    const node = {
      size: [w, 0],
      _procOpen: true, _procH: 0, _procBodyH: 0,
      _procTile: { naturalWidth: iw, naturalHeight: ih, style: {} },
      _procMapBox: { style: {} },
      computeSize: () => [w, 0],
      setSize: () => {},
      setDirtyCanvas: () => {},
    };
    _procFit(node);
    out.push({
      w, iw, ih,
      tw: parseInt(node._procTile.style.width, 10),
      th: parseInt(node._procTile.style.height, 10),
      mw: parseInt(node._procMapBox.style.width, 10),
      bh: node._procBodyH,
    });
  }
}
console.log(JSON.stringify(out));
""" % CAP

    r = subprocess.run(["node", "-e", harness], capture_output=True, text=True)
    if r.returncode != 0:
        _fail(f"_procFit could not be executed: {r.stderr.strip()[:200]}")
    rows = json.loads(r.stdout)

    # ---- THE INVARIANT -----------------------------------------------------
    for row in rows:
        if not (row["tw"] and row["th"] and row["mw"]):
            _fail(f"_procFit left a dimension unset at node {row['w']}px, "
                  f"source {row['iw']}x{row['ih']}: {row}")
        if row["tw"] > CAP or row["th"] > CAP:
            _fail(f"THE LAW: node {row['w']}px, source {row['iw']}x{row['ih']} -> "
                  f"tile {row['tw']}x{row['th']}, past the probe's jpeg ({CAP}px). "
                  f"The pane is upscaling, which is the one thing it may never do.")
        if (row["tw"] > consts["PROC_TILE_MAX"]
                or row["th"] > consts["PROC_TILE_MAX"]):
            _fail(f"the geometry ignores its own cap: node {row['w']}px -> tile "
                  f"{row['tw']}x{row['th']}, but PROC_TILE_MAX is "
                  f"{consts['PROC_TILE_MAX']:.0f}")
        if row["mw"] < MAP_READABLE:
            _fail(f"the minimap is {row['mw']}px wide at node {row['w']}px - "
                  f"under the {MAP_READABLE}px floor this guard owns. It is a "
                  f"control INSTRUMENT (v594): squeezed to a strip it cannot be "
                  f"read, and being read is its entire job. (This floor is the "
                  f"guard's, deliberately - checking against PROC_MAP_MIN would "
                  f"just confirm that the source agrees with itself.)")
        if row["mw"] > consts["PROC_MAP_MAX"]:
            _fail(f"the minimap is {row['mw']}px at node {row['w']}px, past its "
                  f"own ceiling ({consts['PROC_MAP_MAX']:.0f}) - it is taking "
                  f"width the preview needs")
        if row["bh"] < consts["PROC_BODY_H"]:
            _fail(f"body height {row['bh']} fell below its floor "
                  f"({consts['PROC_BODY_H']:.0f})")

    # a wide node must actually REACH the cap - a pane that stays small forever
    # satisfies the invariant and defeats the point
    tmax = consts["PROC_TILE_MAX"]
    wide = [r_ for r_ in rows if r_["w"] >= 900 and (r_["iw"], r_["ih"]) == (768, 768)]
    if not wide or max(max(r_["tw"], r_["th"]) for r_ in wide) != tmax:
        _fail(f"no node width ever reaches the {tmax:.0f}px cap on a square "
              f"source - the pane is capped somewhere else and the ceiling is "
              f"decoration")

    hit = min(r_["w"] for r_ in rows
              if (r_["iw"], r_["ih"]) == (768, 768) and r_["tw"] >= tmax)
    print(f"PASS: v597 -- {len(rows)} geometries swept (node 300..1600px x 6 "
          f"aspects): no tile edge passes PROC_TILE_MAX ({tmax:.0f}px) and none "
          f"ever passes the probe jpeg ({CAP}px - the law); the cap is reached "
          f"from node width {hit}px; the minimap stays readable "
          f"({consts['PROC_MAP_MIN']:.0f}..{consts['PROC_MAP_MAX']:.0f}px); the "
          f"pair is centred and each view has its own framed panel")


if __name__ == "__main__":
    main()
