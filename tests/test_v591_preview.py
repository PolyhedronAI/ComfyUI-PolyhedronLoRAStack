"""v591 guard: THE PREVIEW GROWS TO THE SAMPLER'S SIZE, AND THE PROCESS VIEW
BOWS OUT WHEN THE WORK IS DONE.

Frank's ask, 2026-07-14, with a screenshot: the Power Upscale result viewer
showed the same frames SMALLER than the render sampler does. The reason is
structural, not cosmetic - the sampler rides ComfyUI's native `node.imgs` +
`setSizeForImage`, which sizes the NODE to the image (~512 px on his 768x768
runs). The Power Upscale viewer is a DOM widget: it inherited whatever width
the node happened to carry (~390 px on his canvas) and never asked for more.

Two laws:

1. The first payload of a run WIDENS a too-narrow node to PV_TARGET_W. It never
   SHRINKS one - the v531 law stands, a node pulled wider stays wider. The
   aspect ratio keeps coming from the frames, so the target is a WIDTH, not a
   box: a 16:9 render lands 512x288, not a squashed square.

2. The process view folds away when the result frames land, and RE-ARMS. It is
   a window into work that is happening; once the work is over it costs node
   height to show a frozen tile and a HUD reading "Chunk 9/9". An errored run
   keeps it (no result payload arrives) - that is the run where you want to see
   where it stopped.

THE LIMIT OF THIS GUARD, said out loud (lesson 7): this is a STRUCTURE pin on
JavaScript source. Python cannot execute it, so this proves the number is
there, that it reaches the sizing arithmetic, that the node's own width sits in
the same max() (which is what makes growth-not-shrinkage true), and that the
fold-away is wired to the result payload. It does NOT prove the browser widens
the node. Only a run with F12 open proves that, and the viewer says so itself:
"node widened to 512 px - sampler preview parity (v592)". Measure, then believe.

1st AMENDMENT (v592) - v591 was green and still wrong. It hung the widen on the
RESULT payload, which arrives when the work is OVER. During the only minutes
anyone watches the node - the run itself - nothing widened, and the process pane
sat in a hard-coded 160px box that had never heard of 512. Frank sent a mid-run
screenshot: still a postage stamp. A law that only takes effect after the thing
it governs has finished is not a law. So:

  * the widen is one function, called by BOTH views - the process pane on its
    first probe (seconds in) and the result viewer on its payload;
  * the process pane's height follows its TILE's aspect at the node's width,
    with the old 160 demoted to a floor;
  * PV_TARGET_W may not exceed _PROBE_MAX_EDGE (the backend thumbnails the probe
    jpeg to that edge) - a 512px pane fed by a 384px jpeg is a big BLURRY
    preview, which is a promise broken more quietly than a small one;
  * the pixel stage gets no minimap: its "tiles" are CHUNKS of frames, a count
    in time, and its rect is the whole canvas - so v579's map came up solid
    orange, said nothing, and ate the width the tile needed.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PU_JS = ROOT / "web" / "js" / "ph_power_upscale.js"
PU_PY = ROOT / "nodes" / "ph_power_upscale.py"

# The v590 shape: width inherited, never grown. The checker must throw it.
_PROBE_BROKEN = """
const PV_MIN_H = 96, PV_MAX_H = 768, NODE_MIN_W = 300;
function _pvApply(node, list) {
    node._pvFrames[0].onload = () => {
        const boxW = Math.max(NODE_MIN_W, node.size[0]);
        node.setSize([node.size[0], node.computeSize()[1]]);
    };
}
"""


def _fail(msg):
    print(f"[test_v591_preview] FAIL: {msg}")
    sys.exit(1)


def _target_width(src):
    """-> the PV_TARGET_W literal, or None."""
    m = re.search(r"const\s+PV_TARGET_W\s*=\s*(\d+)\s*;", src)
    return int(m.group(1)) if m else None


def _grow_only(src):
    """Is the preview width a max() that INCLUDES the node's current width and
    the target? Both terms present = it can only ever grow. Drop node.size[0]
    and a wide node snaps back to 512 (a shrink, and the v531 law is dead);
    drop PV_TARGET_W and nothing grows at all (that is v590).

    v592, and this is embarrassing in the right way: the first draft searched
    the WHOLE file for `const want = Math.max(...)`. v591 has another `want`
    sixty lines further down - a height clamp in a different function - and the
    checker matched THAT, then reported the wrong failure. A guard that matches
    a name out of its scope is the exact disease v590 was cut to kill. It now
    reads inside _widenToTarget, and falls back to the v591 shape by its own
    distinctive name only when that function does not exist."""
    scope = _fn_body(src, "_widenToTarget")
    if scope is not None:
        m = re.search(r"Math\.max\(([^;]*?)\)\s*;", scope)          # v592 shape
    else:
        m = re.search(r"boxW\s*=\s*Math\.max\(([^;]*?)\)\s*;", src)  # v591 shape
    if not m:
        return False
    terms = m.group(1)
    return ("PV_TARGET_W" in terms) and ("node.size[0]" in terms)


def _fn_body(src, name):
    m = re.search(r"function\s+" + name + r"\s*\([^)]*\)\s*\{(.*?)\n\}", src, re.S)
    return m.group(1) if m else None


def _widens_from_both(src):
    """v592: the widen must fire from the PROCESS pane (seconds into the run,
    which is when anyone is actually looking) and from the RESULT viewer (the
    belt, for a node that only ever sees the payload). v591 had only the
    second - green, and still a postage stamp for the whole run."""
    if not re.search(r"function\s+_widenToTarget\s*\(", src):
        return False
    proc = _fn_body(src, "_procApply")
    res = _fn_body(src, "_pvApply")
    if not proc or not res:
        return False
    # v596: matched on the CALL, not on its exact argument list. The first
    # version pinned the literal "_widenToTarget(node)" and broke the moment the
    # process pane started passing its own target width - a legitimate change
    # that the guard read as a regression. A pin that cannot survive a new
    # argument is pinning the punctuation, not the law.
    return ("_widenToTarget(" in proc) and ("_widenToTarget(" in res)


def _pane_follows_aspect(src):
    """The process pane's height must come from the tile's own shape, not from
    a constant. _procFit reads naturalWidth/naturalHeight and is wired to the
    jpeg's decode - a height decided before the image exists is a guess."""
    fit = _fn_body(src, "_procFit")
    if not fit:
        return False
    if "naturalWidth" not in fit or "naturalHeight" not in fit:
        return False
    return bool(re.search(r"tile\.onload\s*=", src)) and "_procFit(node)" in src


def _map_always(src):
    """2nd AMENDMENT (v594). This function used to be _no_map_on_pixel and it
    pinned the OPPOSITE law: hide the minimap when there is "nothing to locate".
    Frank overruled it, and he is right - the minimap is a CONTROL INSTRUMENT.
    A single rectangle filling the frame says "no grid here, one tile is the
    canvas", and that is precisely what a person checking their tile_size needs
    to see. Hidden, a mis-set tile_size looks exactly like a correct one: both
    show nothing. So the pane may never decide the map has nothing to say."""
    body = _fn_body(src, "_procApply")
    if not body:
        return False
    m = re.search(r"_procMapBox\.style\.display\s*=\s*([^;]+);", body)
    if not m:
        return False
    return '"flex"' in m.group(1) and "none" not in m.group(1)


def _probe_edge(py_src):
    m = re.search(r"^_PROBE_MAX_EDGE\s*=\s*(\d+)", py_src, re.M)
    return int(m.group(1)) if m else None


def _folds_away(src):
    """Does _procHide exist, does it re-arm, and is it called from _pvApply
    (the result payload = 'the upscale is through')?"""
    if not re.search(r"function\s+_procHide\s*\(", src):
        return False
    body = re.search(r"function\s+_procHide\s*\([^)]*\)\s*\{(.*?)\n\}", src, re.S)
    if not body:
        return False
    b = body.group(1)
    if "_procSeen = false" not in b:      # without the re-arm it never comes back
        return False
    if "_procH = 0" not in b:             # without the height it still eats space
        return False
    apply_fn = re.search(r"function\s+_pvApply\s*\([^)]*\)\s*\{(.*?)\n\}", src, re.S)
    return bool(apply_fn) and "_procHide(node)" in apply_fn.group(1)


def main():
    src = PU_JS.read_text(encoding="utf-8")

    # ---- 0: the checkers must prove they can fail (v581 §5) -----------------
    if _target_width(_PROBE_BROKEN) is not None:
        _fail("the width checker finds a target in source that has none")
    if _grow_only(_PROBE_BROKEN):
        _fail("the grow-only checker passes the v590 shape (width inherited, "
              "never grown) - it is measuring nothing")
    if _folds_away(_PROBE_BROKEN):
        _fail("the fold-away checker passes source with no _procHide at all")

    # ---- 1: parity width ----------------------------------------------------
    w = _target_width(src)
    if w is None:
        _fail("PV_TARGET_W is gone - the viewer is back to inheriting whatever "
              "width the node happened to carry")
    if w != 512:
        _fail(f"PV_TARGET_W is {w}, expected 512 - the render sampler's preview "
              f"size is the whole point of the number (Frank's ask). Change it "
              f"here AND here, deliberately, or not at all.")
    if not _grow_only(src):
        _fail("the preview width no longer grows-but-never-shrinks: the max() "
              "must carry BOTH node.size[0] (the v531 law: a run never shrinks "
              "a node the user pulled wider) AND PV_TARGET_W (the parity)")

    # ---- 2: the aspect comes from the FRAMES, not from a square -------------
    if not re.search(r"naturalWidth", src) or not re.search(r"naturalHeight", src):
        _fail("the viewer no longer reads the frame's own aspect - 512 is a "
              "WIDTH, not a box; a 16:9 render must not land squashed")

    # ---- 3: the process view bows out and re-arms ---------------------------
    if not _folds_away(src):
        _fail("the process view does not fold away on the result payload (or "
              "does not re-arm): _procHide must exist, zero its height, clear "
              "_procSeen, and be called from _pvApply - 'when the upscale is "
              "through, the lower preview can disappear again'")

    # ---- 3b (v592): ...but while the work RUNS, it must be full size ---------
    if not _widens_from_both(src):
        _fail("the widen does not fire from the process pane. v591 hung it on "
              "the RESULT payload - which lands when the run is OVER - so the "
              "node stayed narrow for the whole run, which is the only part "
              "anyone watches. _widenToTarget(node) must be called from BOTH "
              "_procApply and _pvApply.")
    if not _pane_follows_aspect(src):
        _fail("the process pane's height is not driven by its tile's aspect: "
              "_procFit must read naturalWidth/naturalHeight and hang off the "
              "jpeg's onload. A hard-coded body height shows a 512px frame as "
              "a stamp (that was v591's PROC_BODY_H = 160).")
    if not _map_always(src):
        _fail("the minimap can be hidden. v594: it is a CONTROL INSTRUMENT and "
              "it is always on. One tile filling the frame IS the answer - and "
              "a hidden map makes a fat-fingered tile_size look identical to a "
              "correct one. (This check pinned the opposite law in v592. The "
              "guard changes WITH the law, or it enforces a dead one.)")

    # ---- 3c (v592): the pane may not promise more than the probe delivers ----
    edge = _probe_edge(PU_PY.read_text(encoding="utf-8"))
    if edge is None:
        _fail("_PROBE_MAX_EDGE is gone from the backend - the frontend cannot "
              "size a pane against a jpeg whose size nobody states")
    if w > edge:
        _fail(f"PV_TARGET_W ({w}) exceeds _PROBE_MAX_EDGE ({edge}) - the pane "
              f"would outrun even the probe's CEILING.")
    # v594, and this is the correction of a false comfort: _PROBE_MAX_EDGE is a
    # CEILING, not a resolution. PIL's thumbnail() only ever scales DOWN, so the
    # latent2rgb path ships whatever the latent's own grid is - 96px on a 768
    # tile - and the v592 pane blew that up 5.2x. The guard was green over it,
    # because it had confused a cap with a promise.
    # There is no structural pin for "sharp"; there is only a pin that the pane
    # must SAY what it is showing, and that the sharp path exists at all. Both
    # live in test_v594_sharp.

    # ---- 4: the widen is PROVABLE in the field (measure > believe) ----------
    if "sampler preview parity" not in src:
        _fail("the widen must announce itself once per session - a layout "
              "change nobody can confirm in F12 is a claim, not a measurement")

    # ---- 5: the file carries its own version (the v531 doctrine) ------------
    m = re.search(r"ph_power_upscale\.js v(\d+) loaded", src)
    if not m:
        _fail("the load banner is gone")
    if int(m.group(1)) < 592:
        _fail(f"the JS banner says v{m.group(1)} but this file changed in v592 "
              f"- the banner moves WITH the file, or F12 lies about what is "
              f"loaded (v531)")

    print(f"PASS: v591/v592 -- parity width {w} <= probe edge {edge}, widened "
          f"from BOTH views (process pane first), pane height follows the "
          f"tile's aspect, minimap ALWAYS on (v594: a control instrument), "
          f"process view folds away on the result and re-arms, "
          f"banner v{m.group(1)}")


if __name__ == "__main__":
    main()
