"""Guard #107 -- deferred thumbnails (v684, stage L2).

v683 stopped the LISTING from opening every file. This stage stops the
GRID from fetching every thumbnail: each tile's URL is parked in a data
attribute and only assigned to img.src when an IntersectionObserver says
the tile is near view. `loading="lazy"` does not do this job here -- the
loader's DOM widget is absolutely positioned on the ComfyUI canvas, so
the browser considers the whole thing visible and fetches the lot.

DRIVEN, not read: _thumbReveal is lifted out of ph_media_loader.js and
executed in node against fabricated observer entries. What is proven:

  * An intersecting tile with a parked URL gets it assigned, exactly once.
  * A tile that is NOT intersecting is left alone -- no src, no fetch.
  * A revealed tile is UNOBSERVED immediately, so scrolling it back into
    view can never re-fetch it, and the watch set only ever shrinks.
  * An entry with no parked URL (already revealed) is unobserved too and
    assigns nothing.
  * Running the same entry twice reveals ONCE.

STATIC, because they are structural and cheap to lose:
  * the tile parks the URL instead of assigning it, and observes,
  * with a direct-assignment fallback when the API is absent,
  * the watch set is dropped before the tiles under it are discarded,
  * and again in _destroy (the v624 leak law: every observer, listener
    and timer this UI creates dies with the node).

MUTATIONS (wound in a COPY, catch proven):
  M1 the isIntersecting check is dropped -> offscreen tiles fetch anyway.
  M2 the parked URL is not deleted       -> a tile can re-fetch forever.
  M3 unobserve is never called           -> the watch set only grows.
  M4 the renderGrid disconnect is removed -> removed <img> stay reachable.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOADER_JS = os.path.join(ROOT, "web", "js", "ph_media_loader.js")

NAME = "test_v684_lazy_thumbs"


def _fail(msg):
    print("[%s] FAIL -- %s" % (NAME, msg))
    sys.exit(1)


def _need(ok, msg):
    if not ok:
        _fail(msg)


def _lift(src):
    m = re.search(r"function _thumbReveal\(entries, unobserve\) \{.*?\n\}",
                  src, flags=re.S)
    _need(m is not None, "_thumbReveal not found")
    return m.group(0)


_DRIVER = r"""
%s
function img(parked) {
    const o = { dataset: {}, src: "" };
    if (parked) o.dataset.phSrc = parked;
    return o;
}
function entry(target, isIntersecting) { return { target, isIntersecting }; }
(async () => {
    const out = {};
    const seen = [];
    const unob = (el) => { seen.push(el); };

    // a visible tile with a parked URL -> assigned once, then unobserved
    const a = img("URL_A");
    const b = img("URL_B");          // offscreen
    const c = img(null);             // already revealed earlier
    const n = _thumbReveal([entry(a, true), entry(b, false), entry(c, true)], unob);
    out.revealed = n;
    out.aSrc = a.src;
    out.aParked = a.dataset.phSrc === undefined ? null : a.dataset.phSrc;
    out.bSrc = b.src;
    out.bStillParked = b.dataset.phSrc || null;
    out.cSrc = c.src;
    out.unobserved = seen.length;    // a and c, never b

    // the SAME entry again: nothing left to reveal
    const again = _thumbReveal([entry(a, true)], unob);
    out.again = again;
    out.aSrcAfter = a.src;

    console.log(JSON.stringify(out));
})();
"""


def run_driven(src, tag=""):
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False,
                                     encoding="utf-8") as f:
        f.write(_DRIVER % _lift(src))
        path = f.name
    try:
        p = subprocess.run(["node", path], capture_output=True, text=True,
                           timeout=60)
        lines = p.stdout.strip().splitlines()
        if p.returncode != 0 or not lines:
            _fail("%snode produced no verdict: %s"
                  % (tag, (p.stderr.strip() or "no output")[:300]))
        try:
            out = json.loads(lines[-1])
        except ValueError:
            _fail("%sunreadable verdict: %r" % (tag, lines[-1][:200]))
    finally:
        os.unlink(path)

    _need(out["revealed"] == 1,
          "%sexactly one parked, intersecting tile may be revealed, got %s"
          % (tag, out["revealed"]))
    _need(out["aSrc"] == "URL_A",
          "%sthe visible tile must get its parked URL, got %r"
          % (tag, out["aSrc"]))
    _need(out["aParked"] is None,
          "%sthe parked URL must be cleared once assigned (else it can be "
          "assigned again on the next intersection)" % tag)
    _need(out["bSrc"] == "" and out["bStillParked"] == "URL_B",
          "%san offscreen tile must NOT be fetched, got src=%r"
          % (tag, out["bSrc"]))
    _need(out["cSrc"] == "",
          "%san already-revealed entry must assign nothing" % tag)
    _need(out["unobserved"] == 2,
          "%severy intersecting tile must be unobserved (2 expected), got %s"
          % (tag, out["unobserved"]))
    _need(out["again"] == 0 and out["aSrcAfter"] == "URL_A",
          "%sthe same tile must never be revealed twice" % tag)


def run_static(src, tag=""):
    _need("img.dataset.phSrc = url" in src and "obs.observe(img)" in src,
          "%sthe tile must PARK its thumbnail URL and observe, not assign it"
          % tag)
    _need(re.search(r"else \{ img\.src = url; \}", src) is not None,
          "%sthere must be a direct-assignment fallback where the observer "
          "API is missing" % tag)
    _need("THUMB_MARGIN" in src and "rootMargin: THUMB_MARGIN" in src,
          "%sthe observer must pre-load a margin outside the visible strip"
          % tag)
    _need("root: this.gridEl" in src,
          "%sthe observer must be rooted in the grid's own scroller -- the "
          "page viewport is the wrong frame on the ComfyUI canvas" % tag)
    # the leak law, both halves
    i_disc = src.find("this._thumbObs?.disconnect()")
    i_html = src.find('this.gridEl.innerHTML = "";')
    _need(i_disc >= 0 and i_html >= 0 and i_disc < i_html,
          "%sthe watch set must be dropped BEFORE the tiles under it are "
          "discarded" % tag)
    i_destroy = src.find("_destroy() {")
    _need(i_destroy >= 0
          and "this._thumbObs?.disconnect()" in src[i_destroy:i_destroy + 900],
          "%s_destroy must disconnect the thumbnail observer (v624 leak law)"
          % tag)


def _mutate(src, kind):
    if kind == "M1":
        out = src.replace("if (!e || !e.isIntersecting) continue;",
                          "if (!e) continue;")
    elif kind == "M2":
        out = src.replace("            delete img.dataset.phSrc;\n", "")
    elif kind == "M3":
        out = src.replace(
            "        try { unobserve(img); } catch (err) { /* observer already gone */ }\n",
            "")
    elif kind == "M4":
        out = src.replace(
            "        try { this._thumbObs?.disconnect(); } catch (e) { /* ignore */ }\n"
            '        this.gridEl.innerHTML = "";',
            '        this.gridEl.innerHTML = "";')
    else:
        raise AssertionError(kind)
    _need(out != src, "%s could not be injected" % kind)
    return out


def main():
    src = open(LOADER_JS, encoding="utf-8").read()
    run_driven(src)
    run_static(src)

    caught = 0
    for kind in ("M1", "M2", "M3"):
        try:
            run_driven(_mutate(src, kind), tag="[%s] " % kind)
        except SystemExit:
            caught += 1
        else:
            print("[%s] NOTE -- mutation %s survived" % (NAME, kind))
    try:
        run_static(_mutate(src, "M4"), tag="[M4] ")
    except SystemExit:
        caught += 1
    else:
        print("[%s] NOTE -- mutation M4 survived" % NAME)

    _need(caught == 4, "only %d/4 mutations were caught" % caught)
    print("[%s] PASS -- offscreen tiles are never fetched, a visible one is "
          "revealed exactly once and unobserved immediately, the parked URL "
          "is cleared so no tile can re-fetch, the fallback keeps old "
          "browsers on the old behaviour, and the watch set dies both with "
          "the grid and with the node, 4/4 mutations caught" % NAME)


if __name__ == "__main__":
    main()
