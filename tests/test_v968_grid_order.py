#!/usr/bin/env python3
"""v968 -- the tile grid gets an order of its own, under the SAME ordering law.

Second half of public issue #4: "images are currently sorted by creation date
(descending) only, with no way to change this behaviour". Measured before
answering:

  * The grid really was fixed -- uls_routes._scan_media_fast sorts by mtime,
    reverse, and nothing downstream re-ordered it.
  * But the order a BATCH runs in was never the grid's: the Batch dialog has
    had a four-mode sort selector all along, defaulting to "name (natural)".
    The reporter read the browser's order as the pipeline's. That is a
    discoverability defect, and it is ours, not his.
  * While measuring, a real bug fell out: the JS mirror of order_names took
    mtime for BOTH time modes, because the listing shipped no ctime. So
    "Date created" previewed one order and ran another. /uls/media/list now
    carries ctime and the mirror reads it.

The rule this guard defends: ONE ordering law. A grid preset may only be
(a mode the law already knows) + (reverse yes/no) -- never a sort of its own.

  G1  the presets exist, and every one names a mode the law knows
  G2  "newest" is first and is the default -- a pre-v968 workflow looks unchanged
  G3  the grid order lives in node.properties, NEVER in widgets_values
  G4  the grid renders from the ORDERED list, and so does the dimension probe
  G5  the listing ships ctime, and the mirror uses it for "created"
  G6  the handle sits in the existing pager row (no second window, no new row)
  G7  the wire order of the route is untouched -- other consumers unaffected

MUTATION PROBE (run 20.09.2026, each mutation applied and reverted):
  1. GRID_ORDERS[0] swapped to "number"            -> G2 caught
  2. a preset given mode "size"                    -> G1 caught
  3. gridOrder setter writing to widgets_values    -> G3 caught
  4. renderGrid reading this._files again          -> G4 caught
  5. _syncDims reading this._files again           -> G4 caught
  6. ctime dropped from _scan_media_fast           -> G5 caught
  7. the order button appended to its own <div>    -> G6 caught
  8. entries.sort reversed in _scan_media_fast     -> G7 caught
"""
# v380 (public build): the media routes live in nodes/ph_media_routes.py here
# (uls_routes.py stays the Stack's file). Same source text, different path.
import os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, "web", "js", "ph_media_loader.js")
ROUTES = os.path.join(ROOT, "nodes", "ph_media_routes.py")
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def _strip_js(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    return re.sub(r"^\s*//.*$", "", src, flags=re.M)


def main():
    print("v968: the grid orders itself, under the one law")
    js = open(JS, encoding="utf-8").read()
    code = _strip_js(js)
    routes = open(ROUTES, encoding="utf-8").read()

    # ── G1: presets, all in the law's vocabulary ───────────────────────────
    m = re.search(r"const GRID_ORDERS = \[(.*?)\n\];", code, re.S)
    block = m.group(1) if m else ""
    check(m is not None, "G1 GRID_ORDERS is defined")
    presets = re.findall(r'\{\s*key:\s*"(\w+)".*?mode:\s*"([^"]+)"[^}]*?rev:\s*(true|false)',
                         block, re.S)
    check(len(presets) >= 4, "G1 there are at least four presets (got %d)" % len(presets))

    # the modes the ONE law understands, read off _orderNames itself
    om = re.search(r"_orderNames\(names, mode, byName\) \{(.*?)\n    \}", code, re.S)
    obody = om.group(1) if om else ""
    known = set(re.findall(r'mode === "([^"]+)"', obody))
    known.add("name (natural)")          # the default branch
    for key, mode, _rev in presets:
        check(mode in known,
              "G1 preset %r uses a mode the law knows (%r)" % (key, mode))

    # ── G2: the default is what the grid did before ────────────────────────
    check(presets and presets[0][0] == "newest",
          "G2 the FIRST preset is 'newest' -- the pre-v968 behaviour")
    check(presets and presets[0][2] == "true" and presets[0][1].startswith("mtime"),
          "G2 'newest' is mtime reversed, i.e. exactly the old wire order")
    check(re.search(r"return _gridOrder\(k\);", code) is not None
          and re.search(r"GRID_ORDERS\.find\(\(o\) => o\.key === key\) \|\| GRID_ORDERS\[0\]", code)
          is not None,
          "G2 an unknown or absent key falls back to the first preset")

    # ── G3: properties, not widgets_values ─────────────────────────────────
    gm = re.search(r"set gridOrder\(key\) \{(.*?)\n    \}", code, re.S)
    gbody = gm.group(1) if gm else ""
    check(gm is not None, "G3 gridOrder has a setter")
    check("this.node.properties" in gbody and "ph_media_order" in gbody,
          "G3 it writes node.properties.ph_media_order")
    check("widgets_values" not in gbody and "widgets" not in gbody,
          "G3 it never touches widgets_values -- slots and widgets are positional")

    # ── G4: everything that pages reads the ORDERED list ───────────────────
    check(re.search(r"_orderedFiles\(\) \{", code) is not None,
          "G4 _orderedFiles exists")
    rm = re.search(r"\n    renderGrid\(\) \{(.*?)\n    \}\n", code, re.S)
    rbody = rm.group(1) if rm else ""
    check(rm is not None, "G4 renderGrid found")
    check("this._orderedFiles()" in rbody,
          "G4 renderGrid pages the ORDERED list")
    check(re.search(r"this\._files\b", rbody) is None,
          "G4 renderGrid no longer pages the raw _files")
    sm = re.search(r"async _syncDims\(\) \{(.*?)\n    \}\n", code, re.S)
    sbody = sm.group(1) if sm else ""
    check("_orderedFiles()" in sbody,
          "G4 the dimension probe follows the same order (else it probes unseen tiles)")
    check(re.search(r"this\._files\.slice\(", sbody) is None,
          "G4 the dimension probe no longer slices the raw _files")
    # the ordered list must be a COPY -- _files stays the truth
    om2 = re.search(r"_orderedFiles\(\) \{(.*?)\n    \}", code, re.S)
    obody2 = om2.group(1) if om2 else ""
    check(".sort(" not in obody2,
          "G4 _orderedFiles never sorts _files in place")

    # ── G5: ctime on the wire, and used ────────────────────────────────────
    check(re.search(r'"ctime":\s*getattr\(st,\s*"st_ctime",\s*st\.st_mtime\)', routes)
          is not None,
          "G5 _scan_media_fast ships ctime from the same stat call")
    check('mode === "created"' in obody and "ctime" in obody,
          "G5 the JS mirror reads ctime for the 'created' mode")

    # ── G6: one row, no second window ──────────────────────────────────────
    check("_makeOrderButton()" in code, "G6 the handle is built once")
    check(rbody.count("this.pagerEl.append") >= 2
          and "this._makeOrderButton()" in rbody,
          "G6 the handle goes into the EXISTING pager row, on both branches")
    check(re.search(r"document\.createElement\(\"dialog\"\)|position:\s*fixed",
                    _strip_js(re.search(r"_makeOrderButton\(\) \{(.*?)\n    \}", code, re.S).group(1)))
          is None,
          "G6 the handle opens no window of its own")

    # ── G7: the route's wire order -- newest first, and REPRODUCIBLE ───────
    # v969: this used to pin `sort(key=mtime, reverse=True)`. That sort is
    # stable, so files sharing an mtime kept os.scandir's order and the FILE
    # SYSTEM decided the listing. Field, 20.09.2026: six files unpacked from
    # one archive all carried the same minute and came out in Explorer's
    # directory order -- correct, but not reproducible. The tie-break makes
    # the answer a function of the data. Driven below, not just read.
    check(re.search(r'entries\.sort\(key=lambda d: \(-d\["mtime"\], '
                    r'natural_sort_key\(d\["name"\]\)\)\)', routes) is not None,
          "G7 the listing sorts by (-mtime, natural name)")
    check("natural_sort_key" in routes.split("def _scan_media_fast")[0],
          "G7 natural_sort_key is imported, not redefined locally")

    # ── G8: DRIVEN -- equal timestamps must not leave it to the disk ───────
    import tempfile, os as _os
    text = routes
    parts = []
    for sig in ("def _media_kind(", "def _within(", "def _scan_media_fast("):
        i = text.find(sig)
        check(i >= 0, "G8 helper %r found" % sig)
        parts.append(text[i:text.index("\n\n\n", i)])
    sys.path.insert(0, os.path.join(ROOT, "nodes"))
    from ph_media_util import (IMAGE_EXTS, VIDEO_EXTS, AUDIO_EXTS,
                               natural_sort_key)
    ns = {"os": _os, "time": __import__("time"),
          "_MEDIA_IMAGE_EXTS": IMAGE_EXTS, "_MEDIA_VIDEO_EXTS": VIDEO_EXTS,
          "_MEDIA_AUDIO_EXTS": AUDIO_EXTS, "natural_sort_key": natural_sort_key}
    exec("\n\n".join(parts), ns)

    d = tempfile.mkdtemp(prefix="pls_tie_")
    # Written in an order that is neither alphabetical nor natural, so a
    # scandir-order result cannot accidentally look correct.
    written = ["zulu.png", "img10.png", "alpha.png", "img2.png", "img1.png"]
    for n in written:
        open(_os.path.join(d, n), "wb").write(b"\0")
        _os.utime(_os.path.join(d, n), (1_700_000_000, 1_700_000_000))   # ALL equal
    got = [e["name"] for e in ns["_scan_media_fast"](d)]
    want = ["alpha.png", "img1.png", "img2.png", "img10.png", "zulu.png"]
    check(got == want,
          "G8 equal mtimes -> natural name order, got %r" % (got,))

    # and a normal folder still answers newest-first
    for i, n in enumerate(["old.png", "mid.png", "new.png"]):
        p = _os.path.join(d, n)
        open(p, "wb").write(b"\0")
        _os.utime(p, (1_800_000_000 + i * 1000, 1_800_000_000 + i * 1000))
    got2 = [e["name"] for e in ns["_scan_media_fast"](d)][:3]
    check(got2 == ["new.png", "mid.png", "old.png"],
          "G8 distinct mtimes -> still newest-first, got %r" % (got2,))

    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    print("v968 grid order: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
