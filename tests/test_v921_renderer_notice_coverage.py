#!/usr/bin/env python3
"""v921 -- the renderer notice must cover EVERY hand-drawn node.

THE WOUND THIS GUARD CLOSES (found 08.09.2026)
----------------------------------------------
uls_compat.js has warned since v301 that ComfyUI's Vue renderer ("Nodes 2.0")
never calls onDrawForeground, leaving hand-drawn node UIs empty. But its
POLY_CANVAS_NODES set still listed the two classes that existed back then --
UltimateLoraStack and ULSAccelerator -- while the tree had grown to twenty
hand-drawn nodes. For the other eighteen NOTHING fired: no toast, no in-node
notice, no console line. A user on the Vue renderer saw empty boxes and had
no way to learn why.

The set is a hand-kept list next to a growing tree, so it drifts silently.
This guard makes the drift impossible to ship: the set must equal what the
TREE says is hand-drawn, measured by tools/renderer_scan.py -- the same
scanner the tool uses, imported rather than reimplemented (a second
implementation would be a second truth, v368 lesson).

THE SECOND HALF IS THE DANGEROUS ONE
------------------------------------
Listing a node is only safe if we can also OBSERVE it draw. The verdict is
"this node never drew -> the renderer is broken", so a node we cannot observe
would be accused wrongly -- a false "your renderer is broken" toast in front
of every user of the classic renderer, which is worse than the silence we are
fixing. v921 therefore installs its own onDrawForeground wrapper
(installDrawProbe) and the probe skips any node whose wrapper is not provably
installed. This guard pins that safety: the wrapper must set both flags, must
be installed from nodeCreated, and the probe must keep its skip.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import renderer_scan  # noqa: E402  (path set above)

COMPAT = os.path.join(ROOT, "web", "js", "uls_compat.js")

failures = []


def check(cond, msg):
    if cond:
        print("  ok   %s" % msg)
    else:
        print("  FAIL %s" % msg)
        failures.append(msg)


def listed_nodes(src):
    """The POLY_CANVAS_NODES literal, as a set of class names."""
    m = re.search(r"const POLY_CANVAS_NODES = new Set\(\[(.*?)\]\);",
                  src, re.S)
    if not m:
        return None
    return set(re.findall(r'"([A-Za-z0-9_]+)"', m.group(1)))


def main():
    print("v921: renderer notice covers every hand-drawn node")
    src = open(COMPAT, encoding="utf-8", errors="replace").read()

    listed = listed_nodes(src)
    check(listed is not None, "POLY_CANVAS_NODES literal is parseable")
    if listed is None:
        return 1

    drawn = renderer_scan.hand_drawn(ROOT)
    check(len(drawn) > 2,
          "the tree has more than the two v301 classes (found %d)" % len(drawn))

    missing = sorted(drawn - listed)
    check(not missing,
          "every hand-drawn node is listed%s"
          % ("" if not missing else " -- MISSING: %s" % ", ".join(missing)))

    # A node listed but NOT hand-drawn would get the notice wrongly: its UI
    # renders fine under both renderers, so a warning about it is noise.
    extra = sorted(listed - drawn)
    check(not extra,
          "no node is listed that the tree does not draw by hand%s"
          % ("" if not extra else " -- EXTRA: %s" % ", ".join(extra)))

    # ---- the false-alarm safety -------------------------------------------
    check("function installDrawProbe" in src,
          "installDrawProbe exists (central draw evidence)")
    probe = src.split("function installDrawProbe", 1)[-1].split("\nconst ", 1)[0]
    check("_ulsDrawFired = true" in probe,
          "the wrapper records the draw (_ulsDrawFired)")
    check("_ulsCompatHooked = true" in probe,
          "the wrapper marks itself installed (_ulsCompatHooked)")
    check(re.search(r"prev\s*\?\s*prev\.apply", probe) is not None,
          "the wrapper CHAINS the previous onDrawForeground, never replaces it")
    check("catch" in probe,
          "the wrapper cannot throw into ComfyUI")

    check(re.search(r"installDrawProbe\(node\)", src) is not None,
          "nodeCreated installs the wrapper")
    check(re.search(r"setTimeout\(\s*\(\)\s*=>\s*installDrawProbe", src)
          is not None,
          "installation is deferred a tick (runs after other extensions)")

    check(re.search(r"!n\._ulsCompatHooked\s*&&\s*!n\._ulsDrawFired\s*\)\s*continue",
                    src) is not None,
          "the probe SKIPS nodes it cannot observe (no verdict, no false alarm)")

    # The v306 verdict path must survive this cut: a node that DID draw
    # proves the canvas renderer alive, and silent siblings are then merely
    # offscreen. Losing that would turn a scrolled-away node into an alarm.
    check('if (drew) { COMPAT.lastJudgement = "drew"; return; }' in src,
          "the v306 'someone drew -> all good' verdict is intact")
    check("nodeInViewport" in src,
          "the v306 viewport evidence is still consulted")

    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    print("v921: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
