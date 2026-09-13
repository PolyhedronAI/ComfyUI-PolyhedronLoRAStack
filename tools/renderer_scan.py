#!/usr/bin/env python3
"""Renderer scan -- which node survives ComfyUI's Vue renderer ("Nodes 2.0")?

Background (07.09.2026, a public user asked): the Vue node renderer never
calls `onDrawForeground`, so every node whose UI is hand-drawn on the
LiteGraph canvas shows up EMPTY under it. `web/js/uls_compat.js` warns about
this (toast + an in-node notice widget) for every class listed in its
POLY_CANVAS_NODES set.

Classes reported here:

  A   no JS binding at all -- standard widgets only. Renders under BOTH
      renderers, nothing to do.
  B   a JS file targets the node, but only for helpers (tooltips, toasts,
      hydration, migration). Node body is standard widgets.
  C1  DOM widgets (addDOMWidget / createElement), no canvas drawing. Core's
      Vue renderer HAS a DOM widget wrapper
      (src/renderer/extensions/vueNodes/widgets/components/WidgetDOM.vue,
      checked 07.09.2026), so these should appear -- open questions are
      sizing (our computeSize hooks) and hidden widgets (v585).
  C2  hand-drawn PARTS on the canvas (onDrawForeground/onDrawBackground).
      Those parts are missing under Nodes 2.0 -- and exactly the set
      uls_compat.js must know about.

v940 correction (measured 10.09.2026 in a real browser, all 20 C2 nodes side
by side under both renderers): C2 does NOT mean "empty". Standard widgets and
DOM widgets render under Nodes 2.0; what is missing is only what the canvas
code paints. Only two nodes painted their whole body -- the Stack and the
Engine -- and both have a Nodes 2.0 view since v937/v938 (marked [view]).
For the other eighteen the missing parts are extras (segment colours, a hint
line, inline output dots). The real defect the measurement found was a
different one: widgets hidden the classic way came BACK under Nodes 2.0 --
fixed in v940 by web/js/uls_vue_hidden.js.

The way forward is NOT to write Vue components (they would only render under
the new renderer and abandon LiteGraph): it is to move C2 nodes onto DOM
widgets, which render under both. C1 is the proof that this works.

v921: this file lives IN the tree and exposes scan() as a function, because
test_v921_renderer_notice_coverage.py imports it. A second implementation of
the same scan would be a second truth; the guard and the tool must agree by
construction. The baseline file is found by glob, so a new cut does not have
to touch this file.

    python3 tools/renderer_scan.py [path/to/carrier]
"""
import collections
import glob
import os
import re
import sys

ID_RE = re.compile(r'"(ULS[A-Za-z0-9_]+|PH[A-Za-z0-9_]+|Polyhedron[A-Za-z0-9_]+'
                   r'|WAN[A-Za-z0-9_]+|Wan[A-Za-z0-9_]+|UltimateLoraStack)"')
SIGNALS = {
    "draw":   r"onDrawForeground|onDrawBackground",
    "mouse":  r"onMouseDown|onMouseMove|onMouseUp|onDblClick",
    "dom":    r"addDOMWidget|document\.createElement",
    "widget": r"addCustomWidget|addWidget\(",
}

# uls_compat.js is the WARNING layer, not a node UI. It necessarily mentions
# every hand-drawn class and necessarily contains draw signals (it wraps
# onDrawForeground). Counting it as a node's own binding would make every
# listed node look hand-drawn -- including the DOM ones. Excluded by name.
# Files that NAME node classes without being their UI: uls_compat.js lists them
# for its notice; uls_vue_hidden.js (v940) names the pack's classes for its
# options.hidden rule.
NOT_A_NODE_UI = {"uls_compat.js", "uls_vue_hidden.js"}


def node_ids(root):
    """The node id census, from whichever NODE_IDS baseline the tree carries."""
    hits = sorted(glob.glob(os.path.join(root, "NODE_IDS_baseline_*.txt")))
    if not hits:
        raise RuntimeError("no NODE_IDS_baseline_*.txt under %r" % root)
    if len(hits) > 1:
        raise RuntimeError("more than one NODE_IDS baseline: %r" % hits)
    out = []
    for line in open(hits[0], encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.split()[0])
    return out, os.path.basename(hits[0])


def scan(root):
    """-> (per_node dict, ordered id list, baseline filename).

    per_node[id] = {files, draw, mouse, dom, widget, lines}
    """
    ids, base = node_ids(root)
    idset = set(ids)
    per = collections.defaultdict(lambda: {"files": set(), "draw": 0,
                                           "mouse": 0, "dom": 0,
                                           "widget": 0, "lines": 0})
    jsdir = os.path.join(root, "web", "js")
    for name in sorted(os.listdir(jsdir)):
        if not name.endswith(".js") or name in NOT_A_NODE_UI:
            continue
        src = open(os.path.join(jsdir, name), encoding="utf-8",
                   errors="replace").read()
        targets = {t for t in ID_RE.findall(src) if t in idset}
        if not targets:
            continue
        sig = {k: len(re.findall(v, src)) for k, v in SIGNALS.items()}
        for t in targets:
            per[t]["files"].add(name)
            per[t]["lines"] += src.count("\n")
            for k in sig:
                per[t][k] += sig[k]
    return per, ids, base


def klass(per, node):
    d = per.get(node)
    if not d:
        return "A"
    if d["draw"]:
        return "C2"
    if d["dom"]:
        return "C1"
    return "B"


VIEW_RE = re.compile(r"registerVueView\(\s*([A-Za-z_][A-Za-z_0-9]*|\"[^\"]+\")")
CONST_RE = re.compile(r"const\s+([A-Z_][A-Z_0-9]*)\s*=\s*\"([A-Za-z_0-9]+)\"")


def vue_views(root):
    """v940: classes that register a Nodes 2.0 view (web/js, registerVueView)."""
    out = set()
    jsdir = os.path.join(root, "web", "js")
    for name in sorted(os.listdir(jsdir)):
        if not name.endswith(".js"):
            continue
        src = open(os.path.join(jsdir, name), encoding="utf-8", errors="replace").read()
        consts = dict(CONST_RE.findall(src))
        for arg in VIEW_RE.findall(src):
            out.add(arg.strip('"') if arg.startswith('"') else consts.get(arg, arg))
    return out


def hand_drawn(root):
    """The C2 set: node classes whose UI is painted on the LiteGraph canvas."""
    per, ids, _ = scan(root)
    return {n for n in ids if klass(per, n) == "C2"}


TITLE = {
    "A":  "A   no JS binding -- renders under BOTH renderers",
    "B":  "B   JS helpers only (tooltip/toast/hydration) -- standard widgets",
    "C1": "C1  DOM widgets, no canvas drawing -- Vue has a DOM wrapper",
    "C2": "C2  hand-drawn parts on the canvas -- those parts are missing under "
          "Nodes 2.0 (widgets still render; [view] = has its own DOM view)",
}


def main(root):
    per, ids, base = scan(root)
    views = vue_views(root)
    groups = collections.defaultdict(list)
    for n in ids:
        groups[klass(per, n)].append(n)
    print("census: %s (%d nodes)" % (base, len(ids)))
    for k in ("A", "B", "C1", "C2"):
        print("\n### %s  (%d)" % (TITLE[k], len(groups[k])))
        for n in sorted(groups[k]):
            d = per.get(n)
            if d:
                print("  %-26s %-34s draw=%-2d dom=%-3d ~%d lines%s"
                      % (n, ",".join(sorted(d["files"])), d["draw"],
                         d["dom"], d["lines"],
                         "  [view]" if n in views else ""))
            else:
                print("  %s" % n)
    ok = len(groups["A"]) + len(groups["B"]) + len(groups["C1"])
    print("\n%d/%d nodes without canvas drawing (A+B+C1), %d with painted parts "
          "(C2), %d of them with their own Nodes 2.0 view."
          % (ok, len(ids), len(groups["C2"]),
             len([n for n in groups["C2"] if n in views])))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
