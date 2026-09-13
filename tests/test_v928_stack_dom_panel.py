#!/usr/bin/env python3
"""v928 -- the Stack's Nodes 2.0 panel is a second VIEW, never second data.

FIELD FINDING 09.09.2026
------------------------
With Nodes 2.0 on, the LoRA Stack showed sockets and nothing else. Measured
cause: the node builds its whole body in onDrawForeground (545 lines) with
hit-testing in onMouseDown/Move/Up (~800 lines) and calls addDOMWidget
exactly zero times. The Vue renderer never calls onDrawForeground, so nothing
of the node's UI survives. Nodes built on DOM widgets came through the same
field check intact -- hence the move to DOM.

WHAT MUST STAY TRUE
-------------------
The dangerous failure here is not a missing button, it is a SECOND TRUTH.
If the panel kept its own copy of the rows, or wrote uls_config itself, or
carried its own LoRA picker, the two views would drift apart the first time
one of them is fixed -- and the workflow on disk would be the casualty. So:

  * state lives in node._uls, persisted ONLY via node._ulsSync()
  * the panel writes no JSON, touches no widget value, serializes nothing
  * picker and weight dialog are IMPORTED from uls_node.js, not rebuilt
  * the panel exists only while LiteGraph.vueNodesMode is true (v934), so
    the classic renderer keeps the matured painted UI unchanged

The last point protects the user's daily working surface: an unreadable
flag must fail toward "leave today alone", never toward "show both".
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOM = os.path.join(ROOT, "web", "js", "uls_stack_dom.js")
NODE = os.path.join(ROOT, "web", "js", "uls_node.js")
# v936: the switching logic lives in the shared switch point. Pins about WHEN
# the panel exists read it there; pins about WHAT the panel is read the Stack.
VIEWS = os.path.join(ROOT, "web", "js", "uls_vue_views.js")


def _strip(src):
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    src = re.sub(r"^\s*//.*$", "", src, flags=re.M)
    return re.sub(r"\s//.*$", "", src, flags=re.M)

failures = []



def calls(src, name):
    """True if `name` is CALLED somewhere, not merely declared or imported.

    Three separate mutations slipped past this guard because a plain substring
    search finds the name in its own `function name(...)` header or in the
    `import { name }` line. A guard that greps a symbol proves the symbol
    exists; it never proves the code uses it. So: strip the import block and
    every declaration header, then look for a call.
    """
    body = src.split('} from "./uls_node.js";', 1)[-1]
    body = re.sub(r"function\s+" + re.escape(name) + r"\s*\(", "", body)
    return re.search(re.escape(name) + r"\s*\(", body) is not None


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def main():
    print("v928: Stack DOM panel -- one truth, two views")
    for p in (DOM, NODE, VIEWS):
        if not os.path.exists(p):
            print("  FAIL missing %s" % p)
            return 1
    dom = open(DOM, encoding="utf-8", errors="replace").read()
    node = open(NODE, encoding="utf-8", errors="replace").read()
    views = open(VIEWS, encoding="utf-8", errors="replace").read()
    vcode = _strip(views)

    # ---- the panel exists and is a DOM widget on the Stack ----------------
    # v936, RE-JUSTIFIED: the Stack no longer installs the widget itself -- it
    # REGISTERS its view with the shared switch point, which installs it.
    check(calls(dom, "registerVueView"),
          "the Stack registers its view with the switch point (called, not "
          "just imported)")
    check("addDOMWidget" in vcode, "the switch point installs it as a DOM widget")
    check('"UltimateLoraStack"' in dom, "it targets the Stack class")
    check(re.search(r"nodeCreated\s*\(", vcode) is not None,
          "the switch point hooks nodeCreated")

    # ---- one truth: state and persistence stay with the node --------------
    check("_ulsSync" in dom,
          "it persists through the node's own _ulsSync()")
    # Comments may NAME uls_config (they explain why we don't touch it); what
    # must not exist is a write. Strip comments, then look for actual access.
    code = re.sub(r"/\*.*?\*/", "", dom, flags=re.S)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
    code = re.sub(r"\s//.*$", "", code, flags=re.M)
    check("uls_config" not in code,
          "no access to the uls_config widget in code (only _ulsSync may)")
    check(re.search(r"\.value\s*=", code) is None,
          "it assigns no widget value anywhere")
    check("widgets.find" not in code,
          "it does not reach into the node's widget list to write")
    check("JSON.stringify" not in dom,
          "it serializes nothing -- no second format")
    check("serialize: false" in vcode and "widget.serialize = false" in vcode,
          "the DOM widget itself stores nothing in the workflow")
    check(re.search(r"node\._uls\b", dom) is not None,
          "it edits node._uls, the existing state object")

    # ---- one truth: shared dialogs are imported, not rebuilt --------------
    m = re.search(r"import\s*\{([^}]*)\}\s*from\s*\"\./uls_node\.js\"", dom)
    check(m is not None, "it imports helpers from uls_node.js")
    if m:
        imported = {x.strip() for x in m.group(1).split(",") if x.strip()}
        for fn in ("openLoraSelect", "showWeightInput", "newRow"):
            check(fn in imported, "reuses %s instead of rebuilding it" % fn)
        check("export { openLoraSelect" in node or
              re.search(r"export\s*\{[^}]*openLoraSelect", node) is not None,
              "uls_node.js exports those helpers")

    # ---- the classic renderer must be left alone -------------------------
    # v934, RE-JUSTIFIED (not softened). The promise was always "the classic
    # renderer keeps the matured painted UI unchanged". v923-v928 kept it by
    # hiding the panel from a draw-time fact (_ulsDrawFired, a 1500 ms watch,
    # computeSize [0,-4]) -- and in the field (10.09.2026) the frontend's
    # container around the still-existing widget swallowed every click: the
    # painted node was dead. The pins that proved that mechanism proved the
    # wrong thing. The promise now holds by construction: under the classic
    # renderer the widget does not exist, the renderer is read from
    # LiteGraph.vueNodesMode, and nothing guesses from frames or timers.
    # Behaviour is driven in tests/test_v934_stack_dom_classic_safe.py and in
    # a real browser by tools/browser_probe_stack.py.
    # v936: the same promises, read where the logic now lives -- and the two
    # "gone" lists run over BOTH files, so the old mechanism cannot come back
    # through either door.
    check("vueNodesMode" in vcode,
          "the renderer is read from LiteGraph.vueNodesMode (switch point)")
    for where, src_ in (("Stack", code), ("switch point", vcode)):
        check("Comfy.VueNodes.Enabled" not in src_,
              "%s: no dependency on the setting id (v922: store not ready)" % where)
        for gone in ("_ulsDrawFired", "_ulsLastDraw", "QUIET_MS", "REVEAL_MS",
                     "computeSize", "onDrawForeground"):
            check(gone not in src_,
                  "%s: no draw-time guessing, no size games: %s" % (where, gone))
    check(calls(views, "attach"), "the view is attached on demand")
    check(calls(views, "syncNode"), "one place decides per node")

    # ---- the painted view must remain intact ------------------------------
    check(node.count("onDrawForeground") >= 2,
          "the painted UI in uls_node.js is still there (nothing ripped out)")
    check("_ulsDrawFired = true" in node,
          "the v921 draw evidence in uls_node.js is untouched")


    # ---- v924: the panel must not be contradicted by the notice -----------
    # Field 09.09.2026: the Stack showed working DOM rows AND a notice saying
    # the UI needs the LiteGraph renderer. A node with a working replacement
    # view is not broken; warning about it tells the user to abandon a UI that
    # is visibly in front of them.
    compat = open(os.path.join(ROOT, "web", "js", "uls_compat.js"),
                  encoding="utf-8", errors="replace").read()
    check("_ulsDomPanel = true" in vcode,
          "the switch point marks a registered node as having a replacement view")
    check("_ulsDomPanel" in compat,
          "uls_compat.js honours that mark")
    # v943, RE-JUSTIFIED: the skip condition grew a second reason (VUE_USABLE,
    # the nodes measured usable under Nodes 2.0). The promise is unchanged:
    # the DOM-panel mark alone is enough to be skipped -- it must be one of the
    # OR-ed reasons of the skipping if, not ANDed with anything.
    m = re.search(r"if\s*\(([^\n]*?n\._ulsDomPanel[^\n]*?)\)\s*\{[^}\n]*continue", compat)
    check(m is not None and "&&" not in m.group(1),
          "a node with a DOM panel is skipped by the renderer probe")
    check("function dropNotice" in compat,
          "an already-injected notice is removed, not just suppressed")

    # ---- v924: parity with the painted view -------------------------------
    # Test the CALL, not the mention: an imported name stays in the import
    # line even after the call is swapped for a home-grown one, and a guard
    # that only greps the name stays green on exactly that regression
    # (mutation test, 09.09.2026 -- second time this bit).
    body = dom.split("} from \"./uls_node.js\";", 1)[-1]
    for label, needle, why in [
        ("row ordering", "Move up", "the painted view has up/down arrows"),
        ("group pill", "showGroupModePopup(",
         "groups open the SHARED popup, not a rebuilt one"),
        ("weight steps", "Shift = CLIP strength",
         "weight arrows carry the shift=CLIP behaviour"),
        ("flat/group pill", "flatMode", "the header pill exists"),
        ("apply pill", "applyNext(", "apply mode cycles through the shared "
                                     "helper"),
        ("apply normalise", "applyNorm(",
         "apply mode is normalised by the shared helper, not re-derived"),
    ]:
        check(needle in body, "%s -- %s" % (label, why))

    # ---- v927: row parity with the painted benchmark ----------------------
    check("split(/[/\\\\]/).pop()" in dom,
          "the row shows the FILE name, like the painted view -- not the path")
    check("charAt(0).toUpperCase()" in dom,
          "the badge falls back to the file's initial (the M/P in the "
          "classic node), same rule as the painted view")
    check(calls(dom, "openPreviewOverlay"),
          "the badge opens the SHARED preview overlay")
    check("Weight / CLIP Strength" in dom,
          "the weight column is captioned like the benchmark")

    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    print("v928: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
