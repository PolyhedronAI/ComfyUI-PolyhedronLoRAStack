#!/usr/bin/env python3
"""v943 -- P0 of the parity concept + F8 (the stale renderer notice).

F8: under Nodes 2.0 loaded from a workflow (Frank's path, 11.09.2026) eleven
painted nodes got "UI needs LiteGraph renderer ... Your rows are safe" -- all
eleven measured usable under Nodes 2.0 by S5 (10.09.). uls_compat.js now
names them (VUE_USABLE): no probe, no notice. Nodes with their own view are
skipped through their _ulsDomPanel mark.

  C  every painted class (POLY_CANVAS_NODES) is classified: it registers a
     Nodes 2.0 view OR is measured usable -- never both, never neither; the
     set that would still get the notice is empty today
  P  the probe skips measured-usable nodes (no probe at nodeCreated, and the
     judgement skips them); the notice texts are generic
  T  tools/browser_parity_sheet.py: present, compiles, classifies F1-F8,
     writes the classic md5 manifest and fails on a changed classic crop
"""
import os
import py_compile
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPAT = os.path.join(ROOT, "web", "js", "uls_compat.js")
TOOL = os.path.join(ROOT, "tools", "browser_parity_sheet.py")
failures = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def js_set(src, name):
    m = re.search(r"const %s = new Set\(\[(.*?)\]\);" % name, src, re.S)
    return set(re.findall(r'"([A-Za-z0-9_]+)"', m.group(1))) if m else None


def main():
    print("v943: parity P0 + the stale renderer notice (F8)")
    src = open(COMPAT, encoding="utf-8").read()
    painted, usable = js_set(src, "POLY_CANVAS_NODES"), js_set(src, "VUE_USABLE")
    check(painted is not None and usable is not None, "C  both sets are parseable")
    if painted is None or usable is None:
        return 1
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import renderer_scan as RS  # noqa: E402
    views = RS.vue_views(ROOT)
    check(usable <= painted, "C  every measured-usable class is a painted class")
    check(not (usable & views), "C  no class is both measured-usable and viewed (%s)" % sorted(usable & views))
    unclassified = sorted(painted - usable - views)
    check(not unclassified,
          "C  every painted class is classified -- view or measured usable (unclassified: %s)" % unclassified)
    # public build (v374): 14 painted classes ship here -- 6 measured usable
    # (AnySwitch x2, Empty Latent, Note, Seed, Wan Sigma Schedule), 8 with a
    # view (Stack, Engine, CLIP Text Encode, Filter, Int, Load CLIP/Model/VAE);
    # the internal tree counts 11 / 9 with its six internal-only nodes.
    # v376 (internal v959): AnySwitch x2 gained a view -> 4 / 10.
    check(len(usable) == 4 and len(painted & views) == 10,
          "C  today: 4 measured usable, 10 with a view (%d / %d)" % (len(usable), len(painted & views)))
    code = re.sub(r"/\*.*?\*/", "", src, flags=re.S)
    code = re.sub(r"^\s*//.*$", "", code, flags=re.M)
    check(re.search(r"if\s*\(\s*n\._ulsDomPanel\s*\|\|\s*VUE_USABLE\.has\(t\)\s*\)\s*\{\s*dropNotice\(n\);\s*continue;", code)
          is not None, "P  the judgement skips viewed AND measured-usable nodes")
    check(re.search(r"if\s*\(\s*VUE_USABLE\.has\(t\)\s*\)\s*return;", code) is not None,
          "P  no probe is installed on a measured-usable node")
    check("Your rows are safe" not in code and "Stack/Engine node is placed" not in code,
          "P  the notice texts are generic (no Stack-only wording)")
    try:
        py_compile.compile(TOOL, doraise=True)
        ok = True
    except Exception:
        ok = False
    check(ok, "T  tools/browser_parity_sheet.py compiles")
    t = open(TOOL, encoding="utf-8").read()
    for code_ in ("F1", "F2", "F3", "F4", "F5", "F8"):
        check('"%s"' % code_ in t, "T  the classifier reports %s" % code_)
    check("classic_md5.json" in t and "--ref" in t and "rc = 1 if (changed or missing)" in t,
          "T  classic crops are fingerprinted and a changed one fails the run")
    check("Comfy.VueNodes.Enabled', v)" in t and "page.reload()" in t,
          "T  Nodes 2.0 is switched on BEFORE the scene loads (Frank's path)")
    return 1 if failures else 0


if __name__ == "__main__":
    rc = main()
    print("v943: %s" % ("FAIL (%d)" % len(failures) if failures else "PASS"))
    sys.exit(rc)
