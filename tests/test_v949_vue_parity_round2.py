#!/usr/bin/env python3
"""v949 -- Nodes 2.0 parity round 2: the three field findings stay fixed.

1. Stack panel paints group colour from the SHARED palette (uls_node.js exports
   GROUP_COLORS, uls_stack_dom.js imports it and uses it for pill, stripe and
   badge; the pill label is the painted 4-letter upper-case form).
2. Live preview hands each frame through the frontend's own preview door
   (b_preview_with_metadata) -- only in Vue mode, as a Blob.
3. The tint bridge carries the field height: bridgedHeight = max(classic px,
   the Vue field's own content height), overflow hidden. Driven in node.

Run: python tests/test_v949_vue_parity_round2.py
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = os.path.join(ROOT, "web", "js")


def read(name):
    with open(os.path.join(JS, name), encoding="utf-8") as f:
        return f.read()


FAILS = []


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def body_after_imports(src):
    # drop the import block so a name in an import line is not counted as use
    return re.sub(r"^import[\s\S]*?from\s+\"[^\"]+\";\s*$", "", src, flags=re.M)


def main():
    print("v949: Nodes 2.0 parity round 2")
    node = read("uls_node.js")
    dom = read("uls_stack_dom.js")
    prev = read("uls_live_preview.js")
    par = read("uls_vue_parity.js")

    # ── 1. shared palette ───────────────────────────────────────────────
    exp = node.split("export {", 1)[1].split("};", 1)[0]
    check("GROUP_COLORS" in exp, "S  uls_node.js exports GROUP_COLORS")
    imp = dom.split('} from "./uls_node.js";', 1)[0]
    check("GROUP_COLORS" in imp, "S  uls_stack_dom.js imports GROUP_COLORS from uls_node.js")
    check(not re.search(r"const\s+GROUP_COLORS\s*=", dom),
          "S  the panel holds no palette copy of its own")
    b = body_after_imports(dom)
    check(re.search(r"GROUP_COLORS\[row\.group\]", b) is not None,
          "S  the row colour is read from GROUP_COLORS[row.group]")
    check(re.search(r"grp\.style\.background\s*=\s*gc\s*\+\s*\"22\"", b) is not None
          and re.search(r"grp\.style\.borderColor\s*=\s*gc\s*\+\s*\"55\"", b) is not None
          and re.search(r"grp\.style\.color\s*=\s*gc;", b) is not None,
          "S  pill fill +22 / border +55 / text = colour, as painted")
    check(re.search(r"row\.group\.slice\(0,\s*4\)\.toUpperCase\(\)", b) is not None,
          "S  pill label is the painted 4-letter upper-case form")
    check(re.search(r"el\.style\.borderLeft\s*=\s*\"3px solid \"\s*\+\s*\(row\.enabled\s*\?\s*gc\s*:\s*\"#282838\"\)", b) is not None,
          "S  3 px group stripe, grey when the row is off (as painted)")
    check(re.search(r"badge\.style\.color\s*=\s*gc", b) is not None,
          "S  preview badge initial in the group colour")

    # ── 2. preview door ─────────────────────────────────────────────────
    check('import { vueMode } from "./uls_vue_views.js";' in prev,
          "S  live preview asks the switchboard (vueMode)")
    pb = body_after_imports(prev)
    check(re.search(r"if\s*\(vueMode\(\)\)\s*\{[^}]*_pushVuePreview\(node,\s*blob\)", pb) is not None,
          "S  the frame goes through the Vue door ONLY in Vue mode")
    check(re.search(r"new CustomEvent\(\"b_preview_with_metadata\"", pb) is not None
          and re.search(r"displayNodeId:\s*String\(node\.id\)", pb) is not None,
          "S  the door is b_preview_with_metadata with the node id")
    check(re.search(r"new Blob\(\[u8\],\s*\{\s*type:\s*\"image/jpeg\"\s*\}\)", pb) is not None
          and re.search(r"img\._ulsBlob\s*=\s*_b64ToBlob\(b64\)", pb) is not None,
          "S  each frame carries a JPEG Blob for the door")
    check(re.search(r"node\.imgs\s*=\s*\[img\];", pb) is not None,
          "S  the classic image area is still fed (node.imgs)")

    # ── 3. height bridge ────────────────────────────────────────────────
    check("export function bridgedHeight(" in par, "S  bridgedHeight exported")
    check(re.search(r"t\.style\[HEIGHT_KEY\]\s*=\s*h;\s*t\.style\.minHeight\s*=\s*h;", par) is not None,
          "S  the classic inline height is COPIED onto the Vue field (v952: no max -- it could only grow)")
    check(re.search(r"t\.style\.overflowY\s*=\s*\"hidden\"", par) is not None,
          "S  the bridged field does not scroll")
    check(re.search(r"const h\s*=\s*src\[HEIGHT_KEY\]\s*\|\|\s*\"\";\s*if\s*\(h\)", par) is not None,
          "S  the bridge fires only where the classic field states a height")

    # driven: bridgedHeight = max(classic, own content); measured with height auto
    fn = re.search(r"export function bridgedHeight\([\s\S]*?\n\}", par).group(0).replace("export ", "")
    harness = fn + r"""
function el(scrollH) { const e = { style: { height: "12px" }, _s: scrollH }; Object.defineProperty(e, "scrollHeight", { get() { return this.style.height === "auto" ? this._s : 5; } }); return e; }
const a = el(838), b = el(300);
const r1 = bridgedHeight("532px", a), r2 = bridgedHeight("532px", b);
console.log(JSON.stringify({ r1, r2, h1: a.style.height, h2: b.style.height }));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as f:
        f.write(harness)
        path = f.name
    try:
        out = subprocess.run(["node", path], capture_output=True, text=True, timeout=60)
    finally:
        os.unlink(path)
    check(out.returncode == 0, "D  harness ran: " + (out.stderr.strip()[-200:] if out.returncode else ""))
    if out.returncode == 0:
        import json
        r = json.loads(out.stdout.strip().splitlines()[-1])
        check(r["r1"] == 838, "D  the Vue field's larger content height wins (838 over 532)")
        check(r["r2"] == 532, "D  the classic height wins where the Vue content is smaller")
        check(r["h1"] == "12px" and r["h2"] == "12px",
              "D  measuring with height 'auto' restores the previous height (no loop)")

    print("v949: %s" % ("PASS" if not FAILS else "FAIL (%d)" % len(FAILS)))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
