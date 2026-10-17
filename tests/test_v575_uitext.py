"""Guard v575 -- UI text carries no build numbers.

FRANK'S LAW, STATED ON THE v574 SCREENSHOT: version tokens (v568, v569,
...) are for US, for building - changelogs, code comments, console
telemetry, guards. A TOOLTIP is read by a person using the node, and
'the v568 sharpness meter' explains nothing to them. Same family as the
'mush' -> 'multi-LoRA interference' convention: public-facing text
speaks substance, not internal history.

AMENDED IN v581 (1st amendment): this guard was BLIND TO THE V3 NODES.
It walked ast for `{"tooltip": "..."}` dict entries and `DESCRIPTION = ...`
assignments -- both V1 shapes. A V3 node declares the same two things as
KEYWORD ARGUMENTS inside io.Schema(...): `description=`, `tooltip=`. Nine
of the thirty-nine registered nodes take the V3 path on Frank's box, and
for all nine this guard has been reporting green over a room it never
entered. It said "no build numbers in tooltips or DESCRIPTIONs" and meant
"...in the V1 ones". Lesson 4: a green guard proves only what it measures.

Measured before amending, not after: zero version tokens were hiding in
those keywords. The gap was real and unexploited. It is now closed anyway,
because the next V3 node is written by someone who trusts this gate.

Scope, precisely:
  * COVERED: every "tooltip" string and every DESCRIPTION class
    attribute in nodes/*.py - parsed with ast (real parsing, not a
    grep that a comment could fool).
  * COVERED SINCE v581: `description=` and `tooltip=` keyword arguments,
    which is how the V3 io.Schema declares the same UI text.
  * EXEMPT: the seven no-touch anchor files. ph_camera.py carries
    v292/v312 in tooltips, and scrubbing them would break the anchor
    composite - the anchor law outranks the UI-text law. The exemption
    is the documented conflict, not an oversight; an anchor-touching
    cleanup release would need its own re-baselining ceremony.
  * NOT covered on purpose: console prints (developer telemetry, house
    style keeps its history there), comments, docstrings, changelogs.

Script-style: exit 0 = pass.
"""
import ast
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Must match the anchor canon (ANCHORS_baseline_*.txt). If the anchor set
# ever changes, this list changes WITH it, in the same cut.
_ANCHOR_NODES = {
    "wan_model_bridge.py", "ph_camera.py", "ph_cardinal_cameras.py",
    "ph_mesh3d.py", "ph_mesh_render.py", "uls_merge_math.py",
    "ph_viewport3d.py",
}

_TOKEN = re.compile(r"\bv\d{3,4}\b")   # v1001: four digits since v1000


def _fail(msg):
    print("[test_v575_uitext] FAIL: " + msg)
    sys.exit(1)


def main():
    offenders = []
    scanned = 0
    # A count the guard must earn. If the V3 walk below silently stops finding
    # keywords -- an io.Schema refactor, a renamed argument -- this number
    # collapses and the guard says so, instead of reporting green over a room
    # it no longer enters. The amendment guards itself.
    v3_seen = 0
    for path in sorted(glob.glob(os.path.join(ROOT, "nodes", "*.py"))):
        name = os.path.basename(path)
        if name in _ANCHOR_NODES:
            continue
        scanned += 1
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if (isinstance(k, ast.Constant) and k.value == "tooltip"
                            and isinstance(v, ast.Constant)):
                        for tok in _TOKEN.findall(str(v.value)):
                            offenders.append((name, "tooltip", tok))
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if (isinstance(t, ast.Name) and t.id == "DESCRIPTION"
                            and isinstance(node.value, ast.Constant)):
                        for tok in _TOKEN.findall(str(node.value.value)):
                            offenders.append((name, "DESCRIPTION", tok))
            # v581: the V3 shape. io.Schema(description=..., ...) and
            # io.Image.Input("x", tooltip=...) carry the same public text
            # through a keyword, which the two branches above never visit.
            if isinstance(node, ast.Call):
                for kw in node.keywords:
                    if (kw.arg in ("description", "tooltip")
                            and isinstance(kw.value, ast.Constant)):
                        v3_seen += 1
                        for tok in _TOKEN.findall(str(kw.value.value)):
                            offenders.append((name, f"{kw.arg}=", tok))
    if scanned < 10:
        _fail(f"only {scanned} node files scanned - the glob broke, this "
              f"guard is asleep")
    if v3_seen < 5:
        _fail(f"only {v3_seen} V3 description=/tooltip= keywords seen - the "
              f"pack has nine V3 nodes, so this walk is not reaching them and "
              f"the v581 amendment is asleep")
    if offenders:
        _fail("build numbers leaked into UI text: " + ", ".join(
            f"{n}:{w}:{t}" for n, w, t in offenders)
            + " - tooltips speak substance, not internal history")

    print(f"[test_v575_uitext] OK - {scanned} non-anchor node files "
          f"ast-scanned, no build numbers in tooltips or DESCRIPTIONs, "
          f"V1 attributes and {v3_seen} V3 description=/tooltip= keywords "
          f"alike (the seven anchors stay frozen by the harder law)")
    sys.exit(0)


if __name__ == "__main__":
    main()
