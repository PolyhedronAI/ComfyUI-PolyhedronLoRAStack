"""Guard v557 -- CLIP Text Encode: the widget-hide fix + free comment markers.

TWO BUGS, both pinned so they can never come back:

  1. `_hide` set computeSize + hidden but NOT `type`. LiteGraph skips a widget
     in the LAYOUT pass by its TYPE marker; without it the canvas widgets
     closed the gap while the textarea (a DOM element positioned from that
     layout) stayed put -> the overlap. The HIDDEN_PREFIX type swap is pinned.
  2. On show, `computeSize = undefined` was written back whenever the original
     widget had none - leaving it zero-height forever. `delete w.computeSize`
     is pinned, and assigning undefined is forbidden.

Plus: the layout is measured in requestAnimationFrame (a textarea reports its
height only after the browser laid it out) and the hide is idempotent.

FREE COMMENT MARKERS: `_comment_rx` + `_clean` are extracted and EXECUTED
against the marker matrix - default "//", custom "#" / "***", a
space-separated list, the empty list (fail soft: nothing stripped), URL
survival, and both "//Comment" and "// Comment". The heal 13 -> 14 is
measured in node. Script-style: exit 0 = pass.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v557_markers_layout] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    py = _read("nodes", "ph_clip_encode.py")
    js = _read("web", "js", "ph_clip_encode.js")

    # ---- marker engine: extracted and RUN ------------------------------------
    src = ["import re"]
    for rx in (r'_DEFAULT_MARKERS = "[^"]*"', r"_RX_CACHE = \{\}",
               r"def _comment_rx\(.*?\n(?=\n\ndef )",
               r"def _clean\(.*?\n(?=\n\ndef )"):
        m = re.search(rx, py, re.S)
        if not m:
            _fail(f"not extractable: {rx}")
        src.append(m.group(0))
    ns = {}
    exec("\n".join(src), ns)  # noqa: S102 - our own source, measured
    clean = ns["_clean"]

    cases = [
        # (text, markers, expected)
        ("// Head\nRAW photo", "//", "RAW photo"),
        ("//Head\nRAW photo", "//", "RAW photo"),          # no space needed
        ("# Head\nRAW photo", "#", "RAW photo"),
        ("*** Head\nRAW photo", "***", "RAW photo"),
        ("// A\n# B\n*** C\nkeep me", "// # ***", "keep me"),
        ("see https://x.io/y now", "//", "see https://x.io/y now"),
        ("// Head\nRAW", "", "// Head RAW"),               # no marker -> fail soft
        ("RAW photo // trailing note", "//", "RAW photo"),  # inline comment
    ]
    for text, markers, want in cases:
        got = clean(text, True, True, markers)
        if got != want:
            _fail(f"_clean({text!r}, markers={markers!r}) -> {got!r}, "
                  f"expected {want!r}")
    # strip_comments Off must ignore the markers entirely
    if clean("// Head\nRAW", False, True, "//") != "// Head RAW":
        _fail("strip_comments Off must keep the comments (historic behaviour)")

    # ---- backend pins ---------------------------------------------------------
    req = py[py.index('"required"'):py.index('"optional"')]
    names = re.findall(r'"([a-z_0-9]+)":\s*\(', req)
    if "comment_markers" not in names:
        _fail("comment_markers is gone")

    # v603 SUPERSEDES the old form of this check, and it is worth being exact about
    # why, because the old form was not wrong -- it was right for its time.
    #
    # v557's law read: "comment_markers must be appended at the END, so every
    # existing index keeps its slot." That was the correct law WHEN THERE WAS NO
    # HEAL. Appending is the only reorder that costs nothing, and v557 had nothing
    # to pay with.
    #
    # v603 moved the filters ABOVE the prompt boxes on Frank's call, which renumbers
    # every slot -- and PAID for it, with _healPreV603 permuting old arrays before
    # LiteGraph ever sees them.
    #
    # So the law is not "never reorder". The law is: NEVER REORDER WITHOUT A HEAL.
    # test_v603_order proves the heal is correct and is actually invoked. This line
    # proves it EXISTS -- so nobody can shuffle INPUT_TYPES, watch this guard stay
    # green, and quietly ship a build that pours every saved prompt into `separator`.
    if names[-1] != "comment_markers":
        if "_healPreV603" not in js:
            _fail("the widget order was changed and there is NO permutation heal in the JS. "
                  "LiteGraph serialises widget values POSITIONALLY: every saved workflow will "
                  "load its prompts into the filter widgets, in silence, with no error and no "
                  "warning. Append at the end (v557's law) or write the heal (v603's).")
    if "_RX_CACHE" not in py:
        _fail("the regex cache is gone (a rebuild per line would be silly)")
    if "re.escape" not in py:
        _fail("markers must be regex-escaped ('***' is not a quantifier)")

    # ---- THE LAYOUT FIX -------------------------------------------------------
    if "HIDDEN_PREFIX" not in js or "w.type = HIDDEN_PREFIX + w.type" not in js:
        _fail("the widget TYPE swap is gone - without it LiteGraph keeps the "
              "widget in the layout pass and the textarea overlaps (v556 bug 1)")
    if "delete w.computeSize" not in js:
        _fail("on show, a widget without an original computeSize must have the "
              "override DELETED, never set to undefined (v556 bug 2)")
    if "w.computeSize = undefined" in js:
        _fail("assigning undefined back to computeSize is exactly the v556 bug")
    if "_pls_hadCS" not in js:
        _fail("the 'did it have its own computeSize?' memory is gone")
    if "requestAnimationFrame" not in js:
        _fail("the fit must be measured AFTER the browser laid the textareas "
              "out (rAF) - measuring too early squeezed the node")
    if "if (hidden === isHidden) return;" not in js:
        _fail("the hide must be idempotent (never stack the type swaps)")
    if "CTE layout:" not in js:
        _fail("the layout marker is gone - the next screenshot must PROVE the fit")

    # ---- heal 13 -> 14, MEASURED in node ---------------------------------------
    # The exact banner version is pinned by the file's NEWEST guard (v560).
    if "nodeType.prototype.configure" not in js:
        _fail("the heal must hook prototype.configure (the live-proven point "
              "from ph_power_upscale), not onConfigure")
    parts = [re.search(rx, js) for rx in (
        r"const LEN_PRE_V557 = \d+;",
        r"function _healPreV557\(wv\) \{[\s\S]*?\n\}",
    )]
    if not all(parts):
        _fail("the heal is not extractable")
    harness = "\n".join(m.group(0) for m in parts) + "\n" + r"""
const v556 = Array.from({length: 13}, (_, i) => "w" + i);
const a = _healPreV557(v556.slice());
if (a.length !== 14 || a[13] !== "//") {
    console.error("FAIL 13->14: " + JSON.stringify(a)); process.exit(1);
}
if (a.slice(0, 13).join(",") !== v556.join(",")) {
    console.error("FAIL: old indices moved"); process.exit(1);
}
const c14 = Array.from({length: 14}, (_, i) => i);
if (_healPreV557(c14.slice()).join(",") !== c14.join(",")) {
    console.error("FAIL: a v557 save must pass through untouched"); process.exit(1);
}
console.log("OK");
"""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                      encoding="utf-8")
    try:
        tmp.write(harness)
        tmp.close()
        proc = subprocess.run(["node", tmp.name], capture_output=True, text=True)
    finally:
        os.unlink(tmp.name)
    if proc.returncode != 0 or "OK" not in proc.stdout:
        _fail("heal harness failed:\n" + proc.stdout + proc.stderr)

    print("PASS: v557 -- layout fix pinned (type swap + delete computeSize + "
          "rAF + idempotent), marker matrix executed, heal 13->14 measured")
    sys.exit(0)


if __name__ == "__main__":
    main()
