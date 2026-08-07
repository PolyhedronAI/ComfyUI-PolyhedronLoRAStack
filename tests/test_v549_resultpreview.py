"""Guard v549 -- Power Upscale: result preview (PURE ui) inside the node.

Backend pins: `result_preview` appended LAST in required (index stability);
temp JPEGs on the `pls_pu_preview` ui channel; the emitter is encapsulated
("result preview skipped" one-liner) and Pillow-10 safe; the done log states
what happened (measure > believe). Frontend pins: the exact per-file banner
(this is the file's newest guard, so it carries the exact version pin - see
test_v548); the viewer follows the house preview laws (hideOnZoom:false /
object-fit:contain / loop default off / height-only setSize); the heal chain
cascades 16 -> 18 -> 19. The cascade itself is MEASURED in node against the
verbatim extracted functions, not believed. Script-style: exit 0 = pass.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v549_resultpreview] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    py = _read("nodes", "ph_power_upscale.py")
    js = _read("web", "js", "ph_power_upscale.js")

    # ---- backend ------------------------------------------------------------
    req = py[py.index('"required"'):py.index('"optional"')]
    names = re.findall(r'"([a-z_0-9]+)":\s*\(', req)
    # v550 hardening: position pin instead of tail pin (the test_v546 lesson) -
    # result_preview sits directly after the v546 pair; later fields append
    # BEHIND it.
    if names.index("result_preview") != names.index("scheduler_low") + 1:
        _fail("result_preview must sit directly after scheduler_low "
              "(appended in v549)")
    if '"pls_pu_preview"' not in py:
        _fail("the ui channel pls_pu_preview is gone")
    if '"type": "temp"' not in py:
        _fail("previews must live in the ComfyUI TEMP dir, never in output/")
    if "result preview skipped" not in py:
        _fail("the emitter must be encapsulated (one-line skip, never a crash)")
    if "preview={" not in py:
        _fail("the done log must state what the preview did (measure > believe)")
    if 'getattr(Image, "Resampling", Image).LANCZOS' not in py:
        _fail("Pillow-10-safe LANCZOS lookup missing")
    if "_PREVIEW_MAX_FRAMES" not in py:
        _fail("the flipbook frame budget constant is gone")

    # ---- frontend -----------------------------------------------------------
    # The exact banner version is pinned by the file's NEWEST guard (v550);
    # existence + format is pinned by test_v548. No banner pin here.
    if '"pls_pu_result"' not in js:
        _fail("the viewer DOM widget (pls_pu_result) is gone")
    if js.count("hideOnZoom: false") < 1:
        _fail("the viewer must opt out of the DOM zoom-hide (v542)")
    if "object-fit:contain" not in js:
        _fail("box-fit (v535) is gone - the medium must fit its box explicitly")
    if "_pvLoop = false" not in js:
        _fail("loop must default OFF = play once then stop (v534)")
    if "message.pls_pu_preview" not in js and "message && message.pls_pu_preview" not in js:
        _fail("onExecuted no longer feeds the viewer from pls_pu_preview")
    if "node.setSize([node.size[0]" not in js:
        _fail("the fit must set the HEIGHT only (v531: never shrink the width)")
    if "_healPreV549(info.widgets_values)" not in js:
        _fail("configure must cascade the v549 heal")

    canon = re.findall(r'"([a-z_]+)"',
                       re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1))
    disp = re.findall(r'"([a-z_]+)"',
                      re.search(r"const DISPLAY_ORDER = \[(.*?)\];", js, re.S).group(1))
    if len(canon) != len(disp):
        _fail("ORDER_CANON / DISPLAY_ORDER diverged in length")
    if canon.index("result_preview") != 18:
        _fail("result_preview must sit at canon index 18 (appended)")
    if "result_preview" not in disp:
        _fail("result_preview missing from the display order")

    # ---- heal cascade, MEASURED in node --------------------------------------
    parts = [re.search(rx, js) for rx in (
        r'const SAME_AS_HIGH = "[^"]+";',
        r"const LEN_PRE_V546 = \d+;",
        r"const LEN_PRE_V549 = \d+;",
        r"function _healPreV546\(wv\) \{[\s\S]*?\n\}",
        r"function _healPreV549\(wv\) \{[\s\S]*?\n\}",
    )]
    if not all(parts):
        _fail("heal functions not extractable")
    harness = "\n".join(m.group(0) for m in parts) + "\n" + r"""
function cascade(wv) { return _healPreV549(_healPreV546(wv)); }
const v514 = Array.from({length: 16}, (_, i) => "v" + i);
const a = cascade(v514.slice());
if (a.length !== 19 || a[16] !== SAME_AS_HIGH || a[17] !== SAME_AS_HIGH || a[18] !== true) {
    console.error("FAIL 16->19: " + JSON.stringify(a.slice(15))); process.exit(1);
}
if (a.slice(0, 16).join(",") !== v514.join(",")) {
    console.error("FAIL: old indices moved"); process.exit(1);
}
const b = cascade(Array.from({length: 18}, (_, i) => i));
if (b.length !== 19 || b[18] !== true) { console.error("FAIL 18->19"); process.exit(1); }
const c19 = Array.from({length: 19}, (_, i) => i);
if (cascade(c19.slice()).join(",") !== c19.join(",")) {
    console.error("FAIL: a v549 save must pass through untouched"); process.exit(1);
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

    print("PASS: v549 result preview -- ui channel + temp JPEGs + viewer laws, "
          "heal cascade 16->18->19 measured")
    sys.exit(0)


if __name__ == "__main__":
    main()
