"""Guard v555 -- Power Upscale: selectable stage-fit method.

The core promise is the BYTE-IDENTICAL default: "lanczos (cpu)" IS the
historic v514 fit, so every old workflow (and every heal) reproduces its
pixels exactly; speed is one explicit click away (torch methods chunked on
the GPU, or the Maxine nvidia_rtx_vsr capsule). The resize machine moved
FROM ph_fast_upscale INTO ph_power_upscale (the import direction Fast -> PU
already existed, so ONE source of truth without a cycle) - this guard pins
the machine's new home, the dispatch, the deliberate lanczos in the rare
per-tile VAE guard (a corrective fit must never spin up a VSR context), the
`fit=` field in the BEGIN telemetry, and the heal cascade
16 -> 18 -> 19 -> 20 -> 21 -> 22 MEASURED in node. Script-style: exit 0 =
pass.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v555_resizemethod] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    pu = _read("nodes", "ph_power_upscale.py")
    js = _read("web", "js", "ph_power_upscale.js")

    # ---- widget + byte-identical default --------------------------------------
    req = pu[pu.index('"required"'):pu.index('"optional"')]
    names = re.findall(r'"([a-z_0-9]+)":\s*\(', req)
    if names.index("resize_method") != names.index("mute_staging_logs") + 1:
        _fail("resize_method must sit directly after mute_staging_logs "
              "(appended in v555)")
    if '"default": "lanczos (cpu)"' not in req:
        _fail("the default must stay 'lanczos (cpu)' - the byte-identical "
              "historic path (doctrine)")
    if 'resize_method="lanczos (cpu)"' not in pu:
        _fail("the signature default lost the byte-identical path")

    # ---- machine home + dispatch ------------------------------------------------
    for marker in ("_METHODS = {", "def _chunks", "def _resize_chunked",
                   "def _vsr_resize"):
        if marker not in pu:
            _fail(f"the resize machine must live in ph_power_upscale: "
                  f"{marker!r} missing")
    # v560: the call also carries per_batch now (the OOM guard).
    if "_esrgan_pass(cur, um, sw, sh, resize_method" not in pu:
        _fail("the stage no longer passes the chosen fit into _esrgan_pass")
    # v559: the dispatch now runs on the CHECKED method (_fit_method overrules
    # a VSR shrink to area), not on the raw widget value.
    if "_resize_chunked(image, width, height, method," not in pu:
        _fail("the torch dispatch is gone")
    if "_fit_method(" not in pu:
        _fail("the stage fit must pass through the v559 downscale guard")
    if "_vsr_resize(image, width, height)" not in pu:
        _fail("the VSR dispatch is gone")
    if "_RESIZE_PER_BATCH" not in pu:
        _fail("the internal GPU-fit chunk constant is gone")

    # ---- the rare per-tile VAE guard stays lanczos, documented -----------------
    if "must never spin up a VSR context" not in pu:
        _fail("the per-tile corrective fit must stay lanczos, and say why")

    # ---- telemetry ---------------------------------------------------------------
    if "fit={resize_method}" not in pu:
        _fail("the BEGIN line must state the chosen fit (measure > believe)")

    # ---- frontend -----------------------------------------------------------------
    # The exact banner version is pinned by the file's NEWEST guard (v560).
    if "_healPreV555(info.widgets_values)" not in js:
        _fail("configure must cascade the v555 heal")
    canon = re.findall(r'"([a-z_ ()]+)"',
                       re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1))
    disp = re.findall(r'"([a-z_ ()]+)"',
                      re.search(r"const DISPLAY_ORDER = \[(.*?)\];", js, re.S).group(1))
    if len(canon) != len(disp):
        _fail("ORDER_CANON / DISPLAY_ORDER diverged in length")
    if canon.index("resize_method") != 21:
        _fail("resize_method must sit at canon index 21 (appended)")
    if "resize_method" not in disp:
        _fail("resize_method missing from the display order")

    # ---- heal cascade 16 -> ... -> 22, MEASURED in node --------------------------
    parts = [re.search(rx, js) for rx in (
        r'const SAME_AS_HIGH = "[^"]+";',
        r"const LEN_PRE_V546 = \d+;", r"const LEN_PRE_V549 = \d+;",
        r"const LEN_PRE_V550 = \d+;", r"const LEN_PRE_V553 = \d+;",
        r"const LEN_PRE_V555 = \d+;",
        r"function _healPreV546\(wv\) \{[\s\S]*?\n\}",
        r"function _healPreV549\(wv\) \{[\s\S]*?\n\}",
        r"function _healPreV550\(wv\) \{[\s\S]*?\n\}",
        r"function _healPreV553\(wv\) \{[\s\S]*?\n\}",
        r"function _healPreV555\(wv\) \{[\s\S]*?\n\}",
    )]
    if not all(parts):
        _fail("heal functions not extractable")
    harness = "\n".join(m.group(0) for m in parts) + "\n" + r"""
function cascade(wv) {
    return _healPreV555(_healPreV553(_healPreV550(_healPreV549(_healPreV546(wv)))));
}
const v514 = Array.from({length: 16}, (_, i) => "v" + i);
const a = cascade(v514.slice());
if (a.length !== 22 || a[18] !== true || a[19] !== "Off" || a[20] !== true ||
    a[21] !== "lanczos (cpu)") {
    console.error("FAIL 16->22: " + JSON.stringify(a.slice(15))); process.exit(1);
}
const b = cascade(Array.from({length: 21}, (_, i) => i));
if (b.length !== 22 || b[21] !== "lanczos (cpu)") {
    console.error("FAIL 21->22"); process.exit(1);
}
const c22 = Array.from({length: 22}, (_, i) => i);
if (cascade(c22.slice()).join(",") !== c22.join(",")) {
    console.error("FAIL: a v555 save must pass through untouched"); process.exit(1);
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

    print("PASS: v555 -- resize_method with the byte-identical lanczos default, "
          "machine moved to PU (one source), heal cascade to 22 measured")
    sys.exit(0)


if __name__ == "__main__":
    main()
