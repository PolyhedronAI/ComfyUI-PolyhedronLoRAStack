"""Guard v550 -- Power Upscale: process view (per-step tile probe).

Backend pins: `process_preview` appended LAST in required; the probe ships
over the send_sync event "polyhedron.pu_tile"; it is throttled
(_PROBE_MIN_INTERVAL) and DISARMS on first failure ("process view disarmed" /
"process view unavailable" one-liners - a preview can never cost a render);
`_current_node_id` is IMPORTED from uls_sampler in BOTH import branches (house
pattern, see test_v546); `_refine_tiles` gained `tile_probe=None` and the call
site passes the probe. Frontend pins: the exact per-file banner (this file's
newest guard carries the exact version pin - see test_v548/test_v549); the api
import; ONE global listener with getNodeById + comfyClass check; the process
DOM widget with hideOnZoom:false; the execution lifecycle tags; the heal chain
now cascades 16 -> 18 -> 19 -> 20, MEASURED in node against the verbatim
extracted functions, not believed. Script-style: exit 0 = pass.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v550_processview] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    py = _read("nodes", "ph_power_upscale.py")
    js = _read("web", "js", "ph_power_upscale.js")

    # ---- backend ------------------------------------------------------------
    req = py[py.index('"required"'):py.index('"optional"')]
    names = re.findall(r'"([a-z_0-9]+)":\s*\(', req)
    # v553 hardening: position pin instead of tail pin (the test_v546 lesson) -
    # process_preview sits directly after result_preview; later fields append
    # BEHIND it.
    if names.index("process_preview") != names.index("result_preview") + 1:
        _fail("process_preview must sit directly after result_preview "
              "(appended in v550)")
    if '_PROBE_EVENT = "polyhedron.pu_tile"' not in py:
        _fail("the probe event name changed - the frontend listener would starve")
    if "_PROBE_MIN_INTERVAL" not in py:
        _fail("the probe throttle constant is gone")
    if "process view disarmed" not in py or "process view unavailable" not in py:
        _fail("the probe must disarm on first failure with one log line")
    if py.count("_current_node_id") < 3:
        _fail("_current_node_id must be IMPORTED from uls_sampler in BOTH "
              "branches and called once (house pattern)")
    if "tile_probe=None" not in py:
        _fail("_refine_tiles lost its tile_probe parameter")
    if "tile_probe=probe" not in py:
        _fail("the call site no longer passes the probe into _refine_tiles")
    if 'str(process_preview) != "Off"' not in py:
        _fail("Off must mean the probe is NEVER built (zero overhead)")

    # ---- frontend -----------------------------------------------------------
    # The exact banner version is pinned by the file's NEWEST guard (v552);
    # existence + format is pinned by test_v548. No banner pin here.
    if 'import { api } from "../../scripts/api.js";' not in js:
        _fail("the api import is gone - no events without it")
    if 'api.addEventListener("polyhedron.pu_tile"' not in js:
        _fail("the global tile listener is gone")
    if "getNodeById" not in js or "node.type !== NODE_TYPE" not in js:
        _fail("the listener must route by node id AND node.type - the "
              "LIVE-PROVEN uls_live_preview pattern (v552 fix: comfyClass may "
              "be unset and silently dropped every event)")
    if '"pls_pu_process"' not in js:
        _fail("the process DOM widget (pls_pu_process) is gone")
    if js.count("hideOnZoom: false") < 2:
        _fail("BOTH panes must opt out of the DOM zoom-hide (v542)")
    if "execution_interrupted" not in js:
        _fail("the lifecycle tags (done/stopped) are gone")
    if "_healPreV550(info.widgets_values)" not in js:
        _fail("configure must cascade the v550 heal")

    canon = re.findall(r'"([a-z_]+)"',
                       re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1))
    disp = re.findall(r'"([a-z_]+)"',
                      re.search(r"const DISPLAY_ORDER = \[(.*?)\];", js, re.S).group(1))
    if len(canon) != len(disp):
        _fail("ORDER_CANON / DISPLAY_ORDER diverged in length")
    if canon.index("process_preview") != 19:
        _fail("process_preview must sit at canon index 19 (appended)")
    if "process_preview" not in disp:
        _fail("process_preview missing from the display order")

    # ---- heal cascade 16 -> 18 -> 19 -> 20, MEASURED in node -----------------
    parts = [re.search(rx, js) for rx in (
        r'const SAME_AS_HIGH = "[^"]+";',
        r"const LEN_PRE_V546 = \d+;",
        r"const LEN_PRE_V549 = \d+;",
        r"const LEN_PRE_V550 = \d+;",
        r"function _healPreV546\(wv\) \{[\s\S]*?\n\}",
        r"function _healPreV549\(wv\) \{[\s\S]*?\n\}",
        r"function _healPreV550\(wv\) \{[\s\S]*?\n\}",
    )]
    if not all(parts):
        _fail("heal functions not extractable")
    harness = "\n".join(m.group(0) for m in parts) + "\n" + r"""
function cascade(wv) { return _healPreV550(_healPreV549(_healPreV546(wv))); }
const v514 = Array.from({length: 16}, (_, i) => "v" + i);
const a = cascade(v514.slice());
if (a.length !== 20 || a[16] !== SAME_AS_HIGH || a[17] !== SAME_AS_HIGH ||
    a[18] !== true || a[19] !== "Off") {
    console.error("FAIL 16->20: " + JSON.stringify(a.slice(15))); process.exit(1);
}
if (a.slice(0, 16).join(",") !== v514.join(",")) {
    console.error("FAIL: old indices moved"); process.exit(1);
}
const b = cascade(Array.from({length: 18}, (_, i) => i));
if (b.length !== 20 || b[18] !== true || b[19] !== "Off") {
    console.error("FAIL 18->20"); process.exit(1);
}
const c = cascade(Array.from({length: 19}, (_, i) => i));
if (c.length !== 20 || c[19] !== "Off") { console.error("FAIL 19->20"); process.exit(1); }
const d20 = Array.from({length: 20}, (_, i) => i);
if (cascade(d20.slice()).join(",") !== d20.join(",")) {
    console.error("FAIL: a v550 save must pass through untouched"); process.exit(1);
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

    print("PASS: v550 process view -- event + throttle + disarm + house-pattern "
          "import, heal cascade 16->18->19->20 measured")
    sys.exit(0)


if __name__ == "__main__":
    main()
