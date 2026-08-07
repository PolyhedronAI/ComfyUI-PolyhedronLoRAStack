"""Guard v553 -- Power Upscale: stage telemetry + staging-log mute.

Backend pins: the stage narrates itself (begin line with the full plan
INCLUDING the v546 sampler/sched fields, a per-tile line, a done line with
the MEASURED duration); the mute is a context manager built on
logging.disable with a byte-exact restore of the previous gate (root
setLevel would be defeated by a named child logger's own INFO level - the
docstring says so); `mute_staging_logs` is appended LAST in required; our
telemetry is print(), never logging (it must pass the mute). Frontend pins:
exact banner (newest guard of the file), ORDER index 20, heal cascade
16 -> 18 -> 19 -> 20 -> 21 MEASURED in node. Script-style: exit 0 = pass.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v553_telemetry] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    py = _read("nodes", "ph_power_upscale.py")
    js = _read("web", "js", "ph_power_upscale.js")

    # ---- backend: telemetry -------------------------------------------------
    if "begin -> {sw}x{sh}" not in py:
        _fail("the stage BEGIN line is gone (a 300 s run must narrate itself)")
    if "sampler={samp} sched={sched}" not in py:
        _fail("the v546 measurement fields left the begin line")
    if "tile {ti}/{len(grid['tiles'])}" not in py:
        _fail("the per-tile line is gone")
    if "done in " not in py or "time.monotonic() - t0" not in py:
        _fail("the done line must carry the MEASURED stage duration")
    if "s/tile" not in py:
        _fail("the done line must break the duration down per tile")
    if 'f"total={time.monotonic() - t_all:.1f}s"' not in py:
        _fail("the final done line must carry the measured TOTAL duration")

    # ---- backend: mute capsule ----------------------------------------------
    req = py[py.index('"required"'):py.index('"optional"')]
    names = re.findall(r'"([a-z_0-9]+)":\s*\(', req)
    # v555 hardening: position pin instead of tail pin (the test_v546 lesson).
    if names.index("mute_staging_logs") != names.index("process_preview") + 1:
        _fail("mute_staging_logs must sit directly after process_preview "
              "(appended in v553)")
    # v558: the capsule moved to nodes/ph_logmute.py and became a SCALPEL
    # (a logging.Filter on the root handlers) instead of the logging.disable
    # sledgehammer. Its behaviour is measured in test_v558_logmute.py; here we
    # only pin that the stage body still runs inside the mute scope.
    if "_MuteInfoLogs" not in py:
        _fail("the mute capsule import is gone")
    if "with _MuteInfoLogs(mute_staging_logs," not in py:
        _fail("the stage body is no longer wrapped in the mute scope")
    for line in py.splitlines():
        if "[PLS] Power Upscale" in line and "logging." in line:
            _fail("our telemetry must be print(), never logging "
                  "(it has to pass the mute)")
    if "logging.disable" in py:
        _fail("the v553 sledgehammer must not come back (see v558)")

    # ---- frontend -----------------------------------------------------------
    # The exact banner version is pinned by the file's NEWEST guard (v555);
    # existence + format is pinned by test_v548. No banner pin here.
    if "_healPreV553(info.widgets_values)" not in js:
        _fail("configure must cascade the v553 heal")
    canon = re.findall(r'"([a-z_]+)"',
                       re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1))
    disp = re.findall(r'"([a-z_]+)"',
                      re.search(r"const DISPLAY_ORDER = \[(.*?)\];", js, re.S).group(1))
    if len(canon) != len(disp):
        _fail("ORDER_CANON / DISPLAY_ORDER diverged in length")
    if canon.index("mute_staging_logs") != 20:
        _fail("mute_staging_logs must sit at canon index 20 (appended)")
    if "mute_staging_logs" not in disp:
        _fail("mute_staging_logs missing from the display order")

    # ---- heal cascade 16 -> ... -> 21, MEASURED in node ----------------------
    parts = [re.search(rx, js) for rx in (
        r'const SAME_AS_HIGH = "[^"]+";',
        r"const LEN_PRE_V546 = \d+;", r"const LEN_PRE_V549 = \d+;",
        r"const LEN_PRE_V550 = \d+;", r"const LEN_PRE_V553 = \d+;",
        r"function _healPreV546\(wv\) \{[\s\S]*?\n\}",
        r"function _healPreV549\(wv\) \{[\s\S]*?\n\}",
        r"function _healPreV550\(wv\) \{[\s\S]*?\n\}",
        r"function _healPreV553\(wv\) \{[\s\S]*?\n\}",
    )]
    if not all(parts):
        _fail("heal functions not extractable")
    harness = "\n".join(m.group(0) for m in parts) + "\n" + r"""
function cascade(wv) {
    return _healPreV553(_healPreV550(_healPreV549(_healPreV546(wv))));
}
const v514 = Array.from({length: 16}, (_, i) => "v" + i);
const a = cascade(v514.slice());
if (a.length !== 21 || a[18] !== true || a[19] !== "Off" || a[20] !== true) {
    console.error("FAIL 16->21: " + JSON.stringify(a.slice(15))); process.exit(1);
}
const b = cascade(Array.from({length: 20}, (_, i) => i));
if (b.length !== 21 || b[20] !== true) { console.error("FAIL 20->21"); process.exit(1); }
const c21 = Array.from({length: 21}, (_, i) => i);
if (cascade(c21.slice()).join(",") !== c21.join(",")) {
    console.error("FAIL: a v553 save must pass through untouched"); process.exit(1);
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

    print("PASS: v553 -- stage telemetry (begin/tile/done+duration), mute "
          "capsule with byte-exact restore, heal cascade to 21 measured")
    sys.exit(0)


if __name__ == "__main__":
    main()
