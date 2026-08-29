"""Guard v537 -- Switch pair: lazy dynamics, blocker paths, bypass robustness."""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _fail(msg):
    print("FAIL: " + msg)
    sys.exit(1)


def _read(*parts):
    return open(os.path.join(ROOT, *parts), encoding="utf-8").read()


def main():
    py = _read("nodes", "ph_switch.py")
    js = _read("web", "js", "ph_switch.js")
    init = _read("__init__.py")

    # -- core mechanisms present -------------------------------------------
    if "from comfy_execution.graph import ExecutionBlocker" not in py:
        _fail("ExecutionBlocker import missing")
    if py.count('"lazy": True') < 3:
        _fail("lazy flags missing (declared input_1, AllLazyAny, inverse input)")
    if "get_input_info" not in py or "_AllLazyAny" not in py:
        _fail("dynamic-input AllContainer gate missing")
    if "class _FlexOutTuple(tuple)" not in py or "len(self) - 1" not in py:
        _fail("FlexOutTuple clamp missing (unbounded outputs)")
    if py.count("def check_lazy_status") != 2:
        _fail("both nodes must define check_lazy_status")
    if "cnt = max(cnt, int(value[1]))" not in py:
        _fail("hidden-PROMPT consumer scan missing (inverse output count)")

    # -- bypass robustness --------------------------------------------------
    m = re.search(r"class ULSAnySwitchInv.*?\"required\"\s*:\s*\{(.*?)\}\s*,",
                  py, re.S)
    if not m or '"input"' in m.group(1):
        _fail("inverse 'input' must NOT be required (mute/bypass safety)")
    if '"input": (_any' not in py:
        _fail("inverse optional any 'input' missing")
    if '"on_missing"' not in py or "use next active" not in py:
        _fail("forward on_missing fallback toggle missing")
    # (internal helpers may return None as a sentinel; what must never happen
    #  is a bare-None RESULT -- both missing-paths must emit blockers)
    if "(blk, blk, blk)" not in py:
        _fail("forward missing-path must emit a blocker triple")
    if "result.append(_blocked())" not in py:
        _fail("inverse non-selected outputs must emit blockers")

    # -- JS <-> Py parity ----------------------------------------------------
    if '_IN = "input_"' not in py or '_OUT = "out_"' not in py:
        _fail("python slot prefixes changed")
    if 'const IN = "input_"' not in js or 'const OUT = "out_"' not in js:
        _fail("js slot prefixes changed (parity with ph_switch.py broken)")
    if "options.max" not in js:
        _fail("select auto-max update missing in JS")
    if "requestAnimationFrame" not in js or js.count("requestAnimationFrame") < 2:
        _fail("double-rAF onConfigure normalization missing")
    # v874: the EXACT banner version is pinned by the file's NEWEST guard
    # (test_v874_switch_width), per this tree's convention -- see the note in
    # test_v550_processview. Existence and format stay pinned here.
    if "[PLS] ph_switch.js v" not in js or " loaded" not in js:
        _fail("self-proving banner missing/malformed in ph_switch.js")
    if "pls_switch" not in py or "pls_switch" not in js:
        _fail("ui status channel key mismatch")

    # -- registration --------------------------------------------------------
    for needle in ("from .nodes.ph_switch import ULSAnySwitch, ULSAnySwitchInv",
                   '"ULSAnySwitch"', '"ULSAnySwitchInv"',
                   "⬡ Polyhedron Switch", "⬡ Polyhedron Switch Inverse"):
        if needle not in init:
            _fail(f"registration incomplete: {needle!r} not in __init__.py")

    print("PASS: v537 switch pair -- lazy dynamics, blocker paths, parity, registration")
    sys.exit(0)


if __name__ == "__main__":
    main()
