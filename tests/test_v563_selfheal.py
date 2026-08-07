"""Guard v563 -- the node must repair ITSELF.

The v562 field failure: a stale browser cache left a 23-value save in a
24-widget node. `_canonToDisplay` bails out on a length mismatch, so nothing
mapped and nothing filled - `vae_tiling` never got a value, ComfyUI rejected
the prompt ("Value not in list: ''"), and the node could not recover on its
own. The length-exact heal cascade is precise but brittle; it needs nets.

Two nets, both EXECUTED here in node:
  1. _padToCanon - ANY short array is topped up with the canonical defaults.
  2. _sanitize   - after configure, every widget is checked against its OWN
                   options; invalid/empty values fall back to their default,
                   valid ones are never touched. Generic: future widgets are
                   covered without new code.

Plus: per_batch can no longer be 0. It used to mean "whole batch" - precisely
the 14.7 GB path v560 removed - and a failed heal could land there by accident.
min=1 in the widget, and the backend hardens against a stale 0 anyway.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v563_selfheal] FAIL: " + msg)
    sys.exit(1)


def main():
    js = open(os.path.join(ROOT, "web", "js", "ph_power_upscale.js"),
              encoding="utf-8").read()
    pu = open(os.path.join(ROOT, "nodes", "ph_power_upscale.py"),
              encoding="utf-8").read()

    for marker in ("const CANON_DEFAULTS", "function _padToCanon",
                   "function _sanitize"):
        if marker not in js:
            _fail(f"the safety net is gone: {marker}")
    if "_padToCanon(info.widgets_values)" not in js:
        _fail("configure must top up a short save BEFORE the display mapping")
    if "_sanitize(this)" not in js:
        _fail("configure must sanitize the widget values afterwards")

    # ---- run both nets ---------------------------------------------------------
    parts = []
    for rx in (r"const ORDER_CANON = \[[\s\S]*?\];",
               r"const CANON_DEFAULTS = \{[\s\S]*?\};",
               r"const CANON_RANGES = \{[\s\S]*?\};",
               r"function _padToCanon\(arr\) \{[\s\S]*?\n\}",
               r"function _sanitize\(node\) \{[\s\S]*?\n\}"):
        m = re.search(rx, js)
        if not m:
            _fail(f"not extractable: {rx}")
        parts.append(m.group(0))
    harness = "\n".join(parts) + r"""
// the exact field failure: a short save in a wider node. Measured against
// ORDER_CANON.length, so this test survives every future widget.
const N = ORDER_CANON.length;
const p = _padToCanon(Array.from({length: N - 1}, (_, i) => i));
if (p.length !== N || p[N - 1] === undefined || p[N - 1] === null) {
    console.error("FAIL pad: " + JSON.stringify(p.slice(-2))); process.exit(1);
}
// even a very short save is topped up completely
const tiny = _padToCanon([1, 2, 3]);
if (tiny.length !== N) { console.error("FAIL pad tiny"); process.exit(1); }
// a full save must pass through untouched
const full = Array.from({length: N}, (_, i) => i);
if (_padToCanon(full.slice()).join(",") !== full.join(",")) {
    console.error("FAIL: a complete save must not be touched"); process.exit(1);
}
// the widget repair
const node = { widgets: [
    { name: "vae_tiling", value: "", options: { values: ["Off","512","640","768"], default: "Off" } },
    { name: "per_batch", value: undefined, options: { default: 8 } },
    { name: "resize_method", value: "bicubic", options: { values: ["bicubic","area"], default: "bicubic" } },
    { name: "seed", value: 12345, options: {} },
    // AMENDED IN v583 (1st amendment): the net had a TYPE hole and a RANGE
    // hole, both measured in the field on 2026-07-13. A stale-cache first run
    // conserved '' into the autosave's slot 25 (final_upscale_by, the first
    // NUMBER widget appended since the net was built); '' is not
    // undefined/null/NaN, so _sanitize waved it through and ComfyUI's
    // validator rejected the prompt before any backend code could run. The
    // widget meanwhile RENDERED the '' as 0.00 (Number('') is 0) - and a save
    // that stores that cast artefact as a NUMBER passes every type check yet
    // sits below the widget's own min: the error merely changes shape.
    // Pinned: '' -> default; a parseable string is rescued as its number;
    // a number outside [min, max] -> default (consistent with the combo rule:
    // out-of-list -> default, never a silent clamp); valid values untouched.
    { name: "final_upscale_by", value: "", options: { default: 1.0, min: 0.25, max: 8.0 } },
    { name: "final_rescue", value: "1.4", options: { default: 1.0, min: 0.25, max: 8.0 } },
    { name: "final_zero", value: 0, options: { default: 1.0, min: 0.25, max: 8.0 } },
    { name: "final_toolarge", value: 99, options: { default: 1.0, min: 0.25, max: 8.0 } },
    { name: "final_valid", value: 2.0, options: { default: 1.0, min: 0.25, max: 8.0 } },
    // AMENDED IN v584 (2nd amendment): the v583 heal read default/min/max from
    // widget.options - fields these mocks carried but a LIVE ComfyUI widget
    // does not guarantee (measured in the field: the healed slot stayed 0.00).
    // The heal now sources from our own tables (CANON_DEFAULTS/CANON_RANGES)
    // when options are bare. Pinned with widgets carrying NO options at all -
    // the field reality:
    { name: "final_upscale_by", value: "", options: {} },
    { name: "final_upscale_by", value: 0, options: {} },
    // AMENDED IN v586 (3rd amendment): the tables covered only slots 18-25 -
    // enough as the v563 PAD table, a measured gap as the v584 heal SOURCE.
    // The field run: a conserved string landed on tile_size (slot 13), found
    // no default source, and rode as NaN all the way to the validator
    // ("invalid literal for int(): 'same as high'"). Pinned: the bare
    // tile_size field case, and completeness below.
    { name: "tile_size", value: "same as high", options: {} },
], setDirtyCanvas: () => {} };
const fixed = _sanitize(node);
if (node.widgets[0].value !== "Off") { console.error("FAIL: empty combo"); process.exit(1); }
if (node.widgets[1].value !== 8) { console.error("FAIL: undefined number"); process.exit(1); }
if (node.widgets[2].value !== "bicubic" || node.widgets[3].value !== 12345) {
    console.error("FAIL: a VALID value was overwritten"); process.exit(1);
}
if (node.widgets[4].value !== 1.0) {
    console.error("FAIL: '' on a number widget must fall to the default (the "
        + "conserved-autosave case)"); process.exit(1);
}
if (node.widgets[5].value !== 1.4) {
    console.error("FAIL: a parseable string must be RESCUED as its number, "
        + "not flattened"); process.exit(1);
}
if (node.widgets[6].value !== 1.0) {
    console.error("FAIL: 0 below min must fall to the DEFAULT, not clamp - "
        + "the rendered-0.00 field case"); process.exit(1);
}
if (node.widgets[7].value !== 1.0) {
    console.error("FAIL: a number above max must fall to the default"); process.exit(1);
}
if (node.widgets[8].value !== 2.0) {
    console.error("FAIL: a valid number was touched"); process.exit(1);
}
if (node.widgets[9].value !== 1.0) {
    console.error("FAIL: '' on a BARE canon widget (no options) must heal "
        + "from CANON_DEFAULTS - the live-widget case"); process.exit(1);
}
if (node.widgets[10].value !== 1.0) {
    console.error("FAIL: 0 on a BARE canon widget must heal via CANON_RANGES "
        + "- the rendered-0.00 live case"); process.exit(1);
}
if (node.widgets[11].value !== 1024) {
    console.error("FAIL: a conserved string on a BARE tile_size must heal to "
        + "the mirrored default 1024 - the NaN-to-validator field case");
    process.exit(1);
}
if (fixed !== 9) { console.error("FAIL: repair count " + fixed); process.exit(1); }
// v586 completeness: every numeric canon slot has a default source, every
// INT/FLOAT slot has a range - the heal must never again sleep for lack of
// a table entry. (Combo slots self-heal through their own value list.)
const NUMERIC = [0,1,2,3,4,5,6,7,8,9,13,14,15,18,20,22,25];
const RANGED  = [1,2,3,4,5,6,7,8,9,13,14,15,22,25];
for (const i of NUMERIC) if (CANON_DEFAULTS[i] === undefined) {
    console.error("FAIL: CANON_DEFAULTS missing numeric slot " + i); process.exit(1);
}
for (const i of RANGED) if (!Array.isArray(CANON_RANGES[i])) {
    console.error("FAIL: CANON_RANGES missing slot " + i); process.exit(1);
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
        _fail("safety-net harness failed:\n" + proc.stdout + proc.stderr)

    # ---- the 0 trap is gone -----------------------------------------------------
    if '"per_batch": ("INT", {"default": 8, "min": 1' not in pu:
        _fail("per_batch must not accept 0 any more - 0 meant 'whole batch', "
              "the 14.7 GB path v560 removed, and a failed heal could land there")
    if "per_batch = 8 if not per_batch or int(per_batch) < 1 else int(per_batch)" not in pu:
        _fail("the backend must harden against a stale per_batch=0 from an old save")
    # v583: final_upscale_by wears the same belt at its consumption site - a
    # conserved ''/0/junk from an old save or a raw API prompt must never
    # reach the size law. The frontend heal above covers loaded graphs; this
    # covers everything that never passes through the frontend.
    if "if not (0.25 <= fby <= 8.0):" not in pu:
        _fail("the backend must harden final_upscale_by against a conserved "
              "out-of-range artefact (the per_batch belt, second wearer)")
    # v584: the JS mirror tables must stay in step with the python widget -
    # they are the heal's source of truth when a live widget carries no
    # options. Two contains-pins, one per side of the mirror.
    if "25: [0.25, 8.0]," not in js:
        _fail("CANON_RANGES must mirror the final_upscale_by range from "
              "INPUT_TYPES - the heal reads it when options are bare")
    if '"min": 0.25, "max": 8.0' not in pu:
        _fail("the python widget range moved - update CANON_RANGES in the JS "
              "mirror in the same cut")
    # v586 mirror spot-checks: tile_size, the measured field victim.
    if "13: 1024," not in js or "13: [64, 4096]," not in js:
        _fail("the tile_size mirror entries are gone - the heal sleeps again")
    if '"tile_size": ("INT", {"default": 1024, "min": 64' not in pu:
        _fail("the python tile_size widget moved - update the JS mirror in "
              "the same cut")

    # The exact banner version is pinned by the file's NEWEST guard (v564).

    print("PASS: v563 -- self-heal executed (23->24 padded, '' -> Off, "
          "undefined -> default, valid values untouched), per_batch 0-trap "
          "removed at both ends")
    sys.exit(0)


if __name__ == "__main__":
    main()
