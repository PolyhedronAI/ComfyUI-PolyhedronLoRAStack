#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v1005 -- Power Upscale: the DOM tail and the marker that lied.

THE WOUND (measured 24.09.2026 in the sandbox browser, frontend 1.49.6):
the Vue frontend serialises the node's two DOM widgets (result viewer,
process pane) as '' at the END of widgets_values. _displayToCanon and
_canonToDisplay both check `length === ORDER_CANON.length`; with 27 widgets
+ 2 DOM entries that was never true, so BOTH permutations were silent no-ops
-- the save carried the v588 canon marker AND display-ordered values, and the
two lies cancelled on load. v1004 appended two widgets (canon 29): an old
27+2 save now passed the length check as "canon", was permuted "to display"
and came out scrambled (denoise 0.97 -> 1, steps 8 -> 1.4, seed -> 1, dpmpp_2m
-> euler, 19 "repairs"). Frank's node showed exactly that.

WHAT v1005 PROMISES, each driven in node against the REAL functions:

  T1  _splitDomTail takes only trailing '' / null entries, at most DOM_TAIL
      many, and _joinDomTail puts them back; a dial value never leaves.
  T2  _saveOrderOf reads the TYPES first: a marked save whose values are in
      display order answers "display-current", never "canon"; the marker
      decides only when the types are inconclusive.
  T3  configure(): a 27+2 marked display-ordered save (the v1003 field file),
      a 29+2 marked display-ordered save (v1004), and a 27-long marked
      CANON save from the classic era all land on the SAME live values.
      Driven through the lifted configure/serialize chain with a stub node.
  T4  serialize(): the permutation really fires now -- the written core is in
      CANON order, the marker is set, the tail is back at the end -- and the
      written save reloads to the same values (round trip).
  T5  the marker is written ONLY by the branch that permuted (length asserted).

A browser probe drives the same three cases live: tools/browser_pu_load_probe.py
(block F). This guard is the fast half; the probe is the one that sees the
frontend.
"""

import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JS = os.path.join(ROOT, "web", "js", "ph_power_upscale.js")
NAME = "test_v1005_dom_tail"
_fails = []


def _fail(msg):
    _fails.append(msg)
    print("  - " + msg)


def _need(cond, msg):
    if not cond:
        _fail(msg)


def _node():
    for c in ("node", "nodejs"):
        try:
            subprocess.run([c, "--version"], capture_output=True, check=True)
            return c
        except Exception:
            pass
    return None


def _grab(js, rx):
    m = re.search(rx, js, re.S)
    if not m:
        _fail("%s: cannot lift %r from the JS" % (NAME, rx[:40]))
        return ""
    return m.group(0)


def main():
    print("[%s]" % NAME)
    js = open(JS, encoding="utf-8").read()

    # ---- source pins (T5 + the configure/serialize shape) ------------------
    cfg = js[js.index("nodeType.prototype.configure = function (info)"):]
    cfg = cfg[:cfg.index("nodeType.prototype.onSerialize")]
    _need("_splitDomTail(info.widgets_values)" in cfg and "_joinDomTail(info.widgets_values, _tail)" in cfg,
          "%s: configure must split the DOM tail before the heals and join it after the permutation" % NAME)
    i_split = cfg.index("_splitDomTail(")
    i_heal = cfg.index("_healPreV546(")
    i_join = cfg.index("_joinDomTail(")
    i_c2d = cfg.index("_canonToDisplay(")
    _need(i_split < i_heal < i_c2d < i_join, "%s: split BEFORE the heals, join AFTER canon->display" % NAME)
    ser = js[js.index("nodeType.prototype.onSerialize = function (o)"):]
    ser = ser[:ser.index("nodeType.prototype.onResize")]
    _need("if (s.core.length === ORDER_CANON.length)" in ser
          and ser.index("s.core.length === ORDER_CANON.length") < ser.index("o.properties[CANON_MARKER] = 588"),
          "%s: T5 the marker is written only inside the branch that asserted the core length" % NAME)
    _need("_joinDomTail(_displayToCanon(s.core), s.tail)" in ser,
          "%s: T4 serialize permutes the CORE and re-appends the tail" % NAME)
    so = js[js.index("function _saveOrderOf(vals, marked)"):]
    so = so[:so.index("\n}\n") + 3]
    _need('if (marked) return "canon"' not in so, "%s: T2 the marker must not short-circuit the type reading" % NAME)
    _need("const byTypes = _saveOrderByTypes(vals)" in so and 'return marked ? "canon" : "unknown"' in so,
          "%s: T2 types first, marker as tie-break" % NAME)

    node = _node()
    if node is None:
        _fail("%s: node is not available -- T1-T4 need it" % NAME)
    else:
        parts = [
            _grab(js, r"const ORDER_CANON = \[[\s\S]*?\];"),
            _grab(js, r"const DISPLAY_ORDER = \[[\s\S]*?\];"),
            _grab(js, r"const DISPLAY_LEGACY_V587 = \[[\s\S]*?\];"),
            _grab(js, r"const DISPLAY_LEGACY_V851 = \[[\s\S]*?\];"),
            _grab(js, r"const CANON_IDX_AT_DISPLAY = [^\n]*\n"),
            _grab(js, r"const DISPLAY_POS_OF_CANON = [^\n]*\n"),
            _grab(js, r"const CANON_DEFAULTS = \{[\s\S]*?\n\};"),
            _grab(js, r"const CANON_RANGES = \{[\s\S]*?\n\};"),
            _grab(js, r"const DOM_TAIL = [^\n]*\n"),
            _grab(js, r"function _splitDomTail[\s\S]*?\n\}"),
            _grab(js, r"function _joinDomTail[\s\S]*?\n\}"),
            _grab(js, r"function _padToCanon[\s\S]*?\n\}"),
            _grab(js, r"function _saveOrderOf[\s\S]*?\n\}"),
            _grab(js, r"function _saveOrderByTypes[\s\S]*?\n\}"),
            _grab(js, r"function _displayEra[\s\S]*?\n\}"),
            _grab(js, r"function _canonToDisplay[\s\S]*?\n\}"),
            _grab(js, r"function _displayToCanon[\s\S]*?\n\}"),
            _grab(js, r"function _tableToCanon[\s\S]*?\n\}"),
            _grab(js, r"function _legacyDisplayToCanon[\s\S]*?\n\}"),
        ]
        for fn in ("_healPreV546", "_healPreV549", "_healPreV550", "_healPreV553",
                   "_healPreV555", "_healPreV560", "_healPreV562", "_healPreV564"):
            parts.append(_grab(js, r"const LEN_PRE_%s = [^\n]*\n" % fn[8:]) if ("const LEN_PRE_%s" % fn[8:]) in js else "")
            parts.append(_grab(js, r"function %s[\s\S]*?\n\}" % fn))
        parts.append(_grab(js, r"const SAME_AS_HIGH = [^\n]*\n") if "const SAME_AS_HIGH" in js else 'const SAME_AS_HIGH = "same as high";')
        # the configure / serialize bodies, lifted as functions over a stub `this`
        cfg_body = cfg[cfg.index("try {"):cfg.index("const r = _configure")]
        ser_body = ser[ser.index("try {"):ser.index("return r;")]
        harness = "\n".join(p for p in parts if p) + """
const CANON_MARKER = "pls_widgets_canon";
const console_ = { info: () => {}, warn: () => {}, log: () => {} };
function runConfigure(info) {
    const self = { _plsDisplayReordered: true };
    const console = console_;
    const _configure = null;
    (function () { %s }).call(self);
    return info.widgets_values;
}
function runSerialize(o) {
    const self = { _plsDisplayReordered: true };
    const _onSerialize = null;
    (function () { %s }).call(self);
    return o;
}
// the v1003 field values, by DISPLAY name (27 widgets of that era)
const V = { dual_moe: false, upscale_by: 1, upscale_by_low: 1, final_upscale_by: 1.4,
    denoise: 0.97, denoise_low: 0.25, steps: 8, steps_low: 5, cfg: 1, cfg_low: 1,
    seed: 956528975482887, control_after_generate: "randomize", sampler_name: "dpmpp_2m",
    sampler_low: "same as high", scheduler: "karras", scheduler_low: "beta",
    tile_size: 1104, tile_overlap: 64, sigma_shift: 8, sigma_shift_low: 5,
    result_preview: true, process_preview: "latent2rgb", mute_staging_logs: true,
    resize_method: "lanczos (cpu)", per_batch: 8, vae_tiling: "Off", pixel_stage: "model + fit" };
// v1008 RE-GROUNDING (declared): slots 27/28 renamed h3_refine -> refine_order,
// h3_audio -> audio_stream; a save stores values by POSITION, so the same
// positional values must land on the renamed widgets.
const V29 = Object.assign({}, V, { refine_order: "before pixel", audio_stream: "denoise" });
const D27 = DISPLAY_ORDER.slice(0, 27), C27 = ORDER_CANON.slice(0, 27);
const saves = {
    A_v1003_display_marked: { vals: D27.map(n => V[n]).concat(["", ""]), marked: true, exp: V },
    B_v1004_display_marked: { vals: DISPLAY_ORDER.map(n => V29[n]).concat(["", ""]), marked: true, exp: V29 },
    C_classic_canon_marked: { vals: C27.map(n => V[n]), marked: true, exp: V },
    D_unmarked_display:     { vals: D27.map(n => V[n]).concat(["", ""]), marked: false, exp: V },
};
const out = { t1: {}, t3: {}, t4: {} };
// T1
const s1 = _splitDomTail([1, "", "x", "", null]);
out.t1.a = JSON.stringify(s1);
out.t1.b = JSON.stringify(_splitDomTail(["", "", "", ""]));      // at most DOM_TAIL.length
out.t1.c = JSON.stringify(_joinDomTail(s1.core, s1.tail));
// T2
out.t2 = _saveOrderOf(saves.A_v1003_display_marked.vals.slice(0, 27), true);
out.t2b = _saveOrderOf(saves.C_classic_canon_marked.vals, true);
out.t2c = _saveOrderOf(_padToCanon(saves.A_v1003_display_marked.vals.slice(0, 27)), true);
// T3: every save lands on the same live DISPLAY values
for (const [tag, s] of Object.entries(saves)) {
    const info = { widgets_values: s.vals.slice(), properties: s.marked ? { pls_widgets_canon: 588 } : {} };
    const live = runConfigure(info);
    const got = {}; DISPLAY_ORDER.forEach((n, i) => { got[n] = live[i]; });
    const bad = Object.keys(s.exp).filter(k => got[k] !== s.exp[k]);
    const nTail = s.vals.filter(v => v === "").length;   // what the save carried
    out.t3[tag] = { bad: bad.map(k => [k, s.exp[k], got[k]]), len: live.length,
                    tailOK: live.length === DISPLAY_ORDER.length + nTail && (nTail === 0 || live[live.length - 1] === "") };
    if (tag === "B_v1004_display_marked") {
        // T4: serialise the live (display) values and reload them
        const o = runSerialize({ widgets_values: live.slice(), properties: {} });
        const core = o.widgets_values.slice(0, ORDER_CANON.length);
        const canonOK = ORDER_CANON.every((n, i) => core[i] === V29[n]);
        const back = runConfigure({ widgets_values: o.widgets_values.slice(), properties: o.properties });
        const got2 = {}; DISPLAY_ORDER.forEach((n, i) => { got2[n] = back[i]; });
        out.t4 = { canonOK, marked: o.properties.pls_widgets_canon, len: o.widgets_values.length,
                   tail: o.widgets_values.slice(ORDER_CANON.length),
                   bad: Object.keys(V29).filter(k => got2[k] !== V29[k]) };
        // T5: a core of the WRONG length must not be marked
        const o2 = runSerialize({ widgets_values: live.slice(0, 20), properties: {} });
        out.t5 = !!(o2.properties && o2.properties.pls_widgets_canon);
    }
}
console.log(JSON.stringify(out));
""" % (cfg_body, ser_body)
        with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as fh:
            fh.write(harness)
            path = fh.name
        try:
            r = subprocess.run([node, path], capture_output=True, text=True, timeout=60)
        finally:
            os.unlink(path)
        if r.returncode != 0:
            _fail("%s: harness failed: %s" % (NAME, (r.stderr or r.stdout)[-600:]))
        else:
            o = json.loads(r.stdout.strip().splitlines()[-1])
            _need(o["t1"]["a"] == '{"core":[1,"","x"],"tail":["",null]}', "%s: T1 split takes only the trailing tail: %s" % (NAME, o["t1"]["a"]))
            _need(o["t1"]["b"] == '{"core":["",""],"tail":["",""]}', "%s: T1 at most DOM_TAIL.length entries leave: %s" % (NAME, o["t1"]["b"]))
            _need(o["t1"]["c"] == '[1,"","x","",null]', "%s: T1 join restores the array" % NAME)
            _need(o["t2"] == "display-current", "%s: T2 a marked display-ordered save must read as display-current, got %r" % (NAME, o["t2"]))
            _need(o["t2c"] == "display-current", "%s: T2 ... also after padding to 29, got %r" % (NAME, o["t2c"]))
            _need(o["t2b"] == "canon", "%s: T2 a marked canon-ordered save reads as canon, got %r" % (NAME, o["t2b"]))
            for tag, res in o["t3"].items():
                _need(not res["bad"], "%s: T3 %s lands wrong values: %s" % (NAME, tag, res["bad"][:6]))
                _need(res["tailOK"], "%s: T3 %s the DOM tail must be back at the end (len %s)" % (NAME, tag, res["len"]))
            t4 = o["t4"]
            _need(t4["canonOK"], "%s: T4 the serialised core must be in CANON order" % NAME)
            _need(t4["marked"] == 588, "%s: T4 the permuted save carries the marker" % NAME)
            _need(t4["tail"] == ["", ""], "%s: T4 the tail follows the core, got %s" % (NAME, t4["tail"]))
            _need(not t4["bad"], "%s: T4 round trip lost values: %s" % (NAME, t4["bad"]))
            _need(not o["t5"], "%s: T5 a core of the wrong length must NOT be marked" % NAME)

    if _fails:
        print("%s: %d failure(s)" % (NAME, len(_fails)))
        return 1
    print("%s: PASS -- T1 tail, T2 types before marker, T3 four saves land alike, T4 round trip, T5 marker only on proof" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
