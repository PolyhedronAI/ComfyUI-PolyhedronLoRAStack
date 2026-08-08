"""v589 guard: the law of proximity, made legal by proof instead of luck.

Three claims, pinned by RETURN VALUES through a node harness:

1. PROXIMITY: final_upscale_by sits directly after upscale_by_low in the
   display - the third size dial with its family (Frank's ask, twice).
2. THE FINGERPRINT: _saveOrderOf reads a save's order from its TYPES at the
   discriminator slots (13/14 vs 16/17, two witnesses per side). Verdicts
   are pinned with real vectors, including the inconclusive case.
3. THE v584 REGRESSION, DEAD: Frank's actual 19:06 dial set, written in the
   v587 display order (the phantom save that shifted his seed into cfg_low),
   runs the full v589 load pipeline - and every value lands on its OWN
   widget. The seed comes home; karras stays a scheduler; 1104 stays a tile.
"""
import pathlib, re, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
JS = ROOT / "web" / "js" / "ph_power_upscale.js"


def _fail(msg):
    print(f"[test_v589_proximity] FAIL: {msg}")
    sys.exit(1)


def main():
    js = JS.read_text(encoding="utf-8")

    disp = re.findall(r'"([a-z_]+)"',
                      re.search(r"const DISPLAY_ORDER = \[(.*?)\];", js,
                                re.S).group(1))
    if disp.index("final_upscale_by") != disp.index("upscale_by_low") + 1:
        _fail("final_upscale_by must sit DIRECTLY after upscale_by_low - the "
              "law of proximity, the whole point of the cut")

    harness = "\n".join([
        re.search(r"const ORDER_CANON = \[[\s\S]*?\];", js).group(0),
        re.search(r"const DISPLAY_ORDER = \[[\s\S]*?\];", js).group(0),
        re.search(r"const DISPLAY_LEGACY_V587 = \[[\s\S]*?\];", js).group(0),
        # v851: the real load chain pads before it maps, so the harness must
        # carry the pad and its default table too (lifted from the SOURCE, never
        # rebuilt - a second implementation would be a second truth).
        re.search(r"const CANON_DEFAULTS = \{[\s\S]*?\n\};", js).group(0),
        re.search(r"function _padToCanon[\s\S]*?\n\}", js).group(0),
        re.search(r"const CANON_IDX_AT_DISPLAY[\s\S]*?;", js).group(0),
        re.search(r"const DISPLAY_POS_OF_CANON[\s\S]*?;", js).group(0),
        re.search(r"function _canonToDisplay[\s\S]*?\n\}", js).group(0),
        # v852: _legacyDisplayToCanon delegates to the shared table mapper now
        re.search(r"function _tableToCanon[\s\S]*?\n\}", js).group(0),
        re.search(r"function _legacyDisplayToCanon[\s\S]*?\n\}", js).group(0),
        re.search(r"function _saveOrderOf[\s\S]*?\n\}", js).group(0),
        # ---- verdicts, pinned -------------------------------------------------
        # marked short-circuits everything:
        'if (_saveOrderOf(["x"], true) !== "canon") {',
        '    console.error("FAIL marked"); process.exit(1); }',
        # pre-Vue short saves are canon by construction:
        'if (_saveOrderOf(Array(18).fill("s"), false) !== "canon") {',
        '    console.error("FAIL short"); process.exit(1); }',
        # a canon-ordered vector: numbers at 13/14, strings at 16/17
        "const canonVec = [true,1.1,0.19,3,1.6,1.3,0.25,5,1.9,42,'randomize',"
        "'dpmpp_2m','karras',1104,64,8.0,'same as high','bong_tangent',true,"
        "'Off',true,'lanczos (cpu)',8,'Off','model final',1.4];",
        'if (_saveOrderOf(canonVec, false) !== "canon") {',
        '    console.error("FAIL canon fingerprint"); process.exit(1); }',
        # Frank's 19:06 dials in the v587 DISPLAY order - the phantom save:
        "const frank = [true,1.10,1.30,0.19,0.25,3,5,1.0,1.0,933250499243096,"
        "'randomize','dpmpp_2m','same as high','karras','bong_tangent',1104,64,"
        "8.0,true,'latent2rgb',true,'none',8,'Off','model final',1.0];",
        'if (_saveOrderOf(frank, false) !== "legacy-display") {',
        '    console.error("FAIL legacy fingerprint"); process.exit(1); }',
        # inconclusive: strings on BOTH witness pairs -> unknown (status quo)
        "const fog = canonVec.slice(); fog[13]='x'; fog[14]='y';",
        'if (_saveOrderOf(fog, false) !== "unknown") {',
        '    console.error("FAIL unknown"); process.exit(1); }',
        # one corrupt slot must NOT flip the verdict (the empty-string saga):
        "const hurt = canonVec.slice(); hurt[13]='';",
        'if (_saveOrderOf(hurt, false) !== "canon") {',
        '    console.error("FAIL witness majority"); process.exit(1); }',
        # ---- the v584 regression, end to end ---------------------------------
        # v851 RE-GROUNDING (declared): this fed 'frank' STRAIGHT into
        # _legacyDisplayToCanon, but the real configure() pads FIRST and only
        # then maps - and the map gates on arr.length === ORDER_CANON.length.
        # With a hand-built vector that gate held by accident as long as the
        # canon never grew; v851 grew it and the shortcut showed. Run the REAL
        # chain now, so this guard also proves that a canon APPEND leaves the
        # v584 recovery of an old save intact.
        "const frankPadded = _padToCanon(frank);",
        "if (frankPadded.length !== ORDER_CANON.length) {",
        '    console.error("FAIL: the pad did not reach canon length"); process.exit(1); }',
        "if (frankPadded[ORDER_CANON.length - 1] !== CANON_DEFAULTS[ORDER_CANON.length - 1]) {",
        '    console.error("FAIL: the padded tail is not the canon default"); process.exit(1); }',
        'if (_saveOrderOf(frankPadded, false) !== "legacy-display") {',
        '    console.error("FAIL: padding must not flip the fingerprint"); process.exit(1); }',
        "let v = _legacyDisplayToCanon(frankPadded);",
        "v = _canonToDisplay(v);   // the v589 display",
        "const at = (n) => v[DISPLAY_ORDER.indexOf(n)];",
        'if (at("seed") !== 933250499243096) {',
        '    console.error("FAIL: the seed did not come home"); process.exit(1); }',
        'if (at("scheduler") !== "karras" || at("scheduler_low") !== "bong_tangent") {',
        '    console.error("FAIL: schedulers scrambled"); process.exit(1); }',
        'if (at("tile_size") !== 1104 || at("tile_overlap") !== 64) {',
        '    console.error("FAIL: tiling scrambled"); process.exit(1); }',
        'if (at("final_upscale_by") !== 1.0 || at("pixel_stage") !== "model final") {',
        '    console.error("FAIL: the moved dial lost its value"); process.exit(1); }',
        'if (at("sampler_low") !== "same as high" || at("denoise") !== 0.19) {',
        '    console.error("FAIL: neighbours scrambled"); process.exit(1); }',
        'console.log("OK");',
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(harness)
        tmp = f.name
    r = subprocess.run(["node", tmp], capture_output=True, text=True, timeout=30)
    if r.returncode != 0 or "OK" not in r.stdout:
        _fail(f"harness: {r.stdout} {r.stderr}")

    # configure must actually ROUTE by the verdict and SAY it.
    if '_saveOrderOf(info.widgets_values, marked)' not in js:
        _fail("configure must read the verdict")
    if 'if (_ord === "legacy-display")' not in js or "_legacyDisplayToCanon(info.widgets_values)" not in js:
        _fail("the legacy-display verdict must route through the legacy map")
    if "fingerprint INCONCLUSIVE" not in js:
        _fail("the unknown verdict must be SPOKEN - a silent guess is v584")

    print("PASS: v589 -- proximity holds; fingerprint verdicts pinned "
          "(marked, short, canon, legacy, unknown, witness-majority); "
          "Frank's phantom save loads with every value on its own widget - "
          "the seed is home")


if __name__ == "__main__":
    main()
