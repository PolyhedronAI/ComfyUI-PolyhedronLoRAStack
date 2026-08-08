"""Guard v852 -- Power Upscale, sigma_shift_low moves under sigma_shift.

A display re-sort is the single most expensive mistake this node knows. v584
moved one widget into the display middle; every pre-v584 save then loaded
shifted by one slot from position 4 on, the seed landed in cfg_low, and the
self-heal net MASKED half the damage by "repairing" the shifted values into
plausible defaults. Nobody sees that in a screenshot -- the node just runs
differently.

So the re-sort is legal only with the whole ceremony, and this guard is one of
its three parts:

  R1  THE CANON DID NOT MOVE. Whatever the display does, the serialised order
      is untouched: sigma_shift_low stays LAST in INPUT_TYPES and in the
      baseline. If this ever fails, the display re-sort became a #577 breach.

  R2  THE DISPLAY IS A PERMUTATION, AND THE TWINS ARE TOGETHER. Checked as a
      RULE over every HIGH/LOW pair, not as a hand-written list, so the next
      twin cannot be forgotten.

  R3  EVERY ORDER WE EVER DISPLAYED SURVIVES AS A FROZEN TABLE. Two now
      (v587, v851). The tables are compared against orders reconstructed HERE
      from the history, not read out of the file and handed back to it.

  R4  THE FINGERPRINT NAMES ALL FOUR LAYOUTS. Built from the type signature of
      each order, with the witness pairs the code claims to use -- and each
      verdict must survive ONE corrupted witness slot, because the '' saga is
      what taught this node that a single junk value must not flip a verdict.

  R5  A SAVE FROM EVERY ERA COMES HOME. The real chain (pad -> fingerprint ->
      map) is run in node for a canon save, a v587-display save, a v851-display
      save and a save in TODAY's display order -- each built by NAME, then read
      back by NAME. Nothing may land in a neighbour's slot.

  R6  A MARKED SAVE IS IMMUNE. The marker is the whole reason a re-sort is
      survivable at all: a marked save is canon by construction, so the
      fingerprint must not even be consulted.
"""
import ast
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JS_PATH = os.path.join(ROOT, "web", "js", "ph_power_upscale.js")
PY_PATH = os.path.join(ROOT, "nodes", "ph_power_upscale.py")

JS = open(JS_PATH, encoding="utf-8").read()
_fails = []


def _fail(msg):
    print("FAIL: %s" % msg)
    _fails.append(msg)


def _ok(msg):
    print("  ok  %s" % msg)


def _names(const):
    body = re.search(r"const %s = \[(.*?)\];" % const, JS, re.S)
    if not body:
        _fail("%s is gone" % const)
        return []
    return re.findall(r'"([a-z_0-9]+)"', body.group(1))


CANON = _names("ORDER_CANON")
DISPLAY = _names("DISPLAY_ORDER")
LEG587 = _names("DISPLAY_LEGACY_V587")
LEG851 = _names("DISPLAY_LEGACY_V851")

# Widget types, written down HERE from what the node declares. The fingerprint
# stands or falls on these, so they are not read back out of the frontend.
BOOLS = {"dual_moe", "result_preview", "mute_staging_logs"}
STRS = {"control_after_generate", "sampler_name", "scheduler", "sampler_low",
        "scheduler_low", "process_preview", "resize_method", "vae_tiling",
        "pixel_stage"}


def _t(name):
    return "B" if name in BOOLS else ("S" if name in STRS else "N")


def _sig(order):
    return "".join(_t(n) for n in order)


# ---------------------------------------------------------------------------
def r1_canon_untouched():
    if not CANON or CANON[-1] != "sigma_shift_low":
        return _fail("ORDER_CANON must still end with sigma_shift_low -- a "
                     "display re-sort may NEVER move the serialised order (#577)")
    tree = ast.parse(open(PY_PATH, encoding="utf-8").read())
    sys.path.insert(0, HERE)
    import test_v577_widget_order as scan
    live = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ULSPowerUpscale":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "INPUT_TYPES":
                    live = scan._order_of(sub)[0]
    if live is None:
        return _fail("could not read INPUT_TYPES")
    if live[-1] != "sigma_shift_low":
        return _fail("INPUT_TYPES no longer ends with sigma_shift_low")
    # the canon list carries control_after_generate, which INPUT_TYPES does not
    if [n for n in CANON if n != "control_after_generate"] != live:
        return _fail("ORDER_CANON drifted from INPUT_TYPES:\n  js %r\n  py %r"
                     % (CANON, live))
    _ok("canon untouched: sigma_shift_low still last in INPUT_TYPES and canon")


def r2_permutation_and_twins():
    if sorted(DISPLAY) != sorted(CANON):
        return _fail("DISPLAY_ORDER is not a permutation of ORDER_CANON")
    # the HIGH partner is not always the name minus "_low" (sampler_low belongs
    # to sampler_name); the exceptions are named, the rest is derived.
    HIGH_OF = {"sampler_low": "sampler_name"}
    pairs = [(HIGH_OF.get(n, n[:-4]), n) for n in CANON if n.endswith("_low")]
    if len(pairs) < 7:
        return _fail("expected at least seven HIGH/LOW twins, found %r" % pairs)
    for hi, lo in pairs:
        if hi not in DISPLAY:
            return _fail("twin %r has no HIGH partner %r" % (lo, hi))
        if DISPLAY.index(lo) != DISPLAY.index(hi) + 1:
            return _fail("'%s' must sit directly under '%s' (the law of "
                         "proximity); it sits %d slots away"
                         % (lo, hi, DISPLAY.index(lo) - DISPLAY.index(hi)))
    _ok("display is a permutation; all %d twins adjacent" % len(pairs))


def r3_frozen_tables():
    # reconstructed from the history, not copied out of the file
    v587 = ["dual_moe", "upscale_by", "upscale_by_low",
            "denoise", "denoise_low", "steps", "steps_low", "cfg", "cfg_low",
            "seed", "control_after_generate", "sampler_name", "sampler_low",
            "scheduler", "scheduler_low", "tile_size", "tile_overlap",
            "sigma_shift", "result_preview", "process_preview",
            "mute_staging_logs", "resize_method", "per_batch", "vae_tiling",
            "pixel_stage", "final_upscale_by"]
    # v589 pulled final_upscale_by up to slot 3; v851 appended sigma_shift_low
    v851 = v587[:3] + ["final_upscale_by"] + v587[3:-1] + ["sigma_shift_low"]
    if LEG587 != v587:
        return _fail("DISPLAY_LEGACY_V587 is not the pre-v589 order VERBATIM")
    if LEG851 != v851:
        return _fail("DISPLAY_LEGACY_V851 is not the v589..v851 order VERBATIM:\n"
                     "  is   %r\n  want %r" % (LEG851, v851))
    if DISPLAY == LEG851:
        return _fail("the current display equals the v851 table -- then this cut "
                     "did not happen, or the table was written from the new order")
    _ok("both historic tables verbatim; the current order differs from v851")


def _orders():
    """The four layouts a save can be in, padded to canon length."""
    pad = ["sigma_shift_low"]
    return {
        "canon": CANON,
        "legacy-display": LEG587 + pad,   # padding appends the canon tail name
        "legacy-display-851": LEG851,
        "display-current": DISPLAY,
    }


def r4_fingerprint_separates():
    sigs = {k: _sig(v) for k, v in _orders().items()}
    for a in sigs:
        for b in sigs:
            if a < b and sigs[a] == sigs[b]:
                return _fail("layouts %r and %r have the SAME type signature %s "
                             "-- the fingerprint cannot tell them apart" % (a, b, sigs[a]))
    _ok("all four layouts have distinct type signatures")

    # Each witness set answers ONE question, so what it must do is SPLIT the
    # four layouts into the two groups that question names -- not single one
    # out. Checked as the partition it claims, or the fingerprint's reasoning
    # is folklore.
    orders = _orders()
    for slots, left, right, what in (
            ((13, 14, 16, 17), {"canon"},
             {"legacy-display", "legacy-display-851", "display-current"},
             "canon vs some display order"),
            ((10, 18, 19, 25), {"canon", "legacy-display"},
             {"legacy-display-851", "display-current"}, "which display era"),
            ((19, 21, 24, 26), {"legacy-display-851"}, {"display-current"},
             "v851 vs today")):
        groups = {}
        for k in left | right:
            groups.setdefault(tuple(_t(orders[k][i]) for i in slots), set()).add(k)
        for sig, members in groups.items():
            if not (members <= left or members <= right):
                _fail("witness slots %r cannot answer '%s': %r share the "
                      "signature %s" % (slots, what, sorted(members), "".join(sig)))
                return
    _ok("each witness set really draws the line it claims to draw")


def _node_bin():
    for cand in ("node", "nodejs"):
        try:
            subprocess.run([cand, "--version"], capture_output=True)
            return cand
        except OSError:
            continue
    return "node"


def _run_js(lines):
    harness = "\n".join([
        re.search(r"const ORDER_CANON = \[[\s\S]*?\];", JS).group(0),
        re.search(r"const DISPLAY_ORDER = \[[\s\S]*?\];", JS).group(0),
        re.search(r"const DISPLAY_LEGACY_V587 = \[[\s\S]*?\];", JS).group(0),
        re.search(r"const DISPLAY_LEGACY_V851 = \[[\s\S]*?\];", JS).group(0),
        re.search(r"const CANON_DEFAULTS = \{[\s\S]*?\n\};", JS).group(0),
        re.search(r"const CANON_IDX_AT_DISPLAY[\s\S]*?;", JS).group(0),
        re.search(r"const DISPLAY_POS_OF_CANON[\s\S]*?;", JS).group(0),
        re.search(r"function _padToCanon[\s\S]*?\n\}", JS).group(0),
        re.search(r"function _tableToCanon[\s\S]*?\n\}", JS).group(0),
        re.search(r"function _legacyDisplayToCanon[\s\S]*?\n\}", JS).group(0),
        re.search(r"function _saveOrderOf[\s\S]*?\n\}", JS).group(0),
        re.search(r"function _displayEra[\s\S]*?\n\}", JS).group(0),
        re.search(r"function _canonToDisplay[\s\S]*?\n\}", JS).group(0),
        re.search(r"function _displayToCanon[\s\S]*?\n\}", JS).group(0),
        "console.info = () => {};",
    ] + lines)
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
        fh.write(harness)
        path = fh.name
    try:
        return subprocess.run([_node_bin(), path], capture_output=True, text=True)
    finally:
        os.unlink(path)


# a value per widget whose TYPE is right and whose identity is visible
def _configure_branch():
    """The if/else-if chain configure() uses to map a save home."""
    body = re.search(
        r"const _ord = _saveOrderOf\(info\.widgets_values, marked\);\n"
        r"([\s\S]*?)\n\s*console\.info\(", JS)
    if not body:
        _fail("could not lift the mapping branch out of configure() -- if it "
              "was restructured, this guard must be re-grounded, not skipped")
        return "/* missing */"
    return body.group(1)


_CONFIGURE_BRANCH = None


def _vector(order):
    out = []
    for n in order:
        t = _t(n)
        out.append("true" if t == "B" else
                   ("'%s'" % n if t == "S" else "%d" % (hash(n) % 9000 + 100)))
    return "[" + ",".join(out) + "]"


def r5_every_era_comes_home():
    global _CONFIGURE_BRANCH
    _CONFIGURE_BRANCH = _configure_branch()
    lines = []
    for verdict, order in _orders().items():
        # a v587 save is 26 long on disk -- the pad adds the tail
        raw = order[:-1] if verdict == "legacy-display" else order
        lines += [
            "{",
            "  const order = %r;" % verdict,
            "  const raw = %s;" % _vector(raw),
            "  const padded = _padToCanon(raw);",
            "  const said = _saveOrderOf(padded, false);",
            "  if (said !== order) {",
            "    console.error('FAIL: a ' + order + ' save was named ' + said);",
            "    process.exit(1); }",
            # the mapping branch is LIFTED from configure(), never retyped -- a
            # second copy would prove itself instead of the shipped path
            "  const info = { widgets_values: padded };",
            "  const _ord = said;",
            _CONFIGURE_BRANCH,
            "  const v = info.widgets_values;",
            "  const names = %s;" % repr(list(order)).replace("'", '"'),
            "  for (let i = 0; i < ORDER_CANON.length; i++) {",
            "    const n = ORDER_CANON[i];",
            "    const from = names.indexOf(n);",
            "    if (from < 0) continue;",
            "    if (v[i] !== raw[from] && !(from >= raw.length)) {",
            "      console.error('FAIL: ' + order + ': ' + n + ' arrived as ' + v[i]",
            "                    + ' instead of ' + raw[from]);",
            "      process.exit(1); }",
            "  }",
            "}",
        ]
    # ONE corrupted witness may not change the verdict AT ALL -- not to a wrong
    # one, and not to "unknown" either. Degrading to unknown means the node
    # loads a display-ordered save as canon, which IS the v584 damage. null is
    # used as the junk value on purpose: the empty string passes every str()
    # test and would have made this check pass without testing anything.
    for verdict, order in _orders().items():
        raw = order[:-1] if verdict == "legacy-display" else order
        lines += [
            "{",
            "  const base = _padToCanon(%s);" % _vector(raw),
            "  for (const slot of [10, 13, 14, 16, 17, 18, 19, 21, 24, 25, 26]) {",
            "    const hurt = base.slice(); hurt[slot] = null;",
            "    const said = _saveOrderOf(hurt, false);",
            "    if (said !== %r) {" % verdict,
            "      console.error('FAIL: junk at slot ' + slot + ' turned a "
            "%s save into ' + said);" % verdict,
            "      process.exit(1); }",
            "  }",
            "}",
        ]
    lines.append("console.log('JS OK');")
    out = _run_js(lines)
    if out.returncode != 0 or "JS OK" not in out.stdout:
        return _fail("era round trip: %s %s" % (out.stdout.strip(), out.stderr.strip()))
    _ok("a save from all four eras comes home; one junk witness cannot flip it")


def r6_marker_is_immune():
    out = _run_js([
        "const junk = new Array(ORDER_CANON.length).fill('x');",
        "if (_saveOrderOf(junk, true) !== 'canon') {",
        "  console.error('FAIL: a MARKED save must be canon, unasked');",
        "  process.exit(1); }",
        "console.log('JS OK');",
    ])
    if out.returncode != 0 or "JS OK" not in out.stdout:
        return _fail("marker: %s %s" % (out.stdout.strip(), out.stderr.strip()))
    if "if (marked) return \"canon\";" not in JS:
        return _fail("the marker short circuit is gone -- it is what makes a "
                     "display re-sort survivable at all")
    _ok("a marked save short-circuits the fingerprint entirely")


def main():
    print("[test_v852_display_resort]")
    r1_canon_untouched()
    r2_permutation_and_twins()
    r3_frozen_tables()
    r4_fingerprint_separates()
    r5_every_era_comes_home()
    r6_marker_is_immune()
    if _fails:
        print("\n%d FAILURE(S)" % len(_fails))
        sys.exit(1)
    print("\nOK -- 6/6")


if __name__ == "__main__":
    main()
