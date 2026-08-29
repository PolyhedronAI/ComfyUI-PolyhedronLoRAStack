"""Guard v851 -- Power Upscale, the LOW expert's own sigma shift.

The Sampler has carried a per-expert shift since v839; the Power Upscale
applied ONE value to both stages. This guard pins the nachgeruestet widget and,
more importantly, the two things that make it different here:

  S1  THE LAW OF SERIALISATION (#577). The new widget is the LAST entry of
      required, and the baseline says so. Checked against INPUT_TYPES itself,
      not against a remembered list.

  S2  ONE RESOLVER, NOT TWO. The -1 sentinel semantics live in uls_sampler's
      _resolve_low_shift. This node must IMPORT it. A local copy is the
      classic drift and is rejected structurally (AST), not by substring --
      the v546 re-grounding of this same promise is what taught us that.

  S3  THE SENTINEL IS THE OLD BEHAVIOUR. Resolution is compared against a
      table written from the SPEC ("<0 = follow high, 0 = off for low,
      positive = its own"), never derived from the implementation.

  S4  ORDER OF APPLICATION -- the trap of this cut. Applying the HIGH shift
      REBINDS 'model'. If stage L's source is read after that rebind, a
      "low" shift lands on an ALREADY shifted model, i.e. twice. The source
      code is checked to capture the low source BEFORE the high patch, and
      the runtime behaviour is checked with a recording stand-in for
      _apply_sigma_shift: every patch must start from a RAW model.

  S5  THE FALLBACK MUST NOT INHERIT SILENTLY. Without a wired model_low the
      L stage runs on the 'model' input. Four cases are run end to end:
        sentinel + no wire  -> nothing extra is built (bit-identical to v850)
        own value + no wire -> a SECOND clone off the RAW model
        OFF + no wire       -> the RAW model, never the shifted one
        any value + wire    -> the WIRED expert, patched from itself
      This is the case where a wrong answer is invisible: the run would
      simply use a differently shaped schedule and nobody would see it.

  S6  NO INERT MODE. Unlike the Sampler there is no 'Continuous' here, so the
      widget may never be silently ignored -- and the node must not grow an
      honesty line pretending otherwise.

  S7  THE FRONTEND SURVIVES THE GROWTH. ORDER_CANON and DISPLAY_ORDER both
      grew by exactly one, at the END of both. The heal tables carry the new
      index. DISPLAY_LEGACY_V587 stays VERBATIM history, and an old save in
      that order still lands correctly after padding.
"""
import ast
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lift as _LIFT          # noqa: E402  -- the shared closed lift (v896)
PY = os.path.join(ROOT, "nodes", "ph_power_upscale.py")
JS = os.path.join(ROOT, "web", "js", "ph_power_upscale.js")
WIDGET = "sigma_shift_low"

# ---- stubs, so uls_sampler imports without ComfyUI (the v830/v839 pattern) --
import types
_tmp = tempfile.mkdtemp()
_fp = types.ModuleType("folder_paths")
_fp.models_dir = _tmp
_fp.get_filename_list = lambda f: []
_fp.get_full_path = lambda f, n: None
_fp.get_folder_paths = lambda f: [_tmp]
sys.modules.setdefault("folder_paths", _fp)
for _name in ("comfy", "comfy.samplers", "comfy.sample", "comfy.utils",
              "comfy.model_management", "comfy.cli_args"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["comfy.cli_args"].args = types.SimpleNamespace(preview_method=None)
sys.modules["comfy.cli_args"].LatentPreviewMethod = object
sys.modules["comfy"].samplers = sys.modules["comfy.samplers"]
sys.modules["comfy"].sample = sys.modules["comfy.sample"]
sys.modules["comfy"].utils = sys.modules["comfy.utils"]
sys.modules["comfy.samplers"].KSampler = types.SimpleNamespace(
    SAMPLERS=["euler"], SCHEDULERS=["simple"])


def _resolver():
    sys.path.insert(0, os.path.join(ROOT, "nodes"))
    from uls_sampler import _resolve_low_shift
    return _resolve_low_shift


_fails = []


def _fail(msg):
    print("FAIL: %s" % msg)
    _fails.append(msg)


def _ok(msg):
    print("  ok  %s" % msg)


# ---------------------------------------------------------------------------
# S1  the widget exists and is LAST in required
# ---------------------------------------------------------------------------
def _widget_order():
    """The node's widget order, read with the SCANNER THE LAW USES.

    Imported from test_v577 rather than rebuilt: a second implementation would
    be a second truth, and it is #577's scanner that decides what a slot IS
    (sockets and forceInput entries occupy none). The v850 cut is the reason
    this matters -- a scanner that misreads one entry hides a whole widget.
    """
    sys.path.insert(0, HERE)
    import test_v577_widget_order as scan
    tree = ast.parse(open(PY, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ULSPowerUpscale":
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == "INPUT_TYPES":
                    # v901 widened this to (names, dynamic, stray sections);
                    # this test only needs the first two.
                    names, dynamic = scan._order_of(sub)[:2]
                    return names, dynamic
    return None, False


def s1_serialisation_law():
    req, dynamic = _widget_order()
    if req is None:
        return _fail("could not read ULSPowerUpscale.INPUT_TYPES")
    if dynamic:
        return _fail("this node became dynamic; the baseline can no longer pin it")
    if WIDGET not in req:
        return _fail("%s is not declared at all" % WIDGET)
    if req[-1] != WIDGET:
        return _fail("%s must be the LAST widget (#577: widget values are "
                     "serialised BY INDEX; an insert renumbers every saved "
                     "workflow). It sits at %d of %d."
                     % (WIDGET, req.index(WIDGET), len(req) - 1))
    _ok("declared LAST of %d widgets" % len(req))

    base = [p for p in os.listdir(ROOT)
            if p.startswith("WIDGET_ORDER_baseline_v") and p.endswith(".txt")]
    if len(base) != 1:
        return _fail("expected exactly ONE widget baseline, found %r" % base)
    line = None
    for ln in open(os.path.join(ROOT, base[0]), encoding="utf-8"):
        if ln.startswith("ULSPowerUpscale\t"):
            line = ln.rstrip("\n").split("\t")[1].split(",")
    if line is None:
        return _fail("the baseline has no ULSPowerUpscale row")
    if line != req:
        return _fail("baseline row and INPUT_TYPES disagree:\n  base %r\n  live %r"
                     % (line, req))
    _ok("baseline %s agrees with INPUT_TYPES" % base[0])

    # THE most consequential number of this cut: the default decides whether
    # every workflow that already exists keeps its behaviour. -1 is the
    # sentinel; anything else silently re-shapes stage L in old graphs.
    spec = re.search(r'"%s":\s*\("FLOAT",\s*(\{[\s\S]*?\})\),' % WIDGET,
                     open(PY, encoding="utf-8").read())
    if not spec:
        return _fail("could not read the %s widget spec" % WIDGET)
    opts = ast.literal_eval(re.sub(r'"tooltip":[\s\S]*$', '', spec.group(1)).rstrip().rstrip(",") + "}")
    for key, want in (("default", -1.0), ("min", -1.0), ("max", 20.0)):
        if opts.get(key) != want:
            return _fail("%s %s is %r, must be %r -- the sentinel must be the "
                         "default AND must fit inside the range, or an untouched "
                         "node stops behaving like v850"
                         % (WIDGET, key, opts.get(key), want))
    _ok("sentinel is the default and fits the range (-1.0 .. 20.0)")


# ---------------------------------------------------------------------------
# S2  the resolver is imported, not copied
# ---------------------------------------------------------------------------
def s2_one_resolver():
    tree = ast.parse(open(PY, encoding="utf-8").read())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("uls_sampler"):
            imported |= {a.name for a in node.names}
    if "_resolve_low_shift" not in imported:
        return _fail("_resolve_low_shift must be IMPORTED from uls_sampler -- the "
                     "sentinel semantics may exist in ONE place only")
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_low_shift":
            return _fail("_resolve_low_shift is re-implemented here; two places that "
                         "compute the same thing WILL drift")
    _ok("resolver imported from uls_sampler, no local copy")


# ---------------------------------------------------------------------------
# S3  the sentinel table, written from the spec
# ---------------------------------------------------------------------------
def s3_sentinel_semantics():
    try:
        _resolve_low_shift = _resolver()
    except Exception as err:  # pragma: no cover
        return _fail("cannot import _resolve_low_shift: %r" % err)

    # (high, low_widget) -> expected, straight from the tooltip's promise
    spec = [
        (8.0, -1.0, 8.0),    # sentinel: follow high
        (0.0, -1.0, 0.0),    # sentinel with high off: still off
        (8.0, 0.0, 0.0),     # explicit OFF for the low expert only
        (8.0, 5.0, 5.0),     # its own value
        (0.0, 5.0, 5.0),     # low may shift while high does not
        (8.0, -0.5, 8.0),    # anything below zero is the sentinel
        (8.0, "junk", 8.0),  # junk heals to the sentinel
        (8.0, None, 8.0),
    ]
    for hi, lo, want in spec:
        got = _resolve_low_shift(hi, lo)
        if float(got) != float(want):
            _fail("resolve(high=%r, low=%r) = %r, spec says %r" % (hi, lo, got, want))
            return
    _ok("all %d sentinel cases match the spec" % len(spec))


# ---------------------------------------------------------------------------
# S4/S5  order of application and the fallback, run for real
# ---------------------------------------------------------------------------
_APPLY_SRC = None


def _extract_apply_block():
    """Lift the shift block out of upscale() so it can be RUN in isolation.

    Running the real node needs comfy; the block itself is pure arithmetic over
    two opaque model handles, so it is executed verbatim with stand-ins. Lifting
    (never retyping) is the house rule -- a retyped copy would test the copy.
    """
    global _APPLY_SRC
    if _APPLY_SRC:
        return _APPLY_SRC
    src = open(PY, encoding="utf-8").read()
    # v894 put a `_dual = bool(dual_moe)` line at the TOP of this block, ahead
    # of the old anchor -- so the lift silently cut it off and the block ran on
    # an undefined name. Anchored on the first line of the block as it now
    # stands. LESSON, and it is the session's recurring one: changing a seam
    # that other benches LIFT means checking those benches in the same cut.
    start = src.index("_dual = bool(dual_moe)")
    end = src.index("stages = uls_tile_math.plan_stages")
    block = src[start:end]
    _APPLY_SRC = "\n".join(l[8:] if l.startswith(" " * 8) else l
                           for l in block.splitlines())
    return _APPLY_SRC


def _run_block(sigma_shift, sigma_shift_low, wired):
    """Execute the lifted block with a RECORDING _apply_sigma_shift."""
    calls = []

    class M(object):
        def __init__(self, tag, shift=None):
            self.tag, self.shift = tag, shift

        def __repr__(self):
            return "<%s shift=%r>" % (self.tag, self.shift)

    def fake_apply(model, shift):
        calls.append((model.tag, model.shift, shift))
        return M(model.tag, shift)

    env = {
        "sigma_shift": sigma_shift,
        "sigma_shift_low": sigma_shift_low,
        "model": M("H"),
        "model_low": M("L") if wired else None,
        "_apply_sigma_shift": fake_apply,
        "print": lambda *a, **k: None,
        # v894 put the low half behind a dual_moe gate, because Single plans one
        # stage tagged 'single' and never reads model_low -- every clone built
        # for stage L there was paid for and never touched. EVERY promise in
        # this guard is about the High + Low behaviour, which v894 left bit for
        # bit intact, so the bench drives dual. The Single side is pinned in
        # test_v894_shift_low_single.py, not weakened here.
        "dual_moe": True,
    }
    env["_resolve_low_shift"] = _resolver()
    block_src = _extract_apply_block()
    _check_block_closed(block_src, env)
    exec(compile(block_src, "<v851-block>", "exec"), env)
    return env, calls


_closed_reported = []


def _check_block_closed(block_src, env):
    """v896: say WHICH name is short, instead of surfacing as a bare NameError
    that reads like a broken tree. This bench went red the moment v894 gave the
    block a new name -- the shape v896 closed for two other guards, recurring
    inside the same session.

    It REPORTS and lets the run continue to the NameError, deliberately: _fail
    here only records, so returning early would hand the caller None where it
    unpacks a tuple. (Which is exactly what my first draft did -- a guard whose
    failure path crashes the guard tells you nothing about the tree.)"""
    missing = _LIFT.free_names(block_src, set(env))
    if missing and not _closed_reported:
        _closed_reported.append(True)
        _fail("the lifted apply block needs names this bench does not provide: "
              "%s -- a GUARD fault, not a tree fault; add them to the bench "
              "environment" % ", ".join(missing))


def s4_no_double_shift():
    # the source must capture the low source BEFORE the high rebind
    block = _extract_apply_block()
    i_src = block.index("_low_src =")
    i_hi = block.index("model = _apply_sigma_shift(model, _hi_shift)")
    if not i_src < i_hi:
        return _fail("stage L's source is read AFTER 'model' is rebound -- a low "
                     "shift would then be applied on top of the high one")
    _ok("low source captured before the high patch")

    # and at runtime: no patch may ever start from an already patched model
    for wired in (False, True):
        for hi, lo in ((8.0, 5.0), (8.0, 0.0), (8.0, -1.0), (0.0, 5.0)):
            _env, calls = _run_block(hi, lo, wired)
            for tag, had, _new in calls:
                if had is not None:
                    _fail("a shift was applied to an ALREADY shifted model "
                          "(hi=%s lo=%s wired=%s, calls=%r)" % (hi, lo, wired, calls))
                    return
    _ok("every patch starts from a raw model, in all runs")


def s5_fallback_cases():
    def eff_low(env):
        """What the stage loop will actually use for stage L (its own rule)."""
        return env["model_low"] if env["model_low"] is not None else env["model"]

    # 1) sentinel + no wire -> nothing extra built; L uses the high-shifted model
    env, calls = _run_block(8.0, -1.0, wired=False)
    if env["model_low"] is not None:
        _fail("sentinel without a wired expert must build NOTHING extra "
              "(bit-identical to v850); got %r" % env["model_low"])
    elif len(calls) != 1:
        _fail("sentinel without a wire should patch exactly once, got %r" % calls)
    elif eff_low(env).shift != 8.0:
        _fail("stage L must follow the high shift under the sentinel")
    else:
        _ok("sentinel + no wire: unchanged from v850 (one patch, shared model)")

    # 2) own value + no wire -> a SECOND clone off the RAW model
    env, calls = _run_block(8.0, 5.0, wired=False)
    m, low = env["model"], eff_low(env)
    if low is m:
        _fail("stage L reuses the HIGH model, so its own shift is silently lost")
    elif m.shift != 8.0 or low.shift != 5.0:
        _fail("expected H=8.0 / L=5.0, got H=%r L=%r" % (m.shift, low.shift))
    elif low.tag != "H":
        _fail("the fallback clone must come from the 'model' input, got %r" % low.tag)
    else:
        _ok("own value + no wire: second clone off the raw model (H=8.0, L=5.0)")

    # 3) OFF + no wire -> the RAW model, never the shifted one
    env, _calls = _run_block(8.0, 0.0, wired=False)
    low = eff_low(env)
    if low.shift is not None:
        _fail("stage L was switched OFF but still runs on a shifted model (%r)" % low)
    elif env["model"].shift != 8.0:
        _fail("stage H lost its shift while switching stage L off")
    else:
        _ok("OFF + no wire: stage L on the raw model, stage H still shifted")

    # 4) wired expert -> patched from ITSELF, never from the high model
    for hi, lo, want_l in ((8.0, 5.0, 5.0), (8.0, -1.0, 8.0)):
        env, _calls = _run_block(hi, lo, wired=True)
        low = env["model_low"]
        if low.tag != "L":
            _fail("the wired expert must be patched from itself, got %r" % low.tag)
            return
        if low.shift != want_l or env["model"].shift != hi:
            _fail("wired hi=%s lo=%s -> H=%r L=%r" % (hi, lo, env["model"].shift, low.shift))
            return
    _ok("wired expert: patched from itself, both sentinel and own value")

    # 5) nothing set at all -> nothing patched
    env, calls = _run_block(0.0, -1.0, wired=False)
    if calls:
        _fail("with no shift set the models must be left alone, got %r" % calls)
    else:
        _ok("no shift set: no model is touched")


# ---------------------------------------------------------------------------
# S6  no inert mode may be claimed
# ---------------------------------------------------------------------------
def s6_scope_is_honest():
    """RE-GROUNDED IN v894 -- was s6_never_inert.

    The old promise was: "this widget is ALWAYS honoured, unlike the Sampler's,
    so it needs no honesty line", and it pinned the words "Always honoured" in
    the tooltip. v894 MEASURED that claim and it is false for Single:
    plan_stages(False, ...) yields one stage tagged 'single', and model_low is
    read only where the tag is 'low'. The widget reached nothing there, while
    the backend still built a second clone for it on every run.

    The reasoning behind the old promise still holds where it was true -- there
    is no 'Continuous' mode here, so in High + Low the value is always honoured
    and no schedule is shared. What was wrong was the SCOPE. So the promise is
    re-grounded, not dropped:

        the tooltip must state the true reach (High + Low), and a value dialled
        on a Single run must be REPORTED, never silently absorbed.

    The old wording is now explicitly forbidden, so it cannot creep back."""
    src = open(PY, encoding="utf-8").read()
    if "handoff_mode" in src:
        return _fail("a handoff mode appeared in the Power Upscale -- if the stages "
                     "ever share a schedule, the twin arithmetic in S5 is no "
                     "longer the whole truth")
    tip = re.search(r'"sigma_shift_low":.*?\}\),', src, re.S)
    if not tip:
        return _fail("the sigma_shift_low widget declaration is gone")
    tip = tip.group(0)
    if "Always honoured" in tip or "can never be inert" in src:
        return _fail("the v851 claim ('always honoured' / 'can never be inert') "
                     "is back. It is false in Single -- measured in v894: one "
                     "stage tagged 'single', model_low never read")
    if "HIGH + LOW ONLY" not in tip:
        return _fail("the tooltip must name the mode in which the dial works")
    # and the run must SAY so rather than absorb it -- the v552/v885 rule
    # against silent degradation, in the other direction.
    said_lines = []
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "print"):
            said_lines.append(" ".join(
                c.value for c in ast.walk(node)
                if isinstance(c, ast.Constant) and isinstance(c.value, str)))
    if not any("sigma_shift_low" in t and "ignored" in t for t in said_lines):
        return _fail("a dialled sigma_shift_low on a Single run must be "
                     "reported at runtime, not silently absorbed")
    _ok("no handoff mode; the tooltip states the true reach and the run says "
        "when the dial is out of scope")


# ---------------------------------------------------------------------------
# S7  the frontend survives the growth
# ---------------------------------------------------------------------------
def _names(js, const):
    body = re.search(r"const %s = \[(.*?)\];" % const, js, re.S).group(1)
    return re.findall(r'"([a-z_0-9]+)"', body)


def s7_frontend():
    js = open(JS, encoding="utf-8").read()
    canon = _names(js, "ORDER_CANON")
    disp = _names(js, "DISPLAY_ORDER")
    legacy = _names(js, "DISPLAY_LEGACY_V587")
    if canon[-1] != WIDGET:
        return _fail("ORDER_CANON must APPEND the new widget, it ends with %r" % canon[-1])
    # v852 RE-GROUNDING (declared): v851 pinned "appended at the END of
    # DISPLAY_ORDER too" -- that was the v851 SITUATION, not the promise. The
    # promise is that the CANON never re-sorts (above) and that the display is a
    # permutation of it whose twins sit together. v852 moved the widget under
    # its HIGH partner through the full ceremony, so the display pin follows.
    if disp.index(WIDGET) != disp.index("sigma_shift") + 1:
        return _fail("%s must sit directly under sigma_shift in DISPLAY_ORDER "
                     "(the law of proximity); it sits at %d, sigma_shift at %d"
                     % (WIDGET, disp.index(WIDGET), disp.index("sigma_shift")))
    if canon.index(WIDGET) != len(canon) - 1:
        return _fail("the CANON position may never move, whatever the display does")
    if sorted(canon) != sorted(disp):
        return _fail("DISPLAY_ORDER is no longer a permutation of ORDER_CANON")
    if WIDGET in legacy:
        return _fail("DISPLAY_LEGACY_V587 is FROZEN history and must not learn about "
                     "a widget that did not exist yet")
    _ok("appended to both live orders; the legacy table stays verbatim")

    idx = canon.index(WIDGET)
    if not re.search(r"\n\s*%d:\s*-1\.0," % idx, js):
        return _fail("CANON_DEFAULTS needs slot %d = -1.0, or a short save heals "
                     "the new widget to a wrong value" % idx)
    if not re.search(r"\n\s*%d:\s*\[-1\.0,\s*20\.0\]," % idx, js):
        return _fail("CANON_RANGES needs slot %d = [-1.0, 20.0]; without the "
                     "negative bound the sanitiser clamps the sentinel away" % idx)
    _ok("heal tables carry slot %d (default -1.0, range [-1.0, 20.0])" % idx)

    # the real load chain, end to end, in node
    harness = "\n".join([
        re.search(r"const ORDER_CANON = \[[\s\S]*?\];", js).group(0),
        re.search(r"const DISPLAY_ORDER = \[[\s\S]*?\];", js).group(0),
        re.search(r"const DISPLAY_LEGACY_V587 = \[[\s\S]*?\];", js).group(0),
        re.search(r"const CANON_DEFAULTS = \{[\s\S]*?\n\};", js).group(0),
        re.search(r"function _padToCanon[\s\S]*?\n\}", js).group(0),
        re.search(r"function _tableToCanon[\s\S]*?\n\}", js).group(0),
        re.search(r"function _legacyDisplayToCanon[\s\S]*?\n\}", js).group(0),
        # a v850 save in CANON order, 26 long, lacking only the new widget
        "const old = ORDER_CANON.slice(0, ORDER_CANON.length - 1)"
        ".map((n, i) => 'V' + i);",
        "const padded = _padToCanon(old);",
        "if (padded.length !== ORDER_CANON.length) {",
        "  console.error('FAIL: pad did not reach canon length'); process.exit(1); }",
        "if (padded[ORDER_CANON.length - 1] !== -1.0) {",
        "  console.error('FAIL: the new slot did not heal to the sentinel');",
        "  process.exit(1); }",
        "for (let i = 0; i < old.length; i++) if (padded[i] !== old[i]) {",
        "  console.error('FAIL: padding moved an existing value at ' + i);",
        "  process.exit(1); }",
        # a legacy-display save (the v584 phantom), padded then mapped
        "const hist = DISPLAY_LEGACY_V587.map((n) => 'D:' + n);",
        "const histPad = _padToCanon(hist);",
        "const back = _legacyDisplayToCanon(histPad);",
        "for (let i = 0; i < ORDER_CANON.length; i++) {",
        "  const n = ORDER_CANON[i];",
        "  const want = DISPLAY_LEGACY_V587.indexOf(n) < 0 ? -1.0 : 'D:' + n;",
        "  if (back[i] !== want) {",
        "    console.error('FAIL: legacy map lost ' + n + ' -> ' + back[i]);",
        "    process.exit(1); }",
        "}",
        "console.log('JS OK');",
    ])
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
        fh.write(harness)
        path = fh.name
    try:
        out = subprocess.run([_node(), path], capture_output=True, text=True)
    finally:
        os.unlink(path)
    if out.returncode != 0 or "JS OK" not in out.stdout:
        return _fail("frontend load chain: %s%s" % (out.stdout.strip(), out.stderr.strip()))
    _ok("a v850 save pads to the sentinel; a legacy-display save still maps home")


def _node():
    for cand in ("node", "nodejs"):
        try:
            subprocess.run([cand, "--version"], capture_output=True)
            return cand
        except OSError:
            continue
    return "node"


def main():
    print("[test_v851_upscale_shift_low]")
    s1_serialisation_law()
    s2_one_resolver()
    s3_sentinel_semantics()
    s4_no_double_shift()
    s5_fallback_cases()
    s6_scope_is_honest()
    s7_frontend()
    if _fails:
        print("\n%d FAILURE(S)" % len(_fails))
        sys.exit(1)
    print("\nOK -- 7/7")


if __name__ == "__main__":
    main()
