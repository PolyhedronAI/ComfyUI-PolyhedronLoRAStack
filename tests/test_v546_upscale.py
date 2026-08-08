"""Guard v546 -- Power Upscale: per-stage sampler/scheduler (stage chain: no gate needed)."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _fail(m): print("FAIL: " + m); sys.exit(1)
def _read(*p): return open(os.path.join(ROOT, *p), encoding="utf-8").read()

def main():
    py = _read("nodes", "ph_power_upscale.py"); js = _read("web", "js", "ph_power_upscale.js")

    # -- backend: appended at the END (index stability), resolver imported not copied ----
    req = py[py.index('"required"'):py.index('"optional"')]
    names = re.findall(r'"([a-z_0-9]+)":\s*\(', req)
    # v549 hardening: position pin instead of tail pin - the v546 pair sits
    # directly after sigma_shift (the v514 tail); later fields append BEHIND it.
    if names.index("sampler_low") != names.index("sigma_shift") + 1 or \
       names.index("scheduler_low") != names.index("sampler_low") + 1:
        _fail("the v546 pair must sit directly after sigma_shift (appended in v546)")
    # v851 RE-GROUNDING (declared): this pinned the LITERAL import text
    # "_low_or, SAME_AS_HIGH". The promise was never that spelling - it is that
    # the resolvers are IMPORTED from uls_sampler and NOT re-implemented here
    # (two places that compute the same thing drift). v851 added
    # _resolve_low_shift to the same import and re-wrapped the line; the
    # substring broke while the promise held. Checked structurally now, and in
    # its STRONGER form: imported from uls_sampler AND absent as a local def.
    import ast as _ast
    _tree = _ast.parse(py)
    _imported = set()
    for _n in _ast.walk(_tree):
        if isinstance(_n, _ast.ImportFrom) and (_n.module or "").endswith("uls_sampler"):
            for _a in _n.names:
                _imported.add(_a.name)
    for _need in ("_apply_sigma_shift", "_resolve_low_shift", "_low_or", "SAME_AS_HIGH"):
        if _need not in _imported:
            _fail("%s must be IMPORTED from uls_sampler (house pattern), not copied" % _need)
    for _n in _ast.walk(_tree):
        if isinstance(_n, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and _n.name in _imported:
            _fail("%s is re-implemented locally - the mirror WILL drift" % _n.name)
    if "_low_or(sampler_low, sampler_name) if is_low" not in py:
        _fail("stage L must resolve its own sampler")
    if "_low_or(scheduler_low, scheduler) if is_low" not in py:
        _fail("stage L must resolve its own scheduler")
    if "sampler={samp} sched={sched}" not in py:
        _fail("the per-stage log line must state what actually ran (measure > believe)")
    # the whole point: NO handoff gate here -- stage L is an independent run
    if "handoff" in py.lower() and "no handoff" not in py.lower() and "boundary split" not in py.lower():
        _fail("a handoff gate must NOT appear in the stage chain")

    # -- frontend: 18 canonical, twins adjacent in DISPLAY, heal for old saves ----------
    canon = re.findall(r'"([a-z_]+)"', re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1))
    disp = re.findall(r'"([a-z_]+)"', re.search(r"const DISPLAY_ORDER = \[(.*?)\];", js, re.S).group(1))
    # v549 hardening: the brittle length pin became INDEX pins. Index stability
    # is the real invariant - new fields APPEND, so this guard now survives them.
    if len(canon) != len(disp): _fail("DISPLAY_ORDER and ORDER_CANON diverged in length")
    V514_16 = ["dual_moe", "upscale_by", "denoise", "steps", "cfg",
               "upscale_by_low", "denoise_low", "steps_low", "cfg_low",
               "seed", "control_after_generate", "sampler_name", "scheduler",
               "tile_size", "tile_overlap", "sigma_shift"]
    if canon[:16] != V514_16: _fail("the 16 v514 fields must keep indices 0-15")
    if sorted(canon) != sorted(disp): _fail("DISPLAY_ORDER is not a permutation of ORDER_CANON")
    # AMENDED IN v585 (1st amendment) - THE DISPLAY LAW, measured 2026-07-13
    # the hard way: the live frontend serialises widgets_values in the WIDGET
    # (display) order, not in ORDER_CANON. The v584 cut moved a widget into
    # the display middle and every pre-v584 save loaded shifted by one slot
    # from position 4 on - the seed landed in cfg_low, and the self-heal net
    # then MASKED half the damage by "repairing" shifted values into wrong
    # defaults.
    # AMENDED IN v589 (2nd amendment) - the re-sort became LEGAL, not waved
    # through: configure stopped GUESSING the save order. Marked saves (v588)
    # are proven canon; unmarked saves are TYPE-fingerprinted (canon and the
    # old display order differ by number-vs-string at slots 13/14 vs 16/17,
    # two witnesses per side); a save in the old display order loads through
    # DISPLAY_LEGACY_V587. Frank's call, rollback anchor v588. The law that
    # REMAINS: the historic order must survive VERBATIM as the legacy map
    # (or every pre-v589 stray save shifts again), and the display stays a
    # permutation - both pinned below.
    # the v589..v851 display order: the v589 permutation with sigma_shift_low
    # still parked at the very end, where v851 appended it.
    DISPLAY_V589_851 = ["dual_moe", "upscale_by", "upscale_by_low",
                        "final_upscale_by",
                        "denoise", "denoise_low", "steps", "steps_low",
                        "cfg", "cfg_low", "seed", "control_after_generate",
                        "sampler_name", "sampler_low", "scheduler",
                        "scheduler_low", "tile_size", "tile_overlap",
                        "sigma_shift", "result_preview", "process_preview",
                        "mute_staging_logs", "resize_method", "per_batch",
                        "vae_tiling", "pixel_stage", "sigma_shift_low"]
    DISPLAY_LEGACY = ["dual_moe", "upscale_by", "upscale_by_low",
                      "denoise", "denoise_low", "steps", "steps_low",
                      "cfg", "cfg_low", "seed", "control_after_generate",
                      "sampler_name", "sampler_low", "scheduler",
                      "scheduler_low", "tile_size", "tile_overlap",
                      "sigma_shift", "result_preview", "process_preview",
                      "mute_staging_logs", "resize_method", "per_batch",
                      "vae_tiling", "pixel_stage", "final_upscale_by"]
    # v852 AMENDMENT (3rd): sigma_shift_low moved out of the tail and under
    # sigma_shift. Legal by the SAME ceremony v589 used and this guard demanded
    # in its own failure text: the order it left behind is frozen as
    # DISPLAY_LEGACY_V851, the fingerprint learned to name it, and this pin was
    # rewritten in the same cut. The law that REMAINS is unchanged: every order
    # we ever displayed must survive VERBATIM as a table, or a stray save from
    # that era shifts. Rollback anchor: v851.
    DISPLAY_V852 = ["dual_moe", "upscale_by", "upscale_by_low",
                    "final_upscale_by",   # v589: the law of proximity
                    "denoise", "denoise_low", "steps", "steps_low",
                    "cfg", "cfg_low", "seed", "control_after_generate",
                    "sampler_name", "sampler_low", "scheduler",
                    "scheduler_low", "tile_size", "tile_overlap",
                    "sigma_shift", "sigma_shift_low",   # v852: the last twin
                    "result_preview", "process_preview",
                    "mute_staging_logs", "resize_method", "per_batch",
                    "vae_tiling", "pixel_stage"]
    if disp != DISPLAY_V852:
        for i, (a, b) in enumerate(zip(disp, DISPLAY_V852)):
            if a != b:
                _fail(f"DISPLAY slot {i} moved: '{a}' where '{b}' is the "
                      f"v852 order. Re-sorts need the full ceremony: freeze "
                      f"the order being left as a table + teach the "
                      f"fingerprint to name it + rewrite this guard, in one "
                      f"cut")
                break
        else:
            _fail("DISPLAY_ORDER changed length - a widget appears or vanishes")
    # every LOW twin must sit directly under its HIGH partner (the point of the
    # display permutation in the first place). Checked as a RULE, not as a list,
    # so the next twin cannot be forgotten.
    for hi, lo in (("upscale_by", "upscale_by_low"), ("denoise", "denoise_low"),
                   ("steps", "steps_low"), ("cfg", "cfg_low"),
                   ("sampler_name", "sampler_low"), ("scheduler", "scheduler_low"),
                   ("sigma_shift", "sigma_shift_low")):
        if disp.index(lo) != disp.index(hi) + 1:
            _fail(f"'{lo}' must sit directly under '{hi}' - the law of "
                  f"proximity is the whole reason this node permutes at all")
    # every order we have ever displayed must still exist as a frozen table
    for table in ("DISPLAY_LEGACY_V587", "DISPLAY_LEGACY_V851"):
        if not re.search(r"const %s = \[" % table, js):
            _fail(f"{table} is gone - every stray save from that era shifts "
                  f"again without it")
    legacy851 = re.findall(
        r'"([a-z_]+)"',
        re.search(r"const DISPLAY_LEGACY_V851 = \[(.*?)\];", js, re.S).group(1))
    if legacy851 != DISPLAY_V589_851:
        _fail("DISPLAY_LEGACY_V851 must be the v589..v851 order VERBATIM - it "
              "is the load path for that era, not a suggestion")
    legacy_js = re.search(r"const DISPLAY_LEGACY_V587 = \[(.*?)\];", js, re.S)
    if not legacy_js:
        _fail("DISPLAY_LEGACY_V587 is gone - every pre-v589 stray save "
              "shifts again without it")
    if re.findall(r'"([a-z_]+)"', legacy_js.group(1)) != DISPLAY_LEGACY:
        _fail("DISPLAY_LEGACY_V587 must be the v514..v588 order VERBATIM - "
              "it is the load path for history, not a suggestion")
    if canon[16:18] != ["sampler_low", "scheduler_low"]: _fail("the v546 pair must sit at indices 16/17 (appended, never inserted)")
    # v547: law of proximity -- EVERY low twin sits directly under its high partner
    for hi, lo in (("upscale_by", "upscale_by_low"), ("denoise", "denoise_low"),
                   ("steps", "steps_low"), ("cfg", "cfg_low"),
                   ("sampler_name", "sampler_low"), ("scheduler", "scheduler_low")):
        if disp.index(lo) != disp.index(hi) + 1:
            _fail(f"display: {lo} must sit directly under {hi} (law of proximity)")
    # v547: the INPUT sockets follow the same law
    ins = re.findall(r'"([a-z_]+)"', re.search(r"const INPUT_DISPLAY_ORDER = \[(.*?)\];", js, re.S).group(1))
    if ins[:2] != ["image", "video"]:
        _fail("what you upscale (image/video) must be the FIRST sockets")
    for hi, lo in (("model", "model_low"), ("upscale_model", "upscale_model_low")):
        if ins.index(lo) != ins.index(hi) + 1:
            _fail(f"input sockets: {lo} must sit directly under {hi}")
    if "target_slot = i" not in js:
        _fail("permuting sockets WITHOUT repairing LLink.target_slot corrupts the links")
    if "_reorderInputsToDisplay(this)" not in js:
        _fail("loaded graphs must be permuted too (configure restores the saved order)")
    if "LEN_PRE_V546 = 16" not in js or "_healPreV546" not in js:
        _fail("pre-v546 saves (16 values) must heal to 18")
    if "_displayToCanon" not in js or "onSerialize" not in js:
        _fail("saves must be written back in canonical order")
    if '"sampler_low", "scheduler_low"' not in js: _fail("new widgets not greyed outside High + Low")
    print("PASS: v546 power upscale -- per-stage sampler/scheduler, twins adjacent, 16->18 heal")
    sys.exit(0)

if __name__ == "__main__":
    main()
