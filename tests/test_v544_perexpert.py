"""Guard v544 -- per-expert sampler/scheduler + display reorder + migration ladder."""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _fail(m): print("FAIL: " + m); sys.exit(1)
def _read(*p): return open(os.path.join(ROOT, *p), encoding="utf-8").read()

def main():
    py = _read("nodes", "uls_sampler.py"); js = _read("web", "js", "uls_sampler.js")
    ml = _read("web", "js", "ph_media_loader.js")

    # -- dialog no longer re-centres (the jump) --------------------------------
    if "align-items:flex-start" not in ml or "padding:24px 0" not in ml:
        _fail("batch dialog still centred -> it jumps when the content height changes")
    if "max-height:92vh" in ml: _fail("card max-height must account for the top padding")

    # -- backend: widgets appended at the END (index stability) ----------------
    req = py[py.index('"required": {'):py.index('"optional"')]
    names = re.findall(r'\n                "([a-z_]+)":', req)
    # RE-GROUNDED v839: the old pin read names[-2:] -- true the day v544
    # appended them, but a LATENCY: it broke on the first later append
    # (sigma_shift_low), same class as the test_v348 triple lesson. The
    # invariant v544 owns: its twins sit TOGETHER, in order, after
    # handoff_mode. Where later appends land is the #577 baseline's law.
    if "sampler_low" not in names or "scheduler_low" not in names:
        _fail("the v544 twins left required (MASTER 6)")
    _i = names.index("sampler_low")
    if names[_i:_i + 2] != ["sampler_low", "scheduler_low"]             or _i < names.index("handoff_mode"):
        _fail("the v544 twins must sit together, in order, after handoff_mode "
              "(index stability, MASTER 6)")
    if 'SAME_AS_HIGH = "same as high"' not in py: _fail("default sentinel missing")
    if "def _low_or(" not in py: _fail("LOW override resolver missing")

    # -- policy: sampler free everywhere, scheduler only in Wan MoE parity -----
    if "scheduler_low is ignored in 'Continuous'" not in py:
        _fail("Continuous must IGNORE scheduler_low and say so (no silent mis-denoise)")
    if "_low_or(scheduler_low, scheduler)" not in py: _fail("scheduler_low never resolved")
    if py.count("_low_or(sampler_low, sampler_name)") < 2:
        _fail("sampler_low must reach the LOW pass in both the steps and the sigma path")
    if "sampler_low=sampler_low" not in py or "scheduler_low=scheduler_low" not in py:
        _fail("dispatch does not thread the LOW overrides")

    # -- frontend: order arrays grew to 19, same name set ----------------------
    order = re.search(r"const ORDER_V404 = \[(.*?)\];", js, re.S).group(1)
    disp = re.search(r"const DISPLAY_ORDER = \[(.*?)\];", js, re.S).group(1)
    o = re.findall(r'"([a-z_]+)"', order); d = re.findall(r'"([a-z_]+)"', disp)
    # RE-GROUNDED v839 (same latency as the length-19 / names[-2:] pins):
    # v544 owns that its twins ARRIVED as appends and the arrays stay
    # permutations -- not the absolute count, which every later append bumps.
    if len(o) != len(d): _fail(f"order arrays diverge ({len(o)}/{len(d)})")
    if sorted(o) != sorted(d): _fail("DISPLAY_ORDER is not a permutation of ORDER_V404")
    _oi = o.index("sampler_low")
    if o[_oi:_oi + 2] != ["sampler_low", "scheduler_low"]             or _oi < o.index("handoff_mode"):
        _fail("serialised order: the v544 twins must sit together after "
              "handoff_mode (every old index stays put)")
    # display: every LOW twin directly under its HIGH partner
    for hi, lo in (("cfg", "cfg_low"), ("sampler_name", "sampler_low"), ("scheduler", "scheduler_low")):
        if d.index(lo) != d.index(hi) + 1:
            _fail(f"display order: {lo} must sit directly under {hi}")
    if d.index("cfg") != d.index("steps") + 1: _fail("display: cfg must follow steps")

    # -- migration ladder: a v543 save (17) must heal to 19 --------------------
    if "LEN_V492_CURRENT = 17" not in js: _fail("v543 legacy length not registered")
    if "_looksV492Current" not in js or "_healV492Current" not in js:
        _fail("v543 -> v544 heal missing")
    if js.count("DEFAULT_SAMPLER_LOW);") < 4 or js.count("DEFAULT_SCHEDULER_LOW);") < 4:
        _fail("older heals must also append the two v544 defaults")
    if 'const DEFAULT_SAMPLER_LOW = SAME_AS_HIGH' not in js: _fail("JS default must be the sentinel")

    # -- greying ---------------------------------------------------------------
    if '"sampler_low", "scheduler_low"' not in js: _fail("new widgets not in DUAL_ONLY")
    if "moe|rebase" not in js: _fail("scheduler_low must grey outside Wan MoE parity")
    # v545: a widget whose VALUE drives another widget's state MUST re-run the state pass
    if 'const STATE_WIDGETS = ["dual_moe", "handoff_mode"]' not in js:
        _fail("handoff_mode must re-run _applyModeState (else scheduler_low stays stale)")
    if js.count("for (const name of STATE_WIDGETS)") != 1:
        _fail("STATE_WIDGETS callback hook missing")
    print("PASS: v544 -- per-expert sampler/scheduler, display reorder, 17->19 migration, dialog pinned")
    sys.exit(0)

if __name__ == "__main__":
    main()
