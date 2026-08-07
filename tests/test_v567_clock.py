"""Guard v567 -- the clock: one truth for console, bar and HUD.

THE MEASURED WOUND (Frank's 11:38 run): the bar counted 26 TICKS - 18 pixel
chunks at ~62 ms and 8 sampler steps at 43-93 s - as equals. Half a second in
it showed 9/26 = "35%" while the truth was 0.07%, then crawled through the
remaining 65% for eleven minutes. Meanwhile the sampler worked in SILENCE:
462 s between 'low tile 1/1' and its done line, not one line in between. And
the node badge, the console total and the bar told three different stories.

v567: a _RunClock owns the bar (value = wall-clock deciseconds, total =
elapsed + eta), every sampler step PRINTS its measured duration plus stage and
run ETA, encode/decode feed the same clock, both probe payloads carry
elapsed/eta so the HUD ticks on the SAME numbers, sub-100ms pixel chunks print
as 'NNms/chunk' and fold their litany, and the done line names why the ComfyUI
badge differs (executor overhead outside the function).

The clock takes `now` as a parameter, so this guard runs the ENTIRE estimation
ladder deterministically - with Frank's measured rates as the test case.

Script-style: exit 0 = pass.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v567_clock] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


class _FakeBar:
    def __init__(self):
        self.calls = []

    def update_absolute(self, value, total=None, preview=None):
        self.calls.append((value, total))


def main():
    pu = _read("nodes", "ph_power_upscale.py")
    js = _read("web", "js", "ph_power_upscale.js")

    # ---- the clock, extracted and RUN with a fake time source -----------------
    # AMENDED IN v576 (1st amendment): the clock moved to nodes/ph_runclock.py
    # so the Sampler shares it (PU re-exports the names). Structure pinned
    # instead of text: the MODULE is the window now - the whole file is
    # exec'd, and the two regex windows (whose anchors were the fragile part,
    # ledger lesson #3) are retired. Everything this guard asserts about the
    # class itself is unchanged.
    src = _read("nodes", "ph_runclock.py")
    ns = {}
    exec(compile(src, "<ph_runclock>", "exec"), ns)  # noqa: S102 - our own source, measured
    Clock, fmt = ns["_RunClock"], ns["_fmt_clock"]

    t = {"now": 0.0}
    bar = _FakeBar()
    c = Clock(bar, now=lambda: t["now"])
    # Frank's run: H 848 single-tile 3 steps, L 1104 single-tile 5 steps.
    c.post("pix:high", 9, 848.0 * 848.0)
    c.post("enc:high", 1, 848.0 * 848.0)
    c.post("step:high", 3, 848.0 * 848.0)
    c.post("dec:high", 1, 848.0 * 848.0)
    c.post("pix:low", 9, 1104.0 * 1104.0)
    c.post("enc:low", 1, 1104.0 * 1104.0)
    c.post("step:low", 5, 1104.0 * 1104.0)
    c.post("dec:low", 1, 1104.0 * 1104.0)

    # rung 4: NOTHING measured -> eta None, bar honest (~1%, never a made-up %)
    if c.eta() is not None:
        _fail("with nothing measured the eta must be None - honest ignorance "
              "beats a confident lie")
    t["now"] = 0.5
    c.push()
    v, tot = bar.calls[-1]
    if v != 5:
        _fail(f"bar value must be wall-clock deciseconds (0.5s -> 5), got {v}")
    if not tot or v / tot > 0.05:
        _fail(f"an unmeasured run must show ~1%, not v565's 35% - got "
              f"{v}/{tot}")

    # measure the pixel chunks (62 ms each) - rung 1 for pix, still no steps
    for _ in range(9):
        t["now"] += 0.062
        c.measure("pix:high", 0.062)
    e = c.eta()
    if e is None:
        _fail("with pix measured, rung 3 must extrapolate the rest - never None")
    # rung 3 extrapolates ~1s/unit-weight -> tiny; the point is it EXISTS and
    # the bar no longer claims 35%.
    v, tot = bar.calls[-1]
    if v / tot > 0.60:
        _fail(f"weight extrapolation must keep the bar honest, got {v}/{tot}")

    # rung 1+2: measure H steps at Frank's 42.8 s -> L scales by pixel area
    for _ in range(3):
        t["now"] += 42.8
        c.measure("step:high", 42.8)
    r_low = c._rate("step:low")
    want = 42.8 * (1104.0 * 1104.0) / (848.0 * 848.0)     # ~72.5 (linear floor)
    if r_low is None or abs(r_low - want) > 0.1:
        _fail(f"an unmeasured step class must scale from a measured one by "
              f"weight (42.8 * 1104^2/848^2 = {want:.1f}), got {r_low}")
    e_low = c.eta("low")
    if e_low is None or not (5 * want * 0.9 <= e_low):
        _fail(f"the stage eta must cover the remaining low posts, got {e_low}")

    # EMA: the class corrects itself once it measures (attention makes big
    # tiles super-linear; Frank measured 92.5 where linear said 72.5)
    t["now"] += 92.5
    c.measure("step:low", 92.5)
    if abs(c.posts["step:low"]["rate"] - 92.5) > 1e-6:
        _fail("first own measurement must SET the class rate")
    t["now"] += 93.1
    c.measure("step:low", 93.1)
    if abs(c.posts["step:low"]["rate"] - (0.5 * 92.5 + 0.5 * 93.1)) > 1e-6:
        _fail("the rate must be an EMA (alpha 0.5), so a model-load-heavy "
              "first step washes out")

    # resize: the VRAM clamp corrects the plan, never below done
    c.measure("pix:low", 0.1)
    c.resize("pix:low", 22)
    if c.posts["pix:low"]["units"] != 22:
        _fail("resize must adopt the true chunk count")
    c.resize("pix:low", 0)
    if c.posts["pix:low"]["units"] != c.posts["pix:low"]["done"]:
        _fail("resize must never drop units below done")

    # monotone value, total >= value+1 always
    for a, b in bar.calls:
        if b is not None and b <= a:
            _fail("total must stay ahead of value (a full bar that is not "
                  "done is a lie)")
    if [a for a, _ in bar.calls] != sorted(a for a, _ in bar.calls):
        _fail("the bar value must be monotone - it is a wall clock")

    if fmt(65) != "1:05" or fmt(3725) != "1:02:05":
        _fail("_fmt_clock must render m:ss / h:mm:ss")

    # ---- the step SPEAKS -------------------------------------------------------
    ref = pu[pu.index("def _refine_tiles"):pu.index("# \u2500\u2500 v549")]
    if "step {step + 1}/{steps} {dt:.1f}s (stage eta " not in ref:
        _fail("the per-step console line is gone - 462 mute seconds between "
              "'tile 1/1' and its done line is the wound this cut closes")
    if "run eta ~" not in ref:
        _fail("the run eta must ride every step line, tilde-marked (an "
              "estimate that does not say so is a lie)")
    if 'clock.measure(f"step:{stage_tag}"' not in ref:
        _fail("the steps no longer feed the clock")
    if 'clock.measure(f"enc:{stage_tag}"' not in ref or \
       'clock.measure(f"dec:{stage_tag}"' not in ref:
        _fail("encode/decode left the clock - 12% of the run would be "
              "unbudgeted again")
    if "pbar.update_absolute" in ref:
        _fail("a tick writer survived in _refine_tiles - two writers on one "
              "bar tell two stories")
    if "done_steps" in ref:
        _fail("the tick counter is back")

    # ---- the chunks format honestly + fold -------------------------------------
    if 'f"{per * 1000:.0f}ms" if per < 0.1' not in pu:
        _fail("a 62 ms chunk must print as '62ms/chunk', never '0.0s'")
    if "folded (" not in pu or "too fast to narrate" not in pu:
        _fail("the sub-100ms chunk litany must fold into one summary line")
    if 'pixel_probe(_tag, _p["k"], _p["n"], j, _n, part,' not in pu:
        _fail("folding is a console courtesy - the probe must still fire per "
              "chunk, with the clock riding along")

    # ---- one clock everywhere ---------------------------------------------------
    if pu.count('"elapsed": int(elapsed)') != 2:
        _fail("BOTH probe payloads must carry the clock (tile + pixel doors)")
    if 'clock.eta(stage_tag)' not in pu or "clock.elapsed()" not in pu:
        _fail("the probes/console must read the SAME clock, not a private one")
    if "ComfyUI badge adds executor overhead" not in pu:
        _fail("the done line must name why the ComfyUI badge differs - three "
              "unexplained numbers were the complaint")
    body = pu[pu.index("def upscale"):]
    if body.index("t_all = time.monotonic()") > body.index("_resolve_input(image, video)"):
        _fail("t_all must start at the TOP of the function so 'total=' and the "
              "badge disagree only by executor overhead")

    # ---- frontend ---------------------------------------------------------------
    # AMENDED IN v579 (1st amendment): the banner moved to v579 because the FILE
    # changed - the result viewer became a canvas (the old <img> re-decoded every
    # frame on an src swap and handed the compositor an 896px texture at every
    # zoom step), and the single-tile minimap stopped painting a solid orange
    # block. The banner IS the Firefox-cache proof: if it does not move with the
    # file, a stale cached copy is indistinguishable from a fresh one.
    # AMENDED IN v582 (1st amendment, part 2): this pinned the LITERAL banner
    # string ("... v579 loaded"), so every banner bump had to hand-edit this
    # guard to stay green - a guard that must be hand-edited on a schedule
    # will one day be hand-edited wrong (the v580 baseline lesson, verbatim).
    # The claim v567 owns: the file carries a self-proving per-file banner
    # (v531 doctrine) and has done so since v567. Pin the STRUCTURE: the
    # banner pattern, and a version that can never fall below v567.
    _m = re.search(r'console\.info\("\[PLS\] ph_power_upscale\.js v(\d+) '
                   r'loaded"\);', js)
    if not _m:
        _fail("per-file banner is gone (v531 doctrine: every JS file caches "
              "individually in Firefox - the banner IS the cache proof)")
    if int(_m.group(1)) < 567:
        _fail(f"per-file banner claims v{_m.group(1)}, which predates the "
              f"banner itself (v567) - a stale or hand-rolled copy")
    if "function _fmtClock(" not in js:
        _fail("the HUD clock formatter is gone")
    if '" · ETA ~" + _fmtClock(d.eta)' not in js:
        _fail("the HUD must show the eta from the payload - same numbers as "
              "the console, that is the congruence Frank asked for")
    if "d.elapsed !== undefined" not in js:
        _fail("an old backend without the clock must not break the pane "
              "(payload feature-test, v531 cache doctrine)")

    # ---- serialisation: this cut adds NO widget ----------------------------------
    canon = re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1)
    names = re.findall(r'"([a-z_ +()]+)"', canon)
    # AMENDED IN v582 (1st amendment): the absolute count len==25 was a TEXT
    # pin on a moving structure (lesson 1). v582 tail-appends a widget and
    # five sibling guards broke on the same line at once. The claim owned
    # here is historical - "v567 added no widget; the order up to pixel_stage
    # stands" - and a POSITION pin states exactly that, while every future
    # tail-append preserves it.
    if len(names) < 25 or names[24] != "pixel_stage":
        _fail(f"the first 25 canon slots up to pixel_stage are v567's "
              f"history and must stand - got {{len(names)}} entries, "
              f"slot 24 = {{names[24] if len(names) > 24 else 'MISSING'}}")
    if "_healPreV567" in js:
        _fail("a heal step for a cut that changes no widget can only do damage")
    harness = "\n".join(
        re.search(rx, js, re.S).group(0) for rx in (
            r"const ORDER_CANON = \[[\s\S]*?\];",
            r"const CANON_DEFAULTS = \{[\s\S]*?\n\};",
            r"const LEN_PRE_V564 = \d+;",
            r"function _healPreV564\(wv\) \{[\s\S]*?\n\}",
            r"function _padToCanon\(arr\) \{[\s\S]*?\n\}")) + r"""
// AMENDED IN v582 (1st amendment, part 3): '!== 25' pinned the canon LENGTH
// of the day - the fourth copy of the same text pin in this cut. v563 shows
// the structural form: measure against ORDER_CANON.length and the harness
// survives every future widget. The historical claim (slot 24 = pixel_stage,
// default 'model + fit') stays pinned verbatim.
const N = ORDER_CANON.length;
const a = _padToCanon(_healPreV564(Array.from({length: 24}, (_, i) => i)));
if (a.length !== N || a[24] !== "model + fit") {
    console.error("FAIL 24->canon"); process.exit(1);
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

    print("PASS: v567 -- the clock (time bar, honest before calibration, EMA "
          "rates, weight-scaled classes, extrapolation rung), steps speak with "
          "stage+run eta, chunks format in ms and fold, both probes carry the "
          "clock, badge difference named, 25 canon entries")
    sys.exit(0)


if __name__ == "__main__":
    main()
