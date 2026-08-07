"""Guard v571 -- weights that mean physics, telemetry that means what ran.

FRANK'S FINISHED v570 RUN (the first final pass to complete) measured two
half-truths this cut retires:

  * 'run eta ~1:07:54' during the stages, against a real ~6 min of final
    pass. v570 killed the cross-kind borrow, but the pass then fell to the
    rung-3 blend carrying a weight of OUTPUT area x scale^2 - 16x too heavy
    for a 4x model. ESRGAN cost lives on INPUT pixels (the RRDB trunk runs
    at input resolution; the upsampler tail is cheap). The tilde excused
    the number formally; it was still a fairy tale.

  * 'peak ~1.9 GB @ per_batch=8' for a pass the budget had clamped to 6
    frames in flight. The line named the widget, not the truth.

THREE CUTS, PINNED HERE:

1. WEIGHTS FOLLOW THE OPERATION. A 'pix' post weighs the stage's INPUT
   canvas (captured BEFORE scaled_size); a 'fit' post weighs the target.
   The final post weighs the last stage canvas - never x scale^2.

2. ONE GATE, TWO PLACES, SAME SPELLING. The up-front 'pix:final' post fires
   iff the final-pass block will run (wire is not None). v570 gated the
   post on scale > 0 - a scale-less model would have run the pass and then
   KeyError'd the clock on its first measure.

3. THE SUMMARY NAMES WHAT FLEW. peak and the label use the EFFECTIVE
   (clamped) chunk returned by the resident pass, and say when the budget
   clamped the widget.

Script-style: exit 0 = pass.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v571_weights] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    pu = _read("nodes", "ph_power_upscale.py")

    # ---- 1: weights follow the operation ------------------------------------------
    plan = pu[pu.index("plans = []"):pu.index("pbar = comfy.utils.ProgressBar")]
    if 'st["pix_in"] = float(w * h)' not in plan:
        _fail("the stage's INPUT canvas must be captured for the pix weight")
    if plan.index('st["pix_in"]') > plan.index("uls_tile_math.scaled_size"):
        _fail("pix_in must be captured BEFORE scaled_size - it is the canvas "
              "the model will actually chew")
    if ('st["pix_in"] if st[\'pix_kind\'] == "pix"' not in pu
            and "st[\"pix_in\"] if st[\"pix_kind\"] == \"pix\"" not in pu):
        _fail("the stage post must weigh pix by INPUT and fit by target")
    if "* _fsc * _fsc" in pu:
        _fail("the x scale^2 over-weight is back - a 4x pass would again "
              "enter the rung-3 blend 16x too heavy")

    # ---- 2: one gate, two places, same spelling ------------------------------------
    _p0 = pu.index('if str(pixel_stage) == "model final":')
    post = pu[_p0:pu.index("clock.push()", _p0)]
    if "if _um_fin is not None:" not in post:
        _fail("the up-front post must gate on the WIRE (is not None) - "
              "exactly like the block, or a scale-less model KeyErrors the "
              "clock on its first measure")
    if 'clock.post("pix:final", pix_chunks, float(w * h))' not in post:
        _fail("the final post must weigh the pass's INPUT (the last stage "
              "canvas), nothing else")

    # ---- 3: the summary names what flew ---------------------------------------------
    res = pu[pu.index("def _esrgan_resident("):pu.index("def _esrgan_pass(")]
    if "mid, info, chunk)" not in res:
        _fail("the resident pass must return its CLAMPED chunk - the caller "
              "cannot know the budget's verdict otherwise")
    body = pu[pu.index("def _esrgan_pass("):pu.index("def _final_canvas(")]
    if "image, mid, path, eff_chunk = _esrgan_resident(" not in body:
        _fail("the pass must unpack the effective chunk")
    if "f in flight" not in body:
        _fail("the summary must name frames IN FLIGHT, not the widget")
    if "budget-clamped" not in body:
        _fail("a clamped widget must be SAID, with the word")
    if "min(int(eff_chunk) or n, n) * mid[0]" not in body:
        _fail("the peak estimate must use the EFFECTIVE chunk")

    # ---- the physics, EXECUTED on the real clock -------------------------------------
    # AMENDED IN v576 (1st amendment): the clock moved to nodes/ph_runclock.py
    # (shared with the Sampler; PU re-exports). The whole module is the window
    # now - the two regex windows and their anchors are retired (lesson #3).
    src = _read("nodes", "ph_runclock.py")
    ns = {}
    exec(compile(src, "<ph_runclock>", "exec"), ns)
    c = ns["_RunClock"](pbar=None, now=lambda: 0.0)
    # model+fit shape with v571 weights: stage L chews 848^2, the final pass
    # chews 1104^2. One measured model rate must scale to the other by the
    # INPUT ratio - the ratio a resident RRDB actually obeys.
    c.post("pix:low", 9, 848.0 * 848.0)
    c.post("pix:final", 11, 1104.0 * 1104.0)
    c.measure("pix:low", 24.0)
    r = c._rate("pix:final")
    want = 24.0 * (1104.0 * 1104.0) / (848.0 * 848.0)
    if r is None or abs(r - want) > 1e-6:
        _fail(f"input-area scaling broke: expected {want:.2f}s, got {r}")
    if not (30.0 < want < 50.0):
        _fail("the physics sanity moved - recheck the scenario numbers")

    print("[test_v571_weights] OK - pix weights = input pixels (executed on "
          "the real clock), one gate two places, the summary names the "
          "clamped truth")
    sys.exit(0)


if __name__ == "__main__":
    main()
