"""Guard v559 -- the upscaler fixes: VSR downscale guard + keep_proportion.

BEHAVIOURAL (the parts that are pure math run for real):

  * `_fit_method` - the reason this cut exists. nvidia_rtx_vsr is a
    SUPER-RESOLUTION effect; with a 4x ESRGAN in front of a 1.05x stage the
    "fit" is a ~3.8x SHRINK, so Maxine was being used as a downscaler.
    On a shrink the method is overruled to `area` (the filter that averages
    the pixels it throws away) - LOUDLY, with a note. A real upscale keeps VSR.

    AMENDED IN v565. This guard used to end that sentence with "a non-VSR
    method is never touched" - and pinned it. That pinned a belief which v559's
    OWN docstring already contradicted three lines earlier ("bicubic/bilinear
    ring"). Frank then ran `fit=bicubic` behind a 4x model in front of a 2x
    stage: a 2:1 downscale through an interpolator that does not antialias,
    throwing away three of every four ESRGAN pixels without averaging them.
    The guard held the door open for it. A guard that pins a measured-wrong
    belief is worse than no guard - it makes the wrong thing permanent. The law
    is corrected below and pinned in full by test_v565.
  * `_proportion_plan` / `_offset` / `_parse_color` - the Resize-v2 gap:
    stretch (distort, the v554 default), crop (cover + cut), pad (contain +
    fill), with the anchor honoured on both, and a fail-soft colour parser.

Text pins hold the tensor-side (`_crop_or_pad`, no torch in this sandbox) and
the contracts: the four widgets appended LAST, the outputs APPENDED (width /
height / mask - existing wires stay intact), the mask riding the SAME geometry
with a 0 border on pad, and the Power Upscale routing its stage fit through
the same guard.

Script-style: exit 0 = pass.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v559_fitguard] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    pu = _read("nodes", "ph_power_upscale.py")
    fu = _read("nodes", "ph_fast_upscale.py")

    # ---- _fit_method: extracted and RUN ---------------------------------------
    m = re.search(r"def _fit_method\(.*?\n(?=\ndef _chunks)", pu, re.S)
    if not m:
        _fail("_fit_method not extractable from ph_power_upscale")
    ns = {}
    exec(m.group(0), ns)  # noqa: S102 - our own source, measured
    fit = ns["_fit_method"]

    # Frank's real case: 4x ESRGAN in front of a 1.05x stage
    method, note = fit(4296, 5364, 1128, 1408, "nvidia_rtx_vsr")
    if method != "area":
        _fail(f"a VSR shrink must be overruled to area, got {method!r}")
    if not note or "upscaler" not in note:
        _fail("the overrule must SAY it happened (no silent trap)")
    if "3.8x" not in note:
        _fail("the note must state the measured shrink factor")
    # a real upscale keeps VSR
    if fit(1024, 1024, 2048, 2048, "nvidia_rtx_vsr") != ("nvidia_rtx_vsr", None):
        _fail("a genuine upscale must keep nvidia_rtx_vsr untouched")
    # equal size is not a shrink
    if fit(1024, 1024, 1024, 1024, "nvidia_rtx_vsr")[0] != "nvidia_rtx_vsr":
        _fail("an equal-area fit must not be treated as a shrink")
    # v565 AMENDMENT, RE-AMENDED IN v568.
    # v565 coerced bicubic/bilinear to `area` on an involuntary shrink. It was
    # right about the aliasing and WRONG about the cure: `area` is a BOX filter -
    # it antialiases by BLURRING. Measured on a 3.6x supersample downscale of real
    # ESRGAN output, `area` keeps 1.24x LESS detail than a windowed kernel. It
    # blurred away exactly the detail the pixel pass was paid to build.
    # v568: _resize_chunked passes antialias=True (F.interpolate scales the kernel
    # support to the ratio - correct antialiasing AND detail preservation), so the
    # kernels are KEPT and the coercion is gone. Only nearest-exact, which has no
    # antialiased form at all, is still overruled.
    for meth in ("area", "lanczos (cpu)", "bicubic", "bilinear"):
        if fit(4000, 4000, 1000, 1000, meth) != (meth, None):
            _fail(f"{meth!r} must survive a shrink untouched - bicubic/bilinear "
                  f"are antialiased in _resize_chunked since v568, area and "
                  f"lanczos always were")
        if fit(4000, 4000, 1000, 1000, meth, involuntary=True) != (meth, None):
            _fail(f"{meth!r} must survive an INVOLUNTARY shrink too - the box "
                  f"filter cure was worse than the disease")
    for inv, want in ((False, "nearest-exact"), (True, "area")):
        m2, note2 = fit(4000, 4000, 1000, 1000, "nearest-exact", involuntary=inv)
        if m2 != want or not note2 or "alias" not in note2:
            _fail("nearest-exact has no antialiased form: kept+warned on an "
                  "explicit resize, overruled to area on an involuntary one")
    # v568: 'none' is a decision, not a filter - _fit_method must not touch it.
    if fit(4000, 4000, 1000, 1000, "none") != ("none", None):
        _fail("'none' must pass through _fit_method untouched (it is routed to "
              "_crop_to, which does no interpolation at all)")

    # ---- proportion math: extracted and RUN -----------------------------------
    ns2 = {}
    for rx in (r"def _proportion_plan\(.*?\n(?=\ndef _offset)",
               r"def _offset\(.*?\n(?=\ndef _parse_color)",
               r"def _parse_color\(.*?\n(?=\ndef _crop_or_pad)"):
        mm = re.search(rx, fu, re.S)
        if not mm:
            _fail(f"not extractable: {rx}")
        exec(mm.group(0), ns2)  # noqa: S102
    plan, off, color = ns2["_proportion_plan"], ns2["_offset"], ns2["_parse_color"]

    if plan(1920, 1080, 1024, 1024, "stretch") != (1024, 1024):
        _fail("stretch must go straight to the target (the v554 behaviour)")
    if plan(1920, 1080, 1024, 1024, "crop") != (1820, 1024):
        _fail("crop must COVER the target (scale by the larger ratio)")
    if plan(1920, 1080, 1024, 1024, "pad") != (1024, 576):
        _fail("pad must CONTAIN the image (scale by the smaller ratio)")
    if plan(1000, 1000, 500, 500, "crop") != (500, 500):
        _fail("a matching aspect must need no cover scaling")

    if (off(1820, 1024, "left", "x"), off(1820, 1024, "center", "x"),
            off(1820, 1024, "right", "x")) != (0, 398, 796):
        _fail("the crop/pad anchor math broke on x")
    if (off(1024, 576, "top", "y"), off(1024, 576, "center", "y"),
            off(1024, 576, "bottom", "y")) != (0, 224, 448):
        _fail("the crop/pad anchor math broke on y")

    if color("0, 0, 0") != (0.0, 0.0, 0.0):
        _fail("black must parse")
    if color("255,255,255") != (1.0, 1.0, 1.0):
        _fail("white must parse")
    if color("nonsense") != (0.0, 0.0, 0.0):
        _fail("an unparseable colour must fail SOFT to black, never abort")

    # ---- contracts -------------------------------------------------------------
    if 'RETURN_TYPES = ("IMAGE", "VIDEO", "INT", "INT", "MASK")' not in fu:
        _fail("the new outputs must be APPENDED (width/height/mask) so every "
              "existing wire survives")
    if 'RETURN_NAMES = ("image", "video", "width", "height", "mask")' not in fu:
        _fail("the output names changed")
    req = fu[fu.index('"required"'):fu.index('"optional"')]
    names = re.findall(r'"([a-z_0-9]+)":\s*\(', req)
    BIRTH = ["size_mode", "upscale_by", "width", "height", "resize_method",
             "device", "divisible_by", "per_batch", "mute_staging_logs"]
    if names[:len(BIRTH)] != BIRTH:
        _fail("the v554 birth order must keep its indices 0-8")
    if names[len(BIRTH):] != ["keep_proportion", "crop_position", "pad_color"]:
        _fail("the v559 widgets must be APPENDED, in this order")
    if "def _crop_or_pad" not in fu:
        _fail("the crop/pad implementation is gone")
    if "is_mask" not in fu or "canvas[:, y:y + h, x:x + w] = frames" not in fu:
        _fail("a padded MASK border must be 0 (it was never part of the source)")
    if '_fit_method(cur_w, cur_h, rw, rh, resize_method)' not in fu:
        _fail("Fast Upscale must route its fit through the shared guard")
    if "_fit_method(int(image.shape[2]), int(image.shape[1])," not in pu:
        _fail("Power Upscale must route its stage fit through the same guard")
    if "def _fit_method" in fu:
        _fail("the guard must live ONCE, in ph_power_upscale (one source)")

    print("PASS: v559 -- VSR downscale guard executed (Frank's 3.8x case -> "
          "area, loudly), proportion geometry executed (stretch/crop/pad + "
          "anchors + fail-soft colour), outputs appended, one source")
    sys.exit(0)


if __name__ == "__main__":
    main()
