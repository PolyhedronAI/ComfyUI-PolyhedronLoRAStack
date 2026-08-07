"""Guard v568 -- the wires are the truth, and the VAE is the wall.

FRANK RAN THE A/B AND SAW NOTHING. 444 seconds of ESRGAN (both stages,
model+fit), and no visible difference against fit only. Measured why:

  * after the fit, the model's output IS ~5x sharper than a plain resize
    (LapVar 0.071 vs 0.014) - the model GRIPS, it was never broken;
  * but a 4x model in front of a 1.1x stage lands its detail at ~2.5 px,
    and the Wan VAE compresses 8x SPATIALLY - it cannot carry anything
    finer than ~8 px. The refine's encode is a low-pass. What survives it,
    the sampler repaints.

So: BELOW ~2x STAGE FACTOR, A PIXEL MODEL BEFORE THE REFINE CHANGES ALMOST
NOTHING. The place for it is AFTER the last decode, where its detail goes
straight to the file. v560 called this waste, v566 called it supersampling.
Both were half right, and the NOTE now says the measured thing.

FOUR CUTS, PINNED HERE:

1. THE WIRES ARE THE TRUTH. v566 let stage L inherit the H upscale model.
   Right for the MoE experts (one pair, two sigma ranges), poison for the
   pixel model: it ran a 4x ESRGAN over the LARGER L frames (848 -> 3392,
   310 s measured) to build material the fit immediately threw away.
   upscale_model_low or nothing. 'model (high only)' is deleted - it was a
   fourth option duplicating a wiring state.

2. resize_method: 'none' (Frank's third ask - and he was right every time).
   No interpolation anywhere. The canvas becomes what the PIXEL STAGE
   produced: the model's factor with a model wired, the input size without
   one (a pure refine that does not grow). The VAE's /8 grid is taken by
   CROP, never by resampling. In Fast Upscale it is the pure model pass -
   the definitive "do the models grip" test, and the node to hang behind a
   Power Upscale.

3. THE BOX FILTER GOES. v565 coerced bicubic/bilinear to `area` on a shrink.
   Right about the aliasing, WRONG about the cure: `area` antialiases by
   BLURRING - measured 1.24x less detail than a windowed kernel on a 3.6x
   supersample downscale. _resize_chunked now passes antialias=True.

4. fp16 where spandrel says supports_half, with a NaN guard that redoes the
   chunk in fp32, loudly.

Script-style: exit 0 = pass.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v568_wirelaw] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    pu = _read("nodes", "ph_power_upscale.py")
    fu = _read("nodes", "ph_fast_upscale.py")
    js = _read("web", "js", "ph_power_upscale.js")

    # ---- 1: the wires are the truth --------------------------------------------
    if "um = (upscale_model_low if is_low else upscale_model)" not in pu:
        _fail("the pixel model must come from the WIRE, with no inheritance")
    if "upscale_model_low if (is_low and upscale_model_low is not None)" in pu:
        _fail("the pixel-model fallback is back - it sends a 4x ESRGAN over the "
              "LARGER stage-L frames to build material the fit throws away")
    if "model_low if (is_low and model_low is not None) else model" not in pu:
        _fail("the MoE EXPERT fallback must stay - one expert pair, two sigma "
              "ranges. Only the PIXEL model follows the wire")
    # AMENDED IN v569 (1st amendment): the pin said "exactly the three laws".
    # v569 appends 'model final' - the option that puts the model where THIS
    # guard's own docstring says it belongs: after the last decode. A pin that
    # would forbid acting on its own measurement is exactly the kind of pinned
    # belief the amendment ceremony exists for. Tail-append only.
    m = re.search(r'"pixel_stage":\s*\(\[(.*?)\]', pu, re.S)
    opts = re.findall(r'"([^"]+)"', m.group(1)) if m else []
    if opts != ["model + fit", "fit only", "model only", "model final"]:
        _fail(f"pixel_stage must carry exactly the four laws (v569 tail-"
              f"appended 'model final'), got {opts}")
    if "model (high only)" in pu[pu.index("def upscale"):]:
        _fail("'model (high only)' survives in the body - it is a fourth way to "
              "say what an unwired socket already says")

    # ---- 2: resize_method 'none', extracted and RUN -----------------------------
    meth = re.search(r"^_METHODS = \{(.*?)^\}", pu, re.S | re.M)
    if not meth or '"none"' not in meth.group(1):
        _fail("'none' is not in the method table")
    if "_NO_RESIZE" not in pu:
        _fail("the no-resize sentinel is gone")
    if "def _crop_to" not in pu:
        _fail("'none' must route to a CROP, not a filter")

    ns = {"torch": __import__("types").SimpleNamespace()}
    try:
        import torch  # noqa: F401
        have_torch = True
    except Exception:
        have_torch = False
    if have_torch:
        import torch
        ns = {"torch": torch}
        exec(re.search(r"def _crop_to\(.*?\n(?=\n\ndef |\ndef )", pu, re.S).group(0),
             ns)
        crop = ns["_crop_to"]
        a = torch.zeros((2, 3004, 3004, 3))
        out = crop(a, 3000, 3000)
        if tuple(out.shape) != (2, 3000, 3000, 3):
            _fail(f"a /8 remainder must be CROPPED, got {tuple(out.shape)}")
        out = crop(torch.zeros((2, 100, 100, 3)), 104, 104)
        if tuple(out.shape) != (2, 104, 104, 3):
            _fail("a short canvas must be edge-padded, never resampled")
        if not torch.equal(crop(a, 3004, 3004), a):
            _fail("an exact match must be a no-op")

    # _fit_method must let 'none' through untouched (pure, exec'd in isolation)
    fm = re.search(r"def _fit_method\(.*?\n(?=\ndef _chunks)", pu, re.S)
    ns2 = {}
    exec(fm.group(0), ns2)  # noqa: S102
    if ns2["_fit_method"](4000, 4000, 1000, 1000, "none") != ("none", None):
        _fail("'none' must pass through _fit_method untouched")
    if "== _NO_RESIZE" in fm.group(0) or "in _NO_RESIZE" in fm.group(0):
        _fail("_fit_method must not USE a module constant - it is extracted and "
              "exec'd in isolation by three guards, so it has to be pure. (The "
              "comment may NAME it; pin the structure, not the word.)")

    # the canvas is DERIVED, before the grids are planned
    if "_derive = (str(pixel_stage)" not in pu:
        _fail("the canvas derivation is gone")
    if 'str(resize_method) == _NO_RESIZE' not in pu:
        _fail("resize_method='none' must trigger the derivation - no filter "
              "means the pixel stage owns the size")
    der = pu[pu.index("_derive = (str(pixel_stage)"):pu.index("# Plan every stage canvas")]
    if "does NOT grow" not in der:
        _fail("'none' with NO model must mean a pure refine at the input size - "
              "and say so on purpose")
    if pu.index("_derive = (str(pixel_stage)") > pu.index("plans = []"):
        _fail("the derivation must precede the grid plan - the canvas is the "
              "contract with the refine")

    # Fast Upscale: the pure model pass
    if "_NO_RESIZE" not in fu:
        _fail("Fast Upscale never learned 'none'")
    if "are IGNORED" not in fu:
        _fail("'none' must announce that width/height/keep_proportion are dead")
    if "never through a VAE" not in fu:
        _fail("the whole point of the pure model pass must be stated where it "
              "is used - the model's detail only survives AFTER the last decode")

    # ---- 3: the box filter is gone ---------------------------------------------
    rc = pu[pu.index("def _resize_chunked"):pu.index("def _vsr_resize")]
    if "antialias=True" not in rc:
        _fail("a shrink through bicubic/bilinear must be ANTIALIASED, not "
              "coerced to a box filter. F.interpolate does not antialias by "
              "default and common_upscale never asks it to")
    if "shrink and method in (\"bicubic\", \"bilinear\")" not in rc:
        _fail("the antialias must fire on a SHRINK only - upscaling with "
              "antialias=True is wrong (there is nothing to alias)")
    fmb = fm.group(0)
    if "averages the pixels it discards" in fmb:
        _fail("the box-filter coercion is back - it blurred away exactly the "
              "detail the pixel pass was paid to build (1.24x, measured)")
    for meth_name in ("bicubic", "bilinear"):
        if ns2["_fit_method"](4000, 4000, 1000, 1000, meth_name,
                              involuntary=True) != (meth_name, None):
            _fail(f"{meth_name!r} must survive an involuntary shrink now that "
                  f"_resize_chunked antialiases it")
    if ns2["_fit_method"](4000, 4000, 1000, 1000, "nearest-exact",
                          involuntary=True)[0] != "area":
        _fail("nearest-exact has NO antialiased form - it must still be "
              "overruled on an involuntary shrink")

    # ---- 4: fp16 with a NaN guard -----------------------------------------------
    res = pu[pu.index("def _esrgan_resident"):pu.index("def _esrgan_pass")]
    if "supports_half" not in res:
        _fail("fp16 must be gated on what the MODEL claims, never assumed")
    if "torch.autocast" not in res:
        _fail("autocast keeps fp32 master weights - nothing is quantised, only "
              "the matmuls drop")
    if "torch.isfinite(part).all()" not in res:
        _fail("the fp16 NaN guard is gone. A silent NaN frame is worse than a "
              "slow one")
    if "back to fp32 for the whole pass" not in res:
        _fail("the fp16 fallback must be LOUD and must redo the bad chunk")
    if "fp16' if half['on'] else 'fp32'" not in res:
        _fail("the telemetry must say WHICH precision ran (measure > believe)")

    # ---- the NOTE tells the measured physics ------------------------------------
    if "compresses 8x SPATIALLY" not in pu:
        _fail("the NOTE must name the VAE as the wall - that is the measured "
              "reason Frank's 444 s of ESRGAN changed nothing")
    if "thrown away" in pu:
        _fail("the v560 'thrown away' framing is back. It is not thrown away by "
              "the FIT (the fit keeps ~5x the sharpness); it is erased by the "
              "VAE. Naming the wrong culprit sent two versions after the wrong "
              "fix")
    if "AFTER the refine" not in pu:
        _fail("the NOTE must name the place where a pixel model DOES work")

    # ---- serialisation: no new widget -------------------------------------------
    canon = re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1)
    names = re.findall(r'"([a-z_ +()]+)"', canon)
    # AMENDED IN v582 (1st amendment): the absolute count len==25 was a TEXT
    # pin on a moving structure (lesson 1). v582 tail-appends a widget and
    # five sibling guards broke on the same line at once. The claim owned
    # here is historical - "v568 added no widget; the order up to pixel_stage
    # stands" - and a POSITION pin states exactly that, while every future
    # tail-append preserves it.
    if len(names) < 25 or names[24] != "pixel_stage":
        _fail(f"the first 25 canon slots up to pixel_stage are v568's "
              f"history and must stand - got {{len(names)}} entries, "
              f"slot 24 = {{names[24] if len(names) > 24 else 'MISSING'}}")
    if "_healPreV568" in js:
        _fail("a heal step for a cut that changes no widget can only do damage")
    req = pu[pu.index('"required"'):pu.index('"optional"')]
    # AMENDED IN v582 (1st amendment, part 2): '[-1] != "pixel_stage"' was a
    # text pin on the moving tail - it broke in three sibling guards the moment
    # v582 tail-appended a widget, exactly like the len==25 pin above. The
    # zero-maintenance form of the same claim: the LAST required widget must BE
    # the last ORDER_CANON entry. That is the serialisation law itself (python
    # declaration order = canon order), it survives every declared tail-append,
    # and it still fires on the real crime: a widget inserted anywhere but the
    # end, or a canon left un-updated.
    _tail_py = re.findall(r'"([a-z_0-9]+)":\s*\(', req)[-1]
    _tail_js = re.findall(r'"([a-z_ +()]+)"',
                          re.search(r"const ORDER_CANON = \[(.*?)\];", js,
                                    re.S).group(1))[-1]
    if _tail_py != _tail_js:
        _fail(f"the widget tail diverged: INPUT_TYPES ends on '{_tail_py}' but "
              f"ORDER_CANON ends on '{_tail_js}' - the serialisation law says "
              f"they are the same list")
    if "_sanitize" not in js:
        _fail("the v563 net must still be there - it is what heals an old save "
              "carrying the deleted 'model (high only)' value")

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

    print("PASS: v568 -- wires are the truth (no pixel-model inheritance, three "
          "laws), resize_method 'none' (crop not filter, canvas derived, pure "
          "model pass in Fast Upscale), antialiased shrink (the box filter is "
          "gone), fp16 with a NaN guard, and a NOTE that names the VAE")
    sys.exit(0)


if __name__ == "__main__":
    main()
