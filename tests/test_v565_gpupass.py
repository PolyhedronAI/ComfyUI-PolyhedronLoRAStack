"""Guard v565 -- the pixel stage runs on the GPU, and it is never mute again.

THE BUG WAS ONE MISSING KEYWORD ARGUMENT.

    comfy/utils.py:1225
    def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8,
                    upscale_amount=4, out_channels=3,
                    output_device="cpu",          # <-- the default
                    pbar=None):

Core's own ImageUpscaleWithModel passes output_device explicitly. v564 passed
neither it nor pbar. So `output` was allocated on the CPU, every ESRGAN tile was
copied GPU->CPU, and the mask build, the `o.add_(ps_view * mask_view)`, the
`out.div_(out_div)` and our own clamp all ran on the CPU over 3072x3072 buffers -
after which the result was uploaded BACK to the GPU for the fit.

MEASURED on Frank's 65-frame run (768 -> 3072 -> fit 1536):
    ~128 M CPU ops per frame -> 8.3 G CPU ops
    ~18.3 GB over the PCIe bus, of which ~8.2 GB was a pure return trip
    218.3 s = 3358 ms/frame

v564's "model resident" fixed the wrong ping-pong: it moved the model WEIGHTS
once instead of 17 times, worth 89 ms/frame (3448 -> 3359 = 2.6 %). The TENSORS
ping-ponged on every single frame. This guard pins the actual fix.

And the SECOND half of the same wound: the pixel stage contributed 0 to the
progress bar, sent no preview event and checked no interrupt. 218 seconds of a
node that looked hung while it was working. Frank's complaint was "the preview
doesn't come"; the truth was "there is nothing to preview yet, and nobody says
so". v565 gives the pixel stage a budget in the bar, a line per chunk, a probe
door of its own, and an escape hatch.

Script-style: exit 0 = pass.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v565_gpupass] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    pu = _read("nodes", "ph_power_upscale.py")
    js = _read("web", "js", "ph_power_upscale.js")

    res = pu[pu.index("def _esrgan_resident"):pu.index("def _esrgan_pass")]

    # ---- A: the pass stays on the GPU ---------------------------------------
    if "output_device=device" not in res:
        _fail("tiled_scale must be told WHERE to put its output. Without it the "
              "default is output_device='cpu' and the blend, the div_ and the "
              "clamp all run on the CPU over 3072x3072 buffers - that IS the "
              "3358 ms/frame")
    if re.search(r"cu\.tiled_scale\([^)]*tile_x=512", res, re.S):
        _fail("the hardcoded 512 tile is back. A tile that covers the frame "
              "makes comfy take its own single-tile branch: no out_div, no mask, "
              "no div_() - and one model call per frame instead of four")
    if "_ESRGAN_TILE_CAP" not in pu:
        _fail("the first-ask tile size is gone")
    if "mm.raise_non_oom(exc)" not in res or "tile //= 2" not in res:
        _fail("core's OOM backoff must sit behind the bigger tile - an over-large "
              "ask may cost one retry, never a run")
    if "upscale_model(a.float())" not in res:
        _fail("core casts the tile to float before the model; we must too "
              "(template parity - a silent fp16 divergence otherwise)")
    if "mm.throw_exception_if_processing_interrupted()" not in res:
        _fail("the pixel pass must be CANCELLABLE. 218 uninterruptible seconds "
              "is indistinguishable from a hang, and that is what it was")
    if "* 3 * 4" not in res or "the output buffer now lives on the GPU" not in res:
        _fail("the GPU output buffer must be PAID FOR in free_memory (it is "
              "scale^2 times the input chunk - 16x at 4x). Wishing it away is "
              "how you OOM")
    if "involuntary=True" not in res:
        _fail("the fit that follows the model is an INVOLUNTARY shrink and must "
              "be marked as one")

    # ---- v564 must survive intact (this cut sits ON it, not over it) --------
    if "mm.free_memory(need, device)          # ONCE, not per chunk" not in res:
        _fail("v564's one-free-per-pass is gone")
    if res.count("upscale_model.to(device)") != 1:
        _fail("v564's one-upload-per-pass is gone")
    if 'upscale_model.to("cpu")' not in res or "finally:" not in res:
        _fail("v564's guaranteed download is gone")

    # ---- B: the pixel stage is never mute again -----------------------------
    body = pu[pu.index("def _esrgan_pass"):pu.index("def _refine_tiles")]
    if "on_chunk=None" not in body:
        _fail("_esrgan_pass lost its on_chunk hook")
    if body.count("on_chunk(i, j,") < 2:
        _fail("BOTH pixel paths (fit-only and the core fallback) must report "
              "every finished chunk - a fast pass that LOOKS frozen is still bad")
    if "on_chunk=on_chunk" not in body:
        _fail("the resident path is no longer handed the hook")
    if "def _make_pixel_probe" not in pu:
        _fail("the pixel probe door is gone - the v550 probe hangs off the "
              "SAMPLER callback and cannot fire before the refine starts")
    if '"stage": "pixel"' not in pu:
        _fail("the pixel probe must tag its stage, or the pane cannot tell it "
              "apart from a refine tile")
    if "pixel view armed" not in pu or "pixel view: first frame sent" not in pu:
        _fail("the pixel door must prove itself like every other probe (v552)")
    if "pixel view disarmed" not in pu:
        _fail("the pixel door must disarm on first failure - a preview may never "
              "cost a render")
    # Pin the SHAPE, not a word - the docstring is allowed to name what the code
    # deliberately does not do. (This guard first failed on its own prose.)
    pix = pu[pu.index("def _make_pixel_probe"):pu.index("def _resolve_input")]
    if "def _make_pixel_probe():" not in pix:
        _fail("the pixel probe must take NO model - pixel frames are already RGB, "
              "and a door that needs a model is a door that can be shut by one")
    if "einsum" in pix or "model.model" in pix:
        _fail("a latent2rgb conversion crept into the pixel door - those frames "
              "are finished RGB, converting them is decoding an image twice")
    # AMENDED IN v567. This guard used to pin the v565 TICK budget
    # ("+ len(plans) * pix_chunks" in the bar total). Ticks were the right fix
    # for "the bar never moves" and the wrong unit for "the bar tells the
    # truth": 18 chunks at ~62 ms and 8 steps at 43-93 s counted as equals put
    # the bar at 35% after half a second (measured on the 11:38 run). v567
    # replaces ticks with a TIME bar (_RunClock); the pixel stage keeps its
    # budget - as a clock post. The bar-truth law is pinned by test_v567.
    # AMENDED IN v570 (3rd amendment): the pin held the SPELLING
    # clock.post(f"pix: and v570 taught the post its KIND - a fit-only stage
    # posts 'fit:<tag>', a model stage 'pix:<tag>', so the final model pass
    # can never again rate-borrow from a 63 ms fit (the measured ETA fairy
    # tale of Frank's v569 run). The belief - the pixel stage IS a clock
    # post - stands. Pin structure, not letters (sixth landing).
    if "clock.post(f\"{st['pix_kind']}:{st['tag']}\"" not in pu:
        _fail("the pixel stage lost its budget - it must be a clock post now "
              "(v567), keyed by its OPERATION kind (v570), or a 218 s pass "
              "leaves the bar at zero (v564) / the ETA borrows across kinds "
              "(v569)")
    if "pixel chunk " not in pu or "eta " not in pu:
        _fail("the per-chunk console line (count + s/chunk + eta) is gone - that "
              "is the live narration Frank asked for")

    # ---- E/F/G: the instruments (added after the interrupted run) ------------
    # Frank's second log: stage=high 1259.9 s, of which the pixel pass was 218.3 s.
    # So 83 % of the stage happened inside _refine_tiles - and the done line
    # divided the WHOLE stage by the tile count, reporting "315.0s/tile" for a
    # per-tile cost that is really 260.4 s. A number that folds two costs together
    # is worse than no number: it sent v564 after the wrong ping-pong.
    if "pixel {pix_dur:.1f}s + refine {ref_dur:.1f}s" not in pu:
        _fail("the done line must separate the pixel pass from the refine - one "
              "number for two costs is how you optimise the wrong one")
    if "encode {enc:.1f}s + sample {smp:.1f}s" not in pu:
        _fail("the per-tile phase breakdown is gone. 1041.6 s went somewhere "
              "inside that loop and NOBODY measured where. Guessing which quarter "
              "to cut is exactly the v564 mistake, repeated")
    if "s/step" not in pu:
        _fail("the per-step cost is the one number that lets the next stage be "
              "projected before it is started")
    ref = pu[pu.index("def _refine_tiles"):pu.index("# \u2500\u2500 v549")]
    if ref.count("_phases()") != 2:
        _fail("BOTH exits of the tile loop (the single-tile fast path and the "
              "blended path) must report their phases - a fast path that reports "
              "nothing is a fast path nobody can check")
    if "_ESRGAN_OUT_BUDGET" not in pu:
        _fail("the pixel chunk must come from a VRAM budget, not from the widget. "
              "v565 moved the output buffer onto the GPU; at 4x on a 1536 frame "
              "that is 453 MB PER FRAME, so per_batch=8 asks torch.empty() for "
              "3.6 GB in one allocation - and no tile backoff can rescue that")
    if "on_chunk(i, j, fit, chunks)" not in pu:
        _fail("the hook must carry the TRUE chunk count - the VRAM clamp changes "
              "it, and a progress bar told the planned number saturates early and "
              "then sits there looking finished")
    # (v567: the count now corrects a clock post - clock.resize - instead of a
    #  bar-total dict; the assertion above pins the CONTRACT, not the consumer.)

    # ---- C: the corrected fit law, extracted and RUN ------------------------
    m = re.search(r"def _fit_method\(.*?\n(?=\ndef _chunks)", pu, re.S)
    if not m:
        _fail("_fit_method not extractable")
    ns = {}
    exec(m.group(0), ns)  # noqa: S102 - our own source, measured
    fit = ns["_fit_method"]

    # AMENDED IN v568 - the second time this block changes, and the reason is the
    # same discipline both times: it pinned a CURE, not the disease.
    # v565 saw the aliasing correctly (F.interpolate does not antialias, comfy's
    # common_upscale never asks it to) and reached for `area`. But `area` is a BOX
    # filter - it antialiases by BLURRING. Measured on a 3.6x supersample downscale
    # of real ESRGAN output: `area` keeps 1.24x LESS detail than a windowed kernel.
    # It blurred away exactly the detail the pixel pass was paid to build, in the
    # very case this assertion was written for. v568 passes antialias=True in
    # _resize_chunked instead: no aliasing AND no blur, so the kernel is KEPT.
    # The aliasing law itself survives - see test_v568, and nearest-exact (which
    # has no antialiased form) is still overruled.
    meth, note = fit(3072, 3072, 1536, 1536, "bicubic", involuntary=True)
    if meth != "bicubic" or note is not None:
        _fail(f"bicubic must now SURVIVE Frank's exact live case (3072->1536, "
              f"involuntary) - _resize_chunked antialiases it. Got {meth!r}")
    if fit(3072, 3072, 1536, 1536, "nearest-exact", involuntary=True)[0] != "area":
        _fail("nearest-exact has no antialiased form and must still be overruled")
    # An upscale is never touched, whoever asks.
    for inv in (False, True):
        if fit(768, 768, 1536, 1536, "bicubic", involuntary=inv) != ("bicubic", None):
            _fail("a genuine UPSCALE must pass through untouched")
    # VSR: the v559 law, untouched.
    if fit(4296, 5364, 1128, 1408, "nvidia_rtx_vsr")[0] != "area":
        _fail("the v559 VSR law must survive this cut")

    # ---- frontend -----------------------------------------------------------
    # The exact banner version is pinned by the file's NEWEST guard (v567);
    # existence + format is pinned by test_v548. No banner pin here.
    mark = re.search(r"const STAGE_MARK = \{(.*?)\};", js, re.S)
    if not mark:
        _fail("the HUD stage map is gone")
    for k in ("high", "low", "pixel"):
        if k + ":" not in mark.group(1):
            _fail(f"the HUD stage map lost {k!r}")
    if 'd.stage === "high" ? "H" : "L"' in js:
        _fail("the old two-way ternary is back - it labels every pixel frame "
              '"L", which is a confident lie')
    if 'STAGE_MARK[d.stage] || "?"' not in js:
        _fail("a stage the pane does not know must render as '?' - useless is "
              "fine, wrong is not")
    if "Chunk " not in js:
        _fail("the pixel HUD must count CHUNKS, not tiles")

    # ---- serialisation: this cut adds NO widget -----------------------------
    canon = re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1)
    names = re.findall(r'"([a-z_ +()]+)"', canon)
    # AMENDED IN v582 (1st amendment): the absolute count len==25 was a TEXT
    # pin on a moving structure (lesson 1). v582 tail-appends a widget and
    # five sibling guards broke on the same line at once. The claim owned
    # here is historical - "v565 added no widget; the order up to pixel_stage
    # stands" - and a POSITION pin states exactly that, while every future
    # tail-append preserves it.
    if len(names) < 25 or names[24] != "pixel_stage":
        _fail(f"the first 25 canon slots up to pixel_stage are v565's "
              f"history and must stand - got {{len(names)}} entries, "
              f"slot 24 = {{names[24] if len(names) > 24 else 'MISSING'}}")
    if "_healPreV565" in js:
        _fail("a heal step for a cut that changes no widget is a heal step that "
              "can only do damage")
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

    # the heal cascade still terminates at v564 + the two nets
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
if (_padToCanon([1, 2, 3]).length !== N) {
    console.error("FAIL padToCanon net"); process.exit(1);
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

    print("PASS: v565 -- pixel pass on the GPU (output_device + frame-sized tile "
          "+ OOM backoff + interrupt), pixel stage visible (budget, console, "
          "probe door), ringing kernels caught on a shrink, 25 canon entries")
    sys.exit(0)


if __name__ == "__main__":
    main()
