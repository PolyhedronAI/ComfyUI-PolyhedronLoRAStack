"""Guard v560 -- the two real time/VRAM killers, pinned.

MEASURED ON THE LIVE RUN (129 frames, 4x model, 848x848 stage):

  1. THE 6-MINUTE STALL / OOM. `_esrgan_pass` upscaled the WHOLE batch first
     and only then fit it down: 129 frames x 3084x3084 x 3 x float32 =
     ~14.7 GB as ONE tensor - per stage. v560 FUSES the upscale and the fit
     per chunk, so the peak holds `per_batch` frames at the intermediate size
     (~0.9 GB at 8). Both operations are per-frame, so the result is
     unchanged. Pinned: the fused loop, no whole-batch upscale call, the
     telemetry line (frames, sizes, duration, ms/frame, peak GB), and the
     note when the model produces far more pixels than the stage needs.

  2. THE TILE COVERAGE TRAP. A tile_size just below the canvas produces
     near-identical tiles: 848 canvas + 768 tile => offsets 0 and 80 => 2x2
     tiles that sample 3.3x the area for the SAME result. v560 prints the
     coverage in the BEGIN line and warns loudly above 1.6x, naming the
     tile_size that would use ONE tile.

Plus: `per_batch` appended LAST with the heal 22 -> 23 measured in node, and
the CLIP encoder's classic conditioning colours (green positive / brown
negative) applied to the textareas.

Script-style: exit 0 = pass.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v560_oom_coverage] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    pu = _read("nodes", "ph_power_upscale.py")
    js = _read("web", "js", "ph_power_upscale.js")
    cjs = _read("web", "js", "ph_clip_encode.js")

    # ---- 1) the fused ESRGAN pass --------------------------------------------
    body = pu[pu.index("def _esrgan_pass"):pu.index("def _refine_tiles")]
    if "for i, j in _chunks(n, per_batch):" not in body:
        _fail("the ESRGAN pass must run in CHUNKS (a 129-frame 4x intermediate "
              "is ~14.7 GB in one go - that was the stall and the OOM)")
    if "_MODEL_UPSCALER.upscale(upscale_model, image[i:j])" not in body:
        _fail("the upscaler must be fed chunk by chunk, never the whole batch")
    if "_MODEL_UPSCALER.upscale(upscale_model, image)" in body:
        _fail("the whole-batch upscale call is back - that is the OOM")
    if "_fit_to(part, width, height" not in body:
        _fail("each chunk must be FIT IMMEDIATELY (fused), so the 4x "
              "intermediate never exists for the whole batch")
    if "del part" not in body:
        _fail("the chunk must be released before the next one")
    if "esrgan {n}f" not in body or "peak ~" not in body:
        _fail("the telemetry must state frames, sizes, duration and the peak "
              "(measure > believe - this is how the stall was found)")
    if "ms/frame" not in body:
        _fail("the per-frame cost must be visible")
    # AMENDED IN v568. This pinned the phrase "pixels than this stage needs" and
    # with it v560's THEORY: that an overshooting model is wasted WORK. Measured:
    # the fit keeps ~5x the sharpness (the model grips) - and then the Wan VAE,
    # which compresses 8x spatially, erases detail finer than ~8px in the refine's
    # encode. The waste is real but it is not the fit's doing, and naming the
    # wrong culprit sent v566 chasing "supersampling". The note stays; it now
    # names the VAE. The physics is pinned in full by test_v568.
    if "more pixels than this" not in body or "stage needs" not in body:
        _fail("the wasted-work note is gone (4x model in front of a 1.1x "
              "stage throws ~96% of the pixels away)")
    if "def _fit_to" not in pu:
        _fail("the shared fit helper is gone")

    # ---- 2) the coverage warning ---------------------------------------------
    if 'coverage={cov:.1f}x' not in pu:
        _fail("the BEGIN line must state the coverage")
    if "WARNING tile_size=" not in pu:
        _fail("a coverage above the threshold must WARN loudly")
    if "would use ONE tile" not in pu:
        _fail("the warning must name the fix (tile_size >= canvas)")
    if "cov > 1.6" not in pu:
        _fail("the coverage threshold is gone")
    # the maths itself, measured on the live case
    tiles, tw, th, sw, sh = 4, 768, 768, 848, 848
    cov = tiles * tw * th / float(sw * sh)
    if not (3.2 < cov < 3.4):
        _fail(f"the live case must measure ~3.3x, got {cov:.2f}")

    # ---- 3) per_batch widget + heal -------------------------------------------
    req = pu[pu.index('"required"'):pu.index('"optional"')]
    names = re.findall(r'"([a-z_0-9]+)":\s*\(', req)
    # v562 hardening: position pin, not tail pin (the recurring lesson - later
    # cuts append behind it; only the INDEX must never move).
    if names.index("per_batch") != names.index("resize_method") + 1:
        _fail("per_batch must sit directly after resize_method (appended in v560)")
    # The exact banner version is pinned by the file's NEWEST guard (v562).
    canon = re.findall(r'"([a-z_ ()]+)"',
                       re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1))
    if canon.index("per_batch") != 22:
        _fail("per_batch must sit at canon index 22 (appended)")
    parts = [re.search(rx, js) for rx in (
        r"const LEN_PRE_V560 = \d+;",
        r"function _healPreV560\(wv\) \{[\s\S]*?\n\}")]
    if not all(parts):
        _fail("the v560 heal is not extractable")
    harness = "\n".join(m.group(0) for m in parts) + "\n" + r"""
const v555 = Array.from({length: 22}, (_, i) => i);
const a = _healPreV560(v555.slice());
if (a.length !== 23 || a[22] !== 8) {
    console.error("FAIL 22->23: " + JSON.stringify(a.slice(20))); process.exit(1);
}
const c23 = Array.from({length: 23}, (_, i) => i);
if (_healPreV560(c23.slice()).join(",") !== c23.join(",")) {
    console.error("FAIL: a v560 save must pass through untouched"); process.exit(1);
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

    # ---- 4) the classic conditioning colours ----------------------------------
    if "POS_TINT" not in cjs or "NEG_TINT" not in cjs:
        _fail("the classic conditioning colours are gone")
    if "_applyTints" not in cjs:
        _fail("the tints are never applied")
    if "[PLS] ph_clip_encode.js v560 loaded" not in cjs:
        _fail("the CLIP encoder banner must carry v560 (the file was touched)")

    print("PASS: v560 -- ESRGAN fused per chunk (14.7 GB -> ~0.9 GB peak) with "
          "telemetry, coverage warning (the 3.3x trap), per_batch heal 22->23 "
          "measured, conditioning colours")
    sys.exit(0)


if __name__ == "__main__":
    main()
