"""Guard v564 -- the model stays resident, and the either/or is explicit.

MEASURED ON THE LIVE RUN: `esrgan 129f 768x768 -> 3072x3072 -> fit 1536x1536
in 444.8s (3448 ms/frame)`. The pure ESRGAN work is about a second per frame;
the rest was a MODEL PING-PONG that v560's chunking introduced by accident:
Core's ImageUpscaleWithModel does free_memory() + model.to(device) +
model.to("cpu") on EVERY call, and v560 calls it once per chunk. 17 chunks =
17 round trips, each one evicting the DIFFUSION model from VRAM so it has to be
reloaded for the refine.

v564 loads the model ONCE for the whole pass (free once, upload once, fit each
chunk immediately - the v560 fusion stays). Any API drift falls back to the
core node per chunk, loudly.

Plus `pixel_stage` (Frank's switch): "model + fit" (the classic recipe) or
"fit only" - which IGNORES the wired model without unplugging it.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v564_resident] FAIL: " + msg)
    sys.exit(1)


def main():
    pu = open(os.path.join(ROOT, "nodes", "ph_power_upscale.py"),
              encoding="utf-8").read()
    js = open(os.path.join(ROOT, "web", "js", "ph_power_upscale.js"),
              encoding="utf-8").read()

    res = pu[pu.index("def _esrgan_resident"):pu.index("def _esrgan_pass")]
    if "mm.free_memory(need, device)          # ONCE, not per chunk" not in res:
        _fail("free_memory must run ONCE for the pass, never per chunk (it "
              "evicts the diffusion model every time)")
    if res.count("upscale_model.to(device)") != 1:
        _fail("the model must be uploaded exactly ONCE")
    if 'upscale_model.to("cpu")' not in res or "finally:" not in res:
        _fail("the model must always come back down (try/finally)")
    if "_MODEL_UPSCALER.upscale" in res:
        _fail("the resident path must NOT go through the core node - that is "
              "the round trip it exists to avoid")
    if "cu.tiled_scale" not in res:
        _fail("the core tiled_scale util must still do the actual work")
    if "_fit_to(part, width, height" not in res:
        _fail("the v560 fusion (fit each chunk immediately) must survive")

    body = pu[pu.index("def _esrgan_pass"):pu.index("def _refine_tiles")]
    if "_esrgan_resident(image, upscale_model" not in body:
        _fail("the pass no longer uses the resident path")
    if "falling back to the core node per chunk" not in body:
        _fail("API drift must fall back LOUDLY, never abort the run")
    if "model {path}" not in body:
        _fail("the telemetry must say WHICH path ran (measure > believe)")

    req = pu[pu.index('"required"'):pu.index('"optional"')]
    names = re.findall(r'"([a-z_0-9]+)":\s*\(', req)
    # AMENDED IN v582 (1st amendment): 'names[-1] != "pixel_stage"' pinned the
    # moving tail - the fifth copy of the same text pin caught in this cut.
    # First replacement attempt pinned "slot 24" and was WRONG within the same
    # cut: this regex also matches the five required SOCKETS (model, positive,
    # negative, vae, upscale_model), so absolute indices here count things that
    # are not widgets. v564's claim, stated with this block's own means, is a
    # NEIGHBOUR pin: pixel_stage was appended directly after vae_tiling (v562,
    # the then-tail). That survives every later tail-append and still fires on
    # the real crime - something wedged between the two.
    if ("pixel_stage" not in names or "vae_tiling" not in names
            or names.index("pixel_stage") != names.index("vae_tiling") + 1):
        _fail("pixel_stage must sit directly after vae_tiling in required "
              "(v564 appended it onto the v562 tail)")
    if '"default": "model + fit"' not in req:
        _fail("the default must stay the historic recipe (model + fit)")
    # AMENDED IN v569 (1st amendment): the pin held the VERBATIM v564 line and
    # v569 extended the null-set - 'model final' also runs its stages without
    # a pixel model (the wire serves the pass BEHIND the last decode instead).
    # The pinned BELIEF ('fit only' ignores the wired model) is not refuted;
    # the pin now checks the semantics instead of the spelling. Text pins are
    # fragile - pin structure. (This lesson's fifth landing this arc.)
    if ('if str(pixel_stage) in ("fit only", "model final") and um is not None:'
            not in pu):
        _fail("'fit only' (and v569's 'model final') must IGNORE the wired "
              "model in the stages (that is the switch)")
    if "pixel={'fit only'" not in pu:
        _fail("the BEGIN line must state what the pixel stage did")

    # The exact banner version is pinned by the file's NEWEST guard (v565);
    # existence + format is pinned by test_v548. No banner pin here.
    canon = re.findall(r'"([a-z_ +()]+)"',
                       re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1))
    if canon.index("pixel_stage") != 24:
        _fail("pixel_stage must sit at canon index 24 (appended)")
    parts = [re.search(rx, js) for rx in (
        r"const LEN_PRE_V564 = \d+;",
        r"function _healPreV564\(wv\) \{[\s\S]*?\n\}")]
    if not all(parts):
        _fail("the v564 heal is not extractable")
    harness = "\n".join(m.group(0) for m in parts) + r"""
const a = _healPreV564(Array.from({length: 24}, (_, i) => i));
if (a.length !== 25 || a[24] !== "model + fit") {
    console.error("FAIL 24->25"); process.exit(1);
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

    print("PASS: v564 -- model resident (one free, one upload, one download), "
          "fusion kept, loud fallback, pixel_stage switch, heal 24->25")
    sys.exit(0)


if __name__ == "__main__":
    main()
