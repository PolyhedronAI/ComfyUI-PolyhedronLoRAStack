"""Guard v566 -- the pixel-stage laws, the paid-for VRAM, and warnings in numbers.

THREE MEASURED WOUNDS, ONE CUT:

1. THE MISCALIBRATED ASK (my own v565 bug). v565 grew the ESRGAN tile to 1024
   but left comfy's activation constant CALIBRATED TO 512 - free_memory cleared
   room for a 512 forward while the 1024 forward wanted ~4x. On Windows that is
   not an OOM: the driver spills to system RAM over PCIe and GRINDS. Measured as
   the per-pixel curve 3.2 / 5.2 / 7.3 us/px over tiles 512 / 768 / 1024 - same
   model, same card. v566 calibrates the ask to the REAL tile, chooses the tile
   against MEASURED free VRAM, and prints the proof line (chunk-1 peak vs free).

2. THE SILENT FALLBACK TRAP. upscale_model_low falls back to the H model - right
   for the MoE experts, poison for the pixel model: it silently sent a 4x ESRGAN
   over the LARGER stage-L frames (16x the H pixel work, ~34 projected minutes)
   to make material the fit immediately halved. Frank's own pre-node workflow
   proves the law: detail ONCE, at the bottom; the second stage grows by filter
   and the refine paints. v566: pixel_stage gains 'model only' (the canvas IS
   the model factor). v568 goes further and lets the WIRES decide - see
   test_v568.

3. THE WARNING WITHOUT NUMBERS. tile 1072 on a 1104 canvas: 32 px past the tile
   edge tipped the grid 1 -> 4 tiles at coverage 3.8x. The old warning offered
   only 'one big tile' - which the s/step measurements show is often SLOWER
   (attention is quadratic in tokens). v566 computes the snug tile and the
   same-cost canvas edge: a 1072 tile at 2x2 carries up to 2080, so a 1104
   canvas pays exactly what a 2080 canvas would.

Script-style: exit 0 = pass.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v566_pixellaw] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    pu = _read("nodes", "ph_power_upscale.py")
    js = _read("web", "js", "ph_power_upscale.js")

    res = pu[pu.index("def _esrgan_resident"):pu.index("def _esrgan_pass")]

    # ---- 1a: the ask is calibrated to the REAL tile ---------------------------
    if "(tile * tile * 3) * image.element_size()" not in res:
        _fail("the free_memory ask must scale with the REAL tile. The 512 "
              "constant cleared room for a 512 forward while the 1024 forward "
              "wanted ~4x - on Windows that is a silent spill to system RAM, "
              "not an OOM, and it GRINDS")
    if "(512 * 512 * 3)" in res:
        _fail("comfy's 512 activation constant is back in the resident path - "
              "that is the exact miscalibration v566 removes")
    if res.index("tile = max(128") > res.index("need = mm.module_size"):
        _fail("the tile must be chosen BEFORE the memory ask - the ask depends "
              "on it")

    # ---- 1b: the VRAM instrument ----------------------------------------------
    if "torch.cuda.mem_get_info" not in res:
        _fail("the free-VRAM measurement is gone - without it the tile choice "
              "and the spill proof are both blind")
    if "vram free" not in res or "after free_memory" not in res:
        _fail("the vram telemetry line is gone (free/total after free_memory, "
              "the ask, the activation estimate)")
    if "reset_peak_memory_stats" not in res or "max_memory_allocated" not in res:
        _fail("the chunk-1 peak measurement is gone - peak vs free-before is "
              "the spill PROOF, not a guess")
    # AMENDED IN v573 (4th amendment of this guard): the pinned sentence
    # ("SPILLED to system RAM") was the v566/v570 verdict text - and under
    # v572's pool quiet it LIED: free reads ~0 because the POOL holds our
    # blocks, and the line called a healthy 22.5 s chunk spilled, costing a
    # needless backoff. The BELIEF stands - one line must say whether the
    # grinding is real - but the judge is now the pure _watch_verdict (the
    # wall clock), exec'd by test_v573 with all three runs' numbers.
    if "_watch_verdict(dt_ck, t_c1, peak, free_i)" not in res:
        _fail("the spill verdict line is gone - the whole point is one line "
              "that says whether the grinding is real")
    if "* 64 * 4 * 2" not in res:
        _fail("the upsample-map activation estimate is gone. (tile*scale)^2 * "
              "64ch * fp32 * in+out separated the clean 768 run from the "
              "grinding 1024 run on the same 16 GB card - it earned the job")
    if "tile = max(256, tile // 2)" not in res:
        _fail("the up-front VRAM-aware tile halving is gone - waiting for the "
              "spill is exactly what v565 did")

    # ---- 1c: the interrupt lives INSIDE the model fn --------------------------
    if "def _model_fn(a):" not in res:
        _fail("the model fn wrapper is gone")
    fn = res[res.index("def _model_fn(a):"):res.index("out, mid = [], None")]
    if "throw_exception_if_processing_interrupted()" not in fn:
        _fail("the interrupt must live INSIDE the model call - a 92 s chunk "
              "answered six clicks with silence; per tiled_scale call the "
              "answer comes in seconds")
    if "upscale_model(a.float())" not in fn:
        _fail("the fp32 template parity left the model fn")
    if res.count("throw_exception_if_processing_interrupted()") < 2:
        _fail("the chunk-boundary interrupt must stay ALONGSIDE the in-fn one")

    # ---- v565 must survive intact (this cut sits ON it) -----------------------
    if "output_device=device" not in res:
        _fail("the v565 output_device fix is gone - that is the 3358 ms/frame")
    if "mm.free_memory(need, device)          # ONCE, not per chunk" not in res:
        _fail("v564's one-free-per-pass is gone")
    if "mm.raise_non_oom(exc)" not in res or "tile //= 2" not in res:
        _fail("core's OOM backoff net is gone")

    # ---- 2a: the option list --------------------------------------------------
    m = re.search(r'"pixel_stage":\s*\(\[(.*?)\]', pu, re.S)
    if not m:
        _fail("pixel_stage widget not found")
    opts = re.findall(r'"([^"]+)"', m.group(1))
    # AMENDED IN v568: 'model (high only)' is GONE. It existed to skip the L
    # pixel model - which is free now that the WIRES decide (leave
    # upscale_model_low unwired and stage L runs a plain fit). A fourth option
    # that duplicates a wiring state is a fourth way to be wrong.
    # AMENDED IN v569 (2nd amendment of this block): 'model final' is ADDED,
    # appended at the tail. It is NOT a duplicated wiring state - it moves the
    # pass to the one place the v568 measurement says it works: BEHIND the
    # last decode, where no VAE erases it. The list may only ever grow at its
    # tail (the serialisation law, applied to value lists).
    if opts != ["model + fit", "fit only", "model only", "model final"]:
        _fail(f"pixel_stage options must be the four laws IN ORDER (default "
              f"first, later additions at the tail), got {opts}")

    # ---- 2b: 'model only' - the canvas IS the model factor --------------------
    if 'str(pixel_stage) == "model only"' not in pu:
        _fail("the 'model only' law is gone")
    # v568: the derivation now has TWO triggers (model only OR resize_method
    # none), so the block is keyed on _derive.
    mo = pu[pu.index("_derive = (str(pixel_stage)"):
            pu.index("# Plan every stage canvas")]
    if 'getattr(um_st, "scale"' not in mo:
        _fail("'model only' must read the factor from the WIRED MODEL - that "
              "is the whole meaning of the option")
    if "ignoring upscale_by" not in mo:
        _fail("ignoring upscale_by must be ANNOUNCED, never silent")
    if "falling" not in mo or "back to upscale_by" not in mo:
        _fail("no model wired must fall back to upscale_by, loudly")
    if pu.index("_derive = (str(pixel_stage)") > pu.index("plans = []"):
        _fail("the factor swap must happen BEFORE the grids are planned - the "
              "canvas is the contract with the refine")

    # ---- 2c: AMENDED IN v568 - the wires are the truth -------------------------
    if "um = (upscale_model_low if is_low else upscale_model)" not in pu:
        _fail("the pixel model must come from the WIRE, with no inheritance. "
              "v566 let stage L inherit the H upscaler, which sent a 4x ESRGAN "
              "over the LARGER L frames (848 -> 3392, 310 s measured) to build "
              "material the fit threw away")
    if "no upscale_model_low" not in pu:
        _fail("an unwired L pixel model must be ANNOUNCED, not just silently "
              "correct")
    # AMENDED IN v569 (3rd amendment of this guard): the verbatim v564 line
    # grew - 'model final' nulls the stage model too (the wire moved behind
    # the last decode). Same amendment as in test_v564_resident, same reason:
    # the belief stands, the spelling changed. Pin structure, not text.
    if ('if str(pixel_stage) in ("fit only", "model final") and um is not None:'
            not in pu):
        _fail("the v564 fit-only law (extended by v569's 'model final') must "
              "survive (test_v564 pins it too)")

    # ---- 3: the warning speaks in numbers - extracted and RUN -----------------
    m = re.search(r"def _grid_advice\(.*?\n(?=\ndef _refine_tiles)", pu, re.S)
    if not m:
        _fail("_grid_advice not extractable")
    ns = {}
    exec(m.group(0), ns)  # noqa: S102 - our own source, measured
    advice = ns["_grid_advice"]

    # Frank's live case: tile 1072 on a 1104x1104 canvas, 2x2 grid, overlap 64.
    snug, ew, eh = advice(1104, 1104, 1072, 64, 2, 2)
    if snug != 584:
        _fail(f"snug tile for the 1104/2x2 case must be 584 (ceil((1104+64)/2), "
              f"/8), got {snug}")
    if (ew, eh) != (2080, 2080):
        _fail(f"the same-cost canvas edge for a 1072 tile at 2x2 must be 2080 "
              f"(2*1072-64) - the '1104 pays what 2080 would' number - got "
              f"{(ew, eh)}")
    # a single-tile axis needs no snug shrink
    if advice(848, 848, 1024, 64, 1, 1)[0] != 848:
        _fail("a 1x1 grid must return the canvas itself as snug")
    # 3x3 on 3072 (the stage-L target grid)
    s3, e3, _ = advice(3072, 3072, 1072, 64, 3, 3)
    if not (1064 <= s3 <= 1072):
        _fail(f"snug for 3072/3x3 must land at ~1067 -> /8 = 1072-ish, got {s3}")
    if e3 != 3088:
        _fail(f"edge for 1072@3x3 must be 3088 (3*1072-2*64), got {e3}")

    if "Two clean choices" not in pu or "keeps the" not in pu:
        _fail("the warning no longer offers BOTH clean choices (one tile / "
              "snug tile)")
    if "would use ONE tile" not in pu:
        _fail("the v560 pin phrase left the warning")
    if "pays the full next grid" not in pu:
        _fail("the canvas-edge lesson (32 px past the edge = the next grid) "
              "left the warning")

    # ---- serialisation: this cut adds NO widget --------------------------------
    canon = re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1)
    names = re.findall(r'"([a-z_ +()]+)"', canon)
    # AMENDED IN v582 (1st amendment): the absolute count len==25 was a TEXT
    # pin on a moving structure (lesson 1). v582 tail-appends a widget and
    # five sibling guards broke on the same line at once. The claim owned
    # here is historical - "v566 added no widget; the order up to pixel_stage
    # stands" - and a POSITION pin states exactly that, while every future
    # tail-append preserves it.
    if len(names) < 25 or names[24] != "pixel_stage":
        _fail(f"the first 25 canon slots up to pixel_stage are v566's "
              f"history and must stand - got {{len(names)}} entries, "
              f"slot 24 = {{names[24] if len(names) > 24 else 'MISSING'}}")
    if "_healPreV566" in js:
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
    if '"default": "model + fit"' not in req:
        _fail("the default must stay the historic recipe")

    # the heal cascade still terminates at v564 + the nets (unchanged JS)
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

    print("PASS: v566 -- ask calibrated to the real tile, VRAM instrument with "
          "spill proof, interrupt inside the model fn, four pixel-stage laws, "
          "loud fallback, grid advice computed against the live 1104/2080 case, "
          "25 canon entries")
    sys.exit(0)


if __name__ == "__main__":
    main()
