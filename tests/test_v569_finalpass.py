"""Guard v569 -- the final pass: the model becomes a refiner.

THE JOB WAS NEVER THE UPSCALE. Frank's factors are 1.10/1.30 - the node is a
REFINER (eyes, buttons, hair). v568 measured where a pixel model can help
with that job: NOT in front of a VAE round trip (the Wan VAE compresses 8x
spatially; the encode is a low-pass; the sampler repaints the rest), but
BEHIND the last decode, where its detail goes straight to the file. v568
drew that conclusion as a two-node chain; v569 moves it INTO the node.

FIVE CUTS, PINNED HERE:

1. pixel_stage gains 'model final', TAIL-APPENDED (the serialisation law,
   applied to value lists). Default unchanged.

2. In 'model final' the STAGES run pure fit - the um-null covers it in BOTH
   places (the _derive block and the stage loop). A model in front of an
   encode would be the exact mistake the mode exists to end.

3. The pass belongs to the LAST stage and takes ITS wire (v568: the wires
   are the truth): upscale_model_low in High + Low, upscale_model in Single.
   No wire -> no pass, announced, and the run behaves like 'fit only'.
   The H wire in High + Low is announced as unused for the pass.

4. THE SIZE LAW lives in the pure _final_canvas (exec'd below, in isolation,
   with real numbers): resize_method='none' -> the output IS the model
   result (a 1x model = pure detail pass at unchanged size); any kernel ->
   supersample back to the dialled canvas. No VAE follows, so no /8 grid.

5. The pass runs BETWEEN the stage loop and the video build, inside the
   mute capsule, with its own clock phase 'pix:final' posted up front, and
   _esrgan_pass(final=True) swaps the stage-NOTE (which names the VAE) for
   the supersample line - that wall is not in this room.

Script-style: exit 0 = pass.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v569_finalpass] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    pu = _read("nodes", "ph_power_upscale.py")
    js = _read("web", "js", "ph_power_upscale.js")

    # ---- 1: the option list, tail-appended --------------------------------------
    m = re.search(r'"pixel_stage":\s*\(\[(.*?)\]', pu, re.S)
    if not m:
        _fail("pixel_stage widget not found")
    opts = re.findall(r'"([^"]+)"', m.group(1))
    if opts != ["model + fit", "fit only", "model only", "model final"]:
        _fail(f"pixel_stage must be the four laws with 'model final' at the "
              f"TAIL, got {opts}")
    req = pu[pu.index('"required"'):pu.index('"optional"')]
    if '"default": "model + fit"' not in req:
        _fail("the default must stay 'model + fit' - a new mode never "
              "hijacks old saves")

    # ---- 2: the stages run pure fit, in BOTH um-null places ---------------------
    if 'if str(pixel_stage) in ("fit only", "model final"):' not in pu:
        _fail("the _derive block must null the stage model for 'model final' "
              "- a derived canvas may not follow a model that will not run")
    if ('if str(pixel_stage) in ("fit only", "model final") and um is not None:'
            not in pu):
        _fail("the stage loop must null the pixel model for 'model final' - "
              "the stages are pure fit, the wire serves the final pass")

    # ---- 3: the wire law of the pass --------------------------------------------
    blk_at = pu.find("if str(pixel_stage) == \"model final\":",
                     pu.index("_free()   # v561"))
    if blk_at < 0:
        _fail("the final-pass block is gone")
    blk = pu[blk_at:pu.index("video_out = _build_video")]
    if "um_fin = upscale_model_low if bool(dual_moe) else upscale_model" not in blk:
        _fail("the pass must take the LAST stage's wire: upscale_model_low "
              "in High + Low, upscale_model in Single (the wires are the "
              "truth)")
    if "NO final pass" not in blk or "'fit only'" not in blk:
        _fail("no wire -> no pass must be ANNOUNCED and named for what the "
              "run then is: 'fit only'")
    if "the H wire only serves stage pre-passes" not in blk:
        _fail("in High + Low the unused H wire must be announced")
    if "_esrgan_pass(cur, um_fin, tw, th, resize_method" not in blk:
        _fail("the pass must run through _esrgan_pass - fp16/resident/"
              "budget/interrupt come from the stack, not from a copy")
    if "final=True" not in blk:
        _fail("the pass must declare itself final - the stage-NOTE names a "
              "VAE that does not follow this pass")
    if "with _MuteInfoLogs(mute_staging_logs" not in blk:
        _fail("the pass must run inside the mute capsule like every stage")
    if 'clock.measure(key, dt)' not in blk or '"pix:final"' not in blk:
        _fail("the pass must feed the v567 clock under 'pix:final'")
    if '"pix:final"' not in pu[:blk_at]:
        _fail("'pix:final' must be POSTED up front (before the stage loop) "
              "so the ETA covers the pass from second one")

    # ---- 4: the size law - REAL MATH, exec'd in isolation -----------------------
    # AMENDED IN v582 (1st amendment): the law gained the user's dial,
    # final_upscale_by. The kernel path used to pin the canvas to (w, h)
    # unconditionally ("never grow it") - between that and 'none', the file
    # size was decided by everything EXCEPT the user. The old pins below that
    # still hold are kept verbatim; the kernel pins now state the new law,
    # and final_by=1.0 is pinned as the old law, bit for bit.
    src = pu[pu.index("def _final_canvas("):pu.index("def _chunks(")]
    ns = {}
    exec(compile(src, "<_final_canvas>", "exec"), ns)   # pure: no module names
    fc = ns["_final_canvas"]
    # Frank's live canvas, the 4x wire, 'none': the output IS the model result.
    if fc(1104, 1104, 4.0, "none") != (4416, 4416, True):
        _fail(f"none + 4x on 1104^2 must yield the raw 4416^2, got "
              f"{fc(1104, 1104, 4.0, 'none')}")
    # 'none' has no kernel to reach any other canvas: the dial is IGNORED.
    if fc(1104, 1104, 4.0, "none", 1.5) != (4416, 4416, True):
        _fail("'none' must ignore final_by - there is no kernel that could "
              "reach any other canvas")
    # The 1x restoration model, 'none': a pure detail pass at unchanged size.
    if fc(1104, 1104, 1.0, "none") != (1104, 1104, False):
        _fail("a 1x model under 'none' must keep the size - THE smoother")
    # Kernel + default 1.0: the OLD law, bit for bit (supersample back).
    if fc(1104, 1104, 4.0, "bicubic") != (1104, 1104, False):
        _fail("kernel + final_by=1.0 must reproduce the pre-v582 law exactly: "
              "supersample back onto the stage canvas")
    if fc(848, 480, 2.0, "lanczos (cpu)") != (848, 480, False):
        _fail("kernel + final_by=1.0 must be independent of the model factor")
    # Kernel + the user's factor: the canvas is the USER'S, model-independent.
    if fc(768, 768, 4.0, "lanczos (cpu)", 1.4) != (1075, 1075, True):
        _fail(f"kernel + final_by=1.4 on 768^2 must land on 1075^2 "
              f"(768*1.4=1075.2 rounds), got "
              f"{fc(768, 768, 4.0, 'lanczos (cpu)', 1.4)}")
    if (fc(768, 768, 4.0, "bicubic", 2.0)
            != fc(768, 768, 8.0, "bicubic", 2.0)):
        _fail("under a kernel the canvas must not depend on the model factor "
              "- swapping a 4x for an 8x changes the imprint, never the size")
    # Below 1.0: supersampled DOWNscale (the sprite path).
    if fc(768, 768, 4.0, "bicubic", 0.5) != (384, 384, True):
        _fail("final_by below 1.0 must shrink the canvas (supersampled "
              "downscale for sprite work)")
    # Non-integer factors round, they never truncate.
    if fc(850, 480, 1.5, "none") != (1275, 720, True):
        _fail("non-integer factors must ROUND the canvas")

    # ---- 5: no /8 talk on the final canvas --------------------------------------
    if "no /8 grid" not in blk:
        _fail("the pass must SAY that no /8 grid applies - nothing follows "
              "it but the file")

    # ---- serialisation ------------------------------------------------------------
    # AMENDED IN v582 (1st amendment, same cut as the size-law pins above):
    # this used to pin len(ORDER_CANON) == 25 absolutely. That was a TEXT pin
    # on a moving structure (lesson 1): v582 tail-appends a widget, and the
    # absolute count broke here and in four sibling guards at once. The claim
    # this guard actually owns is historical - "v569 added no widget; the
    # order up to pixel_stage stands" - and a POSITION pin states exactly
    # that, while every future tail-append preserves it.
    canon = re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1)
    names = re.findall(r'"([a-z_ +()]+)"', canon)
    if len(names) < 25 or names[24] != "pixel_stage":
        _fail(f"the first 25 canon slots up to pixel_stage are v569's "
              f"history and must stand - got {len(names)} entries, "
              f"slot 24 = {names[24] if len(names) > 24 else 'MISSING'}")
    # v582: every pre-v582 save is topped up on slot 25 with 1.0 - which the
    # size law above pins as the OLD law, bit for bit. An old workflow that
    # loads into the new node must run the run it always ran.
    if not re.search(r"25:\s*1\.0", js):
        _fail("CANON_DEFAULTS must top slot 25 up with 1.0 - without it every "
              "pre-v582 save loads a hole where the old behaviour should be")
    if "_healPreV569" in js:
        _fail("a heal step for a cut that changes no widget can only do "
              "damage - the v563 _sanitize net covers value lists")
    # AMENDED IN v582 (1st amendment, part 3): '[-1] != "pixel_stage"' was a
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

    print("[test_v569_finalpass] OK - 'model final' tail-appended, stages "
          "pure fit, the last stage's wire runs once behind the final "
          "decode (size law exec'd with real numbers), clock phase posted, "
          "no new widget")
    sys.exit(0)


if __name__ == "__main__":
    main()
