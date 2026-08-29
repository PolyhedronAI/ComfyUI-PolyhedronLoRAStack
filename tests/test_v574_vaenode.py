"""Guard v574 -- the Polyhedron VAE node: one box, four modes, a verdict
that runs BEFORE the pass.

THE RESEARCHED PREMISE (read from Frank's exact ComfyUI revision ba9ffa0a,
not remembered): stock VAE.decode estimates its need, attempts the FULL
pass, and falls back to hardcoded 256 px tiles only when an OOM exception
fires. Comfy's own Wan-2.1 formula says 65 f @ 1104 px needs ~17.1 GB - a
16 GB card can never run it full - and the measured WDDM behaviour
(v570-v573, three completed runs) is to PAGE and GRIND instead of
throwing. Stock's safety net can therefore simply never fire while the
pass crawls. This node moves the verdict before the pass, with comfy's
own numbers, out loud.

PINNED HERE:

1. THE PURE _vae_budget_verdict - exec'd below with the researched
   numbers. 'on'/'off' obey the user; 'auto' tiles above 85% of free
   (the v565 headroom convention).

2. THE MODE LAW: both/encode/decode/roundtrip, default 'both', the wires
   decide in 'both' (v568 doctrine), single-lane modes IGNORE the other
   input LOUDLY, an unproducible output returns None and SAYS so - the
   truth beats a silent dummy tensor.

3. STOCK REPLICATION, not reinvention: the tiled decode converts to
   LATENT space with stock's exact clamps; the tiled encode passes
   pixel-space values like stock; comfy's encode_tiled/decode_tiled do
   the work. And the SCAR is respected: no WANVIDEOVAE anywhere.

4. THE METER: roundtrip prints the v568 sharpness (Laplacian variance of
   the luma) before and after - structural pin only, torch is absent in
   the gate sandbox.

Script-style: exit 0 = pass.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v574_vaenode] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    pv = _read("nodes", "ph_vae.py")
    init = _read("__init__.py")

    # ---- 1: the verdict, EXECUTED with the researched numbers ---------------------
    src = pv[pv.index("def _vae_budget_verdict("):pv.index("def _luma_sharpness(")]
    ns = {}
    exec(compile(src, "<_vae_budget_verdict>", "exec"), ns)
    v = ns["_vae_budget_verdict"]
    # 65 f @ 1104 px: comfy's own formula says ~17.1 GB against 15.6 free.
    if v(17.1e9, 15.6e9, "auto") != "tiled":
        _fail("the 1104-canvas decode must verdict tiled - stock would "
              "attempt it full and WDDM would grind, not crash")
    # 65 f @ 768 px: ~8.3 GB against 15.6 free -> full is honest.
    if v(8.3e9, 15.6e9, "auto") != "full":
        _fail("the 768-canvas decode fits - auto must not tile it")
    # The 85% headroom boundary is strict.
    if v(13.26e9, 15.6e9, "auto") != "full" or v(13.27e9, 15.6e9, "auto") != "tiled":
        _fail("the 0.85 headroom boundary moved")
    if v(1.0, 100.0, "on") != "tiled" or v(100.0, 1.0, "off") != "full":
        _fail("on/off must obey the user unconditionally")

    # ---- 2: the mode law ------------------------------------------------------------
    m = re.search(r'_MODES = \[(.*?)\]', pv)
    modes = re.findall(r'"([^"]+)"', m.group(1)) if m else []
    if modes != ["both", "encode", "decode", "roundtrip"]:
        _fail(f"the modes must be both/encode/decode/roundtrip in order, "
              f"got {modes}")
    if '"default": "both"' not in pv:
        _fail("'both' must be the default - the toggle exists so ONE node "
              "replaces the stacked pair")
    if "the wired samples are IGNORED" not in pv or \
       "the wired pixels are IGNORED" not in pv:
        _fail("single-lane modes must ignore the other input LOUDLY")
    if pv.count("wiring it \n") > 0:
        pass
    if pv.count("the honest outcome") < 2:
        _fail("both unproducible outputs must announce their None - the "
              "truth beats a silent dummy tensor")

    # ---- 3: stock replication, and the scar ------------------------------------------
    if "vae.temporal_compression_decode()" not in pv or \
       "vae.spacial_compression_decode()" not in pv:
        _fail("the tiled decode must convert with stock's own compression "
              "queries - latent-space tiles, like ba9ffa0a")
    if "tps = max(2, tps // tcomp)" not in pv:
        _fail("stock's temporal min-clamp is gone")
    if "vae.encode_tiled(" not in pv or "vae.decode_tiled(" not in pv:
        _fail("comfy's tilers must do the work - we wrap, we do not "
              "reinvent")
    if "WANVIDEOVAE" in pv.upper().replace("WANVIDEO VAE", "WANVIDEOVAE") \
            and "no WANVIDEOVAE" not in pv:
        _fail("the v122-v244 scar: no WANVIDEOVAE bridging in this node")
    if "memory_used_encode" not in pv or "memory_used_decode" not in pv:
        _fail("the verdict must use comfy's OWN formulas, not our guesses")

    # ---- 4: the meter (structural - no torch in the gate sandbox) --------------------
    if "def _luma_sharpness(" not in pv:
        _fail("the sharpness meter is gone")
    if "0.2126" not in pv or "0.7152" not in pv or "0.0722" not in pv:
        _fail("the luma weights moved - Rec.709, like the measurement arc")
    if "roundtrip sharpness" not in pv:
        _fail("the roundtrip must PRINT its number - a meter that stays "
              "silent is not a meter")

    # ---- registration -----------------------------------------------------------------
    if 'NODE_CLASS_MAPPINGS["ULSVAE"] = ULSVAE' not in init:
        _fail("ULSVAE is not registered")
    # AMENDED IN v578 (1st amendment): the DISPLAY NAME became "Polyhedron VAE
    # Codec" - it sat one letter away from "Polyhedron Load VAE" in the search
    # box, and one loads a file while the other uses it. A display name is FREE
    # to change: LiteGraph only serialises a node's title when the USER
    # overrode it, so old workflows simply pick the new label up. The node_id
    # ULSVAE - the string the workflow FILE stores - did NOT move, and
    # test_v578_node_ids now pins it so it never can by accident.
    if '"\u2b21 Polyhedron VAE Codec"' not in init:
        _fail("the display name must carry the hex glyph, like every node "
              "in the house")
    if "from .nodes.ph_vae import ULSVAE" not in init:
        _fail("the import must follow the house try/except pattern")

    print("[test_v574_vaenode] OK - verdict exec'd on the researched "
          "numbers (17.1 GB tiles, 8.3 GB runs full, 85% boundary strict, "
          "on/off obey), mode law pinned, stock replicated not reinvented, "
          "the scar respected, the meter present and loud")
    sys.exit(0)


if __name__ == "__main__":
    main()
