"""Guard v561 -- the refine loop: single-tile fast path, GPU feather, freeing.

Three more speed/VRAM leaks, measured on the live 129-frame 848x848 stage:

  1. ONE tile covering the whole canvas still built the blend machinery: a
     1.1 GB float32 accumulator, 278M CPU multiplies and 278M CPU divides -
     for a result identical to the decoded tile (the feather is 1.0 everywhere).
     v561 returns the tile directly. This is exactly the configuration the
     coverage warning tells the user to pick, so the fast path is the norm.
  2. The feather was applied AFTER copying the decoded tile to the CPU - 278M
     float ops per tile on the slowest hardware in the box. Now the weight is
     moved to the tile's device and the multiply happens there.
  3. Nothing was released between tiles/stages: a tiled video run allocates and
     frees GB-sized tensors, and the allocator fragments until the NEXT tile
     OOMs on memory it technically has.

Pinned so none of it can creep back. Script-style: exit 0 = pass.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v561_fastpath] FAIL: " + msg)
    sys.exit(1)


def main():
    pu = open(os.path.join(ROOT, "nodes", "ph_power_upscale.py"),
              encoding="utf-8").read()
    body = pu[pu.index("def _refine_tiles"):pu.index("def _free()")
              if "def _free()" in pu[pu.index("def _refine_tiles"):]
              else len(pu)]
    body = pu[pu.index("def _refine_tiles"):pu.index("# \u2500\u2500 v549: result preview")]

    # 1) the fast path
    if "single = (len(grid[\"tiles\"]) == 1" not in body:
        _fail("the single-tile detection is gone")
    if "if not single:" not in body:
        _fail("the accumulator must NOT be allocated for a single full tile "
              "(that is a 1.1 GB tensor for nothing)")
    if "if single:" not in body or "no blending" not in body:
        _fail("the fast path (return the decoded tile directly) is gone")
    if not re.search(r"if not single:\n\s+acc = torch\.zeros", body):
        _fail("the accumulators must be allocated ONLY when blending is needed "
              "(guarded by `if not single:`)")

    # 2) the feather on the device
    if "w2.to(px.device)" not in body:
        _fail("the feather must be applied ON the tile's device, not after a "
              "CPU copy (278M float ops per tile on the CPU)")
    if ".cpu() * w2" in body:
        _fail("the CPU-side multiply is back - that was the leak")

    # 3) freeing
    if "def _free()" not in pu:
        _fail("the allocator-release helper is gone")
    if "empty_cache" not in pu:
        _fail("nothing hands the freed blocks back - the next tile will OOM on "
              "memory it technically has")
    if "del latent, samples, noise" not in body:
        _fail("the GPU tensors must be released before the fit")
    if "_free()   # v561: between stages" not in pu:
        _fail("the stages must release between them too")
    if "single-tile fast path" not in pu:
        _fail("the done line must SAY when the fast path ran (measure > believe)")

    print("PASS: v561 -- single-tile fast path (no 1.1 GB accumulator, no 278M "
          "CPU ops), feather on the device, allocator released between tiles "
          "and stages")
    sys.exit(0)


if __name__ == "__main__":
    main()
