"""Guard v562 -- spatial VAE tiling, the temporal ban, and the honest device.

Fable's review closed three gaps this guard now holds shut:

  1. VAE TILING (the last big peak): each tile pushed the WHOLE frame stack
     through vae.encode/vae.decode. `vae_tiling` (Off/512/640/768, default Off
     - the byte-identical path) routes them through Comfy's tiled entry points.
  2. THE TEMPORAL BAN: a Wan VAE compresses time 4:1, so a temporal tile would
     cut ACROSS that compression and stitch independently decoded time windows
     - stutter seams. `tile_t` / `overlap_t` are pinned to None wherever the
     signature has them. This is the single most important pin in this file.
  3. THE HONEST DEVICE: Comfy hands the decode back on intermediate_device()
     (CPU unless --gpu-only), so v561's "feather on the GPU" is a no-op in the
     default setup. The node now MEASURES and prints the device instead of
     claiming a win it may not have.

Behavioural: the tiled path is exercised against fake VAEs (a modern one with
tile_t, a legacy one without, and one that raises) - no torch needed.
Script-style: exit 0 = pass.
"""
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v562_vaetiling] FAIL: " + msg)
    sys.exit(1)


def main():
    pu = open(os.path.join(ROOT, "nodes", "ph_power_upscale.py"),
              encoding="utf-8").read()
    js = open(os.path.join(ROOT, "web", "js", "ph_power_upscale.js"),
              encoding="utf-8").read()

    # ---- run _vae_ops against fake VAEs ---------------------------------------
    m = re.search(r"_VAE_TILES = \[.*?\n(?=\ndef _vram_note)", pu, re.S)
    if not m:
        _fail("_vae_ops not extractable")
    ns = {"inspect": __import__("inspect")}
    exec(m.group(0), ns)  # noqa: S102 - our own source, measured
    vae_ops = ns["_vae_ops"]

    class Modern:
        def __init__(self):
            self.calls = []

        def encode(self, x):
            self.calls.append(("encode", {}))
            return "latent"

        def decode(self, x):
            self.calls.append(("decode", {}))
            return "px"

        def encode_tiled(self, x, tile_x=512, tile_y=512, overlap=64,
                         tile_t=None, overlap_t=None):
            self.calls.append(("encode_tiled", dict(tile_x=tile_x, tile_y=tile_y,
                                                    overlap=overlap, tile_t=tile_t,
                                                    overlap_t=overlap_t)))
            return "latent"

        def decode_tiled(self, x, tile_x=64, tile_y=64, overlap=16,
                         tile_t=None, overlap_t=None):
            self.calls.append(("decode_tiled", dict(tile_x=tile_x, tile_y=tile_y,
                                                    overlap=overlap, tile_t=tile_t,
                                                    overlap_t=overlap_t)))
            return "px"

    # Off = the historic path, untouched
    v = Modern()
    enc, dec, label = vae_ops(v, "Off")
    enc("x"); dec("y")
    if label != "off" or [c[0] for c in v.calls] != ["encode", "decode"]:
        _fail("vae_tiling=Off must use the plain encode/decode (byte-identical)")

    # 512 = tiled, and THE TEMPORAL BAN holds
    v = Modern()
    enc, dec, label = vae_ops(v, "512")
    enc("x"); dec("y")
    if label != "tiled(512)":
        _fail(f"the label must state the tile size, got {label!r}")
    kinds = [c[0] for c in v.calls]
    if kinds != ["encode_tiled", "decode_tiled"]:
        _fail(f"the tiled entry points were not used: {kinds}")
    e_kw, d_kw = v.calls[0][1], v.calls[1][1]
    if e_kw["tile_t"] is not None or e_kw["overlap_t"] is not None:
        _fail("TEMPORAL TILING on encode - a Wan VAE compresses time 4:1; this "
              "stitches independently decoded time windows and stutters")
    if d_kw["tile_t"] is not None or d_kw["overlap_t"] is not None:
        _fail("TEMPORAL TILING on decode - same ban, same reason")
    if e_kw["tile_x"] != 512:
        _fail("encode takes PIXEL tiles - 512 must arrive as 512")
    if d_kw["tile_x"] != 64:
        _fail("decode takes LATENT tiles - 512 pixels must arrive as 64")

    # a legacy VAE without the tiled path falls back, loudly but safely
    class Legacy:
        def encode(self, x):
            return "latent"

        def decode(self, x):
            return "px"

    enc, dec, label = vae_ops(Legacy(), "512")
    if "off" not in label or enc("x") != "latent":
        _fail("a VAE without a tiled path must fall back to the plain one")

    # a tiled path that raises must not kill the run
    class Angry(Legacy):
        def encode_tiled(self, x, tile_x=512, tile_y=512, overlap=64):
            raise RuntimeError("nope")

        def decode_tiled(self, x, tile_x=64, tile_y=64, overlap=16):
            raise RuntimeError("nope")

    enc, dec, _ = vae_ops(Angry(), "640")
    if enc("x") != "latent" or dec("y") != "px":
        _fail("a failing tiled path must fall back, never abort the run")

    # ---- contract pins ---------------------------------------------------------
    req = pu[pu.index('"required"'):pu.index('"optional"')]
    names = re.findall(r'"([a-z_0-9]+)":\s*\(', req)
    # v564 hardening: position pin (later cuts append behind it).
    if names.index("vae_tiling") != names.index("per_batch") + 1:
        _fail("vae_tiling must sit directly after per_batch (appended in v562)")
    if '"default": "Off"' not in req:
        _fail("the default must be Off - the byte-identical historic path")
    if "NEVER temporal" not in pu:
        _fail("the temporal ban must stay documented where it is enforced")
    if "decode device=" not in pu:
        _fail("the decode device must be MEASURED and printed (v561's 'feather "
              "on the GPU' is a no-op when Comfy returns on the CPU)")
    if "vae={v_label}" not in pu:
        _fail("the BEGIN line must state which VAE path ran")
    if "def _vram_note" not in pu or "mem_get_info" not in pu:
        _fail("the preflight VRAM note is gone (say it BEFORE the OOM)")
    if "vae_encode=v_enc, vae_decode=v_dec" not in pu:
        _fail("the ops are not handed to the refine loop")

    # ---- heal 23 -> 24, measured in node ---------------------------------------
    # The exact banner version is pinned by the file's NEWEST guard (v563).
    canon = re.findall(r'"([a-z_ ()]+)"',
                       re.search(r"const ORDER_CANON = \[(.*?)\];", js, re.S).group(1))
    if canon.index("vae_tiling") != 23:
        _fail("vae_tiling must sit at canon index 23 (appended)")
    parts = [re.search(rx, js) for rx in (
        r"const LEN_PRE_V562 = \d+;",
        r"function _healPreV562\(wv\) \{[\s\S]*?\n\}")]
    if not all(parts):
        _fail("the v562 heal is not extractable")
    harness = "\n".join(x.group(0) for x in parts) + "\n" + r"""
const v560 = Array.from({length: 23}, (_, i) => i);
const a = _healPreV562(v560.slice());
if (a.length !== 24 || a[23] !== "Off") {
    console.error("FAIL 23->24: " + JSON.stringify(a.slice(21))); process.exit(1);
}
const c24 = Array.from({length: 24}, (_, i) => i);
if (_healPreV562(c24.slice()).join(",") !== c24.join(",")) {
    console.error("FAIL: a v562 save must pass through untouched"); process.exit(1);
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

    print("PASS: v562 -- tiled VAE exercised (modern / legacy / failing), the "
          "TEMPORAL BAN enforced, pixel-vs-latent tiles correct, decode device "
          "measured, heal 23->24")
    sys.exit(0)


if __name__ == "__main__":
    main()
