#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1053 -- the VAE node's tile widgets reach MiniMax H3's own tiler.

THE FAULT (read in Core 7a131a3a, comfy/ldm/minimax/vae.py):

    def decode_tiled(self, z, **kwargs):  return self.decode(z)
    def encode_tiled(self, x, **kwargs):  return self.encode(x)

H3 throws every tile argument away and tiles from three attributes of its own
(tiling, tile_size, tile_overlap_min; stock 256 / 64 px). The node's
tile_size / tile_overlap therefore reached every VAE except H3.

WHAT IS PINNED

  T1  the plan: tile on the 16 px grid (>= 128), overlap at least a quarter
      of the tile (H3's own 64/256 ratio), raised -> reported, capped one
      cell under the tile.
  T2  _H3Tiles sets the three attributes for one pass and restores them
      exactly -- also when the pass raises; a non-H3 VAE is never touched.
  T3  Core's REAL split_tiles (when PLS_CORE_ROOT is set) accepts every plan
      and covers the frame; the stock plan and a 512 plan give the tile
      counts the changelog quotes.
  T4  the lanes, end to end with a recording fake VAE: tiled decode/encode
      run with the widget tiles, the full verdict runs with Core's stock
      tiles and says so, and the VAE comes back as it was.
  T5  the console line names tile, overlap and the stock pair.

Script-style: exit 0 = pass.
"""

import importlib
import io
import os
import sys
import types
from contextlib import redirect_stdout

NAME = "v1053"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "nodes"))

_fails = []
_checks = 0


def _need(cond, msg):
    global _checks
    _checks += 1
    if not cond:
        _fails.append(msg)
        print("[%s] FAIL -- %s" % (NAME, msg))


# Core when given (T3 reads its real tiler), else a comfy stub: ph_vae imports
# comfy.model_management at module level and the lanes never consult it.
CORE = os.environ.get("PLS_CORE_ROOT")
HAVE_CORE = bool(CORE) and os.path.isfile(os.path.join(CORE, "comfy", "ldm", "minimax", "vae.py"))
if HAVE_CORE:
    sys.path.insert(0, CORE)
    sys.argv = [sys.argv[0], "--cpu"]          # Core's cli_args: no GPU here
    import comfy.options
    comfy.options.enable_args_parsing()        # main.py does this; else --cpu is ignored
if not HAVE_CORE and "comfy" not in sys.modules:
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.get_free_memory = lambda *a, **k: float(1 << 34)
    mm.intermediate_device = lambda: "cpu"
    mm.get_torch_device = lambda: "cpu"
    comfy.model_management = mm
    sys.modules["comfy"] = comfy
    sys.modules["comfy.model_management"] = mm

import torch  # noqa: E402

pv = importlib.import_module("ph_vae")

# ---------------------------------------------------------------- T1 plan
P = pv._h3_tile_plan
_need(P(256, 64) == (256, 64, False, 64), "T1 stock 256/64 stays 256/64, not raised")
_need(P(512, 64) == (512, 128, True, 64), "T1 512 with 64 overlap -> raised to 128 (a quarter)")
_need(P(512, 160) == (512, 160, False, 160), "T1 an overlap above the quarter is kept")
_need(P(300, 0)[:2] == (288, 80), "T1 tile snaps DOWN to 16, overlap UP to 16 (300/0 -> 288/80)")
_need(P(64, 0)[:2] == (128, 32), "T1 tile never below 128")
_need(P(128, 200)[:2] == (128, 112), "T1 overlap capped one cell under the tile")
for ts in range(64, 2049, 32):
    for ov in (0, 16, 64, 96, 128, 300):
        t, o, _r, _a = P(ts, ov)
        if not (t % 16 == 0 and o % 16 == 0 and o * 4 >= t and o < t):
            _need(False, "T1 plan %s/%s -> %s/%s breaks grid/ratio/cap" % (ts, ov, t, o))
            break


# ---------------------------------------------------------------- fakes
class MiniMaxH3VideoVAE(object):
    """Named like Core's class -- detection asks the class NAME."""

    def __init__(self):
        self.tiling, self.tile_size, self.tile_overlap_min = True, 256, 64
        self.seen = []


class _OtherInner(object):
    def __init__(self):
        self.tiling, self.tile_size, self.tile_overlap_min = True, 256, 64


class _FakeVAE(object):
    def __init__(self, inner):
        self.first_stage_model = inner
        self.vae_dtype = torch.float32

    def _rec(self, lane):
        i = self.first_stage_model
        i.seen.append((lane, bool(i.tiling), int(i.tile_size), int(i.tile_overlap_min)))

    def decode_tiled(self, z, **kw):
        self._rec("decode_tiled")
        return torch.zeros(2, 8, 8, 3)

    def decode(self, z):
        self._rec("decode")
        return torch.zeros(2, 8, 8, 3)

    def encode_tiled(self, x, **kw):
        self._rec("encode_tiled")
        return torch.zeros(1, 24, 1, 2, 2)

    def encode(self, x):
        self._rec("encode")
        return torch.zeros(1, 24, 1, 2, 2)

    def temporal_compression_decode(self):
        return 4

    def spacial_compression_decode(self):
        return 16


# ---------------------------------------------------------------- T2 capsule
h3 = _FakeVAE(MiniMaxH3VideoVAE())
buf = io.StringIO()
with redirect_stdout(buf):
    with pv._H3Tiles(h3, 512, 64, "decode"):
        i = h3.first_stage_model
        _need((i.tiling, i.tile_size, i.tile_overlap_min) == (True, 512, 128),
              "T2 inside the capsule H3 tiles with 512/128")
i = h3.first_stage_model
_need((i.tiling, i.tile_size, i.tile_overlap_min) == (True, 256, 64),
      "T2 after the capsule the VAE is back at 256/64")
i.tiling = False                       # a user-set state must come back too
try:
    with redirect_stdout(io.StringIO()):
        with pv._H3Tiles(h3, 768, 192, "decode"):
            raise RuntimeError("boom")
except RuntimeError:
    pass
_need((i.tiling, i.tile_size, i.tile_overlap_min) == (False, 256, 64),
      "T2 restored also when the pass raises (tiling=False came back)")
i.tiling = True
oth = _FakeVAE(_OtherInner())
with redirect_stdout(io.StringIO()):
    with pv._H3Tiles(oth, 1024, 0, "decode"):
        o = oth.first_stage_model
        _need((o.tile_size, o.tile_overlap_min) == (256, 64), "T2 a non-H3 VAE is never touched")

# T5 console
log = buf.getvalue()
_need("512 px tiles" in log and "128 px overlap" in log and "256/64" in log,
      "T5 the console names tile, overlap and the stock pair: %r" % log)
_need("raised from 64 px" in log, "T5 a raised overlap is said, with the asked value")

# ---------------------------------------------------------------- T3 Core
if HAVE_CORE:
    try:
        cvae = importlib.import_module("comfy.ldm.minimax.vae")
        split = cvae.MiniMaxH3VideoVAE.split_tiles

        def tiles(n, ts, ov):
            ns = types.SimpleNamespace(tile_size=ts, tile_overlap_min=ov, vae_ratio=16)
            idx, lens, ovs = split(ns, n)
            ok = idx[0] == 0 and idx[-1] + lens[-1] == n and all(x >= ov for x in ovs)
            return len(idx), ok

        for n in (768, 1344, 1072, 1920):
            for ts, ov in ((256, 64), P(384, 64)[:2], P(512, 64)[:2], P(1024, 64)[:2]):
                c, ok = tiles(n, ts, ov)
                _need(ok, "T3 Core split_tiles covers %d px with %d/%d" % (n, ts, ov))
        s = tiles(768, 256, 64)[0] * tiles(1344, 256, 64)[0]
        b = tiles(768, 512, 128)[0] * tiles(1344, 512, 128)[0]
        print("[%s] tiles per frame at 1344x768: stock 256/64 = %d, 512/128 = %d" % (NAME, s, b))
        _need(s == 28 and b == 8, "T3 tile counts 1344x768: stock 28, 512/128 8 (got %d / %d)" % (s, b))
        # the stock pair in Core is what we restore to
        import inspect
        sig = inspect.signature(cvae.MiniMaxH3VideoVAE.__init__)
        _need(sig.parameters["tile_size"].default == 256, "T3 Core's stock tile_size is still 256")
        src = inspect.getsource(cvae.MiniMaxH3VideoVAE.decode_tiled)
        _need("return self.decode(" in src, "T3 Core's decode_tiled still discards its arguments "
              "(if this changes, the capsule may be redundant -- re-check)")
    except ImportError as exc:
        print("[%s] note: Core import failed (%s) -- T3 skipped" % (NAME, exc))
else:
    print("[%s] note: PLS_CORE_ROOT not set -- T3 (Core's real tiler) skipped" % NAME)

# ---------------------------------------------------------------- T4 lanes
node = pv.ULSVAE()
lat = torch.zeros(1, 24, 7, 48, 84)
h3 = _FakeVAE(MiniMaxH3VideoVAE())
buf = io.StringIO()
with redirect_stdout(buf):
    node._decode_lane(h3, lat, "tiled", 512, 64, 64, 8, "off")
    node._decode_lane(h3, lat, "full", 512, 64, 64, 8, "off")
    node._encode_lane(h3, torch.zeros(5, 64, 64, 3), "tiled", 384, 0, 64, 8)
    node._encode_lane(h3, torch.zeros(5, 64, 64, 3), "full", 384, 0, 64, 8)
seen = h3.first_stage_model.seen
_need(seen[0] == ("decode_tiled", True, 512, 128), "T4 tiled decode runs with 512/128: %r" % (seen[:1],))
_need(seen[1] == ("decode", True, 256, 64), "T4 full decode runs with Core's 256/64: %r" % (seen[1:2],))
_need(seen[2] == ("encode_tiled", True, 384, 96), "T4 tiled encode runs with 384/96: %r" % (seen[2:3],))
_need(seen[3] == ("encode", True, 256, 64), "T4 full encode runs with Core's 256/64: %r" % (seen[3:4],))
i = h3.first_stage_model
_need((i.tiling, i.tile_size, i.tile_overlap_min) == (True, 256, 64), "T4 the VAE is back at 256/64 after the lanes")
out = buf.getvalue()
_need(out.count("tiles internally at Core's 256/64 px") == 2,
      "T4 the full verdict says what really runs (decode and encode)")
oth = _FakeVAE(_OtherInner())
oth.first_stage_model.seen = []
with redirect_stdout(io.StringIO()) as b2:
    node._decode_lane(oth, lat, "full", 512, 64, 64, 8, "off")
_need("MiniMax H3" not in b2.getvalue(), "T4 no H3 line for another VAE")

print("[%s] %d checks, %d failed" % (NAME, _checks, len(_fails)))
if _fails:
    sys.exit(1)
print("[%s] PASS" % NAME)
