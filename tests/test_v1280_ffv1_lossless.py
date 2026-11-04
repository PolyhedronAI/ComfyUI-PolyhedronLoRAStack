#!/usr/bin/env python3
# -*- coding: ascii -*-
"""Guard v1280 -- Save "FFV1 (lossless archive)" is lossless in the colour.

Up to v1279 the preset wrote yuv420p: a quarter of the colour resolution and a YUV round trip (found 04.11. with FX
CRT -- an RGB phosphor mask lost half its contrast in 4:2:0). Now planar RGB at 16 bit (gbrp16le) from the 16-bit
intermediate.

  F1  the preset: ffv1, gbrp16le, ten_bit (16-bit intermediate), intra, no even-edge demand (odd sizes allowed).
  F2  driven: the node's own PyAV writer encodes a random RGB clip at an ODD size, PyAV reads it back -- every
      pixel within one 16-bit step of the input (max error < 2/65535), and a one-pixel RGB stripe pattern keeps
      its full contrast (4:2:0 would grey it).
  F3  the other presets unchanged (the ladder is append-only in meaning).

Script-style: exit 0 = pass.
"""
import importlib.util
import os
import sys
import tempfile
import types

NAME = "v1280"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
_fails = []


def _need(cond, msg):
    if not cond:
        _fails.append(msg)
    print(("  ok    " if cond else "  FAIL  ") + msg)


import numpy as np  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


pkg = types.ModuleType("pls1280")
pkg.__path__ = [os.path.join(ROOT, "nodes")]
sys.modules["pls1280"] = pkg
U = _load("pls1280.ph_save_util", "nodes/ph_save_util.py")
p = U.PRESETS["FFV1 (lossless archive)"]
_need(p["vcodec"] == "ffv1" and p["pix_fmt"] == "gbrp16le" and p["ten_bit"] and p["intra"] and p["backend"] == "pyav",
      "F1 FFV1 = planar RGB 16 bit from the 16-bit intermediate, intra")
want = {"H.264 MP4 (delivery)": "yuv420p", "H.265 MP4 (delivery, grain)": "yuv420p10le",
        "ProRes 422 HQ (master)": "yuv422p10le", "ProRes 4444 (master + alpha)": "yuva444p10le", "WebM VP9": "yuv420p"}
_need(all(U.PRESETS[k]["pix_fmt"] == v for k, v in want.items()), "F3 the other video presets unchanged")

try:
    import av  # noqa: F401
    S = _load("pls1280.ph_save", "nodes/ph_save.py")
except Exception as e:  # no PyAV / no Core pieces here
    S = None
    print("  skip  F2 (%s)" % e)
if S is not None:
    _need(S._needs_even(p["backend"], p["pix_fmt"]) is False, "F1 odd edges allowed (no chroma subsampling)")
    g = np.random.default_rng(1280)
    H, W = 33, 47
    clip = g.random((3, H, W, 3)).astype(np.float32)
    clip[2] = 0.0
    clip[2, :, 0::3, 0] = 1.0                      # one-pixel R / G / B stripes
    clip[2, :, 1::3, 1] = 1.0
    clip[2, :, 2::3, 2] = 1.0
    path = os.path.join(tempfile.mkdtemp(prefix="pls1280_"), "t.mkv")
    S.ULSSave()._write_pyav(clip, None, None, p, 24.0, 17, path)
    with av.open(path) as c:
        st = c.streams.video[0]
        fmt = st.codec_context.pix_fmt
        back = [f.to_ndarray(format="rgb48le").astype(np.float64) / 65535.0 for f in c.decode(video=0)]
    back = np.stack(back)
    err = float(np.abs(back - clip).max())
    _need(fmt == "gbrp16le" and back.shape == clip.shape, "F2 written as %s, %s" % (fmt, back.shape))
    _need(err < 2.0 / 65535, "F2 every pixel back within one 16-bit step (max error %.2e)" % err)
    stripes = back[2, H // 2, 3:42]
    _need(float(stripes.max(1).min()) > 0.99 and float(stripes.min(1).max()) < 0.01,
          "F2 one-pixel RGB stripes keep their full contrast")

print("[%s] %d failed" % (NAME, len(_fails)))
print("[%s] %s" % (NAME, "PASS" if not _fails else "FAIL"))
sys.exit(1 if _fails else 0)
