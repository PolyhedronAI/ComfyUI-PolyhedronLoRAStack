# -*- coding: ascii -*-
"""Guard v1209 -- the restore's VAE starts on the tile rung learned for this card at this size.

(Frank 21.10., U4 on v1208: the decode ran out of memory at 512 px, 256/32 and 256/16 -- 5:39 lost -- before
128 px / 8 frames ran in 4:48. "Hat fuer mich auch gerade sehr lange gedauert ... Go.")

  L1 nothing learned -> the walk starts at Core's template (rung 0)
  L2 two out-of-memories -> the rung that ran is remembered (op, card, area); the next call at the same size starts
     THERE, says so in the console and runs once
  L3 a larger frame starts at least there (what failed smaller fails larger); a smaller frame starts at the top
  L4 another card or the other op does not inherit it
  L5 the learned rung still runs out of memory -> the walk goes on down and the new rung is remembered
  L6 the file sits in the user directory next to the learned rates; a broken file is read as empty
  Z  ASCII
Script-style: exit 0 = pass.
"""
import contextlib
import importlib
import io
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


CORE = os.environ.get("PLS_CORE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, CORE)
sys.path.insert(0, str(ROOT / "nodes"))
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
TMP = tempfile.mkdtemp(prefix="pls1209_")
fp = types.ModuleType("folder_paths")
for _n in ("get_temp_directory", "get_output_directory", "get_input_directory"):
    setattr(fp, _n, lambda: TMP)
fp.get_user_directory = lambda: TMP
fp.get_filename_list = lambda *a, **k: []
fp.get_full_path = lambda *a, **k: None
fp.models_dir = TMP
fp.add_model_folder_path = lambda *a, **k: None
fp.get_folder_paths = lambda f: [TMP]
sys.modules["folder_paths"] = fp
import comfy.model_management as mm  # noqa: E402
R = importlib.import_module("ph_restore")
mm.get_free_memory = lambda *a, **k: 14.0 * (1024 ** 3)
mm.unload_all_models = lambda: None
mm.soft_empty_cache = lambda *a, **k: None
R._card_tag = lambda: "gpu16"


class _VAE:
    """Runs out of memory on every tile larger than `fits` px."""

    def __init__(self, fits):
        self.fits, self.seen = fits, []

    def _go(self, x, **k):
        self.seen.append(k["tile_x"])
        if k["tile_x"] > self.fits or (k["tile_x"] == self.fits and k["tile_t"] > self.fits_t):
            raise torch.OutOfMemoryError("CUDA out of memory")
        return x

    encode_tiled = _go
    decode_tiled = _go


def _vae(fits, fits_t=999):
    v = _VAE(fits)
    v.fits_t = fits_t
    return v


def _q(fn, *a):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a)
    return r, buf.getvalue()


lat = torch.zeros(1, 16, 4, 232, 132)          # U4's latent grid: 1056x1856 -> 1.96 Mpx per frame
big = torch.zeros(1, 16, 4, 240, 136)
small = torch.zeros(1, 16, 4, 100, 60)

# ---- L1
v = _vae(128, 8)
(_, rung), _ = _q(R.vae_step, v, "decode", lat, "decode")
_need(v.seen[0] == 512 and rung == 3, "L1 nothing learned: the walk starts at 512 px and lands on rung 3 (%s)" % v.seen)

# ---- L2
v = _vae(128, 8)
(_, rung), out = _q(R.vae_step, v, "decode", lat, "decode")
_need(v.seen == [128] and rung == 3, "L2 the same size starts on the learned rung and runs once (%s)" % v.seen)
_need("starts at tiles 128 px / 8 frames" in out and R.RUNGS_FILE in out, "L2 the console says where it starts and why")
d = json.load(open(os.path.join(TMP, "polyhedron", R.RUNGS_FILE)))
e = d["entries"][0]
_need(e["op"] == "decode" and e["card"] == "gpu16" and e["area"] == 232 * 132 * 64 and e["rung"] == 3,
      "L2 the entry: op, card, area, rung (%s)" % e)

# ---- L3
v = _vae(128, 8)
_q(R.vae_step, v, "decode", big, "decode")
_need(v.seen[0] == 128, "L3 a larger frame starts at least on the learned rung (%s)" % v.seen)
v = _vae(512)
(_, rung), _ = _q(R.vae_step, v, "decode", small, "decode")
_need(v.seen == [512] and rung == 0, "L3 a smaller frame starts at the top (%s)" % v.seen)

# ---- L4
_need(R.rung_start("decode", 232 * 132 * 64, "gpu24")[0] == 0, "L4 another card does not inherit the rung")
_need(R.rung_start("encode", 232 * 132 * 64, "gpu16")[0] == 0, "L4 the encode does not inherit the decode's rung")

# ---- L5 (the learned rung 2 still fails -> on to 3, remembered)
R._card_tag = lambda: "gpu12"          # a fresh card: nothing of L1..L3 applies
R.rung_learn("decode", 300 * 300 * 64, 2)
v = _vae(128, 8)
x5 = torch.zeros(1, 16, 4, 300, 300)
(_, rung), out = _q(R.vae_step, v, "decode", x5, "decode")
_need(v.seen == [256, 128] and rung == 3 and R.rung_start("decode", 300 * 300 * 64)[0] == 3,
      "L5 a learned rung that fails walks on and the new rung is remembered (%s)" % v.seen)

# ---- L6
R._card_tag = lambda: "gpu16"
_need(R._rungs_path() == os.path.join(TMP, "polyhedron", R.RUNGS_FILE), "L6 the file sits in user/polyhedron")
open(R._rungs_path(), "w").write("{broken")
_need(R.rung_start("decode", 10 ** 9)[0] == 0, "L6 a broken file reads as empty")

# ---- Z
for f in ("nodes/ph_restore.py", "tests/test_v1209_learned_rung.py"):
    try:
        (ROOT / f).read_bytes().decode("ascii")
        _need(True, "Z ASCII " + f)
    except UnicodeDecodeError:
        _need(False, "Z ASCII " + f)

import shutil  # noqa: E402
shutil.rmtree(TMP, ignore_errors=True)
print("%s: %d check(s) failed" % ("test_v1209_learned_rung", len(FAILS)) if FAILS else "test_v1209_learned_rung: all checks passed")
sys.exit(1 if FAILS else 0)
