# -*- coding: ascii -*-
"""Guard v1206 -- the restore pass clears the card, keeps a temporal floor, steps down on an out-of-memory, talks.

(Frank 21.10., test U4: OOM in the restore decode at 1056x1856 x141 -- the refine model still held the card, the
chunk law saw 3.46 GiB free and planned ONE frame per chunk, the decode found 0 bytes; the encode ran 3 minutes
without a line. "Dann bau ... go.")

  M1 clear_card unloads every model and empties the cache, and reports free before -> after
  M2 chunk_frames: the auto plan never goes below 5 frames -- too little memory stops with the numbers (free GiB,
     Mpx, the way out); enough memory plans >= 5; a MANUAL count is the user's word (1 stays 1)
  M3 vae_step: an out-of-memory clears the card and steps down the tile ladder; a non-memory error is raised at
     once; the last rung failing is raised
  M4 restore() clears the card BEFORE the encode and again BEFORE the decode, both VAE calls go through vae_step,
     every sampled chunk prints a line
  M5 a one-call phase beats: a line every BEAT_S while it runs, a 'done in' line at the end
  Z  ASCII
Script-style: exit 0 = pass.
"""
import contextlib
import importlib
import io
import os
import re
import sys
import time
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


CORE = os.environ.get("PLS_CORE_ROOT") or "/home/claude/core_pin"
sys.path.insert(0, CORE)
sys.path.insert(0, str(ROOT / "nodes"))
import comfy.cli_args  # noqa: E402
comfy.cli_args.args.cpu = True
fp = types.ModuleType("folder_paths")
for _n in ("get_temp_directory", "get_output_directory", "get_input_directory", "get_user_directory"):
    setattr(fp, _n, lambda: "/tmp")
fp.get_filename_list = lambda *a, **k: []
fp.get_full_path = lambda *a, **k: None
fp.models_dir = "/tmp/models"
fp.add_model_folder_path = lambda *a, **k: None
fp.get_folder_paths = lambda f: ["/tmp/models/" + f]
sys.modules.setdefault("folder_paths", fp)
import comfy.model_management as mm  # noqa: E402
R = importlib.import_module("ph_restore")
R._rungs_path = lambda: None          # AMENDED v1209: no learned rungs here (the ladder is walked from the top)

CALLS = []
FREE = {"gib": 3.46}
mm.get_free_memory = lambda *a, **k: FREE["gib"] * (1024 ** 3)
mm.unload_all_models = lambda: CALLS.append("unload")
mm.soft_empty_cache = lambda *a, **k: CALLS.append("empty")


def _quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


# ---- M1
CALLS.clear()
(b, a), out = _quiet(R.clear_card, "test")
_need(CALLS == ["unload", "empty"], "M1 clear_card unloads every model, then empties the cache")
_need("free 3.46 -> 3.46 GiB" in out and b == a, "M1 it reports free before -> after")

# ---- M2 (U4's numbers: 1056x1856 -> latent 132x232, 141 frames -> 36 latent frames)
lat = torch.zeros(1, 16, 36, 132, 232)
FREE["gib"] = 3.46
try:
    _quiet(R.chunk_frames, lat, 0)
    _need(False, "M2 too little memory stops instead of planning 1 frame")
except RuntimeError as e:
    msg = str(e)
    _need("3.46 GiB free" in msg and "1.96 Mpx" in msg and "restore_frames to 5" in msg and "before refine" in msg,
          "M2 too little memory stops with the numbers and the way out (%s...)" % msg[:60])
FREE["gib"] = 15.0
fpc, out = _quiet(R.chunk_frames, lat, 0)
_need(fpc >= R.MIN_AUTO_FRAMES and (fpc - 1) % 4 == 0 and "frames per chunk" in out,
      "M2 enough memory plans >= %d frames, 4n+1 (got %d)" % (R.MIN_AUTO_FRAMES, fpc))
FREE["gib"] = 3.46
_need(_quiet(R.chunk_frames, lat, 1)[0] == 1 and _quiet(R.chunk_frames, lat, 5)[0] == 5,
      "M2 a manual count is the user's word (1 stays 1, 5 stays 5)")
short = torch.zeros(1, 16, 1, 132, 232)
FREE["gib"] = 15.0
_need(_quiet(R.chunk_frames, short, 0)[0] == 1, "M2 a one-frame clip plans its one frame (the floor is the clip)")


# ---- M3
class _OomVAE:
    def __init__(self, fail_until, err=None):
        self.fail_until, self.err, self.seen = fail_until, err, []

    def _go(self, x, **k):
        self.seen.append(k["tile_x"])
        if len(self.seen) <= self.fail_until:
            raise (self.err or torch.OutOfMemoryError("CUDA out of memory"))
        return x + 1

    encode_tiled = _go
    decode_tiled = _go


CALLS.clear()
v = _OomVAE(2)
(res, rung), out = _quiet(R.vae_step, v, "decode", torch.zeros(2), "decode")
_need(rung == 2 and v.seen == [512, 256, 256] and CALLS.count("unload") == 2,
      "M3 two out-of-memories: two clears, the third rung decodes (%s)" % v.seen)
_need("ran out of memory at tiles 512 px" in out and "next rung 256 px" in out, "M3 the step down is said")
v = _OomVAE(1, err=ValueError("shape"))
try:
    _quiet(R.vae_step, v, "encode", torch.zeros(2), "encode")
    _need(False, "M3 a non-memory error is raised at once")
except ValueError:
    _need(v.seen == [512], "M3 a non-memory error is raised at once, no further rung")
v = _OomVAE(99)
try:
    _quiet(R.vae_step, v, "decode", torch.zeros(2), "decode")
    _need(False, "M3 the last rung failing is raised")
except torch.OutOfMemoryError:
    _need(len(v.seen) == len(R.TILE_LADDER), "M3 the last rung failing is raised (all %d rungs tried)" % len(R.TILE_LADDER))
_need(R.TILE_LADDER[0] == R.TILE, "M3 the first rung is Core's template")

# ---- M4 (statics on restore(); the end-to-end run is test_v1194 R3)
src = (ROOT / "nodes" / "ph_restore.py").read_text()
body = src[src.index("def restore("):src.index("def verdict(")]
i_c1 = body.find("clear_card(")
i_enc = body.find('vae_step(vae, "encode"')
i_c2 = body.find('clear_card("before the decode")')
i_dec = body.find('vae_step(vae, "decode"')
_need(0 < i_c1 < i_enc < i_c2 < i_dec, "M4 clear -> encode -> sample -> clear -> decode, in that order")
_need("encode_tiled(" not in body and "decode_tiled(" not in body, "M4 no bare VAE call left in restore()")
# AMENDED v1210 (Frank: "dieses Rumgehopse mag ich nicht"): no line per chunk any more -- a heartbeat over the
# sampling and ONE summary line after it
_need("restore sample done: %d chunk(s)" in body and "restore sample chunk" not in body,
      "M4 the sampling ends in one summary line (no line per chunk)")
# AMENDED v1207: the plan moved BEFORE the encode (plan_chunks from the frames' size) -- still after the clear
_need(i_c1 < body.find("plan_chunks(") < i_enc, "M4 the chunk plan comes after the clear (it plans with what is free)")

# ---- M5
R.BEAT_S = 0.05


def _phase():
    with R._Beat("encode (test)"):
        time.sleep(0.25)


_, out = _quiet(_phase)
_need(len(re.findall(r"restore encode \(test\) \.\.\. 0:00 elapsed", out)) >= 2 and "done in" in out,
      "M5 the phase beats while it runs and says when it is done")

# ---- Z
for f in ("nodes/ph_restore.py", "tests/test_v1206_restore_memory.py"):
    try:
        (ROOT / f).read_bytes().decode("ascii")
        _need(True, "Z ASCII " + f)
    except UnicodeDecodeError:
        _need(False, "Z ASCII " + f)

print("%s: %d check(s) failed" % ("test_v1206_restore_memory", len(FAILS)) if FAILS else "test_v1206_restore_memory: all checks passed")
sys.exit(1 if FAILS else 0)
