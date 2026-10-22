# -*- coding: ascii -*-
"""Guard v1205 -- the Power Upscale lands a canvas a video encoder cannot open (an odd edge) EVEN itself.

(Frank 21.10., test U6: 768x1344 x 1.40 = 1075x1882 -- the save cropped to 1074, the interpolation padded; "Falls das
die anderen Tests nicht beeinflusst, dann ja: bau".)

  E1 _even_down: an odd edge rounds DOWN (= the size the save cropped to), an even one is untouched
  E2 the node end to end (fit-only delivering final pass, refine off): 20x28 x 1.40 -> 28x39 -> lands 28x38; an even
     canvas is untouched; resize_method 'none' keeps the raw size (the note stays, nothing is changed)
  E3 the model pass lands even too (statics, both sites) and only with a kernel; the refine's /32 snap comes FIRST
     (so U1..U5 / the LoRA / ref-noise series -- refine after pixel, canvas snapped -- are untouched)
  Z  ASCII
Script-style: exit 0 = pass.
"""
import contextlib
import importlib
import io
import os
import sys
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
PU = importlib.import_module("ph_power_upscale")

# ---- E1
_need(PU._even_down(1075, 1882) == (1074, 1882, True) and PU._even_down(1074, 1881) == (1074, 1880, True)
      and PU._even_down(1056, 1856) == (1056, 1856, False), "E1 odd rounds down, even untouched")


class _PlainModel:
    class _Inner:
        diffusion_model = torch.nn.Conv2d(3, 3, 1)
    model = _Inner()


class _PlainVAE:
    first_stage_model = torch.nn.Conv2d(3, 3, 1)


node = PU.ULSPowerUpscale()
cond = [[torch.zeros(1, 4, 8), {}]]
ARGS = (False, 2.0, 0.2, 3, 1.0, 1.3, 0.25, 5, 1.9, 7, "euler", "simple", 1024, 64, 0.0)


def run(img, fby, method):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = node.upscale(_PlainModel(), cond, cond, _PlainVAE(), None, *ARGS, image=img, resize_method=method, pixel_stage="fit only",
                         result_preview=False, process_preview="Off", refine_order="off", final_upscale_by=fby)
    return r["result"][0], buf.getvalue()


o, log = run(torch.rand(3, 20, 28, 3), 1.4, "bicubic")
_need(tuple(o.shape[1:3]) == (28, 38) and "landed on 38x28" in log, "E2 20x28 x 1.40 = 28x39 -> lands 28x38, said (%s)" % (tuple(o.shape),))
o2, _ = run(torch.rand(3, 20, 30, 3), 1.4, "bicubic")
_need(tuple(o2.shape[1:3]) == (28, 42), "E2 an even canvas is untouched (28x42)")
o3, log3 = run(torch.rand(3, 20, 28, 3), 1.4, "none")
_need(tuple(o3.shape[1:3]) == (20, 28) and "landed on" not in log3, "E2 resize_method 'none': the raw size, nothing changed")

# ---- E3 statics
src = (ROOT / "nodes" / "ph_power_upscale.py").read_text(encoding="utf-8")
i_snap = src.find("_sw, _sh = _snap_to(tw, th, _jr_snap)")
i_even = src.find("if str(resize_method) != _NO_RESIZE and ((int(tw) & 1) or (int(th) & 1)):")
_need(0 < i_snap < i_even and src.count("= _even_down(tw, th)") == 2, "E3 the model pass lands even too, AFTER the refine's /32 snap (both sites)")
_need(all(c < 128 for c in (ROOT / "tests" / "test_v1205_even_canvas.py").read_bytes()), "Z ASCII")
print("%s: %d check(s) failed" % ("test_v1205_even_canvas", len(FAILS)) if FAILS else "test_v1205_even_canvas: all checks passed")
sys.exit(1 if FAILS else 0)
