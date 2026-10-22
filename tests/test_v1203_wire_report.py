# -*- coding: ascii -*-
"""Guard v1203 -- the Power Upscale names the pixel model that REALLY runs the final pass.

(Frank 21.10., the neon log: "final : L wire, ESRGAN" -- in Single mode the final pass runs the H wire's model
(upscale_model); the report named the L wire (upscale_model_low, here NMKD-Siax) whenever one was plugged.)

  W1 Single, both wires plugged: "final  : H wire", plus the note that the L wire is NOT read
  W2 High + Low, both wires: "final  : L wire", no note
  W3 Single, only the L wire: final none (no H model) and the note
  W4 the report mirrors the pass: upscale() hands dual=bool(dual_moe); the pass picks
     upscale_model_low if dual_moe else upscale_model (both call sites)
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


class _UM:
    def __init__(self, name):
        self.architecture = types.SimpleNamespace(name=name)
        self.scale = 4


def say(h, l, dual):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        PU._say_model_wires(h, l, "model + fit", "lanczos (gpu)", dual=dual)
    return buf.getvalue()


H, L = _UM("ESRGAN"), _UM("ESRGAN")
o1 = say(H, L, False)
_need("final  : H wire" in o1 and "the L wire is NOT read" in o1, "W1 Single, both wires: final = H wire, the L wire named as not read")
o2 = say(H, L, True)
_need("final  : L wire" in o2 and "NOT read" not in o2, "W2 High + Low: final = L wire, no note")
o3 = say(None, L, False)
_need("final  : none -- no upscale model is wired" in o3 and "NOT read" in o3, "W3 Single, only the L wire: no final model, said")
src = (ROOT / "nodes" / "ph_power_upscale.py").read_text(encoding="utf-8")
_need("_say_model_wires(upscale_model, upscale_model_low, pixel_stage, resize_method, dual=bool(dual_moe))" in src
      and src.count("upscale_model_low if bool(dual_moe) else upscale_model") >= 2,
      "W4 the report gets dual_moe; the pass picks low only in High + Low (both call sites)")
_need(all(c < 128 for c in (ROOT / "tests" / "test_v1203_wire_report.py").read_bytes()), "Z ASCII")
print("%s: %d check(s) failed" % ("test_v1203_wire_report", len(FAILS)) if FAILS else "test_v1203_wire_report: all checks passed")
sys.exit(1 if FAILS else 0)
