# -*- coding: ascii -*-
"""Guard v1186 -- how the pixel upscale model is RUN (pixel_speed), and the fp16-aware tile estimate.

  S1 the widget: pixel_speed is the LAST required widget (old workflows keep their order), default 'off',
     the four modes
  S2 _esrgan_resident on a fake 2x model (CPU): 'off' and 'channels_last' give the SAME frames (bit for bit), the
     path string names run=...; an unknown mode runs as 'off'
  S3 compile is refused for a model with size requirements (padding) -- eager, said; and never asked for off CUDA
  S4 the plumbing: both model-carrying _esrgan_pass call sites hand speed=pixel_speed (AST), upscale() takes it
  S5 the activation estimate counts the bytes the forward writes: 2 under fp16 autocast, 4 in fp32 (the v566
     constant was fp32 and stayed after v568) -- for a 1344 edge at 4x that is 7.4 GB, not 14.8
  Z  ASCII of this file
Script-style: exit 0 = pass.
"""
import ast
import importlib
import os
import re
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
torch.set_num_threads(1)
try:
    import comfy.cli_args
    comfy.cli_args.args.cpu = True          # the sandbox has no GPU: Core on the CPU (as g3cpu.py does)
    import comfy.utils  # noqa: F401
    import comfy.model_management  # noqa: F401
    HAVE_CORE = True
except Exception as exc:
    HAVE_CORE = False
    print("  (no Core at %s: %s)" % (CORE, exc))
fp = types.ModuleType("folder_paths")
fp.get_temp_directory = lambda: "/tmp"
fp.get_output_directory = lambda: "/tmp"
fp.get_input_directory = lambda: "/tmp"
fp.get_filename_list = lambda *a, **k: []
fp.get_full_path = lambda *a, **k: None
sys.modules.setdefault("folder_paths", fp)
PU = importlib.import_module("ph_power_upscale")
SRC = (ROOT / "nodes/ph_power_upscale.py").read_text(encoding="utf-8")

# ---- S1
req = PU.ULSPowerUpscale.INPUT_TYPES()["required"]
keys = list(req)
_need(keys.index("pixel_speed") == keys.index("audio_stream") + 1 and keys[-1] in ("pixel_speed", "restore_frames", "restore_order"),   # AMENDED v1199: + restore_model / restore_order
      "S1 pixel_speed sits right after audio_stream (appended in v1186; v1194 appended the restore dials behind it) -- last is %s" % keys[-1])
_need(req["pixel_speed"][1].get("default") == "off" and tuple(req["pixel_speed"][0]) == PU._PIXEL_SPEEDS
      and PU._PIXEL_SPEEDS == ("off", "channels_last", "compile", "compile + channels_last"),
      "S1 default 'off', four modes")


# ---- S2 / S3  a fake spandrel-like 2x model
class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        with torch.no_grad():
            self.conv.weight.zero_()
            for c in range(3):
                self.conv.weight[c, c, 1, 1] = 1.0

    def forward(self, x):
        return torch.nn.functional.interpolate(self.conv(x), scale_factor=2, mode="nearest")


class _Req:
    minimum, multiple_of, square = 0, 1, False


class _ReqPad:
    minimum, multiple_of, square = 0, 8, False


class FakeUpscale:
    scale = 2.0
    supports_half = False

    def __init__(self, req=_Req()):
        self.model = _Net().eval()
        self.size_requirements = req

    def to(self, *a, **k):
        return self

    def __call__(self, x):
        with torch.inference_mode():
            return self.model(x).clamp_(0, 1)


if HAVE_CORE:
    img = torch.rand(3, 40, 56, 3)
    outs, infos = {}, {}
    for mode in ("off", "channels_last", "nonsense"):
        o, mid, info, ch = PU._esrgan_resident(img.clone(), FakeUpscale(), 80, 112, "bicubic", 8, speed=mode)
        outs[mode], infos[mode] = o, info
    _need(torch.equal(outs["off"], outs["channels_last"]) and tuple(outs["off"].shape) == (3, 112, 80, 3),
          "S2 'off' and 'channels_last' give the same frames (%s)" % (tuple(outs["off"].shape),))
    _need("run=off" in infos["off"] and "run=channels_last" in infos["channels_last"] and "run=off" in infos["nonsense"],
          "S2 the path string names the run mode; an unknown mode runs as 'off'")
    o3, _m, info3, _c = PU._esrgan_resident(img.clone(), FakeUpscale(), 80, 112, "bicubic", 8, speed="compile")
    _need(torch.equal(o3, outs["off"]) and "run=compile (eager)" in info3,
          "S3 'compile' off CUDA runs eager and says so (%s)" % info3[-40:])
    o4, _m, info4, _c = PU._esrgan_resident(img.clone(), FakeUpscale(_ReqPad()), 80, 112, "bicubic", 8,
                                            speed="compile + channels_last")
    _need(torch.equal(o4, outs["off"]) and "(eager)" in info4, "S3 a padding model is never compiled (eager)")
    _need("compile skipped: the model pads its input" in SRC, "S3 the refusal is said")
else:
    _need(False, "S2/S3 need Core (comfy.utils) at PLS_CORE_ROOT or /home/claude/core_pin")

# ---- S4
tree = ast.parse(SRC)
calls = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "_esrgan_pass":
        first_model_arg = node.args[1] if len(node.args) > 1 else None
        carries_model = not (isinstance(first_model_arg, ast.Constant) and first_model_arg.value is None)
        has_speed = any(k.arg == "speed" and getattr(k.value, "id", "") == "pixel_speed" for k in node.keywords)
        calls.append((carries_model, has_speed))
_need(len(calls) == 3 and all(hs for cm, hs in calls if cm) and sum(1 for cm, _ in calls if cm) == 2,
      "S4 both model-carrying _esrgan_pass calls hand speed=pixel_speed (%s)" % calls)
_need("pixel_speed" in PU.ULSPowerUpscale.upscale.__code__.co_varnames, "S4 upscale() takes pixel_speed")

# ---- S5
_need(re.search(r"act_bytes = 2 if half_possible else 4", SRC) is not None
      and SRC.count("act = (tile * scale) ** 2 * 64 * act_bytes * 2") == 2,
      "S5 the estimate uses act_bytes (2 fp16 / 4 fp32) in both places")
_need(abs((1344 * 4) ** 2 * 64 * 2 * 2 / 1e9 - 7.4) < 0.05 and abs((1344 * 4) ** 2 * 64 * 4 * 2 / 1e9 - 14.8) < 0.05,
      "S5 a 1344 edge at 4x: 7.4 GB in fp16, 14.8 GB in fp32 -- the one-call frame fits a 16 GB card only with the fp16 count")

# ---- Z
_need(all(c < 128 for c in (ROOT / "tests/test_v1186_pixel_speed.py").read_bytes()), "Z  ASCII tests/test_v1186_pixel_speed.py")

print("%s: %d check(s) failed" % ("test_v1186_pixel_speed", len(FAILS)) if FAILS
      else "test_v1186_pixel_speed: all checks passed")
sys.exit(1 if FAILS else 0)
