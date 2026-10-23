# -*- coding: ascii -*-
"""Guard v1207 -- the restore says who holds the card, hands nothing of its own to it, and stops before the encode.

(Frank 21.10., U4 on v1206: free 2.75 -> 4.55 GiB after clearing, ~11 GiB held outside the model list, and the stop
came after 3 minutes of encoding. "go".)

  C1 to_host: a CPU tensor passes through untouched
  C2 card_report: no CUDA -> (None, {}); with CUDA: torch allocated / reserved, CUDA free / total, the memory held
     outside torch's pool (total - free - reserved) and the models ComfyUI still lists
  C3 clear_card prints that breakdown after the clear
  C4 plan_chunks plans from the FRAMES' size exactly what chunk_frames plans from the encoded latent (U4: 141 frames
     at 1056x1856 -> latent 36 x 132 x 232); too little memory stops in plan_chunks; in restore() the plan comes after
     the clear and BEFORE the encode
  C5 the frames go to the CPU before the clear (restore) and at all three call sites of the node; the refine's
     latent goes to the CPU before the restore
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


CORE = os.environ.get("PLS_CORE_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
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

FREE = {"gib": 15.0}
mm.get_free_memory = lambda *a, **k: FREE["gib"] * (1024 ** 3)
mm.unload_all_models = lambda: None
mm.soft_empty_cache = lambda *a, **k: None


def _quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


# ---- C1
x = torch.zeros(2, 3)
_need(R.to_host(x) is x and R.to_host(None) is None, "C1 a CPU tensor (and a non-tensor) passes through untouched")

# ---- C2
_need(R.card_report() == (None, {}) if not torch.cuda.is_available() else True, "C2 no CUDA -> (None, {})")
G = 1024 ** 3
_real = (torch.cuda.is_available, torch.cuda.memory_allocated, torch.cuda.memory_reserved, torch.cuda.mem_get_info)
torch.cuda.is_available = lambda: True
torch.cuda.memory_allocated = lambda *a: int(4.33 * G)
torch.cuda.memory_reserved = lambda *a: int(4.40 * G)
torch.cuda.mem_get_info = lambda *a: (int(0.50 * G), int(15.90 * G))
_mm_dev = mm.get_torch_device
mm.get_torch_device = lambda: "cuda:0"


class MiniMaxH3:
    pass


class _Patcher:
    model = MiniMaxH3()

    def loaded_size(self):
        return int(3.2 * G)


class _LM:
    model = _Patcher()


mm.current_loaded_models.append(_LM())
line, d = R.card_report()
_need(line is not None and abs(d["other"] - 11.0) < 0.02 and abs(d["allocated"] - 4.33) < 0.01
      and abs(d["cuda_free"] - 0.5) < 0.01 and abs(d["total"] - 15.9) < 0.01,
      "C2 the numbers: allocated 4.33, free 0.50 of 15.90, outside torch's pool 11.00 (%s)" % (d.get("other"),))
_need("held outside torch's pool 11.00 GiB" in line and "MiniMaxH3 3.20 GiB" in line, "C2 the line names it and the listed model")

# ---- C3
(_, _), out = _quiet(R.clear_card, "test")
_need("card after the clear: torch allocated 4.33 GiB" in out, "C3 clear_card prints the breakdown")
mm.current_loaded_models.pop()
torch.cuda.is_available, torch.cuda.memory_allocated, torch.cuda.memory_reserved, torch.cuda.mem_get_info = _real
mm.get_torch_device = _mm_dev

# ---- C4
FREE["gib"] = 15.0
a, _ = _quiet(R.plan_chunks, 141, 1856, 1056, 0)
b, _ = _quiet(R.chunk_frames, torch.zeros(1, 16, 36, 232, 132), 0)
_need(a == b == 9, "C4 plan from the frames == plan from the latent (U4 size, 15 GiB free: %s / %s)" % (a, b))
FREE["gib"] = 40.0
a, _ = _quiet(R.plan_chunks, 6, 38, 50, 0)
_need(a == 9, "C4 a 6-frame clip plans as the 9 frames pad_frames makes of it (got %s)" % a)
a, _ = _quiet(R.plan_chunks, 9, 38, 50, 5)
_need(a == 5, "C4 a manual count passes through plan_chunks")
FREE["gib"] = 4.55
try:
    _quiet(R.plan_chunks, 141, 1856, 1056, 0)
    _need(False, "C4 too little memory stops in plan_chunks")
except RuntimeError as e:
    _need("4.55 GiB free" in str(e), "C4 too little memory stops in plan_chunks, with the numbers")
src = (ROOT / "nodes" / "ph_restore.py").read_text()
body = src[src.index("def restore("):src.index("def verdict(")]
i_clear, i_plan, i_enc = body.find("clear_card("), body.find("plan_chunks("), body.find('vae_step(vae, "encode"')
_need(0 < i_clear < i_plan < i_enc, "C4 restore(): clear -> plan -> encode")

# ---- C5
i_host = body.find("frames = to_host(frames)")
_need(0 < i_host < i_clear, "C5 restore() takes the frames to the CPU before the clear")
pu = (ROOT / "nodes" / "ph_power_upscale.py").read_text()
sites = pu.count("cur = _RS.to_host(cur)\n")
calls = pu.count("cur, _rs_info = _restore_pass(cur,")
_need(sites == calls == 3, "C5 all three call sites hand the frames over on the CPU (%d / %d)" % (sites, calls))
rp = pu[pu.index("def _restore_pass("):pu.index("_RS.restore(")]
_need('_jr_info["vid_lat"] = _jr_info["vid_lat"].detach().to("cpu")' in rp, "C5 the refine's latent goes to the CPU first")

# ---- Z
for f in ("nodes/ph_restore.py", "tests/test_v1207_card_report.py"):
    try:
        (ROOT / f).read_bytes().decode("ascii")
        _need(True, "Z ASCII " + f)
    except UnicodeDecodeError:
        _need(False, "Z ASCII " + f)

print("%s: %d check(s) failed" % ("test_v1207_card_report", len(FAILS)) if FAILS else "test_v1207_card_report: all checks passed")
sys.exit(1 if FAILS else 0)
