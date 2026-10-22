# -*- coding: ascii -*-
"""Guard v1208 -- the restore releases the Dynamic VRAM buffers Core would release only after the node.

(Frank 21.10., U4b on v1207: after the clear torch held 0.07 GiB, no model was listed, 11.28 GiB stayed held outside
torch's pool -- comfy-aimdo's cast buffers of the refine model, which Core's executor frees only in the finally after
a node. "go".)

  D1 Dynamic VRAM off -> nothing is called, [] comes back
  D2 Dynamic VRAM on -> prefetch queues, cast buffers, vbar watermarks -- in Core's order
  D3 clear_card: unload -> the three -> empty cache, and the console line names what was released
  D4 a Core without one of the calls: the others still run, nothing raises
  D5 the three names are the ones Core's executor calls (execution.py of the Core under test)
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
import comfy.model_management as mm  # noqa: E402
import comfy.memory_management as cmm  # noqa: E402
R = importlib.import_module("ph_restore")

CALLS = []
mm.get_free_memory = lambda *a, **k: 4.55 * (1024 ** 3)
mm.unload_all_models = lambda: CALLS.append("unload")
mm.soft_empty_cache = lambda *a, **k: CALLS.append("empty")
mm.reset_cast_buffers = lambda: CALLS.append("cast")
mp = types.ModuleType("comfy.model_prefetch")
mp.cleanup_prefetch_queues = lambda: CALLS.append("prefetch")
sys.modules["comfy.model_prefetch"] = mp
import comfy  # noqa: E402
comfy.model_prefetch = mp
aim = types.ModuleType("comfy_aimdo")
vb = types.ModuleType("comfy_aimdo.model_vbar")
vb.vbars_reset_watermark_limits = lambda: CALLS.append("vbar")
aim.model_vbar = vb
sys.modules["comfy_aimdo"] = sys.modules.get("comfy_aimdo") or aim
sys.modules["comfy_aimdo.model_vbar"] = vb
if sys.modules["comfy_aimdo"] is not aim:
    sys.modules["comfy_aimdo"].model_vbar = vb


def _quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = fn(*a, **k)
    return r, buf.getvalue()


# ---- D1
cmm.aimdo_enabled = False
CALLS.clear()
_need(R.release_dynamic_buffers() == [] and CALLS == [], "D1 Dynamic VRAM off: nothing called, []")

# ---- D2
cmm.aimdo_enabled = True
CALLS.clear()
ran = R.release_dynamic_buffers()
_need(CALLS == ["prefetch", "cast", "vbar"] and ran == ["prefetch queues", "cast buffers", "vbar watermarks"],
      "D2 Dynamic VRAM on: prefetch -> cast -> vbar, in Core's order (%s)" % CALLS)

# ---- D3
CALLS.clear()
_, out = _quiet(R.clear_card, "test")
_need(CALLS == ["unload", "prefetch", "cast", "vbar", "empty"], "D3 clear: unload -> the three -> empty (%s)" % CALLS)
_need("Dynamic VRAM buffers released: prefetch queues, cast buffers, vbar watermarks" in out, "D3 the line names them")
cmm.aimdo_enabled = False
CALLS.clear()
_, out = _quiet(R.clear_card, "test")
_need(CALLS == ["unload", "empty"] and "Dynamic VRAM" not in out, "D3 with Dynamic VRAM off the clear is v1207's")

# ---- D4
cmm.aimdo_enabled = True
_saved = mm.reset_cast_buffers
del mm.reset_cast_buffers
CALLS.clear()
try:
    ran = R.release_dynamic_buffers()
    _need(CALLS == ["prefetch", "vbar"] and "cast buffers" not in ran, "D4 a missing call is skipped, the others run")
except Exception as e:
    _need(False, "D4 a missing call raised %r" % e)
mm.reset_cast_buffers = _saved
cmm.aimdo_enabled = False

# ---- D5
ex = (Path(CORE) / "execution.py").read_text()
fin = ex[ex.find("finally:", ex.find("get_output_data(prompt_id")):][:900]
_need(all(n in fin for n in ("cleanup_prefetch_queues()", "reset_cast_buffers()", "vbars_reset_watermark_limits()")),
      "D5 Core's executor calls the same three after a node (%s)" % os.path.basename(CORE))

# ---- Z
for f in ("nodes/ph_restore.py", "tests/test_v1208_dynamic_release.py"):
    try:
        (ROOT / f).read_bytes().decode("ascii")
        _need(True, "Z ASCII " + f)
    except UnicodeDecodeError:
        _need(False, "Z ASCII " + f)

print("%s: %d check(s) failed" % ("test_v1208_dynamic_release", len(FAILS)) if FAILS else "test_v1208_dynamic_release: all checks passed")
sys.exit(1 if FAILS else 0)
