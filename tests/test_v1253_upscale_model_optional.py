"""test_v1253_upscale_model_optional -- Power Upscale: the pixel upscale model is an OPTIONAL input.

Frank 31.10. (EYES6 branch 2, pixel_stage 'fit only' + resize nvidia_rtx_vsr): the frontend refused the run --
"Required input missing: upscale_model" -- on a stage that never reads it. The kernel always handled an unwired
model as a plain fit (_esrgan_pass, and the low wire was optional from the start); only the declaration was too
strict. Frank: "nicht sauber -- fixen". v1253: optional; ComfyUI's entry `run` hands None on, `upscale` keeps its
signature (every caller passes the model as its fifth positional argument).

  U1 schema: upscale_model is OPTIONAL (type UPSCALE_MODEL), not required; the other required inputs unchanged
  U2 driven: run() without upscale_model -> upscale(upscale_model=None)
  U3 driven: run() with a wired model -> that very object arrives
  U4 driven: the wire report says "plain fit" when a model stage has no model wired
  U5 the socket display order (JS) still lists upscale_model after vae
  U6 FUNCTION is "run", and upscale() keeps upscale_model as its 5th positional parameter
  Z  ASCII
"""
import inspect
import io
import os
import sys
import types
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1253_upscale_model_optional"
FAILS = []


def need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


def load():
    core = os.environ.get("PLS_CORE_ROOT") or "/home/claude/core_pin"
    if core and os.path.isdir(core) and core not in sys.path:
        sys.path.insert(0, core)
    import comfy.cli_args
    comfy.cli_args.args.cpu = True
    fp = types.ModuleType("folder_paths")
    fp.get_full_path = lambda *a, **k: None
    fp.get_filename_list = lambda *a, **k: []
    fp.add_model_folder_path = lambda *a, **k: None
    fp.get_folder_paths = lambda f: ["/tmp/models/" + f]
    fp.models_dir = "/tmp/models"
    fp.get_temp_directory = lambda: "/tmp"
    fp.get_output_directory = lambda: "/tmp"
    sys.modules.setdefault("folder_paths", fp)
    pkg = types.ModuleType("plsn1253")
    pkg.__path__ = [str(ROOT / "nodes")]
    sys.modules["plsn1253"] = pkg
    import importlib
    return importlib.import_module("plsn1253.ph_power_upscale")


def main():
    print(NAME)
    P = load()
    N = P.ULSPowerUpscale
    it = N.INPUT_TYPES()
    req, opt = it["required"], it.get("optional", {})
    need("upscale_model" not in req and opt.get("upscale_model", (None,))[0] == "UPSCALE_MODEL",
         "U1 upscale_model is optional (UPSCALE_MODEL), not required")
    need(all(k in req for k in ("model", "positive", "negative", "vae")), "U1 model / positive / negative / vae stay required")
    seen = {}
    node = N()
    node.upscale = lambda **kw: seen.update(kw) or ("ok",)
    node.run(model="M", positive="P", negative="N", vae="V", dual_moe=False)
    need("upscale_model" in seen and seen["upscale_model"] is None, "U2 run() without the wire -> upscale_model=None")
    marker = object()
    seen.clear()
    node.run(model="M", positive="P", negative="N", vae="V", upscale_model=marker, dual_moe=False)
    need(seen.get("upscale_model") is marker, "U3 a wired model arrives unchanged")
    buf = io.StringIO()
    with redirect_stdout(buf):
        P._say_model_wires(None, None, "model + fit", "lanczos (gpu)", dual=False)
    need("plain fit" in buf.getvalue(), "U4 the wire report says 'plain fit' for a model stage without a model")
    js = (ROOT / "web" / "js" / "ph_power_upscale.js").read_text(encoding="utf-8")
    i = js.index("const INPUT_DISPLAY_ORDER")
    order = js[i:js.index("];", i)]
    need(order.index('"vae"') < order.index('"upscale_model"'), "U5 the display order keeps upscale_model after vae")
    params = list(inspect.signature(N.upscale).parameters)
    need(N.FUNCTION == "run" and params[5] == "upscale_model",
         "U6 FUNCTION run, upscale() keeps upscale_model 5th (got %s)" % params[5])
    for p in (Path(__file__),):
        need(all(b < 128 for b in p.read_bytes()), "Z ASCII %s" % p.name)
    if FAILS:
        print("[%s] FAIL -- %d" % (NAME, len(FAILS)))
        return 1
    print("[%s] PASS -- the pixel model is optional, unwired means a plain fit, said" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
