"""test_v1244_newer_than_core (public cut: no N6, Image Reference is not in this build) -- a model file this ComfyUI cannot read says WHY.

v1244 (Frank 30.10.): the Ritter-Test Qwen workflows were red on an older Core -- the model
family arrived in a later ComfyUI. The pack's loaders now turn the raw Core error ("Could not
detect model type", a KeyError on a weight name, an invalid VAE) into one plain sentence: the
file, the running ComfyUI version, "update ComfyUI". Out-of-memory errors pass through untouched.
Image Reference: vision tokens are kept whenever no reference latent is spliced (Core's rule).

  N1 newer_than_core_error names the file, the ComfyUI version and the cure
  N2 Load Model (diffusion_models): a detection error becomes the plain sentence (driven)
  N3 Load Model: an out-of-memory error passes through unchanged (driven)
  N4 Load VAE: an invalid / unknown VAE becomes the plain sentence (driven)
  N5 Load CLIP and the checkpoint path carry the same wrap (source pin)
  N6 Image Reference plan: keep_vision True with a VAE but no image, False with both
  N7 this file is pure ASCII
Needs the Core on PYTHONPATH (comfy, folder_paths).
"""
import importlib.util
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = "test_v1244_newer_than_core"
FAILS = []


def need(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def load_nodes_module(mod):
    pkg = "pls_v1244_nodes"
    if pkg not in sys.modules:
        p = types.ModuleType(pkg)
        p.__path__ = [os.path.join(ROOT, "nodes")]
        sys.modules[pkg] = p
    spec = importlib.util.spec_from_file_location(pkg + "." + mod, os.path.join(ROOT, "nodes", mod + ".py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[pkg + "." + mod] = m
    spec.loader.exec_module(m)
    return m


def main():
    print(NAME)
    if "--cpu" not in sys.argv:
        sys.argv = [sys.argv[0], "--cpu"]          # the Core reads its flags at import
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import comfy.sd  # noqa: F401
        import folder_paths  # noqa: F401
    except Exception as e:
        print("  SKIP: the Core is not importable (%s)" % e)
        print("[%s] SKIP" % NAME)
        return 0
    import comfy.sd
    import comfy.utils
    import folder_paths
    B = load_nodes_module("ph_basics")

    # N1
    e = B.newer_than_core_error("Load Model", "/x/y/qwen_new.safetensors", RuntimeError("ERROR: Could not detect model type of: q\nhint"))
    t = str(e)
    need(isinstance(e, ValueError) and "qwen_new.safetensors" in t and "update ComfyUI" in t
         and "ComfyUI" in t and "Could not detect model type" in t and "\n" not in t,
         "N1 the sentence names file, version, cause, cure (%r)" % t[:120])

    tmp = tempfile.mkdtemp()
    fpath = os.path.join(tmp, "m.safetensors")
    open(fpath, "wb").write(b"\0" * 16)
    old_gfp, old_ldm = folder_paths.get_full_path, comfy.sd.load_diffusion_model
    try:
        folder_paths.get_full_path = lambda folder, name: fpath
        L = B.ULSLoadModel()

        def boom(*a, **k):
            raise RuntimeError("ERROR: Could not detect model type of: " + fpath)
        comfy.sd.load_diffusion_model = boom
        try:
            L.load(1, "default", model_1="m.safetensors")
            need(False, "N2 Load Model raised nothing")
        except ValueError as ex:
            need("update ComfyUI" in str(ex) and "m.safetensors" in str(ex),
                 "N2 Load Model: detection error -> plain sentence (%r)" % str(ex)[:90])
        except Exception as ex:
            need(False, "N2 Load Model raised %s: %s" % (type(ex).__name__, ex))

        def oom(*a, **k):
            raise RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")
        comfy.sd.load_diffusion_model = oom
        try:
            L.load(1, "default", model_1="m.safetensors")
            need(False, "N3 nothing raised")
        except RuntimeError as ex:
            need("out of memory" in str(ex), "N3 out-of-memory passes through unchanged")
        except Exception as ex:
            need(False, "N3 OOM was rewrapped as %s" % type(ex).__name__)
    finally:
        folder_paths.get_full_path, comfy.sd.load_diffusion_model = old_gfp, old_ldm

    # N4
    old_raise, old_ltf, old_vae = folder_paths.get_full_path_or_raise, comfy.utils.load_torch_file, comfy.sd.VAE
    try:
        folder_paths.get_full_path_or_raise = lambda folder, name: fpath
        comfy.utils.load_torch_file = lambda *a, **k: ({}, {})

        class BadVAE:
            def __init__(self, *a, **k):
                pass

            def throw_exception_if_invalid(self):
                raise RuntimeError("ERROR: VAE is invalid: None")
        comfy.sd.VAE = BadVAE
        try:
            B.ULSLoadVAE().load("v.safetensors")
            need(False, "N4 Load VAE raised nothing")
        except ValueError as ex:
            need("update ComfyUI" in str(ex) and "Load VAE" in str(ex), "N4 Load VAE: invalid VAE -> plain sentence")
        except Exception as ex:
            need(False, "N4 Load VAE raised %s: %s" % (type(ex).__name__, ex))
    finally:
        folder_paths.get_full_path_or_raise, comfy.utils.load_torch_file, comfy.sd.VAE = old_raise, old_ltf, old_vae

    # N5
    src = open(os.path.join(ROOT, "nodes", "ph_basics.py"), encoding="utf-8").read()
    need(src.count('raise newer_than_core_error("Load CLIP"') == 1, "N5 Load CLIP carries the wrap")
    need(src.count('raise newer_than_core_error("Load Model"') == 2, "N5 Load Model wraps diffusion AND checkpoint")
    need(src.count("if _is_oom(e):") == 4, "N5 every wrap lets out-of-memory through")

    # N6 (Image Reference plan) is not part of the public build.

    # N7
    raw = open(os.path.abspath(__file__), "rb").read()
    need(all(b < 128 for b in raw), "N7 %s is pure ASCII" % os.path.basename(__file__))

    if FAILS:
        print("[%s] FAIL -- %d" % (NAME, len(FAILS)))
        return 1
    print("[%s] PASS -- loaders name a model family newer than the Core; the public cut" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
