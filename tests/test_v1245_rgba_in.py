"""test_v1245_rgba_in (public cut: R5 Interpolate + R7 only) -- a four-channel (RGBA) picture never reaches a three-channel network.

v1245 (Frank 30.10., Ritter-Test RT-1): the Qwen-Image 2.1 VAE decodes RGBA; the Cutout handed the
4-channel frame to BiRefNet and died ("shapes (1024,1024,4) (3,)"). The rule now: the network sees
the colour channels, the alpha travels beside it and is put back -- or, in the Cutout, kept in the
mask when it is real (not opaque everywhere).

  R1 the matting engine normalizes only the colour channels (RGBA == RGB result)
  R2 Cutout: an opaque RGBA frame frees exactly like its RGB twin (driven, stand-in selector)
  R3 Cutout: a real input alpha stays transparent in the mask (driven)
  R4 Object Remove: an RGBA picture is filled on RGB, its alpha comes back unchanged (driven, stand-in net)
  R5 Interpolate: the alpha timeline -- sources exact, new frames the linear blend, held pairs copy
  R6 Interpolate splits RGBA before the network and puts the alpha back (source pins);
     Video Matte feeds the colour channels only (source pin)
  R7 this file is pure ASCII
"""
import importlib.util
import io
import os
import sys
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = "test_v1245_rgba_in"
FAILS = []


def need(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def load_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_pkg(mod):
    import types
    pkg = "pls_v1245_nodes"
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
    try:
        import numpy as np
        import torch
    except ImportError:
        print("[%s] SKIP -- numpy/torch absent" % NAME)
        return 0
    if "--cpu" not in sys.argv:
        sys.argv = [sys.argv[0], "--cpu"]          # the Core reads its flags at import
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
    except Exception:
        pass
    sys.path.insert(0, os.path.join(ROOT, "nodes"))
    sys.path.insert(0, HERE)

    # R1
    # R1-R4 (BGR engine, Cutout, Object Remove) are not part of the public build.

    src = open(os.path.join(ROOT, "nodes", "ph_interpolate.py"), encoding="utf-8").read()
    I = load_pkg("ph_interpolate")
    al = torch.tensor([0.0, 1.0, 0.2]).view(3, 1, 1, 1)
    tasks, oc = I._timeline(3, 2)
    out = I._alpha_timeline(al, 2, tasks, ["interp", "hold"], oc)
    vals = [round(float(v), 4) for v in out.view(-1)]
    need(oc == 5 and vals == [0.0, 0.5, 1.0, 1.0, 0.2], "R5 alpha timeline %r" % (vals,))

    # R6
    need("frames, frames_alpha = frames[..., :3].contiguous(), frames[..., 3:4].contiguous()" in src
         and src.count("_alpha_timeline(frames_alpha") == 1
         and src.index("frames_alpha = None") < src.index("chw = frames.permute(0, 3, 1, 2)"),
         "R6 Interpolate splits RGBA before the network and puts the alpha back")
    # R6 (Video Matte) is not part of the public build.

    raw = open(os.path.abspath(__file__), "rb").read()
    need(all(c < 128 for c in raw), "R7 %s is pure ASCII" % os.path.basename(__file__))

    if FAILS:
        print("[%s] FAIL -- %d" % (NAME, len(FAILS)))
        return 1
    print("[%s] PASS -- RGBA in: the networks see colour, the alpha travels beside it (Cutout keeps a real one)" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
