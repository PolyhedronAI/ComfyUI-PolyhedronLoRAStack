"""test_v1246_save_mask_polarity -- Save reads a wired mask in the convention it is told.

v1246 (Frank 30.10., Ritter-Test RT-1): Pixelize's mask (white = visible, the Cutout convention) was
wired into Save, which reads a mask white = transparent (Load Image) -- the sprite came out with its
alpha inverted (figure clear, background black). Save gains `mask_polarity`, APPENDED after
save_output; the default keeps the old reading, so every older workflow saves exactly as before.

  P1 white = visible: a wired mask becomes the alpha as it is (driven, PNG read back)
  P2 the default (white = transparent): alpha = 1 - mask, as before v1246 (driven)
  P3 a mask saved ALONE is written as it came, whatever the polarity (driven)
  P4 an old workflow's value ("" from the shifted slot) validates and saves with the default (driven)
  P5 mask_polarity is the LAST required widget, defaults to the old reading; ph_save.js resets an
     unknown value on load (source pins)
  P6 this file is pure ASCII
"""
import importlib.util
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NAME = "test_v1246_save_mask_polarity"
FAILS = []


def need(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        FAILS.append(msg)


def load_pkg(mod):
    pkg = "pls_v1246_nodes"
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
        sys.argv = [sys.argv[0], "--cpu"]
    try:
        import comfy.options
        comfy.options.enable_args_parsing()
        import folder_paths
        import numpy as np
        import torch
        from PIL import Image
    except Exception as e:
        print("[%s] SKIP -- the Core is not importable (%s)" % (NAME, e))
        return 0
    S = load_pkg("ph_save")
    out = tempfile.mkdtemp()
    folder_paths.set_output_directory(out)
    img = torch.rand((1, 8, 8, 3))
    msk = torch.zeros((1, 8, 8))
    msk[0, 2:6, 2:6] = 1.0                       # white = the figure

    def save(prefix, **kw):
        node = S.ULSSave()
        args = dict(filename_prefix=prefix, media_kind="auto", image_format="png", image_quality=100,
                    video_preset="H.264 MP4 (delivery)", quality=17, frame_rate=24.0, autoplay=False,
                    pingpong=False, loop_count=0, trim_to_audio=False, save_metadata=False, save_output=True)
        args.update(kw)
        node.save(**args)
        files = sorted(f for f in os.listdir(out) if f.startswith(prefix) and f.endswith(".png"))
        return np.array(Image.open(os.path.join(out, files[-1])))

    a1 = save("p1", image=img, mask=msk, mask_polarity=S.MASK_VISIBLE_WHITE)
    need(a1.shape == (8, 8, 4) and a1[3, 3, 3] == 255 and a1[0, 0, 3] == 0, "P1 white = visible -> the figure is opaque, the rest clear")
    a2 = save("p2", image=img, mask=msk)
    need(a2.shape == (8, 8, 4) and a2[3, 3, 3] == 0 and a2[0, 0, 3] == 255, "P2 default -> alpha = 1 - mask (unchanged)")
    a2b = save("p2b", image=img, mask=msk, mask_polarity=S.MASK_TRANSPARENT_WHITE)
    need(np.array_equal(a2, a2b), "P2 the named default equals no value at all")
    a3 = save("p3", mask=msk, mask_polarity=S.MASK_VISIBLE_WHITE)
    need(a3.ndim == 2 and a3[3, 3] == 255 and a3[0, 0] == 0, "P3 a mask saved alone is written as it came")
    need(S.ULSSave.VALIDATE_INPUTS(mask_polarity="") is True, "P4 an old workflow's '' validates")
    a4 = save("p4", image=img, mask=msk, mask_polarity="")
    need(np.array_equal(a4, a2), "P4 '' saves with the old default")

    req = list(S.ULSSave.INPUT_TYPES()["required"].keys())
    spec = S.ULSSave.INPUT_TYPES()["required"]["mask_polarity"]
    need(req[-1] == "mask_polarity" and req[-2] == "save_output" and spec[1]["default"] == S.MASK_TRANSPARENT_WHITE
         and spec[0][0] == S.MASK_TRANSPARENT_WHITE, "P5 appended last, default = the old reading")
    js = open(os.path.join(ROOT, "web", "js", "ph_save.js"), encoding="utf-8").read()
    need('w.name === "mask_polarity"' in js and "!vals.includes(mp.value)) mp.value = vals[0]" in js,
         "P5 ph_save.js resets an unknown value on load")

    raw = open(os.path.abspath(__file__), "rb").read()
    need(all(c < 128 for c in raw), "P6 %s is pure ASCII" % os.path.basename(__file__))
    if FAILS:
        print("[%s] FAIL -- %d" % (NAME, len(FAILS)))
        return 1
    print("[%s] PASS -- Save reads the mask as told; the default keeps every older workflow as it was" % NAME)
    return 0


if __name__ == "__main__":
    sys.exit(main())
