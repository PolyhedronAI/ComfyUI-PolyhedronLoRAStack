# -*- coding: ascii -*-
"""Guard v1164 -- the joint refine's canvas sits on the model's patch grid, so a pinned frame can be patchified.

Frank 18.10., the neon lane, scene 2 (carry) after v1162: the Power Upscale's joint refine died again --
"shape '[1, 24, 1, 1, 58, 2, 33, 2]' is invalid for input of size 188136" in Core's patchify_video
(_cond_video_rows). The refine canvas was snapped /16 (the VAE's factor only): 1075x1882 -> 1072x1872, a latent
of 67x117 -- odd sides. Core pads the noisy latent to the transformer's patch (2), but NOT a pinned frame; the
v1162 re-encode of the pin at 67x117 could not be cut into 2x2 patches. Scene 1 (no pin) never met it.

  G1  _dit_patch reads the transformer's patch from the model (a tuple's spatial part, an int), 1 when absent
  G2  upscale() snaps the joint canvas to VAE x patch; _joint_refine fits to the same grid
  G3  the field case: 1075x1882 on H3 (VAE 16, patch (1, 2, 2)) lands on /32 -- even latent sides
  S   ASCII of this guard
"""
import ast
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = "test_v1164_joint_grid"
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


SRC = (ROOT / "nodes" / "ph_power_upscale.py").read_text(encoding="utf-8")
tree = ast.parse(SRC)


def _seg(name):
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return ast.get_source_segment(SRC, n)
    return ""


ns = {}
exec(_seg("_dit_patch") + "\n\n" + _seg("_vae_spatial") + "\n\n" + _seg("_snap_to"), ns)
patch, vsp, snap_to = ns.get("_dit_patch"), ns["_vae_spatial"], ns["_snap_to"]


def M(p):
    return types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=types.SimpleNamespace(patch_size=p)))


# G1
_need(patch is not None and patch(M((1, 2, 2))) == 2 and patch(M(2)) == 2 and patch(M((1, 1, 1))) == 1
      and patch(types.SimpleNamespace(), 1) == 1 and patch(None, 1) == 1 and patch(M(None), 3) == 3,
      "G1 _dit_patch reads (1,2,2) -> 2, 2 -> 2, nothing -> the default")

# G2
up = _seg("upscale")
jr = _seg("_joint_refine")
_need("_jr_snap *= _dit_patch(model, 1)" in up, "G2 upscale(): the joint canvas grid is VAE x patch")
_need("snap = _vae_spatial(vae, _JOINT_SNAP_DEFAULT) * _dit_patch(model, 1)" in jr,
      "G2 _joint_refine fits the frames to the same grid")

# G3
H3VAE = types.SimpleNamespace(downscale_ratio=(lambda a: a, 16, 16))
g = vsp(H3VAE, 16) * patch(M((1, 2, 2)), 1)
w, h = snap_to(1075, 1882, g)
_need(g == 32 and (w, h) == (1056, 1856) and (w // 16) % 2 == 0 and (h // 16) % 2 == 0,
      "G3 1075x1882 on H3 -> %dx%d on /%d, latent %dx%d (even: a 2x2 patch fits)" % (w, h, g, w // 16, h // 16))
w0, h0 = snap_to(1075, 1882, 16)
_need((w0 // 16) % 2 == 1, "G3 (the field crash, for the record) /16 alone gave latent %dx%d -- odd" % (w0 // 16, h0 // 16))

_need(all(b < 128 for b in Path(__file__).read_bytes()), "S  ASCII")
print("%s: %d check(s) failed" % (NAME, len(FAILS)) if FAILS else "%s: all checks passed" % NAME)
sys.exit(1 if FAILS else 0)
