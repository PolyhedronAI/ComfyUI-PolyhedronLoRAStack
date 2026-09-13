"""v918 guard -- MiniMax H3 live preview through the vendored TAEHV.

  P1  nodes/vendor/taehv/taehv.py is the verbatim madebyollin file (md5 pin),
      LICENSE (MIT) and SOURCE.md travel with it
  P2  TAE_REGISTRY["taeh3"]: file, raw.github url, sha256 + bytes as
      measured in the sandbox 05.09.
  P3  the mode is wired everywhere it must be: MODE_TAE_H3 in ALL_MODES,
      TAE_MODES, TAE_NAME_OF -> "taeh3", and the INPUT_TYPES combo carries
      a _mode_entry for it; the v885 joint line names the new mode
  P4  _H3TaePreviewer contract on a RANDOM-init taeh3 decoder (no weights
      needed): 1x24x1x8x8 latent -> a PIL image of 128x128 (patch 2, x16);
      a joint (nested) latent is unwrapped to its video half first
  P5  uls_sampler.js: _checkTaePreview recognises taeh3 (and still sends
      lighttaew2_1 before taew2_1)
"""
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODES = os.path.join(ROOT, "nodes")
sys.path.insert(0, NODES)


def _fail(msg):
    print("[test_v918_taeh3_preview] FAIL -- " + msg)
    sys.exit(1)


# P1
vd = os.path.join(NODES, "vendor", "taehv")
for f in ("taehv.py", "__init__.py", "LICENSE", "SOURCE.md"):
    if not os.path.exists(os.path.join(vd, f)):
        _fail("P1 vendor file missing: " + f)
md5 = hashlib.md5(open(os.path.join(vd, "taehv.py"), "rb").read()).hexdigest()
if md5 != "9dbf026ac16cdf1a50947efde693f63f":
    _fail("P1 vendored taehv.py changed (md5 %s) -- it must stay verbatim" % md5)
if "MIT License" not in open(os.path.join(vd, "LICENSE"), encoding="utf-8").read():
    _fail("P1 LICENSE is not the MIT text")

# P2 + P3 (source-level: uls_sampler imports comfy at module level)
src = open(os.path.join(NODES, "uls_sampler.py"), encoding="utf-8").read()
m = re.search(r'"taeh3": \{(.*?)\n    \},', src, re.S)
if not m:
    _fail("P2 TAE_REGISTRY has no taeh3 entry")
ent = m.group(1)
for want in ('"file": "taeh3.safetensors"',
             'madebyollin/taehv/"', 'main/safetensors/taeh3.safetensors',
             '"4fd022bfcab08772fe0536b17ea1a3bb"', '"b5625be11e397868d1c5d891863d4c13"',
             '"bytes": 22709752'):
    if want not in ent:
        _fail("P2 registry entry lacks " + want)

for want in ('MODE_TAE_H3 = "Video · TAE (taeh3)"',
             'TAE_MODES = (MODE_TAE_STD, MODE_TAE_LIGHT, MODE_TAE_H3)',
             'MODE_TAE_H3: "taeh3"',
             'MODE_TAE_STD, MODE_TAE_LIGHT, MODE_TAE_H3)',
             '_mode_entry("Video · TAE (taeh3)", "taeh3")',
             'if self.tae_name == "taeh3":',
             'self._tae = _H3TaePreviewer(path, self.device)',
             "'Video \\u00b7 TAE (taeh3)' (v918)"):
    if want not in src:
        _fail("P3 sampler source lacks: " + want)
if src.count("self.use_tae = preview_mode in self.TAE_MODES") != 2:
    _fail("P3 both __init__ and set_mode must derive use_tae from TAE_MODES")
if src.count('self.tae_name = self.TAE_NAME_OF.get(preview_mode, "taew2_1")') != 2:
    _fail("P3 both __init__ and set_mode must derive tae_name from TAE_NAME_OF")

# P4 -- lift the two helpers and run them on a random-init decoder
import torch  # noqa: E402
from PIL import Image  # noqa: E402

blk = re.search(r"def _joint_video_half\(x0\):.*?return Image\.fromarray\(arr\)\n", src, re.S)
if not blk:
    _fail("P4 could not lift _joint_video_half/_H3TaePreviewer")
ns = {"torch": torch, "Image": Image, "__name__": "lifted"}
exec(compile(blk.group(0), "<lifted v918>", "exec"), ns)


class _P(ns["_H3TaePreviewer"]):
    def __init__(self):
        from vendor.taehv import TAEHV, apply_model_with_memblocks
        self._apply = apply_model_with_memblocks
        self.tae = TAEHV(None, arch_name="taeh3").eval()
        self.device = None


p = _P()
if (p.tae.patch_size, p.tae.latent_channels, p.tae.t_upscale) != (2, 24, 4):
    _fail("P4 taeh3 arch from the vendor is not (patch 2, 24 ch, t_up 4)")
lat = torch.randn(1, 24, 3, 8, 8)              # NCTHW, three latent frames
img = p.decode_latent_to_preview(lat)
if not isinstance(img, Image.Image) or img.size != (128, 128) or img.mode != "RGB":
    _fail("P4 decode -> %r %s" % (type(img), getattr(img, "size", None)))


class _Nested:
    is_nested = True

    def __init__(self, v, a):
        self.tensors = (v, a)


nested = _Nested(lat, torch.randn(1, 32, 2, 7))
if ns["_joint_video_half"](nested) is not lat:
    _fail("P4 nested latent must unwrap to tensors[0]")
img2 = p.decode_latent_to_preview(nested)
if img2.size != (128, 128):
    _fail("P4 nested decode -> %s" % (img2.size,))
plain = torch.zeros(1, 2, 3)
if ns["_joint_video_half"](plain) is not plain:
    _fail("P4 a plain tensor must pass through untouched")

# P5
js = open(os.path.join(ROOT, "web", "js", "uls_sampler.js"), encoding="utf-8").read()
m = re.search(r"const tae = bare\.includes\(\"lighttaew2_1\"\) \? \"lighttaew2_1\"\s*\n\s*: bare\.includes\(\"taeh3\"\) \? \"taeh3\" : \"taew2_1\";", js)
if not m:
    _fail("P5 _checkTaePreview does not recognise taeh3 (lighttaew2_1 first)")

print("[test_v918_taeh3_preview] OK - vendor verbatim + MIT, registry pinned "
      "(sha256/bytes), mode wired in both constructors + combo + joint line, "
      "random-init decoder yields a 128x128 RGB preview from 8x8 latent, joint "
      "latent unwrapped, JS knows taeh3 (5 pins)")
