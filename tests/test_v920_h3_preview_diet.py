"""v920 guard -- the taeh3 preview pays for a tile, not for a film.

  P1  _h3_preview_latent_size: unchanged when the output fits 1024 px,
      scaled (aspect kept, >= 2) when it would not; Frank's 84x48 -> 64x37
      (1024x592 output instead of 1344x768)
  P2  _h3_tae_build caches per (path, device): two previewers share ONE
      decoder object; CPU build stays fp32, the CUDA branch calls .half()
  P3  decode_latent_to_preview on a random-init decoder: a 24x1x84x48 latent
      yields a 1024x592 image (the cap is applied), a 24x1x8x8 one 128x128
      (below the cap, untouched)
  P4  MODE_TAE_H3 sets max_frames = TAE_H3_MAX_FRAMES (8) in BOTH constructors
      and the other TAE modes keep 16
  P5  tae_warm: route /pls/sampler/tae_warm is registered off-loop and the JS
      fires it exactly when taeh3 is present; tae_warm on a WAN name says
      "nothing to warm"; on taeh3 with a real file it returns a "warm" status
      (driven here on CPU with a random-init stand-in via the cache)
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NODES = os.path.join(ROOT, "nodes")
sys.path.insert(0, NODES)

import torch  # noqa: E402
from PIL import Image  # noqa: E402


def _fail(msg):
    print("[test_v920_h3_preview_diet] FAIL -- " + msg)
    sys.exit(1)


src = open(os.path.join(NODES, "uls_sampler.py"), encoding="utf-8").read()
blk = re.search(r"H3_PREVIEW_MAX_EDGE = .*?\n(?=TAE_REGISTRY = \{)", src, re.S)
if not blk:
    _fail("v920 block not found")
ns = {"torch": torch, "Image": Image, "time": __import__("time"), "__name__": "lifted",
      "_strip_mode": lambda v: str(v), "tae_lookup": lambda n: None}
# _joint_video_half lives above the block; lift it too
jv = re.search(r"def _joint_video_half\(x0\):.*?\n    return x0\n", src, re.S)
exec(compile(jv.group(0) + "\n" + blk.group(0), "<lifted v920>", "exec"), ns)

# P1
f = ns["_h3_preview_latent_size"]
if f(84, 48) != (64, 37):
    _fail("P1 84x48 -> %r, expected (64, 37)" % (f(84, 48),))
if f(48, 84) != (37, 64):
    _fail("P1 aspect must be kept both ways")
if f(40, 40) != (40, 40):
    _fail("P1 a latent that fits must pass untouched")
if f(64, 64) != (64, 64) or f(65, 64) == (65, 64):
    _fail("P1 cap boundary at 64 latent px (1024 output)")
if min(f(300, 2)) < 2:
    _fail("P1 floor of 2 latent px")
if ns["TAE_H3_MAX_FRAMES"] != 8 or ns["H3_PREVIEW_MAX_EDGE"] != 1024:
    _fail("P1 constants moved")

# P2 -- swap the vendor import for a random-init stand-in through the cache
from vendor.taehv import TAEHV, apply_model_with_memblocks  # noqa: E402

built = []


def _fake_build(path, device=None):
    key = (str(path), str(device))
    hit = ns["_H3_TAE_CACHE"].get(key)
    if hit is not None:
        return hit
    tae = TAEHV(None, arch_name="taeh3").eval()
    built.append(key)
    ns["_H3_TAE_CACHE"][key] = (tae, apply_model_with_memblocks)
    return ns["_H3_TAE_CACHE"][key]


real_build_src = re.search(r"def _h3_tae_build\(path, device=None\):.*?\n    return _H3_TAE_CACHE\[key\]\n", src, re.S).group(0)
if "tae = tae.half()" not in real_build_src or 'startswith("cuda")' not in real_build_src:
    _fail("P2 the CUDA branch must .half() the decoder")
if "_H3_TAE_CACHE.get(key)" not in real_build_src:
    _fail("P2 build must consult the process cache")
ns["_h3_tae_build"] = _fake_build
P = ns["_H3TaePreviewer"]
p1 = P("x.safetensors", None)
p2 = P("x.safetensors", None)
if p1.tae is not p2.tae or built != [("x.safetensors", "None")]:
    _fail("P2 two previewers must share one cached decoder, built once (%r)" % built)
if next(p1.tae.decoder.parameters()).dtype != torch.float32:
    _fail("P2 CPU build must stay fp32")

# P3
img = p1.decode_latent_to_preview(torch.randn(1, 24, 3, 84, 48))
if img.size != (592, 1024):
    _fail("P3 84x48 latent must decode to a 1024x592 preview, got %s" % (img.size,))
img2 = p1.decode_latent_to_preview(torch.randn(1, 24, 1, 8, 8))
if img2.size != (128, 128):
    _fail("P3 a small latent must not be touched, got %s" % (img2.size,))

# P4
if src.count("if preview_mode == self.MODE_TAE_H3:\n            self.max_frames = TAE_H3_MAX_FRAMES") != 2:
    _fail("P4 both constructors must cap the H3 frames")
for pat in ("self.max_frames = 16 if (self.use_tae or self.use_core) else max_frames",
            "self.max_frames = 16 if (self.use_tae or self.use_core) else self._max_frames_base"):
    if pat not in src:
        _fail("P4 the generic 16 for the WAN TAE modes must stay")

# P5
tw = ns["tae_warm"]
if "nothing to warm" not in tw("taew2_1"):
    _fail("P5 a WAN decoder must not be warmed here")
if "not in models/vae_approx" not in tw("taeh3"):
    _fail("P5 missing file must be reported, not raised")
ns["tae_lookup"] = lambda n: "x.safetensors"
tw = ns["tae_warm"]
st = tw("taeh3")
if not st.startswith("taeh3 warm on"):
    _fail("P5 warm status: %r" % st)
if built != [("x.safetensors", "None")]:
    _fail("P5 warm-up must reuse the cached decoder (%r)" % built)

# public build (v374): the sampler routes live in ph_sampler_routes.py (the
# public-only module, since v365); the table there is column-aligned.
_rp = os.path.join(NODES, "ph_sampler_routes.py")
rt = open(_rp if os.path.exists(_rp) else os.path.join(NODES, "uls_routes.py"), encoding="utf-8").read()
if not re.search(r'\("POST",\s*"/pls/sampler/tae_warm",\s*handle_sampler_tae_warm\)', rt):
    _fail("P5 route not registered")
h = re.search(r"async def handle_sampler_tae_warm\(.*?\n\n\n", rt, re.S).group(0)
if "run_in_executor(None, tae_warm, name)" not in h or "get_running_loop()" not in h:
    _fail("P5 warm-up must run off-loop through get_running_loop")
js = open(os.path.join(ROOT, "web", "js", "uls_sampler.js"), encoding="utf-8").read()
m = re.search(r'if \(st && st\.ok && st\.found && tae === "taeh3"\) \{(.*?)\n        \}', js, re.S)
if not m or '_samplerFetch("/pls/sampler/tae_warm", { name: tae })' not in m.group(1):
    _fail("P5 JS must fire tae_warm exactly when taeh3 is present")

print("[test_v920_h3_preview_diet] OK - latent capped to a 1024 px preview "
      "(84x48 -> 64x37), decoder cached per (path, device) and fp16 on CUDA, "
      "8 frames for H3 / 16 for WAN kept, warm-up route off-loop + JS trigger "
      "(5 pins)")
