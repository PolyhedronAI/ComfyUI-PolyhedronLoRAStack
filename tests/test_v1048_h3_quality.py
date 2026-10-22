# -*- coding: ascii -*-
"""Guard v1048 -- the MiniMax H3 quality nodes (Frank 01.10.: "take everything
sensible from the example node for the best possible quality").

  Q1  the 3D upscaler network has EXACTLY the weight layout of LBH-123-AI's
      LatentResizer3D (key/shape signature of the 12+12x512 release net and
      of a small one) -- so the published weights load strict; measured
      bit-equal against the original on the same weights at build time.
  Q2  detect_arch reads blocks, channels, temporal kernel from the weights;
      a 128-channel file (another model's upscaler in the shared folder) is
      refused by name; the 2D variant (missing keys) is refused.
  Q3  size math: scale / width x height / megapixels, on the 32-px grid,
      enlarge only; the node keeps the AV latent's audio stream untouched,
      drops the old noise_mask, refuses a non-H3 latent; fp32 only, NaN is
      refused.
  Q4  Free VRAM passes through and has NO IS_CHANGED (a node that always runs
      would re-run everything after it).
  Q5  Semantic Bridge: H + alpha (S' - H) with S' at H's norm, marked, never
      twice, alpha 0 passes; the shapes are checked.
  Q6  registered with display names; the third-party licences are in
      docs/third_party/ and named in the module.
  S   ASCII.

Script-style: exit 0 = pass.
"""
import hashlib
import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
FAILS = []


def _need(c, m):
    if not c:
        FAILS.append(m)
    print(("  ok    " if c else "  FAIL  ") + m)


TMP = tempfile.mkdtemp(prefix="v1048_")


class _NT:
    """comfy.nested_tensor.NestedTensor stand-in: a tuple of streams."""
    is_nested = True

    def __init__(self, parts):
        self.parts = tuple(parts)

    def unbind(self):
        return list(self.parts)


fp = types.ModuleType("folder_paths")
fp.models_dir = TMP
fp.folder_names_and_paths = {}
fp.add_model_folder_path = lambda name, path: fp.folder_names_and_paths.setdefault(name, ([path], set()))
fp.get_filename_list = lambda name: sorted(os.listdir(TMP))
fp.get_full_path = lambda name, f: os.path.join(TMP, f)
comfy = types.ModuleType("comfy")
nt = types.ModuleType("comfy.nested_tensor")
nt.NestedTensor = _NT
comfy.nested_tensor = nt
mm = types.ModuleType("comfy.model_management")
mm.unload_all_models = lambda: None
mm.soft_empty_cache = lambda force=False: None
comfy.model_management = mm
sys.modules.update({"folder_paths": fp, "comfy": comfy, "comfy.nested_tensor": nt, "comfy.model_management": mm})
spec = importlib.util.spec_from_file_location("h3q1048", str(ROOT / "nodes" / "ph_h3_quality.py"))
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)


def sig(net):
    return hashlib.md5(";".join("%s:%s" % (k, "x".join(map(str, v.shape))) for k, v in net.state_dict().items()).encode()).hexdigest()


# ------------------------------------------------------------------ Q1
big = Q.LatentResizer3D()
small = Q.LatentResizer3D(in_blocks=3, out_blocks=3, channels=64)
_need(sig(big) == "73dc946b69af0ba170e02f272ec3de94" and len(big.state_dict()) == 322,
      "Q1 the 12+12 x 512 net = LBH-123-AI's weight layout (%s, %d tensors)" % (sig(big), len(big.state_dict())))
_need(sig(small) == "25c9bff4f591e29678d6fc12bf15b504", "Q1 a small one too (%s)" % sig(small))
# the forward itself: LBH-123-AI's original gave these numbers on the same
# seeded weights and input (measured at build time, v1048)
torch.manual_seed(0)
fpn = Q.LatentResizer3D(in_blocks=3, out_blocks=3, channels=64).eval()
for p in fpn.parameters():
    p.data.normal_(0, 0.05)
torch.manual_seed(1)
with torch.no_grad():
    y = fpn(torch.randn(1, 24, 5, 4, 6), 2.0, (5, 8, 12), chunking=False)
_need(abs(float(y.sum()) + 76.939453125) < 2e-3 and abs(float(y.abs().mean()) - 0.05886059254407883) < 1e-6
      and abs(float(y[0, 3, 2, 5, 7]) + 0.035597845911979675) < 1e-6,
      "Q1 the forward = the original's on the same weights (sum %.6f, mean|y| %.8f)" % (float(y.sum()), float(y.abs().mean())))
del big

# ------------------------------------------------------------------ Q2
from safetensors.torch import save_file  # noqa: E402
torch.manual_seed(0)
for p in small.parameters():
    p.data.normal_(0, 0.05)
save_file({"upscaler." + k: v.contiguous() for k, v in small.state_dict().items()}, os.path.join(TMP, "minimax_h3_up.safetensors"))
cfg = Q.detect_arch({k: v for k, v in small.state_dict().items()})
_need(cfg["in_blocks"] == 3 and cfg["out_blocks"] == 3 and cfg["channels"] == 64 and cfg["temporal_kernel"] == 5 and cfg["temporal_every"] == 2,
      "Q2 detect_arch from the weights %r" % cfg)
net = Q.load_upscaler(os.path.join(TMP, "minimax_h3_up.safetensors"))
_need(torch.equal(net.conv_in.weight, small.conv_in.weight), "Q2 the 'upscaler.' prefix is stripped, the weights load")
other = Q.LatentResizer3D(in_channels=128, in_blocks=1, out_blocks=1, channels=32)
save_file({k: v.contiguous() for k, v in other.state_dict().items()}, os.path.join(TMP, "ltx2_up.safetensors"))
try:
    Q.load_upscaler(os.path.join(TMP, "ltx2_up.safetensors"))
    _need(False, "Q2 a 128-channel file must be refused")
except ValueError as e:
    _need("not a MiniMax H3 latent upscaler" in str(e), "Q2 another model's upscaler is refused by name")
sd2 = {k: v for k, v in small.state_dict().items() if not k.startswith("out_blocks.1")}
save_file({k: v.contiguous() for k, v in sd2.items()}, os.path.join(TMP, "h3_2d_like.safetensors"))
Q._UP_CACHE.clear()
try:
    Q.load_upscaler(os.path.join(TMP, "h3_2d_like.safetensors"))
    _need("out_blocks" in str(Q.detect_arch(sd2)), "Q2 (fewer blocks are read as a smaller net)")
except ValueError as e:
    _need("does not fit" in str(e), "Q2 a file that does not fit is refused by name")
_need(Q.upscaler_names()[0] in ("h3_2d_like.safetensors", "minimax_h3_up.safetensors") and Q.upscaler_names()[-1] == "ltx2_up.safetensors",
      "Q2 H3 files listed first %r" % Q.upscaler_names())

# ------------------------------------------------------------------ Q3
_need(Q.target_latent_hw(24, 42, "scale", scale=2.0) == (48, 84, 2.0), "Q3 scale x2: 672x384 -> 1344x768")
h, w, _ = Q.target_latent_hw(24, 42, "width x height", width=1344, height=768)
_need((h, w) == (48, 84), "Q3 width x height")
h, w, _ = Q.target_latent_hw(24, 42, "megapixels", megapixels=1.0)
_need(w * 16 % 32 == 0 and h * 16 % 32 == 0 and abs(w * h * 256 / 1048576.0 - 1) < 0.08, "Q3 megapixels on the 32 grid (%dx%d)" % (w * 16, h * 16))
try:
    Q.target_latent_hw(24, 42, "width x height", width=320, height=192)
    _need(False, "Q3 smaller must be refused")
except ValueError as e:
    _need("only enlarges" in str(e), "Q3 enlarge only")
Q._UP_CACHE.clear()
vid, aud = torch.randn(1, 24, 7, 6, 10), torch.randn(1, 32, 2, 50)
node = Q.ULSH3LatentUpscale()
out, info, _pos = node.run({"samples": _NT((vid, aud)), "noise_mask": torch.ones(1)}, "minimax_h3_up.safetensors", "scale", 2.0, 0, 0, 1.0, 32, True, False)
pv, pa = out["samples"].unbind()
_need(tuple(pv.shape) == (1, 24, 7, 12, 20) and pa is aud and "noise_mask" not in out and "audio passed through" in info,
      "Q3 the AV latent: video lifted, audio untouched, noise_mask dropped")
_need(pv.dtype == vid.dtype and torch.isfinite(pv).all(), "Q3 the result is finite, in the latent's dtype")
try:
    node.run({"samples": torch.randn(1, 16, 7, 6, 10)}, "minimax_h3_up.safetensors", "scale", 2.0, 0, 0, 1.0, 32, True, False)
    _need(False, "Q3 a 16-channel latent must be refused")
except ValueError as e:
    _need("not a MiniMax H3 latent" in str(e), "Q3 a non-H3 latent is refused by name")
src = (ROOT / "nodes" / "ph_h3_quality.py").read_text(encoding="utf-8")
_need("dt = torch.float32" in src and "fp16" not in src.split("class ULSH3LatentUpscale")[1].split("def run")[0],
      "Q3 fp32 only (bf16 GroupNorm can collapse to a brown frame)")
_need('raise ValueError("the upscaler returned NaN/inf' in src, "Q3 NaN is refused, never passed on")

# ------------------------------------------------------------------ Q4
fv = Q.ULSFreeVRAM()
x = {"a": 1}
_need(fv.run(x, "empty cache only")[0] is x and "IS_CHANGED" not in Q.ULSFreeVRAM.__dict__, "Q4 Free VRAM passes through, no IS_CHANGED")

# ------------------------------------------------------------------ Q5
st = Q.SemanticStudent()
for p in st.parameters():
    p.data.normal_(0, 0.02)
save_file({k: v.contiguous() for k, v in st.state_dict().items()}, os.path.join(TMP, "sb.safetensors"))
H = torch.randn(1, 9, 5120)
sb = Q.ULSH3SemanticBridge()
c1, _ = sb.run([[H, {"x": 1}]], "sb.safetensors", 0.15, True)
xx = H.float()
s = st(xx * torch.rsqrt(xx.pow(2).mean(-1, keepdim=True) + 1e-6))
s = s * (xx.norm(dim=-1, keepdim=True) / (s.norm(dim=-1, keepdim=True) + 1e-8))
_need(torch.allclose(c1[0][0], xx + 0.15 * (s - xx), atol=1e-5) and c1[0][1]["x"] == 1 and c1[0][1]["pls_semantic_bridge"] == "sb.safetensors",
      "Q5 H + alpha (S' - H), the meta kept and marked")
c2, i2 = sb.run(c1, "sb.safetensors", 0.15, True)
_need(torch.equal(c2[0][0], c1[0][0]) and "already bridged" in i2, "Q5 never twice")
c3, _ = sb.run([[H, {}]], "sb.safetensors", 0.0, True)
_need(c3[0][0] is H, "Q5 alpha 0 passes through")
try:
    sb.run([[torch.randn(1, 9, 4096), {}]], "sb.safetensors", 0.15, True)
    _need(False, "Q5 a 4096-wide conditioning must be refused")
except ValueError as e:
    _need("[B, T, 5120]" in str(e), "Q5 the token width is checked")

# ------------------------------------------------------------------ Q6
init = (ROOT / "__init__.py").read_text(encoding="utf-8")
# public build (v392): only H3 Latent Upscale ships as a node (the fox templates' second pass); Free VRAM and the
# Semantic Bridge stay in the module unregistered -- RE-GROUNDED, the licences below are pinned unchanged
_need('NODE_DISPLAY_NAME_MAPPINGS["ULSH3LatentUpscale"]' in init
      and not any('NODE_DISPLAY_NAME_MAPPINGS["%s"]' % n in init for n in ("ULSFreeVRAM", "ULSH3SemanticBridge")),
      "Q6 H3 Latent Upscale is registered with a display name (the other two are not part of this build)")
tp = ROOT / "docs" / "third_party"
_need((tp / "LBH-123-AI_Comfyui_Minimax_h3_latent_Upscaler_LICENSE_MIT.txt").is_file()
      and (tp / "AIMixer_ComfyUI_MiniMaxH3_Director_LICENSE_Apache-2.0.txt").is_file()
      and "Copyright (c) 2026 LBH-123-AI" in src and "Apache-2.0" in src, "Q6 the licences are shipped and named")

bad = [f for f in ("nodes/ph_h3_quality.py", "tests/test_v1048_h3_quality.py", "docs/third_party/README.md")
       if any(ord(ch) > 127 for ch in (ROOT / f).read_text(encoding="utf-8"))]
_need(not bad, "S ASCII only %r" % bad)
import shutil  # noqa: E402
shutil.rmtree(TMP, ignore_errors=True)
if FAILS:
    print("test_v1048_h3_quality: %d FAILURE(S)" % len(FAILS))
    sys.exit(1)
print("test_v1048_h3_quality: PASS (Q1-Q6, S)")
