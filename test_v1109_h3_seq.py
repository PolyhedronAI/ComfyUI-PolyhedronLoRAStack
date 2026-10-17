# -*- coding: ascii -*-
"""Guard v1109 -- an H3 LoRA in the Diffusers layout lands in a SEQ row too
(Frank's go, 12.10.2026, on the v1108 find: the v930 converter ran in the merge,
bypass, bake and inspector paths, never in SEQ -- such a LoRA alone in a SEQ
row matched nothing).

  H1  SEQ_CONVERTERS holds BOTH entries: the H3 one and the LM one.
  H2  static: the SEQ loader's preparation passes the file's metadata on (the
      H3 files carry their alpha only as a header ratio).
  H3  (core) the real H3 module tree from core (small config, meta device):
      a Diffusers-layout H3 LoRA (2 DiT blocks + 1 token-refiner block, q/k/v,
      to_out, ff, adaln) lands 0 modules as saved and ALL after _seq_prepare,
      every one on the model; qkv fused, SwiGLU halves swapped (the v930
      identity and convention, pinned there); a header lora_alpha read from
      the FILE becomes a per-module alpha; a dict core already understands
      (native H3 names) stays the SAME object (parity with core's loader).

Script-style: exit 0 = pass.
"""
import importlib
import os
import sys
import tempfile
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
NODES = ROOT / "nodes"
sys.path.insert(0, str(NODES))
import uls_lora_convert as C  # noqa: E402

FAILS = []


def _need(c, m):
    print(("  ok    " if c else "  FAIL  ") + m)
    if not c:
        FAILS.append(m)


_need("MiniMax-H3 (Diffusers)" in C.SEQ_CONVERTERS, "H1 the H3 entry runs in SEQ")
_need("LM adapter (PEFT, language_model.)" in C.SEQ_CONVERTERS, "H1 the LM entry still does")
src = (NODES / "uls_stack_node.py").read_text(encoding="utf-8")
_need("meta = safetensors_metadata(path) if path else {}" in src
      and "convert_foreign_lora(lora, metadata=meta, only=SEQ_CONVERTERS" in src,
      "H2 SEQ's preparation hands the file's metadata to the converters")

HID, HEADS, HD, FFN, TD = 64, 2, 32, 96, 48
INNER = HEADS * HD


def h3_diffusers(blocks=2, refiner=1, r=2):
    g = torch.Generator().manual_seed(1109)
    sd = {}

    def mod(name, out, inp):
        sd[name + ".lora_A.weight"] = torch.randn(r, inp, generator=g)
        sd[name + ".lora_B.weight"] = torch.randn(out, r, generator=g)

    for prefix, n, adaln in (("transformer_blocks.%d.", blocks, True), ("token_refiner.refiner_blocks.%d.", refiner, False)):
        for i in range(n):
            p = prefix % i
            for q in ("to_q", "to_k", "to_v"):
                mod(p + "attn." + q, INNER, HID)
            mod(p + "attn.to_out.0", HID, INNER)
            mod(p + "ff.net.0.proj", FFN * 2, HID)
            mod(p + "ff.net.2", HID, FFN)
            if adaln:
                mod(p + "adaln_proj.linear", 6 * HID * 3, TD)
    return sd


def _core_root():
    env = os.environ.get("PLS_CORE_ROOT")
    for c in ([env] if env else []) + [str(ROOT.parent.parent)]:
        if c and os.path.isfile(os.path.join(c, "comfy", "ldm", "minimax", "model.py")):
            return c
    return None


CORE = _core_root()
if not CORE:
    print("  skip  H3 (no core with MiniMax-H3 reachable -- set PLS_CORE_ROOT)")
else:
    sys.path.insert(0, CORE)
    import comfy.cli_args  # noqa: E402
    comfy.cli_args.args.cpu = True
    import torch.nn as nn  # noqa: E402
    import comfy.ops  # noqa: E402
    import comfy.supported_models  # noqa: E402
    from comfy.ldm.minimax.model import MiniMaxH3Model  # noqa: E402
    from safetensors.torch import save_file  # noqa: E402

    pkg_name = "pls1109pack"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(ROOT)]
    sys.modules[pkg_name] = pkg
    sub = types.ModuleType(pkg_name + ".nodes")
    sub.__path__ = [str(NODES)]
    sys.modules[pkg_name + ".nodes"] = sub
    S = importlib.import_module(pkg_name + ".nodes.uls_stack_node")

    class _Base(nn.Module):
        def __init__(s):
            super().__init__()
            s.diffusion_model = MiniMaxH3Model(
                hidden_size=HID, num_layers=2, token_refiner_num_layers=1, num_attention_heads=HEADS,
                attention_head_dim=HD, ffn_hidden_size=FFN, text_dim=HID, time_embed_hidden_size=HID,
                time_embed_dim=TD, dtype=torch.bfloat16, device="meta", operations=comfy.ops.disable_weight_init)
            s.model_config = comfy.supported_models.MiniMaxH3({"image_model": "minimax_h3"})

    model = types.SimpleNamespace(model=_Base())
    raw = h3_diffusers()
    n_mod = len({k.rsplit(".lora_", 1)[0] for k in raw})
    r0 = S._landing(raw, model, None, 1.0, 1.0)
    _need(r0["landed"] == 0 and r0["modules"] == n_mod,
          "H3 as saved: %d/%d land (the hole SEQ had until v1109)" % (r0["landed"], r0["modules"]))
    tmp = tempfile.mkdtemp(prefix="pls1109_")
    path = os.path.join(tmp, "h3_diffusers.safetensors")
    save_file(raw, path, metadata={"lora_alpha": "4"})
    prep, notes = S._seq_prepare(raw, path, model, None)
    r1 = S._landing(prep, model, None, 1.0, 1.0, notes)
    fused_modules = {k.rsplit(".lora_", 1)[0] for k in prep if ".lora_" in k}
    _need(r1["landed"] == r1["modules"] == len(fused_modules) and r1["on_model"] == r1["landed"] and r1["landed"] > 0,
          "H3 after _seq_prepare: %d/%d land, all on the model" % (r1["landed"], r1["modules"]))
    _need(len(fused_modules) == 2 * 5 + 1 * 4,
          "H3 module count after the fusion: 2 blocks x 5 + 1 refiner x 4 = %d" % len(fused_modules))
    _need(any("converted: MiniMax-H3 (Diffusers)" in n for n in notes), "H3 the conversion is named in the notes")
    qkv = "diffusion_model.blocks.0.attn.qkv_proj"
    _need(prep[qkv + ".lora_A.weight"].shape[0] == 3 * 2 and prep[qkv + ".lora_B.weight"].shape == (3 * INNER, 3 * 2),
          "H3 q/k/v fused into qkv_proj (rank 3r, block-diagonal B)")
    b_src = raw["transformer_blocks.0.ff.net.0.proj.lora_B.weight"]
    b_dst = prep["diffusion_model.blocks.0.mlp.fc1.lora_B.weight"]
    _need(torch.equal(b_dst[:FFN], b_src[FFN:]) and torch.equal(b_dst[FFN:], b_src[:FFN]),
          "H3 fc1 rows swapped (Diffusers [value; gate] -> core gate first)")
    _need(float(prep["diffusion_model.blocks.0.mlp.fc2.alpha"]) == 4.0,
          "H3 the header lora_alpha from the FILE became a per-module alpha")
    native = {k: v for k, v in prep.items()}
    same, _n = S._seq_prepare(native, None, model, None)
    _need(same is native, "H3 a dict core already understands stays the SAME object")

if FAILS:
    print("[v1109] FAIL -- %d check(s)" % len(FAILS))
    sys.exit(1)
print("[v1109] PASS -- H3 Diffusers LoRAs land in SEQ rows too")
sys.exit(0)
