# -*- coding: ascii -*-
"""v935 -- the Core-backed half of test_v935_pdd_wiring.py.

Run as a SUBPROCESS by that guard, only when a ComfyUI core is reachable
(PLS_CORE_ROOT, or the ComfyUI root two levels above the pack). It imports the
pack as ComfyUI does -- by package name -- and drives the PDD wiring against
REAL core classes: ModelPatcher, FinalLayer, WrapperExecutor, comfy.lora, the
k-diffusion Euler. Prints one JSON object on its last line; the guard judges.

Why a subprocess: core parses argv at import (we need --cpu) and registers
global state; none of that may leak into the guard process or other guards.
"""
import importlib
import json
import os
import sys
import tempfile

CORE = sys.argv[1]
PACK_DIR = os.path.abspath(sys.argv[2])
sys.argv = ["v935_core_half", "--cpu"]
sys.path.insert(0, CORE)
import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()
sys.path.insert(0, os.path.dirname(PACK_DIR))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.model_patcher  # noqa: E402
import comfy.patcher_extension as PE  # noqa: E402
import comfy.k_diffusion.sampling as KS  # noqa: E402
import folder_paths  # noqa: E402
import safetensors.torch as sft  # noqa: E402
from comfy.ldm.minimax.model import FinalLayer  # noqa: E402

PKG = os.path.basename(PACK_DIR)


def pack(mod):
    return importlib.import_module(PKG + ".nodes." + mod)


S = pack("uls_stack_node")
A = pack("uls_pdd_apply")
RB = pack("uls_adaln_rebase")
M = pack("uls_pdd_math")
SMP = pack("uls_sampler")

OUT = {}
D, F_, VD, AD, R, DENSE, K, NB = 12, 10, 6, 4, 4, 14, 8, 32


class _Blk(nn.Module):
    def __init__(s, ad_in):
        super().__init__()
        s.attn = nn.Module()
        s.attn.qkv_proj = nn.Linear(D, 3 * D)
        s.attn.out_proj = nn.Linear(D, D)
        s.mlp = nn.Module()
        s.mlp.fc1 = nn.Linear(D, 2 * F_)
        s.mlp.fc2 = nn.Linear(F_, D)
        s.adaln_proj = nn.Module()
        s.adaln_proj.linear = nn.Linear(ad_in, 6 * D)


class _DM(nn.Module):
    """MiniMax-H3 module NAMES around core's REAL FinalLayer; forward wraps
    exactly as MiniMaxH3Model.forward does."""

    def __init__(s, curve, drop_fc2=False):
        super().__init__()
        s.blocks = nn.ModuleList([_Blk(K if curve else DENSE)])
        if drop_fc2:
            del s.blocks[0].mlp.fc2
        s.final_layer = FinalLayer(D, 8, VD, AD, 1e-6,
                                   operations=comfy.ops.disable_weight_init)
        for p in s.final_layer.parameters():
            nn.init.normal_(p)
        if curve:
            s.register_buffer("adaln_t_table", torch.randn(9, K))

    def forward(s, x, timestep, context, transformer_options={}, **kw):
        return PE.WrapperExecutor.new_class_executor(
            s._forward, s, PE.get_all_wrappers(
                PE.WrappersMP.DIFFUSION_MODEL, transformer_options)
        ).execute(x, timestep, context, transformer_options, **kw)

    def _forward(s, x, timestep, context, transformer_options={}, **kw):
        # v965: Core master's FinalLayer.forward takes (sigma, sample_sigmas,
        # shifts) as well (commit 2504e68d); Core 7a131a3a takes four. Call
        # the signature this Core has -- read from the function, not guessed.
        import inspect
        n_params = len(inspect.signature(s.final_layer.forward).parameters)
        if n_params >= 7:
            sig = torch.tensor(float(timestep.flatten()[0]) / 1000.0) if torch.is_tensor(timestep) else torch.tensor(0.5)
            return s.final_layer(x, context, (0, 6, 0), (6, 10, 1), sig,
                                 transformer_options.get("sample_sigmas"), (12.0, 3.0))
        return s.final_layer(x, context, (0, 6, 0), (6, 10, 1))


class _MS(object):
    multiplier = 1000.0


class _Cfg(object):
    unet_config = {"image_model": "minimax_h3"}   # as supported_models.MiniMaxH3


class Base(nn.Module):
    def __init__(s, curve=False, drop_fc2=False):
        super().__init__()
        s.diffusion_model = _DM(curve, drop_fc2)
        s.model_sampling = _MS()
        s.model_config = _Cfg()


def patcher(base):
    return comfy.model_patcher.ModelPatcher(
        base, load_device=torch.device("cpu"), offload_device=torch.device("cpu"))


def pdd_file(path, seed=0):
    """A PDD Acc file in the PUBLISHED layout: Diffusers names, lora_down/up,
    dense adaln, the four bank tensors, alpha in the metadata (10.09. probe)."""
    g = torch.Generator().manual_seed(seed)

    def r(*sh):
        return torch.randn(*sh, generator=g) * 0.1
    sd = {}
    for part in ("to_q", "to_k", "to_v"):
        sd["transformer_blocks.0.attn.%s.lora_down.weight" % part] = r(R, D)
        sd["transformer_blocks.0.attn.%s.lora_up.weight" % part] = r(D, R)
    sd["transformer_blocks.0.attn.to_out.0.lora_down.weight"] = r(R, D)
    sd["transformer_blocks.0.attn.to_out.0.lora_up.weight"] = r(D, R)
    sd["transformer_blocks.0.ff.net.0.proj.lora_down.weight"] = r(R, D)
    sd["transformer_blocks.0.ff.net.0.proj.lora_up.weight"] = r(2 * F_, R)
    sd["transformer_blocks.0.ff.net.2.lora_down.weight"] = r(R, F_)
    sd["transformer_blocks.0.ff.net.2.lora_up.weight"] = r(D, R)
    sd["transformer_blocks.0.adaln_proj.linear.lora_down.weight"] = r(R, DENSE)
    sd["transformer_blocks.0.adaln_proj.linear.lora_up.weight"] = r(6 * D, R)
    sd["proj_out.weight"] = r(NB, VD, D)
    sd["proj_out.bias"] = r(NB, VD)
    sd["audio_proj_out.weight"] = r(NB, AD, D)
    sd["audio_proj_out.bias"] = r(NB, AD)
    sft.save_file(sd, path, metadata={"lora_alpha": "2.0", "lora_rank": str(R),
                                      "pdd_block_size": "4"})
    return sd


class Loader(object):
    calls = []

    def load_lora(self, m, c, name, w, wc):
        Loader.calls.append(name)
        return m, c


silence = open(os.devnull, "w")
real_stdout = sys.stdout
sys.stdout = silence                     # the pack prints; the guard reads JSON

loras = tempfile.mkdtemp()
folder_paths.folder_names_and_paths["loras"] = ([loras], {".safetensors"})
PDD = "MiniMax-H3-FL2VA-Acc-8Step.safetensors"
SD = pdd_file(os.path.join(loras, PDD))

# 1 dense model
base = Base()
mp = patcher(base)
m1, _, e1 = S.apply_lora_set(Loader(), mp, None, [PDD], [1.0], mode="SEQ")
st1 = A.state_of(m1)
w0 = base.diffusion_model.blocks[0].attn.out_proj.weight.detach().clone()
m1.patch_model()
w1 = base.diffusion_model.blocks[0].attn.out_proj.weight.detach().clone()
m1.unpatch_model()
want = (SD["transformer_blocks.0.attn.to_out.0.lora_up.weight"]
        @ SD["transformer_blocks.0.attn.to_out.0.lora_down.weight"] * (2.0 / R))
OUT["dense"] = {
    "errs": e1, "loader": list(Loader.calls), "state": st1 is not None,
    "nfe": st1.nfe if st1 else None, "patches": len(m1.patches),
    "delta_exact": bool(torch.allclose(w1 - w0, want, atol=1e-6)),
}

# 2 curve-form model, a basis matching its table
base2 = Base(curve=True)
mp2 = patcher(base2)
bdir = tempfile.mkdtemp()
g = torch.Generator().manual_seed(7)
sft.save_file({"c": torch.randn(DENSE, generator=g),
               "V": torch.randn(DENSE, K, generator=g)},
              os.path.join(bdir, "basis_test.safetensors"),
              metadata={"table_sha256": RB.table_sha256(
                  base2.diffusion_model.adaln_t_table), "trunk": "fl2va"})
RB.BASIS_DIR = bdir
m2, _, e2 = S.apply_lora_set(Loader(), mp2, None, [PDD], [1.0])
b0 = base2.diffusion_model.blocks[0].adaln_proj.linear.bias.detach().clone()
m2.patch_model()
b1 = base2.diffusion_model.blocks[0].adaln_proj.linear.bias.detach().clone()
width = int(base2.diffusion_model.blocks[0].adaln_proj.linear.weight.shape[1])
m2.unpatch_model()
OUT["curve"] = {
    "errs": e2, "state": A.state_of(m2) is not None,
    "bias_patch": "diffusion_model.blocks.0.adaln_proj.linear.bias" in m2.patches,
    "width": width, "bias_moved": not bool(torch.allclose(b0, b1)),
}

# 3..7 refusals and routing
RB.BASIS_DIR = tempfile.mkdtemp()
mp3 = patcher(Base(curve=True))
m3, _, e3 = S.apply_lora_set(Loader(), mp3, None, [PDD], [1.0])
OUT["no_basis"] = {"errs": e3, "untouched": m3 is mp3 and not mp3.patches}

mp4 = patcher(Base(drop_fc2=True))
m4, _, e4 = S.apply_lora_set(Loader(), mp4, None, [PDD], [1.0])
OUT["missing_module"] = {"errs": e4, "untouched": m4 is mp4 and not mp4.patches}

m5, _, e5 = S.apply_lora_set(Loader(), m1, None, [PDD], [1.0])
OUT["second_bank"] = {"errs": e5, "untouched": m5 is m1}

sft.save_file({"diffusion_model.blocks.0.attn.out_proj.lora_A.weight": torch.zeros(2, D),
               "diffusion_model.blocks.0.attn.out_proj.lora_B.weight": torch.zeros(D, 2)},
              os.path.join(loras, "char.safetensors"))
Loader.calls = []
m6, _, e6 = S.apply_lora_set(Loader(), patcher(Base()), None,
                             [PDD, "char.safetensors"], [1.0, 0.8], mode="SEQ")
OUT["mixed"] = {"errs": e6, "loader": list(Loader.calls),
                "state": A.state_of(m6) is not None}


class NotH3(nn.Module):
    def __init__(s):
        super().__init__()
        s.diffusion_model = nn.Linear(2, 2)
        s.model_config = _Cfg()


mp7 = patcher(NotH3())
m7, _, e7 = S.apply_lora_set(Loader(), mp7, None, [PDD], [1.0])
OUT["not_h3"] = {"errs": e7, "untouched": m7 is mp7}

# 8 the heads on core's real FinalLayer, through core's real wrapper path


def forward(model, base_, sig, with_opts=True):
    model.patch_model()
    to = {}
    PE.merge_nested_dicts(to.setdefault("wrappers", {}), model.wrappers,
                          copy_dict1=False)            # sampler_helpers.py
    if with_opts:
        to["sigmas"] = torch.tensor([sig])
    torch.manual_seed(1)
    x = torch.randn(10, D)
    ctx = torch.randn(3, 8)
    try:
        v, _ = base_.diffusion_model(x, torch.tensor([sig * 1000.0]), ctx,
                                     transformer_options=to)
    finally:
        model.unpatch_model()
    fl = base_.diffusion_model.final_layer
    shift, scale = fl.adaln_proj(ctx)
    hv = (fl.norm(x[0:6]) * (1.0 + scale[0]) + shift[0]).float()
    return v, hv


v, hv = forward(m1, base, st1.boundaries[2])
ok_head = bool(torch.allclose(v, F.linear(hv, st1.fused_v[0][2], st1.fused_v[1][2]), atol=1e-5))
v, hv = forward(m1, base, st1.boundaries[6], with_opts=False)
ok_fallback = bool(torch.allclose(v, F.linear(hv, st1.fused_v[0][6], st1.fused_v[1][6]), atol=1e-5))
try:
    forward(m1, base, 0.5)
    ok_offgrid = False
except RuntimeError as ex:
    ok_offgrid = "not a trained block boundary" in str(ex)
OUT["heads"] = {"forward": ok_head, "fallback": ok_fallback, "offgrid": ok_offgrid,
                "native_back": type(base.diffusion_model.final_layer.video_out).__name__ == "Linear"}

# 9 core's Euler visits exactly the block starts
euler = {}
for nfe in M.LEGAL_NFE:
    b = M.boundary_sigmas(M.resolve_partition(nfe=nfe))
    seen = []

    def fn(x, sigma, **kw):
        seen.append(float(sigma.flatten()[0]))
        return x * 0
    KS.sample_euler(fn, torch.zeros(1, 2), torch.tensor(b, dtype=torch.float32),
                    disable=True)
    euler[str(nfe)] = [M.select_block(s_, b) for s_ in seen]
OUT["euler"] = euler

# 10 the sampler's preparation on a real PDD model
mA, sA = SMP._pdd_prepare(m1, 6, "euler", 1.0, 0, 10000, False, None, 1.0)
stA = A.state_of(mA)
res = {"nfe": stA.nfe, "sig_match": [round(float(x), 6) for x in sA]
       == [round(x, 6) for x in stA.boundaries]}
try:
    SMP._pdd_prepare(m1, 12, "euler", 1.0, 0, 10000, False, None, 1.0)
    res["refuse12"] = False
except ValueError:
    res["refuse12"] = True
mB, sB = SMP._pdd_prepare(m1, 20, "euler", 1.0, 0, 10000, False,
                          torch.tensor(A.state_of(A.refit(m1, 4)).boundaries), 1.0)
res["ext_nfe"] = A.state_of(mB).nfe
plain = patcher(Base())
mC, sC = SMP._pdd_prepare(plain, 20, "dpmpp_2m", 0.5, 0, 10000, True, None, 7.0)
res["no_bank_untouched"] = mC is plain and sC is None
OUT["sampler"] = res

sys.stdout = real_stdout
print(json.dumps(OUT))
