# -*- coding: ascii -*-
"""v930 -- foreign LoRA schemas become ComfyUI keys, provably.

An H3 LoRA is trained in the Diffusers layout. Core's converter (43 lines,
three unrelated model families) does not know it, so every such file matched
nothing and applied to zero layers -- quietly.

The two rearrangements are pinned very differently, on purpose:

  qkv fusion IS AN IDENTITY. Three separate deltas stacked vertically must
  equal the fused delta, exactly. A is concatenated, B is block diagonal,
  alpha is tripled. Getting any of that wrong changes the delta, so it is
  proved numerically -- not by reading key names.

  The SwiGLU swap IS A CONVENTION. No arithmetic can tell a swapped pair from
  an unswapped one; both are valid matrices. It is therefore pinned against
  the two witnesses we actually have: core's own _swiglu_eager takes GATE
  first, and the published Comfy-Org turbo LoRA records the mapping in its
  metadata. The guard pins the direction and the fact that A is untouched.

torch only. No comfy.
"""
import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uls_lora_convert as C  # noqa: E402
import uls_merge_policy as MP  # noqa: E402

FAILED = []
CHECKS = []


def _fail(msg):
    FAILED.append(msg)
    CHECKS.append(msg)
    print("FAIL: {}".format(msg))


def _ok(msg):
    CHECKS.append(msg)
    print("ok  : {}".format(msg))


def _need(cond, msg):
    _ok(msg) if cond else _fail(msg)


R, HID, INNER, FFN, ADALN = 8, 64, 96, 128, 48
torch.manual_seed(930)


def build_source(prefix="transformer_blocks.0.", with_alpha=False):
    """A Diffusers-layout H3 LoRA, in the shape the real files have."""
    sd = {}
    parts = {}
    for part, (o, i) in (("attn.to_q", (INNER, HID)), ("attn.to_k", (INNER, HID)),
                         ("attn.to_v", (INNER, HID)),
                         ("attn.to_out.0", (HID, INNER)),
                         ("ff.net.0.proj", (FFN * 2, HID)),
                         ("ff.net.2", (HID, FFN)),
                         ("adaln_proj.linear", (ADALN, HID))):
        A = torch.randn(R, i)
        B = torch.randn(o, R)
        parts[part] = (A, B)
        sd[prefix + part + ".lora_down"] = A
        sd[prefix + part + ".lora_up"] = B
        if with_alpha:
            sd[prefix + part + ".alpha"] = torch.tensor(float(R))
    return sd, parts


def delta(A, B, alpha, rank):
    return (B @ A) * (alpha / rank)


# --- A: detection --------------------------------------------------------------
src, parts = build_source()
_need(C.detect_h3_diffusers(src), "A  a Diffusers-layout H3 LoRA is recognised")

out, notes = C.convert_h3_diffusers(src, metadata={"lora_alpha": "8.0", "lora_rank": "8"})
_need(not C.detect_h3_diffusers(out),
      "A  the CONVERTED dict is not recognised again (no double conversion)")
again, name = C.convert_foreign_lora(out)
_need(name is None and again is out,
      "A  and the registry hands an already-converted dict straight back")

comfy_style = {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": torch.zeros(3, 4),
               "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": torch.zeros(4, 3)}
_need(not C.detect_h3_diffusers(comfy_style),
      "A  a dict already in core's layout is not touched")
_need(C.convert_foreign_lora({"nonsense": torch.zeros(1)})[1] is None,
      "A  an unknown dict is handed through, not guessed at")

# The load-bearing one: another Diffusers transformer spells attention and
# feed-forward identically. Only adaln_proj tells H3 apart.
foreign = {}
for mod in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
            "ff.net.0.proj", "ff.net.2", "norm1.linear"):
    foreign["transformer_blocks.0." + mod + ".lora_down"] = torch.randn(R, HID)
    foreign["transformer_blocks.0." + mod + ".lora_up"] = torch.randn(HID, R)
_need(not C.detect_h3_diffusers(foreign),
      "A  a Flux/SD3-style Diffusers LoRA (same attn+ff names, norm1.linear "
      "instead of adaln_proj) is NOT claimed as H3")
_need(C.convert_foreign_lora(foreign)[1] is None,
      "A  and the registry leaves it completely alone")
partial = {k: v for k, v in src.items() if ".adaln_proj." not in k}
_need(not C.detect_h3_diffusers(partial),
      "A  prefix + attn names WITHOUT adaln are not enough either")

# --- B: the qkv fusion, as an identity -----------------------------------------
base = "diffusion_model.blocks.0.attn.qkv_proj"
Af = out[base + ".lora_A.weight"]
Bf = out[base + ".lora_B.weight"]
alpha_f = float(out[base + ".alpha"])

_need(Af.shape == (3 * R, HID), "B  fused A is (3r, in): %s" % (list(Af.shape),))
_need(Bf.shape == (3 * INNER, 3 * R), "B  fused B is (3*out, 3r): %s" % (list(Bf.shape),))
_need(abs(alpha_f - 24.0) < 1e-6, "B  alpha is tripled (8 -> %.1f)" % alpha_f)
_need(abs(alpha_f / Af.shape[0] - 8.0 / R) < 1e-9,
      "B  and alpha/rank -- the actual scale -- is unchanged")

sep = torch.cat([delta(*parts[p], 8.0, R) for p in C.H3_QKV_PARTS], dim=0)
fus = delta(Af, Bf, alpha_f, Af.shape[0])
_need(torch.allclose(sep, fus, atol=1e-5),
      "B  THE IDENTITY: three separate deltas stacked == the fused delta "
      "(max diff %.2e)" % (sep - fus).abs().max())

# A wrong arrangement must NOT pass this. Two plausible mistakes, driven:
b_wrong = torch.zeros_like(Bf)
row = 0
for i in range(3):
    b_wrong[row:row + INNER, :R] = parts[C.H3_QKV_PARTS[i]][1]   # all in column 0
    row += INNER
_need(not torch.allclose(delta(Af, b_wrong, alpha_f, Af.shape[0]), sep, atol=1e-5),
      "B  a NON-block-diagonal B fails the identity (the test can fail)")
a_wrong = torch.cat([parts[p][0] for p in reversed(C.H3_QKV_PARTS)], dim=0)
_need(not torch.allclose(delta(a_wrong, Bf, alpha_f, Af.shape[0]), sep, atol=1e-5),
      "B  a wrongly ORDERED A fails it too")

# --- C: the SwiGLU convention ---------------------------------------------------
A_ff, B_ff = parts["ff.net.0.proj"]
fc1 = "diffusion_model.blocks.0.mlp.fc1"
Bs = out[fc1 + ".lora_B.weight"]
_need(torch.equal(Bs[:FFN], B_ff[FFN:]) and torch.equal(Bs[FFN:], B_ff[:FFN]),
      "C  fc1's output halves are swapped: Diffusers [value;gate] -> core [gate;value]")
_need(torch.equal(out[fc1 + ".lora_A.weight"], A_ff),
      "C  and A -- the INPUT side -- is untouched by the swap")
for other, src_name in (("mlp.fc2", "ff.net.2"), ("attn.out_proj", "attn.to_out.0"),
                        ("adaln_proj.linear", "adaln_proj.linear")):
    key = "diffusion_model.blocks.0." + other + ".lora_B.weight"
    _need(torch.equal(out[key], parts[src_name][1]),
          "C  %s is NOT swapped (only fc1 is a SwiGLU gate)" % other)

odd = {"transformer_blocks.0.ff.net.0.proj.lora_down": torch.randn(R, HID),
       "transformer_blocks.0.ff.net.0.proj.lora_up": torch.randn(FFN * 2 + 1, R),
       "transformer_blocks.0.attn.to_q.lora_down": torch.randn(R, HID),
       "transformer_blocks.0.attn.to_q.lora_up": torch.randn(INNER, R)}
_out, _notes = C.convert_h3_diffusers(odd)
_need(any("odd output width" in n for n in _notes),
      "C  an odd fc1 width is reported, not silently half-swapped")

# --- D: names, alpha, prefixes --------------------------------------------------
expect = {"attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2", "adaln_proj.linear"}
got = {k.split("blocks.0.")[-1].rsplit(".lora", 1)[0].rsplit(".alpha", 1)[0]
       for k in out if ".lora" in k or k.endswith(".alpha")}
_need(expect <= got, "D  every core module name is produced: %s" % sorted(got))
_need(all(k.startswith("diffusion_model.") for k in out),
      "D  every key carries the diffusion_model. prefix core expects")
_need(not any("transformer_blocks" in k or "lora_down" in k or "lora_up" in k
              for k in out),
      "D  no Diffusers spelling survives the conversion")

tr_src, _p = build_source(prefix="token_refiner.refiner_blocks.1.")
tr_out, _n = C.convert_h3_diffusers(tr_src, metadata={"lora_alpha": "8.0"})
_need(any("token_refiner.blocks.1." in k for k in tr_out),
      "D  token_refiner.refiner_blocks -> token_refiner.blocks")

no_meta, _n = C.convert_h3_diffusers(build_source()[0], metadata=None)
_need(not any(k.endswith(".alpha") for k in no_meta),
      "D  with no metadata NO alpha is invented (core's default stands)")
with_a, _n = C.convert_h3_diffusers(build_source(with_alpha=True)[0], metadata=None)
_need(abs(float(with_a[base + ".alpha"]) - 3.0 * R) < 1e-6,
      "D  a per-module alpha in the file wins and is still tripled for qkv")

# --- E: partial groups and junk are survived ------------------------------------
half = {k: v for k, v in build_source()[0].items() if ".attn.to_v." not in k}
h_out, h_notes = C.convert_h3_diffusers(half)
_need(any("2 of 3 parts" in n for n in h_notes),
      "E  an incomplete qkv group is reported, not fused from what is there")
_need(not any("qkv_proj" in k for k in h_out),
      "E  and no half-fused qkv key is emitted")

junk = dict(build_source()[0])
junk["something.entirely.other"] = torch.zeros(2)
j_out, _n = C.convert_h3_diffusers(junk)
_need("something.entirely.other" in j_out,
      "E  a key the table does not know travels through unchanged")

# --- F: the metadata reader (v930 half of the header pair) ----------------------
import json  # noqa: E402
import struct  # noqa: E402
import tempfile  # noqa: E402

tmp = tempfile.mkdtemp()
p = os.path.join(tmp, "m.safetensors")
head = {"__metadata__": {"lora_alpha": "64.0", "lora_rank": "64"},
        "a.lora_A.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
blob = json.dumps(head).encode()
with open(p, "wb") as fh:
    fh.write(struct.pack("<Q", len(blob)))
    fh.write(blob)
    fh.write(b"\0" * 4)
_need(MP.safetensors_metadata(p).get("lora_alpha") == "64.0",
      "F  metadata is read from the header alone")
for what, bad in (("a pickle", b"\x80\x04junk"), ("a 2-byte file", b"\x01\x02")):
    q = os.path.join(tmp, what.replace(" ", "_"))
    open(q, "wb").write(bad)
    try:
        got = MP.safetensors_metadata(q)
    except Exception as ex:
        _fail("F  %s raised %s instead of answering {}" % (what, type(ex).__name__))
    else:
        _need(got == {}, "F  %s answers {} without raising" % what)
_need(MP.safetensors_metadata(os.path.join(tmp, "absent")) == {},
      "F  an absent path answers {}")

print("\n%d checks, %d failed" % (len(CHECKS), len(FAILED)))
if FAILED:
    print("\n".join(FAILED))
    sys.exit(1)
print("v930 foreign lora conversion: PASS")
