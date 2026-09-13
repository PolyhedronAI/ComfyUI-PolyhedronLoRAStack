# -*- coding: ascii -*-
"""v929 -- files that are NOT plain LoRAs are refused, not half-applied.

The wound: a MiniMax-H3 PDD Acc file dropped into models/loras loads cleanly
through any LoRA path. Its rank-64 trunk applies; its parallel-decoding head
bank (proj_out / audio_proj_out, 32 per-interval copies) is not expressible as
LoRA factors and is dropped. The run looks fine and is half-distilled. Measured
10.09.: uls_merge_policy._foreign_keys reported NOTHING for such a file -- the
v914 family list knows DoRA/LyCORIS/diff, none of which this is.

What is pinned here:
  A  the pure signature match: complete set convicts, partial never does
  B  the header reader answers from the header alone and NEVER raises
  C  the gate is driven for real: a PDD file is refused, a plain LoRA beside
     it still applies, and refusing does not swallow the survivors
  D  the gate sits BEFORE the apply decision, so every path passes it
  E  the refusal reaches the caller's error list through BOTH exits, in the
     shape the debug panes match a row against

No comfy. torch only to write a safetensors fixture (skipped honestly if the
safetensors writer is unavailable -- and the skip is REPORTED, not silent).
"""
import ast
import os
import pathlib
import struct
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lift as _LIFT  # noqa: E402
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


PDD_KEYS = {"proj_out.weight", "proj_out.bias",
            "audio_proj_out.weight", "audio_proj_out.bias"}
LORA_KEYS = {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight",
             "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight",
             "diffusion_model.blocks.0.attn.qkv_proj.alpha"}

# --- A: the pure signature ----------------------------------------------------
hit = MP.payload_family(PDD_KEYS | LORA_KEYS)
_need(hit is not None and hit[0] == "MiniMax-H3 PDD Acc",
      "A  a complete PDD signature is named")
# v935, re-justified: the PDD files stay in models/loras (Frank, 10.09.) and
# the Stack/Engine apply them whole -- so that is where the answer points.
_need(hit is not None and hit[1] == "models/loras",
      "A  the answer says where the file lives (models/loras since v935)")
_need(MP.payload_family(LORA_KEYS) is None,
      "A  a plain LoRA is not convicted")
_need(MP.payload_family([]) is None and MP.payload_family(None) is None,
      "A  an empty or absent key set is not convicted")

partial_ok = True
for drop in sorted(PDD_KEYS):
    if MP.payload_family((PDD_KEYS - {drop}) | LORA_KEYS) is not None:
        partial_ok = False
_need(partial_ok, "A  ALL signature names are required -- no partial conviction")

# The names must be matched WHOLE. A real LoRA names tensors
# `<module>.lora_A.weight`; a suffix match would convict a module called
# proj_out. This is the difference between this table and FOREIGN_SUFFIXES.
suffixed = {"final_layer." + k for k in PDD_KEYS} | LORA_KEYS
_need(MP.payload_family(suffixed) is None,
      "A  match is on exact top-level names, not suffixes")

for label, names, home, reason in MP.PAYLOAD_SIGNATURES:
    _need(bool(names) and bool(home) and bool(reason),
          "A  signature %r carries names, a home and a reason" % label)

# --- B: the header reader -----------------------------------------------------
tmp = tempfile.mkdtemp()


def _write_st(path, names):
    """A minimal valid safetensors file: length prefix + JSON header + data."""
    off = 0
    entries = []
    for n in names:
        entries.append('"%s":{"dtype":"F32","shape":[1],"data_offsets":[%d,%d]}'
                       % (n, off, off + 4))
        off += 4
    header = ("{" + ",".join(entries) + "}").encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
        fh.write(b"\0" * off)


pdd_file = os.path.join(tmp, "MiniMax-H3-FL2VA-Acc-8Step.safetensors")
plain_file = os.path.join(tmp, "char_lora.safetensors")
_write_st(pdd_file, sorted(PDD_KEYS | LORA_KEYS))
_write_st(plain_file, sorted(LORA_KEYS))

names = MP.safetensors_header_names(pdd_file)
_need(names is not None and set(names) == PDD_KEYS | LORA_KEYS,
      "B  the header alone yields every tensor name")
_need(MP.payload_family(MP.safetensors_header_names(pdd_file)) is not None,
      "B  a real PDD file on disk is convicted")
_need(MP.payload_family(MP.safetensors_header_names(plain_file)) is None,
      "B  a real plain LoRA on disk is not")

# metadata is not a tensor
meta_file = os.path.join(tmp, "meta.safetensors")
header = ('{"__metadata__":{"x":"y"},'
          '"a.lora_A.weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
          ).encode("utf-8")
with open(meta_file, "wb") as fh:
    fh.write(struct.pack("<Q", len(header)))
    fh.write(header)
    fh.write(b"\0" * 4)
_need(MP.safetensors_header_names(meta_file) == frozenset({"a.lora_A.weight"}),
      "B  __metadata__ is not reported as a tensor")

# every malformed shape answers None, and none of them raises
junk = []
p = os.path.join(tmp, "pickle.pt")
open(p, "wb").write(b"\x80\x04junk")
junk.append(("a pickle", p))
p = os.path.join(tmp, "short.safetensors")
open(p, "wb").write(b"\x01\x02")
junk.append(("a 2-byte file", p))
p = os.path.join(tmp, "lying.safetensors")
open(p, "wb").write(struct.pack("<Q", 1 << 40) + b"x")
junk.append(("an oversized header length", p))
p = os.path.join(tmp, "truncated.safetensors")
open(p, "wb").write(struct.pack("<Q", 4096) + b"{}")
junk.append(("a truncated header", p))
p = os.path.join(tmp, "notjson.safetensors")
body = b"not json at all!"
open(p, "wb").write(struct.pack("<Q", len(body)) + body)
junk.append(("a non-JSON header", p))
p = os.path.join(tmp, "jsonlist.safetensors")
body = b"[1,2,3]"
open(p, "wb").write(struct.pack("<Q", len(body)) + body)
junk.append(("a JSON non-object header", p))
junk.append(("an absent path", os.path.join(tmp, "nope.safetensors")))

for what, path in junk:
    try:
        got = MP.safetensors_header_names(path)
    except Exception as ex:  # noqa: BLE001 - that is the point
        _fail("B  %s raised %s instead of answering None" % (what, type(ex).__name__))
    else:
        _need(got is None, "B  %s answers None (no answer, not 'clean')" % what)

# --- C/D/E: the gate, driven --------------------------------------------------
SRC = (ROOT / "nodes" / "uls_stack_node.py").read_text(encoding="utf-8")
TREE = ast.parse(SRC)
FN = next(n for n in ast.walk(TREE)
          if isinstance(n, ast.FunctionDef) and n.name == "apply_lora_set")


def _lineno_of(fragment, start=0):
    idx = SRC.index(fragment, start)
    return SRC[:idx].count("\n") + 1


# D: order inside the function is the whole promise -- the gate must run before
# the apply decision AND before the SEQ shortcut, because those are where the
# paths diverge. Pin the ORDER, not a line number.
gate_line = _lineno_of("_payload_errs = []")
decision_line = _lineno_of("use_bypass, mode, _anotes = _apply_decision(")
seq_line = _lineno_of("        return _with_errs(_apply_seq(")
_need(FN.lineno < gate_line < decision_line,
      "D  the gate runs before the apply decision")
_need(gate_line < seq_line,
      "D  the gate runs before the SEQ shortcut")

# D: it must consult the reader through the policy module, not re-implement it.
calls = {n.func.id for n in ast.walk(FN)
         if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
_need("payload_family" in calls,
      "D  apply_lora_set CALLS payload_family (not merely imports it)")
_need("safetensors_header_names" in calls,
      "D  apply_lora_set CALLS safetensors_header_names")

# E: both productive exits carry the refusals out.
exits = [n for n in ast.walk(FN) if isinstance(n, ast.Return)]
wrapped = [n for n in exits
           if isinstance(n.value, ast.Call)
           and isinstance(n.value.func, ast.Name)
           and n.value.func.id == "_with_errs"]
_need(len(wrapped) == 2,
      "E  both productive exits are wrapped in _with_errs (found %d)" % len(wrapped))
raw_exits = [ast.unparse(n) for n in exits
             if isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Name)
             and n.value.func.id in ("_apply_seq", "_apply_concat_or_dare")]
_need(not raw_exits,
      "E  no exit returns an apply path directly, bypassing the error list")

# C: drive the real thing. The module cannot be imported (it needs comfy), so
# lift the function and its helper and run them against stubs -- the shipped
# source, not a restatement.
PROVIDED = {
    "folder_paths", "payload_family", "safetensors_header_names",
    "_apply_seq", "_apply_concat_or_dare", "_apply_decision",
    "_target_is_quantized", "_joint_merge_downgrade", "_joint_latent_parts",
    "_convention_label", "_foreign_keys", "_has_mid_tensor",
    "_cached_load_torch_file", "_convert_lora_like_core",
    "os", "print",
    # v935: the whole-file handlers are injected -- the gate's promise is the
    # ROUTING, the handler itself is driven in test_v935 against real classes.
    "_PAYLOAD_HANDLERS",
}
LIFTED, MISSING = _LIFT.close_over(SRC, ["apply_lora_set"], PROVIDED)
_need(not MISSING, "C  the lift is closed (short of: %s)" % (MISSING or "nothing"))

if MISSING:
    ALS = None
else:
    class _FP(object):
        def get_full_path(self, kind, n):
            path = os.path.join(tmp, n)
            return path if os.path.exists(path) else None

    applied = []
    merged = []
    handled = []
    handler_says = {"refuse": None}

    def _pdd_handler(model, clip, name, weight, apply):
        """Stands in for uls_stack_node._apply_pdd: records, then either
        applies (returns a new model) or refuses (returns the input + error)."""
        handled.append(name)
        if handler_says["refuse"]:
            return model, clip, ["\u2717 Refused (PDD): %s -- %s"
                                 % (NS["_short_name"](name), handler_says["refuse"])]
        return "M+PDD", clip, []

    HANDLERS = {"MiniMax-H3 PDD Acc": _pdd_handler}

    def _apply_seq(loader, model, clip, names, weights, clip_weights=None):
        applied.extend(names)
        return model, clip, []

    def _apply_concat_or_dare(loader, model, clip, names, weights, **kw):
        merged.extend(names)
        return model, clip, []

    NS = {
        "os": os,
        "folder_paths": _FP(),
        "payload_family": MP.payload_family,
        "safetensors_header_names": MP.safetensors_header_names,
        "_apply_seq": _apply_seq,
        "_apply_concat_or_dare": _apply_concat_or_dare,
        "_apply_decision": lambda a, q, m: (False, m, []),
        "_target_is_quantized": lambda m: False,
        "_joint_merge_downgrade": lambda m, r, t, j: (m, r, []),
        "_joint_latent_parts": lambda m: None,
        "_convention_label": lambda c: "kohya",
        "_foreign_keys": lambda td: [],
        "_has_mid_tensor": lambda td: False,
        "_cached_load_torch_file": lambda path: {},
        "_convert_lora_like_core": lambda td: td,
        "print": lambda *a, **k: None,
        "_PAYLOAD_HANDLERS": HANDLERS,
    }
    exec(compile(LIFTED, "<lift:apply_lora_set>", "exec"), NS)  # noqa: S102
    ALS = NS["apply_lora_set"]

if ALS is not None:

    PDD = "MiniMax-H3-FL2VA-Acc-8Step.safetensors"
    PLAIN = "char_lora.safetensors"

    # v935, RE-JUSTIFIED (not softened). v929 promised "a PDD file is never
    # half-applied" and kept it by refusing the file. Since v935 the Stack and
    # Engine can apply it whole, so the gate ROUTES it to that path instead.
    # What this guard pins is the part of the promise that lives in the gate:
    # the file never reaches a plain LoRA loader, on any path; the handler's
    # answer -- applied or refused -- is what comes out; a refusal reaches the
    # error list; and a family WITHOUT a handler is still refused as in v929.
    m, c, errs = ALS(None, "M", "C", [PDD, PLAIN], [1.0, 0.8], mode="SEQ")
    _need(handled == [PDD] and applied == [PLAIN],
          "C  the PDD file goes to its whole-file path, the plain LoRA to the "
          "loader -- never the PDD file to the loader")
    _need(m == "M+PDD" and errs == [],
          "C  the handler's model is the one that travels on")

    applied[:] = []
    handled[:] = []
    handler_says["refuse"] = "no matching adaln basis"
    m, c, errs = ALS(None, "M", "C", [PDD, PLAIN], [1.0, 0.8], mode="SEQ")
    _need(handled == [PDD] and applied == [PLAIN],
          "C  a refusing handler still keeps the PDD file away from the loader")
    _need(len(errs) == 1 and "Refused" in errs[0],
          "C  the refusal reaches the caller's error list")
    _need(NS["_short_name"](PDD) in errs[0],
          "E  the error string contains the row's short name (debug panes match on it)")

    applied[:] = []
    handled[:] = []
    m, c, errs = ALS(None, "M", "C", [PDD], [1.0], mode="SEQ")
    _need(applied == [] and len(errs) == 1 and (m, c) == ("M", "C"),
          "C  a lone refused PDD file applies NOTHING, model/clip untouched")

    # The merged path must be gated too, not just SEQ.
    applied[:] = []
    merged[:] = []
    handled[:] = []
    m, c, errs = ALS(None, "M", "C", [PDD, PLAIN, PLAIN], [1.0, 0.8, 0.8],
                     mode="CONCAT")
    _need(PDD not in merged and merged == [PLAIN, PLAIN] and handled == [PDD],
          "C  the CONCAT/merged path is gated by the same block")
    _need(len(errs) == 1 and "Refused" in errs[0],
          "E  and its exit carries the refusal out too")
    handler_says["refuse"] = None

    # A family with no handler keeps v929's answer: refused, never loaded.
    HANDLERS.clear()
    applied[:] = []
    m, c, errs = ALS(None, "M", "C", [PDD, PLAIN], [1.0, 0.8], mode="SEQ")
    _need(applied == [PLAIN] and len(errs) == 1 and "not a LoRA" in errs[0],
          "C  a payload family without a whole-file path is refused as in v929")
    HANDLERS["MiniMax-H3 PDD Acc"] = _pdd_handler

    # A run with nothing to route must be byte-for-byte the old behaviour.
    applied[:] = []
    handled[:] = []
    m, c, errs = ALS(None, "M", "C", [PLAIN], [1.0], mode="SEQ")
    _need(applied == [PLAIN] and errs == [] and handled == [],
          "C  a run without payload files is unchanged (no new noise)")

print("\n%d checks, %d failed" % (len(CHECKS), len(FAILED)))
if FAILED:
    print("\n".join(FAILED))
    sys.exit(1)
print("v929 payload guard: PASS")
