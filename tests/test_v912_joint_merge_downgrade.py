#!/usr/bin/env python3
# -*- coding: ascii -*-
"""v912 -- the random merge cleanups step back on a joint (distilled) model.

THE WOUND (Frank, 04.09.: "die LoRA-Merge-Methoden lassen mit MiniMax-LoRAs das
neurale Netzwerk mitunter kollabieren"): DARE and RESOLVE were calibrated on
WAN, where CFG > 1 pulls the sample back. MiniMax H3 samples at CFG 1 -- it is
distilled -- so noise written into the delta reaches the output unfiltered.
Measured 04.09. against the SEQ sum (random rank-32 sources, 2/4/8 LoRAs):
DARE +2/+9/+25 % delta energy at 25/41/73 % deviation; RESOLVE shrinks the
delta to 0.69/0.46/0.32 (TIES averages, additive LoRAs vanish). CONCAT equals
SEQ to bf16 rounding (rel 1.5e-3), the int8 requant is ordinary quant noise
(0.85 % -> 1.5 %). So the cut touches ONLY the two random cleanups.

WHAT IS PINNED, and how:

  A  the pure decision `_joint_merge_downgrade` (uls_merge_policy -- its own
     module, because uls_merge_math.py is an anchor file pinned since v598): on a plain
     model everything passes through untouched; on a joint model DARE becomes
     CONCAT, RESOLVE goes off, TRIM stays -- each with a note that names the
     reason. Driven over the whole {mode} x {resolve} x {trim} x {joint} grid.
  B  BEHAVIOUR of `apply_lora_set`, driven: the function is lifted CLOSED with
     tests/_lift.py and run against stubs that RECORD what reaches the merge.
     Two LoRAs + DARE + RESOLVE on a joint model must reach the merge as
     CONCAT without RESOLVE; on a plain model unchanged; SEQ and single-LoRA
     rows must never call the probe (it runs only when a merge is about to
     happen); a probe that raises must fail OPEN (plain).
  C  the note is printed with the reason (captured stdout), so the console
     says what happened and why.

No torch, no comfy: the block is executed against injected values.
"""
import io
import os
import pathlib
import sys
import contextlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nodes"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _lift as _LIFT  # noqa: E402
import uls_merge_policy as MM  # noqa: E402

FAILED = []


def _fail(msg):
    FAILED.append(msg)
    print("FAIL: {}".format(msg))


def _ok(msg):
    print("ok  : {}".format(msg))


def _need(cond, msg):
    _ok(msg) if cond else _fail(msg)


# --- A: the pure decision ----------------------------------------------------
GRID = [(m, r, t, j) for m in ("SEQ", "CONCAT", "DARE", "dare")
        for r in (False, True) for t in (False, True) for j in (False, True)]
bad = []
for m, r, t, j in GRID:
    mode, resolve, notes = MM._joint_merge_downgrade(m, r, t, j)
    up = m.upper()
    if not j:
        if (mode, resolve, notes) != (up, r, []):
            bad.append((m, r, t, j, mode, resolve, notes))
        continue
    want_mode = "CONCAT" if up == "DARE" else up
    if mode != want_mode or resolve is not False:
        bad.append((m, r, t, j, mode, resolve, notes))
    if up == "DARE" and not any(n.startswith("DARE -> CONCAT") for n in notes):
        bad.append(("no DARE note", m, r, t, j))
    if r and not any(n.startswith("RESOLVE switched off") for n in notes):
        bad.append(("no RESOLVE note", m, r, t, j))
    if t and up != "SEQ" and not any(n.startswith("TRIM stays") for n in notes):
        bad.append(("no TRIM note", m, r, t, j))
    if any(MM.JOINT_MERGE_REASON not in n for n in notes if not n.startswith("TRIM")):
        bad.append(("note without reason", m, r, t, j, notes))
_need(not bad, "A  pure decision over %d cases: %s" % (len(GRID), bad[:3] if bad else "all as specified"))
_need(MM._joint_merge_downgrade("CONCAT", False, False, True) == ("CONCAT", False, []),
      "A  joint + CONCAT without cleanups is untouched and silent")

# --- B: apply_lora_set, lifted closed and driven ----------------------------
SRC = (ROOT / "nodes" / "uls_stack_node.py").read_text(encoding="utf-8")
# v929: apply_lora_set now consults folder_paths in its payload gate. The stub
# answers None for every name, so the gate finds nothing to refuse and this
# guard's promises are unchanged -- it is here to keep the lift closed, not to
# test the gate (tests/test_v929_payload_guard.py does that).
class _NoPaths(object):
    def get_full_path(self, kind, name):
        return None


_FOLDER_PATHS = _NoPaths()

PROVIDED = ["_apply_seq", "_apply_concat_or_dare", "_joint_latent_parts",
            "_joint_merge_downgrade", "JOINT_MERGE_REASON", "print",
            "_apply_decision", "_target_is_quantized",
            "folder_paths", "payload_family", "safetensors_header_names",
            "os",   # v913: apply seam, driven as PATCH here
            # v935: the whole-file payload handlers (the PDD path) are injected
            # EMPTY -- no file here is a payload, and this guard's promise is the
            # joint downgrade, not the PDD path (driven in test_v935).
            "_PAYLOAD_HANDLERS"]
DEPS_SRC, MISSING = _LIFT.close_over(SRC, ["apply_lora_set"], PROVIDED)
_need(not MISSING, "B  the lift is closed" if not MISSING else
      "B  the lift is short of %s -- a GUARD fault, not a tree fault" % ", ".join(MISSING))


def _no_apply(txt):
    """v917: the apply decision now announces BAKED too; that line is
    v913/v917's, not this guard's. Strip it before judging silence."""
    return "\n".join(l for l in txt.splitlines()
                     if "BAKED" not in l and "BYPASS" not in l)


def _drive(mode, resolve, trim, names, probe):
    calls = {"seq": [], "merge": [], "probe": 0}

    def _apply_seq(loader, model, clip, n, w, wc=None):
        calls["seq"].append(list(n))
        return model, clip, []

    def _apply_concat_or_dare(loader, model, clip, n, w, **kw):
        calls["merge"].append(dict(kw, names=list(n)))
        return model, clip, []

    def _joint_latent_parts(model):
        calls["probe"] += 1
        if isinstance(probe, Exception):
            raise probe
        return probe

    ns = {"_apply_seq": _apply_seq, "_apply_concat_or_dare": _apply_concat_or_dare,
          "_joint_latent_parts": _joint_latent_parts,
          "_joint_merge_downgrade": MM._joint_merge_downgrade,
          "JOINT_MERGE_REASON": MM.JOINT_MERGE_REASON, "print": print,
          "_apply_decision": MM._apply_decision,        # v913: real policy
          "_target_is_quantized": lambda m: False,     # v913: plain target -> baked
          "folder_paths": _FOLDER_PATHS,                # v929
          "payload_family": MM.payload_family,          # v929
          "safetensors_header_names": MM.safetensors_header_names,
          "os": os,
          "_PAYLOAD_HANDLERS": {}}                      # v935
    exec(compile(DEPS_SRC, "<lifted apply_lora_set>", "exec"), ns)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ns["apply_lora_set"](None, "MODEL", None, names, [1.0] * len(names),
                             mode=mode, trim=trim, resolve=resolve)
    return calls, out.getvalue()


two = ["a.safetensors", "b.safetensors"]
c, txt = _drive("DARE", True, True, two, 2)
_need(len(c["merge"]) == 1 and c["merge"][0]["mode"] == "CONCAT" and c["merge"][0]["resolve"] is False
      and c["merge"][0]["trim"] is True,
      "B  joint + DARE + RESOLVE + TRIM reaches the merge as CONCAT, no RESOLVE, TRIM kept")
_need(MM.JOINT_MERGE_REASON in txt and "DARE -> CONCAT" in txt and "RESOLVE switched off" in txt,
      "C  the console names both downgrades and the reason")

c, txt = _drive("DARE", True, False, two, 1)
_need(len(c["merge"]) == 1 and c["merge"][0]["mode"] == "DARE" and c["merge"][0]["resolve"] is True,
      "B  plain model: DARE + RESOLVE reach the merge unchanged")
_need(_no_apply(txt).strip() == "", "B  plain model: nothing printed (beyond the v917 apply line)")

c, txt = _drive("CONCAT", False, False, two, 2)
_need(len(c["merge"]) == 1 and c["merge"][0]["mode"] == "CONCAT" and _no_apply(txt).strip() == "",
      "B  joint + plain CONCAT: untouched and silent (beyond the v917 apply line)")

c, _ = _drive("DARE", True, False, two, RuntimeError("probe broke"))
_need(len(c["merge"]) == 1 and c["merge"][0]["mode"] == "DARE" and c["merge"][0]["resolve"] is True,
      "B  a probe that raises fails OPEN (merge runs as dialled)")

c, _ = _drive("SEQ", True, True, two, 2)
_need(c["probe"] == 0 and len(c["seq"]) == 1, "B  SEQ never calls the probe")
c, _ = _drive("DARE", True, True, ["only.safetensors"], 2)
_need(c["probe"] == 0 and len(c["seq"]) == 1, "B  a single LoRA never calls the probe")

# --- verdict -----------------------------------------------------------------
if FAILED:
    print("\n{} check(s) FAILED".format(len(FAILED)))
    sys.exit(1)
print("\nall checks passed")
