# -*- coding: ascii -*-
"""Guard -- what "too long" means on MiniMax H3 (v907).

THE WOUND THIS CLOSES. The Token Counter reported "412 / 512" against H3 and
called 512 a limit. It is not one. Measured in core v0.33.4,
comfy/text_encoders/qwen3vl.py: max_length=99999999, pad_to_max_length=False --
this encoder never truncates. Frank spent an entire prompt session trimming a
positive against a cliff that does not exist, and a later session was told the
old prompt had been "silently truncated". Nothing had been.

WHAT IS TRUE INSTEAD, and what this guard pins. H3 is a single-stream
packed-token transformer: text, video and audio share ONE sequence and ONE
position axis (comfy/ldm/minimax/model.py, PackedLayout):

    segments = [("text", text_len)]
    g[:, 0] = torch.arange(text_len)   # one text token costs 1.0 on t
    cursor  = text_len                 # the video starts BEHIND the text

A latent frame costs FRAME_RESCALE * FRAME_PER_TOKEN[k % 5] -- 1.67 or 6.67.
So the prompt PUSHES the clip along the axis. Frank's own run: 378 tokens
against 7 latent frames puts the video at t=378..414.7, while the clip spans
36.7. The prompt occupies 10.3x the video's own extent, and since RoPE encodes
distance, the opening of that prompt pulls weaker than its end.

The 512 is not invented either -- but it is ai-toolkit's `max_text_length`
default for TRAINING, whose own comment reads "the released stack has no
limit". A training span worth knowing, not a cap.

The numbers below are the MEASURED ones, and the helpers are declared mirrors
of the model's constants. If core changes FRAME_PER_TOKEN or FRAME_RESCALE,
this guard goes red -- which is the point.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SRC = os.path.join(ROOT, "nodes", "uls_stack_node.py")

_FAILED = []
_PASSED = []


def check(label, cond, detail=""):
    if cond:
        _PASSED.append(label)
    else:
        _FAILED.append(label + ((" -- " + detail) if detail else ""))


src = open(SRC, encoding="utf-8").read()

# Lift the pure helpers and their constants: this module imports torch at
# module level, so it cannot be imported in the sandbox. The helpers are pure
# and that is exactly why they are separable.
_want_fn = {"h3_video_span", "h3_reach", "h3_latent_frames"}
_want_const = ("H3_FRAME_PER_TOKEN", "H3_FRAME_RESCALE", "H3_TRAIN_SPAN")
tree = ast.parse(src)
keep = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in _want_fn:
        keep.append(node)
    elif isinstance(node, ast.Assign):
        for t in node.targets:
            if getattr(t, "id", None) in _want_const:
                keep.append(node)
                break
missing = _want_fn - {n.name for n in keep if isinstance(n, ast.FunctionDef)}
if missing:
    print("[test_v907_h3_reach] FAIL: helpers not found: %s" % sorted(missing))
    sys.exit(1)

ns = {}
exec(compile(ast.Module(body=keep, type_ignores=[]), "<lift>", "exec"), ns)
span, reach, frames = ns["h3_video_span"], ns["h3_reach"], ns["h3_latent_frames"]

# --- P1: the mirror reproduces the model's own arithmetic -------------------
# 7 latent frames = spans 1.67, 6.67, 6.67, 6.67, 6.67, 1.67, 6.67 = 36.666..
check("P1 seven latent frames span 36.67",
      abs(span(7) - 36.6666666) < 1e-4, "got %r" % span(7))
check("P1 the 5-cycle is honoured (frame 5 is short again)",
      abs(span(6) - (36.6666666 - 6.6666666)) < 1e-4,
      "FRAME_PER_TOKEN[k%%5] gives 1,4,4,4,4 then 1 again; got %r" % span(6))
check("P1 a single latent frame spans 1.67",
      abs(span(1) - 1.6666666) < 1e-4, "got %r" % span(1))
check("P1 no frames means no span", span(0) == 0.0 and span(-3) == 0.0)

# --- P2: Frank's real run, the number that started this ---------------------
r = reach(378, 7)
check("P2 the measured ratio is 10.3", abs(r["ratio"] - 10.3) < 0.05,
      "378 tokens against a 22-frame clip; got %r" % r["ratio"])
check("P2 the video starts at t=378, not 0", r["start"] == 378.0)
check("P2 and ends at 414.7", abs(r["end"] - 414.6666) < 1e-3)

# --- P3: shorter prompt, proportionally closer -----------------------------
check("P3 100 tokens gives 2.7", abs(reach(100, 7)["ratio"] - 2.727) < 0.01)
check("P3 the ratio is linear in prompt length",
      abs(reach(200, 7)["ratio"] - 2 * reach(100, 7)["ratio"]) < 1e-6,
      "doubling the prompt must double how far it pushes the clip")

# --- P4: no video to compare against -> no ratio, and NO number invented ----
r0 = reach(378, 0)
check("P4 without a latent the ratio is None", r0["ratio"] is None,
      "a number here would be exactly the invented figure this replaces")
check("P4 the text span is still honest", r0["text"] == 378.0)

# --- P5: reading the latent, joint and plain -------------------------------
class _T:
    def __init__(self, shape, nested=False, parts=None):
        self.shape = shape
        self.is_nested = nested
        self._parts = parts or ()

    def unbind(self):
        return self._parts


video = _T((1, 24, 7, 48, 84))
audio = _T((1, 32, 2, 37))
joint = _T((), nested=True, parts=(video, audio))
check("P5 a joint AV latent yields the VIDEO half's frames",
      frames({"samples": joint}) == 7,
      "core's VAEDecode takes unbind()[0]; taking [-1] would give the audio")
check("P5 a plain video latent works too", frames({"samples": video}) == 7)
check("P5 nothing wired yields 0", frames(None) == 0 and frames({}) == 0)
check("P5 a 4-D (image) latent yields 0", frames({"samples": _T((1, 4, 64, 64))}) == 0,
      "only a 5-D latent has a time axis to read")
check("P5 an unbindable nested latent yields 0",
      frames({"samples": _T((), nested=True, parts=())}) == 0)

# --- P6: the training span is named, and named as what it is ---------------
check("P6 the training span is 512", ns["H3_TRAIN_SPAN"] == 512)
check("P6 the constants mirror the model",
      tuple(ns["H3_FRAME_PER_TOKEN"]) == (1, 4, 4, 4, 4)
      and abs(ns["H3_FRAME_RESCALE"] - 5.0 / 3.0) < 1e-9)

# --- P7: the report tells the truth about truncation ------------------------
check("P7 the report states that nothing is truncated",
      "no cap" in src and "Nothing is truncated here" in src,
      "the whole point: H3 does not truncate, and the node must say so")
check("P7 the H3 block is gated on the H3 encoder",
      "if _is_h3_encoder(clip):" in src,
      "printing these numbers for WAN would invent meaning where there is none")
# --- P8: v908 -- the H3 test must identify H3, and it must be DRIVEN --------
# v907 searched for "minimax" in the inner tokenizer's NAME. Measured in
# comfy/text_encoders/minimax.py, MiniMaxH3Tokenizer passes
# embedding_key="qwen3vl_32b" -- the same name a plain Qwen3-VL uses. The test
# could never fire, and Frank's field run proved it: the H3 block was absent
# while the header read "qwen3vl_32b". The OUTER class is what distinguishes
# them, so that is what is checked, driven against both shapes.
_h3_src = src.split("def _is_h3_encoder")[1].split("\ndef ")[0]
check("P8 the H3 test does NOT key on the inner tokenizer name",
      'get("name"' not in _h3_src,
      "H3's inner tokenizer is called qwen3vl_32b, exactly like a plain "
      "Qwen3-VL -- the name cannot tell them apart")
check("P8 it keys on the tokenizer CLASS", "type(tk).__name__" in _h3_src)

_fn = [n for n in ast.parse(src).body
       if isinstance(n, ast.FunctionDef) and n.name == "_is_h3_encoder"]
if _fn:
    _ns = {}
    exec(compile(ast.Module(body=_fn, type_ignores=[]), "<h3>", "exec"), _ns)
    _is_h3 = _ns["_is_h3_encoder"]

    class _Tok:
        pass

    class MiniMaxH3Tokenizer(_Tok):
        pass

    class Qwen3VLTokenizer(_Tok):
        pass

    class _Clip:
        def __init__(self, tk):
            self.tokenizer = tk

    check("P8 an H3 tokenizer is recognised",
          _is_h3(_Clip(MiniMaxH3Tokenizer())) is True)
    check("P8 a plain Qwen3-VL tokenizer is NOT",
          _is_h3(_Clip(Qwen3VLTokenizer())) is False,
          "same inner name, different model -- this is the whole point")
    check("P8 no clip is not H3", _is_h3(None) is False)
else:
    check("P8 _is_h3_encoder could be lifted", False)
# NOT a bare substring: "not a cap" also appears in the explanatory comment
# block above, so searching the whole file passed a mutation that rewrote the
# REPORT line to "the encoder limit". Anchor on the line that is printed.
_span_line = [l for l in src.split("\n")
              if "H3_TRAIN_SPAN} tokens" in l]
check("P7 the printed over-span line says it is NOT a cap",
      len(_span_line) == 1 and "not a cap" in _span_line[0],
      "got %r" % (_span_line or "no such line"))
check("P7 the late-lines advice is present",
      "LAST" in src and "pull weaker" in src,
      "the actionable half of the finding")

if _FAILED:
    print("[test_v907_h3_reach] FAIL (%d of %d):"
          % (len(_FAILED), len(_FAILED) + len(_PASSED)))
    for f in _FAILED:
        print("   - " + f)
    sys.exit(1)
print("[test_v907_h3_reach] PASS: %d promises -- H3 does not truncate, and the "
      "number that does mean something is how far the prompt pushes the clip "
      "along the shared position axis." % len(_PASSED))
