# -*- coding: ascii -*-
"""v905 -- the token count comes from the encoder that will actually run.

WHAT WENT WRONG BEFORE THIS CUT
-------------------------------
The counter loaded a UMT5-XXL tokenizer of its own, next to the one ComfyUI
had already loaded, and reported 512 as "the model limit". Both halves were
wrong, and both were measured against core v0.32.0 before this guard existed:

  * WRONG TOKENIZER for anything but WAN. MiniMax H3 encodes with Qwen3-VL,
    Flux2-Klein with Qwen3-4B.
  * WRONG LIMIT. `min_length` is a PAD FLOOR. Every modern family carries
    max_length=99999999 and pad_to_max_length=False; umt5xxl (WAN) has
    min_length=512. They pad UP and never truncate. comfy/ldm/wan/model.py
    assigns self.text_len = 512 and never reads it again.

THE ONE THAT BITES, AND WHY IT IS FIRST
---------------------------------------
Counting a live tokenizer WITHOUT min_length=0 returns the PAD FLOOR as if it
were the prompt. Measured on the real core: KleinTokenizer on a 13-word
sentence gives 512 by default and 29 with min_length=0. A count that silently
reports 512 for every short prompt is worse than the old guess, because it
looks authoritative.

So P1 does not read the source for the string "min_length". It hands the code
a tokenizer that PADS EXACTLY LIKE THE CORE DOES and checks the number that
comes back. Drop the kwarg and the fake pads, the count jumps to the floor,
and the guard goes red. That is the v903 lesson applied on the way in: a
promise about behaviour is RUN, never read.
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _lift  # noqa: E402

SRC_PATH = os.path.join(HERE, "..", "nodes", "uls_stack_node.py")
CTE_PATH = os.path.join(HERE, "..", "nodes", "ph_clip_encode.py")
SRC = open(SRC_PATH, encoding="utf-8").read()

_FAILED = []
_PASSED = []


def check(label, cond, detail=""):
    if cond:
        _PASSED.append(label)
    else:
        _FAILED.append(label + ((" -- " + detail) if detail else ""))


# ---------------------------------------------------------------------------
# the fake core. Shaped from a MEASURED run against comfy v0.32.0:
#   KleinTokenizer -> max_length=99999999 min_length=512 pad_to_max=False
#   default tokenize -> 512 tokens ; min_length=0 -> 29 tokens
# ---------------------------------------------------------------------------
class FakeInner(object):
    """One inner SDTokenizer, padding exactly the way sd1_clip.py does."""

    def __init__(self, key, max_length=99999999, min_length=512,
                 pad_to_max_length=False, pad_token=151643):
        self.embedding_key = key
        self.max_length = max_length
        self.min_length = min_length
        self.pad_to_max_length = pad_to_max_length
        self.pad_token = pad_token

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        ids = [(hash(w) & 0xFFFF, 1.0) for w in text.split()]
        # the core chunks against max_length ...
        cap = self.max_length
        chunks = [ids[i:i + cap] for i in range(0, max(1, len(ids)), cap)] or [[]]
        # ... then pads the last chunk up to min_length, which is what the
        # caller must switch off to see the real number.
        floor = kwargs.get("min_length", self.min_length)
        if floor and len(chunks[-1]) < floor:
            chunks[-1] = chunks[-1] + [(self.pad_token, 1.0)] * (floor - len(chunks[-1]))
        return chunks


class FakeTokenizer(object):
    def __init__(self, *inners):
        for inner in inners:
            setattr(self, inner.embedding_key, inner)
        self._inners = inners

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        return dict((i.embedding_key, i.tokenize_with_weights(text, **kwargs))
                    for i in self._inners)


class FakeCLIP(object):
    def __init__(self, tokenizer, accept_min_length=True):
        self.tokenizer = tokenizer
        self._accept = accept_min_length
        self.calls = []

    def tokenize(self, text, **kwargs):
        self.calls.append(dict(kwargs))
        if "min_length" in kwargs and not self._accept:
            raise TypeError("this tokenizer does not take min_length")
        return self.tokenizer.tokenize_with_weights(text, **kwargs)


# ---------------------------------------------------------------------------
# lift the seam out of the shipped source (no comfy, no torch)
# ---------------------------------------------------------------------------
PROVIDED = {"print", "int", "str", "bool", "sum", "len", "vars", "getattr",
            "isinstance", "dict", "TypeError", "ValueError", "Exception",
            "re", "max", "round", "min"}
seam, missing = _lift.close_over(
    SRC, ["_encoder_facts", "_count_with_clip", "_count_tokens"], PROVIDED)
if missing:
    print("[test_v905_live_encoder_count] FAIL: lift is short of %s"
          % sorted(missing))
    sys.exit(1)

import re  # noqa: E402
NS = {"re": re, "print": lambda *a, **k: None}
exec(compile(seam, SRC_PATH, "exec"), NS)
encoder_facts = NS["_encoder_facts"]
count_with_clip = NS["_count_with_clip"]
count_tokens = NS["_count_tokens"]

TEXT = "a b c d e f g h i j k l m"          # 13 "words"


# ---------------------------------------------------------------------------
# P1 -- THE PAD FLOOR IS NOT THE PROMPT. Run, do not read.
# ---------------------------------------------------------------------------
clip = FakeCLIP(FakeTokenizer(FakeInner("qwen3_4b", min_length=512)))
n, name = count_with_clip(clip, TEXT)
check("P1 real length, not the pad floor", n == 13,
      "got %r (512 means min_length=0 never reached the tokenizer)" % (n,))
check("P1 names the encoder", name == "qwen3_4b", "got %r" % (name,))
check("P1 passes min_length=0 to the live tokenizer",
      clip.calls and clip.calls[0].get("min_length") == 0,
      "kwargs were %r" % (clip.calls,))

# the same fake WITHOUT the kwarg reaching it must pad -- proves the fixture
# can actually show the wound, so a green P1 means something.
padded = FakeTokenizer(FakeInner("qwen3_4b", min_length=512)) \
    .tokenize_with_weights(TEXT)
check("P1 fixture can show the wound",
      sum(len(c) for c in padded["qwen3_4b"]) == 512,
      "the fake does not pad, so P1 could not fail")


# ---------------------------------------------------------------------------
# P2 -- a tokenizer that refuses min_length=0 is DECLINED, not believed
# ---------------------------------------------------------------------------
strict = FakeCLIP(FakeTokenizer(FakeInner("umt5xxl", min_length=512)),
                  accept_min_length=False)
n2, _ = count_with_clip(strict, TEXT)
check("P2 declines rather than reporting a padded count", n2 is None,
      "got %r -- that number would be the floor, not the prompt" % (n2,))


# ---------------------------------------------------------------------------
# P3 -- no cap is reported as NO CAP; a real cap is reported as a cap
# ---------------------------------------------------------------------------
facts = encoder_facts(FakeCLIP(FakeTokenizer(
    FakeInner("umt5xxl", max_length=99999999, min_length=512))))
check("P3 one encoder found", len(facts) == 1, "got %r" % (facts,))
if facts:
    check("P3 99999999 means no cap", facts[0]["cap"] is None,
          "got cap=%r -- the report would claim a limit that does not exist"
          % (facts[0]["cap"],))
    check("P3 the floor is carried through", facts[0]["pad_floor"] == 512,
          "got %r" % (facts[0]["pad_floor"],))
    check("P3 the encoder is named", facts[0]["name"] == "umt5xxl")

capped = encoder_facts(FakeCLIP(FakeTokenizer(
    FakeInner("sam3_clip", max_length=32, min_length=0,
              pad_to_max_length=True))))
check("P3 a REAL cap survives", capped and capped[0]["cap"] == 32,
      "got %r -- a genuinely truncating encoder must still be flagged"
      % (capped,))

check("P3 no clip yields no claim", encoder_facts(None) == [],
      "must be empty, never a guess")


# ---------------------------------------------------------------------------
# P4 -- multi-encoder objects are all read, and the count is the largest
# ---------------------------------------------------------------------------
multi = FakeCLIP(FakeTokenizer(FakeInner("clip_l", max_length=77,
                                         min_length=0, pad_to_max_length=True),
                               FakeInner("t5xxl", min_length=256)))
mfacts = encoder_facts(multi)
check("P4 both encoders reported", len(mfacts) == 2, "got %r" % (mfacts,))
names = sorted(f["name"] for f in mfacts)
check("P4 both named", names == ["clip_l", "t5xxl"], "got %r" % (names,))


# ---------------------------------------------------------------------------
# P5 -- WITHOUT a clip the v904 behaviour is untouched
# ---------------------------------------------------------------------------
NS2 = dict(NS)
NS2["_try_load_umt5_tokenizer"] = lambda: None      # force the fallback
exec(compile(seam, SRC_PATH, "exec"), NS2)
NS2["_try_load_umt5_tokenizer"] = lambda: None
n5, m5 = NS2["_count_tokens"]("hello world")
check("P5 no clip still counts", isinstance(n5, int) and n5 > 0,
      "got %r" % (n5,))
check("P5 no clip still labels the method", m5 in ("heuristic", "exact"),
      "got %r" % (m5,))
check("P5 empty text is free", count_tokens("") == (0, "exact"))


# ---------------------------------------------------------------------------
# P6 -- the clip actually reaches the counter, in BOTH callers
# ---------------------------------------------------------------------------
tree = ast.parse(SRC)


def _calls_with_clip(tree_, func_name):
    """True when `func_name`'s body calls _count_tokens with a 2nd argument."""
    for node in ast.walk(tree_):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Name)
                        and sub.func.id == "_count_tokens"
                        and len(sub.args) >= 2):
                    return True
    return False


check("P6 ULSTokenCounter.count hands its clip over",
      _calls_with_clip(tree, "count"),
      "the socket exists but the number still comes from the guess")

cte = ast.parse(open(CTE_PATH, encoding="utf-8").read())
check("P6 the CLIP Text Encode hands its clip over",
      any(_calls_with_clip(cte, fn.name) for fn in ast.walk(cte)
          if isinstance(fn, ast.FunctionDef)),
      "the encoder is in the signature but the footer still assumes WAN")


# ---------------------------------------------------------------------------
# P7 -- `clip` is a SOCKET, never a widget (#577: widget_values is positional)
# ---------------------------------------------------------------------------
clip_decl = None
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "ULSTokenCounter":
        for fn in node.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "INPUT_TYPES":
                for ret in ast.walk(fn):
                    if isinstance(ret, ast.Return) and isinstance(ret.value, ast.Dict):
                        for k, v in zip(ret.value.keys, ret.value.values):
                            if getattr(k, "value", None) == "optional" and isinstance(v, ast.Dict):
                                for kk, vv in zip(v.keys, v.values):
                                    if getattr(kk, "value", None) == "clip":
                                        clip_decl = vv
check("P7 the clip input exists", clip_decl is not None)
if clip_decl is not None:
    first = clip_decl.elts[0] if isinstance(clip_decl, ast.Tuple) else None
    check("P7 its type is CLIP, so it never becomes a widget",
          isinstance(first, ast.Constant) and first.value == "CLIP",
          "a widget here would renumber every saved workflow")


# ---------------------------------------------------------------------------
if _FAILED:
    print("[test_v905_live_encoder_count] FAIL (%d of %d):"
          % (len(_FAILED), len(_FAILED) + len(_PASSED)))
    for f in _FAILED:
        print("   - " + f)
    sys.exit(1)
print("[test_v905_live_encoder_count] PASS: %d promises -- the count comes "
      "from the live encoder, the pad floor is never mistaken for the prompt, "
      "and a no-cap encoder is never reported as capped." % len(_PASSED))
