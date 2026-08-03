"""Guard v556 -- Polyhedron CLIP Text Encode.

BEHAVIOURAL where it counts: `_clean` and `_compose` are pure text math, so
they are extracted verbatim and EXECUTED against the decision matrix. The
headline case is the ORDER TRAP, measured against Frank's live Show-Any
output: a "// SCENE" header must NOT eat the line behind it - comments are
stripped line-scoped FIRST, newlines are flattened SECOND. A URL
("https://...") must survive the comment pass.

Text pins hold the contracts that need ComfyUI to run: the CORE encoder is
REUSED (never re-implemented), `_count_tokens` is IMPORTED from
uls_stack_node (ONE token truth - a second counter would be a second truth,
and Frank explicitly asked for no collision with ULSTokenCounter), the four
outputs (2 x CONDITIONING + 2 x STRING for the Token Counter), the ui
channel, the birth order, and the ISOLATED registration.

Script-style: exit 0 = pass.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fail(msg):
    print("[test_v556_clipencode] FAIL: " + msg)
    sys.exit(1)


def _read(*p):
    return open(os.path.join(ROOT, *p), encoding="utf-8").read()


def main():
    py = _read("nodes", "ph_clip_encode.py")
    js = _read("web", "js", "ph_clip_encode.js")
    init = _read("__init__.py")

    # ---- extract the pure text math and RUN it -------------------------------
    src = ["import re"]
    for rx in (r"_SEPARATORS = \{[^\n]*\}", r'_DEFAULT_MARKERS = "[^"]*"',
               r"_RX_CACHE = \{\}", r"def _comment_rx\(.*?\n(?=\n\ndef )",
               r"def _clean\(.*?\n(?=\n\ndef )",
               r"def _compose\(.*?\n(?=\n\ndef )"):
        m = re.search(rx, py, re.S)
        if not m:
            _fail(f"not extractable: {rx}")
        src.append(m.group(0))
    ns = {}
    exec("\n".join(src), ns)  # noqa: S102 - our own source, measured
    clean, compose = ns["_clean"], ns["_compose"]

    # THE ORDER TRAP (the reason this node exists)
    txt = "// SCENE & CAMERA\nRAW photo, 8K\n// BACKGROUND\nfoggy morning"
    got = clean(txt, True, True)
    if got != "RAW photo, 8K foggy morning":
        _fail(f"order trap: comments must be stripped BEFORE newlines "
              f"are flattened -> got {got!r}")
    kept = clean(txt, False, True)
    if "// SCENE" not in kept:
        _fail("with strip_comments Off the historic behaviour must remain "
              "(comments stay in the prompt)")
    if clean("a photo of https://x.io/y", True, True) != "a photo of https://x.io/y":
        _fail("the comment pass must spare a URL (// after ':' is not a comment)")
    if clean("  lots   of   space \n\n ", False, True) != "lots of space":
        _fail("whitespace collapse broke")

    # composition
    if compose(["a", "", "b"], "comma", True, True) != "a, b":
        _fail("empty segments must drop out of the join")
    if compose(["a", "b"], "newline", True, True) != "a\nb":
        _fail("separator 'newline' broke")
    if compose(["a"], "comma", True, True, "ext", "append") != "a, ext":
        _fail("external append broke")
    if compose(["a"], "comma", True, True, "ext", "prepend") != "ext, a":
        _fail("external prepend broke")
    if compose(["a"], "comma", True, True, "ext", "replace") != "ext":
        _fail("external replace must drop the segments")
    if compose(["a"], "comma", True, True, "", "replace") != "a":
        _fail("an EMPTY external must never wipe the segments")
    if compose(["// only a comment"], "comma", True, True) != "":
        _fail("a comment-only segment must vanish entirely")

    # ---- one token truth ------------------------------------------------------
    if "from .uls_stack_node import _count_tokens" not in py or \
       "from uls_stack_node import _count_tokens" not in py:
        _fail("_count_tokens must be IMPORTED from uls_stack_node in BOTH "
              "branches - ONE token truth with ULSTokenCounter (no second "
              "counter, no collision)")
    if "def _count_tokens" in py or "def _heuristic" in py:
        _fail("a second token counter crept in - that is a second truth")

    # ---- core encoder reuse ---------------------------------------------------
    if "from nodes import CLIPTextEncode" not in py or \
       "_CORE_ENCODER = CLIPTextEncode()" not in py:
        _fail("the CORE CLIPTextEncode must be REUSED (the _MODEL_UPSCALER "
              "pattern), never re-implemented")
    if "encode_from_tokens" in py:
        _fail("the encode internals must not be re-implemented")

    # ---- contract -------------------------------------------------------------
    # v600 added a FIFTH output (full_text). The four that were here are still
    # here, in the same slots, and that is the whole point: LiteGraph wires by
    # INDEX, so inserting a new output anywhere but the end would silently
    # re-route every existing workflow -- positive would land where negative was
    # expected, and nothing would say a word. Appended, never inserted.
    if 'RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING")' not in py:
        _fail("the output tuple changed. Four were here (2x COND + 2x STRING for the Token "
              "Counter) and v600 appended full_text as the fifth.")
    if 'RETURN_NAMES = ("positive", "negative", "positive_text", "negative_text", "full_text")' \
            not in py:
        _fail("the output ORDER moved. `full_text` must be LAST. LiteGraph wires by slot index, "
              "so putting it anywhere else re-routes every saved workflow without a single "
              "error message -- the worst kind of breakage there is.")
    if '"pls_cte"' not in py:
        _fail("the ui channel pls_cte is gone")
    if "def encode" not in py or "use_negative" not in py:
        _fail("the negative toggle is gone")
    if 'if bool(use_negative) else ""' not in py:
        _fail("use_negative=Off must encode an EMPTY negative (a valid "
              "conditioning - never None)")

    req = py[py.index('"required"'):py.index('"optional"')]
    names = re.findall(r'"([a-z_0-9]+)":\s*\(', req) + \
            re.findall(r'f"(pos_)\{i\}"', py)
    for must in ("clip", "segments", "use_negative", "neg_1", "separator",
                 "strip_comments", "strip_newlines", "external_mode"):
        if must not in names:
            _fail(f"required input {must!r} is gone")
    if "MAX_SEGMENTS = 6" not in py:
        _fail("the segment budget changed - the JS mirrors it")

    # ---- frontend -------------------------------------------------------------
    # The EXACT banner version is pinned by the file's newest guard (v557);
    # here we pin existence + format only (v531 doctrine).
    if not re.search(r"\[PLS\] ph_clip_encode\.js v\d+ loaded", js):
        _fail("per-file load banner missing (v531 doctrine)")
    if "const MAX_SEGMENTS = 6" not in js:
        _fail("the JS segment budget must mirror the backend")
    if "onDrawForeground" not in js or "_counterText" not in js:
        _fail("the wordcount counter is gone -- it must be painted via onDrawForeground "
              "(v613) so no widget layout can hide it")
    if "message.pls_cte" not in js:
        _fail("onExecuted no longer reads pls_cte for the token count")
    if "no tokenizer" not in js and "has no tokenizer" not in js:
        _fail("the honest split (browser counts words, backend counts tokens) "
              "must stay documented")
    # v531 height-only invariant, enforced by MEANING not by one literal: every
    # setSize the node makes must keep the width as the current *.size[0]. The fields
    # are native now, but the pane/visibility refits still call setSize -- so check the
    # contract (width is never touched), not any one call shape.
    import re as _re
    _sizings = list(_re.finditer(r"setSize\(\[\s*([^,]+),", js))
    if not _sizings:
        _fail("the node never sizes itself -- the pane/height fit is gone")
    for _m in _sizings:
        if ".size[0]" not in _m.group(1):
            _fail("a setSize sets the WIDTH to %r -- must be height-only (v531)"
                  % _m.group(1).strip())

    # v602: green = positive, brown = negative -- and Frank said ALWAYS, which means
    # after every load, not just at birth. _applyTints used to run in onNodeCreated
    # alone; a reloaded graph got its tints only because onNodeCreated happens to
    # fire on load too. That is luck, not a contract. It is called on the reload
    # path now, next to _applyVisibility, where it can be seen to be called.
    reload_path = js[js.index("Loading a saved graph"):] if "Loading a saved graph" in js else ""
    if "_applyTints" not in reload_path:
        _fail("the tints must be re-applied on the RELOAD path. Frank asked for positive ALWAYS "
              "green and negative ALWAYS brown -- 'always' includes the graph he opens tomorrow.")
    if "_applyVisibility" not in reload_path:
        _fail("visibility must be re-applied on the reload path -- segments=2 must not show six "
              "boxes after a reload")

    # ---- isolated registration -------------------------------------------------
    if "_CTE_OK" not in init or 'NODE_CLASS_MAPPINGS["ULSCLIPTextEncode"]' not in init:
        _fail("CLIP Text Encode must register through its own _CTE_OK flag")
    if "\u2b21 Polyhedron CLIP Text Encode" not in init:
        _fail("the display name must carry the \u2b21 prefix (house style)")

    print("PASS: v556 CLIP Text Encode -- order trap + composition matrix "
          "executed, ONE token truth, core encoder reused, isolated")
    sys.exit(0)


if __name__ == "__main__":
    main()
