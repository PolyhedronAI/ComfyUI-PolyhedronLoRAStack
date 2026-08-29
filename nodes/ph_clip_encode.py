"""Polyhedron CLIP Text Encode (v556)

ONE node instead of five: segmented positive text + an optional negative,
composed into a single prompt each, encoded through the CORE CLIPTextEncode
(reused, never re-implemented - the _MODEL_UPSCALER pattern), with the
composed strings exposed as STRING outputs for the Token Counter and shown
inside the node.

Design notes worth keeping:

* THE ORDER MATTERS. Comments are stripped LINE-SCOPED and FIRST; only then
  are newlines flattened. The other way round a "// SCENE" header eats the
  whole following line - which is exactly what a strip_newlines-only chain
  does today (measured against Frank's Show-Any output).
* ONE token truth. `_count_tokens` is IMPORTED from uls_stack_node - the very
  function behind ULSTokenCounter. This node only ever DISPLAYS that number;
  the Token Counter stays the authority with its report / limit / over_limit
  contract. A second counter would be a second truth, so there isn't one.
  v905: it is handed the live `clip`, so the number is exact for whichever
  encoder is wired instead of assuming WAN's UMT5-XXL.
* Hidden segments stay SERIALISED. Turning `segments` down hides fields, it
  never drops their text.
"""
import re

try:  # core node reuse (house pattern - immune to internal API drift)
    from nodes import CLIPTextEncode
    _CORE_ENCODER = CLIPTextEncode()
except Exception as _exc:  # pragma: no cover
    _CORE_ENCODER = None
    print(f"[PLS] CLIP Text Encode: core encoder unavailable ({_exc!r})")

try:  # package load (ComfyUI) vs direct module load (tools)
    from .uls_stack_node import _count_tokens
except ImportError:  # pragma: no cover
    import os as _os
    import sys as _sys
    _here = _os.path.dirname(_os.path.abspath(__file__))
    if _here not in _sys.path:
        _sys.path.insert(0, _here)
    from uls_stack_node import _count_tokens

MAX_SEGMENTS = 6
_SEPARATORS = {"comma": ", ", "newline": "\n", "space": " ", "none": ""}
_DEFAULT_MARKERS = "//"
_RX_CACHE = {}


def _comment_rx(markers):
    """Build (and cache) the comment regex for a WHITESPACE-SEPARATED list of
    markers, e.g. '// # ***'. A marker only counts at a line start or after
    whitespace, which is what spares "https://..." by construction. Everything
    from the marker to the end of the line goes - so '//Comment' and
    '// Comment' are both caught (no leading-space rule to remember)."""
    key = str(markers or "")
    if key in _RX_CACHE:
        return _RX_CACHE[key]
    toks = [t for t in key.split() if t]
    rx = (re.compile(r"(^|\s)(?:" + "|".join(re.escape(t) for t in toks) + r").*$")
          if toks else None)   # no marker -> nothing to strip (fail soft)
    _RX_CACHE[key] = rx
    return rx


def _clean(text, strip_comments, strip_newlines, markers=_DEFAULT_MARKERS):
    """Clean ONE block. Comments first (line-scoped), newlines second - the
    reverse order lets a header comment swallow the line behind it."""
    if not text:
        return ""
    rx = _comment_rx(markers) if strip_comments else None
    if rx is not None:
        kept = []
        for line in text.split("\n"):
            cut = rx.sub("", line)
            if cut.strip():
                kept.append(cut)
        text = "\n".join(kept)
    if strip_newlines:
        text = " ".join(text.split("\n"))
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _compose(parts, separator, strip_comments, strip_newlines,
             external="", external_mode="append", markers=_DEFAULT_MARKERS):
    """Pure composition: clean every block, drop the empty ones, fold in the
    external text (append / prepend / replace), join with the separator."""
    blocks = [_clean(p, strip_comments, strip_newlines, markers) for p in parts]
    blocks = [b for b in blocks if b]
    ext = _clean(external or "", strip_comments, strip_newlines, markers)
    if ext:
        mode = str(external_mode)
        if mode == "replace":
            blocks = [ext]
        elif mode == "prepend":
            blocks = [ext] + blocks
        else:
            blocks = blocks + [ext]
    return _SEPARATORS.get(str(separator), ", ").join(blocks)


# ---- v876: does anyone actually read our conditioning? --------------------
POS_SLOT = 0
NEG_SLOT = 1


def slot_consumed(prompt, unique_id, slot):
    """Does any node in the RUNNING graph read output `slot` of this node?

    WHY THIS EXISTS: encode() used to run two full text encodes unconditionally.
    Wired as a pure text composer -- positive_text into another node's prompt
    field, which is exactly how the MiniMax Reference chain uses it -- that is
    two forward passes through a 32B encoder per run whose results are thrown
    away.

    THIRD SEAT OF THIS PROBE, and deliberately so. ph_switch carries
    ULSAnySwitchInv._consumed_outputs, ph_vectorize carries report_consumed,
    and this is a third body of the same idea. The reason is ARCHITECTURAL, not
    laziness: these node modules are kept IMPORT-INDEPENDENT so their guards can
    load them flat -- test_v869 does a bare `import ph_vectorize`. A shared
    module would break that on the first relative import, which is also why
    v868 wrote "ph_switch is left untouched: its promise is its own".

    THE DEFAULT IS INVERTED HERE, on purpose. ph_vectorize treats an unknown
    prompt (an API call, an older frontend) as NOT wired -- the quiet answer, so
    nobody is surprised by an extra image. Here an unknown prompt counts as
    CONSUMED and we encode: skipping an encode that IS needed breaks the run,
    while an unnecessary one costs seconds. The quiet answer follows whichever
    failure is worse, and that is not the same direction every time.
    """
    if not prompt or unique_id is None:
        return True
    uid = str(unique_id)
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        for value in (node.get("inputs") or {}).values():
            if (isinstance(value, (list, tuple)) and len(value) == 2
                    and str(value[0]) == uid and int(value[1]) == int(slot)):
                return True
    return False


def _encode(clip, text):
    if _CORE_ENCODER is None:
        raise RuntimeError("CLIP Text Encode: the core CLIPTextEncode node is "
                           "unavailable (changed ComfyUI Core API?). Nothing "
                           "was encoded.")
    return _CORE_ENCODER.encode(clip, text)[0]


class ULSCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        seg = {}
        for i in range(1, MAX_SEGMENTS + 1):
            seg[f"pos_{i}"] = ("STRING", {
                "multiline": True, "dynamicPrompts": True, "default": "",
                "tooltip": f"Positive segment {i}. Keep your sections apart "
                           "for readability - they are joined into ONE prompt "
                           "by the separator. Hidden segments keep their text."})
        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "The CLIP / text encoder to use "
                                             "(same pin as the core node)."}),
                "segments": ("INT", {"default": 3, "min": 1,
                                     "max": MAX_SEGMENTS, "step": 1,
                                     "tooltip": "How many positive segments are "
                                                "visible. Turning this DOWN only "
                                                "hides fields - their text stays "
                                                "saved."}),
                # ============================================================
                # THIS ORDER IS THE CANON, AND THE CANON IS APPEND-ONLY.
                #
                # v585 wrote this law after measuring it the hard way: the live
                # frontend serialises widgets_values in WIDGET ORDER. v584 moved
                # one widget into the middle and every saved graph loaded shifted
                # by a slot -- the seed landed in cfg_low.
                #
                # v603 broke the law again, in this very file, and Frank's node
                # came back with his prompt inside `separator` and the string
                # "true" in a prompt box. The heal that was supposed to catch it
                # never fired, because the heal was fighting the symptom.
                #
                # The filters DO now appear above the prompt boxes -- Frank asked
                # for that and he was right. But the reorder lives in the JS as a
                # DISPLAY permutation, and ph_clip_encode.js maps display -> canon
                # on save and canon -> display on load. The file on disk stays in
                # THIS order, forever, and every graph ever saved keeps loading.
                #
                # New widgets join at the END. There is no other legal move.
                # ============================================================
                **seg,
                "use_negative": ("BOOLEAN", {"default": True,
                                             "label_on": "On", "label_off": "Off",
                                             "tooltip": "Off hides the negative box "
                                                        "and encodes an EMPTY negative "
                                                        "(a valid conditioning - the "
                                                        "graph stays intact)."}),
                "neg_1": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                     "default": "",
                                     "tooltip": "The negative prompt."}),
                "separator": (list(_SEPARATORS.keys()),
                              {"default": "comma",
                               "tooltip": "How the segments are joined into one "
                                          "prompt (comma = ', ')."}),
                "strip_comments": ("BOOLEAN", {"default": True,
                                               "label_on": "On", "label_off": "Off",
                                               "tooltip": "Remove // comments BEFORE flattening newlines. Your section headers stay in "
                                                          "the box but never reach the model (left in, they would cost tokens). Off = the"
                                                          " historic behaviour."}),
                "strip_newlines": ("BOOLEAN", {"default": True,
                                               "label_on": "On", "label_off": "Off",
                                               "tooltip": "Flatten line breaks into "
                                                          "spaces after the comment pass."}),
                "external_mode": (["append", "prepend", "replace"],
                                  {"default": "append",
                                   "tooltip": "What to do with a wired external text "
                                              "(e.g. Florence2): add it behind, in "
                                              "front, or let it REPLACE the segments."}),
                # v557: belongs to strip_comments (appended LAST - the
                # serialisation law: every existing index keeps its slot).
                "comment_markers": ("STRING", {"default": _DEFAULT_MARKERS,
                                               "multiline": False,
                                               "tooltip": "Which markers start a comment "
                                                          "(used by strip_comments). "
                                                          "SPACE-SEPARATED list, e.g. "
                                                          "'// # ***'. A marker counts at "
                                                          "the line start or after a space, "
                                                          "and everything up to the end of "
                                                          "the line goes - so '//Comment' "
                                                          "and '// Comment' both work, while "
                                                          "'https://...' survives. Empty = "
                                                          "nothing is stripped."}),
            },
            "hidden": {
                # v876: hidden inputs are injected at runtime and never touch
                # widgets_values, so both baselines stand.
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
            "optional": {
                "pos_external": ("STRING", {"forceInput": True, "default": "",
                                            "tooltip": "External positive text (Florence2, "
                                                       "a Stack's trigger words, ...). It "
                                                       "shows up in the node preview."}),
                "neg_external": ("STRING", {"forceInput": True, "default": "",
                                            "tooltip": "External negative text."}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, prompt=None, unique_id=None, **kwargs):
        """v881: the v876 gate must not be able to hang on a stale answer.

        THE WOUND, from Frank's field log (26.08.). The Power Upscale died in
        CORE with `TypeError: 'NoneType' object is not iterable`
        (comfy/sampler_helpers.py:72) because a CONDITIONING input was None --
        and the run's log carried no CLIP Encode line at all: the node had not
        executed.

        WHY. `slot_consumed()` reads the RUNNING GRAPH through the hidden
        PROMPT input. Core builds a node's cache signature (comfy_execution/
        caching.py) as [class_type, IS_CHANGED] followed by that node's OWN
        inputs -- links are recorded as ("ANCESTOR", ...) of its INCOMING
        edges. Nothing about who reads its OUTPUTS is in there. So wiring the
        conditioning output somewhere new leaves the signature untouched, the
        node is served from cache, and the None it produced back when nobody
        read that slot is handed out again.

        A gate whose ANSWER depends on something outside the cache key must
        put that something INTO the key. That is the whole fix: report the
        consumption state, and a rewire becomes a cache miss.

        Returned as a string because Core compares these values for equality
        and stores them in the prompt; a plain, stable scalar is the least
        surprising thing to put there.
        """
        return "pos=%d neg=%d" % (
            int(bool(slot_consumed(prompt, unique_id, POS_SLOT))),
            int(bool(slot_consumed(prompt, unique_id, NEG_SLOT))),
        )

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("positive", "negative", "positive_text", "negative_text", "full_text")
    FUNCTION = "encode"
    CATEGORY = "Polyhedron/Conditioning"
    DESCRIPTION = ("Segmented prompt encoding: the positive prompt is assembled from "
                   "several fields instead of one wall of text, so a trigger block, a "
                   "subject and a style block can be edited without hunting through a "
                   "paragraph. Comments and stray newlines are stripped before encoding, "
                   "and the text outputs hand back exactly what was encoded - what the "
                   "encoder saw is what you can read.")

    def encode(self, clip, segments, use_negative, neg_1, separator,
               strip_comments, strip_newlines, external_mode,
               comment_markers=_DEFAULT_MARKERS,
               pos_external="", neg_external="",
               prompt=None, unique_id=None, **kwargs):
        n = max(1, min(int(segments), MAX_SEGMENTS))
        parts = [kwargs.get(f"pos_{i}", "") for i in range(1, n + 1)]
        pos_text = _compose(parts, separator, strip_comments, strip_newlines,
                            pos_external, external_mode, comment_markers)
        neg_text = (_compose([neg_1], separator, strip_comments, strip_newlines,
                             neg_external, external_mode, comment_markers)
                    if bool(use_negative) else "")

        # v876: encode only what someone reads. See slot_consumed().
        want_pos = slot_consumed(prompt, unique_id, POS_SLOT)
        want_neg = slot_consumed(prompt, unique_id, NEG_SLOT)
        pos_cond = _encode(clip, pos_text) if want_pos else None
        neg_cond = _encode(clip, neg_text) if want_neg else None
        if not (want_pos and want_neg):
            skipped = ([] if want_pos else ["positive"]) + \
                      ([] if want_neg else ["negative"])
            print("[PLS] CLIP Encode: nothing reads %s -- that encode was "
                  "SKIPPED and the output is None. Wire it and it runs again."
                  % " or ".join(skipped))

        # ONE token truth: the very function behind ULSTokenCounter.
        # v905: hand it the clip we are ALREADY encoding with. Until v904 this
        # counted with a UMT5-XXL we loaded ourselves -- right for WAN, wrong
        # for MiniMax H3 (Qwen3-VL) and Flux2-Klein (Qwen3-4B). The encoder is
        # right here in the signature; asking it removes the second tokenizer
        # rather than teaching it to guess which family it is looking at.
        pt, method = _count_tokens(pos_text, clip)
        nt, _ = _count_tokens(neg_text, clip)
        ext = ("pos" if pos_external else "") + ("+neg" if neg_external else "")
        print(f"[PLS] CLIP Encode: pos={pt} tokens ({method}) from "
              f"{len([p for p in parts if p.strip()])}/{n} segment(s) | "
              f"neg={nt} ({'on' if use_negative else 'off'}) | "
              f"external={ext or 'none'} | "
              f"comments={'stripped' if strip_comments else 'kept'}"
              f"{(' markers=' + repr(str(comment_markers))) if strip_comments else ''}")

        # v600: the composed block used to be PAINTED into the node and nowhere
        # else -- a wall of text that pushed the actual POS/NEG editing fields off
        # screen, in a node whose whole job is editing them. It is a WIRE now, not
        # a mural. Byte for byte what the pane used to render, em dash and all, so
        # nothing is lost by the move -- only relocated to where it can be used.
        full_text = pos_text + (("\n\n\u2014 negative \u2014\n" + neg_text)
                                if neg_text else "")

        # v619: the frontend shows the resolved external text in read-only EXT fields, so
        # send it back cleaned the same way it was composed (comments/newlines stripped).
        # Neg external only counts when the negative side is on, matching neg_text.
        pos_ext_clean = _clean(pos_external or "", strip_comments, strip_newlines,
                               comment_markers)
        neg_ext_clean = (_clean(neg_external or "", strip_comments, strip_newlines,
                                comment_markers) if bool(use_negative) else "")

        ui = [{
            "pos_tokens": int(pt), "neg_tokens": int(nt), "method": method,
            "pos_len": len(pos_text), "neg_len": len(neg_text),
            "pos_ext": pos_ext_clean, "neg_ext": neg_ext_clean,
        }]
        return {"ui": {"pls_cte": ui},
                "result": (pos_cond, neg_cond, pos_text, neg_text, full_text)}
