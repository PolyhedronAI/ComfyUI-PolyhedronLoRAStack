#!/usr/bin/env python3
"""
test_v619_external -- the read-only EXT_POS / EXT_NEG display fields.

Frank wanted the RESOLVED external text (pos_external / neg_external -- whether from a
static STRING node, a Join Strings chain, or Florence2) shown ON the CLIP Text Encode node.
The robust path: the BACKEND resolves it (one path for static, combined, and dynamic) and
sends it back over pls_cte; the frontend shows it in read-only, fixed-height, tinted fields
that appear only when the matching input PIN is wired.

What this guards against:
  - the fields becoming CANON / serialized (they are DERIVED from wired inputs, not user
    data -- serializing them would corrupt the frozen append-only canon round-trip);
  - the fields auto-fitting and re-becoming the v600 mural that shoved editing off screen;
  - the external text silently NOT counting (the words-vs-tokens gap Frank flagged);
  - the backend not sending the text at all.
"""
import os
import re
import sys
import json
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JS = open(os.path.join(ROOT, "web/js/ph_clip_encode.js"), encoding="utf-8").read()
PY = open(os.path.join(ROOT, "nodes/ph_clip_encode.py"), encoding="utf-8").read()


def _fail(msg):
    print("[test_v619_external] FAIL: " + msg)
    sys.exit(1)


def _lift(sig):
    s = JS[JS.index(sig):]
    return s[:s.index("\n}") + 2]


# --- the backend must SEND the resolved external text (the quoted pls_cte keys) ---
if '"pos_ext"' not in PY or '"neg_ext"' not in PY:
    _fail("the Python node does not send the resolved external text (the \"pos_ext\"/\"neg_ext\" "
          "keys on pls_cte) -- without them the frontend has nothing to show, and only a run can "
          "resolve a Join Strings / Florence2 chain the frontend cannot read")

# --- the EXT fields must be READ-ONLY and NON-SERIALIZED (derived, not user input) ---
_mk = JS[JS.index("function _makeExtField"):]
_mk = _mk[:_mk.index("\n}") + 2]
if "readOnly = true" not in _mk:
    _fail("the EXT field is not read-only -- it is a display of resolved external text, "
          "never edited")
if "serialize: false" not in _mk or "serialize = false" not in _mk:
    _fail("the EXT field is serialized -- it is DERIVED from wired inputs, not user data; it "
          "must never enter the saved workflow (that is what keeps the canon append-only)")

# --- the EXT fields must NOT be canon (never in CANON or DISPLAY -> they ride last) ---
_canon = re.search(r"const CANON = \[(.*?)\];", JS, re.S).group(1)
_display = re.search(r"const DISPLAY = \[(.*?)\];", JS, re.S).group(1)
for _n in ("pls_ext_pos", "pls_ext_neg"):
    if _n in _canon or _n in _display:
        _fail("an EXT field (%s) crept into CANON/DISPLAY -- it must be a non-canon extra that "
              "rides last and never serializes, or it corrupts the frozen canon round-trip" % _n)

# --- FIXED height with scroll, not auto-fit (the v600 lesson: no unbounded text wall) ---
if "EXT_H" not in _mk or "overflowY" not in _mk:
    _fail("the EXT field is not fixed-height with scroll -- a 400-word caption would push the "
          "editing fields off screen again (the v600 mural we deleted)")

# --- shown only when the PIN is wired ---
if "_extConnected" not in JS or ".link != null" not in JS:
    _fail("the EXT fields are not gated on the input PIN being wired -- Frank asked for them to "
          "appear only when external data flows in")

# --- RUN _counterText: the external text must FOLD INTO the word/char counts ---
_MAX_SEG = int(re.search(r"const MAX_SEGMENTS = (\d+);", JS).group(1))
harness = """
const MAX_SEGMENTS = %d;
%s
%s
%s
%s
%s
%s
const mkw = (name, value) => ({ name, value });
function node(ext) {
    const w = [mkw("segments",1), mkw("pos_1","a cute cat"), mkw("pos_2",""), mkw("pos_3",""),
               mkw("pos_4",""), mkw("pos_5",""), mkw("pos_6",""),
               mkw("comment_markers","//"), mkw("strip_comments",true),
               mkw("use_negative",true), mkw("neg_1","bad hands")];
    const nd = { widgets: w,
                 _cteTokens: { pos: 100, neg: 20, method: "exact", posLen: 0, negLen: 0 } };
    if (ext) nd._cteExt = ext;
    return nd;
}
console.log(JSON.stringify({
    none: _counterText(node(null)),
    ext:  _counterText(node({ pos: "one two three four five", neg: "six seven" })),
}));
""" % (_MAX_SEG,
       _lift("function _w(node, name)"),
       _lift("function _extConnected(node, inputName)"),
       _lift("function _replacing(node, side)"),   # v1050
       _lift("function _countText(rawTxt, node)"),
       _lift("function _liveCount(node)"),
       _lift("function _counterText(node)"))

with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False, encoding="utf-8") as fh:
    fh.write(harness)
    path = fh.name
try:
    res = subprocess.run(["node", path], capture_output=True, text=True, timeout=30)
finally:
    os.unlink(path)
if res.returncode != 0:
    _fail("the counter logic did not run: %s" % res.stderr.strip()[:300])
c = json.loads(res.stdout.strip().splitlines()[-1])


def _words(txt, label):
    m = re.search(label + r" (\d+) words", txt)
    return int(m.group(1)) if m else None


# "a cute cat" = 3 pos words, "bad hands" = 2 neg words. Externals add 5 pos + 2 neg.
_pos_none, _pos_ext = _words(c["none"], "pos"), _words(c["ext"], "pos")
_neg_none, _neg_ext = _words(c["none"], "neg"), _words(c["ext"], "neg")
if _pos_none is None or _pos_ext is None or _neg_none is None or _neg_ext is None:
    _fail("could not read the pos/neg word counts from the footer text")
if _pos_ext != _pos_none + 5:
    _fail("the external POSITIVE text did not fold into the word count (%s -> %s, expected +5) "
          "-- that is the words-vs-tokens gap Frank asked to close" % (_pos_none, _pos_ext))
if _neg_ext != _neg_none + 2:
    _fail("the external NEGATIVE text did not fold into the word count (%s -> %s, expected +2)"
          % (_neg_none, _neg_ext))

print("[test_v619_external] PASS: the backend sends the resolved external text; the EXT fields "
      "are read-only, non-serialized, non-canon (ride last), fixed-height with scroll, and shown "
      "only when the PIN is wired; and the external text folds into the word/char counts (RUN), "
      "closing the words-vs-tokens gap.")
sys.exit(0)
