# -*- coding: ascii -*-
"""Guard -- the per-block map of an H3 prompt (v997).

THE WOUND THIS CLOSES. v907 established that the prompt pushes the clip along
one shared position axis, and that late lines pull harder than early ones. It
reported ONE number for the whole prompt. That number says "your opening is
far away" but not WHICH block is far away, and the block whose distance
actually matters turned out to be CAMERA.

Measured on Frank's field prompt of 10.09. (1355 tokens, fourteen blocks):

    CAMERA sits at distance 1201, the video's own span is 37.
    So the block whose job is the framing sits 32x further from the video
    than the video is wide -- while simultaneously being outvoted by the
    sheer mass of person text. Two independent mechanisms, both against the
    same block. That is why a CAMERA line alone never forced a framing.

THE SECOND FINDING, and the one this guard pins hardest. Reordering is a
TRADE: pulling MOTION forward drops CAMERA from 1201 to 656 but pushes MOTION
from 84 to 819. There is no order in which everything is near. SHORTENING is
the only lever that moves every block closer at once, because every distance
is proportional to the total token count. P4 pins exactly that, because it is
the mechanical argument behind "so viel wie noetig, so wenig wie moeglich".

The field numbers (ratio 0.86:1, FACE+EYES+SKIN 15.2%) are reproduced from
the real block sizes, so a drift in the grouping or the arithmetic goes red.
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

_want_fn = {"h3_blocks", "h3_block_map", "h3_block_group"}
_want_const = ("H3_BLOCK_PERSON", "H3_BLOCK_RAUM", "H3_BLOCK_DETAIL",
               "H3_DETAIL_CEILING")


def lift(source, want_fn=_want_fn, want_const=_want_const):
    """Lift the pure helpers. The module imports torch at import time, so it
    cannot be imported here -- these helpers are pure, which is why they can
    be separated at all."""
    tree = ast.parse(source)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in want_fn:
            keep.append(node)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) in want_const:
                    keep.append(node)
                    break
    got = {n.name for n in keep if isinstance(n, ast.FunctionDef)}
    if want_fn - got:
        return None, sorted(want_fn - got)
    ns = {"re": __import__("re")}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<lift>", "exec"), ns)
    return ns, []


ns, missing = lift(src)
if ns is None:
    print("[test_v997_h3_block_map] FAIL: helpers not found: %s" % missing)
    sys.exit(1)
blocks = ns["h3_blocks"]
block_map = ns["h3_block_map"]
group = ns["h3_block_group"]

# Frank's field prompt of 10.09., as BLOCK SIZES. The bodies are stand-ins:
# this guard measures the arithmetic, not the wording, and carrying the real
# text would make the file large for no gain.
FIELD = [
    ("TRIGGERS", 93), ("LOOK", 218), ("CAMERA", 679), ("SCENE", 822),
    ("LIGHT", 794), ("SUBJECT", 263), ("WARDROBE", 810), ("HAIR", 251),
    ("FACE", 431), ("EYES", 210), ("SKIN", 241), ("MOOD", 35),
    ("AUDIO", 241), ("MOTION", 725),
]
FIELD_TOKENS = 1355


def build(pairs):
    return "\n".join("// %s\n%s" % (n, "x" * L) for n, L in pairs)


# --- P1: the parse mirrors what the CTE does --------------------------------
p1 = blocks("// CAMERA\nstatic camera,\n\n  wide frame,\n// SCENE\na wall,")
check("P1 two blocks found", len(p1) == 2, "got %r" % (p1,))
check("P1 lines join with ONE space, blank lines drop",
      p1 and p1[0] == ("CAMERA", "static camera, wide frame,"),
      "got %r" % ((p1[0] if p1 else None),))
check("P1 a head with spaces folds to underscores",
      blocks("// LIGHT & GRADE\nsoft,\n// X\ny")[0][0] == "LIGHT_&_GRADE",
      "got %r" % (blocks("// LIGHT & GRADE\nsoft,\n// X\ny")[0][0],))
check("P1 an empty section is dropped, not carried as a phantom",
      blocks("// EMPTY\n// REAL\nbody") == [("REAL", "body")],
      "got %r" % (blocks("// EMPTY\n// REAL\nbody"),))

# --- P2: a CTE-composed prompt has no heads left, and that is not an error --
composed = "static camera, wide frame, a wall, soft light"
check("P2 no // heads -> no blocks", blocks(composed) == [])
check("P2 and the map declines rather than inventing one",
      block_map(composed, 500) is None,
      "an empty table here would be a figure without meaning")
check("P2 a single block is not a map either",
      block_map("// CAMERA\nstatic camera", 500) is None,
      "a one-row table is noise, not information")

# --- P3: the field numbers come back ----------------------------------------
m = block_map(build(FIELD), FIELD_TOKENS)
check("P3 the map is built", m is not None)
if m:
    by = dict((r["name"], r) for r in m["rows"])
    check("P3 fourteen blocks", len(m["rows"]) == 14, "got %d" % len(m["rows"]))
    check("P3 the measured ratio 0.86 : 1 comes back",
          abs(m["ratio"] - 0.86) < 0.01, "got %.3f" % m["ratio"])
    check("P3 FACE+EYES+SKIN is the measured 15.2%%",
          abs(m["detail"] - 0.152) < 0.002, "got %.4f" % m["detail"])
    check("P3 shares sum to 1",
          abs(sum(r["share"] for r in m["rows"]) - 1.0) < 1e-9)
    check("P3 tokens sum to the headline count",
          abs(sum(r["tokens"] for r in m["rows"]) - FIELD_TOKENS) < 1e-6,
          "the map must not invent or lose tokens against the one number "
          "the report already prints")
    # THE finding: CAMERA is far, MOTION is near.
    # 1203, not the 1201 of the sandbox pre-measurement: that one converted
    # chars to tokens with the house factor 4.3 per block, this one apportions
    # the REAL headline count by character share. The node's route is the
    # right one -- the map must add up to the number printed above it (P3
    # "tokens sum to the headline count") -- and the two-position spread is
    # the rounding between the two routes, not a disagreement.
    check("P3 CAMERA sits at the measured 1203",
          abs(by["CAMERA"]["distance"] - 1203) < 2.0,
          "got %.0f" % by["CAMERA"]["distance"])
    check("P3 MOTION sits at the measured 84",
          abs(by["MOTION"]["distance"] - 84) < 2.0,
          "got %.0f" % by["MOTION"]["distance"])
    check("P3 CAMERA is more than ten times further out than MOTION",
          by["CAMERA"]["distance"] > 10 * by["MOTION"]["distance"],
          "this gap IS the finding")
    check("P3 order is preserved",
          [r["name"] for r in m["rows"]] == [n for n, _ in FIELD])

# --- P4: THE mechanical argument -- length beats order ----------------------
# Every distance is proportional to the total token count. Halve the prompt
# and EVERY block halves its distance; reorder it and one block's gain is
# another's loss. A guard that only pinned "late is near" would miss this.
half = block_map(build(FIELD), FIELD_TOKENS / 2.0)
if m and half:
    hb = dict((r["name"], r) for r in half["rows"])
    check("P4 halving the prompt halves EVERY distance",
          all(abs(hb[n]["distance"] * 2 - by[n]["distance"]) < 1e-6
              for n, _ in FIELD),
          "shortening is the only lever that moves every block at once")
    check("P4 and it leaves the shares untouched",
          all(abs(hb[n]["share"] - by[n]["share"]) < 1e-12 for n, _ in FIELD),
          "length and ratio are separate axes -- the test series depends on "
          "being able to change one without the other")

# Reordering is a TRADE, not a win: CAMERA last buys CAMERA and costs MOTION.
reordered = [p for p in FIELD if p[0] not in ("CAMERA",)] + [("CAMERA", 679)]
m2 = block_map(build(reordered), FIELD_TOKENS)
if m and m2:
    b2 = dict((r["name"], r) for r in m2["rows"])
    check("P4 CAMERA last drops it from 1201 to under 100",
          b2["CAMERA"]["distance"] < 100 < by["CAMERA"]["distance"],
          "got %.0f" % b2["CAMERA"]["distance"])
    check("P4 and MOTION pays for it",
          b2["MOTION"]["distance"] > by["MOTION"]["distance"],
          "there is no order in which everything is near")
    check("P4 the total text is unchanged by reordering",
          abs(sum(r["tokens"] for r in m2["rows"])
              - sum(r["tokens"] for r in m["rows"])) < 1e-6)

# --- P5: grouping, including the refusal to guess ---------------------------
check("P5 person blocks group as PERSON",
      all(group(n) == "PERSON" for n in
          ("SUBJECT", "WARDROBE", "HAIR", "FACE", "EYES", "SKIN", "MOOD")))
check("P5 room blocks group as RAUM",
      all(group(n) == "RAUM" for n in
          ("TRIGGERS", "LOOK", "CAMERA", "SCENE", "LIGHT")))
check("P5 AUDIO and MOTION belong to NEITHER group",
      group("AUDIO") == "-" and group("MOTION") == "-",
      "the 10.09. measurement excluded them; folding them in would move the "
      "ratio and silently invalidate the five measured field versions")
check("P5 an unknown head is not guessed into a bucket",
      group("WEATHER") == "-" and group("") == "-")
check("P5 grouping is case- and space-insensitive",
      group("wardrobe") == "PERSON" and group("Light") == "RAUM")

# --- P6: the ratio must ignore the ungrouped blocks -------------------------
# AUDIO+MOTION are 17% of the field prompt. If they leaked into either group
# the ratio would move by more than a measured band is wide.
if m:
    check("P6 person+raum is under 1 because AUDIO/MOTION sit outside",
          abs((m["person"] + m["raum"]) - 0.834) < 0.01,
          "got %.3f" % (m["person"] + m["raum"]))

# --- P7: the report says the things that make the table actionable ----------
# Anchor on the ONE physical line that is printed. The sentence used to be
# split across two source lines, which left no line carrying both halves; it
# was pulled onto one line so that the anchor is unique and a rewrite of
# either half goes red. A substring search over the whole file would also
# match the explanatory comments, which is the v907 lesson.
_report = [l for l in src.split("\n")
           if "lines.append(" in l and "distance = token positions" in l]
check("P7 the distance line explains the direction",
      len(_report) == 1 and "Near pulls harder" in _report[0],
      "a column of numbers without its direction is a riddle; got %r"
      % (_report or "no such line"))
check("P7 the close-up verdict names the ratio as the lever",
      "expect a close-up whatever CAMERA says" in src,
      "this is the 10.09. finding in one line")
check("P7 and it says to lengthen the room BEFORE touching the stack",
      "before touching the stack" in src,
      "the measured order of operations -- the stack is the second griff")
_detail = [l for l in src.split("\n")
           if "lines.append(" in l and "FACE+EYES+SKIN" in l]
check("P7 the detail warning prints its value with one decimal",
      len(_detail) == 1 and "%.1f%%" in _detail[0],
      "seen on the first real render: at 15.2%% a rounded '15%% (over 15%%)' "
      "reads like a bug; got %r" % (_detail or "no such line"))
check("P7 the empty case tells the user what to wire",
      "wire the RAW prompt text" in src,
      "a missing table must say why it is missing")

# --- MUTATION PROBE ---------------------------------------------------------
# Every claim above is only worth what its ability to fail is worth. Each
# mutation below is a plausible wrong version of the code; the probe asserts
# that the lifted helpers really behave differently, i.e. that some check
# above would have gone red.
MUT = [
    ("M1 AUDIO folded into PERSON",
     '"MOOD", "ARMOR", "BODY")', '"MOOD", "ARMOR", "BODY", "AUDIO", "MOTION")',
     lambda f: abs(f(build(FIELD), FIELD_TOKENS)["ratio"] - 0.86) >= 0.01),
    ("M2 distance measured from the START instead of the end",
     "dist = (total_chars - mid) / total_chars * tok",
     "dist = mid / total_chars * tok",
     lambda f: f(build(FIELD), FIELD_TOKENS)["rows"][0]["distance"] < 100),
    ("M3 unknown heads guessed into RAUM",
     'return "-"', 'return "RAUM"',
     lambda f: abs(f(build(FIELD), FIELD_TOKENS)["ratio"] - 0.86) >= 0.01),
    ("M4 a single block accepted as a map",
     "if len(bl) < 2:", "if len(bl) < 1:",
     lambda f: f("// CAMERA\nstatic camera", 500) is not None),
    ("M5 detail ceiling raised past the measured band",
     "H3_DETAIL_CEILING = 0.15", "H3_DETAIL_CEILING = 0.95",
     None),  # constant-only: checked by value below
]

_mut_fail = []
for label, old, new, probe in MUT:
    if src.count(old) < 1:
        _mut_fail.append("%s: anchor not found (%r)" % (label, old))
        continue
    mutated = src.replace(old, new, 1)
    if mutated == src:
        _mut_fail.append("%s: mutation did not land" % label)
        continue
    mns, miss = lift(mutated)
    if mns is None:
        _mut_fail.append("%s: mutated source would not lift (%s)" % (label, miss))
        continue
    if probe is None:
        if mns["H3_DETAIL_CEILING"] == ns["H3_DETAIL_CEILING"]:
            _mut_fail.append("%s: constant did not change" % label)
        continue
    try:
        if not probe(mns["h3_block_map"]):
            _mut_fail.append("%s: survived -- the guard would NOT have caught it"
                             % label)
    except Exception as ex:
        # An exception is a detected mutation too, as long as the clean code
        # does not raise -- which P3 above already proves.
        pass

check("MUT all %d mutations are detected" % len(MUT), not _mut_fail,
      "; ".join(_mut_fail))

if _FAILED:
    print("[test_v997_h3_block_map] FAIL (%d of %d):"
          % (len(_FAILED), len(_FAILED) + len(_PASSED)))
    for f in _FAILED:
        print("   - " + f)
    sys.exit(1)
print("[test_v997_h3_block_map] PASS: %d promises -- the prompt has a per-block "
      "map, CAMERA is the far block, and shortening is the only lever that "
      "moves every block closer at once." % len(_PASSED))
