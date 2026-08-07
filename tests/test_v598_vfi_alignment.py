"""Guard v598 -- the alignment advisory, and the pad that must never come back.

THIS GUARD EXISTS BECAUSE THE FIRST CUT OF ph_interpolate.py WAS WRONG.

It padded the frames itself -- replicate, centred -- on the belief, taken from a
truncated grep, that IFNet.forward did not pad at all. It does: line 483,

    ph = ((h - 1) // 64 + 1) * 64
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)          # zeros, bottom-right

and it crops back at line 732. Measured against ground truth on a controlled
8px translation, the "better" pad was 4.18 dB WORSE (arch 4.7): the network was
TRAINED with the zeros ring, so a centred replicate ring is out of distribution.

So this guard pins three things, and the first is the scar:

  1. THE NODE DOES NOT PAD. Zero F.pad calls inside interpolate(). Walked in the
     AST, not grepped -- a grep is what caused this in the first place.

  2. THE ADVISORY'S MODULUS IS READ OFF THE ENGINE, not typed in twice. The
     number 64 is parsed OUT OF rife_arch.py. If the engine's pad ever changes,
     our advice goes stale, and this guard goes red the same day. Two constants
     that agree by hand agree only until one of them is edited.

  3. THE ADVICE IS ARITHMETICALLY THE ENGINE'S OWN. _alignment_cost reproduces
     `((h-1)//mod + 1)*mod - h` exactly, so what we TELL the user the engine will
     do is what the engine does.

WHAT THE ADVISORY IS WORTH (ground truth, 8px translation, common region):

    canvas 1075 (dial 1.40)  ->  49.90 dB whole /  39.80 dB border   [arch 4.7]
    canvas 1088 (mod 64)     ->  59.32 dB whole /  55.90 dB border
                                 +9.42 dB           +16.09 dB
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "nodes", "ph_interpolate.py")
ENGINE = os.path.join(ROOT, "nodes", "vfi", "rife_arch.py")

TREE = ast.parse(open(SRC, encoding="utf-8").read())


def lift(names):
    want, body, found = set(names), [], set()
    for node in TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name in want:
            body.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in want:
                    body.append(node)
                    found.add(t.id)
    assert not want - found, "not found in ph_interpolate.py: %s" % sorted(want - found)
    ns = {"math": __import__("math")}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
                 SRC, "exec"), ns)
    return ns


ns = lift(["_modulus", "_alignment_cost", "_nearest_aligned", "_scale_list",
           "_strip_training_heads", "TRAINING_ONLY_PREFIXES", "CKPT_ARCH"])

# =========================================================================
# 1. THE SCAR: the node must not pad. Ever again.
# =========================================================================
fn = None
for node in ast.walk(TREE):
    if isinstance(node, ast.FunctionDef) and node.name == "interpolate":
        fn = node
assert fn is not None, "ULSInterpolate.interpolate not found"

pads = []
for node in ast.walk(fn):
    if isinstance(node, ast.Call):
        f = node.func
        name = None
        if isinstance(f, ast.Attribute):
            name = f.attr
        elif isinstance(f, ast.Name):
            name = f.id
        if name == "pad":
            pads.append(getattr(node, "lineno", "?"))
assert not pads, (
    "interpolate() calls pad() at line(s) %s. IFNet.forward ALREADY pads to mod 64 with zeros "
    "and crops back (rife_arch.py:483 and :732). A second pad on top measured 4.18 dB WORSE "
    "against ground truth, because the network was trained with the zeros ring. Do not fight "
    "the engine -- warn the user instead." % pads)

# =========================================================================
# 2. Our modulus is READ OFF the engine, not typed in twice.
# =========================================================================
esrc = open(ENGINE, encoding="utf-8").read()
found = re.findall(r"//\s*(\d+)\s*\+\s*1\)\s*\*\s*(\d+)", esrc)
assert found, ("could not find the engine's pad arithmetic in rife_arch.py. Either the file "
               "moved or the engine changed how it pads -- both mean our advice may now be "
               "wrong, and wrong advice is worse than none.")
mods = set()
for a, b in found:
    assert a == b, "engine pad arithmetic is inconsistent: //%s +1)*%s" % (a, b)
    mods.add(int(a))
assert len(mods) == 1, "engine pads to more than one modulus %s -- the advisory cannot be a " \
                       "single number any more" % sorted(mods)
engine_mod = mods.pop()
assert ns["_modulus"]() == engine_mod, (
    "THE ADVISORY HAS GONE STALE. rife_arch.py pads to mod %d; _modulus() says %d. The node "
    "would now tell the user to dial for a boundary the engine no longer uses."
    % (engine_mod, ns["_modulus"]()))

# and it is the same for every arch we ship -- the engine's pad is unconditional
for arch in sorted(set(ns["CKPT_ARCH"].values())):
    assert ns["_modulus"](arch) == engine_mod, "arch %s: the engine pads unconditionally" % arch

# =========================================================================
# 3. The advice is the engine's own arithmetic.
# =========================================================================
checked = 0
for w in range(1, 2200):
    for h in (w, 1075, 1074, 1068, 1088, 768, 1024):
        pw, ph, aligned = ns["_alignment_cost"](w, h)
        ew = ((w - 1) // engine_mod + 1) * engine_mod - w
        eh = ((h - 1) // engine_mod + 1) * engine_mod - h
        assert (pw, ph) == (ew, eh), \
            "advisory says the engine pads %dx%d by (%d,%d); the engine's own arithmetic says " \
            "(%d,%d)" % (w, h, pw, ph, ew, eh)
        assert aligned == (ew == 0 and eh == 0)
        aw, ah = ns["_nearest_aligned"](w, h)
        assert aw % engine_mod == 0 and ah % engine_mod == 0
        assert aw >= w and ah >= h, "the advisory must never suggest cropping - that is the " \
                                    "Save's job and it already has a law about it"
        assert aw - w < engine_mod and ah - h < engine_mod, "the suggestion must be the NEAREST"
        checked += 1

# --- Frank's canvases, by name -------------------------------------------
assert ns["_alignment_cost"](1075, 1075)[2] is False, "1075 is not mod 64 - the black ring fires"
assert ns["_alignment_cost"](1068, 1068)[2] is False, "1068 is even but NOT mod 64 - it still fires"
assert ns["_alignment_cost"](1088, 1088)[2] is True, "1088 is the canvas that costs nothing"
assert ns["_nearest_aligned"](1075, 1075) == (1088, 1088), \
    "from 1075 the node must point at 1088 - the measurement says that is worth 16 dB at the border"

# =========================================================================
# 4. The training heads: stripped by NAME, and only those.
# =========================================================================
strip = ns["_strip_training_heads"]
assert strip(["block0.conv.weight", "teacher.conv0.0.0.weight", "caltime.0.bias",
              "encode.0.weight", "block4.lastconv.bias"]) == \
    ["block0.conv.weight", "encode.0.weight", "block4.lastconv.bias"], \
    "the filter must drop teacher.* and caltime.* and NOTHING else"
for k in ("block0.x", "block4.x", "encode.x"):
    assert strip([k]) == [k], "%s is inference and must survive the filter" % k
# and it must not swallow a genuinely wrong architecture: strict stays strict
assert "strict=True" in open(SRC, encoding="utf-8").read(), \
    "the loader must keep strict=True. Dropping to strict=False would swallow a wrong " \
    "architecture along with the training heads, and a partial load produces video that looks " \
    "almost right - the one result nobody ever debugs."

print("[test_v598_vfi_alignment] PASS: %d canvases. The node calls pad() ZERO times (the engine "
      "already pads to mod %d with zeros and crops back - a second pad measured 4.18 dB worse). "
      "The advisory's modulus is parsed OUT of rife_arch.py, so stale advice goes red. 1075 -> "
      "1088 is the recommendation, and it is worth 9.4 dB / 16.1 dB at the border."
      % (checked, engine_mod))
sys.exit(0)
