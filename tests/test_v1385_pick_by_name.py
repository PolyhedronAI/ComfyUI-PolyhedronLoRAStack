"""Guard v1385 -- Pick Frame takes a frame by its NAME, and a mask goes along.

Frank, 24.11.: no Core nodes in our workflows -- the one workflow carried 19 'Get Image from Batch' (ImageFromBatch).
Polyhedron Pick Frame takes their place and does the job better: a turnaround's view is picked by its name from the
wired names list ('E' of 'S, SE, E, ...') instead of by a number somebody has to count, and a mask follows.

  B   by_name(): the name's index, case-blind, spaces and line breaks in the list; clear errors for no list, no
      name, an unknown name, a list of another length than the batch, a name held twice
  S   select(mode 'name'): that frame as a batch of one; the old modes untouched; the indices on request
  M   select_mask(): the same frames of a mask batch, a single mask as it is, none -> an empty one of the size;
      a mask too short -> an error
  N   both forms (legacy and V3): 'name' appended after count, 'names' and 'mask' optional sockets, a second
      output MASK; the mode list ends with 'name'; the baseline carries the append
  Z   ASCII; version stamps agree

Script-style: exit 0 = pass.
"""
import ast
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAILS = []


def check(c, m):
    if not c:
        FAILS.append(m)


def raises(fn, word, m):
    try:
        fn()
        check(False, m + " (no error)")
    except ValueError as e:
        check(word in str(e), m + ": %s" % e)


sys.path.insert(0, os.path.join(ROOT, "nodes"))
import torch  # noqa: E402
import uls_pick_frame_core as PF  # noqa: E402
import wan_frame_inflate as WF  # noqa: E402

NAMES = "S, SE, E, NE, N, NW, W, SW"
img = torch.arange(8, dtype=torch.float32).view(8, 1, 1, 1).expand(8, 4, 6, 3).clone()
msk = torch.arange(8, dtype=torch.float32).view(8, 1, 1).expand(8, 4, 6).clone() / 10.0

# B
check([PF.by_name(8, NAMES, n) for n in ("S", "E", "w", " sw ")] == [0, 2, 6, 7], "B the index of a name, case-blind, spaces stripped")
check(PF.by_name(3, "front\nside ,\n back", "back") == 2 and PF.names_list(" a, ,b\n") == ["a", "b"], "B line breaks and empty entries")
raises(lambda: PF.by_name(8, "", "E"), "needs the frames' names", "B no list")
raises(lambda: PF.by_name(8, NAMES, " "), "`name` is empty", "B no name")
raises(lambda: PF.by_name(8, NAMES, "X"), "no frame is called 'X' -- the list holds: S, SE, E", "B an unknown name shows the list")
raises(lambda: PF.by_name(7, NAMES, "E"), "8 name(s)", "B a list of another length than the batch")
raises(lambda: PF.by_name(3, "E, W, e", "E"), "2 times", "B a name held twice")

# S
o, note = PF.select(img, "name", 0, 1, NAMES, "W")
check(o.shape == (1, 4, 6, 3) and float(o[0, 0, 0, 0]) == 6.0 and "'W' = frame 6 of 8" in note and "[name]" in note, "S name W = frame 6: %s" % note)
o, note, got = PF.select(img, "name", 5, 3, NAMES, "ne", True)
check(got == [3] and float(o[0, 0, 0, 0]) == 3.0, "S frame_index and count do not count in mode name; the indices on request")
o, note = PF.select(img, "index", -1, 1)
check(float(o[0, 0, 0, 0]) == 7.0 and len(PF.select(img, "middle (legacy)", -1, 1)) == 2, "S the old call and the old modes stand")
o, note, got = PF.select(img, "every Nth", 0, 3, None, "", True)
check(got == [0, 3, 6] and o.shape[0] == 3, "S every Nth with its indices")
check(PF.MODES[:6] == ["middle (legacy)", "last", "first", "index", "range", "every Nth"] and PF.MODES[6:] == ["name"], "S 'name' appended to the modes")

# M
check(torch.equal(PF.select_mask(msk, [6], img[6:7]), msk[6:7]), "M the same frame of the mask")
check(torch.equal(PF.select_mask(msk, [0, 3, 6], img[:3]), torch.cat([msk[0:1], msk[3:4], msk[6:7]])), "M gathered frames")
one = torch.ones((1, 4, 6))
check(PF.select_mask(one, [5], img[5:6]) is one and PF.select_mask(torch.ones((4, 6)), [5], img[5:6]).shape == (1, 4, 6), "M a single mask as it is")
z = PF.select_mask(None, [2], img[2:3])
check(z.shape == (1, 4, 6) and float(z.sum()) == 0.0, "M none wired: an empty mask of the picked size")
raises(lambda: PF.select_mask(msk[:4], [6], img[6:7]), "the mask has 4 frame(s)", "M a mask too short")

# N
node = WF.ULSImagePickFrame
it = node.INPUT_TYPES()
check(list(it["required"]) == ["images", "frame_index", "mode", "count", "name"] and list(it["optional"]) == ["names", "mask"]
      and it["optional"]["names"][1].get("forceInput") is True, "N legacy: name appended, names + mask optional sockets")
check(node.RETURN_TYPES == ("IMAGE", "MASK") and node.RETURN_NAMES == ("image", "mask"), "N legacy: MASK appended")
r = node().pick(img, 0, "name", 1, "E", NAMES, msk)
check(float(r[0][0, 0, 0, 0]) == 2.0 and torch.equal(r[1], msk[2:3]), "N through the node: E, with its mask")
r = node().pick(img, -1)
check(float(r[0][0, 0, 0, 0]) == 4.0 and r[1].shape == (1, 4, 6), "N the old call: the middle frame, an empty mask")
raises(lambda: node().pick(img, 0, "name", 1, "E"), "needs the frames' names", "N mode name without a list")
v3 = open(os.path.join(ROOT, "nodes", "uls_pick_frame_v3.py"), encoding="utf-8").read()
tree = ast.parse(v3)
ins = [c.args[0].value for c in ast.walk(tree) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
       and c.func.attr == "Input" and c.args and isinstance(c.args[0], ast.Constant)]
check(ins == ["images", "frame_index", "mode", "count", "name", "names", "mask"], "N V3: the same inputs in the same order: %s" % ins)
check(v3.count("io.Image.Output(") == 1 and v3.count("io.Mask.Output(") == 1 and "force_input=True, optional=True" in v3
      and "_pf.select(images, mode, frame_index, count, names, name, True)" in v3, "N V3: the mask output, the sockets, the shared rule")
bl = sorted(glob.glob(os.path.join(ROOT, "WIDGET_ORDER_baseline_v*.txt")))
check(len(bl) == 1 and "\nULSImagePickFrame\tframe_index,mode,count,name\n" in open(bl[-1], encoding="utf-8").read(), "N the baseline carries the append")

# Z
for rel in ("nodes/uls_pick_frame_core.py", "tests/test_v1385_pick_by_name.py"):
    with open(os.path.join(ROOT, rel), "rb") as fh:
        check(all(ch < 128 for ch in fh.read()), "Z  ASCII-clean: " + rel)
_pv = re.search(r'PLUGIN_VERSION = "v(\d+)"', open(os.path.join(ROOT, "web", "js", "uls_compat.js")).read())
_pp = re.search(r'^version = "4\.(\d+)\.0"', open(os.path.join(ROOT, "pyproject.toml")).read(), re.M)
# PUBLIC CUT (v409, declared): the public build counts its own way -- vNNN = Major x 100 + Minor (test_v365 V).
check(_pv and _pp and int(_pv.group(1)) >= 409 and int(_pv.group(1)) - 400 == int(_pp.group(1)), "Z  version stamps agree")

if FAILS:
    print("FAIL test_v1385_pick_by_name:")
    for f_ in FAILS:
        print("  - " + f_)
    sys.exit(1)
print("ok test_v1385_pick_by_name")
