"""v874 -- the switch pair stops shrinking itself into an overlap.

THE WOUND (Frank, from the field): after every restart the ULSAnySwitch drew
"on_missing" and "use next active" on top of each other. Not a paint bug -- a
SIZE bug. Both tidy functions ended with

    node.setSize(node.computeSize())

which hard-resets the node to LiteGraph's computed MINIMUM. That runs on every
connection change AND on every workflow load (onConfigure -> double-rAF ->
tidy), so a width the user had dragged out was discarded on every restart, and
the minimum is narrower than that label/value pair needs.

WHAT THIS PINS:
  * the shrinking call is GONE from both lanes;
  * a MIN_W floor exists and is wide enough for the pair that collided;
  * the width formula is GROW-ONLY -- it must consider the node's CURRENT width,
    or a dragged-out node snaps back on the next connection change;
  * the height still FOLLOWS computeSize, because collapsing a spare slot has to
    shrink the box -- that is what tidy() is for. A guard that demanded
    grow-only height would have frozen the node at its largest ever size.

Script-style: exit 0 = pass. Reads the source; no node, no browser.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
JS = (ROOT / "web" / "js" / "ph_switch.js").read_text(encoding="utf-8")
FAILED = []


def _strip_comments(src):
    """JS with its comments removed.

    THREE guards in this session matched their own prose: a docstring or a
    comment that QUOTES the line under test keeps a text search green while the
    code says something else. Any needle that means "this code is gone" has to
    look at code only."""
    out, i, n = [], 0, len(src)
    while i < n:
        two = src[i:i + 2]
        if two == "/*":
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif two == "//":
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src[i] in "\"'`":
            q = src[i]
            j = i + 1
            while j < n and src[j] != q:
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
        else:
            out.append(src[i])
            i += 1
    return "".join(out)


CODE = _strip_comments(JS)


def _fail(m):
    FAILED.append(m)
    print("FAIL: {}".format(m))


def _ok(m):
    print("ok  : {}".format(m))


# --- the shrinking call must be gone ---------------------------------------
shrink = re.findall(r"setSize\(\s*(?:this|node)\.computeSize\(\s*\)\s*\)", CODE)
if shrink:
    _fail("{} call(s) to setSize(computeSize()) remain -- the node still "
          "hard-resets to LiteGraph's minimum and the labels collide again "
          "after every reload".format(len(shrink)))
else:
    _ok("no setSize(computeSize()) left -- nothing hard-resets the node")

# --- the floor -------------------------------------------------------------
m = re.search(r"const\s+MIN_W\s*=\s*(\d+)\s*;", CODE)
if not m:
    _fail("MIN_W is missing -- there is no floor under the width")
else:
    min_w = int(m.group(1))
    # "on_missing" + "use next active" + the toggle + two margins. Measured
    # loosely at LiteGraph's 12px widget font: ~65 + ~85 + ~20 + ~30 = ~200.
    # 240 is the smallest value that still leaves the pair visibly apart.
    if min_w < 240:
        _fail("MIN_W is {} -- too narrow for 'on_missing' against 'use next "
              "active'; that pair is what collided".format(min_w))
    else:
        _ok("MIN_W = {} clears the pair that collided".format(min_w))

# --- grow-only width, following height -------------------------------------
fit = re.search(r"function\s+fitSize\s*\(node\)\s*\{(.*?)\n\}", CODE, re.S)
if not fit:
    _fail("fitSize() is missing -- both lanes need ONE sizing rule, not two "
          "copies of it")
else:
    body = fit.group(1)
    def _split_top(expr):
        """Split on TOP-LEVEL commas only. A plain split(',') tears
        Math.max(MIN_W, want[0], cur) into pieces and the check then reads the
        wrong operand -- which is how the first version of this guard let a
        mutation through."""
        out, depth, cur_s = [], 0, ""
        for ch in expr:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            if ch == "," and depth == 0:
                out.append(cur_s)
                cur_s = ""
            else:
                cur_s += ch
        out.append(cur_s)
        return [x.strip() for x in out]

    setsize = re.search(r"setSize\(\s*\[(.*?)\]\s*\)", body, re.S)
    if "computeSize" not in body:
        _fail("fitSize must still ASK computeSize what the node needs")
    elif not setsize:
        _fail("fitSize does not call setSize with an explicit [w, h]")
    else:
        args = _split_top(setsize.group(1))
        if len(args) != 2:
            _fail("setSize needs exactly [width, height], got {}".format(args))
        else:
            width_expr, height_expr = args
            if "MIN_W" not in width_expr:
                _fail("the WIDTH expression must floor at MIN_W, got: {}"
                      .format(width_expr))
            elif "Math.max" not in width_expr:
                _fail("the WIDTH must be a Math.max, or it is not grow-only")
            elif not re.search(r"\bcur\b|node\.size", width_expr):
                _fail("the WIDTH expression must include the node's CURRENT "
                      "width -- without it a width the user dragged out snaps "
                      "back on the next connection change, which is the bug "
                      "wearing a different hat. Got: {}".format(width_expr))
            elif "want" not in width_expr:
                _fail("the WIDTH must still respect what computeSize needs")
            else:
                _ok("width = max(MIN_W, computed, current) -- floored and "
                    "grow-only")

            # Height must NOT be grow-only: a collapsed spare slot has to
            # shrink it.
            if "Math.max" in height_expr or "node.size" in height_expr:
                _fail("the HEIGHT must follow computeSize, not grow only -- "
                      "collapsing a spare slot has to shrink the box, and "
                      "freezing it at the largest ever size is a different bug")
            elif "want" not in height_expr:
                _fail("the height should come from computeSize's answer")
            else:
                _ok("height still follows computeSize (spare slots collapse)")

# --- both lanes use it -----------------------------------------------------
for lane in ("tidyInputs", "tidyOutputs"):
    fn = re.search(r"function\s+%s\s*\(node\)\s*\{(.*?)\n\}" % lane, CODE, re.S)
    if not fn:
        _fail("{} is missing".format(lane))
    elif "fitSize(node)" not in fn.group(1):
        _fail("{} does not call fitSize -- one lane would keep shrinking"
              .format(lane))
    else:
        _ok("{} sizes through fitSize".format(lane))

# --- this guard now OWNS the banner version (tree convention) --------------
if "[PLS] ph_switch.js v874 loaded" not in JS:
    _fail("the self-proving banner is not at v874 -- the file's newest guard "
          "pins the exact version (see test_v550_processview)")
else:
    _ok("banner pinned at v874 by its newest guard")

older = (ROOT / "tests" / "test_v537_switch.py").read_text(encoding="utf-8")
if "ph_switch.js v537 loaded" in older:
    _fail("test_v537 still pins the OLD exact banner -- two guards pinning one "
          "banner means the next bump breaks the older one every time")
else:
    _ok("test_v537 handed the exact banner pin over, as the convention says")

print("\n{}: {} failure(s)".format(pathlib.Path(__file__).name, len(FAILED)))
sys.exit(1 if FAILED else 0)
