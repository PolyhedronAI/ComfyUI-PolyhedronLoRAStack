"""Guard v581 -- every registered node carries a help panel, or is declared exempt.

THE LAW: a node that ships with no DESCRIPTION shows an empty help panel in
ComfyUI. It is not a crash and no gate ever caught it, which is exactly why 28
of 39 nodes drifted into that state over five hundred builds.

WHAT THIS GUARD PINS -- and the distinction is the whole point:

  It resolves the class ComfyUI ACTUALLY LOADS for each node_id, then reads
  the text OFF THAT CLASS. It does not check that a DESCRIPTION statement
  exists somewhere in the file. It does not trust a class name. It asks the
  question the user asks: I clicked this node, is there help in the panel?

  Lesson 4 of the handover, made concrete: `test_v579_triggers` was green while
  the code returned "hour" instead of "golden hour", because it pinned the
  MECHANISM (a discriminator, a ladder order) and never a RETURN VALUE. So this
  guard pins the string, its length, and its content -- not the presence of an
  assignment statement.

THE V3 TRAP, walked into and backed out of during this very cut:

  A first pass resolved V3_NODE_CLASSES["ULSCamera"] to a class *named*
  ULSCamera -- and found it in ph_camera.py, the V1 anchor file. Wrong file,
  wrong class, wrong conclusion: it reported five nodes as missing help that
  have had it all along. The V3 map lives in nodes/uls_v3_extension.py and
  points at ULSCameraV3 in ph_camera_v3.py. NAME EQUALITY IS NOT IDENTITY.
  This guard reads the real map. If that map cannot be parsed, it FAILS rather
  than quietly falling back to the V1 names -- a fallback here would recreate
  the exact bug it exists to prevent.

EXEMPT, both by a HARDER law, neither by oversight:

  * The 4 anchor-file nodes (ULSCardinalCameras, ULSViewport3D, ULSWanBridge,
    ULSWanBridgeReverse). Writing help into an anchor changes its md5 and
    breaks the composite. The anchor law outranks the help-panel law.
    NOTE: ULSCamera, ULSMeshRender and ULSMeshToFile3D look like anchors and
    are NOT exempt -- under V3 they load from *_v3.py, which is not an anchor,
    and all three have carried a description= all along.

  * ULSMediaLoader. ph_media_loader.py is byte-pinned (216f852d...) for the
    open Bug-B investigation. A help panel is cosmetic; the pin is a live
    measurement with a hash the next session compares against. Stability before
    features -- the same call v577 made when it left this file alone.

  An exemption is a SENTENCE, not a silence. If one of these is ever written,
  this guard fails on the surplus and forces the exemption to be retired out
  loud.

Script-style: exit 0 = pass.
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = os.path.join(ROOT, "nodes")

# Exempt by a harder law. Each entry names the law that outranks this one.
EXEMPT = {
    "ULSCardinalCameras":  "anchor file (composite md5)",
    "ULSViewport3D":       "anchor file (composite md5)",
    "ULSWanBridge":        "anchor file (composite md5)",
    "ULSWanBridgeReverse": "anchor file (composite md5)",
    "ULSMediaLoader":      "Bug-B byte-pin on ph_media_loader.py",
}

# A help panel is text, not a gesture. Anything shorter than this is a label
# that someone typed to make a gate go green.
MIN_CHARS = 80

_TOKEN = re.compile(r"\bv\d{3,4}\b")   # v1001: four digits since v1000
_BANNED = ("mush",)


def _fail(msg):
    print("[test_v581_description_law] FAIL: " + msg)
    sys.exit(1)


def _classes():
    """{class_name: (filename, ClassDef, module_string_constants)}."""
    out = {}
    for fn in sorted(os.listdir(NODES)):
        if not fn.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(NODES, fn), encoding="utf-8").read())
        consts = _module_consts(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                out.setdefault(n.name, (fn, n, consts))
    return out


def _v3_map():
    """node_id -> V3 class name, read from nodes/uls_v3_extension.py.

    Failing here is correct. Falling back to "the class with the same name" is
    what produced a wrong answer once already; see the module docstring.
    """
    p = os.path.join(NODES, "uls_v3_extension.py")
    if not os.path.exists(p):
        _fail("nodes/uls_v3_extension.py is gone - the V3 registry is the only "
              "honest way to know which class a V3 node_id loads")
    src = open(p, encoding="utf-8").read()
    m = re.search(r"V3_NODE_CLASSES\s*(?::[^=]+)?=\s*\{(.*?)\n\}", src, re.S)
    if not m:
        _fail("V3_NODE_CLASSES dict not found in uls_v3_extension.py - this "
              "guard will not guess node_id -> class by name equality")
    out = dict(re.findall(r'"([^"]+)"\s*:\s*([A-Za-z_]\w*)', m.group(1)))
    if not out:
        _fail("V3_NODE_CLASSES parsed to zero entries - the regex has gone "
              "blind and every V3 node would be silently read off its V1 twin")
    return out


def _registered():
    """node_id -> V1 class name, exactly as __init__.py registers it."""
    src = open(os.path.join(ROOT, "__init__.py"), encoding="utf-8").read()
    reg = {}
    for m in re.finditer(
            r'NODE_CLASS_MAPPINGS\[\s*"([^"]+)"\s*\]\s*=\s*(.+)', src):
        expr = m.group(2).strip()
        tail = re.search(r"else\s+([A-Za-z_]\w*)\s*$", expr)
        reg[m.group(1)] = tail.group(1) if tail else expr
    upd = re.search(r"NODE_CLASS_MAPPINGS\.update\(\{(.*?)\}\)", src, re.S)
    if upd:
        for m in re.finditer(r'"(\w+)"\s*:\s*([A-Za-z_]\w*)', upd.group(1)):
            reg[m.group(1)] = m.group(2)
    return reg


def _module_consts(tree):
    """Module-level string constants, so `description=_DESCRIPTION` resolves.

    CAUGHT BY THIS GUARD ON ITS FIRST RUN, and worth writing down: three V3
    nodes (ULSCamera, ULSImagePickFrame, ULSWanFrameInflate) pass a module
    constant, not a literal. A first draft of this guard read the keyword,
    saw an ast.Name instead of an ast.Constant, and reported all three as
    having NO HELP PANEL. They have had one all along.

    The guard was wrong, not the code -- so the guard was fixed. But note the
    tempting wrong fix: treat a non-literal as "fine, something is there". That
    buys a green light and measures nothing, which is precisely the failure of
    test_v579_triggers. Resolve the constant and read the actual string, or
    fail. There is no third option that is honest.
    """
    out = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and isinstance(stmt.value, ast.Constant):
                    if isinstance(stmt.value.value, str):
                        out[t.id] = stmt.value.value
    return out


def _text_of(node, consts):
    """The string a description/DESCRIPTION node actually carries at runtime."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        return consts.get(node.id)          # None if we cannot resolve it
    return None


def _v1_text(cls, consts=None):
    consts = consts or {}
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            for t in stmt.targets:
                if isinstance(t, ast.Name) and t.id == "DESCRIPTION":
                    return _text_of(stmt.value, consts)
    return None


def _v3_text(cls, consts=None):
    consts = consts or {}
    for n in ast.walk(cls):
        if isinstance(n, ast.Call):
            for kw in n.keywords:
                if kw.arg == "description":
                    return _text_of(kw.value, consts)
    return None


def _resolve(nid, reg, v3, classes):
    """Return (text_or_None, filename, path_label) for the class ACTUALLY loaded."""
    if nid in v3:
        cname = v3[nid]
        if cname not in classes:
            _fail(f"{nid}: V3 registry points at class {cname}, which does not "
                  f"exist in nodes/ - the map and the tree disagree")
        fn, cls, consts = classes[cname]
        return _v3_text(cls, consts), fn, "V3"
    cname = reg[nid]
    if cname not in classes:
        _fail(f"{nid}: registered as {cname}, which does not exist in nodes/")
    fn, cls, consts = classes[cname]
    return _v1_text(cls, consts), fn, "V1"


def main():
    classes = _classes()
    v3 = _v3_map()
    reg = _registered()

    if len(reg) < 30:
        _fail(f"only {len(reg)} node_ids parsed out of __init__.py - the "
              f"registry scan broke and this guard is asleep")
    if len(v3) < 5:
        _fail(f"only {len(v3)} V3 nodes resolved - the pack has nine; the V3 "
              f"map is not being read")

    missing, thin, dirty, surplus = [], [], [], []
    covered = 0

    for nid in sorted(reg):
        text, fn, path = _resolve(nid, reg, v3, classes)
        # bool(text) is the whole verdict: a name we could not resolve comes
        # back None and counts as MISSING, not as "probably fine".
        has = bool(text)

        if nid in EXEMPT:
            if has:
                surplus.append((nid, EXEMPT[nid]))
            continue

        if not has:
            missing.append((nid, path, fn))
            continue

        covered += 1
        if len(text) < MIN_CHARS:
            thin.append((nid, len(text)))
        for tok in _TOKEN.findall(text):
            dirty.append((nid, f"build number {tok}"))
        low = text.lower()
        for w in _BANNED:
            if w in low:
                dirty.append((nid, f'banned word "{w}"'))

    if missing:
        _fail("nodes with no help panel: " + ", ".join(
            f"{n} ({p}:{f})" for n, p, f in missing)
            + " - write a DESCRIPTION (V1) or description= (V3 schema), or "
              "declare the exemption here with the law that outranks this one")
    if thin:
        _fail("help text too thin (< %d chars): " % MIN_CHARS + ", ".join(
            f"{n}={c}" for n, c in thin)
            + " - a label is not a help panel")
    if dirty:
        _fail("help text violates the UI-text law: " + ", ".join(
            f"{n}: {w}" for n, w in dirty))
    if surplus:
        _fail("these are listed EXEMPT but now carry help text: " + ", ".join(
            f"{n} (exempt for: {w})" for n, w in surplus)
            + " - good news, but retire the exemption in this same cut. An "
              "exemption nobody removes is a lie the next reader believes")

    # --- negative probe: prove the gate can actually fail ---------------------
    # A guard that has never been shown to fire is a decoration. Feed it a class
    # with no DESCRIPTION and confirm _v1_text returns None -- if this ever
    # returns a string, the reader above is matching something it should not and
    # every "covered" verdict in this run is worthless.
    probe = ast.parse("class P:\n    CATEGORY = 'x'\n").body[0]
    if _v1_text(probe) is not None:
        _fail("negative probe failed: the V1 reader found DESCRIPTION on a "
              "class that has none - this guard cannot be trusted")
    probe3 = ast.parse("class P:\n    def s():\n        return io.Schema(node_id='x')\n").body[0]
    if _v3_text(probe3) is not None:
        _fail("negative probe failed: the V3 reader found description= on a "
              "schema that has none - this guard cannot be trusted")

    # And the constant resolver itself, which is the piece that got this guard
    # wrong on its first run. POSITIVE probe: it must actually resolve a name.
    # NEGATIVE probe: an unresolvable name must come back None, never a
    # placeholder that would sail through the bool(text) check above.
    ptree = ast.parse('_D = "resolved"\nclass P:\n'
                      '    def s():\n        return io.Schema(description=_D)\n')
    pconsts = _module_consts(ptree)
    pcls = [n for n in ptree.body if isinstance(n, ast.ClassDef)][0]
    if _v3_text(pcls, pconsts) != "resolved":
        _fail("positive probe failed: the constant resolver did not resolve "
              "description=_D to its module string - the three V3 nodes that "
              "pass a constant would be misread as having no help panel")
    if _v3_text(pcls, {}) is not None:
        _fail("negative probe failed: an unresolvable name came back as "
              "something truthy - a name this guard cannot read must be a "
              "failure, not a pass")

    print(f"[test_v581_description_law] OK - {covered} of {len(reg)} registered "
          f"nodes carry a help panel, read off the class ComfyUI actually loads "
          f"({len(v3)} via V3 schema, the rest via the V1 attribute); "
          f"{len(EXEMPT)} exempt by a harder law "
          f"(4 anchors + the Bug-B byte-pin), both negative probes fired")
    sys.exit(0)


if __name__ == "__main__":
    main()
