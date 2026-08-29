"""Guard v577 -- the serialisation law, for ALL of them.

THE STANDING EXPOSURE (found by the v576 audit): LiteGraph restores
widgets_values POSITIONALLY. Slip a widget into the MIDDLE of an INPUT_TYPES
and every saved workflow that uses that node shifts one along -- a bool lands
in an int slot, a seed lands in a sampler name. Silently. No exception, no red
box; just wrong values in a workflow that used to work.

The law against it has existed for a long time ("new widgets ONLY at the end")
and it is honoured: there is not one index-based widget access in the whole
frontend. But it was ENFORCED for exactly ONE node -- ph_power_upscale's
ORDER_CANON (25 entries). The other 38 were protected by memory and care.

This gate generalises ORDER_CANON to the tree. The audit measured that 36 of
39 INPUT_TYPES can be read STATICALLY (no torch, no ComfyUI, pure AST), so the
protection is simply available:

  THE LAW: for every node in the baseline, the baseline's widget order must be
  a PREFIX of the current order. Appending at the end -> pass (that is the
  law). Insert, reorder or delete -> FAIL, with the exact position named.

  NEW NODE: absent from the baseline -> FAIL, asking for a regenerated
  baseline. A new node is a declared change; it goes through the ceremony like
  everything else.

  DYNAMIC: three nodes build their widgets at runtime and are out of static
  reach (ULSLoadModel, ULSCLIPTextEncode, ULSAnySwitch). They are DECLARED in
  the baseline. If a fourth node goes dynamic, this guard fires -- a node
  quietly leaving the gate's protection is exactly the thing we want to hear
  about.

  CANON (v770): a node may also SORT its INPUT_TYPES by a declared canon
  before returning it. That reads as "dynamic" to the AST -- there is no
  literal return dict -- but it is the opposite of dynamic: it is the one
  shape in which the rows on screen and the values on disk CANNOT drift
  apart, because both come from the same list. Filing such a node under
  DYNAMIC would drop it out of this gate for being MORE careful, so it gets
  its own line and stays guarded: the gate imports the module, asks the
  class what it really returns, and holds it to the baseline order. That is
  stronger than the AST path, which only ever read the source.

Regenerating the baseline is a DECLARED act: it belongs in the cut that changes
a widget, named in the changelog, next to the migration/heal that carries the
old workflows across.

Script-style: exit 0 = pass.
"""
import ast
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _newest_baseline(pattern, label):
    """AMENDED IN v580 (2nd amendment): the baseline FILENAME was itself a text
    pin. It read `WIDGET_ORDER_baseline_v579.txt`, hard-coded, so every cut had
    to remember to edit this guard -- and a guard that must be hand-edited on a
    schedule is a guard that will one day be hand-edited wrong.

    Lesson 1 of the handover, applied to the guard that enforces lesson 1:
    pin the STRUCTURE, not the string. Resolve the newest baseline by version.

    Exactly one must exist per cut; two is an ambiguity, not a convenience.

    AMENDED IN v581 (3rd amendment): the sentence above was TRUE IN THE DOCSTRING
    AND FALSE IN THE CODE. It said two baselines are an ambiguity, then quietly
    took max() and carried on. That is lesson 4 -- a guard that claims in its own
    prose what it never executes -- committed by the guard that enforces lesson 1.
    A stale baseline left behind by a forgetful cut would have sat in the tree
    unmentioned, and the day someone deleted the newer one, the gate would have
    silently started comparing against history. Two now fails, loudly.
    """
    hits = sorted(glob.glob(os.path.join(ROOT, pattern)))
    if not hits:
        _fail(f"no {label} baseline found ({pattern}) - the gate has no memory "
              f"to compare against")
    if len(hits) > 1:
        _fail(f"{len(hits)} {label} baselines present "
              f"({', '.join(os.path.basename(h) for h in hits)}) - exactly one "
              f"per cut. Delete the stale one; an ambiguity is not a convenience")
    def _v(p):
        stem = os.path.basename(p).rsplit("_v", 1)[-1]
        return int("".join(c for c in stem if c.isdigit()) or 0)
    return max(hits, key=_v)


BASELINE = None   # resolved below, after _fail is defined


def _fail(msg):
    print("[test_v577_widget_order] FAIL: " + msg)
    sys.exit(1)


# Types that occupy a widgets_values slot. Everything else (IMAGE, MODEL,
# LATENT, VAE, CONDITIONING, "*", ...) is a SOCKET: it never appears in
# widgets_values, so it may be added anywhere without shifting a thing.
# CAUGHT BY THE GUARD'S OWN NEGATIVE PROBE, before it ever landed: a first
# draft counted every INPUT_TYPES entry as a widget and therefore forbade a
# legal append to the end of `required` on nodes whose optionals are all
# sockets (ULSSave). The guard was wrong, not the code.
WIDGET_SCALARS = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def _is_widget(spec):
    """Does this INPUT_TYPES entry occupy a widgets_values slot?

    RE-GROUNDED v838 (audit B4): an entry carrying forceInput: True is a
    SOCKET no matter its scalar type -- ComfyUI renders it as an input
    link and it never occupies a widgets_values slot. The LAW is
    unchanged (the baseline pins slot order); the scanner just learned
    what a slot IS. The old coarse reading held four phantom entries in
    the baseline (Everywhere, Inspector, ResolveInspector, TokenCounter)
    -- harmless under prefix comparison while the phantoms sat at the
    END, but ResolveInspector's socket sits FIRST, and the v838
    regeneration surfaced it. The three Outpaint guards (v790/v799/
    v812) have compared against the socket-free runtime order all
    along.

    RE-GROUNDED v850 (BinOp): a combo list may be BUILT rather than
    named -- `([SAME_AS_HIGH] + list(comfy.samplers.KSampler.SAMPLERS),
    {...})`, `([SAME_AS_MAIN] + modes, {...})`, `(["auto"] + types,
    {...})`. That is an ast.BinOp, which no branch below covered, so the
    scanner fell through to False and filed a REAL widget as a socket.
    FOUR nodes were affected -- ULSPowerUpscale and ULSSampler
    (sampler_low / scheduler_low), ULSAttention (attention_first /
    attention_last) and ULSLoadCLIP (type) -- and their entries vanished
    from the baseline SILENTLY.

    MEASURED before the fix, not assumed. Power Upscale and the Sampler
    publish their true serialisation order in the frontend as
    ORDER_CANON / ORDER_V404, and the patched scan reproduces both
    EXACTLY (25/25 and 19/19 once core's control_after_generate is
    discounted). For the other two the argument is structural: this
    scanner only reads a node statically when INPUT_TYPES returns a
    LITERAL dict, and a literal dict's iteration order IS the order
    INPUT_TYPES() yields at runtime -- so scan order and slot order
    cannot diverge there.

    The baseline was BLIND, not WRONG -- no saved workflow shifted, only
    the yardstick was short. Nothing to heal, nothing to migrate."""
    if (isinstance(spec, (ast.Tuple, ast.List)) and len(spec.elts) > 1
            and isinstance(spec.elts[1], ast.Dict)):
        for k, v in zip(spec.elts[1].keys, spec.elts[1].values):
            if (isinstance(k, ast.Constant) and k.value == "forceInput"
                    and isinstance(v, ast.Constant) and v.value is True):
                return False
    t = spec.elts[0] if isinstance(spec, (ast.Tuple, ast.List)) and spec.elts else spec
    if isinstance(t, ast.Constant) and isinstance(t.value, str):
        return t.value in WIDGET_SCALARS            # "IMAGE" -> socket
    if isinstance(t, (ast.List, ast.Tuple)):
        return True                                 # inline combo list
    if isinstance(t, (ast.Name, ast.Attribute, ast.Call, ast.Subscript)):
        return True                                 # referenced combo list
    if isinstance(t, ast.BinOp):
        return True                                 # combo BUILT by expression
    return False


# ComfyUI reads exactly these three sections out of an INPUT_TYPES dict.
# Anything else is not a section it knows.
_SECTIONS = ("required", "optional", "hidden")


def _order_of(fn):
    """Ordered WIDGET names of one INPUT_TYPES, statically.
    (names, dynamic?, stray sections)

    required widgets first, then optional widgets, in declaration order - which
    is exactly the order LiteGraph fills widgets_values in.

    v901: STRAY SECTIONS ARE NOW REPORTED. This scanner used to `continue`
    past any key that was not required/optional, which silently tolerated the
    worst version of the very fault this gate exists to catch: in v900 a new
    widget was written one brace too late and landed as a THIRD top-level
    section beside required and optional. Python accepted it, py_compile
    accepted it, this gate said the widget order was unchanged -- and ComfyUI
    refused to register the node at all, so the whole VAE box came up red with
    UNKNOWN widgets in Frank's graph. A widget that lands outside every
    section is not "no change", it is a node that does not load."""
    order, dynamic, stray = [], False, []
    for n in ast.walk(fn):
        if isinstance(n, ast.Return) and isinstance(n.value, ast.Dict):
            for k, v in zip(n.value.keys, n.value.values):
                sec = getattr(k, "value", None)
                if sec not in _SECTIONS:
                    if isinstance(k, ast.Constant):
                        stray.append(str(sec))
                    else:
                        dynamic = True      # a computed section key
                    continue
                if sec == "hidden":
                    continue                # hidden fills no widgets_values slot
                if not isinstance(v, ast.Dict):
                    dynamic = True          # the whole section is computed
                    continue
                for kk, vv in zip(v.keys, v.values):
                    if not isinstance(kk, ast.Constant):
                        dynamic = True      # a computed key
                        continue
                    if _is_widget(vv):
                        order.append(str(kk.value))
            return order, dynamic, stray
    return order, True, stray               # no literal return dict at all


def _scan():
    """{node_class: (order, dynamic, file)} for every INPUT_TYPES in nodes/.

    Stray sections are collected on the side and checked before anything
    else -- a node that cannot register has no widget order to compare."""
    out, strays = {}, []
    for p in sorted(glob.glob(os.path.join(ROOT, "nodes", "*.py"))):
        tree = ast.parse(open(p, encoding="utf-8").read())
        for c in ast.walk(tree):
            if not isinstance(c, ast.ClassDef):
                continue
            for fn in c.body:
                if isinstance(fn, ast.FunctionDef) and fn.name == "INPUT_TYPES":
                    o, d, st = _order_of(fn)
                    out[c.name] = (o, d, os.path.basename(p))
                    for sec in st:
                        strays.append((c.name, os.path.basename(p), sec))
    for cls, fname, sec in strays:
        _fail(f"{cls} ({fname}): INPUT_TYPES returns a section named "
              f"'{sec}'. ComfyUI reads only "
              f"{'/'.join(_SECTIONS)} -- a widget written one brace too late "
              f"lands here, and the node then fails to register (red box, "
              f"UNKNOWN widgets). This is the v900 fault.")
    return out


def _load_baseline():
    global BASELINE
    path = _newest_baseline("WIDGET_ORDER_baseline_v*.txt", "WIDGET_ORDER")
    BASELINE = os.path.basename(path)
    if not os.path.isfile(path):
        _fail(f"{BASELINE} is missing - the gate has no memory to compare against")
    static, dynamic, canon = {}, set(), {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if parts[0] == "DYNAMIC":
            dynamic.add(parts[1])
        elif parts[0] == "CANON" and len(parts) == 4:
            # CANON <TAB> NodeId <TAB> file <TAB> widget,widget,...
            canon[parts[1]] = (parts[2],
                               [w for w in parts[3].split(",") if w])
        elif len(parts) == 2:
            static[parts[0]] = [w for w in parts[1].split(",") if w]
    return static, dynamic, canon


RT_WIDGET_SCALARS = WIDGET_SCALARS


def _runtime_order(fname, cls_name):
    """The widget order the class ACTUALLY returns, asked of the module.

    The AST path cannot read a canon-sorted INPUT_TYPES -- that is the whole
    point of it -- so this one drives the code instead of reading it."""
    import importlib.util
    os.environ.setdefault("ULS_BGR_HOME", "/tmp/bgr")
    os.environ.setdefault("ULS_SAM_HOME", "/tmp/sam")
    os.environ.setdefault("ULS_DINO_HOME", "/tmp/dino")
    path = os.path.join(ROOT, "nodes", fname)
    spec = importlib.util.spec_from_file_location(
        "ph_order_%s" % cls_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    it = getattr(mod, cls_name).INPUT_TYPES()
    order = []
    for section in ("required", "optional"):
        for name, entry in (it.get(section) or {}).items():
            t = entry[0] if isinstance(entry, (tuple, list)) and entry \
                else entry
            if isinstance(t, str):
                if t in RT_WIDGET_SCALARS:
                    order.append(name)
            elif isinstance(t, (list, tuple)):
                order.append(name)                  # combo list
    return order


def main():
    now = _scan()
    base_static, base_dynamic, base_canon = _load_baseline()

    if not base_static:
        _fail("the baseline holds no static node - it is empty or malformed")

    # --- 1: nothing may quietly leave the gate ---------------------------------
    for name, (order, dyn, f) in sorted(now.items()):
        if name in base_canon:
            continue                              # its own check, below
        if dyn and name not in base_dynamic:
            _fail(f"{name} ({f}) builds its widgets DYNAMICALLY but is not "
                  f"declared as such - it just left the gate's protection "
                  f"unannounced. Declare it in {BASELINE} (DYNAMIC line) and "
                  f"say in the changelog why.")
        if (not dyn) and name in base_dynamic:
            _fail(f"{name} ({f}) is declared DYNAMIC but reads statically now - "
                  f"good news: move it into the guarded set (regenerate "
                  f"{BASELINE}).")

    # --- 2: every node the baseline knows must still obey the law --------------
    for name, want in sorted(base_static.items()):
        if name not in now:
            _fail(f"{name} is in the baseline but has no INPUT_TYPES any more - "
                  f"a removed node is a declared act (regenerate {BASELINE}).")
        have, dyn, f = now[name]
        if dyn:
            continue                              # covered by check 1
        if have[:len(want)] != want:
            # name the FIRST divergence - that is the position that shifts.
            i = 0
            while i < min(len(want), len(have)) and want[i] == have[i]:
                i += 1
            old = want[i] if i < len(want) else "<end>"
            new = have[i] if i < len(have) else "<end>"
            _fail(f"{name} ({f}) BREAKS THE SERIALISATION LAW at widget index "
                  f"{i}: the baseline has '{old}' there, the code now has "
                  f"'{new}'. Every saved workflow using this node would shift "
                  f"its values from here on - silently. New widgets go at the "
                  f"END. If this move is intended, it needs a heal/migration "
                  f"and a regenerated {BASELINE}, named in the changelog.\n"
                  f"       baseline: {want}\n"
                  f"       code now: {have}")

    # --- 2b: canon-sorted nodes obey the SAME law, asked of the code -----------
    for name, (fname, want) in sorted(base_canon.items()):
        if name not in now:
            _fail(f"{name} is declared CANON in the baseline but has no "
                  f"INPUT_TYPES any more - a removed node is a declared act "
                  f"(regenerate {BASELINE}).")
            continue
        try:
            have = _runtime_order(fname, name)
        except Exception as exc:                   # noqa: BLE001
            _fail(f"{name} ({fname}) is declared CANON but its INPUT_TYPES "
                  f"could not be driven: {type(exc).__name__}: {exc}")
            continue
        if have[:len(want)] != want:
            i = 0
            while i < min(len(want), len(have)) and want[i] == have[i]:
                i += 1
            was = want[i] if i < len(want) else "<end>"
            now_ = have[i] if i < len(have) else "<end>"
            _fail(f"{name} ({fname}) BREAKS THE SERIALISATION LAW at widget "
                  f"index {i}: the baseline has '{was}' there, the canon now "
                  f"puts '{now_}'. Every saved workflow using this node would "
                  f"shift its values from here on - silently. Growth goes at "
                  f"the END; a re-founding needs the heal/migration and a "
                  f"regenerated {BASELINE}, named in the changelog.\n"
                  f"       baseline: {want}\n"
                  f"       code now: {have}")

    # --- 3: a NEW node is a declared change ------------------------------------
    unknown = sorted(set(now) - set(base_static) - base_dynamic
                     - set(base_canon))
    if unknown:
        _fail(f"new node(s) not in the baseline: {', '.join(unknown)}. A new "
              f"node is a declared act - regenerate {BASELINE} in THIS cut so "
              f"the gate carries it from now on.")

    # --- 4: the gate must actually cover the tree ------------------------------
    static_guarded = sum(1 for n, (o, d, f) in now.items()
                         if not d and n not in base_canon)
    guarded = static_guarded + len(base_canon)
    # RE-GROUNDED FOR THE PUBLIC BUILD (v372), declared in the changelog.
    # This floor is a CENSUS of the tree the gate guards, not a law: it fires
    # when a node silently goes dynamic and thereby leaves the guarded set.
    # The internal tree's census is 36 at this cut. THIS tree also carries 36
    # node classes, but a different set: THREE are declared dynamic and there
    # is no CANON row (the node that owns it does not exist here), so full
    # coverage is 34. That number is MEASURED, not derived -- my first draft
    # of this line said 35 from arithmetic and the gate said 33 on the first
    # run. Keeping the internal number would make a correct public build look
    # broken; a lower one would let a real loss slip through.
    if guarded < 34:
        _fail(f"only {guarded} nodes are guarded - the audit measured 34. "
              f"Something went dynamic without being declared.")

    print(f"[test_v577_widget_order] PASS: {static_guarded} nodes guarded "
          f"statically, {len(base_canon)} by their declared canon, "
          f"{len(base_dynamic)} declared dynamic; the serialisation law holds "
          f"for every one of them.")
    sys.exit(0)


if __name__ == "__main__":
    main()
